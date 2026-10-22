"""Linux and Windows log analysis for Nashmi Agent.

Pure Python, no Streamlit and no network: the same input always gives the
same output, and nothing here calls an AI model, so it works for guests
without an account or an API key.

Supported input
    Linux    syslog / auth.log / secure (``Mon DD HH:MM:SS host proc[pid]: msg``)
             and ISO-timestamp lines (``journalctl -o short-iso``, rsyslog ISO).
    Windows  Event Viewer CSV export, ``Get-WinEvent | Export-Csv``,
             ``wevtutil qe <log> /f:xml`` output, and PowerShell
             ``ConvertTo-Json``.  Binary ``.evtx`` files are NOT read; the
             caller is told how to export them.

What the results mean
    * ``Event`` fields are copied from the log and are facts about the file.
    * ``Finding.observed`` is a count or sequence taken from the events.
    * ``Finding.meaning`` is a fixed, rule-based explanation (not a verdict).
    * ``Finding.hypothesis`` is only set for correlations that go beyond what
      one event proves; it is always phrased as something to verify.
"""

from __future__ import annotations

import csv
import io
import ipaddress
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

MAX_BYTES = 25 * 1024 * 1024
MAX_EVENTS = 200_000
SEVERITIES = ("info", "low", "medium", "high")
SEV_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}

# thresholds for the correlation rules
BRUTE_FORCE_MIN = 5          # failures from one source before it is reported
BRUTE_FORCE_HIGH = 20
SPRAY_MIN_USERS = 5          # distinct accounts tried from one source
SUCCESS_AFTER_MIN = 3        # failures before a later success is flagged
PERSISTENCE_WINDOW = timedelta(hours=24)


class LogFormatError(ValueError):
    """The input is not a log format this module reads."""


@dataclass(frozen=True)
class Event:
    n: int                  # position in the file, from 0
    line: int               # 1-based line (Linux) or record (Windows) number
    ts: datetime | None     # naive UTC; None when the line has no usable time
    os: str                 # "linux" or "windows"
    host: str
    source: str             # process (Linux) or provider (Windows)
    event_id: str           # Windows event id; empty on Linux
    user: str
    ip: str
    message: str
    raw: str
    kind: str               # rule key, "other" when no rule matched
    severity: str
    title: str
    dst: str = ""           # network captures: destination address
    port: int = 0           # network captures: destination port
    proto: str = ""         # network captures: protocol name
    size: int = 0           # network captures: bytes
    count: int = 1          # network captures: packets in this flow / entry


@dataclass(frozen=True)
class Finding:
    id: str
    rule: str
    title: str
    severity: str
    observed: str           # what the log shows
    meaning: str            # fixed rule text: why this matters
    hypothesis: str         # "" unless this is a correlation to verify
    attack: str             # MITRE ATT&CK technique mapped by the rule
    events: tuple[int, ...]
    ips: tuple[str, ...] = ()
    users: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()
    first: datetime | None = None
    last: datetime | None = None


@dataclass
class Report:
    os: str
    events: list[Event]
    findings: list[Finding]
    total_lines: int
    skipped: int
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)   # network captures: packet timeline, protocol mix


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _valid_ip(value: str) -> str:
    value = (value or "").strip().strip("[]")
    if not value or value in ("-", "::", "0.0.0.0"):
        return ""
    if value.startswith("::ffff:"):
        value = value[7:]
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return ""


def _naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def decode_bytes(data: bytes) -> str:
    """UTF-8, or the UTF-16 that PowerShell writes by default."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


# --------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"), 1)}
SYSLOG_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d\d:\d\d:\d\d)\s+"
    r"(?P<host>\S+)\s+(?P<proc>[^\s\[:]+)(?:\[(?P<pid>\d+)\])?:\s?(?P<msg>.*)$")
ISO_RE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:[.,]\d+)?(?:Z|[+-]\d\d:?\d\d)?)\s+"
    r"(?P<host>\S+)\s+(?P<proc>[^\s\[:]+)(?:\[(?P<pid>\d+)\])?:\s?(?P<msg>.*)$")


def _parse_iso(text: str) -> datetime | None:
    text = text.replace(",", ".").replace("Z", "+00:00")
    text = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", text)
    try:
        return _naive_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _syslog_time(mon: str, day: str, clock: str, now: datetime) -> datetime | None:
    try:
        hh, mm, ss = (int(x) for x in clock.split(":"))
        stamp = datetime(now.year, _MONTHS[mon], int(day), hh, mm, ss)
    except (KeyError, ValueError):
        return None
    if stamp > now + timedelta(days=1):      # "Dec 31" read in January
        stamp = stamp.replace(year=now.year - 1)
    return stamp


_PRIV_GROUPS = {"sudo", "wheel", "admin", "root", "adm"}


def classify_linux(proc: str, msg: str) -> tuple[str, str, str, str, str]:
    """Return (kind, severity, title, user, ip) for one message."""
    m = re.search(r"Failed (?:password|keyboard-interactive/pam|publickey) for "
                  r"(?:invalid user )?(?P<u>.+?) from (?P<ip>\S+)(?: port|$)", msg)
    if m:
        return "failed_login", "low", "Failed SSH login", m["u"], _valid_ip(m["ip"])
    m = re.search(r"Invalid user (?P<u>.*?) from (?P<ip>\S+)", msg)
    if m:
        return ("failed_login", "low", "SSH login for a user that does not exist",
                m["u"], _valid_ip(m["ip"]))
    m = re.search(r"Accepted (?P<how>\S+) for (?P<u>.+?) from (?P<ip>\S+)(?: port|$)", msg)
    if m:
        if m["u"] == "root":
            return ("root_login", "medium", "SSH login as root", "root",
                    _valid_ip(m["ip"]))
        return ("login", "info", "Successful SSH login", m["u"], _valid_ip(m["ip"]))
    if "authentication failure" in msg:
        ip = re.search(r"rhost=(\S+)", msg)
        user = re.search(r"(?<![a-z])user=(\S+)", msg)
        if proc.startswith("sudo"):
            return ("sudo_failed", "medium", "sudo authentication failure",
                    user[1] if user else "", "")
        return ("failed_login", "low", "PAM authentication failure",
                user[1] if user else "", _valid_ip(ip[1]) if ip else "")
    m = re.match(r"\s*(?P<u>\S+)\s*:\s.*COMMAND=(?P<c>.*)$", msg)
    if proc.startswith("sudo") and m:
        if "incorrect password" in msg:
            return "sudo_failed", "medium", "sudo: incorrect password", m["u"], ""
        return "sudo", "info", "sudo command", m["u"], ""
    m = re.search(r"\(to root\) (?P<u>\S+) on", msg)
    if proc.startswith("su") and m:
        return "su_root", "medium", "su to root", m["u"], ""
    m = re.search(r"FAILED SU \(to (?P<t>\S+)\) (?P<u>\S+) on", msg)
    if m:
        return "failed_login", "low", "Failed su", m["u"], ""
    m = re.search(r"new user: name=(?P<u>[^,\s]+)", msg)
    if m:
        return "user_created", "medium", "User account created", m["u"], ""
    m = (re.search(r"add '(?P<u>[^']+)' to (?:shared )?group '(?P<g>[^']+)'", msg)
         or re.search(r"user (?P<u>\S+) added by \S+ to group (?P<g>\S+)", msg))
    if m:
        sev = "high" if m["g"].lower() in _PRIV_GROUPS else "medium"
        return ("group_member_added", sev,
                f"User added to group {m['g']}", m["u"], "")
    m = re.search(r"delete user '(?P<u>[^']+)'", msg)
    if m:
        return "user_deleted", "low", "User account deleted", m["u"], ""
    m = re.search(r"(?:Did not receive identification string from|Connection "
                  r"(?:closed|reset) by (?:authenticating user \S+ |invalid user "
                  r"\S+ )?|Bad protocol version identification .* from) ?(?P<ip>[0-9a-fA-F:.]+)", msg)
    if proc.startswith("sshd") and m and _valid_ip(m["ip"]):
        return ("ssh_probe", "low", "SSH connection without a login attempt",
                "", _valid_ip(m["ip"]))
    return "other", "info", "Log entry", "", ""


def parse_linux(text: str, now: datetime | None = None
                ) -> tuple[list[Event], int, int, list[str]]:
    now = now or datetime.now()
    events: list[Event] = []
    skipped = total = 0
    notes: list[str] = []
    year_guess = False
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        if len(events) >= MAX_EVENTS:
            break
        stamp: datetime | None
        m = ISO_RE.match(line)
        if m:
            stamp = _parse_iso(m["ts"])
        else:
            m = SYSLOG_RE.match(line)
            if not m:
                skipped += 1
                continue
            stamp = _syslog_time(m["mon"], m["day"], m["time"], now)
            year_guess = True
        kind, sev, title, user, ip = classify_linux(m["proc"], m["msg"])
        events.append(Event(
            n=len(events), line=number, ts=stamp, os="linux", host=m["host"],
            source=m["proc"], event_id="", user=user, ip=ip,
            message=m["msg"][:2000], raw=line[:4000], kind=kind, severity=sev,
            title=title))
    if year_guess:
        notes.append("Classic syslog lines carry no year; the current year was "
                     "assumed (the previous one for dates in the future).")
    return events, total, skipped, notes


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WinRule:
    kind: str
    severity: str
    title: str


WIN_RULES: dict[str, WinRule] = {
    "4624": WinRule("login", "info", "Successful logon"),
    "4625": WinRule("failed_login", "low", "Failed logon"),
    "4648": WinRule("explicit_credentials", "low", "Logon with explicit credentials"),
    "4672": WinRule("special_privileges", "info", "Special privileges assigned"),
    "4688": WinRule("process_created", "info", "Process created"),
    "4697": WinRule("service_installed", "medium", "Service installed"),
    "7045": WinRule("service_installed", "medium", "Service installed"),
    "4698": WinRule("task_created", "medium", "Scheduled task created"),
    "4702": WinRule("task_updated", "low", "Scheduled task updated"),
    "4719": WinRule("audit_policy_changed", "high", "System audit policy changed"),
    "4720": WinRule("user_created", "medium", "User account created"),
    "4722": WinRule("user_enabled", "low", "User account enabled"),
    "4723": WinRule("password_change", "low", "Password change attempted"),
    "4724": WinRule("password_reset", "low", "Password reset attempted"),
    "4726": WinRule("user_deleted", "low", "User account deleted"),
    "4728": WinRule("group_member_added", "medium", "Member added to a security group"),
    "4732": WinRule("group_member_added", "medium", "Member added to a security group"),
    "4756": WinRule("group_member_added", "medium", "Member added to a security group"),
    "4740": WinRule("account_lockout", "medium", "Account locked out"),
    "4771": WinRule("failed_login", "low", "Kerberos pre-authentication failed"),
    "4776": WinRule("ntlm_auth", "info", "NTLM credential validation"),
    "1102": WinRule("log_cleared", "high", "Security audit log cleared"),
    "104": WinRule("log_cleared", "high", "Event log cleared"),
    "1116": WinRule("defender_detection", "high", "Defender detected malware"),
    "1117": WinRule("defender_action", "medium", "Defender took action on malware"),
}

# Rule text shown with a finding (fixed text, not generated per file).
KIND_INFO: dict[str, tuple[str, str]] = {
    # kind: (meaning, MITRE ATT&CK)
    "password_spray": ("Trying many accounts from one source with few attempts each is password spraying.",
                       "T1110.003 Password Spraying"),
    "success_after_failures": ("A logon succeeded from a source that had repeatedly failed before it.",
                               "T1110 Brute Force"),
    "new_account_privileged": ("A new account that quickly gains group rights matches a persistence pattern.",
                               "T1136 Create Account"),
    "failed_login": ("Repeated failed logons are the usual trace of password guessing.",
                     "T1110 Brute Force"),
    "login": ("A logon succeeded. Only unusual sources, accounts or times matter.",
              "T1078 Valid Accounts"),
    "root_login": ("Direct administrator logons over SSH are rarely needed and are a common target.",
                   "T1078 Valid Accounts"),
    "rdp_login": ("Logon type 10 is a remote desktop session.",
                  "T1021.001 Remote Desktop Protocol"),
    "explicit_credentials": ("A process used another account's credentials.",
                             "T1078 Valid Accounts"),
    "user_created": ("A new account can be used to keep access.",
                     "T1136 Create Account"),
    "group_member_added": ("Adding a member to a privileged group raises that account's rights.",
                           "T1098 Account Manipulation"),
    "password_reset": ("A password was changed or reset for an account.",
                       "T1098 Account Manipulation"),
    "password_change": ("A password change was attempted for an account.",
                        "T1098 Account Manipulation"),
    "service_installed": ("New services run code automatically and are a persistence method.",
                          "T1543.003 Windows Service"),
    "task_created": ("Scheduled tasks run code automatically and are a persistence method.",
                     "T1053.005 Scheduled Task"),
    "log_cleared": ("Clearing logs removes the evidence an investigation needs.",
                    "T1070.001 Clear Windows Event Logs"),
    "audit_policy_changed": ("Changing the audit policy can stop events from being recorded.",
                             "T1562.002 Disable Windows Event Logging"),
    "account_lockout": ("Accounts lock after many failed logons.", "T1110 Brute Force"),
    "defender_detection": ("The antivirus reported malware on this host.",
                           "T1204 User Execution"),
    "defender_action": ("The antivirus acted on a detection.", ""),
    "suspicious_process": ("The command line matches a pattern attackers commonly use.",
                           "T1059 Command and Scripting Interpreter"),
    "sudo_failed": ("A user failed to elevate privileges.",
                    "T1548.003 Sudo and Sudo Caching"),
    "su_root": ("Someone switched to the administrator account.",
                "T1548 Abuse Elevation Control Mechanism"),
    "ssh_probe": ("A connection opened and closed without a login attempt, typical of scanners.",
                  "T1595 Active Scanning"),
    "user_deleted": ("An account was removed.", "T1531 Account Access Removal"),
    "task_updated": ("A scheduled task was changed.", "T1053.005 Scheduled Task"),
    "user_enabled": ("A disabled account was enabled.", "T1098 Account Manipulation"),
}

SUSPICIOUS_CMD: tuple[tuple[str, str, str], ...] = (
    (r"-enc(?:odedcommand)?\s", "high", "encoded PowerShell command"),
    (r"mimikatz|sekurlsa|lsadump", "high", "credential dumping tool"),
    (r"vssadmin(?:\.exe)?\s+delete\s+shadows", "high", "shadow copies deleted"),
    (r"invoke-expression|\biex\b\s*[\(\$]|downloadstring|downloadfile", "high",
     "PowerShell downloads and runs code"),
    (r"comsvcs(?:\.dll)?.*minidump", "high", "process memory dump via comsvcs"),
    (r"reg(?:\.exe)?\s+save\s+hklm\\(?:sam|security|system)", "high",
     "registry hive saved (credential theft)"),
    (r"certutil(?:\.exe)?.*(?:-urlcache|-decode)", "medium", "certutil used to download or decode"),
    (r"bitsadmin(?:\.exe)?.*/transfer", "medium", "bitsadmin file transfer"),
    (r"net1?(?:\.exe)?\s+user\s+\S+\s+\S+\s+/add", "medium", "local account added from the command line"),
    (r"wmic(?:\.exe)?.*process\s+call\s+create", "medium", "process started through WMI"),
    (r"whoami(?:\.exe)?\s+/(?:all|priv)", "low", "privilege discovery"),
)


def _alias(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


_COLUMN_ALIASES = {
    "ts": ("timecreated", "dateandtime", "datetime", "timestamp", "systemtime", "date", "time"),
    "id": ("id", "eventid"),
    "source": ("providername", "source", "provider"),
    "host": ("machinename", "computer", "computername", "hostname"),
    "message": ("message", "description", "details", "friendlymessage"),
}

_MSG_FIELDS = (
    ("user", r"Account Name:\s+(\S+)", True),            # last one = the target
    ("ip", r"Source Network Address:\s+(\S+)", False),
    ("ip2", r"Client Address:\s+(?:::ffff:)?(\S+)", False),
    ("logon_type", r"Logon Type:\s+(\d+)", False),
    ("process", r"New Process Name:\s+(.+)", False),
    ("cmdline", r"(?:Process )?Command Line:\s+(.+)", False),
    ("service", r"Service Name:\s+(.+)", False),
    ("task", r"Task Name:\s+(.+)", False),
    ("group", r"Group Name:\s+(.+)", False),
    ("member", r"Member:[\s\S]*?Account Name:\s+(\S+)", False),
)


def _fields_from_message(message: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, pattern, last in _MSG_FIELDS:
        found = re.findall(pattern, message)
        if found:
            out[key] = (found[-1] if last else found[0]).strip()
    if out.get("user", "").endswith("$"):
        out["user"] = out["user"]          # machine account, kept as logged
    return out


_WIN_TIME_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
                     "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                     "%d.%m.%Y %H:%M:%S")


def _parse_win_time(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    m = re.search(r"/Date\((\d+)", value)               # PowerShell 5 JSON
    if m:
        return datetime.fromtimestamp(int(m[1]) / 1000, tz=timezone.utc).replace(tzinfo=None)
    iso = _parse_iso(value.replace("Z", "+00:00") if value.endswith("Z") else value)
    if iso:
        return iso
    for fmt in _WIN_TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def classify_windows(event_id: str, source: str, fields: dict[str, str]
                     ) -> tuple[str, str, str]:
    """Return (kind, severity, title) for one Windows event."""
    if event_id in ("1116", "1117") and "defender" not in source.lower():
        return "other", "info", "Log entry"
    rule = WIN_RULES.get(event_id)
    if rule is None:
        return "other", "info", f"Event {event_id}" if event_id else "Log entry"
    kind, sev, title = rule.kind, rule.severity, rule.title
    if event_id == "4624" and fields.get("logon_type") == "10":
        return "rdp_login", "medium", "Remote desktop logon"
    if kind == "group_member_added":
        group = fields.get("group", "") or fields.get("target_group", "")
        if re.search(r"admin", group, re.I):
            sev = "high"
            title = f"Member added to {group}"
    if event_id == "4688" and fields.get("cmdline"):
        for pattern, level, label in SUSPICIOUS_CMD:
            if re.search(pattern, fields["cmdline"] + " ", re.I):
                return "suspicious_process", level, f"Suspicious command: {label}"
    return kind, sev, title


def _win_event(n: int, line: int, row: dict[str, str], fields: dict[str, str],
               raw: str) -> Event:
    event_id = str(row.get("id", "")).strip()
    source = row.get("source", "")
    kind, sev, title = classify_windows(event_id, source, fields)
    ip = _valid_ip(fields.get("ip", "")) or _valid_ip(fields.get("ip2", ""))
    user = fields.get("user", "")
    if kind == "group_member_added" and fields.get("member"):
        user = fields["member"]
    message = row.get("message", "") or " | ".join(
        f"{k}={v}" for k, v in fields.items() if v and k not in ("ip2",))
    return Event(
        n=n, line=line, ts=_parse_win_time(row.get("ts", "")), os="windows",
        host=row.get("host", ""), source=source, event_id=event_id,
        user=user if user not in ("-", "") else "", ip=ip,
        message=message[:2000], raw=raw[:4000], kind=kind, severity=sev, title=title)


def _map_columns(names) -> dict[str, str]:
    lookup = {_alias(n): n for n in names}
    out = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                out[key] = lookup[alias]
                break
    return out


def _rows_to_events(rows: list[dict], total_start: int = 2
                    ) -> tuple[list[Event], int, int]:
    events: list[Event] = []
    skipped = 0
    cols = _map_columns(rows[0].keys()) if rows else {}
    if "id" not in cols:
        raise LogFormatError(
            "No event id column found. Export with Event Viewer (Save All "
            "Events As, CSV) or Get-WinEvent | Export-Csv.")
    for i, source_row in enumerate(rows):
        if len(events) >= MAX_EVENTS:
            break
        row = {k: str(source_row.get(cols[k], "") or "") if k in cols else ""
               for k in _COLUMN_ALIASES}
        if not row["id"].strip():
            skipped += 1
            continue
        fields = _fields_from_message(row["message"])
        raw = json.dumps({k: v for k, v in source_row.items() if v}, ensure_ascii=False)
        events.append(_win_event(len(events), i + total_start, row, fields, raw))
    return events, len(rows), skipped


def _guard_xml(text: str) -> None:
    if re.search(r"<!DOCTYPE|<!ENTITY", text, re.I):
        raise LogFormatError("XML with a DOCTYPE or entities is refused for safety.")


def parse_windows_xml(text: str) -> tuple[list[Event], int, int]:
    import xml.etree.ElementTree as ET
    _guard_xml(text)
    blocks = re.findall(r"<Event\b.*?</Event>", text, re.S)
    events: list[Event] = []
    skipped = 0
    for i, block in enumerate(blocks, 1):
        if len(events) >= MAX_EVENTS:
            break
        try:
            root = ET.fromstring(block)
        except ET.ParseError:
            skipped += 1
            continue

        def local(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        row = {"id": "", "source": "", "host": "", "ts": "", "message": ""}
        fields: dict[str, str] = {}
        for el in root.iter():
            name = local(el.tag)
            if name == "EventID":
                row["id"] = (el.text or "").strip()
            elif name == "Provider":
                row["source"] = el.get("Name", "")
            elif name == "Computer":
                row["host"] = (el.text or "").strip()
            elif name == "TimeCreated":
                row["ts"] = el.get("SystemTime", "")
            elif name == "Data":
                key = el.get("Name") or ""
                if key:
                    fields[key] = (el.text or "").strip()
        xml_map = {"TargetUserName": "user", "IpAddress": "ip", "LogonType": "logon_type",
                   "NewProcessName": "process", "CommandLine": "cmdline",
                   "ServiceName": "service", "TaskName": "task",
                   "MemberName": "member"}
        norm = {xml_map[k]: v for k, v in fields.items() if k in xml_map and v}
        if row["id"] in ("4672", "4688", "4697", "7045") and "user" not in norm:
            if fields.get("SubjectUserName"):
                norm["user"] = fields["SubjectUserName"]
        if row["id"] in ("4728", "4732", "4756"):
            norm["group"] = fields.get("TargetUserName", "")
            norm.pop("user", None)
        elif "user" in norm and row["id"] == "4625":
            pass
        events.append(_win_event(len(events), i, row, norm, block))
    return events, len(blocks), skipped


def parse_windows_json(text: str) -> tuple[list[Event], int, int]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LogFormatError(f"The JSON could not be read: {exc.msg}.") from exc
    if isinstance(data, dict):
        data = [data]
    rows = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    if not rows:
        raise LogFormatError("The JSON holds no event objects.")
    return _rows_to_events(rows, total_start=1)


def parse_windows_csv(text: str) -> tuple[list[Event], int, int]:
    csv.field_size_limit(10_000_000)
    lines = text.splitlines()
    # Get-WinEvent -> Export-Csv may start with a "#TYPE ..." line
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    if not lines:
        raise LogFormatError("The file is empty.")
    sample = lines[0]
    delimiter = max(",;\t", key=sample.count)
    rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    if not rows:
        raise LogFormatError("The CSV has a header but no rows.")
    return _rows_to_events(rows)


# --------------------------------------------------------------------------
# detection and entry point
# --------------------------------------------------------------------------
def detect(text: str) -> str:
    """'linux', 'windows-xml', 'windows-json', 'windows-csv' or ''."""
    head = text.lstrip()[:4000]
    if head.startswith("<") and "<Event" in text[:20000]:
        return "windows-xml"
    if head.startswith(("[", "{")):
        return "windows-json" if re.search(r'"(?:Id|EventID)"', head) else ""
    first = next((ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")), "")
    if re.search(r"event\s*id|^\W*id\W", first, re.I) and first.count(",") + first.count(";") + first.count("\t") >= 2:
        return "windows-csv"
    sample = [ln for ln in text.splitlines()[:200] if ln.strip()]
    hits = sum(bool(SYSLOG_RE.match(ln) or ISO_RE.match(ln)) for ln in sample)
    if sample and hits / len(sample) >= 0.3:
        return "linux"
    return ""


def analyze(data: bytes | str, os_hint: str = "auto",
            now: datetime | None = None) -> Report:
    """Parse a log and run every rule. Raises LogFormatError on bad input."""
    if isinstance(data, bytes):
        if len(data) > MAX_BYTES:
            raise LogFormatError(f"The file is larger than {MAX_BYTES // 2**20} MB.")
        if data[:7] == b"ElfFile":
            raise LogFormatError(
                "This is a binary .evtx file. Export it first: "
                "wevtutil qe Security /f:xml > security.xml  or  "
                "Get-WinEvent -LogName Security | Export-Csv security.csv")
        text = decode_bytes(data)
    else:
        text = data
    if not text.strip():
        raise LogFormatError("The input is empty.")
    kind = detect(text)
    if os_hint == "linux" and kind.startswith("windows"):
        raise LogFormatError("This looks like a Windows export, not Linux syslog.")
    if os_hint == "windows" and kind == "linux":
        raise LogFormatError("This looks like Linux syslog, not a Windows export.")
    if not kind:
        raise LogFormatError(
            "The format was not recognised. Supported: Linux syslog/auth.log, "
            "Windows Event Viewer CSV, wevtutil XML and PowerShell JSON.")
    notes: list[str] = []
    if kind == "linux":
        events, total, skipped, notes = parse_linux(text, now)
        os_name = "linux"
    else:
        os_name = "windows"
        parser = {"windows-xml": parse_windows_xml, "windows-json": parse_windows_json,
                  "windows-csv": parse_windows_csv}[kind]
        events, total, skipped = parser(text)
        if kind == "windows-csv":
            notes.append("Dates written like 03/04/2024 are read as month/day first, then day/month.")
    if not events:
        raise LogFormatError("No readable events were found in the input.")
    truncated = total > len(events) + skipped and len(events) >= MAX_EVENTS
    if truncated:
        notes.append(f"Only the first {MAX_EVENTS:,} events were analysed.")
    if all(e.ts is None for e in events):
        notes.append("No event has a readable time, so the timeline uses file order.")
    report = Report(os=os_name, events=events, findings=[], total_lines=total,
                    skipped=skipped, truncated=truncated, notes=notes)
    report.findings = correlate(events)
    return report


# --------------------------------------------------------------------------
# findings and correlation
# --------------------------------------------------------------------------
def _order(events):
    return sorted(events, key=lambda e: (e.ts is None, e.ts or datetime.min, e.n))


def _span(events) -> tuple[datetime | None, datetime | None]:
    stamps = [e.ts for e in events if e.ts]
    return (min(stamps), max(stamps)) if stamps else (None, None)


def _fmt(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown time"


def _top(counter: Counter, limit: int = 5) -> str:
    return ", ".join(f"{k} ({v})" for k, v in counter.most_common(limit))


def correlate(events: list[Event]) -> list[Finding]:
    findings: list[Finding] = []
    ordered = _order(events)

    def add(rule, title, severity, observed, evidence, kind="", hypothesis="",
            ips=(), users=(), hosts=()):
        meaning, attack = KIND_INFO.get(kind or rule, ("", ""))
        first, last = _span(evidence)
        findings.append(Finding(
            id=f"F{len(findings) + 1:03d}", rule=rule, title=title,
            severity=severity, observed=observed, meaning=meaning,
            hypothesis=hypothesis, attack=attack,
            events=tuple(e.n for e in evidence[:500]),
            ips=tuple(sorted(set(i for i in ips if i)))[:20],
            users=tuple(sorted(set(u for u in users if u)))[:20],
            hosts=tuple(sorted(set(h for h in hosts if h)))[:20],
            first=first, last=last))

    # 1. repeated failures per source address (or per account when local)
    failures = [e for e in ordered if e.kind == "failed_login"]
    by_ip: dict[str, list[Event]] = defaultdict(list)
    by_user_local: dict[str, list[Event]] = defaultdict(list)
    for e in failures:
        if e.ip:
            by_ip[e.ip].append(e)
        elif e.user:
            by_user_local[e.user].append(e)
    for ip, evs in by_ip.items():
        users = Counter(e.user for e in evs if e.user)
        if len(evs) >= BRUTE_FORCE_MIN:
            first, last = _span(evs)
            add("brute_force", f"Repeated failed logons from {ip}",
                "high" if len(evs) >= BRUTE_FORCE_HIGH else "medium",
                f"{len(evs)} failed logons from {ip} between {_fmt(first)} and "
                f"{_fmt(last)}. Accounts tried: {_top(users) or 'none named'}.",
                evs, "failed_login", ips=[ip], users=users, hosts=[e.host for e in evs])
        if len(users) >= SPRAY_MIN_USERS and len(evs) >= BRUTE_FORCE_MIN:
            add("password_spray", f"One source tried {len(users)} different accounts: {ip}",
                "high",
                f"{ip} failed against {len(users)} distinct accounts "
                f"({len(evs)} failures in total).",
                evs, "", ips=[ip], users=users)
    for user, evs in by_user_local.items():
        if len(evs) >= BRUTE_FORCE_MIN:
            add("brute_force", f"Repeated failed logons for account {user}", "medium",
                f"{len(evs)} failed logons for {user} with no source address in the log.",
                evs, "failed_login", users=[user], hosts=[e.host for e in evs])

    # 2. a success from a source that failed before (a hypothesis, not a verdict)
    ok_kinds = {"login", "root_login", "rdp_login"}
    seen: dict[str, list[Event]] = defaultdict(list)
    flagged: set[str] = set()
    for e in ordered:
        if e.kind == "failed_login" and e.ip:
            seen[e.ip].append(e)
        elif e.kind in ok_kinds and e.ip and e.ip not in flagged and len(seen[e.ip]) >= SUCCESS_AFTER_MIN:
            flagged.add(e.ip)
            prior = seen[e.ip]
            add("success_after_failures", f"Successful logon from {e.ip} after {len(prior)} failures",
                "high",
                f"{len(prior)} failed logons from {e.ip} (last at {_fmt(prior[-1].ts)}), then a "
                f"successful logon as {e.user or 'an unnamed account'} at {_fmt(e.ts)}.",
                prior + [e], "",
                hypothesis="The password may have been guessed or reused. Confirm with the account "
                           "owner whether this logon was theirs before treating it as a compromise.",
                ips=[e.ip], users=[e.user] + [p.user for p in prior], hosts=[e.host])

    # 3. new account followed by a privileged group change (persistence pattern)
    created = [e for e in ordered if e.kind == "user_created" and e.user]
    added = [e for e in ordered if e.kind == "group_member_added"]
    for c in created:
        for g in added:
            same = c.user.lower() in (g.message + " " + g.user).lower()
            near = c.ts and g.ts and timedelta(0) <= g.ts - c.ts <= PERSISTENCE_WINDOW
            if same and (near or (c.ts is None and g.n > c.n)):
                add("new_account_privileged", f"Account {c.user} created, then given group rights",
                    "high",
                    f"{c.user} was created at {_fmt(c.ts)} and a group change mentioning it was "
                    f"logged at {_fmt(g.ts)} ({g.title}).",
                    [c, g], "",
                    hypothesis="This pattern can be an attacker keeping access, or a normal admin "
                               "task. Check whether the change was planned.",
                    users=[c.user], hosts=[c.host, g.host])
                break

    # 4. everything else of medium/high weight, one finding per kind
    skip = {"failed_login", "login"}
    groups: dict[str, list[Event]] = defaultdict(list)
    for e in ordered:
        if e.kind not in skip and e.kind != "other" and SEV_RANK[e.severity] >= SEV_RANK["medium"]:
            groups[(e.kind, e.title)].append(e)
    for (kind, title), evs in groups.items():
        sev = max((e.severity for e in evs), key=SEV_RANK.get)
        users = Counter(e.user for e in evs if e.user)
        ips = Counter(e.ip for e in evs if e.ip)
        first, last = _span(evs)
        extra = f" Accounts: {_top(users)}." if users else ""
        extra += f" Sources: {_top(ips)}." if ips else ""
        add(kind, title + (f" ({len(evs)}x)" if len(evs) > 1 else ""), sev,
            f"{len(evs)} event(s) between {_fmt(first)} and {_fmt(last)}.{extra}",
            evs, kind, ips=ips, users=users, hosts=[e.host for e in evs])

    findings.sort(key=lambda f: (-SEV_RANK[f.severity], f.first or datetime.max, f.id))
    return findings


# --------------------------------------------------------------------------
# views over a report
# --------------------------------------------------------------------------
def summary(report: Report) -> dict:
    ev = report.events
    first, last = _span(ev)
    sev = Counter(e.severity for e in ev)
    return {
        "events": len(ev), "skipped": report.skipped, "lines": report.total_lines,
        "first": first, "last": last,
        "ips": len({e.ip for e in ev if e.ip}),
        "users": len({e.user for e in ev if e.user}),
        "hosts": len({e.host for e in ev if e.host}),
        "by_severity": {s: sev.get(s, 0) for s in SEVERITIES},
        "finding_severity": Counter(f.severity for f in report.findings),
    }


def graph(report: Report, max_nodes: int = 45) -> dict:
    """Entities and the links between them.

    Every solid link comes from one event that names both ends (an address
    and an account in the same log entry) or from a rule that flagged both.
    Links to findings marked as hypotheses are returned with ``hypothesis``
    set so the UI can draw them differently.
    """
    weight: Counter = Counter()
    links: dict[tuple[str, str], dict] = {}
    multi_host = len({e.host for e in report.events if e.host}) > 1

    def link(a: str, b: str, kind: str, hypothesis: bool = False):
        key = (a, b)
        item = links.setdefault(key, {"source": a, "target": b, "count": 0,
                                      "kind": kind, "hypothesis": hypothesis})
        item["count"] += 1
        weight[a] += 1
        weight[b] += 1

    for e in report.events:
        if e.ip and e.user:
            link(f"ip:{e.ip}", f"user:{e.user}", "same event")
        if multi_host and e.host and e.ip:
            link(f"ip:{e.ip}", f"host:{e.host}", "same event")
        if multi_host and e.host and e.user:
            link(f"user:{e.user}", f"host:{e.host}", "same event")
    nodes: dict[str, dict] = {}
    for f in report.findings[:15]:
        node = f"finding:{f.id}"
        hyp = bool(f.hypothesis)
        nodes[node] = {"id": node, "label": f"{f.id} {f.title}", "category": "finding",
                       "severity": f.severity, "hypothesis": hyp}
        for ip in f.ips:
            link(node, f"ip:{ip}", "flagged by rule", hyp)
        for user in f.users[:6]:
            link(node, f"user:{user}", "flagged by rule", hyp)
    keep = {n for n, _ in weight.most_common(max_nodes)} | set(nodes)
    total_entities = len(weight)
    out_nodes = []
    for node in keep:
        if node in nodes:
            out_nodes.append(nodes[node])
            continue
        cat, _, label = node.partition(":")
        out_nodes.append({"id": node, "label": label, "category": cat,
                          "weight": weight[node]})
    out_links = [l for l in links.values() if l["source"] in keep and l["target"] in keep]
    return {"nodes": out_nodes, "links": out_links, "total_entities": total_entities,
            "shown_entities": len([n for n in out_nodes if n["category"] != "finding"])}


def report_markdown(report: Report, name: str = "log") -> str:
    s = summary(report)
    lines = [f"# Nashmi log analysis: {name}", "",
             f"- System: {report.os}",
             f"- Events read: {s['events']:,} ({s['skipped']:,} lines skipped)",
             f"- Time range: {_fmt(s['first'])} to {_fmt(s['last'])}", "",
             "Findings are produced by fixed rules. 'Observed' is taken from the "
             "log; 'Hypothesis' is something to verify, not a conclusion.", ""]
    if not report.findings:
        lines.append("No rule matched.")
    for f in report.findings:
        lines += [f"## {f.id} [{f.severity.upper()}] {f.title}", "",
                  f"**Observed:** {f.observed}", ""]
        if f.meaning:
            lines += [f"**Why it matters (rule):** {f.meaning}", ""]
        if f.hypothesis:
            lines += [f"**Hypothesis to verify:** {f.hypothesis}", ""]
        if f.attack:
            lines += [f"**MITRE ATT&CK (mapped by rule):** {f.attack}", ""]
        refs = ", ".join(str(report.events[i].line) for i in f.events[:30])
        lines += [f"**Source lines:** {refs}", ""]
    return "\n".join(lines)
