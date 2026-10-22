"""Network capture analysis for Nashmi Agent (Wireshark files).

Pure Python, no Streamlit, no network and no AI model: the same file always
gives the same result, so it works for guests with no account or key.

Supported input
    * .pcap  (classic libpcap, both byte orders, micro and nanosecond)
    * .pcapng (section / interface / enhanced and simple packet blocks)
    * Wireshark CSV export (File > Export Packet Dissections > As CSV)
    Link types: Ethernet (with VLAN), raw IP, Linux cooked, loopback.
    Decoded: ARP, IPv4, IPv6, TCP, UDP, ICMP, DNS, HTTP request lines,
    FTP/Telnet clear-text logins and the TLS server name (SNI).

Passwords are never stored: when a clear-text login is found only the user
name is kept, so the report cannot leak the secret it warns about.

The results use the same ``Event`` / ``Finding`` / ``Report`` types as
``nashmi_logs`` so one dashboard shows both. Every finding keeps the
"observed" text (taken from packets), the fixed rule text and, for patterns
across packets, a hypothesis to verify.
"""

from __future__ import annotations

import base64
import csv
import io
import ipaddress
import math
import re
import statistics
import struct
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import nashmi_logs as L
from nashmi_logs import Event, Finding, LogFormatError, Report

MAX_PACKETS = 200_000
MAX_DNS_EVENTS = 3000
MAX_FLOW_EVENTS = 5000

# rule thresholds
SCAN_MIN_TARGETS = 20        # distinct (host, port) pairs probed by one source
SWEEP_MIN_HOSTS = 20         # distinct hosts probed by one source
SYN_FLOOD_MIN = 300          # SYNs to one service inside one window
SYN_FLOOD_WINDOW = 10.0
DNS_LABEL_LONG = 50
DNS_NAME_LONG = 100
DNS_MANY_SUBDOMAINS = 100
BEACON_MIN_HITS = 8
BEACON_MAX_CV = 0.1          # interval spread / mean interval
BEACON_MIN_INTERVAL = 5.0
UPLOAD_BYTES = 5 * 1024 * 1024

PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6), b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9), b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}
PCAPNG_MAGIC = bytes.fromhex("0a0d0d0a")
PCAPNG_LE_BOM = bytes.fromhex("4d3c2b1a")

PORT_NAMES = {20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 67: "dhcp",
              80: "http", 110: "pop3", 123: "ntp", 135: "msrpc", 139: "netbios", 143: "imap",
              161: "snmp", 389: "ldap", 443: "https", 445: "smb", 465: "smtps", 587: "smtp",
              993: "imaps", 995: "pop3s", 1433: "mssql", 3306: "mysql", 3389: "rdp", 5432: "postgres",
              5900: "vnc", 8080: "http-alt", 8443: "https-alt"}
INSECURE = {21: "FTP", 23: "Telnet", 110: "POP3", 143: "IMAP", 80: "HTTP"}

NET_INFO: dict[str, tuple[str, str]] = {
    "port_scan": ("One source probing many ports is how attackers find open services.", "T1046 Network Service Discovery"),
    "host_sweep": ("One source probing many hosts maps which machines are alive.", "T1018 Remote System Discovery"),
    "syn_flood": ("A burst of connection requests to one service can exhaust it.", "T1498 Network Denial of Service"),
    "cleartext_credentials": ("Anyone who can see this traffic can read the login. The capture itself proves it was sent unprotected.", "T1040 Network Sniffing"),
    "dns_tunnel": ("Very long or very many unique DNS names can carry data out of a network.", "T1071.004 DNS"),
    "beaconing": ("Connections at a steady rhythm to one place look like software checking in with a controller.", "T1071 Application Layer Protocol"),
    "arp_conflict": ("Two hardware addresses claiming one IP address is how traffic gets intercepted on a LAN.", "T1557.002 ARP Cache Poisoning"),
    "large_upload": ("A large transfer to an outside address can be data leaving the network.", "T1048 Exfiltration Over Alternative Protocol"),
    "insecure_protocol": ("This protocol sends data unencrypted.", "T1040 Network Sniffing"),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ip4(b: bytes) -> str:
    return ".".join(str(x) for x in b)


def _ip6(b: bytes) -> str:
    return str(ipaddress.IPv6Address(b))


def _mac(b: bytes) -> str:
    return ":".join(f"{x:02x}" for x in b)


_DOC_NETS = tuple(ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))


def is_private(ip: str) -> bool:
    """Inside the network. The documentation ranges (TEST-NET) count as outside:
    courses and labs use them as stand-ins for the internet."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if a.version == 4 and any(a in net for net in _DOC_NETS):
        return False
    return a.is_private or a.is_loopback or a.is_link_local or a.is_multicast


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)


def _fmt(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown time"


def _service(port: int, proto: str) -> str:
    return PORT_NAMES.get(port, f"{proto.lower()}/{port}" if port else proto)


# --------------------------------------------------------------------------
# capture file readers: yield (timestamp_seconds, linktype, bytes)
# --------------------------------------------------------------------------
def _read_pcap(data: bytes):
    order, scale = PCAP_MAGICS[data[:4]]
    if len(data) < 24:
        raise LogFormatError("The capture file is truncated.")
    link = struct.unpack(order + "I", data[20:24])[0]
    pos = 24
    while pos + 16 <= len(data):
        sec, frac, incl, _orig = struct.unpack(order + "IIII", data[pos:pos + 16])
        pos += 16
        if incl > 0x40000 or pos + incl > len(data):
            break
        yield sec + frac * scale, link, data[pos:pos + incl]
        pos += incl


def _read_pcapng(data: bytes):
    order = "<"
    ifaces: list[tuple[int, float]] = []     # (linktype, seconds per tick)
    pos = 0
    while pos + 12 <= len(data):
        if data[pos:pos + 4] == PCAPNG_MAGIC:              # section header: it sets the byte order
            order = "<" if data[pos + 8:pos + 12] == PCAPNG_LE_BOM else ">"
            btype = 0x0A0D0D0A
            ifaces = []
        else:
            btype = struct.unpack(order + "I", data[pos:pos + 4])[0]
        length = struct.unpack(order + "I", data[pos + 4:pos + 8])[0]
        if length < 12 or pos + length > len(data):
            break
        body = data[pos + 8:pos + length - 4]
        if btype == 1 and len(body) >= 8:                       # interface description
            link = struct.unpack(order + "H", body[0:2])[0]
            tick, opt = 1e-6, 8
            while opt + 4 <= len(body):
                code, olen = struct.unpack(order + "HH", body[opt:opt + 4])
                if code == 0:
                    break
                if code == 9 and olen >= 1:
                    r = body[opt + 4]
                    tick = 2.0 ** -(r & 0x7F) if r & 0x80 else 10.0 ** -r
                opt += 4 + ((olen + 3) & ~3)
            ifaces.append((link, tick))
        elif btype == 6 and len(body) >= 20:                    # enhanced packet
            iid, hi, lo, cap, _orig = struct.unpack(order + "IIIII", body[:20])
            link, tick = ifaces[iid] if iid < len(ifaces) else (1, 1e-6)
            yield ((hi << 32) | lo) * tick, link, body[20:20 + cap]
        elif btype == 3 and len(body) >= 4 and ifaces:         # simple packet
            yield 0.0, ifaces[0][0], body[4:]
        pos += (length + 3) & ~3


# --------------------------------------------------------------------------
# packet decoding
# --------------------------------------------------------------------------
class Pkt:
    __slots__ = ("n", "ts", "length", "proto", "src", "dst", "sport", "dport", "flags", "payload",
                 "arp", "info")

    def __init__(self, n, ts, length):
        self.n, self.ts, self.length = n, ts, length
        self.proto = "OTHER"
        self.src = self.dst = ""
        self.sport = self.dport = 0
        self.flags = ""
        self.payload = b""
        self.arp = None
        self.info = ""


def _tcp_flags(f: int) -> str:
    names = (("FIN", 1), ("SYN", 2), ("RST", 4), ("PSH", 8), ("ACK", 16), ("URG", 32))
    return ",".join(n for n, bit in names if f & bit)


def _decode_ip(pk: Pkt, ip: bytes) -> None:
    if not ip:
        return
    ver = ip[0] >> 4
    if ver == 4 and len(ip) >= 20:
        ihl = (ip[0] & 0x0F) * 4
        proto = ip[9]
        pk.src, pk.dst = _ip4(ip[12:16]), _ip4(ip[16:20])
        frag = struct.unpack(">H", ip[6:8])[0] & 0x1FFF
        total = struct.unpack(">H", ip[2:4])[0]
        l4 = ip[ihl:total or None]
    elif ver == 6 and len(ip) >= 40:
        proto = ip[6]
        pk.src, pk.dst = _ip6(ip[8:24]), _ip6(ip[24:40])
        frag, l4 = 0, ip[40:]
    else:
        return
    if frag:
        pk.proto = "FRAG"
        return
    if proto == 6 and len(l4) >= 20:
        pk.proto = "TCP"
        pk.sport, pk.dport = struct.unpack(">HH", l4[:4])
        off = (l4[12] >> 4) * 4
        pk.flags = _tcp_flags(l4[13])
        pk.payload = l4[off:]
    elif proto == 17 and len(l4) >= 8:
        pk.proto = "UDP"
        pk.sport, pk.dport = struct.unpack(">HH", l4[:4])
        pk.payload = l4[8:]
    elif proto in (1, 58):
        pk.proto = "ICMP" if proto == 1 else "ICMPv6"
        if l4:
            pk.info = f"type {l4[0]}"
    else:
        pk.proto = f"IP/{proto}"


def decode_frame(n: int, ts: float, link: int, raw: bytes) -> Pkt:
    pk = Pkt(n, ts, len(raw))
    try:
        if link == 1 and len(raw) >= 14:
            et = struct.unpack(">H", raw[12:14])[0]
            off = 14
            while et in (0x8100, 0x88A8) and len(raw) >= off + 4:
                et = struct.unpack(">H", raw[off + 2:off + 4])[0]
                off += 4
            body = raw[off:]
        elif link == 113 and len(raw) >= 16:
            et, body = struct.unpack(">H", raw[14:16])[0], raw[16:]
        elif link in (101, 12, 14) and raw:
            et, body = (0x0800 if raw[0] >> 4 == 4 else 0x86DD), raw
        elif link == 0 and len(raw) >= 4:
            et, body = 0x0800, raw[4:]
            if body and body[0] >> 4 == 6:
                et = 0x86DD
        else:
            return pk
        if et == 0x0806 and len(body) >= 28:
            op = struct.unpack(">H", body[6:8])[0]
            pk.proto = "ARP"
            pk.arp = (op, _mac(body[8:14]), _ip4(body[14:18]), _ip4(body[24:28]))
            pk.src, pk.dst = pk.arp[2], pk.arp[3]
            pk.info = "who-has" if op == 1 else "is-at"
        elif et in (0x0800, 0x86DD):
            _decode_ip(pk, body)
    except (struct.error, ValueError, IndexError):
        pass
    return pk


# --- application layer ----------------------------------------------------
def _dns_name(buf: bytes, pos: int) -> tuple[str, int]:
    labels = []
    while pos < len(buf) and buf[pos]:
        ln = buf[pos]
        if ln & 0xC0:
            return ".".join(labels), pos + 2
        labels.append(buf[pos + 1:pos + 1 + ln].decode("ascii", "replace"))
        pos += 1 + ln
    return ".".join(labels), pos + 1


def parse_dns(payload: bytes):
    """(is_response, rcode, qname, qtype) or None."""
    if len(payload) < 12:
        return None
    flags = struct.unpack(">H", payload[2:4])[0]
    if struct.unpack(">H", payload[4:6])[0] < 1:
        return None
    name, pos = _dns_name(payload, 12)
    if not name or pos + 4 > len(payload):
        return None
    return bool(flags & 0x8000), flags & 0x0F, name.lower(), struct.unpack(">H", payload[pos:pos + 2])[0]


def parse_sni(payload: bytes) -> str:
    try:
        if payload[0] != 0x16 or payload[5] != 0x01:
            return ""
        pos = 43
        pos += 1 + payload[pos]                                       # session id
        pos += 2 + struct.unpack(">H", payload[pos:pos + 2])[0]        # cipher suites
        pos += 1 + payload[pos]                                       # compression
        end = pos + 2 + struct.unpack(">H", payload[pos:pos + 2])[0]
        pos += 2
        while pos + 4 <= end:
            etype, elen = struct.unpack(">HH", payload[pos:pos + 4])
            if etype == 0:
                nlen = struct.unpack(">H", payload[pos + 7:pos + 9])[0]
                return payload[pos + 9:pos + 9 + nlen].decode("ascii", "replace").lower()
            pos += 4 + elen
    except (IndexError, struct.error):
        pass
    return ""


_HTTP_RE = re.compile(rb"^(GET|POST|PUT|HEAD|DELETE|OPTIONS|PATCH) (\S{1,300}) HTTP/1\.[01]\r\n")


def parse_http(payload: bytes):
    """(method, path, host, basic_user) or None. The password is dropped."""
    m = _HTTP_RE.match(payload[:600])
    if not m:
        return None
    head = payload[:4000].split(b"\r\n\r\n", 1)[0]
    host = re.search(rb"(?im)^host:\s*([^\r\n]+)", head)
    auth = re.search(rb"(?im)^authorization:\s*basic\s+([A-Za-z0-9+/=]+)", head)
    user = ""
    if auth:
        try:
            user = base64.b64decode(auth[1]).decode("utf-8", "replace").split(":", 1)[0]
        except Exception:  # noqa: BLE001 - malformed credentials are simply not reported
            user = "(unreadable)"
    return (m[1].decode(), m[2].decode("ascii", "replace"),
            host[1].decode("ascii", "replace").strip() if host else "", user if auth else None)


def parse_cleartext_login(payload: bytes, port: int):
    """('FTP'|'POP3'|'IMAP'|'Telnet', user) for a user-name line, else None."""
    if port in (21, 110) and payload[:5].upper() == b"USER ":
        return ("FTP" if port == 21 else "POP3"), payload[5:60].split(b"\r")[0].decode("ascii", "replace").strip()
    if port == 143:
        m = re.match(rb"\S+ LOGIN (\S+) ", payload[:200], re.I)
        if m:
            return "IMAP", m[1].decode("ascii", "replace").strip('"')
    return None


# --------------------------------------------------------------------------
# Wireshark CSV
# --------------------------------------------------------------------------
_PORTS_RE = re.compile(r"(\d+)\s*(?:→|>|->)\s*(\d+)(?:\s*\[([^\]]+)\])?")
_CSV_ALIASES = {"no": ("no", "number", "frame"), "time": ("time", "frametime"),
                "src": ("source", "src", "ipsrc"), "dst": ("destination", "dst", "ipdst"),
                "proto": ("protocol", "proto"), "len": ("length", "len", "framelen"), "info": ("info",)}


def _csv_packets(text: str):
    lines = text.splitlines()
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    if not rows:
        raise LogFormatError("The CSV has no rows.")
    names = {L._alias(k): k for k in rows[0]}
    col = {}
    for key, aliases in _CSV_ALIASES.items():
        for a in aliases:
            if a in names:
                col[key] = names[a]
                break
    if not {"src", "dst", "proto"} <= set(col):
        raise LogFormatError("This CSV does not look like a Wireshark export (needs Source, Destination, Protocol).")
    relative = True
    pkts = []
    for i, row in enumerate(rows):
        if len(pkts) >= MAX_PACKETS:
            break
        raw_t = (row.get(col.get("time", ""), "") or "").strip()
        ts: float | None
        try:
            ts = float(raw_t)
        except ValueError:
            dt = L._parse_win_time(raw_t) or L._parse_iso(raw_t)
            ts = dt.replace(tzinfo=timezone.utc).timestamp() if dt else None
            relative = relative and ts is None
        pk = Pkt(i + 1, ts or 0.0, int(float(row.get(col.get("len", ""), 0) or 0)))
        pk.src, pk.dst = (row[col["src"]] or "").strip(), (row[col["dst"]] or "").strip()
        proto = (row[col["proto"]] or "OTHER").strip().upper()
        pk.proto = proto
        pk.info = (row.get(col.get("info", ""), "") or "")[:300]
        m = _PORTS_RE.search(pk.info)
        if m:
            pk.sport, pk.dport = int(m[1]), int(m[2])
            if m[3]:
                pk.flags = ",".join(x.strip() for x in m[3].split(","))
        pkts.append(pk)
    return pkts, relative


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def looks_like_capture(data: bytes) -> bool:
    return data[:4] in PCAP_MAGICS or data[:4] == PCAPNG_MAGIC


def analyze(data: bytes | str, name: str = "") -> Report:
    notes: list[str] = []
    relative = False
    if isinstance(data, bytes) and len(data) > L.MAX_BYTES:
        raise LogFormatError(f"The file is larger than {L.MAX_BYTES // 2**20} MB.")
    pkts: list[Pkt] = []
    if isinstance(data, bytes) and looks_like_capture(data):
        reader = _read_pcapng(data) if data[:4] == PCAPNG_MAGIC else _read_pcap(data)
        for n, (ts, link, raw) in enumerate(reader, 1):
            if n > MAX_PACKETS:
                notes.append(f"Only the first {MAX_PACKETS:,} packets were analysed.")
                break
            pkts.append(decode_frame(n, ts, link, raw))
    else:
        text = L.decode_bytes(data) if isinstance(data, bytes) else data
        if not text.strip():
            raise LogFormatError("The input is empty.")
        pkts, relative = _csv_packets(text)
        if relative:
            notes.append("The CSV has relative times, so the timeline shows seconds since the first packet.")
    if not pkts:
        raise LogFormatError("No packets were found in the capture.")
    return _analyze_packets(pkts, relative, notes)


def _analyze_packets(pkts: list[Pkt], relative: bool, notes: list[str]) -> Report:
    t0 = pkts[0].ts
    flows: dict[tuple, dict] = {}
    scans: dict[str, dict] = defaultdict(lambda: {"targets": set(), "hosts": set(), "syn": 0})
    syns: dict[tuple, list[float]] = defaultdict(list)
    dns_seen: dict[tuple, dict] = {}
    dns_by_domain: dict[str, set] = defaultdict(set)
    http_events, cred_events, sni_seen = [], [], {}
    arp_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    beacon: dict[tuple, list[float]] = defaultdict(list)
    protos: Counter = Counter()
    pkt_ts, pkt_len, pkt_proto = [], [], []

    for pk in pkts:
        protos[pk.proto] += 1
        pkt_ts.append(pk.ts - t0)
        pkt_len.append(pk.length)
        pkt_proto.append(pk.proto)
        if pk.proto == "ARP" and pk.arp:
            op, mac, sip, _tip = pk.arp
            if sip not in ("0.0.0.0",) and op in (1, 2):
                arp_map[sip][mac] += 1
            continue
        if not pk.src:
            continue
        key = (pk.proto, pk.src, pk.dst, pk.dport)
        f = flows.get(key)
        if f is None:
            f = flows[key] = {"n": pk.n, "first": pk.ts, "last": pk.ts, "pkts": 0, "bytes": 0, "syn": 0, "synack": 0}
        f["pkts"] += 1
        f["bytes"] += pk.length
        f["last"] = pk.ts
        flags = set(pk.flags.split(",")) if pk.flags else set()
        if pk.proto == "TCP":
            if "SYN" in flags and "ACK" not in flags:
                f["syn"] += 1
                s = scans[pk.src]
                s["syn"] += 1
                s["targets"].add((pk.dst, pk.dport))
                s["hosts"].add(pk.dst)
                syns[(pk.dst, pk.dport)].append(pk.ts)
                beacon[(pk.src, pk.dst, pk.dport)].append(pk.ts)
            elif "SYN" in flags and "ACK" in flags:
                f["synack"] += 1
        elif pk.proto in ("ICMP",) and pk.info.startswith("type 8"):
            scans[pk.src]["hosts"].add(pk.dst)
        # application layer (only when real payload bytes are present)
        pl = pk.payload
        if pl:
            if pk.proto == "UDP" and 53 in (pk.sport, pk.dport):
                d = parse_dns(pl)
                if d:
                    resp, rcode, qname, qtype = d
                    if not resp:
                        k = (pk.src, qname, qtype)
                        e = dns_seen.setdefault(k, {"n": pk.n, "ts": pk.ts, "count": 0, "rcode": 0})
                        e["count"] += 1
                        dns_by_domain[".".join(qname.split(".")[-2:])].add(qname)
            elif pk.proto == "TCP":
                h = parse_http(pl)
                if h:
                    http_events.append((pk, h))
                    if h[3] is not None:
                        cred_events.append((pk, "HTTP Basic", h[3]))
                c = parse_cleartext_login(pl, pk.dport)
                if c:
                    cred_events.append((pk, c[0], c[1]))
                sni = parse_sni(pl)
                if sni:
                    sni_seen.setdefault((pk.src, sni), pk)
        elif pk.proto == "DNS" and pk.info:                      # CSV rows
            m = re.search(r"Standard query (?:0x[0-9a-f]+ )?(\w+) (\S+)", pk.info)
            if m and "response" not in pk.info:
                k = (pk.src, m[2].lower(), m[1])
                e = dns_seen.setdefault(k, {"n": pk.n, "ts": pk.ts, "count": 0, "rcode": 0})
                e["count"] += 1
                dns_by_domain[".".join(m[2].lower().split(".")[-2:])].add(m[2].lower())
        elif pk.info and pk.proto in ("FTP", "TELNET", "HTTP"):   # CSV rows
            m = re.search(r"Request: USER (\S+)", pk.info)
            if m:
                cred_events.append((pk, "FTP", m[1]))
            m = re.search(r"^(GET|POST|PUT|DELETE|HEAD) (\S+) HTTP", pk.info)
            if m:
                http_events.append((pk, (m[1], m[2], "", None)))

    def when(ts):
        return None if relative else _utc(ts)

    events: list[Event] = []
    index_of: dict = {}

    def add(pk_n, ts, kind, sev, title, src="", dst="", port=0, proto="", size=0, count=1, user="", msg="", raw=""):
        events.append(Event(
            n=len(events), line=pk_n, ts=when(ts), os="network", host=dst, source=proto, event_id="",
            user=user, ip=src, message=msg[:2000], raw=(raw or msg)[:4000], kind=kind, severity=sev,
            title=title, dst=dst, port=port, proto=proto, size=size, count=count))
        return len(events) - 1

    # flows (largest first, capped)
    flow_items = sorted(flows.items(), key=lambda kv: -kv[1]["bytes"])[:MAX_FLOW_EVENTS]
    for (proto, src, dst, dport), f in flow_items:
        insecure = INSECURE.get(dport) if proto == "TCP" else None
        kind, sev, title = ("insecure_protocol", "low", f"{insecure} traffic (unencrypted)") if insecure else \
            ("flow", "info", f"{proto} {_service(dport, proto)}")
        index_of[(proto, src, dst, dport)] = add(
            f["n"], f["first"], kind, sev, title, src, dst, dport, proto, f["bytes"], f["pkts"],
            msg=f"{src} -> {dst}:{dport} {proto}  packets={f['pkts']} bytes={f['bytes']}")
    for (src, qname, qtype), e in list(dns_seen.items())[:MAX_DNS_EVENTS]:
        longlabel = max((len(x) for x in qname.split(".")), default=0)
        sev = "low" if longlabel >= DNS_LABEL_LONG or len(qname) >= DNS_NAME_LONG else "info"
        add(e["n"], e["ts"], "dns", sev, "DNS query", src, "", 53, "DNS", 0, e["count"],
            msg=f"{qname} (type {qtype}) x{e['count']}")
    http_idx = []
    for pk, (method, path, host, _u) in http_events[:2000]:
        http_idx.append(add(pk.n, pk.ts, "http", "info", "HTTP request", pk.src, pk.dst, pk.dport, "HTTP",
                            pk.length, 1, msg=f"{method} {host}{path}"))
    cred_idx = []
    for pk, how, user in cred_events[:200]:
        cred_idx.append(add(pk.n, pk.ts, "cleartext_credentials", "high", f"{how} login sent in clear text",
                            pk.src, pk.dst, pk.dport, how, pk.length, 1, user=user,
                            msg=f"user {user} to {pk.dst}:{pk.dport} (password not stored)"))
    for (src, sni), pk in list(sni_seen.items())[:1000]:
        add(pk.n, pk.ts, "tls", "info", "TLS connection", src, pk.dst, pk.dport, "TLS", pk.length, 1, msg=f"server name {sni}")

    findings: list[Finding] = []

    def finding(rule, title, sev, observed, ev_idx, hypothesis="", ips=(), span=None):
        meaning, attack = NET_INFO.get(rule, ("", ""))
        evs = [events[i] for i in ev_idx if i is not None]
        first = min((e.ts for e in evs if e.ts), default=None)
        last = max((e.ts for e in evs if e.ts), default=None)
        findings.append(Finding(
            id=f"F{len(findings) + 1:03d}", rule=rule, title=title, severity=sev, observed=observed,
            meaning=meaning, hypothesis=hypothesis, attack=attack,
            events=tuple(i for i in ev_idx if i is not None)[:500],
            ips=tuple(sorted(set(i for i in ips if i)))[:20], first=first, last=last))

    def flow_idx(pred):
        return [index_of[k] for k in index_of if pred(k)][:200]

    # 1. scans and sweeps
    for src, s in scans.items():
        ports = {p for _h, p in s["targets"]}
        if len(s["targets"]) >= SCAN_MIN_TARGETS and len(ports) >= 10:
            finding("port_scan", f"Port scan from {src}", "high",
                    f"{src} sent {s['syn']} connection requests (SYN) to {len(s['targets'])} different "
                    f"host/port pairs ({len(ports)} distinct ports).",
                    flow_idx(lambda k, src=src: k[1] == src), ips=[src],
                    hypothesis="" if is_private(src) else "")
        elif len(s["hosts"]) >= SWEEP_MIN_HOSTS:
            finding("host_sweep", f"Host sweep from {src}", "medium",
                    f"{src} contacted {len(s['hosts'])} different hosts.",
                    flow_idx(lambda k, src=src: k[1] == src), ips=[src])
    # 2. SYN flood
    for (dst, port), times in syns.items():
        times.sort()
        best, lo = 0, 0
        for hi in range(len(times)):
            while times[hi] - times[lo] > SYN_FLOOD_WINDOW:
                lo += 1
            best = max(best, hi - lo + 1)
        if best >= SYN_FLOOD_MIN:
            finding("syn_flood", f"SYN burst against {dst}:{port}", "high",
                    f"{best} connection requests to {dst}:{port} inside {SYN_FLOOD_WINDOW:g} seconds.",
                    flow_idx(lambda k, dst=dst, port=port: k[2] == dst and k[3] == port), ips=[dst])
    # 3. credentials
    if cred_idx:
        users = Counter(events[i].user for i in cred_idx)
        finding("cleartext_credentials", f"Logins sent in clear text ({len(cred_idx)})", "high",
                f"{len(cred_idx)} login(s) were visible in the packets. Accounts: "
                + ", ".join(f"{u} ({c})" for u, c in users.most_common(5)) + ". Passwords are not shown here.",
                cred_idx, ips=[events[i].ip for i in cred_idx] + [events[i].dst for i in cred_idx])
    # 4. DNS
    long_names = [(k, e) for k, e in dns_seen.items()
                  if max((len(x) for x in k[1].split(".")), default=0) >= DNS_LABEL_LONG or len(k[1]) >= DNS_NAME_LONG]
    if long_names:
        finding("dns_tunnel", f"Unusually long DNS names ({len(long_names)})", "medium",
                f"{len(long_names)} DNS queries have a label of {DNS_LABEL_LONG}+ characters or a name of "
                f"{DNS_NAME_LONG}+ characters, e.g. {long_names[0][0][1][:70]}...",
                [i for i, e in enumerate(events) if e.kind == "dns" and e.severity == "low"][:200],
                hypothesis="Can be DNS tunnelling, or a CDN/anti-spam name that is simply long. Check the domain.",
                ips=[k[0] for k, _ in long_names])
    for dom, names in dns_by_domain.items():
        if len(names) >= DNS_MANY_SUBDOMAINS:
            finding("dns_tunnel", f"{len(names)} unique names under {dom}", "medium",
                    f"{len(names)} different host names under {dom} were queried.",
                    [i for i, e in enumerate(events) if e.kind == "dns" and dom in e.message][:200],
                    hypothesis="Many one-off names under one domain can be data hidden in DNS, or a busy site. Verify the domain.")
    # 5. beaconing (a pattern across packets: always a hypothesis)
    for (src, dst, port), times in beacon.items():
        if len(times) >= BEACON_MIN_HITS:
            times.sort()
            gaps = [b - a for a, b in zip(times, times[1:])]
            mean = statistics.fmean(gaps)
            if mean >= BEACON_MIN_INTERVAL and statistics.pstdev(gaps) / mean <= BEACON_MAX_CV:
                finding("beaconing", f"Regular connections {src} to {dst}:{port}", "medium",
                        f"{len(times)} connections about every {mean:.1f} s (spread {statistics.pstdev(gaps) / mean * 100:.0f}%).",
                        flow_idx(lambda k, s=src, d=dst, p=port: k[1:] == (s, d, p)), ips=[src, dst],
                        hypothesis="A steady rhythm can be malware checking in, or a legitimate poller such as monitoring or sync. Confirm what runs on the source.")
    # 6. ARP
    for ip, macs in arp_map.items():
        if len(macs) >= 2:
            i = add(0, t0, "arp_conflict", "high", "IP claimed by several hardware addresses", ip, "", 0, "ARP", 0,
                    sum(macs.values()), msg=f"{ip} announced by " + ", ".join(sorted(macs)))
            finding("arp_conflict", f"{ip} claimed by {len(macs)} hardware addresses", "high",
                    f"ARP packets announced {ip} from: " + ", ".join(f"{m} ({c})" for m, c in macs.items()) + ".",
                    [i], hypothesis="Can be ARP spoofing, or a failover pair / virtual IP sharing one address. Check your network design.",
                    ips=[ip])
    # 7. large uploads to outside addresses
    out_bytes: Counter = Counter()
    for (proto, src, dst, _dp), f in flows.items():
        if is_private(src) and not is_private(dst) and dst:
            out_bytes[(src, dst)] += f["bytes"]
    for (src, dst), size in out_bytes.items():
        if size >= UPLOAD_BYTES:
            finding("large_upload", f"{size / 2**20:.1f} MB from {src} to {dst}", "medium",
                    f"{size:,} bytes were sent from {src} to the outside address {dst}.",
                    flow_idx(lambda k, s=src, d=dst: k[1] == s and k[2] == d), ips=[src, dst],
                    hypothesis="Could be data leaving the network, or a backup/upload that was expected. Confirm with the owner.")
    # 8. insecure protocols (one finding)
    insecure_idx = [i for i, e in enumerate(events) if e.kind == "insecure_protocol"]
    if insecure_idx:
        names = Counter(events[i].title.split(" ")[0] for i in insecure_idx)
        finding("insecure_protocol", "Unencrypted protocols in use", "low",
                "Seen: " + ", ".join(f"{k} ({v} flows)" for k, v in names.most_common(6)) + ".",
                insecure_idx[:200], ips=[events[i].ip for i in insecure_idx[:50]])

    findings.sort(key=lambda f: (-L.SEV_RANK[f.severity], f.first or datetime.max, f.id))
    end = pkts[-1].ts
    extra = {
        "relative": relative, "duration": end - t0, "packets": len(pkts), "bytes": sum(pkt_len),
        "flows": len(flows), "protocols": dict(protos.most_common()),
        "pkt_t": pkt_ts, "pkt_len": pkt_len, "pkt_proto": pkt_proto,
        "t0": None if relative else _utc(t0),
        "internal": sorted({p.src for p in pkts if p.src and is_private(p.src)})[:500],
    }
    return Report(os="network", events=events, findings=findings, total_lines=len(pkts), skipped=0,
                  truncated=len(pkts) >= MAX_PACKETS, notes=notes, extra=extra)


def graph(report: Report, max_nodes: int = 45) -> dict:
    """Hosts and who talks to whom. Every link is a flow seen in the capture;
    links to findings that carry a hypothesis are marked as such."""
    weight: Counter = Counter()
    links: dict[tuple, dict] = {}

    def link(a, b, kind, hyp=False, w=1):
        it = links.setdefault((a, b), {"source": a, "target": b, "count": 0, "kind": kind, "hypothesis": hyp})
        it["count"] += w
        weight[a] += w
        weight[b] += w

    for e in report.events:
        if e.kind in ("flow", "insecure_protocol", "http", "cleartext_credentials", "tls") and e.ip and e.dst:
            link(f"ip:{e.ip}", f"ip:{e.dst}", "same flow", w=max(1, min(e.count, 50)))
    nodes: dict[str, dict] = {}
    for f in report.findings[:15]:
        node = f"finding:{f.id}"
        hyp = bool(f.hypothesis)
        nodes[node] = {"id": node, "label": f"{f.id} {f.title}", "category": "finding", "severity": f.severity, "hypothesis": hyp}
        for ip in f.ips[:8]:
            link(node, f"ip:{ip}", "flagged by rule", hyp)
    keep = {n for n, _ in weight.most_common(max_nodes)} | set(nodes)
    out = []
    for node in keep:
        if node in nodes:
            out.append(nodes[node])
        else:
            ip = node.split(":", 1)[1]
            out.append({"id": node, "label": ip, "category": "ip" if is_private(ip) else "external", "weight": weight[node]})
    ls = [l for l in links.values() if l["source"] in keep and l["target"] in keep]
    return {"nodes": out, "links": ls, "total_entities": len(weight),
            "shown_entities": len([n for n in out if n["category"] != "finding"])}
