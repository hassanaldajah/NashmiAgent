"""Synthetic Linux and Windows logs for the tests (made up, never real data)."""
import json

def linux_auth() -> str:
    L = []
    for i in range(24):  # brute force from one IP, 6 distinct users
        u = ["root", "admin", "oracle", "test", "ubuntu", "git"][i % 6]
        L.append(f"Oct 22 03:{10 + i // 6:02d}:{i:02d} web01 sshd[{2000 + i}]: Failed password for "
                 f"{'invalid user ' if u not in ('root',) else ''}{u} from 203.0.113.50 port {40000 + i} ssh2")
    L.append("Oct 22 03:20:01 web01 sshd[2100]: Invalid user postgres from 203.0.113.50 port 41000")
    L.append("Oct 22 03:21:09 web01 sshd[2101]: Accepted password for deploy from 203.0.113.50 port 41002 ssh2")
    L.append("Oct 22 03:22:00 web01 sudo:   deploy : TTY=pts/0 ; PWD=/home/deploy ; USER=root ; COMMAND=/usr/bin/useradd -m backdoor")
    L.append("Oct 22 03:22:01 web01 useradd[2200]: new user: name=backdoor, UID=1002, GID=1002, home=/home/backdoor, shell=/bin/bash")
    L.append("Oct 22 03:22:30 web01 usermod[2210]: add 'backdoor' to group 'sudo'")
    L.append("Oct 22 03:23:00 web01 sshd[2300]: Accepted publickey for root from 198.51.100.7 port 5000 ssh2")
    L.append("Oct 22 03:24:00 web01 sudo: pam_unix(sudo:auth): authentication failure; logname=bob uid=1001 euid=0 tty=/dev/pts/1 ruser=bob rhost=  user=bob")
    L.append("Oct 22 03:25:00 web01 sshd[2400]: Did not receive identification string from 192.0.2.99 port 3333")
    L.append("this line is garbage and should be skipped")
    L.append("2024-05-01T10:11:12+0000 web01 sshd[9]: Failed password for root from 10.0.0.5 port 22 ssh2")
    return "\n".join(L)

def win_csv() -> str:
    hdr = "Level,Date and Time,Source,Event ID,Task Category,Message,Computer\n"
    rows = []
    def r(ts, eid, msg, src="Microsoft-Windows-Security-Auditing"):
        msg = msg.replace('"', '""')
        rows.append(f'Information,{ts},{src},{eid},Logon,"{msg}",DC01.corp.local')
    for i in range(8):
        r(f"10/22/2026 02:{i:02d}:11 AM", 4625,
          f"An account failed to log on.\r\nSubject:\r\n\tAccount Name:\t\t-\r\nAccount For Which Logon Failed:\r\n\tAccount Name:\t\tadmin{i % 2}\r\nLogon Type:\t\t3\r\nNetwork Information:\r\n\tSource Network Address:\t198.51.100.23")
    r("10/22/2026 02:30:00 AM", 4624,
      "An account was successfully logged on.\r\nSubject:\r\n\tAccount Name:\t\tDC01$\r\nLogon Type:\t\t10\r\nNew Logon:\r\n\tAccount Name:\t\tadmin0\r\nNetwork Information:\r\n\tSource Network Address:\t198.51.100.23")
    r("10/22/2026 02:31:00 AM", 4720, "A user account was created.\r\nNew Account:\r\n\tAccount Name:\t\tsvc_backup")
    r("10/22/2026 02:32:00 AM", 4732, "A member was added to a security-enabled local group.\r\nMember:\r\n\tSecurity ID: S-1\r\n\tAccount Name:\t\tsvc_backup\r\nGroup:\r\n\tGroup Name:\t\tAdministrators")
    r("10/22/2026 02:33:00 AM", 4688, "A new process has been created.\r\nSubject:\r\n\tAccount Name:\t\tadmin0\r\nProcess Information:\r\n\tNew Process Name:\tC:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\r\n\tProcess Command Line:\tpowershell.exe -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA ")
    r("10/22/2026 02:34:00 AM", 1102, "The audit log was cleared.", "Microsoft-Windows-Eventlog")
    r("10/22/2026 02:35:00 AM", 7045, "A service was installed in the system.\r\nService Name:  EvilSvc", "Service Control Manager")
    r("10/22/2026 02:36:00 AM", 9999, "Something unrelated", "Other")
    return hdr + "\r\n".join(rows) + "\r\n"

def win_xml() -> str:
    def ev(eid, data, t="2026-10-22T02:00:00.000Z"):
        d = "".join(f"<Data Name='{k}'>{v}</Data>" for k, v in data.items())
        return (f"<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System>"
                f"<Provider Name='Microsoft-Windows-Security-Auditing'/><EventID>{eid}</EventID>"
                f"<TimeCreated SystemTime='{t}'/><Computer>WS01</Computer></System><EventData>{d}</EventData></Event>")
    out = [ev(4625, {"TargetUserName": "bob", "IpAddress": "192.0.2.10", "LogonType": "3"}) for _ in range(6)]
    out.append(ev(4624, {"TargetUserName": "bob", "IpAddress": "192.0.2.10", "LogonType": "3"}, "2026-10-22T02:05:00.000Z"))
    out.append(ev(1102, {}, "2026-10-22T02:06:00.000Z"))
    return "\n".join(out)

def win_json() -> str:
    return json.dumps([{"TimeCreated": "2026-10-22T02:00:00Z", "Id": 4625, "ProviderName": "Security",
                        "MachineName": "WS02", "Message": "Account Name:\t\tjoe\r\nSource Network Address:\t192.0.2.77"}])

HOSTILE_XML = "<?xml version='1.0'?><!DOCTYPE x [<!ENTITY a 'aaaa'>]><Event><System><EventID>4625</EventID></System></Event>"
