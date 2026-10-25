"""Synthetic network captures for the tests (made up, never real traffic)."""
import struct, socket, base64

def mac(n): return bytes([0x02,0,0,0,0,n])
def eth(payload, et=0x0800, src=1, dst=2): return mac(dst)+mac(src)+struct.pack(">H", et)+payload
def ip4(src, dst, proto, l4):
    tot = 20+len(l4)
    return struct.pack(">BBHHHBBH4s4s", 0x45,0,tot,0,0,64,proto,0,socket.inet_aton(src),socket.inet_aton(dst))+l4
def tcp(sp, dp, flags, payload=b""): return struct.pack(">HHIIBBHHH", sp,dp,0,0,5<<4,flags,8192,0,0)+payload
def udp(sp, dp, payload): return struct.pack(">HHHH", sp,dp,8+len(payload),0)+payload
def dns_q(name, qtype=1):
    q = b"".join(bytes([len(l)])+l.encode() for l in name.split("."))+b"\0"
    return struct.pack(">HHHHHH", 0x1234,0x0100,1,0,0,0)+q+struct.pack(">HH", qtype,1)
def arp(op, smac, sip, tip): return struct.pack(">HHBBH", 1,0x0800,6,4,op)+mac(smac)+socket.inet_aton(sip)+mac(0)+socket.inet_aton(tip)
SYN, ACK, PSH = 2, 16, 8

def scenario():
    """list of (ts, frame_bytes)"""
    P = []; t = 1_760_000_000.0
    for port in range(1, 41):                                      # port scan
        P.append((t, eth(ip4("10.0.0.5", "10.0.0.9", 6, tcp(40000+port, port, SYN))))); t += 0.05
    for i in range(10):                                            # beaconing: every 30 s
        P.append((t, eth(ip4("10.0.0.7", "203.0.113.9", 6, tcp(50000+i, 443, SYN))))); t += 30
    http = b"POST /login HTTP/1.1\r\nHost: shop.example.com\r\nAuthorization: Basic " + base64.b64encode(b"alice:SuperSecret123") + b"\r\n\r\n"
    P.append((t, eth(ip4("10.0.0.8", "10.0.0.20", 6, tcp(41000, 80, PSH|ACK, http))))); t += 1
    P.append((t, eth(ip4("10.0.0.8", "10.0.0.21", 6, tcp(41001, 21, PSH|ACK, b"USER bob\r\n"))))); t += 1
    P.append((t, eth(ip4("10.0.0.8", "10.0.0.21", 6, tcp(41001, 21, PSH|ACK, b"PASS hunter2\r\n"))))); t += 1
    longname = "a"*60+".evil-tunnel.example"                       # dns tunnelling hint
    P.append((t, eth(ip4("10.0.0.6", "10.0.0.2", 17, udp(53000, 53, dns_q(longname)))))); t += 1
    P.append((t, eth(ip4("10.0.0.6", "10.0.0.2", 17, udp(53001, 53, dns_q("www.example.com")))))); t += 1
    P.append((t, eth(arp(2, 0x11, "10.0.0.1", "10.0.0.6"), 0x0806, src=0x11))); t += 1   # ARP conflict
    P.append((t, eth(arp(2, 0x22, "10.0.0.1", "10.0.0.6"), 0x0806, src=0x22))); t += 1
    for i in range(700):                                            # large upload (~5.6 MB)
        P.append((t, eth(ip4("10.0.0.30", "198.51.100.4", 6, tcp(42000, 8443, ACK, b"x"*8000))))); t += 0.01
    for i in range(400):                                            # SYN burst
        P.append((t, eth(ip4(f"192.0.2.{i % 200 + 1}", "10.0.0.9", 6, tcp(1024+i, 8080, SYN))))); t += 0.005
    return P

def write_pcap(packets, big=False, nano=False):
    o = ">" if big else "<"; magic = {(False,False):0xa1b2c3d4,(False,True):0xa1b23c4d}[(False,nano)]
    out = struct.pack(o+"IHHiIII", magic,2,4,0,0,65535,1)
    for ts, fr in packets:
        sec = int(ts); frac = int(round((ts-sec)*(1e9 if nano else 1e6)))
        out += struct.pack(o+"IIII", sec, frac, len(fr), len(fr)) + fr
    return out

def write_pcapng(packets, big=False):
    o = ">" if big else "<"
    def blk(t, body):
        body += b"\0"*((-len(body)) % 4); n = 12+len(body)
        return struct.pack(o+"II", t, n)+body+struct.pack(o+"I", n)
    out = blk(0x0A0D0D0A, struct.pack(o+"IHHq", 0x1A2B3C4D, 1, 0, -1))
    out += blk(1, struct.pack(o+"HHI", 1, 0, 65535) + struct.pack(o+"HHB", 9, 1, 9)[:5] + b"\0\0\0" + struct.pack(o+"HH", 0, 0))
    for ts, fr in packets:
        v = int(round(ts*1e9)); out += blk(6, struct.pack(o+"IIIII", 0, v >> 32, v & 0xFFFFFFFF, len(fr), len(fr)) + fr)
    return out

CSV = '''"No.","Time","Source","Destination","Protocol","Length","Info"
''' + "\n".join(f'"{i}","{i*0.01:.6f}","10.1.1.5","10.1.1.9","TCP","60","{40000+i} → {i} [SYN] Seq=0 Win=8192 Len=0"' for i in range(1, 31)) + '''
"31","1.000000","10.1.1.8","10.1.1.21","FTP","70","Request: USER carol"
"32","1.100000","10.1.1.6","10.1.1.2","DNS","80","Standard query 0x1a2b A www.example.com"
"33","1.200000","10.1.1.6","10.1.1.2","DNS","80","Standard query response 0x1a2b A www.example.com A 93.184.216.34"
"34","1.300000","10.1.1.8","10.1.1.30","HTTP","120","GET /index.html HTTP/1.1"
'''
