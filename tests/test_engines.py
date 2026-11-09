"""The rule engines: Linux/Windows logs and network captures."""
import random
import struct
import time
from datetime import datetime

import pytest

import capture_samples as X
import log_samples as F
import nashmi_logs as L
import nashmi_net as N

NOW = datetime(2026, 10, 22, 12, 0, 0)


def rules(report):
    return {f.rule for f in report.findings}


def test_linux_auth_findings():
    r = L.analyze(F.linux_auth(), now=NOW)
    assert r.os == "linux" and r.skipped == 1
    assert {"brute_force", "password_spray", "success_after_failures",
            "new_account_privileged", "root_login", "sudo_failed"} <= rules(r)
    # a pattern across events is a hypothesis; a plain count is not
    assert all(f.hypothesis for f in r.findings if f.rule == "success_after_failures")
    assert all(not f.hypothesis for f in r.findings if f.rule == "brute_force")


def test_usernames_with_spaces_still_count():
    log = "\n".join(f"Oct 22 03:10:{i:02d} h sshd[1]: Failed password for invalid user a b c "
                    f"from 203.0.113.9 port {i} ssh2" for i in range(6))
    r = L.analyze(log, now=NOW)
    assert "brute_force" in rules(r)
    assert any(e.user == "a b c" for e in r.events)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_windows_event_viewer_csv(encoding):
    r = L.analyze(F.win_csv().encode(encoding))
    assert r.os == "windows" and len(r.events) == 15
    assert {"brute_force", "log_cleared", "service_installed", "success_after_failures"} <= rules(r)
    assert any(e.kind == "suspicious_process" and e.severity == "high" for e in r.events)


def test_powershell_export_csv_header():
    csv = ('﻿"TimeCreated","Id","ProviderName","LevelDisplayName","MachineName","Message"\r\n'
           '"10/22/2026 8:10:46 PM","7045","Service Control Manager","Information","PC","A service was installed."\r\n')
    r = L.analyze(csv.encode("utf-8"))
    assert r.os == "windows" and r.events[0].event_id == "7045"


def test_windows_xml_and_json():
    assert any(f.rule == "log_cleared" for f in L.analyze(F.win_xml()).findings)
    ev = L.analyze(F.win_json()).events[0]
    assert (ev.user, ev.ip) == ("joe", "192.0.2.77")


@pytest.mark.parametrize("data, needle", [
    (F.HOSTILE_XML, "DOCTYPE"), (b"ElfFile\x00" + b"0" * 50, "evtx"),
    ("  \n", "empty"), ("hello world\nnot a log", "recognised")])
def test_bad_input_is_refused_cleanly(data, needle):
    with pytest.raises(L.LogFormatError, match=f"(?i){needle}"):
        L.analyze(data)


def test_evidence_graph_links_point_at_nodes():
    g = L.graph(L.analyze(F.linux_auth(), now=NOW))
    ids = {n["id"] for n in g["nodes"]}
    assert all(l["source"] in ids and l["target"] in ids for l in g["links"])


@pytest.mark.parametrize("writer", [X.write_pcap, X.write_pcapng,
                                    lambda p: X.write_pcap(p, big=True), lambda p: X.write_pcapng(p, big=True)])
def test_capture_detections(writer):
    r = N.analyze(writer(X.scenario()))
    assert {"port_scan", "beaconing", "cleartext_credentials", "dns_tunnel", "arp_conflict",
            "large_upload", "syn_flood", "insecure_protocol"} <= rules(r)


def test_capture_never_keeps_passwords():
    r = N.analyze(X.write_pcap(X.scenario()))
    text = " ".join([f.observed for f in r.findings] + [e.message + e.raw + e.user for e in r.events])
    assert "SuperSecret123" not in text and "hunter2" not in text
    assert "alice" in text and "bob" in text


CSV_HEAD = '"No.","Time","Source","Destination","Protocol","Length","Info"\n'


@pytest.mark.parametrize("time, length", [("0.1", "abc"), ("0.1", "1e999"), ("0.1", "nan"), ("0.1", ""),
                                          ("1e999", "60"), ("nan", "60"), ("-5", "60"), ("9" * 40, "60")])
def test_wireshark_csv_with_hostile_numbers(time, length):
    """Text, infinity or absurd values in the numeric columns are read as 0, never a crash."""
    row = f'"1","{time}","10.1.1.5","10.1.1.9","TCP","{length}","40000 → 80 [SYN] Seq=0"\n'
    r = N.analyze(CSV_HEAD + row)
    assert len(r.events) >= 1
    L.summary(r), N.graph(r), L.report_markdown(r, "x")


@pytest.mark.parametrize("ts", [0xFFFFFFFF_FFFFFFFF, 1 << 62, 0])
def test_capture_with_impossible_timestamps(ts):
    """A pcapng timestamp past the year 9999 is shown as the epoch instead of failing."""
    frame = X.scenario()[0][1]
    ng = X.write_pcapng([(0.0, frame)])
    body_at = ng.rindex(frame) - 20                       # the enhanced-packet block header
    ng = ng[:body_at + 4] + struct.pack("<II", ts >> 32, ts & 0xFFFFFFFF) + ng[body_at + 12:]
    r = N.analyze(ng)
    assert len(r.events) >= 1 and L.report_markdown(r, "x")


def test_unclosed_xml_events_are_read_in_linear_time():
    """50,000 opening tags with no closing tag used to take minutes (quadratic rescans)."""
    hostile = "<Events>" + "<Event>" * 50_000
    started = time.perf_counter()
    with pytest.raises(L.LogFormatError):
        L.analyze(hostile)
    assert time.perf_counter() - started < 5
    # and the real thing still parses, including an unclosed tail after valid events
    assert any(f.rule == "log_cleared" for f in L.analyze(F.win_xml() + "<Event>" * 1000).findings)


def test_engines_survive_mutated_files():
    """A small fixed-seed fuzz run: damaged files are refused or read, never an unhandled error."""
    rng = random.Random(2026)
    small = X.scenario()[:40]
    seeds = [(N.analyze, X.write_pcap(small)), (N.analyze, X.write_pcapng(small)), (N.analyze, X.CSV.encode()),
             (L.analyze, F.linux_auth().encode()), (L.analyze, F.win_csv().encode()),
             (L.analyze, F.win_xml().encode()), (L.analyze, F.win_json().encode())]
    for _ in range(400):
        engine, data = rng.choice(seeds)
        b = bytearray(data)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(b))
            kind = rng.randint(0, 3)
            if kind == 0:
                del b[i:]
            elif kind == 1:
                b[i:i + 4] = rng.choice((0, 0x7FFFFFFF, 0xFFFFFFFF)).to_bytes(4, "little")
            elif kind == 2:
                b[i:i] = rng.choice((b"\x00", b'"', b"<", b"1e999", b"9" * 300, b"\xff\xfe"))
            else:
                b[i] = rng.randrange(256)
            if not b:
                break
        try:
            report = engine(bytes(b))
        except L.LogFormatError:
            continue
        L.summary(report), L.report_markdown(report, "x")


def test_wireshark_csv_and_corrupt_files():
    assert "port_scan" in rules(N.analyze(X.CSV))
    for bad in (X.write_pcap(X.scenario())[:60], b"\x00\x01\x02" * 50, b"", "a,b\n1,2"):
        with pytest.raises(L.LogFormatError):
            N.analyze(bad)
