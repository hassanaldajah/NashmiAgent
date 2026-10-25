"""The rule engines: Linux/Windows logs and network captures."""
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


def test_wireshark_csv_and_corrupt_files():
    assert "port_scan" in rules(N.analyze(X.CSV))
    for bad in (X.write_pcap(X.scenario())[:60], b"\x00\x01\x02" * 50, b"", "a,b\n1,2"):
        with pytest.raises(L.LogFormatError):
            N.analyze(bad)
