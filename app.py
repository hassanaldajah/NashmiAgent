"""Nashmi Agent: a multi-provider cybersecurity assistant built with Streamlit.

Features
    * Providers: Gemini and Groq (free tiers), plus OpenAI (ChatGPT models)
      and Claude (Anthropic) with the user's own paid key.
    * Streaming replies for every provider.
    * Ten specialised cybersecurity tracks, each with its own chats and a
      search box.
    * File attachments and voice notes in the chat box (text, code, images,
      PDF, video, audio).
    * One-click copy under every answer.
    * API keys read from a ``.env`` file or Streamlit secrets, never shown
      in the browser and never written to the chat history.
    * Mobile-first layout for iOS and Android browsers.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import codecs
import csv
import hashlib
import hmac
import secrets
import html
import io
import ipaddress
import json
import logging
import mimetypes
import os
import random
import re
import tempfile
import time
import urllib.parse
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageDraw

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

try:
    import nashmi_db
except ImportError:  # accounts need sqlalchemy and cryptography
    nashmi_db = None
try:
    import nashmi_exam
except ImportError:  # the Practice Exam track needs this module
    nashmi_exam = None

load_dotenv()

logging.basicConfig(
    level=os.getenv("NASHMI_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("nashmi")
logging.getLogger("fontTools").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "Nashmi Agent"
HISTORY_FILE = Path(
    os.getenv("NASHMI_HISTORY_FILE")
    or Path(__file__).resolve().with_name("chat_history.json")
)
# Public demo mode (set NASHMI_PUBLIC_MODE=1 on the host): keys from .env or
# server secrets are never used, so every visitor must bring their own key,
# and chats are kept only in that visitor's session, never written to disk.
PUBLIC_MODE = os.getenv("NASHMI_PUBLIC_MODE", "0") == "1"


def is_public() -> bool:
    """True on a shared deployment.

    Set by NASHMI_PUBLIC_MODE, and switched on automatically as soon as
    sign-in is configured, so a forgotten setting can never hand the
    owner's API keys or a shared chat file to visitors.
    """
    if PUBLIC_MODE:
        return True
    try:
        return bool(st.secrets.get("auth"))
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        return False


def persistence_enabled() -> bool:
    """Chats and settings go to local files only on a private computer."""
    return not is_public() and os.getenv("NASHMI_PERSIST_HISTORY", "1") != "0"
USAGE_FILE = Path(
    os.getenv("NASHMI_USAGE_FILE")
    or Path(__file__).resolve().with_name("usage_counter.json")
)
SETTINGS_FILE = Path(
    os.getenv("NASHMI_SETTINGS_FILE")
    or Path(__file__).resolve().with_name("settings.json")
)
DETAIL_CHOICES = ("Concise", "Balanced", "Detailed")
LANGUAGE_CHOICES = ("Auto", "Arabic", "English")
DEFAULT_PREFS = {"detail": "Balanced", "language": "Auto"}
# Daily request limits we know for sure (the app also learns new ones from
# the provider's own "quota exceeded" message).
KNOWN_DAILY_LIMITS = {"gemini:gemini-3.8-flash": 20}

TEMPERATURE = 0.3
MAX_HISTORY_MESSAGES = 6
MAX_PROMPT_CHARS = 20_000
MAX_TEXT_CHARS = 15_000
MAX_MEDIA_BYTES = 18 * 1024 * 1024
REQUEST_TIMEOUT = (10, 180)  # (connect, read) seconds
MAX_RETRIES = 2
RETRY_BACKOFF = 1.0
RETRY_STATUS = {429, 500, 502, 503, 504}
NEW_CHAT_TITLE = "New Chat"
HERO_HEIGHT = 300

TEXT_EXTENSIONS = frozenset({
    ".txt", ".log", ".csv", ".json", ".py", ".js", ".ts", ".java", ".c",
    ".cpp", ".cs", ".go", ".rs", ".php", ".rb", ".sh", ".bat", ".ps1",
    ".sql", ".html", ".css", ".xml", ".yaml", ".yml", ".md", ".ini",
    ".conf", ".cfg", ".env",
})
IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
SECRET_LINE_RE = re.compile(
    r"(?im)^(\s*[A-Z0-9_.-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)"
    r"[A-Z0-9_.-]*\s*[=:]\s*).+$"
)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")


@dataclass(frozen=True)
class Provider:
    """Static description of an AI provider."""

    label: str
    id: str
    short: str
    kind: str  # "gemini", "anthropic" or "openai" (OpenAI-compatible)
    models: tuple[str, ...]
    key_url: str
    note: str
    env_keys: tuple[str, ...]
    base_url: str = ""
    media: str = "none"  # "multimodal", "image+pdf", "image" or "none"
    free: bool = True    # False: billed to the user's own account

    def accepts(self, mime_type: str) -> bool:
        """Return True if this provider can receive the given media type."""
        if self.media == "multimodal":
            return mime_type.startswith(("image/", "video/", "audio/")) or (
                mime_type == "application/pdf"
            )
        if self.media == "image+pdf":
            return mime_type in IMAGE_MIMES or mime_type == "application/pdf"
        if self.media == "image":
            return mime_type in IMAGE_MIMES
        return False


PROVIDER_LIST = (
    Provider(
        label="Gemini (Google) - Free",
        id="gemini",
        short="Gemini",
        kind="gemini",
        models=("gemini-3.8-flash",),
        key_url="https://aistudio.google.com/apikey",
        note="Free tier with per-minute and daily limits.",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        media="multimodal",
    ),
    Provider(
        label="Groq (GPT-OSS / Qwen) - Free",
        id="groq",
        short="Groq",
        kind="openai",
        models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ),
        key_url="https://console.groq.com/keys",
        note=(
            "Free tier with rate limits. GPT-OSS is OpenAI's open model, "
            "not ChatGPT itself."
        ),
        env_keys=("GROQ_API_KEY",),
        base_url="https://api.groq.com/openai/v1",
    ),
    Provider(
        label="OpenAI (ChatGPT models) - your own key",
        id="openai",
        short="OpenAI",
        kind="openai",
        models=("gpt-4o-mini", "gpt-4o"),
        key_url="https://platform.openai.com/api-keys",
        note=(
            "Billed to your own OpenAI account. The list shows the models "
            "your key can use; any other model name can be typed under "
            "Custom model."
        ),
        env_keys=("OPENAI_API_KEY",),
        base_url="https://api.openai.com/v1",
        media="image",
        free=False,
    ),
    Provider(
        label="Claude (Anthropic) - your own key",
        id="anthropic",
        short="Claude",
        kind="anthropic",
        models=("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5",
                "claude-fable-5-1"),
        key_url="https://console.anthropic.com/settings/keys",
        note=(
            "Billed to your own Anthropic account. The list shows the "
            "models your key can use."
        ),
        env_keys=("ANTHROPIC_API_KEY",),
        media="image+pdf",
        free=False,
    ),
)
PROVIDERS = {p.label: p for p in PROVIDER_LIST}
PROVIDERS_BY_ID = {p.id: p for p in PROVIDER_LIST}
DEFAULT_PROVIDER_ID = "groq"  # preselected; the others stay in the list
OPENROUTER_FALLBACK = ("openai/gpt-oss-20b:free",)

# ---------------------------------------------------------------------------
# Tracks and prompts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Track:
    """One specialised area of cybersecurity the user can work in."""

    name: str
    blurb: str
    placeholder: str
    focus: str = ""    # what the system prompt tells the model to focus on
    context: str = ""  # short topic list used to wrap each question
    actions: tuple[tuple[str, str], ...] = ()  # (button label, prompt)


MODE_CTF = "CTF"

# What a strong CTF player checks first in each category. The same notes feed
# the system prompt of the CTF track and the per-question expert prompt.
CTF_CATEGORIES = {
    "Web": (
        r"\b(web|http|https|cookie|jwt|session|sqli?|sql injection|xss|csrf|"
        r"ssrf|ssti|xxe|lfi|rfi|idor|graphql|php|flask|node|burp|login page|"
        r"robots\.txt|endpoint|api|url|parameter)\b",
        "view-source, robots.txt, headers and cookies; decode JWTs and "
        "sessions; test injection (SQLi, SSTI, command injection, XXE, "
        "SSRF, LFI/path traversal), auth and logic flaws (IDOR, mass "
        "assignment, race conditions), client side (XSS, CSRF, prototype "
        "pollution). Tools: Burp Suite, curl, ffuf, sqlmap.",
    ),
    "Crypto": (
        r"\b(crypto|cipher|encrypt\w*|decrypt\w*|rsa|aes|des|xor|hash|md5|"
        r"sha\d*|base64|modulus|exponent|ciphertext|plaintext|key|nonce|iv|"
        r"padding oracle|ecb|cbc|ctr|gcm|prng|lcg|caesar|vigenere|"
        r"diffie|elliptic|ecdsa|lattice)\b",
        "tell classical from modern crypto; single-byte and repeating-key "
        "XOR, frequency analysis; RSA flaws (small e, shared modulus, "
        "common factors, Wiener, Hastad, partial key); AES ECB/CBC "
        "(bit flipping, padding oracle, ECB cut-and-paste); hash length "
        "extension; weak PRNG/LCG. Tools: CyberChef, dCode, RsaCtfTool, "
        "SageMath, z3, pycryptodome.",
    ),
    "Pwn": (
        r"\b(pwn|binary exploitation|buffer overflow|stack|heap|rop|ret2\w+|"
        r"shellcode|format string|canary|aslr|pie|nx|got|plt|libc|gdb|"
        r"segfault|nc|netcat|remote service|checksec)\b",
        "run checksec and file; find the bug class (stack overflow, format "
        "string, heap, integer overflow); get the offset with cyclic and "
        "gdb (pwndbg/GEF); ret2win, ROP, ret2libc, shellcode, one_gadget. "
        "Give a pwntools template.",
    ),
    "Reverse": (
        r"\b(reverse|reversing|rev|disassembl\w*|decompil\w*|ghidra|ida|"
        r"radare2?|r2|crackme|keygen|obfuscat\w*|packed|upx|elf|pe file|"
        r"exe|apk|jar|pyc|dotnet|assembly|angr)\b",
        "file, strings, ltrace/strace first; open in Ghidra/IDA/radare2 and "
        "find the comparison that decides success; watch for anti-debug and "
        "packing (UPX); solve constraints with z3 or angr; use the right "
        "decompiler (jadx, dnSpy, pycdc/uncompyle6).",
    ),
    "Forensics": (
        r"\b(forensics?|pcap|pcapng|wireshark|tshark|memory dump|volatility|"
        r"disk image|autopsy|registry|evtx|event log|carv\w*|deleted file|"
        r"timeline|artifact|dmp|e01|raw dump|log file)\b",
        "file, exiftool, binwalk, foremost for unknown files; Wireshark "
        "(follow stream, export objects, filters) and tshark for pcaps; "
        "Volatility 3 for memory (pslist, netscan, cmdline, filescan); "
        "Autopsy/FTK for disks; event logs and registry for Windows.",
    ),
    "Stego": (
        r"\b(stego\w*|steghide|zsteg|lsb|hidden message|hidden in|image|"
        r"png|jpe?g|bmp|gif|wav|mp3|audio|spectrogram|sstv|dtmf|exif)\b",
        "exiftool, strings, binwalk and check for appended data; images: "
        "zsteg, StegSolve, steghide, per-channel and LSB planes, palette; "
        "audio: spectrogram in Audacity or Sonic Visualiser, DTMF, SSTV.",
    ),
    "OSINT": (
        r"\b(osint|geolocat\w*|where was|which city|username|social media|"
        r"twitter|instagram|linkedin|whois|dns|wayback|reverse image|"
        r"google dork\w*|coordinates|landmark|street view)\b",
        "reverse image search (Google Lens, Yandex, TinEye), EXIF and map "
        "clues, Wayback Machine, username and e-mail search, WHOIS/DNS "
        "history, Google dorks, Street View and Mapillary for locations.",
    ),
    "Misc": (
        r"\b(misc|encoding|base32|base58|base85|hex|binary|morse|"
        r"brainfuck|esolang|qr code|barcode|jail|pyjail|sandbox escape|"
        r"scripting|puzzle|trivia|ctf)\b",
        "identify the encoding layers (base64/32/58/85, hex, binary, morse, "
        "esoteric languages like Brainfuck); QR and barcodes; jail escapes "
        "(pyjail, restricted bash); automate repetitive steps with a short "
        "Python script.",
    ),
}

CTF_FOCUS = (
    "CTF (capture the flag) coaching across every category. Think like a "
    "seasoned CTF player: read the challenge title, description, points and "
    "hints for clues; classify the category; do the cheap checks first; "
    "rank hypotheses; give exact commands or a short script; check the "
    "flag format; say what to try next when stuck. Teach why each step "
    "works, so the player learns. When the player asks only for a hint, "
    "give hints and not the full solution. When a file, source code or tool "
    "output is needed, say exactly what to run and paste instead of "
    "guessing. Only help with challenge targets that belong to the CTF or "
    "the player's own lab. Category checklists: "
    + " ".join(f"[{name}] {notes}" for name, (_, notes)
               in CTF_CATEGORIES.items())
)

CTF_PROMPT_TEMPLATE = (
    "CTF challenge help. Detected category: {category}. Player's message:\n"
    "'''{user_text}'''\n\n"
    "Answer as a seasoned CTF player coaching a teammate:\n"
    "1. Restate what is given and the goal, and the flag format if known.\n"
    "2. Name the likely technique and why in a sentence or two. If the "
    "detected category looks wrong, say so and re-classify.\n"
    "3. Give a numbered plan, cheapest checks first, with the exact "
    "commands or a short script.\n"
    "4. Checklist for this category: {notes}\n"
    "5. If something needed is missing (file, source, service output), say "
    "precisely what to run or paste, and do not guess.\n"
    "6. Say what to try next if the first approach fails. If the player "
    "asked only for a hint, give hints only.\n"
    "Match the answer length to the question; for a greeting or an unclear "
    "message reply briefly and ask for the challenge."
)


def detect_ctf_category(text: str) -> str | None:
    """Best-guess CTF category from keywords, or None when nothing fits."""
    best, best_hits = None, 0
    for name, (pattern, _) in CTF_CATEGORIES.items():
        hits = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def build_ctf_prompt(user_text: str) -> str:
    """Rewrite a plain CTF question into a category-aware expert prompt."""
    category = detect_ctf_category(user_text)
    if category:
        notes = CTF_CATEGORIES[category][1]
        label = f"{category} (guessed from keywords)"
    else:
        notes = ("first identify the category from the title, files and "
                 "description, then use that category's usual first checks")
        label = "unknown, identify it first"
    return CTF_PROMPT_TEMPLATE.format(
        category=label, user_text=user_text, notes=notes
    )


MODE_CHAT = "Chat"
MODE_SOC = "SOC Analyzer"
MODE_TOOLS = "Toolbox"
MODE_EXAM = "Practice Exam"

TRACKS = (
    Track(
        name="Security Intelligence",
        blurb="Threat intel, OSINT, log analysis with Pandas and anomaly "
              "detection.",
        placeholder="Ask about threat intel, OSINT, Sigma/YARA, log analysis",
        focus=(
            "writing and explaining code for threat analysis, in whatever "
            "language fits the task (Threat Intel APIs, OSINT, log "
            "analysis, machine learning anomaly detection, YARA/Sigma "
            "rules)."
        ),
        context=(
            "Security Intelligence Labs: threat intel, OSINT, Pandas log "
            "analysis, anomaly detection, Sigma/YARA"
        ),
        actions=(
            (
                "Pandas Security Data Lab",
                "Explain and write Python code that uses Pandas to read a "
                "security dataset and detect anomalous behavior (Anomaly "
                "Detection) as part of a Security Intelligence lab.",
            ),
            (
                "Threat Intel (OSINT) Script",
                "Write Python code that pulls threat intelligence for "
                "suspicious IP addresses and domains and classifies them in "
                "a table.",
            ),
            (
                "AI Attack Detection Model",
                "Write simple Python code using scikit-learn to build a "
                "classifier that distinguishes normal traffic from attack "
                "traffic for a Security Intelligence lab, with an "
                "explanation of every step.",
            ),
        ),
    ),
    Track(
        name="CPSS",
        blurb="Cyber-physical systems security: SCADA, ICS, IoT and "
              "intrusion detection projects.",
        placeholder="Ask about SCADA, ICS, Modbus, MQTT, IoT security",
        focus=(
            "industrial and IoT security (SCADA, ICS, PLC, Modbus, MQTT, "
            "sensor spoofing, intrusion detection in cyber-physical "
            "systems). Suggest practical project ideas with simulation "
            "code in a language that suits them."
        ),
        context=(
            "Cyber-Physical Systems (CPSS) Lab: SCADA, Modbus, MQTT, PLC, "
            "IoT security"
        ),
        actions=(
            (
                "CPSS Project Ideas",
                "Suggest 3 strong, practical project ideas for a "
                "Cyber-Physical Security Systems (CPSS) course that can be "
                "implemented and simulated with Python, and explain each "
                "project idea.",
            ),
            (
                "Modbus / IoT Attack Simulation",
                "Write Python code that simulates an industrial system or "
                "IoT sensors (such as MQTT or Modbus) along with a script "
                "that detects sensor spoofing.",
            ),
            (
                "CPS Intrusion Detection (IDS)",
                "How do I build a simple Python monitoring system that "
                "watches physical sensor readings (temperature/pressure/"
                "power) and detects Replay Attacks or False Data "
                "Injection?",
            ),
        ),
    ),
    Track(
        name=MODE_CHAT,
        blurb="Ask anything, in Arabic or English. Cybersecurity and "
              "programming are my strong side.",
        placeholder="Message Nashmi",
    ),
    Track(
        name="Secure Coding & SQL",
        blurb="Vulnerable versus secure code, explained line by line.",
        placeholder="Ask about SQL injection, secure code, code review",
        focus=(
            "complete code explained line by line, comparing vulnerable "
            "code with secure code (SQL injection, input validation, "
            "authentication, common web vulnerabilities)."
        ),
        context="Secure Coding & SQL Lab: vulnerable versus secure code",
        actions=(
            (
                "SQL Injection Defense Code",
                "Explain how a SQL Injection vulnerability happens in code, "
                "and write Python and SQL code comparing a vulnerable "
                "query with a secure one.",
            ),
            (
                "Network Port Scanner Script",
                "Write an educational Python script that scans open ports "
                "on my local machine to verify service security, with an "
                "explanation of every line.",
            ),
            (
                "Brute Force Detection Script",
                "Write a Python script that reads log files and detects "
                "repeated failed login attempts.",
            ),
        ),
    ),
    Track(
        name="Penetration Testing",
        blurb="Authorized testing methodology, labs, CTFs and reporting.",
        placeholder="Ask about recon, web testing, labs, CTFs, reporting",
        focus=(
            "authorized penetration testing methodology: reconnaissance, "
            "enumeration, web and network testing in legal labs and CTFs, "
            "and professional reporting. Always stress scope, permission "
            "and remediation."
        ),
        context=(
            "Penetration Testing Lab: recon, enumeration, authorized "
            "testing, CTFs, reporting"
        ),
        actions=(
            (
                "Recon Methodology",
                "Walk me through a structured reconnaissance and "
                "enumeration methodology for an authorized penetration "
                "test of a small web application, with the tools used at "
                "each stage.",
            ),
            (
                "Practice Lab Setup",
                "How do I build a safe, isolated home lab with DVWA or "
                "Juice Shop to practice web exploitation legally? Give the "
                "exact setup steps.",
            ),
            (
                "Pentest Report Template",
                "Give me a professional penetration test report structure "
                "with one example finding written in full (description, "
                "impact, evidence, CVSS, remediation).",
            ),
        ),
    ),
    Track(
        name=MODE_CTF,
        blurb="CTF coach for every category: web, crypto, pwn, reverse, "
              "forensics, stego, OSINT and misc. Describe a challenge and "
              "get a step-by-step approach.",
        placeholder="Paste a CTF challenge, a file's output or ask how to "
                    "start (web, crypto, pwn, rev, forensics...)",
        focus=CTF_FOCUS,
        context="CTF challenge solving (web, crypto, pwn, reverse, "
                "forensics, stego, OSINT, misc)",
        actions=(
            (
                "Web: where do I start?",
                "I have a CTF web challenge with a login page. Walk me "
                "through the first checks and the most likely "
                "vulnerabilities, with the exact commands.",
            ),
            (
                "Crypto: identify the cipher",
                "I have a CTF crypto challenge with a ciphertext. How do I "
                "work out which cipher or flaw it uses, step by step?",
            ),
            (
                "Pwn: first steps on a binary",
                "I downloaded a CTF binary and there is a remote service. "
                "What do I run first, and how do I find and exploit a "
                "buffer overflow with pwntools?",
            ),
            (
                "Reverse: crack a crackme",
                "I have a CTF reverse engineering binary that asks for a "
                "password. How do I find the check in Ghidra and recover "
                "the flag, including with z3 or angr?",
            ),
            (
                "Forensics: analyse a pcap or image",
                "I have a CTF forensics file I know nothing about. Give me "
                "the triage workflow for unknown files, pcaps, memory "
                "dumps and disk images.",
            ),
            (
                "Stego and OSINT checklist",
                "Give me a checklist for CTF steganography on images and "
                "audio, and for OSINT geolocation challenges.",
            ),
        ),
    ),
    Track(
        name="Malware Analysis",
        blurb="Static and dynamic analysis, reverse engineering and YARA, "
              "from a defender's view.",
        placeholder="Ask about static analysis, sandboxes, YARA, reversing",
        focus=(
            "defensive malware analysis: static and dynamic analysis, "
            "sandboxing, unpacking, reverse engineering basics, YARA rules "
            "and writing detections. Never provide working malware."
        ),
        context=(
            "Malware Analysis Lab: static and dynamic analysis, sandboxes, "
            "reverse engineering, YARA"
        ),
        actions=(
            (
                "Static Analysis Workflow",
                "Explain a step-by-step static analysis workflow for a "
                "suspicious Windows executable (hashes, strings, PE "
                "headers, imports) and the tools for each step.",
            ),
            (
                "Safe Sandbox Setup",
                "How do I set up an isolated malware analysis lab with a "
                "VM, network isolation and snapshots, and which tools "
                "should I install?",
            ),
            (
                "Write a YARA Rule",
                "Explain how YARA rules work and write an example rule "
                "that detects a suspicious packed executable, line by "
                "line.",
            ),
        ),
    ),
    Track(
        name="Forensics & IR",
        blurb="Digital forensics and incident response: artifacts, "
              "timelines and playbooks.",
        placeholder="Ask about forensics, artifacts, timelines, IR playbooks",
        focus=(
            "digital forensics and incident response (DFIR): disk, memory "
            "and log forensics, Windows artifacts, timelines, evidence "
            "handling and NIST-style incident response playbooks."
        ),
        context=(
            "Digital Forensics & Incident Response Lab: artifacts, "
            "timelines, memory and disk forensics, playbooks"
        ),
        actions=(
            (
                "Ransomware IR Playbook",
                "Write a concise incident response playbook for a "
                "ransomware infection, following the NIST phases.",
            ),
            (
                "Windows Forensic Artifacts",
                "Which Windows artifacts (event logs, prefetch, registry, "
                "MFT) matter most in an investigation, and how do I read "
                "them?",
            ),
            (
                "Build an Event Timeline",
                "Show me how to use Python and Pandas to merge logs from "
                "several sources into a single investigation timeline.",
            ),
        ),
    ),
    Track(
        name="Network Security",
        blurb="Traffic analysis, IDS/IPS, firewalls and segmentation.",
        placeholder="Ask about Wireshark, Suricata, firewalls, protocols",
        focus=(
            "network security: packet and traffic analysis (Wireshark, "
            "Zeek), IDS/IPS rules (Suricata, Snort), firewalls, VPNs, "
            "segmentation and protocol weaknesses."
        ),
        context=(
            "Network Security Lab: traffic analysis, IDS/IPS, firewalls, "
            "segmentation"
        ),
        actions=(
            (
                "Wireshark Triage",
                "Teach me a practical Wireshark workflow to spot port "
                "scans, beaconing and data exfiltration in a capture.",
            ),
            (
                "Suricata Rule",
                "Explain Suricata rule syntax and write a rule that "
                "detects repeated SSH login attempts.",
            ),
            (
                "Segmentation Design",
                "Design a network segmentation plan for a small company "
                "with offices, servers and guest Wi-Fi, and explain the "
                "firewall rules.",
            ),
        ),
    ),
    Track(
        name="Cloud Security",
        blurb="IAM, misconfigurations, containers and Kubernetes hardening.",
        placeholder="Ask about AWS, Azure, IAM, containers, Kubernetes",
        focus=(
            "cloud security on AWS, Azure and GCP: IAM and least "
            "privilege, storage and network misconfigurations, logging, "
            "container and Kubernetes hardening."
        ),
        context=(
            "Cloud Security Lab: IAM, misconfigurations, containers, "
            "Kubernetes"
        ),
        actions=(
            (
                "IAM Least Privilege",
                "Explain least privilege in AWS IAM with a risky policy "
                "next to a hardened one.",
            ),
            (
                "Storage Misconfigurations",
                "What are the most common cloud storage misconfigurations "
                "that cause data leaks, and how do I audit and fix them?",
            ),
            (
                "Container Hardening",
                "Give a Docker and Kubernetes security hardening "
                "checklist with the reason for each item.",
            ),
        ),
    ),
    Track(
        name=MODE_SOC,
        blurb="Upload logs, screenshots or captures and get a SOC report "
              "you can export as PDF.",
        placeholder="",
        focus=(
            "SOC analysis: turning logs, screenshots and media into a "
            "professional incident report with IoCs, MITRE ATT&CK mapping "
            "and remediation."
        ),
        context="SOC Analyzer",
    ),
    Track(
        name=MODE_TOOLS,
        blurb="Look up CVEs in the official NVD, hash and decode data, and "
              "pull indicators out of any text. Everything can be exported.",
        placeholder="",
    ),
    Track(
        name=MODE_EXAM,
        blurb="Build a full exam from a subject or your own slides and PDFs: "
              "multiple choice, drag and drop, image labelling, short "
              "answers and more. Take it, get graded, export it.",
        placeholder="",
    ),
)
TRACKS_BY_NAME = {t.name: t for t in TRACKS}
MODES = tuple(t.name for t in TRACKS)

COMMON_REFS = (
    ("MITRE ATT&CK", "https://attack.mitre.org/"),
    ("NVD (National Vulnerability Database)", "https://nvd.nist.gov/"),
    ("CVE Program", "https://www.cve.org/"),
    ("CISA Known Exploited Vulnerabilities",
     "https://www.cisa.gov/known-exploited-vulnerabilities-catalog"),
)
TRACK_REFS = {
    "Security Intelligence": (
        ("MISP Project", "https://www.misp-project.org/"),
        ("AlienVault OTX", "https://otx.alienvault.com/"),
        ("abuse.ch", "https://abuse.ch/"),
        ("Sigma rules", "https://github.com/SigmaHQ/sigma"),
        ("YARA documentation", "https://yara.readthedocs.io/"),
    ),
    "CPSS": (
        ("NIST SP 800-82 Rev. 3 (OT security)",
         "https://csrc.nist.gov/pubs/sp/800/82/r3/final"),
        ("MITRE ATT&CK for ICS", "https://attack.mitre.org/matrices/ics/"),
        ("CISA Industrial Control Systems",
         "https://www.cisa.gov/topics/industrial-control-systems"),
        ("Modbus specifications", "https://modbus.org/specs.php"),
        ("MQTT specification", "https://mqtt.org/mqtt-specification/"),
    ),
    "Secure Coding & SQL": (
        ("OWASP Top 10", "https://owasp.org/www-project-top-ten/"),
        ("OWASP Cheat Sheet Series", "https://cheatsheetseries.owasp.org/"),
        ("OWASP ASVS",
         "https://owasp.org/www-project-application-security-verification-"
         "standard/"),
        ("CWE (Common Weakness Enumeration)", "https://cwe.mitre.org/"),
    ),
    "Penetration Testing": (
        ("OWASP Web Security Testing Guide",
         "https://owasp.org/www-project-web-security-testing-guide/"),
        ("NIST SP 800-115",
         "https://csrc.nist.gov/pubs/sp/800/115/final"),
        ("PortSwigger Web Security Academy",
         "https://portswigger.net/web-security"),
        ("Exploit-DB", "https://www.exploit-db.com/"),
    ),
    "Malware Analysis": (
        ("Malpedia", "https://malpedia.caad.fkie.fraunhofer.de/"),
        ("MalwareBazaar", "https://bazaar.abuse.ch/"),
        ("VirusTotal", "https://www.virustotal.com/"),
        ("Microsoft PE format",
         "https://learn.microsoft.com/en-us/windows/win32/debug/pe-format"),
        ("FLARE-VM", "https://github.com/mandiant/flare-vm"),
    ),
    "Forensics & IR": (
        ("NIST SP 800-61 Rev. 3 (incident response)",
         "https://csrc.nist.gov/pubs/sp/800/61/r3/final"),
        ("NIST SP 800-86 (forensics in IR)",
         "https://csrc.nist.gov/pubs/sp/800/86/final"),
        ("The DFIR Report", "https://thedfirreport.com/"),
        ("Volatility Foundation", "https://volatilityfoundation.org/"),
    ),
    "Network Security": (
        ("Wireshark documentation", "https://www.wireshark.org/docs/"),
        ("Suricata documentation", "https://docs.suricata.io/"),
        ("Zeek documentation", "https://docs.zeek.org/"),
        ("NIST SP 800-41 (firewalls)",
         "https://csrc.nist.gov/pubs/sp/800/41/r1/final"),
        ("RFC Editor", "https://www.rfc-editor.org/"),
    ),
    "Cloud Security": (
        ("CIS Benchmarks", "https://www.cisecurity.org/cis-benchmarks"),
        ("AWS Security Pillar",
         "https://docs.aws.amazon.com/wellarchitected/latest/"
         "security-pillar/welcome.html"),
        ("Microsoft Azure security",
         "https://learn.microsoft.com/en-us/azure/security/"),
        ("Cloud Controls Matrix (CSA)",
         "https://cloudsecurityalliance.org/research/cloud-controls-matrix"),
        ("Kubernetes security",
         "https://kubernetes.io/docs/concepts/security/"),
        ("NIST SP 800-190 (containers)",
         "https://csrc.nist.gov/pubs/sp/800/190/final"),
    ),
    MODE_CTF: (
        ("CTFtime", "https://ctftime.org/"),
        ("CTF101", "https://ctf101.org/"),
        ("picoCTF", "https://picoctf.org/"),
        ("pwn.college", "https://pwn.college/"),
        ("OverTheWire wargames", "https://overthewire.org/wargames/"),
        ("CyberChef", "https://gchq.github.io/CyberChef/"),
        ("GTFOBins", "https://gtfobins.github.io/"),
        ("PayloadsAllTheThings",
         "https://github.com/swisskyrepo/PayloadsAllTheThings"),
    ),
    MODE_SOC: (
        ("NIST SP 800-61 Rev. 3 (incident response)",
         "https://csrc.nist.gov/pubs/sp/800/61/r3/final"),
        ("Sigma rules", "https://github.com/SigmaHQ/sigma"),
        ("VirusTotal", "https://www.virustotal.com/"),
    ),
}

# Track names used by older versions, so saved chats keep working.
LEGACY_TRACK_NAMES = {
    "General Chat": MODE_CHAT,
    "Security Intelligence Labs": "Security Intelligence",
    "CPSS Projects (Cyber-Physical Systems)": "CPSS",
    "General Code Lab and SQL": "Secure Coding & SQL",
    "SOC Log Analyzer (File Upload)": MODE_SOC,
}

SYSTEM_PROMPT_TEMPLATE = """\
Your name is "Nashmi". You are a Jordanian cybersecurity expert and a \
specialized academic and technical assistant.
You speak in a professional, direct technical style with a friendly tone.
The track currently selected by the student is: {mode}
Focus of this track: {focus}

Language guidelines:
- Reply in the language used by the user.
- If the user writes in Arabic, reply in clear, professional Arabic with a \
friendly Jordanian touch, while keeping technical terms, tool names, \
commands, and code blocks in English.
- If the user writes in English, reply in English.

Response length:
- Match the length of the answer to the question. Never dump long code \
or full projects unless the user clearly asks for code or a full solution.
- If the message is a greeting, very short, or unclear, reply briefly and \
ask what they need.

Code and languages:
- Write code in the language the user asks for, or the one that best fits \
the task (Python, Bash, PowerShell, SQL, C, JavaScript, YARA, Sigma, ...). \
Never default to Python when the user names another language or another \
one is the better tool.
- Put every piece of code in a fenced block with its language tag, so it \
can be copied or saved as a file.
- For attack chains, network layouts or flows you may draw a diagram with a \
```mermaid block.

Expert standards:
- Behave like a senior practitioner. Accuracy comes first: separate facts \
from assumptions, and say plainly when you are unsure or when something \
depends on versions or dates.
- Prefer primary sources: MITRE ATT&CK (technique IDs such as T1059), NIST \
(SP 800-61, 800-53, CSF), OWASP (Top 10, ASVS, Cheat Sheets), CIS \
Benchmarks, CWE/CVE/NVD, CISA advisories, RFCs and vendor documentation.
- End substantial technical answers with a short "References" list of real \
sources (title, publisher and identifier). Add a URL only when it is a \
well-known stable address. Never invent CVE numbers, URLs, paper titles, \
tool options or quotes; if you cannot name a real source, say so and tell \
the user how to verify.

Safety:
- Assume educational, defensive or authorized use (labs, CTFs, your own \
systems). Stress scope and permission where it matters.
- Do not write ready-to-deploy malware or help attack systems the user does \
not own or have permission to test."""

CHAT_SYSTEM_PROMPT = """\
Your name is "Nashmi". You are a friendly, smart assistant who can talk \
about any topic, and you are especially strong in cybersecurity and \
programming.
- Reply in the language the user writes in. For Arabic, use clear Arabic \
with a friendly Jordanian touch, and keep technical terms, commands and \
code in English.
- Match the length of the answer to the question. For greetings or short \
messages, reply briefly. Give code or long explanations only when asked.
- Write code in the language the user asks for or the one that fits the \
task, not always Python. Use fenced code blocks with a language tag.
- Be honest when you do not know something."""

EXPERT_TEMPLATE = (
    "Context: {context}. The student wrote: '{user_text}'. {style}"
)
RESPONSE_STYLE = (
    "Answer exactly what was asked, at a length that matches the message. "
    "For greetings, short, unclear or off-topic messages, reply in one or "
    "two short sentences and ask what they need. Only give code, long "
    "explanations or full solutions when the student clearly asks for them. "
    "Use the programming language that fits the request, not always "
    "Python. For substantial technical answers end with a short References "
    "list of real primary sources."
)

SOC_PROMPT_TEMPLATE = """\
Analyze all attached security files, logs, screenshots, or videos and \
produce a professional SOC report that includes:
1. Executive summary and severity level.
2. Event/media analysis and timeline.
3. Indicators of Compromise (IoCs): IPs: {ips}, Hashes: {hashes}
4. Related MITRE ATT&CK tactics.
5. Containment and defensive remediation steps.

Additional Analyst Notes: {notes}"""


def build_system_prompt(mode: str) -> str:
    """Return the system prompt for the selected track."""
    if mode == MODE_CHAT:
        return CHAT_SYSTEM_PROMPT
    track = TRACKS_BY_NAME.get(mode)
    return SYSTEM_PROMPT_TEMPLATE.format(
        mode=mode, focus=track.focus if track else "cybersecurity."
    )


def wrap_prompt_as_security_expert(user_text: str, mode: str) -> str:
    """Turn a plain question into an expert prompt for the chosen track."""
    if mode == MODE_CHAT:
        return user_text  # normal chat: send the message as written
    if mode == MODE_CTF:
        return build_ctf_prompt(user_text)
    track = TRACKS_BY_NAME.get(mode)
    return EXPERT_TEMPLATE.format(
        context=track.context if track and track.context else mode,
        user_text=user_text,
        style=RESPONSE_STYLE,
    )


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def get_secret(*names: str) -> str:
    """Look up a secret in the environment (.env) and then Streamlit secrets.

    The value is only used server-side; it is never rendered in the page.
    """
    if is_public():
        return ""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    try:
        for name in names:
            value = str(st.secrets.get(name, "")).strip()
            if value:
                return value
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        LOGGER.debug("No Streamlit secrets available")
    return ""


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MediaPart:
    """Binary attachment that will be sent to the model."""

    name: str
    mime_type: str
    data: bytes


@dataclass
class PreparedFiles:
    """Attachments sorted into inline text, media and skipped files."""

    text: str = ""
    raw_text: str = ""
    media: list[MediaPart] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def is_env_file(name: str) -> bool:
    """True for dotenv files such as ``.env``, ``.env.local`` or ``a.env``."""
    lowered = name.lower()
    return lowered.startswith(".env") or lowered.endswith(".env")


MAX_REDACT_LINE = 2000


def redact_secrets(text: str) -> str:
    """Mask values of KEY/TOKEN/SECRET/PASSWORD lines in env-style files.

    The pattern is only run on short lines: on one enormous line it could
    backtrack for minutes and hang the server.
    """
    return "\n".join(
        SECRET_LINE_RE.sub(r"\1[REDACTED]", line)
        if len(line) <= MAX_REDACT_LINE else "[line too long, removed]"
        for line in text.split("\n")
    )


def prepare_attachments(uploaded_files, provider: Provider) -> PreparedFiles:
    """Read uploaded files and split them into text and supported media."""
    prepared = PreparedFiles()
    if not uploaded_files:
        return prepared

    text_blocks: list[str] = []
    raw_blocks: list[str] = []
    media_bytes = 0

    for upload in uploaded_files:
        name = upload.name
        ext = os.path.splitext(name)[1].lower()
        mime = (
            upload.type
            or mimetypes.guess_type(name)[0]
            or "application/octet-stream"
        )
        try:
            data = upload.getvalue()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", name, exc)
            prepared.skipped.append(f"{name} (could not be read)")
            continue

        env_file = is_env_file(name)
        if env_file or ext in TEXT_EXTENSIONS or mime.startswith("text/"):
            decoded = data.decode("utf-8", errors="ignore")
            if env_file:
                decoded = redact_secrets(decoded)
            raw_blocks.append(decoded)
            clipped = decoded[:MAX_TEXT_CHARS]
            if len(decoded) > MAX_TEXT_CHARS:
                clipped += "\n[... truncated ...]"
            text_blocks.append(
                f"\n--- Attached File: {name} ---\n```\n{clipped}\n```\n"
            )
        elif not provider.accepts(mime):
            prepared.skipped.append(
                f"{name} (not supported by {provider.short})"
            )
        elif media_bytes + len(data) > MAX_MEDIA_BYTES:
            limit_mb = MAX_MEDIA_BYTES // (1024 * 1024)
            prepared.skipped.append(f"{name} (over the {limit_mb} MB limit)")
        else:
            media_bytes += len(data)
            prepared.media.append(MediaPart(name, mime, data))

    prepared.text = "\n".join(text_blocks)
    prepared.raw_text = "\n".join(raw_blocks)
    return prepared


# ---------------------------------------------------------------------------
# Provider layer (streaming)
# ---------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """Raised with a user-friendly message when a provider call fails."""


HTTP_HINTS = {
    400: "the request was rejected (check the model name and attachments)",
    401: "the API key is missing or invalid",
    403: "access denied (the key may lack permission for this model)",
    404: "model or endpoint not found (check the model name)",
    413: "the request is too large (try fewer or smaller files)",
    429: "rate limit or quota reached (wait a bit and try again)",
}


def describe_http_error(name: str, status: int | None, body: str) -> str:
    """Build a short, readable error message from an HTTP failure."""
    detail = ""
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else data
        detail = err.get("message", "") if isinstance(err, dict) else str(err)
    except (ValueError, AttributeError):
        detail = (body or "").strip()
    hint = HTTP_HINTS.get(status or 0)
    if hint is None:
        hint = (
            "the provider had a temporary problem"
            if status and status >= 500
            else "unexpected response"
        )
    message = f"{name} API error {status}: {hint}."
    if detail:
        message += f" Details: {detail[:700]}"
    return message


def show_internal_error(prefix: str, exc: Exception) -> None:
    """Technical detail only on a private computer; visitors get a calm note."""
    if is_public():
        st.error(f"{prefix} Please try again.")
    else:
        st.error(f"{prefix} {exc}")


def show_provider_error(exc: Exception) -> None:
    """Show provider errors; a quota/rate limit gets a short calm notice."""
    text = str(exc)
    if " API error 429" in text:
        st.info(
            "وصلت حد الاستخدام المجاني لهذا الموديل. جرّب موديل أو مزود "
            "ثاني، أو انتظر شوي.\n\n"
            "Free limit reached. Wait a bit or switch model/provider."
        )
    else:
        st.error(text)


def prepare_history(messages: list[dict], limit: int = MAX_HISTORY_MESSAGES):
    """Keep the last messages and make sure the list starts with the user."""
    trimmed = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ][-limit:]
    while trimmed and trimmed[0]["role"] != "user":
        trimmed.pop(0)
    return trimmed


def _with_attachment_text(
    msgs: list[dict], files: PreparedFiles
) -> list[dict]:
    """Append the inline text of attachments to the last user message."""
    if files.text and msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += f"\n\n{files.text}"
    return msgs


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _open_stream(provider: Provider, url: str, headers: dict, payload: dict):
    """POST a streaming request, retrying briefly on transient failures."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, headers=headers, json=payload, stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.Timeout as exc:
            raise ProviderError(
                f"{provider.short} did not answer in time. Try again."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                f"Could not reach {provider.short}. Check your connection."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Network error: {exc}") from exc

        if resp.status_code == 200:
            try:
                headers = resp.headers
                info = {
                    "remaining_requests": headers.get(
                        "x-ratelimit-remaining-requests"),
                    "limit_requests": headers.get("x-ratelimit-limit-requests"),
                    "remaining_tokens": headers.get(
                        "x-ratelimit-remaining-tokens"),
                }
                if any(info.values()):
                    st.session_state.setdefault("rate_info", {})[
                        f"{provider.id}:{payload.get('model')}"
                    ] = info
            except Exception:  # noqa: BLE001 - never break a reply for this
                LOGGER.debug("No rate-limit headers available")
            return resp
        if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
            resp.close()
            time.sleep(RETRY_BACKOFF * 2 ** attempt)
            continue
        body = resp.text
        resp.close()
        raise ProviderError(
            describe_http_error(provider.short, resp.status_code, body)
        )
    raise ProviderError(f"{provider.short} is unavailable right now.")


def _iter_sse(resp) -> Iterator[dict]:
    """Yield the JSON payload of each server-sent event."""
    resp.encoding = "utf-8"
    with resp:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed stream chunk")


def _stream_gemini(provider, api_key, model, system_prompt, msgs, files):
    client = genai.Client(api_key=api_key)
    last = len(msgs) - 1
    contents = []
    for index, msg in enumerate(msgs):
        role = "model" if msg["role"] == "assistant" else "user"
        parts = [types.Part(text=msg["content"])]
        if index == last and role == "user":
            parts.extend(
                types.Part.from_bytes(data=m.data, mime_type=m.mime_type)
                for m in files.media
            )
        contents.append(types.Content(role=role, parts=parts))

    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=TEMPERATURE
            ),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
    except genai_errors.APIError as exc:
        message = getattr(exc, "message", None) or str(exc)
        raise ProviderError(
            describe_http_error(provider.short, getattr(exc, "code", None),
                                message)
        ) from exc


def _stream_anthropic(provider, api_key, model, system_prompt, msgs, files):
    messages = [dict(m) for m in msgs]
    if files.media and messages and messages[-1]["role"] == "user":
        blocks = []
        for item in files.media:
            is_pdf = item.mime_type == "application/pdf"
            kind = "document" if is_pdf else "image"
            blocks.append({
                "type": kind,
                "source": {
                    "type": "base64",
                    "media_type": item.mime_type,
                    "data": _b64(item.data),
                },
            })
        blocks.append({"type": "text", "text": messages[-1]["content"]})
        messages[-1]["content"] = blocks

    resp = _open_stream(
        provider,
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        {
            "model": model,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": messages,
            "temperature": TEMPERATURE,
            "stream": True,
        },
    )
    for event in _iter_sse(resp):
        kind = event.get("type")
        if kind == "content_block_delta":
            text = event.get("delta", {}).get("text")
            if text:
                yield text
        elif kind == "error":
            detail = event.get("error", {}).get("message", "unknown error")
            raise ProviderError(f"{provider.short} stream error: {detail}")


def _stream_openai_compatible(provider, api_key, model, system_prompt, msgs,
                              files):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(dict(m) for m in msgs)
    if files.media and messages[-1]["role"] == "user":
        content = [{"type": "text", "text": messages[-1]["content"]}]
        for item in files.media:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{item.mime_type};base64,{_b64(item.data)}"
                },
            })
        messages[-1]["content"] = content

    # OpenAI's newer models only accept their default temperature.
    sampling = {} if provider.id == "openai" else {"temperature": TEMPERATURE}
    resp = _open_stream(
        provider,
        f"{provider.base_url}/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        {
            "model": model,
            "messages": messages,
            "stream": True,
            **sampling,
        },
    )
    thinking = ""
    answered = False
    for event in _iter_sse(resp):
        if "error" in event:
            err = event["error"]
            detail = err.get("message") if isinstance(err, dict) else err
            raise ProviderError(f"{provider.short} stream error: {detail}")
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                answered = True
                yield text
            elif delta.get("reasoning_content"):
                thinking += delta["reasoning_content"]
    if not answered and thinking.strip():
        yield thinking  # the model only produced its reasoning text


STREAMERS = {
    "gemini": _stream_gemini,
    "anthropic": _stream_anthropic,
    "openai": _stream_openai_compatible,
}


def generate_stream(
    provider: Provider,
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list[dict],
    files: PreparedFiles | None = None,
) -> Iterator[str]:
    """Stream the reply text for the conversation from any provider."""
    files = files or PreparedFiles()
    msgs = _with_attachment_text(prepare_history(messages), files)
    if not msgs:
        raise ProviderError("There is no message to send.")
    streamer = STREAMERS[provider.kind]
    counted = False
    for chunk in streamer(provider, api_key, model, system_prompt, msgs, files):
        if not counted:
            counted = True
            record_usage(provider, model)
        yield chunk


@st.cache_data(ttl=3600, show_spinner=False)
def openrouter_free_models() -> list[str]:
    """Live list of OpenRouter models that cost nothing (ids ending :free)."""
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        resp.raise_for_status()
        return sorted(
            m["id"] for m in resp.json()["data"] if m["id"].endswith(":free")
        )
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        LOGGER.warning("Could not load OpenRouter models: %s", exc)
        return []


@st.cache_data(ttl=600, show_spinner=False)
def live_models(provider_id: str, api_key: str) -> list[str]:
    """Ask the provider which chat models this key can really use."""
    if not api_key:
        return []
    try:
        if provider_id == "gemini":
            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
                params={"pageSize": 100},
                timeout=15,
            )
            resp.raise_for_status()
            skip = ("image", "tts", "live", "audio", "embedding")
            found = []
            for m in resp.json().get("models", []):
                name = m.get("name", "").removeprefix("models/")
                ok = "generateContent" in m.get(
                    "supportedGenerationMethods", []
                )
                if ok and "flash" in name and not any(x in name for x in skip):
                    found.append(name)
            return sorted(found, reverse=True)
        if provider_id == "groq":
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            skip = ("whisper", "guard", "orpheus", "tts", "playai")
            return sorted(
                m["id"] for m in resp.json().get("data", [])
                if m.get("active", True)
                and not any(x in m["id"] for x in skip)
            )
        if provider_id == "openai":
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            keep = ("gpt-", "chatgpt-", "o1", "o3", "o4")
            skip = ("audio", "realtime", "transcribe", "tts", "image",
                    "embedding", "moderation", "search", "instruct",
                    "whisper", "dall", "codex", "computer", "preview-")
            return sorted(
                (m["id"] for m in resp.json().get("data", [])
                 if m["id"].startswith(keep)
                 and not any(x in m["id"] for x in skip)),
                reverse=True,
            )
        if provider_id == "anthropic":
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                params={"limit": 100},
                timeout=15,
            )
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])
                    if str(m.get("id", "")).startswith("claude")]
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        LOGGER.warning("Could not load %s models: %s", provider_id, exc)
    return []


# ---------------------------------------------------------------------------
# VirusTotal file scanning (free API: 4 requests/min, 500/day)
# ---------------------------------------------------------------------------
VT_API = "https://www.virustotal.com/api/v3"
VT_MAX_UPLOAD = 32 * 1024 * 1024


def _vt_verdict(stats: dict, results: dict) -> dict:
    malicious = int(stats.get("malicious", 0))
    suspicious = int(stats.get("suspicious", 0))
    total = sum(int(v) for k, v in stats.items() if k != "type-unsupported")
    hits = sorted(
        f"{engine}: {r.get('result') or r.get('category')}"
        for engine, r in (results or {}).items()
        if r.get("category") in ("malicious", "suspicious")
    )
    return {
        "malicious": malicious, "suspicious": suspicious,
        "total": total, "detections": hits,
    }


def vt_scan_file(api_key: str, name: str, data: bytes, allow_upload: bool) -> dict:
    """Check one file on VirusTotal by SHA-256 (and upload it only if allowed)."""
    sha = hashlib.sha256(data).hexdigest()
    base = {
        "name": name, "sha256": sha,
        "link": f"https://www.virustotal.com/gui/file/{sha}",
    }
    headers = {"x-apikey": api_key}
    try:
        resp = requests.get(f"{VT_API}/files/{sha}", headers=headers, timeout=30)
        if resp.status_code == 200:
            attrs = resp.json()["data"]["attributes"]
            return {**base, "status": "found", **_vt_verdict(
                attrs.get("last_analysis_stats", {}),
                attrs.get("last_analysis_results", {}),
            )}
        if resp.status_code == 401:
            return {**base, "status": "error", "error": "VirusTotal key is invalid."}
        if resp.status_code == 429:
            return {**base, "status": "error",
                    "error": "VirusTotal limit reached (4 per minute). Wait a minute."}
        if resp.status_code != 404:
            return {**base, "status": "error",
                    "error": f"VirusTotal answered {resp.status_code}."}
        if not allow_upload:
            return {**base, "status": "unknown"}
        if len(data) > VT_MAX_UPLOAD:
            return {**base, "status": "error",
                    "error": "File is bigger than 32 MB, cannot upload."}
        up = requests.post(
            f"{VT_API}/files", headers=headers,
            files={"file": (name, data)}, timeout=120,
        )
        up.raise_for_status()
        analysis_id = up.json()["data"]["id"]
        for _ in range(12):
            time.sleep(5)
            an = requests.get(
                f"{VT_API}/analyses/{analysis_id}", headers=headers, timeout=30
            )
            an.raise_for_status()
            attrs = an.json()["data"]["attributes"]
            if attrs.get("status") == "completed":
                return {**base, "status": "found", **_vt_verdict(
                    attrs.get("stats", {}), attrs.get("results", {})
                )}
        return {**base, "status": "pending"}
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        LOGGER.warning("VirusTotal scan failed: %s", exc)
        return {**base, "status": "error", "error": "Could not reach VirusTotal."}


def show_vt_result(res: dict) -> None:
    """Draw one VirusTotal result with a simple colour verdict."""
    head = f"**{res['name']}**  \nSHA-256: `{res['sha256']}`"
    status = res["status"]
    if status == "found":
        score = f"{res['malicious']}/{res['total']} engines flagged it"
        if res["malicious"] > 0:
            st.error(f"{head}  \nMalicious: {score}")
        elif res["suspicious"] > 0:
            st.warning(f"{head}  \nSuspicious: {res['suspicious']} engines")
        else:
            st.success(f"{head}  \nNo detections ({res['total']} engines)")
        if res["detections"]:
            with st.expander("Detections"):
                st.write("\n".join(f"- {d}" for d in res["detections"][:40]))
    elif status == "unknown":
        st.info(f"{head}  \nNot known to VirusTotal. Tick the upload box to scan it.")
    elif status == "pending":
        st.info(f"{head}  \nScan is still running. Open the link below in a minute.")
    else:
        st.warning(f"{head}  \n{res.get('error', 'Scan failed.')}")
    st.markdown(f"[Open on VirusTotal]({res['link']})")


def vt_summary_text(results: list[dict]) -> str:
    """One short line per file to give the AI as context."""
    lines = []
    for r in results:
        if r["status"] == "found":
            lines.append(
                f"{r['name']} (sha256 {r['sha256']}): VirusTotal "
                f"{r['malicious']} malicious, {r['suspicious']} suspicious "
                f"of {r['total']} engines. {'; '.join(r['detections'][:8])}"
            )
        else:
            lines.append(f"{r['name']} (sha256 {r['sha256']}): VirusTotal {r['status']}")
    return "VirusTotal results:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF export of the SOC report
# ---------------------------------------------------------------------------

PDF_FONT_CANDIDATES = (
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
)
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
def markdown_to_pdf(text: str, title: str = "Report"):
    """Render simple Markdown text into PDF bytes (None if fpdf2 is missing)."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    regular = bold = ""
    for reg, bld in PDF_FONT_CANDIDATES:
        if Path(reg).exists():
            regular, bold = reg, (bld if Path(bld).exists() else reg)
            break

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    unicode_ok = bool(regular)
    if unicode_ok:
        pdf.add_font("body", "", regular)
        pdf.add_font("body", "B", bold)
        family = "body"
        try:  # proper Arabic shaping when uharfbuzz is installed
            pdf.set_text_shaping(True)
        except Exception:  # noqa: BLE001
            pass
    else:
        family = "Helvetica"

    def clean(line: str) -> str:
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        if not unicode_ok:
            line = line.encode("latin-1", "replace").decode("latin-1")
        return line

    def put(line: str, size: int, style: str = "", indent: float = 0):
        pdf.set_font(family, style, size)
        rtl = bool(ARABIC_RE.search(line))
        pdf.set_x(pdf.l_margin + (0 if rtl else indent))
        pdf.multi_cell(
            0, size * 0.55, line, align="R" if rtl else "L",
            new_x="LMARGIN", new_y="NEXT",
        )

    put(clean(title), 18, "B")
    pdf.ln(3)
    in_code = False
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            pdf.ln(1)
            continue
        if in_code:
            put(clean(line) or " ", 8, "", 4)
        elif not line.strip():
            pdf.ln(3)
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            pdf.ln(2)
            put(clean(line.lstrip("# ").strip()), max(18 - 2 * level, 11), "B")
        elif re.match(r"^\s*[-*]\s+", line):
            put("- " + clean(re.sub(r"^\s*[-*]\s+", "", line)), 10, "", 4)
        elif line.strip() in ("---", "***"):
            pdf.ln(2)
        elif line.lstrip().startswith("|"):
            if not re.match(r"^\s*\|[\s:\-|]+\|\s*$", line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                put(clean("  |  ".join(cells)), 9)
        else:
            put(clean(line), 10)
    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Daily usage counter and the little "limit left" box
# ---------------------------------------------------------------------------
def _usage_day(provider_id: str) -> str:
    """Today's date where the provider's daily quota resets."""
    if provider_id == "gemini":  # Google resets at midnight Pacific time
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/Los_Angeles")
        except Exception:  # noqa: BLE001 - tzdata may be missing on Windows
            tz = timezone(timedelta(hours=-8))
    else:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def _load_usage() -> dict:
    if is_public():
        return st.session_state.setdefault("usage_counter", {})
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_usage(data: dict) -> None:
    if is_public():
        st.session_state["usage_counter"] = data
        return
    try:
        USAGE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Could not save usage counter: %s", exc)


def record_usage(provider: Provider, model: str) -> None:
    """Count one successful request for today."""
    data = _load_usage()
    key = f"{provider.id}:{model}"
    day = _usage_day(provider.id)
    entry = data.get(key) or {}
    used = entry.get("used", 0) if entry.get("day") == day else 0
    data[key] = {"day": day, "used": used + 1}
    _save_usage(data)
    user_id = account_id()
    if user_id:
        try:
            get_db().add_usage(
                user_id, datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                provider.id, model,
            )
        except Exception:  # noqa: BLE001 - counting must never break a reply
            LOGGER.exception("Could not record the usage")


def note_quota_error(provider: Provider, model: str, text: str) -> None:
    """Learn the daily limit from a 429 message and mark it as used up."""
    if " API error 429" not in text:
        return
    data = _load_usage()
    key = f"{provider.id}:{model}"
    found = re.search(r"limit:\s*(\d+)", text)
    limits = data.setdefault("limits", {})
    if found:
        limits[key] = int(found.group(1))
    daily = re.search(r"retry in \d+h", text, re.IGNORECASE) or (
        "per day" in text.lower()
    )
    limit = limits.get(key) or KNOWN_DAILY_LIMITS.get(key)
    if daily and limit:
        data[key] = {"day": _usage_day(provider.id), "used": limit}
    _save_usage(data)


def usage_snapshot(provider: Provider, model: str) -> tuple[int, int | None]:
    """Return (requests used today, daily limit or None if unknown)."""
    data = _load_usage()
    key = f"{provider.id}:{model}"
    entry = data.get(key) or {}
    used = entry.get("used", 0) if entry.get("day") == _usage_day(
        provider.id) else 0
    limit = (data.get("limits") or {}).get(key) or KNOWN_DAILY_LIMITS.get(key)
    return used, limit


def _fmt_num(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def limit_widget_html(settings) -> str:
    """Small 'limit left' widget with a green-yellow-red bar (one line of HTML)."""
    provider, model = settings.provider, settings.model
    used, limit = usage_snapshot(provider, model)
    if limit:
        left = max(limit - used, 0)
        pct = min(used / limit, 1.0) * 100
        color = "#E5383B" if pct >= 75 else (
            "#f0c24b" if pct >= 50 else "#4cc38a"
        )
        text = (
            f"<b style='color:{color}'>{left} {tt('left', 'متبقي')}</b> "
            f"{tt('of', 'من')} {limit} {tt('today', 'اليوم')} "
            f"({used} {tt('used', 'مستخدم')})"
        )
        bar = (
            "<div class='lw-bar'><span class='lw-mark' "
            f"style='left:{pct:.1f}%'></span></div>"
        )
    else:
        text = tt(f"{used} used today &middot; limit unknown",
                  f"{used} \u0645\u0633\u062a\u062e\u062f\u0645 \u0627\u0644\u064a\u0648\u0645 &middot; \u0627\u0644\u062d\u062f \u063a\u064a\u0631 \u0645\u0639\u0631\u0648\u0641")
        bar = "<div class='lw-bar unknown'></div>"
    info = (st.session_state.get("rate_info") or {}).get(
        f"{provider.id}:{model}"
    ) or {}
    extra = ""
    if info.get("remaining_requests"):
        extra += (
            f"<div class='lw-extra'>{tt('requests left', 'الطلبات المتبقية')}: "
            f"{_fmt_num(info['remaining_requests'])} / "
            f"{_fmt_num(info.get('limit_requests', '?'))}</div>"
        )
    if info.get("remaining_tokens"):
        extra += (
            f"<div class='lw-extra'>{tt('tokens left this minute', 'الرموز المتبقية هذه الدقيقة')}: "
            f"{_fmt_num(info['remaining_tokens'])}</div>"
        )
    return f"<div class='limit-widget'><div>{text}</div>{bar}{extra}</div>"

# ---------------------------------------------------------------------------
# SOC helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicators:
    """Indicators of compromise found in text."""

    ips: tuple[str, ...]
    external_ips: tuple[str, ...]
    hashes: tuple[str, ...]


def extract_indicators(text: str) -> Indicators:
    """Find valid IPv4 addresses (public ones separately) and SHA-256s."""
    ips = set()
    for candidate in IPV4_RE.findall(text):
        try:
            ipaddress.IPv4Address(candidate)
        except ValueError:
            continue
        ips.add(candidate)
    ordered = sorted(ips, key=lambda ip: tuple(map(int, ip.split("."))))
    external = [ip for ip in ordered if ipaddress.ip_address(ip).is_global]
    hashes = sorted({h.lower() for h in SHA256_RE.findall(text)})
    return Indicators(tuple(ordered), tuple(external[:10]), tuple(hashes[:10]))


# ---------------------------------------------------------------------------
# Chat history (stored in session and, optionally, on disk)
# ---------------------------------------------------------------------------
def new_chat_record(project: str = "") -> dict:
    """Create an empty chat; the track's start screen shows until it is used."""
    return {"id": uuid.uuid4().hex, "title": NEW_CHAT_TITLE, "messages": [],
            "project": project}


def _is_old_welcome(msg: dict) -> bool:
    """Old versions stored a greeting as the first message of every chat."""
    return msg.get("role") == "assistant" and str(
        msg.get("content", "")
    ).startswith("Hi there!")


def _sanitize_history(data) -> dict[str, list[dict]]:
    """Keep only well-formed chats from a loaded history file."""
    clean: dict[str, list[dict]] = {}
    if not isinstance(data, dict):
        return clean
    for mode, chats in data.items():
        if not isinstance(chats, list):
            continue
        valid = []
        for chat in chats:
            if not isinstance(chat, dict):
                continue
            msgs = [
                m for m in chat.get("messages", [])
                if isinstance(m, dict)
                and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)
            ]
            if msgs and _is_old_welcome(msgs[0]):
                msgs = msgs[1:]
            if msgs:
                valid.append({
                    "id": str(chat.get("id") or uuid.uuid4().hex)[:36],
                    "title": str(chat.get("title") or NEW_CHAT_TITLE),
                    "messages": msgs,
                    "project": str(chat.get("project") or ""),
                })
        if valid:
            name = LEGACY_TRACK_NAMES.get(mode, mode)
            clean.setdefault(name, []).extend(valid)
    return clean


def load_history(path: Path = HISTORY_FILE) -> dict[str, list[dict]]:
    """Load saved chats, backing up a corrupt file instead of losing it."""
    if not persistence_enabled() or not path.exists():
        return {}
    try:
        return _sanitize_history(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        try:
            path.replace(path.with_suffix(".corrupt.json"))
        except OSError:
            LOGGER.debug("Could not back up the corrupt history file")
        return {}


def _atomic_write_json(data, path: Path) -> bool:
    """Write JSON atomically so a crash never leaves a half-written file."""
    tmp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            json.dump(data, tmp, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            LOGGER.debug("Could not restrict file permissions")
        return True
    except OSError as exc:
        LOGGER.warning("Could not save %s: %s", path.name, exc)
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        return False


def save_history(history: dict, path: Path = HISTORY_FILE) -> bool:
    """Write chats to disk (nothing is written in public demo mode)."""
    if not persistence_enabled():
        return True
    return _atomic_write_json(history, path)


def persist(history: dict) -> None:
    """Save history and tell the user (once per call) if that fails."""
    user_id = account_id()
    if user_id:
        sync_db_history(user_id, history)
    elif not save_history(history):
        st.toast("Could not save the chat history to disk.")


# ---------------------------------------------------------------------------
# Accounts: sign-in with Google or Apple, and one database for every user
# ---------------------------------------------------------------------------
def server_setting(*names: str) -> str:
    """Read server-side configuration (database URL, encryption key).

    Unlike API keys these are always read, also in public demo mode.
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    try:
        for name in names:
            value = str(st.secrets.get(name, "")).strip()
            if value:
                return value
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        LOGGER.debug("No Streamlit secrets available")
    return ""


@st.cache_resource(show_spinner=False)
def _open_db(url: str, keys: tuple):
    """Open the database once per server. Failures are raised, not cached."""
    return nashmi_db.Database(url, keys)


def get_db():
    """The shared database, or None when accounts are not configured."""
    url = server_setting("NASHMI_DATABASE_URL", "DATABASE_URL")
    if not url or nashmi_db is None:
        return None
    if time.time() - st.session_state.get("_db_failed", 0) < 30:
        return None  # do not hammer a database that just failed
    keys = tuple(k.strip() for k in
                 server_setting("NASHMI_SECRET_KEY").split(",") if k.strip())
    try:
        return _open_db(url, keys)
    except Exception as exc:  # noqa: BLE001 - never crash the whole app
        st.session_state["_db_failed"] = time.time()
        LOGGER.error("Accounts are disabled: %s", type(exc).__name__)
        return None


def auth_providers() -> list[str]:
    """Sign-in options set up in the Streamlit secrets ([auth.google] ...)."""
    try:
        auth = st.secrets.get("auth")
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        return []
    if not auth:
        return []
    named = [
        name for name in ("google", "apple")
        if hasattr(auth.get(name), "get")
    ]
    if named:
        return named
    return ["default"] if auth.get("client_id") else []


def accounts_enabled() -> bool:
    return bool(auth_providers()) and get_db() is not None


LOGIN_LABELS = {
    "google": "Continue with Google",
    "apple": "Continue with Apple",
    "default": "Sign in",
}


def start_login(provider: str) -> None:
    st.login(None if provider == "default" else provider)


def current_user() -> dict | None:
    """The signed-in user from the sign-in provider, or None for guests.

    The identity key is ``provider:sub``, never the e-mail address, and
    accounts from different providers are never merged automatically.
    """
    try:
        user = st.user
        if not getattr(user, "is_logged_in", False):
            return None
        issuer = str(user.get("iss") or "")
        if "appleid.apple.com" in issuer:
            provider = "apple"
        elif "accounts.google.com" in issuer:
            provider = "google"
        else:
            provider = (urllib.parse.urlparse(issuer).hostname or "oidc")[:20]
        sub = str(user.get("sub") or "")
        if not sub:
            return None
        email = str(user.get("email") or "")
        return {
            "sub": f"{provider}:{sub}",
            "provider": provider,
            "email": email,
            "name": str(user.get("name") or email.split("@")[0] or "Account"),
            "picture": str(user.get("picture") or ""),
            "verified": user.get("email_verified") is True,
        }
    except Exception:  # noqa: BLE001 - sign-in not configured
        return None


ACCOUNT_SESSION_KEYS = ("saved_history", "store", "_digests", "saved_keys")


def _forget_account_data() -> None:
    for key in ACCOUNT_SESSION_KEYS:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if str(key).startswith(("chat_radio_", "chat_search_", "opened_")):
            del st.session_state[key]
    st.session_state["project"] = ""


def account_id() -> int | None:
    """Database id of the signed-in user (None for guests and local mode).

    When the person signs out, or somebody else signs in, everything that
    belonged to the previous identity is dropped from the session.
    """
    user, db = current_user(), get_db()
    if not (user and db) or st.session_state.get("_account_deleted"):
        if not user:
            st.session_state.pop("_account_deleted", None)
        if st.session_state.pop("_account_sub", None) is not None:
            st.session_state.pop("_account_id", None)
            _forget_account_data()
        return None
    if st.session_state.get("_account_sub") != user["sub"]:
        _forget_account_data()
        st.session_state["_account_sub"] = user["sub"]
        try:
            st.session_state["_account_id"] = db.upsert_user(
                user["sub"], user["email"], user["name"]
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Could not open the account")
            st.session_state["_account_id"] = None
        else:
            try:  # the sign-in log must never block signing in
                db.record_login(st.session_state["_account_id"])
            except Exception:  # noqa: BLE001
                LOGGER.exception("Could not log the sign-in")
    return st.session_state.get("_account_id")


PRESENCE_EVERY_SECONDS = 60
ONLINE_WINDOW_MINUTES = 5


def touch_presence() -> None:
    """Tell the database this signed-in user is active (at most once a minute
    per session), so the admin screen can show who is online right now."""
    user_id, db = st.session_state.get("_account_id"), get_db()
    if not (user_id and db):
        return
    if time.time() - st.session_state.get("_presence_at", 0) < PRESENCE_EVERY_SECONDS:
        return
    st.session_state["_presence_at"] = time.time()
    try:
        db.touch_presence(user_id)
    except Exception:  # noqa: BLE001 - presence is best effort
        LOGGER.exception("Could not update presence")


def admin_emails() -> set[str]:
    raw = server_setting("NASHMI_ADMIN_EMAILS")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin() -> bool:
    """Only a verified Google e-mail listed in NASHMI_ADMIN_EMAILS."""
    user = current_user()
    return bool(
        user and account_id() and user["provider"] == "google"
        and user["verified"] and user["email"].strip().lower() in admin_emails()
        and mfa_passed(account_id())
    )


def _chat_digest(track: str, chat: dict) -> str:
    raw = json.dumps(
        [track, chat.get("title"), chat.get("project"), chat.get("messages")],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_user_history() -> dict:
    """Chats of the signed-in user, or the local file for everybody else."""
    user_id = account_id()
    if not user_id:
        return load_history()
    try:
        history = get_db().load_history(user_id)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Could not load the chats")
        st.toast("Could not load your saved chats.")
        history = {}
    st.session_state["_digests"] = {
        chat["id"]: _chat_digest(track, chat)
        for track, chats in history.items() for chat in chats
    }
    return history


def sync_db_history(user_id: int, history: dict) -> None:
    """Write only the chats that changed since the last save."""
    db = get_db()
    digests = st.session_state.setdefault("_digests", {})
    seen: set[str] = set()
    failed = 0
    for track, chats in history.items():
        for chat in chats:
            if not chat["messages"]:  # empty chats live in the session only
                continue
            chat_id = chat.setdefault("id", uuid.uuid4().hex)
            seen.add(chat_id)
            digest = _chat_digest(track, chat)
            if digests.get(chat_id) == digest:
                continue
            try:
                saved = db.save_chat(user_id, track, chat)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Could not save a chat")
                saved = False
            if saved:
                digests[chat_id] = digest
            else:
                failed += 1
    for chat_id in [c for c in digests if c not in seen]:
        try:
            db.delete_chat(user_id, chat_id)
            digests.pop(chat_id, None)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Could not delete a chat")
    if failed:
        st.toast("Some chats could not be saved (limit reached or database "
                 "problem).")


def load_user_store() -> dict:
    user_id = account_id()
    if not user_id:
        return load_store()
    try:
        return _sanitize_store(get_db().load_store(user_id))
    except Exception:  # noqa: BLE001
        LOGGER.exception("Could not load the settings")
        return _sanitize_store(None)


def saved_keys() -> dict[str, str]:
    """API keys the signed-in user chose to save (decrypted, session only)."""
    user_id, db = account_id(), get_db()
    if not (user_id and db and db.cipher.enabled):
        return {}
    if "saved_keys" not in st.session_state:
        try:
            st.session_state["saved_keys"] = db.get_keys(user_id)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Could not load the saved keys")
            st.session_state["saved_keys"] = {}
    return st.session_state["saved_keys"]


def on_save_key(provider_id: str) -> None:
    user_id, db = account_id(), get_db()
    key = str(st.session_state.get(f"acct_key_{provider_id}", "")).strip()
    if not (user_id and db and key):
        return
    try:
        db.set_key(user_id, provider_id, key)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Could not save the key")
        st.toast("Could not save the key.")
        return
    saved_keys()[provider_id] = key
    st.session_state[f"acct_key_{provider_id}"] = ""
    st.toast("Key saved, encrypted.")


def on_remove_key(provider_id: str) -> None:
    user_id, db = account_id(), get_db()
    if not (user_id and db):
        return
    db.delete_key(user_id, provider_id)
    saved_keys().pop(provider_id, None)
    st.toast("Saved key removed.")


def on_delete_account() -> None:
    user_id, db = st.session_state.get("_account_id"), get_db()
    if user_id and db:
        db.delete_user(user_id)
    st.session_state["_account_deleted"] = True
    st.session_state.pop("_profile_ok", None)
    _forget_account_data()


def open_account() -> None:
    st.session_state["view"] = "account"


def open_admin() -> None:
    st.session_state["view"] = "admin"


def sign_out() -> None:
    for key in ("_mfa_ok", "_mfa_codes", "_mfa_secret", "_profile_ok", "_account_deleted",
                "_account_id", "_account_sub", "_presence_at"):
        st.session_state.pop(key, None)
    st.session_state["gate"] = "landing"
    st.logout()


def close_account() -> None:
    st.session_state["view"] = "chat"


def _sanitize_store(data) -> dict:
    """Projects and preferences, keeping only well-formed entries."""
    store = {"prefs": dict(DEFAULT_PREFS), "projects": []}
    if not isinstance(data, dict):
        return store
    prefs = data.get("prefs")
    if isinstance(prefs, dict):
        if prefs.get("detail") in DETAIL_CHOICES:
            store["prefs"]["detail"] = prefs["detail"]
        if prefs.get("language") in LANGUAGE_CHOICES:
            store["prefs"]["language"] = prefs["language"]
    projects = data.get("projects")
    seen: set[str] = set()
    for item in projects if isinstance(projects, list) else []:
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("name"), str)
            and item["name"].strip()
            and item["id"] not in seen
        ):
            seen.add(item["id"])
            store["projects"].append({
                "id": item["id"],
                "name": item["name"].strip()[:60],
                "instructions": str(item.get("instructions", ""))[:2000],
            })
    return store


def load_store(path: Path = SETTINGS_FILE) -> dict:
    if not persistence_enabled() or not path.exists():
        return _sanitize_store(None)
    try:
        return _sanitize_store(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return _sanitize_store(None)


def get_store() -> dict:
    if "store" not in st.session_state:
        st.session_state["store"] = load_user_store()
    return st.session_state["store"]


def persist_store() -> None:
    user_id = account_id()
    if user_id:
        try:
            get_db().save_store(user_id, get_store())
        except Exception:  # noqa: BLE001
            LOGGER.exception("Could not save the settings")
            st.toast("Could not save your settings.")
    elif persistence_enabled() and not _atomic_write_json(get_store(), SETTINGS_FILE):
        st.toast("Could not save your settings to disk.")


def prefs_prompt(project_id: str = "") -> str:
    """Extra system-prompt lines from the user's preferences and project."""
    store = get_store()
    prefs = store["prefs"]
    lines = []
    if prefs["detail"] == "Concise":
        lines.append("Keep answers short and to the point.")
    elif prefs["detail"] == "Detailed":
        lines.append("Give thorough, step-by-step answers with examples.")
    if prefs["language"] == "Arabic":
        lines.append("Always answer in Arabic; keep technical terms and "
                     "code in English.")
    elif prefs["language"] == "English":
        lines.append("Always answer in English.")
    project = next(
        (p for p in store["projects"] if p["id"] == project_id), None
    )
    if project and project["instructions"].strip():
        lines.append(
            f"Instructions from the user for the project "
            f"'{project['name']}' (follow them): "
            f"{project['instructions'].strip()}"
        )
    return ("\n\nUser preferences:\n- " + "\n- ".join(lines)) if lines else ""


def full_system_prompt(mode: str, project_id: str = "") -> str:
    return build_system_prompt(mode) + prefs_prompt(project_id)


def current_mode() -> str:
    """The track picked in the sidebar (also valid inside callbacks)."""
    mode = st.session_state.get("track")
    return mode if mode in TRACKS_BY_NAME else MODES[0]


def radio_key(mode: str) -> str:
    return f"chat_radio_{mode}"


def search_key(mode: str) -> str:
    return f"chat_search_{mode}"


def active_index(mode: str, count: int) -> int:
    """Index of the selected chat, clamped to the available chats."""
    index = st.session_state.get(radio_key(mode), 0)
    return index if isinstance(index, int) and 0 <= index < count else 0


def make_title(text: str, limit: int = 32) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def search_chats(chats: list[dict], query: str) -> list[tuple[int, int]]:
    """Return (chat index, matching messages) for chats that match a query.

    An empty query matches every chat.
    """
    needle = query.strip().lower()
    if not needle:
        return [(i, 0) for i in range(len(chats))]
    found = []
    for i, chat in enumerate(chats):
        hits = sum(needle in m["content"].lower() for m in chat["messages"])
        if hits or needle in chat["title"].lower():
            found.append((i, hits))
    return found


def message_snippets(chat: dict, query: str, limit: int = 6) -> list[str]:
    """Short excerpts around the query inside one chat."""
    needle = query.strip().lower()
    snippets: list[str] = []
    if not needle:
        return snippets
    for msg in chat["messages"]:
        text = " ".join(msg["content"].split())
        pos = text.lower().find(needle)
        if pos < 0:
            continue
        start = max(pos - 30, 0)
        end = start + 90
        who = "You" if msg["role"] == "user" else "Nashmi"
        snippets.append(
            f"{who}: {'...' if start else ''}{text[start:end]}"
            f"{'...' if end < len(text) else ''}"
        )
        if len(snippets) >= limit:
            break
    return snippets


def on_new_chat() -> None:
    mode = current_mode()
    chats = ensure_history(mode)
    project = st.session_state.get("project", "")
    if not chats or chats[0]["messages"]:  # reuse an untouched empty chat
        chats.insert(0, new_chat_record(project))
    else:
        chats[0]["project"] = project
    st.session_state[radio_key(mode)] = 0
    st.session_state[search_key(mode)] = ""
    persist(st.session_state.saved_history)


def on_delete_chat() -> None:
    mode = current_mode()
    chats = ensure_history(mode)
    chats.pop(active_index(mode, len(chats)))
    if not chats:
        chats.append(new_chat_record())
    st.session_state[radio_key(mode)] = 0
    persist(st.session_state.saved_history)


def ensure_history(mode: str) -> list[dict]:
    """Make sure the session has at least one chat for this track."""
    if "saved_history" not in st.session_state:
        st.session_state.saved_history = load_user_history()
    history = st.session_state.saved_history
    if not history.get(mode):
        history[mode] = [new_chat_record(st.session_state.get("project", ""))]
        persist(history)
    flag = f"opened_{mode}"
    if not st.session_state.get(flag):
        # Like ChatGPT or Claude: entering a track starts a fresh chat and
        # the earlier ones stay saved in the list.
        st.session_state[flag] = True
        if history[mode][0]["messages"]:
            history[mode].insert(
                0, new_chat_record(st.session_state.get("project", ""))
            )
            persist(history)
        st.session_state[radio_key(mode)] = 0
    return history[mode]


def on_create_project() -> None:
    name = str(st.session_state.get("new_project_name", "")).strip()
    if not name:
        return
    project_id = uuid.uuid4().hex[:8]
    get_store()["projects"].append({
        "id": project_id,
        "name": name[:60],
        "instructions": str(st.session_state.get("new_project_instr", ""))[:2000],
    })
    st.session_state["project"] = project_id
    st.session_state["new_project_name"] = ""
    st.session_state["new_project_instr"] = ""
    persist_store()


def on_delete_project() -> None:
    project_id = st.session_state.get("project", "")
    store = get_store()
    store["projects"] = [p for p in store["projects"] if p["id"] != project_id]
    history = st.session_state.get("saved_history", {})
    for chats in history.values():
        for chat in chats:
            if chat.get("project") == project_id:
                chat["project"] = ""
    st.session_state["project"] = ""
    persist_store()
    persist(history)


def on_import_data() -> None:
    upload = st.session_state.get("import_data")
    if upload is None:
        return
    if upload.size > 20 * 1024 * 1024:
        st.toast("That file is too large to import.")
        return
    try:
        data = json.loads(upload.getvalue().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        st.toast("That file is not a valid Nashmi export.")
        return
    if not isinstance(data, dict):
        st.toast("That file is not a valid Nashmi export.")
        return
    if "saved_history" not in st.session_state:
        st.session_state.saved_history = load_user_history()
    history = st.session_state.saved_history
    added = 0
    for mode, chats in _sanitize_history(data.get("history", {})).items():
        for chat in chats:
            chat["id"] = uuid.uuid4().hex  # an import never overwrites a chat
        history.setdefault(mode, []).extend(chats)
        added += len(chats)
    store = get_store()
    known = {p["id"] for p in store["projects"]}
    for project in _sanitize_store(data.get("store"))["projects"]:
        if project["id"] not in known:
            store["projects"].append(project)
    persist(history)
    persist_store()
    st.toast(f"Imported {added} chats.")


def on_wipe_all() -> None:
    st.session_state.saved_history = {}
    st.session_state["store"] = _sanitize_store(None)
    for key in list(st.session_state.keys()):
        if str(key).startswith(("chat_radio_", "chat_search_", "opened_")):
            del st.session_state[key]
    st.session_state["project"] = ""
    st.session_state["confirm_wipe"] = False
    persist({})
    persist_store()


# ---------------------------------------------------------------------------
# Images (avatars)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_assistant_avatar() -> Image.Image:
    """Head of the cat, cropped from the embedded picture."""
    cat = Image.open(io.BytesIO(base64.b64decode(CAT_B64))).convert("RGB")
    head = cat.crop((95, 28, 321, 254))
    return head.resize((120, 120), Image.Resampling.LANCZOS)


@st.cache_resource
def get_user_avatar() -> Image.Image:
    """Terminal-prompt icon drawn pixel by pixel."""
    img = Image.new("RGBA", (16, 16), (229, 56, 59, 255))
    draw = ImageDraw.Draw(img)
    white = (255, 255, 255, 255)
    for i in range(4):
        for dx in (0, 1):
            draw.point((3 + i + dx, 4 + i), fill=white)
            draw.point((3 + i + dx, 11 - i), fill=white)
    draw.rectangle([9, 11, 12, 12], fill=white)
    return img.resize((120, 120), Image.Resampling.NEAREST)


# ---------------------------------------------------------------------------
# Theme (responsive, touch friendly)
# ---------------------------------------------------------------------------
FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500;700&display=swap');\n"
)

THEME_CSS = """

:root {
    --bg: #0b111c;
    --panel: #131c2b;
    --panel-2: #0f1724;
    --accent: #E5383B;
    --accent-dim: rgba(229, 56, 59, 0.35);
    --accent-soft: rgba(229, 56, 59, 0.12);
    --line: rgba(164, 177, 205, 0.14);
    --grid-line: rgba(139, 20, 30, 0.20);  /* dark wine-red grid */
    --text: #A4B1CD;
    --muted: #6f7c98;
    --white: #FFFFFF;
    --ok: #4cc38a;
    --warn: #FFB020;
    --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
    --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --safe-top: env(safe-area-inset-top, 0px);
    --safe-bottom: env(safe-area-inset-bottom, 0px);
    --safe-left: env(safe-area-inset-left, 0px);
    --safe-right: env(safe-area-inset-right, 0px);
}

/* ---- Streamlit chrome ---- */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    visibility: visible !important;
}
/* Keep the sidebar toggle reachable so the sidebar can always reopen */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999;
}
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
    color: var(--text) !important;
    min-width: 44px;
    min-height: 44px;
}

/* ---- Smooth sidebar: eased slide, soft shadow, gentle control motion ---- */
[data-testid="stSidebar"] {
    transition: transform 0.45s cubic-bezier(.22, 1, .36, 1),
                min-width 0.45s cubic-bezier(.22, 1, .36, 1),
                max-width 0.45s cubic-bezier(.22, 1, .36, 1),
                box-shadow 0.45s ease !important;
    will-change: transform;
}
[data-testid="stSidebar"][aria-expanded="true"] {
    box-shadow: 10px 0 40px rgba(0, 0, 0, 0.45);
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    transition: transform 0.2s ease, background 0.2s ease,
                border-color 0.2s ease, color 0.2s ease !important;
}
[data-testid="stSidebar"] button:hover { transform: translateX(2px); }
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
    transition: transform 0.25s cubic-bezier(.22, 1, .36, 1),
                background 0.2s ease !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover { transform: scale(1.08); }
@media (prefers-reduced-motion: reduce) {
    [data-testid="stSidebar"], [data-testid="stSidebar"] button {
        transition: none !important;
    }
}

/* ---- Base ---- */
html, body { -webkit-text-size-adjust: 100%; }
* { -webkit-tap-highlight-color: transparent; }
button, [role="button"], label { touch-action: manipulation; }

.stApp {
    color: var(--text);
    font-family: var(--sans);
    overflow-x: hidden;
    background:
        radial-gradient(900px 420px at 50% -10%,
            rgba(229, 56, 59, 0.10), transparent 70%),
        radial-gradient(700px 400px at 100% 100%,
            rgba(214, 38, 43, 0.08), transparent 70%),
        radial-gradient(ellipse 75% 65% at 50% 30%,
            transparent 0%, rgba(11, 17, 28, 0.55) 60%,
            rgba(11, 17, 28, 0.92) 100%),
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
        var(--bg);
    background-size: auto, auto, auto, 40px 40px, 40px 40px, auto;
    background-attachment: fixed;
}
.block-container {
    max-width: 1000px;
    padding: 1rem 1.25rem 3rem;
    padding-left: max(1.25rem, var(--safe-left));
    padding-right: max(1.25rem, var(--safe-right));
}
h1, h2, h3, h4 {
    font-family: var(--sans) !important;
    color: var(--white) !important;
    overflow-wrap: anywhere;
}
img, video, canvas, iframe { max-width: 100%; }
a { color: var(--accent) !important; }
hr { border-color: var(--line); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted); }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--panel-2);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { padding-bottom: 0; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.25rem;
}
.side-label {
    font-family: var(--mono);
    font-size: 0.66rem;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.9rem 0 0.2rem;
}
.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 0.6rem;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 1.02rem;
    color: var(--white);
}
.brand img {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    border: 1px solid var(--line);
}
.brand .prompt { color: var(--accent); }

/* Track list and chat list look like menu rows, not radio buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    gap: 1px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    display: flex;
    box-sizing: border-box;
    width: 100%;
    margin: 0;
    padding: 5px 10px;
    border-radius: 8px;
    border-left: 2px solid transparent;
    cursor: pointer;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background: rgba(164, 177, 205, 0.08);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background: var(--accent-soft);
    border-left-color: var(--accent);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
    font-family: var(--sans);
    font-size: 0.88rem;
    letter-spacing: 0;
    text-transform: none;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
    color: var(--white);
}

[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
    font-family: var(--mono);
    font-size: 0.7rem !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted) !important;
}

/* ---- Inputs ---- */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--panel) !important;
    color: var(--white) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    min-height: 40px;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: none;
}

/* ---- Buttons ---- */
.stButton, .stDownloadButton { width: 100%; }
.stButton > button, .stDownloadButton > button,
.stFormSubmitButton > button {
    width: 100%;
    min-height: 40px;
    height: auto;
    padding: 0.45rem 0.9rem;
    background: transparent;
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 10px;
    font-family: var(--sans);
    font-weight: 500;
    font-size: 0.9rem;
    white-space: normal;
    overflow-wrap: anywhere;
    transition: background 0.15s ease, border-color 0.15s ease,
        color 0.15s ease;
}
.stButton > button:active, .stDownloadButton > button:active {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
}
@media (hover: hover) and (pointer: fine) {
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--accent-soft);
        color: var(--white);
        border-color: var(--accent-dim);
    }
}
.st-key-new_chat, [class*="st-key-quick_"] { width: 100% !important; }
.st-key-new_chat button {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
    font-weight: 600;
}
@media (hover: hover) and (pointer: fine) {
    .st-key-new_chat button:hover {
        background: #cf2e31;
        border-color: #cf2e31;
    }
}
.st-key-new_chat button:disabled {
    opacity: 0.4;
}
/* Suggestion rows on the start screen */
[class*="st-key-quick_"] button {
    justify-content: flex-start;
    text-align: left;
    background: var(--panel);
    padding: 0.7rem 1rem;
}
[class*="st-key-quick_"] button > div { width: 100%; justify-content: flex-start; }
[class*="st-key-quick_"] button p { text-align: left; }
/* Small text buttons (delete chat) */
[data-testid="stSidebar"] button[kind="tertiary"] {
    color: var(--muted);
    border: none;
    min-height: 32px;
    font-size: 0.82rem;
    justify-content: flex-start;
}
[data-testid="stSidebar"] button[kind="tertiary"]:hover {
    color: var(--accent);
    background: transparent;
}

/* ---- Usage line (sidebar footer) ---- */
.limit-widget {
    margin: 0.4rem 0 0;
    font-family: var(--mono);
    font-size: 0.66rem;
    color: var(--muted);
    line-height: 1.5;
}
.limit-widget .lw-bar {
    position: relative;
    height: 4px;
    margin-top: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #4cc38a, #f0c24b, #E5383B);
}
.limit-widget .lw-bar.unknown { opacity: 0.3; }
.limit-widget .lw-mark {
    position: absolute;
    top: -3px;
    width: 3px;
    height: 10px;
    border-radius: 2px;
    background: #ffffff;
    transform: translateX(-50%);
}
.limit-widget .lw-extra { opacity: 0.8; margin-top: 2px; }

/* ---- Top line and start screen ---- */
.topline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 10px;
    margin: 0 0 0.9rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--line);
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--muted);
}
.topline .t-track { color: var(--white); font-weight: 700; }
.topline .t-sep { opacity: 0.5; }
.topline .t-model { overflow-wrap: anywhere; }
.topline .t-key { margin-left: auto; }
.topline .t-key::before {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 6px;
    border-radius: 50%;
    background: var(--ok);
}
.topline .t-key.bad { color: var(--warn); }
.topline .t-key.bad::before { background: var(--warn); }
.empty-title {
    margin: 0.9rem 0 0.15rem;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--white);
}
.empty-blurb { margin: 0 0 1rem; color: var(--text); font-size: 0.95rem; }

/* ---- Chat ---- */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 0.35rem 0;
    gap: 0.75rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.55rem 0.9rem;
}
[data-testid="stChatMessage"] > div { min-width: 0; }
[data-testid^="stChatMessageAvatar"] {
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
}
/* Detect text direction per paragraph (Arabic and English mixed) */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    direction: auto !important;
    text-align: start !important;
    unicode-bidi: plaintext !important;
}
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    direction: auto !important;
    padding-inline-start: 24px;
}
/* Code always stays left-to-right and scrolls instead of overflowing */
[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] code {
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: isolate !important;
}
[data-testid="stChatMessage"] code { color: var(--accent); }
pre, [data-testid="stCode"] {
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
[data-testid="stChatMessage"] pre {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 10px;
}
[data-testid="stChatMessage"] table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
[data-testid="stChatMessage"] img { border-radius: 6px; height: auto; }
[data-testid="stChatMessage"] iframe { display: block; border: 0; }
[class*="st-key-actions_"] [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap;
    align-items: center;
    gap: 0.25rem;
}
[class*="st-key-actions_"] [data-testid="stColumn"] {
    flex: 0 0 auto;
    width: auto;
    min-width: 0;
}
[class*="st-key-actions_"] [data-testid="stColumn"]:first-child {
    flex: 0 0 104px;
    width: 104px;
}
[class*="st-key-actions_"] [data-testid="stPopover"] button {
    min-height: 30px;
    padding: 0.15rem 0.6rem;
    border-color: transparent;
    background: transparent;
    color: var(--muted);
    font-size: 0.8rem;
}
[class*="st-key-actions_"] [data-testid="stPopover"] button:hover {
    color: var(--white);
    background: rgba(164, 177, 205, 0.12);
}

/* ---- Composer ---- */
[data-testid="stBottom"] > div {
    background: linear-gradient(to top, var(--bg) 55%, transparent) !important;
}
[data-testid="stChatInput"] {
    max-width: 100%;
    background: var(--panel);
    border: 1px solid rgba(164, 177, 205, 0.20);
    border-radius: 22px;
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.50),
        0 2px 8px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    cursor: text;
}
[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(229, 56, 59, 0.45);
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.50),
        0 0 0 3px rgba(229, 56, 59, 0.10),
        0 0 26px rgba(91, 124, 255, 0.16);
}
[data-testid="stChatInput"] textarea {
    color: var(--white) !important;
    caret-color: var(--accent);
    min-height: 3rem;
    max-height: 40vh !important;
    line-height: 1.5;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button { color: var(--text); cursor: pointer; }
/* The "+" sits on the right, next to the microphone and the send button */
[data-testid="stChatInput"] > div > div > div:has(> [data-testid="stChatInputFileUploadButton"]) {
    order: 1 !important;
}
[data-testid="stChatInput"] > div > div > div:has([data-testid="stChatInputSubmitButton"]) {
    order: 2 !important;
}
[data-testid="stChatInput"] button:hover { color: var(--accent); }

/* ---- Expanders, uploader, metrics ---- */
[data-testid="stExpander"] details {
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 10px;
}
[data-testid="stExpander"] summary p {
    font-family: var(--sans);
    font-size: 0.88rem;
    color: var(--text);
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-2);
    border: 1px dashed var(--line);
    border-radius: 12px;
    flex-wrap: wrap;
    gap: 0.5rem;
}
[data-testid="stFileUploader"] { max-width: 100%; }
[data-testid="stFileUploaderFile"] { min-width: 0; overflow-wrap: anywhere; }
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.6rem 0.9rem;
}

/* ---- Frames: a soft red-to-blue sheen ---- */
:root {
    --accent-2: #5b7cff;
    --frame: linear-gradient(135deg, rgba(229, 56, 59, 0.62),
        rgba(91, 124, 255, 0.42) 55%, rgba(229, 56, 59, 0.20));
    --frame-hover: linear-gradient(135deg, rgba(229, 56, 59, 0.95),
        rgba(91, 124, 255, 0.75) 55%, rgba(229, 56, 59, 0.40));
    --sheen: inset 0 1px 0 rgba(255, 255, 255, 0.07);
}
[data-testid="stSidebar"] {
    border-right: none;
    background: linear-gradient(180deg, #101a2b 0%, #0b111c 100%);
    box-shadow: inset -1px 0 0 rgba(229, 56, 59, 0.30),
        8px 0 28px rgba(91, 124, 255, 0.07);
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stExpander"] details,
[data-testid="stMetric"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[class*="st-key-quick_"] button {
    border: 1px solid transparent !important;
    background: linear-gradient(var(--panel), var(--panel)) padding-box,
        var(--frame) border-box !important;
    box-shadow: var(--sheen);
    transition: box-shadow 0.2s ease;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-baseweb="input"]:hover,
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
[data-testid="stExpander"] details:hover,
[class*="st-key-quick_"] button:hover {
    background: linear-gradient(var(--panel), var(--panel)) padding-box,
        var(--frame-hover) border-box !important;
    box-shadow: var(--sheen), 0 0 18px rgba(91, 124, 255, 0.18),
        0 0 14px rgba(229, 56, 59, 0.14);
}
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] .stTextInput input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background: linear-gradient(90deg, rgba(229, 56, 59, 0.20),
        rgba(91, 124, 255, 0.08));
}
.st-key-new_chat button {
    background: linear-gradient(135deg, #E5383B 0%, #d2325a 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.28),
        0 0 16px rgba(229, 56, 59, 0.25);
}

/* ---- Account card and link buttons ---- */
.user-card {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 0.5rem;
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: rgba(164, 177, 205, 0.05);
    font-size: 0.82rem;
    color: var(--white);
    min-width: 0;
}
.user-card img, .user-card .avatar-fallback {
    flex: 0 0 34px;
    width: 34px;
    height: 34px;
    border-radius: 50%;
}
.user-card .avatar-fallback {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #E5383B, #5b7cff);
    font-weight: 700;
}
.user-card div { min-width: 0; overflow-wrap: anywhere; }
.user-card span { color: var(--muted); font-size: 0.74rem; }
.stLinkButton a {
    border: 1px solid var(--line);
    border-radius: 10px;
    color: var(--text) !important;
    background: transparent;
}
.stLinkButton a:hover {
    border-color: var(--accent-dim);
    background: var(--accent-soft);
    color: var(--white) !important;
}

/* ---- Exam timer ---- */
.exam-timer {
    margin-top: 0.9rem;
    text-align: right;
    font-family: var(--mono);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--white);
}
.exam-timer.low { color: var(--accent); }

/* ---- Sidebar: calm and compact ---- */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] { min-height: 38px; }
.st-key-side_chats { margin-top: 0.2rem; }
.st-key-side_chats [data-testid="stRadio"] [role="radiogroup"] {
    max-height: 34vh;
    overflow-y: auto;
    padding-right: 2px;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] { padding: 4px 10px; }
.st-key-chat_head [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap;
    align-items: center;
    gap: 0;
}
.st-key-chat_head [data-testid="stColumn"] {
    flex: 1 1 auto;
    width: auto;
    min-width: 0;
}
.st-key-chat_head [data-testid="stColumn"]:last-child {
    flex: 0 0 34px;
    width: 34px;
}
.st-key-chat_head .side-label { margin: 0.1rem 0 0; }
.st-key-chat_head [data-testid="stPopover"] button {
    width: 32px;
    min-height: 28px;
    padding: 0;
    border: none;
    background: transparent;
    color: var(--muted);
}
.st-key-chat_head [data-testid="stPopover"] button:hover {
    color: var(--white);
    background: rgba(164, 177, 205, 0.12);
}
.st-key-side_settings {
    margin-top: 0.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--line);
}
.st-key-user_actions [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap;
    gap: 0.4rem;
}
.st-key-user_actions [data-testid="stColumn"] {
    flex: 1 1 0;
    min-width: 0;
}

.st-key-side_user { margin-top: 0.6rem; }

/* ---- Scrollbar (desktop) ---- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(164, 177, 205, 0.2);
    border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

/* ---- Columns wrap instead of overflowing ---- */
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
[data-testid="stColumn"], [data-testid="column"] {
    flex: 1 1 14rem;
    min-width: min(100%, 14rem);
}

/* ---- Tablets and phones ---- */
@media (max-width: 768px) {
    .block-container {
        padding: calc(3.25rem + var(--safe-top)) 0.9rem 4rem;
        padding-left: max(0.9rem, var(--safe-left));
        padding-right: max(0.9rem, var(--safe-right));
    }
    [data-testid="stSidebar"] {
        width: min(88vw, 21rem) !important;
        min-width: 0 !important;
        padding-bottom: var(--safe-bottom);
    }
    [data-testid="stBottom"] > div {
        padding-bottom: max(0.5rem, var(--safe-bottom)) !important;
    }
    /* 16px stops iOS Safari from zooming into focused fields */
    input, textarea, select, [data-baseweb="select"] input,
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;
    }
    [data-testid^="stChatMessageAvatar"] { width: 1.9rem; height: 1.9rem; }
    .empty-title { font-size: 1.25rem; }
}
@media (max-width: 400px) {
    .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
"""

HERO_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
    html, body {
        margin: 0; height: 100%; overflow: hidden; background: transparent;
        font-family: Consolas, 'Courier New', monospace;
        -webkit-text-size-adjust: 100%;
    }
    #stage {
        position: relative; box-sizing: border-box; width: 100%;
        height: calc(100% - 4px); overflow: hidden; border-radius: 16px;
        border: 1px solid rgba(229, 56, 59, 0.45);
        background: radial-gradient(circle at 50% 45%, #2a1018 0%,
            #0f1724 55%, #070b12 100%);
    }
    #bg { position: absolute; inset: 0; width: 100%; height: 100%; }
    #cat {
        position: absolute; left: 50%; top: 41%;
        width: min(328px, 96%); height: auto;
        image-rendering: pixelated; image-rendering: crisp-edges;
        transform: translate(-50%, -50%); will-change: transform;
        -webkit-mask-image: radial-gradient(ellipse at center, #000 52%,
            transparent 78%);
        mask-image: radial-gradient(ellipse at center, #000 52%,
            transparent 78%);
    }
    .overlay {
        position: absolute; left: 50%; bottom: 10px;
        transform: translateX(-50%); box-sizing: border-box;
        width: calc(100% - 24px); max-width: 560px; text-align: center;
        pointer-events: none; padding: 6px 14px; border-radius: 12px;
        background: rgba(11, 17, 28, 0.85);
        border: 1px solid rgba(229, 56, 59, 0.35);
    }
    .title {
        margin: 0; font-size: 1.15rem; font-weight: 800;
        letter-spacing: 3px; color: #E5383B; text-transform: uppercase;
        text-shadow: 0 0 12px rgba(229, 56, 59, 0.6);
    }
    .subtitle { margin: 2px 0 0; font-size: 0.75rem; color: #A4B1CD; }
    .caret {
        display: inline-block; width: 9px; height: 1rem;
        background: #E5383B; margin-left: 5px; vertical-align: -2px;
        animation: blink 1s steps(2) infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }
    @media (max-width: 480px) {
        .title { font-size: 1rem; letter-spacing: 2px; }
        .subtitle { font-size: 0.68rem; }
    }
    @media (max-width: 380px) { .subtitle { display: none; } }
    @media (prefers-reduced-motion: reduce) { .caret { animation: none; } }
</style>
</head>
<body>
<div id="stage">
    <canvas id="bg"></canvas>
    <canvas id="cat" width="416" height="292"></canvas>
    <div class="overlay">
        <h1 class="title">NASHMI AGENT<span class="caret"></span></h1>
        <p class="subtitle">Your friendly SOC and security coding buddy.
            The cat is watching your pointer.</p>
    </div>
</div>
<script>
(() => {
    'use strict';
    const stage = document.getElementById('stage');
    const cat = document.getElementById('cat');
    const ctx = cat.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    const reduceMotion =
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Eye rectangles measured on the original picture (x, y, w, h)
    const EYES = [
        {x: 169, y: 91, w: 10, h: 27},
        {x: 227, y: 91, w: 10, h: 27}
    ];
    const STEP = 3, MAX_DX = 14, UP = 3, DOWN = 6;
    const img = new Image();
    img.src = 'data:image/png;base64,__CAT_B64__';

    // ---- pointer tracking (mouse, pen and touch; also outside the banner)
    let pointer = null;
    const frameBox = () => {
        try { return window.frameElement.getBoundingClientRect(); }
        catch (_) { return {left: 0, top: 0}; }
    };
    const bind = (target, insideFrame) => {
        const set = (x, y) => {
            const o = insideFrame ? frameBox() : {left: 0, top: 0};
            pointer = {x: x + o.left, y: y + o.top};
        };
        target.addEventListener('pointermove',
            (e) => set(e.clientX, e.clientY), {passive: true});
        const touch = (e) => {
            if (e.touches && e.touches[0]) {
                set(e.touches[0].clientX, e.touches[0].clientY);
            }
        };
        target.addEventListener('touchstart', touch, {passive: true});
        target.addEventListener('touchmove', touch, {passive: true});
    };
    bind(window, true);
    try {
        bind(window.parent.document, false);
    } catch (_) { /* cross-origin */ }

    // ---- eyes
    let ex = 0, ey = 0, blinkUntil = 0;
    let nextBlink = performance.now() + 2500;
    const clamp = (v) => Math.max(-1, Math.min(1, v));

    function drawCat(t) {
        ctx.drawImage(img, 0, 0);
        let tx = 0, ty = 0;
        if (pointer) {
            const fb = frameBox();
            const r = cat.getBoundingClientRect();
            tx = clamp((pointer.x - fb.left - (r.left + r.width / 2)) / 320);
            ty = clamp((pointer.y - fb.top - (r.top + r.height * 0.38)) / 220);
        }
        ex += (tx * MAX_DX - ex) * 0.18;
        ey += ((ty < 0 ? ty * UP : ty * DOWN) - ey) * 0.18;
        const ox = Math.round(ex / STEP) * STEP;
        const oy = Math.round(ey / STEP) * STEP;
        if (t > nextBlink) {
            blinkUntil = t + 130;
            nextBlink = t + 2500 + Math.random() * 3500;
        }
        const blinking = t < blinkUntil;
        for (const e of EYES) {
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(e.x - 3, e.y - 1, e.w + 6, e.h + 2);
            ctx.fillStyle = '#000000';
            if (blinking) ctx.fillRect(e.x + ox, e.y + oy + e.h - 5, e.w, 4);
            else ctx.fillRect(e.x + ox, e.y + oy, e.w, e.h);
        }
    }

    // ---- falling pixel bricks (fewer on small screens)
    const bg = document.getElementById('bg');
    const bctx = bg.getContext('2d');
    const palette = ['#E5383B', '#FFFFFF', '#9b1c21', '#2b3750'];
    const bricks = [];
    function fit() {
        bg.width = stage.clientWidth;
        bg.height = stage.clientHeight;
        bricks.length = 0;
        const count = bg.width < 500 ? 28 : 55;
        for (let i = 0; i < count; i++) {
            bricks.push({
                x: Math.random() * bg.width, y: Math.random() * bg.height,
                s: 4 + Math.floor(Math.random() * 7),
                v: (0.4 + Math.random() * 1.1) * (reduceMotion ? 0 : 1),
                col: palette[Math.floor(Math.random() * palette.length)],
                a: 0.25 + Math.random() * 0.5
            });
        }
    }
    fit();
    let resizeTimer = null;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(fit, 150);
    });

    function loop(t) {
        requestAnimationFrame(loop);
        if (document.hidden) return;
        bctx.clearRect(0, 0, bg.width, bg.height);
        for (const k of bricks) {
            k.y += k.v;
            if (k.y > bg.height + 10) {
                k.y = -10;
                k.x = Math.random() * bg.width;
            }
            bctx.globalAlpha = k.a;
            bctx.fillStyle = k.col;
            bctx.fillRect(Math.round(k.x), Math.round(k.y), k.s, k.s);
        }
        bctx.globalAlpha = 1;
        if (img.complete) drawCat(t);
        const bob = reduceMotion ? 0 : Math.sin(t * 0.003) * 4;
        cat.style.transform = 'translate(-50%, calc(-50% + ' + bob + 'px))';
    }
    requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""


def inject_theme() -> None:
    st.markdown(
        f"<style>\n{FONT_IMPORT}{THEME_CSS}</style>", unsafe_allow_html=True
    )


def embed_html(markup: str, height: int) -> None:
    """Render an HTML page in an iframe.

    Newer Streamlit versions deprecate ``components.html`` in favour of
    ``st.iframe``; both keep same-origin access, which the cat needs to
    follow the pointer outside the banner.
    """
    if hasattr(st, "iframe"):
        st.iframe(markup, height=height)
    else:
        components.html(markup, height=height)


def render_hero() -> None:
    """Show the animated banner with the eye-tracking cat."""
    embed_html(HERO_TEMPLATE.replace("__CAT_B64__", CAT_B64), HERO_HEIGHT)


# ---------------------------------------------------------------------------
# Expert helpers: NVD lookups, code files, IOC extraction, encoders
# ---------------------------------------------------------------------------
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CVSS_KEYS = (
    ("cvssMetricV40", "4.0"), ("cvssMetricV31", "3.1"),
    ("cvssMetricV30", "3.0"), ("cvssMetricV2", "2.0"),
)


@st.cache_data(ttl=3600, show_spinner=False)
def _nvd_record(cve_id: str) -> dict:
    """Read one CVE from the NVD. Network errors are raised, so a failed
    lookup is never cached."""
    resp = requests.get(NVD_URL, params={"cveId": cve_id}, timeout=(5, 15))
    resp.raise_for_status()
    items = resp.json().get("vulnerabilities", [])
    if not items:
        return {"id": cve_id, "found": False}
    cve = items[0].get("cve", {})
    description = next(
        (d.get("value", "") for d in cve.get("descriptions", [])
         if d.get("lang") == "en"), "",
    )
    score, severity, vector, version = "", "", "", ""
    metrics = cve.get("metrics", {})
    for key, ver in CVSS_KEYS:
        if metrics.get(key):
            entry = metrics[key][0]
            data = entry.get("cvssData", {})
            score = data.get("baseScore", "")
            severity = data.get("baseSeverity") or entry.get("baseSeverity", "")
            vector = data.get("vectorString", "")
            version = ver
            break
    cwes = sorted({
        d.get("value", "")
        for w in cve.get("weaknesses", []) for d in w.get("description", [])
        if str(d.get("value", "")).startswith("CWE-")
    })
    return {
        "id": cve.get("id", cve_id),
        "found": True,
        "published": str(cve.get("published", ""))[:10],
        "modified": str(cve.get("lastModified", ""))[:10],
        "description": description,
        "score": score,
        "severity": severity,
        "vector": vector,
        "cvss_version": version,
        "cwes": cwes,
        "refs": [r.get("url", "") for r in cve.get("references", [])][:8],
        "kev": bool(cve.get("cisaExploitAdd")),
        "kev_due": cve.get("cisaActionDue", ""),
    }


def nvd_allowed() -> bool:
    """At most 10 NVD lookups per session and minute."""
    now = time.time()
    stamps = [t for t in st.session_state.get("_nvd_calls", [])
              if now - t < 60]
    allowed = len(stamps) < 10
    if allowed:
        stamps.append(now)
    st.session_state["_nvd_calls"] = stamps
    return allowed


def fetch_cve(cve_id: str) -> dict | None:
    """NVD record for a CVE id, or None when it cannot be reached."""
    if not nvd_allowed():
        return None
    try:
        return _nvd_record(cve_id.upper())
    except (requests.exceptions.RequestException, ValueError) as exc:
        LOGGER.warning("NVD lookup failed for %s: %s", cve_id, exc)
        return None


def cve_facts_text(records: list[dict]) -> str:
    """Verified NVD facts, written so the model treats them as ground truth."""
    lines = ["Verified data from the official NVD (treat it as ground truth "
             "and cite NVD):"]
    for rec in records:
        if rec.get("found") is None:
            lines.append(
                f"- {rec['id']}: the NVD could not be reached. Do not invent "
                "details; tell the user to verify on nvd.nist.gov."
            )
        elif not rec["found"]:
            lines.append(
                f"- {rec['id']}: not found in the NVD. Say so and do not "
                "invent details."
            )
        else:
            cwes = ", ".join(rec["cwes"]) or "none listed"
            lines.append(
                f"- {rec['id']}: CVSS v{rec['cvss_version'] or '?'} "
                f"{rec['score'] or 'n/a'} {rec['severity']} "
                f"({rec['vector'] or 'no vector'}); published "
                f"{rec['published']}; CWE: {cwes}; CISA KEV: "
                f"{'yes' if rec['kev'] else 'no'}; description: "
                f"{rec['description'][:700]}"
            )
    return "\n".join(lines)


def gather_cve_facts(text: str) -> str:
    """Look up CVE ids in the question and show the steps as they happen."""
    ids = list(dict.fromkeys(c.upper() for c in CVE_RE.findall(text)))[:3]
    if not ids:
        return ""
    records: list[dict] = []
    with st.status(
        f"Checking the official NVD for {', '.join(ids)}...", expanded=True
    ) as status:
        for cve_id in ids:
            rec = fetch_cve(cve_id)
            if rec is None:
                records.append({"id": cve_id, "found": None})
                st.write(f"{cve_id}: NVD could not be reached")
            elif not rec["found"]:
                records.append(rec)
                st.write(f"{cve_id}: not found in the NVD")
            else:
                records.append(rec)
                st.write(
                    f"{cve_id}: CVSS {rec['score'] or 'n/a'} "
                    f"{rec['severity']}"
                )
        status.update(label="NVD check finished", state="complete",
                      expanded=False)
    return cve_facts_text(records)


CODE_BLOCK_RE = re.compile(r"```([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)
CODE_EXTENSIONS = {
    "python": "py", "py": "py", "bash": "sh", "sh": "sh", "shell": "sh",
    "zsh": "sh", "powershell": "ps1", "ps1": "ps1", "bat": "bat",
    "batch": "bat", "cmd": "bat", "sql": "sql", "javascript": "js",
    "js": "js", "typescript": "ts", "ts": "ts", "json": "json",
    "yaml": "yml", "yml": "yml", "toml": "toml", "ini": "ini",
    "html": "html", "css": "css", "xml": "xml", "c": "c", "cpp": "cpp",
    "c++": "cpp", "csharp": "cs", "cs": "cs", "java": "java", "go": "go",
    "rust": "rs", "php": "php", "ruby": "rb", "yara": "yar", "svg": "svg",
    "mermaid": "mmd", "markdown": "md", "md": "md", "dockerfile": "Dockerfile",
}


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Fenced code blocks of an answer as (file name, content) pairs."""
    files: list[tuple[str, str]] = []
    for match in CODE_BLOCK_RE.finditer(text):
        code = match.group(2).rstrip("\n")
        if not code.strip():
            continue
        ext = CODE_EXTENSIONS.get(match.group(1).lower(), "txt")
        files.append((f"nashmi_snippet_{len(files) + 1}.{ext}", code + "\n"))
    return files


def zip_files(files: list[tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files:
            archive.writestr(name, content)
    return buf.getvalue()


def pdf_available() -> bool:
    try:
        import fpdf  # noqa: F401
    except ImportError:
        return False
    return True


def answer_pdf(text: str) -> bytes:
    return markdown_to_pdf(text, "Nashmi Answer") or b""


URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}\b"
)
MAX_IOC_CHARS = 200_000
DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b",
    re.IGNORECASE,
)
MD5_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
SHA1_RE = re.compile(r"\b[a-fA-F0-9]{40}\b")
FILE_LIKE_TLDS = frozenset({
    "txt", "py", "js", "exe", "dll", "log", "json", "png", "jpg", "jpeg",
    "gif", "pdf", "doc", "docx", "xls", "xlsx", "zip", "sh", "bat", "ps1",
    "html", "css", "md", "csv", "xml", "yml", "yaml", "ini", "cfg", "conf",
    "tmp", "bin", "jar", "php", "rb", "go", "rs", "cpp", "sys", "dat",
})
DEFANG_PAIRS = (
    ("hxxp", "http"), ("[.]", "."), ("(.)", "."), ("[dot]", "."),
    ("[:]", ":"), ("[@]", "@"), ("[at]", "@"),
)


def extract_all_iocs(text: str, refang: bool = True) -> dict[str, list[str]]:
    """Pull IPs, domains, URLs, e-mails, hashes and CVE ids out of any text."""
    text = text[:MAX_IOC_CHARS]
    if refang:
        for old, new in DEFANG_PAIRS:
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    domains = {
        d.lower() for d in DOMAIN_RE.findall(text)
        if d.rsplit(".", 1)[-1].lower() not in FILE_LIKE_TLDS
    }
    found = {
        "IPv4": list(extract_indicators(text).ips),
        "Domains": sorted(domains),
        "URLs": sorted(set(URL_RE.findall(text))),
        "E-mails": sorted({e.lower() for e in EMAIL_RE.findall(text)}),
        "MD5": sorted({h.lower() for h in MD5_RE.findall(text)}),
        "SHA-1": sorted({h.lower() for h in SHA1_RE.findall(text)}),
        "SHA-256": sorted({h.lower() for h in SHA256_RE.findall(text)}),
        "CVE": sorted({c.upper() for c in CVE_RE.findall(text)}),
    }
    return {kind: items for kind, items in found.items() if items}


def iocs_to_csv(iocs: dict[str, list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["type", "value"])
    for kind, items in iocs.items():
        for item in items:
            writer.writerow([kind, item])
    return buf.getvalue()


def _hex_decode(text: str) -> str:
    return bytes.fromhex("".join(text.split())).decode("utf-8", "replace")


CODECS = {
    "Base64 encode": lambda t: base64.b64encode(t.encode()).decode(),
    "Base64 decode": lambda t: base64.b64decode(
        t.strip() + "=" * (-len(t.strip()) % 4)).decode("utf-8", "replace"),
    "URL encode": lambda t: urllib.parse.quote(t, safe=""),
    "URL decode": urllib.parse.unquote,
    "Hex encode": lambda t: t.encode().hex(),
    "Hex decode": _hex_decode,
    "ROT13": lambda t: codecs.encode(t, "rot13"),
}


# ---------------------------------------------------------------------------
# Diagrams (```mermaid blocks are drawn instead of shown as code)
# ---------------------------------------------------------------------------
MERMAID_RE = re.compile(r"```mermaid[^\n]*\n(.*?)```", re.DOTALL)
MERMAID_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:">
<style>
    html, body { margin: 0; background: transparent; color: #A4B1CD;
                 font: 13px Inter, system-ui, sans-serif; }
    #box { padding: 6px; overflow: auto; text-align: center; }
    #box svg { max-width: 100%; height: auto; }
    pre { white-space: pre-wrap; text-align: left; color: #FFB020;
          font: 12px monospace; }
</style>
</head>
<body>
<div id="box">Drawing the diagram...</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"
        integrity="sha384-Wm9qzEgq4j1jEnuFK2FxKTlwuhbV2QqtGhcchvjDoKxeJ7WWAW7fysBq+1s6myfX" crossorigin="anonymous"></script>
<script>
(async () => {
    const SRC = __SRC__;
    const box = document.getElementById('box');
    const fail = (why) => {
        const pre = document.createElement('pre');
        pre.textContent = 'Could not draw this diagram (' + why + ').\n\n' + SRC;
        box.replaceChildren(pre);
    };
    if (typeof mermaid === 'undefined') { fail('drawing library not loaded'); return; }
    try {
        mermaid.initialize({
            startOnLoad: false, securityLevel: 'strict', theme: 'dark',
            themeVariables: {
                background: 'transparent', primaryColor: '#131c2b',
                primaryBorderColor: '#E5383B', primaryTextColor: '#ffffff',
                lineColor: '#A4B1CD', secondaryColor: '#0f1724',
                tertiaryColor: '#0b111c', fontFamily: 'Inter, sans-serif'
            }
        });
        const out = await mermaid.render('nashmi_diagram', SRC);
        box.innerHTML = out.svg;
    } catch (e) {
        fail(String((e && e.message) || e).slice(0, 120));
    }
})();
</script>
</body>
</html>
"""


def render_mermaid(source: str) -> None:
    """Draw one Mermaid diagram, with its source underneath."""
    lines = source.count("\n") + 1
    wide = re.search(r"\b(?:flowchart|graph)\s+(?:LR|RL)\b", source)
    height = 110 + 10 * lines if wide else 90 + 34 * lines
    height = max(130, min(640, height))
    payload = json.dumps(source).replace("</", "<\\/")
    embed_html(MERMAID_TEMPLATE.replace("__SRC__", payload), height)
    with st.expander("Diagram source"):
        st.code(source, language="text")


def render_rich(text: str) -> None:
    """Show Markdown, drawing any Mermaid blocks as diagrams."""
    position = 0
    for match in MERMAID_RE.finditer(text):
        before = text[position:match.start()]
        if before.strip():
            st.markdown(before)
        render_mermaid(match.group(1).strip())
        position = match.end()
    rest = text[position:]
    if rest.strip() or position == 0:
        st.markdown(rest)


# ---------------------------------------------------------------------------
# Copy button (one tiny iframe under every answer)
# ---------------------------------------------------------------------------
COPY_BUTTON_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    html, body { margin: 0; background: transparent; }
    button {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 9px; border: 0; border-radius: 8px; cursor: pointer;
        background: transparent; color: #6f7c98;
        font: 500 12px Inter, system-ui, sans-serif;
    }
    button:hover { background: rgba(164, 177, 205, 0.12); color: #ffffff; }
    button.done { color: #4cc38a; }
    svg { width: 14px; height: 14px; fill: none; stroke: currentColor;
          stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
</style>
</head>
<body>
<button id="b" type="button" title="Copy answer">
    <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
    <span id="t">Copy</span>
</button>
<script>
(() => {
    const TEXT = __TEXT__;
    const btn = document.getElementById('b');
    const label = document.getElementById('t');
    async function copy() {
        for (const nav of [navigator, window.parent.navigator]) {
            try { await nav.clipboard.writeText(TEXT); return true; }
            catch (_) { /* try the next way */ }
        }
        const box = document.createElement('textarea');
        box.value = TEXT;
        box.style.cssText = 'position:fixed;opacity:0';
        document.body.appendChild(box);
        box.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (_) {}
        box.remove();
        return ok;
    }
    btn.addEventListener('click', async () => {
        const ok = await copy();
        label.textContent = ok ? 'Copied' : 'Copy failed';
        btn.classList.toggle('done', ok);
        setTimeout(() => {
            label.textContent = 'Copy';
            btn.classList.remove('done');
        }, 1600);
    });
})();
</script>
</body>
</html>
"""


COMPOSER_PLACEHOLDER = "Nashmi is waiting for you..."

# Runs once inside a 1px iframe. It reaches the page (same origin) to
#   * focus the writing box when any empty part of the composer is clicked, and
#   * turn the "+" button into a menu of file types before the file dialog.
COMPOSER_FOCUS_TEMPLATE = r"""<!DOCTYPE html>
<html><body><script>
(() => {
    try {
        const doc = window.parent.document;
        if (doc.__nashmiComposer) return;
        doc.__nashmiComposer = true;

        const KINDS = [
            ['Image', 'image/*'],
            ['Audio clip', 'audio/*'],
            ['Video clip', 'video/*'],
            ['Markdown or text', '.md,.markdown,.txt'],
            ['PDF or document', '.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.odt,.rtf'],
            ['Code, logs or data', '.py,.js,.ts,.sh,.ps1,.sql,.json,.csv,.log,.yaml,.yml,.xml,.html,.css,.c,.cpp,.java,.go,.rs,.php,.ini,.conf'],
            ['Any file', '']
        ];

        const style = doc.createElement('style');
        style.textContent = `
            .nashmi-attach { position: fixed; z-index: 1000001; min-width: 210px;
                padding: 6px; border-radius: 14px; background: #131c2b;
                border: 1px solid rgba(164,177,205,0.25);
                box-shadow: 0 14px 40px rgba(0,0,0,0.55); }
            .nashmi-attach button { display: block; width: 100%; text-align: start;
                padding: 9px 12px; border: 0; border-radius: 9px; cursor: pointer;
                background: transparent; color: #A4B1CD; font: 500 14px Inter, system-ui, sans-serif; }
            .nashmi-attach button:hover, .nashmi-attach button:focus-visible {
                background: rgba(229,56,59,0.14); color: #fff; outline: none; }
            .nashmi-attach .title { padding: 6px 12px 4px; font: 11px 'JetBrains Mono', monospace;
                letter-spacing: 1.4px; text-transform: uppercase; color: #6f7c98; }`;
        doc.head.appendChild(style);

        let menu = null, opener = null;
        const closeMenu = () => {
            if (menu) { menu.remove(); menu = null; }
        };

        const openMenu = (button) => {
            closeMenu();
            opener = button;
            const wrap = button.closest('[data-testid="stChatInputFileUploadButton"]');
            const input = wrap && wrap.querySelector('input[type="file"]');
            if (!input) return;
            menu = doc.createElement('div');
            menu.className = 'nashmi-attach';
            menu.setAttribute('role', 'menu');
            const title = doc.createElement('div');
            title.className = 'title';
            title.textContent = 'Attach';
            menu.appendChild(title);
            KINDS.forEach(([label, accept]) => {
                const item = doc.createElement('button');
                item.type = 'button';
                item.setAttribute('role', 'menuitem');
                item.textContent = label;
                item.addEventListener('click', () => {
                    if (accept) input.setAttribute('accept', accept);
                    else input.removeAttribute('accept');
                    closeMenu();
                    input.click();
                });
                menu.appendChild(item);
            });
            doc.body.appendChild(menu);
            const rect = button.getBoundingClientRect();
            const width = menu.offsetWidth;
            const left = Math.max(8, Math.min(rect.right - width, window.parent.innerWidth - width - 8));
            menu.style.left = left + 'px';
            menu.style.bottom = (window.parent.innerHeight - rect.top + 10) + 'px';
            menu.querySelector('button').focus();
        };

        doc.addEventListener('click', (e) => {
            const target = e.target;
            if (!target || !target.closest) return;
            const attach = target.closest('[data-testid="stChatInputFileUploadButton"] button');
            if (attach) {
                e.preventDefault();
                e.stopImmediatePropagation();      // no file dialog yet
                if (menu) closeMenu(); else openMenu(attach);
                return;
            }
            if (menu && !target.closest('.nashmi-attach')) closeMenu();
            const box = target.closest('[data-testid="stChatInput"]');
            if (box && !target.closest('button, textarea, input, label')) {
                const area = box.querySelector('textarea');
                if (area) area.focus();
            }
        }, true);
        doc.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && menu) {
                closeMenu();
                if (opener) opener.focus();
            }
        }, true);
    } catch (_) { /* cross-origin: the normal attach button keeps working */ }
})();
</script></body></html>
"""


def copy_button(text: str) -> None:
    """Show a Copy button that puts the answer on the clipboard."""
    payload = json.dumps(text).replace("</", "<\\/")
    embed_html(COPY_BUTTON_TEMPLATE.replace("__TEXT__", payload), 34)


# ---------------------------------------------------------------------------
# Voice notes (speech to text)
# ---------------------------------------------------------------------------
GROQ_STT_MODEL = "whisper-large-v3-turbo"
STT_PROMPT = (
    "Transcribe this voice note exactly as it was spoken, in its original "
    "language (Arabic, English or a mix). Reply with the transcript only."
)


def key_for(provider: Provider) -> str:
    """Key typed in Settings, else the one from .env or Streamlit secrets."""
    typed = str(st.session_state.get(f"key_{provider.id}", "")).strip()
    return typed or get_secret(*provider.env_keys) or saved_keys().get(
        provider.id, ""
    )


def transcribe_audio(data: bytes, mime: str = "audio/wav") -> str:
    """Turn a voice note into text with Groq Whisper, or Gemini as backup."""
    groq_key = key_for(PROVIDERS_BY_ID["groq"])
    gemini = PROVIDERS_BY_ID["gemini"]
    gemini_key = key_for(gemini)
    try:
        if groq_key:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": ("voice-note.wav", data, mime)},
                data={"model": GROQ_STT_MODEL, "response_format": "json"},
                timeout=REQUEST_TIMEOUT,
            )
            if not resp.ok:
                raise ProviderError(
                    describe_http_error("Groq", resp.status_code, resp.text)
                )
            return str(resp.json().get("text", "")).strip()
        if gemini_key:
            model = gemini.models[0]
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent",
                headers={"x-goog-api-key": gemini_key},
                json={"contents": [{"parts": [
                    {"inline_data": {"mime_type": mime, "data": _b64(data)}},
                    {"text": STT_PROMPT},
                ]}]},
                timeout=REQUEST_TIMEOUT,
            )
            if not resp.ok:
                raise ProviderError(
                    describe_http_error("Gemini", resp.status_code, resp.text)
                )
            record_usage(gemini, model)
            parts = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
    except (requests.exceptions.RequestException, KeyError, IndexError,
            ValueError) as exc:
        raise ProviderError(f"Voice transcription failed: {exc}") from exc
    raise ProviderError(
        "Voice notes need a Groq or Gemini API key. Add one in Settings or "
        "in your .env file."
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Values chosen in the sidebar."""

    provider: Provider
    api_key: str
    model: str
    mode: str


@st.cache_resource
def avatar_data_uri() -> str:
    """Small cat head as a data URI for the sidebar logo."""
    buf = io.BytesIO()
    get_assistant_avatar().resize((64, 64), Image.Resampling.LANCZOS).save(
        buf, "PNG"
    )
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def md_escape(text: str) -> str:
    """Stop user text from turning into markdown in widget labels."""
    return re.sub(r"([\\`*_{}\[\]()#+!~|<>])", r"\\\1", text)


SIDEBAR_RTL_CSS = """
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { direction: rtl; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { direction: ltr; }
[data-testid="stSidebar"] [data-baseweb="select"] input { direction: rtl; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { text-align: right; }
"""

APP_POLISH_CSS = """
/* Sidebar: soft gradient, pill controls without borders */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1626 0%, #0b111c 55%, #150b14 100%) !important;
    border-right: 0 !important; box-shadow: 12px 0 40px rgba(0, 0, 0, 0.35);
}
[data-testid="stSidebar"] [data-testid^="stBaseButton"] {
    border: 0 !important; border-radius: 999px !important;
    background: rgba(255, 255, 255, 0.05) !important; box-shadow: none !important;
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover {
    background: rgba(229, 56, 59, 0.18) !important; transform: translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #ff3b3f, #c81e3a) !important;
    box-shadow: 0 8px 24px rgba(229, 56, 59, 0.4) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] { background: transparent !important; }
[data-testid="stSidebar"] .st-key-new_chat button {
    background: linear-gradient(135deg, #ff3b3f, #c81e3a) !important; color: #fff !important;
    box-shadow: 0 8px 24px rgba(229, 56, 59, 0.4) !important;
}
[data-testid="stSidebar"] .st-key-new_chat button:hover {
    background: linear-gradient(135deg, #ff5256, #d6243f) !important; transform: translateY(-1px);
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] textarea {
    border: 0 !important; border-radius: 999px !important;
    background: rgba(255, 255, 255, 0.05) !important; box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border: 0 !important; border-radius: 22px !important;
    background: rgba(255, 255, 255, 0.04) !important;
}
.st-key-side_brand [data-testid="stHorizontalBlock"] {
    flex-direction: row !important; flex-wrap: nowrap !important;
    gap: 0.3rem !important; align-items: center !important;
}
.st-key-side_brand [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 0 0 auto !important; }
.st-key-side_brand [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; }
.st-key-side_brand button { min-height: 36px; padding: 0 0.8rem; white-space: nowrap; }
/* Main area: soft rounded cards instead of outlined boxes */
[data-testid="stMain"] [data-testid^="stBaseButton"] {
    border: 0 !important; border-radius: 16px !important;
    background: rgba(255, 255, 255, 0.045) !important;
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMain"] [data-testid^="stBaseButton"]:hover {
    background: rgba(229, 56, 59, 0.14) !important; transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(229, 56, 59, 0.14);
}
[data-testid="stMain"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #ff3b3f, #c81e3a) !important;
}
"""


def render_nav(box) -> str:
    """Logo, New chat button and the track list; returns the active track."""
    mode = current_mode()
    with box:
        with st.container(key="side_brand"):
            logo, out = st.columns([3, 2], vertical_alignment="center",
                                   gap="small")
            logo.markdown(
                f'<div class="brand"><img src="{avatar_data_uri()}" alt="">'
                f'<span><span class="prompt">&gt;</span> Nashmi</span></div>',
                unsafe_allow_html=True,
            )
            if is_public() and current_user():
                out.button(tt("Sign out", "تسجيل الخروج"), key="side_signout",
                           on_click=sign_out, type="tertiary",
                           icon=":material/logout:")
        st.button(
            tt("New chat", "محادثة جديدة"), key="new_chat", icon=":material/add:",
            on_click=on_new_chat, disabled=mode in (MODE_SOC, MODE_TOOLS, MODE_EXAM),
        )
        return st.selectbox(
            "Track", MODES, key="track", label_visibility="collapsed",
            on_change=close_account,
            format_func=lambda name: tt(name, TRACK_AR.get(name, name)),
        )


def render_projects_panel() -> None:
    """Create projects and edit the instructions of the selected one."""
    st.text_input("Project name", key="new_project_name",
                  placeholder="e.g. Final-year thesis")
    st.text_area(
        "Instructions (optional)", key="new_project_instr", height=80,
        placeholder="e.g. Always use Bash, cite NIST, keep answers short",
    )
    st.button("Create project", key="create_project",
              on_click=on_create_project)
    project_id = st.session_state.get("project", "")
    project = next(
        (p for p in get_store()["projects"] if p["id"] == project_id), None
    )
    if project:
        st.divider()
        st.caption(f"Selected project: {project['name']}")
        text = st.text_area(
            "Instructions of this project", value=project["instructions"],
            key=f"edit_instr_{project_id}", height=100,
        )
        if text != project["instructions"]:
            project["instructions"] = text[:2000]
            persist_store()
        st.button("Delete project", key="delete_project",
                  on_click=on_delete_project, icon=":material/delete:")


def render_prefs_form() -> None:
    """Answer length and language."""
    store = get_store()
    prefs = store["prefs"]
    st.selectbox(
        "Answer length", DETAIL_CHOICES, key="pref_detail",
        index=DETAIL_CHOICES.index(prefs["detail"]),
    )
    st.selectbox(
        "Answer language", LANGUAGE_CHOICES, key="pref_language",
        index=LANGUAGE_CHOICES.index(prefs["language"]),
    )
    chosen = {
        "detail": st.session_state["pref_detail"],
        "language": st.session_state["pref_language"],
    }
    if chosen != prefs:
        store["prefs"] = chosen
        persist_store()


def render_login_buttons(key_prefix: str) -> None:
    """One button per sign-in provider that is set up."""
    for provider in auth_providers():
        st.button(
            tr("continue_google") if provider == "google" else LOGIN_LABELS.get(provider, "Sign in"),
            key=f"{key_prefix}_login_{provider}",
            on_click=start_login, args=(provider,),
            icon=None if provider == "google" else ":material/login:",
            width="stretch" if key_prefix in ("login", "signup") else "content",
        )


def key_label(provider: Provider) -> str:
    if lang() == "ar":
        return (f"احصل على مفتاح {provider.short} المجاني" if provider.free
                else f"احصل على مفتاح API لـ {provider.short}")
    return (f"Get your free {provider.short} key" if provider.free
            else f"Get your {provider.short} API key")


def render_key_help(provider: Provider) -> None:
    """Explain the bring-your-own-key model and link to the key page."""
    saving = (
        " Signed-in accounts can also save it, encrypted."
        if accounts_enabled() else ""
    )
    st.info(
        "**Bring your own API key.** Nashmi runs on your own key, so your "
        "chats never use anybody else's quota. Gemini and Groq have free "
        "keys; OpenAI and Claude are billed to your own account.\n\n"
        f"1. Create a {'free ' if provider.free else ''}{provider.short} "
        "API key with the button below.\n"
        "2. Open **Model and API key** in the sidebar, pick the provider "
        "and paste it.\n"
        f"3. Start chatting.{saving}",
        icon=":material/key:",
    )
    st.link_button(
        key_label(provider), provider.key_url,
        icon=":material/open_in_new:",
    )


def render_user_card(box) -> None:
    """Profile card at the bottom of the sidebar."""
    with box:
        user = current_user() if account_id() else None
        if user:
            picture = user["picture"]
            avatar = (
                f'<img src="{html.escape(picture, quote=True)}" alt="" '
                'referrerpolicy="no-referrer">'
                if picture.startswith("https://") else
                f'<span class="avatar-fallback">'
                f'{html.escape(user["name"][:1].upper())}</span>'
            )
            st.markdown(
                f'<div class="user-card">{avatar}<div><b>'
                f'{html.escape(user["name"])}</b><br><span>'
                f'{html.escape(user["email"])}</span></div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="user_actions"):
                first, second = st.columns(2)
                first.button(tt("Account", "الحساب"), key="open_account",
                             on_click=open_account,
                             icon=":material/account_circle:")
                if is_admin():
                    second.button(tt("Admin", "المشرف"), key="open_admin",
                                  on_click=open_admin,
                                  icon=":material/monitoring:")
        elif accounts_enabled():
            st.caption(tt("Guest: chats are not saved. Sign in to keep them.",
                          "ضيف: المحادثات لا تُحفظ. سجّل الدخول لتحتفظ بها."))
            render_login_buttons("side")
            st.button(tt("Usage and settings", "الاستخدام والإعدادات"), key="open_account",
                      on_click=open_account,
                      icon=":material/account_circle:", type="tertiary")
        else:
            st.button(tt("Usage and settings", "الاستخدام والإعدادات"), key="open_account",
                      on_click=open_account,
                      icon=":material/account_circle:", type="tertiary")


def render_side_language(box) -> None:
    """Language dropdown for the signed-in app (same choice as on the public pages)."""
    with box:
        st.session_state["side_lang"] = lang()
        st.selectbox(tt("Language", "اللغة"), ["en", "ar"], key="side_lang",
                     format_func=lambda c: {"en": "English", "ar": "العربية"}[c],
                     on_change=on_app_lang_pick)


def render_keys_panel(user_id: int | None) -> None:
    """Saved API keys, one row per provider."""
    db = get_db()
    st.subheader("API keys")
    st.caption(
        "Nashmi never uses a key of ours. Paste your own: it is used only to "
        "call the provider you pick."
    )
    can_save = bool(user_id and db and db.cipher.enabled)
    saved = saved_keys()
    for provider in PROVIDER_LIST:
        with st.container(border=True):
            st.markdown(f"**{provider.short}**")
            if provider.id in saved:
                st.success("Saved in your account (encrypted)",
                           icon=":material/lock:")
                st.button("Remove saved key", key=f"rm_key_{provider.id}",
                          on_click=on_remove_key, args=(provider.id,),
                          icon=":material/delete:")
            elif can_save:
                st.text_input(
                    f"{provider.short} API key", type="password",
                    key=f"acct_key_{provider.id}",
                    label_visibility="collapsed",
                    placeholder=f"Paste your {provider.short} key",
                )
                st.button("Save encrypted", key=f"save_key_{provider.id}",
                          on_click=on_save_key, args=(provider.id,),
                          icon=":material/lock:")
            st.link_button(
                key_label(provider), provider.key_url,
                icon=":material/open_in_new:",
            )
    if not can_save:
        st.info(
            "Sign in to save keys in your account, or paste a key in the "
            "sidebar for this session only."
            if not user_id else
            "Saving keys is off because the server has no encryption key."
        )


def render_usage_panel(user_id: int | None) -> None:
    """Request counts: today, last 14 days, and totals."""
    import pandas as pd

    history = st.session_state.get("saved_history", {})
    chats = [c for items in history.values() for c in items if c["messages"]]
    messages = sum(len(c["messages"]) for c in chats)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if user_id:
        rows = get_db().usage_summary(user_id, 14)
    else:
        rows = []
        for key, entry in _load_usage().items():
            if isinstance(entry, dict) and entry.get("day") and ":" in key:
                provider_id, model = key.split(":", 1)
                rows.append({
                    "day": entry["day"], "provider": provider_id,
                    "model": model, "requests": entry.get("used", 0),
                })
    today_total = sum(r["requests"] for r in rows if r["day"] == today)
    first, second, third, fourth = st.columns(4)
    first.metric("Requests today", today_total)
    second.metric("Last 14 days", sum(r["requests"] for r in rows))
    third.metric("Chats", len(chats))
    fourth.metric("Messages", messages)
    if not rows:
        st.info("No requests yet. Your usage will show up here.")
        return
    frame = pd.DataFrame(rows)
    chart = frame.pivot_table(
        index="day", columns="provider", values="requests", aggfunc="sum"
    ).fillna(0)
    st.markdown("**Requests per day**")
    st.bar_chart(chart)
    st.markdown("**By model**")
    st.dataframe(
        frame[["day", "provider", "model", "requests"]],
        hide_index=True,
    )
    if not user_id:
        st.caption("Guest usage is counted for this session only. Sign in to "
                   "keep a history.")


def render_data_panel(user_id: int | None, user: dict | None) -> None:
    """Export, import and delete."""
    db = get_db()
    st.caption(
        "Public demo: your chats live only in this browser session. Export "
        "them to keep a copy." if is_public() and not user_id
        else "Your chats and projects are saved in your account." if user_id
        else "Your chats and projects are saved on this computer."
    )
    if user_id:
        try:
            payload = json.dumps(db.export_all(user_id), ensure_ascii=False,
                                 indent=2, default=str)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Could not build the export")
            payload = "{}"
    else:
        payload = json.dumps(
            {"history": st.session_state.get("saved_history", {}),
             "store": get_store()},
            ensure_ascii=False, indent=2,
        )
    st.download_button(
        "Export my data (.json)", payload, "nashmi_data.json",
        "application/json", on_click="ignore", key="dl_all_data",
        icon=":material/download:",
    )
    upload = st.file_uploader("Import data (.json)", type="json",
                              key="import_data")
    st.button("Import now", key="do_import", on_click=on_import_data,
              disabled=upload is None)
    st.divider()
    confirm = st.checkbox("I want to delete all my chats and projects",
                          key="confirm_wipe")
    st.button("Delete all chats and projects", key="wipe_all",
              on_click=on_wipe_all, disabled=not confirm,
              icon=":material/delete_forever:")
    if user_id and user:
        st.divider()
        st.markdown("**Delete my account**")
        st.caption("Removes your profile, chats, projects, usage history and "
                   "saved keys from the database. This cannot be undone.")
        typed = st.text_input("Type your e-mail to confirm",
                              key="confirm_delete_email")
        st.button(
            "Delete my account", key="delete_account",
            on_click=on_delete_account, icon=":material/person_remove:",
            disabled=not user["email"]
            or typed.strip().lower() != user["email"].lower(),
        )


def render_security_notes(user_id: int | None) -> None:
    db = get_db()
    encrypted = bool(db and db.cipher.enabled)
    st.markdown(
        "**How your data is protected**\n"
        "- Sign-in is handled by Google or Apple. Nashmi never sees or "
        "stores a password, and any verification code (2-step, passkey, "
        "SMS) is done by them.\n"
        f"- Chats, project instructions and saved keys are encrypted in the "
        f"database: **{'on' if encrypted else 'off on this server'}**.\n"
        "- Each account only ever reads and writes its own rows.\n"
        "- Your API key is never shown again, never logged, and never "
        "written to the chat history.\n"
        "- You can export everything, or delete it, at any time."
    )


def render_account(settings: Settings) -> None:
    """Account page: profile, settings, usage, data and security."""
    user_id = account_id()
    user = current_user() if user_id else None
    st.button("Back to chat", key="close_account", on_click=close_account,
              icon=":material/arrow_back:")
    st.markdown('<div class="empty-title">Account</div>',
                unsafe_allow_html=True)
    if st.session_state.get("_account_deleted"):
        st.success("Your account and data were deleted. Sign out to finish.")
        st.button("Sign out", key="sign_out_deleted", on_click=sign_out,
                  icon=":material/logout:")
        return
    profile, prefs_tab, usage_tab, data_tab = st.tabs(
        ["Profile", "Settings", "Usage", "Data and security"]
    )
    with profile:
        if user:
            record = get_db().get_user(user_id) or {}
            st.markdown(f"### {html.escape(user['name'])}")
            st.write(user["email"] or "No e-mail shared by the provider")
            since = str(record.get("created_at", ""))[:10]
            st.caption(
                f"Signed in with {user['provider'].title()}"
                + (f" - member since {since}" if since else "")
            )
            st.button("Sign out", key="sign_out_page", on_click=sign_out,
                      icon=":material/logout:")
        else:
            st.markdown("### Guest")
            if accounts_enabled():
                st.write("Sign in to keep your chats, projects, usage "
                         "history and saved keys.")
                render_login_buttons("page")
            else:
                st.write("Accounts are not set up on this server. Your chats "
                         "are kept on this computer.")
    with prefs_tab:
        render_prefs_form()
        st.divider()
        st.subheader("Projects")
        st.caption("Group chats and give them standing instructions. Once "
                   "you have a project, a picker appears in the sidebar.")
        render_projects_panel()
        st.divider()
        render_keys_panel(user_id)
    with usage_tab:
        render_usage_panel(user_id)
    with data_tab:
        render_data_panel(user_id, user)
        st.divider()
        render_security_notes(user_id)


def render_settings(box, mode: str):
    """Provider, key and model in one tidy panel. It opens by itself only
    when no key is available yet, and folds away once one is."""
    with box:
        labels = list(PROVIDERS)
        st.session_state.setdefault(
            "provider_choice", PROVIDERS_BY_ID[DEFAULT_PROVIDER_ID].label)
        current = PROVIDERS[st.session_state["provider_choice"]]
        has_key = bool(
            str(st.session_state.get(f"key_{current.id}", "")).strip()
            or get_secret(*current.env_keys)
            or saved_keys().get(current.id, "")
        )
        with st.expander(
            tt(f"AI model: {current.short}", f"نموذج الذكاء الاصطناعي: {current.short}"),
            icon=":material/tune:",
            expanded=is_public() and not has_key,
        ):
            label = st.selectbox(tt("Provider", "المزوّد"), labels, key="provider_choice")
            provider = PROVIDERS[label]

            typed_key = st.text_input(
                tt(f"{provider.short} API key", f"مفتاح API لـ {provider.short}"),
                type="password",
                key=f"key_{provider.id}",
                placeholder=tt("Paste your key here", "الصق مفتاحك هنا"),
                help=(tt("Used for this session only. Signed-in accounts can "
                         "save it, encrypted, from the Account page.",
                         "يُستخدم لهذه الجلسة فقط. يمكن للحسابات المسجّلة حفظه "
                         "مشفّراً من صفحة الحساب.")
                      if is_public() else None),
            )
            env_key = get_secret(*provider.env_keys)
            stored_key = saved_keys().get(provider.id, "")
            api_key = typed_key.strip() or env_key or stored_key
            if env_key and not typed_key.strip():
                st.caption(tt("Key loaded from .env. Type here to override it.",
                              "تم تحميل المفتاح من .env. اكتب هنا لتجاوزه."))
            elif stored_key and not typed_key.strip():
                st.caption(tt("Using the key saved in your account.",
                              "يُستخدم المفتاح المحفوظ في حسابك."))
            st.link_button(
                key_label(provider), provider.key_url,
                icon=":material/key:", width="stretch",
            )

            if provider.id == "openrouter":
                options = openrouter_free_models() or list(OPENROUTER_FALLBACK)
            elif provider.id in ("gemini", "groq", "openai", "anthropic"):
                options = live_models(provider.id, api_key) or list(
                    provider.models
                )
            else:
                options = list(provider.models)
            chosen = st.selectbox(tt("Model", "الموديل"), options, key=f"model_{provider.id}")
            custom = ""
            if st.toggle(tt("Use a custom model name", "استخدم اسم موديل مخصص"),
                         key=f"custom_on_{provider.id}"):
                custom = st.text_input(
                    "Custom model", key=f"custom_{provider.id}",
                    label_visibility="collapsed",
                    placeholder=tt("exact model id", "معرّف الموديل بالضبط"),
                )
            st.caption(tt(provider.note, PROVIDER_NOTE_AR.get(provider.id, provider.note)))
        usage_slot = st.empty()
    return Settings(provider, api_key, custom.strip() or chosen, mode), usage_slot


def render_topline(settings: Settings) -> None:
    """One quiet line: current track, model and whether a key is loaded."""
    ok = bool(settings.api_key)
    st.markdown(
        f'<div class="topline"><span class="t-track">'
        f'{html.escape(tt(settings.mode, TRACK_AR.get(settings.mode, settings.mode)))}</span><span class="t-sep">/</span>'
        f'<span class="t-model">{html.escape(settings.model)}</span>'
        f'<span class="t-key {"ok" if ok else "bad"}">'
        f'{tt("key loaded", "مفتاح محمّل") if ok else tt("no API key", "لا يوجد مفتاح API")}</span></div>',
        unsafe_allow_html=True,
    )


def render_chat_sidebar(box, mode: str, chats: list[dict]) -> int:
    """Search and recent chats for this track; returns the active index."""
    key = radio_key(mode)
    store = get_store()
    ids = [""] + [p["id"] for p in store["projects"]]
    names = {p["id"]: p["name"] for p in store["projects"]}
    if st.session_state.get("project", "") not in ids:
        st.session_state["project"] = ""
    with box:
        project = st.session_state.get("project", "")
        if len(ids) > 1:  # the picker only appears once a project exists
            project = st.selectbox(
                "Project", ids, key="project", label_visibility="collapsed",
                format_func=lambda i: names.get(i, tt("All chats", "كل المحادثات")),
            )
        if project and not any(c.get("project") == project for c in chats):
            chats.insert(0, new_chat_record(project))
            st.session_state[key] = 0
            persist(st.session_state.saved_history)
        query = st.text_input(
            "Search chats",
            key=search_key(mode),
            placeholder=tt("Search chats", "ابحث في المحادثات"),
            label_visibility="collapsed",
        )
        found = search_chats(chats, query)
        if project:
            found = [(i, h) for i, h in found
                     if chats[i].get("project") == project]
        with st.container(key="chat_head"):
            title, more = st.columns([4, 1])
            title.markdown(
                f'<div class="side-label">'
                f'{tt("Results", "النتائج") if query.strip() else tt("Recent", "الأخيرة")}</div>',
                unsafe_allow_html=True,
            )
            with more:
                with st.popover(" ", icon=":material/more_horiz:"):
                    st.button(
                        tt("Delete this chat", "حذف هذه المحادثة"), key=f"delete_chat_{mode}",
                        icon=":material/delete:", on_click=on_delete_chat,
                    )
        if not found:
            st.caption(tt("No chats match your search.", "لا توجد محادثات مطابقة لبحثك."))
            return active_index(mode, len(chats))

        indices = [i for i, _ in found]
        hits = dict(found)
        if st.session_state.get(key) not in indices:
            st.session_state[key] = indices[0]

        def chat_label(i: int) -> str:
            label = md_escape(chats[i]["title"])
            if query.strip() and hits[i]:
                label += f"  ({hits[i]})"
            return label

        index = st.radio(
            "Chats", indices, key=key, format_func=chat_label,
            label_visibility="collapsed",
        )
        snippets = message_snippets(chats[index], query)
        if snippets:
            with st.expander(tt(f"Matches in this chat ({len(snippets)})",
                                f"نتائج في هذه المحادثة ({len(snippets)})"),
                             expanded=True):
                for snippet in snippets:
                    st.caption(snippet)
    return index


def render_references(name: str) -> None:
    """Collapsed list of trusted primary sources for the track."""
    if name == MODE_CHAT:
        return
    refs = TRACK_REFS.get(name, ()) + COMMON_REFS
    with st.expander("Trusted references", icon=":material/menu_book:"):
        st.markdown("\n".join(f"- [{title}]({url})" for title, url in refs))


def render_empty_state(track: Track) -> str | None:
    """Start screen of a fresh chat; returns a clicked suggestion's prompt."""
    st.markdown(
        f'<div class="empty-title">{html.escape(track.name)}</div>'
        f'<div class="empty-blurb">{html.escape(track.blurb)}</div>',
        unsafe_allow_html=True,
    )
    selected = None
    for number, (label, prompt) in enumerate(track.actions):
        if st.button(label, key=f"quick_{track.name}_{number}"):
            selected = prompt
    render_references(track.name)
    return selected


def export_menu(text: str, uid: str) -> None:
    """Download buttons for one answer: Markdown, text, PDF and code files."""
    st.download_button(
        "Markdown (.md)", text, "nashmi_answer.md", "text/markdown",
        key=f"dl_md_{uid}", on_click="ignore", icon=":material/description:",
    )
    st.download_button(
        "Plain text (.txt)", text, "nashmi_answer.txt", "text/plain",
        key=f"dl_txt_{uid}", on_click="ignore", icon=":material/notes:",
    )
    if pdf_available():
        st.download_button(
            "PDF (.pdf)", data=lambda: answer_pdf(text),
            file_name="nashmi_answer.pdf", mime="application/pdf",
            key=f"dl_pdf_{uid}", on_click="ignore",
            icon=":material/picture_as_pdf:",
        )
    files = extract_code_blocks(text)
    for number, (name, code) in enumerate(files):
        st.download_button(
            name, code, name, "text/plain", key=f"dl_code_{uid}_{number}",
            on_click="ignore", icon=":material/code:",
        )
    if len(files) > 1:
        st.download_button(
            "All code files (.zip)", zip_files(files), "nashmi_code.zip",
            "application/zip", key=f"dl_zip_{uid}", on_click="ignore",
            icon=":material/folder_zip:",
        )


def render_answer_actions(text: str, uid: str) -> None:
    """Copy button and Export menu under an answer."""
    with st.container(key=f"actions_{uid}"):
        copy_col, export_col = st.columns(2)
        with copy_col:
            copy_button(text)
        with export_col:
            with st.popover("Export", icon=":material/download:"):
                export_menu(text, uid)


def render_messages(messages: list[dict]) -> None:
    for number, msg in enumerate(messages):
        assistant = msg["role"] == "assistant"
        avatar = get_assistant_avatar() if assistant else get_user_avatar()
        with st.chat_message(msg["role"], avatar=avatar):
            if assistant:
                render_rich(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("files"):
                st.caption("Attached: " + ", ".join(msg["files"]))
            if assistant:
                render_answer_actions(msg["content"], f"m{number}")


def read_chat_input(placeholder: str):
    """Read the composer (text, attachments, voice note) in a version-safe way.

    Returns ``(text, files, audio)``.
    """
    try:
        value = st.chat_input(
            placeholder, accept_file="multiple", file_type=None,
            accept_audio=True, max_chars=MAX_PROMPT_CHARS,
        )
    except TypeError:  # Streamlit without voice notes
        try:
            value = st.chat_input(
                placeholder, accept_file="multiple", file_type=None,
                max_chars=MAX_PROMPT_CHARS,
            )
        except TypeError:  # Streamlit older than 1.43 has no attachments
            value = st.chat_input(placeholder)
    if not value:
        return "", [], None
    if isinstance(value, str):
        return value.strip(), [], None
    text = (getattr(value, "text", "") or "").strip()
    files = list(getattr(value, "files", []) or [])
    return text, files, getattr(value, "audio", None)


def warn_skipped(prepared: PreparedFiles) -> None:
    if prepared.skipped:
        st.warning("Not sent: " + "; ".join(prepared.skipped))


def with_wait_notice(stream: Iterator[str]) -> Iterator[str]:
    """Show a waiting notice until the first piece of the reply arrives."""
    holder = st.empty()
    holder.markdown(
        "⏳ *Nashmi is preparing a response for you...*  \n"
        "*نشمي عم يجهزلك الرد...*"
    )
    try:
        for chunk in stream:
            holder.empty()
            yield chunk
            break
        yield from stream
    finally:
        holder.empty()


def stream_to_text(stream: Iterator[str]) -> str:
    """Render a stream live and return the complete text."""
    result = st.write_stream(stream)
    return result if isinstance(result, str) else "".join(map(str, result))


def handle_submission(
    settings: Settings, chat: dict, prompt: str, files: list,
    extra_names: tuple[str, ...] = (),
) -> None:
    """Send one user message and stream the answer into the chat."""
    if not settings.api_key:
        render_key_help(settings.provider)
        return

    prepared = prepare_attachments(files, settings.provider)
    warn_skipped(prepared)
    messages = chat["messages"]
    names = [f.name for f in files] + list(extra_names)
    messages.append({"role": "user", "content": prompt, "files": names})
    if chat.get("title") == NEW_CHAT_TITLE:
        chat["title"] = make_title(prompt)
    persist(st.session_state.saved_history)

    with st.chat_message("user", avatar=get_user_avatar()):
        st.markdown(prompt)
        if names:
            st.caption("Attached: " + ", ".join(names))

    reply = ""
    with st.chat_message("assistant", avatar=get_assistant_avatar()):
        failed = False
        try:
            content = wrap_prompt_as_security_expert(prompt, settings.mode)
            facts = gather_cve_facts(prompt)
            if facts:
                content += "\n\n" + facts
            outgoing = messages[:-1] + [{"role": "user", "content": content}]
            reply = stream_to_text(with_wait_notice(generate_stream(
                settings.provider, settings.api_key, settings.model,
                full_system_prompt(settings.mode, chat.get("project", "")),
                outgoing, prepared,
            )))
        except ProviderError as exc:
            failed = True
            note_quota_error(settings.provider, settings.model, str(exc))
            show_provider_error(exc)
        except Exception as exc:  # noqa: BLE001 - keep the app alive
            failed = True
            LOGGER.exception("Unexpected error while generating a reply")
            show_internal_error("Request failed.", exc)

        if reply.strip():
            messages.append({"role": "assistant", "content": reply})
            render_answer_actions(reply, f"m{len(messages) - 1}")
        else:
            messages.pop()  # remove the unanswered question so it can retry
            if not failed:
                st.warning(
                    "No reply was received. Try again or pick another model."
                )
        persist(st.session_state.saved_history)

    if reply.strip() and MERMAID_RE.search(reply):
        st.rerun()  # redraw the finished answer with its diagrams


def render_chat_track(settings: Settings, chats_box) -> None:
    mode = settings.mode
    track = TRACKS_BY_NAME[mode]
    chats = ensure_history(mode)
    index = render_chat_sidebar(chats_box, mode, chats)
    chat = chats[index]

    start_screen = st.empty()
    quick_prompt = None
    if chat["messages"]:
        render_messages(chat["messages"])
    else:
        with start_screen.container():
            quick_prompt = render_empty_state(track)
            if not settings.api_key:
                render_key_help(settings.provider)

    typed, files, audio = read_chat_input(COMPOSER_PLACEHOLDER)
    voice_text = ""
    if audio is not None:
        with st.spinner("Transcribing your voice note..."):
            try:
                voice_text = transcribe_audio(
                    audio.getvalue(), getattr(audio, "type", "") or "audio/wav"
                )
            except ProviderError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - keep the app alive
                LOGGER.exception("Voice transcription failed")
                show_internal_error("Could not transcribe the voice note.", exc)
        if not voice_text and not typed and not files:
            return

    prompt = quick_prompt or "\n\n".join(p for p in (typed, voice_text) if p)
    if not prompt and files:
        prompt = "Analyze the attached file(s)."
    if prompt:
        start_screen.empty()
        handle_submission(
            settings, chat, prompt, [] if quick_prompt else files,
            extra_names=("voice note",) if voice_text and not quick_prompt
            else (),
        )


def render_soc_track(settings: Settings) -> None:
    """Upload logs/media and stream a SOC report."""
    provider = settings.provider
    track = TRACKS_BY_NAME[MODE_SOC]
    st.markdown(
        f'<div class="empty-title">{html.escape(track.name)}</div>'
        f'<div class="empty-blurb">{html.escape(track.blurb)}</div>',
        unsafe_allow_html=True,
    )
    render_references(MODE_SOC)
    uploads = st.file_uploader(
        "Logs, screenshots, videos, PCAP/CSV/JSON or any file",
        type=None,
        accept_multiple_files=True,
        key="soc_multi_uploader",
    )
    notes = st.text_input(
        "Notes or a specific question about the files (optional)", ""
    )
    with st.expander("VirusTotal scan (optional, free key)"):
        vt_typed = st.text_input(
            "VirusTotal API key", type="password", key="vt_key_input",
        )
        allow_upload = st.checkbox(
            "Upload files VirusTotal does not know yet (they become visible "
            "to the VirusTotal community)",
            value=False,
        )
    vt_key = vt_typed.strip() or get_secret("VT_API_KEY", "VIRUSTOTAL_API_KEY")
    if not vt_key:
        st.caption(
            "Add VT_API_KEY to .env (free key from virustotal.com) to scan "
            "every uploaded file with VirusTotal."
        )

    for upload in uploads or []:
        if upload.type and upload.type.startswith("image/"):
            st.image(upload, caption=upload.name, width=350)
        elif upload.type and upload.type.startswith("video/"):
            st.video(upload)

    if not (uploads and st.button("Start security analysis",
                                  key="btn_start_soc")):
        return
    if not settings.api_key:
        st.error(
            f"Please enter your {provider.short} API key in the sidebar "
            "Settings or put it in your .env file."
        )
        return

    vt_results: list[dict] = []
    if vt_key:
        with st.spinner("Scanning files with VirusTotal..."):
            for upload in uploads:
                vt_results.append(
                    vt_scan_file(vt_key, upload.name, upload.getvalue(),
                                 allow_upload)
                )
        st.subheader("VirusTotal scan")
        for res in vt_results:
            show_vt_result(res)
    if vt_results:
        notes = f"{notes}\n\n{vt_summary_text(vt_results)}".strip()

    prepared = prepare_attachments(uploads, provider)
    warn_skipped(prepared)
    found = extract_indicators(prepared.raw_text)

    unique, external, hashes = st.columns(3)
    unique.metric("Unique IPs", len(found.ips))
    external.metric("External IPs", len(found.external_ips))
    hashes.metric("SHA-256 Hashes", len(found.hashes))

    prompt = SOC_PROMPT_TEMPLATE.format(
        ips=list(found.external_ips), hashes=list(found.hashes),
        notes=notes,
    )
    report = ""
    try:
        with st.spinner("Nashmi is scanning indicators and media and "
                        "writing the report..."):
            report = stream_to_text(generate_stream(
                provider, settings.api_key, settings.model,
                full_system_prompt(settings.mode),
                [{"role": "user", "content": prompt}], prepared,
            ))
    except ProviderError as exc:
        note_quota_error(provider, settings.model, str(exc))
        show_provider_error(exc)
    except Exception as exc:  # noqa: BLE001 - keep the app alive
        LOGGER.exception("Unexpected error while writing the SOC report")
        show_internal_error("Request failed.", exc)

    if report.strip():
        render_answer_actions(report, "soc_report")
        st.success("Your security report is ready.")
        pdf_bytes = markdown_to_pdf(report, "Nashmi SOC Report")
        if pdf_bytes:
            st.download_button(
                label="Download report (PDF)",
                data=pdf_bytes,
                file_name="Nashmi_SOC_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Run: py -m pip install fpdf2 uharfbuzz  to get PDF reports.")
            st.download_button(
                label="Download report (Markdown)",
                data=report,
                file_name="Nashmi_SOC_Report.md",
                mime="text/markdown",
            )


def show_cve(rec: dict) -> None:
    """Display one NVD record."""
    st.subheader(rec["id"])
    score, severity, kev = st.columns(3)
    score.metric(f"CVSS v{rec['cvss_version'] or '?'}", rec["score"] or "n/a")
    severity.metric("Severity", rec["severity"] or "n/a")
    kev.metric("CISA KEV", "Yes" if rec["kev"] else "No")
    st.markdown(rec["description"] or "_No description._")
    if rec["vector"]:
        st.code(rec["vector"], language=None)
    st.caption(
        f"Published {rec['published']} - last modified {rec['modified']}"
        + (f" - CWE: {', '.join(rec['cwes'])}" if rec["cwes"] else "")
    )
    if rec["kev"] and rec["kev_due"]:
        st.warning(f"Listed as exploited in the wild. CISA due date: "
                   f"{rec['kev_due'][:10]}")
    if rec["refs"]:
        st.markdown("**References**\n" + "\n".join(
            f"- {url}" for url in rec["refs"] if url
        ))
    st.markdown(
        f"Source: [NVD](https://nvd.nist.gov/vuln/detail/{rec['id']})"
    )
    st.download_button(
        "Download record (.json)", json.dumps(rec, indent=2),
        f"{rec['id']}.json", "application/json", on_click="ignore",
        key="dl_cve_json", icon=":material/download:",
    )


def render_toolbox() -> None:
    """Non-AI helpers an analyst needs all day: CVE, hashes, encoders, IOCs."""
    track = TRACKS_BY_NAME[MODE_TOOLS]
    st.markdown(
        f'<div class="empty-title">{html.escape(track.name)}</div>'
        f'<div class="empty-blurb">{html.escape(track.blurb)}</div>',
        unsafe_allow_html=True,
    )
    tab_cve, tab_hash, tab_ioc = st.tabs(
        ["CVE lookup", "Hash and encode", "IOC extractor"]
    )

    with tab_cve:
        with st.form("cve_form", border=False):
            cve_id = st.text_input("CVE ID", placeholder="CVE-2021-44228")
            go = st.form_submit_button("Look up in the NVD")
        if go:
            cve_id = cve_id.strip().upper()
            if not CVE_RE.fullmatch(cve_id):
                st.session_state.pop("cve_result", None)
                st.error("Enter an ID like CVE-2021-44228.")
            else:
                with st.spinner("Asking the NVD..."):
                    rec = fetch_cve(cve_id)
                if rec is None:
                    st.session_state.pop("cve_result", None)
                    st.error("The NVD could not be reached (network or rate "
                             "limit). Try again in a minute.")
                else:
                    st.session_state["cve_result"] = rec
        result = st.session_state.get("cve_result")
        if result is not None:
            if result["found"]:
                show_cve(result)
            else:
                st.warning(f"{result['id']} was not found in the NVD.")

    with tab_hash:
        text = st.text_area("Text", height=110, key="tool_text")
        upload = st.file_uploader("...or a file", key="tool_file")
        data = upload.getvalue() if upload else text.encode("utf-8")
        if data:
            for algo in ("md5", "sha1", "sha256", "sha512"):
                st.markdown(f"**{algo.upper()}**")
                st.code(hashlib.new(algo, data).hexdigest(), language=None)
        st.divider()
        operation = st.selectbox("Encode or decode the text above",
                                 list(CODECS))
        if text:
            try:
                st.code(CODECS[operation](text), language=None)
            except (ValueError, UnicodeError) as exc:
                st.error(f"Could not do that: {exc}")

    with tab_ioc:
        pasted = st.text_area(
            "Paste logs, alerts, an e-mail or a report", height=180,
            key="ioc_text", max_chars=MAX_IOC_CHARS,
        )
        refang = st.checkbox("Undo defanging (hxxp, [.], [at])", value=True)
        if pasted.strip():
            iocs = extract_all_iocs(pasted, refang)
            if not iocs:
                st.info("No indicators found.")
            else:
                total = sum(len(v) for v in iocs.values())
                st.caption(f"{total} indicators in {len(iocs)} groups")
                for kind, items in iocs.items():
                    with st.expander(f"{kind} ({len(items)})", expanded=True):
                        st.code("\n".join(items), language=None)
                left, right = st.columns(2)
                left.download_button(
                    "Download (.csv)", iocs_to_csv(iocs), "iocs.csv",
                    "text/csv", on_click="ignore", key="dl_ioc_csv",
                    icon=":material/table:",
                )
                right.download_button(
                    "Download (.json)", json.dumps(iocs, indent=2),
                    "iocs.json", "application/json", on_click="ignore",
                    key="dl_ioc_json", icon=":material/data_object:",
                )
    render_references(MODE_SOC)


# ---------------------------------------------------------------------------
# Practice Exam: generate a full exam from a subject or study files, take it
# on screen (with drag and drop), get it graded, and export everything.
# ---------------------------------------------------------------------------
EXAM_UPLOAD_TYPES = ["pdf", "pptx", "docx", "txt", "md", "png", "jpg", "jpeg",
                     "webp"]
EXAM_MAX_FILE_BYTES = 30 * 1024 * 1024
EXAM_MAX_IMAGES = 6
EXAM_FORMAT = "nashmi-exam-1"
EXAM_SECTION_BATCH = 8

_DND_DIR = Path(__file__).resolve().parent / "components" / "dnd"
_dnd_component = (
    components.declare_component("nashmi_dnd", path=str(_DND_DIR))
    if (_DND_DIR / "index.html").exists() else None
)


def exam_state() -> dict:
    return st.session_state.setdefault("exam", {"stage": "setup"})


def _clear_exam_widgets() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(("ex_", "exv_", "exdnd_")):
            del st.session_state[key]


def on_new_exam() -> None:
    _clear_exam_widgets()
    st.session_state["exam"] = {"stage": "setup"}


def on_start_exam() -> None:
    exam = exam_state()
    exam["started"] = time.time()
    exam["stage"] = "taking"


def on_retake_exam() -> None:
    _clear_exam_widgets()
    exam = exam_state()
    for key in ("answers", "results"):
        exam.pop(key, None)
    exam["started"] = time.time()
    exam["stage"] = "taking"


def exam_questions(exam: dict) -> list[dict]:
    return [q for s in exam["data"]["sections"] for q in s["questions"]]


def read_exam_answer(q: dict):
    """The answer currently given to a question (None when untouched)."""
    qid, qtype = q["id"], q["type"]
    if qtype in ("order", "match", "image_label"):
        return st.session_state.get(f"exv_{qid}")
    if qtype == "fill":
        return [st.session_state.get(f"ex_{qid}_{i}", "")
                for i in range(len(q["answers"]))]
    return st.session_state.get(f"ex_{qid}")


def exam_answered(q: dict, answer) -> bool:
    if answer in (None, "", [], {}):
        return False
    if q["type"] in ("fill", "match"):
        return any(str(a).strip() for a in answer)
    return True


def finish_exam() -> None:
    exam = exam_state()
    exam["answers"] = {q["id"]: read_exam_answer(q)
                       for q in exam_questions(exam)}
    exam["stage"] = "grading"


def ask_model(settings: Settings, system: str, user: str,
              media: list | None = None) -> str:
    """One non-streamed call to the chosen provider."""
    prepared = PreparedFiles(media=list(media or []))
    stream = generate_stream(
        settings.provider, settings.api_key, settings.model, system,
        [{"role": "user", "content": user}], prepared,
    )
    return "".join(stream)


def ask_json(settings: Settings, system: str, user: str,
             media: list | None = None):
    """Ask for JSON; if the reply is not JSON, ask once more."""
    data = nashmi_exam.extract_json(ask_model(settings, system, user, media))
    if data is None:
        data = nashmi_exam.extract_json(ask_model(
            settings, system,
            user + "\n\nYour previous reply was not valid JSON. Reply with "
                   "ONLY the JSON, nothing else.", media,
        ))
    return data


def load_exam_material(uploads) -> tuple[str, list[dict], list[str]]:
    """Text and pictures of the uploaded study files (cached per session)."""
    cache = st.session_state.setdefault("exam_material", {})
    texts: list[str] = []
    images: list[dict] = []
    warnings: list[str] = []
    for upload in uploads or []:
        data = upload.getvalue()
        if len(data) > EXAM_MAX_FILE_BYTES:
            warnings.append(f"{upload.name}: larger than 30 MB, skipped.")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[digest] = nashmi_exam.extract_material(upload.name, data)
        text, found, warning = cache[digest]
        texts.append(text)
        images.extend(found)
        if warning:
            warnings.append(warning)
    return "\n\n".join(texts)[:nashmi_exam.MAX_MATERIAL_CHARS], images, warnings


def allowed_exam_types(settings: Settings, chosen: list[str],
                       has_images: bool) -> list[str]:
    can_see = has_images and settings.provider.accepts("image/png")
    pool = chosen or list(nashmi_exam.QUESTION_TYPES)
    return [t for t in pool
            if t not in nashmi_exam.IMAGE_TYPES or can_see]


def build_exam(settings: Settings, spec, images: dict, allowed: list[str],
               verify: bool, log) -> dict:
    """Plan the exam, write every section, then double-check the key."""
    rng = random.Random()
    media = [MediaPart(name=i, mime=im["mime"], data=im["data"])
             for i, im in images.items()]
    log("Planning the exam")
    plan = nashmi_exam.clean_plan(
        ask_json(settings, *nashmi_exam.plan_prompt(spec, allowed)),
        spec, allowed,
    )
    total = sum(s["count"] for s in plan["sections"])
    log(f"Blueprint: {total} questions in {len(plan['sections'])} sections")

    sections, previous, warnings = [], [], []
    number = 0
    for section in plan["sections"]:
        log(f"Writing: {section['title']}")
        questions: list[dict] = []
        attempts = 0
        while len(questions) < section["count"] and attempts < 3:
            batch = min(section["count"] - len(questions), EXAM_SECTION_BATCH)
            data = ask_json(
                settings,
                *nashmi_exam.section_prompt(spec, section, batch, previous),
                media if section["type"] in nashmi_exam.IMAGE_TYPES else None,
            )
            items = data.get("questions") if isinstance(data, dict) else data
            attempts += 1
            if not isinstance(items, list):
                continue
            taken = 0
            for raw in items:
                try:
                    q = nashmi_exam.clean_question(
                        raw, section["type"], list(images),
                        section["marks_each"], rng,
                    )
                except ValueError as exc:
                    warnings.append(
                        f"Dropped one question in '{section['title']}': {exc}.")
                    continue
                if nashmi_exam.norm(q["question"]) in {
                        nashmi_exam.norm(p) for p in previous}:
                    continue
                number += 1
                q["id"] = f"q{number}"
                questions.append(q)
                previous.append(q["question"])
                taken += 1
                if taken >= batch:
                    break
        if len(questions) < section["count"]:
            warnings.append(f"'{section['title']}' has {len(questions)} of "
                            f"{section['count']} planned questions.")
        if questions:
            sections.append({"title": section["title"],
                             "type": section["type"],
                             "questions": questions})
    if not sections:
        raise ValueError(
            "The model did not return any usable question. Try again, pick "
            "another model, or reduce the number of question types."
        )
    exam = {
        "title": plan["title"],
        "instructions": plan["instructions"],
        "duration_minutes": plan["duration_minutes"],
        "sections": sections,
    }
    exam["total_marks"] = sum(
        q["marks"] for s in sections for q in s["questions"])
    disputes: dict[str, str] = {}
    closed = [q for s in sections for q in s["questions"]
              if q["type"] in nashmi_exam.VERIFIABLE]
    if verify and closed:
        log("Double-checking the answer key")
        try:
            needs_images = any(q["type"] == "image_mcq" for q in closed)
            disputes = nashmi_exam.find_disputes(
                closed,
                ask_json(settings, *nashmi_exam.verify_prompt(closed),
                         media if needs_images else None),
            )
            if disputes:
                log(f"{len(disputes)} question(s) flagged for review")
        except ProviderError as exc:
            warnings.append(f"The double-check could not run: {exc}")
    return {"data": exam, "disputes": disputes, "warnings": warnings}


def exam_to_json(exam: dict) -> str:
    images = {
        image_id: {"mime": im["mime"],
                   "b64": base64.b64encode(im["data"]).decode("ascii")}
        for image_id, im in exam.get("images", {}).items()
    }
    return json.dumps({"format": EXAM_FORMAT, "exam": exam["data"],
                       "images": images}, ensure_ascii=False)


def exam_from_json(raw: bytes) -> dict:
    """Rebuild an exam from a saved file, re-checking every field."""
    if len(raw) > 30 * 1024 * 1024:
        raise ValueError("This exam file is too large.")
    try:
        blob = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("This is not a valid exam file.") from exc
    if not isinstance(blob, dict) or blob.get("format") != EXAM_FORMAT:
        raise ValueError("This is not a Nashmi exam file.")
    images: dict[str, dict] = {}
    for image_id, entry in list((blob.get("images") or {}).items())[
            :EXAM_MAX_IMAGES]:
        if (not isinstance(entry, dict)
                or entry.get("mime") not in ("image/png", "image/jpeg")
                or not re.fullmatch(r"img\d{1,2}", str(image_id))):
            continue
        try:
            data = base64.b64decode(str(entry.get("b64", "")), validate=True)
            if len(data) > nashmi_exam.IMAGE_MAX_BYTES:
                continue
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception:  # noqa: BLE001 - a bad picture is skipped
            continue
        images[image_id] = {"label": image_id, "mime": entry["mime"],
                            "data": data}
    data = nashmi_exam.rebuild_exam(blob.get("exam"), list(images))
    return {"data": data, "images": images, "disputes": {}, "warnings": []}


def annotate_image(data: bytes, points: list[dict]) -> bytes:
    """The picture with the numbered points drawn on it."""
    from PIL import ImageFont

    img = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    radius = max(12, min(width, height) // 28)
    try:
        font = ImageFont.load_default(size=radius)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    for point in points:
        cx, cy = point["x"] / 1000 * width, point["y"] / 1000 * height
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=(229, 56, 59), outline="white", width=2)
        draw.text((cx, cy), str(point["id"]), fill="white", font=font,
                  anchor="mm")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def _blank_safe(text: str) -> str:
    """Keep ____ blanks from turning into Markdown rules."""
    return re.sub(r"_{3,}", lambda m: "\\_" * len(m.group()), text)


# ------------------------------------------------------------------ setup
def render_exam_setup(settings: Settings) -> None:
    track = TRACKS_BY_NAME[MODE_EXAM]
    st.markdown(
        f'<div class="empty-title">{html.escape(track.name)}</div>'
        f'<div class="empty-blurb">{html.escape(track.blurb)}</div>',
        unsafe_allow_html=True,
    )
    if not settings.api_key:
        render_key_help(settings.provider)

    subject = st.text_input(
        "Subject or topic",
        placeholder="e.g. Network security: firewalls, IDS and VPNs",
    )
    uploads = st.file_uploader(
        "Study material (optional): slides, lecture notes, PDFs, pictures",
        accept_multiple_files=True, type=EXAM_UPLOAD_TYPES, key="exam_files",
    )
    material, found, warnings = load_exam_material(uploads)
    for warning in warnings:
        st.warning(warning)
    if uploads and material:
        st.caption(f"Read {len(material):,} characters from your material.")

    chosen_images: dict[str, dict] = {}
    if found:
        st.markdown("**Pictures found in your material.** Tick the ones the "
                    "exam may use for image questions.")
        columns = st.columns(4)
        for number, image in enumerate(found):
            tag = hashlib.sha1(image["data"]).hexdigest()[:10]
            with columns[number % 4]:
                st.image(image["data"], caption=image["label"], width=150)
                if st.checkbox("Use", value=number < EXAM_MAX_IMAGES,
                               key=f"exam_img_{tag}") \
                        and len(chosen_images) < EXAM_MAX_IMAGES:
                    chosen_images[f"img{len(chosen_images) + 1}"] = image
        if not settings.provider.accepts("image/png"):
            st.info(f"{settings.provider.short} cannot look at pictures, so "
                    "image questions are off. Pick Gemini in the sidebar to "
                    "get them.")

    left, right = st.columns(2)
    language = left.selectbox("Exam language", list(nashmi_exam.LANGUAGES))
    difficulty = right.selectbox("Difficulty", nashmi_exam.DIFFICULTIES,
                                 index=3)
    style = st.selectbox("Exam style", list(nashmi_exam.STYLES))
    first, second, third = st.columns(3)
    n_questions = first.number_input("Questions (0 = Nashmi decides)", 0,
                                     nashmi_exam.MAX_QUESTIONS, 0)
    total_marks = second.number_input("Total marks (0 = Nashmi decides)", 0,
                                      500, 0, step=5)
    duration = third.number_input("Minutes (0 = Nashmi decides)", 0, 300, 0,
                                  step=5)
    types = st.multiselect(
        "Question types (leave empty and Nashmi picks the mix)",
        list(nashmi_exam.QUESTION_TYPES),
        format_func=nashmi_exam.QUESTION_TYPES.get,
    )
    verify = st.checkbox(
        "Double-check the answer key (one extra request, recommended)",
        value=True,
    )
    allowed = allowed_exam_types(settings, types, bool(chosen_images))
    estimate = 1 + min(len(types) or 5, 6) + (1 if verify else 0) + 1
    st.caption(
        f"An exam uses about {estimate} requests of your "
        f"{settings.provider.short} key (planning, one per section, the "
        "double-check and grading)."
        + ("" if settings.provider.free
           else " They are billed to your account.")
    )

    left, right = st.columns([3, 2])
    ready = bool(settings.api_key and (subject.strip() or material))
    if left.button("Generate exam", type="primary", disabled=not ready,
                   key="exam_generate"):
        spec = nashmi_exam.ExamSpec(
            subject=subject.strip() or "the uploaded study material",
            language=language, difficulty=difficulty, style=style,
            n_questions=int(n_questions), total_marks=int(total_marks),
            duration=int(duration), types=types, material_text=material,
            image_ids=list(chosen_images),
        )
        try:
            with st.status("Building your exam...", expanded=True) as status:
                built = build_exam(settings, spec, chosen_images, allowed,
                                   verify, st.write)
                status.update(label="Your exam is ready", state="complete",
                              expanded=False)
        except ProviderError as exc:
            note_quota_error(settings.provider, settings.model, str(exc))
            show_provider_error(exc)
            return
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - keep the app alive
            LOGGER.exception("Exam generation failed")
            show_internal_error("Could not build the exam.", exc)
            return
        built["images"] = {
            image_id: image for image_id, image in chosen_images.items()
            if any(q.get("image") == image_id
                   for q in (x for s in built["data"]["sections"]
                             for x in s["questions"]))
        }
        _clear_exam_widgets()
        st.session_state["exam"] = {**built, "stage": "ready"}
        st.rerun()

    with st.expander("Load a saved exam (.json)"):
        saved = st.file_uploader("Saved exam", type="json",
                                 key="exam_json_upload")
        if saved is not None and st.button("Open this exam", key="exam_open"):
            try:
                loaded = exam_from_json(saved.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                _clear_exam_widgets()
                st.session_state["exam"] = {**loaded, "stage": "ready"}
                st.rerun()


# ------------------------------------------------------------------ ready
def render_exam_downloads(exam: dict, results: dict | None = None) -> None:
    data = exam["data"]
    first, second, third = st.columns(3)
    first.download_button(
        "Exam paper (.md)", nashmi_exam.exam_to_markdown(data),
        "exam_paper.md", "text/markdown", on_click="ignore",
        key="dl_exam_md", icon=":material/description:",
    )
    if pdf_available():
        second.download_button(
            "Exam paper (.pdf)",
            data=lambda: markdown_to_pdf(
                nashmi_exam.exam_to_markdown(data), data["title"]) or b"",
            file_name="exam_paper.pdf", mime="application/pdf",
            on_click="ignore", key="dl_exam_pdf",
            icon=":material/picture_as_pdf:",
        )
        third.download_button(
            "Answer key (.pdf)",
            data=lambda: markdown_to_pdf(
                nashmi_exam.exam_to_markdown(data, with_key=True),
                data["title"] + " - answer key") or b"",
            file_name="answer_key.pdf", mime="application/pdf",
            on_click="ignore", key="dl_exam_key",
            icon=":material/key:",
        )
    st.download_button(
        "Save this exam to retake later (.json)", exam_to_json(exam),
        "nashmi_exam.json", "application/json", on_click="ignore",
        key="dl_exam_json", icon=":material/save:",
    )
    if exam.get("images"):
        st.caption("PDF and Markdown exports contain the text only. Pictures "
                   "stay in the saved .json exam.")


def render_exam_ready(settings: Settings) -> None:
    exam = exam_state()
    data = exam["data"]
    st.markdown(f'<div class="empty-title">{html.escape(data["title"])}</div>',
                unsafe_allow_html=True)
    if data.get("instructions"):
        st.markdown(data["instructions"])
    count = len(exam_questions(exam))
    first, second, third = st.columns(3)
    first.metric("Questions", count)
    second.metric("Total marks", f"{data['total_marks']:g}")
    third.metric("Time", f"{data['duration_minutes']} min")
    for warning in exam.get("warnings", []):
        st.warning(warning)
    if exam.get("disputes"):
        st.warning(
            f"{len(exam['disputes'])} question(s) were flagged by the "
            "double-check. They are marked in your results; review the key "
            "before trusting them."
        )
    left, right = st.columns(2)
    left.button("Start the exam", type="primary", on_click=on_start_exam,
                key="exam_start", icon=":material/play_arrow:")
    right.button("Make another exam", on_click=on_new_exam, key="exam_new1",
                 icon=":material/refresh:")
    render_exam_downloads(exam)


# ----------------------------------------------------------------- taking
def render_exam_question(q: dict, number: int, images: dict) -> None:
    qid, qtype = q["id"], q["type"]
    key = f"ex_{qid}"
    with st.container(border=True):
        st.markdown(f"**Question {number}** · {q['marks']:g} mark"
                    f"{'s' if q['marks'] != 1 else ''}")
        if q.get("context"):
            st.markdown("> " + q["context"].replace("\n", "\n> "))
        st.markdown(_blank_safe(q["question"]))
        if q.get("code"):
            st.code(q["code"], language=q.get("language") or None)
        if qtype == "image_mcq" and q["image"] in images:
            st.image(images[q["image"]]["data"], width=420)

        if qtype in ("mcq", "image_mcq", "code"):
            st.radio("Answer", range(len(q["options"])), index=None, key=key,
                     format_func=lambda i, o=q["options"]: o[i],
                     label_visibility="collapsed")
        elif qtype == "multi":
            st.multiselect("Select all that apply", range(len(q["options"])),
                           key=key,
                           format_func=lambda i, o=q["options"]: o[i],
                           placeholder="Choose one or more")
        elif qtype == "truefalse":
            st.radio("Answer", [True, False], index=None, key=key,
                     horizontal=True, label_visibility="collapsed",
                     format_func=lambda v: "True" if v else "False")
        elif qtype == "fill":
            for i in range(len(q["answers"])):
                st.text_input(f"Blank {i + 1}", key=f"{key}_{i}")
        elif qtype == "numeric":
            unit = f" ({q['unit']})" if q.get("unit") else ""
            st.text_input(f"Your answer{unit}", key=key)
        elif qtype in ("short", "essay"):
            st.text_area("Your answer", key=key, max_chars=8000,
                         height=100 if qtype == "short" else 220)
        elif _dnd_component is None:
            st.error("The drag-and-drop component is missing "
                     "(components/dnd/index.html).")
        elif qtype == "order":
            st.session_state[f"exv_{qid}"] = dnd(
                "order", f"exdnd_{qid}", items=q["shuffled"],
                initial=st.session_state.get(f"exv_{qid}"),
            )
        elif qtype == "match":
            st.session_state[f"exv_{qid}"] = dnd(
                "match", f"exdnd_{qid}",
                left=[p["left"] for p in q["pairs"]], bank=q["bank"],
                initial=st.session_state.get(f"exv_{qid}"),
            )
        elif qtype == "image_label":
            image = images[q["image"]]
            if "uri" not in image:
                image["uri"] = (f"data:{image['mime']};base64,"
                                + base64.b64encode(image["data"]).decode())
            st.session_state[f"exv_{qid}"] = dnd(
                "label", f"exdnd_{qid}", image=image["uri"],
                image_id=q["image"],
                points=[{"id": p["id"], "x": p["x"], "y": p["y"]}
                        for p in q["points"]],
                bank=q["bank"], initial=st.session_state.get(f"exv_{qid}"),
            )


def dnd(mode: str, key: str, **kwargs):
    """Draw one drag-and-drop widget; returns its value (None until used)."""
    return _dnd_component(mode=mode, key=key, default=None, **kwargs)


@st.fragment(run_every="1s")
def exam_timer(deadline: float) -> None:
    remaining = int(deadline - time.time())
    if remaining <= 0:
        st.session_state["exam_time_up"] = True
        st.rerun()
    minutes, seconds = divmod(remaining, 60)
    st.markdown(
        f'<div class="exam-timer{" low" if remaining < 300 else ""}">'
        f'{minutes:02d}:{seconds:02d}</div>',
        unsafe_allow_html=True,
    )


def render_exam_taking(settings: Settings) -> None:
    exam = exam_state()
    data = exam["data"]
    images = exam.get("images", {})
    questions = exam_questions(exam)
    minutes = int(data["duration_minutes"] or 0)
    deadline = exam["started"] + minutes * 60 if minutes else 0
    if deadline and (time.time() >= deadline
                     or st.session_state.pop("exam_time_up", False)):
        st.warning("Time is up. Your answers were submitted.")
        finish_exam()
        st.rerun()

    head, clock = st.columns([4, 1])
    head.markdown(f'<div class="empty-title">{html.escape(data["title"])}'
                  '</div>', unsafe_allow_html=True)
    with clock:
        if deadline:
            exam_timer(deadline)
    if data.get("instructions"):
        st.markdown(data["instructions"])
    answered = sum(exam_answered(q, read_exam_answer(q)) for q in questions)
    st.progress(answered / len(questions),
                text=f"{answered} of {len(questions)} answered")

    tabs = st.tabs([s["title"] for s in data["sections"]])
    number = 0
    for tab, section in zip(tabs, data["sections"]):
        with tab:
            for q in section["questions"]:
                number += 1
                render_exam_question(q, number, images)
    st.divider()
    unanswered = len(questions) - answered
    if unanswered:
        st.caption(f"{unanswered} question(s) still unanswered.")
    st.button("Submit exam", type="primary", on_click=finish_exam,
              key="exam_submit", icon=":material/task_alt:")


# ---------------------------------------------------------------- grading
def grade_exam(settings: Settings, exam: dict) -> dict:
    """Closed questions are graded here; written ones by the model."""
    answers = exam["answers"]
    per_question: dict[str, dict] = {}
    pending = []
    for q in exam_questions(exam):
        answer = answers.get(q["id"])
        given = nashmi_exam.answer_text(q, answer)
        if q["type"] in nashmi_exam.SUBJECTIVE:
            per_question[q["id"]] = {"score": 0.0, "given": given,
                                     "note": "Not graded"}
            if given != "(no answer)":
                pending.append({
                    "id": q["id"], "question": q["question"],
                    "marks": q["marks"], "model_answer": q["model_answer"],
                    "rubric": q.get("rubric", []), "student_answer": given,
                })
            continue
        fraction, note = nashmi_exam.grade_objective(q, answer)
        per_question[q["id"]] = {
            "score": round(fraction * q["marks"], 2),
            "given": given, "note": note,
        }
    if pending:
        marks = {p["id"]: p["marks"] for p in pending}
        try:
            grades = nashmi_exam.parse_grades(
                ask_json(settings, *nashmi_exam.grading_prompt(pending)),
                marks,
            )
        except ProviderError as exc:
            LOGGER.warning("Written answers could not be graded: %s", exc)
            grades = {}
        for qid, grade in grades.items():
            full = marks[qid]
            per_question[qid].update(
                score=grade["score"], feedback=grade["feedback"],
                note=("Correct" if grade["score"] >= full else
                      "Partly correct" if grade["score"] > 0 else "Wrong"),
            )
        for qid in marks.keys() - grades.keys():
            per_question[qid].update(
                note="Needs your review",
                feedback="The model could not grade this answer. Compare it "
                         "with the model answer below.",
                ungraded=True,
            )
    sections = []
    for section in exam["data"]["sections"]:
        sections.append({
            "title": section["title"],
            "score": sum(per_question[q["id"]]["score"]
                         for q in section["questions"]),
            "total": sum(q["marks"] for q in section["questions"]),
        })
    score = sum(s["score"] for s in sections)
    total = sum(s["total"] for s in sections)
    return {"score": round(score, 2), "total": total,
            "percent": score / total * 100 if total else 0,
            "per_question": per_question, "sections": sections}


def render_exam_grading(settings: Settings) -> None:
    exam = exam_state()
    with st.status("Grading your exam...", expanded=True) as status:
        st.write("Marking the closed questions")
        if any(q["type"] in nashmi_exam.SUBJECTIVE
               for q in exam_questions(exam)):
            st.write("Asking Nashmi to read your written answers")
        exam["results"] = grade_exam(settings, exam)
        status.update(label="Graded", state="complete")
    exam["stage"] = "results"
    st.rerun()


# ---------------------------------------------------------------- results
def render_exam_results(settings: Settings) -> None:
    exam = exam_state()
    data, results = exam["data"], exam["results"]
    images = exam.get("images", {})
    percent = results["percent"]
    verdict = ("Excellent" if percent >= 85 else "Good" if percent >= 70
               else "Pass" if percent >= 50 else "Needs more practice")
    st.markdown(f'<div class="empty-title">{html.escape(data["title"])} - '
                f'results</div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    first.metric("Score", f"{results['score']:g} / {results['total']:g}")
    second.metric("Percent", f"{percent:.0f}%")
    third.metric("Result", verdict)
    st.progress(min(percent / 100, 1.0))
    if any(item.get("ungraded") for item in results["per_question"].values()):
        st.warning("Some written answers could not be graded automatically "
                   "and count as 0 until you review them.")
    st.markdown("**By section**")
    st.dataframe(
        [{"Section": s["title"], "Score": f"{s['score']:g} / {s['total']:g}"}
         for s in results["sections"]], hide_index=True,
    )

    number = 0
    for section in data["sections"]:
        st.markdown(f"#### {section['title']}")
        for q in section["questions"]:
            number += 1
            item = results["per_question"][q["id"]]
            colour = {"Correct": "green", "Partly correct": "orange"}.get(
                item["note"], "red")
            label = (f"Question {number} - :{colour}[{item['note']}] "
                     f"({item['score']:g}/{q['marks']:g})")
            with st.expander(label):
                st.markdown(_blank_safe(q["question"]))
                if q.get("code"):
                    st.code(q["code"], language=q.get("language") or None)
                if q["type"] == "image_label":
                    st.image(annotate_image(images[q["image"]]["data"],
                                            q["points"]), width=480)
                elif q["type"] == "image_mcq" and q["image"] in images:
                    st.image(images[q["image"]]["data"], width=420)
                st.markdown(f"**Your answer:** {item['given']}")
                st.markdown(f"**Correct answer:** "
                            f"{nashmi_exam.correct_answer_text(q)}")
                if item.get("feedback"):
                    st.info(item["feedback"])
                if q.get("explanation"):
                    st.markdown(f"**Why:** {q['explanation']}")
                if q.get("reference"):
                    st.caption(f"Source: {q['reference']}")
                if q["id"] in exam.get("disputes", {}):
                    st.warning(exam["disputes"][q["id"]])

    st.divider()
    left, right = st.columns(2)
    left.button("Retake this exam", on_click=on_retake_exam, key="exam_retake",
                icon=":material/replay:")
    right.button("Make another exam", on_click=on_new_exam, key="exam_new2",
                 icon=":material/refresh:", type="primary")
    pdf_data = nashmi_exam.exam_to_markdown(data, results=results)
    if pdf_available():
        st.download_button(
            "Results with answers (.pdf)",
            data=lambda: markdown_to_pdf(pdf_data, data["title"]) or b"",
            file_name="exam_results.pdf", mime="application/pdf",
            on_click="ignore", key="dl_exam_results",
            icon=":material/picture_as_pdf:",
        )
    render_exam_downloads(exam, results)


def render_exam_track(settings: Settings) -> None:
    """Router for the Practice Exam track."""
    if nashmi_exam is None:
        st.error("The exam engine (nashmi_exam.py) is missing.")
        return
    stage = exam_state().get("stage", "setup")
    if stage == "ready":
        render_exam_ready(settings)
    elif stage == "taking":
        render_exam_taking(settings)
    elif stage == "grading":
        render_exam_grading(settings)
    elif stage == "results":
        render_exam_results(settings)
    else:
        render_exam_setup(settings)
    render_references("Practice Exam")


# ---------------------------------------------------------------------------
# Landing page, sign-in page and admin screen
# ---------------------------------------------------------------------------
# Shared by the landing banner and the scroll story. makeLook gives a cat eyes
# that follow the pointer (also the pointer over the rest of the page) and
# glance around by themselves when nobody is moving one.
CAT_JS = """function sizeFrame(h) {
    // Streamlit keeps its own height on the wrapper around this frame, so the wrapper
    // is sized here too, and again after every re-render
    var fe = null;
    try { fe = window.frameElement; } catch (err) { }
    if (!fe) { return; }
    fe.style.height = h + 'px'; fe.setAttribute('height', h);
    var bx = fe.parentElement;
    if (bx) {
        var hp = h + 'px';
        bx.style.setProperty('height', hp, 'important');
        bx.style.setProperty('min-height', '0', 'important');
        bx.style.setProperty('max-height', hp, 'important');
        bx.style.setProperty('flex', '0 0 ' + hp, 'important');
    }
}

function releaseFrameBoxOnExit() {
    // the page resizes Streamlit's wrapper around this frame; when the frame goes
    // away (another page reuses that wrapper) the wrapper must get its size back
    var boxEl = null;
    try { boxEl = window.frameElement && window.frameElement.parentElement; } catch (err) { }
    window.addEventListener('pagehide', function () {
        if (!boxEl) { return; }
        ['height', 'min-height', 'max-height', 'flex'].forEach(function (k) {
            boxEl.style.removeProperty(k);
        });
    });
}

function trackPointer(cb) {
    var own = function (e) { cb(e.clientX, e.clientY); };
    document.addEventListener('pointermove', own, { passive: true });
    document.addEventListener('pointerdown', own, { passive: true });
    try {
        var fe = window.frameElement, pd = window.parent.document;
        var outer = function (e) {
            var f = fe.getBoundingClientRect();
            cb(e.clientX - f.left, e.clientY - f.top);
        };
        pd.addEventListener('pointermove', outer, { passive: true });
        pd.addEventListener('pointerdown', outer, { passive: true });
    } catch (err) { }
}

function makeLook(el, getCenter) {
    var L = { x: 0, y: 0 }, tgt = { x: 0, y: 0 }, last = -1e9, nextWander = 0;
    trackPointer(function (px, py) {
        var c = getCenter();
        var rx = Math.max(140, el.clientWidth / 2);
        var ry = Math.min(320, Math.max(140, el.clientHeight / 2));
        tgt.x = Math.max(-1, Math.min(1, (px - c.x) / rx * 1.8));
        tgt.y = Math.max(-1, Math.min(1, (py - c.y) / ry * 1.8));
        last = performance.now();
    });
    L.update = function (now) {
        if (now - last > 2600 && now > nextWander) {
            tgt.x = (Math.random() * 2 - 1) * 0.9;
            tgt.y = (Math.random() * 2 - 1) * 0.5;
            nextWander = now + 1300 + Math.random() * 1700;
        }
        L.x += (tgt.x - L.x) * 0.16;
        L.y += (tgt.y - L.y) * 0.16;
    };
    return L;
}
"""

# The cat on the line in the story. It has no shemagh. Like a desktop-pet cat it
# runs along the line in the direction it travels (front, back or side view),
# and when it stops it sits, scratches, gets tired and finally falls asleep.
WALKER_JS = """var WW = 40, WHT = 46;
var K = '#0b0b10', WH = '#ffffff', GR = '#d3d8e3', RD = '#E5383B', PK = '#f4a3a8';

function catPx(ctx) {
    return function (x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); };
}

function catHead(px, B, mode, blink, ox, oy, nx, ny) {
    var y, x0, x1;
    function pb(x, y2, w, h, c) { px(x, y2 + B, w, h, c); }
    for (y = 12; y <= 27; y++) {
        var ins = (y <= 13 || y >= 26) ? 2 : (y === 14 || y === 25 ? 1 : 0);
        x0 = 7 + ins; x1 = 32 - ins;
        pb(x0 - 1, y, 1, 1, K); pb(x1 + 1, y, 1, 1, K);
        pb(x0, y, x1 - x0 + 1, 1, WH);
    }
    pb(9, 11, 22, 1, K); pb(9, 28, 22, 1, K);
    var ew = [1, 2, 4, 5, 7], up = mode === 'alert' ? 1 : 0;
    for (y = 6; y <= 10; y++) {
        var w = ew[y - 6], e0 = 11 - Math.floor(w / 2), f0 = 28 - Math.floor(w / 2);
        pb(e0 - 1, y - up, 1, 1, K); pb(e0, y - up, w, 1, WH); pb(e0 + w, y - up, 1, 1, K);
        pb(f0 - 1, y - up, 1, 1, K); pb(f0, y - up, w, 1, WH); pb(f0 + w, y - up, 1, 1, K);
    }
    pb(11, 5 - up, 1, 1, K); pb(28, 5 - up, 1, 1, K);
    pb(11, 8 - up, 1, 2, PK); pb(28, 8 - up, 1, 2, PK);
    var eh = blink ? 1 : 5, ey = blink ? 18 : 16;
    if (mode === 'alert') { eh = 6; ey = 15; }
    if (mode === 'tired') { eh = 2; ey = 19; }
    pb(13 + ox, ey + oy, 2, eh, K); pb(25 + ox, ey + oy, 2, eh, K);
    pb(19 + nx, 22 + ny, 2, 1, RD);
    if (mode === 'tired') { pb(19 + nx, 24 + ny, 2, 2, K); }
    else { pb(18 + nx, 24 + ny, 1, 1, K); pb(19 + nx, 25 + ny, 2, 1, K); pb(21 + nx, 24 + ny, 1, 1, K); }
    pb(3 + nx, 22, 4, 1, GR); pb(4 + nx, 24, 3, 1, GR);
    pb(33 + nx, 22, 4, 1, GR); pb(33 + nx, 24, 3, 1, GR);
}

function drawCatSit(ctx, t, lx, ly, mode) {
    var px = catPx(ctx), k, y;
    ctx.clearRect(0, 0, WW, WHT);
    var B = (mode === 'tired') ? 1 : (Math.sin(t * 1.6) > 0.75 ? 1 : 0);
    function pb(x, y2, w, h, c) { px(x, y2 + B, w, h, c); }
    var blink = (t % 3.6) > 3.45;
    for (k = 0; k < 11; k++) {
        var u = k / 10, wg = Math.sin(t * 3.2 + k * 0.5) * u * 2.2;
        px(28 + 8 * Math.sin(u * 2.0) + wg - 1, 43 - 15 * u + B - 1, 4, 4, K);
    }
    for (k = 0; k < 11; k++) {
        var v = k / 10, wg2 = Math.sin(t * 3.2 + k * 0.5) * v * 2.2;
        px(28 + 8 * Math.sin(v * 2.0) + wg2, 43 - 15 * v + B, 2, 2, WH);
    }
    for (y = 29; y <= 43; y++) {
        var hw = y < 32 ? 6 : (y < 37 ? 7 : 9);
        pb(20 - hw - 1, y, 1, 1, K); pb(20 + hw, y, 1, 1, K);
        pb(20 - hw, y, hw * 2, 1, WH);
    }
    pb(10, 44, 20, 1, K);
    pb(16, 34, 8, 6, GR);
    pb(13, 40, 5, 4, WH);
    pb(12, 40, 1, 5, K); pb(18, 40, 1, 5, K); pb(15, 42, 1, 2, K);
    if (mode === 'scratch') {                          // right paw scratches the head
        var sw = Math.floor(t * 7) % 2;
        pb(27 + sw, 34, 5, 3, K); pb(28 + sw, 34, 3, 3, WH);
        pb(28 + sw, 24, 5, 11, K); pb(29 + sw, 24, 3, 10, WH);
        pb(28 + sw, 22, 5, 3, K); pb(29 + sw, 22, 3, 3, WH);
    } else {
        pb(22, 40, 5, 4, WH);
        pb(21, 40, 1, 5, K); pb(27, 40, 1, 5, K); pb(24, 42, 1, 2, K);
    }
    pb(13, 28, 14, 2, RD); pb(13, 28, 1, 2, K); pb(26, 28, 1, 2, K);
    catHead(px, B, mode, blink, Math.round((lx || 0) * 2), Math.round((ly || 0) * 1.5),
            Math.round(lx || 0), Math.round(ly || 0));
    if (mode === 'alert') { px(33, 1, 2, 6, RD); px(33, 8, 2, 2, RD); }
}

function drawCatSleep(ctx, t) {
    var px = catPx(ctx), y, x;
    ctx.clearRect(0, 0, WW, WHT);
    var br = Math.sin(t * 2) > 0.5 ? 1 : 0;
    for (y = 26; y <= 41 + br; y++) {                  // curled body
        var dy = (y - 33.5) / (8 + br / 2), half = Math.round(15 * Math.sqrt(Math.max(0, 1 - dy * dy)));
        px(20 - half - 1, y, 1, 1, K); px(20 + half, y, 1, 1, K);
        px(20 - half, y, half * 2, 1, WH);
    }
    px(8, 42 + br, 24, 1, K);
    for (y = 26; y <= 41; y++) {                       // the tail wraps round the front
        px(24, y, 0, 0, K);
    }
    px(20, 39, 12, 3, K); px(21, 40, 10, 2, WH); px(31, 37, 2, 4, K); px(30, 38, 2, 3, WH);
    for (y = 26; y <= 40; y++) {                       // tucked head
        for (x = 5; x <= 21; x++) {
            var d = (x - 13) * (x - 13) + (y - 33) * (y - 33);
            if (d <= 42) { px(x, y, 1, 1, WH); }
            else if (d <= 55) { px(x, y, 1, 1, K); }
        }
    }
    px(8, 24, 4, 3, K); px(9, 25, 2, 2, WH); px(16, 24, 4, 3, K); px(17, 25, 2, 2, WH);
    px(9, 33, 3, 1, K); px(15, 33, 3, 1, K);           // closed eyes
    px(13, 36, 2, 1, RD);                              // nose
    var z = (t * 0.9) % 1;                             // floating Z letters
    function zed(zx, zy, a) {
        ctx.globalAlpha = a; px(zx, zy, 5, 1, WH); px(zx + 3, zy + 1, 1, 1, WH);
        px(zx + 2, zy + 2, 1, 1, WH); px(zx + 1, zy + 3, 1, 1, WH); px(zx, zy + 4, 5, 1, WH);
        ctx.globalAlpha = 1;
    }
    zed(27, 18 - Math.round(z * 8), 1 - z);
    zed(32, 10 - Math.round(z * 6), Math.max(0, 0.8 - z));
}

function drawCatRunSide(ctx, t, gait) {
    var px = catPx(ctx), y, k;
    ctx.clearRect(0, 0, WW, WHT);
    var g = Math.sin(gait * Math.PI), bob = Math.abs(g) > 0.6 ? 1 : 0;
    for (k = 0; k < 10; k++) {
        var u = k / 9, wv = Math.sin(t * 9 + k * 0.7) * u * 2;
        px(8 - 7 * u - 1, 21 - 9 * u * u + wv - 1 + bob, 4, 4, K);
    }
    for (k = 0; k < 10; k++) {
        var v = k / 9, wv2 = Math.sin(t * 9 + k * 0.7) * v * 2;
        px(8 - 7 * v, 21 - 9 * v * v + wv2 + bob, 2, 2, WH);
    }
    function leg(x, phase) {
        var sw = Math.round(Math.sin(gait * Math.PI + phase) * 4);
        var lift = Math.max(0, Math.sin(gait * Math.PI + phase)) * 2;
        px(x + sw - 1, 25 - lift + bob, 5, 11, K);
        px(x + sw, 25 - lift + bob, 3, 10, WH);
        px(x + sw - 1, 35 - lift + bob, 6, 1, K);
    }
    leg(10, 0); leg(26, 0); leg(14, Math.PI); leg(22, Math.PI);
    for (y = 18; y <= 28; y++) {
        var bi = (y === 18 || y === 28) ? 2 : 0;
        px(7 + bi, y + bob, 1, 1, K); px(27 - bi, y + bob, 1, 1, K);
        px(8 + bi, y + bob, 19 - bi * 2, 1, WH);
    }
    px(10, 18 + bob, 15, 1, K); px(10, 29 + bob, 15, 1, K);
    px(11, 25 + bob, 12, 2, GR);
    for (y = 14; y <= 26; y++) {
        var ins = (y <= 15 || y >= 25) ? 2 : (y === 16 || y === 24 ? 1 : 0);
        px(26 + ins - 1, y + bob, 1, 1, K); px(37 - ins + 1, y + bob, 1, 1, K);
        px(26 + ins, y + bob, 12 - ins * 2, 1, WH);
    }
    px(28, 13 + bob, 8, 1, K); px(28, 27 + bob, 8, 1, K);
    px(26, 20 + bob, 2, 7, RD);
    px(28, 10 + bob, 3, 3, WH); px(28, 9 + bob, 3, 1, K); px(27, 10 + bob, 1, 3, K); px(31, 10 + bob, 1, 3, K);
    px(33, 10 + bob, 3, 3, WH); px(33, 9 + bob, 3, 1, K); px(32, 10 + bob, 1, 3, K); px(36, 10 + bob, 1, 3, K);
    px(29, 11 + bob, 1, 2, PK); px(34, 11 + bob, 1, 2, PK);
    px(33, 17 + bob, 2, 4, K); px(37, 21 + bob, 2, 1, RD); px(36, 23 + bob, 2, 1, K);
    px(38, 19 + bob, 2, 1, GR); px(38, 22 + bob, 2, 1, GR);
}

function drawCatRunFront(ctx, t, gait) {
    var px = catPx(ctx), y, k;
    ctx.clearRect(0, 0, WW, WHT);
    var g = Math.sin(gait * Math.PI), B = Math.abs(g) > 0.6 ? 1 : 0;
    function pb(x, y2, w, h, c) { px(x, y2 + B, w, h, c); }
    for (k = 0; k < 9; k++) {
        var u = k / 8, wv = Math.sin(t * 8 + k * 0.6) * u * 2.5;
        px(30 + 6 * Math.sin(u * 1.6) + wv - 1, 38 - 17 * u * u + B - 1, 4, 4, K);
    }
    for (k = 0; k < 9; k++) {
        var v = k / 8, wv2 = Math.sin(t * 8 + k * 0.6) * v * 2.5;
        px(30 + 6 * Math.sin(v * 1.6) + wv2, 38 - 17 * v * v + B, 2, 2, WH);
    }
    function paw(x, ph) {
        var s = Math.sin(gait * Math.PI + ph), len = 6 + Math.round(Math.max(0, s) * 4);
        px(x - 1, 33 + B, 5, len + 2, K); px(x, 33 + B, 3, len + 1, WH); px(x - 1, 33 + B + len + 1, 6, 1, K);
    }
    paw(13, 0); paw(23, Math.PI);
    for (y = 28; y <= 37; y++) {
        var hw = y < 31 ? 6 : 7;
        pb(20 - hw - 1, y, 1, 1, K); pb(20 + hw, y, 1, 1, K); pb(20 - hw, y, hw * 2, 1, WH);
    }
    pb(16, 31, 8, 4, GR);
    pb(13, 28, 14, 2, RD); pb(13, 28, 1, 2, K); pb(26, 28, 1, 2, K);
    catHead(px, B, 'run', false, 0, 0, 0, 0);
}

function drawCatRunBack(ctx, t, gait) {
    var px = catPx(ctx), y, k;
    ctx.clearRect(0, 0, WW, WHT);
    var g = Math.sin(gait * Math.PI), B = Math.abs(g) > 0.6 ? 1 : 0;
    function pb(x, y2, w, h, c) { px(x, y2 + B, w, h, c); }
    for (k = 0; k < 9; k++) {                          // tail, drawn first so the body overlaps it
        var u = k / 8, wv = Math.sin(t * 8 + k * 0.6) * u * 2.5;
        px(21 + 8 * Math.sin(u * 1.5) + wv - 1, 36 - 16 * u * u + B - 1, 4, 4, K);
    }
    for (k = 0; k < 9; k++) {
        var v = k / 8, wv2 = Math.sin(t * 8 + k * 0.6) * v * 2.5;
        px(21 + 8 * Math.sin(v * 1.5) + wv2, 36 - 16 * v * v + B, 2, 2, WH);
    }
    function paw(x, ph) {
        var s = Math.sin(gait * Math.PI + ph), len = 5 + Math.round(Math.max(0, s) * 3);
        px(x - 1, 35 + B, 5, len + 2, K); px(x, 35 + B, 3, len + 1, WH); px(x - 1, 35 + B + len + 1, 6, 1, K);
    }
    paw(13, 0); paw(23, Math.PI);
    for (y = 28; y <= 38; y++) {
        var hw = y < 31 ? 6 : 8;
        pb(20 - hw - 1, y, 1, 1, K); pb(20 + hw, y, 1, 1, K); pb(20 - hw, y, hw * 2, 1, WH);
    }
    pb(15, 33, 10, 3, GR);
    for (y = 12; y <= 27; y++) {                       // back of the head: ears, no face
        var ins = (y <= 13 || y >= 26) ? 2 : (y === 14 || y === 25 ? 1 : 0);
        var x0 = 7 + ins, x1 = 32 - ins;
        pb(x0 - 1, y, 1, 1, K); pb(x1 + 1, y, 1, 1, K); pb(x0, y, x1 - x0 + 1, 1, WH);
    }
    pb(9, 11, 22, 1, K); pb(9, 28, 22, 1, K);
    var ew = [1, 2, 4, 5, 7];
    for (y = 6; y <= 10; y++) {
        var w = ew[y - 6], e0 = 11 - Math.floor(w / 2), f0 = 28 - Math.floor(w / 2);
        pb(e0 - 1, y, 1, 1, K); pb(e0, y, w, 1, WH); pb(e0 + w, y, 1, 1, K);
        pb(f0 - 1, y, 1, 1, K); pb(f0, y, w, 1, WH); pb(f0 + w, y, 1, 1, K);
    }
    pb(11, 5, 1, 1, K); pb(28, 5, 1, 1, K);
    pb(13, 28, 14, 2, RD); pb(13, 28, 1, 2, K); pb(26, 28, 1, 2, K);
    pb(16, 18, 8, 1, GR);
}

// Same idea as the desktop-pet cats: sit when still, scratch, get tired, fall asleep,
// run in the direction of travel.
function drawCatState(ctx, state, t, gait, lx, ly) {
    if (state === 'sleep') { drawCatSleep(ctx, t); }
    else if (state === 'scratch') { drawCatSit(ctx, t, lx, ly, 'scratch'); }
    else if (state === 'tired') { drawCatSit(ctx, t, lx, ly, 'tired'); }
    else if (state === 'alert') { drawCatSit(ctx, t, lx, ly, 'alert'); }
    else if (state === 'runS') { drawCatRunFront(ctx, t, gait); }
    else if (state === 'runN') { drawCatRunBack(ctx, t, gait); }
    else if (state === 'runE') { drawCatRunSide(ctx, t, gait); }
    else { drawCatSit(ctx, t, lx, ly, 'sit'); }
}
"""

THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.min.js"
THREE_SRI = "sha384-eWPGgBROV7OvRpuavxpzaTSoHT/gkcVjZ94JzXW7WDFZB9qEb4Q9k+8bfx/c1rTM"

# Animated 3D banner for the landing and sign-in pages: a wireframe shield
# core, an orbiting ring and a particle network that follows the pointer.
SCENE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:">
<style>
    html, body {
        margin: 0; height: 100%; overflow: hidden; background: transparent;
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif;
        -webkit-text-size-adjust: 100%;
    }
    #stage {
        position: relative; box-sizing: border-box; width: 100%;
        height: calc(100% - 4px); overflow: hidden; border-radius: 18px;
        animation: stageIn 0.6s cubic-bezier(.22, 1, .36, 1) both;
        border: 1px solid rgba(229, 56, 59, 0.4);
        background:
            radial-gradient(circle at 10% 70%, rgba(255, 59, 63, 0.30), transparent 38%),
            radial-gradient(circle at 92% 30%, rgba(255, 255, 255, 0.09), transparent 34%),
            radial-gradient(circle at 50% __CAT_TOP__, #8a1223 0%,
            #3f0e1b 28%, #121827 64%, #070b12 100%);
    }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; z-index: 1; }
    .glow {
        position: absolute; left: 50%; top: __CAT_TOP__; z-index: 0;
        width: 86%; aspect-ratio: 1; max-width: 340px;
        transform: translate(-50%, -50%); border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 59, 63, 0.55) 0%,
            rgba(229, 56, 59, 0.22) 38%, transparent 68%);
        animation: pulse 3.2s ease-in-out infinite;
    }
    .catbox {
        position: absolute; left: 50%; top: __CAT_TOP__; z-index: 2;
        width: __CAT_W__; height: __CAT_H__; transform: translate(-50%, -50%);
        animation: floaty 4.2s ease-in-out infinite;
    }
    #cat {
        display: block; width: 100%; height: 100%;
        will-change: transform;
        image-rendering: pixelated; image-rendering: crisp-edges;
        -webkit-mask-image: radial-gradient(ellipse at center, #000 52%,
            transparent 80%);
        mask-image: radial-gradient(ellipse at center, #000 52%,
            transparent 80%);
        filter: drop-shadow(0 0 14px rgba(255, 255, 255, 0.28));
    }
    @media (min-width: 900px) {
        .catbox { width: __CAT_WD__; height: __CAT_HD__; }
    }
    @keyframes pulse { 0%, 100% { opacity: 0.75; transform: translate(-50%, -50%) scale(0.96); }
                       50% { opacity: 1; transform: translate(-50%, -50%) scale(1.06); } }
    @keyframes floaty { 0%, 100% { transform: translate(-50%, -50%); }
                        50% { transform: translate(-50%, calc(-50% - 7px)); } }
    .copy {
        position: absolute; left: 0; right: 0; bottom: 0; z-index: 4;
        padding: 36px 16px 16px; text-align: center; color: #a4b1cd;
        background: linear-gradient(transparent, rgba(7, 11, 18, 0.88) 55%);
        animation: rise 0.9s cubic-bezier(.22, 1, .36, 1) both;
    }
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        border: 1px solid rgba(229, 56, 59, 0.45);
        background: rgba(229, 56, 59, 0.12); color: #fff;
        font: 600 12px ui-monospace, Consolas, monospace; letter-spacing: .04em;
    }
    h1 {
        margin: 8px 0 6px; color: #fff; font-weight: 800; line-height: 1.08;
        font-size: clamp(1.7rem, 7.5vw, 2.9rem);
    }
    h1 span { color: #E5383B; }
    p { margin: 0 auto; max-width: 560px; line-height: 1.5;
        font-size: clamp(0.88rem, 3.3vw, 1.02rem); }
    @keyframes rise { from { opacity: 0; transform: translateY(16px); }
                      to { opacity: 1; transform: none; } }
    @keyframes stageIn { from { opacity: 0; transform: scale(0.985); }
                         to { opacity: 1; transform: none; } }
    @media (prefers-reduced-motion: reduce) {
        #stage, .copy, .glow, .catbox { animation: none; }
    }
</style>
</head>
<body>
<div id="stage">
    <div class="glow"></div>
    <canvas id="c"></canvas>
    <div class="catbox"><canvas id="cat" width="416" height="292" role="img" aria-label="Nashmi, the cat in a red and white shemagh"></canvas></div>
    <div class="copy"__DIR__>__COPY__</div>
</div>
<script>
__CAT_JS__
</script>
<script>
(function () {
    releaseFrameBoxOnExit();
    var cat = document.getElementById('cat'), stage = document.getElementById('stage');
    var box = document.querySelector('.catbox'), cctx = cat.getContext('2d');
    var reduce = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // taller banner on wide screens (the page resizes the frame around it)
    var BASE_H = __HEIGHT__, DESK_K = __DESK_K__;
    function fit() {
        sizeFrame(window.innerWidth >= 900 ? Math.round(BASE_H * DESK_K) : BASE_H);
    }
    window.addEventListener('resize', fit);
    fit();
    setInterval(fit, 1500);

    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem('nashmi_scene') || 'null'); } catch (err) {}
    var phase0 = saved && saved.p ? saved.p : 0, start = performance.now();
    var look = makeLook(stage, function () {
        var r = box.getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height * 0.45 };
    });

    // the original logo cat; only its eyes, nose and mouth are redrawn so it can look around
    var img = new Image(), ready = false;
    img.onload = function () { ready = true; };
    img.src = 'data:image/png;base64,__CAT__';
    function drawLogoCat(t, lx, ly) {
        cctx.imageSmoothingEnabled = false;
        cctx.drawImage(img, 0, 0);
        var blink = (t % 3.8) > 3.66;
        cctx.fillStyle = '#ffffff';
        cctx.fillRect(167, 88, 14, 32); cctx.fillRect(225, 88, 14, 32);
        cctx.fillRect(196, 125, 14, 15);
        var ex = Math.round(lx * 11), ey = Math.round(ly * 7);
        cctx.fillStyle = '#0a0a0a';
        var h = blink ? 3 : 27, y = blink ? 102 : 91;
        cctx.fillRect(169 + ex, y + ey, 10, h); cctx.fillRect(227 + ex, y + ey, 10, h);
        cctx.drawImage(img, 198, 127, 10, 11,
                       198 + Math.round(ex * 0.5), 127 + Math.round(ey * 0.5), 10, 11);
    }
    function paint(now) {
        look.update(now);
        if (ready) {
            drawLogoCat(phase0 + (now - start) / 1000, look.x, look.y);
            cat.style.transform = 'translate(' + (-look.x * 7).toFixed(1) + 'px,' +
                                  (-look.y * 5).toFixed(1) + 'px)';
        }
        if (!reduce || !ready) { requestAnimationFrame(paint); }
    }
    paint(start);
})();
</script>
<script src="__THREE_URL__" integrity="__THREE_SRI__" crossorigin="anonymous"></script>
<script>
(function () {
    if (!window.THREE) { return; }
    var stage = document.getElementById('stage');
    var canvas = document.getElementById('c');
    var reduce = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var renderer;
    try {
        renderer = new THREE.WebGLRenderer(
            { canvas: canvas, antialias: true, alpha: true });
    } catch (err) { canvas.style.display = 'none'; return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    var group = new THREE.Group();
    group.position.y = __OFFSET_Y__;
    scene.add(group);
    var RED = 0xE5383B;

    group.scale.setScalar(__GSCALE__);
    var BRIGHT = 0xFF3B3F;
    // the cat sits in the middle; a red shield and a white shell orbit it
    var coreWire = new THREE.LineSegments(
        new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(2.05, 1)),
        new THREE.LineBasicMaterial({ color: BRIGHT, transparent: true, opacity: 0.8 }));
    var shell = new THREE.LineSegments(
        new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(2.7, 2)),
        new THREE.LineBasicMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.2 }));
    // a black round mesh hugging the cat, drawn under everything else
    var blackMesh = new THREE.LineSegments(
        new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(2.38, 2)),
        new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.85 }));
    blackMesh.renderOrder = -1;
    group.add(blackMesh);
    var shellWrap = new THREE.Group();     // lets the big mesh stretch to the frame on wide screens
    shellWrap.add(shell);
    group.add(coreWire, shellWrap);

    // On a wide screen the red network is spread over the whole frame, edge to edge, so there
    // are no empty corners. On a phone it stays a ball around the cat.
    var sw0 = stage.clientWidth || 400, sh0 = stage.clientHeight || 400, asp0 = sw0 / sh0;
    var wideNet = asp0 >= 1.25;
    var bz0 = asp0 < 0.9 ? 10.5 : (asp0 < 1.6 ? 8.6 : 7.2);
    var fit0 = Math.min(1, (bz0 * 0.4142 * 0.96) / ((3.6 + __OFFSET_Y__) * __GSCALE__));
    var kLocal = 1 / (__GSCALE__ * fit0);
    var XW = 0.4142 * bz0 * asp0 * 1.06 * kLocal, YH = 0.4142 * bz0 * 1.06 * kLocal;
    var OFFL = __OFFSET_Y__ * kLocal * __GSCALE__ * fit0;
    var N = wideNet ? 420 : 190, positions = new Float32Array(N * 3), colors = new Float32Array(N * 3), pts = [];
    for (var i = 0; i < N; i++) {
        var x, y, z;
        if (wideNet) {
            x = (Math.random() * 2 - 1) * XW;
            y = (Math.random() * 2 - 1) * YH - OFFL * kLocal;
            z = (Math.random() * 2 - 1) * 1.6;
        } else {
            var r = 3.2 + Math.random() * 1.5;
            var t = Math.random() * Math.PI * 2;
            var u = Math.acos(2 * Math.random() - 1);
            x = r * Math.sin(u) * Math.cos(t);
            y = r * Math.sin(u) * Math.sin(t);
            z = r * Math.cos(u);
        }
        positions[i * 3] = x; positions[i * 3 + 1] = y; positions[i * 3 + 2] = z;
        // alternate white and bright red points
        colors[i * 3] = 1; colors[i * 3 + 1] = i % 2 ? 1 : 0.23; colors[i * 3 + 2] = i % 2 ? 1 : 0.25;
        pts.push([x, y, z]);
    }
    var pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pg.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    var points = new THREE.Points(pg, new THREE.PointsMaterial(
        { vertexColors: true, size: 0.075, transparent: true, opacity: 0.95 }));
    var lineData = [], MAX_FLOATS = 260 * 6;
    if (wideNet) {                       // join every point to its two nearest neighbours
        for (var a = 0; a < N; a++) {
            var best = [[1e9, -1], [1e9, -1]];
            for (var b = 0; b < N; b++) {
                if (a === b) { continue; }
                var dx = pts[a][0] - pts[b][0], dy = pts[a][1] - pts[b][1], dz = pts[a][2] - pts[b][2];
                var dd = dx * dx + dy * dy + dz * dz;
                if (dd < best[0][0]) { best[1] = best[0]; best[0] = [dd, b]; }
                else if (dd < best[1][0]) { best[1] = [dd, b]; }
            }
            for (var q = 0; q < 2; q++) {
                var bb = best[q][1];
                if (bb >= 0) {
                    lineData.push(pts[a][0], pts[a][1], pts[a][2], pts[bb][0], pts[bb][1], pts[bb][2]);
                }
            }
        }
    } else {
        for (var a = 0; a < N && lineData.length < MAX_FLOATS; a++) {
            for (var b = a + 1; b < N && lineData.length < MAX_FLOATS; b++) {
                var dx = pts[a][0] - pts[b][0], dy = pts[a][1] - pts[b][1],
                    dz = pts[a][2] - pts[b][2];
                if (dx * dx + dy * dy + dz * dz < 1.7) {
                    lineData.push(pts[a][0], pts[a][1], pts[a][2],
                                  pts[b][0], pts[b][1], pts[b][2]);
                }
            }
        }
    }
    var lg = new THREE.BufferGeometry();
    lg.setAttribute('position',
        new THREE.BufferAttribute(new Float32Array(lineData), 3));
    var lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial(
        { color: BRIGHT, transparent: true, opacity: 0.45 }));
    var net = new THREE.Group();
    net.add(points, lines);
    group.add(net);

    var ring = new THREE.Mesh(new THREE.TorusGeometry(3.0, 0.02, 8, 180),
        new THREE.MeshBasicMaterial({ color: BRIGHT, transparent: true, opacity: 0.85 }));
    ring.rotation.x = 1.38;
    var ring2 = new THREE.Mesh(new THREE.TorusGeometry(3.5, 0.015, 8, 180),
        new THREE.MeshBasicMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.55 }));
    ring2.rotation.x = 1.72;
    ring2.rotation.y = 0.1;
    group.add(ring, ring2);

    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem('nashmi_scene') || 'null'); } catch (err) {}
    var phase0 = saved && saved.p ? saved.p : 0, lastSave = 0, baseZ = 8.6;
    if (saved) { group.rotation.y = saved.ry || 0; group.rotation.x = saved.rx || 0; }

    function resize() {
        var w = stage.clientWidth, h = stage.clientHeight;
        if (!w || !h) { return; }
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        baseZ = w / h < 0.9 ? 10.5 : (w / h < 1.6 ? 8.6 : 7.2);
        camera.position.set(0, 0, baseZ);
        if (wideNet) { net.scale.set(1, 1, 1); }
        else { net.scale.set(Math.max(1, Math.min(2.1, (w / h) * 0.7)), 1, 1); }
        var GS = __GSCALE__, OFF = __OFFSET_Y__;       // keep the whole ring inside the banner
        var fitK = Math.min(1, (baseZ * 0.4142 * 0.96) / ((3.6 + OFF) * GS));
        group.scale.setScalar(GS * fitK);
        group.position.y = OFF * fitK;
        // centre the cat (and its glow) on the middle of the rings
        var ndcY = (OFF * fitK) / (0.4142 * baseZ), cy = ((1 - ndcY) / 2 * h).toFixed(1) + 'px';
        var cb = document.querySelector('.catbox'), gl = document.querySelector('.glow');
        if (cb) { cb.style.top = cy; }
        if (gl) { gl.style.top = cy; }

        camera.updateProjectionMatrix();
    }
    resize();
    if (window.ResizeObserver) { new ResizeObserver(resize).observe(stage); }
    else { window.addEventListener('resize', resize); }

    var tx = 0, ty = 0, lastPtr = -1e9;
    trackPointer(function (x, y) {
        var rc = stage.getBoundingClientRect();
        tx = Math.max(-0.5, Math.min(0.5, (x - rc.left) / rc.width - 0.5));
        ty = Math.max(-0.5, Math.min(0.5, (y - rc.top) / rc.height - 0.5));
        lastPtr = performance.now();
    });

    var start = performance.now(), running = false;
    function frame(now) {
        var t = phase0 + (now - start) / 1000;
        var kIn = Math.min(1, (now - start) / 900), ease = 1 - Math.pow(1 - kIn, 3);
        camera.position.z = baseZ + (saved ? 0.7 : 1.9) * (1 - ease);   // gentle push-in
        var idle = now - lastPtr > 2600;
        var ex = idle ? Math.sin(t * 0.4) * 0.1 : tx;
        var ey = idle ? Math.cos(t * 0.33) * 0.05 : ty;
        coreWire.rotation.y = t * 0.35;
        coreWire.rotation.x = t * 0.2;
        coreWire.scale.setScalar(1 + Math.sin(t * 1.6) * 0.03);
        shell.rotation.y = -t * 0.12;
        blackMesh.rotation.y = t * 0.18; blackMesh.rotation.x = t * 0.07;
        net.rotation.y = wideNet ? Math.sin(t * 0.3) * 0.04 : t * 0.06;
        net.rotation.x = wideNet ? 0 : Math.sin(t * 0.2) * 0.1;
        ring.rotation.z = t * 0.25;
        ring2.rotation.z = -t * 0.18;
        group.rotation.y += (ex * 0.9 - group.rotation.y) * 0.05;
        group.rotation.x += (ey * 0.6 - group.rotation.x) * 0.05;
        if (now - lastSave > 400) {      // lets the next page continue this motion
            lastSave = now;
            try {
                sessionStorage.setItem('nashmi_scene', JSON.stringify(
                    { p: t, ry: group.rotation.y, rx: group.rotation.x }));
            } catch (err) {}
        }
        renderer.render(scene, camera);
        if (running) { requestAnimationFrame(frame); }
    }
    function play() {
        if (reduce || running) { return; }
        running = true;
        requestAnimationFrame(frame);
    }
    frame(start);          // always paint one frame (also the reduced-motion look)
    play();
    document.addEventListener('visibilitychange', function () {
        if (document.hidden) { running = false; } else { play(); }
    });
})();
</script>
</body>
</html>
"""

def scene_copy(kind: str) -> str:
    """The words on the banner, in the current language."""
    if kind == "landing":
        return (
            f'<div class="badge">{html.escape(tr("hero_badge"))}</div>'
            f'<h1>{html.escape(tr("hero_a"))} <span>{html.escape(tr("hero_b"))}</span></h1>'
            f'<p>{html.escape(tr("hero_sub"))}</p>'
        )
    return (
        f'<h1><span>&gt;_</span> {html.escape(tr("scene_" + kind + "_h"))}</h1>'
        f'<p>{html.escape(tr("scene_" + kind + "_p"))}</p>'
    )


# height, scene offset, scene scale, cat width (phone / wide screen), cat
# vertical position, extra height factor on wide screens
SCENE_SIZES = {
    "landing": (470, 0.9, 1.0, (230, 250), "37%", 1.2),
    "login": (330, 0.5, 0.82, (190, 230), "35%", 1.3),
    "signup": (330, 0.5, 0.82, (190, 230), "35%", 1.3),
}


def render_scene(kind: str) -> None:
    """Show the animated 3D banner for the landing or sign-in page."""
    height, offset, scale, (cat_w, cat_wd), cat_top, desk_k = SCENE_SIZES[kind]
    markup = (
        SCENE_TEMPLATE.replace("__DIR__", dir_attr())
        .replace("__COPY__", scene_copy(kind))
        .replace("__THREE_URL__", THREE_URL)
        .replace("__THREE_SRI__", THREE_SRI)
        .replace("__OFFSET_Y__", str(offset))
        .replace("__GSCALE__", str(scale))
        .replace("__CAT_W__", f"{cat_w}px")
        .replace("__CAT_H__", f"{round(cat_w * 292 / 416)}px")
        .replace("__CAT_WD__", f"{cat_wd}px")
        .replace("__CAT_HD__", f"{round(cat_wd * 292 / 416)}px")
        .replace("__CAT_TOP__", cat_top)
        .replace("__HEIGHT__", str(height))
        .replace("__DESK_K__", str(desk_k))
        .replace("__CAT_JS__", CAT_JS)
        .replace("__CAT__", CAT_B64)
    )
    embed_html(markup, height)


LANDING_CSS = """
@keyframes lpUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: none; }
}
.lp-hero { text-align: center; padding: 2.2rem 0 0.4rem; }
.lp-badge {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
    border: 1px solid var(--accent-dim); background: var(--accent-soft);
    color: var(--white); font: 600 0.78rem var(--mono); letter-spacing: 0.04em;
}
.lp-title {
    margin: 1rem 0 0.4rem; color: var(--white); font-weight: 800;
    font-size: clamp(2rem, 6vw, 3.2rem); line-height: 1.1;
}
.lp-title span { color: var(--accent); }
.lp-sub {
    max-width: 640px; margin: 0 auto 1.2rem; color: var(--text);
    font-size: clamp(1rem, 2.6vw, 1.15rem); line-height: 1.55;
}
.lp-grid {
    display: grid; gap: 0.9rem; margin: 1.6rem 0 1rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.lp-card {
    padding: 1.1rem 1.15rem; border-radius: 14px;
    background: var(--panel); border: 1px solid var(--line);
}
.lp-card b { display: block; color: var(--white); margin-bottom: 0.3rem; }
.lp-card span { color: var(--text); font-size: 0.93rem; line-height: 1.5; }
.lp-note {
    max-width: 640px; margin: 0.8rem auto 0; text-align: center;
    color: var(--muted); font-size: 0.85rem; line-height: 1.5;
}
.lp-signin {
    max-width: 440px; margin: 0.9rem auto 0.6rem; padding: 1.4rem 1.4rem;
    border-radius: 16px; background: var(--panel);
    border: 1px solid var(--line); text-align: center;
}
.lp-signin h2 {
    margin: 0 0 0.4rem; padding: 0; color: var(--white); font-size: 1.5rem;
    text-align: center;
}
.lp-signin p { margin: 0.2rem 0 0.8rem; color: var(--text); font-size: 0.95rem; }
.lp-signin ul {
    margin: 0.4rem 0 0; padding-left: 1.2rem; text-align: left;
    color: var(--text); font-size: 0.92rem; line-height: 1.7;
}
[class*="st-key-landing_cta"], .st-key-login_card { max-width: 440px; margin: 0 auto; }

.lp-footer {
    margin: 2.2rem auto 0.4rem; padding-top: 1.2rem; text-align: center;
    border-top: 1px solid var(--line); color: var(--muted);
    font-size: 0.85rem; line-height: 1.7;
}
.lp-footer a { color: var(--text); text-decoration: none; }
.lp-footer a:hover { color: var(--accent); }
.lp-footer-sub { margin-top: 0.15rem; }
.lp-social { display: flex; justify-content: center; gap: 0.7rem; margin-bottom: 0.8rem; }
.lp-social a {
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 50%;
    background: var(--panel); border: 1px solid var(--line);
    transition: transform 0.25s cubic-bezier(.22, 1, .36, 1),
                border-color 0.25s ease, background 0.25s ease;
}
.lp-social a:hover {
    transform: translateY(-3px); border-color: var(--accent);
    background: var(--accent-soft);
}
.lp-social svg { width: 20px; height: 20px; fill: var(--white); }

/* Smooth entrance and hover motion */
.lp-card {
    animation: lpUp 0.7s cubic-bezier(.22, 1, .36, 1) both;
    transition: transform 0.25s ease, border-color 0.25s ease,
                box-shadow 0.25s ease;
}
.lp-card:nth-child(2) { animation-delay: 0.08s; }
.lp-card:nth-child(3) { animation-delay: 0.16s; }
.lp-card:nth-child(4) { animation-delay: 0.24s; }
.lp-card:hover {
    transform: translateY(-4px); border-color: var(--accent-dim);
    box-shadow: 0 12px 30px rgba(229, 56, 59, 0.14);
}
.lp-signin { animation: lpUp 0.7s cubic-bezier(.22, 1, .36, 1) both; }
[class*="st-key-landing_cta"], .st-key-login_card {
    animation: lpUp 0.7s 0.1s cubic-bezier(.22, 1, .36, 1) both;
}
[class*="st-key-landing_cta"] button, .st-key-login_card button {
    transition: transform 0.2s ease, border-color 0.2s ease,
                box-shadow 0.2s ease;
}
[class*="st-key-landing_cta"] button:hover, .st-key-login_card button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 22px rgba(229, 56, 59, 0.18);
}
@media (prefers-reduced-motion: reduce) {
    .lp-card, .lp-signin, [class*="st-key-landing_cta"], .st-key-login_card {
        animation: none;
    }
    .lp-card, [class*="st-key-landing_cta"] button, .st-key-login_card button {
        transition: none;
    }
}
.st-key-login_card [data-testid="stButton"],
.st-key-login_card [data-testid="stButton"] button { width: 100%; }
.st-key-login_card [data-testid="stCaptionContainer"] {
    color: var(--text); text-align: center;
}

/* Landing and sign-in pages: no Streamlit chrome, wider page, livelier colour */
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 1180px !important; padding-top: 0 !important; }
.stApp {
    background:
        radial-gradient(900px 520px at 12% 0%, rgba(229, 56, 59, 0.26), transparent 62%),
        radial-gradient(820px 520px at 92% 18%, rgba(255, 255, 255, 0.08), transparent 62%),
        radial-gradient(1000px 640px at 50% 100%, rgba(229, 56, 59, 0.18), transparent 62%),
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
        var(--bg) !important;
    background-size: auto, auto, auto, 40px 40px, 40px 40px, auto !important;
    background-attachment: fixed !important;
}
/* floating pill bar; its wrapper is only as tall as the bar, so the wrapper sticks */
div:has(> .st-key-topbar) { position: sticky; top: 10px; z-index: 60; }
.st-key-topbar {
    margin: 10px 0 14px; padding: 0.3rem 0.45rem 0.3rem 0.8rem;
    border-radius: 999px; background: rgba(14, 21, 36, 0.72);
    -webkit-backdrop-filter: blur(16px) saturate(1.4);
    backdrop-filter: blur(16px) saturate(1.4);
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.35),
                inset 0 0 0 1px rgba(255, 255, 255, 0.07);
}
.st-key-topbar [data-testid="stHorizontalBlock"] {
    flex-direction: row !important; flex-wrap: nowrap !important;
    gap: 0.3rem !important; align-items: center !important;
}
.st-key-topbar [data-testid="stColumn"] {
    min-width: 0 !important; width: auto !important; flex: 0 0 auto !important;
}
.st-key-topbar [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; }
.st-key-topbar [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
.st-key-topbar button {
    border: 0 !important; border-radius: 999px !important; box-shadow: none;
    min-height: 40px; padding: 0 1.15rem; font-weight: 600;
    background: transparent; color: var(--text);
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.st-key-topbar button:hover { background: rgba(255, 255, 255, 0.09); color: var(--white); }
.st-key-topbar [data-testid="stBaseButton-primary"] {
    color: #fff; background: linear-gradient(135deg, #ff3b3f, #c81e3a);
    box-shadow: 0 6px 20px rgba(229, 56, 59, 0.45);
}
.st-key-topbar [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #ff5256, #d6243f); transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(229, 56, 59, 0.55);
}
[data-testid="stImage"] { display: flex; justify-content: center; }
.st-key-login_card { margin-top: 0.4rem; }
@media (max-width: 480px) {
    .st-key-topbar button { padding: 0 0.75rem; min-width: 0; }
    .st-key-topbar { padding-left: 0.5rem; }
}
.lp-about {
    max-width: 640px; margin: 0.6rem auto 0; padding: 0.95rem 1.15rem; text-align: center;
    border-radius: 16px; background: rgba(19, 28, 43, 0.85);
    box-shadow: inset 0 0 0 1px var(--line);
}
.lp-about b { color: var(--white); font-size: 1rem; }
.lp-about p { margin: 0.3rem 0 0.6rem; color: var(--text); font-size: 0.88rem; line-height: 1.55; }
.lp-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; justify-content: center; }
.lp-chips span {
    padding: 2px 10px; border-radius: 999px; font: 600 0.72rem var(--mono);
    color: var(--text); background: rgba(229, 56, 59, 0.14);
}
.st-key-topbar [data-baseweb="select"] > div {
    border: 0 !important; border-radius: 999px !important; min-height: 40px;
    background: transparent !important; box-shadow: none !important;
    font-size: 0.9rem; cursor: pointer;
}
.st-key-topbar [data-testid="stColumn"]:nth-child(2) { flex: 0 0 6rem !important; width: 6rem !important; min-width: 6rem !important; }
.st-key-topbar [data-baseweb="select"] { min-width: 5.2rem; }
.st-key-topbar [data-baseweb="select"] [class*="ValueContainer"], .st-key-topbar [data-baseweb="select"] div[value] { overflow: visible; }
.st-key-topbar [data-baseweb="select"] > div { padding-left: 0.2rem; }
.st-key-topbar [data-baseweb="select"] [data-baseweb="icon"], .st-key-topbar [data-baseweb="select"] svg { width: 16px; height: 16px; }
.st-key-topbar [data-baseweb="select"] > div:hover { background: rgba(255, 255, 255, 0.06) !important; }
.st-key-topbar [data-baseweb="select"] input { caret-color: transparent; cursor: pointer; }
.st-key-topbar .stSelectbox div[data-baseweb="select"] > div,
.st-key-topbar .stSelectbox div[data-baseweb="select"] > div > div {
    background: transparent !important; background-color: transparent !important;
    border: 0 !important; box-shadow: none !important;
}
.st-key-topbar .stSelectbox div[data-baseweb="select"] > div:hover { background-color: rgba(255, 255, 255, 0.06) !important; }
.st-key-topbar [data-baseweb="select"] *, .st-key-topbar [data-baseweb="select"] > div {
    background: transparent !important; background-color: transparent !important;
    border-color: transparent !important; box-shadow: none !important;
}
/* Streamlit's newer select: the visible box is the element with role="group" */
.st-key-topbar .stSelectbox [role="group"] {
    background: transparent !important; background-color: transparent !important;
    border: 0 !important; border-radius: 999px !important; box-shadow: none !important;
    min-height: 40px;
}
.st-key-topbar .stSelectbox [role="group"]:hover { background-color: rgba(255, 255, 255, 0.06) !important; }
.st-key-topbar .stSelectbox input { caret-color: transparent; cursor: pointer; }
.st-key-topbar_home {
    position: absolute !important; width: 1px; height: 1px; overflow: hidden;
    opacity: 0; pointer-events: none; margin: 0 !important;
}
.lp-footer-note {
    max-width: 640px; margin: 0 auto 1.1rem; color: var(--muted);
    font-size: 0.85rem; line-height: 1.5;
}
/* Google's G on the left of "Continue with Google" */
[class*="_login_google"] button {
    padding-left: 3rem !important; background-repeat: no-repeat !important;
    background-position: 1.1rem center !important; background-size: 20px 20px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 48 48%22%3E%3Cpath fill=%22%23EA4335%22 d=%22M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z%22/%3E%3Cpath fill=%22%234285F4%22 d=%22M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z%22/%3E%3Cpath fill=%22%23FBBC05%22 d=%22M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z%22/%3E%3Cpath fill=%22%2334A853%22 d=%22M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z%22/%3E%3C/svg%3E") !important;
}
"""

# The scroll story on the landing page: a cat slides down a line as the
# visitor scrolls, and each card lights up when the cat reaches it.
STORY_CARDS = {
    "en": (
        ("WHO", "Meet Nashmi",
         "Nashmi is an AI agent built only for cybersecurity. Ask in Arabic or "
         "English and get answers that use the field's own language."),
        ("NAME", "Why the name?",
         "In Jordanian Arabic a nashmi is someone gallant and dependable. That "
         "is why the cat wears the red and white shemagh."),
        ("SOC", "SOC Analyzer",
         "Drop in logs, PCAPs, screenshots or alerts. Nashmi writes a structured "
         "incident report with IoCs, MITRE ATT&CK mapping and remediation steps."),
        ("CTF", "CTF Coach",
         "Web, crypto, pwn, reverse, forensics, stego and OSINT. Describe the "
         "challenge and get a step-by-step plan with exact commands, tuned to "
         "the category."),
        ("TOOLS", "Security Toolbox",
         "CVE lookups from the NVD, hashing and encoding, and an IOC extractor. "
         "No AI needed, and everything can be exported."),
        ("EXAM", "Practice Exams",
         "Build an exam from a topic or your own slides and PDFs, take it, and "
         "get it graded."),
        ("PRIVACY", "Your key, your data",
         "Bring your own API key (Gemini, Groq, OpenAI or Claude). Your chats "
         "and saved keys are encrypted."),
        ("MAKER", "Built by Hassan Aldajah",
         "A fourth-year student at the University of Jordan with a deep "
         "interest in the SOC field and in cybersecurity."),
    ),
    "ar": (
        ("\u0645\u0646 \u0647\u0648", "\u062a\u0639\u0631\u0651\u0641 \u0639\u0644\u0649 \u0646\u0634\u0645\u064a",
         "\u0646\u0634\u0645\u064a \u0648\u0643\u064a\u0644 \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a \u0645\u062e\u0635\u0635 \u0644\u0644\u0623\u0645\u0646 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a \u0641\u0642\u0637. "
         "\u0627\u0633\u0623\u0644\u0647 \u0628\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0623\u0648 \u0628\u0627\u0644\u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629 \u0648\u0633\u064a\u062c\u064a\u0628\u0643 \u0628\u0644\u063a\u0629 \u0627\u0644\u0645\u062c\u0627\u0644."),
        ("\u0627\u0644\u0627\u0633\u0645", "\u0644\u064a\u0634 \u0647\u0627\u0644\u0627\u0633\u0645\u061f",
         "\u0641\u064a \u0627\u0644\u0644\u0647\u062c\u0629 \u0627\u0644\u0623\u0631\u062f\u0646\u064a\u0629\u060c \u0627\u0644\u0646\u0634\u0645\u064a \u0647\u0648 \u0627\u0644\u0634\u0647\u0645 \u0627\u0644\u0630\u064a \u064a\u064f\u0639\u062a\u0645\u062f \u0639\u0644\u064a\u0647. "
         "\u0648\u0644\u0647\u0630\u0627 \u062a\u0631\u062a\u062f\u064a \u0627\u0644\u0642\u0637\u0629 \u0627\u0644\u0634\u0645\u0627\u063a \u0627\u0644\u0623\u062d\u0645\u0631 \u0648\u0627\u0644\u0623\u0628\u064a\u0636."),
        ("SOC", "\u0645\u062d\u0644\u0651\u0644 SOC",
         "\u0627\u0631\u0641\u0639 \u0633\u062c\u0644\u0627\u062a (logs) \u0623\u0648 \u0645\u0644\u0641\u0627\u062a PCAP \u0623\u0648 \u0644\u0642\u0637\u0627\u062a \u0634\u0627\u0634\u0629 \u0623\u0648 \u062a\u0646\u0628\u064a\u0647\u0627\u062a\u060c "
         "\u0648\u064a\u0643\u062a\u0628 \u0644\u0643 \u0646\u0634\u0645\u064a \u062a\u0642\u0631\u064a\u0631 \u062d\u0627\u062f\u062b\u0629 \u0645\u0646\u0638\u0651\u0645\u0627\u064b \u0641\u064a\u0647 \u0645\u0624\u0634\u0631\u0627\u062a \u0627\u0644\u0627\u062e\u062a\u0631\u0627\u0642 (IoCs) "
         "\u0648\u062a\u0635\u0646\u064a\u0641 MITRE ATT&CK \u0648\u062e\u0637\u0648\u0627\u062a \u0627\u0644\u0645\u0639\u0627\u0644\u062c\u0629."),
        ("CTF", "\u0645\u062f\u0631\u0651\u0628 CTF",
         "Web \u0648Crypto \u0648Pwn \u0648Reverse \u0648Forensics \u0648Stego \u0648OSINT. \u0635\u0650\u0641 \u0627\u0644\u062a\u062d\u062f\u064a "
         "\u0648\u0633\u064a\u0639\u0637\u064a\u0643 \u062e\u0637\u0629 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629 \u0645\u0639 \u0627\u0644\u0623\u0648\u0627\u0645\u0631 \u0627\u0644\u062f\u0642\u064a\u0642\u0629 \u062d\u0633\u0628 \u0627\u0644\u0641\u0626\u0629."),
        ("\u0627\u0644\u0623\u062f\u0648\u0627\u062a", "\u0635\u0646\u062f\u0648\u0642 \u0623\u062f\u0648\u0627\u062a \u0627\u0644\u0623\u0645\u0646",
         "\u0628\u062d\u062b CVE \u0645\u0646 NVD\u060c \u0648\u062a\u062c\u0632\u0626\u0629 (hash) \u0648\u062a\u0631\u0645\u064a\u0632\u060c \u0648\u0645\u0633\u062a\u062e\u0631\u062c IOC. "
         "\u0628\u062f\u0648\u0646 \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a\u060c \u0648\u064a\u0645\u0643\u0646 \u062a\u0635\u062f\u064a\u0631 \u0643\u0644 \u0634\u064a\u0621."),
        ("\u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646\u0627\u062a", "\u0627\u0645\u062a\u062d\u0627\u0646\u0627\u062a \u062a\u062f\u0631\u064a\u0628\u064a\u0629",
         "\u0627\u0628\u0646\u0650 \u0627\u0645\u062a\u062d\u0627\u0646\u0627\u064b \u0645\u0646 \u0645\u0648\u0636\u0648\u0639 \u0623\u0648 \u0645\u0646 \u0634\u0631\u0627\u0626\u062d\u0643 \u0648\u0645\u0644\u0641\u0627\u062a PDF\u060c "
         "\u0648\u0642\u062f\u0651\u0645\u0647\u060c \u062b\u0645 \u0627\u062d\u0635\u0644 \u0639\u0644\u0649 \u062a\u0635\u062d\u064a\u062d\u0647."),
        ("\u0627\u0644\u062e\u0635\u0648\u0635\u064a\u0629", "\u0645\u0641\u062a\u0627\u062d\u0643 \u0648\u0628\u064a\u0627\u0646\u0627\u062a\u0643",
         "\u0627\u0633\u062a\u062e\u062f\u0645 \u0645\u0641\u062a\u0627\u062d API \u0627\u0644\u062e\u0627\u0635 \u0628\u0643 (Gemini \u0623\u0648 Groq \u0623\u0648 OpenAI \u0623\u0648 Claude). "
         "\u0645\u062d\u0627\u062f\u062b\u0627\u062a\u0643 \u0648\u0645\u0641\u0627\u062a\u064a\u062d\u0643 \u0627\u0644\u0645\u062d\u0641\u0648\u0638\u0629 \u0645\u0634\u0641\u0651\u0631\u0629."),
        ("\u0627\u0644\u0645\u0637\u0648\u0651\u0631", "\u0645\u0646 \u062a\u0637\u0648\u064a\u0631 \u062d\u0633\u0646 \u0627\u0644\u062f\u0639\u062c\u0629",
         "\u0637\u0627\u0644\u0628 \u0641\u064a \u0627\u0644\u0633\u0646\u0629 \u0627\u0644\u0631\u0627\u0628\u0639\u0629 \u0628\u0627\u0644\u062c\u0627\u0645\u0639\u0629 \u0627\u0644\u0623\u0631\u062f\u0646\u064a\u0629\u060c "
         "\u0645\u0647\u062a\u0645 \u0628\u0639\u0645\u0642 \u0628\u0645\u062c\u0627\u0644 \u0645\u0631\u0643\u0632 \u0627\u0644\u0639\u0645\u0644\u064a\u0627\u062a \u0627\u0644\u0623\u0645\u0646\u064a\u0629 (SOC) \u0648\u0627\u0644\u0623\u0645\u0646 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a."),
    ),
}
STORY_HEIGHT = 2100  # first guess; the page shrinks it to fit its content

STORY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<style>
    :root { --tx: 24px; }
    html, body {
        margin: 0; background: transparent; overflow: hidden; color: #a4b1cd;
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif;
        -webkit-text-size-adjust: 100%;
    }
    #wrap {
        position: relative; padding: 30px 12px 50px; min-height: 300px; overflow: hidden;
        background:
            radial-gradient(520px 380px at 88% 6%, rgba(229, 56, 59, 0.20), transparent 70%),
            radial-gradient(560px 420px at 6% 42%, rgba(229, 56, 59, 0.16), transparent 70%),
            radial-gradient(520px 380px at 92% 78%, rgba(255, 255, 255, 0.07), transparent 70%);
    }
    #sky {
        position: absolute; left: 0; top: 0; width: 100%; overflow: hidden;
        pointer-events: none;
        -webkit-mask-image: linear-gradient(transparent, #000 20%, #000 80%, transparent);
        mask-image: linear-gradient(transparent, #000 20%, #000 80%, transparent);
    }
    #sky canvas { position: absolute; left: -40px; top: 0; }
    #pathsvg {
        position: absolute; left: 0; top: 0; width: 100%; height: 100%;
        overflow: visible; z-index: 1; pointer-events: none;
    }
    #pathBg { fill: none; stroke: rgba(164, 177, 205, 0.4); stroke-width: 2;
              stroke-dasharray: 2 7; stroke-linecap: round; }
    #pathFill { fill: none; stroke: #ff4b4f; stroke-width: 2.5; stroke-linecap: round;
                filter: drop-shadow(0 0 6px rgba(229, 56, 59, 0.7)); }
    #rows { position: relative; z-index: 2; }
    .row { position: relative; margin: 0 0 24px; min-height: 64px; }
    .node {
        position: absolute; top: 24px; left: var(--tx); width: 12px; height: 12px;
        margin-left: -7px; background: #0b111c; z-index: 2;
        border: 2px solid rgba(164, 177, 205, 0.5); transform: rotate(45deg);
        transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .row.on .node {
        background: #E5383B; border-color: #fff;
        box-shadow: 0 0 0 6px rgba(229, 56, 59, 0.22);
    }
    .card {
        position: relative; box-sizing: border-box; padding: 16px 18px;
        background: rgba(19, 28, 43, 0.92); border-radius: 14px;
        border: 1px solid rgba(164, 177, 205, 0.16);
        opacity: 0.4; transform: translateY(10px);
        transition: opacity 0.5s ease, transform 0.5s cubic-bezier(.22, 1, .36, 1),
                    border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .row.on .card { opacity: 1; transform: none; }
    .row.active .card {
        border-color: #E5383B; box-shadow: 0 10px 34px rgba(229, 56, 59, 0.22);
    }
    .tag {
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font: 700 11px ui-monospace, Consolas, monospace; letter-spacing: 0.08em;
        color: #fff; background: rgba(229, 56, 59, 0.2);
        border: 1px solid rgba(229, 56, 59, 0.5);
    }
    h3 { margin: 8px 0 4px; color: #fff; font-size: 1.15rem; }
    p { margin: 0; font-size: 0.94rem; line-height: 1.55; }
    #cat {
        position: absolute; left: 0; top: 0; z-index: 3;
        width: 64px; height: 74px;
        transform: translate(-50%, -56%); transform-origin: 50% 60%;
        image-rendering: pixelated; image-rendering: crisp-edges;
        pointer-events: none;
        filter: drop-shadow(0 0 12px rgba(255, 59, 63, 0.45));
    }
    @media (max-width: 699px) {
        :root { --tx: 40px; }
        .row { padding-left: 84px; }
        #cat { width: 46px; height: 53px; }
    }
    @media (min-width: 700px) {
        :root { --tx: 50%; }
        .row .card { width: calc(50% - 82px); }
        .row:nth-child(even) .card { margin-left: auto; }
        /* the line zigzags between the card sides, like a path from card to card */
        .row:nth-child(odd) .node { margin-left: -41px; }
        .row:nth-child(even) .node { margin-left: 27px; }
        #wrap { padding-left: 24px; padding-right: 24px; }
    }
    @media (prefers-reduced-motion: reduce) {
        .card, .node { transition: none; }
    }
</style>
</head>
<body>
<div id="wrap">
    <div id="sky"><canvas id="far"></canvas><canvas id="near"></canvas></div>
    <svg id="pathsvg" aria-hidden="true"><path id="pathBg"></path><path id="pathFill"></path></svg>
    <div id="rows">__CARDS__</div>
    <canvas id="cat" width="40" height="46" role="img" aria-label="Nashmi the cat walking down the line"></canvas>
</div>
<script>
__CAT_JS__
</script>
<script>
(function () {
    releaseFrameBoxOnExit();
    var reduce = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var wrap = document.getElementById('wrap');
    var rows = Array.prototype.slice.call(document.querySelectorAll('.row'));
    var pathBg = document.getElementById('pathBg'), pathFill = document.getElementById('pathFill');
    var cat = document.getElementById('cat'), sky = document.getElementById('sky');
    var far = document.getElementById('far'), near = document.getElementById('near');
    var P = null, fe = null;
    try { P = window.parent; fe = window.frameElement; void P.document.body; }
    catch (err) { P = null; fe = null; }
    var H = 0, W = 0, y0 = 0, y1 = 0, centers = [], skyH = 0, pts = [], pathLen = 0;

    var cctx = cat.getContext('2d');
    var look = makeLook(wrap, function () {
        var r = cat.getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height * 0.4 };
    });

    function rng(seed) {
        var s = seed;
        return function () { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    }
    function drawDust(c, seed, count, minS, maxS, alpha) {
        var g = c.getContext('2d'), r = rng(seed);
        g.clearRect(0, 0, c.width, c.height);
        for (var i = 0; i < count; i++) {
            var sz = Math.round(minS + r() * (maxS - minS));
            var dx = Math.floor(r() * c.width / 4) * 4, dy = Math.floor(r() * c.height / 4) * 4;
            g.fillStyle = r() < 0.6 ? 'rgba(229,56,59,' + alpha + ')'
                : 'rgba(255,255,255,' + (alpha * 0.8).toFixed(2) + ')';
            g.fillRect(dx, dy, sz, sz);
        }
    }

    function layout() {
        var wb = wrap.getBoundingClientRect();
        W = wrap.clientWidth; H = wrap.offsetHeight;
        var nx = [], ny = [];
        rows.forEach(function (r) {
            var n = r.querySelector('.node').getBoundingClientRect();
            nx.push(n.left - wb.left + n.width / 2);
            ny.push(n.top - wb.top + n.height / 2);
        });
        centers = ny; y0 = ny[0]; y1 = ny[ny.length - 1];
        // between two cards the line goes straight, then diagonally, then straight again
        var d = 'M ' + nx[0] + ',' + ny[0];
        for (var i = 1; i < nx.length; i++) {
            var st = Math.round((ny[i] - ny[i - 1]) * 0.3 / 4) * 4;
            d += ' L ' + nx[i - 1] + ',' + (ny[i - 1] + st) +
                 ' L ' + nx[i] + ',' + (ny[i] - st) + ' L ' + nx[i] + ',' + ny[i];
        }
        pathBg.setAttribute('d', d); pathFill.setAttribute('d', d);
        pathLen = pathBg.getTotalLength();
        pathFill.style.strokeDasharray = pathLen + ' ' + pathLen;
        var count = Math.max(2, Math.ceil(pathLen / 3));
        pts = [];
        for (var k = 0; k <= count; k++) {
            var dist = k / count * pathLen, pt = pathBg.getPointAtLength(dist);
            pts.push({ x: pt.x, y: pt.y, dist: dist });
        }
        var vh = (P && P.innerHeight) || 700;
        skyH = Math.min(vh, H);
        sky.style.height = skyH + 'px';
        far.width = near.width = W + 80; far.height = near.height = skyH + 240;
        drawDust(far, 7, 70, 2, 3, 0.30);
        drawDust(near, 19, 34, 4, 6, 0.22);
        if (fe) { sizeFrame(H); }
    }
    layout();
    setInterval(function () { if (fe) { sizeFrame(H); } }, 1500);
    if (window.ResizeObserver) { new ResizeObserver(layout).observe(wrap); }
    window.addEventListener('resize', layout);

    // where the line is at a given height (the line only ever goes down)
    function pointAtY(y) {
        var lo = 0, hi = pts.length - 1;
        while (hi - lo > 1) {
            var mid = (lo + hi) >> 1;
            if (pts[mid].y < y) { lo = mid; } else { hi = mid; }
        }
        var a = pts[lo], b = pts[hi];
        var f = Math.max(0, Math.min(1, (y - a.y) / ((b.y - a.y) || 1)));
        return { x: a.x + (b.x - a.x) * f, y: y, dist: a.dist + (b.dist - a.dist) * f };
    }
    // eight directions like a desktop-pet cat: [pose, mirror, tilt]
    function dirOf(a) {
        if (a > -22.5 && a <= 22.5) { return ['runE', 1, 0]; }
        if (a > 22.5 && a <= 67.5) { return ['runE', 1, 25]; }
        if (a > 67.5 && a <= 112.5) { return ['runS', 1, 0]; }
        if (a > 112.5 && a <= 157.5) { return ['runE', -1, 25]; }
        if (a > 157.5 || a <= -157.5) { return ['runE', -1, 0]; }
        if (a > -157.5 && a <= -112.5) { return ['runE', -1, -25]; }
        if (a > -112.5 && a <= -67.5) { return ['runN', 1, 0]; }
        return ['runE', 1, -25];
    }

    var cy = y0, prevX = null, prevY = null, svx = 0, svy = 0, gait = 0;
    var state = 'sit', stillSince = null, wasMoving = false, alertUntil = 0;
    var flip = 1, rot = 0, rotCur = 0, passed = [], active = -1;
    function frame(now) {
        var r = fe ? fe.getBoundingClientRect() : null;
        var vh = (P && P.innerHeight) || 700, tgt;
        if (r) { tgt = Math.max(y0, Math.min(y1, vh * 0.5 - r.top)); }
        else { tgt = y0 + (Math.sin(now / 3500 - 1.57) * 0.5 + 0.5) * (y1 - y0); }
        var d = tgt - cy;
        cy += d * 0.14;
        if (Math.abs(d) < 0.05) { cy = tgt; }
        var p = pointAtY(cy);
        cat.style.left = p.x + 'px'; cat.style.top = p.y + 'px';
        pathFill.style.strokeDashoffset = (pathLen - p.dist) + 'px';
        if (prevX === null) { prevX = p.x; prevY = p.y; }
        svx += ((p.x - prevX) - svx) * 0.35; svy += ((p.y - prevY) - svy) * 0.35;
        prevX = p.x; prevY = p.y;
        var speed = Math.sqrt(svx * svx + svy * svy);
        if (stillSince === null) { stillSince = now; }
        if (speed > 0.4) {
            if (!wasMoving && now - stillSince > 1500) { alertUntil = now + 380; }   // wakes up, perks up first
            wasMoving = true; stillSince = now;
            gait += speed / 5;                                  // one stride every ~5 px travelled
            var da = dirOf(Math.atan2(svy, svx) * 180 / Math.PI);
            state = now < alertUntil ? 'alert' : da[0]; flip = da[1]; rot = da[2];
        } else {
            if (wasMoving) { wasMoving = false; stillSince = now; }
            var still = (now - stillSince) / 1000;
            state = still < 1.2 ? 'sit' : (still < 3.4 ? 'scratch' :
                    (still < 4.6 ? 'sit' : (still < 8 ? 'tired' : 'sleep')));
            flip = 1; rot = 0;
        }
        rotCur += (rot - rotCur) * 0.3;
        look.update(now);
        drawCatState(cctx, state, now / 1000, gait, look.x, look.y);
        cat.style.transform = 'translate(-50%, -56%) scaleX(' + flip + ') rotate(' + rotCur.toFixed(1) + 'deg)';
        var best = 0, bd = 1e9;
        for (var i = 0; i < rows.length; i++) {
            var on = centers[i] <= cy + 14;
            if (on !== passed[i]) { passed[i] = on; rows[i].classList.toggle('on', on); }
            var dd = Math.abs(centers[i] - cy);
            if (dd < bd) { bd = dd; best = i; }
        }
        if (best !== active) {
            if (active >= 0) { rows[active].classList.remove('active'); }
            rows[best].classList.add('active'); active = best;
        }
        var top = r ? Math.max(0, Math.min(H - skyH, -r.top)) : 0;
        sky.style.transform = 'translateY(' + top + 'px)';
        var prog = r ? Math.max(0, Math.min(1, (vh * 0.5 - r.top - y0) / ((y1 - y0) || 1))) : 0;
        far.style.transform = 'translate(' + (-prog * 20).toFixed(1) + 'px,' + (-prog * 110).toFixed(1) + 'px)';
        near.style.transform = 'translate(' + (-prog * 40).toFixed(1) + 'px,' + (-prog * 230).toFixed(1) + 'px)';
        if (!reduce) { requestAnimationFrame(frame); }
    }
    if (reduce) {
        rows.forEach(function (r) { r.classList.add('on'); });
    }
    frame(0);
    if (!reduce) { requestAnimationFrame(frame); }
})();
</script>
</body>
</html>
"""


def render_story() -> None:
    """The scrolling 'who is Nashmi' story with the cat on the line."""
    cards = "".join(
        f'<div class="row"><i class="node"></i><article class="card"{dir_attr()}>'
        f'<span class="tag">{html.escape(tag)}</span>'
        f'<h3>{html.escape(title)}</h3><p>{html.escape(text)}</p>'
        '</article></div>'
        for tag, title, text in STORY_CARDS[lang()]
    )
    markup = (STORY_TEMPLATE.replace("__CARDS__", cards)
              .replace("__CAT_JS__", CAT_JS + WALKER_JS))
    embed_html(markup, STORY_HEIGHT)

STRINGS = {
    "en": {
        "sign_in": "Sign in", "sign_up": "Sign up", "start": "Start", "back": "Back",
        "sign_out": "Sign out", "lang_btn": "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",
        "hero_badge": "> CYBERSECURITY AI AGENT", "hero_a": "Meet", "hero_b": "Nashmi Agent",
        "hero_sub": ("An AI agent built only for cybersecurity. Analyse incidents, look up "
                     "vulnerabilities, solve CTFs and practise for exams, with your own API "
                     "key and your data kept private."),
        "scene_login_h": "Welcome back", "scene_login_p": "Sign in to pick up where you left off.",
        "scene_signup_h": "Join Nashmi", "scene_signup_p": "Create your account in one tap with Google.",
        "login_title": "Welcome back",
        "login_sub": "Sign in with Google to pick up where you left off.",
        "signup_title": "Create your account",
        "signup_sub": "One tap with Google, so there is no password to invent.",
        "signup_steps": ("Continue with Google, which checks your e-mail for us",
                         "Choose your name and a username", "Start using Nashmi"),
        "login_steps": ("Sign in works only with an account you already created",
                        "Your chats and projects are saved", "Usage history across devices"),
        "google_note": "You will be sent to Google to verify who you are. Nashmi never sees your password.",
        "continue_google": "Continue with Google",
        "privacy": ("Signing in stores your name, e-mail address and sign-in times. Your chats "
                    "and saved keys are encrypted."),
        "rights": "&copy; 2026 Hassan Aldajah. All rights reserved.",
        "questions": "Questions or feedback?",
        "fs_title": "Finish creating your account",
        "fs_p": "No Nashmi account exists yet for {email}. Choose your name and username to create it.",
        "fs_email": "E-mail (checked by Google)", "fs_name": "Full name", "fs_username": "Username",
        "fs_user_help": "3 to 24 characters: lowercase letters, digits, . or _",
        "fs_ph": "e.g. hassan.sec", "fs_create": "Create account",
        "fs_accept": "Please accept the note above to continue.",
        "fs_cancel": "Cancel and sign out",
        "mfa_title": "Turn on two-step verification",
        "mfa_steps": ("1. Open Google Authenticator (or any authenticator app) and scan this "
                      "code.<br>2. Type the 6-digit code the app shows."),
        "mfa_cant": "Cannot scan? Type this key into the app:",
        "mfa_code": "6-digit code", "mfa_verify": "Verify and turn on",
        "mfa_bad": "That code is not right. Wait for the next one and try again.",
        "codes_title": "Save your backup codes",
        "codes_p": ("Each code works once if you lose your phone. They are shown <b>only now</b>, "
                    "so copy them somewhere safe."),
        "codes_saved": "I saved these codes", "codes_continue": "Continue",
        "ch_title": "Enter your verification code",
        "ch_p": ("Open your authenticator app and type the 6-digit code. Lost your phone? Use one "
                 "of your backup codes."),
        "ch_locked": "Too many wrong codes. Try again at {t} UTC.",
        "ch_code": "Code", "ch_ph": "123456 or backup code", "ch_verify": "Verify",
        "ch_bad": "That code is not right.",
    },
    "ar": {
        "sign_in": "\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644",
        "sign_up": "\u0625\u0646\u0634\u0627\u0621 \u062d\u0633\u0627\u0628",
        "start": "\u0627\u0628\u062f\u0623", "back": "\u0631\u062c\u0648\u0639",
        "sign_out": "\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062e\u0631\u0648\u062c",
        "lang_btn": "English",
        "hero_badge": "> \u0648\u0643\u064a\u0644 \u0623\u0645\u0646 \u0633\u064a\u0628\u0631\u0627\u0646\u064a \u0628\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a",
        "hero_a": "\u062a\u0639\u0631\u0651\u0641 \u0639\u0644\u0649", "hero_b": "\u0646\u0634\u0645\u064a",
        "hero_sub": ("\u0648\u0643\u064a\u0644 \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a \u0645\u062e\u0635\u0635 \u0644\u0644\u0623\u0645\u0646 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a \u0641\u0642\u0637. "
                     "\u062d\u0644\u0651\u0644 \u0627\u0644\u062d\u0648\u0627\u062f\u062b\u060c \u0648\u0627\u0628\u062d\u062b \u0639\u0646 \u0627\u0644\u062b\u063a\u0631\u0627\u062a\u060c \u0648\u062d\u0644\u0651 \u062a\u062d\u062f\u064a\u0627\u062a CTF\u060c "
                     "\u0648\u062a\u062f\u0631\u0651\u0628 \u0644\u0644\u0627\u0645\u062a\u062d\u0627\u0646\u0627\u062a\u060c \u0628\u0645\u0641\u062a\u0627\u062d API \u0627\u0644\u062e\u0627\u0635 \u0628\u0643 "
                     "\u0648\u0628\u064a\u0627\u0646\u0627\u062a \u062a\u0628\u0642\u0649 \u062e\u0627\u0635\u0629 \u0628\u0643."),
        "scene_login_h": "\u0623\u0647\u0644\u0627\u064b \u0628\u0639\u0648\u062f\u062a\u0643",
        "scene_login_p": "\u0633\u062c\u0651\u0644 \u0627\u0644\u062f\u062e\u0648\u0644 \u0644\u062a\u0643\u0645\u0644 \u0645\u0646 \u062d\u064a\u062b \u062a\u0648\u0642\u0641\u062a.",
        "scene_signup_h": "\u0627\u0646\u0636\u0645 \u0625\u0644\u0649 \u0646\u0634\u0645\u064a",
        "scene_signup_p": "\u0623\u0646\u0634\u0626 \u062d\u0633\u0627\u0628\u0643 \u0628\u0636\u063a\u0637\u0629 \u0648\u0627\u062d\u062f\u0629 \u0639\u0628\u0631 Google.",
        "login_title": "\u0623\u0647\u0644\u0627\u064b \u0628\u0639\u0648\u062f\u062a\u0643",
        "login_sub": "\u0633\u062c\u0651\u0644 \u0627\u0644\u062f\u062e\u0648\u0644 \u0639\u0628\u0631 Google \u0644\u062a\u0643\u0645\u0644 \u0645\u0646 \u062d\u064a\u062b \u062a\u0648\u0642\u0641\u062a.",
        "signup_title": "\u0623\u0646\u0634\u0626 \u062d\u0633\u0627\u0628\u0643",
        "signup_sub": "\u0628\u0636\u063a\u0637\u0629 \u0648\u0627\u062d\u062f\u0629 \u0639\u0628\u0631 Google\u060c \u0628\u0644\u0627 \u0643\u0644\u0645\u0629 \u0633\u0631 \u062c\u062f\u064a\u062f\u0629 \u062a\u062e\u062a\u0627\u0631\u0647\u0627.",
        "signup_steps": ("\u062a\u0627\u0628\u0639 \u0639\u0628\u0631 Google\u060c \u0641\u0647\u0648 \u064a\u062a\u062d\u0642\u0642 \u0645\u0646 \u0628\u0631\u064a\u062f\u0643 \u0627\u0644\u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a",
                         "\u0627\u062e\u062a\u0631 \u0627\u0633\u0645\u0643 \u0648\u0627\u0633\u0645 \u0645\u0633\u062a\u062e\u062f\u0645", "\u0627\u0628\u062f\u0623 \u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0646\u0634\u0645\u064a"),
        "login_steps": ("\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644 \u064a\u0639\u0645\u0644 \u0641\u0642\u0637 \u0645\u0639 \u062d\u0633\u0627\u0628 \u0623\u0646\u0634\u0623\u062a\u0647 \u0645\u0633\u0628\u0642\u0627\u064b",
                        "\u0645\u062d\u0627\u062f\u062b\u0627\u062a\u0643 \u0648\u0645\u0634\u0627\u0631\u064a\u0639\u0643 \u0645\u062d\u0641\u0648\u0638\u0629",
                        "\u0633\u062c\u0644 \u0627\u0644\u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0639\u0628\u0631 \u0623\u062c\u0647\u0632\u062a\u0643"),
        "google_note": "\u0633\u064a\u062a\u0645 \u062a\u062d\u0648\u064a\u0644\u0643 \u0625\u0644\u0649 Google \u0644\u0644\u062a\u062d\u0642\u0642 \u0645\u0646 \u0647\u0648\u064a\u062a\u0643. \u0644\u0627 \u064a\u0637\u0651\u0644\u0639 \u0646\u0634\u0645\u064a \u0639\u0644\u0649 \u0643\u0644\u0645\u0629 \u0627\u0644\u0633\u0631 \u0623\u0628\u062f\u0627\u064b.",
        "continue_google": "\u0627\u0644\u0645\u062a\u0627\u0628\u0639\u0629 \u0639\u0628\u0631 Google",
        "privacy": ("\u064a\u062d\u0641\u0638 \u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644 \u0627\u0633\u0645\u0643 \u0648\u0628\u0631\u064a\u062f\u0643 \u0627\u0644\u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a \u0648\u0623\u0648\u0642\u0627\u062a \u0627\u0644\u062f\u062e\u0648\u0644. "
                    "\u0645\u062d\u0627\u062f\u062b\u0627\u062a\u0643 \u0648\u0645\u0641\u0627\u062a\u064a\u062d\u0643 \u0627\u0644\u0645\u062d\u0641\u0648\u0638\u0629 \u0645\u0634\u0641\u0651\u0631\u0629."),
        "rights": "&copy; 2026 \u062d\u0633\u0646 \u0627\u0644\u062f\u0639\u062c\u0629. \u062c\u0645\u064a\u0639 \u0627\u0644\u062d\u0642\u0648\u0642 \u0645\u062d\u0641\u0648\u0638\u0629.",
        "questions": "\u0623\u0633\u0626\u0644\u0629 \u0623\u0648 \u0645\u0644\u0627\u062d\u0638\u0627\u062a\u061f",
        "fs_title": "\u0623\u0643\u0645\u0644 \u0625\u0646\u0634\u0627\u0621 \u062d\u0633\u0627\u0628\u0643",
        "fs_p": "\u0644\u0627 \u064a\u0648\u062c\u062f \u062d\u0633\u0627\u0628 \u0641\u064a \u0646\u0634\u0645\u064a \u0644\u0640 {email} \u0628\u0639\u062f. \u0627\u062e\u062a\u0631 \u0627\u0633\u0645\u0643 \u0648\u0627\u0633\u0645 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u0644\u0625\u0646\u0634\u0627\u0626\u0647.",
        "fs_email": "\u0627\u0644\u0628\u0631\u064a\u062f \u0627\u0644\u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a (\u062a\u062d\u0642\u0651\u0642\u062a \u0645\u0646\u0647 Google)",
        "fs_name": "\u0627\u0644\u0627\u0633\u0645 \u0627\u0644\u0643\u0627\u0645\u0644", "fs_username": "\u0627\u0633\u0645 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645",
        "fs_user_help": "\u0645\u0646 3 \u0625\u0644\u0649 24 \u062d\u0631\u0641\u0627\u064b: \u0623\u062d\u0631\u0641 \u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629 \u0635\u063a\u064a\u0631\u0629 \u0648\u0623\u0631\u0642\u0627\u0645 \u0648\u0646\u0642\u0637\u0629 \u0623\u0648 \u0634\u0631\u0637\u0629 \u0633\u0641\u0644\u064a\u0629",
        "fs_ph": "\u0645\u062b\u0627\u0644: hassan.sec", "fs_create": "\u0625\u0646\u0634\u0627\u0621 \u0627\u0644\u062d\u0633\u0627\u0628",
        "fs_accept": "\u064a\u0631\u062c\u0649 \u0627\u0644\u0645\u0648\u0627\u0641\u0642\u0629 \u0639\u0644\u0649 \u0627\u0644\u0645\u0644\u0627\u062d\u0638\u0629 \u0623\u0639\u0644\u0627\u0647 \u0644\u0644\u0645\u062a\u0627\u0628\u0639\u0629.",
        "fs_cancel": "\u0625\u0644\u063a\u0627\u0621 \u0648\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062e\u0631\u0648\u062c",
        "mfa_title": "\u0641\u0639\u0651\u0644 \u0627\u0644\u062a\u062d\u0642\u0642 \u0628\u062e\u0637\u0648\u062a\u064a\u0646",
        "mfa_steps": ("1. \u0627\u0641\u062a\u062d \u062a\u0637\u0628\u064a\u0642 Google Authenticator (\u0623\u0648 \u0623\u064a \u062a\u0637\u0628\u064a\u0642 \u0645\u0635\u0627\u062f\u0642\u0629) "
                      "\u0648\u0627\u0645\u0633\u062d \u0647\u0630\u0627 \u0627\u0644\u0631\u0645\u0632.<br>2. \u0627\u0643\u062a\u0628 \u0627\u0644\u0631\u0645\u0632 \u0627\u0644\u0645\u0643\u0648\u0651\u0646 \u0645\u0646 6 \u0623\u0631\u0642\u0627\u0645 \u0627\u0644\u0630\u064a \u064a\u0639\u0631\u0636\u0647 \u0627\u0644\u062a\u0637\u0628\u064a\u0642."),
        "mfa_cant": "\u0644\u0627 \u062a\u0633\u062a\u0637\u064a\u0639 \u0627\u0644\u0645\u0633\u062d\u061f \u0627\u0643\u062a\u0628 \u0647\u0630\u0627 \u0627\u0644\u0645\u0641\u062a\u0627\u062d \u0641\u064a \u0627\u0644\u062a\u0637\u0628\u064a\u0642:",
        "mfa_code": "\u0627\u0644\u0631\u0645\u0632 \u0627\u0644\u0645\u0643\u0648\u0651\u0646 \u0645\u0646 6 \u0623\u0631\u0642\u0627\u0645", "mfa_verify": "\u062a\u062d\u0642\u0651\u0642 \u0648\u0641\u0639\u0651\u0644",
        "mfa_bad": "\u0627\u0644\u0631\u0645\u0632 \u063a\u064a\u0631 \u0635\u062d\u064a\u062d. \u0627\u0646\u062a\u0638\u0631 \u0627\u0644\u0631\u0645\u0632 \u0627\u0644\u062a\u0627\u0644\u064a \u0648\u062d\u0627\u0648\u0644 \u0645\u062c\u062f\u062f\u0627\u064b.",
        "codes_title": "\u0627\u062d\u0641\u0638 \u0623\u0643\u0648\u0627\u062f \u0627\u0644\u0627\u062d\u062a\u064a\u0627\u0637",
        "codes_p": ("\u0643\u0644 \u0643\u0648\u062f \u064a\u0639\u0645\u0644 \u0645\u0631\u0629 \u0648\u0627\u062d\u062f\u0629 \u0625\u0630\u0627 \u0641\u0642\u062f\u062a \u0647\u0627\u062a\u0641\u0643. \u062a\u0638\u0647\u0631 <b>\u0627\u0644\u0622\u0646 \u0641\u0642\u0637</b>\u060c "
                    "\u0641\u0627\u0646\u0633\u062e\u0647\u0627 \u0641\u064a \u0645\u0643\u0627\u0646 \u0622\u0645\u0646."),
        "codes_saved": "\u062d\u0641\u0638\u062a \u0647\u0630\u0647 \u0627\u0644\u0623\u0643\u0648\u0627\u062f", "codes_continue": "\u0645\u062a\u0627\u0628\u0639\u0629",
        "ch_title": "\u0623\u062f\u062e\u0644 \u0631\u0645\u0632 \u0627\u0644\u062a\u062d\u0642\u0642",
        "ch_p": ("\u0627\u0641\u062a\u062d \u062a\u0637\u0628\u064a\u0642 \u0627\u0644\u0645\u0635\u0627\u062f\u0642\u0629 \u0648\u0627\u0643\u062a\u0628 \u0627\u0644\u0631\u0645\u0632 \u0627\u0644\u0645\u0643\u0648\u0651\u0646 \u0645\u0646 6 \u0623\u0631\u0642\u0627\u0645. "
                 "\u0641\u0642\u062f\u062a \u0647\u0627\u062a\u0641\u0643\u061f \u0627\u0633\u062a\u062e\u062f\u0645 \u0623\u062d\u062f \u0623\u0643\u0648\u0627\u062f \u0627\u0644\u0627\u062d\u062a\u064a\u0627\u0637."),
        "ch_locked": "\u0645\u062d\u0627\u0648\u0644\u0627\u062a \u062e\u0627\u0637\u0626\u0629 \u0643\u062b\u064a\u0631\u0629. \u062d\u0627\u0648\u0644 \u0645\u062c\u062f\u062f\u0627\u064b \u0639\u0646\u062f {t} (UTC).",
        "ch_code": "\u0627\u0644\u0631\u0645\u0632", "ch_ph": "123456 \u0623\u0648 \u0643\u0648\u062f \u0627\u062d\u062a\u064a\u0627\u0637", "ch_verify": "\u062a\u062d\u0642\u0651\u0642",
        "ch_bad": "\u0627\u0644\u0631\u0645\u0632 \u063a\u064a\u0631 \u0635\u062d\u064a\u062d.",
    },
}

# messages from the database layer, translated for the sign-up form
DB_ERRORS_AR = {
    "Please enter your name.": "\u064a\u0631\u062c\u0649 \u0625\u062f\u062e\u0627\u0644 \u0627\u0633\u0645\u0643.",
    "The username needs 3 to 24 characters: lowercase letters, digits, dot or underscore.":
        "\u0627\u0633\u0645 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u0645\u0646 3 \u0625\u0644\u0649 24 \u062d\u0631\u0641\u0627\u064b: \u0623\u062d\u0631\u0641 \u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629 \u0635\u063a\u064a\u0631\u0629 \u0648\u0623\u0631\u0642\u0627\u0645 \u0648\u0646\u0642\u0637\u0629 \u0623\u0648 \u0634\u0631\u0637\u0629 \u0633\u0641\u0644\u064a\u0629.",
    "That username is reserved. Please pick another one.":
        "\u0647\u0630\u0627 \u0627\u0644\u0627\u0633\u0645 \u0645\u062d\u062c\u0648\u0632\u060c \u0627\u062e\u062a\u0631 \u0627\u0633\u0645\u0627\u064b \u0622\u062e\u0631.",
    "That username is already taken.": "\u0627\u0633\u0645 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u0645\u0633\u062a\u062e\u062f\u0645 \u0628\u0627\u0644\u0641\u0639\u0644.",
}


STRINGS["en"].update({
    "about_title": "About this application",
    "about_text": ("Nashmi Agent is an independent project by Hassan Aldajah, a cybersecurity "
                   "student at the University of Jordan. It is still growing and its answers can "
                   "be wrong, so always double-check anything important."),
})
STRINGS["ar"].update({
    "about_title": "عن هذا التطبيق",
    "about_text": ("نشمي مشروع مستقل من تطوير حسن الدعجة، طالب أمن سيبراني "
                   "في الجامعة الأردنية. ما زال يتطور وقد تخطئ إجاباته، "
                   "لذلك تحقّق دائماً من أي معلومة مهمة."),
})


STRINGS["en"].update({
    "login_title": "Welcome to Nashmi",
    "login_sub": "Continue with Google. New here? Your account is created the first time.",
    "login_steps": ("Google checks your e-mail, Nashmi never sees a password",
                    "First time? Choose your name and a username",
                    "Then start using Nashmi"),
    "scene_login_h": "Welcome to Nashmi", "scene_login_p": "Continue with Google to start.",
})
STRINGS["ar"].update({
    "login_title": "أهلاً بك في نشمي",
    "login_sub": "تابع عبر Google. جديد هنا؟ يُنشأ حسابك أول مرة.",
    "login_steps": ("Google تتحقق من بريدك، ونشمي لا يرى كلمة السر",
                    "أول مرة؟ اختر اسمك واسم مستخدم",
                    "ثم ابدأ استخدام نشمي"),
    "scene_login_h": "أهلاً بك في نشمي",
    "scene_login_p": "تابع عبر Google للبدء.",
})


def lang() -> str:
    """'en' or 'ar'. The first visit follows the browser language; the language
    button changes it. Switching only changes words and text direction, never
    the layout or the pictures."""
    current = st.session_state.get("lang")
    if current in ("en", "ar"):
        return current
    guess = "en"
    try:
        if str(st.context.headers.get("Accept-Language", "")).lower().startswith("ar"):
            guess = "ar"
    except Exception:  # noqa: BLE001 - no browser headers (tests)
        pass
    st.session_state["lang"] = guess
    return guess


def tt(en: str, ar: str) -> str:
    """Inline translation for the in-app text."""
    return ar if lang() == "ar" else en


TRACK_AR = {
    "Security Intelligence": "\u0627\u0633\u062a\u062e\u0628\u0627\u0631\u0627\u062a \u0627\u0644\u0623\u0645\u0646",
    "CPSS": "\u0623\u0645\u0646 \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a\u0629-\u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629",
    "Chat": "\u0645\u062d\u0627\u062f\u062b\u0629",
    "Secure Coding & SQL": "\u0627\u0644\u0628\u0631\u0645\u062c\u0629 \u0627\u0644\u0622\u0645\u0646\u0629 \u0648SQL",
    "Penetration Testing": "\u0627\u062e\u062a\u0628\u0627\u0631 \u0627\u0644\u0627\u062e\u062a\u0631\u0627\u0642",
    "CTF": "CTF",
    "Malware Analysis": "\u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0628\u0631\u0645\u062c\u064a\u0627\u062a \u0627\u0644\u062e\u0628\u064a\u062b\u0629",
    "Forensics & IR": "\u0627\u0644\u062a\u062d\u0642\u064a\u0642 \u0627\u0644\u062c\u0646\u0627\u0626\u064a \u0648\u0627\u0644\u0627\u0633\u062a\u062c\u0627\u0628\u0629",
    "Network Security": "\u0623\u0645\u0646 \u0627\u0644\u0634\u0628\u0643\u0627\u062a",
    "Cloud Security": "\u0623\u0645\u0646 \u0627\u0644\u0633\u062d\u0627\u0628\u0629",
    "SOC Analyzer": "\u0645\u062d\u0644\u0651\u0644 SOC",
    "Toolbox": "\u0635\u0646\u062f\u0648\u0642 \u0627\u0644\u0623\u062f\u0648\u0627\u062a",
    "Practice Exam": "\u0627\u0645\u062a\u062d\u0627\u0646 \u062a\u062f\u0631\u064a\u0628\u064a",
}
PROVIDER_NOTE_AR = {
    "gemini": "\u0646\u0633\u062e\u0629 \u0645\u062c\u0627\u0646\u064a\u0629 \u0628\u062d\u062f\u0648\u062f \u0641\u064a \u0627\u0644\u062f\u0642\u064a\u0642\u0629 \u0648\u0641\u064a \u0627\u0644\u064a\u0648\u0645.",
    "groq": "\u0646\u0633\u062e\u0629 \u0645\u062c\u0627\u0646\u064a\u0629 \u0628\u062d\u062f\u0648\u062f \u0645\u0639\u062f\u0651\u0644. GPT-OSS \u0646\u0645\u0648\u0630\u062c \u0645\u0641\u062a\u0648\u062d \u0645\u0646 OpenAI \u0648\u0644\u064a\u0633 ChatGPT \u0646\u0641\u0633\u0647.",
    "openai": "\u062a\u064f\u062d\u0627\u0633\u064e\u0628 \u0639\u0644\u0649 \u062d\u0633\u0627\u0628\u0643 \u0641\u064a OpenAI. \u0627\u0644\u0642\u0627\u0626\u0645\u0629 \u062a\u0639\u0631\u0636 \u0627\u0644\u0646\u0645\u0627\u0630\u062c \u0627\u0644\u062a\u064a \u064a\u0633\u0645\u062d \u0628\u0647\u0627 \u0645\u0641\u062a\u0627\u062d\u0643\u060c \u0648\u064a\u0645\u0643\u0646\u0643 \u0643\u062a\u0627\u0628\u0629 \u0623\u064a \u0627\u0633\u0645 \u0622\u062e\u0631 \u0636\u0645\u0646 \u0627\u0644\u0646\u0645\u0648\u0630\u062c \u0627\u0644\u0645\u062e\u0635\u0635.",
    "anthropic": "\u062a\u064f\u062d\u0627\u0633\u064e\u0628 \u0639\u0644\u0649 \u062d\u0633\u0627\u0628\u0643 \u0641\u064a Anthropic. \u0627\u0644\u0642\u0627\u0626\u0645\u0629 \u062a\u0639\u0631\u0636 \u0627\u0644\u0646\u0645\u0627\u0630\u062c \u0627\u0644\u0645\u062a\u0627\u062d\u0629 \u0644\u0645\u0641\u062a\u0627\u062d\u0643.",
}


def on_app_lang_pick() -> None:
    st.session_state["lang"] = st.session_state["side_lang"]


def tr(key: str, **fmt):
    value = STRINGS[lang()].get(key, STRINGS["en"][key])
    return value.format(**fmt) if fmt and isinstance(value, str) else value


def dir_attr() -> str:
    return ' dir="rtl" lang="ar"' if lang() == "ar" else ' lang="en"'


def db_error(message: str) -> str:
    return DB_ERRORS_AR.get(message, message) if lang() == "ar" else message


# Arabic text sits inside the same boxes; only the reading direction of the words changes
RTL_CSS = """
.st-key-login_card [data-testid="stCaptionContainer"],
.st-key-login_card [data-testid="stWidgetLabel"], .st-key-login_card label,
.st-key-login_card button, .st-key-topbar button, .lp-signin { direction: rtl; }
.st-key-login_card input { direction: ltr; text-align: left; }
.lp-signin ul { text-align: right; padding-left: 0; padding-right: 1.2rem; }
.st-key-login_card [data-testid="stCaptionContainer"] { text-align: center; }
"""


LANG_SHORT = {"en": "EN", "ar": "AR"}


def on_lang_pick() -> None:
    st.session_state["lang"] = st.session_state["topbar_lang"]


def go_gate(name: str) -> None:
    st.session_state["gate"] = name


def current_gate() -> str:
    """'landing', 'login', 'signup' or 'app'. Only a shared (public) server
    gates, and once accounts work there is no guest way in."""
    if not is_public() or current_user():
        return "app"
    gate = st.session_state.get("gate", "landing")
    if gate == "signup":                  # there is no separate sign-up page any more
        gate = "login"
    if gate == "app" and accounts_enabled():
        return "landing"
    return gate


OWNER_NAME = "Hassan Aldajah"
OWNER_LINKS = (
    ("GitHub", "https://github.com/hassanaldajah",
     "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113"
     ".82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042"
     "-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 "
     "1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998."
     "108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31."
     "465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 "
     "1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 "
     "3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 "
     "1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 "
     "1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 "
     "17.592 24 12.297c0-6.627-5.373-12-12-12"),
    ("LinkedIn", "https://www.linkedin.com/in/hassan-aldajah04",
     "M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-"
     "2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-"
     "1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-"
     "1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 "
     "2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H"
     "3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 "
     "23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 "
     ".774 23.2 0 22.222 0h.003z"),
    ("Email", "mailto:hassanaldajah2004@gmail.com",
     "M2 5.5A2.5 2.5 0 0 1 4.5 3h15A2.5 2.5 0 0 1 22 5.5v13a2.5 2.5 0 0 1-"
     "2.5 2.5h-15A2.5 2.5 0 0 1 2 18.5v-13zm2.2.3L12 11.9l7.8-6.1H4.2zM20 "
     "7.9l-7.4 5.8a1 1 0 0 1-1.2 0L4 7.9v10.6c0 .3.2.5.5.5h15c.3 0 .5-.2."
     "5-.5V7.9z"),
)


def render_footer() -> None:
    """Copyright line and the owner's contact links."""
    icons = "".join(
        f'<a href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer" aria-label="{html.escape(label)}" '
        f'title="{html.escape(label)}"><svg viewBox="0 0 24 24" '
        f'aria-hidden="true"><path d="{path}"/></svg></a>'
        for label, url, path in OWNER_LINKS
    )
    note = (f'<div class="lp-footer-note"{dir_attr()}>{html.escape(tr("privacy"))}</div>'
            if accounts_enabled() else "")
    st.markdown(
        '<div class="lp-footer">' + note +
        f'<div class="lp-social">{icons}</div>'
        f'<div{dir_attr()}>{tr("rights")}</div>'
        f'<div class="lp-footer-sub"{dir_attr()}>{html.escape(tr("questions"))} '
        '<a href="mailto:hassanaldajah2004@gmail.com" dir="ltr">'
        'hassanaldajah2004@gmail.com</a></div></div>',
        unsafe_allow_html=True,
    )


MINI_LOGO_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<style>
    html, body { margin: 0; height: 100%; background: transparent; overflow: hidden;
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }
    .row { display: flex; align-items: center; height: 100%; gap: 8px; }
    .logo { position: relative; width: 44px; height: 44px; flex: none; }
    .logo canvas { position: absolute; left: 0; top: 0; width: 44px; height: 44px; }
    .logo img { position: absolute; left: 6px; top: 6px; width: 32px; height: 32px;
        border-radius: 50%; object-fit: cover; image-rendering: pixelated;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.16), 0 0 14px rgba(255, 59, 63, 0.55); }
    .name { color: #fff; font-weight: 800; font-size: 19px; letter-spacing: 0.01em; white-space: nowrap; }
    .name span { color: #E5383B; font-family: ui-monospace, Consolas, monospace; }
    @media (max-width: 170px) { .name span { display: none; } .name { font-size: 16px; } }
    @media (max-width: 135px) { .name { display: none; } }
</style></head>
<body><div class="row" role="button" tabindex="0" title="Home" style="cursor:pointer"><div class="logo"><canvas id="c"></canvas><img alt="Nashmi" src="__AVATAR__"></div>
<div class="name"><span>&gt;_</span> Nashmi</div></div>
<script src="__THREE_URL__" integrity="__THREE_SRI__" crossorigin="anonymous"></script>
<script>
(function () {
    if (!window.THREE) { return; }
    var c = document.getElementById('c'), r;
    try { r = new THREE.WebGLRenderer({ canvas: c, alpha: true, antialias: true }); }
    catch (err) { c.style.display = 'none'; return; }
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.setSize(44, 44, false);
    var sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    cam.position.z = 4.2;
    var ring = new THREE.Mesh(new THREE.TorusGeometry(1.42, 0.035, 8, 80),
        new THREE.MeshBasicMaterial({ color: 0xFF3B3F }));
    var ring2 = new THREE.Mesh(new THREE.TorusGeometry(1.42, 0.02, 8, 80),
        new THREE.MeshBasicMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.7 }));
    sc.add(ring, ring2);
    var dots = new THREE.Group(); sc.add(dots);
    for (var i = 0; i < 3; i++) {
        var d = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 8),
            new THREE.MeshBasicMaterial({ color: i % 2 ? 0xFFFFFF : 0xFF3B3F }));
        var a = i * 2.094;
        d.position.set(Math.cos(a) * 1.42, Math.sin(a) * 1.42, 0);
        dots.add(d);
    }
    var reduce = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    function f(now) {
        var t = now / 1000;
        ring.rotation.x = 1.0 + Math.sin(t * 0.8) * 0.3; ring.rotation.y = t * 0.9;
        ring2.rotation.x = -0.9; ring2.rotation.y = -t * 0.7;
        dots.rotation.z = t * 1.4; dots.rotation.x = 0.5;
        r.render(sc, cam);
        if (!reduce) { requestAnimationFrame(f); }
    }
    f(0);
    if (!reduce) { requestAnimationFrame(f); }
})();
(function () {
    var row = document.querySelector('.row');
    function home() {
        try {
            var b = window.parent.document.querySelector('.st-key-topbar_home button');
            if (b) { b.click(); }
        } catch (err) { }
    }
    row.addEventListener('click', home);
    row.addEventListener('keydown', function (e) { if (e.key === 'Enter') { home(); } });
})();
</script></body></html>
"""


def render_topbar(mode: str = "landing") -> None:
    """Floating pill bar: 3D logo and name on the left, language button and the
    actions that make sense on this page on the right."""
    if mode == "landing":
        actions = ([("landing_signin", "sign_in", "primary", "login", None)]
                   if accounts_enabled() else
                   [("landing_guest", "start", "primary", "app", None)])
    elif mode == "login":
        actions = [("topbar_back", "back", "tertiary", "landing", ":material/arrow_back:")]
    else:                                                  # finishing sign-up / two-step
        actions = [("topbar_signout", "sign_out", "tertiary", None, ":material/logout:")]
    with st.container(key="topbar"):
        cols = st.columns([5, 1.6] + [2] * len(actions),
                          vertical_alignment="center", gap="small")
        with cols[0]:
            # the page style rides with the first element so nothing sits above the bar
            st.markdown(f'<style>{LANDING_CSS}{RTL_CSS if lang() == "ar" else ""}</style>',
                        unsafe_allow_html=True)
            embed_html(MINI_LOGO_TEMPLATE.replace("__THREE_URL__", THREE_URL)
                       .replace("__THREE_SRI__", THREE_SRI)
                       .replace("__AVATAR__", avatar_data_uri()), 48)
            st.button("Home", key="topbar_home", on_click=go_gate, args=("landing",))
        st.session_state["topbar_lang"] = lang()
        cols[1].selectbox("Language", ["en", "ar"], key="topbar_lang",
                          format_func=LANG_SHORT.get, label_visibility="collapsed",
                          on_change=on_lang_pick)
        for col, (key, label, kind, target, icon) in zip(cols[2:], actions):
            if target is None:
                col.button(tr(label), key=key, type=kind, on_click=sign_out,
                           icon=icon, width="stretch")
            else:
                col.button(tr(label), key=key, type=kind, on_click=go_gate,
                           args=(target,), icon=icon, width="stretch")


def render_about() -> None:
    """A small 'about' card under the story."""
    chips = "".join(f"<span>{name}</span>" for name in
                    ("Python", "Streamlit", "PostgreSQL", "Three.js", "OAuth 2.0 + 2FA"))
    st.markdown(
        f'<div class="lp-about"{dir_attr()}><b>{html.escape(tr("about_title"))}</b>'
        f'<p>{html.escape(tr("about_text"))}</p><div class="lp-chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    """First page a visitor of the public site sees."""
    render_topbar("landing")
    render_scene("landing")
    render_story()
    render_about()
    render_footer()


def needs_signup(user_id: int | None) -> bool:
    """True for a signed-in Google user who has not finished signing up."""
    db = get_db()
    if not (user_id and db):
        return False
    if st.session_state.get("_profile_ok") == user_id:
        return False
    try:
        if db.get_profile(user_id):
            st.session_state["_profile_ok"] = user_id
            return False
    except Exception:  # noqa: BLE001 - never lock people out of a broken DB
        LOGGER.exception("Could not read the profile")
        return False
    return True


def narrow_page() -> None:
    """Single-column pages (finishing sign-up, two-step) use a narrower page."""
    st.markdown("<style>.block-container { max-width: 620px !important; }</style>",
                unsafe_allow_html=True)


def render_complete_signup(user_id: int, user: dict) -> None:
    """Mandatory last step: nobody gets into the app without an account."""
    render_topbar("onboarding")
    narrow_page()
    render_scene("signup")
    st.markdown(
        f'<div class="lp-signin"{dir_attr()}><h2>{html.escape(tr("fs_title"))}</h2>'
        f'<p>{tr("fs_p", email="<b>" + html.escape(user["email"] or "Google") + "</b>")}</p></div>',
        unsafe_allow_html=True)
    with st.container(key="login_card"):
        with st.form("complete_signup", border=False):
            st.text_input(tr("fs_email"), value=user["email"], disabled=True)
            full_name = st.text_input(tr("fs_name"), value=user["name"], max_chars=120)
            username = st.text_input(tr("fs_username"), max_chars=24,
                                     placeholder=tr("fs_ph"), help=tr("fs_user_help"))
            agree = st.checkbox(tr("privacy"))
            go = st.form_submit_button(tr("fs_create"), type="primary", width="stretch")
        if go:
            if not agree:
                st.error(tr("fs_accept"))
            else:
                error = get_db().create_profile(user_id, username, full_name)
                if error:
                    st.error(db_error(error))
                else:
                    st.session_state["_profile_ok"] = user_id
                    st.rerun()
        st.button(tr("fs_cancel"), key="signup_cancel", on_click=sign_out,
                  type="tertiary", width="stretch", icon=":material/logout:")
    render_footer()


# ---------------------------------------------------------------------------
# Two-step verification: an authenticator app (TOTP) after the Google sign-in
# ---------------------------------------------------------------------------
MFA_ISSUER = "Nashmi Agent"


def _mfa_libs():
    try:
        import pyotp
        import qrcode
        return pyotp, qrcode
    except ImportError:  # the packages are listed in requirements.txt
        return None, None


def mfa_required() -> bool:
    """On by default. NASHMI_REQUIRE_2FA=0 switches it off (for example if the
    owner is locked out), and it stays off if the libraries are missing."""
    return server_setting("NASHMI_REQUIRE_2FA") != "0" and _mfa_libs()[0] is not None


def mfa_passed(user_id) -> bool:
    return (not mfa_required()) or st.session_state.get("_mfa_ok") == user_id


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().lower().replace(" ", "").encode()).hexdigest()


def new_backup_codes(count: int = 8) -> list[str]:
    return [f"{secrets.token_hex(4)}-{secrets.token_hex(4)}" for _ in range(count)]


def totp_step(secret: str, code: str) -> int | None:
    """The 30-second step a correct code belongs to, or None. A code from the
    previous or next step is accepted to allow for clock drift."""
    pyotp, _ = _mfa_libs()
    code = (code or "").strip().replace(" ", "")
    if not (pyotp and code.isdigit() and len(code) == 6):
        return None
    totp, now = pyotp.TOTP(secret), datetime.now(timezone.utc)
    for offset in (-1, 0, 1):
        moment = now + timedelta(seconds=30 * offset)
        if hmac.compare_digest(totp.at(moment), code):
            return int(totp.timecode(moment))
    return None


def qr_png(uri: str) -> bytes:
    _, qrcode = _mfa_libs()
    buf = io.BytesIO()
    qrcode.make(uri, box_size=6, border=2).save(buf, "PNG")
    return buf.getvalue()


def _mfa_page_start() -> None:
    render_topbar("onboarding")
    narrow_page()


def render_mfa_setup(user_id: int, user: dict) -> None:
    """First sign-in: link an authenticator app, then save the backup codes."""
    pyotp, _ = _mfa_libs()
    _mfa_page_start()
    codes = st.session_state.get("_mfa_codes")
    if codes:
        st.markdown(
            f'<div class="lp-signin"{dir_attr()}><h2>{html.escape(tr("codes_title"))}</h2>'
            f'<p>{tr("codes_p")}</p></div>', unsafe_allow_html=True)
        with st.container(key="login_card"):
            st.code("\n".join(codes), language=None)
            saved = st.checkbox(tr("codes_saved"))
            if st.button(tr("codes_continue"), type="primary", width="stretch",
                         disabled=not saved, key="mfa_codes_done"):
                st.session_state.pop("_mfa_codes", None)
                st.session_state["_mfa_ok"] = user_id
                st.rerun()
        render_footer()
        return
    secret = st.session_state.setdefault("_mfa_secret", pyotp.random_base32())
    uri = pyotp.TOTP(secret).provisioning_uri(
        name=user["email"] or user["name"], issuer_name=MFA_ISSUER)
    st.markdown(
        f'<div class="lp-signin"{dir_attr()}><h2>{html.escape(tr("mfa_title"))}</h2>'
        f'<p>{tr("mfa_steps")}</p></div>', unsafe_allow_html=True)
    with st.container(key="login_card"):
        st.image(qr_png(uri), width=220)
        st.caption(tr("mfa_cant"))
        st.code(secret, language=None)
        with st.form("mfa_setup", border=False):
            code = st.text_input(tr("mfa_code"), max_chars=7, placeholder="123456")
            go = st.form_submit_button(tr("mfa_verify"), type="primary", width="stretch")
        if go:
            step = totp_step(secret, code)
            if step is None:
                st.error(tr("mfa_bad"))
            else:
                codes = new_backup_codes()
                get_db().save_mfa(user_id, secret, [_code_hash(c) for c in codes])
                get_db().mfa_success(user_id, step)
                st.session_state.pop("_mfa_secret", None)
                st.session_state["_mfa_codes"] = codes
                st.rerun()
        st.button(tr("fs_cancel"), key="mfa_cancel", on_click=sign_out,
                  type="tertiary", width="stretch", icon=":material/logout:")
    render_footer()


def render_mfa_challenge(user_id: int, state: dict) -> None:
    """Every sign-in: ask for the code from the authenticator app."""
    _mfa_page_start()
    st.markdown(
        f'<div class="lp-signin"{dir_attr()}><h2>{html.escape(tr("ch_title"))}</h2>'
        f'<p>{html.escape(tr("ch_p"))}</p></div>', unsafe_allow_html=True)
    locked = _as_utc(state["locked_until"])
    waiting = locked is not None and locked > datetime.now(timezone.utc)
    with st.container(key="login_card"):
        if waiting:
            st.error(tr("ch_locked", t=f"{locked:%H:%M}"))
        with st.form("mfa_challenge", border=False):
            code = st.text_input(tr("ch_code"), max_chars=24, placeholder=tr("ch_ph"))
            go = st.form_submit_button(tr("ch_verify"), type="primary", width="stretch",
                                       disabled=waiting)
        if go and not waiting:
            db = get_db()
            step = totp_step(state["secret"], code)
            if step is not None and step > state["last_step"]:
                db.mfa_success(user_id, step)
                st.session_state["_mfa_ok"] = user_id
                st.rerun()
            elif db.mfa_use_backup(user_id, _code_hash(code or "")):
                db.mfa_success(user_id)
                st.session_state["_mfa_ok"] = user_id
                st.rerun()
            else:
                db.mfa_failure(user_id)
                st.error(tr("ch_bad"))
        st.button(tr("sign_out"), key="mfa_signout", on_click=sign_out,
                  type="tertiary", width="stretch", icon=":material/logout:")
    render_footer()


def mfa_gate(user_id: int | None, user: dict | None) -> bool:
    """True when a page was shown and the app must not be rendered yet."""
    if not (user_id and user and mfa_required()) or mfa_passed(user_id):
        return False
    try:
        state = get_db().get_mfa(user_id)
    except Exception:  # noqa: BLE001 - fail closed: no code check, no entry
        LOGGER.exception("Could not read the two-step state")
        _mfa_page_start()
        st.error(tt("Verification is temporarily unavailable. Please try again in a minute.",
                    "التحقق غير متاح مؤقتاً. حاول مرة أخرى بعد دقيقة."))
        st.button(tt("Sign out", "تسجيل الخروج"), key="mfa_unavailable_out",
                  on_click=sign_out, icon=":material/logout:")
        return True
    if state is None or st.session_state.get("_mfa_codes"):
        render_mfa_setup(user_id, user)      # also shows the backup codes once
    else:
        render_mfa_challenge(user_id, state)
    return True


def render_auth_page(mode: str) -> None:
    """Sign-in and sign-up page. On a wide screen the cat banner sits on the
    left and the form on the right; on a phone they stack. Google shows its own
    password / 2-step screen and Nashmi never sees a password. A new Google
    account must finish the sign-up step before it can use the app."""
    render_topbar(mode)
    banner, form = st.columns([1, 1], gap="large", vertical_alignment="center")
    with banner:
        render_scene(mode)
    with form:
        steps = "".join(f"<li>{html.escape(item)}</li>" for item in tr(mode + "_steps"))
        st.markdown(
            f'<div class="lp-signin"{dir_attr()}><h2>{html.escape(tr(mode + "_title"))}</h2>'
            f'<p>{html.escape(tr(mode + "_sub"))}</p><ul>{steps}</ul></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="login_card"):
            render_login_buttons(mode)
            st.caption(tr("google_note"))
    render_footer()


def _as_utc(value):
    """Databases such as SQLite hand back naive datetimes; they are UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _ago(value) -> str:
    value = _as_utc(value)
    if value is None:
        return "-"
    seconds = max(0, int((datetime.now(timezone.utc) - value).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def _stamp(value) -> str:
    value = _as_utc(value)
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else "-"


@st.fragment(run_every=10)
def render_admin_live() -> None:
    """The live part of the admin screen; refreshes itself every 10 s."""
    import pandas as pd

    db = get_db()
    if not (db and is_admin()):  # re-checked on every refresh
        st.error("Admins only.")
        return
    data = db.admin_overview(online_minutes=ONLINE_WINDOW_MINUTES)
    first, second, third, fourth = st.columns(4)
    first.metric("Online now", len(data["online"]))
    second.metric("Sign-ins today", data["logins_today"])
    third.metric("Total users", data["total_users"])
    fourth.metric("New this week", data["new_this_week"])
    st.caption(
        f"Online = active in the last {ONLINE_WINDOW_MINUTES} minutes. "
        f"Updated {datetime.now(timezone.utc):%H:%M:%S} UTC, refreshes "
        "every 10 seconds."
    )
    st.subheader("Online now")
    if data["online"]:
        st.dataframe(pd.DataFrame([
            {"Name": r["name"], "E-mail": r["email"],
             "Last active": _ago(r["last_seen"])}
            for r in data["online"]
        ]), hide_index=True, width="stretch")
    else:
        st.info("Nobody is online right now.")
    st.subheader("Recent sign-ins")
    if data["recent_logins"]:
        st.dataframe(pd.DataFrame([
            {"Name": r["name"], "E-mail": r["email"],
             "When": _stamp(r["at"]), "": _ago(r["at"])}
            for r in data["recent_logins"]
        ]), hide_index=True, width="stretch")
    else:
        st.info("No sign-ins yet.")
def render_admin_stats(db) -> None:
    """Trends over the last 30 days. Counts only, never chat text."""
    import pandas as pd

    stats = db.admin_stats(days=30)
    first, second, third, fourth = st.columns(4)
    first.metric("Active users (24 h)", stats["active_1d"])
    second.metric("Active users (7 d)", stats["active_7d"])
    third.metric("Active users (30 d)", stats["active_30d"])
    fourth.metric("Total chats saved", stats["chats_total"])
    days = pd.to_datetime(stats["days"])

    def chart(title: str, series: list[int], kind: str = "bar") -> None:
        st.subheader(title)
        frame = pd.DataFrame({title: series}, index=days)
        (st.bar_chart if kind == "bar" else st.line_chart)(frame)

    chart("New sign-ups per day", stats["signups_by_day"])
    chart("Sign-ins per day", stats["logins_by_day"])
    chart("Distinct users signing in per day", stats["active_users_by_day"],
          "line")
    chart("AI requests per day", stats["requests_by_day"])
    left, right = st.columns(2)
    with left:
        st.subheader("Requests by model (30 d)")
        if stats["requests_by_model"]:
            st.dataframe(pd.DataFrame(
                stats["requests_by_model"], columns=["Model", "Requests"]),
                hide_index=True, width="stretch")
        else:
            st.info("No requests yet.")
        st.subheader("Chats by track")
        if stats["chats_by_track"]:
            st.dataframe(pd.DataFrame(
                stats["chats_by_track"], columns=["Track", "Chats"]),
                hide_index=True, width="stretch")
        else:
            st.info("No saved chats yet.")
    with right:
        st.subheader("Most active users (30 d)")
        if stats["top_users"]:
            st.dataframe(pd.DataFrame([
                {"Name": r["name"], "E-mail": r["email"],
                 "Requests": r["requests"]} for r in stats["top_users"]
            ]), hide_index=True, width="stretch")
        else:
            st.info("No requests yet.")
        st.metric("Users with a saved API key",
                  stats["users_with_saved_keys"])
    st.caption("Chat contents are encrypted and are never shown here.")


def render_admin_users(db) -> None:
    import pandas as pd

    data = db.admin_overview(online_minutes=ONLINE_WINDOW_MINUTES)
    frame = pd.DataFrame([
        {"Name": r["name"], "Username": r["username"] or "(not finished)",
         "E-mail": r["email"], "Joined": _stamp(r["created_at"]),
         "Last sign-in": _stamp(r["last_login"]), "Sign-ins": r["logins"]}
        for r in data["users"]
    ])
    st.caption(f"{data['total_users']} users, newest activity first.")
    if frame.empty:
        st.info("No users yet.")
        return
    query = st.text_input("Search by name or e-mail", key="admin_search")
    if query.strip():
        needle = query.strip().lower()
        frame = frame[
            frame["Name"].str.lower().str.contains(needle, regex=False)
            | frame["E-mail"].str.lower().str.contains(needle, regex=False)
        ]
    st.dataframe(frame, hide_index=True, width="stretch")
    bad_start = ('=', '+', '-', '@', chr(9), chr(13))
    safe = frame.map(lambda v: ("'" + v) if isinstance(v, str) and v[:1] in bad_start else v)
    st.download_button(
        "Download as CSV", safe.to_csv(index=False).encode("utf-8"),
        file_name="nashmi_users.csv", mime="text/csv",
        icon=":material/download:",
    )


def render_admin_system(db) -> None:
    """What is configured and whether the database answers."""
    from sqlalchemy import text

    started = time.perf_counter()
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency = f"{(time.perf_counter() - started) * 1000:.0f} ms"
        reachable = "Reachable"
    except Exception:  # noqa: BLE001
        reachable, latency = "Not reachable", "-"
    rows = [
        ("Database", "SQLite (local test)" if db.is_sqlite else "PostgreSQL"),
        ("Database status", reachable),
        ("Database latency", latency),
        ("Encryption at rest", "On" if db.cipher.enabled else "Off"),
        ("Sign-in providers", ", ".join(auth_providers()) or "none"),
        ("Public mode", "On" if is_public() else "Off"),
        ("Admins configured", str(len(admin_emails()))),
        ("Streamlit", st.__version__),
    ]
    st.dataframe(pd_rows(rows), hide_index=True, width="stretch")
    if not db.is_sqlite and not db.cipher.enabled:
        st.warning("Encryption is off. Set NASHMI_SECRET_KEY.")


def pd_rows(rows: list[tuple[str, str]]):
    import pandas as pd

    return pd.DataFrame(rows, columns=["Setting", "Value"])


def render_admin() -> None:
    """Admin screen: live activity, statistics, users and system health."""
    st.button("Back to chat", key="close_admin", on_click=close_account,
              icon=":material/arrow_back:")
    st.markdown('<div class="empty-title">Admin</div>'
                '<div class="empty-blurb">Sign-ins, live activity and '
                'statistics. Only you can see this page.</div>',
                unsafe_allow_html=True)
    db = get_db()
    if not (db and is_admin()):
        st.error("Admins only.")
        return
    live, stats_tab, users_tab, system_tab = st.tabs(
        ["Live", "Statistics", "Users", "System"]
    )
    with live:
        render_admin_live()
    with stats_tab:
        render_admin_stats(db)
    with users_tab:
        render_admin_users(db)
    with system_tab:
        render_admin_system(db)


def main() -> None:
    """Entry point executed on every Streamlit rerun."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",  # collapsed on phones, open on desktop
    )
    inject_theme()
    gate = current_gate()
    if gate == "landing":
        render_landing()
        return
    if gate in ("login", "signup"):
        render_auth_page(gate)
        return
    if st.session_state.get("_account_deleted") and current_user():
        st.markdown(
            f'<style>{LANDING_CSS}.block-container {{ max-width: 620px !important; }}</style>',
            unsafe_allow_html=True)
        st.success("Your account and all its data were deleted from the "
                   "database. Sign out to finish.")
        st.button("Sign out", key="deleted_signout", on_click=sign_out,
                  icon=":material/logout:")
        return
    if needs_signup(account_id()):
        render_complete_signup(account_id(), current_user())
        return
    if mfa_gate(account_id(), current_user()):
        return
    st.markdown(
        f"<style>{APP_POLISH_CSS}"
        + (SIDEBAR_RTL_CSS if lang() == "ar" else "") + "</style>",
        unsafe_allow_html=True)
    nav_box = st.sidebar.container(key="side_nav")
    chats_box = st.sidebar.container(key="side_chats")
    settings_box = st.sidebar.container(key="side_settings")
    user_box = st.sidebar.container(key="side_user")

    account_id()  # settle who is signed in before anything is loaded
    touch_presence()
    mode = render_nav(nav_box)
    settings, usage_slot = render_settings(settings_box, mode)
    usage_slot.markdown(limit_widget_html(settings), unsafe_allow_html=True)
    render_user_card(user_box)
    render_side_language(user_box)
    if st.session_state.get("view") == "account":
        render_account(settings)
        return
    if st.session_state.get("view") == "admin" and is_admin():
        render_admin()
        return
    render_hero()
    render_topline(settings)
    if mode == MODE_SOC:
        render_soc_track(settings)
    elif mode == MODE_TOOLS:
        render_toolbox()
    elif mode == MODE_EXAM:
        render_exam_track(settings)
    else:
        render_chat_track(settings, chats_box)
    usage_slot.markdown(  # refresh the usage line after the reply
        limit_widget_html(settings), unsafe_allow_html=True
    )
    embed_html(COMPOSER_FOCUS_TEMPLATE, 1)


# ---------------------------------------------------------------------------
# The cat picture (embedded so you only need this one file).
# Do not edit the block below: it is the image data.
# ---------------------------------------------------------------------------
_CAT_B64_WRAPPED = """
"""
CAT_B64 = "".join(_CAT_B64_WRAPPED.split())

if __name__ == "__main__":
    main()