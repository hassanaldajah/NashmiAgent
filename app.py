"""Nashmi Agent: a multi-provider cybersecurity assistant built with Streamlit.

Features
    * Providers: Gemini, GLM (Z.ai) and Groq (all free tiers).
    * Streaming replies for every provider.
    * File attachments in the chat box (text, code, images, PDF, video).
    * API keys read from a ``.env`` file or Streamlit secrets, never shown
      in the browser and never written to the chat history.
    * Mobile-first layout for iOS and Android browsers.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import ipaddress
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
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

load_dotenv()

logging.basicConfig(
    level=os.getenv("NASHMI_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("nashmi")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "Nashmi Agent"
HISTORY_FILE = Path(
    os.getenv("NASHMI_HISTORY_FILE")
    or Path(__file__).resolve().with_name("chat_history.json")
)
PERSIST_HISTORY = os.getenv("NASHMI_PERSIST_HISTORY", "1") != "0"
USAGE_FILE = Path(
    os.getenv("NASHMI_USAGE_FILE")
    or Path(__file__).resolve().with_name("usage_counter.json")
)
# Daily request limits we know for sure (the app also learns new ones from
# the provider's own "quota exceeded" message).
KNOWN_DAILY_LIMITS = {"gemini:gemini-3.8-flash": 20}

TEMPERATURE = 0.3
MAX_HISTORY_MESSAGES = 6
MAX_TEXT_CHARS = 15_000
MAX_MEDIA_BYTES = 18 * 1024 * 1024
REQUEST_TIMEOUT = (10, 180)  # (connect, read) seconds
MAX_RETRIES = 2
RETRY_BACKOFF = 1.0
RETRY_STATUS = {429, 500, 502, 503, 504}
NEW_CHAT_TITLE = "New Chat"
HERO_HEIGHT = 300

MODE_CHAT = "General Chat"
MODE_SECURITY = "Security Intelligence Labs"
MODE_CPSS = "CPSS Projects (Cyber-Physical Systems)"
MODE_CODE = "General Code Lab and SQL"
MODE_SOC = "SOC Log Analyzer (File Upload)"
MODES = (MODE_CHAT, MODE_SECURITY, MODE_CPSS, MODE_CODE, MODE_SOC)

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
        label="GLM (Z.ai) - Free Flash models",
        id="glm",
        short="GLM",
        kind="openai",
        models=("glm-4.7-flash", "glm-4.5-flash"),
        key_url="https://z.ai/manage-apikey/apikey-list",
        note="Only the Flash models are free. Other GLM models are billed.",
        env_keys=("GLM_API_KEY", "ZAI_API_KEY"),
        base_url="https://api.z.ai/api/paas/v4",
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
)
PROVIDERS = {p.label: p for p in PROVIDER_LIST}
OPENROUTER_FALLBACK = ("openai/gpt-oss-20b:free", "z-ai/glm-5.2:free")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
WELCOME = {
    MODE_CHAT: (
        "Hi there! I'm **Nashmi**. Ask me anything, in Arabic or English. "
        "Switch the Track in the sidebar when you want the cybersecurity "
        "labs or the SOC file analyzer."
    ),
    MODE_SECURITY: (
        "Hi there! You're in **Security Intelligence Labs**. Ask me about "
        "log analysis with Pandas, OSINT and threat intel scripts, or "
        "anomaly detection, or tap a quick action above."
    ),
    MODE_CPSS: (
        "Hi there! You're in **CPSS Projects (Cyber-Physical Systems)**. "
        "I can help you pick a project idea, simulate ICS and IoT attacks "
        "like Modbus or sensor spoofing, or design an intrusion detection "
        "system."
    ),
    MODE_CODE: (
        "Hi there! You're in **General Code Lab and SQL**. I can walk you "
        "through vulnerable versus secure code line by line, from SQL "
        "injection to port scanning and brute-force detection."
    ),
}

QUICK_ACTIONS = {
    MODE_CHAT: (),
    MODE_SECURITY: (
        (
            "Pandas Security Data Lab",
            "Explain and write Python code that uses Pandas to read a "
            "security dataset and detect anomalous behavior (Anomaly "
            "Detection) as part of a Security Intelligence lab.",
        ),
        (
            "Threat Intel (OSINT) Script",
            "Write Python code that pulls threat intelligence for "
            "suspicious IP addresses and domains and classifies them in a "
            "table.",
        ),
        (
            "AI Attack Detection Model",
            "Write simple Python code using scikit-learn to build a "
            "classifier that distinguishes normal traffic from attack "
            "traffic for a Security Intelligence lab, with an explanation "
            "of every step.",
        ),
    ),
    MODE_CPSS: (
        (
            "CPSS Project Ideas",
            "Suggest 3 strong, practical project ideas for a Cyber-Physical "
            "Security Systems (CPSS) course that can be implemented and "
            "simulated with Python, and explain each project idea.",
        ),
        (
            "Modbus / IoT Attack Simulation",
            "Write Python code that simulates an industrial system or IoT "
            "sensors (such as MQTT or Modbus) along with a script that "
            "detects sensor spoofing.",
        ),
        (
            "CPS Intrusion Detection (IDS)",
            "How do I build a simple Python monitoring system that watches "
            "physical sensor readings (temperature/pressure/power) and "
            "detects Replay Attacks or False Data Injection?",
        ),
    ),
    MODE_CODE: (
        (
            "SQL Injection Defense Code",
            "Explain how a SQL Injection vulnerability happens in code, and "
            "write Python and SQL code comparing a vulnerable query with a "
            "secure one.",
        ),
        (
            "Network Port Scanner Script",
            "Write an educational Python script that scans open ports on my "
            "local machine to verify service security, with an explanation "
            "of every line.",
        ),
        (
            "Brute Force Detection Script",
            "Write a Python script that reads log files and detects "
            "repeated failed login attempts.",
        ),
    ),
}

SYSTEM_PROMPT_TEMPLATE = """\
Your name is "Nashmi". You are a Jordanian cybersecurity expert and a \
specialized academic and technical assistant.
You speak in a professional, direct technical style with a friendly tone.
The track currently selected by the student is: {mode}

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

Rules by track:
1. Security Intelligence: focus on writing and explaining Python code for \
threat analysis (Threat Intel APIs, OSINT, Pandas log analysis, machine \
learning anomaly detection, YARA/Sigma rules).
2. CPSS (Cyber-Physical Security Systems): focus on industrial and IoT \
security (SCADA, ICS, PLC, Modbus, MQTT, sensor spoofing, intrusion \
detection in CPS) and suggest practical project ideas with Python \
simulation code.
3. General Code Lab and SQL: provide complete code explained line by line, \
comparing vulnerable code with secure code."""

CHAT_SYSTEM_PROMPT = """\
Your name is "Nashmi". You are a friendly, smart assistant who can talk \
about any topic, and you are especially strong in cybersecurity and \
programming.
- Reply in the language the user writes in. For Arabic, use clear Arabic \
with a friendly Jordanian touch, and keep technical terms, commands and \
code in English.
- Match the length of the answer to the question. For greetings or short \
messages, reply briefly. Give code or long explanations only when asked.
- Be honest when you do not know something."""

EXPERT_TEMPLATES = {
    MODE_SECURITY: (
        "Context: Security Intelligence Labs (threat intel, OSINT, Pandas "
        "log analysis, anomaly detection, Sigma/YARA). "
        "The student wrote: '{user_text}'. {style}"
    ),
    MODE_CPSS: (
        "Context: Cyber-Physical Systems (CPSS) Lab (SCADA, Modbus, MQTT, "
        "PLC, IoT security). The student wrote: '{user_text}'. {style}"
    ),
    MODE_CODE: (
        "Context: Secure Coding & SQL Lab (vulnerable vs secure code). "
        "The student wrote: '{user_text}'. {style}"
    ),
}
EXPERT_FALLBACK = (
    "Context: {mode}. The student wrote: '{user_text}'. {style}"
)
RESPONSE_STYLE = (
    "Answer exactly what was asked, at a length that matches the message. "
    "For greetings, short, unclear or off-topic messages, reply in one or "
    "two short sentences and ask what they need. Only give code, long "
    "explanations or full solutions when the student clearly asks for them."
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
    return SYSTEM_PROMPT_TEMPLATE.format(mode=mode)


def wrap_prompt_as_security_expert(user_text: str, mode: str) -> str:
    """Turn a plain question into an expert prompt for the chosen track."""
    if mode == MODE_CHAT:
        return user_text  # normal chat: send the message as written
    template = EXPERT_TEMPLATES.get(mode)
    if template is None:
        return EXPERT_FALLBACK.format(
            mode=mode, user_text=user_text, style=RESPONSE_STYLE
        )
    return template.format(user_text=user_text, style=RESPONSE_STYLE)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def get_secret(*names: str) -> str:
    """Look up a secret in the environment (.env) and then Streamlit secrets.

    The value is only used server-side; it is never rendered in the page.
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


def redact_secrets(text: str) -> str:
    """Mask values of KEY/TOKEN/SECRET/PASSWORD lines in env-style files."""
    return SECRET_LINE_RE.sub(r"\1[REDACTED]", text)


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
            "max_tokens": 4096,
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

    # GLM Flash models "think" first, which looks like an empty reply for a
    # long time. Turn the thinking phase off so the answer starts at once.
    payload_extra = (
        {"thinking": {"type": "disabled"}} if provider.id == "glm" else {}
    )
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
            "temperature": TEMPERATURE,
            "stream": True,
            **payload_extra,
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
    """Ask Gemini or Groq which chat models this key can really use."""
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
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_usage(data: dict) -> None:
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
        return str(value)


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
            f"<b style='color:{color}'>{left} left</b> of {limit} today "
            f"({used} used)"
        )
        bar = (
            "<div class='lw-bar'><span class='lw-mark' "
            f"style='left:{pct:.1f}%'></span></div>"
        )
    else:
        text = f"{used} used today &middot; limit unknown"
        bar = "<div class='lw-bar unknown'></div>"
    info = (st.session_state.get("rate_info") or {}).get(
        f"{provider.id}:{model}"
    ) or {}
    extra = ""
    if info.get("remaining_requests"):
        extra += (
            "<div class='lw-extra'>requests left: "
            f"{_fmt_num(info['remaining_requests'])} / "
            f"{_fmt_num(info.get('limit_requests', '?'))}</div>"
        )
    if info.get("remaining_tokens"):
        extra += (
            "<div class='lw-extra'>tokens left this minute: "
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
def new_chat_record(mode: str) -> dict:
    """Create an empty chat that starts with the track's welcome message."""
    return {
        "title": NEW_CHAT_TITLE,
        "messages": [{"role": "assistant", "content": WELCOME[mode]}],
    }


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
            if msgs:
                valid.append({
                    "title": str(chat.get("title") or NEW_CHAT_TITLE),
                    "messages": msgs,
                })
        if valid:
            clean[mode] = valid
    return clean


def load_history(path: Path = HISTORY_FILE) -> dict[str, list[dict]]:
    """Load saved chats, backing up a corrupt file instead of losing it."""
    if not PERSIST_HISTORY or not path.exists():
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


def save_history(history: dict, path: Path = HISTORY_FILE) -> bool:
    """Write chats atomically so a crash never leaves a half-written file."""
    if not PERSIST_HISTORY:
        return True
    tmp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            json.dump(history, tmp, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            LOGGER.debug("Could not restrict history file permissions")
        return True
    except OSError as exc:
        LOGGER.warning("Could not save chat history: %s", exc)
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        return False


def persist(history: dict) -> None:
    """Save history and tell the user (once per call) if that fails."""
    if not save_history(history):
        st.toast("Could not save the chat history to disk.")


def radio_key(mode: str) -> str:
    return f"chat_radio_{mode}"


def active_index(mode: str, count: int) -> int:
    """Index of the selected chat, clamped to the available chats."""
    index = st.session_state.get(radio_key(mode), 0)
    return index if isinstance(index, int) and 0 <= index < count else 0


def make_title(text: str, limit: int = 32) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def on_new_chat(mode: str) -> None:
    history = st.session_state.saved_history
    history[mode].insert(0, new_chat_record(mode))
    st.session_state[radio_key(mode)] = 0
    persist(history)


def on_delete_chat(mode: str) -> None:
    history = st.session_state.saved_history
    chats = history[mode]
    chats.pop(active_index(mode, len(chats)))
    if not chats:
        chats.append(new_chat_record(mode))
    st.session_state[radio_key(mode)] = 0
    persist(history)


def ensure_history(mode: str) -> list[dict]:
    """Make sure the session has at least one chat for this track."""
    if "saved_history" not in st.session_state:
        st.session_state.saved_history = load_history()
    history = st.session_state.saved_history
    if not history.get(mode):
        history[mode] = [new_chat_record(mode)]
        persist(history)
    return history[mode]


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
    "family=Inter:wght@400;500;600;800&"
    "family=JetBrains+Mono:wght@400;600;800&display=swap');\n"
)

THEME_CSS = """

:root {
    --bg: #0b111c;
    --panel: #131c2b;
    --panel-2: #0f1724;
    --accent: #E5383B;
    --accent-dim: rgba(229, 56, 59, 0.35);
    --grid-line: rgba(139, 20, 30, 0.20);  /* dark wine-red grid */
    --text: #A4B1CD;
    --white: #FFFFFF;
    --warn: #FFB020;
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
    color: var(--accent) !important;
    min-width: 44px;
    min-height: 44px;
}

/* ---- Base ---- */
html, body { -webkit-text-size-adjust: 100%; }
* { -webkit-tap-highlight-color: transparent; }
button, [role="button"], label { touch-action: manipulation; }

.stApp {
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
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
    max-width: 1080px;
    padding: 1rem 1rem 4rem;
    padding-left: max(1rem, var(--safe-left));
    padding-right: max(1rem, var(--safe-right));
}
h1, h2, h3, h4 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--white) !important;
    overflow-wrap: anywhere;
}
img, video, canvas, iframe { max-width: 100%; }
a { color: var(--accent) !important; }
hr { border-color: var(--accent-dim); }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1724 0%, #0b111c 100%);
    border-right: 1px solid var(--accent-dim);
}
[data-testid="stSidebar"] h2 {
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent) !important;
}
label, .stSelectbox label, .stTextInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text) !important;
}

/* ---- Inputs ---- */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--panel) !important;
    color: var(--white) !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace;
    min-height: 44px;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent);
}

/* ---- Buttons ---- */
.stButton, .stDownloadButton { width: 100%; }
.stButton > button, .stDownloadButton > button {
    width: 100%;
    min-height: 44px;
    height: auto;
    padding: 0.5rem 0.9rem;
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 12px;
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    white-space: normal;
    overflow-wrap: anywhere;
    transition: background 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:active, .stDownloadButton > button:active {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
}
@media (hover: hover) and (pointer: fine) {
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--accent);
        color: #ffffff;
        border-color: var(--accent);
        box-shadow: 0 0 22px rgba(229, 56, 59, 0.55);
        transform: translateY(-2px);
    }
}

/* ---- Track header and chips ---- */
.track-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 12px;
    margin: 2px 0 12px;
    padding: 10px 16px;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(229, 56, 59, 0.10),
        rgba(19, 28, 43, 0.2));
    border: 1px solid var(--accent-dim);
    border-left: 4px solid var(--accent);
}
.track-head .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    color: #ffffff;
    background: var(--accent);
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 800;
}
.track-head .name {
    flex: 1 1 12rem;
    min-width: 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--white);
    overflow-wrap: anywhere;
}
.limit-widget {
    margin-left: auto;
    min-width: 150px;
    max-width: 260px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: var(--text);
    line-height: 1.5;
}
.limit-widget .lw-bar {
    position: relative;
    height: 5px;
    margin-top: 5px;
    border-radius: 3px;
    background: linear-gradient(90deg, #4cc38a, #f0c24b, #E5383B);
}
.limit-widget .lw-bar.unknown { opacity: 0.35; }
.limit-widget .lw-mark {
    position: absolute;
    top: -3px;
    width: 3px;
    height: 11px;
    border-radius: 2px;
    background: #ffffff;
    transform: translateX(-50%);
    box-shadow: 0 0 6px rgba(255, 255, 255, 0.8);
}
.limit-widget .lw-extra { opacity: 0.8; margin-top: 2px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.chip {
    max-width: 100%;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.5px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--accent-dim);
    background: var(--panel-2);
    color: var(--text);
    overflow-wrap: anywhere;
}
.chip b { color: var(--accent); font-weight: 600; }
.chip.bad b { color: var(--warn); }
.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 6px #ffffff;
    margin-right: 7px;
}
.chip.bad .dot { background: var(--warn); box-shadow: 0 0 6px var(--warn); }

/* ---- Chat ---- */
[data-testid="stChatMessage"] {
    background: var(--panel);
    border: 1px solid rgba(164, 177, 205, 0.12);
    border-left: 3px solid var(--accent);
    border-radius: 14px;
    padding: 10px 14px;
}
[data-testid="stChatMessage"] > div { min-width: 0; }
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
    border: 1px solid var(--accent-dim);
    border-radius: 8px;
}
[data-testid="stChatMessage"] table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
[data-testid="stChatMessage"] img { border-radius: 6px; height: auto; }

[data-testid="stBottom"] > div { background: var(--bg) !important; }
[data-testid="stChatInput"] {
    background: var(--panel);
    border: 1px solid var(--accent-dim);
    border-radius: 12px;
    max-width: 100%;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 18px rgba(229, 56, 59, 0.25);
}
[data-testid="stChatInput"] textarea {
    color: var(--white) !important;
    caret-color: var(--accent);
}
/* Empty chat input: hide any placeholder text */
[data-testid="stChatInput"] textarea::placeholder {
    color: transparent !important;
    opacity: 0 !important;
}

/* ---- Uploader ---- */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-2);
    border: 1px dashed var(--accent);
    border-radius: 12px;
    flex-wrap: wrap;
    gap: 0.5rem;
}
[data-testid="stFileUploader"] { max-width: 100%; }
[data-testid="stFileUploaderFile"] { min-width: 0; overflow-wrap: anywhere; }

/* ---- Scrollbar (desktop) ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
    background: var(--accent-dim);
    border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ---- Columns wrap instead of overflowing ---- */
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
[data-testid="stColumn"], [data-testid="column"] {
    flex: 1 1 14rem;
    min-width: min(100%, 14rem);
}

/* ---- Tablets and phones ---- */
@media (max-width: 768px) {
    .block-container {
        padding: calc(3.25rem + var(--safe-top)) 0.75rem 5rem;
        padding-left: max(0.75rem, var(--safe-left));
        padding-right: max(0.75rem, var(--safe-right));
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
    label, .stSelectbox label, .stTextInput label {
        letter-spacing: 0.8px;
    }
    [data-testid="stChatMessage"] {
        padding: 8px 10px;
        gap: 0.5rem;
    }
    [data-testid^="stChatMessageAvatar"] { width: 1.9rem; height: 1.9rem; }
    .track-head { padding: 8px 12px; }
    .track-head .name { font-size: 0.98rem; }
}
@media (max-width: 400px) {
    .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
    .chip { font-size: 0.64rem; padding: 3px 8px; }
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
# UI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Values chosen in the sidebar."""

    provider: Provider
    api_key: str
    model: str
    mode: str


def render_sidebar() -> Settings:
    """Draw provider, key, model and track controls."""
    with st.sidebar:
        st.header("Control Panel")
        label = st.selectbox("AI provider", list(PROVIDERS))
        provider = PROVIDERS[label]

        typed_key = st.text_input(
            f"{provider.short} API key",
            type="password",
            key=f"key_{provider.id}",
        )
        env_key = get_secret(*provider.env_keys)
        api_key = typed_key.strip() or env_key
        if env_key and not typed_key.strip():
            st.caption("Key loaded from .env. Type here to override it.")

        if provider.id == "openrouter":
            options = openrouter_free_models() or list(OPENROUTER_FALLBACK)
        elif provider.id in ("gemini", "groq"):
            options = live_models(provider.id, api_key) or list(
                provider.models
            )
        else:
            options = list(provider.models)
        chosen = st.selectbox("Model", options, key=f"model_{provider.id}")
        custom = st.text_input(
            "Custom model name (optional)", key=f"custom_{provider.id}"
        )
        st.caption(provider.note)
        st.markdown(f"[Get your {provider.short} API key]({provider.key_url})")

        st.divider()
        mode = st.selectbox("Track", list(MODES))

    return Settings(provider, api_key, custom.strip() or chosen, mode)


def render_status_header(settings: Settings) -> None:
    """Show the track title and the key/model/provider chips."""
    ok = bool(settings.api_key)
    mode = html.escape(settings.mode)
    model = html.escape(settings.model)
    provider = html.escape(settings.provider.short)
    st.markdown(
        f"""
<div class="track-head">
    <span class="tag">TRACK</span><span class="name">{mode}</span>\
{limit_widget_html(settings)}
</div>
<div class="chips">
    <span class="chip {'ok' if ok else 'bad'}"><span class="dot"></span>\
API KEY <b>{'LOADED' if ok else 'MISSING'}</b></span>
    <span class="chip">MODEL <b>{model}</b></span>
    <span class="chip">PROVIDER <b>{provider}</b></span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_chat_sidebar(mode: str, chats: list[dict]) -> int:
    """Draw the saved-chat list in the sidebar; return the active index."""
    with st.sidebar:
        st.divider()
        st.subheader("Track Chats")
        st.button(
            "New Chat", key=f"new_chat_{mode}", use_container_width=True,
            on_click=on_new_chat, args=(mode,),
        )
        labels = [f"{i + 1}. {c['title']}" for i, c in enumerate(chats)]
        index = st.radio(
            "Saved conversations",
            options=list(range(len(chats))),
            format_func=lambda i: labels[i],
            key=radio_key(mode),
        )
        questions = [
            m["content"] for m in chats[index]["messages"]
            if m["role"] == "user"
        ]
        if questions:
            with st.expander("Questions in this chat", expanded=False):
                for number, question in enumerate(questions, 1):
                    st.caption(f"{number}. {make_title(question, 45)}")
        st.button(
            "Delete Current Chat", key=f"delete_chat_{mode}",
            use_container_width=True, on_click=on_delete_chat, args=(mode,),
        )
    return index


def render_quick_actions(mode: str) -> str | None:
    """Show the three quick-action buttons; return the chosen prompt."""
    selected = None
    if not QUICK_ACTIONS.get(mode):
        return None
    columns = st.columns(3)
    for number, (column, (label, prompt)) in enumerate(
        zip(columns, QUICK_ACTIONS[mode])
    ):
        if column.button(label, key=f"quick_{mode}_{number}"):
            selected = prompt
    return selected


def render_messages(messages: list[dict]) -> None:
    for msg in messages:
        avatar = (
            get_assistant_avatar()
            if msg["role"] == "assistant"
            else get_user_avatar()
        )
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("files"):
                st.caption("Attached: " + ", ".join(msg["files"]))


def read_chat_input() -> tuple[str, list]:
    """Read the chat box (text plus attachments) in a version-safe way."""
    try:
        value = st.chat_input("", accept_file="multiple", file_type=None)
    except TypeError:  # Streamlit older than 1.43 has no attachments
        value = st.chat_input("")
    if not value:
        return "", []
    if isinstance(value, str):
        return value.strip(), []
    text = (getattr(value, "text", "") or "").strip()
    files = list(getattr(value, "files", []) or [])
    if files and not text:
        text = "Analyze the attached file(s)."
    return text, files


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
    settings: Settings, chat: dict, prompt: str, files: list
) -> None:
    """Send one user message and stream the answer into the chat."""
    if not settings.api_key:
        st.error(
            f"Please enter your {settings.provider.short} API key in the "
            "sidebar or put it in your .env file."
        )
        return

    prepared = prepare_attachments(files, settings.provider)
    warn_skipped(prepared)
    messages = chat["messages"]
    names = [f.name for f in files]
    messages.append({"role": "user", "content": prompt, "files": names})
    if chat.get("title") == NEW_CHAT_TITLE:
        chat["title"] = make_title(prompt)
    persist(st.session_state.saved_history)

    with st.chat_message("user", avatar=get_user_avatar()):
        st.markdown(prompt)
        if names:
            st.caption("Attached: " + ", ".join(names))

    with st.chat_message("assistant", avatar=get_assistant_avatar()):
        reply = ""
        failed = False
        try:
            outgoing = messages[:-1] + [{
                "role": "user",
                "content": wrap_prompt_as_security_expert(
                    prompt, settings.mode
                ),
            }]
            reply = stream_to_text(with_wait_notice(generate_stream(
                settings.provider, settings.api_key, settings.model,
                build_system_prompt(settings.mode), outgoing, prepared,
            )))
        except ProviderError as exc:
            failed = True
            note_quota_error(settings.provider, settings.model, str(exc))
            show_provider_error(exc)
        except Exception as exc:  # noqa: BLE001 - keep the app alive
            failed = True
            LOGGER.exception("Unexpected error while generating a reply")
            st.error(f"Request failed: {exc}")

        if reply.strip():
            messages.append({"role": "assistant", "content": reply})
        else:
            messages.pop()  # remove the unanswered question so it can retry
            if not failed:
                st.warning(
                    "No reply was received. Try again or pick another model."
                )
        persist(st.session_state.saved_history)


def render_chat_track(settings: Settings) -> None:
    mode = settings.mode
    chats = ensure_history(mode)
    index = render_chat_sidebar(mode, chats)
    chat = chats[index]

    quick_prompt = render_quick_actions(mode)
    render_messages(chat["messages"])
    typed_prompt, files = read_chat_input()

    prompt = quick_prompt or typed_prompt
    if prompt:
        attachments = [] if quick_prompt else files
        handle_submission(settings, chat, prompt, attachments)


def render_soc_track(settings: Settings) -> None:
    """Upload logs/media and stream a SOC report."""
    provider = settings.provider
    uploads = st.file_uploader(
        "Upload logs, images, screenshots, videos, PCAP/CSV/JSON, or any "
        "file for SOC analysis",
        type=None,
        accept_multiple_files=True,
        key="soc_multi_uploader",
    )
    notes = st.text_input(
        "Optional notes or specific question about the uploaded files:", ""
    )
    vt_typed = st.text_input(
        "VirusTotal API key (optional, free)", type="password",
        key="vt_key_input",
    )
    vt_key = vt_typed.strip() or get_secret("VT_API_KEY", "VIRUSTOTAL_API_KEY")
    allow_upload = st.checkbox(
        "Upload files VirusTotal does not know yet (they become visible to "
        "the VirusTotal community)",
        value=False,
    )
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
            f"Please enter your {provider.short} API key in the sidebar or "
            "put it in your .env file."
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
                build_system_prompt(settings.mode),
                [{"role": "user", "content": prompt}], prepared,
            ))
    except ProviderError as exc:
        note_quota_error(provider, settings.model, str(exc))
        show_provider_error(exc)
    except Exception as exc:  # noqa: BLE001 - keep the app alive
        LOGGER.exception("Unexpected error while writing the SOC report")
        st.error(f"Request failed: {exc}")

    if report.strip():
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


def main() -> None:
    """Entry point executed on every Streamlit rerun."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",  # collapsed on phones, open on desktop
    )
    inject_theme()
    render_hero()
    settings = render_sidebar()
    header_slot = st.empty()
    with header_slot.container():
        render_status_header(settings)
    if settings.mode == MODE_SOC:
        render_soc_track(settings)
    else:
        render_chat_track(settings)
    with header_slot.container():  # refresh the limit bar after the reply
        render_status_header(settings)


# ---------------------------------------------------------------------------
# The cat picture (embedded so you only need this one file).
# Do not edit the block below: it is the image data.
# ---------------------------------------------------------------------------
_CAT_B64_WRAPPED = """
"""
CAT_B64 = "".join(_CAT_B64_WRAPPED.split())

if __name__ == "__main__":
    main()