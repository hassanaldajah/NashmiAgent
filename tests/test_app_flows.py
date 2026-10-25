"""End-to-end flows through the real app with Streamlit's AppTest.

A signed-in user is faked (no Google, no database) and the model is replaced by a
stub, so nothing leaves the machine. Everything else is the real app."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
import app

st.logout = lambda: st.session_state.__setitem__("_logged_out", True)
USER = {{"name": "Hassan Aldajah", "email": "hassan@example.com", "picture": "", "sub": "t", "provider": "google"}}
signed_in = not st.session_state.get("_logged_out")
app.is_public = lambda: True
app.accounts_enabled = lambda: True
app.current_user = lambda: USER if signed_in else None
app.account_id = lambda: 1 if signed_in else None
class FakeDB:                       # signed-in users always have a database in production
    cipher = type("Cipher", (), {{"enabled": False}})()
    def get_user(self, user_id):
        return {{}}
    def __getattr__(self, name):
        return lambda *a, **k: []          # list-style answers (usage rows, chats)
app.get_db = lambda: FakeDB()
app.load_user_history = lambda: {{}}
app.load_user_store = lambda: app._sanitize_store({{}})
app.my_username = lambda: "hassan"
app.is_admin = lambda: False
app.needs_signup = lambda uid: False
app.otp_gate = lambda uid, user: False
app.mfa_gate = lambda uid, user: False
app.touch_presence = lambda: None

def fake_stream(*args, **kwargs):
    mode = st.session_state.get("_stream", "ok")
    if mode == "401":
        raise app.ProviderError(app.describe_http_error("Groq", 401, '{{"error": {{"message": "Invalid API Key"}}}}'))
    if mode == "net":
        raise app.ProviderError("Could not reach Groq. Check your connection.")
    yield "Hello from the stub model."
app.generate_stream = fake_stream
app.main()
"""


def boot(**state):
    at = AppTest.from_string(SCRIPT, default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at):
    return " ".join(str(x.value) for x in [*at.markdown, *at.error, *at.warning, *at.info, *at.caption])


def send(at, words="hi"):
    at.chat_input[0].set_value(words).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_signed_in_user_card_and_account():
    at = boot()
    assert "Hassan Aldajah" in texts(at) and "@hassan" in texts(at)
    at.button(key="open_account").click().run()
    assert not at.exception and at.session_state["view"] == "account"


def test_new_chat_and_provider_model_switch():
    at = boot()
    at.button(key="new_chat").click().run()
    assert not at.exception
    claude = next(label for label in at.selectbox(key="provider_choice").options if "Claude" in label)
    at.selectbox(key="provider_choice").set_value(claude).run()
    assert not at.exception
    assert at.selectbox(key="model_anthropic").value      # the model list follows the provider


def test_key_entry_then_send_message():
    at = boot()
    at.text_input(key="key_groq").set_value("gsk_test_not_real").run()
    send(at, "What is a SYN flood?")
    assert "Hello from the stub model." in texts(at)


def test_send_without_key_explains_what_to_do():
    at = send(boot(), "hello")
    assert "Hello from the stub model." not in texts(at)
    assert "key" in texts(at).lower()


@pytest.mark.parametrize("mode, words", [("401", "invalid"), ("net", "could not reach")])
def test_wrong_key_and_network_error_are_shown(mode, words):
    at = boot(_stream=mode)
    at.text_input(key="key_groq").set_value("gsk_wrong").run()
    send(at, "hello")
    assert words in texts(at).lower()


def test_language_switch():
    at = boot()
    at.radio(key="side_lang").set_value("ar").run()
    assert not at.exception and at.session_state["lang"] == "ar"


def test_sign_out():
    # a fresh session: AppTest cannot replay a radio whose labels changed language
    at = boot()
    at.button(key="side_signout").click().run()
    assert not at.exception and at.session_state["_logged_out"] is True
