"""Account storage for Nashmi Agent.

One small layer that keeps every user's chats, projects, preferences, usage
counts and (optionally) saved API keys in a SQL database. It works with
SQLite for local tests and with PostgreSQL (Neon, Supabase, ...) in
production, and it knows nothing about Streamlit so it can be tested alone.

Security rules this module enforces:
    * Every query is built with SQLAlchemy and bound parameters (no string
      SQL), and every read or write is scoped by ``user_id``.
    * Chat text, project instructions and API keys are encrypted at rest with
      Fernet (AES-128-CBC + HMAC). Several keys may be supplied (comma
      separated) so a key can be rotated without losing data.
    * Saving an API key is refused unless encryption is configured.
    * A non-SQLite database without an encryption key is refused at start-up.
    * Per-user limits stop one account from filling the database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, MetaData, String, Table, Text,
    and_, create_engine, delete, func, insert, select, update,
)
from sqlalchemy.exc import IntegrityError

try:
    from cryptography.fernet import Fernet, InvalidToken, MultiFernet
except ImportError:  # encryption stays off; the app then refuses real databases
    Fernet = InvalidToken = MultiFernet = None  # type: ignore[assignment]

LOGGER = logging.getLogger("nashmi.db")

MAX_CHATS_PER_USER = 1000
MAX_CHAT_CHARS = 1_500_000
MAX_PROJECTS_PER_USER = 50
MAX_TITLE_CHARS = 200
MAX_INSTRUCTION_CHARS = 2000

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sub", String(255), nullable=False, unique=True),
    Column("email", String(320), nullable=False, default=""),
    Column("name", String(200), nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_login", DateTime(timezone=True), nullable=False),
)
chats = Table(
    "chats", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           nullable=False, index=True),
    Column("track", String(80), nullable=False),
    Column("title", Text, nullable=False),          # encrypted
    Column("project_id", String(16), nullable=True),
    Column("data", Text, nullable=False),           # encrypted JSON messages
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, index=True),
)
projects = Table(
    "projects", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           primary_key=True),
    Column("id", String(16), primary_key=True),
    Column("name", String(60), nullable=False),
    Column("instructions", Text, nullable=False),   # encrypted
)
prefs = Table(
    "prefs", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           primary_key=True),
    Column("data", Text, nullable=False),
)
usage = Table(
    "usage", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           primary_key=True),
    Column("day", String(10), primary_key=True),
    Column("provider", String(40), primary_key=True),
    Column("model", String(120), primary_key=True),
    Column("requests", Integer, nullable=False, default=0),
)
api_keys = Table(
    "api_keys", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           primary_key=True),
    Column("provider", String(40), primary_key=True),
    Column("token", Text, nullable=False),          # encrypted
)

login_events = Table(
    "login_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           nullable=False, index=True),
    Column("at", DateTime(timezone=True), nullable=False, index=True),
)
presence = Table(
    "presence", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"),
           primary_key=True),
    Column("last_seen", DateTime(timezone=True), nullable=False, index=True),
)

ALL_USER_TABLES = (chats, projects, prefs, usage, api_keys,
                   login_events, presence)

LOGIN_EVENT_RETENTION_DAYS = 90


class ConfigError(RuntimeError):
    """The database or the encryption key is missing or wrong."""


class Cipher:
    """Encrypts text for storage. Values carry a short format tag."""

    def __init__(self, keys: Iterable[str]):
        cleaned = [k.strip() for k in keys if k and k.strip()]
        self._fernet = None
        if cleaned:
            if Fernet is None:
                raise ConfigError("Install the 'cryptography' package.")
            try:
                self._fernet = MultiFernet(
                    [Fernet(k.encode("ascii")) for k in cleaned]
                )
            except (ValueError, TypeError) as exc:
                raise ConfigError(
                    "NASHMI_SECRET_KEY is not a valid Fernet key."
                ) from exc

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def encrypt(self, text: str) -> str:
        if self._fernet is None:
            return "p1:" + text
        token = self._fernet.encrypt(text.encode("utf-8"))
        return "e1:" + token.decode("ascii")

    def decrypt(self, value: str) -> str:
        if value.startswith("p1:"):
            return value[3:]
        if value.startswith("e1:"):
            if self._fernet is None:
                raise ConfigError("Stored data is encrypted but no key is set.")
            try:
                return self._fernet.decrypt(value[3:].encode("ascii")).decode(
                    "utf-8"
                )
            except InvalidToken as exc:
                raise ConfigError("The encryption key does not match.") from exc
        return value


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    """All reads and writes for signed-in users."""

    def __init__(self, url: str, secret_keys: Iterable[str] = ()):
        if url.startswith("postgres://"):  # old style used by some hosts
            url = "postgresql://" + url[len("postgres://"):]
        self.is_sqlite = url.startswith("sqlite")
        self.cipher = Cipher(secret_keys)
        if not self.is_sqlite and not self.cipher.enabled:
            raise ConfigError(
                "A production database needs NASHMI_SECRET_KEY so chats and "
                "keys are encrypted at rest."
            )
        self.engine = create_engine(
            url, pool_pre_ping=True,
            connect_args={"check_same_thread": False} if self.is_sqlite else {},
        )
        metadata.create_all(self.engine)

    # ------------------------------------------------------------- users
    def upsert_user(self, sub: str, email: str, name: str) -> int:
        """Create the user on first sign-in; return the database id."""
        sub, email, name = sub[:255], email[:320], name[:200]
        now = _now()
        with self.engine.begin() as conn:
            row = conn.execute(
                select(users.c.id).where(users.c.sub == sub)
            ).first()
            if row:
                conn.execute(
                    update(users).where(users.c.id == row[0])
                    .values(email=email, name=name, last_login=now)
                )
                return int(row[0])
            result = conn.execute(insert(users).values(
                sub=sub, email=email, name=name, created_at=now,
                last_login=now,
            ))
            return int(result.inserted_primary_key[0])

    # ------------------------------------------- sign-in log and presence
    def record_login(self, user_id: int) -> None:
        """Log one sign-in and drop entries older than the retention window."""
        now = _now()
        with self.engine.begin() as conn:
            conn.execute(insert(login_events).values(user_id=user_id, at=now))
            conn.execute(delete(login_events).where(
                login_events.c.at < now - timedelta(
                    days=LOGIN_EVENT_RETENTION_DAYS)
            ))

    def touch_presence(self, user_id: int) -> None:
        """Mark the user as active right now (the live 'who is online')."""
        now = _now()
        with self.engine.begin() as conn:
            updated = conn.execute(
                update(presence).where(presence.c.user_id == user_id)
                .values(last_seen=now)
            ).rowcount
            if not updated:
                try:
                    conn.execute(insert(presence).values(
                        user_id=user_id, last_seen=now))
                except IntegrityError:  # another session inserted it first
                    pass

    def admin_overview(self, online_minutes: int = 5,
                       recent: int = 50, max_users: int = 500) -> dict:
        """Numbers and lists for the admin screen. Callers must check that
        the viewer is an admin: this method does not know who is asking."""
        now = _now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        online_since = now - timedelta(minutes=online_minutes)
        with self.engine.connect() as conn:
            total_users = conn.execute(
                select(func.count()).select_from(users)).scalar() or 0
            new_week = conn.execute(
                select(func.count()).select_from(users)
                .where(users.c.created_at >= now - timedelta(days=7))
            ).scalar() or 0
            logins_today = conn.execute(
                select(func.count()).select_from(login_events)
                .where(login_events.c.at >= day_start)
            ).scalar() or 0
            online = conn.execute(
                select(users.c.name, users.c.email, presence.c.last_seen)
                .join(presence, presence.c.user_id == users.c.id)
                .where(presence.c.last_seen >= online_since)
                .order_by(presence.c.last_seen.desc())
            ).mappings().all()
            recent_logins = conn.execute(
                select(users.c.name, users.c.email, login_events.c.at)
                .join(login_events, login_events.c.user_id == users.c.id)
                .order_by(login_events.c.at.desc()).limit(recent)
            ).mappings().all()
            counts = (
                select(login_events.c.user_id,
                       func.count().label("n"))
                .group_by(login_events.c.user_id).subquery()
            )
            everyone = conn.execute(
                select(users.c.name, users.c.email, users.c.created_at,
                       users.c.last_login, counts.c.n)
                .outerjoin(counts, counts.c.user_id == users.c.id)
                .order_by(users.c.last_login.desc()).limit(max_users)
            ).mappings().all()
        return {
            "total_users": int(total_users),
            "new_this_week": int(new_week),
            "logins_today": int(logins_today),
            "online": [dict(r) for r in online],
            "recent_logins": [dict(r) for r in recent_logins],
            "users": [{
                "name": r["name"], "email": r["email"],
                "created_at": r["created_at"], "last_login": r["last_login"],
                "logins": int(r["n"] or 0),
            } for r in everyone],
        }

    def admin_stats(self, days: int = 30) -> dict:
        """Aggregate statistics for the admin screen. Counts and totals only:
        chat text stays encrypted and is never read here. Callers must check
        that the viewer is an admin."""
        now = _now()
        since = now - timedelta(days=days)
        first_day = (now - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        with self.engine.connect() as conn:
            signups = conn.execute(
                select(users.c.created_at).where(users.c.created_at >= since)
            ).scalars().all()
            logins = conn.execute(
                select(login_events.c.user_id, login_events.c.at)
                .where(login_events.c.at >= since).limit(50_000)
            ).all()
            requests_rows = conn.execute(
                select(usage.c.day, usage.c.provider, usage.c.model,
                       func.sum(usage.c.requests))
                .where(usage.c.day >= first_day)
                .group_by(usage.c.day, usage.c.provider, usage.c.model)
            ).all()
            top_users = conn.execute(
                select(users.c.name, users.c.email,
                       func.sum(usage.c.requests).label("n"))
                .join(usage, usage.c.user_id == users.c.id)
                .where(usage.c.day >= first_day)
                .group_by(users.c.id, users.c.name, users.c.email)
                .order_by(func.sum(usage.c.requests).desc()).limit(10)
            ).mappings().all()
            chat_total = conn.execute(
                select(func.count()).select_from(chats)).scalar() or 0
            chat_tracks = conn.execute(
                select(chats.c.track, func.count())
                .group_by(chats.c.track).order_by(func.count().desc())
            ).all()
            key_users = conn.execute(
                select(func.count(func.distinct(api_keys.c.user_id)))
            ).scalar() or 0

        def day_of(value) -> str:
            value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).strftime("%Y-%m-%d")

        all_days = [(now - timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(days - 1, -1, -1)]
        signup_by_day = dict.fromkeys(all_days, 0)
        for created in signups:
            signup_by_day[day_of(created)] = (
                signup_by_day.get(day_of(created), 0) + 1)
        logins_by_day = dict.fromkeys(all_days, 0)
        active_by_day: dict[str, set] = {d: set() for d in all_days}
        for user_id, at in logins:
            key = day_of(at)
            logins_by_day[key] = logins_by_day.get(key, 0) + 1
            active_by_day.setdefault(key, set()).add(user_id)
        requests_by_day = dict.fromkeys(all_days, 0)
        by_model: dict[str, int] = {}
        for day, provider, model, total in requests_rows:
            requests_by_day[day] = requests_by_day.get(day, 0) + int(total or 0)
            label = f"{provider}: {model}"
            by_model[label] = by_model.get(label, 0) + int(total or 0)

        def distinct_since(window_days: int) -> int:
            cutoff = now - timedelta(days=window_days)
            return len({uid for uid, at in logins
                        if (at if at.tzinfo else
                            at.replace(tzinfo=timezone.utc)) >= cutoff})

        return {
            "days": all_days,
            "signups_by_day": [signup_by_day[d] for d in all_days],
            "logins_by_day": [logins_by_day[d] for d in all_days],
            "active_users_by_day": [len(active_by_day[d]) for d in all_days],
            "requests_by_day": [requests_by_day[d] for d in all_days],
            "requests_by_model": sorted(by_model.items(),
                                        key=lambda kv: kv[1], reverse=True),
            "top_users": [{"name": r["name"], "email": r["email"],
                           "requests": int(r["n"] or 0)} for r in top_users],
            "active_1d": distinct_since(1),
            "active_7d": distinct_since(7),
            "active_30d": distinct_since(30),
            "chats_total": int(chat_total),
            "chats_by_track": [(t, int(n)) for t, n in chat_tracks],
            "users_with_saved_keys": int(key_users),
        }

    def get_user(self, user_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(users).where(users.c.id == user_id)
            ).mappings().first()
        if not row:
            return None
        return {
            "email": row["email"], "name": row["name"],
            "created_at": row["created_at"], "last_login": row["last_login"],
        }

    # ------------------------------------------------------------- chats
    def load_history(self, user_id: int) -> dict[str, list[dict]]:
        """All chats of a user as {track: [chat, ...]}, newest first."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(chats).where(chats.c.user_id == user_id)
                .order_by(chats.c.updated_at.desc())
            ).mappings().all()
        history: dict[str, list[dict]] = {}
        for row in rows:
            try:
                messages = json.loads(self.cipher.decrypt(row["data"]))
                title = self.cipher.decrypt(row["title"])
            except (ConfigError, ValueError) as exc:
                LOGGER.warning("Skipping unreadable chat %s: %s", row["id"], exc)
                continue
            history.setdefault(row["track"], []).append({
                "id": row["id"], "title": title, "messages": messages,
                "project": row["project_id"] or "",
            })
        return history

    def save_chat(self, user_id: int, track: str, chat: dict) -> bool:
        """Insert or update one chat. False when a limit blocks the save."""
        payload = json.dumps(chat.get("messages", []), ensure_ascii=False)
        if len(payload) > MAX_CHAT_CHARS:
            return False
        values = dict(
            track=track[:80],
            title=self.cipher.encrypt(str(chat.get("title", ""))[:MAX_TITLE_CHARS]),
            project_id=(str(chat.get("project") or "")[:16] or None),
            data=self.cipher.encrypt(payload),
            updated_at=_now(),
        )
        chat_id = str(chat["id"])[:36]
        try:
            with self.engine.begin() as conn:
                owned = conn.execute(select(chats.c.id).where(and_(
                    chats.c.id == chat_id, chats.c.user_id == user_id,
                ))).first()
                if owned:
                    conn.execute(update(chats).where(and_(
                        chats.c.id == chat_id, chats.c.user_id == user_id,
                    )).values(**values))
                    return True
                total = conn.execute(
                    select(func.count()).select_from(chats)
                    .where(chats.c.user_id == user_id)
                ).scalar() or 0
                if total >= MAX_CHATS_PER_USER:
                    return False
                conn.execute(insert(chats).values(
                    id=chat_id, user_id=user_id, created_at=_now(), **values
                ))
                return True
        except IntegrityError:  # the id already belongs to somebody else
            LOGGER.warning("Chat id clash for user %s", user_id)
            return False

    def delete_chat(self, user_id: int, chat_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(chats).where(and_(
                chats.c.id == chat_id, chats.c.user_id == user_id,
            )))

    def delete_all_chats(self, user_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(chats).where(chats.c.user_id == user_id))

    # ------------------------------------------------- projects and prefs
    def load_store(self, user_id: int) -> dict:
        with self.engine.connect() as conn:
            project_rows = conn.execute(
                select(projects).where(projects.c.user_id == user_id)
            ).mappings().all()
            pref_row = conn.execute(
                select(prefs.c.data).where(prefs.c.user_id == user_id)
            ).first()
        found = []
        for row in project_rows:
            try:
                instructions = self.cipher.decrypt(row["instructions"])
            except ConfigError as exc:
                LOGGER.warning("Unreadable project %s: %s", row["id"], exc)
                continue
            found.append({"id": row["id"], "name": row["name"],
                          "instructions": instructions})
        try:
            saved_prefs = json.loads(pref_row[0]) if pref_row else {}
        except ValueError:
            saved_prefs = {}
        return {"prefs": saved_prefs, "projects": found}

    def save_store(self, user_id: int, store: dict) -> None:
        wanted = [
            p for p in store.get("projects", [])[:MAX_PROJECTS_PER_USER]
            if p.get("id") and p.get("name")
        ]
        ids = [str(p["id"])[:16] for p in wanted]
        with self.engine.begin() as conn:
            if ids:
                conn.execute(delete(projects).where(and_(
                    projects.c.user_id == user_id,
                    projects.c.id.not_in(ids),
                )))
            else:
                conn.execute(delete(projects).where(projects.c.user_id == user_id))
            for project in wanted:
                pid = str(project["id"])[:16]
                values = dict(
                    name=str(project["name"])[:60],
                    instructions=self.cipher.encrypt(
                        str(project.get("instructions", ""))[:MAX_INSTRUCTION_CHARS]
                    ),
                )
                exists = conn.execute(select(projects.c.id).where(and_(
                    projects.c.user_id == user_id, projects.c.id == pid,
                ))).first()
                if exists:
                    conn.execute(update(projects).where(and_(
                        projects.c.user_id == user_id, projects.c.id == pid,
                    )).values(**values))
                else:
                    conn.execute(insert(projects).values(
                        user_id=user_id, id=pid, **values
                    ))
            blob = json.dumps(store.get("prefs", {}), ensure_ascii=False)
            if conn.execute(select(prefs.c.user_id).where(
                    prefs.c.user_id == user_id)).first():
                conn.execute(update(prefs).where(
                    prefs.c.user_id == user_id).values(data=blob))
            else:
                conn.execute(insert(prefs).values(user_id=user_id, data=blob))

    # ------------------------------------------------------------- usage
    def add_usage(self, user_id: int, day: str, provider: str,
                  model: str) -> None:
        key = and_(
            usage.c.user_id == user_id, usage.c.day == day,
            usage.c.provider == provider[:40], usage.c.model == model[:120],
        )
        with self.engine.begin() as conn:
            if conn.execute(select(usage.c.requests).where(key)).first():
                conn.execute(update(usage).where(key).values(
                    requests=usage.c.requests + 1))
            else:
                conn.execute(insert(usage).values(
                    user_id=user_id, day=day, provider=provider[:40],
                    model=model[:120], requests=1,
                ))

    def usage_summary(self, user_id: int, days: int = 14) -> list[dict]:
        cutoff = (_now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(usage).where(and_(
                    usage.c.user_id == user_id, usage.c.day >= cutoff))
                .order_by(usage.c.day.desc())
            ).mappings().all()
        return [dict(r) for r in rows]

    # -------------------------------------------------------- saved keys
    def get_keys(self, user_id: int) -> dict[str, str]:
        if not self.cipher.enabled:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(api_keys).where(api_keys.c.user_id == user_id)
            ).mappings().all()
        found = {}
        for row in rows:
            try:
                found[row["provider"]] = self.cipher.decrypt(row["token"])
            except ConfigError as exc:
                LOGGER.warning("Unreadable saved key: %s", exc)
        return found

    def set_key(self, user_id: int, provider: str, key: str) -> None:
        if not self.cipher.enabled:
            raise ConfigError("Saving keys needs NASHMI_SECRET_KEY.")
        token = self.cipher.encrypt(key.strip()[:500])
        where = and_(api_keys.c.user_id == user_id,
                     api_keys.c.provider == provider[:40])
        with self.engine.begin() as conn:
            if conn.execute(select(api_keys.c.user_id).where(where)).first():
                conn.execute(update(api_keys).where(where).values(token=token))
            else:
                conn.execute(insert(api_keys).values(
                    user_id=user_id, provider=provider[:40], token=token))

    def delete_key(self, user_id: int, provider: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(api_keys).where(and_(
                api_keys.c.user_id == user_id,
                api_keys.c.provider == provider[:40])))

    # ------------------------------------------------------------ account
    def export_all(self, user_id: int) -> dict:
        """Everything stored about a user, except API keys."""
        user = self.get_user(user_id) or {}
        return {
            "account": {
                "email": user.get("email", ""), "name": user.get("name", ""),
                "created_at": str(user.get("created_at", "")),
            },
            "history": self.load_history(user_id),
            "store": self.load_store(user_id),
            "usage": self.usage_summary(user_id, days=3650),
        }

    def delete_user(self, user_id: int) -> None:
        """Remove the account and every row that belongs to it."""
        with self.engine.begin() as conn:
            for table in ALL_USER_TABLES:
                conn.execute(delete(table).where(table.c.user_id == user_id))
            conn.execute(delete(users).where(users.c.id == user_id))
