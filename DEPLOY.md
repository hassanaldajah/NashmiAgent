# Deploying Nashmi Agent with accounts

Nashmi works in three modes. You choose the mode with settings, not code.

| Mode | When | Chats are stored |
|---|---|---|
| **Local** | you run `Start_Nashmi.bat` on your computer | in `chat_history.json` and `settings.json` next to `app.py` |
| **Public demo** | `NASHMI_PUBLIC_MODE=1`, no database | nowhere (this browser session only) |
| **Public with accounts** | `NASHMI_PUBLIC_MODE=1` + sign-in + database | in your database, one private space per user |

In both public modes the server **never uses an API key of yours**. Every
visitor pastes their own key (the app links to the key page of the
provider they pick): Gemini and Groq have free keys, OpenAI (ChatGPT
models) and Claude (Anthropic) are billed to the visitor's own account. Signed-in users may save a key; it is encrypted.

## 1. Database (PostgreSQL)

Streamlit Community Cloud forgets files between restarts, so accounts need an
external database. Any PostgreSQL works. Neon and Supabase both have a free
tier. Copy the connection string, for example:

```
postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

The tables are created automatically on the first start.

## 2. Encryption key

Chats, project instructions and saved API keys are encrypted before they are
written to the database. Generate a key once and keep it safe (if you lose it
the stored data cannot be read):

```
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

To rotate it later, put the new key **first** and keep the old one after a
comma: `NEW_KEY,OLD_KEY`. A production database without this key is refused.

## 3. Sign-in (Google, optional Apple)

Nashmi does not store passwords and does not send its own one-time codes.
Google and Apple do the verification (2-step, passkeys, codes) on their side.
Accounts from different providers are kept separate on purpose; they are
never merged by e-mail address.

### Google

1. Google Cloud Console, **APIs and Services**, **Credentials**, create an
   **OAuth client ID** of type *Web application*.
2. Authorised redirect URI: `https://YOUR-APP.streamlit.app/oauth2callback`
   (add `http://localhost:8501/oauth2callback` for local tests).
3. Copy the client ID and client secret into the secrets below.

### Apple (experimental, not tested by the author)

Needs a paid Apple Developer account, a *Services ID*, a verified domain and
an `https` return URL. Apple's client secret is a short-lived signed token that
you must regenerate (at most every 6 months). Apple only returns the name and
e-mail when `response_mode=form_post` is used, which Streamlit's sign-in
callback does not support, so the config below asks for `openid` only: users
will appear as "Account" without an e-mail. If this is not worth it for you,
simply leave `[auth.apple]` out and only the Google button is shown.

## 4. Secrets

On Streamlit Community Cloud: **App settings, Secrets**. Locally:
`.streamlit/secrets.toml` (already in `.gitignore`).

```toml
NASHMI_PUBLIC_MODE = "1"
NASHMI_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
NASHMI_SECRET_KEY = "PASTE_THE_FERNET_KEY"

[auth]
redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
cookie_secret = "A_LONG_RANDOM_STRING"

[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

# Optional, see the Apple notes above
[auth.apple]
client_id = "com.example.nashmi.web"
client_secret = "THE_SIGNED_APPLE_JWT"
server_metadata_url = "https://appleid.apple.com/.well-known/openid-configuration"
client_kwargs = { scope = "openid" }
```

Do **not** put `GEMINI_API_KEY`, `GROQ_API_KEY` or similar in these secrets.
Public mode ignores them anyway, but there is no reason to store them.

## 5. What is protected

- Every database query uses bound parameters and is limited to the signed-in
  user's own rows; one account cannot read, overwrite or delete another's.
- Chats, titles, project instructions and saved keys are encrypted at rest
  (Fernet: AES with an HMAC). Without `NASHMI_SECRET_KEY` the app refuses to
  start a production database and refuses to save API keys.
- Per-user limits: 1000 chats, 1.5 million characters per chat, 50 projects.
- When someone signs out, everything of theirs is removed from the session.
- Users can export all their data or delete their account from the
  **Account** page.
- API keys typed in the sidebar live in that browser session only.

## 5b. Security notes for the owner

- **You can read your users' data.** The database holds e-mail and name in
  plain text, and the chats are encrypted with *your* key. This protects users
  if the database leaks or a database admin looks around; it does not hide
  anything from whoever controls the server and the key. Tell your users, and
  publish a short privacy note.
- **Keep the three secrets apart:** database URL, `NASHMI_SECRET_KEY` and the
  Google client secret. Never commit them (`.streamlit/secrets.toml` and `.env`
  are in `.gitignore`). If one leaks, rotate it: a new Google client secret, a
  new database password, or `NEW_KEY,OLD_KEY` for the encryption key.
- Use a long random `cookie_secret`; changing it signs everyone out.
- Public mode is switched on automatically when `[auth]` is present, so the
  server's own API keys and local chat files are never used for visitors.
- Uploaded files (PDF, PPTX, DOCX, images) are parsed on the server. Sizes,
  pages, pictures and pixel counts are capped, and long inputs are cut before
  any pattern matching, but keep the dependencies up to date
  (`py -m pip install -U -r requirements.txt`).
- Fonts are loaded from Google Fonts and the diagram library from jsDelivr
  (pinned version with an integrity hash). Both see the visitor's IP address.
- Every prompt, answer and uploaded document goes to the AI provider the
  visitor chose, under that provider's terms.

## 6. Landing page, sign-in page and admin screen

On a public server (`NASHMI_PUBLIC_MODE=1` or `[auth]` set) visitors first see
a **landing page**, then a **sign-in page** (Google) or "Continue as guest".
On your own computer none of this appears.

Set `NASHMI_ADMIN_EMAILS` (comma separated) to the e-mail addresses that may
open the **Admin** screen. Only a *verified Google* e-mail on that list counts,
and Google does the password / 2-step verification, so Nashmi never sees a
password. Admins get an **Admin** button in the sidebar with four tabs:

- **Live**: who is online now (active in the last 5 minutes), sign-ins today,
  recent sign-ins. Refreshes itself every 10 seconds.
- **Statistics**: active users, sign-ups, sign-ins and AI requests per day,
  requests by model, most active users, chats by track.
- **Users**: everyone who signed in, searchable, with a CSV download.
- **System**: database type, reachability and latency, encryption status.

Chat contents stay encrypted and are never shown to admins. Sign-in records
older than 90 days are deleted automatically. Because this stores e-mail
addresses and sign-in times, say so in your privacy note (the landing page
already shows a short one).

## 7. Run it locally

```
py -m pip install -r requirements.txt
py -m streamlit run app.py
```
