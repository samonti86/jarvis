"""One-shot interactive OAuth setup for M62 Outlook calendar awareness.

Runs the MSAL device-code flow against the user's personal Microsoft account
and saves a refresh token to %LOCALAPPDATA%\\Jarvis\\msal_cache.bin. After
this succeeds, Jarvis can read the calendar silently (no further user
interaction) for as long as the refresh token remains valid (~90 days for
personal accounts; MSAL handles silent renewals before expiry).

Prerequisite: OUTLOOK_CLIENT_ID must be set in .env to the Application (client)
ID of an Azure app registered with:
  - Supported account types: Personal Microsoft accounts only
  - "Allow public client flows" = Yes
  - API permissions: Microsoft Graph → Delegated → Calendars.Read

See .env.example for the full step-by-step.

Usage (from the project root):
    .\\venv\\Scripts\\python.exe scripts\\outlook_auth.py

It will print a URL and a code. Open the URL on any device (your phone is
fine), sign in with the Microsoft account whose calendar you want Jarvis to
read, and enter the code. The script polls until you've completed the flow
or 15 minutes elapse.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from the project root so OUTLOOK_CLIENT_ID is available even if
# this script is run directly (not via main.py).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.outlook_calendar import CLIENT_ID, _OAuthClient  # noqa: E402

if not CLIENT_ID:
    print("ERROR: OUTLOOK_CLIENT_ID is not set in .env.\n"
          "Open .env.example and follow the M62 setup steps to register an "
          "Azure app and copy its Application (client) ID into .env, then "
          "re-run this script.")
    sys.exit(2)

print(f"OUTLOOK_CLIENT_ID is set; starting device-code flow.\n")
client = _OAuthClient(CLIENT_ID)
ok = client.acquire_via_device_code()
if ok:
    print("\nAuthorisation complete. Jarvis can now read your calendar.")
    print("Try: 'Jarvis, what's on my calendar today?'")
    sys.exit(0)
print("\nAuthorisation failed. See errors above; ensure the Azure app has "
      "the Calendars.Read delegated permission and 'Allow public client "
      "flows' = Yes.")
sys.exit(1)
