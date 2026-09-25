#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "google-api-python-client",
#     "google-auth-oauthlib",
#     "google-auth-httplib2",
#     "python-dateutil",
# ]
# ///

"""
Sync Google Tasks to Remind syntax (.rem format).
"""

import argparse
import datetime
import sys
from pathlib import Path

from dateutil import parser as date_parser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/tasks.readonly"]

DEFAULT_CREDENTIAL_LOCATIONS = [
    Path.home() / ".config" / "outlook_to_google_calendar" / "google_credentials.json",
    Path.home() / ".config" / "gtasks_to_rem" / "google_credentials.json",
    Path(__file__).resolve().parent / "google_credentials.json",
]

DEFAULT_TOKEN_PATH = Path.home() / ".config" / "gtasks_to_rem" / "token.json"


def find_credentials(custom_path=None):
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Credentials file not found at: {custom_path}")

    for candidate in DEFAULT_CREDENTIAL_LOCATIONS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find google_credentials.json in any default location:\n"
        + "\n".join(f" - {p}" for p in DEFAULT_CREDENTIAL_LOCATIONS)
        + "\nPlease specify --credentials /path/to/credentials.json"
    )


def get_authenticated_service(credentials_path, token_path):
    creds = None
    token_path = Path(token_path).expanduser().resolve()

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("tasks", "v1", credentials=creds)


def escape_remind(text: str) -> str:
    """Escape characters that have special meaning in Remind MSG expressions."""
    if not text:
        return ""
    # In Remind, [expr] denotes expression evaluation; escape [ and ] as [""]
    return text.replace("[", '["["]').replace("]", '["]"]')


def format_remind_line(task, label="TASK", list_title=""):
    raw_title = (task.get("title") or "Untitled Task").strip().replace("\n", " ")
    title = escape_remind(raw_title)

    raw_notes = (task.get("notes") or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    notes = escape_remind(raw_notes)

    due_str = task.get("due")

    prefix = f"{label} " if label else ""
    summary = f"{prefix}{title}"
    if list_title and list_title not in ("My Tasks", "Mes tâches"):
        escaped_list = escape_remind(list_title)
        summary += f" ({escaped_list})"

    if notes:
        clean_notes = notes.replace("\n", "%_")
        msg_part = f'%"{summary}%"%_{clean_notes}'
    else:
        msg_part = summary

    if due_str:
        try:
            dt = date_parser.isoparse(due_str)
            local_dt = dt.astimezone()
            date_str = f"{local_dt.year:04d}-{local_dt.month:02d}-{local_dt.day:02d}"

            if not (dt.hour == 0 and dt.minute == 0 and dt.second == 0):
                time_str = f" AT {local_dt.strftime('%H:%M')}"
            else:
                time_str = ""

            return f"REM {date_str}{time_str} TAG task MSG {msg_part}", True
        except Exception:
            pass

    # No due date or unparseable
    return f"REM TAG noduedate TAG task MSG {msg_part}", False


def main():
    parser = argparse.ArgumentParser(description="Export Google Tasks to Remind (.rem) format.")
    parser.add_argument("-l", "--label", default="TASK", help="Label prefix for Remind entries (default: TASK)")
    parser.add_argument("--credentials", help="Path to google_credentials.json")
    parser.add_argument("--token", default=str(DEFAULT_TOKEN_PATH), help="Path to save/read token.json")
    parser.add_argument("--include-completed", action="store_true", help="Include completed tasks")
    args = parser.parse_args()

    try:
        cred_path = find_credentials(args.credentials)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        service = get_authenticated_service(cred_path, args.token)
    except Exception as e:
        print(f"Authentication error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        tasklists_result = service.tasklists().list().execute()
        tasklists = tasklists_result.get("items", [])

        dated_tasks = []
        nodue_tasks = []

        for tl in tasklists:
            tl_id = tl["id"]
            tl_title = tl.get("title", "")

            tasks_request = service.tasks().list(
                tasklist=tl_id,
                showCompleted=args.include_completed,
                showHidden=False,
            )
            tasks_result = tasks_request.execute()
            tasks = tasks_result.get("items", [])

            for task in tasks:
                if task.get("status") == "completed" and not args.include_completed:
                    continue
                rem_line, has_due = format_remind_line(task, label=args.label, list_title=tl_title)
                if has_due:
                    dated_tasks.append(rem_line)
                else:
                    nodue_tasks.append(rem_line)
    except HttpError as e:
        if "accessNotConfigured" in str(e) or "has not been used in project" in str(e):
            print(
                "Error: Google Tasks API is not enabled in your Google Cloud project.\n"
                "Please enable it by opening the following link in your browser:\n"
                "https://console.developers.google.com/apis/api/tasks.googleapis.com/overview?project=299506934817\n\n"
                "After clicking 'Enable', wait ~1 minute and retry.",
                file=sys.stderr,
            )
        else:
            print(f"Google API Error: {e}", file=sys.stderr)
        sys.exit(1)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"# Google Tasks exported on {now_str}")
    print()

    if dated_tasks:
        print("# --- Scheduled Tasks ---")
        for line in dated_tasks:
            print(line)
        print()

    if nodue_tasks:
        print("# --- Tasks without due date ---")
        for line in nodue_tasks:
            print(line)
        print()


if __name__ == "__main__":
    main()
