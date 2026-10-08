#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "google-api-python-client",
#     "google-auth",
#     "google-genai",
#     "pypdf",
# ]
# ///

"""
Add event tickets in PDF format to Google Calendar & Google Drive.

Extracts event information (title, date/time, location, description) using Gemini,
uploads the PDF ticket to Google Drive in 'Documents/tickets-billets',
creates the event in Google Calendar, and attaches the Drive file to the event.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from pypdf import PdfReader
from google import genai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

RCLONE_CONF = Path.home() / ".config/rclone/rclone.conf"
CALENDAR_TOKEN = Path.home() / ".config/outlook_to_google_calendar/google_token.json"
REMOTE_DIR = "GoogleDrive:Documents/tickets-billets"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from all pages of a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Warning: pypdf failed to extract text ({e}), trying pdftotext...", file=sys.stderr)

    res = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()

    return ""


def parse_event_with_gemini(ticket_text: str, filename: str) -> dict:
    """Use Gemini 2.5 Flash to extract structured event details."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    current_year = datetime.now().year

    prompt = f"""
You are an intelligent assistant extracting calendar event details from a ticket / booking confirmation PDF.
Filename: {filename}
Current year reference: {current_year}

Ticket text:
\"\"\"
{ticket_text}
\"\"\"

Please extract the event details into valid JSON with this exact schema:
{{
  "title": "Clean, descriptive event/show/concert title",
  "start_iso": "YYYY-MM-DDTHH:MM:SS",
  "end_iso": "YYYY-MM-DDTHH:MM:SS (if end time is not mentioned, assume 2 hours after start_iso)",
  "location": "Venue name and address if available, or venue name",
  "description": "Short summary including placement/seats, ticket/booking numbers, price, attendee names, and special instructions like arriving early."
}}

If multiple tickets/attendees are in the document for the SAME event, combine their info into one event and list the attendees/ticket numbers in the description.
Respond ONLY with the JSON object.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    data = json.loads(response.text)
    if isinstance(data, list):
        data = data[0]
    return data


def upload_to_drive(pdf_path: Path) -> tuple[str, str]:
    """Upload PDF to Google Drive via rclone and retrieve file ID & link."""
    dest = f"{REMOTE_DIR}/{pdf_path.name}"
    print(f"--> Uploading ticket to {dest} via rclone...")
    subprocess.run(["rclone", "copyto", str(pdf_path), dest], check=True)

    # Get file ID from rclone
    res = subprocess.run(["rclone", "lsf", dest, "--format", "i"], capture_output=True, text=True, check=True)
    file_id = res.stdout.strip()
    if not file_id:
        raise RuntimeError(f"Could not retrieve Drive file ID for {dest}")

    web_view_url = f"https://drive.google.com/file/d/{file_id}/view"
    return file_id, web_view_url


def get_calendar_service():
    """Build and return the Google Calendar service using stored token."""
    if not CALENDAR_TOKEN.exists():
        raise FileNotFoundError(f"Calendar token not found at {CALENDAR_TOKEN}")

    with open(CALENDAR_TOKEN) as f:
        token_data = json.load(f)

    creds = Credentials.from_authorized_user_info(token_data)
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(service, event_info: dict, file_id: str, web_view_url: str, filename: str, calendar_id: str = "primary") -> dict:
    """Create a Google Calendar event with attached Google Drive ticket."""
    start_iso = event_info.get("start_iso")
    end_iso = event_info.get("end_iso")

    if not end_iso and start_iso:
        st = datetime.fromisoformat(start_iso)
        end_iso = (st + timedelta(hours=2)).isoformat()

    description = event_info.get("description", "")
    description += f"\n\nBillet Drive : {web_view_url}"

    body = {
        "summary": event_info.get("title", "Événement"),
        "location": event_info.get("location", ""),
        "description": description,
        "start": {
            "dateTime": start_iso,
            "timeZone": "Europe/Paris",
        },
        "end": {
            "dateTime": end_iso,
            "timeZone": "Europe/Paris",
        },
        "attachments": [
            {
                "fileUrl": web_view_url,
                "title": filename,
                "mimeType": "application/pdf",
                "fileId": file_id,
            }
        ],
    }

    created = service.events().insert(
        calendarId=calendar_id,
        body=body,
        supportsAttachments=True,
    ).execute()

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Extract event details from a PDF ticket, upload to Google Drive, and create a linked Google Calendar event."
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF ticket file")
    parser.add_argument("--calendar", "-c", default="primary", help="Target calendar ID (default: primary)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Extract details and print without creating/uploading")

    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"Error: File not found: {args.pdf}")

    print(f"--> Extracting text from {args.pdf.name}...")
    text = extract_pdf_text(args.pdf)
    if not text:
        sys.exit("Error: No text could be extracted from PDF.")

    print("--> Analyzing ticket details with Gemini...")
    event_info = parse_event_with_gemini(text, args.pdf.name)

    print("\n--- Event Details Extracted ---")
    print(f"Titre    : {event_info.get('title')}")
    print(f"Début    : {event_info.get('start_iso')}")
    print(f"Fin      : {event_info.get('end_iso')}")
    print(f"Lieu     : {event_info.get('location')}")
    print(f"Détails  : {event_info.get('description')}")
    print("-------------------------------\n")

    if args.dry_run:
        print("[Dry run] No changes made.")
        return

    file_id, web_view_url = upload_to_drive(args.pdf)
    print(f"--> Ticket uploaded to Drive: {web_view_url} (ID: {file_id})")

    print(f"--> Adding event to Google Calendar ('{args.calendar}')...")
    calendar_service = get_calendar_service()
    created_event = create_calendar_event(
        service=calendar_service,
        event_info=event_info,
        file_id=file_id,
        web_view_url=web_view_url,
        filename=args.pdf.name,
        calendar_id=args.calendar,
    )

    print(f"\n[OK] Event successfully created!")
    print(f"Google Calendar URL : {created_event.get('htmlLink')}")
    print(f"Drive Ticket URL    : {web_view_url}")


if __name__ == "__main__":
    main()
