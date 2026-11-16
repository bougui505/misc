#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-06-02

import datetime
import fcntl
import os
import json
import sys
from exchangelib import Account, Configuration, DELEGATE, OAUTH2, EWSTimeZone
from exchangelib.winzone import MS_TIMEZONE_TO_IANA_MAP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

def get_access_token(client_id, tenant_id, username, cache_path, scopes):
    import msal
    
    cache_path = os.path.expanduser(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    
    cache = msal.SerializableTokenCache()
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache.deserialize(f.read())
            
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=authority,
        token_cache=cache
    )
    
    # Try silent token acquisition from cache
    accounts = app.get_accounts(username=username)
    result = None
    if accounts:
        result = app.acquire_token_silent(scopes, account=accounts[0])
        
    if not result:
        # Fall back to Device Code Flow first (best for headless/containers and bypasses redirect URI issues)
        try:
            print("Initiating Device Code Flow...")
            flow = app.initiate_device_flow(scopes)
            if "user_code" not in flow:
                raise RuntimeError(flow.get('error_description') or "Unknown error")
            print(flow["message"])
            result = app.acquire_token_by_device_flow(flow)
        except Exception as e:
            print(f"Device Code Flow failed ({e}). Trying interactive login...")
            try:
                result = app.acquire_token_interactive(scopes, login_hint=username)
            except Exception as e2:
                raise RuntimeError(f"All authentication flows failed. Device code error: {e}. Interactive error: {e2}")
            
    if "access_token" in result:
        if cache.has_state_changed:
            with open(cache_path, 'w') as f:
                f.write(cache.serialize())
            # Restrict cache file permissions to user only (chmod 600)
            os.chmod(cache_path, 0o600)
        return result["access_token"]
    else:
        error_msg = result.get("error_description", result.get("error"))
        raise RuntimeError(f"Could not acquire access token: {error_msg}")

CONFIG_PATH = os.path.expanduser('~/.config/outlook_to_google_calendar/config.json')

def load_config():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        default_config = {
            "outlook_email": "your_work_email@company.com",
            "outlook_username": "your_login_upn@company.com",
            "outlook_client_id": "YOUR_AZURE_AD_CLIENT_ID",
            "outlook_tenant_id": "YOUR_AZURE_AD_TENANT_ID",
            "token_cache_path": "~/.config/outlook_to_google_calendar/token_cache.bin",
            "ews_server": None,
            "sync_days": 30,
            "google_calendar_name": "Pasteur",
            "google_token_path": "~/.config/outlook_to_google_calendar/google_token.json",
            "google_credentials_path": "~/.config/outlook_to_google_calendar/google_credentials.json"
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default configuration file at {CONFIG_PATH}")
        print("Please fill in your configuration details in that file and run the script again.")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

# --- CONFIGURATION ---
config_data = load_config()

OUTLOOK_EMAIL = config_data["outlook_email"]
OUTLOOK_USERNAME = config_data.get("outlook_username", OUTLOOK_EMAIL)
OUTLOOK_CLIENT_ID = config_data["outlook_client_id"]
OUTLOOK_TENANT_ID = config_data["outlook_tenant_id"]
TOKEN_CACHE_PATH = config_data["token_cache_path"]
EWS_SERVER = config_data.get("ews_server") 
SYNC_DAYS = config_data.get("sync_days", 30)
GOOGLE_CALENDAR_NAME = config_data.get("google_calendar_name", "primary")
GOOGLE_CALENDAR_ID = config_data.get("google_calendar_id")
GOOGLE_TOKEN_PATH = config_data.get("google_token_path", "~/.config/outlook_to_google_calendar/google_token.json")
GOOGLE_CREDENTIALS_PATH = config_data.get("google_credentials_path", "~/.config/outlook_to_google_calendar/google_credentials.json")
TIMEZONE = config_data.get("timezone", "Europe/Paris")

# Map empty timezone ID to fallback IANA timezone to prevent warnings
MS_TIMEZONE_TO_IANA_MAP[''] = TIMEZONE

# Google Calendar API Scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_gcal_service():
    """Authenticates with Google Calendar and returns the service object."""
    token_path = os.path.expanduser(GOOGLE_TOKEN_PATH)
    creds_path = os.path.expanduser(GOOGLE_CREDENTIALS_PATH)
    
    creds = None
    if os.path.exists(token_path):
        creds = GoogleCredentials.from_authorized_user_file(token_path, SCOPES)
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Failed to refresh Google token: {e}")
                print("Falling back to full authentication flow...")
                creds = None
        
    if not creds or not creds.valid:
        if not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Google Client Credentials file not found at {creds_path}. "
                f"Please download credentials.json from Google Cloud Console, rename it to "
                f"google_credentials.json, and place it there."
            )
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0, open_browser=False)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

def get_outlook_events():
    """Connects to Exchange EWS and pulls calendar events for the configured number of days."""
    from exchangelib.credentials import OAuth2AuthorizationCodeCredentials
    from oauthlib.oauth2 import OAuth2Token
    
    server = EWS_SERVER if EWS_SERVER else 'outlook.office365.com'
    scope_resource = server if server != 'outlook.office365.com' else 'outlook.office365.com'
    scopes = [f"https://{scope_resource}/EWS.AccessAsUser.All"]
    
    print("Acquiring OAuth2 access token...")
    token_str = get_access_token(
        client_id=OUTLOOK_CLIENT_ID,
        tenant_id=OUTLOOK_TENANT_ID,
        username=OUTLOOK_USERNAME,
        cache_path=TOKEN_CACHE_PATH,
        scopes=scopes
    )
    token = OAuth2Token(params={'access_token': token_str})
    creds = OAuth2AuthorizationCodeCredentials(access_token=token)
    
    config = Configuration(server=server, credentials=creds, auth_type=OAUTH2)
    account = Account(
        primary_smtp_address=OUTLOOK_EMAIL,
        config=config,
        autodiscover=False,
        access_type=DELEGATE,
        default_timezone=EWSTimeZone(TIMEZONE)
    )
        
    tz = account.default_timezone
    now = datetime.datetime.now(tz)
    start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=SYNC_DAYS)
    
    print(f"Fetching Outlook events from {start_date.date()} to {end_date.date()}...")
    return account.calendar.view(start=start_date, end=end_date)

def parse_gcal_datetime(dt_dict):
    dt_str = dt_dict.get('dateTime') or dt_dict.get('date')
    if not dt_str:
        return None
    if 'T' not in dt_str:
        dt = datetime.datetime.fromisoformat(dt_str)
        return dt.replace(tzinfo=datetime.timezone.utc)
    dt = datetime.datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)

def get_or_create_calendar(gcal_service, name):
    """Retrieves the ID of a Google Calendar by name, creating it if it doesn't exist."""
    if GOOGLE_CALENDAR_ID:
        return GOOGLE_CALENDAR_ID

    print(f"Checking Google Calendar list for '{name}'...")
    matches = []
    page_token = None
    while True:
        calendar_list = gcal_service.calendarList().list(pageToken=page_token).execute()
        for entry in calendar_list.get('items', []):
            if entry.get('summary') == name:
                matches.append(entry)
        page_token = calendar_list.get('nextPageToken')
        if not page_token:
            break

    if matches:
        if len(matches) > 1:
            ids = ", ".join(m['id'] for m in matches)
            print(
                f"WARNING: Found {len(matches)} Google Calendars named '{name}' ({ids}). "
                f"Using '{matches[0]['id']}'. Delete the duplicate(s) in Google Calendar "
                f"or set 'google_calendar_id' in {CONFIG_PATH}."
            )
        return matches[0]['id']

    print(f"Calendar '{name}' not found. Creating a new one...")
    new_calendar = {
        'summary': name,
        'timeZone': 'UTC'
    }
    created_calendar = gcal_service.calendars().insert(body=new_calendar).execute()
    return created_calendar['id']

def _build_gcal_times(item):
    """Convert an exchangelib CalendarItem's start/end into Google Calendar dicts and comparison keys."""
    is_all_day = getattr(item, 'is_all_day', False) or not isinstance(item.start, datetime.datetime)
    if is_all_day:
        start_str = item.start.strftime('%Y-%m-%d')
        end_str = item.end.strftime('%Y-%m-%d')
        return {'date': start_str}, {'date': end_str}, (start_str, end_str), start_str
    start_utc = datetime.datetime.fromtimestamp(item.start.timestamp(), tz=datetime.timezone.utc)
    end_utc = datetime.datetime.fromtimestamp(item.end.timestamp(), tz=datetime.timezone.utc)
    return (
        {'dateTime': start_utc.isoformat()},
        {'dateTime': end_utc.isoformat()},
        (start_utc, end_utc),
        str(start_utc.date()),
    )

def sync_calendars():
    lock_path = os.path.join(os.path.dirname(CONFIG_PATH), 'sync.lock')
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another sync instance is already running. Exiting.")
        return

    try:
        _run_sync()
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

def _run_sync():
    gcal_service = get_gcal_service()
    outlook_events = get_outlook_events()
    
    # Resolve target calendar ID
    calendar_id = 'primary'
    if GOOGLE_CALENDAR_NAME and GOOGLE_CALENDAR_NAME.lower() != 'primary':
        calendar_id = get_or_create_calendar(gcal_service, GOOGLE_CALENDAR_NAME)
        
    print(f"Target Google Calendar ID: {calendar_id}")
    
    # Fetch existing Google Calendar events from 1 day ago to prevent duplicate creation
    # and cover overlapping/recently past events.
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    time_min = (now_utc - datetime.timedelta(days=1)).isoformat()
    
    print(f"Fetching existing Google Calendar events since {time_min}...")
    gcal_events = []
    page_token = None
    while True:
        events_result = gcal_service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            singleEvents=True,
            maxResults=2500,
            pageToken=page_token
        ).execute()
        gcal_events.extend(events_result.get('items', []))
        page_token = events_result.get('nextPageToken')
        if not page_token:
            break
    
    # Map outlook_id to the Google Calendar event
    gcal_by_outlook_id = {}
    for event in gcal_events:
        ext_props = event.get('extendedProperties', {})
        private_props = ext_props.get('private', {})
        outlook_id = private_props.get('outlook_id')
        if outlook_id:
            gcal_by_outlook_id[outlook_id] = event

    for item in outlook_events:
        title = item.subject if item.subject else "Busy"
        try:
            gcal_start, gcal_end, time_key, date_label = _build_gcal_times(item)
            location = item.location if item.location else ''
            
            gcal_event_body = {
                'summary': title,
                'location': location,
                'description': 'Synced from corporate Outlook via Linux EWS Script',
                'start': gcal_start,
                'end': gcal_end,
                'extendedProperties': {
                    'private': {
                        'outlook_id': item.id
                    }
                }
            }
            
            if item.id in gcal_by_outlook_id:
                existing_event = gcal_by_outlook_id[item.id]
                
                # Compare fields to check if an update is needed
                existing_title = existing_event.get('summary', '')
                existing_location = existing_event.get('location', '')
                
                if 'date' in gcal_start:
                    existing_key = (
                        existing_event['start'].get('date'),
                        existing_event['end'].get('date'),
                    )
                else:
                    existing_key = (
                        parse_gcal_datetime(existing_event['start']),
                        parse_gcal_datetime(existing_event['end']),
                    )
                
                title_changed = (title != existing_title)
                location_changed = (location != existing_location)
                time_changed = (time_key != existing_key)
                
                if title_changed or location_changed or time_changed:
                    print(f"Updating event in Google: '{title}' ({date_label})")
                    gcal_service.events().patch(
                        calendarId=calendar_id,
                        eventId=existing_event['id'],
                        body=gcal_event_body
                    ).execute()
            else:
                print(f"Syncing to Google (New): '{title}' ({date_label})")
                gcal_service.events().insert(
                    calendarId=calendar_id,
                    body=gcal_event_body
                ).execute()
        except Exception as e:
            print(f"ERROR syncing event '{title}': {e}", file=sys.stderr)

if __name__ == '__main__':
    sync_calendars()
