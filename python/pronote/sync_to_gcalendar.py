#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pronotepy>=2.15.0",
#     "google-api-python-client>=2.100.0",
#     "google-auth-httplib2>=0.2.0",
#     "google-auth-oauthlib>=1.2.0",
#     "python-dateutil>=2.9.0",
#     "requests>=2.30.0",
#     "beautifulsoup4>=4.8.2",
#     "pycryptodome>=3.9.4",
# ]
# ///
"""
Synchronisation directe Pronote -> Google Calendar via l'API Google Calendar v3.
"""

import os
import sys
import argparse
import datetime
import getpass
from pathlib import Path

import pronotepy
import pronotepy.ent

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]

DEFAULT_CREDENTIALS_PATH = os.path.expanduser("~/.config/outlook_to_google_calendar/google_credentials.json")
DEFAULT_TOKEN_PATH = os.path.expanduser("~/.config/outlook_to_google_calendar/google_token.json")


def load_env_file(filepath: Path):
    if not filepath.exists():
        return
    # Vérifier et forcer les permissions 600 si le fichier contient des mots de passe
    current_mode = filepath.stat().st_mode & 0o777
    if current_mode not in (0o600, 0o400):
        try:
            filepath.chmod(0o600)
            print(f"[Sécurité] Permissions de {filepath.name} restreintes à 0600 (lecture/écriture propriétaire uniquement).")
        except Exception:
            pass

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k not in os.environ:
                os.environ[k] = v


def authenticate_google(credentials_file: str, token_file: str):
    """Authentifie l'utilisateur via OAuth 2.0 et stocke / rafraîchit le token."""
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Rafraîchissement automatique du token Google...")
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                print(f"\n[ERREUR] Le fichier '{credentials_file}' est introuvable !", file=sys.stderr)
                sys.exit(1)
            print("Ouverture du navigateur pour autoriser l'accès à Google Calendar...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w", encoding="utf-8") as token:
            token.write(creds.to_json())
    return creds


def get_or_create_calendar(service, cal_name: str) -> str:
    """Trouve ou crée un calendrier secondaire dédié aux cours Pronote."""
    page_token = None
    while True:
        calendar_list = service.calendarList().list(pageToken=page_token).execute()
        for item in calendar_list.get("items", []):
            if item.get("summary") == cal_name:
                return item["id"]
        page_token = calendar_list.get("nextPageToken")
        if not page_token:
            break

    print(f"Création d'un calendrier dédié dans Google Calendar : '{cal_name}'...")
    new_calendar = {"summary": cal_name, "timeZone": "Europe/Paris"}
    created = service.calendars().insert(body=new_calendar).execute()
    return created["id"]


def sync_pronote_to_google(service, calendar_id: str, lessons, date_from: datetime.date, date_to: datetime.date):
    """Synchronise les cours dans l'agenda Google sélectionné."""
    time_min = datetime.datetime.combine(date_from, datetime.time.min, tzinfo=datetime.timezone.utc).isoformat()
    time_max = datetime.datetime.combine(date_to, datetime.time.max, tzinfo=datetime.timezone.utc).isoformat()

    print("Récupération des événements existants dans l'agenda Google...")
    existing_events = {}
    page_token = None
    while True:
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            pageToken=page_token,
        ).execute()

        for ev in events_result.get("items", []):
            ext_id = ev.get("extendedProperties", {}).get("private", {}).get("pronote_lesson_id")
            if ext_id:
                existing_events[ext_id] = ev

        page_token = events_result.get("nextPageToken")
        if not page_token:
            break

    print(f"{len(existing_events)} cours Pronote déjà enregistrés sur cette période.")

    created_count = 0
    updated_count = 0
    deleted_count = 0

    seen_lesson_ids = set()

    for lesson in lessons:
        lesson_key = f"{lesson.id}_{lesson.start.strftime('%Y%m%d%H%M')}"
        seen_lesson_ids.add(lesson_key)

        subject_name = lesson.subject.name if lesson.subject else "Cours"
        if lesson.canceled:
            summary = f"[ANNULÉ] {subject_name}"
        elif lesson.detention:
            summary = f"[RETENUE] {subject_name}"
        elif lesson.outing:
            summary = f"[SORTIE] {subject_name}"
        else:
            summary = subject_name

        desc_lines = []
        if lesson.teacher_name:
            desc_lines.append(f"Professeur : {lesson.teacher_name}")
        if lesson.group_name:
            desc_lines.append(f"Groupe : {lesson.group_name}")
        if lesson.classroom:
            desc_lines.append(f"Salle : {lesson.classroom}")
        if lesson.status:
            desc_lines.append(f"Statut : {lesson.status}")
        if lesson.memo:
            desc_lines.append(f"Note : {lesson.memo}")
        if lesson.test:
            desc_lines.append("Évaluation / Devoir prévu")

        event_body = {
            "summary": summary,
            "location": lesson.classroom or "",
            "description": "\n".join(desc_lines),
            "start": {
                "dateTime": lesson.start.isoformat(),
                "timeZone": "Europe/Paris",
            },
            "end": {
                "dateTime": lesson.end.isoformat(),
                "timeZone": "Europe/Paris",
            },
            "extendedProperties": {
                "private": {
                    "pronote_lesson_id": lesson_key,
                }
            },
        }

        if lesson_key in existing_events:
            ev_id = existing_events[lesson_key]["id"]
            # Vérifier si modification nécessaire
            old_ev = existing_events[lesson_key]
            if (old_ev.get("summary") != summary or
                old_ev.get("location") != (lesson.classroom or "") or
                old_ev.get("description") != "\n".join(desc_lines)):
                service.events().update(calendarId=calendar_id, eventId=ev_id, body=event_body).execute()
                updated_count += 1
        else:
            service.events().insert(calendarId=calendar_id, body=event_body).execute()
            created_count += 1

    # Supprimer les cours qui ne sont plus dans l'emploi du temps pour cette plage
    for ext_id, ev in existing_events.items():
        if ext_id not in seen_lesson_ids:
            service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
            deleted_count += 1

    print("\n[Bilan de synchronisation]")
    print(f"  + Nouveaux cours ajoutés : {created_count}")
    print(f"  ~ Cours mis à jour        : {updated_count}")
    print(f"  - Cours supprimés        : {deleted_count}")


def main():
    parser = argparse.ArgumentParser(description="Synchronisation directe Pronote -> Google Calendar API")
    parser.add_argument("--config", default="config.env", help="Fichier config.env")
    parser.add_argument(
        "--credentials",
        default=os.environ.get("GOOGLE_CREDENTIALS", DEFAULT_CREDENTIALS_PATH if os.path.exists(DEFAULT_CREDENTIALS_PATH) else "credentials.json"),
        help="Chemin vers le fichier credentials.json Google OAuth2"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GOOGLE_TOKEN", DEFAULT_TOKEN_PATH if os.path.exists(DEFAULT_TOKEN_PATH) else "token.json"),
        help="Chemin vers le fichier token.json Google OAuth2"
    )
    parser.add_argument("--calendar-name", default=None, help="Nom de l'agenda Google dédié")
    parser.add_argument("--days-past", type=int, default=None, help="Nombre de jours passés à synchroniser")
    parser.add_argument("--days-future", type=int, default=None, help="Nombre de jours futurs à synchroniser")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        load_env_file(cfg_path)

    username = os.environ.get("ENT_USERNAME")
    password = os.environ.get("ENT_PASSWORD")
    pronote_url = os.environ.get("PRONOTE_URL", "https://0750611g.index-education.net/pronote/parent.html")
    child_name = os.environ.get("CHILD_NAME", "")

    days_past = args.days_past if args.days_past is not None else int(os.environ.get("DAYS_PAST", "7"))
    days_future = args.days_future if args.days_future is not None else int(os.environ.get("DAYS_FUTURE", "28"))
    cal_name = args.calendar_name or os.environ.get("CALENDAR_NAME")

    if not username:
        username = input("Identifiant ENT Paris Classe Numérique : ").strip()
    if not password:
        password = getpass.getpass("Mot de passe ENT : ")

    # 1. Pronote Fetch
    print("=== Connexion à Pronote ===")
    try:
        client = pronotepy.ParentClient(
            pronote_url=pronote_url,
            username=username,
            password=password,
            ent=pronotepy.ent.paris_classe_numerique,
        )
    except Exception as e:
        print(f"Erreur de connexion Pronote : {e}", file=sys.stderr)
        sys.exit(1)

    if not client.logged_in:
        print("Erreur : connexion Pronote non authentifiée.", file=sys.stderr)
        sys.exit(1)

    target_child = None
    if child_name:
        for c in client.children:
            if child_name.lower() in c.name.lower():
                target_child = c
                break
    if not target_child:
        target_child = client.children[0]

    client.set_child(target_child)
    print(f"Élève sélectionné : {target_child.name}")

    if not cal_name:
        cal_name = f"Pronote - {target_child.name}"

    # 2. Google Calendar Authentication
    print(f"\n=== Connexion Google Calendar (via {args.token}) ===")
    creds = authenticate_google(args.credentials, args.token)
    service = build("calendar", "v3", credentials=creds)
    calendar_id = get_or_create_calendar(service, cal_name)
    print(f"Agenda Google : '{cal_name}' (ID: {calendar_id})")

    today = datetime.date.today()
    date_from = today - datetime.timedelta(days=days_past)
    date_to = today + datetime.timedelta(days=days_future)
    print(f"Récupération des cours du {date_from.strftime('%d/%m/%Y')} au {date_to.strftime('%d/%m/%Y')}...")

    lessons = client.lessons(date_from, date_to)
    print(f"{len(lessons)} cours récupérés depuis Pronote.")

    # 3. Synchronisation
    print("\n=== Synchronisation des cours vers Google Calendar ===")
    sync_pronote_to_google(service, calendar_id, lessons, date_from, date_to)
    print("\nSynchronisation terminée avec succès !")


if __name__ == "__main__":
    main()
