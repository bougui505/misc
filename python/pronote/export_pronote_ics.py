#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pronotepy>=2.15.0",
#     "icalendar>=6.0.0",
#     "python-dateutil>=2.9.0",
#     "requests>=2.30.0",
#     "beautifulsoup4>=4.8.2",
#     "pycryptodome>=3.9.4",
# ]
# ///
"""
Exporte l'emploi du temps Pronote (compte parent) vers un fichier iCalendar (.ics).
Compatible Google Calendar (abonnement URL ou import).
"""

import os
import sys
import argparse
import datetime
import getpass
from pathlib import Path
import pronotepy
import pronotepy.ent
import icalendar


def load_env_file(filepath: Path):
    """Charge les variables clé=valeur d'un fichier sans écraser os.environ déjà défini."""
    if not filepath.exists():
        return
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


def build_calendar_from_lessons(lessons, cal_name="Emploi du temps Pronote"):
    cal = icalendar.Calendar()
    cal.add("prodid", "-//Pronote to Google Calendar//FR")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", cal_name)
    cal.add("x-wr-timezone", "Europe/Paris")

    for lesson in lessons:
        event = icalendar.Event()
        
        # Titre (Matière)
        subject_name = lesson.subject.name if lesson.subject else "Cours"
        if lesson.canceled:
            summary = f"[ANNULÉ] {subject_name}"
        elif lesson.detention:
            summary = f"[RETENUE] {subject_name}"
        elif lesson.outing:
            summary = f"[SORTIE] {subject_name}"
        else:
            summary = subject_name
            
        event.add("summary", summary)
        
        # Horaires
        event.add("dtstart", lesson.start)
        event.add("dtend", lesson.end)
        event.add("dtstamp", datetime.datetime.now(datetime.timezone.utc))
        
        # Identifiant unique
        uid = f"pronote-{lesson.id}@{lesson.start.strftime('%Y%m%d')}"
        event.add("uid", uid)

        # Lieu / Salle
        if lesson.classroom:
            event.add("location", lesson.classroom)

        # Description détaillée
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

        if desc_lines:
            event.add("description", "\n".join(desc_lines))

        # Statut iCal
        if lesson.canceled:
            event.add("status", "CANCELLED")
        else:
            event.add("status", "CONFIRMED")

        cal.add_component(event)

    return cal


def main():
    parser = argparse.ArgumentParser(description="Synchronisation Pronote -> fichier .ics pour Google Calendar")
    parser.add_argument("--config", default="config.env", help="Fichier de configuration d'environnement")
    parser.add_argument("--output", "-o", default=None, help="Chemin du fichier .ics de sortie")
    parser.add_argument("--days-past", type=int, default=None, help="Nombre de jours dans le passé")
    parser.add_argument("--days-future", type=int, default=None, help="Nombre de jours dans le futur")
    args = parser.parse_args()

    # Charger le fichier config s'il existe
    cfg_path = Path(args.config)
    if cfg_path.exists():
        load_env_file(cfg_path)

    # Récupérer les paramètres
    username = os.environ.get("ENT_USERNAME")
    password = os.environ.get("ENT_PASSWORD")
    pronote_url = os.environ.get("PRONOTE_URL", "https://0750611g.index-education.net/pronote/parent.html")
    child_name = os.environ.get("CHILD_NAME", "")
    
    days_past = args.days_past if args.days_past is not None else int(os.environ.get("DAYS_PAST", "7"))
    days_future = args.days_future if args.days_future is not None else int(os.environ.get("DAYS_FUTURE", "28"))
    output_path = args.output or os.environ.get("OUTPUT_ICS_FILE", "pronote.ics")

    if not username:
        username = input("Identifiant ENT Paris Classe Numérique : ").strip()
    if not password:
        password = getpass.getpass("Mot de passe ENT : ")

    print(f"Connexion à Paris Classe Numérique pour l'élève '{child_name}'...")
    try:
        client = pronotepy.ParentClient(
            pronote_url=pronote_url,
            username=username,
            password=password,
            ent=pronotepy.ent.paris_classe_numerique
        )
    except Exception as e:
        print(f"Erreur de connexion : {e}", file=sys.stderr)
        sys.exit(1)

    if not client.logged_in:
        print("Erreur : connexion non authentifiée.", file=sys.stderr)
        sys.exit(1)

    # Sélectionner l'enfant si spécifié
    target_child = None
    if child_name:
        for c in client.children:
            if child_name.lower() in c.name.lower():
                target_child = c
                break
        if not target_child:
            print(f"Enfant '{child_name}' non trouvé. Enfants disponibles : {[c.name for c in client.children]}")
            target_child = client.children[0]
    else:
        target_child = client.children[0]

    client.set_child(target_child)
    print(f"Enfant sélectionné : {target_child.name}")

    # Calcul de la plage de dates
    today = datetime.date.today()
    date_from = today - datetime.timedelta(days=days_past)
    date_to = today + datetime.timedelta(days=days_future)
    print(f"Extraction des cours du {date_from.strftime('%d/%m/%Y')} au {date_to.strftime('%d/%m/%Y')}...")

    lessons = client.lessons(date_from, date_to)
    print(f"{len(lessons)} cours récupérés.")

    # Génération du calendrier ICS
    cal = build_calendar_from_lessons(lessons, cal_name=f"Pronote - {target_child.name}")
    
    out_file = Path(output_path)
    out_file.write_bytes(cal.to_ical())
    print(f"Fichier iCal généré avec succès : {out_file.resolve()}")
    print("Prêt pour importation ou abonnement dans Google Calendar !")


if __name__ == "__main__":
    main()
