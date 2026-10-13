#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pronotepy>=2.15.0",
#     "beautifulsoup4>=4.8.2",
#     "pycryptodome>=3.9.4",
#     "requests>=2.30.0",
# ]
# ///
"""
Test de connexion à Pronote via Paris Classe Numérique (standalone uv script).
"""
import sys
import getpass
import pronotepy
import pronotepy.ent

PRONOTE_URL = "https://0750611g.index-education.net/pronote/parent.html"

def test_connection():
    print(f"URL Pronote : {PRONOTE_URL}")
    print("ENT cible   : Paris Classe Numérique (paris_classe_numerique)\n")
    
    username = input("Identifiant ENT (Paris Classe Numérique) : ").strip()
    if not username:
        print("Erreur: identifiant vide.")
        sys.exit(1)
        
    password = getpass.getpass("Mot de passe ENT : ")
    if not password:
        print("Erreur: mot de passe vide.")
        sys.exit(1)

    print("\nConnexion en cours...")
    try:
        client = pronotepy.ParentClient(
            pronote_url=PRONOTE_URL,
            username=username,
            password=password,
            ent=pronotepy.ent.paris_classe_numerique
        )
    except Exception as e:
        print(f"\n[ÉCHEC] Erreur de connexion : {e}")
        sys.exit(1)

    if not client.logged_in:
        print("\n[ÉCHEC] Connexion échouée (identifiants ou ENT incorrects).")
        sys.exit(1)

    print("\n[SUCCÈS] Connecté avec succès !")
    print(f"Nombre d'enfants trouvés : {len(client.children)}")
    for i, child in enumerate(client.children, 1):
        print(f"  {i}. {child.name}")

    try:
        ical_url = client.export_ical()
        print(f"\nFlux iCal natif Pronote détecté : {ical_url}")
    except Exception:
        print("\nNote: L'export iCal direct Pronote n'est pas activé côté établissement (génération personnalisée requise).")

if __name__ == "__main__":
    test_connection()
