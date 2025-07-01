import firebase_admin
from firebase_admin import credentials, firestore
import os
import secrets
import string

def generate_claim_token():
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))

def initialize_firebase():
    if "FIREBASE_CRED" not in os.environ:
        print("FIREBASE_CRED non défini.")
        return None
    cred_path = os.environ["FIREBASE_CRED"]
    cred = credentials.Certificate(cred_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

def update_missing_user_fields(db):
    users_ref = db.collection("users")
    users = users_ref.stream()

    updated_count = 0

    for user_doc in users:
        user_data = user_doc.to_dict()
        updates = {}

        if "pseudo" not in user_data:
            print(f"Utilisateur sans pseudo (ID: {user_doc.id}), ignoré.")
            continue

        if "pseudo_lower" not in user_data:
            updates["pseudo_lower"] = user_data["pseudo"].lower()

        if "claim_token" not in user_data:
            updates["claim_token"] = generate_claim_token()

        if "claimed" not in user_data:
            updates["claimed"] = False

        if updates:
            print(f"[Update] Mise à jour de {user_data['pseudo']} ({user_doc.id}): {updates}")
            users_ref.document(user_doc.id).update(updates)
            updated_count += 1

    print(f"[OK] {updated_count} utilisateur(s) mis à jour.")

if __name__ == "__main__":
    db = initialize_firebase()
    if db:
        update_missing_user_fields(db)
