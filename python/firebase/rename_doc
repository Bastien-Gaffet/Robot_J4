import firebase_admin
from firebase_admin import credentials, firestore
import os

def initialize_firebase():
    """
    Initializes the Firebase Admin SDK and connects to Firestore.
    Returns a Firestore client if successful, otherwise None.
    """
    try:
        if "FIREBASE_CRED" not in os.environ:
            print("[Firebase] Environment variable FIREBASE_CRED is not set.")
            return None

        key_path = os.environ.get("FIREBASE_CRED")
        cred = credentials.Certificate(str(key_path))
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        print("[Firebase] Successfully connected to Firestore.")
        return db

    except Exception as e:
        print(f"[Firebase] Initialization failed: {e}")
        return None
    
def rename_document(db, collection_name, old_doc_id, new_doc_id):
    old_ref = db.collection(collection_name).document(old_doc_id)
    new_ref = db.collection(collection_name).document(new_doc_id)

    old_doc = old_ref.get()
    if not old_doc.exists:
        print(f"Document {old_doc_id} does not exist.")
        return False

    data = old_doc.to_dict()
    # Créer nouveau doc avec les données existantes
    new_ref.set(data)
    # Supprimer ancien doc
    old_ref.delete()

    print(f"Document '{old_doc_id}' renamed to '{new_doc_id}'.")
    return True

# Exemple d’utilisation :
db = initialize_firebase()
rename_document(db, "users", "IA", "AI")