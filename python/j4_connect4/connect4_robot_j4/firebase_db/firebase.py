import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
from connect4_robot_j4 import GameData
from connect4_robot_j4.constants import MINIMAX_DEPTH

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

def get_game_data(game_state: GameData):
    """
    Extracts game data from the GameData object.
    """
    return {
        "game_id": game_state.game_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_seconds": (game_state.game_end_time - game_state.game_start_time).total_seconds(),
        "first_player": game_state.first_player,
        "moves": game_state.moves,
        "winner": game_state.winner,
        "player_pseudo": game_state.player_pseudo,
        "ai_depth": MINIMAX_DEPTH
    }

def send_game_data(game_state: GameData, db):
    """
    Sends game data to the Firestore database.
    """
    if db is None:
        print("[Firebase] No database connection. Game data not sent.")
        return

    try:
        game_data = get_game_data(game_state)
        db.collection("games").document(game_data["game_id"]).set(game_data)
        print("[Firebase] Game successfully sent to Firestore.")
    except Exception as e:
        print(f"[Firebase] Failed to send game data: {e}")
