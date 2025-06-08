import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
from connect4_robot_j4.game_state import GameState
from connect4_robot_j4.constants import MINIMAX_DEPTH

# 1. Initialization of Firebase Admin SDK
key_path = os.environ.get("FIREBASE_CRED")
cred = credentials.Certificate(str(key_path))
firebase_admin.initialize_app(cred)

# 2. Connection to Firestore
db = firestore.client()

def get_game_data(game_state: GameState):
    """
    Extracts game data from the GameState object.
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

def send_game_data(game_state: GameState):
    """
    Sends game data to the Firestore database.
    """
    game_data = get_game_data(game_state)
    
    # 3. Sending to the "games" collection
    db.collection("games").document(game_data["game_id"]).set(game_data)
    print("Game successfully sent to Firebase!")
