import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
from connect4_robot_j4 import GameData
import secrets, string

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


def get_user_doc_by_pseudo(db, pseudo):
    pseudo_lower = pseudo.lower()
    users_ref = db.collection("users")
    query = users_ref.where("pseudo_lower", "==", pseudo_lower).limit(1)
    docs = query.get()

    if docs:
        return docs[0].id, docs[0]
    else:
        return None, None

def expected_score(player_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

def k_factor(ai_level):
    """
    K varie entre 10 (niveau IA 1) et 40 (niveau IA 7) par exemple.
    Non linéaire : on prend racine carrée multipliée par une constante
    """
    base_k_min = 10
    base_k_max = 40
    # Racine carrée pour la non-linéarité
    k = base_k_min + (base_k_max - base_k_min) * ((ai_level / 7) ** 0.5)
    return k

def update_elo(player_elo, ai_elo, ai_level, player_result):
    """
    player_result: 1 = victoire joueur, 0 = défaite joueur
    """
    K = k_factor(ai_level)
    E_player = expected_score(player_elo, ai_elo)
    new_player_elo = player_elo + K * (player_result - E_player)
    new_player_elo = max(new_player_elo, 100)  # Elo plancher

    # Mise à jour Elo IA aussi (inverse)
    E_ai = expected_score(ai_elo, player_elo)
    ai_result = 1 - player_result
    new_ai_elo = ai_elo + K * (ai_result - E_ai)
    new_ai_elo = max(new_ai_elo, 100)

    return round(new_player_elo), round(new_ai_elo)

def generate_claim_token():
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))

def get_game_data(game_data: GameData):
    """
    Extracts game data from the GameData object.
    """
    return {
        "game_id": game_data.game_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "duration_seconds": (game_data.game_end_time - game_data.game_start_time).total_seconds(),
        "first_player_color": game_data.first_player_color,
        "moves": game_data.moves,
        "result": game_data.result,
        "player1": game_data.player1,
        "player2": game_data.player2,
        "player_pseudo": game_data.player_pseudo,
    }

def get_players_data(game_data, db, ai_depth):
    """
    Fetches user and AI data from Firestore based on game data.
    Uses pseudo-based document lookup for the human player (not UID).
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    player_pseudo = game_data["player_pseudo"]
    ai_pseudo = f"AI ({ai_depth})"
    ai_uid = "AI"  # fixed document ID for the AI

    # Get player UID and document using their pseudo
    player_uid, player_doc_snapshot = get_user_doc_by_pseudo(db, player_pseudo)
    if player_uid is None:
        raise ValueError(f"Player with pseudo '{player_pseudo}' not found in Firestore.")

    player_ref = db.collection("users").document(player_uid)
    ai_ref = db.collection("users").document(ai_uid)

    # Get latest document snapshots
    player_doc = player_doc_snapshot
    ai_doc = ai_ref.get()

    # Retrieve Elo ratings
    player_elo = player_doc.to_dict().get("elo", 500) if player_doc.exists else 500
    ai_elo = ai_doc.to_dict().get("elo", 500) if ai_doc.exists else 500

    # Determine match outcome
    result = game_data["result"]  # should match player_pseudo, AI (X), or None
    if result == player_pseudo:
        player_result = 1
    elif result == ai_uid:
        player_result = 0
    else:
        player_result = 0.5  # draw

    # Update Elo ratings
    new_player_elo, new_ai_elo = update_elo(player_elo, ai_elo, ai_depth, player_result)

    return {
        "timestamp": timestamp,
        "player": {
            "pseudo": player_pseudo,
            "ref": player_ref,
            "doc": player_doc,
            "elo_before": player_elo,
            "elo_after": new_player_elo,
            "elo_entry": {
                "game_id": game_data["game_id"],
                "timestamp": timestamp,
                "elo": new_player_elo
            }
        },
        "ai": {
            "pseudo": ai_pseudo,
            "ref": ai_ref,
            "doc": ai_doc,
            "elo_before": ai_elo,
            "elo_after": new_ai_elo,
            "elo_entry": {
                "game_id": game_data["game_id"],
                "timestamp": timestamp,
                "elo": new_ai_elo
            }
        }
    }


def send_game_data(game_state: GameData, db):
    """
    Sends a completed Connect Four game to Firestore and updates player stats.
    Supports both PvP (player vs player) and PvAI (player vs AI) modes.
    """
    if db is None:
        print("[Firebase] No database connection. Game data not sent.")
        return

    try:
        # Extract game data from game state
        game_data = get_game_data(game_state)
        game_id = game_data["game_id"]

        # Fetch all player-related Firestore documents and data
        data = get_players_data(game_data, db, game_state.ai_depth)
        timestamp = data["timestamp"]

        # Determine result of the game
        result = game_data["result"]  # "draw" or player1_pseudo / player2_pseudo
        player1 = data["player"]
        player2 = data["ai"]

        if result == player1["pseudo"]:
            player_result = 1
        elif result == player2["pseudo"]:
            player_result = 0
        else:
            player_result = 0.5

        # Prepare stats updates
        if player_result == 1:
            p1_update = {"wins": firestore.Increment(1)}
            p2_update = {"losses": firestore.Increment(1)}
        elif player_result == 0:
            p1_update = {"losses": firestore.Increment(1)}
            p2_update = {"wins": firestore.Increment(1)}
        else:
            p1_update = {"draws": firestore.Increment(1)}
            p2_update = {"draws": firestore.Increment(1)}

        # ✅ Write the game document
        game_data["timestamp"] = timestamp  # Use unified timestamp
        db.collection("games").document(game_id).set(game_data)
        print(f"[Firebase] Game {game_id} successfully sent to Firestore.")

        # ✅ Update Player 1 (the human user who initiated the game)
        if player1["doc"].exists:
            player1["ref"].update({
                "elo": player1["elo_after"],
                "elo_history": firestore.ArrayUnion([player1["elo_entry"]]),
                **p1_update
            })
        else:
            start_entry = {
                "game_id": "initial",
                "timestamp": timestamp - datetime.timedelta(seconds=1),
                "elo": 500
            }
            player1["ref"].set({
                "pseudo": player1["pseudo"],
                "pseudo_lower": player1["pseudo"].lower(),
                "elo": player1["elo_after"],
                "elo_history": [start_entry, player1["elo_entry"]],
                "wins": 1 if player_result == 1 else 0,
                "losses": 1 if player_result == 0 else 0,
                "draws": 1 if player_result == 0.5 else 0,
                "claimed": False,
                "claim_token": generate_claim_token()
            })
            print(f"[Token] Claim token for new user '{player1['pseudo']}': {claim_token}")

        # Update Player 2 (either another human or the AI)
        if player2["doc"].exists:
            player2["ref"].update({
                "elo": player2["elo_after"],
                "elo_history": firestore.ArrayUnion([player2["elo_entry"]]),
                **p2_update
            })
        else:
            start_entry2 = {
                "game_id": "initial",
                "timestamp": timestamp - datetime.timedelta(seconds=1),
                "elo": 500
            }

            opponent_doc = {
                "pseudo": player2["pseudo"],
                "elo": player2["elo_after"],
                "elo_history": [start_entry2, player2["elo_entry"]],
                "wins": 1 if player_result == 0 else 0,
                "losses": 1 if player_result == 1 else 0,
                "draws": 1 if player_result == 0.5 else 0
            }

            if not game_state.opponent_is_ai:
                claim_token = generate_claim_token()
                opponent_doc.update({
                    "pseudo_lower": player2["pseudo"].lower(),
                    "claimed": False,
                    "claim_token": claim_token
                })

                # Show the token in the console
                print(f"[Token] Claim token for new user '{player2['pseudo']}': {claim_token}")

            player2["ref"].set(opponent_doc)

        # Elo logs
        print(f"[Elo] {player1['pseudo']}: {player1['elo_before']} → {player1['elo_after']}")
        print(f"[Elo] {player2['pseudo']}: {player2['elo_before']} → {player2['elo_after']}")

    except Exception as e:
        print(f"[Firebase] Failed to send game data: {e}")
