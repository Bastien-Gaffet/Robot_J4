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


def get_game_data(game_state: GameData):
    """
    Extracts game data from the GameData object.
    """
    return {
        "game_id": game_state.game_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "duration_seconds": (game_state.game_end_time - game_state.game_start_time).total_seconds(),
        "first_player": game_state.first_player,
        "moves": game_state.moves,
        "winner": game_state.winner,
        "player_pseudo": game_state.player_pseudo,
        "ai_depth": MINIMAX_DEPTH
    }

def get_players_data(game_data, db):
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    player_pseudo = game_data["player_pseudo"]
    ai_pseudo = "AI"
    ai_depth = game_data["ai_depth"]  # Niveau IA entre 1 et 7
    winner = game_data["winner"]  # Exemple: "AI (Red)" ou "Player (Yellow)"

    player_ref = db.collection("users").document(player_pseudo)
    ai_ref = db.collection("users").document(ai_pseudo)

    player_doc = player_ref.get()
    ai_doc = ai_ref.get()

    player_elo = player_doc.to_dict().get("elo", 500) if player_doc.exists else 500
    ai_elo = ai_doc.to_dict().get("elo", 500) if ai_doc.exists else 500

    # Gérer le résultat pour le calcul Elo
    if "Player" in winner:
        player_result = 1  # victoire joueur
    elif "AI" in winner:
        player_result = 0  # défaite joueur
    else:
        # Match nul
        player_result = 0.5

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
    if db is None:
        print("[Firebase] No database connection. Game data not sent.")
        return

    try:
        game_data = get_game_data(game_state)
        game_id = game_data["game_id"]

        #Récupération centralisée des données joueurs + IA
        data = get_players_data(game_data, db)
        timestamp = data["timestamp"]

        # 🔢 Determine game outcome for stats update
        player_result = 1 if "Player" in game_data["winner"] else 0 if "AI" in game_data["winner"] else 0.5

        if player_result == 1:
            player_update_stats = {"wins": firestore.Increment(1)}
            ai_update_stats = {"losses": firestore.Increment(1)}
        elif player_result == 0:
            player_update_stats = {"losses": firestore.Increment(1)}
            ai_update_stats = {"wins": firestore.Increment(1)}
        else:
            player_update_stats = {"draws": firestore.Increment(1)}
            ai_update_stats = {"draws": firestore.Increment(1)}
            
        # Envoi de la partie (timestamp natif Firestore)
        game_data["timestamp"] = timestamp
        db.collection("games").document(game_id).set(game_data)
        print(f"[Firebase] Game {game_id} successfully sent to Firestore.")

        #Mise à jour joueur
        player = data["player"]
        if player["doc"].exists:
            player["ref"].update({
                "elo": player["elo_after"],
                "elo_history": firestore.ArrayUnion([player["elo_entry"]]),
                **player_update_stats
            })
        else:
            start_entry = {
                "game_id": "initial",
                "timestamp": timestamp - datetime.timedelta(seconds=1),  # juste avant la partie
                "elo": 500
            }

            player["ref"].set({
                "pseudo": player["pseudo"],
                "elo": player["elo_after"],
                "elo_history": [start_entry, player["elo_entry"]],
                "wins": 1 if player_result == 1 else 0,
                "losses": 1 if player_result == 0 else 0,
                "draws": 1 if player_result == 0.5 else 0
            })

        # 🔄 Mise à jour IA
        ai = data["ai"]
        if ai["doc"].exists:
            ai["ref"].update({
                "elo": ai["elo_after"],
                "elo_history": firestore.ArrayUnion([ai["elo_entry"]]),
                **ai_update_stats
            })
        else:
            start_entry_ai = {
                "game_id": "initial",
                "timestamp": timestamp - datetime.timedelta(seconds=1),
                "elo": 500
            }

            ai["ref"].set({
                "pseudo": ai["pseudo"],
                "elo": ai["elo_after"],
                "elo_history": [start_entry_ai, ai["elo_entry"]],
                "wins": 1 if player_result == 0 else 0,
                "losses": 1 if player_result == 1 else 0,
                "draws": 1 if player_result == 0.5 else 0
            })

        # ✅ Log
        print(f"[Elo] {player['pseudo']}: {player['elo_before']} -> {player['elo_after']}")
        print(f"[Elo] {ai['pseudo']}: {ai['elo_before']} -> {ai['elo_after']}")

    except Exception as e:
        print(f"[Firebase] Failed to send game data: {e}")