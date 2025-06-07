import pygame
import random
import datetime
import uuid
from connect4_robot_j4.game_state import GameState
from connect4_robot_j4.minimax.minimax_functions import(
    initialiser_jeu, 
    afficher_plateau, 
    afficher_message
)

def init_game():
    # Creation of the game state
    game_state = GameState()
    game_state.game_start_time = datetime.datetime.now()
    game_state.game_id = str(uuid.uuid4())  # Unique game ID based on timestamp

    # Board initialization and display
    initialiser_jeu()
    afficher_plateau()

    # Random choice of the player who starts
    game_state.joueur_courant = random.choice([1, 2])
    game_state.first_player = game_state.joueur_courant
    if game_state.joueur_courant == 1:
        afficher_message("The computer starts!")
    else:
        afficher_message("You start!")
    pygame.time.delay(1000)

    return game_state
