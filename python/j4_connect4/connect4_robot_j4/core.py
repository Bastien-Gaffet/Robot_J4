import pygame
import random
from connect4_robot_j4.game_state import GameState
from connect4_robot_j4.minimax.minimax_functions import(
    initialiser_jeu, 
    afficher_plateau, 
    afficher_message
)

def init_game():
    # Création de l'état du jeu
    game_state = GameState()
    game_state.game_over = False

    # Initialisation du plateau et affichage
    initialiser_jeu()
    afficher_plateau()

    # Choix al�atoire du joueur qui commence
    game_state.joueur_courant = random.choice([1, 2])
    if game_state.joueur_courant == 1:
        afficher_message("L'ordinateur commence!")
    else:
        afficher_message("Vous commencez!")
    pygame.time.delay(1000)

    return game_state