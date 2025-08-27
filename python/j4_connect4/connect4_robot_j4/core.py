import pygame
import random
import datetime
import uuid
import tkinter as tk
from tkinter import simpledialog, messagebox
import re
from connect4_robot_j4.constants import MINIMAX_DEPTH
from connect4_robot_j4 import GameState
from connect4_robot_j4 import GameData
from connect4_robot_j4.minimax import(
    initialiser_jeu,
    afficher_plateau,
    afficher_message
)

def init_game():
    # Creation of the game state
    game_state = GameState()
    game_data = GameData()
    game_data.game_start_time = datetime.datetime.now()
    game_data.game_id = str(uuid.uuid4())  # Unique game ID
    game_data.player_pseudo = ask_pseudo()
    game_data.ai_depth = MINIMAX_DEPTH

    # Board initialization and display
    initialiser_jeu()
    afficher_plateau()

    # Random choice of the player who starts
    game_state.joueur_courant = random.choice([1, 2])
    if game_state.joueur_courant == 1:
        afficher_message("The computer starts!")
        game_data.first_player_color = "red"
        game_data.player1 = "AI"
        game_data.player2 = game_data.player_pseudo
    else:
        afficher_message("You start!")
        game_data.first_player_color = "yellow"
        game_data.player1 = game_data.player_pseudo
        game_data.player2 = "AI"
    pygame.time.delay(1000)

    return game_state, game_data

def is_valid_pseudo(pseudo):
    # Allows letters with accents, numbers, and spaces, max 16 characters
    if len(pseudo) > 16:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9À-ÿ\s]+", pseudo))

def ask_pseudo():
    root = tk.Tk()
    root.withdraw()

    while True:
        pseudo = simpledialog.askstring("Name or Nickname", "What is your name or nickname? (max 16 characters, letters/numbers only)")
        
        if not pseudo:
            pseudo = "Player1"
            break
        
        pseudo = pseudo.strip()  # remove extra spaces

        if is_valid_pseudo(pseudo):
            break
        else:
            messagebox.showerror("Error", "Invalid name.\nOnly letters (including accents), numbers, and spaces are allowed.\nMaximum 16 characters.")

    root.destroy()
    return pseudo