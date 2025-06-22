import pygame
import random
import datetime
import uuid
import tkinter as tk
from tkinter import simpledialog, messagebox
import re
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

    # Board initialization and display
    initialiser_jeu()
    afficher_plateau()

    # Random choice of the player who starts
    game_state.joueur_courant = random.choice([1, 2])
    game_data.first_player = game_state.joueur_courant
    if game_state.joueur_courant == 1:
        afficher_message("The computer starts!")
    else:
        afficher_message("You start!")
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