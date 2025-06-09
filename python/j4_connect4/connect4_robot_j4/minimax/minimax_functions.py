import numpy as np
import random
import time
import math
import pygame
from connect4_robot_j4.arduino_serial import send_to_arduino
from connect4_robot_j4.arduino_serial import serial_obj
import time
from connect4_robot_j4.constants import MINIMAX_DEPTH

# Constantes globales
TAILLE_CASE = 80
LARGEUR = 7 * TAILLE_CASE
HAUTEUR = (6 + 1) * TAILLE_CASE  # +1 pour la zone de sélection de colonne
RAYON = int(TAILLE_CASE/2 - 5)

# Couleurs
BLEU = (0, 0, 255)
NOIR = (0, 0, 0)
ROUGE = (255, 0, 0)
JAUNE = (255, 255, 0)
BLANC = (255, 255, 255)

# Variables globales
plateau = None
screen = None
font = None
tour = 0

ia_a_joue = False  # Pour suivre si l'IA a déjà joué son coup dans ce tour
dernier_coup_ia = None  # Pour stocker le dernier coup calculé par l'IA
en_attente_detection = False  # Pour suivre si l'IA attend que son coup soit détecté
joueur_courant = 0

def initialiser_jeu():
    """Initialise le jeu et retourne le plateau"""
    global plateau, screen, font, tour

    # Initialisation du plateau 6 lignes x 7 colonnes (vide = 0, joueur = 1, ordinateur = 2)
    plateau = np.zeros((6, 7), dtype=int)
    tour = 0

    # Initialisation de Pygame
    pygame.init()

    # Création de la fenêtre
    screen = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption('Puissance 4 avec IA')

    # Initialisation de la police pour le texte
    font = pygame.font.SysFont('monospace', 20)

    # Premier affichage du plateau
    afficher_plateau()

    return plateau

def afficher_plateau():
    """Affiche le plateau de jeu avec Pygame"""
    # Effacer l'écran avec du blanc
    screen.fill(BLANC)

    # Zone de sélection (rangée supérieure)
    for col in range(7):
        pygame.draw.rect(screen, BLANC, (col * TAILLE_CASE, 0, TAILLE_CASE, TAILLE_CASE))

    # Dessiner la grille de jeu (fond bleu)
    pygame.draw.rect(screen, BLEU, (0, TAILLE_CASE, LARGEUR, 6 * TAILLE_CASE))

    # Dessiner les emplacements du jeu
    for col in range(7):
        for ligne in range(6):
            # Calculer la position pour chaque emplacement
            posX = col * TAILLE_CASE + TAILLE_CASE//2
            posY = (ligne + 1) * TAILLE_CASE + TAILLE_CASE//2

            # Choisir la couleur en fonction de l'état de la case
            couleur = BLANC  # Case vide
            if plateau[ligne][col] == 1:
                couleur = ROUGE  # Joueur
            elif plateau[ligne][col] == 2:
                couleur = JAUNE  # Ordinateur

            # Dessiner le cercle
            pygame.draw.circle(screen, couleur, (posX, posY), RAYON)

    # Afficher le numéro des colonnes
    for col in range(7):
        texte = font.render(str(col + 1), True, NOIR)
        screen.blit(texte, (col * TAILLE_CASE + TAILLE_CASE//2 - 5, 10))

    # Mettre à jour l'affichage
    pygame.display.update()

def coup_valide(colonne):
    """Vérifie si un coup est valide dans la colonne spécifiée"""
    if colonne < 0 or colonne > 6:
        return False
    # Vérifier si la colonne n'est pas pleine
    return plateau[0][colonne] == 0

def placer_jeton(colonne, joueur):
    """Place un jeton dans la colonne spécifiée et retourne la ligne"""
    global tour

    # Animation de chute du jeton
    if joueur == 1:
        couleur_jeton = ROUGE
    else:
        couleur_jeton = JAUNE

    # Trouver la première position libre (du bas vers le haut)
    for ligne in range(5, -1, -1):
        if plateau[ligne][colonne] == 0:
            # Animation de chute
            for l_anim in range(0, ligne + 1):
                # Redessiner le plateau
                afficher_plateau()

                # Dessiner le jeton en mouvement
                posX = colonne * TAILLE_CASE + TAILLE_CASE//2
                posY = l_anim * TAILLE_CASE + TAILLE_CASE + TAILLE_CASE//2
                pygame.draw.circle(screen, couleur_jeton, (posX, posY), RAYON)
                pygame.display.update()

                # Pause courte pour l'animation
                pygame.time.delay(50)

            # Placer finalement le jeton
            plateau[ligne][colonne] = joueur
            tour += 1
            afficher_plateau()
            return ligne
    return -1  # Erreur, la colonne est pleine

def verifier_victoire(joueur):
    """Vérifie s'il y a une victoire pour le joueur spécifié"""
    # Vérifier les lignes horizontales
    for ligne in range(6):
        for col in range(4):
            if (plateau[ligne][col] == joueur and
                plateau[ligne][col+1] == joueur and
                plateau[ligne][col+2] == joueur and
                plateau[ligne][col+3] == joueur):
                return True

    # Vérifier les lignes verticales
    for ligne in range(3):
        for col in range(7):
            if (plateau[ligne][col] == joueur and
                plateau[ligne+1][col] == joueur and
                plateau[ligne+2][col] == joueur and
                plateau[ligne+3][col] == joueur):
                return True

    # Vérifier les diagonales montantes
    for ligne in range(3, 6):
        for col in range(4):
            if (plateau[ligne][col] == joueur and
                plateau[ligne-1][col+1] == joueur and
                plateau[ligne-2][col+2] == joueur and
                plateau[ligne-3][col+3] == joueur):
                return True

    # Vérifier les diagonales descendantes
    for ligne in range(3):
        for col in range(4):
            if (plateau[ligne][col] == joueur and
                plateau[ligne+1][col+1] == joueur and
                plateau[ligne+2][col+2] == joueur and
                plateau[ligne+3][col+3] == joueur):
                return True

    return False

def plateau_plein():
    """Vérifie si le plateau est plein (match nul)"""
    return tour >= 42 or np.all(plateau != 0)

def evaluer_fenetre(fenetre, joueur):
    """Évalue une fenêtre de 4 positions"""
    score = 0
    adversaire = 1 if joueur == 2 else 2

    if fenetre.count(joueur) == 4:
        score += 100
    elif fenetre.count(joueur) == 3 and fenetre.count(0) == 1:
        score += 5
    elif fenetre.count(joueur) == 2 and fenetre.count(0) == 2:
        score += 2

    if fenetre.count(adversaire) == 3 and fenetre.count(0) == 1:
        score -= 4

    return score

def evaluer_position(joueur):
    """Évalue la position actuelle du plateau pour l'algorithme minimax"""
    score = 0

    # Score pour le centre (préférer le centre)
    centre_col = 3
    centre_array = [int(plateau[i][centre_col]) for i in range(6)]
    score += centre_array.count(joueur) * 3

    # Score pour les lignes horizontales
    for ligne in range(6):
        for col in range(4):
            fenetre = [int(plateau[ligne][col+i]) for i in range(4)]
            score += evaluer_fenetre(fenetre, joueur)

    # Score pour les lignes verticales
    for ligne in range(3):
        for col in range(7):
            fenetre = [int(plateau[ligne+i][col]) for i in range(4)]
            score += evaluer_fenetre(fenetre, joueur)

    # Score pour les diagonales montantes
    for ligne in range(3, 6):
        for col in range(4):
            fenetre = [int(plateau[ligne-i][col+i]) for i in range(4)]
            score += evaluer_fenetre(fenetre, joueur)

    # Score pour les diagonales descendantes
    for ligne in range(3):
        for col in range(4):
            fenetre = [int(plateau[ligne+i][col+i]) for i in range(4)]
            score += evaluer_fenetre(fenetre, joueur)

    return score

def est_position_terminale():
    """Vérifie si la position est terminale (victoire ou match nul)"""
    return verifier_victoire(1) or verifier_victoire(2) or plateau_plein()

def minimax(profondeur, alpha, beta, maximizing):
    """Algorithme Minimax avec élagage alpha-beta"""
    # Si on a atteint la profondeur maximum ou que la partie est terminée
    if profondeur == 0 or est_position_terminale():
        if est_position_terminale():
            if verifier_victoire(2):  # IA gagne
                return (None, 1000000)
            elif verifier_victoire(1):  # Joueur gagne
                return (None, -1000000)
            else:  # Match nul
                return (None, 0)
        else:  # Évaluation heuristique
            return (None, evaluer_position(2))

    # Obtenir les colonnes jouables
    coups_valides = [col for col in range(7) if coup_valide(col)]

    # Trier les colonnes (préférer le centre)
    coups_valides.sort(key=lambda x: abs(x-3))

    if maximizing:
        valeur = -math.inf
        colonne = random.choice(coups_valides) if coups_valides else None

        for col in coups_valides:
            # Simuler le coup
            ligne = -1
            for l in range(5, -1, -1):
                if plateau[l][col] == 0:
                    ligne = l
                    plateau[l][col] = 2  # IA joue
                    break

            # Calculer le score avec récursion
            nouveau_score = minimax(profondeur-1, alpha, beta, False)[1]

            # Annuler le coup
            plateau[ligne][col] = 0

            # Mettre à jour la meilleure valeur
            if nouveau_score > valeur:
                valeur = nouveau_score
                colonne = col

            # Élagage alpha-beta
            alpha = max(alpha, valeur)
            if alpha >= beta:
                break

        return colonne, valeur

    else:  # Minimizing
        valeur = math.inf
        colonne = random.choice(coups_valides) if coups_valides else None

        for col in coups_valides:
            # Simuler le coup
            ligne = -1
            for l in range(5, -1, -1):
                if plateau[l][col] == 0:
                    ligne = l
                    plateau[l][col] = 1  # Joueur joue
                    break

            # Calculer le score avec récursion
            nouveau_score = minimax(profondeur-1, alpha, beta, True)[1]

            # Annuler le coup
            plateau[ligne][col] = 0

            # Mettre à jour la meilleure valeur
            if nouveau_score < valeur:
                valeur = nouveau_score
                colonne = col

            # Élagage alpha-beta
            beta = min(beta, valeur)
            if alpha >= beta:
                break

        return colonne, valeur

def afficher_message(message):
    """Affiche un message au-dessus du plateau"""
    # Effacer la zone de message (partie supérieure de l'écran)
    pygame.draw.rect(screen, BLANC, (0, 0, LARGEUR, TAILLE_CASE))

    # Redessiner les numéros de colonnes
    for col in range(7):
        texte = font.render(str(col + 1), True, NOIR)
        screen.blit(texte, (col * TAILLE_CASE + TAILLE_CASE//2 - 5, 10))

    # Afficher le nouveau message
    text_surface = font.render(message, True, NOIR)
    screen.blit(text_surface, (10, 40))  # Positionner plus bas pour éviter les numéros de colonne
    pygame.display.update()

def tour_ordinateur(game_state):
    """Utilise l'algorithme Minimax pour que l'ordinateur joue"""
    # Si l'IA a déjà joué et on attend la détection, ne pas recalculer
    if game_state.ia_a_joue and game_state.en_attente_detection:
        return False

    # Si l'IA a déjà joué pour ce tour, ne rien faire
    if game_state.ia_a_joue:
        return False

    # Rafraîchir l'affichage avant de commencer le calcul
    afficher_message("L'ordinateur réfléchit...")
    pygame.display.update()  # Forcer la mise à jour de l'affichage

    # Permettre à la caméra de se mettre à jour pendant que l'IA réfléchit
    pygame.event.pump()  # Traiter les événements en attente pour éviter que l'interface ne se bloque

    start_time = time.time()

    # Ajuster la profondeur en fonction du nombre de coups joués
    profondeur = min(MINIMAX_DEPTH, 42 - tour)

    # Utiliser l'algorithme minimax
    colonne, score = minimax(profondeur, -math.inf, math.inf, True)

    # S'assurer qu'un coup valide est retourné
    if colonne is None or not coup_valide(colonne):
        colonnes_valides = [c for c in range(7) if coup_valide(c)]
        if colonnes_valides:
            colonne = random.choice(colonnes_valides)
        else:
            return False  # Match nul, aucun coup possible

    elapsed_time = time.time() - start_time
    afficher_message(f"L'ordinateur a choisi la colonne {colonne + 1} (en {elapsed_time:.2f} secondes)")
    entree = colonne + 1

    # Send message to place the AI's token on the board
    send_to_arduino(serial_obj, entree)

    pygame.time.delay(500)

    # Marquer que l'IA a joué et on attend la détection
    game_state.ia_a_joue = True
    game_state.en_attente_detection = True
    game_state.dernier_coup_ia = colonne

    # Placer le jeton physiquement (message pour l'utilisateur)
    print("IA joue en colonne", colonne + 1)

    # Rafraîchir l'affichage une dernière fois
    pygame.display.update()

    return True

def confirmer_coup_ia(game_state):
    """À appeler quand le coup de l'IA a été détecté par la caméra"""
    game_state.ia_a_joue = False
    game_state.en_attente_detection = False

def verifier_coup_ia(colonne_detectee, game_state):
    """Vérifie si le coup détecté correspond au dernier coup calculé par l'IA"""
    return game_state.dernier_coup_ia == colonne_detectee

def time_to_play(game_state):
    """Gère le tour de jeu en fonction du joueur courant"""
    # Si le jeu est terminé, ne rien faire
    if game_state.game_over:
        return

    # Tour de l'IA (joueur 1 = rouge)
    if game_state.joueur_courant == 1:
        afficher_message("C'est au tour de l'IA de jouer")
        # Forcer la mise à jour de l'affichage avant que l'IA ne joue
        pygame.display.update()
        # Laisser le temps pour la mise à jour de l'interface
        pygame.time.delay(100)
        # Faire jouer l'IA
        tour_ordinateur(game_state)
        print("")

    # Tour du joueur humain (joueur 2 = jaune)
    elif game_state.joueur_courant == 2:
        afficher_message("Votre tour! C'est à vous de placer un jeton")
        # Forcer la mise à jour de l'affichage
        pygame.display.update()

    # Rafraîchir l'affichage une dernière fois
    pygame.display.update()