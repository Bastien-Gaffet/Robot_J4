import cv2
import numpy as np
import math
import time
import pygame
import sys
import random
from collections import Counter
from minimax_fonctions import *
import serial.tools.list_ports
import serial
import time

def detect_arduino():
    """
    Detect Arduino port with improved error handling and multiple identification methods.

    Returns:
    - Port name if Arduino is found
    - None if no Arduino is detected
    """
    ports = list(serial.tools.list_ports.comports())

    # Extended port identification methods
    arduino_keywords = [
        "Arduino", "CH340", "USB Serial",
        "Silicon Labs", "CP210x", "FTDI"
    ]

    for port in ports:
        # Check for keywords in description or hardware ID
        for keyword in arduino_keywords:
            if (keyword.lower() in str(port.description).lower() or
                keyword.lower() in str(port.hwid).lower()):
                return port.device

    return None

def setup_arduino_connection():
    """
    Set up Arduino connection with robust error handling.

    Returns:
    - SerialObj if connection successful
    - None if connection fails
    """
    arduino_port = detect_arduino()

    if not arduino_port:
        print("No Arduino detected.")
        return None

    try:
        SerialObj = serial.Serial(
            port=arduino_port,
            baudrate=9600,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1
        )
        print(f"Arduino connected successfully on {arduino_port}")
        return SerialObj

    except serial.SerialException:
        # Silently fail and return None without error messages
        return None
    except Exception:
        # Silently fail for other exceptions too
        return None

# Global variable to store serial connection
SerialObj = setup_arduino_connection()

# Définition des couleurs HSV avec des seuils améliorés
LOWER_RED1 = np.array([0, 100, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 100, 80])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_YELLOW1 = np.array([15, 100, 100])
UPPER_YELLOW1 = np.array([30, 255, 255])
LOWER_YELLOW2 = np.array([30, 100, 100])
UPPER_YELLOW2 = np.array([45, 255, 255])

LOWER_YELLOW3 = np.array([29, 200, 130])
UPPER_YELLOW3 = np.array([36, 255, 200])

LOWER_YELLOW4 = np.array([25, 200, 110])
UPPER_YELLOW4 = np.array([33, 255, 200])

# Constantes pour le traitement d'image
KERNEL = np.ones((7, 7), np.uint8)

# Paramètres du plateau
ROWS, COLS = 6, 7
ROI_X, ROI_Y, ROI_W, ROI_H = 50, 50, 500, 400
MIN_AREA = 300
MAX_AREA = 3000
MIN_CIRCULARITY = 0.6

# Paramètres de stabilisation
BUFFER_SIZE = 20
DETECTION_THRESHOLD = 0.6
SETTLING_TIME = 1.5  # Temps d'attente en secondes après un changement
GRID_UPDATE_INTERVAL = 0.5  # Intervalle en secondes pour mise à jour

# État global du jeu - variables nécessairement globales
last_red_move_matrix = None
last_yellow_move_matrix = None
grid_buffer = []

class GameState:
    def __init__(self):
        self.initialization_phase = True
        self.last_stable_grid = None
        self.last_stable_matrix = None
        self.current_matrix = None
        self.last_print_time = 0
        self.last_stabilization_time = 0
        self.last_grid_update_time = 0
        self.stabilized_matrix = None
        self.grid_changed = False
        self.last_change_time = 0
        # Variables de jeu
        self.joueur_courant = 1  # Initialement, le joueur 1 commence
        self.ia_a_joue = False
        self.en_attente_detection = False
        self.dernier_coup_ia = None
        self.game_over = False

def update_from_camera(current_matrix, previous_matrix, game_state):
    global SerialObj
    """Met à jour le plateau avec les données de la caméra et gère la logique du jeu."""
    # Phase d'initialisation
    if game_state.initialization_phase:
        if all(all(cell == 0 for cell in row) for row in current_matrix):
            print("Initialization successful - empty grid confirmed")
            print("Initial empty matrix:")
            for row in current_matrix:
                print(row)
            print("---------------------")
            game_state.last_stable_matrix = [row[:] for row in current_matrix]
            game_state.last_change_time = time.time()

            if camera.SerialObj is not None:
                print("Connexion série établie avec succès")
                camera.SerialObj.write(f"{game_state.joueur_courant + 7}\n".encode())
            else:
                print("Erreur de connexion Arduino.")

            if hasattr(camera, 'SerialObj') and camera.SerialObj is not None:
                print("le premier joueur est ", game_state.joueur_courant)
                camera.SerialObj.write(f"{game_state.joueur_courant + 7}\n".encode())

            if hasattr(camera, 'SerialObj') and game_state.joueur_courant == 1:
                print("mouvement initialisé")
                camera.SerialObj.write(f"12\n".encode())

                # Si l'IA commence, faire jouer son premier coup
                if game_state.joueur_courant == 1:
                    time_to_play(game_state)

            game_state.initialization_phase = False
        else:
            print("Waiting for empty grid to start game...")
        return False

    # Vérifier si le mouvement est valide selon les règles du jeu
    is_valid, player, column = is_valid_game_move(current_matrix, previous_matrix, game_state)

    if not is_valid:
        return False

    # Si en attente de détection d'un coup IA et que c'est bien l'IA qui a joué
    if game_state.en_attente_detection and player == 1:
        confirmer_coup_ia(game_state)
        print(f"Coup de l'IA en colonne {column + 1} détecté")

    # Mettre à jour l'affichage de pygame
    ligne = placer_jeton(column, player)
    pygame.display.update()

    # Vérifier s'il y a victoire
    if verifier_victoire(player):
        game_state.game_over = True
        message = "Félicitations! Vous avez gagné!" if player == 2 else "L'ordinateur a gagné!"
        afficher_message(message)
        if hasattr(sys.modules[__name__], 'SerialObj') and player == 2:
            camera.SerialObj.write(f"22\n".encode())
        else:
            camera.SerialObj.write(f"21\n".encode())
        pygame.time.delay(3000)
    elif plateau_plein():
        game_state.game_over = True
        afficher_message("Match nul!")
        camera.SerialObj.write(f"20\n".encode())
        pygame.time.delay(3000)
    else:
        # Alternance entre les joueurs (1 → 2, 2 → 1)
        game_state.joueur_courant = 3 - player
        print(f"Tour du joueur {game_state.joueur_courant}")


        if hasattr(sys.modules[__name__], 'SerialObj'):
            print(f"Envoi au port série: joueur {game_state.joueur_courant + 7}")
            camera.SerialObj.write(f"{game_state.joueur_courant + 7}\n".encode())
        if game_state.joueur_courant == 1:
            camera.SerialObj.write(f"12\n".encode())
        else:
            print("impossible d'envoyer la donnée")

        # Rafraîchir l'affichage avant de continuer
        pygame.display.update()

        # Si c'est au tour de l'IA, faire jouer l'IA
        if not game_state.game_over and game_state.joueur_courant == 1:
            time_to_play(game_state)

    return True

def main():
    global SerialObj
    """Fonction principale qui intègre les deux programmes"""
    # Initialiser l'état du jeu
    game_state = GameState()

    # Chargement de la capture vidéo
    cap = cv2.VideoCapture(0)#"http://192.168.68.56:4747/video"
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra. Vérifiez l'URL ou la connexion.")
        return

    # Initialiser le jeu en passant l'état du jeu
    game_state = jouer(game_state)

    # Initialiser le buffer de grille
    global grid_buffer
    grid_buffer = []

    # Attendre que la caméra s'initialise correctement
    print("Initialisation de la caméra...")
    for _ in range(10):  # Capturer quelques images pour stabiliser la caméra
        ret, _ = cap.read()
        if not ret:
            print("Erreur: Impossible de lire une image de la caméra.")
            return
        time.sleep(0.1)
    print("Caméra initialisée!")

    # Boucle principale
    while True:
        # Gérer les événements Pygame pour éviter que l'interface ne paraisse bloquée
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                return

        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire une image de la caméra.")
            break

        # Inverser l'image horizontalement
        frame = cv2.flip(frame, 1)

        # Détection des jetons à chaque frame
        current_grid = detect_tokens(frame)

        # Mise à jour du buffer à chaque frame
        grid_buffer.append(current_grid)
        if len(grid_buffer) > BUFFER_SIZE:
            grid_buffer.pop(0)

        current_time = time.time()
        game_state.grid_changed = False

        # Stabilisation et conversion en matrice seulement une fois par intervalle
        if current_time - game_state.last_grid_update_time >= GRID_UPDATE_INTERVAL:
            if len(grid_buffer) >= BUFFER_SIZE // 2:
                stable_grid = stabilize_grid(current_grid, game_state)
                current_matrix = grid_to_matrix(stable_grid)

                # Vérifier que la structure physique de la grille est valide
                structure_valid = game_state.last_stable_matrix is None or is_valid_move(game_state.last_stable_matrix, current_matrix)

                if structure_valid:
                    # Si un changement est détecté et que le temps de stabilisation est passé
                    if (game_state.last_stable_matrix is None or
                        matrices_are_different(current_matrix, game_state.last_stable_matrix)) and \
                    (current_time - game_state.last_change_time >= SETTLING_TIME):

                        # Tenter de mettre à jour le jeu avec cette nouvelle grille
                        game_updated = update_from_camera(current_matrix, game_state.last_stable_matrix, game_state)

                        # Seulement si le jeu a été mis à jour avec succès, on met à jour la matrice stable
                        if game_updated:
                            game_state.grid_changed = True
                            game_state.current_matrix = current_matrix

                            # Mettre à jour les matrices spécifiques aux joueurs
                            update_player_matrices(current_matrix, game_state.last_stable_matrix)

                            # Deep copy pour éviter les références partagées
                            game_state.last_stable_matrix = [row[:] for row in current_matrix]
                            game_state.last_change_time = current_time

                            # Affichage de la nouvelle grille validée
                            print("Nouvelle grille détectée et validée:")
                            for row in current_matrix:
                                print(row)
                            print("---------------------")

                game_state.last_grid_update_time = current_time

        # Utiliser la dernière grille stable pour l'affichage
        display_grid = game_state.last_stable_grid if game_state.last_stable_grid is not None else current_grid
        camera_overlay = overlay_on_camera(frame, display_grid)

        # Afficher le statut actuel
        status_text = "Grille modifiée!" if game_state.grid_changed else "Grille stable"
        status_color = (0, 0, 255) if game_state.grid_changed else (0, 255, 0)
        cv2.putText(camera_overlay, status_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Afficher le temps restant jusqu'à la prochaine mise à jour
        time_to_next = max(0, GRID_UPDATE_INTERVAL - (current_time - game_state.last_grid_update_time))
        cv2.putText(camera_overlay, f"Prochaine mise a jour: {time_to_next:.1f}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Afficher le joueur courant
        player_text = f"Tour: {'IA (Rouge)' if game_state.joueur_courant == 1 else 'Joueur (Jaune)'}"
        cv2.putText(camera_overlay, player_text,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if game_state.joueur_courant == 1 else (0, 255, 255), 2)

        cv2.imshow("Camera Feed", camera_overlay)
        cv2.setMouseCallback("Camera Feed", lambda event, x, y, flags, param: mouse_callback(event, x, y, flags, param, frame))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Réinitialiser le jeu
            print("Réinitialisation du jeu...")
            game_state = jouer(game_state)
            grid_buffer = []

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

def detect_circles(frame, lower, upper):
    """Détecte les cercles d'une couleur spécifique dans l'image"""
    roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA <= area <= MAX_AREA:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (math.pi * (radius ** 2))
            if circularity >= MIN_CIRCULARITY:
                centers.append((int(cx), int(cy)))

    return centers, mask

def detect_tokens(frame):
    """Détecte tous les jetons rouges et jaunes dans l'image"""
    red_centers1, _ = detect_circles(frame, LOWER_RED1, UPPER_RED1)
    red_centers2, _ = detect_circles(frame, LOWER_RED2, UPPER_RED2)
    red_centers = red_centers1 + red_centers2

    yellow_centers1, _ = detect_circles(frame, LOWER_YELLOW1, UPPER_YELLOW1)
    yellow_centers2, _ = detect_circles(frame, LOWER_YELLOW2, UPPER_YELLOW2)
    yellow_centers3, _ = detect_circles(frame, LOWER_YELLOW3, UPPER_YELLOW3)
    yellow_centers4, _ = detect_circles(frame, LOWER_YELLOW4, UPPER_YELLOW4)
    yellow_centers = yellow_centers1 + yellow_centers2 + yellow_centers3 + yellow_centers4

    # Création d'une grille vide
    grid = {}

    # Définir les dimensions de la cellule
    cell_width = ROI_W / COLS
    cell_height = ROI_H / ROWS

    # Traiter chaque jeton rouge
    for cx, cy in red_centers:
        # Convertir les coordonnées en indices de grille
        col = int(cx / cell_width)
        row = int(cy / cell_height)

        # S'assurer que les indices sont dans les limites
        if 0 <= row < ROWS and 0 <= col < COLS:
            grid[(row, col)] = "red"

    # Traiter chaque jeton jaune
    for cx, cy in yellow_centers:
        col = int(cx / cell_width)
        row = int(cy / cell_height)
        if 0 <= row < ROWS and 0 <= col < COLS:
            grid[(row, col)] = "yellow"

    return grid

def overlay_on_camera(frame, grid):
    """Superpose la grille détectée sur l'image de la caméra"""
    overlay = frame.copy()

    # Dessiner la grille pour visualisation
    for row in range(ROWS):
        for col in range(COLS):
            # Calculer le centre de chaque cellule
            cx = ROI_X + int((col + 0.5) * (ROI_W / COLS))
            cy = ROI_Y + int((row + 0.5) * (ROI_H / ROWS))

            # Dessiner le cadre de la cellule
            cell_w = int(ROI_W / COLS)
            cell_h = int(ROI_H / ROWS)
            cv2.rectangle(overlay,
                         (ROI_X + col * cell_w, ROI_Y + row * cell_h),
                         (ROI_X + (col + 1) * cell_w, ROI_Y + (row + 1) * cell_h),
                         (100, 100, 100), 1)

            # Si un jeton est présent dans cette cellule, le dessiner
            if (row, col) in grid:
                color = grid[(row, col)]
                color_bgr = (0, 0, 255) if color == "red" else (0, 255, 255) if color == "yellow" else (255, 255, 255)
                cv2.circle(overlay, (cx, cy), int(min(cell_w, cell_h) * 0.4), color_bgr, -1)

    # Dessiner le cadre ROI
    cv2.rectangle(overlay, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)

    return overlay

def stabilize_grid(current_grid, game_state):
    """Stabilise la détection de la grille en utilisant un buffer temporel"""
    global grid_buffer

    # Si le buffer n'est pas encore rempli, utiliser la grille actuelle
    if len(grid_buffer) < BUFFER_SIZE:
        return current_grid

    # Créer une grille stabilisée
    stable_grid = {}
    all_positions = set()

    # Collecter toutes les positions détectées dans le buffer
    for grid in grid_buffer:
        all_positions.update(grid.keys())

    # Pour chaque position détectée
    for pos in all_positions:
        # Collecter toutes les couleurs détectées à cette position
        colors = [grid.get(pos, None) for grid in grid_buffer]
        colors = [c for c in colors if c is not None]  # Éliminer les None

        # Si aucune couleur n'a été détectée, passer à la position suivante
        if not colors:
            continue

        # Compter les occurrences de chaque couleur
        color_counts = Counter(colors)
        most_common_color, count = color_counts.most_common(1)[0]

        # Si la couleur la plus fréquente dépasse le seuil, l'utiliser
        if count / len(grid_buffer) >= DETECTION_THRESHOLD:  # Utiliser le buffer complet comme dénominateur
            stable_grid[pos] = most_common_color

    # Vérifier que la grille est valide selon les règles de gravité
    if not is_valid_grid(stable_grid):
        print("Grille non valide, ignorée")
        # Conserver l'ancienne grille si la nouvelle n'est pas valide
        if game_state.last_stable_grid is not None:
            return game_state.last_stable_grid

    # Mettre à jour le timestamp de stabilisation
    game_state.last_stabilization_time = time.time()
    game_state.last_stable_grid = stable_grid

    return stable_grid

def grid_to_matrix(grid):
    """Convertit la représentation en dictionnaire de la grille en matrice 2D"""
    # Créer une matrice vide remplie de zéros
    matrix = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    # Remplir la matrice avec les valeurs du dictionnaire grid
    for (row, col), color in grid.items():
        if 0 <= row < ROWS and 0 <= col < COLS:  # Vérifier les limites
            if color == "red":
                matrix[row][col] = 1
            elif color == "yellow":
                matrix[row][col] = 2

    return matrix

def is_valid_grid(grid):
    """Vérifie que la grille respecte les règles de gravité du Puissance 4"""
    # Convertir en matrice pour faciliter la vérification
    matrix = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    for (row, col), color in grid.items():
        if 0 <= row < ROWS and 0 <= col < COLS:
            if color == "red":
                matrix[row][col] = 1
            elif color == "yellow":
                matrix[row][col] = 2

    # Vérifier les règles de gravité
    for col in range(COLS):
        # Pour chaque colonne, on vérifie de bas en haut
        found_empty = False
        for row in range(ROWS-1, -1, -1):  # De bas en haut
            if matrix[row][col] == 0:  # Case vide
                found_empty = True
            elif found_empty:  # Si on trouve un jeton après une case vide (violation de gravité)
                return False

    return True

def is_valid_game_move(current_matrix, previous_matrix, game_state):
    """Vérifie si le mouvement détecté est valide selon les règles du jeu."""
    # Déterminer joueur et colonne du dernier coup
    last_player = get_last_player(current_matrix, previous_matrix)
    last_move = get_last_move_column(current_matrix, previous_matrix)

    # Si aucun joueur n'est détecté, pas de mouvement valide
    if last_player is None or last_move == -1:
        return False, None, None

    # Convertir last_player en entier
    player_num = 1 if last_player == "red" else 2 if last_player == "yellow" else None

    # Vérifier si c'est bien le tour de ce joueur
    if player_num != game_state.joueur_courant:
        print(f"Détection ignorée: c'est au tour du joueur {game_state.joueur_courant}, mais {player_num} a été détecté")
        return False, None, None

    # Vérification spéciale pour le coup de l'IA
    if game_state.en_attente_detection and player_num == 1:
        if not verifier_coup_ia(last_move, game_state):
            print(f"Coup détecté en colonne {last_move + 1} ne correspond pas au coup de l'IA attendu")
            return False, None, None

    return True, player_num, last_move

def count_tokens(matrix):
    """Compte le nombre total de jetons dans la matrice"""
    count = 0
    for row in range(ROWS):
        for col in range(COLS):
            if matrix[row][col] > 0:  # Un jeton est présent (1 pour rouge, 2 pour jaune)
                count += 1
    return count

def is_valid_move(previous_matrix, current_matrix):
    if previous_matrix is None:
        # Si c'est la première grille, elle doit être vide
        return all(all(cell == 0 for cell in row) for row in current_matrix)

    previous_count = count_tokens(previous_matrix)
    current_count = count_tokens(current_matrix)

    return current_count == previous_count + 1

def matrices_are_different(matrix1, matrix2):
    """Compare deux matrices et retourne True si elles sont différentes"""
    if matrix1 is None or matrix2 is None:
        return True  # Si l'une des matrices est None, considérer comme différentes

    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if matrix1[i][j] != matrix2[i][j]:
                return True  # Différence trouvée

    return False  # Aucune différence trouvée

def get_last_move_column(current_matrix, previous_matrix):
    """Détermine la colonne du dernier coup joué en comparant deux matrices consécutives"""
    # Si une des matrices est None, impossible de déterminer le dernier coup
    if current_matrix is None or previous_matrix is None:
        return -1

    # Parcourir chaque colonne
    for col in range(COLS):
        # Trouver le premier jeton différent dans cette colonne (de bas en haut)
        for row in range(ROWS-1, -1, -1):
            # Si on trouve un jeton dans la matrice actuelle qui n'était pas là avant
            if current_matrix[row][col] != 0 and previous_matrix[row][col] == 0:
                return col

    # Aucun nouveau jeton trouvé
    return -1

def get_last_player(current_matrix, previous_matrix):
    """Détermine quel joueur a joué le dernier coup"""
    # Si une des matrices est None, impossible de déterminer le dernier joueur
    if current_matrix is None or previous_matrix is None:
        return None

    # Parcourir la grille pour trouver le nouveau jeton
    for row in range(ROWS):
        for col in range(COLS):
            # Si on trouve un jeton dans la matrice actuelle qui n'était pas là avant
            if previous_matrix[row][col] == 0 and current_matrix[row][col] != 0:
                # Identifier le joueur en fonction de la valeur
                if current_matrix[row][col] == 1:
                    return "red"
                elif current_matrix[row][col] == 2:
                    return "yellow"

    # Aucun nouveau jeton trouvé
    return None

def is_empty_matrix(matrix):
    """Vérifie si la matrice est vide (aucun jeton n'est placé)"""
    return all(all(cell == 0 for cell in row) for row in matrix)

def update_player_matrices(current_matrix, previous_matrix):
    """Met à jour les matrices de chaque joueur en fonction du changement détecté"""
    global last_red_move_matrix, last_yellow_move_matrix

    # Si pas de matrice précédente, impossible de déterminer le dernier joueur
    if previous_matrix is None:
        return

    # Trouver le nouveau jeton ajouté
    for row in range(ROWS):
        for col in range(COLS):
            # Si un jeton a été ajouté
            if previous_matrix[row][col] == 0 and current_matrix[row][col] != 0:
                # Vérifier la couleur du jeton ajouté
                if current_matrix[row][col] == 1:  # Rouge
                    # Copier profondément la matrice actuelle
                    last_red_move_matrix = [row[:] for row in current_matrix]
                elif current_matrix[row][col] == 2:  # Jaune
                    # Copier profondément la matrice actuelle
                    last_yellow_move_matrix = [row[:] for row in current_matrix]
                return

def get_last_red_move_grid():
    """Renvoie la grille telle qu'elle était après le dernier coup des rouges"""
    global last_red_move_matrix
    return last_red_move_matrix

def get_last_yellow_move_grid():
    """Renvoie la grille telle qu'elle était après le dernier coup des jaunes"""
    global last_yellow_move_matrix
    return last_yellow_move_matrix

def mouse_callback(event, x, y, flags, param, frame):
    """Callback pour les clics de souris - utile pour le débogage des couleurs"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if ROI_X <= x <= ROI_X + ROI_W and ROI_Y <= y <= ROI_Y + ROI_H:
            roi_x, roi_y = x - ROI_X, y - ROI_Y
            roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[roi_y, roi_x]
            print(f"HSV à ce point: H={h}, S={s}, V={v}")

# Lancer le jeu
if __name__ == "__main__":
    main()