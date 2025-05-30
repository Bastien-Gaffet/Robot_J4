import numpy as np
import pygame
import sys

# Paramètres d'affichage
TAILLE_CASE = 80
LARGEUR = 7 * TAILLE_CASE
HAUTEUR = (6 + 2) * TAILLE_CASE  # 6 lignes + 1 pour la sélection + 1 pour les messages
RAYON = int(TAILLE_CASE / 2 - 5)

# Couleurs
BLEU = (0, 0, 255)
NOIR = (0, 0, 0)
ROUGE = (255, 0, 0)
JAUNE = (255, 255, 0)
BLANC = (255, 255, 255)

# Plateau initial
plateau = np.zeros((6, 7), dtype=int)
screen = None
font = None

def choisir_adversaire():
    print("Qui voulez-vous affronter ?")
    print("1 : Humain vs Humain")
    print("2 : Humain vs IA Minimax")
    print("3 : Humain vs IA Entraînée")
    print("4 : Humain vs IA Aléatoire")
    choix = input("Entrez le numéro de votre choix : ")
    if choix == "2":
        import algo_minimax as ia_module
    elif choix == "3":
        import trained_AI as ia_module
    elif choix == "4":
        import random_column as ia_module
    else:
        ia_module = None
    return ia_module

def initialiser_jeu():
    global screen, font
    pygame.init()
    screen = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption('Puissance 4')
    font = pygame.font.SysFont('monospace', 30)

def afficher_plateau(joueur_courant=None):
    screen.fill(BLANC)
    pygame.draw.rect(screen, BLEU, (0, TAILLE_CASE * 2, LARGEUR, 6 * TAILLE_CASE))

    for col in range(7):
        for ligne in range(6):
            couleur = BLANC
            if plateau[ligne][col] == 1:
                couleur = ROUGE
            elif plateau[ligne][col] == 2:
                couleur = JAUNE
            posX = col * TAILLE_CASE + TAILLE_CASE // 2
            posY = (ligne + 2) * TAILLE_CASE + TAILLE_CASE // 2
            pygame.draw.circle(screen, couleur, (posX, posY), RAYON)

        # Numéros de colonnes
        texte = font.render(str(col + 1), True, NOIR)
        screen.blit(texte, (col * TAILLE_CASE + TAILLE_CASE // 2 - 10, TAILLE_CASE + 10))

    # Affichage du joueur courant en haut
    if joueur_courant:
        texte_tour = f"Tour du joueur {joueur_courant} ({'Rouge' if joueur_courant == 1 else 'Jaune'})"
        rendu = font.render(texte_tour, True, NOIR)
        screen.blit(rendu, (10, 10))

    pygame.display.update()

def coup_valide(col):
    return plateau[0][col] == 0

def animer_jeton(col, ligne_finale, joueur):
    couleur = ROUGE if joueur == 1 else JAUNE
    for ligne in range(ligne_finale + 1):
        afficher_plateau()
        posX = col * TAILLE_CASE + TAILLE_CASE // 2
        posY = (ligne + 2) * TAILLE_CASE + TAILLE_CASE // 2
        pygame.draw.circle(screen, couleur, (posX, posY), RAYON)
        pygame.display.update()
        pygame.time.wait(50)

def placer_jeton(col, joueur):
    for ligne in range(5, -1, -1):
        if plateau[ligne][col] == 0:
            animer_jeton(col, ligne, joueur)
            plateau[ligne][col] = joueur
            break

def verifier_victoire(joueur):
    # Horizontale
    for ligne in range(6):
        for col in range(4):
            if all(plateau[ligne][col + i] == joueur for i in range(4)):
                return True
    # Verticale
    for ligne in range(3):
        for col in range(7):
            if all(plateau[ligne + i][col] == joueur for i in range(4)):
                return True
    # Diagonale ↘
    for ligne in range(3):
        for col in range(4):
            if all(plateau[ligne + i][col + i] == joueur for i in range(4)):
                return True
    # Diagonale ↙
    for ligne in range(3):
        for col in range(3, 7):
            if all(plateau[ligne + i][col - i] == joueur for i in range(4)):
                return True
    return False

def afficher_message_fin(message):
    pygame.draw.rect(screen, BLANC, (0, 0, LARGEUR, TAILLE_CASE))
    texte = font.render(message, True, NOIR)
    screen.blit(texte, (40, 10))
    pygame.display.update()
    pygame.time.wait(3000)

def tour_adversaire_automatique(joueur, ia_module):
    col = ia_module.choisir_coup(plateau)
    if col is not None and coup_valide(col):
        placer_jeton(col, joueur)
        return True
    return False

def jouer():
    ia_module = choisir_adversaire()
    initialiser_jeu()
    joueur_courant = 1
    game_over = False
    afficher_plateau(joueur_courant)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if ia_module is None or joueur_courant == 2:
                    x = event.pos[0]
                    col = x // TAILLE_CASE
                    if coup_valide(col):
                        placer_jeton(col, joueur_courant)
                        if verifier_victoire(joueur_courant):
                            afficher_message_fin(f"Joueur {joueur_courant} gagne !")
                            game_over = True
                        joueur_courant = 2 if joueur_courant == 1 else 1
                        afficher_plateau(joueur_courant)

        if not game_over and joueur_courant == 1 and ia_module:
            if tour_adversaire_automatique(1, ia_module):
                if verifier_victoire(1):
                    afficher_message_fin("IA gagne !")
                    game_over = True
                joueur_courant = 2
                afficher_plateau(joueur_courant)

if __name__ == "__main__":
    jouer()
