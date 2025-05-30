import random

def choisir_coup(plateau):
    colonnes_valides = [col for col in range(7) if plateau[0][col] == 0]
    return random.choice(colonnes_valides) if colonnes_valides else None