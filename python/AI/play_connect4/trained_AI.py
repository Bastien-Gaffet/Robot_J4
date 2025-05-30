import numpy as np
import torch
import torch.nn as nn

ROWS, COLS = 6, 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn_puissance4.pth"

class ConvDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, COLS)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def plateau_to_state(plateau):
    # Convertit le plateau 6x7 avec valeurs {0,1,2} en tensor 3x6x7 float
    state = np.zeros((3, ROWS, COLS), dtype=np.float32)
    state[0][plateau == 1] = 1  # joueur 1
    state[1][plateau == 2] = 1  # joueur 2
    state[2][plateau == 0] = 1  # cases vides
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # batch 1

def valid_actions(plateau):
    # Colonnes non pleines
    return [c for c in range(COLS) if plateau[0][c] == 0]

# Chargement du modèle
model = ConvDQN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

def choisir_coup(plateau):
    state = plateau_to_state(plateau)
    with torch.no_grad():
        q_values = model(state)[0]  # shape (COLS,)
        valid_cols = valid_actions(plateau)
        # Masquer les colonnes pleines avec une grosse valeur négative
        q_values_cpu = q_values.cpu().numpy()
        for c in range(COLS):
            if c not in valid_cols:
                q_values_cpu[c] = -float('inf')
        # Choisir la colonne avec la meilleure valeur Q
        best_col = int(np.argmax(q_values_cpu))
    return best_col
