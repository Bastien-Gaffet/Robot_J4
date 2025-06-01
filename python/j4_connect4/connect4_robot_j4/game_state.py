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
        self.grid_buffer = []
        self.last_red_move_matrix = None
        self.last_yellow_move_matrix = None
        self.joueur_courant = 1  # Initially, player 1 starts
        self.ia_a_joue = False
        self.en_attente_detection = False
        self.dernier_coup_ia = None
        self.game_over = False