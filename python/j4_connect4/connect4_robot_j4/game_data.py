class GameData:
    def __init__(self):       
        #data for Firebase
        self.first_player = None
        self.winner = None
        self.moves = []
        self.player_pseudo = "Player1"
        self.game_id = None
        self.game_start_time = None
        self.game_end_time = None