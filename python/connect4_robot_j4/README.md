#🤖 Connect4 Robot J4
A physical Connect 4 game powered by:

- 🎥 Computer vision (OpenCV)
- 🧠 Artificial intelligence (minimax algorithm)
- ⚙️ Arduino-based control (via PySerial)
- 🖥 Graphical interface (Pygame)
- 🦾 Mechanical arm


##📦 Installation
To install and run the python code you can :
- 1. Clone the repository
    ```bash
    git clone https://github.com/Bastien-Gaffet/Robot_J4.git
    cd Robot_J4/python/connect4_robot_j4
    ```
    2. Install in development mode
    ```bash
    pip install -e . 
    ```
- Or just tipe this command (You need to have Git installed on your machine): 
```bash
pip install git+https://github.com/Bastien-Gaffet/Robot_J4.git@Python-main-modification#subdirectory=connect4_robot_j4
```
This will:

Install all required dependencies (pygame, opencv-python, pyserial, etc.)
Make the command connect4 available in your terminal

##▶️ Usage

To start the game, run:
```bash
connect4
```
The program will:

Initialize the game state
Start the camera
Wait for a clean empty grid to begin
Detect player or AI moves and update the game board in real time


##🎮 Controls

r → Reset the game
q → Quit the game


##🧱 Project Structure
connect4_robot_j4/
├── main.py                  # Entry point
├── game_loop.py            # Main game logic
├── core.py                 # Game initialization
├── game_state.py           # Game state container
├── constants.py            # HSV color thresholds, config values
├── camera/                 # Vision system (token detection, grid extraction)
├── minimax/                # AI algorithm
├── arduino_serial/         # Serial communication with Arduino
├── requirements.txt
├── setup.py
└── README.md

##📋 Requirements

Python ≥ 3.8
opencv-python
pygame
pyserial
numpy

You can also install them manually:
bashpip install -r requirements.txt

##⚙️ Developer Notes
To modify the code and have changes reflected without reinstalling:
bashpip install -e .

##🚀 Future Ideas

Score tracking
Match history or logs
GUI-based calibration for camera and detection zones


##👨‍🔬 Author

This project was developed by the Vaucanson Robot J4 Team

##📄 License

This project is licensed - see the LICENSE file for details.