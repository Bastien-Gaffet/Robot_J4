# 🤖 Connect4 Robot J4
A physical Connect 4 game powered by:

- 🎥 **Computer vision** (OpenCV)
- 🧠 **Artificial intelligence** (minimax algorithm)
- ⚙️ **Arduino-based control** (via PySerial)
- 🖥 **Graphical interface** (Pygame)
- 🦾 **Mechanical arm** (MOVEO-3D)


## 📦 Installation
To install and run the python code you can (You need to have Git installed on your machine):
- I Clone the repository
    ```bash
    git clone https://github.com/Bastien-Gaffet/Robot_J4.git
    cd Robot_J4/python/j4_connect4
    ```
- II Install in development mode
    ```bash
    pip install -e . 
    ```
- Or you can also tipe this command : 
```bash
pip install git+https://github.com/Bastien-Gaffet/Robot_J4.git@main#subdirectory=python/j4_connect4
```
- Finally, just type this command; it works as well:
```bash
pip install connect4-robot-j4
```

**This will:**

Install all required dependencies (pygame, opencv-python, pyserial, etc.)
Make the command connect4 available in your terminal

## ▶️ Usage

To **start** the game, run:
```bash
connect4
```
The program will:

1. **Initialize** the game state
2. **Start** the camera
3. **Wait** for a clean empty grid to begin
4. **Detect** player or AI moves and **update** the game board in **real time**


## 🎮 Controls

- **r** → Reset the game
- **q** → Quit the game


## 🧱 Project Structure
```bash
connect4_robot_j4/
├── main.py                 # Entry point
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
```

## 📋 Requirements

```python
Python ≥ 3.8
opencv-python
pygame
pyserial
numpy
```

You can also install them manually:
```bash
pip install -r requirements.txt
```

## ⚙️ Developer Notes
To modify the code and have changes reflected without reinstalling:
```bash
pip install -e .
```

## 🚀 Future Ideas

- Score tracking
- Match history or logs
- GUI-based calibration for camera and detection zones

## 👨‍🔬 Author

This project was developed by the Vaucanson Robot J4 Team

## 📄 License

This project is licensed - see the [LICENSE](https://github.com/Bastien-Gaffet/Robot_J4/blob/main/LICENSE) file for details.