# 🤖 Robot J4 – Connect Four Playing Robotic Arm Using the Minimax Algorithm

## 📌 Project Overview

**Robot J4** is a custom-designed robotic arm built to autonomously play **Connect Four** against a human player. The project combines **mechanical design**, **embedded systems**, and **algorithmic strategy** to showcase human-robot interaction in a fun and intelligent way.

At the heart of the decision-making process lies the **Minimax algorithm**, a classical strategy from **game theory** commonly used in two-player, turn-based games like chess or Connect Four. Robot J4 evaluates potential moves up to a certain depth and selects the most optimal action to either **win or block** the opponent.

This project was awarded the **national final winner** title at the French **Concours Cgénial** and is currently being carried forward by our team for participation in the **EUCYS competition in Latvia**.

## 🎯 Objectives

- Build a robotic arm capable of dropping game pieces into a real-world Connect Four board.
- Create a smooth and interactive human-vs-machine gameplay loop.
- Use the **Minimax algorithm** with a heuristic evaluation function.
- Establish a **serial communication** link between a PC and the robot via UART.
- Design a modular, maintainable Python interface for interaction and logic.

## ⚙️ Technologies Used

- **Python** (game engine, Minimax logic, and user interface)
- **Minimax algorithm** with custom heuristics
- **Arduino** (motor control and robot commands)
- **Serial communication (UART)** for computer-to-robot interaction
- **Mechanical design** (custom robotic arm for token handling)
- **Computer vision** for automated board state recognition
- - *(Optional – future upgrade)* **Miniaturization** using an arduino instead of a desktop or laptop computer

## 🚀 How It Works

1. The first player (robot J4 or human) is determined randomly at the start of the game.  
2. A camera films the Connect Four board in real time, enabling the robot to track the game’s progress and know the current state of the grid.  
3. When it’s J4’s turn, the robot analyzes the board using the Minimax algorithm and chooses the optimal move.  
4. J4 physically drops the token into the selected column.  
5. The human player plays next, and the robot detects their move automatically via the camera feed.  
6. The cycle continues with real-time board recognition until one player wins or the board is full.

## 📦 Repository Structure

- **arduino/**: Source code to control the robotic arm (J4).
- **data/**: Intended for storing saved games, move histories, scores, or logs.
- **docs/**: Technical documentation, circuit diagrams, and design notes.
- **python/**: Contains all Python code, now organized into:
  - **camera/**: Handles token detection, image processing, and game logic.
  - **minimax/**: Implements the Minimax AI algorithm and board management.
- **LICENSE**: contains the full license terms governing the use, distribution, and attribution of this project.
- **README.md**: This file, providing a general overview of the project.

## 🧠 Key Concepts

- **Minimax Algorithm**: Used to simulate future game states and make optimal decisions.
- **Heuristic Evaluation**: Scores board positions based on win/loss chances and threats.
- **Human-Robot Interaction**: Alternating moves with physical execution and human input.
- **Mechanical Design**: Custom robotic arm with real-time motor control.

## 🤝 Acknowledgments

This project was created as part of a high school engineering specialization.  
Special thanks to our teachers, mentors, and supporters for their guidance.

---

**Made with ❤️ by the Vaucanson Robot J4 Team**
