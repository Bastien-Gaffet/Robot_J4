# AI Development Project — PyTorch & TensorBoard Visualization

> **Warning:**  
> This part of the project is currently experimental and has not yet achieved reliable results for playing against a human opponent.

## Overview

This project explores the creation of an artificial intelligence (AI) agent using **PyTorch** in Python. The AI’s training progress can be visualized in real time using **TensorBoard**.

## How to Use

To monitor the training progress of the AI, simply execute the following file:

- **tensorboard-visualisation.bat**

This will launch TensorBoard and allow you to track the AI’s learning metrics as training progresses.

## Additional Information

- Past training sessions are also available within the TensorBoard visualization, enabling you to compare different runs.
- The existing models were trained on an **NVIDIA RTX 3060** GPU.
- The project is still under active development, and further improvements are planned.
- The program should save previous versions in an archive folder, but this logic is not implemented correctly. It is possible that you will need to do it manually to avoid completely overwriting the old model.

## Running the Connect Four Game or Testing the AI

To test the AI or simply play Connect Four games (against a human or AI opponent), you can run the following program:
**play_connect4/puissance4_main.py**
This script allows you to:

- Play a full game of Connect Four
- Choose between different opponent types (e.g. human, Minimax algorithm, random moves, or trained AI)
- Test and evaluate AI behavior in a complete game loop

This module is designed for both gameplay and experimentation with different AI strategies.
---
