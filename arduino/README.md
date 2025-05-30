## Overview of Arduino Programs for Connect 4

- **arduino_main**: The main program used during gameplay. It handles player moves, target positions, and LED control based on the current player's turn.

- **arduino_position**: A tool for freely moving the robot and obtaining real-time position feedback. This helps determine the robot's step positions for accurate token placement. Used during setup and testing.

- **Led**: Used to verify proper color communication with the LED strip, ensuring the LEDs display colors correctly. Used during testing.

- **testled**: Designed to detect the number of LEDs on the LED strip, as this information was initially unknown. Used during testing.

Only **arduino_main** is necessary to run a game, while the other programs are intended for setup and diagnostic purposes.
