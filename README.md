# Ping Pong Game Created Using OpenCV

Experience a touchless Ping Pong game powered by computer vision! This project lets you play Ping Pong using only your hand movements, detected via your webcam—no physical controllers required. It's a fun demonstration of real-time hand tracking and interactive gameplay using Python.

## Overview

This project leverages computer vision and machine learning to create an interactive Ping Pong game. By using your webcam, the game tracks your hand movements and translates them into paddle controls, allowing you to play without touching any hardware. It's a great way to explore OpenCV, Mediapipe, and real-time image processing.

## Technologies Used

- **OpenCV**: For capturing video from the webcam and processing image frames in real time.
- **Mediapipe**: For advanced hand tracking and gesture recognition, enabling touchless control.
- **NumPy**: For efficient numerical operations and array manipulations during image processing.

## Python Packages Required

To run this project, you need to install the following Python packages:

- `opencv-python`: The main OpenCV library for computer vision tasks.
- `mediapipe`: Provides pre-trained models for hand tracking and gesture recognition.
- `numpy`: Used for numerical computations and array operations.
- `cvzone`: A utility library that simplifies many OpenCV and Mediapipe tasks, making it easier to build interactive applications.

### Installation

You can install all required packages using pip. Run the following command in your terminal or command prompt:

```shell
pip install opencv-python mediapipe numpy cvzone
```

Alternatively, you can use Python's module installer:

```shell
python -m pip install opencv-python mediapipe numpy cvzone
```

## How to Run the Game

1. Make sure you have Python installed (version 3.7.X to 3.10.X is required for Mediapipe compatibility).
2. Install the required packages as shown above.
3. Download or clone this repository to your local machine.
4. Place the `ball.png` asset in the `asset/` directory (already included).
5. Run the main script:

```shell
python main.py
```

6. Allow access to your webcam when prompted. Use your hand to control the paddle and enjoy the game!

## Compatibility Note

- **Mediapipe** requires Python version **3.7.X to 3.10.X**. Using an older or newer version may result in installation errors or runtime issues.
- This project was developed and tested with Python `3.9.5`.

## Troubleshooting

- If you encounter issues with package installation, ensure your Python version is within the supported range.
- For webcam access problems, check your system permissions and ensure no other application is using the camera.
- If you see errors related to missing modules, double-check that all required packages are installed.

## Credits

Created as a fun summer camp project to demonstrate the power of computer vision and touchless interaction using Python.
