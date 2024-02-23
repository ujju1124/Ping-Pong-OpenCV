import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone import overlayPNG
import random
import time

# Add Text to Center


def addTextToCenter(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA, custom_x=None, custom_y=None):
    image = image.copy()
    # Get text dimensions
    text_width, text_height = cv2.getTextSize(
        text, font, fontScale, thickness)[0]

    # Calculate center coordinates
    img_center_x = int(FrameWidth / 2)
    img_center_y = int(boardMaxHeight / 2)

    text_x = img_center_x - int(text_width / 2)
    text_y = img_center_y + int(text_height / 2)

    if not (custom_x == None):
        text_x = custom_x

    if not (custom_y == None):
        text_y = custom_y

    image = cv2.putText(image, text, (text_x, text_y),
                        font, fontScale, color, thickness, lineType)

    return image


# Draw the Board
def updateBoard(image):
    image = image.copy()

    # Draw the Board
    mask = np.zeros_like(image)
    mask = cv2.rectangle(
        mask, (0, 0), (FrameWidth, FrameHeight), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.3, mask, 0.7, 0)

    # Draw Bottom Rectangle
    mask = cv2.rectangle(mask, (0, boardMaxHeight),
                         (FrameWidth, FrameHeight), (252, 5, 244), -1)
    image = cv2.addWeighted(image, 1, mask, 1, 0)

    # Draw the Gamer Lines
    image = cv2.line(image, (boardWidth, 0),
                     (boardWidth, boardMaxHeight), (252, 5, 244), 1)
    image = cv2.line(image, (FrameWidth-boardWidth, 0),
                     (FrameWidth-boardWidth, boardMaxHeight), (252, 5, 244), 1)

    # Add Points
    image = cv2.putText(image, str(leftPoint), (boardWidth - 25,
                        boardMaxHeight + 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    image = cv2.putText(image, str(rightPoint), (FrameWidth -
                        boardWidth - 25, boardMaxHeight + 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    # Add the Ball
    image = overlayPNG(image, ballImg, [ballPosX, ballPosY])

    # Add "How to Start Game" text if Game is not Started
    if not (gameStarted):
        image = addTextToCenter(image, "Show Both hands to Start the Game", color=(
            255, 255, 255), custom_y=boardMaxHeight+70)
    else:
        image = addTextToCenter(image, "Play with Index Fingers", color=(
            255, 255, 255), custom_y=boardMaxHeight+70)

    return image


# Update the Bat
def updateBat(image, idxPos, bat, drawCenter=True):
    image = image.copy()

    y1 = idxPos[1] - (batHeight // 2)
    if (y1 <= 0):
        y1 = 0
    y2 = y1 + batHeight

    if (y2 >= boardMaxHeight):
        y1 = boardMaxHeight - batHeight
        y2 = y1 + batHeight

    if (bat == "Left"):  # Left Bat
        x1 = boardWidth - batWidth
        x2 = boardWidth
        globals()["leftBatPosY"] = [y1, y2]
    elif (bat == "Right"):  # Right Bat
        x1 = FrameWidth - boardWidth
        x2 = x1 + batWidth
        globals()["rightBatPosY"] = [y1, y2]

    # Draw the Bat
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (230, 11, 26), -1)

    # Draw Bat center
    image = cv2.circle(image, (x1 + (batWidth//2), y1 +
                       (batHeight // 2)), 6, (255, 0, 255), -1)

    # Add Pointer in the finger
    image = cv2.circle(image, (idxPos[0], idxPos[1]), 6, (255, 0, 255), -1)

    return image


# Global & Const Variables
boardMaxHeight = 420  # y -> 0 - 420
boardWidth = 80  # x -> 80 - WIDTH - 80

FrameWidth = 960
FrameHeight = 540

ballWH = 50
ballPosX, ballPosY = (FrameWidth // 2) - (ballWH //
                                          2), (boardMaxHeight // 2) - (ballWH // 2)
ballSpeed = 16
# For start, Randomly choose the ball direction
ballSpeedX = random.choice([ballSpeed, -ballSpeed])
ballSpeedY = random.choice([ballSpeed, -ballSpeed])
speedUpEvery = 8  # seconds

batHeight = 100
batWidth = 30

leftBatPosY = [0, 0]
rightBatPosY = [0, 0]

leftPoint = 0
rightPoint = 0

gameStarted = False
gameOver = False

startTime = time.time()

# Ball
ballImg = cv2.imread("./asset/ball.png", cv2.IMREAD_UNCHANGED)

# Initializing Hand Tracker
detector = HandDetector(detectionCon=0.5)

# Video Capture
cap_vid = cv2.VideoCapture(1)
cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, FrameWidth)
cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FrameHeight)
windowName = "Ping Pong"

while cap_vid.isOpened():
    ret, frame = cap_vid.read()
    if not ret:
        continue

    img = cv2.flip(frame, cv2.CAP_PROP_XI_DECIMATION_HORIZONTAL)
    # draw=True if Hand Needs to be drawn
    handsT, _ = detector.findHands(img, flipType=False, draw=False)
    hands = []
    for hand in handsT:
        if not any(x["type"] == hand["type"] for x in hands):
            hands.append(hand)

    if (len(hands) == 2) and (not gameStarted):
        gameStarted = True
    # Draw The Board
    img = updateBoard(img)

    for hand in hands:
        if (not gameStarted):
            break

        fingers = detector.fingersUp(hand)
        if (fingers[1] != 1):
            continue

        lmList = hand["lmList"]
        idxPos = lmList[8]

        # img = updateBat(img, idxPos, hand["type"], drawCenter=False) # Without Finger and Bat Center pointed
        img = updateBat(img, idxPos, hand["type"])

    """
    1. In Below condition, we are first checking if the Ball is in any territory (Left / Right) by Checking it touches or crosses the border --- and then checking if the ball's center is between the Y positions of any Bat. Then change the direction
    2. Then checking if the ball is out of the field, if it is, then game over and point increased
    """
    if ((boardWidth - 20) < ballPosX < boardWidth) and ((leftBatPosY[0] < (ballPosY + (ballWH / 2)) < leftBatPosY[1])):  # Left Bat hit
        ballSpeedX = -ballSpeedX
        ballPosX += 30
    # Right Bat hit
    elif ((FrameWidth - boardWidth + 20) > (ballPosX + ballWH) > (FrameWidth - boardWidth)) and ((rightBatPosY[0] < (ballPosY + (ballWH / 2)) < rightBatPosY[1])):
        ballSpeedX = -ballSpeedX
        ballPosX -= 30
    elif (ballPosX + ballWH <= 0):
        gameOver = True
        rightPoint += 1
    elif (ballPosX >= FrameWidth):
        gameOver = True
        leftPoint += 1

    # If Game is Over
    if (gameOver):
        gameStarted = False
        gameOver = False
        ballSpeedX = random.choice([ballSpeed, -ballSpeed])
        ballSpeedY = random.choice([ballSpeed, -ballSpeed])
        ballPosX, ballPosY = (FrameWidth // 2) - (ballWH //
                                                  2), (boardMaxHeight // 2) - (ballWH // 2)

        imgGO = addTextToCenter(
            img, "Game Over", fontScale=2, color=(255, 255, 255), thickness=3)

        cv2.imshow(windowName, imgGO)
        cv2.waitKey(2000)

    # Reset bat positions every time to avoid "Invisible Bat" situation
    leftBatPosY = [0, 0]
    rightBatPosY = [0, 0]

    # Check if Ball Hits the Wall!
    if (ballPosY <= 0) or (ballPosY + ballWH >= boardMaxHeight):
        ballSpeedY = -ballSpeedY

    if (gameStarted):
        ballPosX += ballSpeedX
        ballPosY += ballSpeedY

    # Speed up after every 15 seconds
    if (time.time() - startTime >= 15):
        if (ballSpeedX < 0):
            ballSpeedX -= 1
        else:
            ballSpeedX += 1

        if (ballSpeedY < 0):
            ballSpeedY -= 1
        else:
            ballSpeedY += 1

        startTime = time.time()

    cv2.imshow(windowName, img)
    if (cv2.waitKey(1) & 0xFF == 27):
        break
    if (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1):
        break


# Release Camera
cap_vid.release()
cv2.destroyAllWindows()
