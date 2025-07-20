import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone import overlayPNG
import random
import time
import math

class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_explosion(self, x, y, color=(255, 255, 255), count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 30,
                'max_life': 30,
                'color': color,
                'size': random.uniform(2, 6)
            })
    
    def add_trail(self, x, y, color=(255, 255, 255)):
        self.particles.append({
            'x': x + random.uniform(-5, 5),
            'y': y + random.uniform(-5, 5),
            'vx': random.uniform(-1, 1),
            'vy': random.uniform(-1, 1),
            'life': 15,
            'max_life': 15,
            'color': color,
            'size': random.uniform(1, 3)
        })
    
    def update_and_draw(self, image):
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            particle['vy'] += 0.2  # gravity
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
                continue
            
            alpha = particle['life'] / particle['max_life']
            size = int(particle['size'] * alpha)
            
            if size > 0:
                color = tuple(int(c * alpha) for c in particle['color'])
                cv2.circle(image, (int(particle['x']), int(particle['y'])), 
                          size, color, -1)

class PowerUp:
    def __init__(self, x, y, type_name):
        self.x = x
        self.y = y
        self.type = type_name
        self.active = True
        self.size = 30
        self.animation = 0
        
    def update(self):
        self.animation += 0.1
        
    def draw(self, image):
        if not self.active:
            return
            
        # Pulsating effect
        pulse = int(5 * math.sin(self.animation * 2))
        size = self.size + pulse
        
        # Different colors for different power-ups
        colors = {
            'speed': (0, 255, 255),  # Cyan
            'big_bat': (255, 0, 255),  # Magenta
            'multi_ball': (255, 255, 0)  # Yellow
        }
        
        color = colors.get(self.type, (255, 255, 255))
        cv2.circle(image, (self.x, self.y), size, color, 3)
        cv2.circle(image, (self.x, self.y), size//2, color, -1)

class Ball:
    def __init__(self, x, y, speed_x, speed_y):
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.size = 25
        self.trail = []
        self.glow = 0
        
    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.glow = (self.glow + 0.2) % (2 * math.pi)
        
        # Add to trail
        self.trail.append((int(self.x + self.size//2), int(self.y + self.size//2)))
        if len(self.trail) > 8:
            self.trail.pop(0)
    
    def draw(self, image):
        # Draw trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail)
            size = int(self.size * alpha * 0.3)
            if size > 0:
                color = (int(100 * alpha), int(150 * alpha), int(255 * alpha))
                cv2.circle(image, (tx, ty), size, color, -1)
        
        # Draw glowing ball
        glow_intensity = int(50 + 30 * math.sin(self.glow))
        ball_color = (255, 255, 255)
        glow_color = (100, 200, 255)
        
        # Outer glow
        cv2.circle(image, (int(self.x + self.size//2), int(self.y + self.size//2)), 
                  self.size + 10, glow_color, -1)
        # Main ball
        cv2.circle(image, (int(self.x + self.size//2), int(self.y + self.size//2)), 
                  self.size, ball_color, -1)
        # Inner highlight
        cv2.circle(image, (int(self.x + self.size//2 - 5), int(self.y + self.size//2 - 5)), 
                  8, (255, 255, 255), -1)

def create_gradient_background(width, height):
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        ratio = i / height
        # Dark blue to purple gradient
        b = int(30 + ratio * 20)  # Blue component
        g = int(10 + ratio * 30)  # Green component  
        r = int(50 + ratio * 100)  # Red component
        gradient[i, :] = [b, g, r]
    return gradient

def add_text_with_glow(image, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      scale=1, color=(255, 255, 255), thickness=2, glow_color=(100, 100, 255)):
    # Draw glow effect
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx != 0 or dy != 0:
                cv2.putText(image, text, (pos[0] + dx, pos[1] + dy), 
                           font, scale, glow_color, thickness + 2)
    
    # Draw main text
    cv2.putText(image, text, pos, font, scale, color, thickness)

def addTextToCenter(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                   color=(255, 255, 255), thickness=2, custom_x=None, custom_y=None):
    text_width, text_height = cv2.getTextSize(text, font, fontScale, thickness)[0]
    
    img_height, img_width = image.shape[:2]
    img_center_x = img_width // 2
    img_center_y = min(boardMaxHeight, img_height - 100) // 2
    
    text_x = img_center_x - text_width // 2
    text_y = img_center_y + text_height // 2
    
    if custom_x is not None:
        text_x = custom_x
    if custom_y is not None:
        text_y = custom_y
    
    add_text_with_glow(image, text, (text_x, text_y), font, fontScale, color, thickness)
    return image

def draw_neon_line(image, pt1, pt2, color, thickness=2):
    # Draw outer glow
    cv2.line(image, pt1, pt2, tuple(c//3 for c in color), thickness + 6)
    # Draw middle glow  
    cv2.line(image, pt1, pt2, tuple(c//2 for c in color), thickness + 3)
    # Draw main line
    cv2.line(image, pt1, pt2, color, thickness)

def updateBoard(image, score_animation=0):
    # Get actual image dimensions
    img_height, img_width = image.shape[:2]
    
    # Create gradient background matching image dimensions
    background = create_gradient_background(img_width, img_height)
    
    # Ensure both images have same dimensions and type
    if image.shape == background.shape:
        image = cv2.addWeighted(image, 0.3, background, 0.7, 0)
    else:
        # Resize background to match image if needed
        background = cv2.resize(background, (img_width, img_height))
        image = cv2.addWeighted(image, 0.3, background, 0.7, 0)
    
    # Draw animated grid pattern
    grid_alpha = int(30 + 10 * math.sin(time.time() * 2))
    actual_board_height = min(boardMaxHeight, img_height)
    for i in range(0, img_width, 40):
        cv2.line(image, (i, 0), (i, actual_board_height), (50, 50, 50), 1)
    for i in range(0, actual_board_height, 40):
        cv2.line(image, (0, i), (img_width, i), (50, 50, 50), 1)
    
    # Draw neon game area borders - use actual image dimensions
    actual_board_width = min(boardWidth, img_width // 10)
    actual_board_height = min(boardMaxHeight, img_height - 100)
    
    draw_neon_line(image, (actual_board_width, 0), (actual_board_width, actual_board_height), (252, 5, 244), 3)
    draw_neon_line(image, (img_width-actual_board_width, 0), (img_width-actual_board_width, actual_board_height), (252, 5, 244), 3)
    draw_neon_line(image, (0, actual_board_height), (img_width, actual_board_height), (252, 5, 244), 3)
    
    # Draw center line with animation
    center_x = img_width // 2
    dash_length = 20
    gap_length = 15
    animation_offset = int(time.time() * 50) % (dash_length + gap_length)
    
    for y in range(0, actual_board_height, dash_length + gap_length):
        y_start = y - animation_offset
        y_end = y_start + dash_length
        if y_start < actual_board_height and y_end > 0:
            draw_neon_line(image, (center_x, max(0, y_start)), 
                          (center_x, min(actual_board_height, y_end)), (100, 255, 255), 2)
    
    # Draw bottom area with gradient
    if img_height > actual_board_height:
        bottom_height = img_height - actual_board_height
        bottom_gradient = np.zeros((bottom_height, img_width, 3), dtype=np.uint8)
        for i in range(bottom_height):
            ratio = i / bottom_height
            bottom_gradient[i, :] = [int(100 * ratio), int(50 * ratio), int(150 * ratio)]
        
        image[actual_board_height:, :] = cv2.addWeighted(
            image[actual_board_height:, :], 0.3, bottom_gradient, 0.7, 0)
    
    # Animated score display
    score_glow = int(100 + 50 * math.sin(score_animation))
    score_color = (255, 255, 255)
    glow_color = (score_glow, score_glow, 255)
    
    # Left score
    add_text_with_glow(image, str(leftPoint), (actual_board_width - 50, actual_board_height + 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 4, score_color, 6, glow_color)
    
    # Right score  
    add_text_with_glow(image, str(rightPoint), (img_width - actual_board_width - 50, actual_board_height + 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 4, score_color, 6, glow_color)
    
    # Game status text
    if not gameStarted:
        addTextToCenter(image, "🖐️ Show Both Hands to Start! 🖐️", 
                       fontScale=1.2, color=(0, 255, 255), custom_y=actual_board_height+50)
        addTextToCenter(image, "Get ready for an epic match!", 
                       fontScale=0.8, color=(255, 255, 0), custom_y=actual_board_height+80)
    else:
        addTextToCenter(image, "☝️ Play with Index Fingers ☝️", 
                       fontScale=1, color=(0, 255, 0), custom_y=actual_board_height+50)
    
    return image

def updateBat(image, idxPos, bat, bat_size_multiplier=1.0):
    current_bat_height = int(batHeight * bat_size_multiplier)
    
    y1 = idxPos[1] - (current_bat_height // 2)
    y1 = max(0, min(y1, boardMaxHeight - current_bat_height))
    y2 = y1 + current_bat_height
    
    if bat == "Left":
        x1 = boardWidth - batWidth
        x2 = boardWidth
        globals()["leftBatPosY"] = [y1, y2]
    elif bat == "Right":
        x1 = FrameWidth - boardWidth
        x2 = x1 + batWidth
        globals()["rightBatPosY"] = [y1, y2]
    
    # Draw bat with neon effect
    bat_color = (230, 11, 26) if bat == "Left" else (26, 230, 11)
    glow_color = tuple(c//2 for c in bat_color)
    
    # Outer glow
    cv2.rectangle(image, (x1-3, y1-3), (x2+3, y2+3), glow_color, -1)
    # Main bat
    cv2.rectangle(image, (x1, y1), (x2, y2), bat_color, -1)
    # Inner highlight
    cv2.rectangle(image, (x1+3, y1+5), (x2-3, y1+15), (255, 255, 255), -1)
    
    # Bat center indicator
    center_x = x1 + batWidth // 2
    center_y = y1 + current_bat_height // 2
    cv2.circle(image, (center_x, center_y), 8, (255, 255, 255), 2)
    cv2.circle(image, (center_x, center_y), 4, bat_color, -1)
    
    # Finger position indicator with trail effect
    cv2.circle(image, (idxPos[0], idxPos[1]), 12, (255, 255, 255), 2)
    cv2.circle(image, (idxPos[0], idxPos[1]), 8, (255, 0, 255), -1)
    
    return image

def check_power_up_collision(ball, power_ups):
    ball_center = (ball.x + ball.size//2, ball.y + ball.size//2)
    
    for power_up in power_ups[:]:
        distance = math.sqrt((ball_center[0] - power_up.x)**2 + (ball_center[1] - power_up.y)**2)
        if distance < (ball.size//2 + power_up.size//2) and power_up.active:
            power_up.active = False
            return power_up.type
    return None

# Enhanced Global Variables
FrameWidth = 960  # Reduced to more standard webcam resolution
FrameHeight = 540  # Reduced to more standard webcam resolution
boardMaxHeight = 420
boardWidth = 80

# Game objects
balls = []
power_ups = []
particle_system = ParticleSystem()

# Enhanced ball properties
ballSize = 25
ballSpeed = 12
speedUpEvery = 10

# Bat properties
batHeight = 120
batWidth = 25
bat_size_multiplier = 1.0
bat_effect_timer = 0

leftBatPosY = [0, 0]
rightBatPosY = [0, 0]

leftPoint = 0
rightPoint = 0

gameStarted = False
gameOver = False
multi_ball_active = False
speed_boost_active = False

startTime = time.time()
last_power_up_spawn = time.time()
score_animation = 0

# Initialize first ball
initial_ball = Ball(
    FrameWidth // 2 - ballSize // 2, 
    boardMaxHeight // 2 - ballSize // 2,
    random.choice([ballSpeed, -ballSpeed]),
    random.choice([ballSpeed, -ballSpeed])
)
balls.append(initial_ball)

# Hand detector
detector = HandDetector(detectionCon=0.6, maxHands=2)

# Video capture with more compatible settings
cap_vid = cv2.VideoCapture(0)
cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
# Remove FPS setting as it might not be supported on all cameras
# cap_vid.set(cv2.CAP_PROP_FPS, 60)

windowName = "🏓 EPIC PING PONG 🏓"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

print("🏓 Epic Ping Pong Game Started!")
print("Features:")
print("• Particle effects and neon graphics")
print("• Power-ups (appear every 15 seconds)")
print("• Multiple balls and enhanced physics") 
print("• Improved AI and smoother gameplay")
print("• Show both hands to start!")

while cap_vid.isOpened():
    ret, frame = cap_vid.read()
    if not ret:
        continue
    
    img = cv2.flip(frame, 1)
    handsT, _ = detector.findHands(img, flipType=False, draw=False)
    
    # Process hands
    hands = []
    for hand in handsT:
        if not any(x["type"] == hand["type"] for x in hands):
            hands.append(hand)
    
    # Start game when both hands are detected
    if len(hands) == 2 and not gameStarted:
        gameStarted = True
        particle_system.add_explosion(FrameWidth//2, boardMaxHeight//2, (0, 255, 0), 25)
    
    # Update animations
    score_animation += 0.1
    if bat_effect_timer > 0:
        bat_effect_timer -= 1
    else:
        bat_size_multiplier = max(1.0, bat_size_multiplier - 0.02)
    
    # Draw board
    img = updateBoard(img, score_animation)
    
    # Handle hand input for bats
    for hand in hands:
        if not gameStarted:
            break
        
        fingers = detector.fingersUp(hand)
        if fingers[1] != 1:  # Index finger must be up
            continue
        
        lmList = hand["lmList"]
        idxPos = lmList[8]
        palmPos = hand["center"]  # Use palm center instead of index finger
        img = updateBat(img, palmPos, hand["type"], bat_size_multiplier)
    
    # Spawn power-ups
    if gameStarted and time.time() - last_power_up_spawn > 15:
        power_up_types = ['speed', 'big_bat', 'multi_ball']
        power_up_type = random.choice(power_up_types)
        x = random.randint(boardWidth + 50, FrameWidth - boardWidth - 50)
        y = random.randint(50, boardMaxHeight - 50)
        power_ups.append(PowerUp(x, y, power_up_type))
        last_power_up_spawn = time.time()
    
    # Update and draw power-ups
    for power_up in power_ups[:]:
        power_up.update()
        power_up.draw(img)
        if not power_up.active:
            power_ups.remove(power_up)
    
    # Update balls
    for ball in balls[:]:
        if not gameStarted:
            break
            
        # Add trail particles
        if random.random() < 0.3:
            particle_system.add_trail(ball.x + ball.size//2, ball.y + ball.size//2, 
                                    (100, 200, 255))
        
        # Check power-up collisions
        power_up_hit = check_power_up_collision(ball, power_ups)
        if power_up_hit:
            particle_system.add_explosion(ball.x + ball.size//2, ball.y + ball.size//2, 
                                        (255, 255, 0), 20)
            if power_up_hit == 'speed':
                ball.speed_x *= 1.5
                ball.speed_y *= 1.5
            elif power_up_hit == 'big_bat':
                bat_size_multiplier = 2.0
                bat_effect_timer = 300  # 5 seconds at 60fps
            elif power_up_hit == 'multi_ball':
                # Add two more balls
                for _ in range(2):
                    new_ball = Ball(ball.x, ball.y, 
                                   random.choice([ballSpeed, -ballSpeed]),
                                   random.choice([ballSpeed, -ballSpeed]))
                    balls.append(new_ball)
        
        # Ball collision with bats
        ball_center_y = ball.y + ball.size // 2
        
        # Left bat collision
        if (boardWidth - 20 < ball.x < boardWidth and 
            leftBatPosY[0] < ball_center_y < leftBatPosY[1]):
            ball.speed_x = abs(ball.speed_x)  # Ensure ball goes right
            ball.x = boardWidth + 5
            particle_system.add_explosion(ball.x, ball.y + ball.size//2, (230, 11, 26), 10)
        
        # Right bat collision  
        elif (FrameWidth - boardWidth < ball.x + ball.size < FrameWidth - boardWidth + 20 and
              rightBatPosY[0] < ball_center_y < rightBatPosY[1]):
            ball.speed_x = -abs(ball.speed_x)  # Ensure ball goes left
            ball.x = FrameWidth - boardWidth - ball.size - 5
            particle_system.add_explosion(ball.x + ball.size, ball.y + ball.size//2, (26, 230, 11), 10)
        
        # Ball out of bounds
        elif ball.x + ball.size <= 0:
            rightPoint += 1
            particle_system.add_explosion(ball.x, ball.y, (26, 230, 11), 25)
            balls.remove(ball)
            if not balls:  # If no balls left, end game
                gameOver = True
        elif ball.x >= FrameWidth:
            leftPoint += 1
            particle_system.add_explosion(ball.x, ball.y, (230, 11, 26), 25)
            balls.remove(ball)
            if not balls:  # If no balls left, end game
                gameOver = True
        
        # Wall collisions (top/bottom)
        if ball.y <= 0 or ball.y + ball.size >= boardMaxHeight:
            ball.speed_y = -ball.speed_y
            particle_system.add_explosion(ball.x + ball.size//2, ball.y + ball.size//2, 
                                        (255, 255, 255), 8)
        
        ball.update()
        ball.draw(img)
    
    # Handle game over
    if gameOver:
        gameStarted = False
        gameOver = False
        
        # Reset to single ball
        balls.clear()
        new_ball = Ball(
            FrameWidth // 2 - ballSize // 2,
            boardMaxHeight // 2 - ballSize // 2,
            random.choice([ballSpeed, -ballSpeed]),
            random.choice([ballSpeed, -ballSpeed])
        )
        balls.append(new_ball)
        
        # Game over screen
        game_over_text = f"🏆 POINT SCORED! 🏆"
        winner_text = f"Left: {leftPoint} | Right: {rightPoint}"
        
        addTextToCenter(img, game_over_text, fontScale=2, color=(255, 255, 0), thickness=4)
        addTextToCenter(img, winner_text, fontScale=1.5, color=(255, 255, 255), 
                       thickness=3, custom_y=img.shape[0]//2 + 60)
        
        cv2.imshow(windowName, img)
        cv2.waitKey(2000)
    
    # Reset bat positions
    leftBatPosY = [0, 0] 
    rightBatPosY = [0, 0]
    
    # Speed increase over time
    if gameStarted and time.time() - startTime >= speedUpEvery:
        for ball in balls:
            ball.speed_x += 1 if ball.speed_x > 0 else -1
            ball.speed_y += 1 if ball.speed_y > 0 else -1
        startTime = time.time()
        particle_system.add_explosion(FrameWidth//2, 50, (255, 255, 0), 15)
    
    # Update and draw particle system
    particle_system.update_and_draw(img)
    
    # Show frame
    cv2.imshow(windowName, img)
    
    # Exit conditions
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        break

cap_vid.release()
cv2.destroyAllWindows()
print("Thanks for playing Epic Ping Pong! 🏓")