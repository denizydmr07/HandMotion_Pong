import threading
from enum import Enum
import random
import queue
import cv2
import pygame
import mediapipe as mp
import pyautogui
import numpy as np

class Direction(Enum):
    UP = 1
    STAY = 0
    DOWN = 2

# Game settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
BALL_SIZE = 10
PADDLE_SPEED = 8
BALL_SPEED_X = 8
BALL_SPEED_Y = 8
WALL_WIDTH = 10
SCORE_FONT_SIZE = 36

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0 , 255)
BLACK = (0, 0, 0)

# global variable to stop the camera thread
STOP_CAMERA = False
# lock for the camera thread
lock = threading.Lock()


def draw_objects(screen, ball, paddles,  bot_score, player_score):
    # drawing ball
    pygame.draw.rect(screen, WHITE, ball)
    # drawing paddles
    pygame.draw.rect(screen, BLUE, paddles[0])
    pygame.draw.rect(screen, RED, paddles[1])
    # drawing a circle and line in the middle
    pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), 140, 1)
    pygame.draw.line(screen, WHITE, (SCREEN_WIDTH // 2, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT), 1)
    # drawing scores
    score_display = pygame.font.Font(None, SCORE_FONT_SIZE).render(f"Bot: {bot_score}", True, WHITE)
    screen.blit(score_display, (SCREEN_WIDTH - score_display.get_width() - 10, 10))
    score_display = pygame.font.Font(None, SCORE_FONT_SIZE).render(f"Player: {player_score}", True, WHITE)
    screen.blit(score_display, (10, 10))
    


def move_ball(ball, ball_speed):
    # moving ball according to ball_speed
    ball.x += ball_speed[0]
    ball.y += ball_speed[1]

def move_paddle(paddle, direction):
    # moving paddle according to direction
    if direction == Direction.UP and paddle.top > 0:
        paddle.y -= PADDLE_SPEED
    elif direction == Direction.DOWN and paddle.bottom < SCREEN_HEIGHT:
        paddle.y += PADDLE_SPEED

def decide_bot_direction(paddle, ball):
    # deciding the direction of bot_paddle according to relative positions
    if ball[1] < paddle.top:
        return Direction.UP
    elif ball[1] > paddle.bottom:
        return Direction.DOWN
    else:
        return Direction.STAY

def reset_ball(ball):
    # reseting ball position
    ball.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    return ball

def process_input(hand_landmarks):
    # return STAY if no hand detected in the frame
    if not hand_landmarks:
        return Direction.STAY
    
    # we just need the y coordinate of tip and mcp
    tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    mcp_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y
    
    # deciding the direction according to relative positions 
    # of index finger tip and mcp
    if tip_y - mcp_y < -0.15:
        return Direction.UP
    elif tip_y - mcp_y > 0.15:
        return Direction.DOWN
    else:
        return Direction.STAY
    
def decide_input(direction_queue,):
    # initialize mediapipe components
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
    
    # initialize camera
    cap = cv2.VideoCapture(0)
    
    # main loop
    while True:
        # get image from camera
        success, image = cap.read()

        # if no image, continue
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # flip image
        image = cv2.flip(image, 1)

        # convert image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # process image
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # convert image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        _hand_landmarks = []
        # draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                _hand_landmarks = hand_landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # process hand landmarks and put the result in the queue
        direction_queue.put(process_input(hand_landmarks=_hand_landmarks))

        # show image
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        # if STOP_CAMERA is True, break the loop
        with lock:
            if STOP_CAMERA:
                break


def main(direction_queue,):
    # initialize pygame
    pygame.init()

    # initialize screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("1-Player Pong")

    # initialize clock
    clock = pygame.time.Clock()

    # initialize ball and ball_speed
    ball = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BALL_SIZE, BALL_SIZE)
    ball = reset_ball(ball)
    ball_speed = [0,0]

    # initialize paddles and paddles list
    bot_paddle = pygame.Rect(SCREEN_WIDTH - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
    player_paddle = pygame.Rect(0, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
    paddles = [player_paddle,bot_paddle]

    # initialize scores
    bot_score = 0
    player_score = 0

    # initialize collision count to adjust ball speed
    collision_count = 1

    # initialize previous direction to smooth paddle movement
    prev_direction = Direction.STAY

    # initialize count for initializing ball speed
    count_for_initializing_ball_speed = 0

    # main loop
    while True:
        # check if there is any input
        for event in pygame.event.get():
            # if input is quit, quit the game
            if event.type == pygame.QUIT:
                pygame.quit()
                # set STOP_CAMERA to True to stop the camera
                global STOP_CAMERA
                STOP_CAMERA = True
                quit()
            
        # check if collision is mulyiply of 5, if so, increse speed
        if collision_count % 5 == 0:
            ball_speed[0] = ball_speed[0] * 1.1
            ball_speed[1] = ball_speed[1] * 1.1
            collision_count += 1

        # Move the ball
        move_ball(ball, ball_speed)

        # move player paddle

        # since game has more fps than camera, we need to check if there is any input
        # if there is no input, we will use the previous input to smooth the paddle movement
        if not direction_queue.empty():
            direction = direction_queue.get()
            prev_direction = direction
        else:
            direction = prev_direction
        move_paddle(player_paddle, direction)

        # move bot paddle
        direction = decide_bot_direction(bot_paddle, ball)
        move_paddle(bot_paddle, direction)

        # Check collision with walls
        if ball.centery <= 0 or ball.centery >= SCREEN_HEIGHT:
            ball_speed[1] = -ball_speed[1]

        # Check collision with paddles
        for paddle in paddles:
            if ball.colliderect(paddle):
                collision_count += 1
                ball_speed[0] = -ball_speed[0]


        # Check if the ball is out of bounds
        if ball.centerx <= 0:
            # if ball is out of bounds, increase bot score and reset the ball
            # and ball speed
            bot_score += 1
            ball = reset_ball(ball)
            ball_speed = [0, 0]
            count_for_initializing_ball_speed = 0
        if ball.centerx >= SCREEN_WIDTH:
            # if ball is out of bounds, increase player score and reset the ball
            # and ball speed
            player_score += 1
            ball = reset_ball(ball)
            ball_speed = [0, 0]
            count_for_initializing_ball_speed = 0

        # Clear the screen and draw objects
        screen.fill(BLACK)
        draw_objects(screen, ball, paddles, bot_score, player_score)

        # Update the screen
        pygame.display.flip()
        clock.tick(60)

        # if ball is resetted, wait for 0.5 second and then initialize the ball speed
        if count_for_initializing_ball_speed == 30:
            ball_speed = [random.choice((BALL_SPEED_X, -BALL_SPEED_X)), random.choice((BALL_SPEED_Y, -BALL_SPEED_Y))]
            count_for_initializing_ball_speed += 1

        # if ball is resetted, increase the count
        if ball_speed == [0, 0]:
            count_for_initializing_ball_speed += 1

if __name__ == "__main__":
        # initialize queue for communication between threads
    direction_queue = queue.Queue()

    # initialize threads
    camera_thread = threading.Thread(target=decide_input, args=(direction_queue,))
    game_thread = threading.Thread(target=main, args=(direction_queue,))

    # start threads
    game_thread.start()
    camera_thread.start()

    
    # wait for threads to finish
    game_thread.join()
    camera_thread.join()



    print("Done")
