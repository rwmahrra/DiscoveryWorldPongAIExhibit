import numpy as np
import cv2
import math
import keyboard
from utils import normalize_states
import matplotlib.pyplot as plt
from random import choice, randint


class Pong:
    PADDING = 10
    MAX_SCORE = 21
    WIDTH = 160
    HEIGHT = 192
    SPEEDUP = 3
    ACTIONS = ["UP", "DOWN"]
    GRACE_PADDLE = 4

    @staticmethod
    def read_key(up, down):
        if keyboard.is_pressed(up):
            return "UP"
        elif keyboard.is_pressed(down):
            return "DOWN"
        else:
            return "NONE"

    @staticmethod
    def random_action():
        return choice(Pong.ACTIONS)

    class Paddle:
        EDGE_BUFFER = 10

        def __init__(self, side):
            self.side = side
            self.x = 0
            self.y = int(Pong.HEIGHT / 2)
            self.w = 3
            self.h = 15
            self.speed = 2 * Pong.SPEEDUP
            if side == "left":
                self.x = Pong.PADDING
            elif side == "right":
                self.x = Pong.WIDTH - Pong.PADDING

        def reset(self):
            self.x = 0
            self.y = int(Pong.HEIGHT / 2)
            self.w = 3
            self.h = 15
            self.speed = 2 * Pong.SPEEDUP
            if self.side == "left":
                self.x = Pong.PADDING
            elif self.side == "right":
                self.x = Pong.WIDTH - Pong.PADDING

        def up(self):
            self.y -= self.speed
            if self.y < Pong.Paddle.EDGE_BUFFER:
                self.y = Pong.Paddle.EDGE_BUFFER

        def down(self):
            self.y += self.speed
            max = Pong.HEIGHT - Pong.Paddle.EDGE_BUFFER
            if self.y > max:
                self.y = max

        def handle_action(self, action):
            if action == "UP":
                self.up()
            elif action == "DOWN":
                self.down()

    class Ball:  # 0, 30, -45, 225
        BOUNCE_ANGLES = [0, 60, 45, 30, -30, -45, -60]

        def __init__(self):
            self.x = math.floor(Pong.WIDTH / 2)
            self.y = math.floor(Pong.HEIGHT / 2)
            self.speed = 1 * Pong.SPEEDUP
            self.velocity = (0, 0)
            self.w = 4
            self.right = None
            self.h = 4

        def reset(self):
            self.right = None
            self.x = math.floor(Pong.WIDTH / 2)
            self.y = math.floor(Pong.HEIGHT / 2)
            self.speed = 1 * Pong.SPEEDUP
            self.velocity = (0, 0)
            self.w = 4
            self.h = 4

        def get_vector(self, deg, scale):
            rad = math.pi * deg / 180
            return scale * math.cos(rad), scale * math.sin(rad)

        def reverse_vector(self, vector):
            x, y = vector
            return -x, -y

        def bounce(self, x=False, y=False):
            xv, yv = self.velocity
            if x:
                xv = -xv
            if y:
                yv = -yv
            self.velocity = xv, yv

        def bounce_angle(self, pos):
            segment = int(round(pos * 3))
            angle = Pong.Ball.BOUNCE_ANGLES[segment]
            velocity = self.get_vector(angle, self.speed)
            if self.right:
                velocity = self.get_vector(-angle, self.speed)
                velocity = self.reverse_vector(velocity)
            self.velocity = velocity
            self.speed += 0.1 * Pong.SPEEDUP

        def update(self):
            if self.velocity == (0, 0):
                angle = choice(Pong.Ball.BOUNCE_ANGLES)
                self.right = True
                if randint(0, 1) == 1:
                    angle += 180
                    self.right = False
                self.velocity = self.get_vector(angle, self.speed)
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            if self.y > Pong.HEIGHT:
                self.y = Pong.HEIGHT
                self.bounce(y=True)
            if self.y < 0:
                self.y = 0
                self.bounce(y=True)

    def __init__(self):
        # Holds last preprocessed, non-diffed state
        self.last_processed_screen = None
        # Holds last raw screen pixels for rendering
        self.last_screen = None
        # Holds last state passed to the agents
        self.last_state = None

        self.score_left = 0
        self.score_right = 0
        self.left = Pong.Paddle("left")
        self.right = Pong.Paddle("right")
        self.ball = Pong.Ball()

    def reset(self):
        self.last_processed_screen = None
        self.score_left = 0
        self.score_right = 0
        self.left.reset()
        self.right.reset()
        self.ball.reset()
        screen = self.render()
        self.last_screen = screen
        state = self.preprocess_screen(screen)
        self.last_processed_screen = state
        self.last_state = state
        return state

    def get_score(self):
        return self.score_left, self.score_right

    def get_bot_data(self, left=False, right=False):
        if left:
            return self.left, self.ball
        if right:
            return self.right, self.ball

    def check_collision(self, ball, paddle):
        if abs(ball.x - paddle.x) < (paddle.w + 2) / 2:
            hit_position = ball.y - paddle.y
            if abs(hit_position) < (paddle.h + Pong.GRACE_PADDLE) / 2:
                return True, hit_position / ((paddle.h + Pong.GRACE_PADDLE) / 2)
        return False, 0


    def step(self, left_action, right_action):
        reward_l = 0
        reward_r = 0
        self.left.handle_action(left_action)
        self.right.handle_action(right_action)

        collide_left, pos = self.check_collision(self.ball, self.left)
        if collide_left and not self.ball.right:
            self.ball.bounce_angle(pos)
            self.ball.right = True
        collide_right, pos = self.check_collision(self.ball, self.right)
        if collide_right and self.ball.right:
            self.ball.bounce_angle(pos)
            self.ball.right = False

        if self.ball.x < 0:
            self.score_right += 1
            reward_l = -1.0
            reward_r = 1.0
            self.ball.reset()
            self.left.reset()
            self.right.reset()
        elif self.ball.x > Pong.WIDTH:
            self.score_left += 1
            reward_l = 1.0
            reward_r = -1.0
            self.ball.reset()
            self.left.reset()
            self.right.reset()
        self.ball.update()
        done = False
        if self.score_right >= Pong.MAX_SCORE or self.score_left >= Pong.MAX_SCORE:
            done = True

        screen = self.render()
        self.last_screen = screen
        processed_screen = self.preprocess_screen(screen)
        state = processed_screen - self.last_processed_screen
        self.last_processed_screen = processed_screen
        self.last_state = state
        return state, (reward_l, reward_r), done

    def preprocess_screen(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        h, w = state.shape
        state = cv2.resize(state, (w//2, h//2))
        state[state < 250] = 0

        return state

    def show(self, scale=1,  duration=1):
        l, r = self.get_score()
        to_render = cv2.resize(self.last_screen, (int(Pong.WIDTH * scale), int(Pong.HEIGHT * scale)))
        cv2.imshow(f"Pong", to_render)
        cv2.waitKey(duration)

    def show_state(self, scale=1, duration=1):
        l, r = self.get_score()
        h, w = self.last_state.shape
        to_render = cv2.resize(self.last_state, (int(w * scale), int(h * scale)))
        cv2.imshow(f"Pong State", normalize_states(to_render))
        cv2.waitKey(duration)

    def draw_rect(self, screen, x, y, w, h, color):
        screen[y:y+h+1, x:x+w+1] = color

    def render(self):
        screen = np.zeros((Pong.HEIGHT, Pong.WIDTH, 3), dtype=np.float32)
        screen[:, :] = (0.0, 0.4, 0.7)

        # Draw middle grid lines
        #self.draw_rect(screen, 0, int(Pong.HEIGHT/2), int(Pong.WIDTH), 1, 255)
        #self.draw_rect(screen, int(Pong.WIDTH/2), 0, 1, int(Pong.HEIGHT), 255)

        self.draw_rect(screen, int(self.left.x - int(self.left.w / 2)), int(self.left.y - int(self.left.h / 2)),
                       self.left.w, self.left.h, 255)
        self.draw_rect(screen, int(self.right.x - int(self.right.w / 2)), int(self.right.y - int(self.right.h / 2)),
                       self.right.w, self.right.h, 255)
        self.draw_rect(screen, int(self.ball.x - int(self.ball.w / 2)), int(self.ball.y - int(self.ball.h / 2)),
                       self.ball.w, self.ball.h, 255)
        return screen


