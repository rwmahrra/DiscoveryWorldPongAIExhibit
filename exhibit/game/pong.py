import numpy as np
import cv2
import math
import keyboard
import time
from random import choice, randint
from exhibit.shared.config import Config

if Config.ENABLE_AUDIO:
    import pygame.mixer


class Pong:
    """
    This class captures all of the game logic for Pong.
    It was used instead of OpenAI Gym or various other publicly available alternatives
    in order to allow for complete flexibility.
    """
    VOLLEY_SPEEDUP = Config.VOLLEY_SPEEDUP
    PADDING = Config.PADDING  # Distance between screen edge and player paddles (px)
    MAX_SCORE = Config.MAX_SCORE  # Points one side must win to finish game
    WIDTH = Config.WIDTH  # Game window width (px)
    HEIGHT = Config.HEIGHT  # Game window height (px)
    SPEEDUP = Config.SPEEDUP  # Flat multiplier to game movement speeds
    ACTIONS = Config.ACTIONS

    # Cache game sounds. Loaded on first instance's init
    sounds = None

    @staticmethod
    def read_key(up, down):
        """
        Converts keyboard state to internal action state
        :param up: key code for "up" control
        :param down: key code for "down" control
        :return: Action code: 0 for up, 1 for down, 2 for nothing
        """
        if keyboard.is_pressed(up):
            return 0
        elif keyboard.is_pressed(down):
            return 1
        else:
            return 2

    @staticmethod
    def random_action():
        return choice(Pong.ACTIONS)

    @staticmethod
    def load_sounds():
        if Config.ENABLE_AUDIO:
            pygame.mixer.init()
            Pong.sounds = {}
            Pong.sounds["return"] = pygame.mixer.Sound(Config.AUDIO_DIR + "return.ogg")
            Pong.sounds["score"] = pygame.mixer.Sound(Config.AUDIO_DIR + "score.ogg")
            Pong.sounds["bounce"] = pygame.mixer.Sound(Config.AUDIO_DIR + "bounce.ogg")

    @staticmethod
    def play_sound(sound):
        if Config.ENABLE_AUDIO and Pong.sounds is not None and sound in Pong.sounds:
            try:
                playback = Pong.sounds[sound].play()
            except Exception as e:
                print(e)


    class Paddle:
        EDGE_BUFFER = 0  # Pixel distance from screen edges that paddle is allowed to reach
        HEIGHT = Config.PADDLE_HEIGHT  # px
        WIDTH = Config.PADDLE_WIDTH  # px
        SPEED = 3  # base speed (in px/tick)

        def __init__(self, side):
            self.side = side
            self.x = int(Pong.WIDTH / 2)
            self.y = 0
            self.w = self.WIDTH
            self.h = self.HEIGHT
            self.velocity = [0, 0]
            self.speed = self.SPEED * Pong.SPEEDUP
            if self.side == "top":
                self.y = Config.TOP_PADDLE_Y
            elif self.side == "bottom":
                self.y = Config.BOTTOM_PADDLE_Y

        def reset(self, hit_practice=False):
            """
            Clean game state to initial configuration.
            Should be called before a new game.
            hit_practice: Spawns paddle at random valid position each round, for training generalization
            """
            self.x = int(Pong.WIDTH / 2)
            self.y = 0
            if hit_practice:
                # Generate all positions the paddle could end up in, then pick one.
                # This allows us to better generalize to all paddle positions in training.
                # (yes, it's redundant to do this on every reset, but I don't want to add another class field
                # just for this test setup)
                base_start = int(Pong.WIDTH / 2)
                valid_starts = [base_start]
                start = base_start + self.speed
                while start < Pong.HEIGHT - Pong.Paddle.EDGE_BUFFER:
                    valid_starts.append(start)
                    start += self.speed
                start = base_start - self.speed
                while start > Pong.Paddle.EDGE_BUFFER:
                    valid_starts.append(start)
                    start -= self.speed
                self.y = choice(valid_starts)

            self.w = self.WIDTH
            self.h = self.HEIGHT
            self.speed = self.SPEED * Pong.SPEEDUP
            if self.side == "top":
                self.y = Config.TOP_PADDLE_Y
            elif self.side == "bottom":
                self.y = Config.BOTTOM_PADDLE_Y

        def left(self):
            """
            Handle up action
            """
            self.velocity[0] -= self.speed

        def right(self):
            """
            Handle down action
            """
            self.velocity[0] += self.speed

        def update(self):
            """
            Run game tick housekeeping logic
            """
            # First blindly increment position by velocity
            self.x += self.velocity[0]
            self.y += self.velocity[1]  # Y should never actually change, but it feels right to include it
            self.velocity = [0, 0]  # We don't actually want velocity to persist from tick to tick, so deplete it all

            # Then back up position if we cross the screen border
            max = Pong.WIDTH - Pong.Paddle.EDGE_BUFFER
            if self.x > max:
                self.x = max
            if self.x < Pong.Paddle.EDGE_BUFFER:
                self.x = Pong.Paddle.EDGE_BUFFER

        def handle_action(self, action):
            """
            Parse action and modify state accordingly
            :param action: String representation of action ("UP", "DOWN", "NONE")
            :return:
            """
            if action == "LEFT":
                self.left()
            elif action == "RIGHT":
                self.right()
            elif action == "NONE":
                pass

    class Ball:
        DIAMETER = Config.BALL_DIAMETER
        SPEED = 2
        BOUNCE_ANGLES = [-90, -30, -45, -60, -120, -135, -150]  # True to original Atari Pong
        START_ANGLES = [90]

        def spawn_hit_practice(self):
            """
            Set initial position on the left side with a random trajectory towards
            the right side. Useful for training a right model to hit from various
            trajectories without coupling to an opponent strategy.
            """
            self.x = randint(0, Pong.WIDTH)
            self.y = 5
            self.speed = self.SPEED * Pong.SPEEDUP
            self.velocity = self.get_vector(Pong.Ball.SPEED + (Pong.VOLLEY_SPEEDUP * choice(list(range(12)))), choice(Pong.Ball.BOUNCE_ANGLES))
            self.w = self.DIAMETER
            self.up = True
            self.h = self.DIAMETER

        def __init__(self, hit_practice=False):
            """
            Set basic state.
            :param hit_practice: Overrides ball reset to use "spawn_hit_practice"
                                 for alternative training mode. See method for details
            """
            self.hit_practice = hit_practice
            if self.hit_practice:
                self.spawn_hit_practice()
            else:
                self.x = round((Pong.WIDTH / 2) - 1)
                self.y = round((Pong.HEIGHT / 2) - 1)
                self.speed = self.SPEED * Pong.SPEEDUP
                self.velocity = (0, 0)
                self.w = self.DIAMETER
                self.up = None
                self.h = self.DIAMETER

        def reset(self):
            """
            Reset to initial state before starting a new game.
            """
            if self.hit_practice:
                self.spawn_hit_practice()
            else:
                self.up = None
                self.x = round(Pong.WIDTH / 2)
                self.y = round(Pong.HEIGHT / 2)
                self.speed = self.SPEED * Pong.SPEEDUP
                self.velocity = (0, 0)
                self.w = self.DIAMETER
                self.h = self.DIAMETER

        def get_vector(self, deg, scale):
            """
            Simple trig helper to build a vector from angle and magnitude
            :param deg: unit circle degrees
            :param scale: desired magnitude
            :return: float tuple representing vector
            """
            rad = math.pi * deg / 180
            return scale * math.cos(rad), scale * math.sin(rad)

        def reverse_vector(self, vector):
            """
            Simple helper to mirror vector
            :param vector: float tuple to mirror
            :return: mirrored float tuple
            """
            x, y = vector
            return -x, -y

        def bounce(self, x=False, y=False):
            """
            Selectively mirrors a vector dimension
            :param x: should x dimension bounce?
            :param y: should y dimension bounce?
            :return: selectively mirrored vector
            """
            xv, yv = self.velocity
            if x:
                xv = -xv
            if y:
                yv = -yv
            Pong.play_sound("bounce")
            self.velocity = xv, yv

        def bounce_angle(self, pos):
            """
            Implement Atari Pong logic to bounce based on which part of the paddle makes contact with the ball.
            This divides the paddle into eight equal regions from top to bottom.
            The center two regions bounce the ball straight ahead. Each successive region above and below the center
            bounces the ball at a steeper angle: 30 degrees, then 45, then 60, in the respective direction from center.
            :param pos: paddle-relative position
            :return:
            """
            # Translate segment to fraction of the paddle that is impacted
            segment = int(round(pos * 3))

            # In edge cases where the ball is barely grazing the edge of the paddle, don't wrap around to the other side
            segment = min(segment, 3)
            segment = max(segment, -3)

            angle = Pong.Ball.BOUNCE_ANGLES[segment]
            velocity = self.get_vector(angle, self.speed)
            if self.up:
                velocity = self.get_vector(-angle, self.speed)
                #velocity = self.reverse_vector(velocity)
            self.velocity = velocity
            self.speed += Pong.VOLLEY_SPEEDUP * Pong.SPEEDUP

        def update(self):
            """
            Run game tick housekeeping logic
            """
            if self.velocity == (0, 0):
                angle = choice(Pong.Ball.START_ANGLES)
                self.up = False
                if randint(0, 1) == 1:
                    angle += 180
                    self.up = True
                self.velocity = self.get_vector(angle, self.speed)
            self.x += self.velocity[0]
            self.y += self.velocity[1]

            if self.x > Pong.WIDTH:
                self.x = Pong.WIDTH
                self.bounce(x=True)
            if self.x < 0:
                self.x = 0
                self.bounce(x=True)

    def __init__(self, hit_practice=False):
        """
        Initialize basic game state
        :param hit_practice: Trigger training mode with a single paddle and randomly spawned balls
                             See the Ball class's hit_practice method.
        """
        if Pong.sounds == None:
            Pong.load_sounds()

        # Holds last raw screen pixels for rendering
        self.last_screen = None
        self.hit_practice = hit_practice
        self.score_bottom = 0
        self.score_top = 0
        self.bottom = Pong.Paddle("bottom") if not self.hit_practice else None
        self.top = Pong.Paddle("top")
        self.ball = Pong.Ball(hit_practice=hit_practice)
        self.frames = 0

    def reset(self):
        """
        Reset game state
        """
        self.score_bottom = 0
        self.score_top = 0
        if not self.hit_practice: self.bottom.reset()
        self.top.reset()
        self.ball.reset()
        screen = self.render()
        self.last_screen = screen
        return screen

    def get_score(self):
        """
        Fetch score tuple at the current frame
        :return: integer tuple: (left score, right score)
        """
        return self.score_bottom, self.score_top

    def get_bot_data(self, bottom=False, top=False):
        """
        Returns internal objects for the hard-coded opponent bot with perfect game state knowledge
        :param bottom: hard-coded bot operates bottom paddle
        :param top: hard-coded bot operates top paddle
        :return: Bot's paddle object and the ball object, used to directly calculate optimal move from object positions
        """
        if bottom:
            return self.bottom, self.ball
        if top:
            return self.top, self.ball

    def check_collision(self, ball, paddle):
        """
        Implement simple rectangle-rectangle collision math
        :param ball: Ball object to check
        :param paddle: Paddle object to check
        :return: Tuple of boolean indicating collision and float indicating collision position relative to paddle
        """

        # In retrospect, I regret making positions central vs in a corner - it adds a lot of ugly half translations
        ball_r = ball.DIAMETER / 2
        paddle_half_w = paddle.w / 2
        paddle_half_h = paddle.h / 2

        next_ball_y = ball.y + ball.velocity[1]
        crosses_y_left = (next_ball_y - ball_r <= paddle.y + paddle_half_h) and (next_ball_y - ball_r >= paddle.y - paddle_half_h)
        crosses_y_right = (next_ball_y + ball_r <= paddle.y + paddle_half_h) and (next_ball_y + ball_r >= paddle.y - paddle_half_h)

        intersects_y_left = (ball.y + ball_r <= paddle.y - paddle_half_h) and (next_ball_y + ball_r >= paddle.y - paddle_half_h)
        intersects_y_right = (ball.y - ball_r >= paddle.y + paddle_half_h) and (next_ball_y - ball_r <= paddle.y + paddle_half_h)

        if crosses_y_left or crosses_y_right or intersects_y_left or intersects_y_right:
            next_ball_x = ball.x + ball.velocity[0]
            paddle_left = min(paddle.x - paddle_half_w, paddle.x - paddle_half_w + paddle.velocity[0])
            paddle_right = max(paddle.x + paddle_half_w, paddle.x + paddle_half_w + paddle.velocity[0])
            collide_x_right = (next_ball_x + ball_r <= paddle_right) and (next_ball_x + ball_r >= paddle_left)
            collide_x_bottom = (next_ball_x - ball_r <= paddle_right) and (next_ball_x - ball_r >= paddle_left)
            if collide_x_right or collide_x_bottom:
                return True, (ball.x - paddle.x) / (paddle.w / 2)

        return False, 0

    def step_hit_practice(self, top_action, frames=3):
        """
        Game tick if running hit practice
        :param top_action: Action from top agent
        :param frames: Frames to run before the next action is accepted
        :return: Tuple containing:
                 (screen state,
                 (left points scored this action, right points scored this action),
                 boolean indicating if game is over)
        """
        reward_l = 0
        reward_r = 0
        done = False
        for i in range(frames):
            if not done:
                self.top.handle_action(top_action)

                collide_right, pos = self.check_collision(self.ball, self.top)
                if collide_right and self.ball.up:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.up = False

                if self.ball.y > Pong.HEIGHT:
                    Pong.play_sound("score")
                    self.score_top += 1
                    reward_l -= 1.0
                    reward_r += 1.0
                    self.ball.reset()
                    self.top.reset(hit_practice=True)
                elif self.ball.y < 0:
                    Pong.play_sound("score")
                    self.score_bottom += 1
                    reward_l += 1.0
                    reward_r -= 1.0
                    self.ball.reset()
                    self.top.reset(hit_practice=True)
                self.ball.update()
                self.top.update()
                done = False
                if self.score_top >= Pong.MAX_SCORE or self.score_bottom >= Pong.MAX_SCORE:
                    done = True
        screen = self.render()
        #self.show(self.render(), 3)
        return screen, (reward_l, reward_r), done

    def step(self, bottom_action, top_action, frames=3):
        """
        Game tick housekeeping
        :param bottom_action: Action from bottom agent
        :param top_action: Action from top agent
        :param frames: Frames to run before the next action is accepted
        :return: Tuple containing:
                 (screen state,
                 (left points scored this action, right points scored this action),
                 boolean indicating if game is over)
        """
        if self.hit_practice:
            return self.step_hit_practice(top_action, frames=frames)
        reward_l = 0
        reward_r = 0
        done = False
        for i in range(frames):
            if not done:
                self.bottom.handle_action(bottom_action)
                self.top.handle_action(top_action)

                collide_bottom, pos = self.check_collision(self.ball, self.bottom)
                if collide_bottom and not self.ball.up:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.up = True
                collide_top, pos = self.check_collision(self.ball, self.top)
                if collide_top and self.ball.up:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.up = False

                if self.ball.y > Pong.HEIGHT:
                    Pong.play_sound("score")
                    self.score_top += 1
                    reward_l -= 1.0
                    reward_r += 1.0
                    self.ball.reset()
                    self.bottom.reset()
                    self.top.reset()
                elif self.ball.y < 0:
                    Pong.play_sound("score")
                    self.score_bottom += 1
                    reward_l += 1.0
                    reward_r -= 1.0
                    self.ball.reset()
                    self.bottom.reset()
                    self.top.reset()
                self.bottom.update()
                self.top.update()
                self.ball.update()
                done = False
                if self.score_top >= Pong.MAX_SCORE or self.score_bottom >= Pong.MAX_SCORE:
                    done = True

            screen = self.render()
            self.last_screen = screen
            self.last_frame_time = time.time()

        self.last_screen = screen
        self.show(self.render(), 3)

        self.frames += 1
        return screen, (reward_l, reward_r), done

    def show(self, screen, scale=1,  duration=1):
        """
        Render last game frame through OpenCV
        :param scale: Multiplier to scale up/scale down rendered frame
        :param duration: Length of time to show frame and block thread (set to 0 to require keystroke to continue)
        :return:
        """
        l, r = self.get_score()
        to_render = cv2.resize(screen, (int(Pong.WIDTH * scale), int(Pong.HEIGHT * scale)))
        cv2.imshow(f"Pong", to_render/255)
        cv2.waitKey(duration)

    def get_packet_info(self):
        """
        Return all info necessary for regular update messages
        :return: Tuple representing ((puck_x, puck_y), paddle1_y, paddle2_y, paddle1_score, paddle2_score, game_frame))
        """
        return (self.ball.x, self.ball.y), self.bottom.y, self.top.y, self.score_bottom, self.score_top,  self.frames

    def get_screen(self):
        """
        Get last rendered screen. For exhibit to hook into.
        :return: np array of RGB pixel values
        """
        return self.last_screen

    def draw_rect(self, screen, x, y, w, h, color):
        """
        Utility to draw a rectangle on the screen state ndarray
        :param screen: ndarray representing the screen
        :param x: leftmost x coordinate
        :param y: Topmost y coordinate
        :param w: width (px)
        :param h: height (px)
        :param color: RGB int tuple
        :return:
        """
        screen[max(y,0):y+h, max(x,0):x+w] = color

    def render(self):
        """
        Render the current game pixel state by hand in an ndarray
        :return: ndarray of RGB screen pixels
        """
        screen = np.zeros((Pong.HEIGHT, Pong.WIDTH, 3), dtype=np.float32)
        screen[:, :] = (140, 60, 0) # BGR for a deep blue

        # Draw middle grid lines
        self.draw_rect(screen, 0, round(Pong.HEIGHT/2 - 1), round(Pong.WIDTH), 1, 255)
        #self.draw_rect(screen, int(Pong.WIDTH/2), 0, 1, int(Pong.HEIGHT), 255)

        if not self.hit_practice:
            self.draw_rect(screen, round(self.bottom.x - round(self.bottom.w / 2)), round(self.bottom.y - round(self.bottom.h / 2)),
                           self.bottom.w, self.bottom.h, 255)
        self.draw_rect(screen, round(self.top.x - round(self.top.w / 2)), round(self.top.y - round(self.top.h / 2)),
                       self.top.w, self.top.h, 255)
        self.draw_rect(screen, round(self.ball.x - round(self.ball.w / 2)), round(self.ball.y - round(self.ball.h / 2)),
                       self.ball.w, self.ball.h, 255)
        return screen


