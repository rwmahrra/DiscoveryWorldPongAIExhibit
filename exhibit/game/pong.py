import numpy as np
import cv2
import base64
import math
import keyboard
from random import choice, randint, random
import time
from exhibit import motion
from exhibit.shared.config import Config

from exhibit.shared.utils import Timer

if Config.instance().ENABLE_AUDIO:
    import pygame.mixer

from exhibit.shared.config import Config


class Pong:
        
    """
    This class captures all of the game logic for Pong.
    It was used instead of OpenAI Gym or various other publicly available alternatives
    in order to allow for complete flexibility.
    """

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
        return choice(Config.instance().ACTIONS)

    @staticmethod
    def load_sounds():
        if Config.instance().ENABLE_AUDIO:
            pygame.mixer.init()
            Pong.sounds = {}
            Pong.sounds["return"] = pygame.mixer.Sound(Config.instance().AUDIO_DIR + "return.ogg")
            Pong.sounds["score"] = pygame.mixer.Sound(Config.instance().AUDIO_DIR + "score.ogg")
            Pong.sounds["bounce"] = pygame.mixer.Sound(Config.instance().AUDIO_DIR + "bounce.ogg")

    @staticmethod
    def play_sound(sound):
        if Config.instance().ENABLE_AUDIO and Pong.sounds is not None and sound in Pong.sounds:
            try:
                playback = Pong.sounds[sound].play()
            except Exception as e:
                print(e)

    class Paddle:
        EDGE_BUFFER = 0  # Pixel distance from screen edges that paddle is allowed to reach
        SPEED = 6  # base speed (in px/tick)

        def __init__(self, side, config=None):
            self.config = config
            self.side = side
            self.x = int(self.config.WIDTH / 2)
            self.y = 0
            self.w = self.config.PADDLE_WIDTH + self.config.DEPLOYMENT_PADDLE_ADVANTAGE
            self.h = self.config.PADDLE_HEIGHT
            self.velocity = [0, 0]
            self.speed = self.SPEED * self.config.SPEEDUP
            if self.side == "top":
                self.y = self.config.TOP_PADDLE_Y
            elif self.side == "bottom":
                self.y = self.config.BOTTOM_PADDLE_Y

        def reset(self, hit_practice=False):
            """
            Clean game state to initial configuration.
            Should be called before a new game.
            hit_practice: Spawns paddle at random valid position each round, for training generalization
            """
            self.x = self.config.WIDTH / 2
            self.y = 0
            if hit_practice:
                # Generate all positions the paddle could end up in, then pick one.
                # This allows us to better generalize to all paddle positions in training.
                # (yes, it's redundant to do this on every reset, but I don't want to add another class field
                # just for this test setup)
                base_start = self.config.WIDTH / 2
                valid_starts = [base_start]
                start = base_start + self.speed
                while start < self.config.HEIGHT - Pong.Paddle.EDGE_BUFFER:
                    valid_starts.append(start)
                    start += self.speed
                start = base_start - self.speed
                while start > Pong.Paddle.EDGE_BUFFER:
                    valid_starts.append(start)
                    start -= self.speed
                self.y = choice(valid_starts)

            self.w = self.config.PADDLE_WIDTH + self.config.DEPLOYMENT_PADDLE_ADVANTAGE
            self.h = self.config.PADDLE_HEIGHT
            self.speed = self.SPEED * self.config.SPEEDUP
            if self.side == "top":
                self.y = self.config.TOP_PADDLE_Y
            elif self.side == "bottom":
                self.y = self.config.BOTTOM_PADDLE_Y

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

        def absolute_move(self, motion_position):
            position = motion_position * self.config.WIDTH
            self.x = position

        def MotionMove(self, motion_position):
            #print(f'depthMove with value = {depth}')
            desiredPos = motion_position * self.config.WIDTH #((depth-500)/2000) * Pong.HEIGHT
            distance = desiredPos - self.x
            vel = (self.speed * (distance/self.config.WIDTH) * 25)
            self.velocity[0] += vel

        def update(self):
            """
            Run game tick housekeeping logic
            """
            # First blindly increment position by velocity
            next_x = self.x + self.velocity[0]
            next_y = self.y + self.velocity[1]  # Y should never actually change, but it feels right to include it
            self.velocity = [0, 0]  # We don't actually want velocity to persist from tick to tick, so deplete it all

            # Then back up position if we cross the screen border
            max = self.config.WIDTH - Pong.Paddle.EDGE_BUFFER
            if not math.ceil(next_x - self.w / 2) > max and not math.ceil(next_x + self.w / 2) < Pong.Paddle.EDGE_BUFFER:
                self.x = next_x

        def handle_action(self, action, motion_position=None):
            """
            Parse action and modify state accordingly
            :param action: String representation of action ("UP", "DOWN", "NONE")
            :return:
            """
            print(f'action {action}')
            print(motion_position)
            if action == "LEFT":
                self.left()
            elif action == "RIGHT":
                self.right()
            elif action == "NONE":
                pass
            elif action == "ABSOLUTE":
                self.MotionMove(motion_position = motion_position)

    class Ball:
        def spawn_hit_practice(self):
            """
            Set initial position on the left side with a random trajectory towards
            the right side. Useful for training a right model to hit from various
            trajectories without coupling to an opponent strategy.
            """
            self.x = randint(0, self.config.WIDTH)
            self.y = self.config.HEIGHT - 5
            self.speed = self.config.BALL_SPEED * self.config.SPEEDUP
            self.velocity = self.get_vector(choice(self.config.BALL_BOUNCE_ANGLES), self.config.BALL_SPEED + (self.config.VOLLEY_SPEEDUP * choice(list(range(12)))))
            self.w = self.config.BALL_DIAMETER
            self.up = True
            self.h = self.config.BALL_DIAMETER

        def __init__(self, hit_practice=False, config=None):
            """
            Set basic state.
            :param hit_practice: Overrides ball reset to use "spawn_hit_practice"
                                 for alternative training mode. See method for details
            """
            self.config = config
            self.hit_practice = hit_practice
            if self.hit_practice:
                self.spawn_hit_practice()
            else:                
                # start the ball not in the center to give more reaction time
                self.x = round((self.config.WIDTH / 6)*5) 

                self.y = round((self.config.HEIGHT / 2) - 1)
                self.speed = self.config.BALL_SPEED * self.config.SPEEDUP
                self.velocity = (0, 0)
                self.start_up = True
                self.w = self.config.BALL_DIAMETER
                self.h = self.config.BALL_DIAMETER
                self.up = None
            
            self.delay_counter = 0
            self.angle = 0
            self.reset()

        def reset(self):
            """
            Reset to initial state before starting a new game.
            """
            if self.hit_practice:
                self.spawn_hit_practice()
            else:
                self.y = round(self.config.HEIGHT / 2) 
                self.speed = self.config.BALL_SPEED * self.config.SPEEDUP
                self.velocity = (0, 0)
                self.x = (self.config.WIDTH - 1) / 2
                self.y = (self.config.HEIGHT - 1) / 2
            self.delay_counter = 0

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

            angle = self.config.BALL_BOUNCE_ANGLES[segment]
            velocity = self.get_vector(angle, self.speed)
            if self.up:
                velocity = self.get_vector(-angle, self.speed)
            self.velocity = velocity
            self.speed += self.config.VOLLEY_SPEEDUP * self.config.SPEEDUP

        def update(self):
            """
            Run game tick housekeeping logic
            """
            if self.velocity == (0, 0):
                #self.velocity = self.get_vector(angle, self.speed)
                
                # if self.start_up == True and self.delay_counter == 0:
                if self.delay_counter == 0:
                    # change to your side
                    self.y = round(self.config.HEIGHT / 2) # just spawn in the middle
                    self.up = True
                    self.start_up = True # Start down on the next volley
                # elif self.delay_counter == 0:
                #     self.y = round((self.config.HEIGHT / 6)*5)
                #     self.up = False
                #     self.start_up = True # Start up on the next volley
                    
                if self.delay_counter <= 15: # a delay so that players can see the ball there before it launches
                    self.delay_counter += 1 # give a delay before the ball starts off again
                    #print(f'delay count {self.delay_counter}')
                else:
                    angle = choice(self.config.BALL_START_ANGLES)
                    self.velocity = self.get_vector(angle + 180, self.speed) if self.up else self.get_vector(angle, self.speed)
                    #print(f'velocity set to to {self.velocity} using angle {self.angle}, self.right equals {self.right} (true is unchanged)')
            else:
                # Higher positions = lower on screen, so negative y velocity is up
                self.up = self.velocity[1] < 0
                self.x += self.velocity[0]
                self.y += self.velocity[1]
                if self.x > self.config.WIDTH:
                    self.x = self.config.WIDTH
                    self.bounce(x=True)
                if self.x < 0:
                    self.x = 0
                    self.bounce(x=True)

    def __init__(self, config=None, hit_practice=False, level = 1, max_score = Config.instance().MAX_SCORE):
        """
        Initialize basic game state
        :param hit_practice: Trigger training mode with a single paddle and randomly spawned balls
                             See the Ball class's hit_practice method.
        """
        if config is None:
            config = Config.instance()
        
        config.SPEEDUP = 1 #+ 0.4 # (0.4*level) # uncomment this to make it faster each level
        config.MAX_SCORE = max_score

        # if Pong.sounds == None:
        #     Pong.load_sounds()

        self.config = config

        # Holds last raw screen pixels for rendering
        self.last_screen = None
        self.hit_practice = hit_practice
        self.score_bottom = 0
        self.score_top = 0
        self.bottom = Pong.Paddle("bottom", config=config) if not self.hit_practice else None
        self.top = Pong.Paddle("top", config=config)
        self.ball = Pong.Ball(hit_practice=hit_practice, config=config)
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
        screen_full = self.render()
        screen_top = self.render(top_only=True)
        screen_bottom = self.render(bottom_only=True)
        self.last_screen = screen_full
        return screen_bottom, screen_top

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
        ball_r = ball.h / 2
        paddle_half_w = paddle.w / 2
        paddle_half_h = paddle.h / 2

        next_ball_y = ball.y + ball.velocity[1]
        crosses_y_bottom = (next_ball_y - ball_r <= paddle.y + paddle_half_h) and (next_ball_y - ball_r >= paddle.y + paddle_half_h)
        crosses_y_top = (next_ball_y + ball_r <= paddle.y - paddle_half_h) and (next_ball_y + ball_r >= paddle.y - paddle_half_h)

        intersects_y_bottom = (ball.y + ball_r <= paddle.y - paddle_half_h) and (next_ball_y + ball_r >= paddle.y - paddle_half_h)
        intersects_y_top = (ball.y - ball_r >= paddle.y + paddle_half_h) and (next_ball_y - ball_r <= paddle.y + paddle_half_h)
        if crosses_y_bottom or crosses_y_top or intersects_y_bottom or intersects_y_top:
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

                collide_up, pos = self.check_collision(self.ball, self.top)
                if collide_up and self.ball.up:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.up = False

                if self.ball.y > self.config.HEIGHT - 1:
                    Pong.play_sound("score")
                    self.score_top += 1
                    # No distance based loss for bottom since it doesn't exist in hit practice
                    reward_l -= 1.0 #+ abs(self.bottom.x - self.ball.x) / self.config.WIDTH
                    reward_r += 1.0
                    self.ball.reset()
                    self.top.reset(hit_practice=True)
                elif self.ball.y < 0:
                    Pong.play_sound("score")
                    self.score_bottom += 1
                    reward_l += 1.0
                    reward_r -= 1.0 + abs(self.top.x - self.ball.x) / self.config.WIDTH
                    self.ball.reset()
                    self.top.reset(hit_practice=True)
                self.ball.update()
                self.top.update()
                done = False
                if self.score_top >= self.config.MAX_SCORE or self.score_bottom >= self.config.MAX_SCORE:
                    done = True
        screen = self.render()
        return screen, (reward_l, reward_r), done

    def step(self, bottom_action, top_action, frames=3, motion_position=None):
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
                self.bottom.handle_action(bottom_action, motion_position)
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

                if self.ball.y > self.config.HEIGHT - 1:
                    Pong.play_sound("score")
                    self.score_top += 1
                    reward_l -= 1.0 + abs(self.top.x - self.ball.x) / self.config.WIDTH
                    reward_r += 1.0
                    self.ball.reset()
                    self.bottom.reset()
                    self.top.reset()
                elif self.ball.y < 0:
                    Pong.play_sound("score")
                    self.score_bottom += 1
                    reward_l += 1.0
                    reward_r -= 1.0 + abs(self.bottom.x - self.ball.x) / self.config.WIDTH
                    self.ball.reset()
                    self.bottom.reset()
                    self.top.reset()
                self.bottom.update()
                self.top.update()
                self.ball.update()
                done = False
                if self.score_top >= self.config.MAX_SCORE or self.score_bottom >= self.config.MAX_SCORE:
                    done = True

            screen_full = self.render()
            screen_top = self.render(top_only=True)
            screen_bottom = self.render(bottom_only=True)
            self.last_screen = screen_full
            self.last_frame_time = time.time()

        self.last_screen = screen_full
        self.show(self.render(), duration=3)

        self.frames += 1
        return (screen_bottom, screen_top), (reward_l, reward_r), done

    def show(self, screen, scale=1,  duration=100):
        """
        Render last game frame through OpenCV
        :param scale: Multiplier to scale up/scale down rendered frame
        :param duration: Length of time to show frame and block thread (set to 0 to require keystroke to continue)
        :return:
        """
        l, r = self.get_score()
        to_render = cv2.resize(screen, (int(self.config.WIDTH * scale), int(self.config.HEIGHT * scale)))
        cv2.imshow(f"Pong", to_render/255)
        cv2.waitKey(duration)

    def get_packet_info(self):
        """
        Return all info necessary for regular update messages
        :return: Tuple representing ((puck_x, puck_y), paddle1_y, paddle2_y, paddle1_score, paddle2_score, game_frame))
        """
        return (self.ball.x, self.ball.y), self.bottom.x, self.top.x, self.score_bottom, self.score_top,  self.frames

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
        """
        IMPORTANT NOTE:
        The pixel indexing is a little quirky in this project.
        Index 0 corresponds to the center of the first pixel. Objects are centered around their coordinates.
        Even-dimensioned objects therefore have half-pixel coordinates since their center falls on the line between
        two pixels. So a two-pixel wide line on the first row of pixels would be at (0.5, 0): centered halfway between 
        the 0th and 1st pixel.
        
        To render this line, we would draw a rectangle beginning at x-(w/2),y-(h/2) with dimensions (w,h): so with
        the above example, it would go from (-0.5, -0.5) with size (2, 1). At the pixel level, we want to actually fill
        in (0,0) and (1,0)
        
        (Since odd length objects will also be rendered starting from their coordinate minus half their width, they will
        also fall on half-coordinates when rendered. If I had to rewrite this, I would shift everything that half-
        coordinate so everything is truly pixel aligned rather than pixel border aligned.) 
        """
        y = math.ceil(y)
        x = math.ceil(x)
        screen[max(y, 0):y+h, max(x, 0):x+w] = color

    def render(self, top_only=False, bottom_only=False):
        """
        Render the current game pixel state by hand in an ndarray
        :return: ndarray of RGB screen pixels
        """
        screen = np.zeros((self.config.HEIGHT, self.config.WIDTH, 3), dtype=np.float32)
        screen[:, :] = (140, 60, 0)  # BGR for a deep blue

        # Draw middle grid line
        self.draw_rect(screen, 0, (self.config.HEIGHT)/2 - 1, self.config.WIDTH, 2, 255)

        if top_only:
            self.draw_rect(screen, self.top.x - self.top.w / 2, (self.top.y - (self.top.h / 2)),
                           self.top.w, self.top.h, 255)
        elif bottom_only:
            self.draw_rect(screen, self.bottom.x - self.bottom.w / 2, (self.bottom.y - (self.bottom.h / 2)),
                           self.bottom.w, self.bottom.h, 255)
        else:
            if not self.hit_practice:
                self.draw_rect(screen, self.bottom.x - self.bottom.w / 2, (self.bottom.y - (self.bottom.h / 2)),
                               self.bottom.w, self.bottom.h, 255)
            self.draw_rect(screen, self.top.x - self.top.w / 2, (self.top.y - (self.top.h / 2)),
                           self.top.w, self.top.h, 255)
        self.draw_rect(screen, self.ball.x - self.ball.w / 2, (self.ball.y - (self.ball.h / 2)),
                       self.ball.w, self.ball.h, 255)
        return screen


