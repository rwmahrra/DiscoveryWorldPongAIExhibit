import numpy as np
import cv2
import base64
import math
import keyboard
from random import choice, randint, random
import time
from exhibit.shared.config import Config
import pyrealsense2 as rs

if Config.ENABLE_AUDIO:
    import pygame.mixer

from exhibit.shared.config import Config


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
    #SPEEDUP = self.speedUp
    ACTIONS = Config.ACTIONS
    #SPEEDUP = 1
    # Cache game sounds. Loaded on first instance's init
    sounds = None
    
    align_to = rs.stream.color
    align = rs.align(align_to)  

    depth_feed = ""

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

    # get_depth() is not the currently used method
    # get_depth() gets the distance the center of the image is from the camera. 
    def get_depth():
        for i in range(50):
            # try 50 times. the intel realsense d435 can be finicky
            frames = Pong.pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth: continue # if it didn't work, try again

            depth_filtered = Pong.decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            
            # get dimensions of image after filtering (it usually shrinks)
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * Pong.crop_percentage_w)), int(w/2 + (w * Pong.crop_percentage_w))
            hs, he = int(h/2 - (h * Pong.crop_percentage_h)), int(h/2 + (h * Pong.crop_percentage_h))
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            depth_cropped_3d = np.dstack((depth_cropped,depth_cropped,depth_cropped))
            mean = 0.5
            try :
                mean = depth_cropped[~(depth_cropped > Pong.clipping_distance) &~(depth_cropped <= 0.1)].mean()
            except Exception as ex :
                print('failed to take mean of depth_cropped, could be zero values')
                print(ex)
                mean = 0.5
            
            return mean/2000 # This is where the range of values and how we scale that distance is set
        return 0.5 # middle value for when no player found
    
    # This is the currently used method.
    # This finds the "biggest blob"/island and gets the x coordinate of their center of mass
    # It filters out everything below a certain height
    def get_human_x():
        #try to get the frame 50 times
        for i in range(50): 
            try:
                frames = Pong.pipeline.wait_for_frames()
            except Exception:
                continue

            depth = frames.get_depth_frame()
            #color = frames.get_color_frame()
            if not depth: continue

            # filtering the image to make it less noisy and inconsistent
            depth_filtered = Pong.decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            
            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * Pong.crop_percentage_w)/2), int(w/2 + (w * Pong.crop_percentage_w)/2)
            hs, he = int(h/2 - (h * Pong.crop_percentage_h)/2), int(h/2 + (h * Pong.crop_percentage_h)/2)
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            #depth_cropped = depth_image

            cutoffImage = np.where((depth_cropped < Pong.clipping_distance) & (depth_cropped > 0.1), True, False)

            #print(f'cutoffImage shape is {cutoffImage.shape}, depth_cropped shape is {depth_cropped.shape}');
            avg_x = 0;
            avg_x_array = np.array([])
            countB = 0;
            for a in range(np.size(cutoffImage,0)):
                for b in range(np.size(cutoffImage,1)):
                    if cutoffImage[a,b] :
                        avg_x += b
                        #print(b)
                        avg_x_array = np.append(avg_x_array,b)
                        countB = countB+1
            # if we got no pixels in depth, return dumb value
            if countB <= 40: 
                return 0.5
            avg_x_array.sort()
            islands = []
            i_min = 0
            i_max = 0
            p = avg_x_array[0]
            for index in range(np.size(avg_x_array,0)) :
                n = avg_x_array[index]
                if n > p+1 and not i_min == i_max : # if the island is done
                    islands.append(avg_x_array[i_min:i_max])
                    i_min = index
                i_max = index
                p = n
            if not i_min == i_max: islands.append(avg_x_array[i_min:i_max])
            

            #print(islands)
            bigIsland = np.array([])
            for array in islands:
                if np.size(array,0) > np.size(bigIsland,0): bigIsland = array
            
            #print(np.median(bigIsland))
            m = (np.median(bigIsland))

            # DISPLAYING ******************************************************************************

            aligned_frames = Pong.align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            #print(color_frame)
            color_image = np.asanyarray(color_frame.get_data())

            grey_color = 153
            grey_color2 = 40
            depth_cropped_3d = np.dstack((depth_image,depth_image,depth_image))
            bg_removed = np.where((depth_cropped_3d < Pong.clipping_distance) & (depth_cropped_3d > 0.1), color_image, grey_color )

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW)

            depth_cropped_3d_actual = np.dstack((depth_cropped,depth_cropped,depth_cropped))
            depth_cropped_3d_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped_3d_actual, alpha=0.03), cv2.COLORMAP_RAINBOW)
            depth_cropped_3d_colormap = np.where((depth_cropped_3d_actual < Pong.clipping_distance) & (depth_cropped_3d_actual > 0.1), depth_cropped_3d_colormap, grey_color2 )

            # Uncomment these lines to have a window showing the camera feeds
            # images = np.hstack((bg_removed, depth_colormap))
            
            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example',  images)
            # cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
            depth_cropped_3d_colormap = cv2.line(depth_cropped_3d_colormap, (int(m),h), (int(m),0), (255,255,255), 1)
            # cv2.imshow('Filtered',  depth_cropped_3d_colormap)
            
            buffer = cv2.imencode('.jpg', depth_cropped_3d_colormap)[1].tostring()
            Pong.depth_feed = base64.b64encode(buffer).decode()

            # *****************************************************************************************
            # we multiply by 1.4 and subtract -0.2 so that the player can reach the edges of the Pong game.
            # In other words, we shrunk the frame so that the edges of the pong game can be reached without leaving the camera frame
            return (m/(np.size(cutoffImage,1)) * 1.4) -0.2
        return 0.5 # dummy value if we can't successfully get a good one

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
            self.x = 0
            self.y = int(Pong.HEIGHT / 2)
            self.w = self.WIDTH
            self.h = self.HEIGHT
            self.velocity = [0, 0]
            self.speed = self.SPEED * Pong.SPEEDUP
            #print(f'Pong.SPEEDUP is equal to {Pong.SPEEDUP}')
            if side == "left":
                self.x = Pong.PADDING
            elif side == "right":
                self.x = Pong.WIDTH - Pong.PADDING

        def reset(self, hit_practice=False):
            """
            Clean game state to initial configuration.
            Should be called before a new game.
            hit_practice: Spawns paddle at random valid position each round, for training generalization
            """
            self.x = 0
            #self.y = int(Pong.HEIGHT / 2)
            if hit_practice:
                # Generate all positions the paddle could end up in, then pick one.
                # This allows us to better generalize to all paddle positions in training.
                # (yes, it's redundant to do this on every reset, but I don't want to add another class field
                # just for this test setup)
                base_start = int(Pong.HEIGHT / 2)
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
            if self.side == "left":
                self.x = Config.LEFT_PADDLE_X
            elif self.side == "right":
                self.x = Config.RIGHT_PADDLE_X

        def up(self):
            """
            Handle up action
            """
            self.velocity[1] -= self.speed

        def down(self):
            """
            Handle down action
            """
            self.velocity[1] += self.speed
            
        # we need this method because controlling from the depth camera is not fixed speed
        def depthMove(self, depth):
            #print(f'depthMove with value = {depth}')
            desiredPos = depth * Pong.HEIGHT #((depth-500)/2000) * Pong.HEIGHT
            distance = desiredPos - self.y
            vel = (self.speed * (distance/Pong.HEIGHT) * 25)
            self.velocity[1] += vel
        
        
        def update(self):
            """
            Run game tick housekeeping logic
            """
            # First blindly increment position by velocity
            self.x += self.velocity[0]  # X should never actually change, but it feels right to include it
            self.y += self.velocity[1]
            self.velocity = [0, 0]  # We don't actually want velocity to persist from tick to tick, so deplete it all

            # Then back up position if we cross the screen border
            max = Pong.HEIGHT - Pong.Paddle.EDGE_BUFFER
            if self.y > max:
                self.y = max
            if self.y < Pong.Paddle.EDGE_BUFFER:
                self.y = Pong.Paddle.EDGE_BUFFER

        def handle_action(self, action, depth=None):
            """
            Parse action and modify state accordingly
            :param action: String representation of action ("UP", "DOWN", "NONE")
            :return:
            """
            if action == "UP":
                self.up()
            elif action == "DOWN":
                self.down()
            elif action == "NONE":
                pass
            elif action == "DEPTH":
                self.depthMove(depth = depth)

    class Ball:
        DIAMETER = Config.BALL_DIAMETER
        SPEED = 2
        BOUNCE_ANGLES = [0, 60, 45, 30, -30, -45, -60]  # True to original Atari Pong
        START_ANGLES = [0, 30, 20, 15, 10, 5, -5, -10, -15, -20, -30]#[0]

        def spawn_hit_practice(self):
            """
            Set initial position on the left side with a random trajectory towards
            the right side. Useful for training a right model to hit from various
            trajectories without coupling to an opponent strategy.
            """
            self.x = 5
            self.y = randint(0, Pong.HEIGHT)
            self.speed = self.SPEED * Pong.SPEEDUP
            self.velocity = self.get_vector(choice(Pong.Ball.BOUNCE_ANGLES), Pong.Ball.SPEED + (Pong.VOLLEY_SPEEDUP * choice(list(range(12)))))
            self.w = self.DIAMETER
            self.right = True
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
                # start the ball not in the center to give more reaction time
                self.x = round((Pong.WIDTH / 6)*5) 

                self.y = round((Pong.HEIGHT / 2) - 1)
                self.speed = self.SPEED * Pong.SPEEDUP
                self.velocity = (0, 0)
                self.w = self.DIAMETER
                self.right = None
                self.h = self.DIAMETER
            
            self.delay_counter = 0
            self.angle = 0

        def reset(self):
            """
            Reset to initial state before starting a new game.
            """
            if self.hit_practice:
                self.spawn_hit_practice()
            else:
                self.right = None
                # self.x = round(Pong.WIDTH / 6)
                self.y = round(Pong.HEIGHT / 2) 
                self.speed = self.SPEED * Pong.SPEEDUP
                self.velocity = (0, 0)
                self.w = self.DIAMETER
                self.h = self.DIAMETER
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

            angle = Pong.Ball.BOUNCE_ANGLES[segment]
            velocity = self.get_vector(angle, self.speed)
            if self.right:
                velocity = self.get_vector(-angle, self.speed)
                velocity = self.reverse_vector(velocity)
            self.velocity = velocity
            self.speed += Pong.VOLLEY_SPEEDUP * Pong.SPEEDUP

        def update(self):
            """
            Run game tick housekeeping logic
            """
            if self.velocity == (0, 0):

                self.angle = choice(Pong.Ball.START_ANGLES)
                
                if Pong.score_left + Pong.score_right == 1 and self.delay_counter == 0:
                    # change to your side
                    self.x = round(Pong.WIDTH / 6) # if flipping which is first also change line 347ish
                    self.right = True
                elif self.delay_counter == 0:
                    self.x = round((Pong.WIDTH / 6)*5)
                    self.right = False
                    
                if self.delay_counter <= 15: # a delay so that players can see the ball there before it launches
                    self.delay_counter += 1 # give a delay before the ball starts off again
                    #print(f'delay count {self.delay_counter}')
                else:
                    self.velocity = self.get_vector(self.angle, self.speed) if self.right else self.get_vector(self.angle + 180, self.speed)
                    #print(f'velocity set to to {self.velocity} using angle {self.angle}, self.right equals {self.right} (true is unchanged)')
                
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            if self.y > Pong.HEIGHT:
                self.y = Pong.HEIGHT
                self.bounce(y=True)
            if self.y < 0:
                self.y = 0
                self.bounce(y=True)

    def __init__(self, hit_practice=False, level = 1, pipeline = None, decimation_filter = None, crop_percentage_w = None, crop_percentage_h = None, clipping_distance = None, max_score = Config.MAX_SCORE):

        if pipeline == None:
            print('pipeline equal to None')
        
        Pong.SPEEDUP = 1 #+ 0.4 # (0.4*level) # uncomment this to make it faster each level
        print(f'Pong environment init level {level} and SPEEDUP is {Pong.SPEEDUP}')
        Pong.MAX_SCORE = max_score
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
        Pong.score_left = 0
        Pong.score_right = 0
        self.left = Pong.Paddle("left") if not self.hit_practice else None
        self.right = Pong.Paddle("right")
        self.ball = Pong.Ball(hit_practice=hit_practice)
        self.frames = 0

        # For managing our depth camera later
        Pong.decimation_filter = decimation_filter
        Pong.pipeline = pipeline
        Pong.crop_percentage_w = crop_percentage_w
        Pong.crop_percentage_h = crop_percentage_h
        Pong.clipping_distance = clipping_distance

    def reset(self):
        """
        Reset game state
        """
        Pong.score_left = 0
        Pong.score_right = 0
        if not self.hit_practice: self.left.reset()
        self.right.reset()
        self.ball.reset()
        screen = self.render()
        self.last_screen = screen
        return screen

    def get_score(self):
        """
        Fetch score tuple at the current frame
        :return: integer tuple: (left score, right score)
        """
        return Pong.score_left, Pong.score_right

    def get_bot_data(self, left=False, right=False):
        """
        Returns internal objects for the hard-coded opponent bot with perfect game state knowledge
        :param left: hard-coded bot operates left paddle
        :param right: hard-coded bot operates right paddle
        :return: Bot's paddle object and the ball object, used to directly calculate optimal move from object positions
        """
        if left:
            return self.left, self.ball
        if right:
            return self.right, self.ball

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

        next_ball_x = ball.x + ball.velocity[0]
        crosses_x_left = (next_ball_x - ball_r <= paddle.x + paddle_half_w) and (next_ball_x - ball_r >= paddle.x - paddle_half_w)
        crosses_x_right = (next_ball_x + ball_r <= paddle.x + paddle_half_w) and (next_ball_x + ball_r >= paddle.x - paddle_half_w)

        intersects_x_left = (ball.x + ball_r <= paddle.x - paddle_half_w) and (next_ball_x + ball_r >= paddle.x - paddle_half_w)
        intersects_x_right = (ball.x - ball_r >= paddle.x + paddle_half_w) and (next_ball_x - ball_r <= paddle.x + paddle_half_w)

        if crosses_x_left or crosses_x_right or intersects_x_left or intersects_x_right:
            next_ball_y = ball.y + ball.velocity[1]
            paddle_bottom = min(paddle.y - paddle_half_h, paddle.y - paddle_half_h + paddle.velocity[1])
            paddle_top = max(paddle.y + paddle_half_h, paddle.y + paddle_half_h + paddle.velocity[1])
            collide_y_top = (next_ball_y + ball_r <= paddle_top) and (next_ball_y + ball_r >= paddle_bottom)
            collide_y_bottom = (next_ball_y - ball_r <= paddle_top) and (next_ball_y - ball_r >= paddle_bottom)
            if collide_y_top or collide_y_bottom:
                return True, (ball.y - paddle.y) / (paddle.h / 2)

        """
        ball_left = ball.x - (ball.w / 2)
        ball_right = ball.x + (ball.w / 2)
        ball_top = ball.y + (ball.h / 2)
        ball_bottom = ball.y - (ball.h / 2)
        paddle_left = paddle.x - (paddle.w / 2)
        paddle_right = paddle.x + (paddle.w / 2)
        paddle_top = paddle.y + (paddle.h / 2)
        paddle_bottom = paddle.y - (paddle.h / 2)
        left_collide = ball_left > paddle_left and ball_left < paddle_right
        right_collide = ball_right > paddle_left and ball_right < paddle_right
        top_collide = ball_top > paddle_bottom and ball_top < paddle_top
        bottom_collide = ball_bottom < paddle_top and ball_bottom > paddle_bottom
        if left_collide or right_collide:
            if top_collide or bottom_collide:
                return True, (ball.y - paddle.y) / (paddle.h / 2)
        """
        return False, 0


    def step_hit_practice(self, right_action, frames=3):
        """
        Game tick if running hit practice
        :param right_action: Action from right agent
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
                self.right.handle_action(right_action)

                collide_right, pos = self.check_collision(self.ball, self.right)
                
                if collide_right and self.ball.right:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.right = False

                if self.ball.x < 0:
                    Pong.play_sound("score")
                    Pong.score_right += 1
                    reward_l -= 1.0
                    reward_r += 1.0
                    self.ball.reset()
                    self.right.reset(hit_practice=True)
                elif self.ball.x > Pong.WIDTH:
                    Pong.play_sound("score")
                    Pong.score_left += 1
                    reward_l += 1.0
                    reward_r -= 1.0
                    self.ball.reset()
                    self.right.reset(hit_practice=True)
                self.ball.update()
                self.right.update()
                done = False
                if Pong.score_right >= Pong.MAX_SCORE or Pong.score_left >= Pong.MAX_SCORE:
                    print(f'The scores have hit the max_score of {Pong.MAX_SCORE} with AI: {Pong.score_right} and Human: {Pong.score_left}')
                    done = True
        screen = self.render()
        #self.show(self.render(), 3)
        return screen, (reward_l, reward_r), done

    def step(self, left_action, right_action, frames=3, depth=None):
        """
        Game tick housekeeping
        :param left_action: Action from left agent
        :param right_action: Action from right agent
        :param frames: Frames to run before the next action is accepted
        :return: Tuple containing:
                 (screen state,
                 (left points scored this action, right points scored this action),
                 boolean indicating if game is over)
        """
        if self.hit_practice:
            return self.step_hit_practice(right_action, frames=frames)
        reward_l = 0
        reward_r = 0
        done = False
        for i in range(frames):
            if not done:
                self.left.handle_action(left_action, depth)
                self.right.handle_action(right_action)

                collide_left, pos = self.check_collision(self.ball, self.left)
                if collide_left and not self.ball.right:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.right = True
                collide_right, pos = self.check_collision(self.ball, self.right)
                if collide_right and self.ball.right:
                    Pong.play_sound("return")
                    self.ball.bounce_angle(pos)
                    self.ball.right = False

                if self.ball.x < 0:
                    Pong.play_sound("score")
                    Pong.score_right += 1
                    reward_l -= 1.0
                    reward_r += 1.0
                    self.ball.reset()
                    self.left.reset()
                    self.right.reset()
                elif self.ball.x > Pong.WIDTH:
                    Pong.play_sound("score")
                    Pong.score_left += 1
                    reward_l += 1.0
                    reward_r -= 1.0
                    self.ball.reset()
                    self.left.reset()
                    self.right.reset()
                self.left.update()
                self.right.update()
                self.ball.update()
                done = False
                #if Pong.score_right >= Pong.MAX_SCORE or Pong.score_left >= Pong.MAX_SCORE:
                # changed this to make it consisten number of balls/playsfor exhibit
                # switch to if statement above to change to the traditional "first to x points"
                if Pong.score_right + Pong.score_left >= Pong.MAX_SCORE:
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
        # print('show()')
        cv2.imshow(f"Pong", to_render/255)
        cv2.waitKey(duration)

    def get_packet_info(self):
        """
        Return all info necessary for regular update messages
        :return: Tuple representing ((puck_x, puck_y), paddle1_y, paddle2_y, paddle1_score, paddle2_score, game_frame))
        """
        return ((self.ball.x, self.ball.y), self.left.y, self.right.y, Pong.score_left, Pong.score_right,  self.frames)

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
        #print('rendering Pong game screen')
        # Draw middle grid line
        # Note: subtract 1 from game width because pixels index starting from 0, and we want this to be symmetrical
        self.draw_rect(screen, round((Pong.WIDTH)/2 - 1), 0, 2, round(Pong.HEIGHT), 255)

        if not self.hit_practice:
            self.draw_rect(screen, round(self.left.x - round(self.left.w / 2)), round(self.left.y - round(self.left.h / 2)),
                           self.left.w, self.left.h, 255)
        self.draw_rect(screen, round(self.right.x - round(self.right.w / 2)), round(self.right.y - round(self.right.h / 2)),
                       self.right.w, self.right.h, 255)
        self.draw_rect(screen, round(self.ball.x - round(self.ball.w / 2)), round(self.ball.y - round(self.ball.h / 2)),
                       self.ball.w, self.ball.h, 255)
        return screen


