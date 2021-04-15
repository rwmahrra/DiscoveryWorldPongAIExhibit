import paho.mqtt.client as mqtt
import numpy as np
import json

from src.shared import utils
from src.shared.config import Config


class AISubscriber:
    """
    MQTT compliant game state subscriber.
    Always stores the latest up-to-date combination of game state factors.
    """

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("puck/position")
        client.subscribe("player1/score")
        client.subscribe("player2/score")
        client.subscribe("paddle1/position")
        client.subscribe("paddle2/position")
        client.subscribe("game/level")
        client.subscribe("game/frame")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = json.loads(msg.payload)
        if topic == "puck/position":
            self.puck_x = payload["x"]
            self.puck_y = payload["y"]
        if topic == "paddle1/position":
            self.paddle1_y = payload["position"]
        if topic == "paddle2/position":
            self.paddle2_y = payload["position"]
        if topic == "game/level":
            self.game_level = payload["level"]
        if topic == "game/frame":
            self.frame = payload["frame"]
            self.trailing_frame = self.latest_frame
            self.latest_frame = self.render_latest_preprocessed()

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
        screen[max(y, 0):y + h + 1, max(x, 0):x + w + 1] = color

    def publish(self, topic, message):
        """
        Use the state subscriber to send a message since we have the connection open anyway
        :param topic: MQTT topic
        :param message: payload object, will be JSON stringified
        :return:
        """
        self.client.publish(topic, payload=json.dumps(message))

    def render_latest(self):
        """
        Render the current game pixel state by hand in an ndarray
        :return: ndarray of RGB screen pixels
        """
        screen = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.float32)
        screen[:, :] = (0, 60, 140)

        self.draw_rect(screen, int(Config.LEFT_PADDLE_X - int(Config.PADDLE_WIDTH / 2)), int(self.paddle1_y - int(Config.PADDLE_HEIGHT / 2)),
                  Config.PADDLE_WIDTH, Config.PADDLE_HEIGHT, 255)
        self.draw_rect(screen, int(Config.RIGHT_PADDLE_X - int(Config.PADDLE_WIDTH / 2)), int(self.paddle2_y - int(Config.PADDLE_HEIGHT / 2)),
                 Config.PADDLE_WIDTH, Config.PADDLE_HEIGHT, 255)
        self.draw_rect(screen, int(self.puck_x - int(Config.BALL_DIAMETER / 2)), int(self.puck_y - int(Config.BALL_DIAMETER / 2)),
                  Config.BALL_DIAMETER, Config.BALL_DIAMETER, 255)
        return screen

    def render_latest_preprocessed(self):
        """
        Render the current game pixel state by hand in an ndarray
        Scaled down for AI consumption
        :return: ndarray of RGB screen pixels
        """
        latest = self.render_latest()
        return utils.preprocess(latest)

    def render_latest_diff(self):
        """
        Render the current game pixel state, subtracted from the previous
        Guarantees that adjacent frames are used for the diff
        :return: ndarray of RGB screen pixels
        """
        if self.trailing_frame is None:
            return self.latest_frame
        return self.latest_frame - self.trailing_frame

    def ready(self):
        """
        Determine if all state attributes have been received since initialization
        :return: Boolean indicating that all state values are populated.
        """
        return self.puck_x is not None \
               and self.puck_y is not None \
               and self.paddle1_y is not None \
               and self.paddle2_y is not None \
               and self.game_level is not None

    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = lambda client, userdata, flags, rc : self.on_connect(client, userdata, flags, rc)
        self.client.on_message = lambda client, userdata, msg : self.on_message(client, userdata, msg)
        print("Initializing subscriber")
        self.client.connect_async("localhost", port=1883, keepalive=60)
        self.puck_x = None
        self.puck_y = None
        self.paddle1_y = None
        self.paddle2_y = None
        self.game_level = None
        self.frame = None
        self.latest_frame = None
        self.trailing_frame = None
        self.client.loop_start()
