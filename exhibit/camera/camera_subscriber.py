import time

import paho.mqtt.client as mqtt
import numpy as np
import json

from exhibit.shared import utils
from exhibit.shared.config import Config
import cv2
import math

class CameraSubscriber:
    """
    MQTT compliant game state subscriber.
    Always stores the latest up-to-date combination of game state factors.
    """

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("depth/request")
        # client.subscribe("paddle2/action")
        # client.subscribe("paddle1/frame")
        # client.subscribe("paddle2/frame")

    # get depth camera feed into browser
    def emit_depth_feed(self, feed):
        self.client.publish("depth/feed", payload=json.dumps({"feed": feed}))
        #print(f'emitting depth feed: {feed}')

    def emit_camposition(self, data):
        self.client.publish("depth/camposition", payload=json.dumps({"camposition": data}))


    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = json.loads(msg.payload)
        if topic == "depth/request":
            pass
            #publish(get_human_x())
            #TODO publish the values to depth/camposition
        # if topic == "paddle1/frame":
        #     self.paddle1_frame = payload["frame"]
        #     if Config.instance().NETWORK_TIMESTAMPS:
        #         print(f'{time.time_ns() // 1_000_000} F{self.paddle1_frame} RECV AI->GM')
        # if topic == "paddle2/action":
        #     self.paddle2_action = int(payload["action"])
        # if topic == "paddle2/frame":
        #     self.paddle2_frame = payload["frame"]


    
    def publish(self, state, request_action=False):
        
        self.client.publish("depth/camposition", payload=json.dumps({"camposition": state}))


    def __init__(self, config, trigger_event=None):
        """
        :param trigger_event: Function to call each time a new state is received
        """
        self.config = config
        self.trigger_event = trigger_event
        self.client = mqtt.Client(client_id="ai_module")
        self.client.on_connect = lambda client, userdata, flags, rc : self.on_connect(client, userdata, flags, rc)
        self.client.on_message = lambda client, userdata, msg : self.on_message(client, userdata, msg)
        print("Initializing subscriber")
        self.client.connect_async("localhost", port=1883, keepalive=60)
        self.puck_x = None
        self.puck_y = None
        self.bottom_paddle_x = None
        self.top_paddle_x = None
        self.game_level = None
        self.frame = 0
        self.latest_frame = None
        self.trailing_frame = None

    def start(self):
        self.client.loop_forever()
