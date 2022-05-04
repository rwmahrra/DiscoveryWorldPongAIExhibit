from distutils.command.config import config
import time

import paho.mqtt.client as mqtt
import numpy as np
import json

from exhibit.shared import utils
from exhibit.shared.config import Config


class MotionSubscriber:
    """
    MQTT compliant game state subscriber
    """

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        # none of the motion function rely on game state so no subscriptions are needed yet
        # client.subscribe("puck/position")

    # again - we might not need this
    # def on_message(self, client, userdata, msg):
    #     topic = msg.topic
    #     payload = json.loads(msg.payload)
    #     if topic == "puck/position":
    #         self.puck_x = payload["x"]
    #         self.puck_y = payload["y"]
        
    def publish(self, topic, message, qos=0):
        """
        Use the subscriber to send a message to update game variables elsewhere
        """
        p = json.dumps(message)
        self.client.publish(topic, payload=p, qos=qos)

    def __init__(self, config):
        self.config = config
        self.client = mqtt.Client(client_id="motion_module")
        self.client.on_connect = lambda client, userdata, flags, rc : self.on_connect(client, userdata, flags, rc)
        self.client.connect_async("localhost", port=1883, keepalive=60)
        
    def start(self):
        self.client.loop_forever()