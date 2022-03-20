import json
import paho.mqtt.client as mqtt
import numpy as np
import time
from exhibit.shared.utils import Config

class GameSubscriber:
    def emit_state(self, state, request_action=False):
        (puck_x, puck_y), bottom_x, top_x, score_left, score_right, frame = state

        self.client.publish("puck/position", payload=json.dumps({"x": puck_x, "y": puck_y}))
        self.client.publish("paddle1/position", payload=json.dumps({"position": bottom_x}))
        self.client.publish("paddle2/position", payload=json.dumps({"position": top_x}))
        self.client.publish("player1/score", payload=json.dumps({"score": score_left}))
        self.client.publish("player2/score", payload=json.dumps({"score": score_right}))

        if request_action:
            self.client.publish("game/frame", payload=json.dumps({"frame": frame}))
            if Config.instance().NETWORK_TIMESTAMPS:
                print(f'{time.time_ns() // 1_000_000} F{frame} SEND GM->AI')

    # get depth camera feed into browser
    def emit_depth_feed(self, feed):
        self.client.publish("depth/feed", payload=json.dumps({"feed": feed}))
        #print(f'emitting depth feed: {feed}')

    def emit_level(self, level):
        self.client.publish("game/level", payload=json.dumps({"level": level}), qos=2)

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("paddle1/action")
        client.subscribe("paddle2/action")
        client.subscribe("paddle1/frame")
        client.subscribe("paddle2/frame")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = json.loads(msg.payload)
        if topic == "paddle1/action":
            self.paddle1_queued_action = int(payload["action"])
        if topic == "paddle1/frame":
            self.paddle1_frame = payload["frame"]
            self.paddle1_queued_timestep = payload["frame"]
            if Config.instance().NETWORK_TIMESTAMPS:
                print(f'{time.time_ns() // 1_000_000} F{self.paddle1_frame} RECV AI->GM')
        if topic == "paddle2/action":
            self.paddle2_action = int(payload["action"])
        if topic == "paddle2/frame":
            self.paddle2_frame = payload["frame"]

    def get_paddle1_action(self, frame):
        if self.paddle1_queued_action is not None and self.paddle1_queued_timestep == frame - Config.instance().AI_FRAME_DELAY:
            self.paddle1_action = self.paddle1_queued_action
            self.paddle1_queued_action = None
            self.paddle1_queued_timestep = None
        return self.paddle1_action

    def __init__(self):
        print("init GameSubscriber")
        self.client = mqtt.Client(client_id="game_module")
        self.client.connect_async("localhost", port=1883, keepalive=60)
        self.client.on_connect = lambda client, userdata, flags, rc : self.on_connect(client, userdata, flags, rc)
        self.client.on_message = lambda client, userdata, msg : self.on_message(client, userdata, msg)
        self.client.loop_start()
        self.paddle1_queued_action = None
        self.paddle1_queued_timestep = None
        self.paddle1_action = 2  # ID for "NONE"
        self.paddle1_prob = np.array([0, 1])
        self.paddle1_frame = None
        self.paddle2_action = 2  # ID for "NONE"
        self.paddle2_prob = np.array([0, 1])
        self.paddle2_frame = None
