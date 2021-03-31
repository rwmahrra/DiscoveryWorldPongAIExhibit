import paho.mqtt.client as mqtt


class StateSubscriber:
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

    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))
        topic = msg.topic
        if topic == "puck/position":
            self.puck_x = msg.payload["x"]
            self.puck_y = msg.payload["y"]
        if topic == "paddle1/position":
            self.paddle1_y = msg.payload["position"]
        if topic == "paddle2/position":
            self.paddle2_y = msg.payload["position"]
        if topic == "game/level":
            self.game_level = msg.payload["level"]

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
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        self.puck_x = None
        self.puck_y = None
        self.paddle1_y = None
        self.paddle2_y = None
        self.game_level = None

