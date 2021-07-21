from exhibit.ai.model import PGAgent
from exhibit.shared.config import Config
from exhibit.game.game_subscriber import GameSubscriber

import time
from exhibit.ai.ai_subscriber import AISubscriber
import numpy as np
import cv2


class AIDriver:
    #MODEL = 'validation/canstop_randomstart_6850.h5'#'../../validation/newhit_10k.h5'
    MODEL_1 = f'./validation/canstop_randomstart_3k.h5'
    MODEL_2 = f'./validation/canstop_randomstart_6850.h5'
    MODEL_3 = f'./validation/canstop_randomstart_10k.h5'
    level = 1
    def publish_inference(self):

        #print("testing*** check if level changed")
        if (AIDriver.level != self.state.game_level):
            temp = AIDriver.level
            AIDriver.level = self.state.game_level
            print(f'level changed to {AIDriver.level}')
            if self.state.game_level == 0:
                self.agent.load(AIDriver.MODEL_1)
            elif self.state.game_level == 1 and temp != 0:
                self.agent.load(AIDriver.MODEL_1)
            elif self.state.game_level == 2:
                self.agent.load(AIDriver.MODEL_2)
            elif self.state.game_level == 3:
                self.agent.load(AIDriver.MODEL_3)
            else:
                self.agent.load(AIDriver.MODEL_2)
        
        # Get latest state diff
        diff_state = self.state.render_latest_diff()
        current_frame_id = self.state.frame
        # Infer on flattened state vector
        x = diff_state.ravel()
        action, probs = self.agent.act(x)

        # Publish prediction
        if self.paddle1:
            self.state.publish("paddle1/action", {"action": str(action)})
            self.state.publish("paddle1/frame", {"frame": current_frame_id})
        elif self.paddle2:
            self.state.publish("paddle2/action", {"action": str(action)})
            self.state.publish("paddle2/frame", {"frame": current_frame_id})

        model_activation = self.agent.get_activation_packet()
        self.state.publish("ai/activation", model_activation)

        if len(self.frame_diffs) > 1000:
            print(
                f"Frame distribution: mean {np.mean(self.frame_diffs)}, stdev {np.std(self.frame_diffs)} counts {np.unique(self.frame_diffs, return_counts=True)}")
            self.frame_diffs = []

    def __init__(self, paddle1=True):
        self.paddle1 = paddle1
        self.paddle2 = not self.paddle1
        self.agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
        self.agent.load(AIDriver.MODEL_1)
        self.state = AISubscriber(trigger_event=lambda: self.publish_inference())
        self.last_frame_id = self.state.frame
        self.last_tick = time.time()
        self.frame_diffs = []
        self.state.start()
        
        #self.level=1


if __name__ == "__main__":
    instance = AIDriver()
