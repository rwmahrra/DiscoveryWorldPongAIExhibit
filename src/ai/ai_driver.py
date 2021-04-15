from src.ai.model import PGAgent
from src.shared.config import Config
import time
from src.ai.ai_subscriber import AISubscriber
import numpy as np

if __name__ == "__main__":
    agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
    state = AISubscriber()

    # Allow subscriber to connect
    while not state.ready():
        print("Waiting to connect to message broker...")
        time.sleep(1)

    agent.load('../../validation/6px_7k.h5')

    last_frame_id = state.frame
    last_tick = time.time()
    frame_diffs = []
    while True:
        next_planned_tick = last_tick + (1 / Config.AI_INFERENCES_PER_SECOND)

        # Get latest state diff
        current_frame_id = state.frame
        if not current_frame_id == last_frame_id:
            diff_state = state.render_latest_diff()
            if last_frame_id is not None:
                frame_diff = current_frame_id - last_frame_id
                frame_diffs.append(frame_diff)
            last_frame_id = current_frame_id
            # Infer on flattened state vector
            x = diff_state.ravel()
            action, probs = agent.act(x)

            # Publish prediction
            state.publish("paddle2/action", {"action": Config.ACTIONS[action]})
            state.publish("paddle2/frame", {"frame": current_frame_id})

            # Sleep until gap time in inference interval is passed
            now = time.time()
            to_sleep = (next_planned_tick - now)
            if len(frame_diffs) > 1000:
                print(frame_diffs)
                print(f"Frame distribution: mean {np.mean(frame_diffs)}, stdev {np.std(frame_diffs)} counts {np.unique(frame_diffs, return_counts=True)}")
                frame_diffs = []
            if to_sleep < 0:
                print(f"Warning: skipping inference ticks: running {-int(1000 * to_sleep)} ms behind."
                      f" Consider adjusting inference rate.")
            else:
                time.sleep(to_sleep)
                #while time.time() < next_planned_tick - ((1 / Config.AI_INFERENCES_PER_SECOND) / 10): pass # BUSY WAIT
            last_tick = time.time()


