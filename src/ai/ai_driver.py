from src.ai.model import PGAgent
from src.shared.config import Config
import time
from src.shared.state_subscriber import StateSubscriber
import numpy as np

if __name__ == "__main__":
    agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
    state = StateSubscriber()

    # Allow subscriber to connect
    while not state.ready():
        print("Waiting to connect to message broker...")
        time.sleep(1)

    agent.load('../../validation/6px_7k.h5')

    start_state = state.render_latest_preprocessed()
    last_frame_id = state.frame
    last_state = start_state
    last_tick = time.time()
    frame_diffs = []
    while True:
        next_planned_tick = last_tick + (1 / Config.AI_INFERENCES_PER_SECOND)

        # Get latest state diff
        current_state = state.render_latest_preprocessed()
        current_frame_id = state.frame
        diff_state = current_state - last_state
        last_state = current_state
        if last_frame_id is not None:
            frame_diff = current_frame_id - last_frame_id
            frame_diffs.append(frame_diff)
        last_frame_id = current_frame_id
        # Infer on flattened state vector
        x = diff_state.ravel()
        action, probs = agent.act(x)

        # Publish prediction
        state.publish("paddle2/action", {"action": Config.ACTIONS[action]})

        # Sleep until gap time in inference interval is passed
        now = time.time()
        to_sleep = (next_planned_tick - now)
        if to_sleep < 0:
            print(f"Warning: skipping inference ticks: running {-int(1000 * to_sleep)} ms behind."
                  f" Consider adjusting inference rate.")
        else:
            time.sleep(to_sleep)
        if len(frame_diffs) > 1000:
            print(f"Frame distribution: mean {np.mean(frame_diffs)}, stdev {np.std(frame_diffs)}")
            frame_diffs = []
        last_tick = time.time()


