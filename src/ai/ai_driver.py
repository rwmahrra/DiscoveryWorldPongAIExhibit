from src.ai.model import PGAgent
from src.shared.config import Config
import time
from src.shared.state_subscriber import StateSubscriber

if __name__ == "__main__":
    agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
    state = StateSubscriber()

    # Allow subscriber to connect
    while not state.ready():
        print("Waiting to connect to message broker...")
        time.sleep(1)

    agent.load('../../validation/6px_7k.h5')

    start_state = state.render_latest_preprocessed()
    last_state = start_state
    last_tick = time.time()
    while True:
        next_planned_tick = (last_tick * 1000) + Config.AI_INFER_INTERVAL_MS

        # Get latest state diff
        current_state = state.render_latest_preprocessed()
        diff_state = current_state - last_state
        last_state = current_state
        # Infer on flattened state vector
        x = diff_state.ravel()
        action, probs = agent.act(x)

        # Publish prediction
        state.publish("paddle2/action", {"action": Config.ACTIONS[action]})

        # Sleep until gap time in inference interval is passed
        now = time.time() * 1000
        to_sleep = (next_planned_tick - now)
        if to_sleep < 0:
            print(f"Warning: skipping inference ticks: running {-int(to_sleep)} ms behind."
                  f" Consider adjusting inference rate.")
        else:
            to_sleep /= 1000 # Convert to MS
            time.sleep(to_sleep / 1000)
        last_tick = time.time()


