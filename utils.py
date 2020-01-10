import time
import numpy as np

class Timer:
    timers = {}

    @staticmethod
    def start(name):
        Timer.timers[name] = time.time()

    @staticmethod
    def stop(name):
        t = time.time() - Timer.timers[name]
        print(f'Finished {name} in {round(t, 5)}s')


def encode_action(action):
    if action == "UP":
        return np.asarray([1, 0], dtype=np.float32)
    elif action == "DOWN":
        return np.asarray([0, 1], dtype=np.float32)
    elif action == "NONE":
        return np.asarray([0, 0], dtype=np.float32)
    else:
        raise NameError(f"Action {action} does not exist")
