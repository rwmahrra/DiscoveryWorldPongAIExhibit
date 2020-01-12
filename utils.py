import time
import numpy as np
import os

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


def get_last_file():
    id = get_resume_index()
    if id:
        return os.path.join("models", f"{id}.h5")
    else:
        return None

def get_resume_index():
    files = [f for f in os.listdir("models") if os.path.isfile(os.path.join("models", f))]
    ids = [int(os.path.split(f)[1].split('.')[0]) for f in files]
    max = -1
    for id in ids:
        if id > max:
            max = id
    if max == -1:
        return None
    return max
