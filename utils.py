import time

class Timer:
    timers = {}

    @staticmethod
    def start(name):
        Timer.timers[name] = time.time()

    @staticmethod
    def stop(name):
        t = time.time() - Timer.timers[name]
        print(f'Finished {name} in {round(t, 5)}s')
