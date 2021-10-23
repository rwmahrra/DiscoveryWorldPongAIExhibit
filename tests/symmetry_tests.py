from exhibit.game.player import BotPlayer
from exhibit.game.pong import Pong
from exhibit.shared.config import Config
import cv2
import numpy as np

cfg = Config.instance()
ACT = cfg.ACTIONS[2]

"""
These tests assert that the screen can be vertically flipped and have the exact same pixel layout.
This is necessary to ensure that the game is totally fair,
and that the exact same AI model can be trained against itself.
"""

def flip(y):
    # 0 -> Pong.HEIGHT - 1
    # Pong.HEIGHT - 1 -> 0
    # 79 -> 80
    return -y + (cfg.HEIGHT - 1)


def setup_custom(up=True, angle=None, hit_practice=False):
    default_angle = 90 if up else -90
    cfg = Config()
    cfg.RANDOMIZE_START = False
    if hit_practice:
        cfg.ENV_TYPE = hit_practice
    if angle is not None:
        cfg.BALL_START_ANGLES = [angle]
    else:
        cfg.BALL_START_ANGLES = [default_angle]
    env = Pong(config=cfg)
    env.reset()
    return env


def test_basic_symmetry():
    envup = setup_custom(up=True)
    envdown = setup_custom(up=False)
    for i in range(100):
        try:
            assert(envup.ball.y - flip(envdown.ball.y) < 0.00001) # Should be equal, except for float rounding
            upr = envup.render()
            downr = envdown.render()
            downr = np.flip(downr, axis=0)
            # For additional sanity, we could view each frame diff to make sure they match up
            #cv2.imshow("test", upr - downr)
            #cv2.waitKey(0)
            assert(np.all((upr - downr) == 0))
        except:
            cv2.waitKey(0)
        print("UP")
        envup.step(ACT, ACT)
        print("DOWN")
        envdown.step(ACT, ACT)


def test_angular_symmetry():
    for angle in [30, 45, 60]:
        envup = setup_custom(up=True, angle=angle)
        envdown = setup_custom(up=False, angle=-angle)
        for i in range(500):
            assert(envup.ball.y - flip(envdown.ball.y) < 0.00001)
            upr = envup.render()
            downr = envdown.render()
            downr = np.flip(downr, axis=0)
            assert(np.all((upr - downr) == 0))
            # For additional sanity, we could view each frame diff to make sure they match up
            #cv2.imshow("test", upr - downr)
            #cv2.waitKey(0)
            envup.step(ACT, ACT)
            envdown.step(ACT, ACT)

def test_symmetry_with_gameplay():
    envup = setup_custom(up=True, angle=60)
    envdown = setup_custom(up=False, angle=-60)
    agenttop = BotPlayer(env=envup, bottom=True)
    agentbottom = BotPlayer(env=envup, top=True)
    revagenttop = BotPlayer(env=envdown, bottom=True)
    revagentbottom = BotPlayer(env=envdown, top=True)
    for i in range(500):
        try:
            assert (envup.ball.y - flip(envdown.ball.y) < 0.00001)
            upr = envup.render()
            downr = envdown.render()
            downr = np.flip(downr, axis=0)
            envup.step(cfg.ACTIONS[agenttop.act()[0]], cfg.ACTIONS[agentbottom.act()[0]], frames=2)
            envdown.step(cfg.ACTIONS[revagenttop.act()[0]], cfg.ACTIONS[revagentbottom.act()[0]], frames=2)
            cv2.imshow("test", upr - downr)
            assert (np.all((upr - downr) == 0))
            cv2.waitKey(0)
        except:
            cv2.waitKey(0)

def test_hit_practice_symmetry():
    envup = setup_custom(up=True, angle=60, hit_practice=True)
    envdown = setup_custom(up=False, angle=-60, hit_practice=True)
    agenttop = BotPlayer(env=envup, top=True)
    revagentbottom = BotPlayer(env=envdown, bottom=True)
    for i in range(500):
        try:
            assert (envup.ball.y - flip(envdown.ball.y) < 0.00001)
            upr = envup.render()
            downr = envdown.render()
            downr = np.flip(downr, axis=0)
            print(agenttop.act())
            envup.step(cfg.ACTIONS[agenttop.act()[0]], None, frames=1)
            envdown.step(None, cfg.ACTIONS[revagentbottom.act()[0]], frames=1)
            cv2.imshow("test", upr - downr)
            assert (np.all((upr - downr) == 0))

        except:
            cv2.waitKey(0)

if __name__ == "__main__":
    #test_basic_symmetry()
    #test_angular_symmetry()
    test_symmetry_with_gameplay()