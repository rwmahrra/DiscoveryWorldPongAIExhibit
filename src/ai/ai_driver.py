from src.ai.model import PGAgent
from src.shared.config import Config
import numpy as np

agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
agent.load('../../validation/6px_7k.h5')


def draw_rect(screen, x, y, w, h, color):
    """
    Utility to draw a rectangle on the screen state ndarray
    :param screen: ndarray representing the screen
    :param x: leftmost x coordinate
    :param y: Topmost y coordinate
    :param w: width (px)
    :param h: height (px)
    :param color: RGB int tuple
    :return:
    """
    screen[max(y, 0):y + h + 1, max(x, 0):x + w + 1] = color


def render_latest(state):
    """
    Render the current game pixel state by hand in an ndarray
    :return: ndarray of RGB screen pixels
    """
    screen = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.float32)
    screen[:, :] = (0, 60, 140)

    draw_rect(screen, int(state.left.x - int(state.left.w / 2)), int(state.left.y - int(state.left.h / 2)),
              state.left.w, state.left.h, 255)
    draw_rect(screen, int(state.right.x - int(state.right.w / 2)), int(state.right.y - int(state.right.h / 2)),
              state.right.w, state.right.h, 255)
    draw_rect(screen, int(state.ball.x - int(state.ball.w / 2)), int(state.ball.y - int(state.ball.h / 2)),
              state.ball.w, state.ball.h, 255)
    return screen
