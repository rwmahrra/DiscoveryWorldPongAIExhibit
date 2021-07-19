from exhibit.game.pong import Pong
from exhibit.shared.config import Config
import cv2
import numpy as np

env = Pong()
initial_state = env.reset()

cv2.imshow("test", initial_state)
cv2.waitKey(0)

flipped = np.flip(initial_state, axis=1)
cv2.imshow("test", flipped)
cv2.waitKey(0)

diff = initial_state - flipped
cv2.imshow("test", diff)
cv2.waitKey(0)