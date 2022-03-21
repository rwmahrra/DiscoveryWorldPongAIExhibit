
from exhibit.game.pong import Pong
from exhibit.shared.config import Config
from exhibit.game.player import RandomPlayer
from exhibit.shared.utils import preprocess
import cv2
import numpy as np

VAL_GAMES = 75
TRAIN_GAMES = 75

env = Pong(hit_practice=True)
config = Config.instance()

random_agent = RandomPlayer()

X_train = []
y_paddle_train = []
y_ball_train = []

for i in range(TRAIN_GAMES):
    done = False
    game_frames = []
    game_paddles = []
    env.reset()
    while not done:
        move = random_agent.act()
        state, reward, done = env.step(None, config.ACTIONS[move], frames=1)
        game_frames.append(preprocess(state).flatten())
        game_paddles.append(env.top.x)

    game_frames = np.asarray(game_frames)
    game_paddles = np.asarray(game_paddles)
    game_ends = np.ones_like(game_paddles) * env.ball.x
    X_train.append(game_frames)
    y_paddle_train.append(game_paddles)
    y_ball_train.append(game_ends)

X_val = []
y_paddle_val = []
y_ball_val = []

X_train = np.concatenate(X_train)
y_paddle_train = np.concatenate(y_paddle_train)
y_ball_train = np.concatenate(y_ball_train)

print(X_train.shape)
X_train_prev = np.vstack((np.zeros((80*96)), X_train))
X_train_current = np.vstack((X_train, np.zeros((80*96))))
X_train = (X_train_current - X_train_prev)[:-1]
print(X_train.shape)

for i in range(VAL_GAMES):
    done = False
    game_frames = []
    game_paddles = []
    env.reset()
    while not done:
        move = random_agent.act()
        state, reward, done = env.step(None, config.ACTIONS[move], frames=1)
        game_frames.append(preprocess(state).flatten())
        game_paddles.append(env.top.x)

    game_frames = np.asarray(game_frames)
    game_paddles = np.asarray(game_paddles)
    game_ends = np.ones_like(game_paddles) * env.ball.x
    X_val.append(game_frames)
    y_paddle_val.append(game_paddles)
    y_ball_val.append(game_ends)

X_val = np.concatenate(X_val)
y_paddle_val = np.concatenate(y_paddle_val)
y_ball_val = np.concatenate(y_ball_val)

y_paddle_train = (y_paddle_train - (config.WIDTH / 2)) / (config.WIDTH / 2)
y_ball_train = (y_ball_train - (config.WIDTH / 2)) / (config.WIDTH / 2)
y_paddle_val = (y_paddle_val - (config.WIDTH / 2)) / (config.WIDTH / 2)
y_ball_val = (y_ball_val - (config.WIDTH / 2)) / (config.WIDTH / 2)

print(X_val.shape, X_train.shape, y_paddle_train.shape, y_paddle_val.shape, y_ball_train.shape, y_ball_val.shape)

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

frame_input = Input((80*96))
x = Dense(100, activation='relu')(frame_input)
out_ball = Dense(1, activation='linear')(x)
out_paddle = Dense(1, activation='linear')(x)
model = Model(inputs=frame_input, outputs=(out_paddle, out_ball))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, (y_paddle_train, y_ball_train), epochs=10)
model.evaluate(X_val, (y_paddle_val, y_ball_val))

y_paddle_pred, y_ball_pred = model.predict(X_val)
print(y_ball_val, y_ball_pred)

import matplotlib.pyplot as plt
spread = np.arange(y_ball_val.shape[0])
print(spread.shape)
response = np.abs(y_ball_val.squeeze() - y_ball_pred.squeeze())
print(response.shape)
plt.scatter(spread, response)
plt.show()

plt.scatter(spread, y_ball_val)
plt.scatter(spread, y_ball_pred)
plt.show()

plt.scatter(spread, y_paddle_val)
plt.scatter(spread, y_paddle_pred)
plt.show()

plt.scatter(y_ball_val.squeeze(), y_ball_pred.squeeze())
plt.show()
