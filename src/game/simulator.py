from src.shared import utils
import numpy as np
import asyncio
import paho.mqtt.client as mqtt
from src.game.pong import Pong
from src.game.player import BotPlayer
import time
import json

from src.shared.config import Config

"""
Wraps both the OpenAI Gym Atari Pong environment and the custom
Pong environment in a common interface, useful to test the same training setup
against both environments
"""

CUSTOM = 0
ATARI = 1
HIT_PRACTICE = 2
CUSTOM_ACTIONS = ["UP", "DOWN", "NONE"]
ATARI_ACTIONS = [2, 3]  # Control indices for "UP", "DOWN"
ATARI_ACTION_SIZE = 2
CUSTOM_ACTION_SIZE = 2
CUSTOM_STATE_SHAPE = Pong.HEIGHT // 2, Pong.WIDTH // 2
ATARI_STATE_SHAPE = 80, 80
CUSTOM_STATE_SIZE = Pong.HEIGHT // 2 * Pong.WIDTH // 2
ATARI_STATE_SIZE = 80 * 80


def step(env, env_type, action_l=None, action_r=None, frames=10):
    if env_type == CUSTOM:
        return env.step(action_l, action_r, frames=frames)
    if env_type == HIT_PRACTICE:
        return env.step(None, CUSTOM_ACTIONS[action_r], frames=frames)
    elif env_type == ATARI:
        state, reward, done, _unused = env.step(ATARI_ACTIONS[action_r])
        return state, (0, reward), done
    else:
        raise NotImplementedError


def simulate_game(env_type=CUSTOM, left=None, right=None, batch=1, visualizer=None, marker_h=False, marker_v=False, subscriber=None):
    env = None
    state_size = None
    games_remaining = batch

    if env_type == CUSTOM:
        from src.game.pong import Pong
        env = Pong(marker_h=marker_h, marker_v=marker_v)
        state_size = CUSTOM_STATE_SIZE
        state_shape = CUSTOM_STATE_SHAPE
        if type(left) == BotPlayer: left.attach_env(env)
        if type(right) == BotPlayer: right.attach_env(env)
    elif env_type == HIT_PRACTICE:
        from src.game.pong import Pong
        env = Pong(hit_practice=True)
        state_size = CUSTOM_STATE_SIZE
        state_shape = CUSTOM_STATE_SHAPE
        if type(right) == BotPlayer: right.attach_env(env)
    elif env_type == ATARI:
        import gym
        env = gym.make("Pong-v0")
        if left is not None: raise NotImplementedError("Atari env does not support custom left player")
        state_size = ATARI_STATE_SIZE
        state_shape = ATARI_STATE_SHAPE
    else:
        raise NotImplementedError

    # Training data
    states = []
    actions_l = []
    actions_r = []
    rewards_l = []
    rewards_r = []
    probs_l = []
    probs_r = []

    # Prepare to collect fun data for visualizations
    render_states = []
    model_states = []
    score_l = 0
    score_r = 0
    last_state = np.zeros(state_shape)
    state = env.reset()
    if visualizer is not None:
        visualizer.base_render(utils.preprocess(state))

    # Emit state over MQTT and keep a running timer to track the interval
    subscriber.emit_state(env.get_packet_info())
    last_state_emit = time.time()
    last_frame_time = time.time()
    i = 0
    while True:
        next_frame_time = last_frame_time + (1 / Config.GAME_FPS)
        render_states.append(state.astype(np.uint8))
        current_state = utils.preprocess(state)
        diff_state = current_state - last_state
        model_states.append(diff_state.astype(np.uint8))
        last_state = current_state
        action_l, prob_l, action_r, prob_r = None, None, None, None
        x = diff_state.ravel()
        states.append(x)

        # Checking for defined left and right agents here is clunky but necessary to support single-agent environments
        # (e.g. "hit practice", where a single paddle is confronted with a barrage of balls at random trajectories.)

        if left is not None:
            action_l, prob_l = left.act()
        if right is not None:
            action_r, prob_r = right.act()


        if visualizer is not None and i % 1 == 0:
            visualizer.render_frame(diff_state, current_state, prob_r)
        state, reward, done = step(env, env_type, action_l=action_l, action_r=action_r)

        reward_l = float(reward[0])
        reward_r = float(reward[1])

        # Save observations
        probs_l.append(prob_l)
        probs_r.append(prob_r)
        actions_l.append(action_l)
        actions_r.append(action_r)
        rewards_l.append(reward_l)
        rewards_r.append(reward_r)

        env.show(1, 1)

        if reward_r < 0: score_l -= reward_r
        if reward_r > 0: score_r += reward_r

        if done:
            games_remaining -= 1
            print('Score: %f - %f.' % (score_l, score_r))
            utils.write(f'{score_l},{score_r}', f'../../analytics/scores.csv')
            if games_remaining == 0:
                metadata = (render_states, model_states, (score_l, score_r))
                return states, (actions_l, probs_l, rewards_l), (actions_r, probs_r, rewards_r), metadata
            else:
                score_l, score_r = 0, 0
                state = env.reset()
        now = time.time()
        if (now - last_state_emit) * 1000 > Config.STATE_PACKET_INTERVAL_MS:
            subscriber.emit_state(env.get_packet_info())
            last_state_emit = now
        to_sleep = next_frame_time - time.time()
        if to_sleep < 0:
            print(f"Warning: render tick is lagging behind by {-int(to_sleep * 1000)} ms.")
        else:
            time.sleep(to_sleep)
        last_frame_time = time.time()

        i += 1

