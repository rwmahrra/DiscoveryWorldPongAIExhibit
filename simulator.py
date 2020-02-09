import utils
import numpy as np
from pong import Pong
from player import BotPlayer

CUSTOM = 0
ATARI = 1
CUSTOM_ACTIONS = ["UP", "DOWN"]
ATARI_ACTIONS = [2, 3]  # Control indices for "UP", "DOWN"
ATARI_ACTION_SIZE = 2
CUSTOM_ACTION_SIZE = 2
CUSTOM_STATE_SHAPE = Pong.HEIGHT // 2, Pong.WIDTH // 2
ATARI_STATE_SHAPE = 80, 80
CUSTOM_STATE_SIZE = Pong.HEIGHT // 2 * Pong.WIDTH // 2
ATARI_STATE_SIZE = 80 * 80


def preprocess(state, env_type):
    if env_type == CUSTOM:
        return utils.preprocess_custom(state)
    elif env_type == ATARI:
        return utils.preprocess_gym(state)
    else:
        raise NotImplementedError


def step(env, env_type, action_l=None, action_r=None, frames=5):
    if env_type == CUSTOM:
        return env.step(CUSTOM_ACTIONS[action_l], CUSTOM_ACTIONS[action_r], frames=frames)
    elif env_type == ATARI:
        state, reward, done, _unused = env.step(ATARI_ACTIONS[action_r])
        return state, (0, reward), done
    else:
        raise NotImplementedError


def simulate_game(env_type=CUSTOM, left=None, right=None):
    env = None
    state_size = None

    if env_type == CUSTOM:
        from pong import Pong
        env = Pong()
        state_size = CUSTOM_STATE_SIZE
        state_shape = CUSTOM_STATE_SHAPE
        if type(left) == BotPlayer: left.attach_env(env)
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
    while True:
        render_states.append(state.astype(np.uint8))
        current_state = preprocess(state, env_type)
        x = current_state - last_state
        model_states.append(x.astype(np.uint8))
        last_state = current_state

        action_l, prob_l, action_r, prob_r = None, None, None, None
        x = x.ravel()
        if left is not None: action_l, prob_l = left.act(x)
        if right is not None: action_r, prob_r = right.act(x)

        states.append(x)

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

        if reward_r < 0: score_l -= reward_r
        if reward_r > 0: score_r += reward_r

        if done:
            print('Score: %f - %f.' % (score_l, score_r))
            metadata = (render_states, model_states, (score_l, score_r))
            return states, (actions_l, probs_l, rewards_l), (actions_r, probs_r, rewards_r), metadata
