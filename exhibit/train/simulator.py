import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
from exhibit.ai.model import PGAgent
from exhibit.shared import utils
from exhibit.game.pong import Pong
from exhibit.shared.config import Config
from itertools import starmap
import functools


def simulate_game(batch, env_type=Config.instance().CUSTOM, structure=(200,), bottom_path=None, top_path=None):
    """
    Wraps both the OpenAI Gym Atari Pong environment and the custom
    Pong environment in a common interface, useful to test the same training setup
    against both environments
    """
    config = Config.instance()
    env = None
    state_size = config.CUSTOM_STATE_SIZE
    state_shape = config.CUSTOM_STATE_SHAPE
    action_size = config.CUSTOM_ACTION_SIZE

    if top_path is not None:
        agent_top = PGAgent(state_size, action_size, name="agent_top", learning_rate=0,
                            structure=structure, verbose=False)
        agent_top.load(top_path)
    if bottom_path is not None:
        agent_bottom = PGAgent(state_size, action_size, name="agent_bottom", learning_rate=0,
                            structure=structure, verbose=False)
        agent_bottom.load(bottom_path)

    if env_type == config.CUSTOM:
        env = Pong()
        state_size = config.CUSTOM_STATE_SIZE
        state_shape = config.CUSTOM_STATE_SHAPE
        #if type(bottom_path) == BotPlayer: bottom_path.attach_env(env)
        #if type(top_path) == BotPlayer: top_path.attach_env(env)
    elif env_type == config.HIT_PRACTICE:
        env = Pong(hit_practice=True)
        state_size = config.CUSTOM_STATE_SIZE
        state_shape = config.CUSTOM_STATE_SHAPE
        #if type(top_path) == BotPlayer: top_path.attach_env(env)

    # Training data
    states = []
    states_flipped = []
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
    i = 0
    while True:
        render_states.append(state.astype(np.uint8))
        current_state = utils.preprocess_custom(state)
        diff_state = current_state - last_state
        model_states.append(diff_state.astype(np.uint8))
        diff_state_rev = np.flip(diff_state, axis=1)
        last_state = current_state
        action_l, prob_l, action_r, prob_r = None, None, None, None
        x = diff_state.ravel()
        x_flip = diff_state_rev.ravel()
        if bottom_path is not None: action_l, prob_l = agent_bottom.act(x_flip)
        if top_path is not None: action_r, prob_r = agent_top.act(x)
        states.append(x)

        state, reward, done = None, None, None
        if env_type == config.HIT_PRACTICE:
            state, reward, done = env.step(None, config.ACTIONS[action_r], frames=config.AI_FRAME_INTERVAL)
        else:
            state, reward, done = env.step(config.ACTIONS[action_l], config.ACTIONS[action_r], frames=config.AI_FRAME_INTERVAL)

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
            #print('Score: %f - %f.' % (score_l, score_r))
            utils.write(f'{score_l},{score_r}', f'analytics/scores.csv')
            metadata = (render_states, model_states, (score_l, score_r))
            return states, (actions_l, probs_l, rewards_l), (actions_r, probs_r, rewards_r), metadata
        i += 1


def simulate_game_batch(pool, env_type=Config.instance().CUSTOM, structure=(200,), bottom_path=None, top_path=None, batch=1):
    job_args = (env_type, structure, bottom_path, top_path)
    job_definitions = [(job_index, *job_args) for job_index in range(batch)]
    if pool is not None:
        results = pool.starmap(simulate_game, job_definitions)
    else:
        results = starmap(simulate_game, job_definitions)
    results = functools.reduce(lambda a, b: (a[0]+b[0], (a[1][0]+b[1][0], a[1][1]+b[1][1], a[1][2]+b[1][2]),
                                             (a[2][0]+b[2][0], a[2][1]+b[2][1], a[2][2]+b[2][2]),
                                             (a[3][0]+b[3][0], a[3][1]+b[3][1],
                                              (a[3][2][0]+b[3][2][0],a[3][2][1]+b[3][2][1]))), results)
    states, left, right, metadata = results
    return states, left, right, metadata
