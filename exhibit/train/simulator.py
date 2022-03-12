from exhibit.shared import utils
import numpy as np
from exhibit.game.pong import Pong
from exhibit.game.player import BotPlayer
from exhibit.shared.config import Config

def simulate_game(config, env_type=Config.instance().CUSTOM, left=None, right=None, batch=1, visualizer=None):
    """
    Wraps both the OpenAI Gym Atari Pong environment and the custom
    Pong environment in a common interface, useful to test the same training setup
    against both environments
    """
    env = None
    state_size = None
    games_remaining = batch
    state_shape = config.CUSTOM_STATE_SHAPE

    if env_type == config.CUSTOM:
        env = Pong()
        state_size = config.CUSTOM_STATE_SIZE
        state_shape = config.CUSTOM_STATE_SHAPE
        if type(left) == BotPlayer: left.attach_env(env)
        if type(right) == BotPlayer: right.attach_env(env)
    elif env_type == config.HIT_PRACTICE:
        env = Pong(hit_practice=True)
        state_size = config.CUSTOM_STATE_SIZE
        state_shape = config.CUSTOM_STATE_SHAPE
        if type(right) == BotPlayer: right.attach_env(env)

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
    reward_r = 0
    reward_l = 0
    if visualizer is not None:
        visualizer.base_render(utils.preprocess_custom(state))
    i = 0

    # Simulate latency. When the AI moves (every AI_FRAME_INTERVAL frames), it becomes the queued_action
    # and sets the queue_timestamp
    queued_action = None
    queue_timestamp = None
    selected_action = 2  # Stationary action
    action_l, prob_l, action_r, prob_r = None, None, None, None
    while True:
        if queue_timestamp == i - config.AI_FRAME_DELAY:
            selected_action = queued_action
            queue_timestamp = None
            queued_action = None

        if i % config.AI_FRAME_INTERVAL == 0:
            render_states.append(state.astype(np.uint8))
            current_state = utils.preprocess_custom(state)
            diff_state = current_state - last_state
            model_states.append(diff_state.astype(np.uint8))
            diff_state_rev = np.flip(diff_state, axis=1)
            last_state = current_state

            x = diff_state.ravel()
            x_flip = diff_state_rev.ravel()
            if left is not None: action_l, _, prob_l = left.act(x_flip)
            if right is not None: action_r, _, prob_r = right.act(x)
            states.append(x)
            queued_action = action_r
            queue_timestamp = i

        if env_type == config.HIT_PRACTICE:
            state, reward, done = env.step(None, config.ACTIONS[selected_action], frames=1)
        else:
            state, reward, done = env.step(config.ACTIONS[action_l], config.ACTIONS[action_r], frames=1)

        reward_l = reward_l + float(reward[0])
        reward_r = reward_r + float(reward[1])

        if action_r is not None and (i + 1) % config.AI_FRAME_INTERVAL == 0:
            # Save observations
            probs_l.append(prob_l)
            probs_r.append(prob_r)
            actions_l.append(action_l)
            actions_r.append(action_r)
            rewards_l.append(reward_l)
            rewards_r.append(reward_r)
            if reward_r < 0:
                score_l += reward_l
            if reward_r > 0:
                score_r -= reward_l
            reward_l = 0
            reward_r = 0

        if done:
            # Stash last action/prob/reward results because we didn't complete a move cycle
            if (i + 1) % config.AI_FRAME_INTERVAL != 0:
                probs_l.append(prob_l)
                probs_r.append(prob_r)
                actions_l.append(action_l)
                actions_r.append(action_r)
                rewards_l.append(reward_l)
                rewards_r.append(reward_r)
                if reward_r < 0: score_l -= reward_r
                if reward_r > 0: score_r += reward_r
                reward_l = 0
                reward_r = 0
            games_remaining -= 1
            print('Score: %f - %f.' % (score_l, score_r))
            utils.write(f'{score_l},{score_r}', f'analytics/scores.csv')
            if games_remaining == 0:
                metadata = (render_states, model_states, (score_l, score_r))
                return states, (actions_l, probs_l, rewards_l), (actions_r, probs_r, rewards_r), metadata
            else:
                score_l, score_r = 0, 0
                action_l, prob_l, action_r, prob_r = None, None, None, None
                state = env.reset()
        i += 1
