from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from src.shared.utils import write
import numpy as np


class PGAgent:
    """
    Policy gradient agent
    Partly adapted from https://github.com/keon/policy-gradient/blob/master/pg.py
    """

    def __init__(self, state_size, action_size, name="PGAgent", learning_rate=0.001, structure=(200,)):
        """
        Set basic variables and construct the model
        :param state_size: Pixels in flattened input state
        :param action_size: Number of possible action types to output
        :param name: Agent name, used in some graphing/visualizing
        :param learning_rate: Model learning rate
        :param structure: Tuple of integers. a dense hidden layer with n layers is crated for each tuple element n
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = learning_rate
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.structure = structure
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        """
        Helper to construct model with Keras based on configuration
        """
        model = Sequential()
        model.add(Dense(self.structure[0], activation='relu', init='he_uniform', input_shape=(self.state_size,)))
        if len(self.structure) > 1:
            for layer in self.structure[1:]:
                model.add(Dense(layer, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def act(self, state):
        """
        Infer action from state
        :param state: ndarray representing game state
        :return: (action id, confidence)
        """
        state = state.reshape([1, state.shape[0]])
        prob = self.model.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        """
        "Smears" the reward values back through time so frames leading up to a reward are associated to that reward
        :param rewards: vector representing reward at each frame
        :return: discounted reward vector
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards.astype(np.float32)

    def train(self, states, actions, probs, rewards):
        """
        Train the model on a batch of game data. Imlements the "REINFORCE" algorithm.
        :param states: states from each frame
        :param actions: inferred actions from each frame
        :param probs: confidence probabilities from each frame
        :param rewards: rewards from each frame
        :return:
        """
        gradients = []
        for i in range(len(actions)):
            action = actions[i]
            prob = probs[i]
            y = np.zeros([self.action_size])
            y[action] = 1
            gradients.append(np.array(y).astype('float32') - prob)

        gradients = np.vstack(gradients)
        rewards = np.vstack(rewards)
        rewards = self.discount_rewards(rewards)
        gradients *= rewards

        X = np.squeeze(np.vstack([states]))
        Y = probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        result = self.model.train_on_batch(X, Y)
        write(str(result), f'analytics/{self.name}.csv')

    def load(self, name):
        """
        Load weights from an h5 file
        :param name: path to load weights
        """
        print(f"Loading {name}")
        self.model.load_weights(name)

    def save(self, name):
        """
        Export weights to an h5 file
        :param name: path to save weights
        """
        self.model.save_weights(name)

