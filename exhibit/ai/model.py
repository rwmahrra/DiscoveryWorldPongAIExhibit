import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from exhibit.shared.config import Config
from exhibit.shared.utils import write
import numpy as np


class PGAgent:
    """
    Policy gradient agent
    Partly adapted from https://github.com/keon/policy-gradient/blob/master/pg.py
    """

    def __init__(self, state_size, action_size, name="PGAgent", learning_rate=0.001, structure=(200,), verbose=True):
        """
        Set basic variables and construct the model
        :param state_size: Pixels in flattened input state
        :param action_size: Number of possible action types to output
        :param name: Agent name, used in some graphing/visualizing
        :param learning_rate: Model learning rate
        :param structure: Tuple of integers. a dense hidden layer with n layers is crated for each tuple element n
        """
        self.verbose = verbose
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
        if self.verbose: self.model.summary()
        self.last_state = None
        self.last_hidden_activation = None
        self.last_output = None

        # Truncated model to only calculate hidden layer activations
        # (it's hard to get at them otherwise since the model is compiled)
        self.hl_model = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)

    def _build_model(self):
        """
        Helper to construct model with Keras based on configuration
        """
        model = Sequential()
        model.add(Dense(self.structure[0], activation='relu', input_shape=(self.state_size,)))
        if len(self.structure) > 1:
            for layer in self.structure[1:]:
                model.add(Dense(layer, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def act(self, state):
        """
        Infer action from state
        :param state: ndarray representing game state
        :return: (action id, confidence vector)
        """
        state = state.reshape([1, state.shape[0]])
        self.last_hidden_activation = self.hl_model.predict(state, batch_size=1).squeeze()
        prob = self.model.predict(state, batch_size=1).flatten()
        self.last_output = prob
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        state_ravel = state.reshape(Config.instance().CUSTOM_STATE_SHAPE)
        self.last_state = np.rot90(state_ravel, axes=(0,1), k=1).flatten()

        return action, prob

    def get_structure_packet(self):
        """
        Returns the state of the model suitable for realtime visualization
        :return: Model weights (list of 2d lists), biases (list of 1d lists),
        """
        layers = []
        i = 0
        for w in self.model.weights:
            l = None
            if i == 0: # Rotate first weight matrix as temporary solution for rotated
                l = np.rot90(w.numpy().reshape(*Config.CUSTOM_STATE_SHAPE, -1), axes=(0,1), k=1)
                l = l.reshape(Config.CUSTOM_STATE_SIZE, 200).tolist()
            else:
                l = w.numpy().tolist()

            layers.append(l)
            i += 1
        return layers

    def get_activation_packet(self):
        """
        Returns the state of the model suitable for realtime visualization
        :return: Model weights (list of 2d lists), biases (list of 1d lists),
        """
        import time
        t1 = time.perf_counter()
        # First, get input activations (the last preprocessed state)
        input_activation = self.last_state.tolist()

        # Then, get hidden layer activations (using the truncated model)
        hidden_activations = self.last_hidden_activation.tolist()

        # Finally, store output activations (model prediction)
        output_activations = self.last_output.tolist()

        return [input_activation, hidden_activations, output_activations]


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
        if self.verbose: print(f"Loading {name}")
        self.model.load_weights(name)

    def save(self, name):
        """
        Export weights to an h5 file
        :param name: path to save weights
        """
        self.model.save_weights(name)

