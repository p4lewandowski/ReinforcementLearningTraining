from collections import deque
import numpy as np
import random
from itertools import compress

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, action_size, model, get_legal_actions):
        self.action_size = action_size
        self.get_legal_actions = get_legal_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        # Function adds information to the memory about last action and its results
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        if self.epsilon and (random.uniform(0, 1) < self.epsilon):
            # Choose only from valid actions
            chosen_action = random.choice(self.get_legal_actions(state))
            # chosen_action = random.randrange(self.action_size)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def get_best_action(self, state):
        """
        Compute the best action to take in a state.
        """
        if state.ndim < 2:
            state = np.expand_dims(state, axis=0)

        # Check what action can be made at all
        possible_actions = self.get_legal_actions(state)
        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        q_values = self.model.predict(state)[0]  # 0 as we have just one state that is considered
        # Make small values for impossible actions
        for i, val in enumerate(q_values):
            if i not in possible_actions:
                q_values[i] = -999999

        # Take the biggest
        max_q_val = max(q_values)
        is_potential_actions = [val == max_q_val for val in q_values]
        actions = list(range(self.action_size))

        actions_filtered = list(compress(actions, is_potential_actions))

        return random.choice(actions_filtered)

    def replay(self, batch_size):
        """
        Function learn network using randomly selected actions from the memory.
        First calculates Q value for the next state and choose action with the biggest value.
        Target value is calculated according to:
                Q(s,a) := (r + gamma * max_a(Q(s', a)))
        except the situation when the next action is the last action, in such case Q(s, a) := r.
        In order to change only those weights responsible for chosing given action, the rest values should be those
        returned by the network for state state.
        The network should be trained on batch_size samples.
        Also every time the function replay is called self.epsilon value should be updated according to equation:
        self.epsilon *= self.epsilon_decay
        """
        # Not enough samples
        if len(self.memory) < batch_size:
            return

        # Untangle batches
        batch = np.array(random.sample(self.memory, batch_size)).T
        states = np.vstack(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.vstack(batch[3])
        dones = batch[4]

        # comments for x64 batch size
        # Check Q values for current states # 64 x 4
        target = self.model.predict(states)

        # Calculate Q values for the next states and choose one with the biggest value # 64 x 4 --> 64 x 1
        next_pred = self.model.predict(next_states)

        # Exclude values for nonexisting action in state
        for id, val in enumerate(next_pred):
            possible_actions = self.get_legal_actions(next_states[id])
            for action in range(self.action_size):
                if action not in possible_actions:
                    next_pred[id][action] = 0

        # Pick max q_val
        next_q_val = np.max(next_pred, axis=1)

        # For subsequent targets, for action taken, replace val with reward and possible Q Value from next state
        for i in range(batch_size):
            target[i][actions[i]] = rewards[i] + ((1 - dones[i]) * self.gamma * next_q_val[i])

        # Update the model # 64 x 64 with target 4 x
        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

        # Update the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0


def build_model(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    model.summary()
    return model
