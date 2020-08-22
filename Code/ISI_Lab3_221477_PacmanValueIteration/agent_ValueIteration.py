import numpy as np
import pickle
from tqdm import tqdm


class agent_ValueIteration:
    def __init__(self, model, gamma, theta):
        self.model = model
        self.gamma = gamma
        self.theta = theta

        self.policy = dict()
        self.V = dict()
        self._init_policy()

    def _init_policy(self):
        # init with a policy with first avail action for each state
        for state in self.model.get_all_states():
            self.V[state] = 0

    def value_iteration(self):
        """
                This function calculate optimal policy for the specified model using Value Iteration approach:

                'model' - model of the environment, use following functions:
                    get_all_states - return list of all states available in the environment
                    get_possible_actions - return list of possible actions for the given state
                    get_next_states - return list of possible next states with a probability for transition from state by taking
                                      action into next_state

                'gamma' - discount factor for model
                'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is
                          smaller than theta
                Function returns optimal policy and value function for the policy
           """

        while True:
            delta = 0
            for s in tqdm(self.model.get_all_states()):
                v = self.V[s]
                V_s_a = {a: 0 for a in self.model.get_possible_actions(s)}
                for a in self.model.get_possible_actions(s):
                    for s_next_probability, s_next in self.model.get_next_states(s, a):
                        V_s_a[a] += s_next_probability * (
                                self.model.get_reward(s, a, s_next) + self.gamma * self.V[s_next])
                self.V[s] = max(list(V_s_a.values()))
                delta = max(delta, np.abs(v - self.V[s]))  # previous vs new
                self.policy[s] = max(V_s_a.keys(), key=(lambda k: V_s_a[k]))
            print(delta)
            if delta < self.theta:
                break
        self.save_policy()

    def load_policy(self):
        with open('policy.pkl', 'rb') as f:
            self.policy = pickle.load(f)

    def save_policy(self):
        with open('policy.pkl', 'wb') as f:
            pickle.dump(self.policy, f, pickle.HIGHEST_PROTOCOL)

    def get_action_for_state(self, state):
        return self.policy[state]
