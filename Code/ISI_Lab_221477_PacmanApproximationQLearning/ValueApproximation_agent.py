from collections import defaultdict, deque
import random
from itertools import compress
import numpy as np
import json
import functools

PLAYER = 0
GHOSTS = 1
FOODS = 2
BOARD = 3

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class PathFinder:
    '''
    Modified distance/path-finding for binary matrices
    Source: https://www.geeksforgeeks.org/shortest-path-in-a-binary-maze/
    '''

    def __init__(self, board):
        self.board = []
        for row in board:
            self.board.append([0 if x == 'w' else 1 for x in row])
        self.width = len(self.board[0])
        self.height = len(self.board)

    # To store self.boardrix cell cordinates
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # A data structure for queue used in BFS
    class queueNode:
        def __init__(self, pt, dist):
            self.pt = pt  # The cordinates of the cell
            self.dist = dist  # Cell's distance from the source

    # Check whether given cell(row,col)
    # is a valid cell or not
    def isValid(self, row, col):
        return (row >= 0) and (row < self.height) and (col >= 0) and (col < self.width)

    # Function to find the shortest path between
    # a given source cell to a destination cell.
    def BFS(self, src, dest):
        src = self.Point(*src)
        dest = self.Point(*dest)

        rowNum = [-1, 0, 0, 1]
        colNum = [0, -1, 1, 0]

        # check source and destination cell
        # of the self.boardrix have value 1
        if self.board[src.x][src.y] != 1 or self.board[dest.x][dest.y] != 1:
            return -1

        visited = [[False for i in range(self.width)] for j in range(self.height)]

        # Mark the source cell as visited
        visited[src.x][src.y] = True

        # Create a queue for BFS
        q = deque()

        # Distance of source cell is 0
        s = self.queueNode(src, 0)
        q.append(s)  # Enqueue source cell

        # Do a BFS starting from source cell
        while q:
            curr = q.popleft()  # Dequeue the front cell

            # If we have reached the destination cell,
            # we are done
            pt = curr.pt
            if pt.x == dest.x and pt.y == dest.y:
                return curr.dist

            # Otherwise enqueue its adjacent cells
            for i in range(4):
                row = pt.x + rowNum[i]
                col = pt.y + colNum[i]

                # if adjacent cell is valid, has path
                # and not visited yet, enqueue it.
                if (self.isValid(row, col) and self.board[row][col] == 1 and not visited[row][col]):
                    visited[row][col] = True
                    Adjcell = self.queueNode(self.Point(row, col), curr.dist + 1)
                    q.append(Adjcell)

        # Return -1 if destination cannot be reached
        return -1


class ValueApproximationAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

        # Used for features
        self.weights = None
        self.maxDistance = None
        self.PathFinder = None
        self.maxFood = None

    def get_qvalue(self, state, action):
        return np.dot(self.state_to_features(state, action), self.weights)

    def get_value(self, state):
        if isinstance(state, str):
            state_ = json.loads(state)

        possible_actions = self.get_legal_actions(state_)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        return max([self.get_qvalue(state, a) for a in possible_actions])

    def update(self, state, action, reward, next_state):
        # numpy array with features
        features = np.array(self.state_to_features(state, action))

        # difference - some real value
        difference = reward + self.discount * self.get_value(next_state) - self.get_qvalue(state, action)
        # Element to element addition
        self.weights += self.alpha * difference * features

    @functools.lru_cache(maxsize=None)
    def state_to_features(self, state, action):
        if isinstance(state, str):
            state = json.loads(state)

        foods = state[FOODS]
        pacman = state[PLAYER]
        ghosts = state[GHOSTS]

        initialization = False
        if not self.PathFinder:
            board = state[BOARD]
            width = len(board[0])
            height = len(board)
            self.PathFinder = PathFinder(board)
            self.maxDistance = width // 2 + width * (height // 2 + 1)  # slalom
            initialization = True

        pacman_new = [pacman['x'], pacman['y']]
        if action == DOWN:
            pacman_new[1] = pacman_new[1] + 1
        elif action == UP:
            pacman_new[1] = pacman_new[1] - 1
        elif action == LEFT:
            pacman_new[0] = pacman_new[0] - 1
        elif action == RIGHT:
            pacman_new[0] = pacman_new[0] + 1

        if foods:
            next_foods_distances = []
            for food in foods:
                next_foods_distances.append(self.PathFinder.BFS((food['y'], food['x']), (pacman_new[1], pacman_new[0])))
            # ***FEATURE*** Stepped on food
            hasEaten = 1 if 0 in next_foods_distances else 0
            # ***FEATURE*** Distance to the closest food
            food_distance = min(next_foods_distances)
            food_distance = food_distance / self.maxDistance
        # If no food left
        else:
            food_distance = 0  # distance is zero
            hasEaten = 1  # We have eaten all we could

        # ***FEATURE*** How many ghosts are near - up to 1 steps from pacman
        next_ghost_distances = []
        for ghost in ghosts:
            next_ghost_distances.append(self.PathFinder.BFS((ghost['y'], ghost['x']), (pacman_new[1], pacman_new[0])))
        areGhostsNear = sum(distance < 2 for distance in next_ghost_distances) / len(ghosts)

        features = [food_distance, areGhostsNear, hasEaten]

        if initialization:
            self.weights = np.zeros(len(features))

        return features

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        values = [self.get_qvalue(state, a) for a in possible_actions]
        highest_value = max(values)
        is_potential_actions = [val == highest_value for val in values]
        actions_filtered = list(compress(possible_actions, is_potential_actions))

        return random.choice(actions_filtered)

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probability, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if epsilon and (random.uniform(0, 1) < epsilon):
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0


def play_and_train(env, agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False

    while not done:
        # get agent to pick action given state state.
        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward
