import pygame
import random
import copy
from collections import defaultdict
from copy import deepcopy
from agent_QLearning import QLearningAgent, play_and_train
from itertools import chain, combinations, product
from tqdm import tqdm
import matplotlib.pyplot as plt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

PLAYER = 0
GHOSTS = 1
FOODS = 2

class Pacman:

    def __init__(self, board):
        """
            Pacman:
        """

        self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))

        self.board = board
        self.width = len(self.board[0])
        self.height = len(self.board)

        self.display_mode_on = True

        self.cell_size = 60
        pygame.init()
        self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))
        self.player_pos = dict()
        self.ghosts = []
        self.foods = []
        self.walls = []
        self.possible_positions = []
        self.all_states_raw = []
        self.all_states = defaultdict(list)
        self.score = 0
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] == 'p':
                    self.player_pos['x'] = x
                    self.player_pos['y'] = y
                    self.init_player_pos = self.player_pos.copy()
                elif self.board[y][x] == 'g':
                    ghost = dict()
                    ghost['x'] = x
                    ghost['y'] = y
                    ghost['direction'] = random.choice([LEFT, DOWN])
                    self.ghosts.append(ghost)
                elif self.board[y][x] == '*':
                    food = dict()
                    food['x'] = x
                    food['y'] = y
                    self.foods.append(food)
                if self.board[y][x] == 'w':
                    wall = dict()
                    wall['x'] = x
                    wall['y'] = y
                    self.walls.append(wall)
                else:
                    position = dict()
                    position['x'] = x
                    position['y'] = y
                    self.possible_positions.append(position)

        self.init_foods = self.foods.copy()
        self.init_ghosts = self.ghosts.copy()
        self.__draw_board()
        self.__init_all_states()

    def reset(self):
        """ resets state of the environment """
        self.foods = copy.deepcopy(self.init_foods)
        self.ghosts = copy.deepcopy(self.init_ghosts)
        self.player_pos = self.init_player_pos.copy()
        self.score = 0
        return self.__get_state()

    def __init_all_states(self):
        # Foods - can decrease in number
        food_positions = list(chain(*map(lambda x: combinations(self.foods, x), range(0, len(self.foods) + 1))))
        food_positions = [list(elem) for elem in food_positions]  # tuples to list
        # Ghost positions - the same number in each state
        ghost_positions = list(product(self.possible_positions, repeat=len(self.ghosts)))
        # Player positions
        player_positions = deepcopy(self.possible_positions)

        for pacman_position in player_positions:
            for ghost_position in ghost_positions:
                for food_position in food_positions:
                    self.all_states_raw.append([pacman_position, list(ghost_position), food_position])

        for i, state in enumerate(self.all_states_raw):
            self.all_states[i] = state

    def get_all_states(self):
        """ return a list of all possible states """
        return self.all_states

    def is_terminal(self, state):
        """
        return true if state is terminal or false otherwise
        state is terminal when ghost is on the same position as pacman or all capsules are eaten
        """
        player_pos = state[PLAYER]
        # if on ghost
        for ghost_pos in state[GHOSTS]:
            if ghost_pos == player_pos:
                return True
        # if last food and player on it
        foods_pos = state[FOODS]
        if len(foods_pos) == 1:
            if foods_pos[0] == player_pos:
                return True

        return False

    def get_possible_actions_for_position(self, XY_dict):
        actions = []

        # Move left
        if XY_dict['x'] > 0 and self.board[XY_dict['y']][XY_dict['x'] - 1] != 'w':
            actions.append(LEFT)
        # Move right
        if XY_dict['x'] + 1 < self.width and self.board[XY_dict['y']][XY_dict['x'] + 1] != 'w':
            actions.append(RIGHT)
        # Move up
        if XY_dict['y'] > 0 and self.board[XY_dict['y'] - 1][XY_dict['x']] != 'w':
            actions.append(UP)
        # Move down
        if XY_dict['y'] + 1 < self.height and self.board[XY_dict['y'] + 1][XY_dict['x']] != 'w':
            actions.append(DOWN)

        return actions

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        state = self.__hash_to_state(state)
        try:
            player_pos = state[PLAYER]
        except:
            return []
        return self.get_possible_actions_for_position(player_pos)

    def move_position(self, position, action):
        position = deepcopy(position)
        if action == LEFT:
            position['x'] -= 1
        elif action == RIGHT:
            position['x'] += 1
        elif action == UP:
            position['y'] -= 1
        elif action == DOWN:
            position['y'] += 1

        return position

    def get_next_states(self, hash_state, action):
        """
        return a set of possible next states and probabilities of moving into them
        assume that ghost can move in each possible direction with the same probability, ghost cannot stay in place
        """
        state = self.__hash_to_state(hash_state)
        # If terminal state - we cannot move from it
        if self.is_terminal(state):
            return [(1, hash_state)]

        current_state = deepcopy(state)
        player_pos = current_state[PLAYER]
        player_pos = self.move_position(player_pos, action)
        # replace the value now
        current_state[PLAYER] = player_pos

        # If terminal state already - finish
        if self.is_terminal(current_state):
            return [(1, self.__state_to_hash(current_state))]

        # If moved on the food - remove the food
        if player_pos in current_state[FOODS]:
            current_state[FOODS].remove(player_pos)

        # If not, move the ghosts
        # First determine actions
        ghosts = current_state[GHOSTS]
        ghosts_actions = []
        for ghost in ghosts:
            ghosts_actions.append(self.get_possible_actions_for_position(ghost))
        ghosts_actions = list(product(*ghosts_actions))
        # construct new ghost positions by moving ghosts
        out_ghosts = []
        for action_set in ghosts_actions:
            temp = []
            for ghost, action in zip(ghosts, action_set):
                temp.append(self.move_position(ghost, action))
            out_ghosts.append(temp)
        # Construct new state possibilities
        states = [[player_pos, out_ghost, current_state[FOODS]] for out_ghost in out_ghosts]
        prob_states = [(1 / len(states), self.__state_to_hash(state)) for state in states]

        return prob_states

    def get_reward(self, state, action, next_state):
        """
        return the reward after taking action in state and landing on next_state
            -1 for each step
            10 for eating capsule
            -500 for eating ghost
            500 for eating all capsules
        """
        next_state = self.__hash_to_state(next_state)
        player_pos = next_state[PLAYER]
        ghosts = next_state[GHOSTS]
        foods = next_state[FOODS]

        # If on ghost
        for ghost in ghosts:
            if ghost['x'] == player_pos['x'] and ghost['y'] == player_pos['y']:
                return -500
        # If on food
        for food in foods:
            if food['x'] == player_pos['x'] and food['y'] == player_pos['y']:
                # If last food
                if len(foods) == 1:
                    return 500
                return 10
        # If just step
        return -1

    def get_state(self):
        '''
        Function returns current state of the game
        :return: state
        '''
        ghosts = deepcopy(self.ghosts)
        [g_dict.pop('direction', None) for g_dict in ghosts]  # filter out directions
        return self.__state_to_hash([self.player_pos, ghosts, self.foods])

    def step(self, action):
        '''
        Function apply action. Do not change this code
        :returns:
        state - current state of the game
        reward - reward received by taking action (-1 for each step, 10 for eating capsule, -500 for eating ghost, 500 for eating all capsules)
        done - True if it is end of the game, False otherwise
        score - temporarily score of the game, later it will be displayed on the screen
        '''

        width = len(self.board[0])
        height = len(self.board)

        # move player according to action

        if action == LEFT and self.player_pos['x'] > 0:
            if self.board[self.player_pos['y']][self.player_pos['x'] - 1] != 'w':
                self.player_pos['x'] -= 1
        if action == RIGHT and self.player_pos['x'] + 1 < width:
            if self.board[self.player_pos['y']][self.player_pos['x'] + 1] != 'w':
                self.player_pos['x'] += 1
        if action == UP and self.player_pos['y'] > 0:
            if self.board[self.player_pos['y'] - 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] -= 1
        if action == DOWN and self.player_pos['y'] + 1 < height:
            if self.board[self.player_pos['y'] + 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return self.__get_state(), reward, True, self.score

        # check if player eats food

        for food in self.foods:
            if food['x'] == self.player_pos['x'] and food['y'] == self.player_pos['y']:
                self.score += 10
                reward = 10
                self.foods.remove(food)
                break
        else:
            self.score -= 1
            reward = -1

        # move ghosts
        for ghost in self.ghosts:
            moved = False
            ghost_moves = [LEFT, RIGHT, UP, DOWN]
            if ghost['x'] > 0 and self.board[ghost['y']][ghost['x'] - 1] != 'w':
                if ghost['direction'] == LEFT:
                    if RIGHT in ghost_moves:
                        ghost_moves.remove(RIGHT)
            else:
                if LEFT in ghost_moves:
                    ghost_moves.remove(LEFT)

            if ghost['x'] + 1 < width and self.board[ghost['y']][ghost['x'] + 1] != 'w':
                if ghost['direction'] == RIGHT:
                    if LEFT in ghost_moves:
                        ghost_moves.remove(LEFT)
            else:
                if RIGHT in ghost_moves:
                    ghost_moves.remove(RIGHT)

            if ghost['y'] > 0 and self.board[ghost['y'] - 1][ghost['x']] != 'w':
                if ghost['direction'] == UP:
                    if DOWN in ghost_moves:
                        ghost_moves.remove(DOWN)
            else:
                if UP in ghost_moves:
                    ghost_moves.remove(UP)

            if ghost['y'] + 1 < height and self.board[ghost['y'] + 1][ghost['x']] != 'w':
                if ghost['direction'] == DOWN:
                    if UP in ghost_moves:
                        ghost_moves.remove(UP)
            else:
                if DOWN in ghost_moves:
                    ghost_moves.remove(DOWN)

            ghost['direction'] = random.choice(ghost_moves)

            if ghost['direction'] == LEFT and ghost['x'] > 0:
                if self.board[ghost['y']][ghost['x'] - 1] != 'w':
                    ghost['x'] -= 1
            if ghost['direction'] == RIGHT and ghost['x'] + 1 < width:
                if self.board[ghost['y']][ghost['x'] + 1] != 'w':
                    ghost['x'] += 1
            if ghost['direction'] == UP and ghost['y'] > 0:
                if self.board[ghost['y'] - 1][ghost['x']] != 'w':
                    ghost['y'] -= 1
            if ghost['direction'] == DOWN and ghost['y'] + 1 < height:
                if self.board[ghost['y'] + 1][ghost['x']] != 'w':
                    ghost['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return self.__get_state(), reward, True, self.score

        self.__draw_board()

        if len(self.foods) == 0:
            reward = 500
            self.score += 500

        if self.display_mode_on:
            clock.tick(5)  # to see better
        return self.__get_state(), reward, len(self.foods) == 0, self.score

    def __draw_board(self):
        '''
        Function displays current state of the board. Do not change this code
        '''
        if self.display_mode_on:
            self.screen.fill((0, 0, 0))

            y = 0

            for line in board:
                x = 0
                for obj in line:
                    if obj == 'w':
                        color = (0, 255, 255)
                        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, 60, 60))
                    x += 60
                y += 60

            color = (255, 255, 0)
            # pygame.draw.rect(self.screen, color, pygame.Rect(self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15, 30, 30))
            self.screen.blit(self.player_image,
                             (self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15))

            color = (255, 0, 0)
            for ghost in self.ghosts:
                # pygame.draw.rect(self.screen, color, pygame.Rect(ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15, 30, 30))
                self.screen.blit(self.ghost_image,
                                 (ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15))

            color = (255, 255, 255)

            for food in self.foods:
                pygame.draw.ellipse(self.screen, color,
                                    pygame.Rect(food['x'] * self.cell_size + 25, food['y'] * self.cell_size + 25, 10,
                                                10))

            pygame.display.flip()

    def __get_state(self):
        '''
        Function returns current state of the game
        :return: state
        '''
        ghosts = deepcopy(self.ghosts)
        [g_dict.pop('direction', None) for g_dict in ghosts]  # filter out directions
        return self.__state_to_hash([self.player_pos, ghosts, self.foods])

    def __hash_to_state(self, state):
        return deepcopy(self.all_states[state])

    def __state_to_hash(self, state):
        return next(i for i, in_state in enumerate(self.all_states_raw) if state == in_state)

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True


board = ["*   g",
         " www ",
         " w*  ",
         " www ",
         "p    "]

clock = pygame.time.Clock()
pacman = Pacman(board)
state = pacman.reset()
agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions=pacman.get_possible_actions)
'''
Apply QLearning algorithm for Pacman
'''

done = False
pacman.display_mode_on = False

##### Training #####
for i in tqdm(range(10000)):
    play_and_train(pacman, agent)

##### Playing #####
agent.turn_off_learning()
rewards = []
for i in tqdm(range(1000)):
    # if i % 100 == 0:
    #     pacman.display_mode_on = True
    #     play_and_train(pacman, agent)
    #     pacman.display_mode_on = False
    # else:
    rewards.append(play_and_train(pacman, agent))

plt.plot(rewards)
plt.ylabel('Reward')
plt.show()
