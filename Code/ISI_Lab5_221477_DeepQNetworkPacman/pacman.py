import pygame
import random
import copy
from copy import deepcopy
from agent_DeepQNetwork import DQNAgent, build_model
import matplotlib.pyplot as plt
import numpy as np
import json
import functools

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
            Smol Pacman
        """
        self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))

        self.display_mode_on = True
        self.cell_size = 60
        pygame.init()
        self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))

        self.board = board
        self.width = len(self.board[0])
        self.height = len(self.board)
        self.board_shape = (self.width, self.height)
        self.state_size = 3 * self.width * self.height
        self.possible_actions = 4

        self.player_pos = dict()
        self.ghosts = []
        self.foods = []
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

        self.init_foods = self.foods.copy()
        self.init_ghosts = self.ghosts.copy()
        self.__draw_board()

    def reset(self):
        """ resets state of the pacmanironment """
        self.foods = copy.deepcopy(self.init_foods)
        self.ghosts = copy.deepcopy(self.init_ghosts)
        self.player_pos = self.init_player_pos.copy()
        self.score = 0
        return self.__get_state(json.dumps((self.player_pos, self.ghosts, self.init_foods)))

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
                return self.__get_state(
                    json.dumps((self.player_pos, self.ghosts, self.foods))), reward, True, self.score

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
                return self.__get_state(
                    json.dumps((self.player_pos, self.ghosts, self.foods))), reward, True, self.score

        self.__draw_board()

        if len(self.foods) == 0:
            reward = 500
            self.score += 500

        if self.display_mode_on:
            clock.tick(5)  # to see better
        return self.__get_state(json.dumps((self.player_pos, self.ghosts, self.foods))), reward, len(
            self.foods) == 0, self.score

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
            pygame.event.wait()
            pygame.display.flip()

    def get_possible_actions_for_state(self, state):
        player_pos = np.argmax(state[:int(self.state_size / 3)])
        XY = np.unravel_index(player_pos, self.board_shape)
        XY_dict = {'x': XY[0], 'y': XY[1]}

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

    def position_to_binary_vector(self, positions):
        vector = np.zeros((self.width * self.height))
        if positions is not None:
            np.put(vector, [positions], 1)

        return vector

    @functools.lru_cache(maxsize=None)
    def __get_state(self, state):
        '''
        Function returns current state of the game
        :return: state
        '''
        player_pos, ghosts, foods = json.loads(state)
        # State defined as vector consisting of binary sub-vectors:
        # - player position on board
        # - ghosts positions on board
        # - food positions on board

        # First identify and copy positions
        ghosts = deepcopy(ghosts)
        [g_dict.pop('direction', None) for g_dict in ghosts]  # filter out directions
        foods = deepcopy(foods)
        player = deepcopy(player_pos)

        # Get indices - set for ghost in case they stand at the same position
        ghosts_ids = list(
            set(np.ravel_multi_index(np.array([list(x.values()) for x in ghosts]).T, dims=self.board_shape)))
        if foods:
            foods_ids = np.ravel_multi_index(np.array([list(x.values()) for x in foods]).T, dims=self.board_shape)
        player_ids = np.ravel_multi_index(list(player.values()), dims=self.board_shape)

        # Now construct vectors
        ghosts_vec = self.position_to_binary_vector(ghosts_ids)
        if foods:
            foods_vec = self.position_to_binary_vector(foods_ids)
        else:
            foods_vec = self.position_to_binary_vector(None)
        player_vec = self.position_to_binary_vector(player_ids)

        # Concat all
        state = np.concatenate((player_vec, ghosts_vec, foods_vec))

        return state

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True


board = ["*   g",
         " www ",
         " w* *",
         " www ",
         " p   "]

clock = pygame.time.Clock()
pacman = Pacman(board)
pacman.display_mode_on = False

##### Model and agent #####

learning_rate = 0.005
model = build_model(pacman.state_size, pacman.possible_actions, learning_rate)
agent = DQNAgent(pacman.possible_actions, model, pacman.get_possible_actions_for_state)
agent.epsilon = 0.75

##### Training #####
episodes = 10000
batch_size = 64
train_reward_treshold = 444

e = 0
for e in range(episodes):
    summary = []
    for _ in range(100):
        total_reward = 0
        state = pacman.reset()

        for _ in range(1000):
            action = agent.get_action(state)
            next_state, reward, done, _ = pacman.step(action)
            total_reward += reward

            # add to experience memory
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        agent.replay(batch_size)

        summary.append(total_reward)
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(e, np.mean(summary), agent.epsilon))
    if np.mean(summary) > train_reward_treshold:
        print("You Win!")
        break

##### Test #####
agent.turn_off_learning()
e_test = 100

test_reward = []
for _ in range(e_test):
    total_reward = 0
    state = pacman.reset()

    for _ in range(1000):
        action = agent.get_action(state)
        next_state, reward, done, _ = pacman.step(action)
        total_reward += reward

        # add to experience memory
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    test_reward.append(total_reward)

plt.plot(test_reward)
plt.ylabel("Rewards")
plt.xlabel("Test episodes")
plt.title('Training for {} episodes, testing for {} games'.format(e, e_test))
plt.show()
