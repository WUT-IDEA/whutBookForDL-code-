# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 18-3-5 下午3:28
'''

from __future__ import print_function
from six.moves import xrange

import numpy as np
from copy import deepcopy
import time
import os
import math
import random

STAND_POINT = 2
BARRICADE = -1
TRESSURE = 1


class QLearnig:
    def __init__(self, map):
        self.map = map
        self.initial_map = deepcopy(map)
        self.map_size = list(map.shape)
        self.dim = len(self.map_size)
        self.actions = None
        self.Qtable = None
        self.epsilon = 0.99
        self.learning_rate = 1e-1
        self.training_episodes = 50
        self.reward = 10.0
        self.discount = 0.9
        self.reach_terminal = False
        self.draw_dic = {BARRICADE: '^', 0: '-', STAND_POINT: 'P', TRESSURE: 'T'}
        self.log_text = 'log.txt'
        self.time_sleep = 0.2
        self.path = None

        # other parameter
        self.current_site = None
        self.trap_sites = None
        self.treasure_site = None

        # initialize
        self.initialize_parameters()
        self.initialize_actions()

        self.create_Qtable(heuristic=True)  # [num of action, num of state]

    def initialize_parameters(self):
        self.map = deepcopy(self.initial_map)
        self.current_site = self.find_site_matched_condition(map=self.map, condition=STAND_POINT)
        self.trap_sites = self.find_site_matched_condition(map=self.map, condition=BARRICADE)
        self.treasure_site = self.find_site_matched_condition(map=self.map, condition=TRESSURE)
        self.reach_terminal = False
        self.path = []

    def initialize_actions(self):
        self.actions = []
        for i in xrange(self.dim):
            tmp = [0] * self.dim
            tmp[i] = STAND_POINT
            self.actions.append(tuple(tmp))
            tmp[i] = BARRICADE
            self.actions.append(tuple(tmp))
        random.shuffle(self.actions)

    def find_site_matched_condition(self, map, condition):
        coordinate = np.where(map == condition)
        array = []
        for i in xrange(len(coordinate[0])):
            site = []
            for j in xrange(len(coordinate)):
                site.append(coordinate[j][i])
            array.append(site)
        array = array[0] if len(array) == 1 else array
        return np.asarray(array)

    def distance(self, site):
        # heuristic method
        def elud_distance(site1, site2):
            return np.sqrt(np.sum((site1 - site2) ** 2))

        return 1.0 / max(elud_distance(site, self.treasure_site), 1e-10)

    def recursion_DFS(self, head, tail):
        if len(tail) == 0:
            self.Qtable[:, head] = self.distance(np.asarray(head))
            return
            # iteration = tail[0]
        for i in xrange(tail[0]):
            self.recursion_DFS(head + [i], tail[1:])

    def create_Qtable(self, heuristic=False):
        action_num = len(self.actions)
        Qtable_shape = deepcopy(self.map_size)
        Qtable_shape.insert(0, action_num)
        self.Qtable = np.zeros(Qtable_shape)
        # if heuristic == True:
        #     # design your heuristic method for accelerating learning
        #     # self.recurrent(head=[], tail=Qtable_shape[1:])
        # else:
        #     pass

    def index_of_available_action(self):
        x, y = self.current_site
        if np.random.uniform() > self.epsilon or \
                self.current_site == (0, 0):
            return np.random.randint(4)
        else:
            return np.argmax(self.Qtable[:, x, y])

    def move_one_step(self, action):
        return self.current_site + action

    def visit_by_indexList(self, matrix, ind_list, first_index_default=False):
        '''
        :param matrix: map
        :param ind_list: list type of indices
        :param first_index_default: for batch training, map shape is [None, [map_shape]]
                                    if False,  map shape is [map_shape]
        :return: the coordinate of the matrix by ind_list
        '''
        if first_index_default == True:
            for i in xrange(self.dim):
                matrix = matrix[:, ind_list[i]]
        else:
            for i in xrange(self.dim):
                matrix = matrix[ind_list[i]]
        return matrix

    def tensor_equal(self, tensor1, tensor2):
        return sum(tensor1 == tensor2) == len(tensor1)

    def move(self, action):
        new_coordinate = self.move_one_step(action)
        movable_flag = True
        for i in xrange(self.dim):
            movable_flag &= 0 <= new_coordinate[i] < self.map_size[i]

        if movable_flag:
            movable_flag &= not self.visit_by_indexList(self.map, new_coordinate) == BARRICADE
            if movable_flag:
                if self.tensor_equal(new_coordinate, self.treasure_site):
                    return new_coordinate, self.reward
                else:
                    return new_coordinate, 0.0
                    # heuristic method
                    # return new_coordinate, self.distance(new_coordinate)
            else:
                return self.current_site, 0.0
        else:
            return self.current_site, 0.0

    def advanced_move(self):
        '''
        :return: return reachable sites of map
        '''
        candidate = []
        action_ind = []

        for ind, actn in enumerate(self.actions):
            next_site, reward = self.move(actn)
            if self.tensor_equal(next_site, self.current_site) or \
                    self.visit_by_indexList(self.map, next_site) == BARRICADE:
                # self.tensor_equal(next_site, self.current_site), stay still?
                # self.visit_by_indexList(self.map, next_site) == BARRICADE, is barricade?
                pass
            else:
                candidate.append((ind, next_site, reward))
                action_ind.append(ind)

        try:
            choice_num = len(candidate)
        except:
            raise Exception('No valid path in the game.')

        if np.random.uniform() > self.epsilon or self.tensor_equal(self.current_site, [0] * self.dim):
            return candidate[np.random.randint(choice_num)]
        else:
            subTensor = self.visit_by_indexList(self.Qtable, self.current_site, first_index_default=True)
            subTensor = np.asarray([subTensor[i] if i in action_ind else BARRICADE for i in xrange(len(subTensor))])
            # 不要使用 np.argmax，可能会使得模型左右/上下反复移动。
            # 这与actions的设计相关。但是设计actions较难，这样随机比较容易
            choice_num = np.random.choice(np.where(subTensor == max(subTensor))[0])
            # print(subTensor, self.actions[choice_num])
            for item in candidate:
                if item[0] == choice_num:
                    return item

    def run(self):
        for i in xrange(1, 1 + self.training_episodes):
            self.draw_map()
            while not self.reach_terminal:

                action_ind, next_site, reward = self.advanced_move()

                if not self.tensor_equal(next_site, self.treasure_site):
                    feedback = reward + self.discount * \
                               max(self.visit_by_indexList(self.Qtable, next_site, first_index_default=True))
                else:
                    feedback = reward
                    self.reach_terminal = True
                self.Qtable[action_ind][tuple(self.current_site)] += \
                    self.learning_rate * \
                    (feedback - self.visit_by_indexList(self.Qtable[action_ind], self.current_site))

                # update map matrix
                self.update_map(next_site)

                self.current_site = next_site
                self.path.append(next_site)

                self.draw_map()
            print('Training epoch %s / %s, it costs %s steps' % (i, self.training_episodes, len(self.path)))
            # self.write_log(epoch=i, log=self.path)
            self.initialize_parameters()

    def update_map(self, future_site):
        self.map[tuple(self.current_site)] = 0
        self.map[tuple(future_site)] = STAND_POINT

    def draw_map(self):
        easy_map = np.reshape(deepcopy(self.map), newshape=(-1))
        end_length = self.map_size[-1]

        def ind2str(ind, ch):
            if (ind + 1) % end_length == 0:
                return self.draw_dic[ch] + '\n'
            else:
                return self.draw_dic[ch]

        easy_map = [ind2str(ind, ch) for ind, ch in enumerate(easy_map)]
        easy_map = ''.join(easy_map)
        # print'\r{}'.format(easy_map),
        os.system('clear')
        print(easy_map)
        time.sleep(self.time_sleep)

    def write_log(self, epoch, log):
        line = 'Epoch %s, step num: %s\nsteps follow:\n(0, 0) -> ' % (epoch, len(self.path))
        for next_site in log:
            line += ' -> ('
            for i in xrange(self.dim):
                line += '%s, ' % (next_site[i])
            line += ') '
        with open(self.log_text, 'a') as stream:
            stream.write(line + '\n\n')


# two dimension search leaning
game_map = np.zeros(shape=(6, 6), dtype=int)

game_map[0, 1] = BARRICADE
game_map[1, 1] = BARRICADE
game_map[0, 0] = STAND_POINT
game_map[-1, -1] = TRESSURE
print(game_map)

# # one dimension search leaning
# game_map = np.zeros(shape=(4,), dtype=int)
# game_map[0] = 1
# game_map[-1] = 2
# print(game_map)

simple = QLearnig(game_map)
simple.run()
