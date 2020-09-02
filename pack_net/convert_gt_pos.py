# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import copy

sys.path.append('../')
import tools




class PackEngine:
    def __init__(self, container_width, container_height, max_blocks_num, reward_type='C+P+S-msc-soft', is_train=True):
        self.container_width = container_width
        self.container_height = container_height
        self.height_map = np.zeros(container_width).astype(int)

        self.container = np.zeros((container_width, container_height)).astype('int')

        # actions are triggered by letters
        self.value_action_map = [ i for i in range(container_width) ]
        self.nb_actions = container_width

        # for running the engine
        self.score = 0
        self.time = 0
        self.num_blocks = 0
        self.max_blocks_num = max_blocks_num
        # clear after initializing
        self.block_dim = 2
        # self.positions = []
        self.positions = np.zeros((max_blocks_num, self.block_dim)).astype(int)
        self.blocks = []
        self.stable = [False for i in range(max_blocks_num)]
        self.is_train = is_train

        self.reward_type = reward_type

        self.valid_size = 0
        self.empty_size = 0

        self.bounding_box = [0, 0]
        # greedy
        # self.level_free_space = [ [0] for i in range(container_height) ]
        # mcs
        self.level_free_space = [ [ 0, self.container_width-1 ] \
                                    for i in range(self.container_height) ]
        self.clear()


    def step(self, action, block, draw_img=False):
        '''
        LB
        action: 1D int, place position (x)
        block: 2D float, the block (w, h)
        '''
        # if the place position can place the block, punish it
        block = block.int()

        if True:
            self.blocks.append( block.cpu().numpy() )

            self.positions, self.container, self.stable, self.height_map, self.bounding_box, self.valid_size, self.empty_size, self.level_free_space = \
                tools.calc_one_position_greedy(self.blocks, self.num_blocks, self.positions, self.container, self.reward_type, self.stable, self.height_map,
                                    self.bounding_box, self.valid_size, self.empty_size, self.level_free_space)

            self.num_blocks += 1


            stable_num = np.sum(self.stable)
            box_size = np.max(self.height_map) * self.container_width
            # elif block_dim == 3:
                # box_size = np.max(heightmap) * container_size[0] * container_size[1]

            if self.num_blocks == 0:
                C = 0
                P = 0
                S = 0    
            else:
                C = self.valid_size / box_size
                P = self.valid_size / (self.empty_size + self.valid_size)
                S = stable_num / self.num_blocks

            reward = (C+P+S) / 3

        # Update time and reward
        self.time += 1

        done = False
        if self.time == self.max_blocks_num:
            self.clear()
            done = True
        
        state = np.copy(self.height_map)

        return state, reward, done, self.positions[self.num_blocks-1][0]


    def clear(self):
        # self.positions = []

        self.container = np.zeros_like(self.container)

        # for running the engine
        self.score = 0
        self.time = 0
        self.num_blocks = 0
        # self.positions = []
        self.positions = np.zeros_like(self.positions)
        self.blocks = []
        self.stable = [False for i in range(self.max_blocks_num)]

        self.valid_size = 0
        self.empty_size = 0

        self.bounding_box = [0, 0]
        # self.level_free_space = [ [0] for i in range(container_height) ]
        self.level_free_space = [ [ 0, self.container_width-1 ] \
                                    for i in range(self.container_height) ]

        return self.height_map


def load_data_gt(data_file, blocks_num, num_samples):
    '''
    Data initialization
    ----
    params
    ----
        xxx
    Returns
    ---
        gt_blocks: num-samples x block-dim x blocks-num
        gt_position: num-samples x block-dim x blocks-num
    '''
    
    gt_blocks = np.loadtxt(data_file + 'gt_blocks.txt').astype('float32')
    gt_positions = np.loadtxt(data_file + 'gt_pos.txt').astype('float32')

    gt_positions = torch.from_numpy(gt_positions)
    gt_positions = gt_positions.view(num_samples, -1, blocks_num)

    gt_blocks = torch.from_numpy(gt_blocks)
    gt_blocks = gt_blocks.view(num_samples, -1, blocks_num)
    
    order = [blocks_num - i-1 for i in range(blocks_num)  ]

    # inverse order
    gt_blocks = gt_blocks[:,:,order]
    gt_positions = gt_positions[:,:,order]

    if use_cuda:
        gt_positions = gt_positions.cuda().detach()
        gt_blocks = gt_blocks.cuda().detach()

    return gt_blocks, gt_positions


def load_data_rand(data_file, blocks_num, num_samples):
    '''
    Data initialization
    ----
    params
    ----
        xxx
    Returns
    ---
        gt_blocks: num-samples x block-dim x blocks-num
        gt_position: num-samples x block-dim x blocks-num
    '''
    
    gt_positions = np.loadtxt(data_file + 'pos.txt').astype('float32')

    gt_positions = torch.from_numpy(gt_positions)
    gt_positions = gt_positions.view(num_samples, -1, blocks_num).numpy()

    all_blocks = np.loadtxt( data_file + 'blocks.txt').astype('int')

    block_dim = 2
    rotate_types = np.math.factorial(block_dim)

    data_size = int(len(all_blocks) / rotate_types)
    all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
    all_blocks = all_blocks.transpose(0, 1, 3, 2)
    all_blocks = all_blocks.reshape( data_size, -1, block_dim )

    gt_blocks = all_blocks[:,:blocks_num]
    gt_blocks = torch.from_numpy(gt_blocks)
    
    print(gt_blocks.shape)
    return gt_blocks, gt_positions


def convert_data( engine, num_samples, blocks_num, reward_type, is_train_data=True):
    
    if is_train_data:
        gt_blocks, positions = load_data_rand('../data/rand_2d/pack-train-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    else:
        gt_blocks, positions = load_data_rand('../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)

    
    # gt_blocks = gt_blocks.transpose(2,1)
    # positions = positions.transpose(2,1)
    
    gt_height_map = []
    gt_pos_map = []

    for i_episode in tqdm(range(num_samples)):
        # if i_episode > 1:
        #     break

        block_index = 0
        for t in count():
            block = gt_blocks[i_episode][block_index][None,None,:]
            # NOTE
            # pos = positions[i_episode][block_index]
            pos = [2,3,4]

            # Select and perform an action
            pos_x = int(pos[0])

            height_map = copy.copy(engine.height_map)

            state, reward, done, pos_x = engine.step(pos_x, gt_blocks[i_episode][block_index])
            
            container = engine.container

            pos_map = np.zeros(container_width)
            pos_map[pos_x] = 1

            block_index += 1
            
            gt_height_map.append(height_map)
            gt_pos_map.append(pos_map)

            if done:
                break
            
    # np.savetxt('./data/gt_2d/pack-valid-%d-%d-5-1-5/mcs_height_map.txt' % (blocks_num, num_samples), gt_height_map)
    # np.savetxt('./data/gt_2d/pack-valid-%d-%d-5-1-5/mcs_pos_map.txt' % (blocks_num, num_samples), gt_pos_map)
    # return
    # if num_samples == 10000:
    #     np.savetxt('./data/gt_2d/pack-valid-%d-%d-5-1-5/mcs_height_map.txt' % (blocks_num, num_samples), gt_height_map)
    #     np.savetxt('./data/gt_2d/pack-valid-%d-%d-5-1-5/mcs_pos_map.txt' % (blocks_num, num_samples), gt_pos_map)
    # else:    
    #     np.savetxt('./data/gt_2d/pack-train-%d-%d-5-1-5/mcs_height_map.txt'  % (blocks_num, num_samples) , gt_height_map)
    #     np.savetxt('./data/gt_2d/pack-train-%d-%d-5-1-5/mcs_pos_map.txt'  % (blocks_num, num_samples) , gt_pos_map)
    
    # np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/lb_height_map.txt' % (blocks_num, num_samples), gt_height_map)
    # np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/lb_pos_map.txt' % (blocks_num, num_samples), gt_pos_map)

    if reward_type == 'C+P+S-mcs-soft':
        if is_train_data:
            np.savetxt('../data/rand_2d/pack-train-%d-%d-7-1-5/mcs_height_map.txt'  % (blocks_num, num_samples) , gt_height_map)
            np.savetxt('../data/rand_2d/pack-train-%d-%d-7-1-5/mcs_pos_map.txt'  % (blocks_num, num_samples) , gt_pos_map)
        else:    
            np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/mcs_height_map.txt' % (blocks_num, num_samples), gt_height_map)
            np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/mcs_pos_map.txt' % (blocks_num, num_samples), gt_pos_map)
    elif reward_type == 'C+P+S-lb-soft':
        if is_train_data:
            np.savetxt('../data/rand_2d/pack-train-%d-%d-7-1-5/lb_height_map.txt'  % (blocks_num, num_samples) , gt_height_map)
            np.savetxt('../data/rand_2d/pack-train-%d-%d-7-1-5/lb_pos_map.txt'  % (blocks_num, num_samples) , gt_pos_map)
        else:    
            np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/lb_height_map.txt' % (blocks_num, num_samples), gt_height_map)
            np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/lb_pos_map.txt' % (blocks_num, num_samples), gt_pos_map)

    print('Complete')



if __name__ == '__main__':
    # use to generate MCS data
    reward_type = 'C+P+S-mcs-soft'

    is_train = True
    num_samples = 1280
    blocks_num = 10

    container_width, container_height = 5, 100 # container_size
    container_size = (container_width, container_height)
    engine = PackEngine(container_width, container_height, blocks_num, reward_type )

    convert_data( engine, num_samples, blocks_num, reward_type, is_train)
    