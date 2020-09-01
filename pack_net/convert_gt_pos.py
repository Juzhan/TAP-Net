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


# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda: print("....Using Gpu....")

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor

class PackEngine:
    def __init__(self, container_width, container_height, max_blocks_num, is_train=True):
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

        # if self.is_train == False:
        #     while action + block[0] > self.container_width:
        #         action -= 1
        
        # if action + block[0] > self.container_width:
        #     reward = -1
        # else:
        if True:
            # valid_size

            # get height_map

            # self.container, self.height_map, self.stable, \
            #     box_size, valid_size, empty_size, pos, self.num_blocks = add_block(block.cpu().numpy(), action, self.container, self.height_map, self.stable, self.num_blocks)

            self.blocks.append( block.cpu().numpy() )

            self.positions, self.container, self.stable, self.heightmap, self.valid_size, self.empty_size, self.level_free_space = tools.calc_one_position_mcs(
                self.blocks, self.num_blocks, self.positions, self.container, 'C+P+S-mcs-sof',
                self.stable, self.height_map, self.valid_size, self.empty_size, self.level_free_space )


            self.num_blocks += 1

            # update to next height map
            # min_height = self.height_map.min()
            # min_height = int(min_height)
            # print(self.height_map)
            

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
            if draw_img:
                tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
                    save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
                    save_name='convert_gt_%s' % (note) )

            self.clear()
            done = True
        
        # min_height = self.height_map.min()
        # min_height = int(min_height)
        
        state = np.copy(self.height_map)

        return state, reward, done, self.positions[self.num_blocks-1][0]



    def step_lb(self, action, block, draw_img=False):
        '''
        LB
        action: 1D int, place position (x)
        block: 2D float, the block (w, h)
        '''
        # if the place position can place the block, punish it
        block = block.int()

        # if self.is_train == False:
        #     while action + block[0] > self.container_width:
        #         action -= 1
        
        # if action + block[0] > self.container_width:
        #     reward = -1
        # else:
        if True:
            # valid_size

            # get height_map

            # self.container, self.height_map, self.stable, \
            #     box_size, valid_size, empty_size, pos, self.num_blocks = add_block(block.cpu().numpy(), action, self.container, self.height_map, self.stable, self.num_blocks)

            self.blocks.append( block.cpu().numpy() )

            self.positions, self.container, self.stable, self.height_map, self.bounding_box, self.valid_size, self.empty_size, self.level_free_space = \
                tools.calc_one_position_greedy(self.blocks, self.num_blocks, self.positions, self.container, 'C+P+S-lb-soft', self.stable, self.height_map,
                                    self.bounding_box, self.valid_size, self.empty_size, self.level_free_space)

            self.num_blocks += 1

            # update to next height map
            # min_height = self.height_map.min()
            # min_height = int(min_height)
            # print(self.height_map)
            

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

            # self.positions.append( pos )

            # punish bad
            # if reward < 0.8:
            #     reward = reward / 2

            # tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
            #     save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
            #     save_name='p' )

            # input()
            # a = 3
        # Update time and reward
        self.time += 1

        done = False
        if self.time == self.max_blocks_num:
            if draw_img:
                tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
                    save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
                    save_name='convert_gt_%s' % (note) )

            self.clear()
            done = True
        
        # min_height = self.height_map.min()
        # min_height = int(min_height)
        
        state = np.copy(self.height_map)

        return state, reward, done, self.positions[self.num_blocks-1][0]


    def step_(self, action, block, draw_img=False):
        '''
        action: 1D int, place position (x)
        block: 2D float, the block (w, h)
        '''
        # if the place position can place the block, punish it
        block = block.int()

        # if self.is_train == False:
        #     while action + block[0] > self.container_width:
        #         action -= 1
        
        if action + block[0] > self.container_width:
            reward = -1
        else:
        # if True:
            # valid_size

            # get height_map

            self.container, self.height_map, self.stable, \
                box_size, valid_size, empty_size, pos, self.num_blocks = add_block(block.cpu().numpy(), action, self.container, self.height_map, self.stable, self.num_blocks)



        # for block_index in range(blocks_num):
        #     positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space = \
        #         tools.calc_one_position_greedy(self.blocks, self.num_blocks, self.positions, self.container, 'C+P+S-lb-soft', self.stable, self.height_map,
        #                             bounding_box, valid_size, empty_size, level_free_space)


            # update to next height map
            # min_height = self.height_map.min()
            # min_height = int(min_height)
            # print(self.height_map)

            if box_size == 0:
                C = 0
            else:
                C = valid_size / box_size
            
            if valid_size + empty_size == 0:
                P = 0
            else:
                P = valid_size / (valid_size + empty_size)
            
            if self.num_blocks == 0:
                S = 0
            else:
                S = np.sum(self.stable) / self.num_blocks
            
            # reward = (C+P+S) / 3
            reward = C

            self.blocks.append( block.cpu().numpy() )
            self.positions.append( pos )

            # punish bad
            # if reward < 0.8:
            #     reward = reward / 2

            # tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
            #     save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
            #     save_name='p' )

            # input()
            # a = 3
        # Update time and reward
        self.time += 1

        done = False
        if self.time == self.max_blocks_num:
            if draw_img:
                tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
                    # save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
                    save_name='p_%s' % (note) )
                
            self.clear()
            done = True
        
        # min_height = self.height_map.min()
        # min_height = int(min_height)
        
        state = np.copy(self.height_map)

        return state, reward, done



    def clear(self):
        # self.positions = []
        

        self.height_map = np.zeros_like(self.height_map)

        self.container = np.zeros_like(self.container)

        # for running the engine
        self.score = 0
        self.time = 0
        self.num_blocks = 0
        # self.positions = []
        self.positions = np.zeros_like(self.positions)
        self.blocks = []
        self.stable = [False for i in range(max_blocks_num)]

        self.valid_size = 0
        self.empty_size = 0

        self.bounding_box = [0, 0]
        # self.level_free_space = [ [0] for i in range(container_height) ]
        self.level_free_space = [ [ 0, self.container_width-1 ] \
                                    for i in range(self.container_height) ]

        return self.height_map


def load_data(data_file, blocks_num, num_samples):
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

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()        
        self.conv_state = nn.Conv1d(1, 128, kernel_size=1)
        self.conv_block = nn.Conv1d(1, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.lin1 = nn.Linear(1792, 512)
        self.head = nn.Linear(512, engine.nb_actions)
        
        # self.conv_state = nn.Conv1d(1, 16, kernel_size=1)
        # self.conv_block = nn.Conv1d(1, 16, kernel_size=1)
        # self.bn1 = nn.BatchNorm1d(16)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.conv3 = nn.Conv1d(32, 32, kernel_size=1)
        # self.bn3 = nn.BatchNorm1d(32)
        # self.lin1 = nn.Linear(224, 128)
        # self.head = nn.Linear(128, engine.nb_actions)

    def forward(self, state, block):
        '''
        state: batch_size x 1 x container_width
        block: batch_size x 1 x 2
        '''
        encode_state = self.conv_state(state)
        encode_block = self.conv_block(block)

        output = F.relu(self.bn1( torch.cat((encode_state, encode_block), dim=-1) ))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.lin1(output.view(output.size(0), -1)))
        # return self.head(output.view(output.size(0), -1))
        return torch.softmax(self.head(output.view(output.size(0), -1)), dim=1)


BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 20

IS_TRAIN = True

LEARN_RATE = 5e-4
note = 'with next'

CHECKPOINT_FILE = './checkpoints/%s_checkpoint.pth.tar' % (note)
BEST_CHECKPOINT_FILE = './checkpoints/%s_best_checkpoint.pth.tar' % (note)

print(note)
container_width, container_height = 5, 100 # container_size
container_size = (container_width, container_height)
max_blocks_num = 10
engine = PackEngine(container_width, container_height, max_blocks_num)


steps_done = 0

policy_net = DQN()

# print(policy_net)

if use_cuda:
    policy_net.cuda()

loss = nn.MSELoss()
optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARN_RATE)



last_sync = 0


def add_block(block, pos_x, container, height_map, stable, current_blocks_num, is_train=True):
    '''
    Parameters:
    ---
        block: (w, h)
        pos-x: int
        container: 2d array
        height-map: 1d array (container-w)
        stable: 1d array, store the stable state  
        current_blocks_num
        is_train
    Returns:
    ---
        container
        height-map
        stable
        box-size, valid-size, empty-size,
        (pos-x, pos-z)
        current_blocks_num
    '''
    
    container_width, container_height = container.shape
    block_width, block_height = block
    
    block_width = int(block_width)
    block_height = int(block_height)
    pos_x = int(pos_x)
    
    if is_train == False:
        while pos_x + block_width > container_width:
            pos_x -= 1

    pos_z = int(height_map[pos_x:pos_x+block_width].max())
    

    block_id = current_blocks_num + 1
    block_id = int(block_id)


    if pos_x + block_width <= container_width:
        support = container[pos_x:pos_x+block_width, pos_z-1]
        if pos_z == 0:
            stable[block_id - 1] = True
        else:
            stable[block_id - 1] = tools.is_stable_2d(support, pos_x, block_width)
        
        container[ pos_x:pos_x+block_width, pos_z:pos_z+block_height ] = block_id

        under_block = container[pos_x:pos_x+block_width, :pos_z]
        container[pos_x:pos_x+block_width, :pos_z][ under_block == 0 ] = -1

        height_map[pos_x:pos_x+block_width] = height_map[pos_x:pos_x+block_width].max() + block_height

    else:
        pos_x = container_width
        stable[block_id - 1] = False

    if block_width != 0:
        current_blocks_num += 1

    # C P S
    box_size = (height_map.max() * container_width)
    valid_size = (container >= 1).sum()
    py_size = (container != 0).sum()
    empty_size = py_size - valid_size

    # if box_size == 0:
    #     C = 0
    # else:
    #     C = valid_size / box_size
    
    # if py_size == 0:
    #     P = 0
    # else:
    #     P = valid_size / py_size

    # if current_blocks_num == 0:
    #     S = 0
    # else:
    #     S = np.sum(stable) / current_blocks_num
    
    # reward = (C+P+S) / 3
    
    return container, height_map, stable, box_size, valid_size, empty_size, (pos_x, pos_z), current_blocks_num


def calc_positions(net, blocks, container_size, is_train=True):
    """
    Parameters:
    ---
        net: net
        blocks: batch-size x block-dim x blocks-num
        container-size: 2d array/list
    """
    batch_size = blocks.shape[0]
    blocks_num = blocks.shape[-1]
    
    container_width, container_height = container_size

    # batch_size x blocks_num x block_dim
    blocks = blocks.transpose(2, 1)

    containers = [np.zeros(container_size) for i in range(batch_size)]
    height_maps = [np.zeros(container_width) for i in range(batch_size)]
    stables = [np.zeros(blocks_num) for i in range(batch_size)]
    current_blocks_nums = np.zeros(batch_size)
    positions = np.zeros((batch_size, blocks_num, 2))
    
    box_size = np.zeros(batch_size)
    empty_size = np.zeros(batch_size)
    packing_height = np.zeros(batch_size)
    ratio = np.zeros(batch_size)
    stable_num = np.zeros(batch_size)
    valid_size = np.zeros(batch_size)

    for block_index in range(blocks_num):
        # batch_size x container_width
        real_height_map = np.array(height_maps)
        # batch_size x container_width
        real_height_map -= real_height_map.min(axis=1)[:, None].repeat(container_width, axis=1)
        # batch_size x 1 x container_width
        real_height_map = real_height_map[:, None, :]
        real_height_map = torch.from_numpy(real_height_map)
        
        block = blocks[:, block_index, :][:, None, :]

        action = net(
            Variable( real_height_map ).type(FloatTensor),
            block
        ).data.max(1)[1].view(-1,1).type(LongTensor)

        for batch_index in range(batch_size):
            containers[batch_index], height_maps[batch_index], stables[batch_index], \
                box_size[batch_index], valid_size[batch_index], empty_size[batch_index], \
                positions[batch_index][block_index], current_blocks_nums[batch_index] = \
                    add_block(blocks[batch_index][block_index], action[batch_index], \
                        containers[batch_index], height_maps[batch_index], \
                            stables[batch_index], current_blocks_nums[batch_index], is_train )

        # state, reward, done = engine.step(action[0,0])
        # state = FloatTensor(state[None,None,:,:])
    


    # def calc_C(packing_data):
    #     valid_size, box_size, empty_size, stable_num, packing_height = packing_data
    #     return valid_size / box_size


    # def calc_P(packing_data):
    #     valid_size, box_size, empty_size, stable_num, packing_height = packing_data
    #     return valid_size / (valid_size + empty_size)

    # def calc_S(packing_data):
    #     valid_size, box_size, empty_size, stable_num, packing_height = packing_data
    #     blocks_num = 10
    #     return stable_num / blocks_num

    for batch_index in range(batch_size):
        packing_height[batch_index] = height_maps[batch_index].max()
        stable_num[batch_index] = stables[batch_index].sum()
        
        
        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, save_name='gt')

        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, \
        #     save_title='C %.3f   S %.3f   P %.3f' % (C[batch_index], S[batch_index], P[batch_index]), \
        #         save_name='./img/gt_%d' % batch_index)
        # break
    C = valid_size / box_size
    P = valid_size / (valid_size + empty_size)
    S = stable_num / blocks_num


    for batch_index in range(batch_size):
        if batch_index > 3:
            break
        tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, \
            save_title='C %.3f   S %.3f   P %.3f' % (C[batch_index], S[batch_index], P[batch_index]), \
                save_name='./img/gt_%d' % batch_index)


    # Accumulate reward
    ratio = (C+P+S) / 3

    # ratio = torch.from_numpy(rewards) 


    return ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height 


def train(num_samples, blocks_num):
    
    # gt_blocks, positions = load_data('./data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    # gt_blocks, positions = load_data_rand('../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    if num_samples == 10000:
        # gt_blocks, positions = load_data('./data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
        gt_blocks, positions = load_data_rand('../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    else:
        # gt_blocks, positions = load_data('./data/gt_2d/pack-train-%d-%d-5-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)

        gt_blocks, positions = load_data_rand('../data/rand_2d/pack-train-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    
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

            # state, reward, done = engine.step(pos_x, gt_blocks[i_episode][block_index], draw_img=True)
            state, reward, done, pos_x = engine.step_lb(pos_x, gt_blocks[i_episode][block_index])
            
            container = engine.container

            pos_map = np.zeros(container_width)
            pos_map[pos_x] = 1

            block_index += 1
            
            # tools.draw_container_2d(gt_blocks[i_episode][:block_index], positions[i_episode][:block_index], 
            #     container_size,  save_title='%s\n%s' % (height_map, pos_map), save_name='p')
            # input('GO')

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

    if num_samples == 10000:
        np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/lb_height_map.txt' % (blocks_num, num_samples), gt_height_map)
        np.savetxt('../data/rand_2d/pack-valid-%d-%d-7-1-5/lb_pos_map.txt' % (blocks_num, num_samples), gt_pos_map)
    else:    
        np.savetxt('../data/rand_2d/pack-train-%d-%d-7-1-5/lb_height_map.txt'  % (blocks_num, num_samples) , gt_height_map)
        np.savetxt('../data/rand_2d/pack-train-%d-%d-7-1-5/lb_pos_map.txt'  % (blocks_num, num_samples) , gt_pos_map)
    print('Complete')



def valid(num_samples, blocks_num, checkpoint_file=None):
    
    blocks, positions = load_data('./data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    
    # blocks = blocks.transpose(2,1)
    # positions = positions.transpose(2,1)
    
    if checkpoint_file is None:
        checkpoint_file = BEST_CHECKPOINT_FILE

    checkpoint = torch.load(checkpoint_file)
    policy_net.load_state_dict(checkpoint['state_dict'])
    print('load checkpoint file')
    
    ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height = calc_positions(policy_net, blocks, container_size, is_train=False)

    np.savetxt('./render/box_size.txt', box_size)
    np.savetxt('./render/empty_size.txt', empty_size)
    np.savetxt('./render/packing_height.txt', packing_height)
    np.savetxt('./render/ratio.txt', ratio)
    np.savetxt('./render/stable_num.txt', stable_num)
    np.savetxt('./render/valid_size.txt', valid_size)

    print(np.mean(ratio))


if __name__ == '__main__':
    is_train = True

    if is_train:
        num_samples = 10000
        blocks_num = 10
    
        train(num_samples, blocks_num)
    # else:
    #     num_samples = 10000
    #     blocks_num = 10
    
    #     valid(num_samples, blocks_num)