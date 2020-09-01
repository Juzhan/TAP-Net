import numpy as np
import os
from tqdm import tqdm
import time

from RL import load_data

import sys
sys.path.append('../')
import tools
import generate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


    # num_sample x block_num x dim
    gt_blocks = gt_blocks.cpu().numpy().transpose(0,2,1)
    # num_sample x block_num x dim
    gt_positions = gt_positions.cpu().numpy().transpose(0,2,1)

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
    print(gt_blocks.shape)
    return gt_blocks, gt_positions



def test_random(blocks_num, reward_type, num_samples, container_size):
    
    block_dim = len(container_size)


    # all_blocks, all_positions = load_data('./data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    all_blocks, all_positions = load_data_rand('../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    
    # import IPython
    # IPython.embed()


    for max_first in [False]:

        my_ratio=[]
        my_valid_size=[]
        my_box_size=[]
        my_stable_num=[]
        my_empty_size=[]

        if max_first == True:
            max_str = 'max'
        else:
            max_str = 'rand'

        print(max_str, reward_type, blocks_num)

        for block_index in tqdm(range(num_samples)):

            blocks = all_blocks[block_index]
            # positions = all_positions[block_index]

            if reward_type == 'C+P+S-lb-soft':
                positions, _, _, ratio, scores = tools.calc_positions_greedy( blocks, container_size, reward_type )
                # positions, _, _, ratio, scores = tools.calc_positions_lb_greedy( blocks[order], container_size, reward_type )
            elif reward_type == 'C+P+S-mul-soft' or reward_type == 'C+P+S-mcs-soft':
                positions, _, _, ratio, scores = tools.calc_positions_mcs( blocks, container_size, reward_type )
                # positions, _, _, ratio, scores = tools.calc_positions_mcs( blocks[order], container_size, reward_type )
            valid_size, box_size, empty_size, stable_num, packing_height = scores



            my_ratio.append(ratio/3)
            my_valid_size.append(valid_size)
            my_box_size.append(box_size)
            my_stable_num.append(stable_num)
            my_empty_size.append(empty_size)


        tools.draw_container_2d(blocks, positions, container_size, save_name='random')

        my_valid_size = np.array(my_valid_size)
        my_box_size = np.array(my_box_size)
        my_stable_num = np.array(my_stable_num)
        my_empty_size = np.array(my_empty_size)
        
        C = np.mean(my_valid_size / my_box_size)
        P = np.mean(my_valid_size / (my_valid_size + my_empty_size))
        S = np.mean(my_stable_num / 10)
    
        print("C: %.3f   P: %.3f  S: %.3f   R: %.3f" % ( C, P, S, np.mean(my_ratio) )  )
        
        # print(np.mean(my_ratio))



if __name__ == "__main__":
    test_random(10, 'C+P+S-lb-soft', 10000, [5, 100])

    # ss = test_random(70, 10, 'C+P+S-lb-soft', 'rand', [1, 50], [50, 500])
    # for s in ss:
    #     print(s)