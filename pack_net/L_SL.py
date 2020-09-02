'''
Local pack net trained using supervised learning

You can set you training parameters in main function.

if note == 'SL_rand' and heightmap_type == 'diff':
    
    The code will generate a folder named 'SL_rand_diff/', 
    the training result and checkpoints will store unber this folder

'''

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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import copy
import time

# sys.path.append('../')
# import tools


def is_stable_2d(support, obj_left, obj_width):
    '''
    check if the obj is stable
    ---
    params:
    ---
        support: obj_width x 1 array / list / tnesor, the container data under obj
        obj_left: a float number, left index of obj's position (x of obj.position)
        obj_width: a float number, width of obj
    return:
    ---
        is_stable: bool, stable or not
    '''
    object_center = obj_left + obj_width/2
    left_index = obj_left
    right_index = obj_left + obj_width
    for left in support:
        if left <= 0:
            left_index += 1
        else:
            break
    for right in reversed(support):
        if right <= 0:
            right_index -= 1
        else:
            break
    if left_index + 1 == right_index and obj_width == 1:
        return True
    if object_center <= left_index or object_center >= right_index:
        return False
    return True


class PackDataset(Dataset):
    def __init__(self, data_file, blocks_num, data_size, heightmap_type):
        '''
        Data initialization
        ----
        params
        ----
        '''
        super(PackDataset, self).__init__()

        gt_blocks = np.loadtxt(data_file + 'gt_blocks.txt').astype('float32')
        gt_positions = np.loadtxt(data_file + 'gt_pos.txt').astype('float32')
        gt_height_map = np.loadtxt(data_file + 'mcs_height_map.txt').astype('float32')
        gt_pos_map = np.loadtxt(data_file + 'mcs_pos_map.txt').astype('float32')


        gt_positions = torch.from_numpy(gt_positions)
        gt_positions = gt_positions.view(data_size, -1, blocks_num)

        gt_blocks = torch.from_numpy(gt_blocks)
        gt_blocks = gt_blocks.view(data_size, -1, blocks_num)

        gt_height_map = torch.from_numpy(gt_height_map)
        gt_height_map = gt_height_map.view(data_size, -1)
        gt_height_map = gt_height_map.view(data_size, blocks_num, -1)
        
        gt_pos_map = torch.from_numpy(gt_pos_map)
        gt_pos_map = gt_pos_map.view(data_size, -1)
        gt_pos_map = gt_pos_map.view(data_size, blocks_num, -1)
        
        order = [blocks_num - i-1 for i in range(blocks_num)  ]

        # inverse order
        gt_blocks = gt_blocks[:,:,order]
        gt_positions = gt_positions[:,:,order]

        gt_blocks = gt_blocks.transpose(2, 1)
        gt_positions = gt_positions.transpose(2, 1)


        gt_blocks = gt_blocks.contiguous()
        gt_positions = gt_positions.contiguous()
        gt_height_map = gt_height_map.contiguous()
        gt_pos_map = gt_pos_map.contiguous()

        gt_blocks = gt_blocks.view(data_size * blocks_num, 1, -1)
        gt_positions = gt_positions.view(data_size * blocks_num, 1, -1)
        gt_height_map = gt_height_map.view(data_size * blocks_num, 1, -1)
        gt_pos_map = gt_pos_map.view(data_size * blocks_num, 1, -1)
        
        total_size= data_size * blocks_num
        if heightmap_type == 'diff':
            for i in range(total_size):
                tmp = gt_height_map[i].clone()[0]
                tmp[:-1] = tmp[1:]
                tmp[-1] = gt_height_map[i][0][-1]

                gt_height_map[i] = tmp - gt_height_map[i]
        elif heightmap_type == 'zero':
            for i in range(total_size):
                gt_height_map[i] = gt_height_map[i] - torch.min(gt_height_map[i])
        
        self.gt_blocks = gt_blocks
        self.gt_positions = gt_positions
        self.gt_pos_map = gt_pos_map
        self.gt_height_map = gt_height_map
        
        self.data_size = data_size * blocks_num

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (self.gt_blocks[idx], self.gt_positions[idx], self.gt_height_map[idx], self.gt_pos_map[idx] )


class PackDataset_rand(Dataset):
    def __init__(self, data_file, blocks_num, data_size, heightmap_type):
        '''
        Data initialization
        ----
        params
        ----
        '''
        super(PackDataset_rand, self).__init__()

        gt_positions = np.loadtxt(data_file + 'pos.txt').astype('float32')

        gt_positions = torch.from_numpy(gt_positions)
        gt_positions = gt_positions.view(data_size, -1, blocks_num)

        all_blocks = np.loadtxt( data_file + 'blocks.txt').astype('int')

        block_dim = 2
        rotate_types = np.math.factorial(block_dim)

        data_size = int(len(all_blocks) / rotate_types)
        all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
        all_blocks = all_blocks.transpose(0, 1, 3, 2)
        all_blocks = all_blocks.reshape( data_size, -1, block_dim )

        gt_blocks = all_blocks[:,:blocks_num]

        gt_blocks = torch.from_numpy( gt_blocks.astype('float32') ).transpose(2,1)

        # ============
        gt_height_map = np.loadtxt(data_file + 'mcs_height_map.txt').astype('float32')
        gt_pos_map = np.loadtxt(data_file + 'mcs_pos_map.txt').astype('float32')

        gt_height_map = torch.from_numpy(gt_height_map)
        gt_height_map = gt_height_map.view(data_size, -1)
        gt_height_map = gt_height_map.view(data_size, blocks_num, -1)
        
        gt_pos_map = torch.from_numpy(gt_pos_map)
        gt_pos_map = gt_pos_map.view(data_size, -1)
        gt_pos_map = gt_pos_map.view(data_size, blocks_num, -1)
        

        gt_blocks = gt_blocks.transpose(2, 1)
        gt_positions = gt_positions.transpose(2, 1)


        gt_blocks = gt_blocks.contiguous()
        gt_positions = gt_positions.contiguous()
        gt_height_map = gt_height_map.contiguous()
        gt_pos_map = gt_pos_map.contiguous()

        gt_blocks = gt_blocks.view(data_size * blocks_num, 1, -1)
        gt_positions = gt_positions.view(data_size * blocks_num, 1, -1)
        gt_height_map = gt_height_map.view(data_size * blocks_num, 1, -1)
        gt_pos_map = gt_pos_map.view(data_size * blocks_num, 1, -1)
        
        total_size= data_size * blocks_num
        if heightmap_type == 'diff':
            for i in range(total_size):
                tmp = gt_height_map[i].clone()[0]
                tmp[:-1] = gt_height_map[i].clone()[0][1:]
                tmp[-1] = gt_height_map[i][0][-1]

                gt_height_map[i] = tmp - gt_height_map[i]
        elif heightmap_type == 'zero':
            for i in range(total_size):
                gt_height_map[i] = gt_height_map[i] - torch.min(gt_height_map[i])

        self.gt_blocks = gt_blocks
        self.gt_positions = gt_positions
        self.gt_pos_map = gt_pos_map
        self.gt_height_map = gt_height_map
        
        self.data_size = data_size * blocks_num

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (self.gt_blocks[idx], self.gt_positions[idx], self.gt_height_map[idx], self.gt_pos_map[idx] )


def load_data_gt(data_file, blocks_num, data_size):
    '''
    Data initialization
    ----
    params
    ----
        xxx
    Returns
    ---
        gt-blocks: data-size x block-dim x blocks-num
        gt-position: data-size x block-dim x blocks-num

        gt-height-map: data-size x blocks-num x block-dim
        gt-position: data-size x blocks-num x block-dim
    '''
    
    gt_blocks = np.loadtxt(data_file + 'gt_blocks.txt').astype('float32')
    gt_positions = np.loadtxt(data_file + 'gt_pos.txt').astype('float32')

    gt_height_map = np.loadtxt(data_file + 'mcs_height_map.txt').astype('float32')
    gt_pos_map = np.loadtxt(data_file + 'mcs_pos_map.txt').astype('float32')

    gt_positions = torch.from_numpy(gt_positions)
    gt_positions = gt_positions.view(data_size, -1, blocks_num)

    gt_blocks = torch.from_numpy(gt_blocks)
    gt_blocks = gt_blocks.view(data_size, -1, blocks_num)

    gt_height_map = torch.from_numpy(gt_height_map)
    gt_height_map = gt_height_map.view(data_size, -1)
    gt_height_map = gt_height_map.view(data_size, blocks_num, -1)
    
    gt_pos_map = torch.from_numpy(gt_pos_map)
    gt_pos_map = gt_pos_map.view(data_size, -1)
    gt_pos_map = gt_pos_map.view(data_size, blocks_num, -1)
    
    order = [blocks_num - i-1 for i in range(blocks_num)  ]

    # inverse order
    gt_blocks = gt_blocks[:,:,order]
    gt_positions = gt_positions[:,:,order]

    # if use_cuda:
    #     gt_positions = gt_positions.cuda().detach()
    #     gt_blocks = gt_blocks.cuda().detach()
    #     gt_height_map = gt_height_map.cuda().detach()
    #     gt_pos_map = gt_pos_map.cuda().detach()


    return gt_blocks, gt_positions, gt_height_map, gt_pos_map


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
    gt_positions = gt_positions.view(num_samples, -1, blocks_num)

    all_blocks = np.loadtxt( data_file + 'blocks.txt').astype('int')

    block_dim = 2
    rotate_types = np.math.factorial(block_dim)

    data_size = int(len(all_blocks) / rotate_types)
    all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
    all_blocks = all_blocks.transpose(0, 1, 3, 2)
    all_blocks = all_blocks.reshape( data_size, -1, block_dim )

    gt_blocks = all_blocks[:,:blocks_num]

    # gt_blocks = gt_blocks[:, ::-1, :]

    gt_blocks = torch.from_numpy( gt_blocks.astype('float32') ).transpose(2,1).cuda().detach()
    print(gt_blocks.shape)
    return gt_blocks, gt_positions



class DQN(nn.Module):

    def __init__(self, output_size, is_diff_height):
        super(DQN, self).__init__()      
        if is_diff_height:  
            self.conv_height_map = nn.Conv1d(1, 128, kernel_size=1)
            self.conv_block = nn.Conv1d(1, 128, kernel_size=1)
            self.bn1 = nn.BatchNorm1d(128)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(256)
            self.conv3 = nn.Conv1d(256, 256, kernel_size=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.lin1 = nn.Linear(1536, 512)
            self.head = nn.Linear(512, output_size)
        else:
            self.conv_height_map = nn.Conv1d(1, 128, kernel_size=1)
            self.conv_block = nn.Conv1d(1, 128, kernel_size=1)
            self.bn1 = nn.BatchNorm1d(128)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(256)
            self.conv3 = nn.Conv1d(256, 256, kernel_size=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.lin1 = nn.Linear(1792, 512)
            self.head = nn.Linear(512, output_size)
        
        self.is_diff_height = is_diff_height

    def forward(self, height_map, block):
        '''
        height_map: batch_size x 1 x container_width
        block: batch_size x 1 x 2
        '''
        # if self.is_zero_height:
        #     container_width = height_map.shape[-1]
        #     height_map -= height_map.min(-1)[0].unsqueeze(-1).repeat(1, 1, container_width)
        if self.is_diff_height:
            height_map = height_map[:,:,:-1]
        encode_height_map = self.conv_height_map(height_map)
        encode_block = self.conv_block(block)

        output = F.relu(self.bn1( torch.cat((encode_height_map, encode_block), dim=-1) ))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.lin1(output.view(output.size(0), -1)))
        # return self.head(output.view(output.size(0), -1))
        return F.softmax(self.head(output.view(output.size(0), -1)), dim=1)



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
            stable[block_id - 1] = is_stable_2d(support, pos_x, block_width)
        
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

    
    return container, height_map, stable, box_size, valid_size, empty_size, (pos_x, pos_z), current_blocks_num


def calc_positions(net, blocks, container_size, save_dir, heightmap_type, is_train=True):
    """
    Parameters:
    ---
        net: net
        blocks: batch-size x block-dim x blocks-num
        container-size: 2d array/list
    """

    use_cuda = blocks.is_cuda

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor

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
        real_height_map = np.array(height_maps).astype('float32')
        # batch_size x container_width
        if heightmap_type == 'diff':
            
            tmp = copy.deepcopy(real_height_map)
            tmp[:, :-1] = tmp[:, 1:]
            tmp[:, -1] = real_height_map[:,-1]

            real_height_map = tmp - real_height_map
            
        elif heightmap_type == 'zero':
            real_height_map -= real_height_map.min(axis=1)[:, None].repeat(container_width, axis=1)
        # batch_size x 1 x container_width
        real_height_map = real_height_map[:, None, :]
        real_height_map = torch.from_numpy(real_height_map)

        block = blocks[:, block_index, :][:, None, :]

        if use_cuda:
            real_height_map = real_height_map.cuda()

        action = net(
            real_height_map,
            block
        ).data.max(1)[1].view(-1,1).type(LongTensor)

        for batch_index in range(batch_size):
            containers[batch_index], height_maps[batch_index], stables[batch_index], \
                box_size[batch_index], valid_size[batch_index], empty_size[batch_index], \
                positions[batch_index][block_index], current_blocks_nums[batch_index] = \
                    add_block(blocks[batch_index][block_index], action[batch_index], \
                        containers[batch_index], height_maps[batch_index], \
                            stables[batch_index], current_blocks_nums[batch_index], is_train )

    for batch_index in range(batch_size):
        packing_height[batch_index] = height_maps[batch_index].max()
        stable_num[batch_index] = stables[batch_index].sum()

        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, \
        #     save_title='C %.3f   S %.3f   P %.3f' % (C[batch_index], S[batch_index], P[batch_index]), \
        #         save_name='./img/gt_%d' % batch_index)
    C = valid_size / box_size
    P = valid_size / (valid_size + empty_size)
    S = stable_num / blocks_num


    for batch_index in range(batch_size):
        if batch_index > 3:
            break
        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, \
        #     save_title='C %.3f   S %.3f   P %.3f' % (C[batch_index], S[batch_index], P[batch_index]), \
        #         save_name= save_dir + '/img/gt_%d' % batch_index)


    # Accumulate reward
    ratio = (C+P+S) / 3

    # ratio = torch.from_numpy(rewards) 


    return ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height 


def train(net, train_size, valid_size, blocks_num, heightmap_type, batch_size, epoch_num, learning_rate, save_dir, use_cuda, note):

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        os.makedirs(save_dir + '/img')

    
    loss = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

    if 'rand' in note:
        print('rand')
        train_data_file = '../data/rand_2d/pack-train-%d-%d-7-1-5/' % (blocks_num, train_size)
        valid_data_file = '../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, valid_size)
        train_data = PackDataset_rand( train_data_file, blocks_num, train_size, heightmap_type )
        valid_data = PackDataset_rand( valid_data_file, blocks_num, valid_size, heightmap_type )
    else:
        train_data_file = './data/gt_2d/pack-train-%d-%d-5-1-5/' % (blocks_num, train_size)
        valid_data_file = './data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, valid_size)
        train_data = PackDataset( train_data_file, blocks_num, train_size, heightmap_type )
        valid_data = PackDataset( valid_data_file, blocks_num, valid_size, heightmap_type )

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, valid_size, shuffle=False, num_workers=0)

    log_step = int(tarin_size / batch_size)
    if log_step > 100:
        log_step = int(100)
    if log_step == 0:
        log_step = int(1)
    
    my_loss = []
    total_losses = []
    times = []
    best_loss = 100

    
    for epoch in range(epoch_num):
        epoch_start = time.time()
        start = epoch_start
        
        for batch_idx, batch in enumerate(train_loader):

            blocks, positions, height_maps, gt_pos_maps = batch
            
            if use_cuda:
                blocks = blocks.cuda().detach()
                positions = positions.cuda().detach()
                height_maps = height_maps.cuda().detach()
                gt_pos_maps = gt_pos_maps.cuda().detach()
                
            gt_pos_maps = gt_pos_maps.view(-1, gt_pos_maps.shape[-1])
            
            net_predict_map = net(height_maps,  blocks)

            net_loss = loss( net_predict_map, gt_pos_maps )

            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()

            my_loss.append(torch.mean(net_loss.detach()).item())
            
            if (batch_idx + 1) % log_step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(my_loss[-log_step:])
                total_losses.append(mean_loss)

                print('    Epoch %d  Batch %d/%d, loss: %2.4f, took: %2.4fs' %
                      (epoch, batch_idx, len(train_loader), mean_loss, times[-1] ))

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'SL.pt')
        torch.save(net.state_dict(), save_path)

        mean_loss = np.mean(total_losses)
        # valid network
        with torch.no_grad():
            for batch in train_loader:
                blocks, positions, height_maps, gt_pos_maps = batch
                if use_cuda:
                    blocks = blocks.cuda().detach()
                    positions = positions.cuda().detach()
                    height_maps = height_maps.cuda().detach()
                    gt_pos_maps = gt_pos_maps.cuda().detach()
            
            net_valid_map = net(height_maps, blocks)

            valid_loss = loss(net_valid_map, gt_pos_maps)

            mean_valid = valid_loss.detach().item()

        if mean_valid < best_loss:
            best_loss = mean_valid
            save_path = os.path.join(save_dir, 'SL.pt')
            torch.save(net.state_dict(), save_path)

        print('Epoch %d,  mean epoch valid: %2.4f  | loss: %2.4f, took: %2.4fs '\
              '(%2.4fs / %d batches)' % \
              (epoch, mean_valid, mean_loss, time.time() - epoch_start,
              np.mean(times), log_step  ))

        import matplotlib.pyplot as plt
        plt.close('all')
        plt.title('Loss')
        plt.plot(range(len(total_losses)), total_losses, '-')
        plt.savefig(save_dir + '/img/loss.png' , bbox_inches='tight', dpi=400)

    np.savetxt(save_dir + '/loss.txt', total_losses)



def valid(actor, data_size, blocks_num, heightmap_type, checkpoint_path, use_cuda, save_dir):
    
    # blocks, _, _, _ = load_data_gt('./data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, data_size), blocks_num, data_size)
    blocks, _ = load_data_rand('../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, data_size), blocks_num, data_size)

    if use_cuda:
        blocks = blocks.cuda()
    
    if not os.path.exists(save_dir + '/render'):
        os.makedirs(save_dir + '/render')


    path = os.path.join(checkpoint_path, 'SL.pt')
    print(path)
    actor.load_state_dict(torch.load(path))
    print('load checkpoint file')
    
    ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height \
        = calc_positions(actor, blocks, container_size, save_dir, heightmap_type, is_train=False)

    np.savetxt( save_dir + '/render/box_size.txt', box_size)
    np.savetxt( save_dir + '/render/empty_size.txt', empty_size)
    np.savetxt( save_dir + '/render/packing_height.txt', packing_height)
    np.savetxt( save_dir + '/render/ratio.txt', ratio)
    np.savetxt( save_dir + '/render/stable_num.txt', stable_num)
    np.savetxt( save_dir + '/render/valid_size.txt', valid_size)

    C = np.mean(valid_size / (box_size))
    P = np.mean(valid_size / (valid_size + empty_size))
    S = np.mean(stable_num / 10 )

    print("C: %.3f   P: %.3f  S: %.3f   R: %.3f" % ( C, P, S, np.mean(ratio) )  )


if __name__ == '__main__':
    
    # ====== setting begin =========
    # if gpu is to be used
    use_cuda = True
    if use_cuda: 
        print("....Using Gpu....")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    batch_size = 256
    epoch_num = 200
    learning_rate = 5e-4

    container_width, container_height = 5, 100 # container_size
    container_size = (container_width, container_height)
    blocks_num = 10

    is_train = True

    heightmap_type = 'diff'
    if heightmap_type == 'diff':
        is_diff_height = True
    else:
        is_diff_height = False

    # note = 'SL_gt'
    # note = 'SL_lb'
    # note = 'SL_mcs'
    note = 'SL_rand'

    save_dir = './%s_%s' % (note,  heightmap_type)

    pack_net = DQN(container_width, is_diff_height)
    if use_cuda:
        pack_net = pack_net.cuda()

    if is_train:
        tarin_size = 1280
        valid_size = 100
        train(pack_net, tarin_size, valid_size, blocks_num, heightmap_type, batch_size, epoch_num, learning_rate, save_dir, use_cuda, note)
    else:
        print('valid')
        tarin_size = 100
        valid_size = 10000
        checkpoint_dir = os.path.join(save_dir, 'checkpoints/199')
        valid(pack_net, valid_size, blocks_num, heightmap_type, checkpoint_dir, use_cuda, save_dir)
        