
"""Defines the main task for the PACK
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.spatial import ConvexHull
from matplotlib.path import Path

from tqdm import tqdm
import multiprocessing
import networkx as nx
import itertools

import tools
import generate

class PACKDataset(Dataset):
    def __init__(self, data_file, blocks_num, num_samples, seed, input_type, heightmap_type, allow_rot, container_width, mix_data_file=None, unit=1, no_precedence=False):
        '''
        Data initialization
        ----
        params
        ----
            input_type: str, the type of input data
                'simple':   [idx][w,(l),h]     [0,1,1,0,...][   ]

                'rot':      [idx][w,(l),h]     [0,1,1,0,...][   ]
                'rot-old':  [idx][w,(l),h]     [0,1,1,0,...][0]
   
                'bot':      [idx][w,(l),h]     [0,1,1,0,...][0,0,1,1,...][0,1,0,1,...]
                'bot-rot':  [idx][w,(l),h]     [0,1,1,0,...][0,0,0,0,0..][0,0,0,0,0..]

                'mul':      [idx][w,(l),h][  ] [0,1,1,0,...] # old
                'mul-with': [idx][w,(l),h][id] [0,1,1,0,...] # old

                'mul':      [idx][w,(l),h][  ] [0,1,1,0,...][0,0,1,1,...][0,1,0,1,...]
                'mul-with': [idx][w,(l),h][id] [0,1,1,0,...][0,0,1,1,...][0,1,0,1,...]

            allow_rot: bool, allow to rotate
                False:  the final dim of input will be blocks_num * 1
                True:   the final dim of input will be blocks_num * rotate_types
        '''
        super(PACKDataset, self).__init__()
        if seed is None:
            seed = np.random.randint(123456)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if mix_data_file is None:
            deps_move = np.loadtxt(data_file + 'dep_move.txt').astype('float32')
            rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('float32')
            rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('float32')
            
            blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')
            positions = np.loadtxt(data_file + 'pos.txt').astype('float32')
            container_index = np.loadtxt(data_file + 'container.txt').astype('float32')

        else:
            num_mid = int(num_samples / 2)
            print('Mixing... %d' % num_mid)
            deps_move = np.loadtxt(data_file + 'dep_move.txt').astype('float32')[:num_mid]
            positions = np.loadtxt(data_file + 'pos.txt').astype('float32')[:num_mid]
            container_index = np.loadtxt(data_file + 'container.txt').astype('float32')[:num_mid]

            rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('float32')
            rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('float32')
            blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')

            rot_num_mid = int( len(blocks) / 2)
            rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('float32')[:rot_num_mid]
            rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('float32')[:rot_num_mid]
            blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')[:rot_num_mid]

            mix_deps_move = np.loadtxt(mix_data_file + 'dep_move.txt').astype('float32')[:num_mid]
            mix_positions = np.loadtxt(mix_data_file + 'pos.txt').astype('float32')[:num_mid]
            mix_container_index = np.loadtxt(mix_data_file + 'container.txt').astype('float32')[:num_mid]

            mix_rotate_deps_small = np.loadtxt(mix_data_file + 'dep_small.txt').astype('float32')[:rot_num_mid]
            mix_rotate_deps_large = np.loadtxt(mix_data_file + 'dep_large.txt').astype('float32')[:rot_num_mid]
            mix_blocks = np.loadtxt(mix_data_file + 'blocks.txt').astype('float32')[:rot_num_mid]


            deps_move = np.vstack( (deps_move, mix_deps_move) )
            rotate_deps_small = np.vstack( (rotate_deps_small, mix_rotate_deps_small) )
            rotate_deps_large = np.vstack( (rotate_deps_large, mix_rotate_deps_large) )
            blocks = np.vstack( (blocks, mix_blocks) )
            positions = np.vstack( (positions, mix_positions) )
            container_index = np.vstack( (container_index, mix_container_index) )



        positions = torch.from_numpy(positions)
        positions = positions.view(num_samples, -1, blocks_num)
        
        deps_move = torch.from_numpy(deps_move)
        deps_move = deps_move.view(num_samples, -1, blocks_num)
        deps_move = deps_move.transpose(2, 1)
      
        block_dim = positions.shape[1]
        rotate_types = np.math.factorial(block_dim)

        # data_size = int(len(blocks) / rotate_types)
        # num_samples x rotate_types x block_dim x blocks_num
        blocks = blocks.reshape( num_samples, -1, block_dim, blocks_num)
        # num_samples x rotate_types x blocks_num x block_dim
        blocks = blocks.transpose(0, 1, 3, 2)
        # num_samples x (rotate_types * blocks_num) x block_dim
        blocks = blocks.reshape( num_samples, -1, block_dim )
        # num_samples x block_dim x (blocks_num * rotate_types)
        blocks = blocks.transpose(0,2,1)
        blocks = torch.from_numpy(blocks)

        # resolution
        blocks = blocks * unit
        # if unit<1:
        blocks = blocks.ceil()#.int()

        rotate_deps_small = rotate_deps_small.reshape( num_samples, -1, blocks_num, blocks_num )
        rotate_deps_large = rotate_deps_large.reshape( num_samples, -1, blocks_num, blocks_num )
        rotate_deps_small = rotate_deps_small.transpose(0,1,3,2)
        rotate_deps_large = rotate_deps_large.transpose(0,1,3,2)
        rotate_deps_small = rotate_deps_small.reshape( num_samples, blocks_num*rotate_types, blocks_num )
        rotate_deps_large = rotate_deps_large.reshape( num_samples, blocks_num*rotate_types, blocks_num )
        rotate_deps_small = rotate_deps_small.transpose(0,2,1)
        rotate_deps_large = rotate_deps_large.transpose(0,2,1)
        rotate_deps_small = torch.from_numpy(rotate_deps_small)
        rotate_deps_large = torch.from_numpy(rotate_deps_large)


        # check rotate type:
        if allow_rot == False:
            blocks = blocks[:,:,:blocks_num]
            rotate_types = 1

        blocks_index = torch.arange(blocks_num)
        blocks_index = blocks_index.unsqueeze(0).unsqueeze(0)
        # num_samples x 1 x (blocks_num * rotate_types)
        blocks_index = blocks_index.repeat(num_samples, 1, rotate_types).float()

        # import IPython
        
        container_index = torch.from_numpy(container_index)
        container_index = container_index.unsqueeze(1)
        container_index = container_index.repeat(1, 1, rotate_types).float()
        # num_samples x block_dim x (blocks_num * rotate_types)
        positions = positions.repeat(1,1,rotate_types)
        # num_samples x blocks_num x (blocks_num * rotate_types)
        deps_move = deps_move.repeat(1,1,rotate_types)
        
        # # # random shuffle
        # order = [ o for o in range(blocks_num) ]
        # np.random.shuffle(order)
        # order = order * rotate_types
        # order = np.array(order)
        # for r in range(1, rotate_types):
        #     order[r*blocks_num: ] += blocks_num
        # blocks = blocks[:,:,order]
        # blocks_index = blocks_index[:,:,order]
        # deps_move = deps_move[:,:,order]
        # rotate_deps_small = rotate_deps_small[:,:,order]
        # rotate_deps_large = rotate_deps_large[:,:,order]
        # container_index = container_index[:,:,order]
        # positions = positions[:,:,order]

        # No precedence setting
        if no_precedence == True:
            deps_move = torch.zeros_like(deps_move)
            rotate_deps_small = torch.zeros_like(rotate_deps_small)
            rotate_deps_large = torch.zeros_like(rotate_deps_large)

        # conbine the data into our final input
        if input_type == 'simple':
            # num_samples x (1 + block_dim) x (blocks_num * 1)
            self.static = torch.cat( (blocks_index, blocks), 1 )
            # num_samples x (blocks_num) x (blocks_num * 1)
            self.dynamic = deps_move
        elif input_type == 'rot':
            # num_samples x (1 + block_dim) x (blocks_num * rotate_types)
            self.static = torch.cat( (blocks_index, blocks), 1 )
            # num_samples x (blocks_num) x (blocks_num * rotate_types)
            self.dynamic = deps_move
        elif input_type == 'bot':
            # num_samples x (1 + block_dim) x (blocks_num * rotate_types)
            self.static = torch.cat( (blocks_index, blocks), 1 )
            # num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
            self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )
        elif input_type == 'bot-rot':
            # num_samples x (1 + block_dim) x (blocks_num * rotate_types)
            self.static = torch.cat( (blocks_index, blocks), 1 )
            # num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
            rotate_deps_small = torch.zeros_like(rotate_deps_small)
            rotate_deps_large = torch.zeros_like(rotate_deps_large)
            self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )
        
        elif input_type == 'use-static' or input_type == 'use-pnet':
            # num_samples x (1 + block_dim) x (blocks_num * rotate_types)
            self.static = torch.cat( (blocks_index, blocks), 1 )
            # num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
            rotate_deps_small = torch.zeros_like(rotate_deps_small)
            rotate_deps_large = torch.zeros_like(rotate_deps_large)
            self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )

        elif input_type == 'mul' or input_type == 'mul-with':
            # num_samples x (1 + block_dim + 1) x (blocks_num * rotate_types)
            self.static = torch.cat( (blocks_index, blocks, container_index), 1 )
            # num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
            self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )

        elif input_type == 'rot-old':
            rotate_state = torch.zeros_like(blocks_index)
            # num_samples x (1 + block_dim) x (blocks_num * rotate_types)
            self.static = torch.cat( (blocks_index, blocks), 1 )
            # num_samples x (blocks_num + 1) x (blocks_num * rotate_types)
            self.dynamic = torch.cat( (deps_move, rotate_state), 1 )

        else:
            print('Dataset OHHHHH')
        
        print('    Static shape:  ', self.static.shape)
        print('    Dynamic shape: ', self.dynamic.shape)

        static_dim = block_dim
        heightmap_num = 1
        
        if heightmap_type == 'diff':
            if block_dim == 2:
                heightmap_width = container_width * unit - 1
            elif block_dim == 3:
                heightmap_num = 2
                heightmap_width = container_width * unit
                heightmap_length = container_width * unit
        else:
            heightmap_width = container_width * unit
            heightmap_length = container_width * unit

        # if unit < 1:
        heightmap_width = np.ceil(heightmap_width).astype(int)
        if block_dim==3: heightmap_length = np.ceil(heightmap_length).astype(int)


        if input_type == 'mul' or input_type == 'mul-with':
            if block_dim == 2:
                heightmap_width = heightmap_width * 2
            else:
                heightmap_num = heightmap_num * 2

        if input_type == 'mul-with':
            static_dim = static_dim + 1

        if block_dim == 2:
            self.decoder_static = torch.zeros(num_samples, static_dim, 1, requires_grad=True)
            self.decoder_dynamic = torch.zeros(num_samples, heightmap_width, 1, requires_grad=True)
        elif block_dim == 3:
            self.decoder_static = torch.zeros(num_samples, static_dim, 1, requires_grad=True)
            self.decoder_dynamic = torch.zeros(num_samples, heightmap_num, heightmap_width, heightmap_length, requires_grad=True)

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.decoder_static[idx], self.decoder_dynamic[idx])


def update_mask(mask, dynamic, static, chosen_idx, input_type, allow_rot):
    """
    Return two mask
        new_mask for current selection
        chosen_mask for next selection
    """
    # mask: batch_size x size
    # dynamic: batch_size x _size x size
    # chosen_idx: batch_size

    batch_size = chosen_idx.shape[0]

    if input_type == 'simple':
        # real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'rot' or input_type == 'rot-old':
        # real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'bot' or input_type == 'bot-rot':
        # real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'use-static' or input_type == 'use-pnet':
        # real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'mul' or input_type == 'mul-with':
        # real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 2))
    else:
        print('mask OHHHHHH')

    if allow_rot == False:
        rotate_types = 1
    else:
        rotate_types = np.math.factorial(block_dim)

    blocks_num = int(dynamic.shape[-1] / rotate_types)

    # real index get real index (the order of input set)
    real_chosen_idx = chosen_idx.clone()
    while (real_chosen_idx >= blocks_num).any():
        real_chosen_idx[ real_chosen_idx >= blocks_num ] -= blocks_num

    chosen_mask = mask.clone()
    # let the chosen object not selectable
    for i in range(rotate_types):
        chosen_mask = chosen_mask.scatter(1, (real_chosen_idx + blocks_num * i).unsqueeze(1), 0)    

    new_mask = chosen_mask.clone()
    move_mask = dynamic[:, :blocks_num, :].sum(1)
    rotate_small_mask = dynamic[:, blocks_num:blocks_num*2, :].sum(1)
    rotate_large_mask = dynamic[:, blocks_num*2:blocks_num*3, :].sum(1)
    rotate_mask = rotate_small_mask * rotate_large_mask
    dynamic_mask = rotate_mask + move_mask
    new_mask[ dynamic_mask.ne(0) ] = 0.
    
    return new_mask.float(), chosen_mask

def update_dynamic(dynamic, static, chosen_idx, input_type, allow_rot):
    """Updates the (load, demand) dataset values."""
    # TODO all of them
    batch_size = chosen_idx.shape[0]

    if input_type == 'simple':
        real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
        update_time = 1
    elif input_type == 'rot' or input_type == 'rot-old':
        real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
        update_time = 1
    elif input_type == 'bot' or input_type == 'bot-rot':
        real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
        update_time = 3
    elif input_type == 'use-static' or input_type == 'use-pnet':
        real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 1))
        update_time = 3
    elif input_type == 'mul' or input_type == 'mul-with':
        real_chosen_idx = static[:,0,:][ range(batch_size), chosen_idx ].long()
        block_dim = int((static.shape[1] - 2))
        update_time = 3
    else:
        print('dynamic OHHHHHHHH')

    # TODO rotate mask
    if allow_rot == False:
        rotate_types = 1
    else:
        rotate_types = np.math.factorial(block_dim)

    blocks_num = int(dynamic.shape[-1] / rotate_types)

    # batch_size x size x sourceL
    new_dynamic = dynamic.clone()

    for i in range(update_time):
        chosen_idx = (real_chosen_idx + blocks_num * i).unsqueeze(1).unsqueeze(1).repeat(1,1, dynamic.shape[-1] )
        new_dynamic.scatter_(1, chosen_idx, 0)
    
    return new_dynamic

def reward(static, tour_indices, reward_type, input_type, allow_rot, container_width, container_height, packing_strategy='LB_GREEDY'):
    """
    Deprecated, we output reward after container packed all the blocks in actor network
    --------------
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. w, h) data (batch_size, sourceL, num_cities)
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """
    
    use_cuda = static.is_cuda
    # TODO new type of input_type
    if input_type == 'simple':
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'rot' or input_type == 'rot-old':
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'bot' or input_type == 'bot-rot':
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'use-static' or input_type == 'use-pnet':
        block_dim = int((static.shape[1] - 1))
    elif input_type == 'mul' or input_type == 'mul-with':
        block_dim = int((static.shape[1] - 2))
    else:
        print('REWARD OHHHHHH')
    
    if block_dim == 3:
        container_size = [container_width, container_width, container_height]
    else:
        container_size = [container_width, container_height]
    if input_type == 'mul' or input_type == 'mul-with':
        if block_dim == 3:
            container_size_a = [container_width, container_width,container_height]
            container_size_b = container_size_a
        else:
            container_size_a = [container_width, container_height]
            container_size_b = container_size_a

    # batch_size x (1 + block_dim + block_dim) x (blocks_num * rotate_types)
    # static
    dim1 = static.shape[1]
    dim2 = static.shape[2]

    if allow_rot == False:
        rotate_types = 1
    else:
        rotate_types = np.math.factorial(block_dim)

    if packing_strategy == 'MACS' or packing_strategy == 'MUL':
        calc_position_fn = tools.calc_positions_mus
    elif reward_type == 'C+P+S-SL-soft' or reward_type == 'C+P+S-RL-soft' or reward_type == 'C+P+S-G-soft' or reward_type == 'C+P+S-LG-soft':
        calc_position_fn = tools.calc_positions_net
    else:
        # calc_position_fn = tools.calc_positions_greedy
        calc_position_fn = tools.calc_positions_lb_greedy

    blocks_num = int(dim2 / rotate_types)
    
    # batch_size x blocks_num  // tour_indices and expand as(static)
    idx = tour_indices.unsqueeze(1).repeat(1, dim1, rotate_types)

    # batch_size x (1 + block_dim) x blocks_num
    sample_solution = torch.gather(static.data, 2, idx)[:,:,:blocks_num]
    sample_solution = sample_solution.cpu().numpy()
    
    batch_size = sample_solution.shape[0]
    scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()
        
    for batch_index in range(batch_size):
        # if input_type == 'simple':
        blocks = sample_solution[batch_index,1:1+block_dim,:]
        blocks = blocks.transpose(1,0)
        if input_type == 'mul' or input_type == 'mul-with':
            target_ids = sample_solution[batch_index,-1,:]
            blocks_a = blocks[target_ids == 0]
            blocks_b = blocks[target_ids == 1]

            if len(blocks_a) == 0:
                scores_a = 0
            else:
                _, _, _, scores_a, _ = calc_position_fn(blocks_a, container_size_a, reward_type)
            if len(blocks_b) == 0:
                scores_b = 0
            else:
                _, _, _, scores_b, _ = calc_position_fn(blocks_b, container_size_b, reward_type)
            scores[batch_index] = (scores_a + scores_b) / 2
        else:
            _, _, _, scores[batch_index], _ = calc_position_fn(blocks, container_size, reward_type)
    
    return -scores

def create_dataset_gt(blocks_num, train_size, valid_size, obj_dim, target_container_width, target_container_height, initial_container_width, initial_container_height, input_type, arm_size, size_range, seed=None):
    blocks_num = int(blocks_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    if obj_dim == 2:
        initial_container_size = [initial_container_width, initial_container_height]
        target_container_size = [target_container_width, target_container_height]
        train_dir = './data/gt_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/gt_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
    elif obj_dim == 3:
        initial_container_size = [initial_container_width, initial_container_width, initial_container_height]
        target_container_size = [target_container_width, target_container_width, target_container_height]
        train_dir = './data/gt_3d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/gt_3d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(train_dir + 'blocks.txt') and os.path.exists(valid_dir  + 'blocks.txt'):
        return train_dir, valid_dir
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    # random_distribution = None
    # if initial_container_width < -1:
    random_distribution = generate.generate_height_prob(obj_dim, blocks_num, size_range, initial_container_width, target_container_width)
    prob, key = random_distribution

    # print('random_distribution: \n', prob, '\n', key)

    def generate_data(data_dir, data_size):

        container_ids = np.ones(blocks_num)
        container_ids[:int(blocks_num/2)] = 0
        np.random.shuffle(container_ids)
        
        blocks_f = open(data_dir + 'blocks.txt', 'w')
        pos_f = open(data_dir + 'pos.txt', 'w')
        container_f = open(data_dir + 'container.txt', 'w')
        
        deps_move_f = open(data_dir + 'dep_move.txt', 'w')
        rotate_deps_small_f = open(data_dir + 'dep_small.txt', 'w')
        rotate_deps_large_f = open(data_dir + 'dep_large.txt', 'w')

        for data_index in tqdm(range(data_size)):
            if random_distribution is not None:
                target_container_size[-1] = np.random.choice( key, p=prob )
            rotate_blocks, positions, deps_move, rotate_deps_small, rotate_deps_large = \
                    generate.generate_blocks_with_GT(blocks_num, target_container_size, initial_container_size, 
                                                        arm_size, size_range, input_type, data_index, allow_rot=True)

            for blocks_index, blocks in enumerate(rotate_blocks):
                blocks_f.writelines(arr2str( blocks ) )
                rotate_deps_small_f.writelines(arr2str( rotate_deps_small[blocks_index] ))
                rotate_deps_large_f.writelines(arr2str( rotate_deps_large[blocks_index] ))
                
            pos_f.writelines(arr2str( positions ) )
            deps_move_f.writelines( arr2str( deps_move ) )
            container_f.writelines( arr2str( container_ids.astype('int') )  )

        blocks_f.close()
        rotate_deps_small_f.close()
        rotate_deps_large_f.close()
        
        pos_f.close()
        deps_move_f.close()
        container_f.close()

    if not os.path.exists(train_dir + 'blocks.txt'):
        generate_data(train_dir, train_size)
    if not os.path.exists(valid_dir + 'blocks.txt'):
        generate_data(valid_dir, valid_size)
    return train_dir, valid_dir

def get_mix_dataset( blocks_num, train_size, valid_size, obj_dim, initial_container_width, size_range, seed=None):
    blocks_num = int(blocks_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    if obj_dim == 2:
        train_dir_1 = './data/gt_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        train_dir_2 = './data/rand_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

        valid_dir_1 = './data/gt_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir_2 = './data/rand_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        
    elif obj_dim == 3:
        train_dir_1 = './data/gt_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        train_dir_2 = './data/rand_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

        valid_dir_1 = './data/gt_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir_2 = './data/rand_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

    return train_dir_1, train_dir_2, valid_dir_1, valid_dir_2

def create_dataset( blocks_num, train_size, valid_size, obj_dim, initial_container_width, initial_container_height, arm_size, size_range, seed=None):
    blocks_num = int(blocks_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    if obj_dim == 2:
        initial_container_size = [initial_container_width, initial_container_height]
        train_dir = './data/rand_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/rand_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
    elif obj_dim == 3:
        initial_container_size = [initial_container_width, initial_container_width, initial_container_height]
        train_dir = './data/rand_3d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/rand_3d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(initial_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(train_dir + 'blocks.txt') and os.path.exists(valid_dir  + 'blocks.txt'):
        return train_dir, valid_dir
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)


    random_distribution = None
    random_num = False
    if initial_container_width == -1:
        random_distribution = generate.generate_deps_prob(obj_dim, blocks_num, 0)
    elif initial_container_width < -1:
        random_distribution = generate.generate_deps_prob(obj_dim, blocks_num, 0)
        random_num = True

        
    
    def generate_data(data_dir, data_size):


        print('here')

        blocks_f = open(data_dir + 'blocks.txt', 'w')
        pos_f = open(data_dir + 'pos.txt', 'w')
        container_f = open(data_dir + 'container.txt', 'w')
        
        deps_move_f = open(data_dir + 'dep_move.txt', 'w')
        rotate_deps_small_f = open(data_dir + 'dep_small.txt', 'w')
        rotate_deps_large_f = open(data_dir + 'dep_large.txt', 'w')

        for _ in tqdm(range(data_size)):
            
            container_ids = np.ones(blocks_num)
            container_ids[:int(blocks_num/2)] = 0
            np.random.shuffle(container_ids)
            # container_f.writelines( arr2str( container_ids.astype('int') )  )

            # continue
            rotate_blocks, positions, deps_move, rotate_deps_small, rotate_deps_large = \
                generate.generate_blocks(blocks_num, initial_container_size, arm_size, size_range, random_distribution, random_num)

            for blocks_index, blocks in enumerate(rotate_blocks):
                blocks_f.writelines(arr2str( blocks ) )
                rotate_deps_small_f.writelines(arr2str( rotate_deps_small[blocks_index] ))
                rotate_deps_large_f.writelines(arr2str( rotate_deps_large[blocks_index] ))
                
            pos_f.writelines(arr2str( positions ) )
            deps_move_f.writelines( arr2str( deps_move ) )
            container_f.writelines( arr2str( np.random.random_integers(0,1, blocks_num) )  )

        blocks_f.close()
        rotate_deps_small_f.close()
        rotate_deps_large_f.close()
        
        pos_f.close()
        deps_move_f.close()
        container_f.close()

    
    if not os.path.exists(train_dir + 'blocks.txt'):
        generate_data(train_dir, train_size)
    if not os.path.exists(valid_dir + 'blocks.txt'):
        generate_data(valid_dir, valid_size)
    return train_dir, valid_dir


def render(static, tour_indices, save_path, dynamic, valid_time, **kwargs):
    """Plots"""

    # batch_size x (1 + block_dim) x (blocks_num * rotate_types)
    dim1 = static.shape[1]
    dim2 = static.shape[2]

    if kwargs['input_type'] == 'simple':
        block_dim = int((static.shape[1] - 1))
    elif kwargs['input_type'] == 'rot' or kwargs['input_type'] == 'rot-old':
        block_dim = int((static.shape[1] - 1))
    elif kwargs['input_type'] == 'bot' or kwargs['input_type'] == 'bot-rot':
        block_dim = int((static.shape[1] - 1))
    elif kwargs['input_type'] == 'use-static' or kwargs['input_type'] == 'use-pnet':
        block_dim = int((static.shape[1] - 1))
    elif kwargs['input_type'] == 'mul' or kwargs['input_type'] == 'mul-with':
        block_dim = int((static.shape[1] - 2))
    else:
        print('Render OHHHH')

    if kwargs['allow_rot'] == False:
        rotate_types = 1
    else:
        rotate_types = np.math.factorial(block_dim)

    blocks_num = int(dim2 / rotate_types)

    # batch_size x blocks_num
    # tour_indices   expand as(static)
    idx = tour_indices.unsqueeze(1).repeat(1, dim1, rotate_types)

    all_blocks = static.data[:,1:1+block_dim,:].cpu().numpy()
    all_blocks = all_blocks.transpose(0, 2, 1).astype('int')

    # batch_size x (1 + block_dim + block_dim) x blocks_num
    sample_solution = torch.gather(static.data, 2, idx)[:,:,:blocks_num]
    sample_solution = sample_solution.cpu().numpy()

    batch_size = sample_solution.shape[0]
    
    data_len = len(tour_indices)

    container_width  = kwargs['container_width']  * kwargs['unit']
    container_height = kwargs['container_height'] * kwargs['unit']
    initial_container_width  = kwargs['initial_container_width']  * kwargs['unit']
    initial_container_height = kwargs['initial_container_height'] * kwargs['unit']

    container_width  = np.ceil(container_width).astype(int)
    container_height = np.ceil(container_height).astype(int)
    initial_container_width  = np.ceil(initial_container_width).astype(int)
    initial_container_height = np.ceil(initial_container_height).astype(int)

    if block_dim == 3:
        container_size = [container_width, container_width, container_height]
        initial_container_size = [initial_container_width, initial_container_width, initial_container_height]
    else:
        container_size = [container_width, container_height]
        initial_container_size = [initial_container_width, initial_container_height]
    if kwargs['input_type'] == 'mul' or kwargs['input_type'] == 'mul-with':
        if block_dim == 3:
            container_size_a = [container_width, container_width, initial_container_height]
            container_size_b = container_size_a
        else:
            container_size_a = [container_width, container_height]
            container_size_b = container_size_a
            
    my_ratio = []
    my_valid_size = []
    my_box_size = []
    my_empty_size = []
    my_stable_num = []
    my_packing_height = []

    if kwargs['packing_strategy'] == 'MACS' or kwargs['packing_strategy'] == 'MUL':
        calc_position_fn = tools.calc_positions_mcs
    elif kwargs['reward_type'] == 'C+P+S-SL-soft' or kwargs['reward_type'] == 'C+P+S-RL-soft' or \
         kwargs['reward_type'] == 'C+P+S-G-soft'  or kwargs['reward_type'] == 'C+P+S-LG-soft':
        calc_position_fn = tools.calc_positions_net
    else:
        # calc_position_fn = tools.calc_positions_greedy
        calc_position_fn = tools.calc_positions_lb_greedy


    for i in range(data_len):
        plt.close('all')
        fig = plt.figure()
        
        order = tour_indices[i].long().cpu().numpy()

        if kwargs['input_type'] == 'simple':
            blocks = sample_solution[i,1:1+block_dim,:]
        else:
            blocks = sample_solution[i,1:1+block_dim,:]
        
        blocks = blocks.transpose(1,0)

        my_all_blocks = all_blocks[i]
        
        if kwargs['input_type'] == 'mul' or kwargs['input_type'] == 'mul-with':
            target_ids = sample_solution[i,-1,:]
            blocks_a = blocks[target_ids == 0]
            blocks_b = blocks[target_ids == 1]


            if len(blocks_a) == 0:
                ratio_a, valid_size_a, box_size_a, empty_num_a, stable_num_a, packing_height_a = 0, 0, 0, 0, 0, 0
            else:
                _, _, _, ratio_a, scores_a = calc_position_fn(blocks_a, container_size_a, kwargs['reward_type'])
                valid_size_a, box_size_a, empty_num_a, stable_num_a, packing_height_a = scores_a
            if len(blocks_b) == 0:
                ratio_b, valid_size_b, box_size_b, empty_num_b, stable_num_b, packing_height_b = 0, 0, 0, 0, 0, 0
            else:
                _, _, _, ratio_b, scores_b = calc_position_fn(blocks_b, container_size_b, kwargs['reward_type'])
                valid_size_b, box_size_b, empty_num_b, stable_num_b, packing_height_b = scores_b

            ratio = (ratio_a + ratio_b) / 2 
            valid_size = (valid_size_a + valid_size_b) / 2 
            box_size = (box_size_a + box_size_b) / 2 
            empty_num = (empty_num_a + empty_num_b) / 2 
            stable_num = (stable_num_a + stable_num_b) / 2 
            packing_height = (packing_height_a + packing_height_b) / 2
        else:
            positions, container, stable, ratio, scores = calc_position_fn(blocks, container_size, kwargs['reward_type'])
            valid_size, box_size, empty_num, stable_num, packing_height = scores

        
        my_ratio.append(ratio)
        my_valid_size.append(valid_size)
        my_box_size.append(box_size)
        my_empty_size.append(empty_num)
        my_stable_num.append(stable_num)
        my_packing_height.append(packing_height)
        if i >= 6:
            continue

        C = valid_size / box_size
        P = valid_size / (valid_size + empty_num)
        S = stable_num / blocks_num


        # # paint
        # if input_type == 'mul' or input_type == 'mul-with':
        #     target_ids = sample_solution[i,-1,:]
        #     blocks_a = blocks[target_ids == 0]
        #     blocks_b = blocks[target_ids == 1]
            
        #     # calc the real index of blocks
        #     def calc_real_order(order, blocks_num):
        #         '''
        #         Calc the real index order of blocks
        #         ---
        #         params:
        #         ----
        #             order: 1 x n int array / list, the select order of blocks (contain rotation index)
        #             blocks_num: int, number of blocks
        #         return:
        #         ----
        #             real_order: 1 x n int array, the real block index (without rotation index)
        #             rotate_state: 1 x n bool array, the rotation state of each block
        #         '''
        #         real_order = []
        #         rotate_state = []
        #         for o in order:
        #             if o < blocks_num:
        #                 real_order.append(o)
        #                 rotate_state.append(False)
        #             else:
        #                 tmp = o
        #                 while tmp >= blocks_num:
        #                     tmp = tmp-blocks_num
        #                 real_order.append(tmp)
        #                 rotate_state.append(True)

        #         real_order = np.array(real_order)
        #         rotate_state = np.array(rotate_state)                        
        #         return real_order, rotate_state

        #     real_order, rotate_state = calc_real_order(order, blocks_num)

        #     order_a = real_order[ target_ids == 0 ]
        #     order_b = real_order[ target_ids == 1 ]

        #     rotate_state_a = rotate_state[ target_ids == 0 ]
        #     rotate_state_b = rotate_state[ target_ids == 1 ]

        #     blocks_num_a = np.sum(target_ids == 0)
        #     blocks_num_b = np.sum(target_ids == 1)

        #     if len(blocks_a) == 0:
        #         positions_a, container_a, stable_a = None, None, None
        #         feasibility_a = None
        #     else:
        #         # feasibility_a = tools.check_feasibility(blocks_a, blocks_num_a, initial_container_size, order_a, reward_type)
        #         positions_a, container_a, stable_a, _, _ = calc_position_fn(blocks_a, container_size_a, reward_type)
        #         feasibility_a = None
        #     if len(blocks_b) == 0:
        #         positions_b, container_b, stable_b = None, None, None
        #         feasibility_b = None
        #     else:
        #         # feasibility_b = tools.check_feasibility(blocks_b, blocks_num_b, initial_container_size, order_b, reward_type)
        #         positions_b, container_b, stable_b, _, _ = calc_position_fn(blocks_b, container_size_b, reward_type)
        #         feasibility_b = None
            
        
        #     if positions_a is None:
        #         C_a = 0
        #         P_a = 0
        #         S_a = 0
        #     else:
        #         C_a = valid_size_a / box_size_a
        #         P_a = valid_size_a / (valid_size_a + empty_num_a)
        #         S_a = stable_num_a / blocks_num_a

        #     if positions_b is None:
        #         C_b = 0
        #         P_b = 0
        #         S_b = 0
        #     else:
        #         C_b = valid_size_b / box_size_b
        #         P_b = valid_size_b / (valid_size_b + empty_num_b)
        #         S_b = stable_num_b / blocks_num_b
            

        #     if block_dim == 3:
        #         for view in ['front', 'right']:
        #             tools.draw_container_voxel( container_a[:,:,:25], blocks_num,
        #                 blocks_num_to_draw=blocks_num_a, 
        #                 order=order_a,
        #                 rotate_state=rotate_state_a,
        #                 reward_type=reward_type,
        #                 view_type=view, 
        #                 feasibility=feasibility_a,
        #                 save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C_a, P_a, S_a),
        #                 save_name=save_path[:-4] + '-a-%d' % (i) )
        #             tools.draw_container_voxel( container_b[:,:,:25], blocks_num, 
        #                 blocks_num_to_draw=blocks_num_b, 
        #                 order=order_b,
        #                 rotate_state=rotate_state_b,
        #                 view_type=view, 
        #                 reward_type=reward_type, 
        #                 feasibility=feasibility_b,
        #                 save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C_b, P_b, S_b),
        #                 save_name=save_path[:-4] + '-b-%d' % (i) )
        #     else:
        #         tools.draw_container_2d(blocks_a, positions_a, container_size_a, 
        #             order=order_a, 
        #             rotate_state=rotate_state_a, 
        #             stable=stable_a,
        #             reward_type=reward_type,
        #             feasibility=feasibility_a,
        #             save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C_a, P_a, S_a),
        #             save_name=save_path[:-4] + '-a-%d' % (i) )
        #         tools.draw_container_2d(blocks_b, positions_b, container_size_b, 
        #             order=order_b, 
        #             rotate_state=rotate_state_b, 
        #             stable=stable_b,
        #             feasibility=feasibility_b,
        #             reward_type=reward_type,
        #             save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C_b, P_b, S_b),
        #             save_name=save_path[:-4] + '-b-%d' % (i) )

        # else:
        #     positions, container, stable, _, _ = calc_position_fn(blocks, container_size, reward_type)
        #     new_order = []
        #     rotate_state = []
        #     # calc the real index of blocks
        #     for o in order:
        #         if o < blocks_num:
        #             new_order.append(o)
        #             rotate_state.append(False)
        #         else:
        #             tmp = o
        #             while tmp >= blocks_num:
        #                 tmp = tmp-blocks_num
        #             new_order.append(tmp)
        #             rotate_state.append(True)

        #     # feasibility = tools.check_feasibility(my_all_blocks, blocks_num, initial_container_size, order, reward_type)
        #     feasibility = None #tools.check_feasibility(my_all_blocks, blocks_num, initial_container_size, order, reward_type)
            
        #     order = new_order

        #     if block_dim == 3:
        #         tools.draw_container_voxel( container[:,:,:25], blocks_num, order=order,
        #             reward_type=reward_type, 
        #             rotate_state=rotate_state,
        #             feasibility=feasibility,
        #             save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C, P, S),
        #             save_name=save_path[:-4] + '-%d' % (i) )
        #     else:
        #         tools.draw_container_2d(blocks, positions, container_size, reward_type=reward_type,
        #             order=order, rotate_state=rotate_state, stable=stable,
        #             feasibility=feasibility,
        #             save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C, P, S),
        #             save_name=save_path[:-4] + '-%d' % (i) )


    np.savetxt( save_path[:-13] + '-ratio.txt',             my_ratio)
    np.savetxt( save_path[:-13] + '-valid_size.txt',        my_valid_size)
    np.savetxt( save_path[:-13] + '-box_size.txt',          my_box_size)
    np.savetxt( save_path[:-13] + '-empty_size.txt',        my_empty_size)
    np.savetxt( save_path[:-13] + '-stable_num.txt',        my_stable_num)
    np.savetxt( save_path[:-13] + '-packing_height.txt',    my_packing_height)
    total_time = np.array([valid_time])
    np.savetxt( save_path[:-13] + '-time.txt', total_time)
    
    ids = tour_indices.cpu().numpy()
    np.savetxt( save_path[:-13] + '-ids.txt', ids)
    
    # # draw precedence graph
    # if dynamic is not None:    
    #     for _i in range(data_len):
    #         if _i >= 6:
    #             break
    #         deps = dynamic[_i][:blocks_num, :blocks_num]

    #         tools.draw_dep(deps, save_name=save_path[:-4] + '-dep-%d' % (_i) )

