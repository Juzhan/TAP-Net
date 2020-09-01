#
#   generate Ground Truth packing
#   from GT generate the inital layout and dependence
#


import itertools
import numpy as np
import networkx as nx
import math

from tools import draw_container_voxel, calc_positions
import tools
import copy
import time

def generate_blocks_with_GT(blocks_num, gt_packing_size, initial_container_size, arm_size, size_range, input_type, data_index, allow_rot=True):
    '''
    Generate the blocks data in a container
    ----
    params:
        blocks_num: int, the number of blocks want to generate
        container_size: 1 x 2/3 int array / list, the size of container, block_dim = len(container_size)
    return:
    ----
        rotate_blocks: (n x dim!) x dim int array, store all posible rotation state of these blocks 
        positions: n x dim int array, the positions of each block in initial container
        deps_move: n x n int array, movement dependence of blocks in initial container
        rotate_deps_small: n x n int array, rotation dependence of blocks in initial container on the small size (such as left size)
        rotate_deps_large: n x n int array, rotation dependence of blocks in initial container on the large size (such as right size)
    '''
    # np.random.seed(
    #     np.random.randint(12345679)
    # )

    block_dim = len(gt_packing_size)
    
    min_box = size_range[0]
    max_box = size_range[1]

    def check_block_size(block, size_range):
        min_size = size_range[0]
        max_size = size_range[1]
        for width in block:
            if width >= min_size and width < max_size:
                continue
            return False
        return True

    def check_all_blocks_size(blocks, size_range):
        for block in blocks:
            if check_block_size(block, size_range) == False:
                return False
        return True
    

    you_find_it = False

    while you_find_it == False:
        # generate Candidate Perfect Packing Solution
        if initial_container_size[0] <= 0:
            container_width = np.random.randint(5, 11)
            if block_dim == 2:
                initial_container_size = [ container_width, blocks_num * max_box + 10 ]
            elif block_dim == 3:
                initial_container_size = [ container_width, container_width, blocks_num * max_box + 10 ]

        while True:
            if block_dim == 3:
                gt_blocks, gt_positions, gt_container = BPP_Generator_3D(blocks_num, gt_packing_size, size_range)
            else:
                # gt_blocks, gt_positions, gt_container = BPP_Generator_2D(blocks_num, gt_packing_size, size_range)
                gt_blocks, gt_positions, gt_container = BPP_Generator_2D_easy(blocks_num, gt_packing_size, size_range)
            if check_all_blocks_size(gt_blocks, size_range):
                break

        gt_deps, _, _, _, _, _, _ = calc_dependent(gt_blocks, gt_positions, gt_container, arm_size)

        # rotate the blocks randomly
        rotates = []
        for p in itertools.permutations( range(block_dim) ):
            rotates.append(list(p))
        rotates = np.array(rotates)

        loop_time = 20
        while True:
            loop_time -= 1
            if loop_time < 0:
                break

            cpps_to_init_solution = []
            my_deps = copy.copy(gt_deps.transpose())
            while np.sum(my_deps) > 0 or len(cpps_to_init_solution) < blocks_num:
                # get the 0 deps idex
                candidate_idx = np.where( np.sum(my_deps, axis=1) == 0)[0]
                
                idx = np.random.choice(candidate_idx)
                if idx in cpps_to_init_solution:
                    continue
                else:
                    cpps_to_init_solution.append(idx)
                    my_deps[:,idx] = 0
            # get a sequence to generate initial layout        
            blocks = gt_blocks[cpps_to_init_solution]
            
            
            if allow_rot:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i][rotates[np.random.randint(0, len(rotates))]]
            
            # calc the positions in initial container
            positions, container, stable, _, _ = tools.calc_positions_lb_greedy(blocks, initial_container_size, 'C+P+S-lb-hard')
            # check stability
            if np.sum(stable) < blocks_num: continue
            deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down = \
                            calc_dependent(blocks, positions, container, arm_size)

            deps_move = deps_move.transpose()
            deps_left = deps_left.transpose()
            deps_right = deps_right.transpose()
            deps_forward = deps_forward.transpose()
            deps_backward = deps_backward.transpose()
            deps_up = deps_up.transpose()
            deps_down = deps_down.transpose()
            # remove of all of the deps if the order can work
            work = True
            for s in reversed(range(blocks_num)):
                if work == False:
                    break
                tmp_deps_move = np.sum(deps_move[s])
                tmp_deps_left = np.sum(deps_left[s])
                tmp_deps_right = np.sum(deps_right[s])
                tmp_deps_forward = np.sum(deps_forward[s])
                tmp_deps_backward = np.sum(deps_backward[s])
                tmp_deps_up = np.sum(deps_up[s])
                tmp_deps_down = np.sum(deps_down[s])

                x = tmp_deps_forward * tmp_deps_backward
                y = tmp_deps_left * tmp_deps_right
                z = tmp_deps_up * tmp_deps_down
                if input_type == 'simple':
                    x = 0
                    y = 0
                    z = 0
                if tmp_deps_move == 0 and x==0 and y==0 and z==0:
                    deps_move[:, s] = 0
                    deps_left[:, s] = 0
                    deps_right[:, s] = 0
                    deps_forward[:, s] = 0
                    deps_backward[:, s] = 0
                    deps_up[:, s] = 0
                    deps_down[:, s] = 0
                else:
                    work = False
                    
            if work == True:
                you_find_it = True
                break
        
    
    # # draw the Candidate Perfect Packing Solution
    # if data_index<300:
    #     init_to_cpps_order = []
    #     init_to_cpps_rotate = []
    #     for id in range(blocks_num):
    #         o = cpps_to_init_solution.index(id)
    #         init_to_cpps_order.append( o )
    #         if (blocks[o] == gt_blocks[id]).all():  init_to_cpps_rotate.append(False)
    #         else:                                   init_to_cpps_rotate.append(True)
    #     init_to_cpps_order  = np.array(init_to_cpps_order)
    #     init_to_cpps_rotate = np.array(init_to_cpps_rotate)
    #     # draw
    #     # all_yellow = np.array(['#ffd966'] * 10)
    #     tools.draw_container_2d(gt_blocks, gt_positions, gt_packing_size, order=init_to_cpps_order, 
    #                             save_name='visual/_add/%d_cpps' % data_index)
    #     # note = '10-41'
    #     # tools.draw_container_2d(gt_blocks, gt_positions, gt_packing_size, order=init_to_cpps_order, 
    #     #                         rotate_state=init_to_cpps_rotate, save_name='visual/cpps-2d-note-%s/%d_cpps' % (note, data_index))
    #     # tools.draw_container_voxel(gt_container, blocks_num, save_name='visual/3d-gt')


    # now to store the order
    # re-calc the deps
    # positions, container, stable, _, _ = tools.calc_positions_lb_greedy(blocks, initial_container_size, 'C+P+S-lb-hard')
    deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down = \
                            calc_dependent(blocks, positions, container, arm_size)

    rotate_blocks = []
    blocks = blocks.transpose()
    deps_left = deps_left.flatten()
    deps_right = deps_right.flatten()
    deps_forward = deps_forward.flatten()
    deps_backward = deps_backward.flatten()
    deps_up = deps_up.flatten()
    deps_down = deps_down.flatten()

    rotate_deps_small = []
    rotate_deps_large = []

    if block_dim == 3:
        for p in itertools.permutations( range(block_dim) ):
            if p[-1] == 0:
                rotate_deps_small.append(deps_left)
                rotate_deps_large.append(deps_right)
            elif p[-1] == 1:
                rotate_deps_small.append(deps_forward)
                rotate_deps_large.append(deps_backward)
            elif p[-1] == 2:
                rotate_deps_small.append(deps_down)
                rotate_deps_large.append(deps_up)
            rotate_blocks.append( blocks[list(p)].flatten() )
    else:
        for p in itertools.permutations( range(block_dim) ):
            if p[-1] == 0:
                rotate_deps_small.append(deps_left)
                rotate_deps_large.append(deps_right)
            elif p[-1] == 1:
                rotate_deps_small.append(deps_down)
                rotate_deps_large.append(deps_up)
            rotate_blocks.append( blocks[list(p)].flatten() )

    rotate_blocks = np.array(rotate_blocks)
    rotate_deps_small = np.array(rotate_deps_small)
    rotate_deps_large = np.array(rotate_deps_large)

    positions = np.array(positions)
    positions = positions.transpose().flatten()
    deps_move = deps_move.transpose().flatten()

    return rotate_blocks, positions, deps_move, rotate_deps_small, rotate_deps_large

def BPP_Generator_3D( blocks_num, gt_packing_size, size_range ):
    '''
    Return blocks, positions, container
    '''
    # np.random.seed(
    #     np.random.randint(12345679)
    # )
    # gt_packing_size[-1] = blocks_num
    
    min_size = size_range[0]
    max_size = size_range[1]

    block_dim = len(gt_packing_size)
    container = np.ones(gt_packing_size).astype(int)
    blocks = np.zeros((blocks_num, block_dim)).astype(int)
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    volumes = np.zeros(blocks_num).astype(int)
    
    blocks[0] = gt_packing_size
    volumes[0] = np.sum(container)

    def normalize(l):
        return ( np.array(l) / np.sum(l) ).tolist()
    
    for block_index in range(1, blocks_num):
        if block_index == 1:
            chosen_blk_idx = 0
        else:
            # chosen_blk_idx = np.argmax(volumes)    # choose the biggest block
            # choose a block randomly by volumes
            ids = [i for i in range(block_index)]
            prob = volumes[:block_index]
            prob = normalize(prob)
            chosen_blk_idx = np.random.choice(ids, p=prob)
        x, y, z = positions[chosen_blk_idx]
        X, Y, Z = blocks[chosen_blk_idx]
        blk_cube = container[x:x+X, y:y+Y, z:z+Z]
        # blk_cube = container[ np.where(container == chosen_blk_idx+1) ]

        # choose an axis to split
        # split_axis = np.random.randint(0, 3)                # choose a random axis
        # split_axis = np.argmax(blocks[chosen_blk_idx])      # choose the longest axis to split
        prob = normalize([X, Y, Z])
        split_axis = np.random.choice([0, 1, 2], p=prob)    # choose an axis randomly by length

        # choose the split position on the chosen axis
        axis_max = blocks[chosen_blk_idx, split_axis]
        min_ = min_size
        max_ = np.min( [axis_max, max_size] )

        if min_ == max_:
            split_pos = min_
        else:
            split_pos = np.random.randint(min_, max_)
        
        # split the block
        splits = np.split(blk_cube, [split_pos], axis=split_axis)
        blocks[chosen_blk_idx] = np.array(splits[0].shape)
        blocks[block_index] = np.array(splits[1].shape)
        positions[block_index] = positions[chosen_blk_idx]
        positions[block_index][split_axis] += split_pos

        # update the volumes and the container
        volumes[chosen_blk_idx] = np.size(splits[0])
        volumes[block_index] = np.size(splits[1])
        x, y, z = positions[block_index]
        X, Y, Z = blocks[block_index]
        container[x:x+X, y:y+Y, z:z+Z] = block_index + 1

    return blocks, positions, container

def BPP_Generator_2D( blocks_num, gt_packing_size, size_range ):
    '''
    Return blocks, positions, container
    '''
    # np.random.seed(
    #     np.random.randint(12345679)
    # )
    # gt_packing_size[-1] = blocks_num

    min_size = size_range[0]
    max_size = size_range[1]
    
    block_dim = len(gt_packing_size)
    container = np.ones(gt_packing_size).astype(int)
    blocks    = np.zeros((blocks_num, block_dim)).astype(int)
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    volumes   = np.zeros(blocks_num).astype(int)
    
    blocks[0] = gt_packing_size
    volumes[0] = np.sum(container)

    def normalize(l):
        return ( np.array(l) / np.sum(l) ).tolist()

    for block_index in range(1, blocks_num):
        # Step 1: choose a block to split
        if block_index == 1:
            chosen_blk_idx = 0
        else:
            # chosen_blk_idx = np.argmax(volumes) # choose the biggest block
            # choose a block randomly by volumes
            ids = []
            for id in range(block_index):
                # remove those too small to split
                X, Z = blocks[id]
                if X>=2*min_size or Z>=2*min_size: ids.append(id)
            ids = np.array(ids)
            prob = volumes[ids]
            prob = normalize(prob)
            chosen_blk_idx = np.random.choice(ids, p=prob)
        x, z = positions[chosen_blk_idx]
        X, Z = blocks[chosen_blk_idx]
        blk_cube = container[x:x+X, z:z+Z]
        
        # Step 2: choose an axis to split
        # split_axis = np.random.randint(0, 3)            # choose a random axis
        # split_axis = np.argmax(blocks[biggest_blk_idx]) # choose the longest axis to split
        prob = normalize([X, Z])
        split_axis = np.random.choice([0, 1], p=prob)   # choose an axis randomly by length
        # estimate whether the axis can be splited or not
        axis_length = blocks[chosen_blk_idx, split_axis]
        if axis_length < 2 * min_size: 
            if split_axis==0: split_axis=1
            elif split_axis==1: split_axis=0
        axis_length = blocks[chosen_blk_idx, split_axis]

        # Step 3: choose the split position randomly on the chosen axis
        # split by uniform distribution
        if axis_length < 2 * max_size -1:
            # split by uniform distribution
            # max_ = np.min( [axis_length, max_size] ) - min_size + 1
            max_ = axis_length - min_size + 1
            split_pos = np.random.randint(min_size, max_)
        else:
            # if it's too large, split it by Gaussian distribution
            mu = 0.5
            sigma = 0.16
            prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, axis_length - 2*min_size)
            prob = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
            prob = prob / np.sum(prob)
            choices = [i for i in range(min_size, axis_length-min_size)]
            split_pos = np.random.choice(choices, p=prob)

        # Step 4 :split the block
        splits = np.split(blk_cube, [split_pos], axis=split_axis)
        blocks[chosen_blk_idx] = np.array(splits[0].shape)
        blocks[block_index] = np.array(splits[1].shape)
        positions[block_index] = positions[chosen_blk_idx]
        positions[block_index][split_axis] += split_pos

        # Step 5: update the volumes and the container
        volumes[chosen_blk_idx] = np.size(splits[0])
        volumes[block_index] = np.size(splits[1])
        x, z = positions[block_index]
        X, Z = blocks[block_index]
        container[x:x+X, z:z+Z] = block_index + 1

    return blocks, positions, container

def BPP_Generator_2D_easy( blocks_num, gt_packing_size, size_range ):
    '''
    Return blocks, positions, container
    '''
    # np.random.seed(
    #     np.random.randint(12345679)
    # )
    # gt_packing_size[-1] = blocks_num

    min_size = size_range[0]
    max_size = size_range[1]
    
    block_dim = len(gt_packing_size)
    container = np.ones(gt_packing_size).astype(int)
    blocks    = np.zeros((blocks_num, block_dim)).astype(int)
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    volumes   = np.zeros(blocks_num).astype(int)
    
    blocks[0] = gt_packing_size
    volumes[0] = np.sum(container)

    def normalize(l):
        return ( np.array(l) / np.sum(l) ).tolist()

    for block_index in range(1, blocks_num):
        # Step 1: choose a block to split
        if block_index == 1:
            chosen_blk_idx = 0
        else:
            # chosen_blk_idx = np.argmax(volumes) # choose the biggest block
            # choose a block randomly by volumes
            ids = []
            for id in range(block_index):
                # first split those larger than max_size
                X, Z = blocks[id]
                if X>=max_size or Z>=max_size: ids.append(id)
            if ids==[]:
                for id in range(block_index):
                    # remove those cannot be splited
                    X, Z = blocks[id]
                    if X>=2*min_size or Z>=2*min_size: ids.append(id)
            ids = np.array(ids)
            prob = volumes[ids]
            prob = normalize(prob)
            chosen_blk_idx = np.random.choice(ids, p=prob)
        x, z = positions[chosen_blk_idx]
        X, Z = blocks[chosen_blk_idx]
        blk_cube = container[x:x+X, z:z+Z]
        
        # Step 2: choose an axis to split
        # split_axis = np.random.randint(0, 3)            # choose a random axis
        # split_axis = np.argmax(blocks[biggest_blk_idx]) # choose the longest axis to split
        # first split those larger than max_size
        if X>=2*max_size and X>Z: split_axis = 0
        elif Z>=2*max_size and Z>x: split_axis = 1
        else:
            prob = normalize([X, Z])
            split_axis = np.random.choice([0, 1], p=prob)   # choose an axis randomly by length
            axis_length = blocks[chosen_blk_idx, split_axis]
            if axis_length < 2 * min_size: 
                if split_axis==0: split_axis=1
                elif split_axis==1: split_axis=0
        axis_length = blocks[chosen_blk_idx, split_axis]

        # Step 3: choose the split position randomly on the chosen axis
        if axis_length < 2 * max_size -1:
            # split by uniform distribution
            max_ = np.min( [axis_length, max_size] ) - min_size + 1
            split_pos = np.random.randint(min_size, max_)
        else:
            # if it's too large, split it by Gaussian distribution
            mu = 0.5
            sigma = 0.16
            prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, axis_length - 2*min_size)
            prob = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
            prob = prob / np.sum(prob)
            choices = [i for i in range(min_size, axis_length-min_size)]
            split_pos = np.random.choice(choices, p=prob)

        # Step 4 :split the block
        splits = np.split(blk_cube, [split_pos], axis=split_axis)
        blocks[chosen_blk_idx] = np.array(splits[0].shape)
        blocks[block_index] = np.array(splits[1].shape)
        positions[block_index] = positions[chosen_blk_idx]
        positions[block_index][split_axis] += split_pos

        # Step 5: update the volumes and the container
        volumes[chosen_blk_idx] = np.size(splits[0])
        volumes[block_index] = np.size(splits[1])
        x, z = positions[block_index]
        X, Z = blocks[block_index]
        container[x:x+X, z:z+Z] = block_index + 1

    return blocks, positions, container


def calc_dependent_2D_old(blocks, positions, container):
    '''
    return
    ----
        deps_move:      movement dep
        deps_left:      rotation dep along x axis, left
        deps_right:     rotation dep along x axis, right
        deps_forward:   rotation dep along y axis, forward
        deps_backward:  rotation dep along y axis, backward
        deps_up:        rotation dep along z axis, up
        deps_down:      rotation dep along z axis, down
    '''

    blocks_num = len(blocks)
    container_x, container_z = container.shape
    # calculate dependences by searching surrounded area
    deps = np.zeros((blocks_num, blocks_num))
    deps_left = np.zeros((blocks_num, blocks_num))
    deps_right = np.zeros((blocks_num, blocks_num))
    deps_forward = np.zeros((blocks_num, blocks_num))
    deps_backward = np.zeros((blocks_num, blocks_num))

    for block_index, block in enumerate(blocks):
        [block_x, block_z] = block
        [x, z] = positions[block_index]
        x = int(x)
        z = int(z)
        # watch below
        support = container[x:x+block_x, 0:z]
        support_obj = []
        # for _x, _y in zip(range(block_x), range(block_y)):
        for _x in range(block_x):
            for _z in reversed(range(z)):
                obj = support[_x][_z]
                if obj <= 0:
                    continue
                elif obj in support_obj:
                    break
                else:
                    support_obj.append(obj)
                    deps[ block_index, int(obj)-1 ] = 1
                    break
        # above
        top = container[x:x+block_x, z+block_z:]
        top_obj = []
        for _x in range(block_x):
            for _z in range(container_z - z - block_z):
                obj = top[_x][_z]
                if obj <= 0:
                    continue
                elif obj in top_obj:
                    break
                else:
                    top_obj.append(obj)
                    deps[ int(obj)-1, block_index ] = 1
                    break

        z_mid = z + int((block_z-1)/2)
        
        x_mid_1 = x + int((block_x-1)/2)
        x_mid_2 = x + int(block_x/2)
        if x_mid_1 == x_mid_2:
            x_mid_2 += 1

        # left (rotate to x as top)
        if (x == 0):
            # next to the container wall
            deps_left[ block_index, block_index ] = 1
        else:
            left = container[x-1, z_mid:container_z]
            for obj in left[left>0]:
                deps_left[ int(obj)-1, block_index ] = 1

        # right
        if (x+block_x == container_x):
            deps_right[ block_index, block_index] = 1
        else:
            right = container[x+block_x, z_mid:container_z]
            for obj in right[right>0]:
                deps_right[ int(obj)-1, block_index] = 1

    deps_up = np.zeros_like(deps_left)
    deps_down = np.zeros_like(deps_left)
    deps_move = deps

    return deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down

def calc_dependent_2D(blocks, positions, container, arm_size):
    '''
    return
    ----
        deps_move:      movement dep
        deps_left:      rotation dep along x axis, left
        deps_right:     rotation dep along x axis, right
        deps_forward:   rotation dep along y axis, forward
        deps_backward:  rotation dep along y axis, backward
        deps_up:        rotation dep along z axis, up
        deps_down:      rotation dep along z axis, down
    '''

    blocks_num = len(blocks)
    container_x, container_z = container.shape
    # calculate dependences by searching surrounded area
    deps = np.zeros((blocks_num, blocks_num))
    deps_left = np.zeros((blocks_num, blocks_num))
    deps_right = np.zeros((blocks_num, blocks_num))
    deps_forward = np.zeros((blocks_num, blocks_num))
    deps_backward = np.zeros((blocks_num, blocks_num))

    for block_index, block in enumerate(blocks):
        [block_x, block_z] = block
        [x, z] = positions[block_index]
        x = int(x)
        z = int(z)
        # watch below
        support = container[x:x+block_x, 0:z]
        support_objs = np.unique(support)
        for obj in support_objs:
            if obj <= 0: continue
            deps[ block_index, int(obj)-1 ] = 1
        # above
        top = container[x:x+block_x, z+block_z:]
        top_objs = np.unique(top)
        for obj in top_objs:
            if obj <= 0: continue
            deps[ int(obj)-1, block_index ] = 1

        z_mid = z + int((block_z-1)/2)
        
        x_mid_1 = x + int((block_x-1)/2)
        x_mid_2 = x + int(block_x/2)
        if x_mid_1 == x_mid_2:
            x_mid_2 += 1

        # left (rotate to x as top)
        if (x < arm_size):
            # next to the container wall
            deps_left[ block_index, block_index ] = 1
        else:
            left = container[x-arm_size:x, z_mid:container_z]
            left_objs = np.unique(left)
            for obj in left_objs:
                if obj <= 0: continue
                deps_left[ int(obj)-1, block_index ] = 1

        # right
        if (x+block_x > container_x-arm_size ):
            deps_right[ block_index, block_index] = 1
        else:
            right = container[x+block_x:x+block_x+arm_size, z_mid:container_z]
            right_objs = np.unique(right)
            for obj in right_objs:
                if obj <= 0: continue
                deps_right[ int(obj)-1, block_index] = 1

    deps_up = np.zeros_like(deps_left)
    deps_down = np.zeros_like(deps_left)
    deps_move = deps

    return deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down

def calc_dependent_3D(blocks, positions, container):
    '''
    return
    ----
        deps_move:      movement dep
        deps_left:      rotation dep along x axis, left
        deps_right:     rotation dep along x axis, right
        deps_forward:   rotation dep along y axis, forward
        deps_backward:  rotation dep along y axis, backward
        deps_up:        rotation dep along z axis, up
        deps_down:      rotation dep along z axis, down
    '''

    blocks_num = len(blocks)
    container_x, container_y, container_z = container.shape
    # calculate dependences by searching surrounded area
    deps = np.zeros((blocks_num, blocks_num))
    deps_left = np.zeros((blocks_num, blocks_num))
    deps_right = np.zeros((blocks_num, blocks_num))
    deps_forward = np.zeros((blocks_num, blocks_num))
    deps_backward = np.zeros((blocks_num, blocks_num))
    for block_index, block in enumerate(blocks):
        [block_x, block_y, block_z] = block
        [x, y, z] = positions[block_index]
        # watch below
        support = container[x:x+block_x, y:y+block_y, 0:z]
        support_obj = []
        # for _x, _y in zip(range(block_x), range(block_y)):
        for _x, _y in itertools.product(range(block_x), range(block_y)):
            for _z in reversed(range(z)):
                obj = support[_x][_y][_z]
                if obj <= 0:
                    continue
                elif obj in support_obj:
                    break
                else:
                    support_obj.append(obj)
                    deps[ block_index, int(obj)-1 ] = 1
                    break
        # above
        top = container[x:x+block_x, y:y+block_y, z+block_z:]
        top_obj = []
        for _x, _y in itertools.product(range(block_x), range(block_y)):
            for _z in range(container_z - z - block_z):
                obj = top[_x][_y][_z]
                if obj <= 0:
                    continue
                elif obj in top_obj:
                    break
                else:
                    top_obj.append(obj)
                    deps[ int(obj)-1, block_index ] = 1
                    break
        

        z_mid = z + int((block_z-1)/2)
        y_mid_1 = y + int((block_y-1)/2)
        y_mid_2 = y + int(block_y/2)
        if y_mid_1 == y_mid_2:
            y_mid_2 += 1
        
        x_mid_1 = x + int((block_x-1)/2)
        x_mid_2 = x + int(block_x/2)
        if x_mid_1 == x_mid_2:
            x_mid_2 += 1

        # left (rotate to x as top)
        if (x == 0):
            # next to the container wall
            deps_left[ block_index, block_index ] = 1
        else:
            left = container[x-1, y_mid_1:y_mid_2, z_mid:container_z]
            for obj in left[left>0]:
                deps_left[ int(obj)-1, block_index ] = 1

        # right
        if (x+block_x == container_x):
            deps_right[ block_index, block_index] = 1
        else:
            right = container[x+block_x, y_mid_1:y_mid_2, z_mid:container_z]
            for obj in right[right>0]:
                deps_right[ int(obj)-1, block_index] = 1

        # forward (rotate to y as top)
        if (y == 0):
            deps_forward[ block_index, block_index ] = 1
        else:
            forward = container[x_mid_1:x_mid_2, y-1, z_mid:container_z]
            for obj in forward[forward>0]:
                deps_forward[ int(obj)-1, block_index ] = 1

        # backward
        if (y+block_y == container_y):
            deps_backward[ block_index, block_index ] = 1
        else:
            backward = container[x_mid_1:x_mid_2, y+block_y, z_mid:container_z]
            for obj in backward[backward>0]:
                deps_backward[ int(obj)-1, block_index ] = 1
    
    deps_up = np.zeros_like(deps_left)
    deps_down = np.zeros_like(deps_left)
    deps_move = deps

    return deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down

def calc_dependent(blocks, positions, container, arm_size=1):
    '''
    return
    ----
        deps_move:      movement dep
        deps_left:      rotation dep along x axis, left
        deps_right:     rotation dep along x axis, right
        deps_forward:   rotation dep along y axis, forward
        deps_backward:  rotation dep along y axis, backward
        deps_up:        rotation dep along z axis, up
        deps_down:      rotation dep along z axis, down
    '''

    block_dim = len(container.shape)
    if block_dim == 3:
        return calc_dependent_3D(blocks, positions, container)
    else:
        return calc_dependent_2D(blocks, positions, container, arm_size)

def generate_blocks(blocks_num, container_size, arm_size, size_range, random_distribution=None, random_num=False):
    '''
    Generate the blocks data in a container
    ----
    params:
        blocks_num: int, the number of blocks want to generate
        container_size: 1 x 2/3 int array / list, the size of container, block_dim = len(container_size)
    return:
    ----
        rotate_blocks: (n x dim!) x dim int array, store all posible rotation state of these blocks 
        positions: n x dim int array, the positions of each block in initial container
        deps_move: n x n int array, movement dependence of blocks in initial container
        rotate_deps_small: n x n int array, rotation dependence of blocks in initial container on the small size (such as left size)
        rotate_deps_large: n x n int array, rotation dependence of blocks in initial container on the large size (such as right size)
    '''    
    # np.random.seed(
    #     np.random.randint(12345679)
    # )
    block_dim = len(container_size)

    min_box = size_range[0]
    max_box = size_range[1]
    size_list = [ i for i in range(min_box, max_box) ]
    
    container_width = container_size[0]

    def random_dependence(blocks_num, random_distribution):
        nodes = blocks_num
        edges_num = np.random.randint( nodes )

        move_dist, left_dist, right_dist = random_distribution

        # build the precedence graph
        G_move = nx.DiGraph()
        for block_index in range(blocks_num):
            G_move.add_node(block_index)

        # rot
        G_left = G_move.copy()
        G_right = G_move.copy()
        G_forw = G_move.copy()
        G_back = G_move.copy()
        
        # build move
        if random_num:
            edges_num = np.random.randint( 1, np.max(move_dist[1]) + 1 )
        else:
            edges_num = np.random.choice( move_dist[1], p=move_dist[0] )

        while edges_num > 0:
            a = np.random.randint(nodes)
            b=a
            while b==a:
                b = np.random.randint(nodes)
            G_move.add_edge(a,b)
            if nx.is_directed_acyclic_graph(G_move):
                edges_num -= 1
            else:
                G_move.remove_edge(a,b)

        def random_rotation_deps( G, distribution ):
            g = copy.deepcopy(G_move)
            # edges_num = np.random.randint( nodes )
            if random_num:
                edges_num = np.random.randint( 1, np.max(distribution[1]) + 1 )
            else:
                edges_num = np.random.choice( distribution[1], p=distribution[0] )
            while edges_num > 0:
                a = np.random.randint(nodes)
                while True:
                    b = np.random.randint(nodes)
                    if not G_move.has_edge(b,a):
                        break
                G.add_edge(a,b)
                g.add_edge(a,b)
                if nx.is_directed_acyclic_graph(G) and \
                    nx.is_directed_acyclic_graph(g):
                    edges_num -= 1
                else:
                    G.remove_edge(a,b)
                    g.remove_edge(a,b)
            return G

        def graph_to_deps(G):
            deps = np.zeros((blocks_num, blocks_num)).astype(int)
            for e in G.edges:
                deps[ e[0], e[1] ] = 1
            return deps


        G_left = random_rotation_deps(G_left, left_dist )
        G_right = random_rotation_deps(G_right, right_dist )

        # TODO 3d
        G_forw = random_rotation_deps(G_forw, left_dist )
        G_back = random_rotation_deps(G_back, right_dist )

        move = graph_to_deps(G_move)
        left = graph_to_deps(G_left)
        right =graph_to_deps(G_right)
        forw = graph_to_deps(G_forw)
        back = graph_to_deps(G_back)
        up = np.zeros_like(back)
        down = np.zeros_like(back)

        return move, left, right, forw, back, up, down

    # Gaussian distribution
    if len(size_list) == 4:
        prob_blocks = [0.15, 0.35, 0.35, 0.15]
    elif len(size_list) == 5:
        prob_blocks = [0.08, 0.26, 0.32, 0.26, 0.08]
    else:
        mu = 0.5
        sigma = 0.16
        prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(size_list))
        prob_blocks = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
        prob_blocks = prob_blocks / np.sum(prob_blocks)

    
    if container_width >= 0:
        # a = time.time()
        while True:
            blocks = np.random.choice(size_list, (blocks_num, block_dim), p=prob_blocks )
            # blocks = np.random.randint(min_box, max_box, (blocks_num, block_dim))

            if container_width == 0:
                container_width = np.random.randint(5, 11)
                if block_dim == 2:
                    container_size = [ container_width, blocks_num * max_box + 10 ]
                elif block_dim == 3:
                    container_size = [ container_width, container_width, blocks_num * max_box + 10 ]

            # put every blocks into the intial container
            # positions, container, place = tools.calc_positions(blocks, container_size, True)
            positions, container, stable, _, _ = tools.calc_positions_lb_greedy(blocks, container_size, 'C+P+S-lb-hard')
            if np.sum(stable) == blocks_num:
                break
        # print('t1: ', time.time() - a)
        # a = time.time()
        deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down = \
                                    calc_dependent(blocks, positions, container, arm_size)
        # print('t2: ', time.time() - a)
    else:
        blocks = np.random.choice(size_list, (blocks_num, block_dim), p=prob_blocks )
        deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down = random_dependence(blocks_num, random_distribution)
        
        if block_dim == 2:
            container_size = [ 7, blocks_num * max_box + 10 ]
        elif block_dim == 3:
            container_size = [ 7, 7, blocks_num * max_box + 10 ]
            
        positions, container, place = tools.calc_positions(blocks, container_size, True)
    

    rotate_blocks = []
    blocks = blocks.transpose()
    deps_left = deps_left.flatten()
    deps_right = deps_right.flatten()
    deps_forward = deps_forward.flatten()
    deps_backward = deps_backward.flatten()
    deps_up = deps_up.flatten()
    deps_down = deps_down.flatten()

    rotate_deps_small = []
    rotate_deps_large = []

    if block_dim == 3:
        for p in itertools.permutations( range(block_dim) ):
            if p[-1] == 0:
                rotate_deps_small.append(deps_left)
                rotate_deps_large.append(deps_right)
            elif p[-1] == 1:
                rotate_deps_small.append(deps_forward)
                rotate_deps_large.append(deps_backward)
            elif p[-1] == 2:
                rotate_deps_small.append(deps_down)
                rotate_deps_large.append(deps_up)
            rotate_blocks.append( blocks[list(p)].flatten() )
    else:
        for p in itertools.permutations( range(block_dim) ):
            if p[-1] == 0:
                rotate_deps_small.append(deps_left)
                rotate_deps_large.append(deps_right)
            elif p[-1] == 1:
                rotate_deps_small.append(deps_down)
                rotate_deps_large.append(deps_up)
            rotate_blocks.append( blocks[list(p)].flatten() )


    rotate_blocks = np.array(rotate_blocks)
    rotate_deps_small = np.array(rotate_deps_small)
    rotate_deps_large = np.array(rotate_deps_large)

    positions = np.array(positions)
    positions = positions.transpose().flatten()
    deps_move = deps_move.transpose().flatten()

    return rotate_blocks, positions, deps_move, rotate_deps_small, rotate_deps_large


# ===================
# NOTE prob

def generate_height_prob(block_dim, blocks_num, size_range, initial_container_width, target_container_width):
    '''
    statistics of the size the random data, used to generate the CPPS data(gt_data)
    use the rand_data of 10 blocks (in .../pack-valid-10-10000-...)
    '''
    
    print('gt blocks num %d' % blocks_num)
    [min_size, max_size] = size_range
    if block_dim == 2:
        data_file = './data/rand_2d/pack-valid-10-10000-%d-%d-%d/' % (initial_container_width, min_size, max_size)
        bottom_size = target_container_width
    elif block_dim == 3:
        data_file = './data/rand_3d/pack-valid-10-10000-%d-%d-%d/' % (initial_container_width, min_size, max_size)
        bottom_size = target_container_width * target_container_width

    total_blocks = np.loadtxt(data_file + 'blocks.txt').astype('int')
    rotate_types = np.math.factorial(block_dim)
    data_size = int( len(total_blocks) / rotate_types )
    total_blocks = total_blocks.reshape( data_size, -1, block_dim, 10)
    total_blocks = total_blocks.transpose(0, 1, 3, 2)
    total_blocks = total_blocks.reshape( data_size, -1, block_dim )[:,:10,:]

    def calc_block_size(block):
        size = 1
        for width in block:
            size *= width
        return size

    def calc_blocks_size_sum(blocks):
        size = 0
        for block in blocks:
            size += calc_block_size(block)
        return size

    def calc_height_distribution(total_blocks):
        sizes = {}
        for blocks in total_blocks:
            size = int(calc_blocks_size_sum(blocks) / bottom_size * (blocks_num/10) )
            # size = int(calc_blocks_size_sum(blocks) / bottom_size ) 
            if size not in sizes:
                sizes[size] = 1
            else:
                sizes[size] += 1

        return sizes

    sizes = calc_height_distribution(total_blocks)
    print('statistic height: ', sizes)
    # if size_range == [10, 41]:
    #     for size in list(sizes.keys()):
    #         if size >= 227: 
    #             print('delete height: ', size, ': ', sizes[size])
    #             del sizes[size]
    if size_range == [100, 401]:
        for size in list(sizes.keys()):
            if size >= 1400 or size<= 1000: 
                print('delete height: ', size, ': ', sizes[size])
                del sizes[size]
    sizes_value = np.array(list( sizes.values() ))
    sizes_keys = np.array(list( sizes.keys() ))

    def normalization(x):
        return x / np.sum(x)

    return normalization(sizes_value), sizes_keys

def generate_deps_prob(block_dim, blocks_num, initial_container_width):
    if block_dim == 2:
        data_file = './data/rand_2d/pack-valid-10-10000-%d-1-5/' % initial_container_width
    elif block_dim == 3:
        data_file = './data/rand_3d/pack-valid-10-10000-%d-1-5/' % initial_container_width

    deps_move = np.loadtxt(data_file + 'dep_move.txt').astype('int')
    rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('int')
    rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('int')

    rotate_types = np.math.factorial(block_dim)
    data_size = len(deps_move)
    
    deps_move = deps_move.reshape( len(deps_move), blocks_num, -1 )
    deps_move = deps_move.transpose(0,2,1)

    num_samples = data_size

    rotate_deps_small = rotate_deps_small.reshape( num_samples, -1, blocks_num, blocks_num )
    rotate_deps_large = rotate_deps_large.reshape( num_samples, -1, blocks_num, blocks_num )

    rotate_deps_small = rotate_deps_small.transpose(0,1,3,2)
    rotate_deps_large = rotate_deps_large.transpose(0,1,3,2)

    rotate_deps_small = rotate_deps_small.reshape( num_samples, blocks_num*rotate_types, blocks_num )
    rotate_deps_large = rotate_deps_large.reshape( num_samples, blocks_num*rotate_types, blocks_num )

    rotate_deps_small = rotate_deps_small.transpose(0,2,1)
    rotate_deps_large = rotate_deps_large.transpose(0,2,1)


    def calc_deps_num(total_deps):
        deps_num = {}
        for deps in total_deps:
            num = np.sum(deps)
            if num not in deps_num:
                deps_num[num] = 1
            else:
                deps_num[num] += 1
        return deps_num

    def normalization(x):
        return x / np.sum(x)

    def calc_deps_distribution(deps_type):

        if deps_type == 'move':
            deps_num = calc_deps_num(deps_move)
        elif deps_type == 'left':
            deps_num = calc_deps_num( rotate_deps_small[:,:, blocks_num:blocks_num*2] )
        elif deps_type == 'right':
            deps_num = calc_deps_num( rotate_deps_large[:,:, blocks_num:blocks_num*2] )

        value = np.array(list( deps_num.values() ))
        keys = np.array(list( deps_num.keys() ))
        return normalization(value), keys

    return calc_deps_distribution('move'), calc_deps_distribution('left'), calc_deps_distribution('right')





# ====================
# NOTE

def generate_order_graph(blocks, positions, container_size, arm_size, allow_bot, find_order_type, reward_type, target_container_size=None):
    '''
    Generate the picking order from a container by graph
    ----
    params:
    ----
        blocks: n x 2/3 int array, the original blocks
        positions: n x 2/3 int array/list, the position of blocks in a container
        container_size: 1 x 2/3 int array / list, the size of container, block_dim = len(container_size)
        allow_bot: bool, True if you condiser the left and right
        # max_first: bool, True if you want to select the the max out_degree node in the precedence graph
        find_order_type: str, 'rand' 'max' 'best'
    return:
    ----
        solution: 1 x n int list, store the selected order(include rotation) of the blocks
            so you may need to use tools.calc_real_order to get the original index of blocks
    '''
    np.random.seed(
        np.random.randint(8888)
    )
    blocks_num = len(blocks)
    block_dim = len(container_size)

    rotates = []
    for p in itertools.permutations( range(block_dim) ):
        rotates.append(list(p))
    rotates = np.array(rotates)
    

    container = np.zeros(container_size)
    if block_dim == 2:
        for i in range(len(blocks)):
            px, pz = positions[i]
            bx, bz = blocks[i]
            container[px:px+bx, pz:pz+bz] = i+1
    
    elif block_dim == 3:
        for i in range(len(blocks)):
            px, py, pz = positions[i]
            bx, by, bz = blocks[i]
            container[px:px+bx, py:py+by, pz:pz+bz] = i+1

    deps_move, deps_left, deps_right, deps_forward, deps_backward, deps_up, deps_down = \
        calc_dependent( blocks, positions, container, arm_size )
    
    deps_move = deps_move.astype('int')
    deps_left = deps_left.astype('int')
    deps_right = deps_right.astype('int')
    deps_forward = deps_forward.astype('int')
    deps_backward = deps_backward.astype('int')
    deps_up = deps_up.astype('int')
    deps_down = deps_down.astype('int')

    # build the precedence graph
    G_move = nx.DiGraph()
    for rot_index, rot in enumerate(rotates):
        for block_index, block in enumerate(blocks):
            G_move.add_node(rot_index * blocks_num + block_index)

    G_left = G_move.copy()
    G_right = G_move.copy()
    G_all = G_move.copy()



    # dep_move first
    deps = deps_move
    for i in range(blocks_num):
        for j in range(blocks_num):
            if deps[i,j] == True:
                for r_i in range( len(rotates) ):
                    for r_j in range( len(rotates) ):
                        G_move.add_edge( r_i*blocks_num+i, r_j*blocks_num+j)
                        G_all.add_edge( r_i*blocks_num+i, r_j*blocks_num+j)

    # then deps rotate 
    for rot_index, rot in enumerate(rotates):
        if allow_bot == False:
            break
        if block_dim == 2:
            if rot[-1] == 0:
                deps_a = deps_left
                deps_b = deps_right
            elif rot[-1] == 1:
                continue
                # deps_a = deps_up
                # deps_b = deps_up
        else:
            if rot[-1] == 0:
                deps_a = deps_left
                deps_b = deps_right
            elif rot[-1] == 1:
                deps_a = deps_forward
                deps_b = deps_backward
            elif rot[-1] == 2:
                continue
                # deps_a = deps_up
                # deps_b = deps_up

        for i in range(blocks_num):
            for j in range(blocks_num):
                if deps_a[i,j] == True:
                    if i == j:
                        G_left.add_edge( rot_index*blocks_num+i, rot_index*blocks_num+j)
                        G_all.add_edge( rot_index*blocks_num+i, rot_index*blocks_num+j)
                    else:
                        for r_i in range( len(rotates) ):
                            G_left.add_edge( r_i*blocks_num+i, rot_index*blocks_num+j)
                            G_all.add_edge( r_i*blocks_num+i, rot_index*blocks_num+j)
                            # G_left.add_edge( rot_index*blocks_num+i, r_j*blocks_num+j)
                            # G_all.add_edge( rot_index*blocks_num+i, r_j*blocks_num+j)

                if deps_b[i,j] == True:
                    if i == j:
                        G_right.add_edge( rot_index*blocks_num+i, rot_index*blocks_num+j)
                        G_all.add_edge( rot_index*blocks_num+i, rot_index*blocks_num+j)
                    else:
                        for r_i in range( len(rotates) ):
                            G_right.add_edge( r_i*blocks_num+i, rot_index*blocks_num+j)
                            G_all.add_edge( r_i*blocks_num+i, rot_index*blocks_num+j)
                        # for r_j in range( len(rotates) ):
                        #     G_right.add_edge( rot_index*blocks_num+i, r_j*blocks_num+j)
                        #     G_all.add_edge( rot_index*blocks_num+i, r_j*blocks_num+j)

    # import IPython
    # IPython.embed()

    solution = []
    # remove nodes from graph

    if target_container_size is not None:
        test_container = tools.Container(target_container_size, blocks_num, reward_type, 'full', container_size)

    def best_to_pack(valid_nodes, blocks_num):
        test_list = []

        test_list = valid_nodes

        # # only concen
        # if find_order_type == 'max':
        #     test_list = valid_nodes
        # elif find_order_type == 'best':
        #     test_list = valid_nodes
        # else:
        #     lucky_index = valid_nodes[np.random.choice(len(valid_nodes))]
        #     # test here
        #     # select all the states of item          
        #     origin_index = lucky_index % blocks_num
        #     for i in range(len(rotates)):
        #         test_index = origin_index + i*blocks_num
        #         if test_index in valid_nodes:
        #             test_list.append(test_index)
        
        max_out_index = test_list[0]
        max_ratio = 0
        tmp = []
        for test_index in test_list:
            origin_index = test_index % blocks_num
            rotate_type = int(test_index / blocks_num)
            if test_index >= blocks_num:
                test_block = blocks[origin_index][ rotates[rotate_type] ]
            else:
                test_block = blocks[origin_index]
            
            test_container_copy = copy.deepcopy(test_container)
            test_container_copy.add_new_block(test_block)
            ratio = test_container_copy.calc_ratio()
            tmp.append(ratio)
            if max_ratio < ratio:
                max_ratio = ratio
                max_out_index = test_index

        # print(test_list, max_out_index, tmp)

        origin_index = max_out_index % blocks_num
        if max_out_index >= blocks_num:
            rotate_type = int(max_out_index / blocks_num)
            block = blocks[origin_index][rotates[rotate_type]]
        else:
            block = blocks[origin_index]

        test_container.add_new_block(block)

        return max_out_index
    
    search_start = time.time()
    valid_nodes_num_list = []

    while G_move.number_of_nodes() > 0:        
        valid_nodes = []

        # get nodes with in_degree == 0 for dep_move
        nodes = np.array(G_move.in_degree)
        mask = np.where(nodes[:,1] == 0)[0]
        # get the out_degree of these nodes
        nodes = np.array(G_move.out_degree)[mask]

        nodes_ids = nodes[:,0]
        # check feasibility
        # import IPython
        # IPython.embed()
        
        for node_id in nodes_ids:
            if G_left.in_degree(node_id) == 0 or G_right.in_degree(node_id) == 0:
                valid_nodes.append(node_id)
                try:
                    G_all.remove_edge(node_id, node_id)
                except nx.exception.NetworkXError as e:
                    pass
        
        # if len(valid_nodes) == 1:
        #     nodes = [0,0]
        # else:

        valid_nodes_num_list.append(len(valid_nodes))
        
        # if max_first == True:
        if find_order_type == 'max':
            # if len(valid_nodes) == 1:
            #     max_out_index = valid_nodes[0]
            # else:
            nodes = np.array(G_all.out_degree(valid_nodes))
            # select the node with max out_degree
            max_out = np.max(nodes[:,1])
            max_out_ids = np.array(valid_nodes)[nodes[:,1] == max_out]
            
            # max_out_index = luky_to_pack(max_out_ids, blocks_num)
            max_out_index = max_out_ids[np.random.choice(len(max_out_ids))]
        elif find_order_type == 'best':
            max_out_index = best_to_pack(valid_nodes, blocks_num)
        else:
            # max_out_index = luky_to_pack(valid_nodes, blocks_num)
            max_out_index = valid_nodes[np.random.choice(len(valid_nodes))]
            
            # print(test_container.calc_ratio())

        solution.append(max_out_index)

        # get the basic id of node, the remove all of the rotate state of the nodes
        node_id = max_out_index % blocks_num
        
        for r in range(len(rotates)):
            G_move.remove_node(node_id + r * blocks_num)
            G_left.remove_node(node_id + r * blocks_num)
            G_right.remove_node(node_id + r * blocks_num)
            G_all.remove_node(node_id + r * blocks_num)
            
        # print('Order: ',  test_container.calc_ratio())

    search_time = time.time() - search_start
    mean_valid_nodes_num = np.mean(valid_nodes_num_list)

    return solution, search_time, mean_valid_nodes_num


# ============================

# NOTE sub graph

def decompose_deps_graphs(Gm, Gl, Gr, Gf, Gb, child_graph_size, traverse='BFS'):
    gm = Gm.copy()
    gl = Gl.copy()
    gr = Gr.copy()
    gf = Gf.copy()
    gb = Gb.copy()
    sub_gm = []
    sub_gl = []
    sub_gr = []
    sub_gf = []
    sub_gb = []
    sub_graphs_nodes_list = []

    after_nodes_list = list(Gm.nodes)
    
    def decompose(sg_index):
        # get the ndoes atfer current sub_graphs 

        nodes = sub_graphs_nodes_list[sg_index]
        pre_nodes = nodes
        for i in range(sg_index):
            pre_nodes = pre_nodes + sub_graphs_nodes_list[i]
        subgm = Gm.subgraph(nodes).copy()
        subgl = Gl.subgraph(nodes).copy()
        subgr = Gr.subgraph(nodes).copy()
        subgf = Gf.subgraph(nodes).copy()
        subgb = Gb.subgraph(nodes).copy()
        # check rotations
        for node in subgl.nodes():
            for pre_node in Gl.predecessors(node):
                # if not pre_node in pre_nodes:
                if pre_node in after_nodes_list:
                    subgl.add_edge(node, node)
        for node in subgr.nodes():
            for pre_node in Gr.predecessors(node):
                # if not pre_node in pre_nodes:
                if pre_node in after_nodes_list:
                    subgr.add_edge(node, node)
        for node in subgf.nodes():
            for pre_node in Gf.predecessors(node):
                # if not pre_node in pre_nodes:
                if pre_node in after_nodes_list:
                    subgf.add_edge(node, node)
        for node in subgb.nodes():
            for pre_node in Gb.predecessors(node):
                # if not pre_node in pre_nodes:
                if pre_node in after_nodes_list:
                    subgb.add_edge(node, node)
        # add the sub-graphs
        sub_gm.append(subgm)
        sub_gl.append(subgl)
        sub_gr.append(subgr)
        sub_gf.append(subgf)
        sub_gb.append(subgb)
        # remove from the graphs
        for node in nodes:
            gm.remove_node(node)
            gl.remove_node(node)
            gr.remove_node(node)
            gf.remove_node(node)
            gb.remove_node(node)

    sub_graph_nodes = []
    gm_copy = gm.copy()
    sg_index = 0
    if traverse == 'BFS':
        while gm_copy.number_of_nodes() > 0:
            # get nodes with in_degree==0 in Gm
            if gm_copy.number_of_nodes() == 1:
                nodes = list(gm_copy.nodes)
            else:
                nodes = np.array(gm_copy.in_degree)
                mask = np.where(nodes[:,1]==0)[0]
                nodes = nodes[mask][:, 0]
            for node in nodes:
                sub_graph_nodes.append(node)    # add to sub_graph
                gm_copy.remove_node(node)       # remove from the graph
                after_nodes_list.remove(node)   # remove from the nodes_list
                # check if the sub graph full
                if len(sub_graph_nodes) == child_graph_size:
                    sub_graphs_nodes_list.append(sub_graph_nodes)
                    decompose(sg_index)
                    sg_index = sg_index + 1
                    sub_graph_nodes = []
    elif traverse == 'DFS':
        stack = []
        while gm_copy.number_of_nodes() > 0:
            # get nodes with in_degree==0 in Gm
            if gm_copy.number_of_nodes() == 1:
                nodes = list(gm_copy.nodes)
            else:
                nodes = np.array(gm_copy.in_degree)
                mask = np.where(nodes[:,1]==0)[0]
                nodes = nodes[mask][:, 0]
            for node in nodes:
                if not node in stack:
                    stack.append(node)
            while len(stack) > 0:
                node = stack.pop()
                sub_graph_nodes.append(node)
                gm_copy.remove_node(node)
                for child in gm.successors(node):
                    if len(list(gm_copy.predecessors(child))) == 0:
                        if not node in stack:
                            stack.append(child)
                if len(sub_graph_nodes) == child_graph_size:
                    sub_graphs_nodes_list.append(sub_graph_nodes)
                    decompose(sg_index)
                    sg_index = sg_index + 1
                    sub_graph_nodes = []

    # the last sub graph, which is not full
    if len(sub_graph_nodes) > 0:
        sub_graphs_nodes_list.append(sub_graph_nodes)
        decompose(sg_index)

    return sub_gm, sub_gl, sub_gr, sub_gf, sub_gb, sub_graphs_nodes_list

def generate_graph_and_decompose(blocks, positions, initial_container_size, allow_bot, child_graph_size, traverse='BFS'):
    """
    Generate precedence graphs and decompose them to several sub-graphs
    ---
    returns:
    --- 
        sub_deps_move
        sub_deps_left
        sub_deps_right
        sub_deps_forward
        sub_deps_backward
        sub_graphs_nodes_list
    """
    blocks_num = len(blocks)
    block_dim = len(initial_container_size)

    container = np.zeros(initial_container_size).astype(int)
    if block_dim == 2:
        for i in range(len(blocks)):
            px, pz = positions[i]
            bx, bz = blocks[i]
            container[px:px+bx, pz:pz+bz] = i+1

    elif block_dim == 3:
        for i in range(len(blocks)):
            px, py, pz = positions[i]
            bx, by, bz = blocks[i]
            container[px:px+bx, py:py+by, pz:pz+bz] = i+1

    deps_move, deps_left, deps_right, deps_forward, deps_backward, _, _ = calc_dependent(blocks, positions, container)

    # build the precedence graphs
    G_move = nx.DiGraph()
    for block_index in range(blocks_num):
        G_move.add_node(block_index)
    G_left = G_move.copy()
    G_right = G_move.copy()
    G_forward = G_move.copy()
    G_backward = G_move.copy()

    # movement deps
    for i in range(blocks_num):
        for j in range(blocks_num):
            if deps_move[i,j] == True:
                G_move.add_edge( i, j )

    # rotation deps
    # 2D
    if allow_bot and block_dim == 2:
        for i in range(blocks_num):
            for j in range(blocks_num):
                if deps_left[i, j] == True:
                    G_left.add_edge( i, j )
                if deps_right[i, j] == True:
                    G_right.add_edge( i, j )
    # 3D
    elif allow_bot and block_dim == 3:
        for i in range(blocks_num):
            for j in range(blocks_num):
                if deps_left[i, j] == True:
                    G_left.add_edge( i, j )
                if deps_right[i, j] == True:
                    G_right.add_edge( i, j )
                if deps_forward[i, j] == True:
                    G_forward.add_edge( i, j )
                if deps_backward[i, j] == True:
                    G_backward.add_edge( i, j )

    # Decompose
    sub_gm, sub_gl, sub_gr, sub_gf, sub_gb, sub_graphs_nodes_list = decompose_deps_graphs(
                    G_move, G_left, G_right, G_forward, G_backward, child_graph_size, traverse)

    # convert to deps matrix
    num_sub_graphs = len(sub_gm)
    sub_deps_move = np.zeros((num_sub_graphs, child_graph_size, child_graph_size)).astype(int)
    sub_deps_left = np.zeros((num_sub_graphs, child_graph_size, child_graph_size)).astype(int)
    sub_deps_right = np.zeros((num_sub_graphs, child_graph_size, child_graph_size)).astype(int)
    sub_deps_forward = np.zeros((num_sub_graphs, child_graph_size, child_graph_size)).astype(int)
    sub_deps_backward = np.zeros((num_sub_graphs, child_graph_size, child_graph_size)).astype(int)

    def G_to_deps(G, deps):
        nodes = list(G.nodes())
        for edge in G.edges():
            deps[  nodes.index(edge[0]), nodes.index(edge[1])] = 1

    for index, sg in enumerate(sub_gm):
        G_to_deps(sg, sub_deps_move[index, :, :])
    for index, sg in enumerate(sub_gl):
        G_to_deps(sg, sub_deps_left[index, :, :])
    for index, sg in enumerate(sub_gr):
        G_to_deps(sg, sub_deps_right[index, :, :])
    for index, sg in enumerate(sub_gf):
        G_to_deps(sg, sub_deps_forward[index, :, :])
    for index, sg in enumerate(sub_gb):
        G_to_deps(sg, sub_deps_backward[index, :, :])

    # sort the index of nodes_list, then the deps_matrix can match the nodes order in list
    for index in range(len(sub_graphs_nodes_list)):
        sub_graphs_nodes_list[index].sort()

    return sub_deps_move, sub_deps_left, sub_deps_right, sub_deps_forward, sub_deps_backward, sub_graphs_nodes_list


class InitialContainer(object):
    def __init__(self, blocks, positions, blocks_num, initial_container_size, allow_bot, child_graph_size, input_type='bot'):

        block_dim = len(initial_container_size)

        container = np.zeros(initial_container_size).astype(int)
        if block_dim == 2:
            for i in range(blocks_num):
                px, pz = positions[i]
                bx, bz = blocks[i]
                container[px:px+bx, pz:pz+bz] = i+1

        elif block_dim == 3:
            for i in range(blocks_num):
                px, py, pz = positions[i]
                bx, by, bz = blocks[i]
                container[px:px+bx, py:py+by, pz:pz+bz] = i+1

        rotate_types = np.math.factorial(block_dim)

        self.input_type = input_type
        self.rotate_types = rotate_types
        self.block_dim = block_dim
        self.blocks_num = blocks_num
        
        self.blocks = blocks
        self.positions = positions
        self.container = container
        self.all_bot = allow_bot
        self.child_graph_size = child_graph_size        
    
        deps_move, deps_left, deps_right, deps_forward, deps_backward, _, _ = calc_dependent(blocks[:blocks_num], positions, container)

        # build the precedence graphs
        G_move = nx.DiGraph()
        for block_index in range(blocks_num):
            G_move.add_node(block_index)
        G_left = G_move.copy()
        G_right = G_move.copy()
        G_forward = G_move.copy()
        G_backward = G_move.copy()

        # movement deps
        for i in range(blocks_num):
            for j in range(blocks_num):
                if deps_move[i,j] == True:
                    G_move.add_edge( i, j )

        # rotation deps
        # 2D
        if allow_bot and block_dim == 2:
            for i in range(blocks_num):
                for j in range(blocks_num):
                    if deps_left[i, j] == True:
                        G_left.add_edge( i, j )
                    if deps_right[i, j] == True:
                        G_right.add_edge( i, j )
        # 3D
        elif allow_bot and block_dim == 3:
            for i in range(blocks_num):
                for j in range(blocks_num):
                    if deps_left[i, j] == True:
                        G_left.add_edge( i, j )
                    if deps_right[i, j] == True:
                        G_right.add_edge( i, j )
                    if deps_forward[i, j] == True:
                        G_forward.add_edge( i, j )
                    if deps_backward[i, j] == True:
                        G_backward.add_edge( i, j )
        
        self.G_move = G_move
        self.G_left = G_left
        self.G_right = G_right
        self.G_forward = G_forward
        self.G_backward = G_backward

        self.gm = self.G_move.copy()
        self.gl = self.G_left.copy()
        self.gr = self.G_right.copy()
        self.gf = self.G_forward.copy()
        self.gb = self.G_backward.copy()

        self.after_nodes_list = list(self.G_move.nodes)
        self.sub_graph_nodes = []

    def sub_deps_graph(self):

        sub_gm = []
        sub_gl = []
        sub_gr = []
        sub_gf = []
        sub_gb = []
        
        def decompose(nodes_list):
            nodes = nodes_list
            subgm = self.G_move.subgraph(nodes).copy()
            subgl = self.G_left.subgraph(nodes).copy()
            subgr = self.G_right.subgraph(nodes).copy()
            subgf = self.G_forward.subgraph(nodes).copy()
            subgb = self.G_backward.subgraph(nodes).copy()
            # check rotations
            for node in subgl.nodes():
                for pre_node in self.G_left.predecessors(node):
                    if pre_node in self.after_nodes_list:
                        subgl.add_edge(node, node)
            for node in subgr.nodes():
                for pre_node in self.G_right.predecessors(node):
                    if pre_node in self.after_nodes_list:
                        subgr.add_edge(node, node)
            for node in subgf.nodes():
                for pre_node in self.G_forward.predecessors(node):
                    if pre_node in self.after_nodes_list:
                        subgf.add_edge(node, node)
            for node in subgb.nodes():
                for pre_node in self.G_backward.predecessors(node):
                    if pre_node in self.after_nodes_list:
                        subgb.add_edge(node, node)
            # add the sub-graphs
            sub_gm.append(subgm)
            sub_gl.append(subgl)
            sub_gr.append(subgr)
            sub_gf.append(subgf)
            sub_gb.append(subgb)
            # remove from the graphs
            for node in nodes:
                try:
                    self.gm.remove_node(node)
                    self.gl.remove_node(node)
                    self.gr.remove_node(node)
                    self.gf.remove_node(node)
                    self.gb.remove_node(node)
                except nx.NetworkXException as e:
                    pass
                    # print(e)

        gm_copy = self.gm.copy()
        
        stop = False
        while gm_copy.number_of_nodes() > 0:
            if stop == True:
                break
            # get nodes with in_degree==0 in Gm
            if gm_copy.number_of_nodes() == 1:
                nodes = list(gm_copy.nodes)
            else:
                nodes = np.array(gm_copy.in_degree)
                mask = np.where(nodes[:,1]==0)[0]
                nodes = nodes[mask][:, 0]
            for node in nodes:
                if len(self.sub_graph_nodes) == self.child_graph_size:
                    decompose( self.sub_graph_nodes )
                    stop = True
                    break
                
                self.sub_graph_nodes.append(node)    # add to sub_graph
                gm_copy.remove_node(node)       # remove from the graph
                self.after_nodes_list.remove(node)   # remove from the nodes_list
                # check if the sub graph full
                if len(self.sub_graph_nodes) == self.child_graph_size:
                    decompose( self.sub_graph_nodes )
                    stop = True
                    break

        sub_deps_move = np.zeros((self.child_graph_size, self.child_graph_size)).astype(int)
        sub_deps_left = np.zeros((self.child_graph_size, self.child_graph_size)).astype(int)
        sub_deps_right = np.zeros((self.child_graph_size, self.child_graph_size)).astype(int)
        sub_deps_forward = np.zeros((self.child_graph_size, self.child_graph_size)).astype(int)
        sub_deps_backward = np.zeros((self.child_graph_size, self.child_graph_size)).astype(int)

        def G_to_deps(G, deps):
            nodes = list(G.nodes())
            for edge in G.edges():
                deps[  nodes.index(edge[0]), nodes.index(edge[1])] = 1

        for index, sg in enumerate(sub_gm):
            G_to_deps(sg, sub_deps_move)
        for index, sg in enumerate(sub_gl):
            G_to_deps(sg, sub_deps_left)
        for index, sg in enumerate(sub_gr):
            G_to_deps(sg, sub_deps_right)
        for index, sg in enumerate(sub_gf):
            G_to_deps(sg, sub_deps_forward)
        for index, sg in enumerate(sub_gb):
            G_to_deps(sg, sub_deps_backward)
        
        self.sub_graph_nodes.sort()

        return sub_deps_move, sub_deps_left, sub_deps_right, sub_deps_forward, sub_deps_backward

    def convert_to_input(self):
        '''
        convert the current sub_graph to the input of network
        ----
        returns:
        ----
            static
            dynamic
        '''        
        # sub_deps_move, sub_deps_left, sub_deps_right, sub_deps_forward, sub_deps_backward, self.sub_graph_nodes
        move, left, right, forward, backward = self.sub_deps_graph()
        
        up = np.zeros_like(move)
        down = np.zeros_like(move)

        rotate_order = tools.calc_rotate_order( self.sub_graph_nodes, self.block_dim, self.blocks_num)

        static_index = []
        for r in range(self.rotate_types):
            for i in range(self.child_graph_size):
                static_index.append(i)
        static_index = np.array(static_index).reshape( self.child_graph_size * self.rotate_types, 1)

        static = np.concatenate( ( static_index, self.blocks[rotate_order]), 1 ).transpose(1, 0)

        if self.input_type == 'bot':
            dynamic = np.zeros( ( self.child_graph_size * 3, len(rotate_order)) )
        else:
            dynamic = np.zeros( ( self.child_graph_size, len(rotate_order) ) )

        if self.block_dim == 3:
            for p_index, p in enumerate(itertools.permutations( range(self.block_dim) )):
                if p[-1] == 0:
                    dynamic[:, p_index * self.child_graph_size : (p_index+1)*self.child_graph_size ] = np.concatenate( (move, left, right), 0 )
                elif p[-1] == 1:
                    dynamic[:, p_index * self.child_graph_size : (p_index+1)*self.child_graph_size ] = np.concatenate( (move, forward, backward), 0 )
                elif p[-1] == 2:
                    dynamic[:, p_index * self.child_graph_size : (p_index+1)*self.child_graph_size ] = np.concatenate( (move, up, down), 0 )
        elif self.block_dim == 2:
            for p_index, p in enumerate(itertools.permutations( range(self.block_dim) )):
                if p[-1] == 0:
                    dynamic[:, p_index*self.child_graph_size : (p_index+1)*self.child_graph_size ] = np.concatenate( (move, left, right), 0 )
                elif p[-1] == 1:
                    dynamic[:, p_index*self.child_graph_size : (p_index+1)*self.child_graph_size ] = np.concatenate( (move, up, down), 0 )
        return static, dynamic

    def remove_block(self, block_id):
        '''
        Remove a block from container (actually we just remove from the list, the graph will update when you call convert_to_input)
        ---
        params:
        ---
            block_id: int
        '''
        try:
            self.sub_graph_nodes.remove(block_id)
        except Exception as e:
            pass
            # print(e)
    
    def is_last_graph(self):
        return len(self.after_nodes_list) == 0



if __name__ == "__main__":

    blocks_num = 10
    gt_packing_size = [50, 235]
    initial_container_size = [70, 500]
    size_range = [10, 50]
    input_type = 'rot'

    for h in range(170, 180):
        gt_packing_size[1] = h
        t = []
        for _ in range(10):
            tt = time.time()
            rotate_blocks, positions, _, _, _ = generate_blocks_with_GT(blocks_num, gt_packing_size, 
                                                        initial_container_size, size_range, input_type)
            t.append(time.time()-tt)
        print(h, ': ', np.mean(t))
    # gt_blocks, gt_positions, gt_container = BPP_Generator_2D(blocks_num, gt_packing_size, size_range)

    # print(gt_blocks)
    # print(gt_positions)