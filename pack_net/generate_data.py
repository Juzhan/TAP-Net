
import numpy as np
import os
import sys

from tqdm import tqdm

sys.path.append('../')
import generate

def create_dataset_gt(blocks_num, train_size, valid_size, obj_dim, target_container_width, target_container_height, size_range, seed=None):
    blocks_num = int(blocks_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    if obj_dim == 2:
        target_container_size = [target_container_width, target_container_height]
        train_dir = './data/gt_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/gt_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
    elif obj_dim == 3:
        target_container_size = [target_container_width, target_container_width, target_container_height]
        train_dir = './data/gt_3d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/gt_3d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(train_dir + 'blocks.txt') and os.path.exists(valid_dir  + 'blocks.txt'):
        return train_dir, valid_dir
    
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    init_container_width = 7
    init_container_size = (init_container_width, 50)

    random_distribution = generate.generate_height_prob(obj_dim, blocks_num, size_range, init_container_width, target_container_width)
    prob, key = random_distribution

    def generate_data(data_dir, data_size):

        gt_blocks_f = open(data_dir + 'gt_blocks.txt', 'w')
        gt_pos_f = open(data_dir + 'gt_pos.txt', 'w')
        

        for _ in tqdm(range(data_size)):
            # gt_blocks, gt_positions = generate.generate_blocks_with_GT(blocks_num, target_container_size, init_container_size, 1, size_range, 'bot', key, prob)
            gt_blocks, gt_positions = generate.generate_blocks_with_GT(blocks_num, target_container_size, init_container_size, 1, size_range, 'bot', 0, True )


            gt_pos_f.writelines(arr2str( gt_positions ) )
            gt_blocks_f.writelines(arr2str( gt_blocks ) )


        gt_pos_f.close()
        gt_blocks_f.close()

    if not os.path.exists(train_dir + 'blocks.txt'):
        generate_data(train_dir, train_size)
    if not os.path.exists(valid_dir + 'blocks.txt'):
        generate_data(valid_dir, valid_size)
    return train_dir, valid_dir



def create_dataset_rand(blocks_num, train_size, valid_size, obj_dim, target_container_width, target_container_height, size_range, seed=None):
    blocks_num = int(blocks_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    if obj_dim == 2:
        target_container_size = [target_container_width, target_container_height]
        train_dir = './data/rand_2d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/rand_2d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
    elif obj_dim == 3:
        target_container_size = [target_container_width, target_container_width, target_container_height]
        train_dir = './data/rand_3d/pack-train-' + str(blocks_num) + '-' + str(train_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'
        valid_dir = './data/rand_3d/pack-valid-' + str(blocks_num) + '-' + str(valid_size) + '-' + str(target_container_width) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(train_dir + 'blocks.txt') and os.path.exists(valid_dir  + 'blocks.txt'):
        return train_dir, valid_dir
    
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    def generate_data(data_dir, data_size):

        gt_blocks_f = open(data_dir + 'blocks.txt', 'w')
        gt_pos_f = open(data_dir + 'pos.txt', 'w')

        size_min = size_range[0]
        size_max = size_range[1]
        

        for _ in tqdm(range(data_size)):
            gt_blocks = np.random.randint( size_min, size_max, (blocks_num * obj_dim) )
            gt_positions = np.random.randint( size_min, size_max, (blocks_num * obj_dim) )


            gt_pos_f.writelines(arr2str( gt_positions ) )
            gt_blocks_f.writelines(arr2str( gt_blocks ) )


        gt_pos_f.close()
        gt_blocks_f.close()

    if not os.path.exists(train_dir + 'blocks.txt'):
        generate_data(train_dir, train_size)
    if not os.path.exists(valid_dir + 'blocks.txt'):
        generate_data(valid_dir, valid_size)
    return train_dir, valid_dir



if __name__ == "__main__":

    obj_dim = 2
    
    blocks_num = 10
    target_container_width = 5
    target_container_height = 20 # whatever

    size_range = [1, 5]

    train_size = 64000
    valid_size = 10000
    
    # generate gt dataset
    # create_dataset_gt(blocks_num, train_size, valid_size, obj_dim, target_container_width, target_container_height, size_range)

    # generate random dataset
    create_dataset_rand(blocks_num, train_size, valid_size, obj_dim, target_container_width, target_container_height, size_range)