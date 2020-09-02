'''
Simple example
'''

import numpy as np

import sys
sys.path.append('../')
import tools


blocks = np.array([
    [2, 1], 
    [2, 2],
    [2, 1],
])
blocks = np.array([
    [1, 2], 
    [2, 3],
    [1, 2],
])
blocks = np.array([
    [3, 2], 
    [1, 1],
    [1, 2],
])

name = 'c'

container_width = 4
container_size= (container_width, 6)


positions, container, stable, ratio, [valid_size, box_size, empty_size, stable_num, packing_height] = tools.calc_positions_lb_greedy(blocks, container_size, "C+P+S-lb-soft")

tools.draw_container_2d(blocks, positions, container_size, "C+P+S-lb-soft", save_name=name )