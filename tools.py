import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import networkx as nx
import itertools
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import copy
from matplotlib import rcParams, rc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F


# rcParams['font.family'] = 'Calibri'
# rcParams['font.calibri'] = ['cal']
# rc('text',usetex=True)

# rcParams['mathtext.fontset'] = 'custom'
# rcParams['mathtext.it'] = 'STIXGeneral:italic'
# rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
###patch end###


def calc_colors(max_len):
    mid_len = int(max_len/2)
    colors = np.zeros( (max_len, 3) )

    start_color = [237, 125, 49]
    # mid_color = [84, 130, 53]
    mid_color = [84, 250, 180]
    # end_color = [255, 217, 102]
    end_color = [160, 16, 40]
    
    for i in range(3): 
        colors[:mid_len, i] = np.linspace( start_color[i] , mid_color[i], mid_len)
        colors[mid_len:, i] = np.linspace( end_color[i] , end_color[i], max_len - mid_len)
    
    colors_str = []
    for color in colors:
        color = color.astype('int')
        colors_str.append( 
            '#%02X%02X%02X' % (color[0], color[1], color[2])
         )

    colors_str = ['#ffd966', '#a9d08e', '#f4b084', '#9bc2e6', '#ff7c80', '#c6b5f0', 
            '#a0c8c0', '#f5f4c2', '#c0bed3', '#dd9286',  '#91aec6',
            'silver', 'khaki', 'lime', 'coral',
            'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender',  ]

    return np.array(colors_str)

def calc_labels(max_len):
    labels = [ str(i) for i in range(1, max_len) ]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return np.array(labels)

# labels = [ str(i) for i in range(1, 51) ]

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


colors = ['#ffd966', '#a9d08e', '#f4b084', '#9bc2e6', '#ff7c80', '#c6b5f0', 
    '#a0c8c0', '#f5f4c2', '#c0bed3', '#dd9286',  '#91aec6',
    'silver', 'khaki', 'lime', 'coral',
    'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender',  ]

labels = np.array(labels)
colors = calc_colors(len(labels))

def get_cube_data(pos=[0,0,0], size=[1,1,1], face_index=[0]):
    l, w, h = size
    a, b, c = pos
    x = [ [a, a + l, a + l, a, a],
            [a, a + l, a + l, a, a],
            [a, a + l, a + l, a, a],
            [a, a + l, a + l, a, a]
        ]
    y = [ [b, b, b + w, b + w, b],
            [b, b, b + w, b + w, b],  
            [b, b, b, b, b],
            [b + w, b + w, b + w, b + w, b + w]
             ]   
    z = [ [c, c, c, c, c],                       
            [c + h, c + h, c + h, c + h, c + h],
            [c, c, c + h, c + h, c],
            [c, c, c + h, c + h, c] ]
    x = np.array(x)[face_index]
    y = np.array(y)[face_index]
    z = np.array(z)[face_index]
    return x, y, z

def draw_container_voxel_old(container, blocks_num, colors=colors, reward_type='comp', 
    order=None, 
    rotate_state=None,
    feasibility=None, 
    draw_top = True,
    view_type='front',
    blocks_num_to_draw=None,
    save_title='3D packing', save_name='result'):

    colors = calc_colors(blocks_num)
    # if feasibility is not None:
    #     feasibility = np.ones(blocks_num)
    
    if blocks_num_to_draw is None:
        blocks_num_to_draw = blocks_num
    container_width = container.shape[0]
    container_length = container.shape[1]
    container_height = container.shape[2]

    if rotate_state is None:
        rotate_state = np.zeros(blocks_num)

    if order is None:
        order = [i for i in range(blocks_num)]
        
    rotate_state = np.array(rotate_state)

    edges_not_rotate_color = np.empty( np.sum(rotate_state == False)  ).astype('object')
    edges_rotate_color = np.empty( np.sum(rotate_state == True) ).astype('object')

    for i in range(len(edges_not_rotate_color)):
        edges_not_rotate_color[i] = '#00225515'
    for i in range(len(edges_rotate_color)):
        edges_rotate_color[i] = '#002255'

    blocks_color = np.empty_like(container).astype('object')
    voxels = np.zeros_like(container).astype('bool')

    voxels_rotate = np.zeros_like(container).astype('bool')
    voxels_not_rotate = np.zeros_like(container).astype('bool')

    place_order = []
    for i in range(blocks_num_to_draw):
        block_index = order[i]
        block_voxel = (container == i+1)
        blocks_color[block_voxel] = colors[block_index]

        # if rotate_state is not None:
        if rotate_state[i] == True:
            voxels_rotate = voxels_rotate | block_voxel
        else:
            voxels_not_rotate = voxels_not_rotate | block_voxel
        # else:
        #     voxels = voxels | block_voxel
        place_order.append(block_index)


    plt.close('all')
    fig = plt.figure( figsize=(4,7) )
    fig.subplots_adjust(left=0, right=1, bottom=-0.10)
    ax = fig.gca(projection='3d')
    
    xticks = [ i for i in range(0, container_width+1)]
    yticks = [ i for i in range(0, container_length+1)]
    zticks = [ i for i in range(0, container_height+1)]
    xlabels = [ '' for i in range(0, container_width+1)]
    ylabels = [ '' for i in range(0, container_length+1)]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    
    ax.set_xlim3d(0, container_width)
    ax.set_ylim3d(0, container_length)
    ax.set_zlim3d(0, container_height)

    # matplotlib.rc('axes',edgecolor='#00000000')
    # ax.tick_params(axis='both', which='major', pad=-1)
    # ax.scatter([0],[0],[0], marker='o', s=30, c='k')
    
    zorder = 2000
    w, l, h = container.shape

    ax.plot([w,w], [0,0], [0,h], 'k-', linewidth=2, zorder=zorder)
    ax.plot([0,0], [0,0], [0,h], 'r-', linewidth=2, zorder=zorder*2)
    if view_type == 'front':
        ax.plot([w,w], [l,l], [0,h], 'k-', linewidth=2, zorder=-1)
        ax.plot([0,0], [l,l], [0,h], 'k-', linewidth=2, zorder=zorder)
    else:
        ax.plot([w,w], [l,l], [0,h], 'k-', linewidth=2, zorder=zorder)
        ax.plot([0,0], [l,l], [0,h], 'k-', linewidth=2, zorder=-1)
    # top
    top_h = h
    for z in range(h):
        if (container[:,:,z] == 0).all():
            top_h = z
            break

    gap = 0.0
    X, Y, Z = get_cube_data(pos=[-gap,-gap,top_h], 
        size=[container_width + 2*gap, container_length + 2*gap, 0.001], 
        face_index=[0, 2])
    
    if draw_top:
        ax.plot_surface(X, Y, Z, color='gray', shade=False, edgecolor='#10101050', alpha=0.2)
    
    # bottom
    ax.plot([0,w], [0,0], [0,0], 'k-', linewidth=2, zorder=zorder)
    ax.plot([0,w], [l,l], [0,0], 'k-', linewidth=2, zorder=-1)
    if view_type == 'front':
        ax.plot([0,0], [0,l], [0,0], 'k-', linewidth=2, zorder=zorder)
        ax.plot([w,w], [0,l], [0,0], 'k-', linewidth=2, zorder=-1)
    else:
        ax.plot([0,0], [0,l], [0,0], 'k-', linewidth=2, zorder=-1)
        ax.plot([w,w], [0,l], [0,0], 'k-', linewidth=2, zorder=zorder)
    
    if draw_top:    
        zlabels = [ str(i) if i==top_h else '' for i in range(0, container_height+1)]
    else:
        zlabels = [ '' for i in range(0, container_height+1)]
    ax.set_zticklabels(zlabels)

    ax.voxels(voxels_rotate, facecolors=blocks_color, edgecolor=edges_rotate_color, alpha=1, zorder=-1, shade=False)
    ax.voxels(voxels_not_rotate, facecolors=blocks_color, edgecolor=edges_not_rotate_color, alpha=1, zorder=-1, shade=False)

    if len(place_order) > 0 and feasibility is not None:
        place_order_str = ''
        for i in range(len(place_order) - 1):
            if feasibility[i] == False:
                label = str(1 + int(place_order[i]))
                place_order_str = place_order_str + r'$\overline{\bf{' + label + '}}$ '
            else:
                label = str(1 + int(place_order[i]))    
                if rotate_state[i] == False:
                    place_order_str = place_order_str + label + ' '
                else:
                    place_order_str = place_order_str + r'$\bf{' + label + '}$ '

        if feasibility[len(place_order) - 1] == False:
            label = str(1 + int(place_order[-1]))
            place_order_str = place_order_str + r'$\overline{\bf{' + label + '}}$'
        else:
            label = str(1 + int(place_order[-1]))
            if rotate_state[len(place_order) - 1] == False:
                place_order_str = place_order_str + label
            else:
                place_order_str = place_order_str + r'$\bf{' + label + '}$'
        
        save_title += '\n' + place_order_str
        ax.set_title(save_title, fontsize=20, y=0.95)
    else:
        ax.set_title(save_title + '\n', fontsize=20, y=0.95)
    
    

    if view_type == 'front':
        ax.view_init(13, -130)
        plt.savefig(save_name + '-' + view_type + '.png', bbox_inches=0, dpi=400)
    else:
        ax.view_init(13, -40)
        plt.savefig(save_name + '-' + view_type + '.png', bbox_inches=0, dpi=400)

    # for an in range(0, 360, 20):
    #     ax.view_init(13, an)
    #     plt.savefig(save_name + '-%d.png' % an, bbox_inches='tight', dpi=400)
    # ax.view_init(90, 0)
    # plt.savefig(save_name + '-top.png', bbox_inches='tight', dpi=400)


def draw_container_voxel(container, blocks_num, colors=colors, reward_type='comp', 
    order=None, 
    rotate_state=None,
    feasibility=None, 
    draw_top = True,
    view_type='front',
    blocks_num_to_draw=None,
    save_title='3D packing', save_name='result'):

    colors = calc_colors(blocks_num)

    if blocks_num_to_draw is None:
        blocks_num_to_draw = blocks_num
    container_width = container.shape[0]
    container_length = container.shape[1]
    container_height = container.shape[2]

    if rotate_state is None:
        rotate_state = np.zeros(blocks_num)

    if order is None:
        order = [i for i in range(blocks_num)]
        
    rotate_state = np.array(rotate_state)

    edges_not_rotate_color = np.empty( np.sum(rotate_state == False)  ).astype('object')
    edges_rotate_color = np.empty( np.sum(rotate_state == True) ).astype('object')

    for i in range(len(edges_not_rotate_color)):
        # edges_not_rotate_color[i] = '#00225515'
        edges_not_rotate_color[i] = '#00225500'
    for i in range(len(edges_rotate_color)):
        # edges_rotate_color[i] = '#002255'
        edges_rotate_color[i] = '#00225500'

    blocks_color = np.empty_like(container).astype('object')
    voxels = np.zeros_like(container).astype('bool')

    voxels_rotate = np.zeros_like(container).astype('bool')
    voxels_not_rotate = np.zeros_like(container).astype('bool')

    place_order = []
    for i in range(blocks_num_to_draw):
        block_index = order[i]
        block_voxel = (container == i+1)
        blocks_color[block_voxel] = colors[block_index]

        # if rotate_state is not None:
        if rotate_state[i] == True:
            voxels_rotate = voxels_rotate | block_voxel
        else:
            voxels_not_rotate = voxels_not_rotate | block_voxel
        # else:
        #     voxels = voxels | block_voxel
        place_order.append(block_index)


    plt.close('all')
    fig = plt.figure( figsize=(3,5) )
    # fig.subplots_adjust(left=0, right=1)
    fig.subplots_adjust(left=0, right=1, bottom=-0.00)
    # ax = fig.gca(projection='3d')
    
    ax = Axes3D(fig)

    xticks = [ i for i in range(0, container_width+1)]
    yticks = [ i for i in range(0, container_length+1)]
    zticks = [ i for i in range(0, container_height+1)]
    xlabels = [ '' for i in range(0, container_width+1)]
    ylabels = [ '' for i in range(0, container_length+1)]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    
    # ax.set_xticklabels(xlabels)
    # ax.set_yticklabels(ylabels)
    
    ax.set_xlim3d(0, container_width)
    ax.set_ylim3d(0, container_length)
    ax.set_zlim3d(0, container_height)

    plt.grid(True, alpha=0.3, lw=1 )

    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)
    ax.set_axisbelow(True)
    
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # for tic in ax.xaxis.get_major_ticks():
    #     tic.tick1line.set_visible(False)
    #     tic.tick2line.set_visible(False)
    # for tic in ax.yaxis.get_major_ticks():
    #     tic.tick1line.set_visible(False)
    #     tic.tick2line.set_visible(False)
    # for tic in ax.zaxis.get_major_ticks():
    #     tic.tick1line.set_visible(False)
    #     tic.tick2line.set_visible(False)

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # plt.tick_params(axis='both', which='both', bottom=False, top=False,
    #             left=False, right=False)
        
    # plt.set_hatch(hatch)
    # ax.spines['top'].set_visible(False)
    
    # plt.xticks(range(0,container_width+1))
    # plt.yticks(range(0,container_height+1))

    # matplotlib.rc('axes',edgecolor='#00000000')
    # ax.tick_params(axis='both', which='major', pad=-1)
    # ax.scatter([0],[0],[0], marker='o', s=30, c='k')
    
    zorder = 2000
    w, l, h = container.shape

    ax.plot([w,w], [0,0], [0,h], 'k-', linewidth=2, zorder=zorder)
    ax.plot([0,0], [0,0], [0,h], 'r-', linewidth=2, zorder=zorder*2)
    if view_type == 'front':
        ax.plot([w,w], [l,l], [0,h], 'k-', linewidth=2, zorder=-1)
        ax.plot([0,0], [l,l], [0,h], 'k-', linewidth=2, zorder=zorder)
    else:
        ax.plot([w,w], [l,l], [0,h], 'k-', linewidth=2, zorder=zorder)
        ax.plot([0,0], [l,l], [0,h], 'k-', linewidth=2, zorder=-1)
    # top
    top_h = h
    for z in range(h):
        if (container[:,:,z] == 0).all():
            top_h = z
            break

    gap = 0.0
    X, Y, Z = get_cube_data(pos=[-gap,-gap,top_h], 
        size=[container_width + 2*gap, container_length + 2*gap, 0.001], 
        face_index=[0, 2])
    
    # if draw_top:
    #     ax.plot_surface(X, Y, Z, color='gray', shade=False, edgecolor='#10101050', alpha=0.2)
    
    # bottom
    ax.plot([0,w], [0,0], [0,0], 'k-', linewidth=2, zorder=zorder)
    ax.plot([0,w], [l,l], [0,0], 'k-', linewidth=2, zorder=-1)
    if view_type == 'front':
        ax.plot([0,0], [0,l], [0,0], 'k-', linewidth=2, zorder=zorder)
        ax.plot([w,w], [0,l], [0,0], 'k-', linewidth=2, zorder=-1)
    else:
        ax.plot([0,0], [0,l], [0,0], 'k-', linewidth=2, zorder=-1)
        ax.plot([w,w], [0,l], [0,0], 'k-', linewidth=2, zorder=zorder)
    
    if draw_top:
        zlabels = [ str(i) if i==top_h else '' for i in range(0, container_height+1)]
    else:
        zlabels = [ '' for i in range(0, container_height+1)]
    # ax.set_zticklabels(zlabels)

    ax.voxels(voxels_rotate, facecolors=blocks_color, edgecolor=edges_rotate_color, alpha=1, zorder=-1, shade=False)
    ax.voxels(voxels_not_rotate, facecolors=blocks_color, edgecolor=edges_rotate_color, alpha=1, zorder=-1, shade=False)


    # ax.voxels(voxels_rotate, facecolors=blocks_color, edgecolor=edges_rotate_color, alpha=1, zorder=-1, shade=False)
    # ax.voxels(voxels_not_rotate, facecolors=blocks_color, edgecolor=edges_not_rotate_color, alpha=1, zorder=-1, shade=False)


    if view_type == 'front':
        ax.view_init(13, -130)
        plt.savefig(save_name + '-' + view_type + '.png', bbox_inches=0, dpi=400)
    else:
        ax.view_init(13, -40)
        plt.savefig(save_name + '-' + view_type + '.png', bbox_inches=0, dpi=400)

    # for an in range(0, 360, 20):
    #     ax.view_init(13, an)
    #     plt.savefig(save_name + '-%d.png' % an, bbox_inches='tight', dpi=400)
    # ax.view_init(90, 0)
    # plt.savefig(save_name + '-top.png', bbox_inches='tight', dpi=400)




def draw_container_2d_old( blocks, positions, container_size, reward_type='comp', 
        order=None, 
        stable=None, 
        feasibility=None,
        rotate_state=[], 
        labels=labels, colors=colors, 
        save_title='', save_name='./a' ):

    container_width = container_size[0]
    container_height = 15
    blocks_num = len(blocks)

    # if feasibility is not None:
    #     feasibility = np.ones(blocks_num)
    colors = calc_colors(blocks_num)
    
    if order is None:
        order = [i for i in range(blocks_num)]
    if stable is None:
        stable = [True for i in range(blocks_num)]
    
    rotate_blocks = []
    place_order = []
    
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    plt.xlim(0,container_width)
    plt.ylim(0,container_height)
    plt.grid(True, alpha=0.4 )
    plt.xticks(range(0,container_width+1), fontsize=5)
    plt.yticks(range(0,container_height+1), fontsize=5)
    for block_i in range(blocks_num):
        if reward_type == 'hard' or reward_type == 'pyrm-hard' or reward_type == 'CPS' or reward_type=='pyrm-hard-sum' or reward_type=='pyrm-hard-SUM':
            if stable[block_i] == False:
                continue
        x = positions[block_i][0]
        y = positions[block_i][1]
        w = blocks[block_i][0]
        h = blocks[block_i][1]
        if len(rotate_state) > 0:
            if rotate_state[block_i] == True:
                rotate_blocks.append([ x, y, w, h, colors[order][block_i]])

        # ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=colors[order][block_i], edgecolor='#00225515', linestyle='-'))
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=colors[order][block_i], edgecolor='black', linestyle='-'))
        ax.text(x+w/2, y+h/2, str(labels[order][block_i]), ha='center', va='center', zorder=2000)

        place_order.append(order[block_i])

    for rotate_block in rotate_blocks:
        x, y, w, h, color = rotate_block
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linestyle='-', linewidth=1, hatch='\\\\'))


    if len(place_order) > 0 and feasibility is not None:
        place_order_str = ''
        for i in range(len(place_order) - 1):
            if feasibility[i] == False:
                label = str(1 + int(place_order[i]))
                place_order_str = place_order_str + r'$\overline{\bf{' + label + '}}$ '
            else:
                label = str(1 + int(place_order[i]))    
                if rotate_state[i] == False:
                    place_order_str = place_order_str + label + ' '
                else:
                    place_order_str = place_order_str + r'$\bf{' + label + '}$ '

        if feasibility[len(place_order) - 1] == False:
            label = str(1 + int(place_order[-1]))
            place_order_str = place_order_str + r'$\overline{\bf{' + label + '}}$'
        else:
            label = str(1 + int(place_order[-1]))
            if rotate_state[len(place_order) - 1] == False:
                place_order_str = place_order_str + label
            else:
                place_order_str = place_order_str + r'$\bf{' + label + '}$'
        
        save_title += '\n' + place_order_str
        ax.set_title(save_title, fontsize=15)
    else:
        ax.set_title(save_title + '\n', fontsize=15)
    plt.savefig(save_name + '.png', bbox_inches='tight', dpi=400)
    plt.cla()

def draw_container_2d( blocks, positions, container_size, reward_type='"C+P+S-lb-soft"', 
        order=None, 
        stable=None, 
        feasibility=None,
        rotate_state=[], 
        labels=labels, colors=colors, 
        save_title='', save_name='./a' ):

    container_width = container_size[0]
    if container_size[1] > 70:  container_height = 200
    else:                       container_height = 25   # 15
        
    blocks_num = len(blocks)

    # colors = calc_colors(blocks_num)
    
    if order is None:   order  = [i for i in range(blocks_num)]
    if stable is None:  stable = [True for i in range(blocks_num)]
    
    rotate_blocks = []
    place_order = []
    
    plt.close('all')
    if container_height >= 200: fig = plt.figure(figsize=(5, 20))
    else:                       fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1, aspect='equal')
    plt.xlim(0,container_width)
    plt.ylim(0,container_height)
    plt.grid(True, alpha=0.6, lw=1 )

    for axis in ['top','bottom','left','right']:
        if container_height >= 200: ax.spines[axis].set_linewidth(5)
        else:                       ax.spines[axis].set_linewidth(2) # 1

    ax.set_axisbelow(True)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    
    # plt.set_hatch(hatch)
    ax.spines['top'].set_visible(False)
    
    plt.xticks(range(0,container_width+1))
    plt.yticks(range(0,container_height+1))
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for block_i in range(blocks_num):
        # if reward_type == 'hard' or reward_type == 'pyrm-hard' or reward_type == 'CPS' or \
        #    reward_type == 'pyrm-hard-sum' or reward_type == 'pyrm-hard-SUM':
        if 'hard' in reward_type:
            if stable[block_i] == False:
                continue
        x = positions[block_i][0]
        y = positions[block_i][1]
        w = blocks[block_i][0]
        h = blocks[block_i][1]
        if len(rotate_state) > 0:
            if rotate_state[block_i] == True:
                rotate_blocks.append([ x, y, w, h, colors[order][block_i]])

        # ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=colors[order][block_i], edgecolor='#00225515', linestyle='-'))
        
        if container_height >= 200: 
            ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=colors[order][block_i], edgecolor='black', 
                                                            linewidth=2, linestyle='-'))
            ax.text(x+w/2, y+h/2, str(labels[order][block_i]), ha='center', va='center', zorder=2000, fontsize=40)
        else:
            ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=colors[order][block_i], edgecolor='black', 
                                                            linewidth=1, linestyle='-'))
            ax.text(x+w/2, y+h/2, str(labels[order][block_i]), ha='center', va='center', zorder=2000, fontsize=15) #10)

        place_order.append(order[block_i])

    for rotate_block in rotate_blocks:
        x, y, w, h, color = rotate_block
        if container_height >= 200: 
            lw = 2
            hatch = '\\'
            hatch_lw = 6
        else:
            lw = 1
            hatch = '\\\\\\'
            hatch_lw = 2
        
        blk = patches.Rectangle((x, y), w, h, facecolor='white', edgecolor='black', 
                                                linestyle='-', linewidth=lw, hatch=hatch)
        blk.set_edgecolor(color)
        ax.add_patch(blk)
        plt.rcParams['hatch.linewidth'] = hatch_lw
        blk = patches.Rectangle((x, y), w, h, facecolor='#11223300', edgecolor='black', 
                                                linestyle='-', linewidth=lw)
        blk.set_edgecolor('black')
        ax.add_patch(blk)

    plt.savefig(save_name + '.png', bbox_inches='tight', dpi=400)
    plt.savefig(save_name + '.pdf', bbox_inches='tight', dpi=400)
    plt.cla()



def draw_dep(deps, colors=colors, save_name='d'):
    '''
    draw dependence grph
    ----
    params:
        deps: n x n int array, store the dependence relation of blocks
            deps[i][j] == True ==> block_i can move after block_j is moved
    '''
    blocks_num = deps.shape[-1]

    colors = calc_colors(blocks_num)
    
    ang = np.linspace(0, np.pi * 2, blocks_num + 1)[:-1]
    pos = {}
    labels = {}
    for i in range(blocks_num):
        pos[i] = ( np.cos(ang[i]), np.sin(ang[i])  )
        labels[i] = str(1 + i)
        
    plt.close('all')        
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    G = nx.DiGraph()
    for i in range(len(deps)):
        G.add_node(i)
    for i in range(len(deps)):
        for j in range(len(deps)):
            if deps[i][j] == True:
                G.add_edge(j,i)
    nx.draw(G, pos, labels=labels, node_size=600, node_color=colors[:blocks_num], edge_color='black', font_size=15 )
    plt.savefig(os.path.abspath('./')+'/' + save_name + '.png', bbox_inches='tight', dpi=400)


# =================================== 

def is_stable(block, position, container):
    '''
    check for 3D packing
    ----
    '''
    if (position[2]==0):
        return True
    x_1 = position[0]
    x_2 = x_1 + block[0] - 1
    y_1 = position[1]
    y_2 = y_1 + block[1] - 1
    z = position[2] - 1
    obj_center = ( (x_1+x_2)/2, (y_1+y_2)/2 )

    # valid points right under this object
    points = []
    for x in range(x_1, x_2+1):
        for y in range(y_1, y_2+1):
            if (container[x][y][z] > 0):
                points.append([x, y])
    if(len(points) > block[0]*block[1]/2):
        return True
    if(len(points)==0 or len(points)==1):
        return False
    elif(len(points)==2):
        # whether the center lies on the line of the two points
        a = obj_center[0] - points[0][0]
        b = obj_center[1] - points[0][1]
        c = obj_center[0] - points[1][0]
        d = obj_center[1] - points[1][1]
        # same ratio and opposite signs
        if (b==0 or d==0):
            if (b!=d): return False
            else: return (a<0)!=(c<0) 
        return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )
    else:
        # calculate the convex hull of the points
        points = np.array(points)
        try:
            convex_hull = ConvexHull(points)
        except:
            # error means co-lines
            min_point = points[np.argmin( points[:,0] )]
            max_point = points[np.argmax( points[:,0] )]
            points = np.array( (min_point, max_point) )
            a = obj_center[0] - points[0][0]
            b = obj_center[1] - points[0][1]
            c = obj_center[0] - points[1][0]
            d = obj_center[1] - points[1][1]
            if (b==0 or d==0):
                if (b!=d): return False
                else: return (a<0)!=(c<0)
            return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )

        hull_path = Path(points[convex_hull.vertices])
        return hull_path.contains_point((obj_center))

def calc_positions(blocks, container_size, hard_condition=False):
    '''
    calculate the position in hard condition
    ---
    params:
    ---
        blocks: n x 2/3 array / list, blocks with some order
        container_size: 1 x 2/3 array / list, size of the container
    return:
    ---
        positions: n x 2/3 array, positions of blocks
        container: a 2/3-dimension array, final state of container
        place: n x 1 bool list, the blocks' placed state
    '''
    block_dim = blocks.shape[-1]
    if block_dim == 2:
        return calc_positions_2d(blocks, container_size, hard_condition)

    container = np.zeros(list(container_size))
    
    # level_free_space_x = [ [0] for i in range(container_size[2]) ] #(60, x)
    # level_free_space_y = [ [0] for i in range(container_size[2]) ] #(60, y)
    #(60, y, x)
    level_free_space = [ [ [0] for i in range(container_size[1]) ] for j in range(container_size[2]) ]

    blocks_num = len(blocks)
    positions = np.zeros((blocks_num, 3)).astype(int)
    stable = [False] * blocks_num # [ False for i in range(blocks_num)]

    for block_index, block in enumerate(blocks):
        block = block.astype('int')
        [block_x, block_y, block_z] = block
        isSettle = False
        # place from bottom to top
        # 每一层开始找位置
        for z, free_space_x in enumerate(level_free_space):
            if isSettle: break
            for y, free_space in enumerate(free_space_x):
                if isSettle: break
                elif y+block_y > container_size[1]: break 
                elif len(free_space)==0: continue
                for x in free_space:
                    if isSettle: break
                    while (x+block_x <= container_size[0]):
                        if (container[x:x+block_x, y:y+block_y, z:z+block_z] == 0).all():
                            # check stability
                            if not is_stable(np.array(block), np.array([x, y, z]), container):
                                if hard_condition == True:
                                    x += 1
                                    continue
                            else:
                                stable[block_index] = True

                            # update container and postions
                            container[x:x+block_x, y:y+block_y, z:z+block_z] = block_index + 1
                            under_space = container[x:x+block_x, y:y+block_y, 0:z]
                            container[x:x+block_x, y:y+block_y, 0:z][ under_space==0 ] = -1
                            positions[block_index] = [x, y, z]
                            # update free spaces
                            for zz, yy in itertools.product(range(block_z), range(block_y)):
                                try:
                                    level_free_space[z + zz][y + yy].remove(x)
                                except:
                                    pass
                                if (x+block_x < container_size[0]) and (container[x+block_x, y+yy, z+zz] == 0):
                                    level_free_space[z + zz][y + yy].append(x+block_x)
                            isSettle = True
                            break
                        else:
                            x += 1
    return positions, container, stable

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
        if left == 0:
            left_index += 1
        else:
            break
    for right in reversed(support):
        if right == 0:
            right_index -= 1
        else:
            break
    # 如果物体中心线不在范围内，不稳定
    if object_center <= left_index or object_center >= right_index:
        return False
    return True

def calc_positions_2d(blocks, container_size, hard_condition=False):
    '''
    calculate the position
    ---
    params:
    ---
        blocks: n x 2 array / list, blocks with some order
        container_width: a float number
        container_height: a float number
    return:
    ---
        positions: n x 2 array, positions of blocks
        container: w x h array, final state of container
        place: n x 1 list, the blocks' placed state
    '''
    block_dim = blocks.shape[-1]
    blocks_num = len(blocks)

    container_width, container_height = container_size
    container = np.zeros((container_width, container_height))
    level_free_space = [ [0] for i in range(container_height) ]

    positions = np.zeros((blocks_num, block_dim))
    stable = [ False for i in range(blocks_num)]

    for block_index, block in enumerate(blocks):
        block_width = int(block[0])
        block_height = int(block[1])
        isSettle = False
        # place from bottom to top
        # and free_space from level_free_space
        for y, free_space in enumerate(level_free_space):
            if isSettle:
                break
            y = int(y)
            if len(free_space) == 0:
                continue
            for x in free_space:
                x = int(x)
                if (isSettle):
                    continue
                while (x+block_width <= container_width):
                    if (container[x:x+block_width, y:y+block_height] == 0).all():
                        # check stable
                        if y>0:
                            support = container[x:x+block_width, y-1]
                            if not is_stable_2d(support, x, block_width):
                                if hard_condition == True:
                                    x += 1
                                    continue
                            else:
                                stable[block_index] = True
                        else:
                            stable[block_index] = True
                            
                        container[x:x+block_width, y:y+block_height] = block_index + 1
                        under_space = container[x:x+block_width, 0:y]
                        container[x:x+block_width, 0:y][ under_space==0 ] = -1
                        positions[block_index] = [x,y]
                        # update free_space
                        for height in range(block_height):
                            try:
                                level_free_space[y + height].remove(x)
                            except:
                                pass
                            if (x+block_width < container_width) and (container[x+block_width, y+height] == 0):
                                level_free_space[y + height].append(x+block_width)
                        isSettle = True
                        break
                    else:
                        x += 1
    return positions, container, stable

def calc_ratio(blocks, reward_type, container_size):
    '''
    calculate the ratio
    ---
    params:
    ---
        blocks: n x 2/3 array / list, blocks with some order
        reward_type: str
            'comp'
            'soft'
            'hard'
            'pyrm'
            'pyrm-soft'
            'pyrm-hard'
            'CPS'
        container_shape: 1 x 2/3 array / list, the shape of container
    return:
    ---
        ratio
        valid-size
        box-size
        empty-size
        stable-num
        packing_height
    '''
    blocks_num = len(blocks)
    block_dim = blocks.shape[-1]

    hard_condition = False
    if reward_type == 'hard' or reward_type == 'pyrm-hard' or reward_type == 'CPS' or reward_type == 'pyrm-hard-sum' or reward_type=='pyrm-hard-SUM':
        hard_condition = True
    _, container, stable = calc_positions(blocks, container_size, hard_condition)
    
    stable_num = np.sum(stable)
    
    valid_size = np.sum(container>=1)

    if block_dim == 3:
        for k in range(container.shape[2]):
            if (np.all(container[:, :, k] == 0)):
                # no blocks lie over z=k
                box = container[:, :, :k]
                break
        for j in range(container.shape[1]):
            if (np.all(container[:, j, :] == 0)):
                # no blocks lie over y=j
                box = box[:, :j, :]
                break
        for i in range(container.shape[0]):
            if (np.all(container[i, :, :] == 0)):
                # no blocks lie over x=i
                box = box[:i, :, :]
                break
    else:
        for j in range(container.shape[1]):
            # 直到没有物体在这一行
            if not (container[:, j] >=1 ).any():
                box = container[:, :j]
                break
        for i in range(container.shape[0]):
            # 直到没有物体在这一列
            if not (container[i, :] >= 1).any():
                box = box[:i, :]
                break
            
    if block_dim == 3:
        box_size = box.shape[0] * box.shape[1] * box.shape[2]
        # box_size = box.shape[-1] * container_size[0] * container_size[1]
        
    else:
        box_size = box.shape[0] * box.shape[1]
        # box_size = box.shape[-1] * container_size[0]
    
    packing_height = box.shape[-1]

    empty_size = np.sum( box == -1 )

    C = valid_size / box_size

    P = valid_size / (empty_size + valid_size)
    S = stable_num / blocks_num

    if reward_type == 'comp':
        ratio = C
    elif reward_type == 'soft':
        ratio = C * S
    elif reward_type == 'hard':
        ratio = C * S
    elif reward_type == 'pyrm':
        ratio = C + P
    elif reward_type == 'pyrm-soft':
        ratio = (C + P) * S
    elif reward_type == 'pyrm-hard':
        ratio = (C + P) * S
        
    elif reward_type == 'pyrm-soft-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-soft-SUM':	ratio = 2*C + P + S

    elif reward_type == 'pyrm-hard-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-hard-SUM':	ratio = 2*C + P + S

    elif reward_type == 'CPS':
        ratio = C * P * S
    else:
        print("Reward ERROR......................")
    
    return ratio, valid_size, box_size, empty_size, stable_num, packing_height
    # return ratio, valid_size, packing_height, empty_size, stable_num

def calc_real_order(order, blocks_num):
    '''Calc the real index order of blocks
    ---
    params:
    ----
        order: 1 x n int array / list, the select order of blocks (contain rotation index)
        blocks_num: int, number of blocks
    return:
    ----
        real_order: 1 x n int array, the real block index (without rotation index)
        rotate_state: 1 x n bool array, the rotation state of each block
    '''
    real_order = []
    rotate_state = []
    for o in order:
        if o < blocks_num:
            real_order.append(o)
            rotate_state.append(False)
        else:
            tmp = o
            while tmp >= blocks_num:
                tmp = tmp-blocks_num
            real_order.append(tmp)
            rotate_state.append(True)

    real_order = np.array(real_order)
    rotate_state = np.array(rotate_state)                        
    return real_order, rotate_state

def calc_rotate_order(order, block_dim, blocks_num):
    '''Calc the rotate index order of blocks
    ---
    params:
    ----
        order: 1 x n int array / list, the select order of blocks (no contain rotation index)
        blocks_dim: int, dimension of blocks
        blocks_num: int, number of blocks
    return:
    ----
        rotate_order: 1 x (n * rotate_types) int array, the block index (with rotation index)
    '''
    if block_dim == 2:
        rotate_types = 2
    elif block_dim == 3:
        rotate_types = 6

    rotate_order = []
    for r in range(rotate_types):
        for o in order:
            rotate_order.append(o + r*blocks_num )
    return rotate_order


'''
# ===========================
# NOTE greedy empty maxinal space

def calc_positions_greedy_3d(blocks, container_size, reward_type):

    blocks_num = len(blocks)
    blocks = blocks.astype('int')
    positions = np.zeros((blocks_num, 3)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num

    # states to calculate the scores
    bounding_box = [0, 0, 0]
    valid_size = 0
    empty_size = 0

    # level free space (z, y, x), to stores the empty-maximal-spaces' corners
    level_free_space = [ [ [0] for i in range(container_size[1]) ] for j in range(container_size[2]) ]

    # settle each block by the order
    for block_index, block in enumerate(blocks):

        block_x, block_y, block_z = block
        valid_size += block_x * block_y * block_z

        # empty-maximal-spaces list
        ems_list = []
        for z, free_space_x in enumerate(level_free_space):
            if z+block_z > container_size[2]: break
            # elif z>0 and level_free_space[z-1]==free_space_x: continue
            elif z>0 and (level_free_space[z-1] == [[0]]*container_size[1]): continue
            
            for y, free_space in enumerate(free_space_x):
                if y+block_y > container_size[1]: break
                # elif y>0 and free_space_x[y-1]==free_space: continue
                elif y>0 and free_space_x[y-1]==[0]: continue

                for x in free_space:
                    if x+block_x > container_size[0]: break
                    if y>0 and x in free_space_x[y-1] and (container[x:, y, z]==container[x:, y-1, z]).all(): continue
                    if z>0 and x in level_free_space[z-1][y] and (container[x:, y:, z]==container[x:, y:, z-1]).all(): continue
                    ems_list.append([x, y, z])
        # add the space started uppon the settled blcoks
        for blk_i in range(block_index):
            x, y, z = positions[blk_i]
            xx, yy, zz = blocks[blk_i]
			# if container[x+xx, y, z]==0 and not [x+xx, y, z] in ems_list:
			# 	ems_list.append([x+xx, y, z])   # same with free space
            if y+yy < container.shape[1]:
                if  container[x, y+yy, z]==0 and not [x, y+yy, z] in ems_list:
                   ems_list.append([x, y+yy, z])
            if z+zz < container.shape[-1]:
                if container[x, y, z+zz]==0 and not [x, y, z+zz] in ems_list:
                    ems_list.append([x, y, z+zz])
            # except Exception as e:
            #     print(e)
            #     import IPython
            #     IPython.embed()

        # search postions in each ems
        ems_num = len(ems_list)
        pos_ems = np.zeros((ems_num, 3)).astype(int)
        bbox_ems = [[0, 0, 0]] * ems_num
        is_settle_ems = [False] * ems_num
        is_stable_ems = [False] * ems_num
        empty_ems = [empty_size] * ems_num
        compactness_ems  = [0.0] * ems_num
        pyramidality_ems = [0.0] * ems_num
        X = int(container_size[0] - block_x + 1)
        Y = int(container_size[1] - block_y + 1)
        for ems_index, ems in enumerate(ems_list):
            bbox_ems[ems_index] = bounding_box.copy()
            # using buttom-left strategy in each ems
            _z = int(ems[2])
            for _x, _y in itertools.product( range( int(ems[0]), X), range( int(ems[1]), Y) ):
                if is_settle_ems[ems_index]: break
                if (container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] == 0).all():
                    if not is_stable(block, np.array([_x, _y, _z]), container):
                        if reward_type=='hard' or reward_type=='pyrm-hard' or reward_type == 'CPS' or reward_type=='pyrm-hard-sum' or reward_type=='pyrm-hard-SUM':
                            continue
                    else:
                        is_stable_ems[ems_index] = True
                    pos_ems[ems_index] = np.array([_x, _y, _z])
                    is_settle_ems[ems_index] = True
            # calculate the compactness and pyramidality for each ems if the block has been settled
            if is_settle_ems[ems_index]:
                _x, _y, _z = pos_ems[ems_index]
                if _x+block_x > bbox_ems[ems_index][0]: bbox_ems[ems_index][0] = _x+block_x
                if _y+block_y > bbox_ems[ems_index][1]: bbox_ems[ems_index][1] = _y+block_y
                if _z+block_z > bbox_ems[ems_index][2]: bbox_ems[ems_index][2] = _z+block_z
                # compactness
                # bbox_size = bbox_ems[ems_index][0] * bbox_ems[ems_index][1] * bbox_ems[ems_index][2]

                bbox_size = bbox_ems[ems_index][-1] * container_size[0] * container_size[1]
                
                compactness_ems[ems_index] = valid_size / bbox_size
                # pyramidality
                under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
                empty_ems[ems_index] += np.sum(under_space==0)
                if reward_type.startswith('pyrm') or reward_type=='CPS':
                    pyramidality_ems[ems_index] = valid_size / (empty_ems[ems_index] + valid_size)

        # if the block has not been settled
        if np.sum(is_settle_ems) == 0:
            valid_size -= block_x * block_y * block_z
            stable[block_index] = False
            continue

        # get the best ems
        ratio_ems = [c+p for c,p in zip(compactness_ems, pyramidality_ems)]
        best_ems_index = np.argmax(ratio_ems)
        while not is_settle_ems[best_ems_index]:
            ratio_ems.remove(ratio_ems[best_ems_index])
            best_ems_index = np.argmax(ratio_ems)

        # update the positions, stable list, bin, and the container
        _x, _y, _z = pos_ems[best_ems_index]
        positions[block_index] = pos_ems[best_ems_index]
        stable[block_index] = is_stable_ems[best_ems_index]
        bounding_box = bbox_ems[best_ems_index]
        empty_size = empty_ems[best_ems_index]
        container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
        under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
        container[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space==0 ] = -1

        # update the level_free_space
        for zz, yy in itertools.product( range(block_z), range(block_y) ):
            if _x in level_free_space[_z + zz][_y + yy]:
                level_free_space[_z + zz][_y + yy].remove(_x)
            if ( _x+block_x < container_size[0] ) and ( container[_x+block_x, _y+yy, _z+zz] == 0 ):
                level_free_space[_z + zz][_y + yy].append(_x + block_x)

    # calculate the compactness, pyramidality, stability, and the final ratio

    # box_size = bounding_box[0] * bounding_box[1] * bounding_box[2]
    box_size = bounding_box[-1] * container_size[0] * container_size[1]

    packing_height = bounding_box[-1]

    stable_num = np.sum(stable)
    
    C = valid_size / box_size

    P = valid_size / (empty_size + valid_size)
    S = stable_num / blocks_num

    if 	 reward_type == 'comp':			ratio = C
    elif reward_type == 'soft':			ratio = C * S
    elif reward_type == 'hard':			ratio = C * S
    elif reward_type == 'pyrm':			ratio = C + P
    elif reward_type == 'pyrm-soft':	ratio = (C + P) * S
    elif reward_type == 'pyrm-hard':	ratio = (C + P) * S
    
    elif reward_type == 'mcs-soft':	ratio = (C + P) * S
    elif reward_type == 'mcs-hard':	ratio = (C + P) * S
    
    elif reward_type == 'pyrm-soft-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-soft-SUM':	ratio = 2*C + P + S

    elif reward_type == 'pyrm-hard-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-hard-SUM':	ratio = 2*C + P + S

    elif reward_type == 'C+P-soft':	        ratio = C + P
    elif reward_type == 'C+P+S-soft':	    ratio = C + P + S

    elif reward_type == 'CPS':          ratio = C * P * S
    else:								print('GREEDY................')

    # return ratio, valid_size, box_size, empty_size, stable_num
    scores = [valid_size, box_size, empty_size, stable_num, packing_height]
    return positions, container, stable, ratio, scores

def calc_positions_greedy_2d(blocks, container_size, reward_type):

    block_dim = len(container_size)
    blocks_num = len(blocks)
    blocks = blocks.astype('int')
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num

    # states to calculate the scores
    bounding_box = [0, 0]
    valid_size = 0
    empty_size = 0

    # level free space (z, x), to stores the empty-maximal-spaces' corners
    level_free_space = [ [0] for i in range(container_size[-1]) ]

    # settle each block by the order
    for block_index, block in enumerate(blocks):

        block_x, block_z = block
        valid_size += block_x * block_z

        # empty-maximal-spaces list
        ems_list = []
        for z, free_space_x in enumerate(level_free_space):
            if z+block_z > container_size[-1]: break

            # elif z>0 and level_free_space[z-1]==free_space_x: continue
            elif z>0 and level_free_space[z-1]==[0]: continue

            for x in free_space_x:
                if x+block_x > container_size[0]: break
                if z>0 and x in level_free_space[z-1] and (container[x:, z]==container[x:, z-1]).all(): continue
                ems_list.append([x, z])
        # add the space started uppon the settled blcoks
        for blk_i in range(block_index):
            x, z = positions[blk_i]
            xx, zz = blocks[blk_i]
			# if container[x+xx, y, z]==0 and not [x+xx, y, z] in ems_list:
			# 	ems_list.append([x+xx, y, z])   # same with free space
            # if container[x, y+yy, z]==0 and not [x, y+yy, z] in ems_list:
            #     ems_list.append([x, y+yy, z])
            if z+zz < container_size[-1]:
                if container[x, z+zz]==0 and not [x, z+zz] in ems_list:
                    ems_list.append([x, z+zz])

        # search postions in each ems
        ems_num = len(ems_list)
        pos_ems = np.zeros((ems_num, block_dim)).astype(int)
        bbox_ems = [[0, 0]] * ems_num
        is_settle_ems = [False] * ems_num
        is_stable_ems = [False] * ems_num
        empty_ems = [empty_size] * ems_num
        compactness_ems  = [0.0] * ems_num
        pyramidality_ems = [0.0] * ems_num
        stability_ems    = [0.0] * ems_num
        X = int(container_size[0] - block_x + 1)
        for ems_index, ems in enumerate(ems_list):
            bbox_ems[ems_index] = bounding_box.copy()
            # using buttom-left strategy in each ems
            _z = int(ems[-1])
            for _x  in range( int(ems[0]), X):
                if is_settle_ems[ems_index]: break
                if (container[_x:_x+block_x, _z:_z+block_z] == 0).all():
                    if _z>0:
                        support = container[_x:_x+block_x, _z-1]
                        if not is_stable_2d(support, _x, block_x):
                            if reward_type.endswith('hard'):
                                continue
                        else:
                            is_stable_ems[ems_index] = True
                    else:
                        is_stable_ems[ems_index] = True
                    pos_ems[ems_index] = np.array([_x, _z])
                    is_settle_ems[ems_index] = True
            # calculate the compactness and pyramidality for each ems if the block has been settled
            if is_settle_ems[ems_index]:
                _x, _z = pos_ems[ems_index]
                if _x+block_x > bbox_ems[ems_index][0]: bbox_ems[ems_index][0] = _x+block_x
                if _z+block_z > bbox_ems[ems_index][-1]: bbox_ems[ems_index][-1] = _z+block_z
                # compactness
                # bbox_size = bbox_ems[ems_index][0] * bbox_ems[ems_index][-1]
                bbox_size = bbox_ems[ems_index][-1] * container_size[0]
                
                compactness_ems[ems_index] = valid_size / bbox_size
                # pyramidality
                under_space = container[_x:_x+block_x, 0:_z]
                empty_ems[ems_index] += np.sum(under_space==0)
                if 'P' in reward_type:
                    pyramidality_ems[ems_index] = valid_size / (empty_ems[ems_index] + valid_size)
                if 'S' in reward_type:
                    stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[ems_index])
                    stability_ems[ems_index] = stable_num / (block_index + 1)

        # if the block has not been settled
        if np.sum(is_settle_ems) == 0:
            valid_size -= block_x * block_z
            stable[block_index] = False
            continue

        # get the best ems
        ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
        best_ems_index = np.argmax(ratio_ems)
        while not is_settle_ems[best_ems_index]:
            ratio_ems[best_ems_index] = -1.0
            best_ems_index = np.argmax(ratio_ems)

        # update the positions, stable list, bin, and the container
        _x, _z = pos_ems[best_ems_index]
        positions[block_index] = pos_ems[best_ems_index]
        stable[block_index] = is_stable_ems[best_ems_index]
        bounding_box = bbox_ems[best_ems_index]
        empty_size = empty_ems[best_ems_index]
        container[_x:_x+block_x, _z:_z+block_z] = block_index + 1
        under_space = container[_x:_x+block_x, 0:_z]
        container[_x:_x+block_x, 0:_z][ under_space==0 ] = -1

        # update the level_free_space
        for zz in range(block_z):
            if _x in level_free_space[_z + zz]:
                level_free_space[_z + zz].remove(_x)
            if ( _x+block_x < container_size[0] ) and ( container[_x+block_x, _z+zz] == 0 ):
                level_free_space[_z + zz].append(_x + block_x)

    # calculate the compactness, pyramidality, stability, and the final ratio
    # box_size = bounding_box[-1] * bounding_box[0]
    box_size = bounding_box[-1] * container_size[0]

    packing_height = bounding_box[-1]

    
    stable_num = np.sum(stable)
    C = valid_size / box_size
    P = valid_size / (empty_size + valid_size)
    S = stable_num / blocks_num

    if 	 reward_type == 'comp':			ratio = C
    elif reward_type == 'soft':			ratio = C * S
    elif reward_type == 'hard':			ratio = C * S
    elif reward_type == 'pyrm':			ratio = C + P
    elif reward_type == 'pyrm-soft':	ratio = (C + P) * S
    elif reward_type == 'pyrm-hard':	ratio = (C + P) * S

    elif reward_type == 'mcs-soft':	ratio = (C + P) * S
    elif reward_type == 'mcs-hard':	ratio = (C + P) * S
    
    elif reward_type == 'pyrm-soft-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-soft-SUM':	ratio = 2*C + P + S

    elif reward_type == 'pyrm-hard-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-hard-SUM':	ratio = 2*C + P + S

    elif reward_type == 'C+P-soft':	        ratio = C + P
    elif reward_type == 'C+P+S-soft':	    ratio = C + P + S

    elif reward_type == 'CPS':          ratio = C * P * S
    else:								print('GREEDY................')

    # return ratio, valid_size, box_size, empty_size, stable_num
    scores = [valid_size, box_size, empty_size, stable_num, packing_height]
    return positions, container, stable, ratio, scores

def calc_positions_greedy(blocks, container_size, reward_type):

    block_dim = len(container_size)
    if block_dim == 3:
        return calc_positions_greedy_3d(blocks, container_size, reward_type)
    else:
        return calc_positions_greedy_2d(blocks, container_size, reward_type)
'''

# ============================


def check_feasibility_2d(blocks, blocks_num, initial_container_size, order, reward_type='pyrm-hard'):
    block_dim = len(initial_container_size)

    rotate_types = np.math.factorial(block_dim)

    real_order, rotate_state = calc_real_order(order, blocks_num)
    
    positions, container, stable = calc_positions_2d(blocks[:blocks_num], initial_container_size, True)

    container_x, container_z = initial_container_size
    feasibility = np.ones(blocks_num)

    rotate_permutation = []
    for p in itertools.permutations(range(block_dim)):
        rotate_permutation.append(p)

    for o_idx, o in enumerate(order):
        x, z = positions[real_order][o_idx]
        x = int(x)
        z = int(z)
        block_width, block_height = blocks[real_order][o_idx]
        
        if rotate_state[o_idx] == True:
            # check feasibility
            orgin_b = blocks[real_order][o_idx]
            rot_b = blocks[order][o_idx]
            real_o = int((o-real_order[o_idx]) / blocks_num)

            # TODO real index
            p = rotate_permutation[real_o]
            if (orgin_b[list(p)] == rot_b).all():
                if p[-1] == 0:
                    # get left and right container
                    z_mid = int(z + (orgin_b[-1]-1) / 2)
                    if x == 0:
                        left = False
                    else:
                        left = ((container[x-1, z_mid:container_z])==0).all()
                    
                    if x + block_width >= container_x:
                        right = False
                    else:
                        right = ((container[x+block_width, z_mid:container_z])==0).all()
                    
                    if left + right > 0:
                        feasibility[o_idx] = True
                    else:
                        feasibility[o_idx] = False
                else:
                    feasibility[o_idx] = True
        container[x:x+block_width, z:z+block_height] = 0
        container[x:x+block_width, 0:z][ np.where(container[x:x+block_width, 0:z] == -1)  ] = 0

    return feasibility

def check_feasibility_3d(blocks, blocks_num, initial_container_size, order, reward_type='pyrm-hard'):
    '''Return fesibility
    '''
    block_dim = len(initial_container_size)
    
    rotate_types = np.math.factorial(block_dim)

    real_order, rotate_state = calc_real_order(order, blocks_num)

    positions, container, stable = calc_positions(blocks[:blocks_num], initial_container_size, True)

    container_x, container_y, container_z = initial_container_size
    feasibility = np.ones(blocks_num)

    rotate_permutation = []
    for p in itertools.permutations(range(block_dim)):
        rotate_permutation.append(p)
    
    for o_idx, o in enumerate(order):
        x, y, z = positions[real_order][o_idx]
        x = int(x)
        y = int(y)
        z = int(z)
        [block_x, block_y, block_z] = blocks[real_order][o_idx]
        if rotate_state[o_idx] == True:
            # check feasibility
            orgin_b = blocks[real_order][o_idx]
            rot_b = blocks[order][o_idx]
            
            real_o = int((o-real_order[o_idx]) / blocks_num)

            p = rotate_permutation[real_o]
            if (orgin_b[list(p)] == rot_b).all():
                # get left and right container
                z_mid = z + int((block_z-1)/2)
                y_mid_1 = y + int((block_y-1)/2)
                y_mid_2 = y + int(block_y/2)
                if y_mid_1 == y_mid_2:
                    y_mid_2 += 1
                
                x_mid_1 = x + int((block_x-1)/2)
                x_mid_2 = x + int(block_x/2)
                if x_mid_1 == x_mid_2:
                    x_mid_2 += 1

                if p[-1] == 1:
                    if x == 0:
                        left = False
                    else:
                        left = ((container[x-1, y_mid_1:y_mid_2, z_mid:container_z])==0).all()           
                    if x + block_x >= container_x:
                        right = False
                    else:
                        right = ((container[x+block_x, y_mid_1:y_mid_2, z_mid:container_z])==0).all()
                    if left + right > 0:
                        feasibility[o_idx] = True
                    else:
                        feasibility[o_idx] = False
                elif p[-1] == 0:  
                    if y == 0:
                        forward = False
                    else:
                        forward = (container[x_mid_1:x_mid_2, y-1, z_mid:container_z]==0).all()
                    
                    if y + block_y >= container_y:
                        backward = False
                    else:
                        backward = (container[x_mid_1:x_mid_2, y+block_y, z_mid:container_z]==0).all()
                        
                    if forward + backward > 0:
                        feasibility[o_idx] = True
                    else:
                        feasibility[o_idx] = False
                elif p[-1] == 2:  
                    feasibility[o_idx] = True

        container[x:x+block_x, y:y+block_y, z:z+block_z] = 0
        container[x:x+block_x, y:y+block_y, 0:z][ np.where(container[x:x+block_x, y:y+block_y, 0:z] == -1) ] = 0


    return feasibility
    
def check_feasibility(blocks, blocks_num, initial_container_size, order, reward_type='pyrm-hard'):
    if len(initial_container_size) == 2:
        return check_feasibility_2d(blocks, blocks_num, initial_container_size, order, reward_type)
    else:
        return check_feasibility_3d(blocks, blocks_num, initial_container_size, order, reward_type)

# ===============================

# NOTE choose from Left-Bottom corners by greedy strategy
# Abandoned. Unefficient because of using level_free_space
# Please use calc_positions_lb_greedy()

def calc_one_position_greedy_2d(blocks, block_index, positions, container, reward_type, stable, heightmap,
                                bounding_box, valid_size, empty_size, level_free_space):
    """
    calculate the latest block's position in the container
    ---
    params:
    ---
        blocks: n x 2 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 2 array, positions of blocks, those after block_index are still [0, 0]
        container: width x height array, the container
        reward_type: string
            'C+P-lb-soft'
            'C+P-lb-hard'
            'C+P+S-lb-soft'
            'C+P+S-lb-hard'
        stable: n x 1 bool list, the blocks' stability state
        heightmap: 1 x width array, heightmap of the container
        bounding_box: 1 x 2 list, bounding_box of the packed blocks
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 2 array, updated positions
        container: width x height array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: 1 x width array, updated heightmap
        bounding_box: updated bounding_box
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """
    container_size = np.array(container.shape)
    # Initialize at the first block
    if (container==0).all():
        # states to calculate the scores
        bounding_box = [0, 0]
        valid_size = 0
        empty_size = 0
        # level free space (z, x), to stores the empty-maximal-spaces' corners
        level_free_space = [ [0] for i in range(container_size[-1]) ]
        # height map
        heightmap = np.zeros(container_size[0]).astype(int)

    block = blocks[block_index].astype(int)
    block_dim = len(block)
    block_x, block_z = block
    valid_size += block_x * block_z

    # empty-maximal-spaces list
    ems_list = []
    for z, free_space_x in enumerate(level_free_space):
        if z+block_z > container_size[-1]: break
        elif z>0 and (container[:, z-1]==0).all(): break
        for x in free_space_x:
            if x+block_x > container_size[0]: break
            if z>0 and x in level_free_space[z-1] and (container[x:, z]==container[x:, z-1]).all(): continue
            ems_list.append([x, z])
    # add the space started uppon the settled blcoks
    for blk_i in range(block_index):
        x, z = positions[blk_i]
        xx, zz = blocks[blk_i]
        # if container[x+xx, y, z]==0 and not [x+xx, y, z] in ems_list:
        # 	ems_list.append([x+xx, y, z])   # same with free space
        # if container[x, y+yy, z]==0 and not [x, y+yy, z] in ems_list:
        #     ems_list.append([x, y+yy, z])
        if z+zz < container.shape[-1]:
            if container[x, z+zz]==0 and not [x, z+zz] in ems_list:
                ems_list.append([x, z+zz])

    # search positions in each ems
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    bbox_ems = [[0, 0]] * ems_num
    is_settle_ems = [False] * ems_num
    is_stable_ems = [False] * ems_num
    empty_ems = [empty_size] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    X = int(container_size[0] - block_x + 1)
    for ems_index, ems in enumerate(ems_list):
        bbox_ems[ems_index] = bounding_box.copy()
        # using buttom-left strategy in each ems
        _z = int(ems[-1])
        for _x  in range( int(ems[0]), X):
            if is_settle_ems[ems_index]: break
            if (container[_x:_x+block_x, _z:_z+block_z] == 0).all():
                if _z>0:
                    support = container[_x:_x+block_x, _z-1]
                    if not is_stable_2d(support, _x, block_x):
                        if reward_type.endswith('hard'):
                            continue
                    else:
                        is_stable_ems[ems_index] = True
                else:
                    is_stable_ems[ems_index] = True
                pos_ems[ems_index] = np.array([_x, _z])
                is_settle_ems[ems_index] = True
        # calculate the compactness and pyramidality for each ems if the block has been settled
        if is_settle_ems[ems_index]:
            _x, _z = pos_ems[ems_index]
            if _x+block_x > bbox_ems[ems_index][0]: bbox_ems[ems_index][0] = _x+block_x
            if _z+block_z > bbox_ems[ems_index][-1]: bbox_ems[ems_index][-1] = _z+block_z
            # compactness
            # bbox_size = bbox_ems[ems_index][0] * bbox_ems[ems_index][-1]
            bbox_size = bbox_ems[ems_index][-1] * container_size[0]
            
            compactness_ems[ems_index] = valid_size / bbox_size
            # pyramidality
            under_space = container[_x:_x+block_x, 0:_z]
            empty_ems[ems_index] += np.sum(under_space==0)
            if 'P' in reward_type:
                pyramidality_ems[ems_index] = valid_size / (empty_ems[ems_index] + valid_size)
            if 'S' in reward_type:
                stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[ems_index])
                stability_ems[ems_index] = stable_num / (block_index + 1)
        
    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_z
        stable[block_index] = False
        return positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space

    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the positions, stable list, bin, and the container
    _x, _z = pos_ems[best_ems_index]
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    bounding_box = bbox_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]
    container[_x:_x+block_x, _z:_z+block_z] = block_index + 1
    under_space = container[_x:_x+block_x, 0:_z]
    container[_x:_x+block_x, 0:_z][ under_space==0 ] = -1

    # update the level_free_space
    for zz in range(block_z):
        if _x in level_free_space[_z + zz]:
            level_free_space[_z + zz].remove(_x)
        if ( _x+block_x < container_size[0] ) and ( container[_x+block_x, _z+zz] == 0 ):
            level_free_space[_z + zz].append(_x + block_x)

    # update the height map
    heightmap[_x:_x+block_x] = _z+block_z

    return positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space

def calc_one_position_greedy_3d(blocks, block_index, positions, container, reward_type, stable, heightmap,
                                bounding_box, valid_size, empty_size, level_free_space):
    """
    calculate the latest block's position in the container
    ---
    params:
    ---
        blocks: n x 3 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 3 array, positions of blocks, those after block_index are still [0, 0]
        container: 3-dimension array, the container
        reward_type: string
            'C+P-lb-soft'
            'C+P-lb-hard'
            'C+P+S-lb-soft'
            'C+P+S-lb-hard'
        stable: n x 1 bool list, the blocks' stability state
        heightmap: width x depth array, heightmap of the container
        bounding_box: 1 x 3 list, bounding_box of the packed blocks
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 3 array, updated positions
        container: 3-dimension array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: width x depth array, updated heightmap
        bounding_box: updated bounding_box
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """
    container_size = np.array(container.shape)
    # Initialize at the first block
    if (container==0).all():
        # states to calculate the scores
        bounding_box = [0, 0, 0]
        valid_size = 0
        empty_size = 0
        # level free space (z, y, x), to stores the empty-maximal-spaces' corners
        level_free_space = [ [ [0] for i in range(container_size[1]) ] for j in range(container_size[2]) ]
        # height map
        heightmap = np.zeros(container_size[:2]).astype(int)

    block = blocks[block_index].astype(int)
    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # empty-maximal-spaces list
    ems_list = []
    for z, free_space_x in enumerate(level_free_space):
        if z+block_z > container_size[2]: break
        elif z>0 and (container[:, :, z-1] == 0).all(): break
        for y, free_space in enumerate(free_space_x):
            if y+block_y > container_size[1]: break
            # elif y>0 and free_space_x[y-1]==free_space: continue
            elif y>0 and free_space_x[y-1]==[0]: continue

            for x in free_space:
                if x+block_x > container_size[0]: break
                if y>0 and x in free_space_x[y-1] and (container[x:, y, z]==container[x:, y-1, z]).all(): continue
                if z>0 and x in level_free_space[z-1][y] and (container[x:, y:, z]==container[x:, y:, z-1]).all(): continue
                ems_list.append([x, y, z])
    # add the space started uppon the settled blcoks
    for blk_i in range(block_index):
        x, y, z = positions[blk_i]
        xx, yy, zz = blocks[blk_i]
        # if container[x+xx, y, z]==0 and not [x+xx, y, z] in ems_list:
        # 	ems_list.append([x+xx, y, z])   # same with free space
        if y+yy < container.shape[1]:
            if  container[x, y+yy, z]==0 and not [x, y+yy, z] in ems_list:
                ems_list.append([x, y+yy, z])
        if z+zz < container.shape[-1]:
            if container[x, y, z+zz]==0 and not [x, y, z+zz] in ems_list:
                ems_list.append([x, y, z+zz])
    
    # search postions in each ems
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, 3)).astype(int)
    bbox_ems = [[0, 0, 0]] * ems_num
    is_settle_ems = [False] * ems_num
    is_stable_ems = [False] * ems_num
    empty_ems = [empty_size] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        bbox_ems[ems_index] = bounding_box.copy()
        # using buttom-left strategy in each ems
        _z = int(ems[2])
        for _x, _y in itertools.product( range( int(ems[0]), X), range( int(ems[1]), Y) ):
            if is_settle_ems[ems_index]: break
            if (container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] == 0).all():
                if not is_stable(block, np.array([_x, _y, _z]), container):
                    if reward_type.endswith('hard'):
                        continue
                else:
                    is_stable_ems[ems_index] = True
                pos_ems[ems_index] = np.array([_x, _y, _z])
                is_settle_ems[ems_index] = True
        # calculate the compactness and pyramidality for each ems if the block has been settled
        if is_settle_ems[ems_index]:
            _x, _y, _z = pos_ems[ems_index]
            if _x+block_x > bbox_ems[ems_index][0]: bbox_ems[ems_index][0] = _x+block_x
            if _y+block_y > bbox_ems[ems_index][1]: bbox_ems[ems_index][1] = _y+block_y
            if _z+block_z > bbox_ems[ems_index][2]: bbox_ems[ems_index][2] = _z+block_z
            # compactness
            # bbox_size = bbox_ems[ems_index][0] * bbox_ems[ems_index][1] * bbox_ems[ems_index][2]

            bbox_size = bbox_ems[ems_index][-1] * container_size[0] * container_size[1]
            
            compactness_ems[ems_index] = valid_size / bbox_size
            # pyramidality
            under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
            empty_ems[ems_index] += np.sum(under_space==0)
            if 'P' in reward_type:
                pyramidality_ems[ems_index] = valid_size / (empty_ems[ems_index] + valid_size)
            if 'S' in reward_type:
                stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[ems_index])
                stability_ems[ems_index] = stable_num / (block_index + 1)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space
    
    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the positions, stable list, bin, and the container
    _x, _y, _z = pos_ems[best_ems_index]
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    bounding_box = bbox_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]
    container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
    under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
    container[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space==0 ] = -1

    # update the level_free_space
    for zz, yy in itertools.product( range(block_z), range(block_y) ):
        if _x in level_free_space[_z + zz][_y + yy]:
            level_free_space[_z + zz][_y + yy].remove(_x)
        if ( _x+block_x < container_size[0] ) and ( container[_x+block_x, _y+yy, _z+zz] == 0 ):
            level_free_space[_z + zz][_y + yy].append(_x + block_x)

    # update the height map
    heightmap[_x:_x+block_x, _y:_y+block_y] = _z + block_z

    return positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space

def calc_one_position_greedy(blocks, block_index, positions, container, reward_type, stable, heightmap,
                                bounding_box, valid_size, empty_size, level_free_space):
    """
    calculate the latest block's position in the container
    ---
    params:
    ---
        blocks: n x 2/3 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 2/3 array, positions of blocks, those after block_index are still [0, 0]
        container: width (x depth) x height array, the container
        reward_type: string
            'C+P-lb-soft'
            'C+P-lb-hard'
            'C+P+S-lb-soft'
            'C+P+S-lb-hard'
        stable: n x 1 bool list, the blocks' stability state
        heightmap: 1 x width array, heightmap of the container
        bounding_box: 1 x 2 list, bounding_box of the packed blocks
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 2 array, updated positions
        container: width x height array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: 1 x width array, updated heightmap
        bounding_box: updated bounding_box
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """
    block_dim = len(container.shape)
    if block_dim == 3:
        return calc_one_position_greedy_3d(blocks, block_index, positions, container, reward_type, stable, heightmap,
                                bounding_box, valid_size, empty_size, level_free_space)
    else:
        return calc_one_position_greedy_2d(blocks, block_index, positions, container, reward_type, stable, heightmap,
                                bounding_box, valid_size, empty_size, level_free_space)

def calc_positions_greedy(blocks, container_size, reward_type):
    '''
    calculate the positions
    ---
    params:
    ---
        blocks: n x 2/3 array, blocks with an order
        container_size: 1 x 2/3 array, size of the container
        reward_type: string
            'C+P-lb-soft'
            'C+P-lb-hard'
            'C+P+S-lb-soft'
            'C+P+S-lb-hard'
    return:
    ---
        positions: n x 2/3 array, positions of the blocks
        container: a 2/3-dimension array, the final state of the container
        stable: n x 1 bool list, each element indicates whether a block is placed(hard)/stable(soft) or not
        ratio: float, C / C*S1 / C*S2 / C+P / (C+P)*S1 / (C+P)*S2, calculated by the following scores
        scores: 5 integer numbers: valid_size, box_size, empty_size, stable_num and packing_height
    '''
    # Initialize
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num
    heightmap = np.zeros(container_size[:-1]).astype(int)
    bounding_box = [0, 0]
    valid_size = 0
    empty_size = 0
    if block_dim == 2:
        level_free_space = [ [0] for i in range(container_size[-1]) ]
    elif block_dim == 3:
        level_free_space = [ [ [0] for i in range(container_size[1]) ] for j in range(container_size[2]) ]

    for block_index in range(blocks_num):
        positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space = \
            calc_one_position_greedy(blocks, block_index, positions, container, reward_type, stable, heightmap,
                                bounding_box, valid_size, empty_size, level_free_space)

    stable_num = np.sum(stable)
    if block_dim == 2:
        box_size = np.max(heightmap) * container_size[0]
    elif block_dim == 3:
        box_size = np.max(heightmap) * container_size[0] * container_size[1]

    if blocks_num == 0:
        C = 0
        P = 0
        S = 0    
    else:
        C = valid_size / box_size
        P = valid_size / (empty_size + valid_size)
        S = stable_num / blocks_num

    if   reward_type == 'C+P-lb-soft':      ratio = C + P + S
    elif reward_type == 'C+P-lb-hard':      ratio = C + P + S
    elif reward_type == 'C+P+S-lb-soft':    ratio = C + P + S
    elif reward_type == 'C+P+S-lb-hard':    ratio = C + P + S
    else:                                   print("GREEDY................")

    scores = [valid_size, box_size, empty_size, stable_num, np.max(heightmap)]
    return positions, container, stable, ratio, scores

# ===============================

# NOTE choose from Left-Bottom corners by greedy strategy

def calc_one_position_lb_greedy_2d(block, block_index, container_size, reward_type, 
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy in 2D cases
    ---
    params:
    ---
        static params:
            block: int * 2 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 2 array, size of the container
            reward_type: string, options:
                'C+P-lb-soft'
                'C+P-lb-hard'
                'C+P+S-lb-soft'
                'C+P+S-lb-hard'
        dynamic params:
            container: width * height array, the container state
            positions: int * 2 array, coordinates of the blocks, [0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width * 1 array, heightmap of the container
            valid_size: int, sum of the packed blocks' size
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width * height array, updated container
        positions: int * 2 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width * 1 array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
    """
    block_dim = len(block)
    block_x, block_z = block
    valid_size += block_x * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    hm_diff = heightmap.copy()
    hm_diff = np.insert(hm_diff, 0, hm_diff[0])
    hm_diff = np.delete(hm_diff, len(hm_diff)-1)
    hm_diff = heightmap - hm_diff
    # get the x coordinates of all left-bottom corners
    ems_x_list = np.nonzero(hm_diff)
    ems_x_list = np.insert(ems_x_list, 0, 0)
    # get ems_list
    for x in ems_x_list:
        if x+block_x > container_size[0]: break
        z = np.max( heightmap[x:x+block_x] )
        ems_list.append( [x, z] )
    # firt consider the most bottom, then left
    def bottom_first(pos): return pos[1]
    ems_list.sort(key=bottom_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask  = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _z):
        # check if the pos visited
        if [_x, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _z-1]==0).all(): return
        visited.append([_x, _z])
        # if (container[_x:_x+block_x, _z:_z+block_z] == 0).all():
        if (container[_x:_x+block_x, _z] == 0).all():
            if _z > 0:
                support = container[_x:_x+block_x, _z-1]
                if not is_stable_2d(support, _x, block_x):
                    if reward_type.endswith('hard'):
                        return
                else:
                    is_stable_ems[index] = True
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _z])
            heightmap_ems[index][_x:_x+block_x] = _z + block_z
            is_settle_ems[index] = True

    # calculate socres
    def calc_C_P_S(index):
        _x, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        # if _z+block_x > height: height = _z+block_z
        bbox_size = height * container_size[0]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, 0:_z]
        under_space_mask[index] = under_space==0
        empty_ems[index] += np.sum(under_space_mask[index])
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        # stability
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    for ems_index, ems in enumerate(ems_list):
        # using buttom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        _z = int(ems[-1])
        for _x  in range( int(ems[0]), X ):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _z)
        if is_settle_ems[ems_index]: 
            calc_C_P_S(ems_index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _z = pos_ems[best_ems_index]
    container[_x:_x+block_x, _z:_z+block_z] = block_index + 1
    container[_x:_x+block_x, 0:_z][ under_space_mask[best_ems_index] ] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size

def calc_one_position_lb_greedy_3d(block, block_index, container_size, reward_type, 
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy in 2D cases
    ---
    params:
    ---
        static params:
            block: int * 3 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 3 array, size of the container
            reward_type: string, options:
                'C+P-lb-soft'
                'C+P-lb-hard'
                'C+P+S-lb-soft'
                'C+P+S-lb-hard'
        dynamic params:
            container: width * length * height array, the container state
            positions: int * 3 array, coordinates of the blocks, [0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width * length array, heightmap of the container
            valid_size: int, sum of the packed blocks' size
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width * length * height array, updated container
        positions: int * 3 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width * length array, updated heightmap
        valid_size: int, updated valid_size
        empty_size: int, updated empty_size
    """
    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    # x coordinate
    hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
    hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
    hm_diff_x = heightmap - hm_diff_x
    # y coordinate
    hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
    hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
    hm_diff_y = heightmap - hm_diff_y

    # get the xy coordinates of all left-deep-bottom corners
    ems_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
    ems_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()
    ems_xy_list = []
    ems_xy_list.append([0,0])
    for xy in ems_x_list:
        x, y = xy
        if y!=0 and [x, y-1] in ems_x_list:
            if heightmap[x, y] == heightmap[x, y-1] and \
                hm_diff_x[x, y] == hm_diff_x[x, y-1]:
                continue
        ems_xy_list.append(xy)
    for xy in ems_y_list:
        x, y = xy
        if x!=0 and [x-1, y] in ems_y_list:
            if heightmap[x, y] == heightmap[x-1, y] and \
                hm_diff_x[x, y] == hm_diff_x[x-1, y]:
                continue
        if xy not in ems_xy_list:
            ems_xy_list.append(xy)

    # sort by y coordinate, then x
    def y_first(pos): return pos[1]
    ems_xy_list.sort(key=y_first, reverse=False)

    # get ems_list
    for xy in ems_xy_list:
        x, y = xy
        if x+block_x > container_size[0] or \
            y+block_y > container_size[1]: continue
        z = np.max( heightmap[x:x+block_x, y:y+block_y] )
        ems_list.append( [ x, y, z ] )
    
    # firt consider the most bottom, sort by z coordinate, then y last x
    def z_first(pos): return pos[2]
    ems_list.sort(key=z_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask  = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _y, _z):
        # check if the pos visited
        if [_x, _y, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _y:_y+block_y, _z-1]==0).all(): return
        visited.append([_x, _y, _z])
        if (container[_x:_x+block_x, _y:_y+block_y, _z] == 0).all():
            if not is_stable(block, np.array([_x, _y, _z]), container):
                if reward_type.endswith('hard'):
                    return
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _y, _z])
            heightmap_ems[index][_x:_x+block_x, _y:_y+block_y] = _z + block_z
            is_settle_ems[index] = True

    # calculate socres
    def calc_C_P_S(index):
        _x, _y, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        bbox_size = height * container_size[0] *container_size[1]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
        under_space_mask[index] = under_space==0
        empty_ems[index] += np.sum(under_space_mask[index])
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        # stability
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        # using buttom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        X0, Y0, _z = ems
        for _x, _y  in itertools.product( range(X0, X), range(Y0, Y) ):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _y, _z)
        if is_settle_ems[ems_index]: calc_C_P_S(ems_index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _y, _z = pos_ems[best_ems_index]
    container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
    container[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space_mask[best_ems_index] ] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size

def calc_one_position_lb_greedy(block, block_index, container_size, reward_type, 
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy
    ---
    params:
    ---
        static params:
            block: int * 2/3 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 2/3 array, size of the container
            reward_type: string, options:
                'C+P-lb-soft'
                'C+P-lb-hard'
                'C+P+S-lb-soft'
                'C+P+S-lb-hard'
        dynamic params:
            container: width (* depth) * height array, the container state
            positions: int * 2/3 array, coordinates of the blocks, [0, 0]/[0, 0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width (* depth) * 1 array, heightmap of the container
            valid_size: int, sum of the packed blocks' volume
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width (* depth) * height array, updated container
        positions: n * 2/3 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width (* depth) * 1 array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
    """
    block_dim = len(container_size)
    if block_dim == 2:
        return calc_one_position_lb_greedy_2d(block, block_index, container_size, reward_type, 
                                            container, positions, stable, heightmap, valid_size, empty_size)
    elif block_dim == 3:
        return calc_one_position_lb_greedy_3d(block, block_index, container_size, reward_type, 
                                            container, positions, stable, heightmap, valid_size, empty_size)

def calc_positions_lb_greedy(blocks, container_size, reward_type):
    '''
    calculate the positions to pack a group of blocks into a container by lb-greedy
    ---
    params:
    ---
        blocks: n x 2/3 array, blocks with an order
        container_size: 1 x 2/3 array, size of the container
        reward_type: string
            'C+P-lb-soft'
            'C+P-lb-hard'
            'C+P+S-lb-soft'
            'C+P+S-lb-hard'
    return:
    ---
        positions: int x 2/3 array, packing positions of the blocks
        container: width (* depth) * height array, the final state of the container
        stable: n x 1 bool list, each element indicates whether a block is placed(hard)/stable(soft) or not
        ratio: float, C / C*S / C+P / (C+P)*S / C+P+S, calculated by the following scores
        scores: 5 integer numbers: valid_size, box_size, empty_size, stable_num and packing_height
    '''
    # Initialize
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num
    heightmap = np.zeros(container_size[:-1]).astype(int)
    valid_size = 0
    empty_size = 0

    # try to use cuda to accelerate but fail
    # container_tensor = torch.from_numpy(container)
    # container_tensor = container_tensor.cuda().detach()

    for block_index in range(blocks_num):
        container, positions, stable, heightmap, valid_size, empty_size = \
            calc_one_position_lb_greedy(blocks[block_index], block_index, container_size, reward_type, 
                                        container, positions, stable, heightmap, valid_size, empty_size)

    stable_num = np.sum(stable)
    if block_dim == 2:      box_size = np.max(heightmap) * container_size[0]
    elif block_dim == 3:    box_size = np.max(heightmap) * container_size[0] * container_size[1]

    C = valid_size / box_size
    P = valid_size / (empty_size + valid_size)
    S = stable_num / blocks_num

    if   reward_type == 'C+P-lb-soft':      ratio = C + P + S
    elif reward_type == 'C+P-lb-hard':      ratio = C + P + S
    elif reward_type == 'C+P+S-lb-soft':    ratio = C + P + S
    elif reward_type == 'C+P+S-lb-hard':    ratio = C + P + S
    else:                                   print("Unknown reward type for lb_greedy packing")

    scores = [valid_size, box_size, empty_size, stable_num, np.max(heightmap)]
    return positions, container, stable, ratio, scores

# ===============================

# NOTE consider multi-corners of an EMS
# Able to use Maximize-Accessible-Convex-Space (MACS) strategy

def calc_one_position_mcs_2d(blocks, block_index, positions, container, reward_type, stable,
                             heightmap, valid_size, empty_size, level_free_space):
    """
    calculate the latest block's position in the container
    by the policy of maximizing remaining empty spaces
    ---
    params:
    ---
        blocks: n x 2 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 2 array, positions of blocks, those after block_index are still [0, 0]
        container: 2-dimension array, the container
        reward_type: string
            'mcs-soft'
            'mcs-hard'
            'C+P-mul-soft'
            'C+P-mul-hard'
            'C+P-mcs-soft'
            'C+P-mcs-hard'
            'C+P+S-mul-soft'
            'C+P+S-mul-hard'
            'C+P+S-mcs-soft'
            'C+P+S-mcs-hard'
        stable: n x 1 bool list, the blocks' stability state
        heightmap: width x 1 array, heightmap of the container
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 2 array, updated positions
        container: 2-dimension array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: width x 1 array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """
    container_size = np.array(container.shape)
    # Initialize at the first block
    # if block_index == 0:
    #     blocks_num = len(blocks)
    #     block_dim = len(blocks[0])
    #     positions = np.zeros((blocks_num, block_dim)).astype(int)
    #     stable = [False] * blocks_num
    #     valid_size = 0
    #     empty_size = 0
    #     # level free space (z, y, x), storing each row's empty spaces' two corners
    #     level_free_space = [ [ 0, container_size[0]-1 ] \
    #                                 for i in range(container_size[-1]) ]
    #     # height map, width x 1
    #     heightmap = np.zeros(container_size[:-1]).astype(int)

    # initialize the block
    block = blocks[block_index].astype(int)
    block_dim = len(block)
    block_x, block_z = block
    valid_size += block_x * block_z

    # get empty-maximal-spaces list from level_free_space
    # each ems represented as two corners on its bottom
    ems_list = []
    for z, free_space in enumerate(level_free_space):
        if z+block_z > container_size[-1]: break
        elif z>0 and level_free_space[z-1]==free_space: continue
        # split into [x1, x2] forms
        spaces = [free_space[i:i+2] for i in range(0, len(free_space), 2)]
        for space in spaces:
            x1, x2 = space
            if x1+block_x > container_size[0]: break
            if z>0 and x1 in level_free_space[z-1]:
                idx = level_free_space[z-1].index(x1) + 1
                if idx % 2 == 1 and x2 == level_free_space[z-1][idx]: continue
            ems_list.append([x1, z, x2, z])
    # add the space starting next to the setteld blocks
    for blk_i in range(block_index):
        x, z = positions[blk_i]
        xx, zz = blocks[blk_i]
        # upon z-axis(top)
        if z+zz < container_size[-1]:
            # full
            if (container[x:x+xx, z+zz] == 0).all():
                if not [x, z+zz, x+xx-1, z+zz] in ems_list:
                    ems_list.append([x, z+zz, x+xx-1, z+zz])
            # part
            else:
                # left
                if container[x, z+zz] == 0:
                    if x>0 and container[x-1, z+zz] == 0:
                        for x2 in range(x, x+xx):
                            if x2 == container_size[0] - 1: break
                            if container[x2+1, z+zz] != 0: break
                        ems_list.append([x, z+zz, x2, z+zz])
                # right
                if container[x+xx-1, z+zz] == 0:
                    if x+xx < container_size[0] and container[x+xx, z+zz] == 0:
                        for x1 in reversed(range(x, x+xx)):
                            if x1 == 0: break
                            if container[x1-1, z+zz] != 0: break
                        ems_list.append([x1, z+zz, x+xx-1, z+zz])
    

    # varients to store results of searching ems corners
    ems_num = len(ems_list) * 2
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    empty_ems = [empty_size] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0]   * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _z):
        # check if the pos visited
        if [_x, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _z-1]==0).all(): return
        visited.append([_x, _z])
        if (container[_x:_x+block_x, _z:_z+block_z] == 0).all():
            if _z > 0:
                support = container[_x:_x+block_x, _z-1]
                if not is_stable_2d(support, _x, block_x):
                    if reward_type.endswith('hard'):
                        return
                else:
                    is_stable_ems[index] = True
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _z])
            heightmap_ems[index][_x:_x+block_x] = _z + block_z
            is_settle_ems[index] = True

    def calc_C_P_S(index):
        _x, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        if _z+block_x > height: height = _z+block_z
        bbox_size = height * container_size[0]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, 0:_z]
        empty_ems[index] += np.sum(under_space==0)
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    def update_level_free_space(pos):
        _x, _z = pos
        xx = _x + block_x - 1
        # updated_level_free_space = level_free_space.copy()
        updated_level_free_space = copy.deepcopy(level_free_space)
        # _z ~ _z+block_z
        for zz in range(_z, _z+block_z):
            free_space = updated_level_free_space[zz]
            if _x in free_space:
                idx = free_space.index(_x)
                # odd
                if (idx+1) % 2 == 1:
                    if xx in free_space:
                        if block_x == 1:
                            if free_space[idx+1] == _x:
                                free_space.remove(_x)
                                free_space.remove(_x)
                            else:
                                free_space[idx] = _x + 1
                        else:
                            free_space.remove(_x)
                            free_space.remove(xx)
                    else:
                        free_space[idx] = xx + 1
                # even
                else:
                    free_space[idx] = _x - 1
            else:
                if xx in free_space:
                    idx = free_space.index(xx)
                    free_space[idx] = _x - 1
                else:
                    free_space.append(_x - 1)
                    free_space.append(xx + 1)
                    free_space.sort()
        # 0 ~ _z
        for zz in range(0, _z):
            free_space = updated_level_free_space[zz]
            spaces = [free_space[i:i+2] for i in range(0, len(free_space), 2)]
            for space in spaces:
                x1, x2 = space
                if x1 == x2:
                    if x1 >= _x and x1 <= xx: 
                        free_space.remove(x1)
                        free_space.remove(x1)
                elif block_x == 1:  # _x == xx
                    if _x == x1: free_space[free_space.index(x1)] = _x + 1
                    elif _x == x2: free_space[free_space.index(x2)] = xx - 1
                elif _x <= x1 and x2 <= xx:
                    free_space.remove(x1)
                    free_space.remove(x2)
                elif _x <= x1 and x1 <= xx: free_space[free_space.index(x1)] = xx + 1
                elif _x <= x2 and x2 <= xx: free_space[free_space.index(x2)] = _x - 1
        return updated_level_free_space

    def update_container(ctn, pos):
        _x, _z = pos
        ctn[_x:_x+block_x, _z:_z+block_z] = block_index + 1
        under_space = ctn[_x:_x+block_x, 0:_z]
        ctn[_x:_x+block_x, 0:_z][ under_space==0 ] = -1

    def calc_maximal_usable_spaces(lfs, H):
        score = 0
        for h in range(H):
            level_max_empty = 0
            free_space = lfs[h]
            spaces = [free_space[i:i+2] for i in range(0, len(free_space), 2)]
            for space in spaces:
                length = space[1] - space[0]
                if length > level_max_empty:
                    level_max_empty = length
            score += level_max_empty
        return score

    # search positions in each ems, from 2 corners of each
    X = int(container_size[0] - block_x + 1)
    for ems_index, ems in enumerate(ems_list):
        X1, Z, X2, _ = ems
        # search from the 2 corners of the ems
        # left corner
        if X1 < X:
            index = ems_index * 2 + 0
            heightmap_ems[index] = heightmap.copy()
            for _x in range(X1, X):
                if is_settle_ems[index]: break
                check_position(index, _x, Z)
            if is_settle_ems[index]: calc_C_P_S(index)
        # right corner
        if X2 - block_x + 2 > 0:
            index = ems_index * 2 + 1
            heightmap_ems[index] = heightmap.copy()
            for _x in reversed(range(0, X2 - block_x + 2)):
                if is_settle_ems[index]: break
                check_position(index, _x, Z)
            if is_settle_ems[index]: calc_C_P_S(index)
    
    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_z
        stable[block_index] = False
        return positions, container, stable, heightmap, valid_size, empty_size, level_free_space

    # get the best ems
    if reward_type.startswith('mcs'):
        ratio_ems = [0.0] * ems_num
    else:
        ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_score = np.max(ratio_ems)
    best_ems_indexes = [i for i,s in enumerate(ratio_ems) if s==best_score]
    # updated_containers = [0] * len(best_ems_indexes)
    updated_lfs_ems = [0] * len(best_ems_indexes)
    maximal_usable_spaces_ems = [0] * len(best_ems_indexes)
    if len(best_ems_indexes) > 1 and 'mcs' in reward_type:
        max_height = np.max(heightmap_ems)
        for i, index in enumerate(best_ems_indexes):
            if is_settle_ems[index]:
                # updated_containers[i] = container.copy()
                # update_container(updated_containers[i], pos_ems[index])
                updated_lfs_ems[i] = update_level_free_space(pos_ems[index])
                maximal_usable_spaces_ems[i] = calc_maximal_usable_spaces(updated_lfs_ems[i], max_height)
        best_free_space_index = np.argmax(maximal_usable_spaces_ems)
        best_ems_index = best_ems_indexes[best_free_space_index]
        while not is_settle_ems[best_ems_index]:
            maximal_usable_spaces_ems[best_free_space_index] = -1
            best_free_space_index = np.argmax(maximal_usable_spaces_ems)
            best_ems_index = best_ems_indexes[best_free_space_index]
    else:
        best_ems_index = best_ems_indexes[0]
        while not is_settle_ems[best_ems_index]:
            best_ems_indexes.remove(best_ems_indexes[0])
            best_ems_index = best_ems_indexes[0]

    # update the positions, stable list, the container, and the level_free_space
    _x, _z = pos_ems[best_ems_index]
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]
    update_container(container, positions[block_index])
    level_free_space = update_level_free_space(positions[block_index])

    # update the height map
    heightmap[_x:_x+block_x] = _z + block_z

    return positions, container, stable, heightmap, valid_size, empty_size, level_free_space

def calc_one_position_mcs_3d(blocks, block_index, positions, container, reward_type, stable,
                             heightmap, valid_size, empty_size, level_free_space):
    """
    calculate the latest block's position in the container
    by the policy of maximizing remaining empty spaces
    ---
    params:
    ---
        blocks: n x 3 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 3 array, positions of blocks, those after block_index are still [0, 0]
        container: 3-dimension array, the container
        reward_type: string
            'mcs-soft'
            'mcs-hard'
            'C+P-mul-soft'
            'C+P-mul-hard'
            'C+P-mcs-soft'
            'C+P-mcs-hard'
            'C+P+S-mul-soft'
            'C+P+S-mul-hard'
            'C+P+S-mcs-soft'
            'C+P+S-mcs-hard'
        stable: n x 1 bool list, the blocks' stability state
        heightmap: width x depth array, heightmap of the container
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 3 array, updated positions
        container: 3-dimension array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: width x depth array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """
    container_size = np.array(container.shape)
    # # Initialize at the first block
    # if block_index == 0:
    #     blocks_num = len(blocks)
    #     block_dim = len(blocks[0])
    #     positions = np.zeros((blocks_num, block_dim)).astype(int)
    #     stable = [False] * blocks_num
    #     valid_size = 0
    #     empty_size = 0
    #     # level free space (z, y, x), storing each row's empty spaces' two corners
    #     level_free_space = [ [ [ 0, container_size[0]-1 ] \
    #                                 for i in range(container_size[1]) ] \
    #                                     for j in range(container_size[2]) ]
    #     # height map, X*Y
    #     heightmap = np.zeros(container_size[:-1]).astype(int)
        
    # initialize the block
    block = blocks[block_index].astype(int)
    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from level_free_space
    # each ems represented as two corners on its bottom
    ems_list = []
    for z, free_space_x in enumerate(level_free_space):
        if z+block_z > container_size[-1]: break
        elif z>0 and level_free_space[z-1]==free_space_x: continue
        for y, free_space in enumerate(free_space_x):
            if y+block_y > container_size[1]: break
            elif y>0 and free_space_x[y-1]==free_space: continue
            # split into [x1, x2] forms
            spaces = [free_space[i:i+2] for i in range(0, len(free_space), 2)]
            for space in spaces:
                x1, x2 = space
                if x1+block_x > container_size[0]: break
                if y>0 and x1 in free_space_x[y-1]:
                    idx = free_space_x[y-1].index(x1) + 1
                    if idx % 2 == 1 and x2 == free_space_x[y-1][idx]: continue
                if z>0 and x1 in level_free_space[z-1][y]:
                    idx = level_free_space[z-1][y].index(x1) + 1
                    if idx % 2 == 1 and x2 == level_free_space[z-1][y][idx]: continue
                xspace = True
                for y2 in range(y, container_size[1]):
                    if y2 == container_size[1] - 1: break
                    if not (container[x1:x2+1, y2+1, z] == 0).all(): break
                    if xspace and not (x1 in free_space_x[y2+1] and x2 in free_space_x[y2+1]):
                        # next to settled blocks along x-axis
                        xspace = False
                        ems_list.append([x1, y, z, x2, y2, z])
                ems_list.append([x1, y, z, x2, y2, z])

    # add the space starting next to the settled blocks
    for blk_i in range(block_index):
        x, y, z = positions[blk_i]
        xx, yy, zz = blocks[blk_i]
        # upon along y-axis
        if y+yy < container_size[1]:
            # full
            if (container[x:x+xx, y+yy, z] == 0).all():
                if (x>0 and container[x-1, y+yy, z] == 0) or \
                    (x+xx<container_size[0] and container[x+xx, y+yy, z] == 0):
                    for y2 in range(y+yy, container_size[1]):
                        if y2 == container_size[1] - 1: break
                        if not (container[x:x+xx, y2+1, z] == 0).all(): break
                    ems_list.append([x, y+yy, z, x+xx-1, y2, z])
            # part
            else:
                # left
                if container[x, y+yy, z] == 0:
                    if x>0 and container[x-1, y+yy, z] == 0:
                        for x2 in range(x, x+xx):
                            if x2 == container_size[0] - 1: break
                            if container[x2+1, y+yy, z] != 0: break
                        for y2 in range(y+yy, container_size[1]):
                            if y2 == container_size[1] - 1: break
                            if not (container[x1:x2+1, y2+1, z] == 0).all(): break
                        ems_list.append([x, y+yy, z, x2, y2, z])
                # right
                if container[x+xx-1, y+yy, z] == 0:
                    if x+xx < container_size[0] and container[x+xx, y+yy, z] == 0:
                        for x1 in reversed(range(x, x+xx)):
                            if x1 == 0: break
                            if container[x1-1, y+yy, z] != 0: break
                        for y2 in range(y+yy, container_size[1]):
                            if y2 == container_size[1] - 1: break
                            if not (container[x1:x+xx, y2+1, z] == 0).all(): break
                        ems_list.append([x1, y+yy, z, x+xx-1, y2, z])
        # under along y-axis
        if y > 0:
            # full
            if (container[x:x+xx, y-1, z] == 0).all():
                if (x>0 and container[x-1, y-1, z] == 0) or \
                    (x+xx<container_size[0] and container[x+xx, y-1, z] == 0):
                    for y1 in reversed(range(0, y)):
                        if y1 == 0: break
                        if not (container[x:x+xx, y1-1, z] == 0).all(): break
                    ems_list.append([x, y1, z, x+xx-1, y-1, z])
            # part
            else:
                # left
                if container[x, y-1, z] == 0:
                    if x>0 and container[x-1, y-1, z] == 0:
                        for x2 in range(x, x+xx):
                            if x2 == container_size[0] - 1: break
                            if container[x2+1, y-1, z] != 0: break
                        for y1 in reversed(range(0, y)):
                            if y1 == 0: break
                            if not (container[x:x2+1, y1-1, z] == 0).all(): break
                        ems_list.append([x, y1, z, x2, y-1, z])
                # right
                if container[x+xx-1, y-1, z] == 0:
                    if x+xx < container_size[0] and container[x+xx, y-1, z] == 0:
                        for x1 in reversed(range(x, x+xx)):
                            if x1 == 0: break
                            if container[x1-1, y-1, z] != 0: break
                        for y1 in reversed(range(0, y)):
                            if y1 == 0: break
                            if not (container[x1:x+xx, y1-1, z] == 0).all(): break
                        ems_list.append([x1, y1, z, x+xx-1, y-1, z])
        # upon along z-axis(top)
        if z+zz < container_size[2]:
            # full
            if (container[x:x+xx, y:y+yy, z+zz] == 0).all():
                if not [x, y, z+zz, x+xx-1, y+yy-1, z+zz] in ems_list:
                    ems_list.append([x, y, z+zz, x+xx-1, y+yy-1, z+zz])
            else:
                # build the histogram map
                hotmap = (container[x:x+xx, y:y+yy, z+zz] == 0).astype(int)
                histmap = np.zeros_like(hotmap).astype(int)
                for i in reversed(range(xx)):
                    for j in range(yy):
                        if i==xx-1: histmap[i, j] = hotmap[i, j]
                        elif hotmap[i, j] == 0: histmap[i, j] = 0
                        else: histmap[i, j] = histmap[i+1, j] + hotmap[i, j]
                # scan the histogram map
                for i in range(xx):
                    for j in range(yy):
                        if histmap[i, j] == 0: continue
                        if j>0 and histmap[i, j] == histmap[i, j-1]: continue
                        if i>0 and (container[x+i, y+j:y+yy, z+zz] == container[x+i-1, y+j:y+yy, z+zz]).all(): continue
                        i2 = i + histmap[i, j] -1
                        # look right
                        for j2 in range(j, yy):
                            if j2 == yy-1: break
                            if histmap[i, j2+1] < histmap[i, j]: break
                        if i>0 and (container[x+i, y+j:y+j2, z+zz] == container[x+i-1, y+j:y+j2, z+zz]).all(): continue
                        # look left
                        for j1 in reversed(range(0, j+1)):
                            if j1 == 0: break
                            if histmap[i, j1-1] < histmap[i, j]: break
                        if not [x+i, y+j1, z, x+i2, y+j2, z] in ems_list:
                            ems_list.append([x+i, y+j1, z, x+i2, y+j2, z])


    # varients to store results of searching ems corners
    ems_num = len(ems_list) * 4
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    empty_ems = [empty_size] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0]   * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _y, _z):
        # check if the pos visited
        if [_x, _y, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _y:_y+block_y, _z-1]==0).all(): return
        visited.append([_x, _y, _z])
        if (container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] == 0).all():
            if not is_stable(block, np.array([_x, _y, _z]), container):
                if reward_type.endswith('hard'):
                    return
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _y, _z])
            heightmap_ems[index][_x:_x+block_x, _y:_y+block_y] = _z + block_z
            is_settle_ems[index] = True

    # calculate the compactness and pyramidality
    def calc_C_P_S(index):
        _x, _y, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        if _z+block_x > height: height = _z+block_z
        bbox_size = height * container_size[0] * container_size[1]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
        empty_ems[index] += np.sum(under_space==0)
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    # update the level_free_space
    def update_level_free_space(pos):
        _x, _y, _z = pos
        xx = _x + block_x - 1
        # updated_level_free_space = level_free_space.copy()
        updated_level_free_space = copy.deepcopy(level_free_space)
        # _z ~ _z+block_z
        for zz, yy in itertools.product( range(_z, _z+block_z), range(_y, _y+block_y) ):
            free_space = updated_level_free_space[zz][yy]
            if _x in free_space:
                idx = free_space.index(_x)
                # odd
                if (idx+1) % 2 == 1:
                    if xx in free_space:
                        if block_x == 1:
                            if free_space[idx+1] == _x:
                                free_space.remove(_x)
                                free_space.remove(_x)
                            else:
                                free_space[idx] = _x + 1
                        else:
                            free_space.remove(_x)
                            free_space.remove(xx)
                    else:
                        free_space[idx] = xx + 1
                # even
                else:
                    free_space[idx] = _x - 1
            else:
                if xx in free_space:
                    idx = free_space.index(xx)
                    free_space[idx] = _x - 1
                else:
                    free_space.append(_x - 1)
                    free_space.append(xx + 1)
                    free_space.sort()
        # 0 ~ _z
        for zz, yy in itertools.product( range(0, _z), range(_y, _y+block_y) ):
            free_space = updated_level_free_space[zz][yy]
            spaces = [free_space[i:i+2] for i in range(0, len(free_space), 2)]
            for space in spaces:
                x1, x2 = space
                if x1 == x2:
                    if x1 >= _x and x1 <= xx: 
                        free_space.remove(x1)
                        free_space.remove(x1)
                elif block_x == 1:  # _x == xx
                    if _x == x1: free_space[free_space.index(x1)] = _x + 1
                    elif _x == x2: free_space[free_space.index(x2)] = xx - 1
                elif _x <= x1 and x2 <= xx:
                    free_space.remove(x1)
                    free_space.remove(x2)
                elif _x <= x1 and x1 <= xx: free_space[free_space.index(x1)] = xx + 1
                elif _x <= x2 and x2 <= xx: free_space[free_space.index(x2)] = _x - 1
        return updated_level_free_space

    def update_container(ctn, pos):
        _x, _y, _z = pos
        ctn[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
        under_space = ctn[_x:_x+block_x, _y:_y+block_y, 0:_z]
        ctn[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space==0 ] = -1

    def calc_maximal_usable_spaces(ctn, H):
        score = 0
        for h in range(H):
            level_max_empty = 0
            # build the histogram map
            hotmap = (ctn[:, :, h] == 0).astype(int)
            histmap = np.zeros_like(hotmap).astype(int)
            for i in reversed(range(container_size[0])):
                for j in range(container_size[1]):
                    if i==container_size[0]-1: histmap[i, j] = hotmap[i, j]
                    elif hotmap[i, j] == 0: histmap[i, j] = 0
                    else: histmap[i, j] = histmap[i+1, j] + hotmap[i, j]
            # scan the histogram map
            for i in range(container_size[0]):
                for j in range(container_size[1]):
                    if histmap[i, j] == 0: continue
                    if j>0 and histmap[i, j] == histmap[i, j-1]: continue
                    # look right
                    for j2 in range(j, container_size[1]):
                        if j2 == container_size[1] - 1: break
                        if histmap[i, j2+1] < histmap[i, j]: break
                    # look left
                    for j1 in reversed(range(0, j+1)):
                        if j1 == 0: break
                        if histmap[i, j1-1] < histmap[i, j]: break
                    area = histmap[i, j] * (j2 - j1 + 1)
                    if area > level_max_empty: level_max_empty = area
            score += level_max_empty
        return score

    # search positions in each ems, from 4 coreners of each
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        X1, Y1, Z, X2, Y2, _ = ems
        # search from the 4 corners of the ems
        # left-bottom corner
        if X1 < X and Y1 < Y:
            index = ems_index * 4 + 0
            heightmap_ems[index] = heightmap.copy()
            for _x, _y in itertools.product(range(X1, X), range(Y1, Y)):
                if is_settle_ems[index]: break
                check_position(index, _x, _y, Z)
            if is_settle_ems[index]: calc_C_P_S(index)
        # right-bottom corner
        if X2-block_x+2 > 0 and Y1 < Y:
            index = ems_index * 4 + 1
            heightmap_ems[index] = heightmap.copy()
            for _y, _x in itertools.product(range(Y1, Y), reversed(range(0, X2-block_x+2))):
                if is_settle_ems[index]: break
                check_position(index, _x, _y, Z)
            if is_settle_ems[index]: calc_C_P_S(index)
        # right-up corner
        if X2-block_x+2 > 0 and Y2-block_y+2 > 0:
            index = ems_index * 4 + 2
            heightmap_ems[index] = heightmap.copy()
            for _x, _y in itertools.product(reversed(range(0, X2-block_x+2)), reversed(range(0, Y2-block_y+2))):
                if is_settle_ems[index]: break
                check_position(index, _x, _y, Z)
            if is_settle_ems[index]: calc_C_P_S(index)
        # left-up corner
        if X1 < X and Y2-block_y+2 > 0:
            index = ems_index * 4 + 3
            heightmap_ems[index] = heightmap.copy()
            for _y, _x in itertools.product(reversed(range(0, Y2-block_y+2)), range(X1, X)):
                if is_settle_ems[index]: break
                check_position(index, _x, _y, Z)
            if is_settle_ems[index]: calc_C_P_S(index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return positions, container, stable, heightmap, valid_size, empty_size, level_free_space

    # get the best ems
    if reward_type.startswith('mcs'):
        ratio_ems = [0.0] * ems_num
    else:
        ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_score = np.max(ratio_ems)
    best_ems_indexes = [i for i,s in enumerate(ratio_ems) if s==best_score]
    updated_containers = [0] * len(best_ems_indexes)
    maximal_usable_spaces_ems = [0] * len(best_ems_indexes)
    if len(best_ems_indexes) > 1 and 'mcs' in reward_type:
        max_height = np.max(heightmap_ems)
        for i, index in enumerate(best_ems_indexes):
            if is_settle_ems[index]:
                updated_containers[i] = container.copy()
                update_container(updated_containers[i], pos_ems[index])
                maximal_usable_spaces_ems[i] = calc_maximal_usable_spaces(updated_containers[i], max_height)
        best_free_space_index = np.argmax(maximal_usable_spaces_ems)
        best_ems_index = best_ems_indexes[best_free_space_index]
        while not is_settle_ems[best_ems_index]:
            maximal_usable_spaces_ems[best_free_space_index] = -1
            best_free_space_index = np.argmax(maximal_usable_spaces_ems)
            best_ems_index = best_ems_indexes[best_free_space_index]
    else:
        best_ems_index = best_ems_indexes[0]
        while not is_settle_ems[best_ems_index]:
            best_ems_indexes.remove(best_ems_indexes[0])
            best_ems_index = best_ems_indexes[0]

    # update the positions, stable list, the container, and the level_free_space
    _x, _y, _z = pos_ems[best_ems_index]
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]
    update_container(container, positions[block_index])
    level_free_space = update_level_free_space(positions[block_index])

    # update the height map
    heightmap[_x:_x+block_x, _y:_y+block_y] = _z + block_z

    return positions, container, stable, heightmap, valid_size, empty_size, level_free_space

def calc_one_position_mcs(blocks, block_index, positions, container, reward_type, stable,
                             heightmap, valid_size, empty_size, level_free_space):
    """
    calculate the latest block's position in the container
    by the policy of maximizing remaining empty spaces
    ---
    params:
    ---
        blocks: n x 3 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 3 array, positions of blocks, those after block_index are still [0, 0]
        container: 3-dimension array, the container
        reward_type: string
            'mcs-soft'
            'mcs-hard'
            'C+P-mul-soft'
            'C+P-mul-hard'
            'C+P-mcs-soft'
            'C+P-mcs-hard'
            'C+P+S-mul-soft'
            'C+P+S-mul-hard'
            'C+P+S-mcs-soft'
            'C+P+S-mcs-hard'
        stable: n x 1 bool list, the blocks' stability state
        heightmap: width x depth array, heightmap of the container
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 3 array, updated positions
        container: 3-dimension array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: width x depth array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """
    block_dim = len(container.shape)
    if block_dim == 2:
        return calc_one_position_mcs_2d(blocks, block_index, positions, container, reward_type, stable,
                             heightmap, valid_size, empty_size, level_free_space)
    elif block_dim == 3:
        return calc_one_position_mcs_3d(blocks, block_index, positions, container, reward_type, stable,
                             heightmap, valid_size, empty_size, level_free_space)

def calc_positions_mcs(blocks, container_size, reward_type):
    '''
    calculate the positions
    ---
    params:
    ---
        blocks: n x 3 array, blocks with some order
        container_size: 1 x 3 array, size of the container
        reward_type: string
            'mcs-soft'
            'mcs-hard'
            'C+P-mul-soft'
            'C+P-mul-hard'
            'C+P-mcs-soft'
            'C+P-mcs-hard'
            'C+P+S-mul-soft'
            'C+P+S-mul-hard'
            'C+P+S-mcs-soft'
            'C+P+S-mcs-hard'
    return:
    ---
        positions: n x 3 array, positions of blocks
        container: a 3-dimension array, final state of container
        place: n x 1 bool list, the blocks' placed state
        ratio: float, C / C*S1 / C*S2 / C+P / (C+P)*S1 / (C+P)*S2, the ratio calculated by the following 4 scores
        scores: 4 float numbers: valid-size, box-size, empty-size, stable-num and packing_height
    '''
    # Initialize at the first block
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    stable = [False] * blocks_num
    valid_size = 0
    empty_size = 0
    heightmap = np.zeros(container_size[:-1]).astype(int)
    container = np.zeros(container_size)
    
    if block_dim == 2:
        # level free space (z, x), storing each row's empty spaces' two corners
        level_free_space = [ [ 0, container_size[0]-1 ] \
                                    for i in range(container_size[-1]) ]
    elif block_dim == 3:
        # level free space (z, y, x), storing each row's empty spaces' two corners
        level_free_space = [ [ [ 0, container_size[0]-1 ] \
                                    for i in range(container_size[1]) ] \
                                        for j in range(container_size[2]) ]

    for block_index in range(blocks_num):
        positions, container, stable, heightmap, valid_size, empty_size, level_free_space = \
            calc_one_position_mcs (blocks, block_index, positions, container, reward_type, 
                                    stable, heightmap, valid_size, empty_size, level_free_space)

    stable_num = np.sum(stable)
    if block_dim == 2:
        box_size = np.max(heightmap) * container_size[0]
    elif block_dim == 3:
        box_size = np.max(heightmap) * container_size[0] * container_size[1]
    
    if blocks_num == 0:
        C = 0
        P = 0
        S = 0
    else:
        C = valid_size / box_size
        P = valid_size / (empty_size + valid_size)
        S = stable_num / blocks_num

    if 	 reward_type == 'comp':			ratio = C
    elif reward_type == 'soft':			ratio = C * S
    elif reward_type == 'hard':			ratio = C * S
    elif reward_type == 'pyrm':			ratio = C + P
    elif reward_type == 'pyrm-soft':	ratio = (C + P) * S
    elif reward_type == 'pyrm-hard':	ratio = (C + P) * S

    elif reward_type == 'mcs-soft':	ratio = (C + P) * S
    elif reward_type == 'mcs-hard':	ratio = (C + P) * S
    
    elif reward_type == 'pyrm-soft-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-soft-SUM':	ratio = 2*C + P + S

    elif reward_type == 'pyrm-hard-sum':	ratio = C + P + S
    elif reward_type == 'pyrm-hard-SUM':	ratio = 2*C + P + S

    elif reward_type == 'CPS':              ratio = C * P * S

    # Final
    elif reward_type == 'mcs-soft':         ratio = C + P + S
    elif reward_type == 'mcs-hard':         ratio = C + P + S
    elif reward_type == 'C+P-mul-soft':     ratio = C + P + S
    elif reward_type == 'C+P-mul-hard':     ratio = C + P + S
    elif reward_type == 'C+P-mcs-soft':     ratio = C + P + S
    elif reward_type == 'C+P-mcs-hard':     ratio = C + P + S
    elif reward_type == 'C+P+S-mul-soft':   ratio = C + P + S
    elif reward_type == 'C+P+S-mul-hard':   ratio = C + P + S
    elif reward_type == 'C+P+S-mcs-soft':   ratio = C + P + S
    elif reward_type == 'C+P+S-mcs-hard':   ratio = C + P + S

    else:								    print('Unkown reward type')

    # return ratio, valid_size, box_size, empty_size, stable_num
    scores = [valid_size, box_size, empty_size, stable_num, np.max(heightmap)]
    return positions, container, stable, ratio, scores

# ===============================

# NOTE pack-net to pack object


# ===============================

class DQN(nn.Module):
    '''Structure for L-Pnet'''
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

from pack_net import LG_RL

def calc_one_position_net(blocks, block_index, positions, container, reward_type, stable, 
                        heightmap, valid_size, empty_size, level_free_space=None, net=None ):
    """
    calculate the latest block's position in the container
    ---
    params:
    ---
        blocks: n x 2 array, blocks with some order
        block_index: int, index of the block to pack, previous were already packed
        positions: n x 2 array, positions of blocks, those after block_index are still [0, 0]
        container: 2-dimension array, the container
        reward_type: string
        stable: n x 1 bool list, the blocks' stability state
        heightmap: width x 1 array, heightmap of the container
        valid_size: int, sum of the packed blocks' volume
        empty_size: int, size of the empty space under packed blocked
        level_free_space: lists, each element is a new list to manage each level's free space
    return:
    ---
        positions: n x 2 array, updated positions
        container: 2-dimension array, updated container
        stable: n x 1 bool list, updated stable
        heightmap: width x 1 array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
        level_free_space: updated level_free_space
    """

    block = blocks[block_index].astype(int)
    
    container_width, container_height = container.shape
    block_width, block_height = block
    
    block_width = int(block_width)
    block_height = int(block_height)

    tensor_heightmap = torch.from_numpy( heightmap.astype('float32')[None, None, :] ).cuda()
    tensor_block = torch.from_numpy( block.astype('float32')[None, None, :] ).cuda()

    if net is None:
        if reward_type == 'C+P+S-SL-soft':
            net = DQN(container_width, True).cuda().eval()
            net.load_state_dict(torch.load('./pack_net/SL_rand_diff/checkpoints/199/SL.pt'))
        elif reward_type == 'C+P+S-RL-soft':
            net = DQN(container_width, True).cuda().eval()
            checkpoint = torch.load('./pack_net/RL_rand_diff/checkpoints/199/actor.pt')
            net.load_state_dict(checkpoint)


    pos_x = net(tensor_heightmap, tensor_block).data.max(1)[1]

    pos_x = int(pos_x)
    
    while pos_x + block_width > container_width:
        pos_x -= 1

    pos_z = int(heightmap[pos_x:pos_x+block_width].max())
    

    block_id = block_index + 1
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

        heightmap[pos_x:pos_x+block_width] = heightmap[pos_x:pos_x+block_width].max() + block_height

    else:
        pos_x = container_width
        stable[block_id - 1] = False

    positions[block_id - 1] = [pos_x, pos_z]

    # C P S
    box_size = (heightmap.max() * container_width)
    valid_size = (container >= 1).sum()
    py_size = (container != 0).sum()
    empty_size = py_size - valid_size

    # return container, heightmap, stable, box_size, valid_size, empty_size, (pos_x, pos_z), current_blocks_num

    return positions, container, stable, heightmap, valid_size, empty_size, level_free_space

def calc_positions_LG_net(blocks, container_size, reward_type):
    '''
    calculate the positions
    ---
    params:
    ---
        blocks: n x ? array, blocks with some order
        container_size: 1 x ? array, size of the container
        reward_type: string
            'C+P+S-SL-soft'
            'C+P+S-RL-soft'
            'C+P+S-LG-soft'
            'C+P+S-LG-gt-soft'
    return:
    ---
        positions: n x ? array, positions of blocks
        container: a ?-dimension array, final state of container
        place: n x 1 bool list, the blocks' placed state
        ratio: float, C / C*S1 / C*S2 / C+P / (C+P)*S1 / (C+P)*S2, the ratio calculated by the following 4 scores
        scores: 4 float numbers: valid-size, box-size, empty-size, stable-num and packing_height
    '''
    # Initialize at the first block
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    
    container_width, container_height = container_size
    tensor_block = torch.from_numpy( blocks.astype('float32')[None, :, :] ).transpose(2,1).cuda()
    
    if reward_type == 'C+P+S-G-soft' or reward_type == 'C+P+S-G-gt-soft':
        net = LG_RL.PackRNN(2, 128, container_width, 128, container_width, container_height, 'diff', pack_net_type='G').cuda().eval()
        net.load_state_dict(torch.load('./pack_net/G_rand_diff/checkpoints/199/actor.pt'))
        # net.load_state_dict(torch.load('./pack/10/2d-bot-C+P+S-G-soft-width-5-diff-pre_eval-note-sh-R-2020-05-16-23-18/checkpoints/85/actor_pnet.pt'))
        ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height, containers, stables = LG_RL.calc_positions(net, tensor_block, container_size, False)

    elif reward_type == 'C+P+S-LG-soft' or reward_type == 'C+P+S-LG-gt-soft':
        net = LG_RL.PackRNN(2, 128, container_width, 128, container_width, container_height, 'diff', pack_net_type='LG').cuda().eval()
        net.load_state_dict(torch.load('./pack_net/LG_rand_diff/checkpoints/199/actor.pt'))
        ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height, containers, stables = LG_RL.calc_positions(net, tensor_block, container_size, False)
    
    scores = [valid_size, box_size, empty_size, stable_num, packing_height]
    return positions[0], containers[0], stables[0], ratio, scores

def calc_positions_net(blocks, container_size, reward_type):
    '''
    calculate the positions
    ---
    params:
    ---
        blocks: n x ? array, blocks with some order
        container_size: 1 x ? array, size of the container
        reward_type: string
            'C+P+S-SL-soft'
            'C+P+S-RL-soft'
            'C+P+S-LG-soft'
            'C+P+S-LG-gt-soft'
    return:
    ---
        positions: n x ? array, positions of blocks
        container: a ?-dimension array, final state of container
        place: n x 1 bool list, the blocks' placed state
        ratio: float, the ratio calculated by the following 4 scores
        scores: 4 float numbers: valid-size, box-size, empty-size, stable-num and packing_height
    '''
    # Initialize at the first block
    # return calc_positions_LG_net(blocks, container_size, reward_type)
    if reward_type == 'C+P+S-LG-soft' or reward_type == 'C+P+S-LG-gt-soft' or \
    reward_type == 'C+P+S-G-soft' or reward_type == 'C+P+S-G-gt-soft':
        return calc_positions_LG_net(blocks, container_size, reward_type)
    
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    stable = [False] * blocks_num
    valid_size = 0
    empty_size = 0
    heightmap = np.zeros(container_size[:-1]).astype(int)
    container = np.zeros(container_size)
    
    container_width, container_height = container_size
    
    if reward_type == 'C+P+S-SL-soft':
        net = DQN(container_size[0], True).cuda().eval()
        net.load_state_dict(torch.load('./pack_net/SL_rand_diff/checkpoints/199/SL.pt'))
        
    elif reward_type == 'C+P+S-RL-soft':
        net = DQN(container_size[0], True).cuda().eval()
        checkpoint = torch.load('./pack_net/RL_rand_diff/checkpoints/199/actor.pt')
        net.load_state_dict(checkpoint)

        
    if block_dim == 2:
        # level free space (z, x), storing each row's empty spaces' two corners
        level_free_space = [ [ 0, container_size[0]-1 ] \
                                    for i in range(container_size[-1]) ]
    elif block_dim == 3:
        # level free space (z, y, x), storing each row's empty spaces' two corners
        level_free_space = [ [ [ 0, container_size[0]-1 ] \
                                    for i in range(container_size[1]) ] \
                                        for j in range(container_size[2]) ]
    for block_index in range(blocks_num):
        positions, container, stable, heightmap, valid_size, empty_size, level_free_space = \
            calc_one_position_net(blocks, block_index, positions, container, reward_type, 
                                    stable, heightmap, valid_size, empty_size, level_free_space, net)



    stable_num = np.sum(stable)
    if block_dim == 2:
        box_size = np.max(heightmap) * container_size[0]
    elif block_dim == 3:
        box_size = np.max(heightmap) * container_size[0] * container_size[1]
    
    if blocks_num == 0:
        C = 0
        P = 0
        S = 0
    
    else:
        C = valid_size / box_size
        P = valid_size / (empty_size + valid_size)
        S = stable_num / blocks_num

    if reward_type == 'C+P+S-SL-soft':   ratio = (C + P + S)/3
    elif reward_type == 'C+P+S-RL-soft':   ratio = (C + P + S)/3
    elif reward_type == 'C+P+S-LG-soft':   ratio = (C + P + S)/3
    elif reward_type == 'C+P+S-LG-gt-soft': ratio = (C + P + S)/3
    elif reward_type == 'C+P+S-G-soft':   ratio = (C + P + S)/3
    elif reward_type == 'C+P+S-G-gt-soft': ratio = (C + P + S)/3
    else:								    print('Unkown reward type')

    # return ratio, valid_size, box_size, empty_size, stable_num
    scores = [valid_size, box_size, empty_size, stable_num, np.max(heightmap)]
    return positions, container, stable, ratio, scores





# ===============================
# NOTE container class

class Container(object):
    '''
    Container class to place the object
    '''
    def __init__(self, container_size, blocks_num, reward_type, heightmap_type='full', 
                 initial_container_size=None, max_height=None, packing_strategy='LB_GREEDY'):
        block_dim = len(container_size)

        self.reward_type = reward_type

        if reward_type == 'C+P+S-mul-soft' or reward_type == 'C+P+S-mul-hard':
            packing_strategy = 'MUL'
        elif reward_type == 'C+P+S-mcs-soft' or reward_type == 'C+P+S-mcs-hard':
            packing_strategy = 'MACS'
        
        self.block_dim = len(container_size)

        if max_height is None:
            self.max_height = 2 * container_size[0]
        else:
            self.max_height = max_height

        self.positions = np.zeros((blocks_num, block_dim)).astype('int')
        self.container = np.zeros(container_size).astype('int')
        self.heightmap = np.zeros(container_size[:-1]).astype('int')
        self.blocks = []
        self.bounding_box = np.zeros(self.block_dim)
        self.stable = [False] * blocks_num
        self.rotate_state = [False] * blocks_num
        self.valid_size = 0
        self.empty_size = 0

        if packing_strategy == 'MACS' or packing_strategy == 'MUL':    
            if block_dim == 2:
                # level free space (z, y, x), storing each row's empty spaces' two corners
                self.level_free_space = [ [ 0, container_size[0]-1 ] \
                                            for i in range(container_size[-1]) ]
            elif block_dim == 3:
                # level free space (z, y, x), storing each row's empty spaces' two corners
                self.level_free_space = [ [ [ 0, container_size[0]-1 ] \
                                            for i in range(container_size[1]) ] \
                                                for j in range(container_size[2]) ]
        elif packing_strategy == 'LB' or packing_strategy == 'LB_GREEDY':
            if block_dim == 2:
                self.level_free_space = [ [0] for i in range(container_size[1]) ]
            elif block_dim == 3:
                self.level_free_space = [ [ [0] for i in range(container_size[1]) ] for j in range(container_size[2]) ]

        self.current_blocks_num = 0

        self.blocks_num = blocks_num
        self.container_size = container_size
        self.initial_container_size = initial_container_size
        self.packing_strategy = packing_strategy
        self.heightmap_type = heightmap_type

    def add_new_block(self, block, is_rotate=False):
        '''
        Add a block into current container
        ---
        params:
        ---
            block: int * 2/3 array, a block
        returns:
        ---
            heightmap: (width) or (width x depth) array, heightmap of the container
        '''
        self.blocks.append(block)
        blocks = np.array(self.blocks).astype('int')

        self.rotate_state[ self.current_blocks_num ] = is_rotate
        
        if self.packing_strategy == 'MACS' or self.packing_strategy == 'MUL':
            positions, container, stable, heightmap, valid_size, empty_size, level_free_space = calc_one_position_mcs(
                blocks, self.current_blocks_num, self.positions, self.container, self.reward_type,
                self.stable, self.heightmap, self.valid_size, self.empty_size, self.level_free_space )
        elif self.packing_strategy == 'LB':
            positions, container, stable, heightmap, bounding_box, valid_size, empty_size, level_free_space = calc_one_position_greedy(
                blocks, self.current_blocks_num, self.positions, self.container, self.reward_type,
                self.stable, self.heightmap, self.bounding_box, self.valid_size, self.empty_size, self.level_free_space )
        elif self.packing_strategy == 'LB_GREEDY':
            container, positions, stable, heightmap, valid_size, empty_size = calc_one_position_lb_greedy(
                block.astype(int), self.current_blocks_num, self.container_size, self.reward_type, 
                self.container, self.positions, self.stable, self.heightmap, self.valid_size, self.empty_size)
        # p-net
        elif self.reward_type == 'C+P+S-SL-soft' or self.reward_type == 'C+P+S-RL-soft':
                positions, container, stable, heightmap, valid_size, empty_size, level_free_space = calc_one_position_net(
                    blocks, self.current_blocks_num, self.positions, self.container, self.reward_type,
                    self.stable, self.heightmap, self.valid_size, self.empty_size ) #, self.level_free_space self.net )
        elif self.reward_type == 'C+P+S-LG-soft'  or self.reward_type == 'C+P+S-LG-gt-soft' or \
            self.reward_type == 'C+P+S-G-soft'  or self.reward_type == 'C+P+S-G-gt-soft':
            positions, container, stable, _, score = calc_positions_LG_net(
                blocks, self.container_size, self.reward_type )
            valid_size, _, empty_size, _, heightmap = score
            level_free_space = self.level_free_space


        self.positions = positions
        self.container = container
        self.stable = stable
        self.heightmap = heightmap
        # self.bounding_box = bounding_box
        self.valid_size = valid_size
        self.empty_size = empty_size
        if self.packing_strategy != 'LB_GREEDY':
            self.level_free_space = level_free_space
        self.current_blocks_num = self.current_blocks_num + 1

        # heightmap type that pass to the network decoder
        if self.heightmap_type == 'full':
            hm = self.heightmap
        elif self.heightmap_type == 'zero':
            hm = self.heightmap - np.min(self.heightmap)
        elif self.heightmap_type == 'diff':
            if len(heightmap.shape) == 2:
                # x coordinate
                hm_diff_x = np.insert(self.heightmap, 0, heightmap[0, :], axis=0)
                hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
                hm_diff_x = heightmap - hm_diff_x
                # hm_diff_x = np.delete(hm_diff_x, 0, axis=0)
                # y coordinate
                hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
                hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
                hm_diff_y = heightmap - hm_diff_y
                # hm_diff_y = np.delete(hm_diff_y, 0, axis=1)
                # combine
                width = self.heightmap.shape[0]
                length = self.heightmap.shape[1]
                hm = np.zeros( (2, width, length) ).astype(int)
                hm[0] = hm_diff_x
                hm[1] = hm_diff_y
            else:
                hm = self.heightmap
                hm = np.insert(hm, len(hm)-1, hm[-1])
                hm = np.delete(hm, 0)
                hm = hm - self.heightmap
                hm = np.delete(hm, len(hm)-1)
        return hm

    def add_new_block_at(self, block, pos_x, is_rotate=False):
        '''
        Add a block into current container
        ---
        params:
        ---
            block: 2/3 int array, a block
            pos_x: int, x axis position
        returns:
        ---
            heightmap: (width) or (width x depth) array, heightmap of the container
        '''
        self.blocks.append(block)
        blocks = np.array(self.blocks).astype('int')

        self.rotate_state[ self.current_blocks_num ] = is_rotate

        pos_x = int(pos_x)

        block_width, block_height = block.astype('int')
        if pos_x + block_width > self.container_size[0]:
            while pos_x + block_width > self.container_size[0]:
                pos_x -= 1

        # calc z
        pos_z = np.max(self.heightmap[ pos_x : pos_x+block_width ])
        # add into container
        self.positions[ self.current_blocks_num ] = (pos_x, pos_z)
        self.container[ pos_x:pos_x+block_width, pos_z:pos_z+block_height ] = self.current_blocks_num + 1
        under = self.container[ pos_x:pos_x+block_width, :pos_z ]
        self.container[ pos_x:pos_x+block_width, :pos_z ][ under == 0 ] = -1

        # check stable
        if pos_z == 0:
            self.stable[self.current_blocks_num] = True
        else:
            support = self.container[ pos_x:pos_x+block_width, pos_z-1 ]
            self.stable[self.current_blocks_num] = is_stable_2d(support, pos_x, block_width)
        
        self.heightmap[pos_x : pos_x+block_width] = pos_z + block_height

        self.valid_size += block_width * block_height
        self.empty_size += np.sum(  self.container[ pos_x:pos_x+block_width, :pos_z ] == -1 )

        self.current_blocks_num = self.current_blocks_num + 1

        # heightmap type that pass to the network decoder
        heightmap = copy.deepcopy(self.heightmap)
        if self.heightmap_type == 'full':
            hm = self.heightmap
        elif self.heightmap_type == 'zero':
            hm = self.heightmap - np.min(self.heightmap)
        elif self.heightmap_type == 'diff':
            if len(heightmap.shape) == 2:
                # x coordinate
                hm_diff_x = np.insert(self.heightmap, 0, heightmap[0, :], axis=0)
                hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
                hm_diff_x = heightmap - hm_diff_x
                # hm_diff_x = np.delete(hm_diff_x, 0, axis=0)
                # y coordinate
                hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
                hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
                hm_diff_y = heightmap - hm_diff_y
                # hm_diff_y = np.delete(hm_diff_y, 0, axis=1)
                # combine
                width = self.heightmap.shape[0]
                length = self.heightmap.shape[1]
                hm = np.zeros( (2, width, length) ).astype(int)
                hm[0] = hm_diff_x
                hm[1] = hm_diff_y
            else:
                hm = self.heightmap
                hm = np.insert(hm, len(hm)-1, hm[-1])
                hm = np.delete(hm, 0)
                hm = hm - self.heightmap
                hm = np.delete(hm, len(hm)-1)
        return hm

    def get_heightmap(self, is_full=None):
        heightmap = copy.deepcopy(self.heightmap)
        if is_full is not None:
            return heightmap
        if self.heightmap_type == 'full':
            hm = self.heightmap
        elif self.heightmap_type == 'zero':
            hm = self.heightmap - np.min(self.heightmap)
        elif self.heightmap_type == 'diff':
            if len(heightmap.shape) == 2:
                # x coordinate
                hm_diff_x = np.insert(self.heightmap, 0, heightmap[0, :], axis=0)
                hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
                hm_diff_x = heightmap - hm_diff_x
                # hm_diff_x = np.delete(hm_diff_x, 0, axis=0)
                # y coordinate
                hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
                hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
                hm_diff_y = heightmap - hm_diff_y
                # hm_diff_y = np.delete(hm_diff_y, 0, axis=1)
                # combine
                width = self.heightmap.shape[0]
                length = self.heightmap.shape[1]
                hm = np.zeros( (2, width, length) ).astype(int)
                hm[0] = hm_diff_x
                hm[1] = hm_diff_y
            else:
                hm = self.heightmap
                hm = np.insert(hm, len(hm)-1, hm[-1])
                hm = np.delete(hm, 0)
                hm = hm - self.heightmap
                hm = np.delete(hm, len(hm)-1)
        return hm

    def clear_container(self):
        self.positions = np.zeros((self.blocks_num, self.block_dim)).astype('int')
        self.container = np.zeros(self.container_size).astype('int')
        self.heightmap = np.zeros(self.container_size[:-1]).astype('int')
        self.blocks = []
        self.bounding_box = np.zeros(self.block_dim)
        self.stable = [False] * self.blocks_num
        self.valid_size = 0
        self.empty_size = 0
        
        if self.packing_strategy == 'MACS' or self.packing_strategy == 'MUL':    
            if self.block_dim == 2:
                # level free space (z, y, x), storing each row's empty spaces' two corners
                self.level_free_space = [ [ 0, self.container_size[0]-1 ] \
                                            for i in range(self.container_size[-1]) ]
            elif self.block_dim == 3:
                # level free space (z, y, x), storing each row's empty spaces' two corners
                self.level_free_space = [ [ [ 0, self.container_size[0]-1 ] \
                                            for i in range(self.container_size[1]) ] \
                                                for j in range(self.container_size[2]) ]
        elif self.packing_strategy == 'LB' or self.packing_strategy == 'LB_GREEDY':
            if self.block_dim == 2:
                self.level_free_space = [ [0] for i in range(self.container_size[1]) ]
            elif self.block_dim == 3:
                self.level_free_space = [ [ [0] for i in range(self.container_size[1]) ] for j in range(self.container_size[2]) ]


        self.current_blocks_num = 0

    def calc_CPS(self):
        if self.current_blocks_num == 0:
            return 0, 0, 0

        height = np.max(self.heightmap)

        if self.block_dim == 2:
            box_size = self.container_size[0] * height
        elif self.block_dim == 3:
            box_size = self.container_size[0] * self.container_size[1] * height 
        
        valid_size = self.valid_size
        empty_size = self.empty_size
        stable_num = np.sum(self.stable)

        C = valid_size / box_size
        P = valid_size / (empty_size + valid_size)
        S = stable_num / self.current_blocks_num
        return C, P, S

    def calc_ratio(self):
        '''
        Calc the current ratio
        ---
        returns:
        ---
            ratio: float, compute by reward type
        '''
        reward_type = self.reward_type
    
        C, P, S = self.calc_CPS()
        
        if 	 reward_type == 'comp':			ratio = C
        elif reward_type == 'soft':			ratio = C * S
        elif reward_type == 'hard':			ratio = C * S
        elif reward_type == 'pyrm':			ratio = C + P
        elif reward_type == 'pyrm-soft':	ratio = (C + P) * S
        elif reward_type == 'pyrm-hard':	ratio = (C + P) * S

        elif reward_type == 'mcs-soft':	ratio = (C + P) * S
        elif reward_type == 'mcs-hard':	ratio = (C + P) * S
        
        elif reward_type == 'pyrm-soft-sum':	ratio = C + P + S
        elif reward_type == 'pyrm-soft-SUM':	ratio = 2*C + P + S

        elif reward_type == 'pyrm-hard-sum':	ratio = C + P + S
        elif reward_type == 'pyrm-hard-SUM':	ratio = 2*C + P + S

        elif reward_type == 'CPS':          ratio = C * P * S

        elif reward_type == 'mcs-soft':         ratio = C + P + S
        elif reward_type == 'mcs-hard':         ratio = C + P + S
        elif reward_type == 'C+P-mul-soft':     ratio = C + P + S
        elif reward_type == 'C+P-mul-hard':     ratio = C + P + S
        elif reward_type == 'C+P-mcs-soft':     ratio = C + P + S
        elif reward_type == 'C+P-mcs-hard':     ratio = C + P + S
        elif reward_type == 'C+P+S-mul-soft':   ratio = C + P + S
        elif reward_type == 'C+P+S-mul-hard':   ratio = C + P + S
        elif reward_type == 'C+P+S-mcs-soft':   ratio = C + P + S
        elif reward_type == 'C+P+S-mcs-hard':   ratio = C + P + S

        elif reward_type == 'C+P-lb-soft':      ratio = C + P + S
        elif reward_type == 'C+P-lb-hard':      ratio = C + P + S
        elif reward_type == 'C+P+S-lb-soft':    ratio = C + P + S
        elif reward_type == 'C+P+S-lb-hard':    ratio = C + P + S

        
        elif reward_type == 'C+P+S-SL-soft':      ratio = C + P + S
        elif reward_type == 'C+P+S-RL-soft':    ratio = C + P + S
        elif reward_type == 'C+P+S-LG-soft':    ratio = C + P + S
        elif reward_type == 'C+P+S-G-soft':    ratio = C + P + S

        else:								print('Container ................')

        if reward_type == 'C+P-lb-soft':
            return (C+P)/2
        else:
            return ratio / 3

        return ratio

    def draw_container(self,  save_name, order=None):
        if self.initial_container_size is None:
            print('Do not know the initial_conainer_size')
            return

        blocks = np.array(self.blocks).astype(int)
        if order is None:
            order = [ i for i in range(self.current_blocks_num) ]
        # feasibility = check_feasibility(blocks, self.blocks_num, self.initial_container_size, order, self.reward_type)
        feasibility = None

        C, P, S = self.calc_CPS()
        save_title="Compactness: %.3f\nPyramidality: %.3f\nStability: %.3f\n" % (C, P, S)

        if self.block_dim == 2:
            draw_container_2d(blocks, self.positions.astype('int'), 
                self.container_size, self.reward_type,
                order, self.stable, 
                feasibility, self.rotate_state,
                save_title=save_title,
                save_name=save_name )
        elif self.block_dim == 3:
            draw_container_voxel( self.container, self.current_blocks_num, 
            reward_type=self.reward_type,
            order = order,
            rotate_state=self.rotate_state,
            feasibility=feasibility,
            save_title=save_title,
            save_name=save_name )

