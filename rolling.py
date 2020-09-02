import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import itertools
from tqdm import tqdm

import tools
import generate
# from model import DRL

def get_dataset( blocks_num, train_size, valid_size, obj_dim, initial_container_width, initial_container_height, arm_size, size_range, seed=None):
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
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)
    
    def generate_data(data_dir, data_size):
        blocks_f = open(data_dir + 'blocks.txt', 'w')
        pos_f = open(data_dir + 'pos.txt', 'w')
        container_f = open(data_dir + 'container.txt', 'w')
        
        deps_move_f = open(data_dir + 'dep_move.txt', 'w')
        rotate_deps_small_f = open(data_dir + 'dep_small.txt', 'w')
        rotate_deps_large_f = open(data_dir + 'dep_large.txt', 'w')

        for _ in tqdm(range(data_size)):
            rotate_blocks, positions, deps_move, rotate_deps_small, rotate_deps_large = generate.generate_blocks(blocks_num, initial_container_size, arm_size, size_range)

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

class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, int(hidden_size), kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

class HeightmapEncoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, map_width):
        super(HeightmapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, int(hidden_size/4), stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(int(hidden_size/4), int(hidden_size/2), stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(int(hidden_size/2), int(hidden_size), kernel_size= math.ceil(map_width/4))

    def forward(self, input):

        output = F.leaky_relu(self.conv1(input))
        output = F.leaky_relu(self.conv2(output))
        output = self.conv3(output).squeeze(-1)
        return output  # (batch, hidden_size, seq_len)

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, decoder_hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, 2 * encoder_hidden_size + decoder_hidden_size), requires_grad=True))

        self.decoder_input_type = decoder_input_type
        self.input_type = input_type

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        encoder_hidden = torch.cat( (static_hidden, dynamic_hidden), 1 )
        batch_size, hidden_size = decoder_hidden.size()

        decoder_hidden = decoder_hidden.unsqueeze(2).repeat(1, 1, static_hidden.shape[-1])

        # expand_as(static_hidden)
        hidden = torch.cat((encoder_hidden, decoder_hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.decoder_input_type = decoder_input_type
        self.input_type = input_type

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, decoder_hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, 4 * encoder_hidden_size), requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU( decoder_hidden_size, decoder_hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention( encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context

        enc_attn = self.encoder_attn( static_hidden, dynamic_hidden, rnn_out)
        
        encoder_hidden = torch.cat( (static_hidden, dynamic_hidden), 1)
        context = enc_attn.bmm( encoder_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as( encoder_hidden )
        
        energy = torch.cat(( encoder_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh

class DRL(nn.Module):
    def __init__(self, static_size, dynamic_size, encoder_hidden_size, decoder_hidden_size, 
                use_cuda, input_type, allow_rot, container_width, container_height, block_dim,
                reward_type, decoder_input_type, heightmap_type, packing_strategy, 
                containers,
                update_fn, mask_fn, num_layers=1, dropout=0., unit=1):
        super(DRL, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        print('    static size: %d, dynamic size: %d' % (static_size, dynamic_size))
        print('    encoder hidden size: %d' % (encoder_hidden_size))
        print('    decoder hidden size: %d' % (decoder_hidden_size))

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, encoder_hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, encoder_hidden_size)
        
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
        
        if input_type == 'mul' or input_type == 'mul-with':
            if block_dim == 2:
                heightmap_width = heightmap_width * 2
            else:
                heightmap_num = heightmap_num * 2
        
        if decoder_input_type == 'shape_only':
            self.decoder = Encoder(static_size, decoder_hidden_size)
        elif decoder_input_type == 'heightmap_only':
            if block_dim == 2:
                self.dynamic_decoder = Encoder(heightmap_width, int(decoder_hidden_size))
            elif block_dim == 3:
                self.dynamic_decoder = HeightmapEncoder(heightmap_num, int(decoder_hidden_size), (heightmap_width, heightmap_length))
        elif decoder_input_type == 'shape_heightmap':
            self.static_decoder = Encoder(static_size, int(decoder_hidden_size/2))
            if block_dim == 2:
                self.dynamic_decoder = Encoder(heightmap_width, int(decoder_hidden_size/2))
            elif block_dim == 3:
                self.dynamic_decoder = HeightmapEncoder(heightmap_num, int(decoder_hidden_size/2), (heightmap_width, heightmap_length))

        # if use_heightmap:
        #     if only_heightmap:
        #         if block_dim == 2:
        #             self.dynamic_decoder = Encoder(container_width, int(decoder_hidden_size))
        #         elif block_dim == 3:
        #             self.dynamic_decoder = HeightmapEncoder(1, int(decoder_hidden_size), container_width)
        #     else:
        #         self.static_decoder = Encoder(static_size, int(decoder_hidden_size/2))
        #         if block_dim == 2:
        #             self.dynamic_decoder = Encoder(container_width, int(decoder_hidden_size/2))
        #         elif block_dim == 3:
        #             self.dynamic_decoder = HeightmapEncoder(1, int(decoder_hidden_size/2), container_width)
        # else:
        #     self.decoder = Encoder(static_size, decoder_hidden_size)

        self.pointer = Pointer(encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.use_cuda = use_cuda
        self.input_type = input_type
        self.allow_rot = allow_rot
        self.block_dim = block_dim
        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.reward_type = reward_type
        self.container_width = container_width
        self.container_height = container_height
        self.decoder_input_type = decoder_input_type
        self.heightmap_type = heightmap_type
        self.packing_strategy = packing_strategy
        
        self.containers = containers

    def forward(self, static, dynamic, decoder_input, last_hh=None, one_step=False):
        batch_size, _, sequence_size = static.size()

        if self.allow_rot == False:
            rotate_types = 1
        else:
            if self.block_dim == 2:
                rotate_types = 2
            elif self.block_dim == 3:
                rotate_types = 6
        blocks_num = int(dynamic.shape[-1] / rotate_types)

        # if self.block_dim == 3:
        #     container_size = [self.container_width, self.container_width, self.container_height]
        # else:
        #     container_size = [self.container_width, self.container_height]
        # if self.input_type == 'mul' or self.input_type == 'mul-with':
        #     if self.block_dim == 3:
        #         container_size_a = [self.container_width, self.container_width, self.container_height]
        #         container_size_b = container_size_a
        #     else:
        #         container_size_a = [self.container_width, self.container_height]
        #         container_size_b = container_size_a

        # if self.input_type == 'mul' or self.input_type == 'mul-with':
        #     containers_a = [tools.Container(container_size_a, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]
        #     containers_b = [tools.Container(container_size_b, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]
        # else:
        #     containers = [tools.Container(container_size, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]


        mask = torch.ones(batch_size, sequence_size)
        if self.use_cuda:
            mask = mask.cuda()

        current_mask = mask.clone()
        move_mask = dynamic[:, :blocks_num, :].sum(1)
        rotate_small_mask = dynamic[:, blocks_num:blocks_num*2, :].sum(1)
        rotate_large_mask = dynamic[:, blocks_num*2:blocks_num*3, :].sum(1)
        rotate_mask = rotate_small_mask * rotate_large_mask
        dynamic_mask = rotate_mask + move_mask
        current_mask[ dynamic_mask.ne(0) ] = 0.


        max_steps = sequence_size if self.mask_fn is None else 1000

        if self.input_type == 'mul':
            static_hidden = self.static_encoder(static[:,1:-1,:])
        elif self.input_type == 'rot-old':
            static_hidden = self.static_encoder(static)
        else:
            static_hidden = self.static_encoder(static[:,1:,:])

        dynamic_hidden = self.dynamic_encoder(dynamic)

        # if self.use_heightmap:
        if 'heightmap' in self.decoder_input_type:
            decoder_static, decoder_dynamic = decoder_input

        if one_step == True:
            max_steps = 1
            
        for _ in range(max_steps):

            if not mask.byte().any():
                break

            if self.decoder_input_type == 'shape_only':
                decoder_hidden = self.decoder(decoder_input)
            elif self.decoder_input_type == 'heightmap_only':
                decoder_hidden = self.dynamic_decoder(decoder_dynamic)
            elif self.decoder_input_type == 'shape_heightmap':
                decoder_static_hidden = self.static_decoder(decoder_static)
                decoder_dynamic_hidden = self.dynamic_decoder(decoder_dynamic)
                decoder_hidden = torch.cat( (decoder_static_hidden, decoder_dynamic_hidden), 1 )

            # if self.use_heightmap:
            #     if self.only_heightmap:
            #         decoder_hidden = self.dynamic_decoder(decoder_dynamic)
            #     else:
            #         decoder_static_hidden = self.static_decoder(decoder_static)
            #         decoder_dynamic_hidden = self.dynamic_decoder(decoder_dynamic)
            #         decoder_hidden = torch.cat( (decoder_static_hidden, decoder_dynamic_hidden), 1 )
            # else:
            #     decoder_hidden = self.decoder(decoder_input)


            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + current_mask.log(), dim=1)
            

            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, static, ptr.data, self.input_type, self.allow_rot)
                dynamic_hidden = self.dynamic_encoder(dynamic)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                current_mask, mask = self.mask_fn(mask, dynamic, static, ptr.data, self.input_type, self.allow_rot)
                current_mask = current_mask.detach()
                mask = mask.detach()


            if self.input_type == 'mul':
                static_part = static[:,1:-1,:]
            elif self.input_type == 'rot-old':
                static_part = static
            else:
                static_part = static[:,1:,:]

            # if self.use_heightmap:
            if 'heightmap' in self.decoder_input_type:
                decoder_static = torch.gather( static_part, 2,
                                ptr.view(-1, 1, 1)
                                .expand(-1, self.static_size, 1)).detach()

                is_rotate = (ptr < blocks_num).cpu().numpy().astype('bool')
                blocks = decoder_static.transpose(2,1).squeeze(1).cpu().numpy()

                # now get the selected blocks and update heightmap
                heightmaps = []
                for batch_index in range(batch_size):
                    heightmaps.append(self.containers[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] ))
                    # heightmaps.append(containers[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] ))
                if self.block_dim == 2:
                    if self.use_cuda:
                        decoder_dynamic = torch.FloatTensor(heightmaps).cuda().unsqueeze(2)
                    else:
                        decoder_dynamic = torch.FloatTensor(heightmaps).unsqueeze(2)
                elif self.block_dim == 3:
                    if self.use_cuda:
                        decoder_dynamic = torch.FloatTensor(heightmaps).cuda()
                    else:
                        decoder_dynamic = torch.FloatTensor(heightmaps)
                    if self.heightmap_type != 'diff':
                        decoder_dynamic = decoder_dynamic.unsqueeze(1)

            else:
                decoder_input = torch.gather(static_part, 2,
                                ptr.view(-1, 1, 1)
                                .expand(-1, self.static_size, 1)).detach()
                # check rotate or not
                is_rotate = (ptr < blocks_num).cpu().numpy().astype('bool')
                # now get the selected blocks and update containers
                blocks = decoder_input.transpose(2,1).squeeze(1).cpu().numpy()
                for batch_index in range(batch_size):
                    self.containers[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] )
                    # containers[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] )


        # if self.use_heightmap:
        if 'heightmap' in self.decoder_input_type:
            return ptr, [decoder_static, decoder_dynamic], last_hh

        return ptr, decoder_input, last_hh

class RollingDataset(object):
    def __init__(self, data_file, total_blocks_num, net_blocks_num, num_samples, block_dim, seed, input_type, heightmap_type, allow_rot, \
                    container_width, initial_container_width, initial_container_height, mix_data_file=None, unit=1):
        
        if seed is None:
            seed = np.random.randint(123456)
        np.random.seed(seed)
        torch.manual_seed(seed)

        deps_move = np.loadtxt(data_file + 'dep_move.txt').astype('int')
        rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('int')
        rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('int')
        
        blocks = np.loadtxt(data_file + 'blocks.txt').astype('int')
        positions = np.loadtxt(data_file + 'pos.txt').astype('int')
        container_index = np.loadtxt(data_file + 'container.txt').astype('int')

        rotate_types = np.math.factorial(block_dim)
        
        data_size = int( len(blocks) / rotate_types )

        blocks = blocks.reshape( data_size, -1, block_dim, total_blocks_num)
        blocks = blocks.transpose(0, 1, 3, 2)
        blocks = blocks.reshape( data_size, -1, block_dim )

        deps_move = deps_move.reshape( len(deps_move), total_blocks_num, -1 )
        deps_move = deps_move.transpose(0,2,1)

        positions = positions.reshape( len(positions), -1, total_blocks_num )
        positions = positions.transpose(0,2,1)
    

        if block_dim == 2:
            initial_container_size = [ initial_container_width, initial_container_height ]
        elif block_dim == 3:
            initial_container_size = [ initial_container_width, initial_container_width, initial_container_height ]

        initial_containers = []
        for batch_index in range(num_samples):
            initial_container = generate.InitialContainer(blocks[batch_index], positions[batch_index], total_blocks_num, initial_container_size, True, net_blocks_num, input_type)
            initial_containers.append(initial_container)

        self.initial_containers = initial_containers
        
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

        if input_type == 'mul' or input_type == 'mul-with':
            if block_dim == 2:
                heightmap_width = heightmap_width * 2
            else:
                heightmap_num = heightmap_num * 2

        if input_type == 'mul-with':
            static_dim = static_dim + 1

        if block_dim == 2:
            self.decoder_static = torch.zeros(1, static_dim, 1, requires_grad=True)
            self.decoder_dynamic = torch.zeros(1, heightmap_width, 1, requires_grad=True)
        elif block_dim == 3:
            self.decoder_static = torch.zeros(1, static_dim, 1, requires_grad=True)
            self.decoder_dynamic = torch.zeros(1, heightmap_num, heightmap_width, heightmap_width, requires_grad=True)

        self.num_samples = num_samples

def str2bool(v):
      return v.lower() in ('true', '1')

def validate(actor, task, num_nodes, valid_data, batch_size,
        reward_type, input_type, 
        allow_rot, obj_dim,
        container_width, initial_container_width, 
        total_blocks_num, network_blocks_num,
        use_cuda, decoder_input_type,
        **kwargs):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()

    date = datetime.datetime.now()
    now = '%s' % date.date()
    now += '-%s' % date.hour
    now += '-%s' % date.minute
    now = str(now)
    save_dir = os.path.join(task, '%d' % num_nodes, 
        str(obj_dim) + 'd-' + input_type + '-'  + reward_type + '-width-' + str(container_width) + '-note-' + kwargs['note'] + '-' + now)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + '/render'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_dir + '/render/0'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    my_valid_size = []
    my_box_size = []
    my_empty_size = []
    my_stable_num = []
    my_packing_height = []
    my_time = []

    for i in tqdm(range( valid_data.num_samples )):
        initial_container = valid_data.initial_containers[i]

        start = time.time()
        with torch.no_grad():
            one_step = True
            last_hh = None
            decoder_static = valid_data.decoder_static
            decoder_dynamic = valid_data.decoder_dynamic

            # valid_size = []
            # box_size = []
            # empty_size = [] 
            # stable_num = []
            # packing_height = []  

            while one_step == True:
                static, dynamic = initial_container.convert_to_input()
                
                static = torch.FloatTensor(static).unsqueeze(0)
                dynamic = torch.FloatTensor(dynamic).unsqueeze(0)

                if initial_container.is_last_graph():
                    one_step = False
                if use_cuda:
                    static = static.cuda()
                    dynamic = dynamic.cuda()
                    decoder_static = decoder_static.cuda()
                    decoder_dynamic = decoder_dynamic.cuda()

                # if use_heightmap:
                if 'heightmap' in decoder_input_type:
                    ptr, [decoder_static, decoder_dynamic], last_hh = actor(static, dynamic, [decoder_static, decoder_dynamic], last_hh, one_step)
                else:
                    ptr, decoder_static, last_hh = actor(static, dynamic, decoder_static, last_hh, one_step)

                # check actor.containers[0], if ptr can place and just stable, but overheight 
                # we should place in a new container
                # if container is full: 
                    # container = actor.containers[0]

                    # valid_size.append(container.valid_size)
                    # height = np.max(container.heightmap)
                    # if container.block_dim == 2:
                    #     box_size = container.container_size[0] * height
                    # elif container.block_dim == 3:
                    #     box_size = container.container_size[0] * container.container_size[1] * height 

                    # box_size.append(box_size)
                    # empty_size.append(container.empty_size)
                    # stable_num.append( np.sum(container.stable) * ( network_blocks_num / total_blocks_num) )
                    # packing_height.append(container.bounding_box[-1])
                    # container.draw_container()
                    # container.clear_container()
                    # continue 

                # get real block id
                ptr = ptr.cpu().numpy().astype('int')[0]

                while ptr >= network_blocks_num:
                    ptr -= network_blocks_num
                
                initial_container.remove_block( initial_container.sub_graph_nodes[ptr] )

        my_time.append(time.time() - start)
        container = actor.containers[0]
        
        my_valid_size.append(container.valid_size)
        
        height = np.max(container.heightmap)
        if container.block_dim == 2:
            box_size = container.container_size[0] * height
        elif container.block_dim == 3:
            box_size = container.container_size[0] * container.container_size[1] * height 

        my_box_size.append(box_size)
        my_empty_size.append(container.empty_size)
        my_stable_num.append( np.sum(container.stable) * ( network_blocks_num / total_blocks_num) )
        my_packing_height.append(container.bounding_box[-1])

        if i < 6:
            container.draw_container(save_path + '/%d' % i)
            
        actor.containers[0].clear_container()

    np.savetxt( save_path + '/batch-valid_size.txt', my_valid_size)
    np.savetxt( save_path + '/batch-box_size.txt', my_box_size)
    np.savetxt( save_path + '/batch-empty_size.txt', my_empty_size)
    np.savetxt( save_path + '/batch-stable_num.txt', my_stable_num)
    np.savetxt( save_path + '/batch-packing_height.txt', my_packing_height)
    np.savetxt( save_path + '/batch-time.txt', my_time)

def train_pack(args):
    import pack
    
    if args.input_type == 'simple':
        STATIC_SIZE = args.obj_dim
        DYNAMIC_SIZE = args.num_nodes
    elif args.input_type == 'rot':
        STATIC_SIZE = args.obj_dim
        DYNAMIC_SIZE = args.num_nodes
    elif args.input_type == 'bot' or args.input_type == 'bot-rot':
        STATIC_SIZE = args.obj_dim
        DYNAMIC_SIZE = args.num_nodes * 3
    elif args.input_type == 'mul':
        STATIC_SIZE = args.obj_dim
        DYNAMIC_SIZE = args.num_nodes
    elif args.input_type == 'mul-with':
        STATIC_SIZE = args.obj_dim + 1
        DYNAMIC_SIZE = args.num_nodes
    elif args.input_type == 'rot-old':
        STATIC_SIZE = args.obj_dim + 1
        DYNAMIC_SIZE = args.num_nodes + 1
    else:
        print('TRAIN OHHH')

    print('Loading data...')
    use_cuda = args.use_cuda
    size_range = [ args.min_size, args.max_size ]

    if args.obj_dim == 2:
        container_size = [args.container_width, args.container_height]
        initial_container_size = [args.initial_container_width, args.initial_container_height]
    elif args.obj_dim == 3:
        container_size = [args.container_width, args.container_width, args.container_height]
        initial_container_size = [args.initial_container_width, args.initial_container_width, args.initial_container_height]

    containers = [tools.Container(container_size, args.total_blocks_num, args.reward_type, args.heightmap_type, \
                    initial_container_size, packing_strategy=args.packing_strategy)]

    train_file, valid_file = get_dataset(
        args.total_blocks_num,
        args.train_size,
        args.valid_size,
        args.obj_dim,
        args.initial_container_width,
        args.initial_container_height,
        args.arm_size,
        size_range,
        seed=args.seed,
    )

    # if args.just_generate == True:
    #     return
    print(valid_file)

    valid_data = RollingDataset(
        valid_file,
        args.total_blocks_num,
        args.num_nodes,
        args.valid_size,
        args.obj_dim,
        args.seed + 1,
        args.input_type,
        args.heightmap_type,
        args.allow_rot,
        args.container_width,
        args.initial_container_width,
        args.initial_container_height,
        unit=args.unit
    )

    actor = DRL(STATIC_SIZE,
                DYNAMIC_SIZE,
                args.encoder_hidden_size,
                args.decoder_hidden_size,
                args.use_cuda,
                args.input_type,
                args.allow_rot,
                args.container_width,
                args.container_height,
                args.obj_dim,
                args.reward_type,
                args.decoder_input_type,
                args.heightmap_type,
                args.packing_strategy,
                containers,
                pack.update_dynamic,
                pack.update_mask,
                args.num_layers,
                args.dropout,
                args.unit
            )

    if use_cuda:
        actor = actor.cuda()

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path))
        print('Loading pre-train model: ', path)

    kwargs = vars(args)
    kwargs['valid_data'] = valid_data
    kwargs['network_blocks_num'] = args.num_nodes

    validate(actor, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    
    # Task settings
    parser.add_argument('--task', default='pack')
    parser.add_argument('--note', default='rolling-debug-and-test', type=str)
    parser.add_argument('--just_test', default=True, type=str2bool)
    parser.add_argument('--just_generate', default=False, type=str2bool)
    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--cpu_threads', default=0, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--seed', default=12345, type=int)

    # Training/testing settings
    parser.add_argument('--train_size',default=10, type=int)
    parser.add_argument('--valid_size', default=10, type=int)
    parser.add_argument('--epoch_num', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # Data settings
    parser.add_argument('--obj_dim', default=2, type=int)
    parser.add_argument('--gt_data', default=False, type=str2bool)
    parser.add_argument('--mix_data', default=False, type=str2bool)
    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
    # sizes of blocks and containers
    parser.add_argument('--unit', default=1, type=int)
    parser.add_argument('--arm_size', default=1, type=int)
    parser.add_argument('--min_size', default=1, type=int)
    parser.add_argument('--max_size', default=5, type=int)
    parser.add_argument('--container_width', default=5, type=int)
    parser.add_argument('--container_height', default=250, type=int)
    parser.add_argument('--initial_container_width', default=7, type=int)
    parser.add_argument('--initial_container_height', default=250, type=int)

    # Packing settings
    parser.add_argument('--packing_strategy', default='LB_GREEDY', type=str)
    parser.add_argument('--reward_type', default='C+P+S-lb-soft', type=str)

    # Network settings
    # ---- TODO: network reward
    parser.add_argument('--input_type', default='bot', type=str)
    parser.add_argument('--allow_rot', default=True, type=str2bool)
    parser.add_argument('--decoder_input_type', default='shape_heightmap', type=str) # shape_heightmap, shape_only, heightmap_only
    parser.add_argument('--heightmap_type', default='diff', type=str)     # full, zero, diff

    # Network parameters
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--n_process_blocks', default=3, type=int)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--encoder_hidden', dest='encoder_hidden_size', default=128, type=int)
    parser.add_argument('--decoder_hidden', dest='decoder_hidden_size', default=256, type=int)

    # ROLLING
    parser.add_argument('--total_blocks_num', default=20, type=int)

    # parser.add_argument('--use_all_gt', default=False, type=str2bool)

    # parser.add_argument('--use_heightmap', default=True, type=str2bool)
    # parser.add_argument('--only_heightmap', default=False, type=str2bool)
    # parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    if args.cpu_threads != 0:
        torch.set_num_threads(args.cpu_threads)

    if args.task == 'pack':
        print('Reward type:      %s' % args.reward_type)
        print('Input type:       %s' % args.input_type)
        print('Mix data:         %s' % args.mix_data)
        print('Gt  data:         %s' % args.gt_data)
        print('Decoder input:    %s' % args.decoder_input_type)
        print('Heightmap_type:   %s' % args.heightmap_type)
        print('Target container: %s' % args.container_width)
        print('Init container:   %s' % args.initial_container_width)
        print('Unit:             %s' % args.unit)
        print('Packing strategy: %s' % args.packing_strategy)
        print('note:             %s' % args.note)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        train_pack(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)

