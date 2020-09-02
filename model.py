import torch
import torch.nn as nn
import torch.nn.functional as F
import tools
import math
import numpy as np
from pack_net import LG_RL

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

    def __init__(self, input_size, hidden_size, map_size):
        super(HeightmapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, int(hidden_size/4), stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(int(hidden_size/4), int(hidden_size/2), stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(int(hidden_size/2), int(hidden_size), kernel_size=( math.ceil(map_size[0]/4), math.ceil(map_size[1]/4) ) )

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

        # if input_type == 'use-static' or input_type == 'use-pnet':
        #     self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, encoder_hidden_size + decoder_hidden_size), requires_grad=True))
        # else:
        self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, 2 * encoder_hidden_size + decoder_hidden_size), requires_grad=True))

        self.decoder_input_type = decoder_input_type
        self.input_type = input_type

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        # if self.input_type == 'use-static' or self.input_type == 'use-pnet':
        #     encoder_hidden = static_hidden
        # else:
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

        # if self.input_type == 'use-static' or self.input_type == 'use-pnet':
        #     self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, decoder_hidden_size + encoder_hidden_size), requires_grad=True))
        # else:
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
        
        # if self.input_type == 'use-static' or self.input_type == 'use-pnet':
        #     encoder_hidden = static_hidden
        # else:
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
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, encoder_hidden_size, decoder_hidden_size, 
                use_cuda, input_type, allow_rot, container_width, container_height, block_dim, 
                reward_type, decoder_input_type, heightmap_type, packing_strategy, 
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
            if block_dim==3: heightmap_length = container_width * unit
        
        heightmap_width = math.ceil(heightmap_width)
        if block_dim==3: heightmap_length = math.ceil(heightmap_length)

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

        self.pointer = Pointer(encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type, num_layers, dropout)

        if input_type == 'use-pnet':
            self.encoder_RNN = nn.GRU(encoder_hidden_size*2, encoder_hidden_size*2, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)        

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
        # if unit < 1: 
        self.container_width = math.ceil(container_width * unit)
        # else: 
        #     self.container_width = container_width * unit
        self.container_height = container_height
        self.decoder_input_type = decoder_input_type
        self.heightmap_type = heightmap_type
        self.packing_strategy = packing_strategy
        # Used as a proxy initial state in the decoder when not specified

    def forward(self, static, dynamic, decoder_input, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, _, sequence_size = static.size()

        if self.allow_rot == False:
            rotate_types = 1
        else:
            if self.block_dim == 2:
                rotate_types = 2
            elif self.block_dim == 3:
                rotate_types = 6
        blocks_num = int(dynamic.shape[-1] / rotate_types)

        if self.block_dim == 3:
            container_size = [self.container_width, self.container_width, self.container_height]
        else:
            container_size = [self.container_width, self.container_height]
        if self.input_type == 'mul' or self.input_type == 'mul-with':
            if self.block_dim == 3:
                container_size_a = [self.container_width, self.container_width, self.container_height]
                container_size_b = container_size_a
            else:
                container_size_a = [self.container_width, self.container_height]
                container_size_b = container_size_a

        if self.input_type == 'mul' or self.input_type == 'mul-with':
            containers_a = [tools.Container(container_size_a, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]
            containers_b = [tools.Container(container_size_b, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]
        else:
            containers = [tools.Container(container_size, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]

        # Always use a mask - if no function is provided, we don't update it
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

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        dynamic_hidden = self.dynamic_encoder(dynamic)

        if self.input_type == 'mul':
            static_hidden = self.static_encoder(static[:,1:-1,:])
        elif self.input_type == 'rot-old':
            static_hidden = self.static_encoder(static)
        elif self.input_type == 'use-static':
            static_hidden = self.static_encoder(static[:,1:,:])
        elif self.input_type == 'use-pnet':
            static_hidden = self.static_encoder(static[:,1:,:])
            # batch_size x dim_num x encoder_hidden_size
            static_hidden = static_hidden.transpose(2, 1)
            dynamic_hidden = dynamic_hidden.transpose(2, 1)
            # RNN for pointer
            encoder_hidden, last_hh = self.encoder_RNN( torch.cat( (static_hidden, dynamic_hidden), dim=2 ) )
            static_hidden = encoder_hidden[:, :, :self.encoder_hidden_size]
            dynamic_hidden = encoder_hidden[:, :, self.encoder_hidden_size:]

            static_hidden = static_hidden.transpose(2, 1)
            dynamic_hidden = dynamic_hidden.transpose(2, 1)
        else:
            static_hidden = self.static_encoder(static[:,1:,:])

        if 'heightmap' in self.decoder_input_type:
            decoder_static, decoder_dynamic = decoder_input

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

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + current_mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest

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
                if self.input_type == 'use-static' or self.input_type == 'use-pnet':
                    pass            
                else:
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

            if self.input_type == 'mul' or self.input_type == 'mul-with':
                target_ids = static[:,-1,:]

                target_ids = torch.gather( target_ids, 1,
                            ptr.view(-1, 1)
                            .expand(-1, 1)).detach()

            if 'heightmap' in self.decoder_input_type:
                decoder_static = torch.gather( static_part, 2,
                                ptr.view(-1, 1, 1)
                                .expand(-1, self.static_size, 1)).detach()   
                is_rotate = (ptr < blocks_num).cpu().numpy().astype('bool')

                if self.input_type == 'mul-with':
                    blocks = decoder_static.transpose(2,1).squeeze(1).cpu().numpy()[:,:self.block_dim]
                else:
                    blocks = decoder_static.transpose(2,1).squeeze(1).cpu().numpy()
                
                if self.input_type == 'mul' or self.input_type == 'mul-with':                    
                    # now get the selected blocks and update heightmap
                    heightmaps_a = []
                    heightmaps_b = []
                
                    for batch_index in range(batch_size):
                        target_id = target_ids[batch_index]

                        if target_id == 0:
                            heightmaps_a.append( containers_a[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] ) )
                            heightmaps_b.append( containers_b[batch_index].get_heightmap() )
                        elif target_id == 1:
                            heightmaps_a.append( containers_a[batch_index].get_heightmap() )
                            heightmaps_b.append( containers_b[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] ) )

                    if self.block_dim == 2:
                        if self.use_cuda:
                            heightmaps_a = torch.FloatTensor(heightmaps_a).cuda().unsqueeze(2)
                            heightmaps_b = torch.FloatTensor(heightmaps_b).cuda().unsqueeze(2)
                        else:
                            heightmaps_a = torch.FloatTensor(heightmaps_a).unsqueeze(2)
                            heightmaps_b = torch.FloatTensor(heightmaps_b).unsqueeze(2)
                        decoder_dynamic = torch.cat( (heightmaps_a, heightmaps_b), 1 )
                    elif self.block_dim == 3:
                        if self.use_cuda:
                            heightmaps_a = torch.FloatTensor(heightmaps_a).cuda()#.unsqueeze(1)
                            heightmaps_b = torch.FloatTensor(heightmaps_b).cuda()#.unsqueeze(1)
                        else:
                            heightmaps_a = torch.FloatTensor(heightmaps_a)#.unsqueeze(1)
                            heightmaps_b = torch.FloatTensor(heightmaps_b)#.unsqueeze(1)
                        if self.heightmap_type != 'diff':
                            heightmaps_a = heightmaps_a.unsqueeze(1)
                            heightmaps_b = heightmaps_b.unsqueeze(1)
                        decoder_dynamic = torch.cat( (heightmaps_a, heightmaps_b), 1 )

                else:
                    # now get the selected blocks and update heightmap
                    heightmaps = []
                    for batch_index in range(batch_size):
                        heightmaps.append(containers[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] ))
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
                if self.input_type == 'mul-with':
                    blocks = decoder_input.transpose(2,1).squeeze(1).cpu().numpy()[:,:self.block_dim]
                else:
                    blocks = decoder_input.transpose(2,1).squeeze(1).cpu().numpy()
                
                if self.input_type == 'mul' or self.input_type == 'mul-with':                    
                    # now get the selected blocks and update heightmap
                    heightmaps_a = []
                    heightmaps_b = []
                
                    for batch_index in range(batch_size):
                        target_id = target_ids[batch_index]
                        if target_id == 0:
                            containers_a[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] )
                        elif target_id == 1:
                            containers_b[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] )
                else:
                    for batch_index in range(batch_size):
                        containers[batch_index].add_new_block(blocks[batch_index], is_rotate[batch_index] )

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

        # now we can return the reward at the same time
        scores = torch.zeros(batch_size).detach()
        if self.use_cuda:
            scores = scores.cuda()

        if self.input_type == 'mul' or self.input_type == 'mul-with':
            for batch_index in range(batch_size):
                scores[batch_index] += containers_a[batch_index].calc_ratio()
                scores[batch_index] += containers_b[batch_index].calc_ratio()
                scores[batch_index] /= 2.0
        else:
            for batch_index in range(batch_size):
                scores[batch_index] = containers[batch_index].calc_ratio()
        
        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp, None, -scores


class DRL_RNN(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """
    def __init__(self, static_size, dynamic_size, encoder_hidden_size, decoder_hidden_size, 
                use_cuda, input_type, allow_rot, container_width, container_height, block_dim, 
                reward_type, decoder_input_type, heightmap_type, packing_strategy, 
                update_fn, mask_fn, num_layers=1, dropout=0., unit=1):
 
        super(DRL_RNN, self).__init__()

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
            if block_dim==3: heightmap_length = container_width * unit
        
        heightmap_width = math.ceil(heightmap_width)
        if block_dim==3: heightmap_length = math.ceil(heightmap_length)

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

        self.pointer = Pointer(encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type, num_layers, dropout)



        if input_type == 'use-pnet':
            self.encoder_RNN = nn.GRU(encoder_hidden_size, encoder_hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)        

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
        # Used as a proxy initial state in the decoder when not specified
        self.packing_strategy = packing_strategy

        if self.reward_type == 'C+P+S-G-soft':
            self.pack_net = LG_RL.PackRNN(2, 128, container_width, 128, container_width, container_height, heightmap_type, pack_net_type='G')
            if packing_strategy[:3] == 'pre':
                self.pack_net.load_state_dict(torch.load('./pack_net/G_rand_diff/checkpoints/199/actor.pt'))
                print('pre')
            if packing_strategy[-4:] == 'eval':
                print('eval')
                self.pack_net.eval()
            else:
                print('train')
            
        elif self.reward_type == 'C+P+S-LG-soft':
            self.pack_net = LG_RL.PackRNN(2, 128, container_width, 128, container_width, container_height, heightmap_type, pack_net_type='LG')
            print('LG')
            if packing_strategy[:3] == 'pre':
                self.pack_net.load_state_dict(torch.load('./pack_net/LG_rand_diff/checkpoints/199/actor.pt'))
                print('pre')
            if packing_strategy[-4:] == 'eval':
                self.pack_net.eval()
                print('eval')
            else:
                print('train')
        else:
            print('========> Error in DRL_RNN')            

    def forward(self, static, dynamic, decoder_input, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, _, sequence_size = static.size()

        if self.allow_rot == False:
            rotate_types = 1
        else:
            if self.block_dim == 2:
                rotate_types = 2
            elif self.block_dim == 3:
                rotate_types = 6
        blocks_num = int(dynamic.shape[-1] / rotate_types)

        if self.block_dim == 3:
            container_size = [self.container_width, self.container_width, self.container_height]
        else:
            container_size = [self.container_width, self.container_height]

        if self.input_type == 'mul' or self.input_type == 'mul-with':
            if self.block_dim == 3:
                container_size_a = [self.container_width, self.container_width, self.container_height]
                container_size_b = container_size_a
            else:
                container_size_a = [self.container_width, self.container_height]
                container_size_b = container_size_a

        # Always use a mask - if no function is provided, we don't update it
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

        # Structures for holding the output sequences
        tour_idx, tour_logp, pack_logp = [], [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        dynamic_hidden = self.dynamic_encoder(dynamic)

        if self.input_type == 'mul':
            encoder_static = static[:,1:-1,:]
            static_hidden = self.static_encoder(static[:,1:-1,:])
        elif self.input_type == 'rot-old':
            encoder_static = static
            static_hidden = self.static_encoder(static)
        elif self.input_type == 'use-static':
            encoder_static = static[:, 1:, :]
            static_hidden = self.static_encoder(static[:,1:,:])
        elif self.input_type == 'use-pnet':
            encoder_static = static[:, 1:, :]
            static_hidden = self.static_encoder(static[:,1:,:])
            # batch_size x dim_num x encoder_hidden_size
            static_hidden = static_hidden.transpose(2, 1)
            # RNN for pointer
            static_hidden, last_hh = self.encoder_RNN(static_hidden)
            static_hidden = static_hidden.transpose(2, 1)
        else:
            encoder_static = static[:, 1:, :]
            static_hidden = self.static_encoder(static[:,1:,:])

        if 'heightmap' in self.decoder_input_type:
            decoder_static, decoder_dynamic = decoder_input

        # if self.heightmap_type == 'diff':
        #     decoder_dynamic = decoder_dynamic[:,:-1, :]

        self.P = None
        self.C = None
        self.S = None

        all_blocks = [] * batch_size
        # all_rotate = [] * batch_size
        # all_order = [] * batch_size

        for current_block_num in range(max_steps):

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

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + current_mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest

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
                if self.input_type == 'use-static' or self.input_type == 'use-pnet':
                    pass            
                else:
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

            if self.input_type == 'mul' or self.input_type == 'mul-with':
                target_ids = static[:,-1,:]

                target_ids = torch.gather( target_ids, 1,
                            ptr.view(-1, 1)
                            .expand(-1, 1)).detach()

                
            # if self.reward_type == 'C+P+S-LG-soft' or 'C+P+S-G-soft':
            decoder_static = torch.gather( static_part, 2,
                            ptr.view(-1, 1, 1)
                            .expand(-1, self.static_size, 1)) #.detach()
            # pack current blocks

            # TODO multi ?
            is_rotate = (ptr < blocks_num).cpu().numpy().astype('bool')
            blocks = decoder_static.detach().transpose(2,1).squeeze(1).cpu().numpy()

            all_blocks.append(decoder_static)
            pack_blocks = torch.cat(all_blocks, dim=-1)
            positions, place_logp, scores = self.pack_net( pack_blocks, current_block_num+1 )
            
            # update in container
            heightmaps = []
            
            for batch_index in range(batch_size):
                heightmaps.append( self.pack_net.engines[batch_index].get_heightap(self.heightmap_type) )
                # heightmaps.append(containers[batch_index].add_new_block_at(blocks[batch_index], place_pos[batch_index], is_rotate[batch_index] ))
            if self.block_dim == 2:
                if self.use_cuda:
                    decoder_dynamic = torch.FloatTensor(heightmaps).cuda().unsqueeze(2)
                else:
                    decoder_dynamic = torch.FloatTensor(heightmaps).unsqueeze(2)
            elif self.block_dim == 3:
                if self.use_cuda:
                    decoder_dynamic = torch.FloatTensor(heightmaps).cuda().unsqueeze(1)
                else:
                    decoder_dynamic = torch.FloatTensor(heightmaps).unsqueeze(1)


            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))


        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
        # pack_logp = torch.cat(pack_logp, dim=1)  # (batch_size, seq_len)
        pack_logp = place_logp  # (batch_size, seq_len)
        # return None
        return tour_idx, tour_logp, pack_logp, scores.detach()


class DRL_L(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, encoder_hidden_size, decoder_hidden_size, 
                use_cuda, input_type, allow_rot, container_width, container_height, block_dim, 
                reward_type, decoder_input_type, heightmap_type, packing_strategy, 
                update_fn, mask_fn, num_layers=1, dropout=0., unit=1.0):
        super(DRL_L, self).__init__()

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
            if block_dim==3: heightmap_length = container_width * unit
        
        heightmap_width = math.ceil(heightmap_width)
        if block_dim==3: heightmap_length = math.ceil(heightmap_length)

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

        self.pointer = Pointer(encoder_hidden_size, decoder_hidden_size, decoder_input_type, input_type, num_layers, dropout)

        if input_type == 'use-pnet':
            self.encoder_RNN = nn.GRU(encoder_hidden_size, encoder_hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)        

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

        self.heightmap_type = heightmap_type
        self.decoder_input_type = decoder_input_type
        self.packing_strategy = packing_strategy

        if reward_type == 'C+P+S-RL-soft':          
            print('RL')
            if heightmap_type == 'diff':
                self.pack_net = tools.DQN(self.container_width, is_diff_height=True)
                if packing_strategy[:3] == 'pre':
                    self.pack_net.load_state_dict(torch.load('./pack_net/RL_rand_diff/checkpoints/199/actor.pt'))
                if packing_strategy[-4:] == 'eval':
                    print('eval')
                    self.pack_net.eval()
            elif heightmap_type == 'full':
                self.pack_net = tools.DQN(self.container_width, is_diff_height=False)
                if packing_strategy[:3] == 'pre':
                    self.pack_net.load_state_dict(torch.load('./pack_net/RL_rand_full/checkpoints/199/actor.pt'))
                if packing_strategy[-4:] == 'eval':
                    print('eval')
                    self.pack_net.eval()
            elif heightmap_type == 'zero':
                self.pack_net = tools.DQN(self.container_width, is_diff_height=False)
                if packing_strategy[:3] == 'pre':
                    self.pack_net.load_state_dict(torch.load('./pack_net/RL_rand_zero/checkpoints/199/actor.pt'))
                if packing_strategy[-4:] == 'eval':
                    print('eval')
                    self.pack_net.eval()
            else:
                print('=====> Error in DRL_one_step init pack')
            print('one  step RL')
        else:
            print('SL')
            if heightmap_type == 'diff':
                self.pack_net = tools.DQN(self.container_width, is_diff_height=True)
                if packing_strategy[:3] == 'pre':
                    self.pack_net.load_state_dict(torch.load('./pack_net/SL_rand_diff/checkpoints/199/SL.pt'))
                if packing_strategy[-4:] == 'eval':
                    print('eval')
                    self.pack_net.eval()
            elif heightmap_type == 'full':
                self.pack_net = tools.DQN(self.container_width, is_diff_height=False)
                if packing_strategy[:3] == 'pre':
                    self.pack_net.load_state_dict(torch.load('./pack_net/SL_rand_full/checkpoints/199/SL.pt'))
                if packing_strategy[-4:] == 'eval':
                    print('eval')
                    self.pack_net.eval()
            elif heightmap_type == 'zero':
                self.pack_net = tools.DQN(self.container_width, is_diff_height=False)
                if packing_strategy[:3] == 'pre':
                    self.pack_net.load_state_dict(torch.load('./pack_net/SL_rand_zero/checkpoints/199/SL.pt'))
                if packing_strategy[-4:] == 'eval':
                    print('eval')
                    self.pack_net.eval()
            else:
                print('=====> Error in DRL_L init pack')

    def forward(self, static, dynamic, decoder_input, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, _, sequence_size = static.size()

        if self.allow_rot == False:
            rotate_types = 1
        else:
            if self.block_dim == 2:
                rotate_types = 2
            elif self.block_dim == 3:
                rotate_types = 6
        blocks_num = int(dynamic.shape[-1] / rotate_types)

        if self.block_dim == 3:
            container_size = [self.container_width, self.container_width, self.container_height]
        else:
            container_size = [self.container_width, self.container_height]

        if self.input_type == 'mul' or self.input_type == 'mul-with':
            if self.block_dim == 3:
                container_size_a = [self.container_width, self.container_width, self.container_height]
                container_size_b = container_size_a
            else:
                container_size_a = [self.container_width, self.container_height]
                container_size_b = container_size_a

        if self.input_type == 'mul' or self.input_type == 'mul-with':
            containers_a = [tools.Container(container_size_a, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]
            containers_b = [tools.Container(container_size_b, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]
        else:
            containers = [tools.Container(container_size, blocks_num, self.reward_type, self.heightmap_type, packing_strategy=self.packing_strategy) for _ in range(batch_size)]

        # Always use a mask - if no function is provided, we don't update it
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

        # Structures for holding the output sequences
        tour_idx, tour_logp, pack_logp = [], [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        dynamic_hidden = self.dynamic_encoder(dynamic)

        if self.input_type == 'mul':
            static_hidden = self.static_encoder(static[:,1:-1,:])
        elif self.input_type == 'rot-old':
            static_hidden = self.static_encoder(static)
        elif self.input_type == 'use-static':
            static_hidden = self.static_encoder(static[:,1:,:])
        elif self.input_type == 'use-pnet':
            static_hidden = self.static_encoder(static[:,1:,:])
            # batch_size x dim_num x encoder_hidden_size
            static_hidden = static_hidden.transpose(2, 1)
            # RNN for pointer
            static_hidden, last_hh = self.encoder_RNN(static_hidden)
            static_hidden = static_hidden.transpose(2, 1)
        else:
            static_hidden = self.static_encoder(static[:,1:,:])

        if 'heightmap' in self.decoder_input_type:
            decoder_static, decoder_dynamic = decoder_input

        # if self.heightmap_type == 'diff':
        #     decoder_dynamic = decoder_dynamic[:,:-1, :]

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

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + current_mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest

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
                if self.input_type == 'use-static' or self.input_type == 'use-pnet':
                    pass            
                else:
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

            if self.input_type == 'mul' or self.input_type == 'mul-with':
                target_ids = static[:,-1,:]

                target_ids = torch.gather( target_ids, 1,
                            ptr.view(-1, 1)
                            .expand(-1, 1)).detach()

            if self.reward_type == 'C+P+S-SL-soft' or 'C+P+S-RL-soft':
                
                # if self.use_heightmap: TODO
                decoder_static = torch.gather( static_part, 2,
                                ptr.view(-1, 1, 1)
                                .expand(-1, self.static_size, 1)) #.detach()

                # get heightmap
                heightmaps = []
                for batch_index in range(batch_size):
                    heightmaps.append(containers[batch_index].heightmap )

                if self.block_dim == 2:
                    if self.use_cuda:
                        pack_net_heightmap = torch.FloatTensor(heightmaps).cuda().unsqueeze(2).transpose(2, 1)
                    else:
                        pack_net_heightmap = torch.FloatTensor(heightmaps).unsqueeze(2).transpose(2, 1)
                elif self.block_dim == 3:
                    if self.use_cuda:
                        pack_net_heightmap = torch.FloatTensor(heightmaps).cuda().unsqueeze(1)
                    else:
                        pack_net_heightmap = torch.FloatTensor(heightmaps).unsqueeze(1)
                # get net place position
                if self.heightmap_type == 'diff':
                    tmp = pack_net_heightmap.clone()
                    tmp[:, :, :-1] = tmp[:, :, 1:]
                    tmp[:, :, -1] = pack_net_heightmap[:, :, -1]
                    pack_net_heightmap = tmp - pack_net_heightmap
                    # pack_net_heightmap = pack_net_heightmap[:, :, :-1]

                elif self.heightmap_type == 'zero':
                    pack_net_heightmap -= torch.min(pack_net_heightmap, dim=-1)[0].unsqueeze(-1).expand(-1, 1, self.container_width).float()
                
                pack_pos_porb = self.pack_net( pack_net_heightmap, decoder_static.transpose(2, 1))                

                place_pos = pack_pos_porb.data.max(1)[1]
                place_prob = pack_pos_porb.max(1)[0]
                place_logp = place_prob.log()

                # TODO multi ?
                is_rotate = (ptr < blocks_num).cpu().numpy().astype('bool')
                blocks = decoder_static.detach().transpose(2,1).squeeze(1).cpu().numpy()
                # update in container
                heightmaps = []
                
                for batch_index in range(batch_size):
                    heightmaps.append(containers[batch_index].add_new_block_at(blocks[batch_index], place_pos[batch_index], is_rotate[batch_index] ))
                if self.block_dim == 2:
                    if self.use_cuda:
                        decoder_dynamic = torch.FloatTensor(heightmaps).cuda().unsqueeze(2)
                    else:
                        decoder_dynamic = torch.FloatTensor(heightmaps).unsqueeze(2)
                elif self.block_dim == 3:
                    if self.use_cuda:
                        decoder_dynamic = torch.FloatTensor(heightmaps).cuda().unsqueeze(1)
                    else:
                        decoder_dynamic = torch.FloatTensor(heightmaps).unsqueeze(1)

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            pack_logp.append(place_logp.unsqueeze(1))

        # now we can return the reward at the same time
        # scores = torch.zeros(batch_size).detach()
        scores = np.zeros(batch_size)

        if self.input_type == 'mul' or self.input_type == 'mul-with':
            for batch_index in range(batch_size):
                scores[batch_index] += containers_a[batch_index].calc_ratio()
                scores[batch_index] += containers_b[batch_index].calc_ratio()
                scores[batch_index] /= 2.0
        else:
            for batch_index in range(batch_size):
                scores[batch_index] = containers[batch_index].calc_ratio()
                # containers[batch_index].draw_container('./img/%s' % batch_index)
                
        scores = torch.from_numpy( scores.astype('float32') ).detach()
        if self.use_cuda:
            scores = scores.cuda()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
        pack_logp = torch.cat(pack_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp, pack_logp, -scores



if __name__ == '__main__':
    raise Exception('Cannot be called from main')
