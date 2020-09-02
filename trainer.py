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
import matplotlib.pyplot as plt

from model import DRL, DRL_RNN, DRL_L, Encoder

def str2bool(v):
      return v.lower() in ('true', '1')

class criticAttention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(criticAttention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        logit = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        logit = torch.softmax(logit, dim=2)
        return logit 

class realCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size, num_layers=1, n_process_blocks=3, dropout=0.2):
        super(realCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True)

        self.attention = criticAttention(hidden_size)

        self.n_process_blocks = n_process_blocks

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.num_layers = num_layers
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static, dynamic, decoder_input, last_hh=None):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        decoder_hidden = self.decoder(decoder_input)

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        for i in range(self.n_process_blocks):
            prob = self.attention(static_hidden, dynamic_hidden, rnn_out)
            # Given a summary of the output, find an  input context
            context = prob.bmm(static_hidden.permute(0, 2, 1))
            # Calculate the next output using Batch-matrix-multiply ops
            rnn_out = context.squeeze(1)

        output = self.fc(rnn_out)
        return output

def validate(data_loader, actor, save_dir, **kwargs):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if kwargs['render_fn'] is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):
        encoder_static, encoder_dynamic, decoder_static, decoder_dynamic = batch

        if kwargs['use_cuda']:
            encoder_static = encoder_static.cuda()
            encoder_dynamic = encoder_dynamic.cuda()
            decoder_static = decoder_static.cuda()
            decoder_dynamic = decoder_dynamic.cuda()

        # Full forward pass through the dataset
        if kwargs['decoder_input_type'] == 'shape_only':
            decoder_input = decoder_static
        else:
            decoder_input = [decoder_static, decoder_dynamic]

        with torch.no_grad():
            start = time.time()
            tour_indices, tour_logp, pack_logp, reward = actor(encoder_static, encoder_dynamic, decoder_input)
            valid_time = time.time() - start

        reward = reward.mean().item()
        rewards.append(reward)

        if kwargs['render_fn'] is not None:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            kwargs['render_fn'](encoder_static, tour_indices, path, encoder_dynamic, valid_time, **kwargs)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    date = datetime.datetime.now()
    now = '%s' % date.date()
    now += '-%s' % date.hour
    now += '-%s' % date.minute
    now = str(now)
    save_dir = os.path.join('pack', '%d' % kwargs['num_nodes'], 
        str(kwargs['obj_dim']) + 'd-' + kwargs['input_type'] + '-'  + kwargs['reward_type'] + '-' + kwargs['packing_strategy'] + '-width-' + str(kwargs['container_width']) + '-note-' + kwargs['note'] + '-' + now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=kwargs['actor_lr'])
    critic_optim = optim.Adam(critic.parameters(), lr=kwargs['critic_lr'])

    train_loader = DataLoader(kwargs['train_data'], kwargs['batch_size'], shuffle=True, num_workers=0)
    valid_loader = DataLoader(kwargs['valid_data'], len(kwargs['valid_data']), shuffle=False, num_workers=0)

    best_reward = np.inf
    my_rewards = []
    my_losses = []

    train_size = kwargs['train_size']
    log_step = int(train_size / kwargs['batch_size'])
    if log_step > 100:
        log_step = int(100)
    if log_step == 0:
        log_step = int(1)

    for epoch in range(kwargs['epoch_num']):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start
        valid_dir = os.path.join(save_dir, 'render', '%s' % epoch)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)

        for batch_idx, batch in enumerate(train_loader):
            if kwargs['task'] == 'test':
                break
            encoder_static, encoder_dynamic, decoder_static, decoder_dynamic = batch

            use_cuda = kwargs['use_cuda']
            if use_cuda:
                encoder_static = encoder_static.cuda().detach()
                encoder_dynamic = encoder_dynamic.cuda().detach()
                decoder_static = decoder_static.cuda().detach()
                decoder_dynamic = decoder_dynamic.cuda().detach()

            # Full forward pass through the dataset
            if kwargs['decoder_input_type']=='shape_only':
                decoder_input = decoder_static
            else:
                decoder_input = [decoder_static, decoder_dynamic]

            tour_indices, tour_logp, pack_logp, reward = actor(encoder_static, encoder_dynamic, decoder_input)

            # Sum the log probabilities for each city in the tour
            # reward = reward_fn(static, tour_indices, kwargs['reward_type'], kwargs['input_type'], kwargs['allow_rot'], kwargs['container_width'], kwargs['container_height'] )

            # Query the critic for an estimate of the reward
            if kwargs['input_type'] == 'mul-with':
                static_input = encoder_static[:,1:,:]
            elif kwargs['input_type'] == 'rot-old':
                static_input = encoder_static
            else:
                static_input = encoder_static[:,1:1+kwargs['obj_dim'],:]

            critic_x0 = torch.zeros_like(static_input[:,:,0]).unsqueeze(2)
            critic_est = critic(static_input, encoder_dynamic, critic_x0).view(-1)

            advantage = (reward - critic_est)


            if kwargs['packing_strategy'] == 'pre_train' or kwargs['packing_strategy'] == 'none_train':
                actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1) + advantage.detach() * pack_logp.sum(dim=1) )
            else:
                actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))

            # actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), kwargs['max_grad_norm'])
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), kwargs['max_grad_norm'])
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % log_step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-log_step:])
                mean_reward = np.mean(rewards[-log_step:])
                my_rewards.append(mean_reward)
                my_losses.append(mean_loss)

                print('    Epoch %d  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (epoch, batch_idx, len(train_loader), mean_reward, mean_loss, times[-1]))


        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # save rendering of validation set tours
        valid_dir = os.path.join(save_dir, 'render', '%s' % epoch)
        mean_valid = validate(
            valid_loader, 
            actor,
            valid_dir,
            **kwargs
        )

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Epoch %d,  mean epoch reward: %2.4f    valid: %2.4f, loss: %2.4f, took: %2.4fs '\
              '(%2.4fs / %d batches)' % \
              (epoch, mean_reward, mean_loss, mean_valid, time.time() - epoch_start,
              np.mean(times), log_step  ))

        import matplotlib.pyplot as plt
        plt.close('all')
        plt.title('Reward')
        plt.plot(range(len(my_rewards)), my_rewards, '-')
        plt.savefig(save_dir + '/reward.png' , bbox_inches='tight', dpi=400)

    np.savetxt(save_dir + '/reawrds.txt', my_rewards)
    np.savetxt(save_dir + '/losses.txt', my_losses)
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.title('Reward')
    # try:
    #     xticks = [ i for i in range( 0, len(my_rewards),  log_step ) ]
    #     xlabels = [ str(i* 10 * log_step / int(train_size / kwargs['batch_size']) ) for i,_ in enumerate(xticks)  ]
    #     plt.xticks(xticks, xlabels)
    # except:
    #     pass
    plt.plot(range(len(my_rewards)), my_rewards, '-')
    plt.savefig(save_dir + '/reward.png', bbox_inches='tight', dpi=400)

def train_pack(**kwargs):

    import pack as pack
    from pack import PACKDataset
    
    if kwargs['input_type'] == 'simple':
        STATIC_SIZE = kwargs['obj_dim']
        DYNAMIC_SIZE = kwargs['num_nodes']
    elif kwargs['input_type'] == 'rot':
        STATIC_SIZE = kwargs['obj_dim']
        DYNAMIC_SIZE = kwargs['num_nodes']
    elif kwargs['input_type'] == 'bot' or kwargs['input_type'] == 'bot-rot':
        STATIC_SIZE = kwargs['obj_dim']
        DYNAMIC_SIZE = kwargs['num_nodes'] * 3
    elif kwargs['input_type'] == 'use-static' or kwargs['input_type'] == 'use-pnet':
        STATIC_SIZE = kwargs['obj_dim']
        DYNAMIC_SIZE = kwargs['num_nodes'] * 3
    elif kwargs['input_type'] == 'mul':
        STATIC_SIZE = kwargs['obj_dim']
        DYNAMIC_SIZE = kwargs['num_nodes'] * 3
    elif kwargs['input_type'] == 'mul-with':
        STATIC_SIZE = kwargs['obj_dim'] + 1
        DYNAMIC_SIZE = kwargs['num_nodes'] * 3
    elif kwargs['input_type'] == 'rot-old':
        STATIC_SIZE = kwargs['obj_dim'] + 1
        DYNAMIC_SIZE = kwargs['num_nodes'] + 1
    else:
        print('TRAIN OHHH')

    print('Loading data...')
    use_cuda = kwargs['use_cuda']

    size_range = [ kwargs['min_size'], kwargs['max_size'] ]

    if kwargs['dataset'] == 'MIX':
        train_file_1, train_file_2, valid_file_1, valid_file_2 = pack.get_mix_dataset(
            kwargs['num_nodes'],
            kwargs['train_size'],
            kwargs['valid_size'],
            kwargs['obj_dim'],
            kwargs['initial_container_width'],
            kwargs['initial_container_height'],
            size_range,
            kwargs['seed']
        )
    elif kwargs['dataset'] == 'PPSG':
        train_file, valid_file = pack.create_dataset_gt(
            kwargs['num_nodes'],
            kwargs['train_size'],
            kwargs['valid_size'],
            kwargs['obj_dim'],
            kwargs['container_width'],
            kwargs['container_height'],
            kwargs['initial_container_width'],
            kwargs['initial_container_height'],
            kwargs['input_type'],
            kwargs['arm_size'],
            size_range,
            kwargs['seed']
        )    
    else:
        train_file, valid_file = pack.create_dataset(
            kwargs['num_nodes'],
            kwargs['train_size'],
            kwargs['valid_size'],
            kwargs['obj_dim'],
            kwargs['initial_container_width'],
            kwargs['initial_container_height'],
            kwargs['arm_size'],
            size_range,
            seed=kwargs['seed'],
        )

    if kwargs['dataset'] == 'MIX':
        print(train_file_1, train_file_2)
        print(valid_file_1, valid_file_2)
    else:
        print(train_file)
        print(valid_file)

    # end here if the task is generate
    if kwargs['task'] == 'generate':
        return

    if kwargs['dataset'] == 'MIX':
        train_data = PACKDataset(
            train_file_1,
            kwargs['num_nodes'],
            kwargs['train_size'],
            kwargs['seed'],
            kwargs['input_type'],
            kwargs['heightmap_type'],
            kwargs['allow_rot'],
            kwargs['container_width'],
            mix_data_file=train_file_2,
            unit=kwargs['unit'],
            no_precedence=kwargs['no_precedence']
            )

        valid_data = PACKDataset(
            valid_file_1,
            kwargs['num_nodes'],
            kwargs['valid_size'],
            kwargs['seed'] + 1,
            kwargs['input_type'],
            kwargs['heightmap_type'],
            kwargs['allow_rot'],
            kwargs['container_width'],
            mix_data_file=valid_file_2,
            unit=kwargs['unit'],
            no_precedence=kwargs['no_precedence']
            )
    else:
        train_data = PACKDataset(
            train_file,
            kwargs['num_nodes'],
            kwargs['train_size'],
            kwargs['seed'],
            kwargs['input_type'],
            kwargs['heightmap_type'],
            kwargs['allow_rot'],
            kwargs['container_width'],
            unit=kwargs['unit'],
            no_precedence=kwargs['no_precedence']
            )

        valid_data = PACKDataset(
            valid_file,
            kwargs['num_nodes'],
            kwargs['valid_size'],
            kwargs['seed'] + 1,
            kwargs['input_type'],
            kwargs['heightmap_type'],
            kwargs['allow_rot'],
            kwargs['container_width'],
            unit=kwargs['unit'],
            no_precedence=kwargs['no_precedence']
            )

    if kwargs['reward_type'] == 'C+P+S-G-soft' or kwargs['reward_type'] == 'C+P+S-LG-soft':
        network = DRL_RNN
    elif kwargs['reward_type'] == 'C+P+S-SL-soft' or kwargs['reward_type'] == 'C+P+S-RL-soft':
        network = DRL_L
    else:
        network = DRL

    actor = network(STATIC_SIZE,
                DYNAMIC_SIZE,
                kwargs['encoder_hidden_size'],
                kwargs['decoder_hidden_size'],
                kwargs['use_cuda'],
                kwargs['input_type'],
                kwargs['allow_rot'],
                kwargs['container_width'],
                kwargs['container_height'],
                kwargs['obj_dim'],
                kwargs['reward_type'],
                kwargs['decoder_input_type'],
                kwargs['heightmap_type'],
                kwargs['packing_strategy'],
                pack.update_dynamic,
                pack.update_mask,
                kwargs['num_layers'],
                kwargs['dropout'],
                kwargs['unit']
                )

    critic = realCritic(STATIC_SIZE, DYNAMIC_SIZE, kwargs['encoder_hidden_size'], kwargs['num_layers'],
                                                     kwargs['n_process_blocks'], kwargs['dropout'])

    if use_cuda:
        actor = actor.cuda()
        critic = critic.cuda()


    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = pack.reward
    kwargs['render_fn'] = pack.render

    if kwargs['checkpoint']:
        path = os.path.join(kwargs['checkpoint'], 'actor.pt')
        actor.load_state_dict(torch.load(path))

        path = os.path.join(kwargs['checkpoint'], 'critic.pt')
        critic.load_state_dict(torch.load(path))

        print('Loading pre-train model', path)

    train(actor, critic, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transport and Pack')

    # Task settings
    parser.add_argument('--task', default='test', type=str) # train, test, generate
    parser.add_argument('--note', default='debug', type=str)
    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--cpu_threads', default=0, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--seed', default=12345, type=int)

    # Training/testing settings
    parser.add_argument('--train_size',default=13, type=int)
    parser.add_argument('--valid_size', default=10, type=int)
    parser.add_argument('--epoch_num', default=2, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # Data settings
    parser.add_argument('--obj_dim', default=2, type=int)
    parser.add_argument('--nodes', dest='num_nodes', default=6, type=int)
    parser.add_argument('--total_obj_num', default=10, type=int)            # if more, do Rolling @TODO
    parser.add_argument('--dataset', default='RAND', type=str)              # RAND, PPSG, MIX
    # sizes of blocks and containers
    parser.add_argument('--unit', default=1.0, type=float)
    parser.add_argument('--arm_size', default=1, type=int) # size of robotic arm to pass and rotate a block
    parser.add_argument('--min_size', default=1, type=int)
    parser.add_argument('--max_size', default=5, type=int)
    parser.add_argument('--container_width', default=5, type=int)
    parser.add_argument('--container_length', default=5, type=int)  # for 3D
    parser.add_argument('--container_height', default=50, type=int)
    parser.add_argument('--initial_container_width', default=7, type=int)
    parser.add_argument('--initial_container_length', default=7, type=int)  # for 3D
    parser.add_argument('--initial_container_height', default=50, type=int)

    # Packing settings
    parser.add_argument('--packing_strategy', default='LB_GREEDY', type=str)
    parser.add_argument('--reward_type', default='C+P+S-lb-soft', type=str)

    # Network settings
    # ---- TODO: network reward
    parser.add_argument('--input_type', default='bot', type=str)
    parser.add_argument('--allow_rot', default=True, type=str2bool)
    parser.add_argument('--decoder_input_type', default='shape_heightmap', type=str) # shape_heightmap, shape_only, heightmap_only
    parser.add_argument('--heightmap_type', default='diff', type=str)     # full, zero, diff
    parser.add_argument('--no_precedence', default=False, type=str2bool)    # if true, set all deps to 0 

    # Network parameters
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--n_process_blocks', default=3, type=int)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--encoder_hidden', dest='encoder_hidden_size', default=128, type=int)
    parser.add_argument('--decoder_hidden', dest='decoder_hidden_size', default=256, type=int)


    args = parser.parse_args()

    if args.cpu_threads != 0:
        torch.set_num_threads(args.cpu_threads)

    print('Reward type:      %s' % args.reward_type)
    print('Input type:       %s' % args.input_type)
    print('Dataset:          %s' % args.dataset)
    print('Decoder input:    %s' % args.decoder_input_type)
    print('Heightmap_type:   %s' % args.heightmap_type)
    print('Target container: %s' % args.container_width)
    print('Init container:   %s' % args.initial_container_width)
    print('Packing strategy: %s' % args.packing_strategy)
    print('note:             %s' % args.note)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    kwargs = vars(args)
    train_pack(**kwargs)