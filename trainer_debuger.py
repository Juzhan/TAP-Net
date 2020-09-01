from trainer import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transport and Pack')

    # Task settings
    parser.add_argument('--task', default='test', type=str) # train, test, generate
    parser.add_argument('--note', default='debug', type=str)
    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--seed', default=12345, type=int)

    # Training/testing settings
    parser.add_argument('--train_size',default=13, type=int)
    parser.add_argument('--valid_size', default=10, type=int)
    parser.add_argument('--epoch_num', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # Data settings
    parser.add_argument('--obj_dim', default=2, type=int)
    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
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