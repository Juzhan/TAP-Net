# TAP-Net
TAP-Net: Transport-and-Pack using Reinforcement Learning

![overview](./doc/overview.png)


### Introduction

We introduce the transport-and-pack(TAP) problem, a frequently encountered instance of real-world packing, and develop a neural optimization solution based on reinforcement learning. 
Given an initial spatial configuration of boxes, we seek an efficient method to iteratively transport and pack the boxes compactly into a target container. Due to obstruction and accessibility constraints, our problem has to add a new search dimension, i.e., finding an optimal em transport sequence, to the already immense search space for packing alone. Using a learning-based approach, a trained network can learn and encode solution patterns to guide the solution of new problem instances instead of executing an expensive online search. In our work, we represent the transport constraints using a precedence graph and train a neural network, coined TAP-Net, using reinforcement learning to reward efficient and stable packing. 

For more details, please refer to our [paper]().

![network](./doc/network.png)

### Usage

### File structure:

    data/
    pack/
    compare/    <--- place to draw the table
    p/          <--- some models we trained before

    tools.py
    generate.py
    model.py
    pack.py
    train.py

`tools.py`: consists of 3 parts: painting functions, packing functions and feasibility functions

`generate.py`: use to generate training data

`model.py`: define the network sturcture

`pack.py`: update dymanic data and calculate reward value

`train.py`: the main code of this work, loading data and train the network, we set some args to contorll the training

    some important args
    
        --nodes: the number of blocks
        --batch_size: batch size, default as 128
        --train_size: the number of training data
        --valid_size: the number of testing data
        --epoch_num: epoch num
        --checkpoint: the location of pre-trained network model, if the path for a model of network is "./netwrok/model/actor.pt", then the checkpoint should be "./network/model/"

        --use_cuda: default as True
        --cuda: the ids of GPU you want to use

        --obj_dim: the block's dimension
        --reward_type: the reward type you want to use:
            'comp', 'soft', 'hard', 'pyrm', 'pymr-soft', 'pyrm-hard'
        --input_type: the data type you want to use:
            'simple', 'rot', 'bot', 'mul', 'mul-with'
            more detial you can go to pack.py > PACKDataset 
        --allow_rot: True if you want to rotation
        --container_width: the target container width, for 2D, the container size will be container_width x inf, for 3D, the container size will be container_width x container_width x inf
            
        --mix_data: Ture if you want to trian/test on mix dataset
        --gt_data: Ture if you want to trian/test on gt dataset

        --min_size:
        --max_size:
        --initial_container_width:

        --encoder_hidden_size:
        --decoder_hidden_size:

        --just_test: Ture if you want to test a network, False if you want to train a network
        --just_generate: True if you want to generate some data, False otherwise
        --note: any words you want to say, when the train.py is running, it will create a folder under pack/ to store the result, and the note string will be a part of the folder's name
        
Example:

    python trainer.py --just_test=False --valid_size=10 --train_size=128000 --epoch_num=50 \
        --mix_data=False \
        --gt_data=True \
        --container_width=9 \
        --cuda=2 \
        --nodes=10 \
        --note=*some-useless-words* \
        --input_type=bot \
        --reward_type=comp \
        --allow_rot=True \
        --obj_dim=2 \


This command means you want to train the network in 2D condition, using the gt dataset, the training data size is 128000, the task is packing 10 blocks into a container, the container size is 9 x inf, reward type is 'comp', and the constrain of the packing is robot feasibility (--input_type=bot)

Then it will generate a folder unber `pack/` folder to store the training or testing result.
The folder structure of `pack` looks like this:

    pack/
        10/  <--- nodes=10
            2d-bot-comp-note-*some-useless-words*-2019-12-xx-xx-xx/ <--- the folder create by our code
                checkpoints/    <--- store models of each epoch
                    0/          <--- the epoch num
                    1/
                    ...
                render/         <--- store the testing result of each epoch
                    0/  
                    1/
                    ...

The `data/` folder contain the training and testing data, `*_gt/` means the gt dataset, and `*_bot/` means the random dataset.
The folder structure of `pack` looks like this:

    data/
        pack_2d_bot/
            pack-train-5-10/
                blocks.txt      <--- blocks data, container each rotation type
                dep_move.txt    <--- movement dependency
                dep_large.txt   <--- rotation dependency of large size
                dep_small.txt   <--- rotation dependency of small size
                container.txt   <--- only use in multi-target container task, just some random generated target container id for blocks

Under these folders, we have many kinds for dataset, `pack_train_10_1000` means the training dataset with 1000 data, each data has 10 blocks, `pack_valid_5_10` means the testing dataset with 10 data, each data has 5 blocks.
