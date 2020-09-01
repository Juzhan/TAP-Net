# Pack net

## File structure:
    
* **L_SL.py**: Code to train local pack net with supervised learning
* **L_RL.py**: Code to train local pack net with reinforcement learning
* **LG_RL.py**: Code to train global/(local+global) pack net with reinforcement learning
* **generate_data.py**: Code to generate total random data, it will generate the blocks which packed in a container, then save the blocks size and position
* **data/**: Training and testing data folder, each dataset contains the positions of blocks and 

## Training/testing
* First we should generate training and testing dataset using the [`generate.sh`](../generate.sh) located in project root folder:
```
(project root)> ../generate.sh
```
* Training/testing **local pack net using supervised learning**, you can modify the training adn testing setting on main function:
```
(current folder)> python L_SL.py
```
* Training/testing **local pack net using reinforcement learning**, you can modify the training adn testing setting on main function:
```
(current folder)> python L_RL.py
```
* Training/testing **global pack net using reinforcement learning**, you can modify the training adn testing setting on main function:
```
(current folder)> python LG_RL.py
```
* Training/testing **local+global pack net using reinforcement learning**, you can modify the training adn testing setting on main function:
```
(current folder)> python LG_RL.py
```

![pack net structure, (a) local pnet, (b)global pnet, (c)local+global pnet](./source/pnet.png)
