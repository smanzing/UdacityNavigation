# UdacityNavigation

## Project Details

In this project, we train an agent to collect yellow bananas while avoiding blue bananas in a square world.

### Environment

**Reward**: A reward of +1 is given for collecting yellow bananas, a reward of -1 for collecting blue bananas.

**Duration**: The task is episodic. The task terminates after the agent has taken a maximum number of actions.

**State Space**: The state space has 37 dimensions which consist of the agent's velocity and a ray-based perception of objects in front of the agent.

**Action Space**: The 4 discrete actions are moving forward, moving backward, turning left, and turning right.

**Success Criteria**: The environment is solved, if the agent reaches an average score of +13 over 100 consecutive episodes. 


## Getting Started

Follow the instructions below to set up your python environment to run the code in this repository.
Note that the installation was only tested for __Mac__.

1. Initialize the git submodules in the root folder of this repository. 

	```bash
	git submodule init
	git submodule update
	```
 
2. Create (and activate) a new conda environment with Python 3.6.

	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	
3. Install the base Gym library and the **box2d** environment group:

	```bash
	pip install gym
	pip install Box2D gym
	```

4. Navigate to the `external/Value-based-methods/python/` folder.  Then, install several dependencies.

    ```bash
    cd external/Value-based-methods/python
    pip install .
    ```
    **Note**: If an error appears during installation regarding the torch version, please remove the torch version 0.4.0 in
    external/Value-based-methods/python/requirements.txt and try again.

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    **Note**: Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

6. Download the [Unity environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)  in the **root folder** of this repository.

    
## Instructions

After you have successfully installed the dependencies, open the jupyter notebook [Navigation.ipynb](Navigation.ipynb) 
and follow the instructions to learn how to train the agent.

Below, you can find an overview of the files in the repository.

### Repository Overview

The folder **external** contains the repository [Value-based Methods](https://github.com/udacity/Value-based-methods#dependencies) 
as submodule. The repository is only used for installation purposes.

The folder **rl_lib** contains the library that implements the agent and the algorithms for training the agent: 

- [training.py](rl_lib/training.py): The file contains the main function **dqn** for training the agent. 
- [agent.py](rl_lib/agent.py): Implementation of the agent that interacts and learns from the environment. 
                               The member function **learn** implements the DQN algorithm and extensions, e.g., DoubleDQN.
- [model.py](rl_lib/model.py): Neural network model that represents the policy of the agent.
- [replay.py](rl_lib/replay.py): Implements a fixed-sized buffer to store the agent's experience over time.

The jupyter notebook [Navigation.ipynb](Navigation.ipynb) shows how to train the agent using the standard DQN algorithm and Double DQN
with prioritized experience replay.

## References 

The implementation is based on the research papers [[1]](#1), [[2]](#2), and [[3]](#3) and the code provided by Udacity, see
[Value-based Methods](https://github.com/udacity/Value-based-methods#dependencies).


<a id="1">[1]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* 
**Human-level control through deep reinforcement learning.**
*Nature* **518**, 529–533 (2015).

<a id="1">[2]</a>
Hasselt, H. van and Guez, A. and Silver, D.
**Deep Reinforcement Learning with Double Q-Learning**
*Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence*, 2094–2100 (2016).

<a id="1">[3]</a>
Schaul, T. and Quan, J. and Antonoglou, I. and Silver, D.
**Prioritized Experience Replay**
*arXiv* **10.48550/ARXIV.1511.05952**, (2015).