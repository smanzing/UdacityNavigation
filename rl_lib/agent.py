import numpy as np
import random
from enum import Enum
from typing import List

from rl_lib.model import QNetwork
from rl_lib.replay import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class DQNExtensions(Enum):
    DoubleDQN = 0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int, replay_buffer: ReplayBuffer,
                 gamma: float = 0.99, tau: float = 1e-3, learning_rate: float = 5e-4, update_every: int = 4,
                 dqn_extensions: List[DQNExtensions] = []):
        """
        @param state_size: dimension of each state
        @param action_size: dimension of each action
        @param seed:  random seed
        @param replay_buffer: replay buffer
        @param gamma: discount factor
        @param tau: for soft update of target parameters
        @param learning_rate: learning rate
        @param update_every: how often to update the target network
        @param dqn_extensions: if empty, the standard DQN algorithm is used for learning.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.dqn_extensions = dqn_extensions

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = replay_buffer
        # Initialize time step (helper for updating the target network every update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        # if self.t_step == 0:
        # If enough samples are available in memory, get random subset and learn
        #    if len(self.memory) > BATCH_SIZE:
        #        experiences = self.memory.sample()
        #        self.learn(experiences, self.gamma)
        if self.memory.is_size_of_memory_sufficient_to_draw_batch():
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if DQNExtensions.DoubleDQN in self.dqn_extensions:
            a_max = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, a_max)
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
