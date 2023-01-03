import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, random_seed, alpha, beta):
        """
        @param action_size: dimension of each action
        @param buffer_size: maximum size of buffer
        @param batch_size: size of each training batch
        @param random_seed: random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(random_seed)

        self.alpha = alpha
        self.beta = beta

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if len(self.memory) > self.batch_size:
            max_priority = max([experience.priority for experience in self.memory])
        else:
            max_priority = 1.0
        e = self.experience(state, action, reward, next_state, done, max_priority)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # experiences = random.choices(self.memory, k=self.batch_size)
        priorities = np.array([experience.priority for experience in self.memory]).astype('float64')
        probabilites = priorities ** self.alpha / np.sum(priorities ** self.alpha)
        batch_indices = np.random.choice(len(self.memory), size=self.batch_size, p=probabilites)

        states = torch.from_numpy(
            np.vstack([self.memory[i].state for i in batch_indices if self.memory[i] is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([self.memory[i].action for i in batch_indices if self.memory[i] is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([self.memory[i].reward for i in batch_indices if self.memory[i] is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([self.memory[i].next_state for i in batch_indices if self.memory[i] is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([self.memory[i].done for i in batch_indices if self.memory[i] is not None]).astype(
                np.uint8)).float().to(
            device)
        importance_sampling_weights = torch.from_numpy(
            np.vstack([(len(self.memory) * probabilites[i]) ** (-self.beta) for i in batch_indices if
                       self.memory[i] is not None])).float().to(
                device)
        normalized_importance_sampling_weigths = importance_sampling_weights / max(importance_sampling_weights)
        return states, actions, rewards, next_states, dones, batch_indices, normalized_importance_sampling_weigths

    def update_priorities(self, batch_indices, updated_priorities):
        for i, priority in zip(batch_indices, updated_priorities):
            # add offset of 1e-6 to priority to prevent experiences from being starved for selection due
            # to zero TD error
            self.memory[i] = self.memory[i]._replace(priority=priority.item() + 1e-6)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def is_size_of_memory_sufficient_to_draw_batch(self) -> bool:
        return len(self.memory) > self.batch_size
