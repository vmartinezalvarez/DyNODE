import os
import time
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(dir_path):
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return dir_path

class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, batch_length=40, dynode_batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.dynode_batch_size = dynode_batch_size
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.episode_length = 1000
        self.prev_idx = 0
        self.idx = 0
        self.full = False
        
    def add(self, obs, action, reward, next_obs, done, episode_done=False):
       
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

        if episode_done:
            self.last_episode_length = self.idx - self.prev_idx
            self.prev_idx = self.idx
            self.episode_length = min(self.last_episode_length, self.episode_length)
        
    def sample(self):
        
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses   = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_dynode(self):

        inter =  self.capacity//self.episode_length if self.full else self.idx//self.episode_length

        numbs = np.concatenate([np.arange((i)*self.episode_length, 
                                ((i+1)*self.episode_length)-self.batch_length) for i in range(inter)])
        
        idxs = np.random.choice(numbs, size=self.dynode_batch_size)
        idxses = np.asarray([np.arange(idx, idx + self.batch_length) for idx in idxs])
        vec_idx = idxses.transpose().reshape(-1)

        obses = torch.as_tensor(self.obses[vec_idx].reshape(self.batch_length, self.dynode_batch_size,
                                *self.obs_shape), device=self.device).float()
        actions = torch.as_tensor(self.actions[vec_idx].reshape(self.batch_length, self.dynode_batch_size,
                                *self.action_shape), device=self.device)
        rewards = torch.as_tensor(self.rewards[vec_idx].reshape(self.batch_length, self.dynode_batch_size, 1),
                                device=self.device)
        next_obses = torch.as_tensor(self.next_obses[vec_idx].reshape(self.batch_length, self.dynode_batch_size,
                                *self.obs_shape),  device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[vec_idx].reshape(self.batch_length, self.dynode_batch_size, 1),
                                device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def __len__(self):
        return self.capacity 