import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""
def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# why retain graph? Do not auto free memory for one loss when computing multiple loss
# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
def update_params(optim, loss):
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer:
    """
    Convert to numpy
    """
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='actor', chkpt_dir='tmp/sac'):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.name = name
        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, action_dim)
        ).apply(init_weights)

    def forward(self, s):
        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def greedy_act(self, s):  # no softmax more efficient
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
        return greedy_actions.item()

    def sample_act(self, s):
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        actions_probs = F.softmax(actions_logits, dim=-1)
        actions_distribution = Categorical(actions_probs)
        action = actions_distribution.sample()
        return action.item()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='critic', chkpt_dir='tmp/sac'):
        super(Critic, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        if chkpt_dir is not None:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.qnet1 = DuelQNet(state_dim, action_dim, n_hidden_units)
        self.qnet2 = DuelQNet(state_dim, action_dim, n_hidden_units)

    def forward(self, s):  # S: N x F(state_dim) -> Q: N x A(action_dim) Q(s,a)
        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        return q1, q2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class DuelQNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='DuelQNet', chkpt_dir='tmp/sac'):
        super(DuelQNet, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        if chkpt_dir is not None:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        ).apply(init_weights)

        # self.q_head = nn.Linear(n_hidden_units, action_dim)

        self.action_head = nn.Linear(n_hidden_units, action_dim).apply(init_weights)
        self.value_head = nn.Linear(n_hidden_units, 1).apply(init_weights)

    def forward(self, s):
        s = self.shared_mlp(s)
        a = self.action_head(s)
        v = self.value_head(s)
        return v + a - a.mean(1, keepdim=True)
        # return self.q_head(s)
