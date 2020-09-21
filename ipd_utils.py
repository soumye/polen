# coding: utf-8

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import random
from envs import IPD
from tensorboardX import SummaryWriter

def sigmoid_inv(x):
    return torch.log(x/(1-x))

def get_gradient(objective, z):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (z), create_graph=True)[0]
    return grad_objective

def phi(x1,x2):
    return [x1*x2, x1*(1-x2), (1-x1)*x2,(1-x1)*(1-x2)]

def true_objective(theta1, theta2, ipd, gamma = 0.96):
    p1 = torch.sigmoid(theta1.cpu())
    p2 = torch.sigmoid(theta2.cpu()[[0,1,3,2,4]])
    p0 = (p1[0], p2[0])
    p = (p1[1:], p2[1:])
    # create initial laws, transition matrix and rewards:
    P0 = torch.stack(phi(*p0), dim=0).view(1,-1)
    P = torch.stack(phi(*p), dim=1)
    R = torch.from_numpy(ipd.payout_mat).view(-1,1).float()
    # the true value to optimize:
    objective = (P0.mm(torch.inverse(torch.eye(4) - gamma*P))).mm(R)
    return -objective

def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
    def __init__(self, args):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []
        self.args = args

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self, dice_both=False):
        # bsz x traj_len
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(self.args.gamma * torch.ones(*rewards.size()), dim=1)/self.args.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        if not dice_both:
            dependencies = torch.cumsum(self_logprobs, dim=1)
            # logprob of each stochastic nodes:
            stochastic_nodes = self_logprobs
        else:
            dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)
            # logprob of each stochastic nodes:
            stochastic_nodes = self_logprobs + other_logprobs

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if not self.args.not_use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)

class PenDataset(Dataset):
    def __init__(self, size, embedding_size):
        self.size = size
        self.z1s = sigmoid_inv(torch.rand(self.size , embedding_size))
        self.z2s = sigmoid_inv(torch.rand(self.size , embedding_size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.z1s[idx], self.z2s[idx]

class ReplayBuffer:
    """
    Replay buffer...
    """
    def __init__(self, size, dim):
        self._storage = []
        self._dim = dim
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, z1, z2):
        data = (z1, z2)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def add_surround(self, z1, z2, num_samples, var=1):
        for _ in range(num_samples):
            # Generate z1s & z2s aroud z1, z2 sampled from the buffer
            z1_sampled = torch.rand(self._dim)*var + z1
            z2_sampled = torch.rand(self._dim)*var + z2
            self.add(z1_sampled, z2_sampled)

    def sample(self):
        # import ipdb; ipdb.set_trace()
        return self._storage[random.randint(0, len(self._storage) - 1)]

    def sample_batch(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
    def _encode_sample(self, idxes):
        z1s, z2s = [], []
        for i in idxes:
            z1, z2 = self._storage[i]
            z1s.append(z1)
            z2s.append(z2)
        return torch.stack(z1s), torch.stack(z2s)