# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipd_utils import *

class Agent():
    def __init__(self, args, device, ipd, val=None):
        # init z and its optimizer
        self.args = args
        self.device = device
        self.ipd = ipd
        if val is None:
            self.z = nn.Parameter(torch.zeros(self.args.embedding_size, requires_grad=True).to(self.device))
        else:
            self.z = nn.Parameter(val.requires_grad_().to(self.device))
        self.z_optimizer = torch.optim.Adam((self.z,),lr=self.args.lr_out)
        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(self.args.embedding_size, requires_grad=True).to(self.device))
        self.value_optimizer = torch.optim.Adam((self.values,),lr=self.args.lr_v)

    def z_update(self, objective, eval=False):
        self.z_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        if not eval:
            self.z_optimizer.step()
        return self.z.grad

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def in_lookahead(self, other_z, other_values, policy):
        (s1, s2), _ = self.ipd.reset()
        other_memory = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = policy.act(s1, self.z, self.values)
            a2, lp2, v2 = policy.act(s2, other_z, other_values)
            (s1, s2), (r1, r2),_,_ = self.ipd.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        other_objective = other_memory.dice_objective()
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead(self, other_z, other_values, policy, eval=False):
        (s1, s2), _ = self.ipd.reset()
        memory = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = policy.act(s1, self.z, self.values)
            a2, lp2, v2 = policy.act(s2, other_z, other_values)
            (s1, s2), (r1, r2),_,_ = self.ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

        # update self z
        objective = memory.dice_objective()
        grad_z = self.z_update(objective, eval)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)
        return grad_z

    def in_lookahead_exact(self, theta, other_z):
        other_objective = true_objective(theta + other_z, theta + self.z, self.ipd)
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead_exact(self, theta, other_z, eval=False):
        objective = true_objective(theta + self.z, theta + other_z, self.ipd)
        grads = self.z_update(objective, eval)
        return grads
    
    def in_lookahead_pen(self, pen, other_z):
        other_objective = pen.predict(other_z, self.z)
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead_pen(self, pen, other_z, eval=False):
        objective = pen.predict(self.z, other_z)
        grads = self.z_update(objective, eval)
        return grads