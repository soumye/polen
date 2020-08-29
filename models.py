# coding: utf-8

import numpy as np
import torch
from shutil import rmtree
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class PolicyEvaluationNetwork(nn.Module):
    def __init__(self, args):
        super(PolicyEvaluationNetwork, self).__init__()
        self.linear1 = nn.Linear(2*args.embedding_size, args.pen_hidden, bias=True)
        self.linear2 = nn.Linear(args.pen_hidden, args.pen_hidden, bias=True)
        # Output a distribution of Returns ( Look at Distributional RL too ?? )
        self.head1 = nn.Linear(args.pen_hidden, args.nbins)
        self.head2 = nn.Linear(args.pen_hidden, args.nbins)

    def init_weights(self):
        # Glorot Initialization
        torch.nn.init.xavier_normal_(self.linear1.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.linear2.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.head1.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.head2.weight, std=0.1)

    def forward(self, z1, z2):
        # Compute Distribution over returns in log space. To get probs take softmax.
        # TODO: Do we need to input $\theta$ as well??
        x = F.relu(self.linear1(torch.cat((z1,z2), dim=1)))
        x = F.relu(self.linear2(x))
        return self.head1(x), self.head2(x)

class SteerablePolicy():
    def __init__(self, args):
        # self.theta = nn.Parameter(torch.randn(5, requires_grad=True))
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam(params=(self.theta,),lr=args.lr_theta)
    
    def policy_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def act(self, batch_states, z, values=None):
        batch_states = torch.from_numpy(batch_states).long()
        conditioned_vec = self.theta + z
        probs = torch.sigmoid(conditioned_vec)[batch_states]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        if values is not None:    
            return actions.numpy().astype(int), log_probs_actions, values[batch_states]
        else:
            return actions.numpy().astype(int), log_probs_actions