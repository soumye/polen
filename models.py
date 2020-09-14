# coding: utf-8

import numpy as np
import torch
from shutil import rmtree
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class PolicyEvaluationNetwork(nn.Module):
    def __init__(self, args, no_avg=False):
        super(PolicyEvaluationNetwork, self).__init__()
        self.linear1 = nn.Linear(2*args.embedding_size, args.pen_hidden, bias=True)
        self.linear2 = nn.Linear(args.pen_hidden, args.pen_hidden, bias=True)
        # Output a distribution of Returns ( Look at Distributional RL too ?? )
        self.out = nn.Linear(args.pen_hidden, args.nbins)
        if no_avg:
            min = -75.0
        else:
            min = -3.0
        max = 0.0
        h = (max - min)/args.nbins
        self.mids = min + h/2 + h*torch.tensor(range(args.nbins))

    def init_weights(self):
        # Glorot Initialization
        torch.nn.init.xavier_normal_(self.linear1.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.linear2.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.out.weight, std=0.1)

    def forward(self, z1, z2):
        # Compute Distribution over returns in log space. To get probs take softmax.
        # TODO: Do we need to input $\theta$ as well??
        x = F.relu(self.linear1(torch.cat((z1,z2), dim=1)))
        x = F.relu(self.linear2(x))
        return self.out(x)
    
    def predict(self, z1, z2):
        """ For getting objective to minimize and get gradients .
        Returns: -ve of Value_1(z1, z2)
        """        
        h1 = self.forward(z1.unsqueeze(0), z2.unsqueeze(0))
        return -(F.softmax(h1)*self.mids).sum()

class PolicyEvaluationNetwork_2(nn.Module):
    def __init__(self, args):
        ''' PEN without Distributional Output
        '''
        super(PolicyEvaluationNetwork_2, self).__init__()
        self.linear1 = nn.Linear(2*args.embedding_size, args.pen_hidden, bias=True)
        self.linear2 = nn.Linear(args.pen_hidden, args.pen_hidden, bias=True)
        self.linear3 = nn.Linear(args.pen_hidden, args.pen_hidden, bias=True)
        # Output Scalar Predicted Returns
        self.out = nn.Linear(args.pen_hidden, 1)

    def init_weights(self):
        # Glorot Initialization
        torch.nn.init.xavier_normal_(self.linear1.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.linear2.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.linear3.weight, std=0.1)
        torch.nn.init.xavier_normal_(self.out.weight, std=0.1)

    def forward(self, z1, z2):
        # Compute Distribution over returns in log space. To get probs take softmax.
        x = F.relu(self.linear1(torch.cat((z1,z2), dim=1)))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.out(x)
    
    def predict(self, z1, z2):
        return -self.forward(z1.unsqueeze(0), z2.unsqueeze(0)).squeeze(0)


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
    
    def act_parallel(self, batch_states, batch_z, values=None):
        batch_states = torch.from_numpy(batch_states).long()
        batch_conditioned_vec = self.theta + batch_z
        probs = torch.sigmoid(batch_conditioned_vec).gather(1, batch_states.view(-1,1)).squeeze(1)
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        if values is not None:    
            return actions.numpy().astype(int), log_probs_actions, values[batch_states]
        else:
            return actions.numpy().astype(int), log_probs_actions