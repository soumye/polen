# coding: utf-8

import numpy as np
import torch
from shutil import rmtree
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class PolicyEvaluationNetwork(nn.Module):
    def __init__(self, args, device, no_avg=False):
        super(PolicyEvaluationNetwork, self).__init__()
        self.device = device
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
        self.init_weights()

    def init_weights(self):
        # Glorot Initialization
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.xavier_normal_(self.out.weight)

    def forward(self, z1, z2):
        # Compute Distribution over returns in log space. To get probs take softmax.
        # TODO: Do we need to input $\theta$ as well??
        z1, z2 = z1.to(self.device), z2.to(self.device)
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
    def __init__(self, args, device):
        super(PolicyEvaluationNetwork_2, self).__init__()
        ''' PEN without Distributional Output
        '''
        self.device = device
        self.linear1 = nn.Linear(2*args.embedding_size, args.pen_hidden, bias=True)
        self.linear2 = nn.Linear(args.pen_hidden, args.pen_hidden, bias=True)
        # self.linear3 = nn.Linear(args.pen_hidden, args.pen_hidden, bias=True)
        # Output Scalar Predicted Returns
        self.out = nn.Linear(args.pen_hidden, 1)
        self.init_weights()

    def init_weights(self):
        # Glorot Initialization
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        # torch.nn.init.xavier_normal_(self.linear3.weight)
        torch.nn.init.xavier_normal_(self.out.weight)

    def forward(self, z1, z2):
        # Compute Distribution over returns in log space. To get probs take softmax.
        z1, z2 = z1.to(self.device), z2.to(self.device)
        x = torch.tanh(self.linear1(torch.cat((z1,z2), dim=1)))
        x = torch.tanh(self.linear2(x))
        # x = torch.tanh(self.linear3(x))
        return self.out(x)
    
    def predict(self, z1, z2):
        return -self.forward(z1.unsqueeze(0), z2.unsqueeze(0)).squeeze(0)

class SteerablePolicyNet(nn.Module):
    def __init__(self, args, device):
        super(SteerablePolicyNet, self).__init__()
        """ Steerable Policy net with parameters theta. Takes in conditioning vector z and state to output conditioned policy params(5)
        """        
        self.device = device
        self.linear1 = nn.Linear(args.embedding_size, args.policy_hidden)
        self.linear2 = nn.Linear(args.policy_hidden, args.policy_hidden)
        # Output Defect Probabilities for states
        # self.out = nn.Linear(args.policy_hidden, 5, bias=True)
        self.out = nn.Linear(args.policy_hidden, 5, bias=False)
        self.ln1 = nn.LayerNorm(args.policy_hidden)
        self.ln2 = nn.LayerNorm(args.policy_hidden)
        self.lnout = nn.LayerNorm(5)
        # self.bn1 = nn.BatchNorm1d(args.policy_hidden, affine=False)
        # self.bn2 = nn.BatchNorm1d(args.policy_hidden, affine=False)
        # self.bn3 = nn.BatchNorm1d(5)
        self.init_weights()

    def init_weights(self):
        # Glorot Initialization
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.xavier_normal_(self.out.weight)

        # torch.nn.init.uniform_(self.linear1.weight, -1, 1)
        # torch.nn.init.uniform_(self.linear2.weight, -1, 1)
        # torch.nn.init.uniform_(self.out.weight, -1, 1)

        # torch.nn.init.normal_(self.linear1.weight, mean=1.0, std=0.1)
        # torch.nn.init.normal_(self.linear2.weight, mean=1.0, std=0.1)
        # torch.nn.init.normal_(self.out.weight, mean=1.0, std=0.1)
        # self.linear1.weight.data.copy_(torch.eye(5))
        # self.linear2.weight.data.copy_(torch.eye(5))
        # self.out.weight.data.copy_(torch.eye(5))
        
    
    # def policy_update(self, objective):
    #     self.theta_optimizer.zero_grad()
    #     objective.backward(retain_graph=True)
    #     self.theta_optimizer.step()
    
    def forward(self, x):
        x = x.to(self.device)
        # x = self.ln1(torch.tanh(self.linear1(x)))
        # x = self.ln2(torch.tanh(self.linear2(x)))
        x = torch.tanh(self.ln1(self.linear1(x)))
        x = torch.tanh(self.ln2(self.linear2(x)))
        return self.out(x)
        # return self.lnout(self.out(x))
    
    def fwd(self, z):
        """ Takes as input the strategy vector z and outputs the defect logits
        """
        return self.forward(z.unsqueeze(0)).squeeze(0)

    def act(self, batch_states, z, values=None):
        batch_states = torch.from_numpy(batch_states).long()
        conditioned_vec = self.fwd(z)
        probs = torch.sigmoid(conditioned_vec)[batch_states]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        if values is not None:    
            return actions.cpu().numpy().astype(int), log_probs_actions, values[batch_states]
        else:
            return actions.cpu().numpy().astype(int), log_probs_actions
    
    def act_parallel(self, batch_states, batch_z, values=None):
        assert (batch_states.shape[0] == batch_z.shape[0])
        batch_states = torch.from_numpy(batch_states).long()
        batch_conditioned_vec = self.forward(batch_z)
        probs = torch.sigmoid(batch_conditioned_vec).gather(1, batch_states.view(-1,1)).squeeze(1)
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        if values is not None:    
            return actions.numpy().astype(int), log_probs_actions, values[batch_states]
        else:
            return actions.numpy().astype(int), log_probs_actions

class SteerablePolicy():
    def __init__(self, args, device):
        # self.theta = nn.Parameter(torch.randn(5, requires_grad=True))
        self.device = device
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True).to(self.device))
        
    # def policy_update(self, objective):
    #     self.theta_optimizer.zero_grad()
    #     objective.backward(retain_graph=True)
    #     self.theta_optimizer.step()
    
    def fwd(self, z):
        return self.theta + z

    def act(self, batch_states, z, values=None):
        batch_states = torch.from_numpy(batch_states).long()
        conditioned_vec = self.theta + z.to(self.device)
        probs = torch.sigmoid(conditioned_vec)[batch_states]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        if values is not None:    
            return actions.cpu().numpy().astype(int), log_probs_actions, values[batch_states]
        else:
            return actions.cpu().numpy().astype(int), log_probs_actions
    
    def act_parallel(self, batch_states, batch_z, values=None):
        assert (batch_states.shape[0] == batch_z.shape[0])
        batch_states = torch.from_numpy(batch_states).long()
        batch_conditioned_vec = self.theta + batch_z.to(self.device)
        probs = torch.sigmoid(batch_conditioned_vec).gather(1, batch_states.view(-1,1)).squeeze(1)
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        if values is not None:    
            return actions.numpy().astype(int), log_probs_actions, values[batch_states]
        else:
            return actions.numpy().astype(int), log_probs_actions