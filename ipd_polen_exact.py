# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from copy import deepcopy
from envs import IPD

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_out', default=0.2, type=float, help='learning rate for out iters')
    parser.add_argument('-lr_in', default=0.3, type=float, help='learning rate for out iters')
    parser.add_argument('-gamma', default=0.96, type=float, help='learning rate for out iters')
    parser.add_argument('-n_update', default=50, type=int, help='learning rate for out iters')
    parser.add_argument('-len_rollout', default=200, type=int, help='learning rate for out iters')
    parser.add_argument('-batch_size', default=64, type=int, help='learning rate for out iters')
    parser.add_argument('-seed', default=42, type=int, help='learning rate for out iters')
    parser.add_argument('-lookaheads', default=1, type=int, help='learning rate for out iters')
    parser.add_argument('-embedding_size', default=5, type=int, help='learning rate for out iters')

    return parser.parse_args()

def phi(x1,x2):
    return [x1*x2, x1*(1-x2), (1-x1)*x2,(1-x1)*(1-x2)]

def true_objective(theta1, theta2, ipd):
    p1 = torch.sigmoid(theta1)
    p2 = torch.sigmoid(theta2[[0,1,3,2,4]])
    p0 = (p1[0], p2[0])
    p = (p1[1:], p2[1:])
    # create initial laws, transition matrix and rewards:
    P0 = torch.stack(phi(*p0), dim=0).view(1,-1)
    P = torch.stack(phi(*p), dim=1)
    R = torch.from_numpy(ipd.payout_mat).view(-1,1).float()
    # the true value to optimize:
    objective = (P0.mm(torch.inverse(torch.eye(4) - args.gamma*P))).mm(R)
    return -objective

def get_gradient(objective, z):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (z), create_graph=True)[0]
    return grad_objective

class Agent():
    def __init__(self, z=None):
        # init z and its optimizer. z is the embedding
        self.z = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.z_optimizer = torch.optim.Adam((self.z,),lr=args.lr_out)

    def z_update(self, objective):
        self.z_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.z_optimizer.step()

    def in_lookahead(self, theta, other_z):
        other_objective = true_objective(theta + other_z, theta + self.z, ipd)
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead(self, theta, other_z):
        objective = true_objective(theta + self.z, theta + other_z, ipd)
        self.z_update(objective)

class SteerablePolicy():
    def __init__(self):
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,),lr=args.lr_out)

    def act(self, batch_states, z):
        batch_states = torch.from_numpy(batch_states).long()
        conditioned_vec = self.theta + z
        probs = torch.sigmoid(conditioned_vec)[batch_states]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions

class PolicyEvaluationNetwork(nn.module):
    def __init__(self, args):
        super(PolicyEvaluationNetwork, self).__init__()
        self.linear1 = nn.Linear(args.embedding_size, 200, bias=True)
        self.linear2 = nn.Linear(args.embedding_size, 200, bias=True)
        # Output a distribution of Returns ( Look at Distributional RL too ?? )
        self.linear3 = nn.Linear(200, 5)
        # TODO: 
        pass

    def init_weights(self):
        # TODO: 
        torch.nn.init.normal_(self.linear1.weight, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, std=0.1)
        torch.nn.init.normal_(self.linear3.weight, std=0.1)

    def forward(self, z1, z2):
        # TODO:         
        

class Polen(args):
    def _init(self, args):
            self.agent1 = Agent()
            self.agent2 = Agent()
            self.args = args
            self.policy = SteerablePolicy()
            self.pen = PolicyEvaluationNetwork(self.args)

    def step(self):
        # just to evaluate progress:
        (s1, s2), _ = ipd.reset()
        score1 = 0
        score2 = 0
        for t in range(self.args.len_rollout):
            a1, lp1 = self.policy.act(s1, self.agent1.z)
            a2, lp2 = self.policy.act(s2, self.agent2.z)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            # cumulate scores
            score1 += np.mean(r1)/float(self.args.len_rollout)
            score2 += np.mean(r2)/float(self.args.len_rollout)
        return (score1, score2)

    def train(self):
        print("start iterations with", self.args.lookaheads, "lookaheads:")
        joint_scores = []
        for update in range(self.args.n_update):
            # 1. Learn steerable policy & PEN by maximizing rollouts

            # 2. Do on Lola Updates
            # copy other's parameters:
            z1_ = torch.tensor(self.agent1.z.detach(), requires_grad=True)
            z2_ = torch.tensor(self.agent2.z.detach(), requires_grad=True)

            for k in range(self.args.lookaheads):
                # estimate other's gradients from in_lookahead:
                grad2 = self.agent1.in_lookahead(z2_)
                grad1 = self.agent2.in_lookahead(z1_)
                # update other's theta
                z2_ = z2_ - self.args.lr_in * grad2
                z1_ = z1_ - self.args.lr_in * grad1

            # update own parameters from out_lookahead:
            self.agent1.out_lookahead(z2_)
            self.agent2.out_lookahead(z1_)

            # evaluate:
            score = self.step()
            joint_scores.append(0.5*(score[0] + score[1]))

            # print
            if update%10==0 :
                p1 = [p.item() for p in torch.sigmoid(self.policy.theta + self.agent1.z)]
                p2 = [p.item() for p in torch.sigmoid(self.policy.theta + self.agent2.z)]
                print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
        return joint_scores
    
def plot(scores, args):
    "Plotting"
    plt.plot(scores, 'b', label=str(args.lookaheads)+" lookaheads")
    plt.legend()
    plt.xlabel('rollouts')
    plt.ylabel('joint score')
    plt.show()

if __name__=="__main__":

    args = get_args()
    global ipd
    ipd = IPD(max_steps=args.len_rollout, batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    polen = Polen(args)
    scores = polen.train()
    plot(scores, args)




