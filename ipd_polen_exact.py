# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from copy import deepcopy
from envs import IPD

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_out', default=0.2, type=float, help='')
    parser.add_argument('-lr_in', default=0.3, type=float, help='')
    parser.add_argument('-lr_pen', default=0.3, type=float, help='Adam lr for PEN')
    parser.add_argument('-gamma', default=0.96, type=float, help='')
    parser.add_argument('-n_iter', default=50, type=int, help='No of algo iteration')
    parser.add_argument('-len_rollout', default=200, type=int, help='Length of rollouts')
    parser.add_argument('-batch_size', default=64, type=int, help='')
    parser.add_argument('-seed', default=42, type=int, help='Random Seed')
    parser.add_argument('-lookaheads', default=1, type=int, help='No of lola lookaheads')
    parser.add_argument('-embedding_size', default=5, type=int, help='Size of the embedding z')
    parser.add_argument('-nbins', default=15, type=int, help='No of bins to discretize return space')
    parser.add_argument('-pen_hidden', default=80, type=int, help='Hidden size of PEN')
    parser.add_argument('-n_policy', default=30, type=int, help='No of policy updates for each algo iteration')
    parser.add_argument('-n_pen', default=30, type=int, help='')
    parser.add_argument('-nsamples_bin', default=30, type=int, help='No of rollouts for given z1/z2 to compute histogram of returns. Usually 2*nbins')


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
    def __init__(self, args,  z=None):
        # init z and its optimizer. z is the embedding
        # TODO: How to initialize z. 0 is bad
        self.z = nn.Parameter(torch.zeros(args.embedding_size, requires_grad=True))
        self.z = nn.Parameter(torch.randn(args.embedding_size, requires_grad=True))
        self.z_optimizer = torch.optim.Adam(params=(self.z,),lr=args.lr_out)

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
    def __init__(self, args):
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam(params=(self.theta,),lr=args.lr_out)

    def act(self, batch_states, z):
        batch_states = torch.from_numpy(batch_states).long()
        conditioned_vec = self.theta + z
        probs = torch.sigmoid(conditioned_vec)[batch_states]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions

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

class Polen():
    def __init__(self, args):
            """
            1. Initialize Agent embeddings z, poliy theta & pen psi
            """
            self.args = args
            self.agent1 = Agent(self.args)
            self.agent2 = Agent(self.args)
            self.policy = SteerablePolicy(self.args)
            self.pen = PolicyEvaluationNetwork(self.args)
            self.pen_optimizer = torch.optim.Adam(params=self.pen.parameters(), lr=args.lr_pen)
            self.ipd2 = IPD(max_steps=args.len_rollout, batch_size=args.nsamples_bin)

    def rollout(self, nsteps):
        # just to evaluate progress:
        (s1, s2), _ = ipd.reset()
        score1 = 0
        score2 = 0
        for t in range(nsteps):
            a1, lp1 = self.policy.act(s1, self.agent1.z)
            a2, lp2 = self.policy.act(s2, self.agent2.z)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            # cumulate scores
            score1 += np.mean(r1)/float(self.args.len_rollout)
            score2 += np.mean(r2)/float(self.args.len_rollout)
        return (score1, score2)

    def rollout_binning(self, nsteps, z1, z2):
        # just to evaluate progress:
        (s1, s2), _ = self.ipd2.reset()
        score1 = torch.zeros(self.args.nsamples_bin, dtype=torch.float)
        score2 = torch.zeros(self.args.nsamples_bin, dtype=torch.float)
        for t in range(nsteps):
            a1, lp1 = self.policy.act(s1, z1)
            a2, lp2 = self.policy.act(s2, z2)
            (s1, s2), (r1, r2),_,_ = self.ipd2.step((a1, a2))
            # cumulate scores
            score1 += r1
            score2 += r2
        score1 = score1/nsteps
        score2 = score2/nsteps
        hist1 = torch.histc(score1, bins=self.args.nbins, min=-3, max=0)
        hist2 = torch.histc(score2, bins=self.args.nbins, min=-3, max=0)
        return hist1, hist2

    
    def train(self):
        print("start iterations with", self.args.lookaheads, "lookaheads:")
        joint_scores = []
        for update in range(self.args.n_iter):
            # 1a. For fixed z1 & z2, Learn steerable policy theta & PEN by maximizing rollouts
            for _ in range(self.args.n_policy):
                # Update steerable policy parameters. True possible for IPD
                policy_loss = self.policy_update_true()
                # self.policy_update_pg()

            # 1b. Train the PEN
            # TODO: Convert this to a parallel version so one call to PEN is required
            for _ in range(self.args.n_pen):
                # randomly generate z1, z2. Maybe generation centered on z0, z1 would be better.
                z1 = torch.randn(self.args.embedding_size)
                z2 = torch.randn(self.args.embedding_size)
                # Experiment with smaller length of rollouts for estimation
                hist1, hist2 = self.rollout_binning(self.args.len_rollout, z1, z2)

                # Compute the KL Div
                w1, w2 = self.pen.forward(self.agent1.z.unsqueeze(0), self.agent2.z.unsqueeze(0))
                w1 = F.softmax(w1.squeeze(), dim=0)
                w2 = F.softmax(w2.squeeze(), dim=0)
                # F.kl_div(Q.log(), P, None, None, 'sum')
                self.pen_optimizer.zero_grad()
                pen_loss = (hist1* (hist1 / w1).log()).sum() + (hist2* (hist2 / w2).log()).sum()
                pen_loss.backward()
                self.pen_optimizer.step()
        
            # 2. Do on Lola Updates
            self.lola_update_exact()
            
            # evaluate:
            score = self.rollout(self.args.len_rollout)
            joint_scores.append(0.5*(score[0] + score[1]))

            # Logging
            if update%10==0 :
                p1 = [p.item() for p in torch.sigmoid(self.policy.theta + self.agent1.z)]
                p2 = [p.item() for p in torch.sigmoid(self.policy.theta + self.agent2.z)]
                print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]), '(agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
        
        return joint_scores
    
    def policy_update_true(self):
        # Batching not needed here.
        self.policy.theta_optimizer.zero_grad()
        theta1 = self.agent1.z + self.policy.theta
        theta2 = self.agent2.z + self.policy.theta
        objective = -(true_objective(theta1, theta2, ipd) + true_objective(theta2,theta1, ipd))
        objective.backward()
        self.policy.theta_optimizer.step()
        return objective
    
    def policy_update_pg(self):
        """
        TODO:
        Will need batching
        """
        pass

    def lola_update_exact(self):
        """
        Do Lola Updates
        """
        # copy other's parameters:
        z1_ = self.agent1.z.clone().detach().requires_grad_(True)
        z2_ = self.agent2.z.clone().detach().requires_grad_(True)
        # z1_ = torch.tensor(self.agent1.z.detach(), requires_grad=True)
        # z2_ = torch.tensor(self.agent2.z.detach(), requires_grad=True)

        for k in range(self.args.lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = self.agent1.in_lookahead(self.policy.theta, z2_)
            grad1 = self.agent2.in_lookahead(self.policy.theta, z1_)
            # update other's theta
            z2_ = z2_ - self.args.lr_in * grad2
            z1_ = z1_ - self.args.lr_in * grad1

        # update own parameters from out_lookahead:
        self.agent1.out_lookahead(self.policy.theta, z2_)
        self.agent2.out_lookahead(self.policy.theta, z1_)


def plot(scores, args):
    """Plotting"""
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
