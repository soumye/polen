# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import os
from shutil import rmtree
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from copy import deepcopy
from envs import IPD
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from models import PolicyEvaluationNetwork, SteerablePolicy
from plotting import plot, plot_many

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_theta', default=0.2, type=float, help='lr for theta using Naive PG')
    parser.add_argument('-lr_out', default=0.2, type=float, help='lr for outer loops of z')
    parser.add_argument('-lr_in', default=0.3, type=float, help='lr for inner loops of z')
    parser.add_argument('-lr_v', default= 0.1, type=float, help='lr for vf for baseline')
    parser.add_argument('-lr_pen', default=0.3, type=float, help='Adam lr for PEN')
    parser.add_argument('-gamma', default=0.96, type=float, help='')
    parser.add_argument('-n_iter', default=200, type=int, help='No of algo iteration')
    parser.add_argument('-len_rollout', default=150, type=int, help='Length of rollouts')
    parser.add_argument('-batch_size', default=128, type=int, help='')
    parser.add_argument('-seed', default=4, type=int, help='Random Seed')
    parser.add_argument('-lookaheads', default=1, type=int, help='No of lola lookaheads')
    parser.add_argument('-embedding_size', default=5, type=int, help='Size of the embedding z')
    parser.add_argument('-nbins', default=15, type=int, help='No of bins to discretize return space')
    parser.add_argument('-logdir', default='logdir/', type=str, help='Logging Directory')
    parser.add_argument('-pen_hidden', default=80, type=int, help='Hidden size of PEN')
    parser.add_argument('-n_policy', default=1, type=int, help='No of policy updates for each algo iteration')
    parser.add_argument('-n_pen', default=10, type=int, help='')
    parser.add_argument('-nsamples_bin', default=30, type=int, help='No of rollouts for given z1/z2 to compute histogram of returns. Usually 2*nbins')
    parser.add_argument('-not_plot', action='store_true', help='Disable Plotting')
    parser.add_argument('-not_use_baseline', action='store_true', help='Disable Baseline in Dice')
    parser.add_argument('-only_self', action='store_true', help='Enable Plotting')
    parser.add_argument('-use_exact_lola', action='store_true', help='Disable Plotting')
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

    def dice_objective(self, only_self=False):
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
        if only_self:
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

def get_gradient(objective, z):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (z), create_graph=True)[0]
    return grad_objective

class Agent():
    def __init__(self, args):
        # init z and its optimizer
        self.args = args
        self.z = nn.Parameter(torch.zeros(self.args.embedding_size, requires_grad=True))
        self.z_optimizer = torch.optim.Adam((self.z,),lr=self.args.lr_out)
        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(self.args.embedding_size, requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,),lr=self.args.lr_v)

    def z_update(self, objective):
        self.z_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.z_optimizer.step()

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def in_lookahead(self, other_z, other_values, policy):
        (s1, s2), _ = ipd.reset()
        other_memory = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = policy.act(s1, self.z, self.values)
            a2, lp2, v2 = policy.act(s2, other_z, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        other_objective = other_memory.dice_objective()
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead(self, other_z, other_values, policy):
        (s1, s2), _ = ipd.reset()
        memory = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = policy.act(s1, self.z, self.values)
            a2, lp2, v2 = policy.act(s2, other_z, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

        # update self z
        objective = memory.dice_objective()
        self.z_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)

    def in_lookahead_exact(self, theta, other_z):
        other_objective = true_objective(theta + other_z, theta + self.z, ipd)
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead_exact(self, theta, other_z):
        objective = true_objective(theta + self.z, theta + other_z, ipd)
        self.z_update(objective)

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
            if os.path.exists(args.logdir):
                rmtree(args.logdir)
            writer = SummaryWriter(args.logdir)
            self.writer = SummaryWriter(args.logdir)

    def rollout(self, nsteps):
        # just to evaluate progress:
        (s1, s2), _ = ipd.reset()
        score1 = 0
        score2 = 0
        for t in range(nsteps):
            a1, lp1, v1 = self.policy.act(s1, self.agent1.z, self.agent1.values)
            a2, lp2, v2 = self.policy.act(s2, self.agent2.z, self.agent2.values)
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
            # if update % self.args.n_policy == 0:
            for t in range(self.args.n_policy):
                # Update steerable policy parameters. True possible for IPD
                if self.args.use_exact_lola:
                    policy_loss = self.policy_update_true()
                else:
                    policy_loss = self.policy_update_pg()
                # self.writer.add_scalar('PolicyObjective V1 plus V2', -policy_loss, update/self.args.n_policy)
                self.writer.add_scalar('PolicyObjective V1 plus V2', -policy_loss, update*self.args.n_policy + t )

            # 1b. Train the PEN
            # TODO: Convert this to a parallel version so one call to PEN is required
            for t in range(self.args.n_pen):
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
                # pen_loss = (hist1* (hist1 / w1).log()).sum() + (hist2* (hist2 / w2).log()).sum()
                pen_loss = F.kl_div(hist1, w1) + F.kl_div(hist2, w2)
                pen_loss.backward()
                self.pen_optimizer.step()
                self.writer.add_scalar('PEN Loss: KL1 plus KL2', pen_loss, update*self.args.n_pen + t )
        
            # 2. Do on Lola Updates
            if self.args.use_exact_lola:
                self.lola_update_exact()
            else:
                self.lola_update_dice()
            
            # evaluate:
            score = self.rollout(self.args.len_rollout)
            avg_score = 0.5*(score[0] + score[1])
            self.writer.add_scalar('Avg Score of Agent', avg_score, update)
            joint_scores.append(avg_score)

            # Logging
            if update%10==0 :
                p0 = [p.item() for p in torch.sigmoid(self.policy.theta)]
                p1 = [p.item() for p in torch.sigmoid(self.policy.theta + self.agent1.z)]
                p2 = [p.item() for p in torch.sigmoid(self.policy.theta + self.agent2.z)]
                
                print('After update', update, '------------')
                print('score (%.3f,%.3f)\n' % (score[0], score[1]) , 'Default = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}\n' % (p0[0], p0[1], p0[2], p0[3], p0[4]), '(agent1) = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}\n' % (p1[0], p1[1], p1[2], p1[3], p1[4]), '(agent2) = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
                # print('theta: ', self.policy.theta, '\n', 'z1: ',  self.agent1.z, '\n', 'z2: ',  self.agent2.z)
        return joint_scores
    
    def policy_update_true(self):
        # Batching not needed here.
        self.policy.theta_optimizer.zero_grad()
        theta_ = self.policy.theta.clone().detach()
        theta1 = self.agent1.z + self.policy.theta
        theta2 = self.agent2.z + self.policy.theta
        objective = true_objective(self.agent1.z + self.policy.theta, self.agent2.z + theta_, ipd) + true_objective(self.agent2.z + self.policy.theta, self.agent1.z + theta_, ipd)
        objective.backward()
        self.policy.theta_optimizer.step()
        return objective
    
    def policy_update_pg(self):
        """
        Policy Gradient for theta using Dice Objective. Treat theta as shared parameters
        """

        # First using Agent 1
        (s1, s2), _ = ipd.reset()
        memory1 = Memory(self.args)
        memory2 = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = self.policy.act(s1, self.agent1.z, self.agent1.values)
            a2, lp2, v2 = self.policy.act(s2, self.agent2.z, self.agent2.values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            memory1.add(lp1, lp2, v1, torch.from_numpy(r1).float())
            memory2.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        # update self z
        objective = memory1.dice_objective(only_self=self.args.only_self) +  memory2.dice_objective(only_self=self.args.only_self)
        self.policy.policy_update(objective)
        # # update self value:
        self.agent1.value_update(memory1.value_loss())
        self.agent2.value_update(memory2.value_loss())
        return objective

    def lola_update_dice(self):
        """
        Do Lola-DiCE Updates
        """
        # copy other's parameters:
        z1_ = self.agent1.z.clone().detach().requires_grad_(True)
        z2_ = self.agent2.z.clone().detach().requires_grad_(True)
        values1_ = self.agent1.values.clone().detach().requires_grad_(True)
        values2_ = self.agent2.values.clone().detach().requires_grad_(True)

        for k in range(self.args.lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = self.agent1.in_lookahead(z2_, values2_, self.policy)
            grad1 = self.agent2.in_lookahead(z1_, values1_, self.policy)
            # update other's theta
            z2_ = z2_ - self.args.lr_in * grad2
            z1_ = z1_ - self.args.lr_in * grad1

        # update own parameters from out_lookahead:
        self.agent1.out_lookahead(z2_, values2_, self.policy)
        self.agent2.out_lookahead(z1_, values1_, self.policy)
    
    def lola_update_exact(self):
        """
        Do Lola Updates
        """
        # copy other's parameters:
        z1_ = self.agent1.z.clone().detach().requires_grad_(True)
        z2_ = self.agent2.z.clone().detach().requires_grad_(True)

        for k in range(self.args.lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = self.agent1.in_lookahead_exact(self.policy.theta, z2_)
            grad1 = self.agent2.in_lookahead_exact(self.policy.theta, z1_)
            # update other's theta
            z2_ = z2_ - self.args.lr_in * grad2
            z1_ = z1_ - self.args.lr_in * grad1

        # update own parameters from out_lookahead:
        self.agent1.out_lookahead_exact(self.policy.theta, z2_)
        self.agent2.out_lookahead_exact(self.policy.theta, z1_)

if __name__=="__main__":
    args = get_args()
    global ipd
    ipd = IPD(max_steps=args.len_rollout, batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    polen = Polen(args)
    scores = polen.train()
    polen.writer.close()
    if not args.not_plot:
        plot(scores, args.lookaheads)


# if __name__=="__main__":
#     args = get_args()
#     global ipd
#     ipd = IPD(max_steps=args.len_rollout, batch_size=args.batch_size)
#     scores = []
#     for args.seed in [1234,635,234,2445,678]:
#         print('for Seed', args.seed)
#         torch.manual_seed(args.seed)
#         polen = Polen(args)
#         scores.append(polen.train())
#         polen.writer.close()
#     if not args.not_plot:
#         plot_many(scores, args)