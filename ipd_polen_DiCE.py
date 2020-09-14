# coding: utf-8

import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
from shutil import rmtree
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from envs import IPD
from models import PolicyEvaluationNetwork, SteerablePolicy, PolicyEvaluationNetwork_2
from plotting import *
from ipd_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_theta', default=0.02, type=float, help='lr for theta using Naive PG')
    parser.add_argument('-lr_out', default=0.2, type=float, help='lr for outer loops of z')
    parser.add_argument('-lr_in', default=0.3, type=float, help='lr for inner loops of z')
    parser.add_argument('-lr_v', default= 0.1, type=float, help='lr for vf for baseline')
    parser.add_argument('-lr_pen', default=0.3, type=float, help='Adam lr for PEN')
    parser.add_argument('-gamma', default=0.96, type=float, help='')
    parser.add_argument('-n_iter', default=200, type=int, help='No of algo iteration')
    parser.add_argument('-len_rollout', default=200, type=int, help='Length of rollouts')
    parser.add_argument('-batch_size', default=128, type=int, help='')
    parser.add_argument('-seed', default=4, type=int, help='Random Seed')
    parser.add_argument('-lookaheads', default=2, type=int, help='No of lola lookaheads')
    parser.add_argument('-embedding_size', default=5, type=int, help='Size of the embedding z')
    parser.add_argument('-nbins', default=25, type=int, help='No of bins to discretize return space')
    parser.add_argument('-logdir', default='logdir/', type=str, help='Logging Directory')
    parser.add_argument('-pen_hidden', default=80, type=int, help='Hidden size of PEN')
    parser.add_argument('-n_policy', default=1, type=int, help='No of policy updates for each algo iteration')
    parser.add_argument('-n_pen', default=10, type=int, help='')
    parser.add_argument('-pen_train_size', default=10000, type=int, help='')
    parser.add_argument('-pen_test_size', default=1000, type=int, help='')
    parser.add_argument('-pen_batch_size', default=16, type=int, help='Batch size for training PEN')
    parser.add_argument('-pen_epochs', default=2, type=int, help='Batch size for training PEN')
    parser.add_argument('-nsamples_bin', default=75, type=int, help='No of rollouts for given z1/z2 to compute histogram of returns. Usually 2*nbins')
    parser.add_argument('-not_plot', action='store_true', help='Disable Plotting')
    parser.add_argument('-use_cuda', action='store_true', help='Use CUDA')
    parser.add_argument('-use_kl', action='store_true', help='Use KL for pen loss')
    parser.add_argument('-not_use_baseline', action='store_true', help='Disable Baseline in Dice')
    parser.add_argument('-only_self', action='store_true', help='Enable Plotting')
    parser.add_argument('-use_exact_lola', action='store_true', help='Disable Plotting')
    return parser.parse_args()

class Agent():
    def __init__(self, args, device):
        # init z and its optimizer
        self.args = args
        self.device = device
        self.z = nn.Parameter(torch.zeros(self.args.embedding_size, requires_grad=True).to(self.device))
        self.z_optimizer = torch.optim.Adam((self.z,),lr=self.args.lr_out)
        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(self.args.embedding_size, requires_grad=True).to(self.device))
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
    
    def in_lookahead_pen(self, pen, other_z):
        other_objective = pen.predict(other_z, self.z)
        grad = get_gradient(other_objective, other_z)
        return grad

    def out_lookahead_pen(self, pen, other_z):
        objective = pen.predict(self.z, other_z)
        self.z_update(objective)
        return

class Polen():
    def __init__(self, args):
            """
            1. Initialize Agent embeddings z, poliy theta & pen psi
            """
            self.args = args
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.args.use_cuda) else "cpu")
            print('Using Device: ', self.device)
            self.agent1 = Agent(self.args, self.device)
            self.agent2 = Agent(self.args, self.device)
            self.policy = SteerablePolicy(self.args, self.device)
            if self.args.use_kl:
                self.pen = PolicyEvaluationNetwork(self.args, self.device, no_avg=True)
            else:
                self.pen = PolicyEvaluationNetwork_2(self.args, self.device)
            self.pen.to(self.device)
            self.pen_optimizer = torch.optim.Adam(params=self.pen.parameters(), lr=args.lr_pen)
            self.ipd2 = IPD(max_steps=args.len_rollout, batch_size=args.nsamples_bin)
            self.ipd3 = IPD(max_steps=args.len_rollout, batch_size=args.pen_batch_size)
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
        score1 = np.zeros(self.args.nsamples_bin)
        score2 = np.zeros(self.args.nsamples_bin)
        gamma_t = 1
        for t in range(nsteps):
            a1, lp1 = self.policy.act(s1, z1)
            a2, lp2 = self.policy.act(s2, z2)
            (s1, s2), (r1, r2),_,_ = self.ipd2.step((a1, a2))
            # cumulate scores
            score1 += gamma_t*r1
            score2 += gamma_t*r2
            gamma_t = gamma_t*self.args.gamma
        # score1 = torch.tensor(score1*(1-self.args.gamma), dtype=torch.float32)
        # score2 = torch.tensor(score2*(1-self.args.gamma), dtype=torch.float32)
        # hist1 = torch.histc(score1, bins=self.args.nbins, min=-3.0, max=0.0)
        # hist2 = torch.histc(score2, bins=self.args.nbins, min=-3.0, max=0.0)

        score1 = torch.tensor(score1, dtype=torch.float32)
        score2 = torch.tensor(score2, dtype=torch.float32)
        hist1 = torch.histc(score1, bins=self.args.nbins, min=-75.0, max=0.0)
        hist2 = torch.histc(score2, bins=self.args.nbins, min=-75.0, max=0.0)
        return hist1, hist2, score1.mean(), score2.mean()

    def rollout_binning_batch(self, nsteps, z1s, z2s):
        # just to evaluate progress:
        (s1, s2), _ = self.ipd3.reset()
        score1 = np.zeros(z1s.shape[0])
        score2 = np.zeros(z1s.shape[0])
        gamma_t = 1
        for t in range(nsteps):
            a1, lp1 = self.policy.act_parallel(s1, z1s)
            a2, lp2 = self.policy.act_parallel(s2, z2s)
            (s1, s2), (r1, r2),_,_ = self.ipd3.step((a1, a2))
            # cumulate scores
            try:
                score1 += gamma_t*r1
                score2 += gamma_t*r2
                gamma_t = gamma_t*self.args.gamma
            except:
                import ipdb; ipdb.set_trace()
        score1 = torch.tensor(score1, dtype=torch.float32)
        score2 = torch.tensor(score2, dtype=torch.float32)
        return score1, score2
    
    def train(self):
        print("start iterations with", self.args.lookaheads, "lookaheads:")
        joint_scores = []

        # Train PEN 
        pen_train = PenDataset(self.args.pen_train_size, self.args.embedding_size)
        pen_test = PenDataset(self.args.pen_test_size, self.args.embedding_size)
        dloader_train = DataLoader(pen_train, batch_size=self.args.pen_batch_size)
        # dloader_test = DataLoader(pen_test, batch_size=self.args.pen_test_size//10)
        pen_steps = 0
        for epoch in range(1,self.args.pen_epochs + 1):
            for t, sampled_batch in enumerate(dloader_train):
                z1s, z2s = sampled_batch
                pen_train_loss = self.pen_update_no_kl(z1s, z2s, use_true_vf=True)
                self.writer.add_scalar('PEN_LOSS/Train', pen_train_loss, pen_steps*self.args.pen_batch_size)
                if t%100 == 0:
                    #Compute test stats
                    pen_test_loss = self.pen_update_no_kl(pen_test.z1s, pen_test.z2s, eval=True, use_true_vf=True)
                    self.writer.add_scalar('PEN_LOSS/Test', pen_test_loss, pen_steps*self.args.pen_batch_size)
                    print('Epoch: {}, After {} iter,  Train Loss: {} , Test Loss: {}'.format(epoch, t, pen_train_loss, pen_test_loss))
                pen_steps += 1
                # self.writer.add_scalar('MSE V1 and R1', mse_1, t)
                # self.writer.add_scalar('MSE V2 and R2', mse_2, t)

        
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

            # # 1b. Train the PEN
            # if update%4==0 :
            #     for t in range(self.args.n_pen):
            #         pen_loss = self.pen_update()
            #         self.writer.add_scalar('PEN Loss: KL1 plus KL2', pen_loss, update*self.args.n_pen/4 + t )

            # 2. Do on Lola Updates
            if self.args.use_exact_lola:
                # self.lola_update_exact()
                grad_diff, grad_cos_1, grad_cos_2, grads = self.lola_update_pen()
                self.writer.add_scalar('LOLA_Grads/Grad_MSE', grad_diff, update)
                self.writer.add_scalar('LOLA_Grads/Grad_COS_1', grad_cos_1, update)
                self.writer.add_scalar('LOLA_Grads/Grad_COS_2', grad_cos_2, update)
                self.writer.add_scalar('LOLA_Grads/Norm of Gradients from PEN', grads, update)
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

                # self.writer.add_figure('Scores', plot_bar(p0, p1, p2),update) 
                print('After update', update, '------------')
                v_env_1, v_env_2 = score[0], score[1]
                v_true_1, v_true_2 = -true_objective(self.policy.theta + self.agent1.z, self.policy.theta + self.agent2.z, ipd).data, -true_objective(self.policy.theta + self.agent2.z, self.policy.theta + self.agent1.z, ipd).data
                v_pen_1, v_pen_2 = -self.pen.predict(self.agent1.z, self.agent2.z).data, -self.pen.predict(self.agent2.z, self.agent1.z).data
                print('Sampled Score in Env (%.3f,%.3f)' % (v_env_1, v_env_2))
                print('True Value ({}, {})'.format(v_true_1, v_true_2))
                print('PEN Value ({}, {})'.format(v_pen_1, v_pen_2))
                print('Default = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}\n' % (p0[0], p0[1], p0[2], p0[3], p0[4]), '(agent1) = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}\n' % (p1[0], p1[1], p1[2], p1[3], p1[4]), '(agent2) = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
                self.writer.add_scalar('Value_Agent_1/Sampled in env', v_env_1, update)
                self.writer.add_scalar('Value_Agent_1/True Value', v_true_1, update)
                self.writer.add_scalar('Value_Agent_1/PEN Value', v_pen_1, update)
                self.writer.add_scalar('Value_Agent_2/Sampled in env', v_env_2, update)
                self.writer.add_scalar('Value_Agent_2/True Value', v_true_2, update)
                self.writer.add_scalar('Value_Agent_2/PEN Value', v_pen_2, update)
        return joint_scores, p0, p1, p2

    def pen_update(self):
        # randomly generate z1, z2. Maybe generation centered on z0, z1 would be better.
        z1 = sigmoid_inv(torch.rand(self.args.embedding_size))
        z2 = sigmoid_inv(torch.rand(self.args.embedding_size))

        # Experiment with smaller length of rollouts for estimation
        hist1, hist2, avg_return_1, avg_return_2 = self.rollout_binning(self.args.len_rollout, z1, z2)

        # Compute the KL Div
        w1 = self.pen.forward(self.agent1.z.unsqueeze(0), self.agent2.z.unsqueeze(0))
        w2 = self.pen.forward(self.agent2.z.unsqueeze(0), self.agent1.z.unsqueeze(0))

        v1 = self.pen.predict(self.agent1.z, self.agent2.z)
        v2 = self.pen.predict(self.agent2.z, self.agent1.z)

        w1 = F.softmax(w1.squeeze(), dim=0)
        w2 = F.softmax(w2.squeeze(), dim=0)
        # F.kl_div(Q.log(), P, None, None, 'sum')
        mse_1 = (-v1 - avg_return_1/(1-self.args.gamma))**2
        mse_2 = (-v2 - avg_return_2/(1-self.args.gamma))**2

        self.pen_optimizer.zero_grad()
        # pen_loss = (hist1* (hist1 / w1).log()).sum() + (hist2* (hist2 / w2).log()).sum()
        # F.kl_div(Q.log(), P) computes KL(P || Q) 
        pen_loss = F.kl_div(w1.squeeze().log(), hist1) + F.kl_div(w2.squeeze().log(), hist2)
        pen_loss.backward()
        self.pen_optimizer.step()

        # print('z1: ', torch.sigmoid(z1))
        # print('z2: ', torch.sigmoid(z2))
        # print('return_1: ', avg_return_1)
        # print('return_2: ', avg_return_2)
        # print('v1: ', v1)
        # print('v2: ', v2)
        # print('-----------------')
        # print('loss:', pen_loss)

        return pen_loss, mse_1, mse_2
    
    def pen_update_no_kl(self, z1s, z2s, eval=False, use_true_vf=True):
        # randomly generate z1, z2. Maybe generation centered on z0, z1 would be better.
        # z1 = sigmoid_inv(torch.rand(self.args.embedding_size))
        # z2 = sigmoid_inv(torch.rand(self.args.embedding_size))
        
        # Experiment with smaller length of rollouts for estimation
        if use_true_vf:
            target_1, target_2 = [], []
            # TODO: See if can parallelize this
            for i in range(z1s.shape[0]):
                target_1.append(-true_objective(self.policy.theta.cpu()  + z1s[i], self.policy.theta.cpu()  + z2s[i], ipd))
                target_2.append(-true_objective(self.policy.theta.cpu() + z2s[i], self.policy.theta.cpu()  + z1s[i], ipd))
            target_1 = torch.tensor(target_1).to(self.device)
            target_2 = torch.tensor(target_2).to(self.device)
        else:
            target_1, target_2 = self.rollout_binning_batch(self.args.len_rollout, z1s, z2s)

        # Compute the PEN Values
        w1 = self.pen.forward(z1s, z2s)
        w2 = self.pen.forward(z2s, z1s)

        pen_loss = ((w1.squeeze(1) - target_1)**2 + (w2.squeeze(1) - target_2)**2).mean()

        if not eval:
            self.pen_optimizer.zero_grad()
            pen_loss.backward()
            self.pen_optimizer.step()
        # print('z1: ', torch.sigmoid(z1))
        # print('z2: ', torch.sigmoid(z2))
        # print('return_1: ', avg_return_1)
        # print('return_2: ', avg_return_2)
        # print('w1: ', w1)
        # print('w2: ', w2)
        # print('-----------------')
        # print('loss:', pen_loss)
        return pen_loss
    
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
    
    def lola_update_pen(self):
        """
        Do Lola Updates using pen
        """
        # copy other's parameters:
        z1_ = self.agent1.z.clone().detach().requires_grad_(True)
        z2_ = self.agent2.z.clone().detach().requires_grad_(True)
        grad_diff = 0
        grad_cos_1 = 0
        grad_cos_2 = 0
        grads = 0
        for k in range(self.args.lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = self.agent1.in_lookahead_pen(self.pen, z2_)
            grad1 = self.agent2.in_lookahead_pen(self.pen, z1_)
            grad2_ex = self.agent1.in_lookahead_exact(self.policy.theta, z2_)
            grad1_ex = self.agent2.in_lookahead_exact(self.policy.theta, z1_)
            # update other's theta
            z2_ = z2_ - self.args.lr_in * grad2
            z1_ = z1_ - self.args.lr_in * grad1
            grads = grads + torch.norm(grad1)
            grad_diff += F.mse_loss(grad1, grad1_ex) + F.mse_loss(grad2, grad2_ex)
            grad_cos_1 += F.cosine_similarity(grad1.unsqueeze(0), grad1_ex.unsqueeze(0))
            grad_cos_2 += F.cosine_similarity(grad2.unsqueeze(0), grad2_ex.unsqueeze(0))
        # update own parameters from out_lookahead:
        self.agent1.out_lookahead_pen(self.pen, z2_)
        self.agent2.out_lookahead_pen(self.pen, z1_)
        if self.args.lookaheads > 0:
            return grad_diff/self.args.lookaheads, grad_cos_1/self.args.lookaheads, grad_cos_2/self.args.lookaheads, grads/self.args.lookaheads
        else:
            return grad_diff, grad_cos_1, grad_cos_2, grads

if __name__=="__main__":
    args = get_args()
    global ipd
    
    ipd = IPD(max_steps=args.len_rollout, batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    polen = Polen(args)
    scores, p0, p1, p2 = polen.train()
    # Debugging
    print('tft vs tft', polen.pen(torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]]), torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]]))                                                                                                                                            )
    print('tft vs DD', polen.pen(torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]]), torch.tensor([[20.0, 20.0,20.0, 20.0, 20.0]]))                                                                                                                                                )
    print('DD vs CC', polen.pen(torch.tensor([[20.0, 20.0, 20.0, 20.0, 20.0]]), torch.tensor([[-20.0, -20.0, -20.0, -20.0, -20.0]]))                                                                                                                                             )
    print('DD vs tft', polen.pen(torch.tensor([[20.0, 20.0, 20.0, 20.0, 20.0]]), torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]]))                                                                                                                                               )
    print('DD vs rnd', polen.pen(torch.tensor([[20.0, 20.0, 20.0, 20.0, 20.0]]), torch.tensor([[-20.0, -20.0, 20.0, 20.0, 20.0]])))
    if not args.not_plot:
        plot(scores, args.lookaheads)
    polen.writer.close()


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