# coding: utf-8

import numpy as np
import argparse
import torch
import os
import copy
import time
from tqdm import tqdm
from shutil import rmtree
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from envs import IPD
from ipd_agent import Agent
from models import *
from plotting import *
from ipd_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', default='logdir/',   type=str, help='Logging Directory')
    parser.add_argument('-savedir', default='saved_models_2/', type=str, help='Logging Directory')
    parser.add_argument('-embedding_size', default=5, type=int, help='Size of the embedding z')
    parser.add_argument('-gamma', default=0.96, type=float, help='')
    parser.add_argument('-len_rollout', default=100, type=int, help='Length of rollouts')
    parser.add_argument('-ipd_batch_size', default=128, type=int, help='Batch size of IPD estimates')
    parser.add_argument('-seed', default=911, type=int, help='Random Seed')
    parser.add_argument('-lookaheads', default=2, type=int, help='No of lola lookaheads')
    parser.add_argument('-grads', default='pen', type=str, help='pen, dice, true: Options for LOLA gradients')
    # Learning Rates
    parser.add_argument('-lr_theta', default=0.02, type=float, help='lr for theta using Naive PG')
    parser.add_argument('-lr_out', default=0.2, type=float, help='lr for outer loops of z')
    parser.add_argument('-lr_in', default=0.3, type=float, help='lr for inner loops of z')
    parser.add_argument('-lr_v', default= 0.1, type=float, help='lr for vf for baseline')
    parser.add_argument('-lr_pen', default=0.01, type=float, help='Adam lr for PEN')
    # Ratios
    parser.add_argument('-n_iter', default=200, type=int, help='No of algo iteration')
    parser.add_argument('-n_policy', default=0, type=int, help='No of policy updates for each algo iteration')
    parser.add_argument('-n_pen', default=5, type=int, help='No of pen refining steps per iteration')
    #Policy HPS
    parser.add_argument('-policy_hidden', default=128, type=int, help='Hidden size of Policy Net')
    #PEN HPS
    parser.add_argument('-pen_hidden', default=128, type=int, help='Hidden size of PEN')
    parser.add_argument('-pen_train_size', default=50000, type=int, help='PEN train size')
    parser.add_argument('-pen_test_size', default=1000, type=int, help='PEN test size')
    parser.add_argument('-buffer_size', default=30000, type=int, help='Size of Buffer for PEN Refining')
    parser.add_argument('-pen_batch_size', default=64, type=int, help='Batch size for training PEN')
    parser.add_argument('-pen_epochs', default=3, type=int, help='Epochs for training PEN')
    parser.add_argument('-nbins', default=3, type=int, help='No of bins to discretize return space for KL Variant')
    parser.add_argument('-nsamples_return', default=5, type=int,\
        help='No of rollouts for givenz1/z2 to compute target of return. Usually 2*nbins')
    # Booleans
    parser.add_argument('-not_use_baseline', action='store_true', help='Disable Baseline in Dice')
    parser.add_argument('-not_use_policy_net', action='store_true', help='Disable Baseline in Dice')
    parser.add_argument('-dice_both', action='store_true', help='Use both dependencies in Dice')
    parser.add_argument('-use_pg', action='store_true', help='Use Policy gradient/Dice for theta updates')
    parser.add_argument('-use_kl', action='store_true', help='Use KL for pen loss')
    parser.add_argument('-pen_true_vf', action='store_true', help='Use true vf for training PEN')
    parser.add_argument('-fix_z2', action='store_true', help='Fix z2')
    parser.add_argument('-gpu', action='store_true', help='Use CUDA')
    parser.add_argument('-no_plot', action='store_true', help='Disable Plotting')
    return parser.parse_args()

class Polen():
    def __init__(self, args):
            """
            1. Initialize Agents with embeddings z, shared policy theta & pen
            """
            self.args = args
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.args.gpu) else "cpu")
            print('Using Device: ', self.device)
            # Default ipd object has batch_size=args.pen_batch_size. For any other
            # pass that batch_size in the reset, step functions
            self.ipd = IPD(max_steps=args.len_rollout, batch_size=args.pen_batch_size)

            self.agent1 = Agent(self.args, self.device, self.ipd)
            if self.args.fix_z2:
                self.agent2 = Agent(self.args, self.device, self.ipd, val=torch.tensor([-20.0, 20.0, -20.0, 20.0, -20.0]))
            else:
                self.agent2 = Agent(self.args, self.device, self.ipd)
            
            if self.args.not_use_policy_net:
                self.policy = SteerablePolicy(self.args, self.device)
                self.policy_optimizer = torch.optim.Adam(params=(self.policy.theta,),lr=args.lr_theta)
            else:
                self.policy = SteerablePolicyNet(self.args, self.device)
                self.policy.to(self.device)
                self.policy_optimizer = torch.optim.Adam(params=self.policy.parameters(), lr=args.lr_theta)
            z
            self.buffer = ReplayBuffer(size=self.args.buffer_size, dim=self.args.embedding_size)
            if self.args.use_kl:
                self.pen = PolicyEvaluationNetwork(self.args, self.device, no_avg=True)
            else:
                self.pen = PolicyEvaluationNetwork_2(self.args, self.device)
            self.pen.to(self.device)
            self.pen_optimizer = torch.optim.Adam(params=self.pen.parameters(), lr=args.lr_pen)
            # self.pen_scheduler = torch.optim.lr_scheduler.StepLR(self.pen_optimizer, step_size=10, gamma=0.9)
            self.pth_1 = '2l_tanh/' + ('true_vf/' if self.args.pen_true_vf else ('sampled/num_' + str(self.args.nsamples_return)))
            self.pth_2 = ('_KL_' if self.args.use_kl else '') + '_policy_h_' + str(self.args.policy_hidden) + '_h_' + str(self.args.pen_hidden) + '_train_' + str(self.args.pen_train_size) + '_bsz_' + str(self.args.pen_batch_size) + '_ep_' + str(self.args.pen_epochs) + '_lr_'+ str(self.args.lr_pen) + '_seed_' + str(self.args.seed) + ('_gpu_' if self.args.gpu else '') + '.pth'
            self.args.logdir = args.logdir + self.pth_1 + 'look_' + str(args.lookaheads) + '/n_policy_' + str(args.n_policy) + '_n_pen' + str(args.n_pen) + '_lr_theta_' + str(args.lr_theta) + '_lr_in_' + str(args.lr_in)  + '_lr_out_' + str(args.lr_out) + '_emb_' + str(args.embedding_size) + self.pth_2
            if os.path.exists(self.args.logdir):
                print('Removing existing')
                rmtree(self.args.logdir)
            self.writer = SummaryWriter(self.args.logdir)
  
    def train(self):
        """ Main function to 
        1. Initialize PEN
        2. Do LOLA updates while refining PEN.
        """ 

        print(f"start iterations with {self.args.lookaheads} lookaheads")
        joint_scores = []
        PATH = self.args.savedir + self.pth_1 + self.pth_2
        # PATH = 'saved_models/2l_tanh/true_vf/_h_128_train_50000_bsz_128_ep_3_lr_0.002_seed_911.pth'
        if os.path.exists(PATH):
            print('Loading Saved Model from : ', PATH)
            self.pen.load_state_dict(torch.load(PATH))
        elif self.args.grads == 'pen':
            print('Training PEN from Sratch------------')
            pen_train = PenDataset(self.args.pen_train_size, self.args.embedding_size)
            pen_test = PenDataset(self.args.pen_test_size, self.args.embedding_size)
            dloader_train = DataLoader(pen_train, batch_size=self.args.pen_batch_size)

            pen_steps = 0
            for epoch in range(1,self.args.pen_epochs + 1):
                for t, sampled_batch in enumerate(dloader_train):
                    z1s, z2s = sampled_batch
                    pen_train_loss = self.pen_update(z1s, z2s, use_true_vf=self.args.pen_true_vf)
                    self.writer.add_scalar('PEN_LOSS/Train', pen_train_loss, pen_steps*self.args.pen_batch_size)
                    if t%100 == 0:
                        #Compute test stats
                        pen_test_loss = self.pen_update(pen_test.z1s, pen_test.z2s, eval=True, use_true_vf=self.args.pen_true_vf)
                        self.writer.add_scalar('PEN_LOSS/Test', pen_test_loss, pen_steps*self.args.pen_batch_size)
                        print('Epoch: {}, After {} iter,  Train Loss: {} , Test Loss: {}'.format(
                            epoch, t, pen_train_loss, pen_test_loss))
                    pen_steps += 1
                    # self.writer.add_scalar('MSE V1 and R1', mse_1, t)
                    # self.writer.add_scalar('MSE V2 and R2', mse_2, t)
            print('Saving Model to : ', PATH)
            torch.save(self.pen.state_dict(), PATH)
        
        # Start main algo iterations
        for update in range(self.args.n_iter):
            # 1a. For fixed z1 & z2, Learn steerable policy theta & PEN by maximizing rollouts
            if self.args.n_policy and update % self.args.n_policy == 0:
            # for t in range(self.args.n_policy):
                # Update steerable policy parameters. True possible for IPD
                if not self.args.use_pg:
                    policy_loss = self.policy_update_true(step=update)
                else:
                    policy_loss = self.policy_update_pg()
                self.writer.add_scalar('PolicyObjective V1 plus V2', -policy_loss, update)
                # self.writer.add_scalar('PolicyObjective V1 plus V2', -policy_loss, update*self.args.n_policy + t )
            
            if self.args.grads == 'pen':
                self.buffer.add_surround(self.agent1.z.data, self.agent2.z.data, num_samples=self.args.pen_batch_size)
                # Refine the PEN
                if update%1==0 :
                    for t in range(self.args.n_pen):
                        z1s, z2s = self.buffer.sample_batch(batch_size=self.args.pen_batch_size)
                        # # Generate z1s & z2s aroud z1, z2 sampled from the buffer
                        # z1s = (torch.rand((self.args.pen_batch_size,self.args.embedding_size)).to(self.device)) + z1_sample
                        # z2s = (torch.rand((self.args.pen_batch_size,self.args.embedding_size)).to(self.device)) + z2_sample
                        pen_train_loss = self.pen_update(z1s.to(self.device), z2s.to(self.device), use_true_vf=self.args.pen_true_vf)
                        self.writer.add_scalar('PEN_Refine/Train', pen_train_loss, update*self.args.n_pen + t)
                    # self.pen_scheduler.step()
                # print('New PEN lr rate ', self.pen_optimizer.param_groups[0]['lr'])

            # 2. Do Lola Updates
            if self.args.grads == 'pen':
                in_grads, out_grads = self.lola_update_pen(fix_z2=self.args.fix_z2)
                self.writer.add_scalar('LOLA_Grads/MSE', in_grads[0], update)
                self.writer.add_scalar('LOLA_Grads/COS_Similarity_1', in_grads[1], update)
                self.writer.add_scalar('LOLA_Grads/COS_Similarity_2', in_grads[2], update)
                self.writer.add_scalar('LOLA_Grads/Norm_PEN', in_grads[3], update)
                
                self.writer.add_scalar('Outer_LOLA_Grads/MSE', out_grads[0], update)
                self.writer.add_scalar('Outer_LOLA_Grads/COS_Similarity_1', out_grads[1], update)
                self.writer.add_scalar('Outer_LOLA_Grads/COS_Similarity_2', out_grads[2], update)
                self.writer.add_scalar('Outer_LOLA_Grads/Norm_PEN', out_grads[3], update)
            elif self.args.grads == 'dice':
                self.lola_update_dice()
            elif self.args.grads == 'true':
                self.lola_update_exact()
            else:
                raise NotImplementedError 
            
            # Evaluate:
            score = self.rollout(self.args.len_rollout)
            avg_score = 0.5*(score[0] + score[1])
            self.writer.add_scalar('Avg Score of Agent', avg_score, update)
            joint_scores.append(avg_score)

            # Logging
            if update%10==0 :
                # import ipdb; ipdb.set_trace()
                p0 = [p.item() for p in torch.sigmoid(self.policy.fwd(torch.zeros_like(self.agent1.z)))]
                p1 = [p.item() for p in torch.sigmoid(self.policy.fwd(self.agent1.z))]
                p2 = [p.item() for p in torch.sigmoid(self.policy.fwd(self.agent2.z))]
                self.writer.add_image('Defect Probs', plot_scatter(p0, p1, p2), update) 
                print('After update', update, '------------')
                v_env_1, v_env_2 = score[0], score[1]
                v_true_1 = -true_objective(self.policy.fwd(self.agent1.z), self.policy.fwd(self.agent2.z), self.ipd).data
                v_true_2 = -true_objective(self.policy.fwd(self.agent2.z), self.policy.fwd(self.agent1.z), self.ipd).data
                v_pen_1  = -self.pen.predict(self.agent1.z, self.agent2.z).data
                v_pen_2 =  -self.pen.predict(self.agent2.z, self.agent1.z).data
                print('Sampled Score in Env (%.3f,%.3f)' % (v_env_1, v_env_2))
                print('True Value ({}, {})'.format(v_true_1, v_true_2))
                print('PEN Value ({}, {})'.format(v_pen_1, v_pen_2))
                print('Default = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}\n' % (p0[0], p0[1], p0[2], p0[3], p0[4]),\
                    '(agent1) = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}\n' % (p1[0], p1[1], p1[2], p1[3], p1[4]),\
                    '(agent2) = {S: %.3f, DD: %.3f, DC: %.3f, CD: %.3f, CC: %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
                self.writer.add_scalar('Value_Agent_1/Sampled in env', v_env_1, update)
                self.writer.add_scalar('Value_Agent_1/True Value', v_true_1, update)
                self.writer.add_scalar('Value_Agent_1/PEN Value', v_pen_1, update)
                self.writer.add_scalar('Value_Agent_2/Sampled in env', v_env_2, update)
                self.writer.add_scalar('Value_Agent_2/True Value', v_true_2, update)
                self.writer.add_scalar('Value_Agent_2/PEN Value', v_pen_2, update)
        return joint_scores, p0, p1, p2
   
    def pen_update(self, z1s, z2s, eval=False, use_true_vf=True):
        """ Method to update PEN with MSE loss

        Args:
            z1s (bsz x dimension):
            z2s (bsz x dimension):
            eval (bool, optional): Run in eval mode. No gradient steps on PEN. Defaults to False.
            use_true_vf (bool, optional): Whether to use True Value Function as target or 
            use returns computed from the environment. Defaults to True.

        Returns:
            pen_loss: The PEN MSE loss computed for the batch
        """
        if use_true_vf:
            target_1, target_2 = [], []
            # TODO: See if can parallelize this
            for i in range(z1s.shape[0]):
                target_1.append(-true_objective(self.policy.fwd(z1s[i]), self.policy.fwd(z2s[i]), self.ipd))
                target_2.append(-true_objective(self.policy.fwd(z2s[i]), self.policy.fwd(z1s[i]), self.ipd))
            target_1 = torch.tensor(target_1).to(self.device)
            target_2 = torch.tensor(target_2).to(self.device)
        else:
            target_1, target_2 = torch.zeros(z1s.shape[0]), torch.zeros(z2s.shape[0])
            for _ in range(self.args.nsamples_return):
                t_1, t_2 = self.rollout_binning_batch(self.args.len_rollout, z1s, z2s)
                target_1 +=  t_1
                target_2 += t_2
            target_1 = target_1/self.args.nsamples_return
            target_2 = target_2/self.args.nsamples_return

        # Compute the PEN Values
        w1 = self.pen.forward(z1s, z2s)
        w2 = self.pen.forward(z2s, z1s)
        pen_loss = ((w1.squeeze(1) - target_1)**2 + (w2.squeeze(1) - target_2)**2).mean()

        if not eval:
            self.pen_optimizer.zero_grad()
            pen_loss.backward()
            self.pen_optimizer.step()
        return pen_loss

    def lola_update_pen(self, fix_z2=False):
        """ Do lola updates on agent parameters using Gradients from the PEN

        Args:
            fix_z2 (bool, optional): Fix Agent 2's parameters and do learning only
            for Agent 1. To make the environment stationary for Debugging. Defaults to False.

        Returns:
            in_grads: List of gradient statistics in inner loop of lola
            out_grads: List of gradient statistics in outer loop of lola
        """ 
        # copy other's parameters:
        z1_ = self.agent1.z.clone().detach().requires_grad_(True)
        z2_ = self.agent2.z.clone().detach().requires_grad_(True)
        grad_diff = 0
        grad_cos_1 = 0
        grad_cos_2 = 0
        grads = 0
        out_grad_cos_1 = 0
        out_grad_cos_2 = 0
        # Update clones using lookaheads
        for k in range(self.args.lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = self.agent1.in_lookahead_pen(self.pen, z2_)
            grad1 = self.agent2.in_lookahead_pen(self.pen, z1_)
            grad2_ex = self.agent1.in_lookahead_exact(self.policy, z2_)
            grad1_ex = self.agent2.in_lookahead_exact(self.policy, z1_)
            # update other's theta
            z2_ = z2_ - self.args.lr_in * grad2
            z1_ = z1_ - self.args.lr_in * grad1
            grads = grads + torch.norm(grad1)
            grad_diff += F.mse_loss(grad1, grad1_ex) + F.mse_loss(grad2, grad2_ex)
            grad_cos_1 += F.cosine_similarity(grad1.unsqueeze(0), grad1_ex.unsqueeze(0))
            grad_cos_2 += F.cosine_similarity(grad2.unsqueeze(0), grad2_ex.unsqueeze(0))
        # update own parameters from out_lookahead:
        # Need to clone because grad is just a reference
        grad_z1_pen = self.agent1.out_lookahead_pen(self.pen, z2_).clone()
        grad_z1_exact = self.agent1.out_lookahead_exact(self.policy, z2_, eval=True)

        # if we fix_z2, then equal to just doing eval on it
        grad_z2_pen = self.agent2.out_lookahead_pen(self.pen, z1_, eval=fix_z2).clone()
        grad_z2_exact = self.agent2.out_lookahead_exact(self.policy, z1_, eval=True)

        out_grad_cos_1 = F.cosine_similarity(grad_z1_pen.unsqueeze(0), grad_z1_exact.unsqueeze(0))
        out_grad_cos_2 = F.cosine_similarity(grad_z2_pen.unsqueeze(0), grad_z2_exact.unsqueeze(0))
        out_grad_diff = F.mse_loss(grad_z1_pen, grad_z1_exact) + F.mse_loss(grad_z2_pen, grad_z2_exact)
        out_grad_norm = torch.norm(grad_z1_pen) + torch.norm(grad_z2_pen)

        out_grads = [out_grad_diff, out_grad_cos_1, out_grad_cos_2, out_grad_norm]
        if self.args.lookaheads > 0:
            in_grads = [grad_diff/self.args.lookaheads, grad_cos_1/self.args.lookaheads, grad_cos_2/self.args.lookaheads, grads/self.args.lookaheads]
        else:
            in_grads = [grad_diff, grad_cos_1, grad_cos_2, grads]
        return in_grads, out_grads

    def pen_update_kl(self):
        """ Method to update PEN with KL Dive loss. DEPRECATED CURRENTLY

        Args:
            z1s (bsz x dimension):
            z2s (bsz x dimension):
            eval (bool, optional): Run in eval mode. No gradient steps on PEN. Defaults to False.
            use_true_vf (bool, optional): Whether to use True Value Function as target or 
            use returns computed from the environment. Defaults to True.

        Returns:
            pen_loss: The PEN MSE loss computed for the batch
        """  
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
        mse_1 = (-v1 - avg_return_1/(1-self.args.gamma))**2
        mse_2 = (-v2 - avg_return_2/(1-self.args.gamma))**2

        self.pen_optimizer.zero_grad()
        # Note: F.kl_div(Q.log(), P) computes KL(P || Q) 
        pen_loss = F.kl_div(w1.squeeze().log(), hist1) + F.kl_div(w2.squeeze().log(), hist2)
        pen_loss.backward()
        self.pen_optimizer.step()

        return pen_loss, mse_1, mse_2
 
    def policy_update_true(self, step):
        # Batching not needed here.
        self.policy_optimizer.zero_grad()
        # if self.args.not_use_policy_net:
        #     theta_ = self.policy.theta.clone().detach()
        #     objective = true_objective(self.agent1.z + self.policy.theta, self.agent2.z + theta_, self.ipd)\
        #         + true_objective(self.agent2.z + self.policy.theta, self.agent1.z + theta_, self.ipd)
        # else:
        # objective = true_objective(self.policy.fwd(self.agent1.z), self.policy.fwd(self.agent2.z), self.ipd)\
            # + true_objective(self.policy.fwd(self.agent2.z), self.policy.fwd(self.agent1.z), self.ipd)
        objective = true_objective(self.policy.fwd(self.agent1.z), self.policy.fwd(self.agent2.z).detach(), self.ipd)\
            + true_objective(self.policy.fwd(self.agent2.z), self.policy.fwd(self.agent1.z).detach(), self.ipd)
        # TODO: See what this retain_graph does. Some multiprocessing stuff.
        objective.backward(retain_graph=True)
        self.policy_optimizer.step()

        # grad_norm = self.policy.linear1.weight.grad.norm()+ self.policy.linear2.weight.grad.norm()
        # self.writer.add_scalar('Policy/Grad_Norms', grad_norm, step)

        return objective
    
    def policy_update_pg(self):
        """
        Policy Gradient for theta using Dice Objective. Treat theta as shared parameters
        """
        # First using Agent 1
        (s1, s2), _ = self.ipd.reset(self.args.ipd_batch_size)
        memory1 = Memory(self.args)
        memory2 = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = self.policy.act(s1, self.agent1.z, self.agent1.values)
            a2, lp2, v2 = self.policy.act(s2, self.agent2.z, self.agent2.values)
            (s1, s2), (r1, r2),_,_ = self.ipd.step((a1, a2), self.args.ipd_batch_size)
            memory1.add(lp1, lp2, v1, torch.from_numpy(r1).float())
            memory2.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        # update self z
        objective = memory1.dice_objective(dice_both=self.args.dice_both) +  memory2.dice_objective(dice_both=self.args.dice_both)
        # self.policy.policy_update(objective)
        self.policy_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.policy_optimizer.step()
        # # update self value:
        self.agent1.value_update(memory1.value_loss())
        self.agent2.value_update(memory2.value_loss())
        return objective

    def lola_update_exact(self):
        """
        Do Lola Updates using true value function
        """
        # copy other's parameters:
        z1_ = self.agent1.z.clone().detach().requires_grad_(True)
        z2_ = self.agent2.z.clone().detach().requires_grad_(True)

        for k in range(self.args.lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = self.agent1.in_lookahead_exact(self.policy, z2_)
            grad1 = self.agent2.in_lookahead_exact(self.policy, z1_)
            # update other's theta
            z2_ = z2_ - self.args.lr_in * grad2
            z1_ = z1_ - self.args.lr_in * grad1

        # update own parameters from out_lookahead:
        self.agent1.out_lookahead_exact(self.policy, z2_)
        self.agent2.out_lookahead_exact(self.policy, z1_)

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

    def rollout(self, nsteps):
        # just to evaluate progress:
        (s1, s2), _ = self.ipd.reset(self.args.ipd_batch_size)
        score1 = 0
        score2 = 0
        for t in range(nsteps):
            a1, lp1, v1 = self.policy.act(s1, self.agent1.z, self.agent1.values)
            a2, lp2, v2 = self.policy.act(s2, self.agent2.z, self.agent2.values)
            (s1, s2), (r1, r2),_,_ = self.ipd.step((a1, a2), self.args.ipd_batch_size)
            # cumulate scores
            score1 += np.mean(r1)/float(self.args.len_rollout)
            score2 += np.mean(r2)/float(self.args.len_rollout)
        return (score1, score2)

    def rollout_binning(self, nsteps, z1, z2):
        # just to evaluate progress:
        (s1, s2), _ = self.ipd.reset(self.args.nsamples_return)
        score1 = np.zeros(self.args.nsamples_return)
        score2 = np.zeros(self.args.nsamples_return)
        gamma_t = 1
        for t in range(nsteps):
            a1, lp1 = self.policy.act(s1, z1)
            a2, lp2 = self.policy.act(s2, z2)
            (s1, s2), (r1, r2),_,_ = self.ipd.step((a1, a2), self.args.nsamples_return)
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
        """Compute the Value Targets by sampling in the environment using 
        Steerable policy conditioned on strategy vectors.

        Args:
            nsteps (int): No of time-steps in the environment
            z1s (bsz x dim): Batch of z1s
            z2s (bsz x dim): Batch of z2s

        Returns:
            score1 (bsz): Value Targets for agent 1
            score2 (bsz): Value Targets for agent 2
        """
        bsz = z1s.shape[0]
        (s1, s2), _ = self.ipd.reset(batch_size=bsz)
        score1 = np.zeros(bsz)
        score2 = np.zeros(bsz)
        gamma_t = 1
        for t in range(nsteps):
            a1, lp1 = self.policy.act_parallel(s1, z1s)
            a2, lp2 = self.policy.act_parallel(s2, z2s)
            (s1, s2), (r1, r2),_,_ = self.ipd.step((a1, a2), batch_size=bsz)
            # Accrue discounted scores
            score1 += gamma_t*r1
            score2 += gamma_t*r2
            gamma_t = gamma_t*self.args.gamma
        score1 = torch.tensor(score1, dtype=torch.float32)
        score2 = torch.tensor(score2, dtype=torch.float32)
        return score1, score2
  
if __name__=="__main__":
    start_time = time.time()
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    polen = Polen(args)
    scores, p0, p1, p2 = polen.train()
    # Debugging
    # print('tft vs tft', polen.pen(torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]]), torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]])))
    # print('tft vs DD',  polen.pen(torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]]), torch.tensor([[20.0, 20.0,20.0, 20.0, 20.0]])))
    # print('DD vs CC', polen.pen(torch.tensor([[20.0, 20.0, 20.0, 20.0, 20.0]]), torch.tensor([[-20.0, -20.0, -20.0, -20.0, -20.0]])))
    # print('DD vs tft', polen.pen(torch.tensor([[20.0, 20.0, 20.0, 20.0, 20.0]]), torch.tensor([[-20.0, 20.0, -20.0, 20.0, -20.0]])))
    # print('DD vs rnd', polen.pen(torch.tensor([[20.0, 20.0, 20.0, 20.0, 20.0]]), torch.tensor([[-20.0, -20.0, 20.0, 20.0, 20.0]])))
    if not args.no_plot:
        plot(scores, args.lookaheads, polen.args.logdir)
        plot_scatter(p0, p1, p2, polen.args.logdir)
    polen.writer.close()
    print("--- Execution Finished in %s seconds ---" % (time.time() - start_time))

