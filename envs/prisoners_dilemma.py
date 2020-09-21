"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple

from .common import OneHot


class IteratedPrisonersDilemma(gym.Env):
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = np.array([[-2,0],[-3,-1]])
        self.states = np.array([[1,2],[3,4]])

        self.action_space = Tuple([
            Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        ])
        self.observation_space = Tuple([
            OneHot(self.NUM_STATES) for _ in range(self.NUM_AGENTS)
        ])
        self.step_count = None
    
    def available_actions(self, batch_size):
        return [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

    def reset(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self.step_count = 0
        init_state = np.zeros(batch_size)
        observation = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions(batch_size)]
        return observation, info

    def step(self, action, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        ac0, ac1 = action
        self.step_count += 1
        r0 = self.payout_mat[ac0, ac1]
        r1 = self.payout_mat[ac1, ac0]
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observation = [s0, s1]
        reward = [r0, r1]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions(batch_size)]
        return observation, reward, done, info
