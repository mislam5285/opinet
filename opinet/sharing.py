"""
Implements the agent and game classes for the Sharing Game. Each class inherit 
from the general agent and game class, respectively.
"""

import numpy as np

from opinet import Agent, Game

class SharingAgent(Agent):
    """
    Describes a agent in the Sharing Game
    """
    def __init__(self, init_stances, alphas, betas, gammas, strategies, T, 
                 stance_mult=None, diff_mult=None):
        Agent.__init__(self, init_stances, alphas, betas, gammas, T)

        # record strateiges
        strategies_map = {
            'truthful': self.get_truthful_actions,
            'extreme': self.get_extreme_actions,
            'linear': self.get_linear_actions
        }
        self.get_actions = strategies_map[strategies]

        # for linear strategies, uniformly initialize sharing traits
        self.stance_mult = stance_mult
        self.diff_mult = diff_mult

    def get_truthful_actions(self, G, t, prev_actions=None):
        """
        Agents share current stances.
        """ 
        actions = self.stances[t]

        return actions

    def get_extreme_actions(self, G, t, prev_actions=None):
        """
        Agents share -1 if stance < 0 and +1 if stance > 0 and 0 is stance = 0
        """
        stances = self.stances[t]
        actions = np.empty((self.n))
        actions[stances < 0] = -1
        actions[stances > 0] = 1
        actions[stances == 0] = 0

        return actions

    def get_linear_actions(self, G, t, prev_actions):
        """
        s = diff_mult * (average stance difference) + stance_mult * current
        stance
        """
        diffs = abs(self.get_stances_diffs(t))
        diffs[G != 1] = np.NAN
        avg_diffs = np.nanmean(diffs, axis=1)

        actions = (self.stances[t] * self.stance_mult + avg_diffs *
                   self.diff_mult)
        actions = np.clip(actions, -1, 1)

        return actions

class SharingGame(Game):
    """
    Describes an instance of a Sharing Game.
    """
    def __init__(self, agents, E_mat, T, calc_utilities=True, 
                 keep_actions=True):
        Game.__init__(self, agents, T, calc_utilities)

        # must be one agent for each node
        assert(E_mat.shape[0] == self.n)

        # store graph as adjacency matrix; 1 => edge, NAN => no edge
        self.G = np.copy(E_mat)
        self.G[self.G == 0] = np.NAN
        np.fill_diagonal(self.G, np.NAN) 

        # record actions
        self.keep_actions = keep_actions
        if self.keep_actions:
            # utilities[t,i] = utility of agent i in time t
            self.actions = np.empty((T, self.n))
        else:
            self.actions = None
        
        self.curr_actions = np.empty(self.n)

    def update_actions(self, G):
        actions = self.agents.get_actions(G, self.t, self.actions)

        if self.keep_actions:
            self.actions[self.t] = actions

        self.curr_actions = actions

    def run(self):
        while self.t < self.T:

            if self.t != 0:
                self.update_stances(self.G)
            self.update_actions(self.G)
            self.update_utilities(self.G)
            self.t += 1

        return self.agents.stances, self.actions, self.utilities
            

