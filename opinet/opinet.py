"""
This is the core of the opinet infrastructure. It implements the general agent
and game classes from which specific agents and games inherit.
"""

import numpy as np

import warnings

class Agent(object):
    """
    Describes a general agent.
    """
    def __init__(self, init_stances, alphas, betas, gammas, T):
        # number of agents
        self.n = init_stances.shape[0]

        # alpha, beta, gamma or each agent
        self.alphas = np.array(alphas)
        self.betas = np.array(betas)
        vectorized = np.vectorize(lambda f, x: f(x))        
        self.apply_gammas = lambda R: vectorized(gammas, R)

        # stances[t,i] is stance of agent i at time t
        self.stances = np.empty((T, self.n))
        self.stances[1:] = np.NAN
        self.stances[:1] = init_stances

        # cache matrix of stance differences to avoid recomputing
        self.stance_diffs_t = -1
        self.stance_diffs = np.empty((self.n, self.n))

        # check values       
        assert(np.all(self.alphas <= 1))
        assert(np.all(self.alphas >= -1))

        assert(np.all(self.betas <= 1))
        assert(np.all(self.betas >= 0))

        assert(np.all(self.stances[:1] <= 1))
        assert(np.all(self.stances[:1] >= -1))

    def get_stances_diffs(self, t):
        """
        Calculates all pairwise differences in stances at time t
        """
        if self.stance_diffs_t == t:
            # already calculated in time t
            diffs = self.stance_diffs
        else:
            # need to calculate for first time
            stances = self.stances[t]
            diffs = np.empty((self.n, self.n))
            stance_matrix = np.tile(stances, (self.n, 1))
            diffs = np.swapaxes(stance_matrix, 0, 1) - stance_matrix

            # update cache
            self.stance_diffs_t = t
            self.stance_diffs = diffs

        return diffs

class Game(object):
    """
    Describes a general game.
    """
    def __init__(self, agents, T, calc_utilities=True):
        # agents
        self.agents = agents

        # size of graph
        self.n = self.agents.n

        # calculate utilities? - slower
        self.calc_utilities = calc_utilities

        if self.calc_utilities:
            # utilities[t,i] = utility of agent i in time t
            self.utilities = np.empty((T, self.n))
            self.utilities[:1] = np.NAN
        else:
            self.utilities = None

        # current round and total rounds
        self.t = 0
        self.T = T

    def update_stances(self, G):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_info_stances = np.nanmean(self.agents.stances[self.t-1] * G, 
                                          axis=1)
        stances = self.agents.betas * avg_info_stances + (1 - 
            self.agents.betas) * self.agents.stances[self.t-1]
        stances[np.isnan(stances)] = self.agents.stances[self.t-1][np.isnan(
            stances)]

        self.agents.stances[self.t] = stances

    def update_utilities(self, G):
        if self.calc_utilities:
            info_vals = self.agents.apply_gammas(np.nansum(G))
            diffs = abs(self.agents.get_stances_diffs(self.t))
            diffs[G != 1] = np.NAN

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff_vals = self.agents.alphas * np.nanmean(diffs, axis=1)

            utilities = info_vals + diff_vals
            utilities[np.isnan(utilities)] = info_vals[np.isnan(utilities)]

            self.utilities[self.t] = utilities

