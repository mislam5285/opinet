# docs
# undo repeated, cython, parallelize

import numpy as np
cimport numpy as np

import types
import warnings

# compile time data types
ctypedef np.int_t IDX_t
ctypedef np.float_t FLT_t

class SharingAgent(object):
    """
    Describes a agent in the Sharing Game
    """
    def __init__(object self, np.ndarray[FLT_t, ndim=1] init_stances_in, np.ndarray[FLT_t, ndim=1] alphas_in, np.ndarray[FLT_t, ndim=1] betas_in, list gammas_in, IDX_t T_in):
        # attribute type defs
        cdef IDX_t n 

        cdef np.ndarray[FLT_t, ndim=1] alphas
        cdef np.ndarray[FLT_t, ndim=1] betas

        cdef np.ndarray[FLT_t, ndim=2] stances
        cdef IDX_t stance_diffs_t 
        cdef np.ndarray[FLT_t, ndim=2] stance_diffs

        # number of agens
        self.n = init_stances_in.shape[0]

        # alpha, beta, gamma or each agent
        self.alphas = alphas_in
        self.betas = betas_in
        vectorized = np.vectorize(lambda f, x: f(x))        
        self.apply_gammas = lambda R: vectorized(gammas_in, R)

        # stances[t,i] is stance of agent i at time t
        self.stances = np.empty((T_in, self.n))
        self.stances[:1] = init_stances_in

        # cache matrix of stance differences to avoid recomputing
        self.stance_diffs_t = -1
        self.stance_diffs = np.empty((self.n, self.n))

    def get_stances_diffs(object self, IDX_t t):
        """
        Calculates all pairwise differences in stances at time t
        """
        # type defs
        cdef np.ndarray[FLT_t, ndim=2] diffs
        cdef np.ndarray[FLT_t, ndim=1] stances
        cdef np.ndarray[FLT_t, ndim=2] stance_matrix

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

    def get_approx_opt_actions(object self, np.ndarray[FLT_t, ndim=2] G, IDX_t t):
        """
        Linearizes impact of adding/removing agent, returning approx optimal action for each agent
        """ 
        # type defs
        cdef np.ndarray[FLT_t, ndim=1] R
        cdef np.ndarray[FLT_t, ndim=1] gamma_null
        cdef np.ndarray[FLT_t, ndim=1] marginal_add
        cdef np.ndarray[FLT_t, ndim=1] marginal_remove
        cdef np.ndarray[FLT_t, ndim=2] diffs
        cdef np.ndarray[FLT_t, ndim=2] alpha_diffs
        cdef np.ndarray[FLT_t, ndim=2] diffs_remove
        cdef np.ndarray[FLT_t, ndim=2] diffs_add
        cdef np.ndarray[IDX_t, ndim=2] CS
        cdef np.ndarray[FLT_t, ndim=2] DS
        cdef np.ndarray[IDX_t, ndim=1] action_types
        cdef np.ndarray[IDX_t, ndim=1] actions

        # information differences
        R = np.nansum(G, axis=1)
        gamma_null = self.apply_gammas(R)
        marginal_add = self.apply_gammas(R + 1) - gamma_null
        marginal_remove = self.apply_gammas(R - 1) - gamma_null

        diffs = abs(self.get_stances_diffs(t))
        
        alpha_diffs = self.alphas * diffs
        
        diffs_remove = -1 * alpha_diffs
        diffs_remove[G != 1] = -np.inf

        diffs_add = np.copy(alpha_diffs)
        np.fill_diagonal(diffs_add, -np.inf)
        diffs_add[G == 1] = -np.inf

        CS = np.empty((3, self.n), dtype=int)
        CS[0] = -1
        CS[1] = np.argmax(diffs_add, axis=1)
        CS[2] = np.argmax(diffs_remove, axis=1)

        print "CS before\n", CS

        CS[1][np.isneginf(np.max(diffs_add, axis=1))] = -1
        CS[2][np.isneginf(np.max(diffs_remove, axis=1))] = -1

        print "CS after\n", CS

        DS = np.empty((3, self.n))
        DS[0] = 0
        DS[1] = marginal_add + self.alphas * abs(self.stances[t] - self.stances[t][CS[1]])  / (R + 1)
        DS[2] = marginal_remove - self.alphas * abs(self.stances[t] - self.stances[t][CS[2]]) / np.maximum(R - 1, 1)
        
        print "DS before\n", DS

        DS[1][CS[1] == -1] = np.NAN
        DS[2][CS[2] == -1] = np.NAN

        print "DS after\n", DS

        action_types = np.nanargmax(DS, axis=0)
        actions = CS[action_types, range(self.n)]

        return actions

class SharingGame(object):
    """
    Describes an instance of a Sharing Game.
    """
    def __init__(object self, object agents_in, np.ndarray[FLT_t, ndim=2] init_E_mat_in, IDX_t T_in, IDX_t calc_utilities_in=1):
        """
        Args:
            agents (list): list of DataFrames of action probabilities for each agent
            default_mutation_rate (float): default mutation rate

            keep history of stances, graph structure, utilities ()
        """
        # attribute type defs
        cdef np.ndarray[FLT_t, ndim=1] agents
        cdef np.ndarray[FLT_t, ndim=2] utilities
        cdef np.ndarray[FLT_t, ndim=3] G
        cdef IDX_t n
        cdef IDX_t t
        cdef IDX_t T
        cdef IDX_t calc_utilities

        # agents
        self.agents = agents_in

        # size of graph
        self.n = self.agents.n

        # must be one agent for each node
        assert(init_E_mat_in.shape[0] == self.n)

        # utilities[t,i] = utility of agent i in time t
        self.utilities = np.empty((T_in, self.n))
        self.utilities[:] = np.NAN

        # store graph as adjacency matrix; 1 => edge, NAN => no edge
        self.G = np.zeros((T_in, self.n, self.n))
        self.G[0] = np.array(init_E_mat_in)
        self.G[0][self.G[0] == 0] = np.NAN
        np.fill_diagonal(self.G[0], np.NAN)

        # current round and total rounds
        self.t = 0
        self.T = T_in

        # calculate utilities - slower
        self.calc_utilities = calc_utilities_in

    def update_stances(object self):
        """
        Args:
            agents (list): list of DataFrames of action probabilities for each agent
            default_mutation_rate (float): default mutation rate
        """
        # type defs
        cdef np.ndarray[FLT_t, ndim=1] avg_info_stances
        cdef np.ndarray[FLT_t, ndim=1] stances

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_info_stances = np.nanmean(self.agents.stances[self.t-1] * self.G[self.t-1], axis=1)
        stances = self.agents.betas * avg_info_stances + (1 - self.agents.betas) * self.agents.stances[self.t-1]
        stances[np.isnan(stances)] = self.agents.stances[self.t-1][np.isnan(stances)]

        self.agents.stances[self.t] = stances

    def update_structure(object self):
        """
        Args:
            agents (list): list of DataFrames of action probabilities for each agent
            default_mutation_rate (float): default mutation rate
        """
        # type defs
        cdef np.ndarray[FLT_t, ndim=2] G
        cdef np.ndarray[IDX_t, ndim=1] updates
        cdef IDX_t update
 
        G = np.copy(self.G[self.t-1])
        updates = self.agents.get_approx_opt_actions(G, self.t)
        for i in range(self.n):
            update = updates[i] 
            if update != -1:
                G[i][update] = 1 if np.isnan(G[i][update]) else np.NAN

        self.G[self.t] = G

    def update_utilities(object self):
        """
        Args:
            agents (list): list of DataFrames of action probabilities for each agent
            default_mutation_rate (float): default mutation rate
        """
        # type defs
        cdef np.ndarray[FLT_t, ndim=2] G
        cdef np.ndarray[FLT_t, ndim=1] info_vals
        cdef np.ndarray[FLT_t, ndim=2] diffs
        cdef np.ndarray[FLT_t, ndim=1] diff_vals
        cdef np.ndarray[FLT_t, ndim=1] utilities


        if self.calc_utilities:
            G = self.G[self.t]

            info_vals = self.agents.apply_gammas(np.nansum(G))
            diffs = abs(self.agents.get_stances_diffs(self.t))
            diffs[G != 1] = np.NAN

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff_vals = self.agents.alphas * np.nanmean(diffs, axis=1)

            utilities = info_vals + diff_vals
            utilities[np.isnan(utilities)] = info_vals[np.isnan(utilities)]

            self.utilities[self.t] = utilities

    def run(object self):
        """
        Args:
            new_agents (DataFrame or list): action probabilities for single or multiple new agents
        """
        while self.t < self.T - 1:
            print "time:", self.t
            self.t += 1
            self.update_stances()
            self.update_structure()
            self.update_utilities()

        print "G:\n", self.G
        print "stances:\n", self.agents.stances
        print "utilities:\n", self.utilities


