import numpy as np

from opinet import Agent, Game

class FollowingAgent(Agent):
    """
    Describes a agent in the Following Game
    """
    def __init__(self, init_stances, alphas, betas, gammas, strategies, T):
        Agent.__init__(self, init_stances, alphas, betas, gammas, T)

        # record strategies
        strategies_map = {
            'approx_opt': self.get_approx_opt_actions
        }
        self.get_actions = strategies_map[strategies]

    def get_approx_opt_actions(self, G, t):
        """
        Linearizes impact of adding/removing agent, returning approx optimal action for each agent
        """ 
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

        CS[1][np.isneginf(np.max(diffs_add, axis=1))] = -1
        CS[2][np.isneginf(np.max(diffs_remove, axis=1))] = -1

        DS = np.empty((3, self.n))
        DS[0] = 0
        DS[1] = marginal_add + self.alphas * abs(self.stances[t] - self.stances[t][CS[1]])  / (R + 1)
        DS[2] = marginal_remove - self.alphas * abs(self.stances[t] - self.stances[t][CS[2]]) / np.maximum(R - 1, 1)
        DS[1][CS[1] == -1] = np.NAN
        DS[2][CS[2] == -1] = np.NAN

        action_types = np.nanargmax(DS, axis=0)
        actions = CS[action_types, range(self.n)]

        return actions

class FollowingGame(Game):
    """
    Describes an instance of a Following Game.
    """
    def __init__(self, agents, init_E_mat, T, calc_utilities=True, keep_G=True):
        """
        Args:
            agents (list): list of DataFrames of action probabilities for each agent
            default_mutation_rate (float): default mutation rate

            keep history of stances, graph structure, utilities ()
        """
        Game.__init__(self, agents, T, calc_utilities)

        # must be one agent for each node
        assert(init_E_mat.shape[0] == self.n)

        # store graph as adjacency matrix; 1 => edge, NAN => no edge
        G_init = np.copy(init_E_mat)
        G_init[G_init == 0] = np.NAN
        np.fill_diagonal(G_init, np.NAN) 
        self.G_curr = G_init

        self.keep_G = keep_G
        if self.keep_G:
            self.G = np.zeros((T, self.n, self.n))
            self.G[0] = self.G_curr

    def update_structure(self, G):
        """
        Args:
            agents (list): list of DataFrames of action probabilities for each agent
            default_mutation_rate (float): default mutation rate
        """
        G_copy = np.copy(G) if self.keep_G else G

        updates = self.agents.get_actions(G_copy, self.t)
        for i in range(self.n):
            update = updates[i] 
            if update != -1:
                G_copy[i][update] = 1 if np.isnan(G_copy[i][update]) else np.NAN

        if self.keep_G:
            self.G[self.t] = G_copy

        self.G_curr = G_copy

    def run(self):
        """
        Args:
            new_agents (DataFrame or list): action probabilities for single or multiple new agents
        """
        while self.t < self.T:
            # print "t:\n", self.t
            # print "stances:\n", self.agents.stances
            # print "G:\n", self.G_curr
            # print "utilities:\n", self.utilities
            if self.t != 0:
                self.update_stances(self.G_curr)
            self.update_structure(self.G_curr)
            self.update_utilities(self.G_curr)
            self.t += 1

        # print "t:\n", self.t
        # print "stances:\n", self.agents.stances
        # print "G:\n", self.G_curr
        # print "utilities:\n", self.utilities

        G_out = self.G if self.keep_G else self.G_curr

        return self.agents.stances, G_out, self.utilities


