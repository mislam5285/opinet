"""
Graph manipulation utilities and experiment helper functions.
"""

import numpy as np

from following import FollowingAgent, FollowingGame
from sharing import SharingAgent, SharingGame

def mat_to_edge_list(E_mat):
    edges = []
    n = E_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if E_mat[i,j] == 1:
                edges.append((i,j))
    
    return edges

def convergence_time(stances, convergence_eps):
    stance_deltas = np.absolute(stances[1:] - stances[:-1])
    avg_stance_deltas = np.mean(stance_deltas, axis=1)
    times_below_threshold = np.where(avg_stance_deltas <= convergence_eps)[0]
    if times_below_threshold.size != 0:
        convergence_time = np.min(times_below_threshold) + 1
    else:
        convergence_time = np.NAN

    return convergence_time

def run_sharing_experiment(init_stances, alphas, betas, gammas, strategies, T, 
                           E_mat, calc_utilities=True, keep_actions=True):
    agents = SharingAgent(init_stances, alphas, betas, gammas, strategies, T)
    game = SharingGame(agents, E_mat, T, calc_utilities, keep_actions)
    stances, actions, utilities = game.run()

    return stances, actions, utilities

def run_following_experiment(init_stances, alphas, betas, gammas, strategies, T,
                             E_mat, calc_utilities=True, keep_G=True):
    agents = FollowingAgent(init_stances, alphas, betas, gammas, strategies, T)
    game = FollowingGame(agents, E_mat, T, calc_utilities, keep_G)
    stances, G, utilities = game.run()

    return stances, G, utilities
