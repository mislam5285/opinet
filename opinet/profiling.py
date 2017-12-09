"""
For investigation and performance testing.
"""

import cProfile
import numpy as np
import pstats

from following import FollowingAgent, FollowingGame
from sharing import SharingAgent, SharingGame

# n, T = 4, 10
# alphas, betas, gammas = [-1.0] * n, [0.5] * n, [lambda R: 0.5 * R] * n
# init_E_mat = np.array(np.random.binomial(n=1, p=0.3, size=(n, n)), dtype=float)
# init_stances = np.random.uniform(low=-1, high=1, size=n)

# n, T = 1000, 3
# alphas, betas, gammas = np.array([-1] * n, dtype=float), np.array([0] * n, dtype=float), [lambda R: 0.3 * R] * n
# # init_E_mat = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
# init_E_mat = np.zeros((n, n))
# init_stances = np.array([0.5] * (n / 2) +  [-0.5] * (n / 2))

# agents = FollowingAgent(init_stances, alphas, betas, gammas, 'approx_opt', T)
# game = FollowingGame(agents, init_E_mat, T, calc_utilities=True, keep_G=False)

# cProfile.run("game.run()", "stats")

# p = pstats.Stats("stats")
# p.strip_dirs().sort_stats("cumulative").print_stats()

# p.print_stats()

# agents_c = following_c.FollowingAgent(init_stances, alphas, betas, gammas, T)
# game_c = following_c.FollowingGame(agents_c, init_E_mat, T, calc_utilities_in=True)

# cProfile.run("game_c.run()", "stats")

# p = pstats.Stats("stats")
# p.strip_dirs().sort_stats("cumulative").print_stats()

# p.print_stats()

####### SHARING ########

# n, T = 4, 5
# alphas, betas, gammas = [-0.8] * n, [0.5] * n, [lambda R: 0.0 * R] * n
# E_mat = np.random.binomial(n=1, p=0.7, size=(n, n)).astype(float)
# init_stances = np.random.uniform(low=-1, high=1, size=n)
# print "init_stance\n", init_stances

# agents = SharingAgent(init_stances, alphas, betas, gammas, 'truthful', T)
# game = SharingGame(agents, E_mat, T, calc_utilities=True, keep_actions=True)

# cProfile.run("game.run()", "stats")

# p = pstats.Stats("stats")
# p.strip_dirs().sort_stats("cumulative").print_stats()

# p.print_stats()

