import numpy as np

from sharing import SharingAgent, SharingGame

def initialize_mults(n, method):
    """
    """
    assert (method in ['uniform', 'truncated_normal'])
    if method == 'uniform':
        stance_mult = np.random.uniform(-1, 1, n)
        diff_mult = np.random.uniform(-1, 1, n)
    elif method == 'truncated_normal':
        mean, sd = 0, 0.5
        stance_mult = np.random.normal(mean, sd, n)
        diff_mult = np.random.normal(mean, sd, n)

    return stance_mult, diff_mult

def update_mults(stance_mult, diff_mult, utilities, mutation_rate):
    """
    """
    n = stance_mult.shape[0]
    avg_utilities = np.mean(utilities, axis=0)

    # scale to make all utilities positive, then normalize
    avg_scaled_utilities = avg_utilities - min(avg_utilities)
    normalized_utilities = avg_scaled_utilities / np.sum(avg_scaled_utilities)

    # choose agents to reproduce
    reproducing = np.random.choice(range(n), n, replace=True, p=normalized_utilities)
    stance_mult_new = stance_mult[reproducing]
    diff_mult_new = diff_mult[reproducing]

    # mutate multiples
    stance_mult_new = stance_mult_new + np.random.normal(0, mutation_rate, n)
    diff_mult_new = diff_mult_new + np.random.normal(0, mutation_rate, n)

    return stance_mult_new, diff_mult_new

def run_evolving_experiment(init_stances, alphas, betas, gammas, strategies, T, E_mat, n_rounds, initialization='uniform', mutation_rate=0.05):
    """
    """
    n = init_stances.shape[0]
    stance_mult, diff_mult = initialize_mults(n, initialization)

    for r in range(n_rounds):
        # run the game with current multiples
        agents = SharingAgent(init_stances, alphas, betas, gammas, 'linear', T, stance_mult=stance_mult, diff_mult=diff_mult)
        game = SharingGame(agents, E_mat, T, calc_utilities=True, keep_actions=False)
        _, _, utilities = game.run()
        
        # birth new agents, updating multiples based on utilities in this round
        stance_mult, diff_mult = update_mults(stance_mult, diff_mult, utilities, mutation_rate)

    return utilities, stance_mult, diff_mult

