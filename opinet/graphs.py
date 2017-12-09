"""
Utility module to create adjacency matrix representation of frequently used
graph topologies. 
"""

import numpy as np
import networkx as nx

def complete_graph(n):
    mat = np.ones((n, n), dtype=float)
    np.fill_diagonal(mat, np.NAN)

    return mat

def gnp_graph(n, avg_degree, directed=False):
    p = avg_degree / float(n-1)
    G = nx.adjacency_matrix(nx.fast_gnp_random_graph(n, p, directed=directed))
    mat = np.array(G.todense(), dtype=float)

    return mat

def barabasi_albert_graph(n, avg_degree, directed=False):
    m = int(avg_degree / 2) + 1
    G = nx.adjacency_matrix(nx.barabasi_albert_graph(n, m))
    mat = np.array(G.todense(), dtype=float)
    
    return mat