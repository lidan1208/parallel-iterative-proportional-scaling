"""
Efficiency experiment (rectangular grid): Test the performance of serial/parallel CovIPS on rectangular grid graphs using i.i.d. standard normal data.
Data generation: generate_n01 (covariance = identity matrix)
"""

import numpy as np
import pandas as pd
import time
import os
import gc
import networkx as nx
from common import GGM_Fitter, fast_cov, generate_n01

def generate_rectangular_grid(rows, cols):
    """
    Generate edge matrix for a rectangular grid graph.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.

    Returns
    -------
    emat : ndarray, shape (2, n_edges)
        Edge matrix.
    n_nodes : int
        Total number of nodes.
    n_edges : int
        Number of edges.
    density : float
        Graph density (edges / maximum possible edges).
    """
    G = nx.grid_2d_graph(rows, cols)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    edges = [[mapping[u], mapping[v]] for u, v in G.edges()]
    emat = np.array(edges).T
    n_nodes = rows * cols
    n_edges = len(edges)
    density = n_edges / (n_nodes * (n_nodes - 1) // 2)
    return emat, n_nodes, n_edges, density

def rectangular_grid_experiment(n_repeats, n_obs, grid_sizes=None, debug=False, seed=None):
    """
    Run serial/parallel CovIPS on rectangular grids and collect performance metrics.

    Parameters
    ----------
    n_repeats : int
        Number of repetitions for each grid size.
    n_obs : int
        Number of samples (i.i.d. standard normal).
    grid_sizes : list of tuple, optional
        List of (rows, cols). Default: [(20,25), (40,25), (40,50)].
    debug : bool
        Passed to GGM_Fitter.
    seed : int or None
        Random seed for data generation (passed to generate_n01).

    Returns
    -------
    df : pandas.DataFrame
        Summary results with columns: method, grid_size, n_nodes, avg_time, std_time, avg_iter, n_repeats, avg_speedup.
    """
    if grid_sizes is None:
        grid_sizes = [(20, 25), (40, 25), (40, 50)]

    results = []
    for rows, cols in grid_sizes:
        emat, n_nodes, n_edges, density = generate_rectangular_grid(rows, cols)
        sim_data = generate_n01(n_obs, n_nodes, seed=seed)
        S_sim = fast_cov(sim_data)

        serial_times, parallel_times = [], []
        serial_iters, parallel_iters = [], []
        for rep in range(n_repeats):
            K_init = np.diag(1 / np.diag(S_sim))

            # Serial
            fitter_s = GGM_Fitter(parallel=False, verbose=False, debug=debug, max_memory_mb=4096)
            t0 = time.time()
            res_s = fitter_s.fit(S_sim, emat, n_obs, K_init=K_init.copy())
            serial_times.append(time.time() - t0)
            serial_iters.append(res_s['iter'])

            # Parallel
            fitter_p = GGM_Fitter(parallel=True, n_jobs=min(os.cpu_count(), 4), verbose=False,
                                  debug=debug, parallel_eps_factor=1, max_memory_mb=4096)
            t0 = time.time()
            res_p = fitter_p.fit(S_sim, emat, n_obs, K_init=K_init.copy())
            parallel_times.append(time.time() - t0)
            parallel_iters.append(res_p['iter'])

            del fitter_s, fitter_p
            gc.collect()

        results.append({
            "method": "serial",
            "grid_size": f"{rows}x{cols}",
            "n_nodes": n_nodes,
            "avg_time": np.mean(serial_times),
            "std_time": np.std(serial_times),
            "avg_iter": np.mean(serial_iters),
            "n_repeats": n_repeats
        })
        results.append({
            "method": "parallel",
            "grid_size": f"{rows}x{cols}",
            "n_nodes": n_nodes,
            "avg_time": np.mean(parallel_times),
            "std_time": np.std(parallel_times),
            "avg_iter": np.mean(parallel_iters),
            "n_repeats": n_repeats,
            "avg_speedup": np.mean(serial_times) / np.mean(parallel_times)
        })

    return pd.DataFrame(results)
