"""
Efficiency experiment (random tree): Test the performance of serial/parallel CovIPS on random trees with extra edges using i.i.d. standard normal data
Data generation: generate_n01
"""


def generate_random_tree_with_extra_edges(n_nodes, extra_edge_prob=0.0, seed=None):
   """Generate random tree based on Prüfer sequence and add extra edges"""
    if seed is not None:
        np.random.seed(seed)

    if n_nodes <= 1:
        return np.array([], dtype=int).reshape(2,0), 0, 0.0
    prufer = np.random.randint(1, n_nodes+1, size=n_nodes-2)
    degree = np.ones(n_nodes, dtype=int)
    for v in prufer:
        degree[v-1] += 1
    edges = []
    for v in prufer:
        for u in range(1, n_nodes+1):
            if degree[u-1] == 1:
                edges.append((u, v))
                degree[u-1] -= 1
                degree[v-1] -= 1
                break
    remaining = [i+1 for i in range(n_nodes) if degree[i]==1]
    edges.append((remaining[0], remaining[1]))
    emat = np.array(edges).T - 1   # 0-based

  # Add extra edges
    if extra_edge_prob > 0:
        max_possible = n_nodes*(n_nodes-1)//2
        target = int(extra_edge_prob * max_possible)
        if target > 0:
            all_pairs = [(i,j) for i in range(n_nodes) for j in range(i+1,n_nodes)]
            chosen = np.random.choice(len(all_pairs), min(target, len(all_pairs)), replace=False)
            extra = [all_pairs[idx] for idx in chosen]
            edge_set = set()
            all_edges = []
            for i in range(emat.shape[1]):
                u,v = emat[0,i], emat[1,i]
                tup = (u,v) if u<v else (v,u)
                if tup not in edge_set:
                    all_edges.append((u,v))
                    edge_set.add(tup)
            for u,v in extra:
                tup = (u,v) if u<v else (v,u)
                if tup not in edge_set:
                    all_edges.append((u,v))
                    edge_set.add(tup)
            emat = np.array(all_edges).T
    n_edges = emat.shape[1]
    density = n_edges / (n_nodes*(n_nodes-1)//2)
    return emat, n_edges, density

def random_tree_experiment(n_repeats, n_obs, n_vars_list=None, extra_probs=None, debug=False，seed=None):
    if n_vars_list is None:
        n_vars_list = [500, 1000, 2000, 4000, 6000]
    if extra_probs is None:
        extra_probs = [0.001, 0.005, 0.010]

    results = []
    for prob in extra_probs:
        for n_var in n_vars_list:
            emat, n_edges, density = generate_random_tree_with_extra_edges(n_var, prob)
            sim_data = generate_n01(n_obs, n_var,seed=seed)
            S_sim = fast_cov(sim_data)

            serial_times, parallel_times = [], []
            serial_iters, parallel_iters = [], []
            for rep in range(n_repeats):
                K_init = np.diag(1/np.diag(S_sim))
                # serial
                fitter_s = GGM_Fitter(parallel=False, verbose=False, debug=debug, max_memory_mb=4096)
                t0 = time.time()
                res_s = fitter_s.fit(S_sim, emat, n_obs, K_init=K_init.copy())
                serial_times.append(time.time()-t0)
                serial_iters.append(res_s['iter'])
                # parallel
                fitter_p = GGM_Fitter(parallel=True, n_jobs=min(os.cpu_count(),4), verbose=False,
                                      debug=debug, parallel_eps_factor=1, max_memory_mb=4096)
                t0 = time.time()
                res_p = fitter_p.fit(S_sim, emat, n_obs, K_init=K_init.copy())
                parallel_times.append(time.time()-t0)
                parallel_iters.append(res_p['iter'])
                del fitter_s, fitter_p
                gc.collect()

            results.append({"method":"serial", "extra_prob":prob, "n_var":n_var,
                            "avg_time":np.mean(serial_times), "avg_iter":np.mean(serial_iters),
                            "n_repeats":n_repeats})
            results.append({"method":"parallel","extra_prob":prob,"n_var":n_var,
                            "avg_time":np.mean(parallel_times), "avg_iter":np.mean(parallel_iters),
                            "avg_speedup":np.mean(serial_times)/np.mean(parallel_times),
                            "n_repeats":n_repeats})

    df = pd.DataFrame(results)
    return df
