import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import lstsq, solve
from scipy.optimize import minimize
import argparse


# ============== Kernels ==============

def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def fourth_order_kernel(u):
    phi = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    return 0.5 * (3 - u**2) * phi


def get_kernel(order=2):
    if order == 4:
        return fourth_order_kernel, 4
    else:
        return gaussian_kernel, 2


def optimal_bandwidth_init(n, d, q=2):
    return n ** (-1.0 / (2 * q + d))


# ============== d=0 LOOCV (intercept only) ==============

def loocv_d0(y):
    """
    LOOCV for d=0 model: Y independent of X.
    CV = (1/n) Σᵢ (Yᵢ - Ȳ₋ᵢ)² where Ȳ₋ᵢ is leave-one-out mean.
    
    Simplifies to: n/(n-1)² * Σᵢ (Yᵢ - Ȳ)²
    """
    n = len(y)
    y_bar = np.mean(y)
    residuals = y - y_bar
    cv = np.mean((n / (n - 1) * residuals) ** 2)
    return cv


# ============== B-spline basis ==============

def compute_knots(x, n_knots, degree=3):
    knot_positions = np.linspace(0, 100, n_knots + 2)[1:-1]
    internal_knots = np.percentile(x, knot_positions)
    x_min, x_max = x.min(), x.max()
    return np.concatenate([
        np.repeat(x_min, degree + 1),
        internal_knots,
        np.repeat(x_max, degree + 1)
    ])


def bspline_basis_1d(x, knots, degree=3):
    return BSpline.design_matrix(x, knots, degree).toarray()


def tensor_product_basis(X, knots_list, degree=3):
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    n, p = X.shape
    
    basis = bspline_basis_1d(X[:, 0], knots_list[0], degree)
    for j in range(1, p):
        basis_j = bspline_basis_1d(X[:, j], knots_list[j], degree)
        basis = np.einsum('ij,ik->ijk', basis, basis_j).reshape(n, -1)
    
    return basis


def compute_knots_from_data(X, n_knots, degree=3):
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    p = X.shape[1]
    
    if np.isscalar(n_knots):
        n_knots = [n_knots] * p
    
    return [compute_knots(X[:, j], n_knots[j], degree) for j in range(p)]


# ============== LOOCV methods ==============

def bspline_loocv_hat(X, y, n_knots, degree=3):
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    
    knots_list = compute_knots_from_data(X, n_knots, degree)
    B = tensor_product_basis(X, knots_list, degree)
    
    BtB = B.T @ B + 1e-10 * np.eye(B.shape[1])
    Bty = B.T @ y
    
    coeffs = solve(BtB, Bty, assume_a='pos')
    y_hat = B @ coeffs
    
    # Compute diagonal of hat matrix efficiently
    X_solve = solve(BtB, B.T, assume_a='pos')  # Shape (k, n)
    H_diag = np.einsum('ij,ji->i', B, X_solve)  # Diagonal of B @ X
    
    residuals = y - y_hat
    loocv = np.mean((residuals / (1 - H_diag)) ** 2)
    
    return loocv


def nw_loocv(X, y, bandwidth, kernel_order=4):
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    n = len(y)
    
    kernel_fn, _ = get_kernel(kernel_order)
    p = X.shape[1]
    h = np.full(p, bandwidth) if np.isscalar(bandwidth) else np.asarray(bandwidth)
    
    squared_errors = []
    for i in range(n):
        u = (X[i, :] - X) / h
        weights = np.prod(kernel_fn(u), axis=1)
        weights[i] = 0
        
        w_sum = weights.sum()
        if np.abs(w_sum) > 1e-10:
            y_pred = (weights @ y) / w_sum
        else:
            y_pred = np.mean(np.delete(y, i))
        
        squared_errors.append((y[i] - y_pred) ** 2)
    
    return np.mean(squared_errors)


# ============== Projection optimization ==============

def make_beta(C, d, p):
    C = C.reshape(p - d, d)
    return np.vstack([np.eye(d), C])


def projection_matrix(beta):
    return beta @ np.linalg.solve(beta.T @ beta, beta.T)


def subspace_distance(beta_true, beta_est):
    """
    Compute Frobenius norm between projection matrices.
    If beta_est is None (d=0), the estimated projection is the zero matrix.
    """
    P_true = projection_matrix(beta_true)
    
    if beta_est is None:
        # d_est = 0: projection onto trivial subspace is zero matrix
        return np.linalg.norm(P_true, 'fro')
    
    P_est = projection_matrix(beta_est)
    return np.linalg.norm(P_true - P_est, 'fro')


def project_cv_bspline_hat(C, X, y, d, n_knots, degree):
    n, p = X.shape
    beta = make_beta(C, d, p)
    X_proj = X @ beta
    return bspline_loocv_hat(X_proj, y, n_knots, degree)


def project_cv_nw(params, X, y, d, kernel_order):
    n, p = X.shape
    n_C = (p - d) * d
    
    C = params[:n_C]
    h = np.exp(params[n_C])
    
    beta = make_beta(C, d, p)
    X_proj = X @ beta
    
    return nw_loocv(X_proj, y, h, kernel_order)


def optimize_bspline(X, y, d, n_knots, degree, method, n_restarts, rng):
    """Optimize projection for a fixed number of knots."""
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    n, p = X.shape
    
    cv_fn = project_cv_bspline_hat
    
    # Track minimize stats
    minimize_time_total = 0.0
    minimize_count = 0
    minimize_iters_total = 0
    
    if d >= p:
        cv = bspline_loocv_hat(X, y, n_knots, degree)
        return np.eye(p), cv, minimize_time_total, minimize_count, minimize_iters_total
    
    n_C = (p - d) * d
    
    best_cv = np.inf
    best_beta = None
    
    for _ in range(n_restarts):
        C0 = rng.standard_normal(n_C) * 0.1
        
        t_min_start = time.perf_counter()
        res = minimize(
            cv_fn, C0, args=(X, y, d, n_knots, degree),
            method='CG', 
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        minimize_time_total += time.perf_counter() - t_min_start
        minimize_count += 1
        minimize_iters_total += res.nit
        
        if res.fun < best_cv:
            best_cv = res.fun
            best_beta = make_beta(res.x, d, p)
    
    return best_beta, best_cv, minimize_time_total, minimize_count, minimize_iters_total


def optimize_bspline_with_knots(X, y, d, degree, method, n_restarts, rng, model=None):
    """
    Optimize projection AND number of knots jointly via grid search over knots.
    """
    best_cv = np.inf
    best_beta = None
    best_n_knots = None
    
    # Track minimize stats across all knot candidates
    total_minimize_time = 0.0
    total_minimize_count = 0
    total_minimize_iters = 0

    n_knots = int(X.shape[0] ** (1 / (2*4 + d)))  # assuming 4 continuous derivatives
    knot_candidates = [n_knots]
    
    for n_knots in knot_candidates:
        knot_rng = np.random.default_rng(rng.integers(0, 2**31))
        beta, cv, min_time, min_count, min_iters = optimize_bspline(X, y, d, n_knots, degree, method, n_restarts, knot_rng)
        
        total_minimize_time += min_time
        total_minimize_count += min_count
        total_minimize_iters += min_iters
        
        if cv < best_cv:
            best_cv = cv
            best_beta = beta
            best_n_knots = n_knots
    
    return best_beta, best_cv, best_n_knots, total_minimize_time, total_minimize_count, total_minimize_iters


def optimize_nw(X, y, d, kernel_order, n_restarts, rng):
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    n, p = X.shape
    
    _, q = get_kernel(kernel_order)
    h_init = optimal_bandwidth_init(n, d, q)
    
    # Track minimize stats
    minimize_time_total = 0.0
    minimize_count = 0
    minimize_iters_total = 0
    
    if d >= p:
        def cv_h(log_h):
            return nw_loocv(X, y, np.exp(log_h), kernel_order)
        
        t_min_start = time.perf_counter()
        res = minimize(cv_h, np.log(h_init), method='CG',
                      options={'maxiter': 200, 'gtol': 1e-6})
        minimize_time_total += time.perf_counter() - t_min_start
        minimize_count += 1
        minimize_iters_total += res.nit
        
        h_opt = np.exp(res.x[0])
        cv = nw_loocv(X, y, h_opt, kernel_order)
        return np.eye(p), h_opt, cv, minimize_time_total, minimize_count, minimize_iters_total
    
    n_C = (p - d) * d
    
    best_cv = np.inf
    best_beta = None
    best_h = None
    
    for _ in range(n_restarts):
        C0 = rng.standard_normal(n_C) * 0.1
        log_h0 = np.log(h_init) + rng.standard_normal() * 0.3
        params0 = np.concatenate([C0, [log_h0]])
        
        t_min_start = time.perf_counter()
        res = minimize(
            project_cv_nw, params0, args=(X, y, d, kernel_order),
            method='CG',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        minimize_time_total += time.perf_counter() - t_min_start
        minimize_count += 1
        minimize_iters_total += res.nit
        
        if res.fun < best_cv:
            best_cv = res.fun
            best_beta = make_beta(res.x[:(p-d)*d], d, p)
            best_h = np.exp(res.x[-1])
    
    return best_beta, best_h, best_cv, minimize_time_total, minimize_count, minimize_iters_total


def search_dimension_single(X, y, max_d, degree, kernel_order, method, n_restarts, rng, model=None):
    """
    Search for best dimension starting from d=0.
    """
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    n, p = X.shape
    
    best_cv = np.inf
    best_d = None
    best_beta = None
    best_h = None
    best_n_knots = None
    cv_path = {}
    
    # Track minimize stats across all dimensions
    total_minimize_time = 0.0
    total_minimize_count = 0
    total_minimize_iters = 0
    
    # Start with d=0 (intercept-only model)
    cv_d0 = loocv_d0(y)
    cv_path[0] = {'cv': cv_d0, 'n_knots': None, 'h': None}
    best_cv = cv_d0
    best_d = 0
    best_beta = None
    best_h = None
    
    # Search d=1, 2, ..., max_d
    for d in range(1, min(max_d + 1, p)):
        if method == 'hat':
            beta, cv, n_knots, min_time, min_count, min_iters = optimize_bspline_with_knots(
                X, y, d, degree, method, n_restarts, rng, model=model
            )
            cv_path[d] = {'cv': cv, 'n_knots': n_knots, 'h': None}
            h = None
        else:  # nw
            beta, h, cv, min_time, min_count, min_iters = optimize_nw(X, y, d, kernel_order, n_restarts, rng)
            cv_path[d] = {'cv': cv, 'n_knots': None, 'h': h}
        
        total_minimize_time += min_time
        total_minimize_count += min_count
        total_minimize_iters += min_iters
        
        if cv < best_cv:
            best_cv = cv
            best_d = d
            best_beta = beta
            best_h = h
            if method == 'hat':
                best_n_knots = n_knots
        else:
            break
    
    return best_d, best_beta, best_h, best_cv, cv_path, total_minimize_time, total_minimize_count, total_minimize_iters


# ============== Data generation ==============
 

## True model structural dimensions
MODEL_TO_DIM = {
    'M1': 1, 
    'M2': 2, 
    'M3_func1': 3,
    'M3_func2': 3
}


def generate_X(n, p, distribution, rng):
    if distribution == 'uniform':
        X = rng.uniform(0, 10, (n, p))
    elif distribution == 'normal':
        X = rng.standard_normal((n, p))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return X


def generate_data(n, p, model, distribution, sigma, rng):
    """
    Generate data for a given model.
    
    Parameters
    ----------
    model : str
        One of 'M1', 'M2', 'M3_func1', 'M3_func2'
    """
    X = generate_X(n, p, distribution, rng)
    
    if model == 'M1':
        # y = z^3 + z^2 + z where z = β'X
        d_true = 1
        beta_true = np.zeros((p, 1))
        beta_true[0, 0] = 1.0
        beta_true[1, 0] = 1.0
        beta_true[2, 0] = 1.0
        beta_true[3, 0] = 1.0
        
        z = (X @ beta_true)[:, 0]
        y = z**3 + z**2 + z
        
    elif model == 'M2':
        # y = sin(x_1) * cos(2*x_2)
        d_true = 2
        beta_true = np.zeros((p, 2))
        beta_true[0, 0] = 1.0
        beta_true[1, 1] = 1.0
        
        z = X @ beta_true
        y = np.sin(z[:, 0]) * np.cos(2*z[:, 1])
        
    elif model == 'M3_func1':
        # y = 3*z_1^2 + 1.5*(cos(z_2) + sin(z_3))^2
        d_true = 3
        beta_true = np.zeros((p, 3))
        beta_true[0, 0] = 1.0
        beta_true[1, 1] = 1.0
        beta_true[2, 2] = 1.0
        
        z = X @ beta_true
        y = 3*z[:, 0]**2 + 1.5*(np.cos(z[:, 1]) + np.sin(z[:, 2]))**2
        
    elif model == 'M3_func2':
        # Y = z_1/(0.25 + (z_2 + 1.5)^2) + 2*z_3^3 + ε with ε ~ N(0, sigma)

        d_true = 3
        beta_true = np.zeros((p, 3))
        beta_true[0, 0] = 1.0
        beta_true[1, 1] = 1.0
        beta_true[2, 2] = 1.0
        
        z = X @ beta_true
        y = z[:, 0] / (0.25 + (z[:, 1] + 1.5)**2) + 2*z[:, 2]**3
        
    else:
        raise ValueError(f"model must be 'M1', 'M2', 'M3_func1', or 'M3_func2', got {model}")
    
    y += sigma * rng.standard_normal(n) # add noise
    
    return X, y, beta_true, d_true


# ============== Experiment ==============

def run_experiment_single(model, distribution, n_reps=100, p=10, degree=3, 
                          kernel_order=4, n_restarts=1, sigma=0.2, 
                          global_seed=42, output_file=None, method='all', sample_size=None):
    """
    Run experiment for a single (model, distribution) combination.
    
    Parameters
    ----------
    model : str
        One of 'M1', 'M2', 'M3_func1', 'M3_func2'
    distribution : str
        One of 'uniform', 'normal'
    output_file : str, optional
        CSV filename to save results
    
    Returns
    -------
    df : pd.DataFrame
        Results for this combination
    """
    
    # Sample sizes based on method
    if method == 'hat':
        sample_sizes_hat = [50, 100, 400, 1000, 2000, 5000]
        sample_sizes_nw = []
    elif method == 'nw':
        sample_sizes_hat = []
        sample_sizes_nw = [50, 100, 400, 1000, 2000, 5000]
    else:  # 'all' - run both methods
        sample_sizes_hat = [50, 100, 400, 1000, 2000, 5000]
        sample_sizes_nw = [50, 100, 400, 1000, 2000, 5000]
    
    all_sample_sizes = sorted(set(sample_sizes_hat + sample_sizes_nw))
    
    # Filter to single sample size if specified
    if sample_size is not None:
        all_sample_sizes = [n for n in all_sample_sizes if n == sample_size]
        sample_sizes_hat = [n for n in sample_sizes_hat if n == sample_size]
        sample_sizes_nw = [n for n in sample_sizes_nw if n == sample_size]
        if not all_sample_sizes:
            raise ValueError(f"Sample size {sample_size} not valid for method {method}")
    
    # Pre-generate seeds for all possible sample sizes
    all_possible_sample_sizes = [50, 100, 400, 1000, 2000, 5000]
    
    master_rng = np.random.default_rng(global_seed)
    seeds = {}
    for n in all_possible_sample_sizes:
        for rep in range(n_reps):
            seeds[(n, rep)] = master_rng.integers(0, 2**31)
    
    d_true = MODEL_TO_DIM[model]
    max_d = d_true + 1
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model} (d={d_true}), DISTRIBUTION: {distribution}")
    print('='*70)
    
    results = []
    total_start_time = time.perf_counter()
    
    for n_idx, n in enumerate(all_sample_sizes):
        # Determine methods
        methods_for_n = []
        if n in sample_sizes_hat:
            methods_for_n.append(('BS (hat)', 'hat'))
        if n in sample_sizes_nw:
            methods_for_n.append(('NW', 'nw'))
        
        # Initialize collectors
        method_results = {}
        for method_name, _ in methods_for_n:
            method_results[method_name] = {
                'times_all': [],
                'times_correct': [],
                'd_correct': 0,
                'subspace_errors_all': [],
                'subspace_errors': [],
                'minimize_times_all': [],
                'minimize_times_correct': [],
                'minimize_counts_all': [],
                'minimize_counts_correct': [],
                'minimize_iters_all': [],
                'minimize_iters_correct': []
            }
        
        print(f"\n  n={n} (sample size {n_idx+1}/{len(all_sample_sizes)}):")
        sample_start_time = time.perf_counter()
        
        for rep in range(n_reps):
            rep_seed = seeds[(n, rep)]
            
            # Generate data
            data_rng = np.random.default_rng(rep_seed)
            X, y, beta_true, _ = generate_data(n, p, model, distribution, sigma, data_rng)
            
            # Run each method
            for method_name, method_code in methods_for_n:
                opt_rng = np.random.default_rng(rep_seed + 1)
                
                t0 = time.perf_counter()
                d_est, beta_est, h_est, cv, cv_path, min_time, min_count, min_iters = search_dimension_single(
                    X, y, max_d=max_d, degree=degree,
                    kernel_order=kernel_order, method=method_code,
                    n_restarts=n_restarts, rng=opt_rng,
                    model=model
                )
                t = time.perf_counter() - t0
                
                res = method_results[method_name]
                err = subspace_distance(beta_true, beta_est)
                
                # Always record time, subspace error, and minimize stats
                res['times_all'].append(t)
                res['subspace_errors_all'].append(err)
                res['minimize_times_all'].append(min_time)
                res['minimize_counts_all'].append(min_count)
                res['minimize_iters_all'].append(min_iters)
                
                # Record separately when dimension is correct
                if d_est == d_true:
                    res['d_correct'] += 1
                    res['times_correct'].append(t)
                    res['subspace_errors'].append(err)
                    res['minimize_times_correct'].append(min_time)
                    res['minimize_counts_correct'].append(min_count)
                    res['minimize_iters_correct'].append(min_iters)
            
            # Progress logging
            if (rep + 1) % 10 == 0 or rep == n_reps - 1:
                elapsed = time.perf_counter() - sample_start_time
                avg_per_rep = elapsed / (rep + 1)
                remaining_reps = n_reps - (rep + 1)
                eta_seconds = avg_per_rep * remaining_reps
                eta_min = eta_seconds / 60
                
                total_elapsed = time.perf_counter() - total_start_time
                total_elapsed_min = total_elapsed / 60
                
                print(f"    Rep {rep+1}/{n_reps} | Elapsed: {elapsed:.1f}s | ETA for n={n}: {eta_min:.1f}min | Total elapsed: {total_elapsed_min:.1f}min", flush=True)
        
        # Print summaries
        for method_name in method_results:
            res = method_results[method_name]
            d_correct = res['d_correct']
            times_all = res['times_all']
            times_correct = res['times_correct']
            subspace_errors_all = res['subspace_errors_all']
            subspace_errors = res['subspace_errors']
            minimize_times_all = res['minimize_times_all']
            minimize_times_correct = res['minimize_times_correct']
            minimize_counts_all = res['minimize_counts_all']
            minimize_counts_correct = res['minimize_counts_correct']
            minimize_iters_all = res['minimize_iters_all']
            minimize_iters_correct = res['minimize_iters_correct']
            
            print(f"    {method_name}: Dim acc {d_correct}/{n_reps} ({100*d_correct/n_reps:.0f}%)", end="")
            
            # Print "when correct" metrics
            if times_correct:
                se_time = np.std(times_correct) / np.sqrt(len(times_correct))
                print(f", Time(corr) {np.mean(times_correct):.2f}±{se_time:.2f}s", end="")
            if subspace_errors:
                se_suberr = np.std(subspace_errors) / np.sqrt(len(subspace_errors))
                print(f", SubErr(corr) {np.mean(subspace_errors):.4f}±{se_suberr:.4f}", end="")
            if minimize_times_correct and minimize_counts_correct:
                avg_min_time_corr = [t/c if c > 0 else 0 for t, c in zip(minimize_times_correct, minimize_counts_correct)]
                se_min_time_corr = np.std(avg_min_time_corr) / np.sqrt(len(avg_min_time_corr))
                print(f", MinTime(corr) {np.mean(avg_min_time_corr):.4f}±{se_min_time_corr:.4f}s", end="")
            if minimize_iters_correct and minimize_counts_correct:
                avg_min_iters_corr = [i/c if c > 0 else 0 for i, c in zip(minimize_iters_correct, minimize_counts_correct)]
                se_min_iters_corr = np.std(avg_min_iters_corr) / np.sqrt(len(avg_min_iters_corr))
                print(f", MinIters(corr) {np.mean(avg_min_iters_corr):.1f}±{se_min_iters_corr:.1f}", end="")
            if minimize_times_correct and minimize_iters_correct:
                time_per_iter_corr = [t/i if i > 0 else 0 for t, i in zip(minimize_times_correct, minimize_iters_correct)]
                se_time_per_iter_corr = np.std(time_per_iter_corr) / np.sqrt(len(time_per_iter_corr))
                print(f", TimePerIter(corr) {np.mean(time_per_iter_corr)*1000:.2f}±{se_time_per_iter_corr*1000:.2f}ms", end="")
            
            # Print "all" metrics
            if times_all:
                se_time_all = np.std(times_all) / np.sqrt(len(times_all))
                print(f", Time(all) {np.mean(times_all):.2f}±{se_time_all:.2f}s", end="")
            if subspace_errors_all:
                se_suberr_all = np.std(subspace_errors_all) / np.sqrt(len(subspace_errors_all))
                print(f", SubErr(all) {np.mean(subspace_errors_all):.4f}±{se_suberr_all:.4f}", end="")
            if minimize_times_all and minimize_counts_all:
                avg_min_time_all = [t/c if c > 0 else 0 for t, c in zip(minimize_times_all, minimize_counts_all)]
                se_min_time_all = np.std(avg_min_time_all) / np.sqrt(len(avg_min_time_all))
                print(f", MinTime(all) {np.mean(avg_min_time_all):.4f}±{se_min_time_all:.4f}s", end="")
            if minimize_iters_all and minimize_counts_all:
                avg_min_iters_all = [i/c if c > 0 else 0 for i, c in zip(minimize_iters_all, minimize_counts_all)]
                se_min_iters_all = np.std(avg_min_iters_all) / np.sqrt(len(avg_min_iters_all))
                print(f", MinIters(all) {np.mean(avg_min_iters_all):.1f}±{se_min_iters_all:.1f}", end="")
            if minimize_times_all and minimize_iters_all:
                time_per_iter_all = [t/i if i > 0 else 0 for t, i in zip(minimize_times_all, minimize_iters_all)]
                se_time_per_iter_all = np.std(time_per_iter_all) / np.sqrt(len(time_per_iter_all))
                print(f", TimePerIter(all) {np.mean(time_per_iter_all)*1000:.2f}±{se_time_per_iter_all*1000:.2f}ms", end="")
            print()
            
            # Compute avg minimize time and iterations per call
            avg_min_time_corr = [t/c if c > 0 else 0 for t, c in zip(minimize_times_correct, minimize_counts_correct)] if minimize_times_correct else []
            avg_min_time_all = [t/c if c > 0 else 0 for t, c in zip(minimize_times_all, minimize_counts_all)] if minimize_times_all else []
            avg_min_iters_corr = [i/c if c > 0 else 0 for i, c in zip(minimize_iters_correct, minimize_counts_correct)] if minimize_iters_correct else []
            avg_min_iters_all = [i/c if c > 0 else 0 for i, c in zip(minimize_iters_all, minimize_counts_all)] if minimize_iters_all else []
            # Compute time per iteration (total time / total iters for each rep)
            time_per_iter_corr = [t/i if i > 0 else 0 for t, i in zip(minimize_times_correct, minimize_iters_correct)] if minimize_times_correct else []
            time_per_iter_all = [t/i if i > 0 else 0 for t, i in zip(minimize_times_all, minimize_iters_all)] if minimize_times_all else []
            
            # Store result
            results.append({
                'Method': method_name,
                'Model': model,
                'X_dist': distribution,
                'n': n,
                'n_reps': n_reps,
                'd_true': d_true,
                'n_correct': d_correct,
                'dim_accuracy': d_correct / n_reps,
                # Metrics when dimension correct
                'time_mean': np.mean(times_correct) if times_correct else np.nan,
                'time_std': np.std(times_correct) / np.sqrt(len(times_correct)) if times_correct else np.nan,
                'subspace_error_mean': np.mean(subspace_errors) if subspace_errors else np.nan,
                'subspace_error_std': np.std(subspace_errors) / np.sqrt(len(subspace_errors)) if subspace_errors else np.nan,
                'minimize_time_mean': np.mean(avg_min_time_corr) if avg_min_time_corr else np.nan,
                'minimize_time_std': np.std(avg_min_time_corr) / np.sqrt(len(avg_min_time_corr)) if avg_min_time_corr else np.nan,
                'minimize_iters_mean': np.mean(avg_min_iters_corr) if avg_min_iters_corr else np.nan,
                'minimize_iters_std': np.std(avg_min_iters_corr) / np.sqrt(len(avg_min_iters_corr)) if avg_min_iters_corr else np.nan,
                'time_per_iter_mean': np.mean(time_per_iter_corr) if time_per_iter_corr else np.nan,
                'time_per_iter_std': np.std(time_per_iter_corr) / np.sqrt(len(time_per_iter_corr)) if time_per_iter_corr else np.nan,
                # Metrics for all runs
                'time_all_mean': np.mean(times_all) if times_all else np.nan,
                'time_all_std': np.std(times_all) / np.sqrt(len(times_all)) if times_all else np.nan,
                'subspace_error_all_mean': np.mean(subspace_errors_all) if subspace_errors_all else np.nan,
                'subspace_error_all_std': np.std(subspace_errors_all) / np.sqrt(len(subspace_errors_all)) if subspace_errors_all else np.nan,
                'minimize_time_all_mean': np.mean(avg_min_time_all) if avg_min_time_all else np.nan,
                'minimize_time_all_std': np.std(avg_min_time_all) / np.sqrt(len(avg_min_time_all)) if avg_min_time_all else np.nan,
                'minimize_iters_all_mean': np.mean(avg_min_iters_all) if avg_min_iters_all else np.nan,
                'minimize_iters_all_std': np.std(avg_min_iters_all) / np.sqrt(len(avg_min_iters_all)) if avg_min_iters_all else np.nan,
                'time_per_iter_all_mean': np.mean(time_per_iter_all) if time_per_iter_all else np.nan,
                'time_per_iter_all_std': np.std(time_per_iter_all) / np.sqrt(len(time_per_iter_all)) if time_per_iter_all else np.nan,
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    column_order = ['Method', 'Model', 'X_dist', 'n', 'n_reps', 'd_true', 'n_correct',
                    'dim_accuracy', 
                    'time_mean', 'time_std', 
                    'subspace_error_mean', 'subspace_error_std',
                    'minimize_time_mean', 'minimize_time_std',
                    'minimize_iters_mean', 'minimize_iters_std',
                    'time_per_iter_mean', 'time_per_iter_std',
                    'time_all_mean', 'time_all_std',
                    'subspace_error_all_mean', 'subspace_error_all_std',
                    'minimize_time_all_mean', 'minimize_time_all_std',
                    'minimize_iters_all_mean', 'minimize_iters_all_std',
                    'time_per_iter_all_mean', 'time_per_iter_all_std']
    df = df[column_order]
    
    # Save to file
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")
    
    return df

#### Minimal call for local experimenting

def run_experiments(
    models=['M1', 'M2', 'M3_func1', 'M3_func2'],
    sample_sizes=[50, 100, 400, 1000, 2000, 5000],
    p_values=[5, 10],
    distributions=['uniform', 'normal'],
    methods=['hat', 'nw'],
    n_reps=100,
    global_seed=42,
    output_file='results.csv'
):
    """
    Run experiments across multiple configurations.
    
    Parameters
    ----------
    models : list
        Models to test, e.g. ['M1', 'M2', 'M3_func1', 'M3_func2']
    sample_sizes : list
        Sample sizes to test, e.g. [50, 100, 400, 1000]
    p_values : list
        Number of predictors, e.g. [5, 10]
    distributions : list
        X distributions, e.g. ['uniform', 'normal']
    methods : list
        Methods to run, e.g. ['hat', 'nw'] or ['hat'] or ['nw']
    n_reps : int
        Number of replications per configuration
    global_seed : int
        Random seed
    output_file : str or None
        CSV filename to save combined results (None to skip saving)
    
    Returns
    -------
    df : pd.DataFrame
        Combined results for all configurations
    """
    all_results = []
    
    # Determine method parameter for run_experiment_single
    if set(methods) == {'hat', 'nw'}:
        method_arg = 'all'
    elif 'hat' in methods:
        method_arg = 'hat'
    elif 'nw' in methods:
        method_arg = 'nw'
    else:
        raise ValueError("methods must contain 'hat' and/or 'nw'")
    
    # Iterate over all configurations
    for model in models:
        for distribution in distributions:
            for p in p_values:
                # Set sigma based on model
                sigma = 2.0 if model == 'M3_func2' else 0.2
                
                # Create temp output filename
                temp_output = f"temp_{model}_{distribution}_p{p}.csv"
                
                # Run experiment for this configuration
                df = run_experiment_single(
                    model=model,
                    distribution=distribution,
                    n_reps=n_reps,
                    p=p,
                    sigma=sigma,
                    global_seed=global_seed,
                    output_file=None,  # Don't save individual files
                    method=method_arg,
                    sample_size=None  # Run all sample sizes
                )
                
                # Filter to requested sample sizes
                df = df[df['n'].isin(sample_sizes)]
                
                # Add p column
                df.insert(4, 'p', p)
                
                all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    column_order = ['Method', 'Model', 'X_dist', 'p', 'n', 'n_reps', 'd_true', 'n_correct',
                    'dim_accuracy', 
                    'time_mean', 'time_std', 
                    'subspace_error_mean', 'subspace_error_std',
                    'minimize_time_mean', 'minimize_time_std',
                    'minimize_iters_mean', 'minimize_iters_std',
                    'time_per_iter_mean', 'time_per_iter_std',
                    'time_all_mean', 'time_all_std',
                    'subspace_error_all_mean', 'subspace_error_all_std',
                    'minimize_time_all_mean', 'minimize_time_all_std',
                    'minimize_iters_all_mean', 'minimize_iters_all_std',
                    'time_per_iter_all_mean', 'time_per_iter_all_std']
    combined_df = combined_df[column_order]
    
    # Save combined results
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"\nAll results saved to '{output_file}'")
    
    return combined_df


# Example: Quick test with minimal configuration
df = run_experiments(
    models=['M1'],
    sample_sizes=[100, 400],
    p_values=[5],
    distributions=['uniform'],
    methods=['hat'],
    n_reps=50,
    output_file='quick_test.csv'
)

#### Below is for command-line execution (used for ALICE cluster experiments)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Run dimension reduction experiments')
#     parser.add_argument('--model', type=str, required=True,
#                        choices=['M1', 'M2', 'M3_func1', 'M3_func2'],
#                        help='Model to run')
#     parser.add_argument('--distribution', type=str, required=True,
#                        choices=['uniform', 'normal'],
#                        help='Distribution for X')
#     parser.add_argument('--method', type=str, default='all',
#                        choices=['hat', 'nw', 'all'],
#                        help='Method to run')
#     parser.add_argument('--sample_size', type=int, default=None,
#                        help='Single sample size to run (default: all)')
#     parser.add_argument('--n_reps', type=int, default=100,
#                        help='Number of replications')
#     parser.add_argument('--p', type=int, default=10,
#                        help='Number of predictors')
#     parser.add_argument('--seed', type=int, default=42,
#                        help='Global random seed')
#     parser.add_argument('--output', type=str, default=None,
#                        help='Output CSV file')
    
#     args = parser.parse_args()
    
#     # Set sigma based on model
#     sigma = 2.0 if args.model == 'M3_func2' else 0.2
    
#     df = run_experiment_single(
#         model=args.model,
#         distribution=args.distribution,
#         n_reps=args.n_reps,
#         p=args.p,
#         sigma=sigma,
#         global_seed=args.seed,
#         output_file=args.output,
#         method=args.method,
#         sample_size=args.sample_size
#     )
    
#     print("\n" + "="*70)
#     print("COMPLETED")
#     print("="*70)
