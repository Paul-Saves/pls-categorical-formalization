"""
Reproducible Gaussian Process Emulation Benchmark for Categorical PLS
====================================================================
Evaluates the 4 regularization strategies across 3 benchmark functions (f1, f2, f3)
as reported in Table 6 of the manuscript:
"Partial Least Squares for Qualitative Correlation Matrices: Geometry, Invariance, and Fast Spectral Repair"
Author: Paul Saves
"""

import numpy as np
from scipy.optimize import minimize
import time

def f1_harmonic(x, c, L=15):
    """Harmonic coupled function f1."""
    return np.sin(2 * np.pi * x) + np.cos(2 * np.pi * c / L) + 0.1 * x * (c / L)

def f2_multimodal(x, c, L=15):
    """Multimodal Griewank-type function f2."""
    return (x**2) / 4000.0 - np.cos(x) + 1.0 + np.sin(4 * np.pi * c / L)

def f3_step_quadratic(x, c, L=15):
    """Step / quadratic discontinuous function f3."""
    c_arr = np.asarray(c)
    step = np.where(c_arr > L / 2, 0.5, 0.0)
    return (x - 0.5)**2 + step + 0.05 * (c_arr / L)**2

def build_incidence_E(L, D_in):
    """Vertex-edge incidence matrix of complete graph K_L."""
    E = np.zeros((L, D_in))
    m = 0
    for j in range(L):
        for jp in range(j + 1, L):
            E[j, m] = 1.0
            E[jp, m] = 1.0
            m += 1
    return E

def compute_pls_rotation(X_cat, y, D_out, x_cont=None):
    """Compute NIPALS PLS rotation matrix and row normalize.
    
    If x_cont is provided, PLS is fitted on the joint design matrix
    X_joint = [X_cont, X_cat] to break the purely categorical Krylov
    degeneracy established in Proposition 2, after which the categorical
    rotation block G_cat is extracted and row-normalized.
    """
    nt, D_in = X_cat.shape
    if x_cont is not None:
        X_cont_mat = np.atleast_2d(x_cont).T if np.ndim(x_cont) == 1 else np.asarray(x_cont)
        d_cont = X_cont_mat.shape[1]
        X_mat = np.hstack([X_cont_mat, X_cat])
    else:
        d_cont = 0
        X_mat = X_cat

    p_tot = X_mat.shape[1]
    X_c = X_mat - np.mean(X_mat, axis=0)
    y_c = y - np.mean(y)

    W = np.zeros((p_tot, D_out))
    P = np.zeros((p_tot, D_out))
    E_k = X_c.copy()
    f_k = y_c.copy()

    for k in range(D_out):
        w_k = E_k.T @ f_k
        norm_w = np.linalg.norm(w_k)
        if norm_w < 1e-12:
            w_k = np.random.randn(p_tot)
            w_k /= np.linalg.norm(w_k)
        else:
            w_k /= norm_w
        
        t_k = E_k @ w_k
        norm_t = np.linalg.norm(t_k)
        if norm_t < 1e-12:
            break
        t_k /= norm_t
        p_k = E_k.T @ t_k
        q_k = np.dot(f_k, t_k)
        E_k = E_k - np.outer(t_k, p_k)
        f_k = f_k - t_k * q_k
        W[:, k] = w_k
        P[:, k] = p_k

    PW = P.T @ W
    try:
        G_raw = W @ np.linalg.inv(PW)
    except np.linalg.LinAlgError:
        G_raw = W

    if d_cont > 0:
        G_cat = G_raw[d_cont:, :]
    else:
        G_cat = G_raw

    row_norms = np.linalg.norm(G_cat, axis=1, keepdims=True)
    row_norms[row_norms < 1e-12] = 1.0
    G_star = G_cat / row_norms
    return G_star

def reconstruct_and_repair(th_vec, G_star, L, strategy, eps=1e-4):
    """Reconstruct L x L correlation matrix and apply chosen repair strategy."""
    off = (G_star**2) @ th_vec
    Theta = np.eye(L)
    m = 0
    for j in range(L):
        for jp in range(j + 1, L):
            Theta[j, jp] = off[m]
            Theta[jp, j] = off[m]
            m += 1

    if strategy == 'nugget':
        e = np.linalg.eigvalsh(Theta)
        lam_min = np.min(e)
        eta = max(0.0, -lam_min + eps)
        return Theta + eta * np.eye(L)
    elif strategy == 'clip':
        V, S, Vt = np.linalg.svd(Theta)
        S_clip = np.maximum(S, eps)
        Th_c = V @ np.diag(S_clip) @ Vt
        d = np.sqrt(np.maximum(1e-12, np.diag(Th_c)))
        return Th_c / np.outer(d, d)
    elif strategy == 'exact':
        e = np.linalg.eigvalsh(Theta)
        lam_min = np.min(e)
        eta = max(0.0, -lam_min + eps)
        return (Theta + eta * np.eye(L)) / (1.0 + eta)
    elif strategy == 'gersh':
        row_sums = np.sum(np.abs(Theta), axis=1) - 1.0
        eta = max(0.0, float(np.max(row_sums)) - 1.0 + eps)
        return (Theta + eta * np.eye(L)) / (1.0 + eta)
    elif strategy == 'mdg':
        off_diag = []
        for j in range(L):
            for jp in range(j + 1, L):
                off_diag.append(Theta[j, jp])
        rho_bar = max(0.0, float(np.mean(off_diag)))
        row_sums = np.sum(np.abs(Theta - rho_bar), axis=1) - np.abs(1.0 - rho_bar)
        eta_mdg = max(0.0, float(np.max(row_sums)) + rho_bar - 1.0 + eps)
        eta_g = max(0.0, float(np.max(np.sum(np.abs(Theta), axis=1) - 1.0)) - 1.0 + eps)
        eta = min(eta_g, eta_mdg)
        return (Theta + eta * np.eye(L)) / (1.0 + eta)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def run_single_replication(fn, L=15, ell=3, nt=45, n_test=30, noise_std=0.01, seed=42):
    """Run 1 replication for all 4 strategies and report condition numbers and test RMSE."""
    rng = np.random.RandomState(seed)
    D_in = L * (L - 1) // 2
    D_out = ell * (ell - 1) // 2

    # Balanced training design
    cats_train = np.array([j for j in range(L) for _ in range(nt // L)])
    rng.shuffle(cats_train)
    x_train = rng.uniform(0.0, 1.0, nt)
    y_clean_train = fn(x_train, cats_train, L)
    y_train = y_clean_train + rng.normal(0.0, noise_std, nt)

    # Test design
    cats_test = rng.randint(0, L, n_test)
    x_test = rng.uniform(0.0, 1.0, n_test)
    y_clean_test = fn(x_test, cats_test, L)

    # Encode X_cat
    E = build_incidence_E(L, D_in)
    M_train = np.zeros((nt, L))
    for i, c in enumerate(cats_train):
        M_train[i, c] = 1.0
    X_cat_train = M_train @ E

    M_test = np.zeros((n_test, L))
    for i, c in enumerate(cats_test):
        M_test[i, c] = 1.0

    # PLS rotation on joint continuous-categorical design matrix
    G_star = compute_pls_rotation(X_cat_train, y_train, D_out, x_cont=x_train)

    results = {}
    strategies = ['nugget', 'clip', 'exact', 'gersh', 'mdg']

    D_x_train = (x_train[:, None] - x_train[None, :])**2
    D_x_test = (x_test[:, None] - x_train[None, :])**2

    for strat in strategies:
        def neg_log_lik(params):
            theta_q = params[0]
            th_vec = params[1:]
            Th = reconstruct_and_repair(th_vec, G_star, L, strat)
            R_q = np.exp(-theta_q * D_x_train)
            K = R_q * (M_train @ Th @ M_train.T) + (noise_std**2) * np.eye(nt)
            try:
                L_chol = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, y_train))
            sig2 = np.dot(y_train, alpha) / nt
            if sig2 <= 0:
                return 1e10
            log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
            return 0.5 * log_det + 0.5 * nt * np.log(sig2)

        # 5 multi-starts
        best_nll = 1e10
        best_params = None
        starts = [
            [5.0, 0.2, 0.2, 0.2],
            [1.0, -0.3, 0.1, 0.4],
            [10.0, 0.5, -0.2, -0.2],
            [2.0, 0.0, 0.0, 0.0],
            [15.0, 0.7, 0.7, 0.7]
        ]
        bounds = [(0.05, 50.0), (-0.95, 0.95), (-0.95, 0.95), (-0.95, 0.95)]

        for x0 in starts:
            res = minimize(neg_log_lik, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 50})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x

        if best_params is None:
            best_params = starts[0]

        theta_q_opt = best_params[0]
        th_opt = reconstruct_and_repair(best_params[1:], G_star, L, strat)

        # Training Covariance Condition Number
        R_train = np.exp(-theta_q_opt * D_x_train)
        K_train = R_train * (M_train @ th_opt @ M_train.T) + (noise_std**2) * np.eye(nt)
        e_K = np.linalg.eigvalsh(K_train)
        cond_K = np.max(e_K) / max(1e-12, np.min(e_K))

        # Prediction on test set
        R_test = np.exp(-theta_q_opt * D_x_test)
        K_cross = R_test * (M_test @ th_opt @ M_train.T)
        try:
            L_chol = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, y_train))
            y_pred = K_cross @ alpha
            rmse = np.sqrt(np.mean((y_pred - y_clean_test)**2))
        except np.linalg.LinAlgError:
            rmse = 5.0  # Breakdown penalty

        results[strat] = {'cond': cond_K, 'rmse': rmse}

    return results

if __name__ == '__main__':
    print("=" * 78)
    print("RUNNING GP EMULATION BENCHMARK (TEST REPLICATIONS)")
    print("=" * 78)
    test_funcs = [('f1_harmonic', f1_harmonic), ('f2_multimodal', f2_multimodal), ('f3_step_quadratic', f3_step_quadratic)]

    for fname, fn in test_funcs:
        print(f"\n--- Benchmark Function: {fname} (5 replications smoke test) ---")
        conds = {'nugget': [], 'clip': [], 'exact': [], 'gersh': [], 'mdg': []}
        rmses = {'nugget': [], 'clip': [], 'exact': [], 'gersh': [], 'mdg': []}
        for rep in range(5):
            res = run_single_replication(fn, seed=100 + rep)
            for s in res:
                conds[s].append(res[s]['cond'])
                rmses[s].append(res[s]['rmse'])
        for s in ['nugget', 'clip', 'exact', 'gersh', 'mdg']:
            print(f"  {s:8s} | Cond: {np.mean(conds[s]):.2e} +/- {np.std(conds[s]):.2e} | RMSE: {np.mean(rmses[s]):.4f} +/- {np.std(rmses[s]):.4f}")
    print("\nBenchmark smoke test completed successfully.")
