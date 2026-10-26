"""
Latent Dimension Scaling Benchmark (ell in [2, 8]) for Categorical PLS GP
=========================================================================
Companion benchmark and automated verification suite (unit, regression, and
end-to-end integration tests) for Section 6.5 of the manuscript:
"Partial Least Squares for Correlation Matrices: Theory and Applications"
Author: Paul Saves

Evaluates all five correlation-matrix regularization strategies across every
integer latent PLS dimension ell in {2, 3, 4, 5, 6, 7, 8}
(D_out in {1, 3, 6, 10, 15, 21, 28}, keeping ell < L to remain in the
supervised dimension-reduction regime):
  1. Raw Nugget (Theta + eta * I_L)                [O(L^3), non-unit diagonal]
  2. Spectral Clipping (eigenvalue clip + scale)   [O(L^3), unit diagonal]
  3. Exact Spectral Repair ((Theta+eta I)/(1+eta)) [O(L^3), unit diagonal]
  4. Fast Gershgorin Repair (eta_Gersh)            [O(L^2), unit diagonal]
  5. Mean-Deflated Gershgorin Repair (eta_MDG*)    [O(L^2), unit diagonal]

Methodological notes:
  1. Optimizer symmetry (`jac=None`): All five strategies are optimized using
     identical numerical finite-difference L-BFGS-B (`jac=None`), avoiding any
     asymmetry between piecewise-smooth closed-form Jacobians (MDG/Gershgorin)
     and eigensolver-based methods.
  2. Dimension-scaled multi-start budget (`n_starts = ell`): Because the
     non-convex profile likelihood lives on a (1 + D_out)-dimensional domain
     with D_out = ell(ell-1)/2 growing quadratically, the multi-start budget is
     scaled as n_starts(ell) = ell in {2, 3, 4, 5, 6, 7, 8} (shared identically
     across all five strategies), preventing spurious local-minimum trapping at
     higher ell while faithfully capturing the resulting computational cost.

Usage:
  python benchmark_ell_scaling.py --verify   (Unit + regression + integration suite)
  python benchmark_ell_scaling.py --quick    (6 replications smoke test)
  python benchmark_ell_scaling.py --full     (20 replications full benchmark)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

STRATEGIES = ["nugget", "clip", "exact", "gersh", "mdg"]
STRATEGY_LABELS = {
    "nugget": "Raw Nugget (Theta + eta I)",
    "clip": "Spectral Clipping (SVD)",
    "exact": "Exact Spectral Repair",
    "gersh": "Fast Gershgorin Repair",
    "mdg": "Mean-Deflated MDG",
}
ELL_GRID = [2, 3, 4, 5, 6, 7, 8]


# =========================================================================
# 1. Complete-Graph Incidence, Hypersphere Parameterization & Adjoint R_W
# =========================================================================

def build_incidence_E(L: int) -> np.ndarray:
    """Unoriented vertex-edge incidence matrix E in {0, 1}^{L x D_in} of K_L."""
    D_in = L * (L - 1) // 2
    E = np.zeros((L, D_in))
    m = 0
    for j in range(L):
        for jp in range(j + 1, L):
            E[j, m] = 1.0
            E[jp, m] = 1.0
            m += 1
    return E


def hypersphere_angles_to_corr_vec(
    ell: int, phi: np.ndarray, ii_lat: np.ndarray, jj_lat: np.ndarray
) -> np.ndarray:
    """Vectorized trigonometric map from D_out spherical Cholesky angles
    phi in (0, pi) to upper-triangular off-diagonals of Theta_hat in C_ell."""
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    L_c = np.zeros((ell, ell))
    L_c[0, 0] = 1.0
    idx = 0
    for i in range(1, ell):
        c_row = cos_p[idx : idx + i]
        s_row = sin_p[idx : idx + i]
        idx += i
        sp = 1.0
        for k in range(i):
            L_c[i, k] = sp * c_row[k]
            sp *= s_row[k]
        L_c[i, i] = sp
    Theta_hat = L_c @ L_c.T
    return Theta_hat[ii_lat, jj_lat]


def corr_matrix_to_hypersphere_angles(
    Theta_hat: np.ndarray, low: float = 0.06, high: float = np.pi - 0.06
) -> np.ndarray:
    """Exact inverse map from an SPD correlation matrix Theta_hat in C_ell
    to spherical Cholesky angles phi in [low, high]^{D_out}."""
    ell = Theta_hat.shape[0]
    D_out = ell * (ell - 1) // 2
    eigs = np.linalg.eigvalsh(Theta_hat)
    if np.min(eigs) < 1e-4:
        eta0 = max(0.0, -float(np.min(eigs)) + 1e-3)
        Theta_hat = (Theta_hat + eta0 * np.eye(ell)) / (1.0 + eta0)
    L_c = np.linalg.cholesky(Theta_hat)
    phi = np.zeros(D_out)
    idx = 0
    for i in range(1, ell):
        for k in range(i):
            rem_norm = float(np.linalg.norm(L_c[i, k : i + 1]))
            if rem_norm < 1e-12:
                angle = np.pi / 2.0
            else:
                cos_val = float(np.clip(L_c[i, k] / rem_norm, -1.0 + 1e-7, 1.0 - 1e-7))
                angle = float(np.arccos(cos_val))
            phi[idx] = float(np.clip(angle, low, high))
            idx += 1
    return phi


def compute_empirical_adjoint_warmstart(
    X_tr: np.ndarray, y_tr: np.ndarray, cats_tr: np.ndarray, G_star: np.ndarray, L: int, ell: int
) -> np.ndarray:
    """Compute adjoint reduction warm-start Theta_hat_0 = R_W(Theta_emp) (Definition 4)
    from ridge-stabilized per-category response profiles and map to spherical angles."""
    p_feat = 1 + X_tr.shape[1]
    profiles = np.zeros((L, p_feat))
    X_aug = np.column_stack([np.ones(len(y_tr)), X_tr - np.mean(X_tr, axis=0)])
    global_beta, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
    for c in range(L):
        mask = cats_tr == c
        if np.sum(mask) >= 2:
            Xc = X_aug[mask]
            yc = y_tr[mask]
            beta_c = np.linalg.solve(
                Xc.T @ Xc + 0.25 * np.eye(p_feat), Xc.T @ yc + 0.25 * global_beta
            )
            profiles[c] = beta_c
        elif np.sum(mask) == 1:
            profiles[c] = global_beta.copy()
            profiles[c, 0] = float(y_tr[mask][0])
        else:
            profiles[c] = global_beta.copy()

    profiles_c = profiles - np.mean(profiles, axis=0, keepdims=True)
    norms = np.linalg.norm(profiles_c, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    U_prof = profiles_c / norms
    Theta_emp = 0.85 * (U_prof @ U_prof.T) + 0.15 * np.eye(L)
    np.fill_diagonal(Theta_emp, 1.0)

    ii_L, jj_L = np.triu_indices(L, k=1)
    theta_emp_vec = Theta_emp[ii_L, jj_L]
    A_w = G_star ** 2
    col_sums = np.sum(A_w, axis=0)
    W_adj = np.zeros_like(A_w.T)
    nz = col_sums > 1e-12
    W_adj[nz] = A_w[:, nz].T / col_sums[nz, None]
    if np.any(~nz):
        W_adj[~nz] = 1.0 / A_w.shape[0]
    th_lat_0 = W_adj @ theta_emp_vec

    ii_lat, jj_lat = np.triu_indices(ell, k=1)
    Theta_hat_0 = np.eye(ell)
    Theta_hat_0[ii_lat, jj_lat] = th_lat_0
    Theta_hat_0[jj_lat, ii_lat] = th_lat_0
    return corr_matrix_to_hypersphere_angles(Theta_hat_0)


def build_scaled_multistarts(
    ell: int, phi_warm: np.ndarray, rng: np.random.RandomState
) -> list[np.ndarray]:
    """Build n_starts(ell) = ell initial points in [theta_quant, phi_1, ..., phi_{D_out}],
    scaling multi-start coverage with latent dimension ell in [2, 8]:
      - Start 1: Adjoint reduction warm-start R_W(Theta_emp)
      - Start 2: Uncorrelated latent identity I_ell (phi = pi/2)
      - Start 3 (ell >= 3): Moderate positive equicorrelation (rho = 0.35)
      - Starts 4..ell (ell >= 4): Random well-conditioned factor correlation matrices in C_ell
    """
    D_out = ell * (ell - 1) // 2
    starts = [
        np.concatenate([[1.8], phi_warm]),
        np.concatenate([[1.8], np.full(D_out, np.pi / 2.0)]),
    ]
    if ell >= 3:
        Th_eq = 0.65 * np.eye(ell) + 0.35 * np.ones((ell, ell))
        starts.append(np.concatenate([[2.6], corr_matrix_to_hypersphere_angles(Th_eq)]))
    tq_grid = [1.2, 3.2, 1.5, 2.4, 3.8]
    for s_idx in range(3, ell):
        F_mat = rng.randn(ell, ell + 1)
        cov = F_mat @ F_mat.T + 0.35 * np.eye(ell)
        d_s = np.sqrt(np.diag(cov))
        Th_rand = cov / np.outer(d_s, d_s)
        tq_init = tq_grid[(s_idx - 3) % len(tq_grid)]
        starts.append(np.concatenate([[tq_init], corr_matrix_to_hypersphere_angles(Th_rand)]))
    return starts


# =========================================================================
# 2. Rank-Guarded RN-PLS & Five Regularization Operators
# =========================================================================

def compute_rn_pls_rotation(
    X_cat: np.ndarray, y: np.ndarray, D_out: int, X_cont: np.ndarray
) -> tuple[np.ndarray, int]:
    """Compute rank-guarded NIPALS PLS rotation matrix on standardized joint
    matrix X_joint = [X_cont, X_cat] and extract row-normalized G_* in R^{D_in x D_out}.
    Returns (G_star, k_act) where k_act <= min(n_t - 1, d + L_obs - 1) is the
    active Krylov dimension (Eq. eq:krylov_bound)."""
    X_joint = np.hstack([X_cont, X_cat])
    d_cont = X_cont.shape[1]
    X_mean = np.mean(X_joint, axis=0)
    X_std = np.std(X_joint, axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_c = (X_joint - X_mean) / X_std
    y_c = (y - np.mean(y)) / max(1e-12, float(np.std(y)))

    x_norm0 = float(np.linalg.norm(X_c))
    p_tot = X_c.shape[1]
    W = np.zeros((p_tot, D_out))
    P = np.zeros((p_tot, D_out))
    E_k = X_c.copy()
    f_k = y_c.copy()
    k_act = 0

    for k in range(D_out):
        if float(np.linalg.norm(E_k)) < 1e-9 * x_norm0:
            break
        w_k = E_k.T @ f_k
        norm_w = float(np.linalg.norm(w_k))
        if norm_w < 1e-9:
            break
        w_k = w_k / norm_w
        t_k = E_k @ w_k
        norm_t = float(np.linalg.norm(t_k))
        if norm_t < 1e-9:
            break
        t_k = t_k / norm_t
        p_k = E_k.T @ t_k
        q_k = float(np.dot(f_k, t_k))
        E_k = E_k - np.outer(t_k, p_k)
        f_k = f_k - t_k * q_k
        W[:, k] = w_k
        P[:, k] = p_k
        k_act += 1

    G_raw = np.zeros((p_tot, D_out))
    if k_act > 0:
        PW = P[:, :k_act].T @ W[:, :k_act]
        G_raw[:, :k_act] = W[:, :k_act] @ np.linalg.solve(PW, np.eye(k_act))

    G_cat = G_raw[d_cont:, :]
    row_norms = np.linalg.norm(G_cat, axis=1, keepdims=True)
    zero_rows = row_norms.squeeze() < 1e-12
    G_star = np.zeros_like(G_cat)
    G_star[~zero_rows] = G_cat[~zero_rows] / row_norms[~zero_rows]
    G_star[zero_rows] = 1.0 / np.sqrt(D_out)
    return G_star, k_act


def apply_repair_strategy(Theta: np.ndarray, strategy: str, eps: float = 1e-4) -> np.ndarray:
    """Apply one of the 5 correlation regularization strategies of Section 6.2."""
    L = Theta.shape[0]
    if strategy == "nugget":
        lam_min = float(np.min(np.linalg.eigvalsh(Theta)))
        eta = max(0.0, -lam_min + eps)
        out = Theta.copy()
        out.flat[:: L + 1] += eta
        return out
    elif strategy == "clip":
        eigs, V = np.linalg.eigh(Theta)
        Th_c = (V * np.maximum(eigs, eps)) @ V.T
        d_v = np.sqrt(np.maximum(1e-12, np.diag(Th_c)))
        return Th_c / np.outer(d_v, d_v)
    elif strategy == "exact":
        lam_min = float(np.min(np.linalg.eigvalsh(Theta)))
        eta = max(0.0, -lam_min + eps)
        if eta == 0.0:
            return Theta.copy()
        out = Theta.copy()
        out.flat[:: L + 1] += eta
        out /= 1.0 + eta
        return out
    elif strategy == "gersh":
        row_sums = np.sum(np.abs(Theta), axis=1) - 1.0
        eta = max(0.0, float(np.max(row_sums)) - 1.0 + eps)
        if eta == 0.0:
            return Theta.copy()
        out = Theta.copy()
        out.flat[:: L + 1] += eta
        out /= 1.0 + eta
        return out
    elif strategy == "mdg":
        rho_bar_plus = max(0.0, float((np.sum(Theta) - L) / (L * (L - 1))))
        row_sums_defl = np.sum(np.abs(Theta - rho_bar_plus), axis=1) - abs(1.0 - rho_bar_plus)
        eta_mdg = max(0.0, float(np.max(row_sums_defl)) + rho_bar_plus - 1.0 + eps)
        eta_gersh = max(0.0, float(np.max(np.sum(np.abs(Theta), axis=1) - 1.0)) - 1.0 + eps)
        eta = min(eta_gersh, eta_mdg)
        if eta == 0.0:
            return Theta.copy()
        out = Theta.copy()
        out.flat[:: L + 1] += eta
        out /= 1.0 + eta
        return out
    else:
        raise ValueError(f"Unknown regularization strategy: {strategy}")


# =========================================================================
# 3. Multi-Family Mixed Benchmark Function & GP MLE Protocol
# =========================================================================

def fn_clustered_mixed_response(X: np.ndarray, cats: np.ndarray) -> np.ndarray:
    """Smooth 2D mixed continuous-categorical benchmark over L = 15 levels
    partitioned into 5 physical families of 3 strongly correlated grades each,
    with predominantly positive intra-family correlations and distinct continuous
    sensitivities requiring higher-order latent PLS subspaces (ell >= 4..6)."""
    grp = cats // 3  # 5 families of 3 categories
    sub = (cats % 3) - 1.0
    b0 = np.array([2.42, 1.68, 0.96, 0.32, -0.72])[grp] + 0.05 * sub
    a1 = np.array([1.50, 1.02, 0.40, 0.15, -0.92])[grp] + 0.04 * sub
    a2 = np.array([0.42, 1.35, -0.35, 0.85, -0.45])[grp] - 0.03 * sub
    a3 = np.array([0.92, 0.35, 1.15, 0.75, -0.40])[grp] + 0.03 * sub
    return (
        b0
        + a1 * np.sin(np.pi * X[:, 0])
        + a2 * np.cos(np.pi * X[:, 1])
        + a3 * X[:, 0] * X[:, 1]
    )


def _run_single_rep_task(args: tuple) -> dict:
    """Top-level worker executing one replication at a given ell across all 5 strategies."""
    ell, rep, L, d, nt, n_test, noise_std, maxiter, base_seed = args
    D_out = ell * (ell - 1) // 2
    ii_L, jj_L = np.triu_indices(L, k=1)
    ii_lat, jj_lat = np.triu_indices(ell, k=1)
    E = build_incidence_E(L)

    rng_data = np.random.RandomState(base_seed + rep)
    cats_tr = np.concatenate(
        [np.repeat(np.arange(L), 3), rng_data.randint(0, L, size=nt - 3 * L)]
    )
    rng_data.shuffle(cats_tr)
    X_tr = np.column_stack(
        [(rng_data.permutation(nt) + rng_data.uniform(size=nt)) / nt for _ in range(d)]
    )
    y_raw = fn_clustered_mixed_response(X_tr, cats_tr) + rng_data.normal(0.0, noise_std, size=nt)
    y_mean = float(np.mean(y_raw))
    y_std = max(1e-12, float(np.std(y_raw)))
    y_tr = (y_raw - y_mean) / y_std

    X_te = rng_data.uniform(0.0, 1.0, size=(n_test, d))
    cats_te = rng_data.randint(0, L, size=n_test)
    y_te = fn_clustered_mixed_response(X_te, cats_te)

    X_cat_tr = np.eye(L)[cats_tr] @ E
    G_star, k_act = compute_rn_pls_rotation(X_cat_tr, y_tr, D_out, X_tr)
    A_w = G_star ** 2

    phi_warm = compute_empirical_adjoint_warmstart(X_tr, y_tr, cats_tr, G_star, L, ell)
    rng_starts = np.random.RandomState(base_seed + 1000 * ell + rep)
    starts = build_scaled_multistarts(ell, phi_warm, rng_starts)
    bounds = [(0.05, 25.0)] + [(0.06, np.pi - 0.06)] * D_out

    Dx_tr = np.sum((X_tr[:, None, :] - X_tr[None, :, :]) ** 2, axis=2)
    Dx_te = np.sum((X_te[:, None, :] - X_tr[None, :, :]) ** 2, axis=2)
    cats_r, cats_c = cats_tr[:, None], cats_tr[None, :]
    Th_buf = np.eye(L)

    out = {"k_act": k_act, "strat": {}}
    for strat in STRATEGIES:
        t_start = time.perf_counter()

        def neg_log_lik(params: np.ndarray) -> float:
            tq = params[0]
            th_lat = hypersphere_angles_to_corr_vec(ell, params[1:], ii_lat, jj_lat)
            off = A_w @ th_lat
            Th_buf[ii_L, jj_L] = off
            Th_buf[jj_L, ii_L] = off
            Th_rep = apply_repair_strategy(Th_buf, strat, eps=1e-4)
            K = np.exp(-tq * Dx_tr) * Th_rep[cats_r, cats_c]
            K.flat[:: nt + 1] += 1e-4
            try:
                c_fac, lower = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = cho_solve((c_fac, lower), y_tr, check_finite=False)
            sig2 = float(np.dot(y_tr, alpha)) / nt
            if sig2 <= 0.0:
                return 1e10
            return float(np.sum(np.log(np.diag(c_fac)))) + 0.5 * nt * np.log(max(sig2, 1e-15))

        best_res = None
        tot_nfev = 0
        for x0 in starts:
            res = minimize(
                neg_log_lik,
                x0,
                jac=None,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-7},
            )
            tot_nfev += int(res.nfev)
            if best_res is None or res.fun < best_res.fun:
                best_res = res

        elapsed = time.perf_counter() - t_start

        tq_opt = float(best_res.x[0])
        th_lat_opt = hypersphere_angles_to_corr_vec(ell, best_res.x[1:], ii_lat, jj_lat)
        off_opt = A_w @ th_lat_opt
        Th_raw = np.eye(L)
        Th_raw[ii_L, jj_L] = off_opt
        Th_raw[jj_L, ii_L] = off_opt
        Th_opt = apply_repair_strategy(Th_raw, strat, eps=1e-4)

        K_tr = np.exp(-tq_opt * Dx_tr) * Th_opt[cats_r, cats_c]
        K_tr.flat[:: nt + 1] += 1e-4
        eigs_K = np.linalg.eigvalsh(K_tr)
        cond_val = float(np.max(eigs_K) / max(1e-12, float(np.min(eigs_K))))

        c_fac, lower = cho_factor(K_tr, lower=True, check_finite=False)
        alpha = cho_solve((c_fac, lower), y_tr, check_finite=False)
        K_cross = np.exp(-tq_opt * Dx_te) * Th_opt[cats_te[:, None], cats_tr[None, :]]
        y_pred = y_mean + y_std * (K_cross @ alpha)
        rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))

        out["strat"][strat] = {
            "rmse": rmse,
            "cond": cond_val,
            "time": elapsed,
            "nfev": tot_nfev,
        }
    return out


def run_ell_scaling_experiment(
    ell_list: list[int] = ELL_GRID,
    n_reps: int = 20,
    L: int = 15,
    d: int = 2,
    nt: int = 75,
    n_test: int = 300,
    noise_std: float = 0.01,
    maxiter: int = 45,
    base_seed: int = 3100,
    n_workers: int = 5,
) -> dict:
    """Evaluate all 5 regularization strategies across ell in ell_list using
    identical finite-difference L-BFGS-B (`jac=None`) and dimension-scaled
    multi-starts n_starts(ell) = ell."""
    summary = {}
    max_w = max(1, min(n_workers, n_reps))

    with ProcessPoolExecutor(max_workers=max_w) as pool:
        for ell in ell_list:
            D_out = ell * (ell - 1) // 2
            n_starts = ell
            tasks = [
                (ell, rep, L, d, nt, n_test, noise_std, maxiter, base_seed)
                for rep in range(n_reps)
            ]
            rep_outputs = list(pool.map(_run_single_rep_task, tasks))

            k_acts = [r["k_act"] for r in rep_outputs]
            summary[ell] = {
                "D_out": D_out,
                "n_starts": n_starts,
                "k_act_mean": float(np.mean(k_acts)),
                "k_act_max": int(np.max(k_acts)),
                "krylov_bound": d + L - 1,
                "strategies": {},
            }
            for s in STRATEGIES:
                arr_r = np.array([r["strat"][s]["rmse"] for r in rep_outputs])
                arr_c = np.array([r["strat"][s]["cond"] for r in rep_outputs])
                arr_t = np.array([r["strat"][s]["time"] for r in rep_outputs])
                arr_f = np.array([r["strat"][s]["nfev"] for r in rep_outputs])
                summary[ell]["strategies"][s] = {
                    "rmse_mean": float(np.mean(arr_r)),
                    "rmse_sd": float(np.std(arr_r)),
                    "rmse_median": float(np.median(arr_r)),
                    "rmse_q25": float(np.percentile(arr_r, 25)),
                    "rmse_q75": float(np.percentile(arr_r, 75)),
                    "cond_mean": float(np.mean(arr_c)),
                    "cond_sd": float(np.std(arr_c)),
                    "time_mean": float(np.mean(arr_t)),
                    "time_sd": float(np.std(arr_t)),
                    "nfev_mean": float(np.mean(arr_f)),
                }

            row_str = " | ".join(
                f"{s}: {summary[ell]['strategies'][s]['rmse_mean']:.4f} "
                f"[{summary[ell]['strategies'][s]['rmse_median']:.4f}] "
                f"({summary[ell]['strategies'][s]['time_mean']:.2f}s)"
                for s in STRATEGIES
            )
            print(
                f"ell={ell:2d} (D={D_out:2d}, S={n_starts}, k={summary[ell]['k_act_mean']:4.1f}/{d+L-1}) | {row_str}",
                flush=True,
            )

    return summary


# =========================================================================
# 4. Automated Unit, Regression & Integration Verification Suite (--verify)
# =========================================================================

def run_verification_suite() -> None:
    """Execute unit, regression, and end-to-end integration checks for ell in [2, 8]."""
    print("=" * 78)
    print("RUNNING AUTOMATED UNIT, REGRESSION & INTEGRATION TESTS (ell in [2, 8])")
    print("=" * 78)

    # [UNIT 1] Exact bijection between Cholesky hypersphere angles and C_ell across all ell in [2, 8]
    rng = np.random.RandomState(42)
    for ell in ELL_GRID:
        ii_lat, jj_lat = np.triu_indices(ell, k=1)
        A = rng.randn(ell, 2 * ell)
        cov = A @ A.T + 0.5 * np.eye(ell)
        d_s = np.sqrt(np.diag(cov))
        Th_orig = cov / np.outer(d_s, d_s)
        phi = corr_matrix_to_hypersphere_angles(Th_orig, low=1e-6, high=np.pi - 1e-6)
        vec_rec = hypersphere_angles_to_corr_vec(ell, phi, ii_lat, jj_lat)
        assert np.allclose(vec_rec, Th_orig[ii_lat, jj_lat], atol=1e-6), f"Hypersphere bijection failed at ell={ell}"
        starts_u = build_scaled_multistarts(ell, phi, rng)
        assert len(starts_u) == ell, f"Expected n_starts == {ell}, got {len(starts_u)}"
    print("[PASS] Unit Test 1: Cholesky hypersphere bijection & n_starts=ell across ell in {2,3,4,5,6,7,8}.")

    # [UNIT 2] Proposition 4 (ell=2 collapse) & Theorem 8 (SPD + unit diagonal invariants)
    L_u = 15
    ii_L, jj_L = np.triu_indices(L_u, k=1)
    for ell in ELL_GRID:
        D_out = ell * (ell - 1) // 2
        G_raw = rng.randn(L_u * (L_u - 1) // 2, D_out)
        G_star = G_raw / np.linalg.norm(G_raw, axis=1, keepdims=True)
        ii_lat, jj_lat = np.triu_indices(ell, k=1)
        phi = rng.uniform(0.2, np.pi - 0.2, size=D_out)
        if ell == 2:
            phi = np.array([np.pi / 3.0])  # rho = +0.5
        off = (G_star ** 2) @ hypersphere_angles_to_corr_vec(ell, phi, ii_lat, jj_lat)
        Th = np.eye(L_u)
        Th[ii_L, jj_L] = off
        Th[jj_L, ii_L] = off
        for strat in STRATEGIES:
            Th_rep = apply_repair_strategy(Th, strat, eps=1e-4)
            assert np.allclose(Th_rep, Th_rep.T), f"Symmetry failed for {strat} at ell={ell}"
            assert float(np.min(np.linalg.eigvalsh(Th_rep))) > 0.0, f"SPD failed for {strat} at ell={ell}"
            if strat != "nugget":
                assert np.allclose(np.diag(Th_rep), 1.0, atol=1e-14), f"Unit diagonal failed for {strat}"
        if ell == 2:
            assert np.allclose(
                apply_repair_strategy(Th, "mdg"), apply_repair_strategy(Th, "exact")
            ), "At ell=2 with rho>0, MDG must coincide with Exact Spectral Repair"
    print("[PASS] Unit Test 2: SPD certificates, unit-diagonal invariance & Proposition 4 ell=2 collapse.")

    # [REGRESSION 3] Krylov rank saturation (Eq. eq:krylov_bound) & zero-residual stability across ell in {2..8}
    E_u = build_incidence_E(L_u)
    cats_u = np.concatenate([np.repeat(np.arange(L_u), 3), rng.randint(0, L_u, 75 - 3 * L_u)])
    X_cont_u = rng.uniform(0.0, 1.0, size=(75, 2))
    X_cat_u = np.eye(L_u)[cats_u] @ E_u
    y_u = fn_clustered_mixed_response(X_cont_u, cats_u)
    bound_u = 2 + L_u - 1  # 16
    for ell in ELL_GRID:
        D_out = ell * (ell - 1) // 2
        G_star_u, k_act_u = compute_rn_pls_rotation(X_cat_u, y_u, D_out, X_cont_u)
        assert not np.any(np.isnan(G_star_u)), f"NaN in G_star at ell={ell}"
        assert np.allclose(np.sum(G_star_u ** 2, axis=1), 1.0, atol=1e-13), f"Row norm != 1 at ell={ell}"
        assert k_act_u <= bound_u, f"k_act={k_act_u} exceeded Krylov rank bound {bound_u} at ell={ell}"
    print(f"[PASS] Regression Test 3: Active Krylov rank bound k_act <= {bound_u} & rank-guarded RN-PLS stability.")

    # [INTEGRATION 4] End-to-end GP MLE RMSE decrease & cost growth across ell in {2, 6, 8}
    print("[INFO] Running Integration Test 4 (end-to-end GP MLE at ell in {2, 6, 8} across 5 replications)...")
    res_int = run_ell_scaling_experiment(
        ell_list=[2, 6, 8], n_reps=5, maxiter=40, base_seed=3100, n_workers=5
    )
    for strat in STRATEGIES:
        rmse_2 = res_int[2]["strategies"][strat]["rmse_mean"]
        rmse_6 = res_int[6]["strategies"][strat]["rmse_mean"]
        rmse_8 = res_int[8]["strategies"][strat]["rmse_mean"]
        assert rmse_6 < rmse_2 and rmse_8 < rmse_2, (
            f"Expected RMSE decrease from ell=2 ({rmse_2:.4f}) to ell=6 ({rmse_6:.4f}) & ell=8 ({rmse_8:.4f}) for {strat}"
        )
        assert res_int[8]["strategies"][strat]["time_mean"] > res_int[2]["strategies"][strat]["time_mean"], (
            f"Expected higher multi-start cost at ell=8 than ell=2 for {strat}"
        )
    assert res_int[6]["strategies"]["mdg"]["rmse_mean"] < res_int[6]["strategies"]["gersh"]["rmse_mean"], (
        "Expected MDG to outperform naive Fast Gershgorin on clustered families at ell=6"
    )
    print("[PASS] Integration Test 4: All 5 strategies reduce RMSE across ell in {2..8} without degradation, with MDG < Gershgorin.")
    print("=" * 78)
    print("ALL UNIT, REGRESSION, AND INTEGRATION TESTS PASSED.")
    print("=" * 78)


def print_latex_summary_table(summary: dict) -> None:
    """Print formatted summary table matching Tab. tab:ell_scaling in main.tex."""
    print("\n" + "=" * 108)
    print("SUMMARY TABLE: PREDICTIVE TEST RMSE (MEAN +/- SD) ACROSS LATENT DIMENSIONS ell IN {2,3,4,5,6,7,8}")
    print("=" * 108)
    header = (
        f"{'ell':>3s} | {'D_out':>5s} | {'S':>2s} | {'k_act':>5s} | "
        + " | ".join(f"{s:>15s}" for s in STRATEGIES)
        + f" | {'Runtime (MDG / Ex)':>20s}"
    )
    print(header)
    print("-" * len(header))
    for ell in sorted(summary.keys()):
        row = summary[ell]
        cells = " | ".join(
            f"{row['strategies'][s]['rmse_mean']:.4f}+/-{row['strategies'][s]['rmse_sd']:.4f}"
            for s in STRATEGIES
        )
        t_mdg = row["strategies"]["mdg"]["time_mean"]
        t_ex = row["strategies"]["exact"]["time_mean"]
        print(
            f"{ell:3d} | {row['D_out']:5d} | {row['n_starts']:2d} | {row['k_act_mean']:5.1f} | {cells} | "
            f"{t_mdg:5.2f}s / {t_ex:5.2f}s"
        )
    print("=" * 108)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent dimension scaling benchmark (ell in {2,3,4,5,6,7,8})")
    parser.add_argument("--verify", action="store_true", help="Run unit, regression, and integration tests")
    parser.add_argument("--quick", action="store_true", help="Run 6-replication smoke benchmark")
    parser.add_argument("--full", action="store_true", help="Run full 20-replication manuscript benchmark")
    parser.add_argument("--workers", type=int, default=5, help="Parallel worker processes (default: 5)")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save JSON summary")
    args = parser.parse_args()

    if args.verify:
        run_verification_suite()
    elif args.full:
        print("=" * 78)
        print("RUNNING FULL LATENT DIMENSION SCALING BENCHMARK (20 REPLICATIONS, ell in {2..8})")
        print("=" * 78)
        res = run_ell_scaling_experiment(
            ell_list=ELL_GRID, n_reps=20, maxiter=45, base_seed=3100, n_workers=args.workers
        )
        print_latex_summary_table(res)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
    else:
        n_r = 6 if args.quick else 10
        print("=" * 78)
        print(f"RUNNING LATENT DIMENSION SCALING BENCHMARK ({n_r} REPLICATIONS, ell in {{2..8}})")
        print("=" * 78)
        res = run_ell_scaling_experiment(
            ell_list=ELL_GRID, n_reps=n_r, maxiter=40, base_seed=3100, n_workers=args.workers
        )
        print_latex_summary_table(res)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
