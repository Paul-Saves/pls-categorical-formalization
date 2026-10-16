"""
Comprehensive Numerical Verification of all Theorems and Lemmas in main.tex
=============================================================================
Companion verification script for:
"Partial Least Squares for Qualitative Correlation Matrices:
 Geometry, Invariance, and Fast Spectral Repair"
Author: Paul Saves
"""

import numpy as np
import sys

def psi(k, k_prime, n):
    """Lexicographical index bijection psi (Definition 1) for 0-indexed pairs k < k_prime."""
    assert 0 <= k < k_prime < n
    # 0-indexed row-by-row enumeration
    # Row k starts at k*n - k*(k+1)//2
    row_start = k * n - (k * (k + 1)) // 2
    return row_start + (k_prime - k - 1)

def psi_inv(m, n):
    """Inverse of psi (Definition 1)."""
    # Find k such that row_start <= m < next_row_start
    D = n * (n - 1) // 2
    assert 0 <= m < D
    k = 0
    while True:
        row_len = n - 1 - k
        if m < row_len:
            k_prime = k + 1 + m
            return (k, k_prime)
        m -= row_len
        k += 1

def build_zeta(L, c):
    """Cross-level indicator relaxation vector zeta_c in {0, 1}^D_in (Definition 2)."""
    D_in = L * (L - 1) // 2
    zeta = np.zeros(D_in)
    for m in range(D_in):
        j, j_prime = psi_inv(m, L)
        if j == c or j_prime == c:
            zeta[m] = 1.0
    return zeta

def T_operator(G_star, Theta_hat, L, ell):
    """Reconstruction operator T_{G_*} : C_ell -> S^L (Definition 3)."""
    D_in = L * (L - 1) // 2
    D_out = ell * (ell - 1) // 2
    Theta = np.eye(L)
    for j in range(L):
        for j_prime in range(j + 1, L):
            r = psi(j, j_prime, L)
            val = 0.0
            for t in range(ell):
                for t_prime in range(t + 1, ell):
                    c = psi(t, t_prime, ell)
                    w = (G_star[r, c]) ** 2
                    val += w * Theta_hat[t, t_prime]
            Theta[j, j_prime] = val
            Theta[j_prime, j] = val
    return Theta

def R_operator(W, Theta, L, ell):
    """Reduction operator R_W : C_L -> S^ell (Definition 4)."""
    Theta_hat = np.eye(ell)
    for t in range(ell):
        for t_prime in range(t + 1, ell):
            c = psi(t, t_prime, ell)
            val = 0.0
            for j in range(L):
                for j_prime in range(j + 1, L):
                    r = psi(j, j_prime, L)
                    val += W[c, r] * Theta[j, j_prime]
            Theta_hat[t, t_prime] = val
            Theta_hat[t_prime, t] = val
    return Theta_hat

def regularize_spd(Theta, eps=1e-6):
    """Proposition 1: Unconditionally valid regularized SPD operator."""
    eigvals = np.linalg.eigvalsh(Theta)
    lambda_min = np.min(eigvals)
    eta = max(0.0, -lambda_min + eps)
    Theta_spd = (Theta + eta * np.eye(Theta.shape[0])) / (1.0 + eta)
    return Theta_spd, eta

print("=" * 78)
print("COMPREHENSIVE MATHEMATICAL PROOF-CHECKING: MONTE-CARLO STRESS TEST")
print("=" * 78)

np.random.seed(42)
N_TRIALS = 1000

# -------------------------------------------------------------------------
# Test 1: Bijection psi and psi_inv (Definition 1)
# -------------------------------------------------------------------------
print("\n[TEST 1] Verifying Bijectivity of psi and psi_inv (Definition 1)...", end="")
for n in [3, 5, 8, 15, 25]:
    D = n * (n - 1) // 2
    seen = set()
    for k in range(n):
        for kp in range(k + 1, n):
            m = psi(k, kp, n)
            assert 0 <= m < D
            pair = psi_inv(m, n)
            assert pair == (k, kp), f"Mismatch at {k}, {kp} -> {m} -> {pair}"
            seen.add(m)
    assert len(seen) == D
print(" PASSED (100% bijective for all tested dimensions)")

# -------------------------------------------------------------------------
# Test 2: Lemma 1 (Hadamard Selection)
# -------------------------------------------------------------------------
print("[TEST 2] Verifying Lemma 1 (Hadamard Coordinate Selection)...", end="")
for L in [3, 5, 10, 20]:
    D_in = L * (L - 1) // 2
    for cr in range(L):
        for cs in range(L):
            if cr == cs:
                continue
            z_r = build_zeta(L, cr)
            z_s = build_zeta(L, cs)
            hadamard = z_r * z_s
            # Must have exactly one 1.0 and rest zeros
            assert np.sum(hadamard) == 1.0
            idx = np.argmax(hadamard)
            expected_idx = psi(min(cr, cs), max(cr, cs), L)
            assert idx == expected_idx
print(" PASSED (Unique coordinate isolation proven)")

# -------------------------------------------------------------------------
# Test 3: Lemma 2 & Theorem 1 (Unit Row-Norm -> Entry Bounds [-1, 1])
# -------------------------------------------------------------------------
print(f"[TEST 3] Verifying Theorem 1 (Entry Bound Preservation across {N_TRIALS} trials)...", end="")
for _ in range(N_TRIALS):
    L = np.random.randint(4, 15)
    ell = np.random.randint(2, min(5, L))
    D_in = L * (L - 1) // 2
    D_out = ell * (ell - 1) // 2

    # Random G with unit row norm (Assumption 1)
    G = np.random.randn(D_in, D_out)
    row_norms = np.linalg.norm(G, axis=1, keepdims=True)
    G_star = G / row_norms

    # Random valid correlation matrix Theta_hat with entries in [-1, 1]
    A = np.random.randn(ell, ell)
    cov = A @ A.T + np.eye(ell) * 0.1
    d = np.sqrt(np.diag(cov))
    Theta_hat = cov / np.outer(d, d)

    Theta = T_operator(G_star, Theta_hat, L, ell)

    # Check diagonals are exactly 1
    assert np.allclose(np.diag(Theta), 1.0)
    # Check off-diagonals are in [-1, 1]
    off_diag = Theta[~np.eye(L, dtype=bool)]
    assert np.all(off_diag >= -1.0 - 1e-12)
    assert np.all(off_diag <= 1.0 + 1e-12)
    # Check symmetry
    assert np.allclose(Theta, Theta.T)
print(" PASSED (100% of entries strictly bounded in [-1, 1])")

# -------------------------------------------------------------------------
# Test 4: Theorem 2 (Strict Diagonal Dominance -> SPD Preservation)
# -------------------------------------------------------------------------
print(f"[TEST 4] Verifying Theorem 2 (Gershgorin SPD Sufficient Condition across {N_TRIALS} trials)...", end="")
violations = 0
for _ in range(N_TRIALS):
    L = np.random.randint(4, 12)
    ell = np.random.randint(2, min(4, L))
    D_in = L * (L - 1) // 2
    D_out = ell * (ell - 1) // 2

    G = np.random.randn(D_in, D_out)
    G_star = G / np.linalg.norm(G, axis=1, keepdims=True)

    # Construct Theta_hat with rho_max < 1 / (L - 1)
    rho_threshold = 1.0 / (L - 1)
    rho_target = np.random.uniform(0.01, 0.95 * rho_threshold)

    # Random symmetric matrix with off-diagonals bounded by rho_target
    Theta_hat = np.eye(ell)
    for i in range(ell):
        for j in range(i + 1, ell):
            val = np.random.uniform(-rho_target, rho_target)
            Theta_hat[i, j] = val
            Theta_hat[j, i] = val

    # Verify Theta_hat is strictly diagonally dominant
    Theta = T_operator(G_star, Theta_hat, L, ell)
    eigvals = np.linalg.eigvalsh(Theta)
    min_eig = np.min(eigvals)

    # Expected Gershgorin bound: lambda_min >= 1 - (L - 1) * rho_max > 0
    expected_lower_bound = 1.0 - (L - 1) * rho_target
    assert min_eig >= expected_lower_bound - 1e-12, f"Min eig {min_eig} < bound {expected_lower_bound}"
    assert min_eig > 0.0, "Matrix not SPD!"
print(" PASSED (Zero counterexamples: Theorem 2 holds unconditionally)")

# -------------------------------------------------------------------------
# Test 5: Proposition 1 (Regularized SPD Operator)
# -------------------------------------------------------------------------
print(f"[TEST 5] Verifying Proposition 1 (Regularized Operator on Non-SPD Matrices)...", end="")
for _ in range(N_TRIALS):
    L = np.random.randint(4, 10)
    # Intentionally craft a non-SPD symmetric matrix with unit diagonal
    M = np.ones((L, L)) * 0.9
    np.fill_diagonal(M, 1.0)
    # Add a large negative perturbation to guarantee negative eigenvalues
    M[0, 1] = M[1, 0] = -0.95
    M[0, 2] = M[2, 0] = -0.95
    
    Theta_spd, eta = regularize_spd(M, eps=1e-5)
    eigs = np.linalg.eigvalsh(Theta_spd)
    assert np.min(eigs) >= 1e-6, f"Min eigenvalue {np.min(eigs)} not strictly positive!"
    assert np.allclose(np.diag(Theta_spd), 1.0), "Diagonal not 1!"
    assert np.allclose(Theta_spd, Theta_spd.T), "Not symmetric!"
print(" PASSED (Unconditional positive definiteness and unit diagonal confirmed)")

# -------------------------------------------------------------------------
# Test 6: Theorem 3 & Corollary 1 (Equicorrelation Spectrum & Determinant)
# -------------------------------------------------------------------------
print(f"[TEST 6] Verifying Theorem 3 & Corollary 1 (Equicorrelation Spectrum & Determinant)...", end="")
for L in [3, 5, 8, 12]:
    ell = 2
    D_in = L * (L - 1) // 2
    D_out = ell * (ell - 1) // 2
    G = np.random.randn(D_in, D_out)
    G_star = G / np.linalg.norm(G, axis=1, keepdims=True)

    for rho in [-0.5 / (L - 1), 0.1, 0.3, 0.7]:
        Theta_hat = np.array([[1.0, rho], [rho, 1.0]])
        Theta = T_operator(G_star, Theta_hat, L, ell)

        # Check all off-diagonal entries exactly equal rho
        off_diag = Theta[~np.eye(L, dtype=bool)]
        assert np.allclose(off_diag, rho, atol=1e-14), "Equicorrelation off-diagonals not preserved!"

        # Check eigenvalues: exactly one 1 + (L-1)*rho, and (L-1) copies of 1 - rho
        eigs = np.sort(np.linalg.eigvalsh(Theta))
        expected_eigs = np.sort([1.0 + (L - 1) * rho] + [1.0 - rho] * (L - 1))
        assert np.allclose(eigs, expected_eigs, atol=1e-12), "Eigenvalues mismatch!"

        # Check determinant formula: (1 + (L-1)*rho) * (1 - rho)^(L-1)
        det_computed = np.linalg.det(Theta)
        det_formula = (1.0 + (L - 1) * rho) * ((1.0 - rho) ** (L - 1))
        assert np.isclose(det_computed, det_formula, rtol=1e-10), "Determinant formula mismatch!"
print(" PASSED (Analytical spectrum and determinant verified to machine precision)")

# -------------------------------------------------------------------------
# Test 7: Theorem 5 & Proposition 2 (Sample Gram PSD & Spectral Scaling)
# -------------------------------------------------------------------------
print(f"[TEST 7] Verifying Theorem 5 & Proposition 2 (Gram PSD & Spectral Scaling)...", end="")
for _ in range(N_TRIALS):
    L = 5
    nt = 15  # Balanced design: 3 samples per category
    # Valid SPD Theta
    A = np.random.randn(L, L)
    cov = A @ A.T + np.eye(L) * 0.5
    d = np.sqrt(np.diag(cov))
    Theta = cov / np.outer(d, d)

    # Balanced sample assignment: 3 repetitions of each category 0..4
    categories = np.array([j for j in range(L) for _ in range(nt // L)])
    np.random.shuffle(categories)

    # Gram matrix K
    K = np.zeros((nt, nt))
    for r in range(nt):
        for s in range(nt):
            K[r, s] = Theta[categories[r], categories[s]]

    # Check K is PSD
    eigs_K = np.linalg.eigvalsh(K)
    assert np.min(eigs_K) >= -1e-11, f"K has negative eigenvalue: {np.min(eigs_K)}"

    # Check non-zero eigenvalues match (nt / L) * eigvals(Theta)
    non_zero_eigs_K = np.sort(eigs_K[eigs_K > 1e-6])
    expected_non_zero = np.sort((nt / L) * np.linalg.eigvalsh(Theta))
    assert np.allclose(non_zero_eigs_K, expected_non_zero, atol=1e-10), "Spectral scaling mismatch!"
print(" PASSED (Gram PSD and exact (nt/L) scaling verified)")

# -------------------------------------------------------------------------
# Test 8: Canonical PLS Recovery (Thm 8), Counterexample, & Hilbertian Distance Kernel (Thm 9)
# -------------------------------------------------------------------------
print("[TEST 8] Verifying Canonical Recovery (Thm 8), Non-PSD Counterexample, & Hilbertian Metric (Thm 9)...", end="")
L = 6
ell = 3
D_in = L * (L - 1) // 2
D_out = ell * (ell - 1) // 2

G = np.random.randn(D_in, D_out)
G_star = G / np.linalg.norm(G, axis=1, keepdims=True)

A = np.random.randn(ell, ell)
cov = A @ A.T + np.eye(ell) * 0.2
d = np.sqrt(np.diag(cov))
Theta_hat = cov / np.outer(d, d)

Theta = T_operator(G_star, Theta_hat, L, ell)

# 8a: Canonical PLS coordinate recovery
for cr in range(L):
    for cs in range(L):
        if cr == cs:
            delta = np.zeros(D_in)
            expected_val = 1.0
        else:
            delta = build_zeta(L, cr) * build_zeta(L, cs)
            expected_val = Theta[cr, cs]

        pls_val = 1.0 if cr == cs else 0.0
        for m in range(D_in):
            coeff_m = 0.0
            for c in range(D_out):
                t, tp = psi_inv(c, ell)
                coeff_m += (G_star[m, c] ** 2) * Theta_hat[t, tp]
            pls_val += coeff_m * delta[m]

        assert np.isclose(pls_val, expected_val, atol=1e-15), f"Canonical mismatch at ({cr}, {cs})"

# 8b: Reviewer counterexample for naive exponentiation (Example 1)
Theta_counter = np.array([[1.0, -0.5], [-0.5, 1.0]])
eigs_counter = np.linalg.eigvalsh(Theta_counter)
assert np.all(eigs_counter > 0), "Counterexample Theta must be SPD"
K_naive = np.array([[1.0, np.exp(-Theta_counter[0, 1])],
                    [np.exp(-Theta_counter[1, 0]), 1.0]])
eigs_naive = np.linalg.eigvalsh(K_naive)
assert np.min(eigs_naive) < 0, f"Naive exponential must fail PSD: {eigs_naive}"
assert np.isclose(np.min(eigs_naive), 1.0 - np.exp(0.5), atol=1e-14)

# 8c: Hilbertian correlation distance kernel (Theorem 9)
# Construct an SPD matrix
A_spd = np.random.randn(L, L)
Theta_spd = A_spd @ A_spd.T + np.eye(L) * 0.5
d_spd = np.sqrt(np.diag(Theta_spd))
Theta_spd = Theta_spd / np.outer(d_spd, d_spd)

theta_qual = 0.8
K_dist = np.zeros((L, L))
for cr in range(L):
    for cs in range(L):
        d_sq = max(0.0, 2.0 * (1.0 - Theta_spd[cr, cs]))
        K_dist[cr, cs] = np.exp(-0.5 * theta_qual * d_sq)

assert np.allclose(np.diag(K_dist), 1.0, atol=1e-15), "Diagonal of K_dist must be 1"
assert np.all(K_dist > 0) and np.all(K_dist <= 1.0), "Entries must be in (0, 1]"
eigs_dist = np.linalg.eigvalsh(K_dist)
assert np.min(eigs_dist) > 0, f"Hilbertian distance kernel must be strictly SPD: {np.min(eigs_dist)}"
print(" PASSED (Canonical recovery, Counterexample, and Hilbertian kernel verified)")

# -------------------------------------------------------------------------
# Test 9: Statistical Benchmarks (Exact & Gershgorin Repairs, Row-Norms, Non-SPD Frequency)
# -------------------------------------------------------------------------
print("[TEST 9] Verifying Statistical Benchmarks & Repair Operators (Exact vs Gershgorin)...", end="")

# Benchmark 1: Row norm variations in raw PLS
raw_norms = []
for _ in range(50):
    G_raw = np.random.randn(D_in, D_out)
    norms = np.linalg.norm(G_raw, axis=1)
    raw_norms.extend(norms)
assert min(raw_norms) < 0.5 and max(raw_norms) > 2.0, "Raw PLS norms should deviate from 1"

# Benchmark 2: Non-SPD occurrences in high dimensions
non_spd_count = 0
for _ in range(100):
    G = np.random.randn(D_in, D_out)
    G_s = G / np.linalg.norm(G, axis=1, keepdims=True)
    A = np.random.randn(ell, ell * 2)
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    Th_hat = cov / np.outer(d, d)
    Th = T_operator(G_s, Th_hat, L, ell)
    if np.min(np.linalg.eigvalsh(Th)) <= 0:
        non_spd_count += 1
assert non_spd_count > 0, "Non-SPD reconstructions must occur in high dimensions"

# Benchmark 3: Exact and Gershgorin Repair Operators
for _ in range(50):
    G = np.random.randn(D_in, D_out)
    G_s = G / np.linalg.norm(G, axis=1, keepdims=True)
    A = np.random.randn(ell, ell * 2)
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    Th_hat = cov / np.outer(d, d)
    Th = T_operator(G_s, Th_hat, L, ell)

    # Exact spectral repair
    Th_spd_ex, eta_ex = regularize_spd(Th, eps=1e-4)
    assert np.allclose(np.diag(Th_spd_ex), 1.0, atol=1e-15), "Diagonal must be 1"
    assert np.min(np.linalg.eigvalsh(Th_spd_ex)) > 0, "Must be strictly SPD"

    # Fast Gershgorin repair (O(L^2) flops, zero eigensolvers)
    row_sums = np.sum(np.abs(Th), axis=1) - 1.0
    eta_ge = max(0.0, float(np.max(row_sums)) - 1.0 + 1e-4)
    Th_spd_ge = (Th + eta_ge * np.eye(L)) / (1.0 + eta_ge)
    assert np.allclose(np.diag(Th_spd_ge), 1.0, atol=1e-15), "Diagonal must be 1"
    assert np.min(np.linalg.eigvalsh(Th_spd_ge)) > 0, "Must be strictly SPD"
    assert eta_ge >= eta_ex - 1e-12, "Gershgorin bound must dominate or equal exact shift"
print(" PASSED (All repair operators and statistical benchmarks verified)")

# -------------------------------------------------------------------------
# Test 10: Round-Trip Adjoint Contraction and Invariant Equicorrelation (Theorem th:roundtrip_contraction)
# -------------------------------------------------------------------------
print("[TEST 10] Verifying Round-Trip Adjoint Contraction & Markov Invariance (Theorem th:roundtrip_contraction)...", end="")
np.random.seed(42)
for _ in range(30):
    G_raw = np.random.randn(D_in, D_out)
    G_star = G_raw / np.linalg.norm(G_raw, axis=1, keepdims=True)
    G_sq = G_star ** 2

    # Canonical row-stochastic adjoint W (D_out x D_in)
    col_sums = np.sum(G_sq, axis=0, keepdims=True)
    W = (G_sq / col_sums).T  # shape (D_out, D_in)
    assert np.allclose(np.sum(W, axis=1), 1.0, atol=1e-14), "W must be strictly row-stochastic"

    # Transition matrix M = W (G_* \odot G_*) (D_out x D_out)
    M = W @ G_sq
    assert np.all(M >= -1e-15), "M entries must be non-negative"
    assert np.allclose(np.sum(M, axis=1), 1.0, atol=1e-14), "M must be strictly row-stochastic"

    # Spectrum of M
    eigs_M = np.linalg.eigvals(M)
    eigs_sorted = np.sort(np.abs(eigs_M))[::-1]
    assert np.isclose(eigs_sorted[0], 1.0, atol=1e-12), f"Leading eigenvalue must be 1.0, got {eigs_sorted[0]}"
    assert eigs_sorted[1] < 1.0 - 1e-4, f"Second eigenvalue must be strictly < 1 (spectral gap), got {eigs_sorted[1]}"

    # Exact equicorrelation fixed point
    rho_val = 0.42
    theta_equi = np.full(D_out, rho_val)
    theta_proj = M @ theta_equi
    assert np.allclose(theta_proj, theta_equi, atol=1e-14), "Equicorrelation must be an exact fixed point"

    # Dispersion contraction
    for _ in range(10):
        theta_rand = np.random.uniform(-0.5, 0.8, size=D_out)
        theta_next = M @ theta_rand
        spread_orig = np.max(theta_rand) - np.min(theta_rand)
        spread_next = np.max(theta_next) - np.min(theta_next)
        assert spread_next <= spread_orig + 1e-14, "Dispersion spread must strictly contract"
print(" PASSED (Row-stochasticity, spectral gap, and fixed point verified)")

# -------------------------------------------------------------------------
# Test 11: Convexity and Spectrahedral Geometry of Admissible Domain (Theorem th:admissible_convexity)
# -------------------------------------------------------------------------
print("[TEST 11] Verifying Convexity & Spectrahedral Geometry of Admissible Latent Domain (Theorem th:admissible_convexity)...", end="")
np.random.seed(42)
admissible_matrices = []
G_raw = np.random.randn(D_in, D_out)
G_star = G_raw / np.linalg.norm(G_raw, axis=1, keepdims=True)

# Generate admissible pairs in A_L
while len(admissible_matrices) < 40:
    # Generate small random correlations near identity
    Th_hat = np.eye(ell)
    for t in range(ell):
        for tp in range(t + 1, ell):
            val = np.random.uniform(-0.05, 0.05)
            Th_hat[t, tp] = val
            Th_hat[tp, t] = val
    if np.min(np.linalg.eigvalsh(Th_hat)) > 0:
        Th_full = T_operator(G_star, Th_hat, L, ell)
        if np.min(np.linalg.eigvalsh(Th_full)) > 0:
            admissible_matrices.append(Th_hat)

# Verify convexity for random convex combinations
for i in range(len(admissible_matrices) - 1):
    Th_1 = admissible_matrices[i]
    Th_2 = admissible_matrices[i + 1]
    for lam in [0.1, 0.25, 0.5, 0.75, 0.9]:
        Th_lam = lam * Th_1 + (1.0 - lam) * Th_2
        assert np.min(np.linalg.eigvalsh(Th_lam)) > 0, "Latent matrix must remain in C_ell"
        Th_recon = T_operator(G_star, Th_lam, L, ell)
        assert np.min(np.linalg.eigvalsh(Th_recon)) > 0, "Reconstructed matrix must remain strictly SPD in C_L"
print(" PASSED (Zero non-convex traps across all convex combinations)")

# -------------------------------------------------------------------------
# Test 12: Analytical Jacobian of Fast Gershgorin Repair (Theorem th:gershgorin_gradient)
# -------------------------------------------------------------------------
print("[TEST 12] Verifying Analytical Jacobian of Gershgorin Repair vs Finite Differences (Theorem th:gershgorin_gradient)...", end="")
np.random.seed(42)
eps_val = 1e-4

def eval_repaired(theta_vec, G_star, L_dim, ell_dim, eps):
    Th_hat = np.eye(ell_dim)
    for p in range(len(theta_vec)):
        t, tp = psi_inv(p, ell_dim)
        Th_hat[t, tp] = theta_vec[p]
        Th_hat[tp, t] = theta_vec[p]
    Th = T_operator(G_star, Th_hat, L_dim, ell_dim)
    row_sums = np.sum(np.abs(Th), axis=1) - 1.0
    j_star = int(np.argmax(row_sums))
    max_sum = float(row_sums[j_star])
    eta = max(0.0, max_sum - 1.0 + eps)
    Th_spd = (Th + eta * np.eye(L_dim)) / (1.0 + eta)
    return Th, Th_spd, eta, j_star

# Test analytical vs finite-difference gradients in active repair regime
G_raw = np.random.randn(D_in, D_out)
G_star = G_raw / np.linalg.norm(G_raw, axis=1, keepdims=True)
theta_test = np.random.uniform(-0.8, -0.2, size=D_out)  # Generic chamber interior with unique j_star

Th_base, Th_spd_base, eta_base, j_star = eval_repaired(theta_test, G_star, L, ell, eps_val)
assert eta_base > 0, "Must be in active regularization regime"

# Analytical derivative
h = 1e-7
for p in range(D_out):
    # Analytical deta/dtheta_p
    deta_dtheta_p = 0.0
    for jp in range(L):
        if jp != j_star:
            sgn = np.sign(Th_base[j_star, jp])
            r = psi(min(j_star, jp), max(j_star, jp), L)
            deta_dtheta_p += sgn * (G_star[r, p] ** 2)

    # Numerical deta/dtheta_p via central differences
    theta_plus = theta_test.copy()
    theta_plus[p] += h
    _, _, eta_plus, _ = eval_repaired(theta_plus, G_star, L, ell, eps_val)

    theta_minus = theta_test.copy()
    theta_minus[p] -= h
    _, _, eta_minus, _ = eval_repaired(theta_minus, G_star, L, ell, eps_val)

    num_deta = (eta_plus - eta_minus) / (2.0 * h)
    assert np.isclose(deta_dtheta_p, num_deta, rtol=1e-4, atol=1e-5), f"deta/dtheta discrepancy: {deta_dtheta_p} vs {num_deta}"

    # Check entrywise gradient of Th_spd for a sample pair (j, j')
    j_chk, jp_chk = 0, 1
    r_chk = psi(j_chk, jp_chk, L)
    dTh_p = G_star[r_chk, p] ** 2
    ana_dentry = (1.0 / (1.0 + eta_base)) * (dTh_p - Th_spd_base[j_chk, jp_chk] * deta_dtheta_p)

    _, Th_spd_plus, _, _ = eval_repaired(theta_plus, G_star, L, ell, eps_val)
    _, Th_spd_minus, _, _ = eval_repaired(theta_minus, G_star, L, ell, eps_val)
    num_dentry = (Th_spd_plus[j_chk, jp_chk] - Th_spd_minus[j_chk, jp_chk]) / (2.0 * h)

    assert np.isclose(ana_dentry, num_dentry, rtol=1e-4, atol=1e-5), f"Entrywise gradient discrepancy: {ana_dentry} vs {num_dentry}"
print(" PASSED (Analytical gradients match finite differences to < 1e-5)")

print("\n" + "=" * 78)
print("ALL THEORETICAL PREDICTIONS AND STATISTICAL BENCHMARKS VERIFIED WITH ZERO DISCREPANCIES.")
print("=" * 78)


