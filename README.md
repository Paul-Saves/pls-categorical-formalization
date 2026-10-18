# Partial Least Squares for Qualitative Correlation Matrices: Geometry, Invariance, and Fast Spectral Repair

[![Lean 4](https://img.shields.io/badge/Lean_4-Formalized-blue.svg)](https://leanprover.github.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion repository and open-source reproducibility package for the manuscript:

> **Partial Least Squares for Qualitative Correlation Matrices: Geometry, Invariance, and Fast Spectral Repair**  
> **Author:** Paul Saves  
> *Targeted for submission to Technometrics / Statistics and Computing*

---

## 🔬 Overview

When modeling computer experiments with qualitative (categorical) factors using Gaussian processes (Kriging), supervised dimension reduction via Partial Least Squares (PLS) is a standard technique for projecting high-dimensional categorical spaces onto low-dimensional latent spaces. 

However, unconstrained PLS reconstruction frequently violates positive definiteness, producing non-SPD correlation matrices that trigger Cholesky factorization breakdowns during Maximum Likelihood Estimation (MLE). This repository provides complete mathematical, formal, and numerical foundations to resolve this challenge:

1. **Mathematical Guarantees & Row Normalization (RN-PLS):** A rigorous geometric characterization of the Row-Normalized PLS architecture. Enforcing unit row norms (sum_c [G_*]_{rc}^2 = 1) unconditionally preserves the correlation hypercube [-1, 1]^(L x L). We establish the complete graph incidence Gram identity E E^T = (L-2)I_L + 1_L 1_L^T, prove the exact Krylov rank bound rank(K_ell) <= 2 under balanced pure designs, and prove the 1-parameter equicorrelation collapse at ell = 2.
2. **Markov Contraction & Invariant Mean:** Round-trip adjoint PLS reduction R_W and reconstruction T_{G_*} form a reversible Markov transition operator M = W (G_* (.) G_*) satisfying detailed balance D_col M = M^T D_col = A^T A, strictly preserving the full-space mean correlation theta_full.
3. **Fast Certified O(L^2) Spectral Repair Operators:**
   - **Fast Gershgorin Repair:** A closed-form diagonal shift-and-scaling operator Theta_SPD = (Theta + eta I_L) / (1 + eta) that certifies strict positive definiteness and preserves exact unit diagonals in O(L^2) flops without eigenvalue decompositions.
   - **Mean-Deflated Gershgorin (MDG):** By deflating the mean positive correlation rho_+, MDG eliminates the conservative over-shrinkage cap of naive Gershgorin repair on clustered topologies, matching exact spectral projection while executing in O(L^2) flops.
4. **Universal Schoenberg--Bernstein Distance Kernels:** Grounded in conditionally negative definite metric geometry, we prove that the chordal distance d_Theta^2(c, c') = 2(1 - [Theta]_{c, c'}) defines an isometric Hilbertian embedding, establishing universal strict positive definiteness for categorical Matern, Gaussian, and Rational Quadratic kernels.
5. **Analytical Jacobians & Clarke Generalized Subdifferentials:** Closed-form directional gradients and non-smooth polyhedral chamber characterizations for fast quasi-Newton (L-BFGS-B) optimization. Adjoint reduction warm-starts Theta_0 = R_W(Theta_emp) reduce MLE iterations by 3x.
6. **Machine-Certified Formal Proofs in Lean 4 & Aristotle ATP:** Formalization in Lean 4 (`FormalProofs.lean`) and a 29-goal automated theorem proving bundle (`AristotleExport.lean`, `aristotle_goals.json`), with 17 goals fully proven and 12 formally typed specifications.
7. **Aerospace Engineering Application:** Structural mechanics surrogate modeling of a mixed continuous--categorical cantilever beam across 12 real aerospace alloys and composites.

---

## 📁 Repository Structure

```text
├── main.tex                       # Complete LaTeX manuscript (Technometrics template)
├── sample.bib                     # Curated BibTeX bibliography
├── sn-jnl.cls                     # Springer Nature journal class
├── requirements.txt               # Minimal Python dependencies
│
├── generate_all_tables.py         # Master reproduction script for Tables 2, 3, 4, 5, C2
├── generate_all_figures.py        # Master reproduction script for Figures 1, 2, 3, 4
├── benchmark_gp_experiments.py    # Multi-start L-BFGS-B GP emulation benchmark
├── verify_theorems.py             # 12-suite Monte Carlo theorem verification harness
│
├── FormalProofs.lean              # Lean 4 formalization (definitions, lemmas, theorems)
├── AristotleExport.lean           # Standalone Aristotle ATP proof goal specifications
├── aristotle_verifier.py          # Formal verification orchestrator & JSON bundle generator
├── aristotle_goals.json           # Machine-readable Aristotle ATP goals (29 goals)
├── aristotle_bundle.json          # Complete Lean 4 Mathlib formalization export
│
└── figures/                       # Publication-quality vector graphics (PDF & PNG)
    ├── fig1_phase_diagram.pdf     # Admissible domain A_L slices & cluster Frobenius distortion
    ├── fig2_repair_tradeoff.pdf   # Profile likelihood landscape, chambers, & L-BFGS-B convergence
    ├── fig3_benchmark_boxplots.pdf# Budget scaling (nt/L) & condition number stability boxplots
    └── fig4_runtime_scaling.pdf   # Pareto MLE runtime scaling (L <= 80) & Cantilever correlation heatmap
```

---

## 🚀 Quick Start & Reproducibility

### 1. Installation

Clone this repository and install the minimal Python dependencies:

```bash
git clone https://github.com/Paul-Saves/pls-categorical-formalization.git
cd pls-categorical-formalization
pip install -r requirements.txt
```

### 2. Reproduce All Manuscript Tables

To execute a rapid validation (~20-30 seconds):
```bash
python generate_all_tables.py --quick
```

To reproduce the exact full Monte Carlo counts reported in the manuscript (1,000 trials for Tables 2 & C2; 200 instances for Table 3; 50 replications x 5 multi-starts for Table 4):
```bash
python generate_all_tables.py --full
```

You can also run individual table reproductions:
```bash
python generate_all_tables.py --table 4    # Table 2: Frequency & severity of non-SPD reconstructions
python generate_all_tables.py --table 5    # Table 3: Condition numbers across 200 non-SPD instances
python generate_all_tables.py --table 6    # Table 4: Multi-function GP emulation benchmark (f1, f2, f3)
python generate_all_tables.py --table 3    # Table C2: Empirical distribution of unconstrained PLS row norms
```

### 3. Reproduce All Vector Figures

Generate all 4 publication vector figures in `figures/` (both PDF and 300 DPI PNG):
```bash
python generate_all_figures.py
```

### 4. Mathematical Proof-Checking & Monte Carlo Stress Testing

Run the companion verification suite to validate all 12 mathematical test suites (spectral gaps, Markov reversibility, equicorrelation spectra, MDG zero-shrinkage, Hilbertian chordal embeddings, and Hellmann--Feynman gradients):
```bash
python verify_theorems.py
```

### 5. Formal Verification in Lean 4 & Aristotle ATP Bundle

Inspect formal declarations and verify the Aristotle proof bundle using:
```bash
python aristotle_verifier.py
```

To build the Lean 4 proof environment directly (requires `elan` and `lake` with Mathlib v4.14.0):
```bash
lake build
```

---

## 🧩 Aristotle Formal Verification Matrix (17/29 Proven, 58.6% Machine-Certified)

The mapping between the manuscript theorems in `main.tex` and the Lean 4 proof goals in `AristotleExport.lean` is summarized below:

| Goal | Target ID | Manuscript Reference | Mathematical Statement | Formal Status |
|:---:|:---|:---|:---|:---:|
| **01** | `LEMMA_CONVEX_COMB_CONST` | Section 3.1, Foundation (Lemma) | Convex combination of identical constant $\rho$ equals $\rho$ | :white_check_mark: **PROVEN** |
| **02** | `LEMMA_CONVEX_COMB_ABS_LE` | Section 3.1, Foundation (Lemma) | Convex combination preserves $[-B, B]$ interval | :white_check_mark: **PROVEN** |
| **03** | `LEMMA_3_CONVEX_COMBINATION_WEIGHTS` | Section 3.1, Lemma 3 (`lem:convex`) | Squared row weights $w_r^{(t,t')}$ are non-negative and sum to 1 | :white_check_mark: **PROVEN** |
| **04** | `LEMMA_3_ENTRY_BOUND_PRESERVATION` | Section 3.2, Lemma 3 (`lem:entry_bound`) | Reconstructed matrix unconditionally preserves correlation hypercube $[-1, 1]$ | :white_check_mark: **PROVEN** |
| **05** | `PROP_4_L2_TRIVIAL_WEIGHT` | Section 3.1, Proposition 4 (`prop:l2_degeneracy`) | Latent dimension $\ell=2$ yields single normalized weight $([G_*]_{r,1})^2 \equiv 1$ | :white_check_mark: **PROVEN** |
| **06** | `PROP_4_L2_RN_PLS_EQUICORRELATION_COLLAPSE` | Section 3.1, Proposition 4 (`prop:l2_degeneracy`) | Collapse of $\ell=2$ RN-PLS to 1-parameter equicorrelation via $([G_*]_{r,1})^2 \equiv 1$ | :white_check_mark: **PROVEN** |
| **07** | `PROP_2_INCIDENCE_GRAM_HOMOTHETY` | Section 2.3, Proposition 2 (`prop:rank_bound`) | Incidence Gram homothety $E E^\top u = (L-2)u$ on centered subspace $\mathds{1}_L^\perp$ | :microscope: *FORMALIZED SPEC* |
| **08** | `PROP_2_INCIDENCE_ALL_ONES_EIGENVALUE` | Section 2.3, Proposition 2 (`prop:rank_bound`) | Complete graph incidence Gram maximal eigenvalue $\lambda_1 = 2(L-1)$ | :microscope: *FORMALIZED SPEC* |
| **09** | `THM_5_ROUNDTRIP_ADJOINT_MARKOV_CONTRACTION` | Section 3.2, Theorem 5 (`th:roundtrip_contraction`) | Round-trip transition matrix $M = W (G_* \odot G_*)$ is row-stochastic with $\rho(M)=1$ | :white_check_mark: **PROVEN** |
| **10** | `THM_5_MARKOV_DETAILED_BALANCE` | Section 3.2, Theorem 5 (`th:roundtrip_contraction`) | Detailed balance $D_{\mathrm{col}} M = M^\top D_{\mathrm{col}} = A^\top A$ and reversible symmetry | :microscope: *FORMALIZED SPEC* |
| **11** | `THM_5_MARKOV_STATIONARY_DISTRIBUTION` | Section 3.2, Theorem 5 (`th:roundtrip_contraction`) | Column degrees vector $\bm{d}$ forms exact stationary distribution $\bm{d}^\top M = \bm{d}^\top$ | :microscope: *FORMALIZED SPEC* |
| **12** | `THM_5_MARKOV_INVARIANT_MEAN_CORRELATION` | Section 3.2, Theorem 5 (`th:roundtrip_contraction`) | Reconstructed full-space mean correlation $\bar{\theta}_{\mathrm{full}} = \bm{\pi}^\top \hat{\bm{\theta}}$ is an exact invariant | :microscope: *FORMALIZED SPEC* |
| **13** | `THM_6_SPD_SUFFICIENT_CONDITION` | Section 4.1, Theorem 6 (`th:spd_condition`) | Gershgorin dominant bound $\rho_{\max} < 1/(L-1)$ ensures positive definiteness | :microscope: *FORMALIZED SPEC* |
| **14** | `THM_7_ADMISSIBLE_DOMAIN_CONVEX_SPECTRAHEDRON` | Section 4.2, Theorem 7 (`th:admissible_convexity`) | Convexity of the positive definite elliptope cone under linear combinations | :microscope: *FORMALIZED SPEC* |
| **15** | `THM_8_REPAIR_UNIT_DIAGONAL` | Section 4.3, Theorem 8 (`th:reg_spd`) | Shift-and-scaling repair operator preserves exact unit diagonal $[\Theta_{\mathrm{SPD}}]_{jj} \equiv 1$ | :white_check_mark: **PROVEN** |
| **16** | `THM_8_GERSHGORIN_DOMINANCE` | Section 4.3, Theorem 8 & Remark 4 (`th:reg_spd`) | Gershgorin shift $\eta_{\mathrm{Gersh}}$ guarantees strict diagonal dominance | :white_check_mark: **PROVEN** |
| **17** | `THM_8_LEDOIT_WOLF_SHRINKAGE_EQUIVALENCE` | Section 4.3, Theorem 8 & Remark 4 (`th:reg_spd`) | Exact identity with Ledoit-Wolf shrinkage $\alpha = \eta / (1+\eta) \in [0, 1)$ | :white_check_mark: **PROVEN** |
| **18** | `THM_8_REPAIR_COMMUTES_WITH_THETA` | Section 4.3, Theorem 8 (`th:reg_spd`) | Commutativity $[\Theta_{\mathrm{SPD}}, \Theta] = \mathbf{0}$ and simultaneous diagonalizability | :microscope: *FORMALIZED SPEC* |
| **19** | `THM_8_REPAIR_PRESERVES_EIGENVECTORS` | Section 4.3, Theorem 8 (`th:reg_spd`) | Identical eigenvector preservation with shifted eigenvalues $(\lambda + \eta)/(1+\eta)$ | :microscope: *FORMALIZED SPEC* |
| **20** | `THM_8_REPAIR_PRESERVES_OFFDIAGONAL_RATIOS` | Section 4.3, Theorem 8 (`th:reg_spd`) | Strict cancellation of shrinkage factors across all off-diagonal correlation ratios | :white_check_mark: **PROVEN** |
| **21** | `COR_9_HETEROSCEDASTIC_REPAIR_DIAGONAL` | Section 4.3, Corollary 9 (`cor:heteroscedastic_repair`) | Exact marginal variance preservation $[\Sigma_{\mathrm{SPD}}]_{jj} = \sigma_j^2$ under heteroscedastic repair | :white_check_mark: **PROVEN** |
| **22** | `THM_10_GERSHGORIN_REPAIR_ANALYTICAL_JACOBIAN` | Section 4.5, Theorem 10 (`th:gershgorin_gradient`) | Quotient rule directional derivative of the regularized repair operator | :white_check_mark: **PROVEN** |
| **23** | `PROP_12_EQUICORRELATION_SPECTRUM` | Section 4.4, Proposition 12 (`prop:equicorr_spectrum`) | Equicorrelation all-ones eigenvector and eigenvalue $\lambda_1 = 1 + (L-1)\rho$ | :microscope: *FORMALIZED SPEC* |
| **24** | `PROP_13_SAMPLE_GRAM_PSD` | Section 5.1, Proposition 13 (`prop:gram_psd`) | Sample Gram matrix $K = M \Theta M^\top$ is symmetric positive semidefinite | :microscope: *FORMALIZED SPEC* |
| **25** | `LEMMA_14_CANONICAL_COORDINATE_RECOVERY` | Section 5.1, Lemma 14 (`lem:canonical_pls_recov`) | Indicator bilinear pairing isolates exact categorical PLS coordinate | :microscope: *FORMALIZED SPEC* |
| **26** | `THM_15_HILBERTIAN_DISTANCE_IDENTITY` | Section 5.3, Theorem 15 (`th:hilbert_kernel_psd`) | Chordal distance identity $\|v_i - v_j\|^2 = 2(1 - \langle v_i, v_j \rangle)$ | :white_check_mark: **PROVEN** |
| **27** | `THM_15_SCHOENBERG_KERNEL_BOUNDS` | Section 5.3, Theorem 15 (`th:hilbert_kernel_psd`) | Distance kernel entries strictly bounded in $(0, 1]$ | :white_check_mark: **PROVEN** |
| **28** | `THM_15_RATIONAL_QUADRATIC_BOUNDS` | Section 5.3, Theorem 15 (`th:hilbert_kernel_psd`) | Completely monotonic Rational Quadratic categorical kernel base bounds | :white_check_mark: **PROVEN** |
| **29** | `PROP_16_CLARKE_SUBDIFFERENTIAL_LIPSCHITZ` | Appendix B.1, Proposition 16 (`prop:clarke_subdiff`) | Shift function Lipschitz constant bound $\mathrm{Lip}_2(\eta_{\mathrm{Gersh}}) \le L - 1$ | :white_check_mark: **PROVEN** |

---

## 📊 Summary of Manuscript Tables & Figures

### Empirical Tables

| Table | Title & Scope | Nature | Provenance in Code / Text |
|---|---|---|---|
| **Table 1** | Methodological Positioning Taxonomy | Analytical / SOTA | Section 2.3 (`tab:sota_positioning`) |
| **Table 2** | Frequency & Severity of Non-SPD $\Theta$ across $(L, \ell)$ | Monte Carlo ($N=1{,}000$) | `generate_all_tables.py --table 4` & Section 6.1 |
| **Table 3** | Regularization Strategy Comparison ($L=15, \ell=3$) | Benchmarks ($N=200$) | `generate_all_tables.py --table 5` & Section 6.3 |
| **Table 4** | Condition Number $\kappa(K_{\mathrm{train}})$ and RMSE on $f_1, f_2, f_3$ | 50 Replications $\times$ 5 Starts | `generate_all_tables.py --table 6` & Section 6.3 |
| **Table 5** | Aerospace Cantilever Beam Emulation Benchmark ($L=12, d=3$) | 30 Replications | `benchmark_gp_experiments.py` & Section 6.4 |
| **Table C1** | Operational Duality of Bidirectional PLS Operators | Analytical / Algebraic | Appendix C.1 (`tab:operator_comparison`) |
| **Table C2** | Empirical Distribution of Raw PLS Row Norms ($N=1{,}000$) | Monte Carlo ($N=1{,}000$) | `generate_all_tables.py --table 3` & Appendix C.2 |

### Publication Figures

| Figure | Description | File Path |
|---|---|---|
| **Figure 1** | Exact 2D cross-sections of admissible domain $\mathcal{A}_L \subset \mathcal{C}_3$ and cluster Frobenius distortion vs $\rho_{\mathrm{in}}$. | `figures/fig1_phase_diagram.pdf` |
| **Figure 2** | Concentrated profile likelihood $\mathcal{L}(\bm{\theta})$ contours, polyhedral chambers, and L-BFGS-B convergence trajectories. | `figures/fig2_repair_tradeoff.pdf` |
| **Figure 3** | Predictive test error scaling across training budgets $n_t/L \in \{2, 3, 4, 6\}$ and training condition number $\log_{10} \kappa$ boxplots. | `figures/fig3_benchmark_boxplots.pdf` |
| **Figure 4** | Pareto MLE training time scaling ($L \le 80$) and learned $12 \times 12$ correlation matrix heatmap on the Aerospace Cantilever Beam. | `figures/fig4_runtime_scaling.pdf` |

---

## 📑 Citation

If you use this work, its code, or the formal proofs in your research, please cite:

```bibtex
@article{saves2026pls,
  title={Partial Least Squares for Qualitative Correlation Matrices: Geometry, Invariance, and Fast Spectral Repair},
  author={Saves, Paul},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
