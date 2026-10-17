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

When modeling computer experiments with categorical factors using Gaussian processes (Kriging), supervised dimension reduction via Partial Least Squares (PLS) is a popular technique for projecting high-dimensional categorical spaces onto low-dimensional latent spaces. 

However, unconstrained PLS reconstruction frequently produces non-positive-definite (non-SPD) correlation matrices, causing numerical breakdowns during Maximum Likelihood Estimation (MLE). This repository provides:
1. **Mathematical Guarantees:** A rigorous characterization of the Row-Normalized PLS (RN-PLS) operator, hypercube preservation, Markov contraction, and the spectrahedral convexity of the admissible latent domain $\mathcal{A}_L$.
2. **Fast Gershgorin Repair:** A certified, closed-form $O(L^2)$ diagonal shift-and-scaling operator that guarantees positive definiteness and exact unit diagonals without computing eigenvalues.
3. **Analytical Gradients:** Closed-form directional Jacobians and Clarke subdifferentials almost everywhere for seamless integration into quasi-Newton solvers (L-BFGS-B).
4. **Interactive Theorem Proving:** Complete formalization in Lean 4 (`FormalProofs.lean`, `AristotleExport.lean`).

---

## 📁 Repository Structure

```text
├── main.tex                       # Complete LaTeX manuscript
├── sample.bib                     # Curated BibTeX bibliography
├── requirements.txt               # Minimal Python dependencies
│
├── generate_all_tables.py         # Master reproduction script for Tables 3, 4, 5, 6
├── generate_all_figures.py        # Master reproduction script for Figures 1, 2, 3, 4
├── benchmark_gp_experiments.py    # Multi-start L-BFGS-B GP emulation benchmark
├── verify_theorems.py             # Numerical stress-testing and theorem verification
│
├── FormalProofs.lean              # Lean 4 formalization of definitions, lemmes, and theorems
├── AristotleExport.lean           # Lean 4 goals formatted for automated theorem provers
├── aristotle_verifier.py          # Formal verification orchestrator & JSON goal exporter
├── aristotle_goals.json           # Machine-readable Aristotle ATP goals
│
└── figures/                       # Publication-quality vector graphics (PDF & PNG)
    ├── fig1_elliptope_geometry.pdf
    ├── fig2_eigenvalue_spectra.pdf
    ├── fig3_benchmark_boxplots.pdf
    └── fig4_runtime_scaling.pdf
```

---

## 🚀 Quick Start & Reproducibility

### 1. Installation

Clone this repository and install the minimal dependencies:

```bash
git clone https://github.com/Paul-Saves/pls-categorical-formalization.git
cd pls-categorical-formalization
pip install -r requirements.txt
```

### 2. Reproduce All Manuscript Tables

To execute a rapid validation (~20 seconds):
```bash
python generate_all_tables.py --quick
```

To reproduce the exact full Monte-Carlo counts reported in the manuscript (1,000 trials for Tables 3 & 4; 200 instances for Table 5; 50 replications $\times$ 5 multi-starts for Table 6):
```bash
python generate_all_tables.py --full
```

You can also run individual tables:
```bash
python generate_all_tables.py --table 3    # Raw PLS row norms distribution
python generate_all_tables.py --table 4    # Frequency and severity of non-SPD reconstructions
python generate_all_tables.py --table 5    # Condition numbers across 200 non-SPD instances
python generate_all_tables.py --table 6    # Multi-function GP emulation benchmark (f1, f2, f3)
```

### 3. Reproduce All Vector Figures

Generate all 4 publication vector figures in `figures/`:
```bash
python generate_all_figures.py
```

### 4. Mathematical Proof-Checking

Run the companion verification suite to validate all algebraic identities, matrix invariants, and non-SPD counterexamples:
```bash
python verify_theorems.py
```

### 5. Formal Verification in Lean 4 & Aristotle ATP Bundle

Verify the Lean 4 proof bundle using the Lean package manager `lake`:
```bash
lake build
```
Or inspect the formal declarations and automated theorem prover goals using:
```bash
python aristotle_verifier.py
```

### 🧩 Aristotle Formal Verification Matrix (17/29 Proven, 58.6% Machine-Certified)

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
| **13** | `THM_7_SPD_SUFFICIENT_CONDITION` | Section 4.1, Theorem 7 (`th:spd_condition`) | Gershgorin dominant bound $\rho_{\max} < 1/(L-1)$ ensures positive definiteness | :microscope: *FORMALIZED SPEC* |
| **14** | `THM_8_ADMISSIBLE_DOMAIN_CONVEX_SPECTRAHEDRON` | Section 4.2, Theorem 8 (`th:admissible_convexity`) | Convexity of the positive definite elliptope cone under linear combinations | :microscope: *FORMALIZED SPEC* |
| **15** | `THM_9_REPAIR_UNIT_DIAGONAL` | Section 4.3, Theorem 9 (`th:reg_spd`) | Shift-and-scaling repair operator preserves exact unit diagonal $[\Theta_{\mathrm{SPD}}]_{jj} \equiv 1$ | :white_check_mark: **PROVEN** |
| **16** | `THM_9_GERSHGORIN_DOMINANCE` | Section 4.3, Theorem 9 & Remark 4 (`th:reg_spd`) | Gershgorin shift $\eta_{\mathrm{Gersh}}$ guarantees strict diagonal dominance | :white_check_mark: **PROVEN** |
| **17** | `THM_9_LEDOIT_WOLF_SHRINKAGE_EQUIVALENCE` | Section 4.3, Theorem 9 & Remark 4 (`th:reg_spd`) | Exact identity with Ledoit-Wolf shrinkage $\alpha = \eta / (1+\eta) \in [0, 1)$ | :white_check_mark: **PROVEN** |
| **18** | `THM_9_REPAIR_COMMUTES_WITH_THETA` | Section 4.3, Theorem 9 (`th:reg_spd`) | Commutativity $[\Theta_{\mathrm{SPD}}, \Theta] = \mathbf{0}$ and simultaneous diagonalizability | :microscope: *FORMALIZED SPEC* |
| **19** | `THM_9_REPAIR_PRESERVES_EIGENVECTORS` | Section 4.3, Theorem 9 (`th:reg_spd`) | Identical eigenvector preservation with shifted eigenvalues $(\lambda + \eta)/(1+\eta)$ | :microscope: *FORMALIZED SPEC* |
| **20** | `THM_9_REPAIR_PRESERVES_OFFDIAGONAL_RATIOS` | Section 4.3, Theorem 9 (`th:reg_spd`) | Strict cancellation of shrinkage factors across all off-diagonal correlation ratios | :white_check_mark: **PROVEN** |
| **21** | `COR_9_HETEROSCEDASTIC_REPAIR_DIAGONAL` | Section 4.3, Corollary (`cor:heteroscedastic_repair`) | Exact marginal variance preservation $[\Sigma_{\mathrm{SPD}}]_{jj} = \sigma_j^2$ under heteroscedastic repair | :white_check_mark: **PROVEN** |
| **22** | `THM_10_GERSHGORIN_REPAIR_ANALYTICAL_JACOBIAN` | Section 4.5, Theorem 10 (`th:gershgorin_gradient`) | Quotient rule directional derivative of the regularized repair operator | :white_check_mark: **PROVEN** |
| **23** | `PROP_11_EQUICORRELATION_SPECTRUM` | Section 4.4, Proposition 11 (`prop:equicorr_spectrum`) | Equicorrelation all-ones eigenvector and eigenvalue $\lambda_1 = 1 + (L-1)\rho$ | :microscope: *FORMALIZED SPEC* |
| **24** | `PROP_12_SAMPLE_GRAM_PSD` | Section 5.1, Proposition 12 (`prop:gram_psd`) | Sample Gram matrix $K = M \Theta M^\top$ is symmetric positive semidefinite | :microscope: *FORMALIZED SPEC* |
| **25** | `LEMMA_14_CANONICAL_COORDINATE_RECOVERY` | Section 5.1, Lemma 14 (`lem:canonical_pls_recov`) | Indicator bilinear pairing isolates exact categorical PLS coordinate | :microscope: *FORMALIZED SPEC* |
| **26** | `THM_16_HILBERTIAN_DISTANCE_IDENTITY` | Section 5.3, Theorem 16 (`th:hilbert_kernel_psd`) | Chordal distance identity $\|v_i - v_j\|^2 = 2(1 - \langle v_i, v_j \rangle)$ | :white_check_mark: **PROVEN** |
| **27** | `THM_16_SCHOENBERG_KERNEL_BOUNDS` | Section 5.3, Theorem 16 (`th:hilbert_kernel_psd`) | Distance kernel entries strictly bounded in $(0, 1]$ | :white_check_mark: **PROVEN** |
| **28** | `THM_16_RATIONAL_QUADRATIC_BOUNDS` | Section 5.3, Theorem 16 (`th:hilbert_kernel_psd`) | Completely monotonic Rational Quadratic categorical kernel base bounds | :white_check_mark: **PROVEN** |
| **29** | `PROP_17_CLARKE_SUBDIFFERENTIAL_LIPSCHITZ` | Appendix B.1, Proposition 17 (`prop:clarke_subdiff`) | Shift function Lipschitz constant bound $\mathrm{Lip}_2(\eta_{\mathrm{Gersh}}) \le L - 1$ | :white_check_mark: **PROVEN** |

---

## 📊 Summary of Experimental Tables & Provenance

| Table | Content | Nature | Provenance in Code / Text |
|---|---|---|---|
| **Table 1** | Methodological Positioning Taxonomy | Analytical / Theoretical | Section 2.3 in `main.tex` |
| **Table 2** | Operational Duality of PLS Operators | Analytical / Algebraic | Appendix C.1 (`subsec:app_operator_duality`) |
| **Table 3** | Raw PLS Row Norms Distribution | Empirical Monte Carlo ($N=1{,}000$) | `generate_all_tables.py:run_table_3` & Appendix C.2 |
| **Table 4** | Frequency & Severity of Non-SPD $\Theta$ | Empirical Wishart ($N=1{,}000$) | `generate_all_tables.py:run_table_4` & Section 6.1 |
| **Table 5** | Regularization Strategy Comparison | Non-SPD Benchmarks ($N=200$) | `generate_all_tables.py:run_table_5` & Section 6.2 |
| **Table 6** | Multi-Function GP Benchmark ($f_1, f_2, f_3$) | MLE Optimization (750 trajectories / strategy) | `benchmark_gp_experiments.py` & Section 6.2 |

---

## 📑 Citation

If you use this work or its code, please cite:

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

This project is licensed under the MIT License - see the LICENSE file for details.
