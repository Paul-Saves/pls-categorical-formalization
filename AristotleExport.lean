/-
  AristotleExport.lean
  ====================
  Standalone formal proof goal specifications and automated theorem prover (ATP)
  bundles for Aristotle / Lean SMT / Lean 4.

  Manuscript:
  "Partial Least Squares for Qualitative Correlation Matrices:
   Geometry, Invariance, and Fast Spectral Repair"
  Author: Paul Saves

  Target Formal Verification Systems:
  - Aristotle Automated Prover Engine (Harmonix / DeepMind Formal Math)
  - Lean 4 (v4.14.0) with Mathlib 4.14.0
  - Lean-SMT / cvc5 / Z3 backend
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Order.BigOperators.Group.Finset

set_option linter.unusedVariables false
set_option linter.docPrime false

namespace AristotleExport

open Matrix BigOperators Finset

/-!
  ### Foundational Definitions, Predicates, and Norms
-/

/-- Number of independent off-diagonal parameters: D(n) = n(n-1)/2. -/
def D_dim (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Row-normalization predicate for the PLS rotation weight matrix G_*. -/
def UnitRowNorm {Din Dout : ℕ} (G : Matrix (Fin Din) (Fin Dout) ℝ) : Prop :=
  ∀ r : Fin Din, ∑ c : Fin Dout, (G r c)^2 = 1

/-- Affine space S_1^n: Real symmetric matrices with unit diagonals. -/
def InAffineSymmetricUnitDiag (n : ℕ) (M : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  M.IsSymm ∧ (∀ i : Fin n, M i i = 1)

/-- Correlation hypercube H_n: Matrices in S_1^n with off-diagonal entries in [-1, 1]. -/
def InCorrelationHypercube (n : ℕ) (M : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  InAffineSymmetricUnitDiag n M ∧ (∀ i j : Fin n, |M i j| ≤ 1)

/-- Correlation elliptope C_n: Strictly positive definite matrices in S_1^n. -/
def InCorrelationElliptope (n : ℕ) (M : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  InAffineSymmetricUnitDiag n M ∧ M.PosDef

/-- Strictly Diagonally Dominant predicate with positive diagonals. -/
def StrictlyDiagonallyDominant (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i : Fin n, ∑ j : Fin n, (if j ≠ i then |A i j| else 0) < A i i

/-- Eigenvalue predicate for real matrices. -/
def IsEigenvalue {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (lambda : ℝ) : Prop :=
  ∃ v : Fin n → ℝ, v ≠ 0 ∧ mulVec A v = lambda • v

/-- Classical Gershgorin circle consequence:
    Strictly diagonally dominant real symmetric matrices with positive diagonals are SPD. -/
axiom sdd_implies_posdef (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
    (h_sym : A.IsSymm) (h_diag_pos : ∀ i, A i i > 0)
    (h_sdd : StrictlyDiagonallyDominant n A) : A.PosDef


/-!
  =============================================================================
  ARISTOTLE GOAL 1: Foundational Convex Combination Identity
  =============================================================================
  TARGET_ID: LEMMA_CONVEX_COMB_CONST
  DIFFICULTY: Level 1 (Elementary Ring / BigOperators)
  HINTS: [rw [← sum_mul], rw [hw_sum], rw [one_mul]]
  STATUS: PROVEN
-/

theorem aristotle_goal_convex_comb_const {n : ℕ} (w : Fin n → ℝ) (c : ℝ)
    (hw_sum : ∑ i, w i = 1) :
    ∑ i, w i * c = c := by
  rw [← sum_mul, hw_sum, one_mul]


/-!
  =============================================================================
  ARISTOTLE GOAL 2: Convex Combination Absolute Boundedness
  =============================================================================
  TARGET_ID: LEMMA_CONVEX_COMB_ABS_LE
  DIFFICULTY: Level 2 (Linear Arithmetic + Absolute Value Order)
  HINTS: [abs_le, Finset.sum_le_sum, linarith, convex_comb_const]
  STATUS: PROVEN
-/

theorem aristotle_goal_convex_comb_abs_le {n : ℕ} (w x : Fin n → ℝ) (B : ℝ)
    (hw_nonneg : ∀ i, 0 ≤ w i)
    (hw_sum : ∑ i, w i = 1)
    (hx_le : ∀ i, |x i| ≤ B) :
    |∑ i, w i * x i| ≤ B := by
  have h_bound : ∀ i, - (w i * B) ≤ w i * x i ∧ w i * x i ≤ w i * B := by
    intro i
    have hxi := abs_le.mp (hx_le i)
    constructor
    · have h_mul := mul_le_mul_of_nonneg_left hxi.1 (hw_nonneg i)
      linarith
    · exact mul_le_mul_of_nonneg_left hxi.2 (hw_nonneg i)
  have h_upper : (∑ i, w i * x i) ≤ B := by
    calc (∑ i, w i * x i) ≤ ∑ i, w i * B := Finset.sum_le_sum (fun i _ => (h_bound i).2)
    _ = B := aristotle_goal_convex_comb_const w B hw_sum
  have h_lower : -B ≤ ∑ i, w i * x i := by
    have h_neg_sum : -B = ∑ i, - (w i * B) := by
      calc -B = - (∑ i, w i * B) := by rw [aristotle_goal_convex_comb_const w B hw_sum]
      _ = ∑ i, - (w i * B) := by rw [← sum_neg_distrib]
    rw [h_neg_sum]
    exact Finset.sum_le_sum (fun i _ => (h_bound i).1)
  exact abs_le.mpr ⟨h_lower, h_upper⟩


/-!
  =============================================================================
  ARISTOTLE GOAL 3: Unit Row-Norm Convex Weights (Lemma 2)
  =============================================================================
  TARGET_ID: LEMMA_2_CONVEX_COMBINATION_WEIGHTS
  DIFFICULTY: Level 1 (Definitional Expansion)
  PAPER_REF: Section 3.2, Lemma 2
  HINTS: [exact ⟨fun c => sq_nonneg _, h_norm r⟩]
  STATUS: PROVEN
-/

theorem aristotle_goal_convex_weights {Din Dout : ℕ} (G : Matrix (Fin Din) (Fin Dout) ℝ)
    (h_norm : UnitRowNorm G) (r : Fin Din) :
    (∀ c : Fin Dout, 0 ≤ (G r c)^2) ∧ (∑ c : Fin Dout, (G r c)^2 = 1) := by
  exact ⟨fun c => sq_nonneg (G r c), h_norm r⟩


/-!
  =============================================================================
  ARISTOTLE GOAL 4: Entry Bound Preservation on Hypercube (Lemma 3)
  =============================================================================
  TARGET_ID: LEMMA_3_ENTRY_BOUND_PRESERVATION
  DIFFICULTY: Level 2 (Matrix Structure + Convex Combination Bound)
  PAPER_REF: Section 3.2, Lemma 3
  STATUS: PROVEN
-/

theorem aristotle_goal_entry_bound
    {Din Dout : ℕ}
    (w : Fin Din → Fin Dout → ℝ)
    (hw_nonneg : ∀ r c, 0 ≤ w r c)
    (hw_sum : ∀ r, ∑ c, w r c = 1)
    (theta_hat : Fin Dout → ℝ)
    (h_bound : ∀ c, |theta_hat c| ≤ 1) :
    ∀ r : Fin Din, |∑ c, w r c * theta_hat c| ≤ 1 := by
  intro r
  exact aristotle_goal_convex_comb_abs_le (w r) theta_hat 1 (hw_nonneg r) (hw_sum r) h_bound


/-!
  =============================================================================
  ARISTOTLE GOAL 5: Strict Diagonal Dominance => Positive Definiteness (Theorem 1)
  =============================================================================
  TARGET_ID: THM_1_SPD_SUFFICIENT_CONDITION
  DIFFICULTY: Level 3 (Gershgorin + Diagonals)
  PAPER_REF: Section 4.1, Theorem 1
  STATUS: PROOF_STRUCTURE_FORMALIZED
-/

theorem aristotle_goal_spd_from_dominance
    (L : ℕ) (A : Matrix (Fin L) (Fin L) ℝ)
    (h_sym : A.IsSymm)
    (h_diag : ∀ i, A i i = 1)
    (rho_max : ℝ)
    (h_offdiag : ∀ i j, i ≠ j → |A i j| ≤ rho_max)
    (h_rho_bound : (L : ℝ) - 1 > 0 ∧ rho_max < 1 / ((L : ℝ) - 1)) :
    A.PosDef := by
  sorry


/-!
  =============================================================================
  ARISTOTLE GOAL 6: Sample Gram Matrix Positive Semidefiniteness (Proposition 2)
  =============================================================================
  TARGET_ID: PROP_2_SAMPLE_GRAM_PSD
  DIFFICULTY: Level 3 (Quadratic Form Factoring vᵀ (M Θ Mᵀ) v = uᵀ Θ u ≥ 0)
  PAPER_REF: Section 5.1, Proposition 2
  STATUS: PROOF_STRUCTURE_FORMALIZED
-/

theorem aristotle_goal_gram_psd
    {L nt : ℕ} (Theta : Matrix (Fin L) (Fin L) ℝ)
    (h_posdef : Theta.PosDef)
    (M : Matrix (Fin nt) (Fin L) ℝ) :
    let K := M * Theta * M.transpose
    K.PosSemidef := by
  sorry


/-!
  =============================================================================
  ARISTOTLE GOAL 7: Equicorrelation Spectrum & Determinant Scaling (Theorem 3)
  =============================================================================
  TARGET_ID: THM_3_EQUICORRELATION_SPECTRUM
  DIFFICULTY: Level 3 (Matrix Perturbation with All-Ones Vector)
  PAPER_REF: Section 4.3, Theorem 3
  STATUS: PROOF_STRUCTURE_FORMALIZED
-/

theorem aristotle_goal_equicorrelation_eigenvalue
    {L : ℕ} (rho : ℝ) (hL : L ≥ 2) :
    let J : Matrix (Fin L) (Fin L) ℝ := fun _ _ => 1
    let Theta := (1 - rho) • (1 : Matrix (Fin L) (Fin L) ℝ) + rho • J
    let ones : Fin L → ℝ := fun _ => 1
    mulVec Theta ones = (1 + ((L : ℝ) - 1) * rho) • ones := by
  sorry


/-!
  =============================================================================
  ARISTOTLE GOAL 8: Regularized Operator Unit Diagonal Preservation (Theorem 2)
  =============================================================================
  TARGET_ID: THM_2_REPAIR_UNIT_DIAGONAL
  DIFFICULTY: Level 1 (Diagonal Arithmetic)
  PAPER_REF: Section 4.2, Theorem 2
  STATUS: PROVEN
-/

noncomputable def repair_operator (n : ℕ) (Theta : Matrix (Fin n) (Fin n) ℝ) (eta : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  (1 / (1 + eta)) • (Theta + eta • (1 : Matrix (Fin n) (Fin n) ℝ))

theorem aristotle_goal_repair_unit_diagonal
    {L : ℕ} (Theta : Matrix (Fin L) (Fin L) ℝ)
    (h_diag : ∀ i, Theta i i = 1)
    (eta : ℝ) (h_eta : 1 + eta ≠ 0) :
    ∀ i : Fin L, repair_operator L Theta eta i i = 1 := by
  intro i
  dsimp [repair_operator]
  rw [h_diag i, Matrix.one_apply_eq, mul_one]
  exact one_div_mul_cancel h_eta


/-!
  =============================================================================
  ARISTOTLE GOAL 9: Gershgorin Shift Strict Dominance Guarantee (Theorem 2)
  =============================================================================
  TARGET_ID: THM_2_GERSHGORIN_DOMINANCE
  DIFFICULTY: Level 2 (Diagonal Shift Strict Inequality)
  PAPER_REF: Section 4.2, Theorem 2 & Remark 3
  STATUS: PROVEN
-/

theorem aristotle_goal_gershgorin_shift_dominance
    (row_sum_offdiag eta eps : ℝ)
    (heps : eps > 0)
    (heta : eta ≥ row_sum_offdiag - 1 + eps) :
    row_sum_offdiag < 1 + eta := by
  linarith


/-!
  =============================================================================
  ARISTOTLE GOAL 10: Hilbertian Squared Distance Euclidean Expansion (Theorem 4)
  =============================================================================
  TARGET_ID: THM_4_HILBERTIAN_DISTANCE_IDENTITY
  DIFFICULTY: Level 1 (Euclidean Norm Squared Ring Expansion)
  PAPER_REF: Section 5.2.2, Theorem 4
  STATUS: PROVEN
-/

theorem aristotle_goal_hilbertian_distance_identity
    (norm_vi_sq norm_vj_sq inner_prod : ℝ)
    (hvi : norm_vi_sq = 1)
    (hvj : norm_vj_sq = 1) :
    norm_vi_sq + norm_vj_sq - 2 * inner_prod = 2 * (1 - inner_prod) := by
  rw [hvi, hvj]
  ring


/-!
  =============================================================================
  ARISTOTLE GOAL 11: Schoenberg Distance Kernel Envelope Bounds (Theorem 4)
  =============================================================================
  TARGET_ID: THM_4_SCHOENBERG_KERNEL_BOUNDS
  DIFFICULTY: Level 2 (Exponential Monotonicity on Correlation Domain)
  PAPER_REF: Section 5.2.2, Theorem 4
  STATUS: PROVEN
-/

theorem aristotle_goal_schoenberg_kernel_envelope
    (theta_qual rho : ℝ)
    (htheta : theta_qual > 0)
    (hrho_lower : -1 ≤ rho)
    (hrho_upper : rho ≤ 1) :
    let k := Real.exp (-theta_qual * (1 - rho))
    0 < k ∧ k ≤ 1 ∧ (rho = 1 → k = 1) := by
  dsimp
  refine ⟨Real.exp_pos _, ?_, ?_⟩
  · rw [Real.exp_le_one_iff]
    have h1 : 0 ≤ 1 - rho := by linarith
    have h2 : 0 ≤ theta_qual * (1 - rho) := mul_nonneg (le_of_lt htheta) h1
    linarith
  · intro hrho1
    have hzero : -theta_qual * (1 - rho) = 0 := by
      calc -theta_qual * (1 - rho) = -theta_qual * (1 - 1) := by rw [hrho1]
      _ = 0 := by ring
    rw [hzero, Real.exp_zero]


/-!
  =============================================================================
  ARISTOTLE GOAL 12: Ledoit-Wolf Linear Shrinkage Equivalence (Theorem 2 & Remark 3)
  =============================================================================
  TARGET_ID: THM_2_LEDOIT_WOLF_SHRINKAGE_EQUIVALENCE
  DIFFICULTY: Level 1 (Affine Convex Shrinkage Parameter Identity)
  PAPER_REF: Section 4.2, Theorem 2 and Remark 3
  STATUS: PROVEN
-/

theorem aristotle_goal_ledoit_wolf_shrinkage_equivalence
    (eta : ℝ) (heta : eta ≥ 0) :
    let alpha := eta / (1 + eta)
    (0 ≤ alpha ∧ alpha < 1) ∧
    (1 - alpha = 1 / (1 + eta)) ∧
    (alpha = (1 / (1 + eta)) * eta) := by
  dsimp
  have h_denom_pos : 0 < 1 + eta := by linarith
  have h_alpha_nonneg : 0 ≤ eta / (1 + eta) := div_nonneg heta (le_of_lt h_denom_pos)
  have h_alpha_lt_one : eta / (1 + eta) < 1 := by
    rw [div_lt_one h_denom_pos]
    linarith
  have h_one_sub_alpha : 1 - eta / (1 + eta) = 1 / (1 + eta) := by
    have h_div_self : (1 + eta) / (1 + eta) = 1 := div_self (ne_of_gt h_denom_pos)
    calc 1 - eta / (1 + eta)
      _ = (1 + eta) / (1 + eta) - eta / (1 + eta) := by rw [h_div_self]
      _ = (1 + eta - eta) / (1 + eta) := by rw [← sub_div]
      _ = 1 / (1 + eta) := by ring
  have h_alpha_factor : eta / (1 + eta) = (1 / (1 + eta)) * eta := by
    ring
  exact ⟨⟨h_alpha_nonneg, h_alpha_lt_one⟩, h_one_sub_alpha, h_alpha_factor⟩


/-!
  =============================================================================
  ARISTOTLE GOAL 13: Canonical PLS Coordinate Recovery (Lemma 4)
  =============================================================================
  TARGET_ID: LEMMA_4_CANONICAL_COORDINATE_RECOVERY
  DIFFICULTY: Level 2 (Indicator Vector Bilinear Pairing Identity)
  PAPER_REF: Section 5.1, Lemma 4
  STATUS: PROOF_STRUCTURE_FORMALIZED
-/

/-- Pairwise level selection operator delta (Definition in Lemma 4). -/
def delta_operator {D : ℕ} (m_target : Fin D) : Fin D → ℝ :=
  fun m => if m = m_target then 1 else 0

/-- Lemma 4 (Canonical Coordinate Recovery):
    Bilinear pairing with the indicator delta operator isolates the exact coordinate. -/
theorem aristotle_goal_canonical_recovery
    {D : ℕ} (theta_vec : Fin D → ℝ) (m_target : Fin D) :
    ∑ m : Fin D, (delta_operator m_target m) * (theta_vec m) = theta_vec m_target := by
  sorry


/-!
  =============================================================================
  ARISTOTLE GOAL 14: Round-Trip Adjoint Contraction (Theorem th:roundtrip_contraction)
  =============================================================================
  TARGET_ID: THM_ROUNDTRIP_ADJOINT_CONTRACTION
  DIFFICULTY: Level 2 (Row-Stochastic Composite Matrix Operator)
  PAPER_REF: Section 3.3, Theorem th:roundtrip_contraction
  STATUS: PROOF_STRUCTURE_FORMALIZED
-/

theorem aristotle_goal_roundtrip_stochastic
    {Din Dout : ℕ}
    (G : Matrix (Fin Din) (Fin Dout) ℝ)
    (h_norm : UnitRowNorm G)
    (W : Matrix (Fin Dout) (Fin Din) ℝ)
    (h_W_stoch : ∀ p, (∀ r, 0 ≤ W p r) ∧ (∑ r, W p r = 1)) :
    let M : Matrix (Fin Dout) (Fin Dout) ℝ := fun p q => ∑ r : Fin Din, W p r * (G r q)^2
    (∀ p q, 0 ≤ M p q) ∧ (∀ p, ∑ q, M p q = 1) := by
  dsimp
  constructor
  · intro p q
    exact Finset.sum_nonneg (fun r _ => mul_nonneg (h_W_stoch p).1 r (sq_nonneg (G r q)))
  · intro p
    have h_swap : ∑ q, (∑ r, W p r * (G r q)^2) = ∑ r, (∑ q, W p r * (G r q)^2) := Finset.sum_comm
    rw [h_swap]
    have h_inner : ∀ r, (∑ q, W p r * (G r q)^2) = W p r := by
      intro r
      rw [← mul_sum, h_norm r, mul_one]
    have h_sum_inner : (∑ r, ∑ q, W p r * (G r q)^2) = ∑ r, W p r := by
      exact Finset.sum_congr rfl (fun r _ => h_inner r)
    rw [h_sum_inner]
    exact (h_W_stoch p).2


/-!
  =============================================================================
  ARISTOTLE GOAL 15: Admissible Latent Domain Convexity (Theorem th:admissible_convexity)
  =============================================================================
  TARGET_ID: THM_ADMISSIBLE_DOMAIN_CONVEXITY
  DIFFICULTY: Level 2 (Spectrahedral Linear Preimage Intersection)
  PAPER_REF: Section 4.1, Theorem th:admissible_convexity
  STATUS: PROOF_STRUCTURE_FORMALIZED
-/

theorem aristotle_goal_admissible_domain_convex
    {n : ℕ} (M1 M2 : Matrix (Fin n) (Fin n) ℝ)
    (h_sym1 : M1.IsSymm) (h_sym2 : M2.IsSymm)
    (h_diag1 : ∀ i, M1 i i = 1) (h_diag2 : ∀ i, M2 i i = 1)
    (lambda : ℝ) (h_lam : 0 < lambda ∧ lambda < 1) :
    let M := lambda • M1 + (1 - lambda) • M2
    M.IsSymm ∧ (∀ i, M i i = 1) := by
  dsimp
  constructor
  · rw [Matrix.isSymm_add]
    · exact ⟨IsSymm.smul lambda h_sym1, IsSymm.smul (1 - lambda) h_sym2⟩
  · intro i
    rw [Matrix.add_apply, Matrix.smul_apply, Matrix.smul_apply, h_diag1 i, h_diag2 i]
    ring


/-!
  =============================================================================
  ARISTOTLE GOAL 16: Analytical Gradient of Gershgorin Repair (Theorem th:gershgorin_gradient)
  =============================================================================
  TARGET_ID: THM_GERSHGORIN_REPAIR_GRADIENT
  DIFFICULTY: Level 2 (Non-Smooth Shift Quotient Rule Derivative)
  PAPER_REF: Section 4.2, Theorem th:gershgorin_gradient
  STATUS: PROVEN
-/

theorem aristotle_goal_gershgorin_repair_gradient
    (theta_entry eta dtheta_entry deta : ℝ)
    (h_eta : eta > 0) :
    let repaired_entry := theta_entry / (1 + eta)
    (1 / (1 + eta)) * (dtheta_entry - repaired_entry * deta) =
    (dtheta_entry * (1 + eta) - theta_entry * deta) / (1 + eta)^2 := by
  dsimp
  have h_denom : 1 + eta ≠ 0 := by linarith
  have h_sq : (1 + eta)^2 = (1 + eta) * (1 + eta) := sq (1 + eta)
  rw [h_sq]
  field_simp
  ring

end AristotleExport
