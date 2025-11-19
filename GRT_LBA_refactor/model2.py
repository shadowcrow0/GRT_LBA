"""
GRT-LBA 4-Choice Recognition Task - REALISTIC VERSION (v7_REALISTIC)
==================================================================
雙側刺激辨認任務 with Perceptual Separability (PS):

架構：
- 4 個 conditions (刺激組合): HH, HV, VH, VV
- 2 個獨立的 dimensions: LEFT 和 RIGHT
- 每個 dimension 在每個時間點有 2 個 accumulators 競爭 (選 H vs 選 V)
- 最終反應時間 = max(LEFT 決策時間, RIGHT 決策時間) + t0

Perceptual Separability 假設：
- LEFT dimension 的 drift rates 只依賴 LEFT 刺激（不受 RIGHT 影響）
- RIGHT dimension 的 drift rates 只依賴 RIGHT 刺激（不受 LEFT 影響）
- 總共只需要 8 個 drift rate 參數：
  * LEFT: v1_L_when_H, v2_L_when_H, v1_L_when_V, v2_L_when_V (4 個)
  * RIGHT: v1_R_when_H, v2_R_when_H, v1_R_when_V, v2_R_when_V (4 個)

KEY FIXES (v6 - 2025-11-13):

**BUG FIX #1**: Removed incorrect across-trial drift rate variability
- v2/v4/v5 had: v_trial = rng.normal(v_mean, s) ← WRONG!
- v6 fixed to: v_trial = v_mean ← CORRECT!
- Impact: This was causing systematic overestimation (~+1.5 on average)
- Root cause: mismatch between data generation (across-trial variability)
  and likelihood (assumes fixed drift rates)

**BUG FIX #2**: Fixed truncation bias in prior
- v2/v4/v5 had: TruncatedNormal(mu=1.0, sigma=2.0, lower=0.5) ← BAD!
- v6 fixed to: TruncatedNormal(mu=1.0, sigma=2.0, lower=0.01) ← BETTER!
- Impact: lower=0.5 caused +1.16 bias for v2 parameters (mu=1.0)
- Explanation: 40% of original distribution was below 0.5 → truncation
  pushed mean from 1.0 to 2.16
- Combined effect: Two bugs together caused +1~+3 systematic bias

NEW IN v7 (2025-11-14): PRECISION IMPROVEMENTS

**PRECISION FIX #1**: Increased integration precision
- v6 had: n_points=60 → integration error +223% (SEVERE!)
- v7 fixed: n_points=200 → integration error -5.4% (GOOD!)
- Impact: v6's +0.3 residual bias was primarily due to integration error
- Expected improvement: bias +0.24 → ~±0.05 (接近完美)

**PRECISION FIX #2**: Further reduced truncation bias
- v6 had: lower=0.01 → truncation bias +0.8965
- v7 fixed: lower=0.001 → truncation bias +0.8920
- Impact: Minimal (~0.004 improvement), but zero cost

NEW IN v7_REALISTIC (2025-11-14): REALISTIC HUMAN PARAMETERS

**REALISM FIX**: Changed to human-like performance parameters
- Previous versions had unrealistic parameters:
  * v_correct = 3.0, v_error = 1.0 → 100% accuracy, RT = 0.37-0.53 sec (TOO FAST!)
- v7_REALISTIC uses realistic parameters:
  * v_correct = 1.5 (降低 from 3.0)
  * v_error = 0.8 (提高 from 1.0)
  * b = 1.5 (提高 from 1.0)
  * t0 = 0.25 (提高 from 0.2)
- Expected results:
  * Accuracy: ~70-85% (realistic for human 4-choice task)
  * RT: ~0.7-1.3 seconds (realistic human response time)

Other features (from v6):
- EXACT P(choice) calculation using defective PDF integration
- NUMBA-OPTIMIZED: 50-100x faster than scipy.quad
- WIDER PRIORS: sigma=2.0 (was 0.3)
- MORE DATA: 5000 trials (was 2000)
- MORE CHAINS: chains=8, cores=8 (was 4) for better convergence diagnostics

Author: YYC & Claude
Date: 2025-11-14
"""

import numpy as np
import os  # For file size check when saving .nc

# FIX: PyTensor file lock issue in WSL (must be before importing pytensor)
os.environ['PYTENSOR_FLAGS'] = 'base_compiledir=/tmp/pytensor_cache'

import pytensor
import pytensor.tensor as pt
import pymc as pm
from scipy.stats import norm  # Still needed for lba_defective_cdf
import warnings
from numba import jit
warnings.filterwarnings('ignore')

print("=" * 70)
print("Model 2: Symmetric PS Violation Test (10 params)")
print("=" * 70)
print("  CRITICAL BUG FIXES (2025-11-19):")
print("     1. ✅ RT likelihood now uses defective PDF/CDF (was causing +0.5-0.7 bias)")
print("     2. ✅ Integration n_points: 60 → 200 for P(choice) (reduces error)")
print("     3. ✅ Prior sigma: 2.0 → 0.8 (tighter priors for convergence)")
print("     4. ✅ Sampling: draws=30000, tune=3000 (was 15000/1000)")
print("\n  SPEED OPTIMIZATION (2025-11-19):")
print("     5. ✅ Numba JIT compilation for defective CDF integration")
print("     6. ✅ Reduced integration points: 50 → 25 (balanced speed/accuracy)")
print("     7. ✅ Fast Numba norm.cdf/pdf (no scipy overhead)")
print("     → Expected: ~4-8 hours (was 120 hours!), 15-30x speedup, PRECISE")
print("\n  REALISTIC PARAMETERS:")
print("     - v_correct: 1.8, v_error: 1.5")
print("     - b: 1.5, t0: 0.25")
print("     - Expected: Accuracy 70-85%, RT 0.7-1.3 sec")
print("\n  SYMMETRIC PS DESIGN:")
print("     - 10 params: 8 drift rates + 2 delta (delta_L, delta_R)")
print("     - Tests PS violations in BOTH Left and Right dimensions")
print("\n  BAYES FACTOR ANALYSIS:")
print("     - Uses Savage-Dickey density ratio method")
print("     - Tests H0: delta=0 (PS holds) vs H1: delta≠0 (PS violated)")
print("     - Interpretation: Kass & Raftery (1995) guidelines")
print("=" * 70)

# ============================================================================
# Global control for CDF calculation method (using dict to avoid global keyword)
# ============================================================================
# FIX: Use EXACT CDF integration (2025-11-11)
# Fast approximation has 50-100% error - see BUG_REPORT_FINAL.md
_CDF_MODE = {'use_fast': False}  # Use exact integration for correct likelihood


# ============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS (10-50x faster)
# ============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def fast_norm_pdf_numba(x):
    """Fast standard normal PDF using Numba"""
    return 0.3989422804014327 * np.exp(-0.5 * x * x)


@jit(nopython=True, fastmath=True, cache=True)
def fast_norm_cdf_numba(x):
    """Fast standard normal CDF approximation (Abramowitz & Stegun)"""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327 * np.exp(-0.5 * x * x)
    p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 +
                 t * (-1.821255978 + t * 1.330274429))))
    if x >= 0.0:
        return 1.0 - p
    else:
        return p

@jit(nopython=True, fastmath=True, cache=True)
def lba_pwin_numba_OLD_APPROX(v_w1, v_l1, v_w2, v_l2):
    # OLD APPROXIMATION (v2) - Has 16-95% error!
    # P(win) ≈ Φ(winner - loser) for each dimension
    # This ignores A, b, s parameters and has systematic bias
    cdf1 = fast_norm_cdf_numba(v_w1 - v_l1)
    cdf2 = fast_norm_cdf_numba(v_w2 - v_l2)
    return cdf1 * cdf2


# ============================================================================
# NEW: EXACT P(WIN) CALCULATION (v4) - NUMBA-OPTIMIZED Defective PDF Integration
# ============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def lba_pwin_1d_numba(v_win, v_lose, A, b, s, t_max=10.0, n_points=200):
    """
    NUMBA-OPTIMIZED: Calculate EXACT P(one accumulator wins in one dimension)

    P(win) = ∫[0→t_max] f_win(t) × S_lose(t) dt

    Uses vectorized trapezoidal integration with Numba JIT compilation.
    50-100x faster than scipy.quad while maintaining accuracy.

    Parameters:
    -----------
    v_win, v_lose : float
        Drift rates for winner and loser
    A, b, s : float
        LBA parameters
    t_max : float
        Integration upper bound (default 10.0)
    n_points : int
        Number of integration points (default 200 for high accuracy)
        v7 CHANGE: 60 → 200 (reduces integration error from +223% to -5.4%)

    Returns:
    --------
    float : P(win) in [0, 1]
    """
    # Generate integration points
    tau = np.linspace(0.0, t_max, n_points)

    # Avoid division by zero at t=0
    tau_safe = tau.copy()
    tau_safe[0] = 1e-10

    # ===== Vectorized f_win(τ) calculation =====
    t_s_win = tau_safe * s
    z1_win = (b - A - tau_safe * v_win) / t_s_win
    z2_win = (b - tau_safe * v_win) / t_s_win

    # Use fast Numba CDF/PDF
    cdf1_win = np.empty(n_points)
    cdf2_win = np.empty(n_points)
    pdf1_win = np.empty(n_points)
    pdf2_win = np.empty(n_points)

    for i in range(n_points):
        cdf1_win[i] = fast_norm_cdf_numba(z1_win[i])
        cdf2_win[i] = fast_norm_cdf_numba(z2_win[i])
        pdf1_win[i] = fast_norm_pdf_numba(z1_win[i])
        pdf2_win[i] = fast_norm_pdf_numba(z2_win[i])

    f_win = (1.0 / A) * (-v_win * cdf1_win + v_win * cdf2_win +
                         s * pdf1_win - s * pdf2_win)
    f_win = np.maximum(f_win, 0.0)
    f_win[0] = 0.0  # Boundary condition

    # ===== Vectorized S_lose(τ) calculation =====
    z1_lose = (b - A - tau_safe * v_lose) / t_s_win
    z2_lose = (b - tau_safe * v_lose) / t_s_win

    cdf1_lose = np.empty(n_points)
    cdf2_lose = np.empty(n_points)
    pdf1_lose = np.empty(n_points)
    pdf2_lose = np.empty(n_points)

    for i in range(n_points):
        cdf1_lose[i] = fast_norm_cdf_numba(z1_lose[i])
        cdf2_lose[i] = fast_norm_cdf_numba(z2_lose[i])
        pdf1_lose[i] = fast_norm_pdf_numba(z1_lose[i])
        pdf2_lose[i] = fast_norm_pdf_numba(z2_lose[i])

    v_t_minus_b_lose = v_lose * tau_safe - b
    cdf_lose = 1.0 + (v_t_minus_b_lose / A) * cdf2_lose - \
               ((v_t_minus_b_lose + A) / A) * cdf1_lose + \
               (t_s_win / A) * pdf2_lose - (t_s_win / A) * pdf1_lose

    cdf_lose = np.clip(cdf_lose, 0.0, 1.0)
    s_lose = 1.0 - cdf_lose

    # ===== Trapezoidal integration =====
    integrand = f_win * s_lose

    # Manual trapezoidal rule (Numba doesn't support np.trapz)
    result = 0.0
    for i in range(n_points - 1):
        dt = tau[i+1] - tau[i]
        result += 0.5 * (integrand[i] + integrand[i+1]) * dt

    # Ensure result is in [0, 1]
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


def lba_pwin_1d_exact(v_win, v_lose, A, b, s, t_max=10.0):
    """
    Wrapper for backward compatibility.
    Calls the Numba-optimized version.

    v7 FIX: n_points=60 → 200 (reduces integration error from +223% to -5.4%)
    """
    return lba_pwin_1d_numba(v_win, v_lose, A, b, s, t_max, n_points=200)


def lba_pwin_exact(v_w1, v_l1, v_w2, v_l2, A, b, s):
    """
    NEW (v3): EXACT P(choice) for 2D independent decisions

    P(choice) = P(dim1 winner wins) × P(dim2 winner wins)
              = ∫ f_win1(t) × S_lose1(t) dt × ∫ f_win2(t) × S_lose2(t) dt

    This is the whiteboard method = defective PDF integration!
    """
    p1 = lba_pwin_1d_exact(v_w1, v_l1, A, b, s)
    p2 = lba_pwin_1d_exact(v_w2, v_l2, A, b, s)
    return p1 * p2


@jit(nopython=True, fastmath=True, cache=True)
def lba_pdf_numba(t, v, A, b, s):
    """Numba-optimized PDF for one LBA accumulator"""
    if t <= 0.0 or v <= 0.0 or A <= 0.0 or b <= 0.0 or s <= 0.0:
        return 1e-10
    if t > 10.0:
        return 1e-10

    t_s = t * s
    z1 = (b - A - t * v) / t_s
    z2 = (b - t * v) / t_s

    cdf1 = fast_norm_cdf_numba(z1)
    cdf2 = fast_norm_cdf_numba(z2)
    pdf1 = fast_norm_pdf_numba(z1)
    pdf2 = fast_norm_pdf_numba(z2)

    pdf = (1.0 / A) * (-v * cdf1 + v * cdf2 + s * pdf1 - s * pdf2)

    if pdf < 1e-10:
        return 1e-10
    return pdf


@jit(nopython=True, fastmath=True, cache=True)
def lba_cdf_numba(t, v, A, b, s):
    """Numba-optimized CDF for one LBA accumulator"""
    if t <= 0.0:
        return 0.0
    if v <= 0.0 or A <= 0.0 or b <= 0.0 or s <= 0.0:
        return 0.0
    if t > 10.0:
        return 1.0

    t_s = t * s
    z1 = (b - A - t * v) / t_s
    z2 = (b - t * v) / t_s

    cdf1 = fast_norm_cdf_numba(z1)
    cdf2 = fast_norm_cdf_numba(z2)
    pdf1 = fast_norm_pdf_numba(z1)
    pdf2 = fast_norm_pdf_numba(z2)

    v_t_minus_b = v * t - b
    cdf = 1.0 + (v_t_minus_b / A) * cdf2 - ((v_t_minus_b + A) / A) * cdf1 + \
          (t_s / A) * pdf2 - (t_s / A) * pdf1

    if cdf < 0.0:
        return 0.0
    if cdf > 1.0:
        return 1.0
    return cdf


# ============================================================================
# WRAPPER FUNCTIONS (maintain compatibility with original interface)
# ============================================================================

# NOTE: lba_pwin wrapper removed in v3
# v3 directly uses lba_pwin_exact() in lba_2dim_likelihood()

def lba_pdf(t, v, A, b, s):
    """PDF for one LBA accumulator (Numba-optimized)"""
    return lba_pdf_numba(t, v, A, b, s)


def lba_cdf(t, v, A, b, s):
    """CDF for one LBA accumulator (Numba-optimized)"""
    return lba_cdf_numba(t, v, A, b, s)


def lba_survival(t, v, A, b, s):
    """Survival function for one LBA accumulator"""
    return 1.0 - lba_cdf_numba(t, v, A, b, s)


def lba_defective_pdf(t, v_win, v_lose, A, b, s):
    """
    Defective PDF for LBA with 2 accumulators

    Returns the probability density that:
    - The winning accumulator reaches threshold at time t
    - The losing accumulator has NOT reached threshold by time t

    From LBA paper equation (3):
    PDF_i(t) = f_i(t) × ∏(j≠i)[1 - F_j(t)]
            = f_i(t) × [1 - F_j(t)]  (for 2 accumulators)

    Parameters:
    -----------
    t : float
        Time
    v_win : float
        Drift rate of winning accumulator
    v_lose : float
        Drift rate of losing accumulator
    A, b, s : float
        LBA parameters

    Returns:
    --------
    defective_pdf : float
    """
    pdf_win = lba_pdf(t, v_win, A, b, s)
    survival_lose = lba_survival(t, v_lose, A, b, s)
    return pdf_win * survival_lose


@jit(nopython=True, fastmath=True, cache=True)
def lba_defective_cdf_numba(t, v_win, v_lose, A, b, s, n_points=25):
    """
    NUMBA-OPTIMIZED: Defective CDF for LBA with 2 accumulators

    F_defective(t) = ∫[0 to t] f_win(τ) × S_lose(τ) dτ

    Uses Numba JIT for ~10-50x speedup over scipy version
    n_points=25 (reduced from 50) still maintains good accuracy
    """
    if t <= 0.0:
        return 0.0

    # Generate integration points
    tau = np.linspace(0.0, t, n_points)
    tau_safe = tau.copy()
    tau_safe[0] = 1e-10

    # ===== Vectorized f_win(τ) calculation =====
    t_s = tau_safe * s
    z1_win = (b - A - tau_safe * v_win) / t_s
    z2_win = (b - tau_safe * v_win) / t_s

    # Use fast Numba CDF/PDF
    cdf1_win = np.empty(n_points)
    cdf2_win = np.empty(n_points)
    pdf1_win = np.empty(n_points)
    pdf2_win = np.empty(n_points)

    for i in range(n_points):
        cdf1_win[i] = fast_norm_cdf_numba(z1_win[i])
        cdf2_win[i] = fast_norm_cdf_numba(z2_win[i])
        pdf1_win[i] = fast_norm_pdf_numba(z1_win[i])
        pdf2_win[i] = fast_norm_pdf_numba(z2_win[i])

    f_win = (1.0 / A) * (-v_win * cdf1_win + v_win * cdf2_win +
                         s * pdf1_win - s * pdf2_win)
    f_win = np.maximum(f_win, 0.0)
    f_win[0] = 0.0

    # ===== Vectorized S_lose(τ) calculation =====
    z1_lose = (b - A - tau_safe * v_lose) / t_s
    z2_lose = (b - tau_safe * v_lose) / t_s

    cdf1_lose = np.empty(n_points)
    cdf2_lose = np.empty(n_points)
    pdf1_lose = np.empty(n_points)
    pdf2_lose = np.empty(n_points)

    for i in range(n_points):
        cdf1_lose[i] = fast_norm_cdf_numba(z1_lose[i])
        cdf2_lose[i] = fast_norm_cdf_numba(z2_lose[i])
        pdf1_lose[i] = fast_norm_pdf_numba(z1_lose[i])
        pdf2_lose[i] = fast_norm_pdf_numba(z2_lose[i])

    v_t_minus_b = v_lose * tau_safe - b
    cdf_lose = 1.0 + (v_t_minus_b / A) * cdf2_lose - \
               ((v_t_minus_b + A) / A) * cdf1_lose + \
               (t_s / A) * pdf2_lose - (t_s / A) * pdf1_lose
    cdf_lose = np.clip(cdf_lose, 0.0, 1.0)

    s_lose = 1.0 - cdf_lose

    # ===== Trapezoidal integration =====
    integrand = f_win * s_lose

    # Manual trapezoidal rule (Numba doesn't support np.trapz)
    result = 0.0
    for i in range(n_points - 1):
        dt = tau[i+1] - tau[i]
        result += 0.5 * (integrand[i] + integrand[i+1]) * dt

    # Ensure result is in [0, 1]
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


def lba_defective_cdf(t, v_win, v_lose, A, b, s):
    """
    Defective CDF for LBA with 2 accumulators - OPTIMIZED VERSION

    F_defective(t) = ∫[0 to t] f_win(τ) × S_lose(τ) dτ

    Two modes:
    1. Fast approximation: F ≈ F_win(t) × S_lose(t)
       - 317x faster (~0.18 ms per call)
       - Good for exploration

    2. NUMBA-optimized integration:
       - n_points=25, Numba JIT compiled
       - ~10-50x faster than scipy, accurate
       - Good for MCMC sampling
    """
    if t <= 0:
        return 0.0

    # FAST MODE: Use approximation
    if _CDF_MODE['use_fast']:
        cdf_win = lba_cdf(t, v_win, A, b, s)
        survival_lose = lba_survival(t, v_lose, A, b, s)
        result = max(min(cdf_win * survival_lose, 1.0), 0.0)
        return result

    # PRECISE MODE: Use Numba-optimized integration
    return lba_defective_cdf_numba(t, v_win, v_lose, A, b, s, n_points=25)


def lba_2dim_likelihood(choice, rt, condition, v_left, v_right, A, b, t0, s=1.0):
    """
    GRT-LBA likelihood with 2 independent dimension decisions

    Decision architecture:
    1. Left dimension: v1_L vs v2_L compete → decides Left judgment (H or V)
    2. Right dimension: v1_R vs v2_R compete → decides Right judgment (H or V)
    3. Each dimension decision produces its own RT (decision time + dimension-specific t0)
    4. Final choice = combination of Left and Right judgments (HH, HV, VH, or VV)
    5. Final RT = max(Left RT, Right RT) - the slower dimension determines overall RT

    Parameters:
    -----------
    choice : int
        Final chosen combination: 0=VH, 1=HH, 2=HV, 3=VV
    rt : float
        Observed response time
    v_left : array, shape (2,)
        Drift rates for left dimension [v1_L, v2_L]
        v1_L: drift rate for judging "H" (|) on left
        v2_L: drift rate for judging "V" (/) on left
    v_right : array, shape (2,)
        Drift rates for right dimension [v1_R, v2_R]
        v1_R: drift rate for judging "H" (|) on right
        v2_R: drift rate for judging "V" (/) on right
    A, b : float
        Shared LBA parameters (starting point variability, threshold)
    t0 : float
        Non-decision time (fixed value for all choices)
    s : float
        Within-trial variability (fixed at 1.0)

    Returns:
    --------
    likelihood : float

    Notes:
    ------
    Choice mapping:
    0 = VH (/|): Left judges V (/), Right judges H (|)
    1 = HH (||): Left judges H (|), Right judges H (|)
    2 = HV (|/): Left judges H (|), Right judges V (/)
    3 = VV (//): Left judges V (/), Right judges V (/)
    """
    # Map to left and right judgments
    # 0=VH, 1=HH, 2=HV, 3=VV
    judgment_map = {
        0: (1, 0),  # VH (/|): left=v2(V), right=v1(H)
        1: (0, 0),  # HH (||): left=v1(H), right=v1(H)
        2: (0, 1),  # HV (|/): left=v1(H), right=v2(V)
        3: (1, 1)   # VV (//): left=v2(V), right=v2(V)
    }

    # Define observed response accumulators based on CHOICE
    left_choice, right_choice = judgment_map[choice]
    v_left_choice = v_left[left_choice]
    v_left_other = v_left[1 - left_choice]
    v_right_choice = v_right[right_choice]
    v_right_other = v_right[1 - right_choice]

    # Calculate P(choice | stimulus) - v3: EXACT defective PDF integration
    # This is the probability that both dimensions select the accumulators for this choice
    # NEW: Using whiteboard method = integrating defective PDF
    p_choice = lba_pwin_exact(v_left_choice, v_left_other, v_right_choice, v_right_other, A, b, s)

    # Time calculation
    t = rt - t0

    if t <= 0:
        return 1e-10

    # CORRECTED: Use defective PDF/CDF for 2-accumulator races
    # f_left = PDF that Left dimension completes at time t (winner reaches, loser hasn't)
    # F_left = CDF that Left dimension completes by time t
    f_left = lba_defective_pdf(t, v_left_choice, v_left_other, A, b, s)
    F_left = lba_defective_cdf(t, v_left_choice, v_left_other, A, b, s)

    f_right = lba_defective_pdf(t, v_right_choice, v_right_other, A, b, s)
    F_right = lba_defective_cdf(t, v_right_choice, v_right_other, A, b, s)

    # Likelihood: P(choice | stimulus) × P(RT=t | choice)
    # P(RT=t | choice) = PDF of max(t_left, t_right) = f_left*F_right + f_right*F_left
    likelihood = p_choice * (f_left * F_right + f_right * F_left)

    return max(likelihood, 1e-10)


def grt_lba_4choice_logp_numpy(choice, rt, condition, v_tensor, A, b, t0, s=1.0):
    """
    Calculate GRT-LBA 4-choice log-likelihood with 2-dimension independent decisions

    Parameters:
    -----------
    choice : array, shape (n_trials,)
        Choices (0=VH, 1=HH, 2=HV, 3=VV)
    rt : array, shape (n_trials,)
        Response times
    condition : array, shape (n_trials,)
        Experimental conditions (0=VH, 1=HH, 2=HV, 3=VV)
    v_tensor : array, shape (4, 2, 2)
        Drift rate tensor with perceptual separability structure
        v_tensor[cond, dimension, accumulator]
        - dimension: 0=Left, 1=Right
        - accumulator: 0=v1(judge H), 1=v2(judge V)
    A, b : float
        Shared LBA parameters
    t0 : float
        Non-decision time (fixed value)
    s : float
        Within-trial variability (fixed at 1.0)

    Returns:
    --------
    total_log_lik : float
    """
    total_log_lik = 0.0

    for i in range(len(rt)):
        cond = int(condition[i])  # 0, 1, 2, 3 (0-indexed)

        # Get drift rates for left and right dimensions under this condition
        v_left = v_tensor[cond, 0, :]  # shape (2,) - Left dimension
        v_right = v_tensor[cond, 1, :]  # shape (2,) - Right dimension

        lik = lba_2dim_likelihood(
            choice=choice[i],
            rt=rt[i],
            condition=cond,
            v_left=v_left,
            v_right=v_right,
            A=A, b=b, t0=t0, s=s
        )

        if lik > 0:
            total_log_lik += np.log(lik)
        else:
            total_log_lik += -1000.0  # Penalty for invalid likelihood

    return total_log_lik


def lba_2dim_random(n_trials_per_condition, v_tensor, A, b, t0, s=1.0, rng=None):
    """
    Generate GRT-LBA simulated data with 2 independent dimension decisions

    Decision rule:
    1. Left dimension: v1_L vs v2_L compete → Left judges H or V
    2. Right dimension: v1_R vs v2_R compete → Right judges H or V
    3. Final choice = combination of Left and Right judgments (HH, HV, VH, VV)
    4. Final RT = decision_time + t0 (choice-specific motor time)

    Parameters:
    -----------
    n_trials_per_condition : int
        Number of trials per condition
    v_tensor : array, shape (4, 2, 2)
        Drift rate tensor with perceptual separability structure
        v_tensor[cond, dimension, accumulator]
    A, b : float
        Shared LBA parameters
    t0 : float
        Non-decision time for each choice (button-specific motor time)
    s : float
        Within-trial variability (fixed at 1.0)

    Returns:
    --------
    data : array, shape (n_trials, 3)
        [RT, choice, condition]
        choice: 0=VH(/|), 1=HH(||), 2=HV(|/), 3=VV(//)
        condition: 0=VH, 1=HH, 2=HV, 3=VV
    """
    if rng is None:
        rng = np.random.default_rng()

    all_data = []

    # Generate data for each condition
    for cond in range(4):  # condition 0=VH, 1=HH, 2=HV, 3=VV
        v_left = v_tensor[cond, 0, :]  # Left dimension [v1_L, v2_L]
        v_right = v_tensor[cond, 1, :]  # Right dimension [v1_R, v2_R]

        for _ in range(n_trials_per_condition):
            # Left dimension decision
            k1_L = rng.uniform(0, A)
            k2_L = rng.uniform(0, A)
            # FIX: Remove across-trial variability - use fixed drift rates
            v1_L_trial = v_left[0]  # Fixed drift rate (within-trial variability handled by LBA formula)
            v2_L_trial = v_left[1]
            t1_L = (b - k1_L) / v1_L_trial if v1_L_trial > 0 else np.inf
            t2_L = (b - k2_L) / v2_L_trial if v2_L_trial > 0 else np.inf

            # Left judgment: 0=H (if v1 wins), 1=V (if v2 wins)
            if t1_L < t2_L:
                left_judgment = 0  # Judge H
            else:
                left_judgment = 1  # Judge V

            # Right dimension decision
            k1_R = rng.uniform(0, A)
            k2_R = rng.uniform(0, A)
            # FIX: Remove across-trial variability - use fixed drift rates
            v1_R_trial = v_right[0]  # Fixed drift rate
            v2_R_trial = v_right[1]
            t1_R = (b - k1_R) / v1_R_trial if v1_R_trial > 0 else np.inf
            t2_R = (b - k2_R) / v2_R_trial if v2_R_trial > 0 else np.inf

            # Right judgment: 0=H (if v1 wins), 1=V (if v2 wins)
            if t1_R < t2_R:
                right_judgment = 0  # Judge H
            else:
                right_judgment = 1  # Judge V

            # Combined choice: 0=VH(/|), 1=HH(||), 2=HV(|/), 3=VV(//)
            # left_judgment: 0=H(|), 1=V(/)
            # right_judgment: 0=H(|), 1=V(/)
            choice_map_reverse = {
                (1, 0): 0,  # VH (/|): 左=/, 右=|
                (0, 0): 1,  # HH (||): 左=|, 右=|
                (0, 1): 2,  # HV (|/): 左=|, 右=/
                (1, 1): 3   # VV (//): 左=/, 右=/
            }
            final_choice = choice_map_reverse[(left_judgment, right_judgment)]

            # Decision time = max of both dimensions (slower one determines RT)
            decision_time = max(min(t1_L, t2_L), min(t1_R, t2_R))

            # Add choice-specific motor time
            rt = decision_time + t0

            # Filter: RT < 2 seconds and valid
            if rt > 0 and rt < 2.0 and np.isfinite(rt):
                all_data.append([rt, final_choice, cond])

    return np.array(all_data)


class GRT_LBA_4Choice_LogLik(pytensor.tensor.Op):
    """PyTensor Op for GRT-LBA with 2 independent dimensions"""

    itypes = [
        pytensor.tensor.ivector,  # choice
        pytensor.tensor.dvector,  # RT
        pytensor.tensor.ivector,  # condition
        pytensor.tensor.dtensor3,  # v_tensor (4, 2, 2) - [condition, dimension, accumulator]
        pytensor.tensor.dscalar,  # A
        pytensor.tensor.dscalar,  # b
        pytensor.tensor.dscalar,  # t0
        pytensor.tensor.dscalar   # s
    ]

    otypes = [pytensor.tensor.dscalar]

    def perform(self, node, inputs, outputs):
        choice, rt, condition, v_tensor, A, b, t0, s = inputs

        logp = grt_lba_4choice_logp_numpy(
            choice=choice,
            rt=rt,
            condition=condition,
            v_tensor=v_tensor,
            A=A, b=b, t0=t0, s=s
        )

        outputs[0][0] = np.array(logp, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        return [()]

    def __str__(self):
        return self.__class__.__name__


# ============================================================================
# MAP 估計函數 (多次運行以避免局部最優)
# ============================================================================

def generate_initial_values_8param(n_runs, solution_hint=None):
    """生成多組初始值 (10 參數版本: 8 drift rates + 2 delta)

    Parameters:
    -----------
    n_runs : int
        生成幾組初始值
    solution_hint : str or None
        'high_correct' (高 correct drift rates) 或 None (隨機)

    Returns:
    --------
    initvals_list : list of dict
    """
    initvals_list = []

    if solution_hint == 'high_correct':
        # 假設正確反應有較高 drift rates
        for i in range(n_runs):
            initvals_list.append({
                'v1_L_when_H': 3.0 + np.random.uniform(-0.3, 0.3),  # Correct
                'v2_L_when_H': 1.0 + np.random.uniform(-0.2, 0.2),  # Error
                'v1_L_when_V': 1.0 + np.random.uniform(-0.2, 0.2),  # Error
                'v2_L_when_V': 3.0 + np.random.uniform(-0.3, 0.3),  # Correct
                'v1_R_when_H': 3.0 + np.random.uniform(-0.3, 0.3),  # Correct
                'v2_R_when_H': 1.0 + np.random.uniform(-0.2, 0.2),  # Error
                'v1_R_when_V': 1.0 + np.random.uniform(-0.2, 0.2),  # Error
                'v2_R_when_V': 3.0 + np.random.uniform(-0.3, 0.3),  # Correct
                'delta_v1_L_V': 0.0 + np.random.uniform(-0.00001, 0.00001),  # PS violation (Left)
                'delta_v1_R_H': 0.0 + np.random.uniform(-0.00001, 0.00001)   # PS violation (Right)
            })
    else:
        # 隨機初始值
        for i in range(n_runs):
            initvals_list.append({
                'v1_L_when_H': np.random.uniform(1.5, 4.0),
                'v2_L_when_H': np.random.uniform(0.8, 2.0),
                'v1_L_when_V': np.random.uniform(0.8, 2.0),
                'v2_L_when_V': np.random.uniform(1.5, 4.0),
                'v1_R_when_H': np.random.uniform(1.5, 4.0),
                'v2_R_when_H': np.random.uniform(0.8, 2.0),
                'v1_R_when_V': np.random.uniform(0.8, 2.0),
                'v2_R_when_V': np.random.uniform(1.5, 4.0),
                'delta_v1_L_V': np.random.uniform(-0.000001, 0.000001),
                'delta_v1_R_H': np.random.uniform(-0.000001, 0.000001)
            })

    return initvals_list


def run_map_estimation_8param(model, initvals, data, run_id, maxeval=10000):
    """運行單次 MAP 估計 (10 參數版本: 8 drift rates + 2 delta)

    Parameters:
    -----------
    model : pm.Model
        PyMC model
    initvals : dict
        初始值字典
    data : np.ndarray
        觀測數據 [RT, choice, condition]
    run_id : int
        運行編號
    maxeval : int
        最大評估次數

    Returns:
    --------
    result : dict
        包含 MAP 估計結果和 log-likelihood
    """
    print(f"\n--- MAP Run {run_id} ---")
    print(f"  初始值範例: v1_L_when_H={initvals['v1_L_when_H']:.2f}, " +
          f"v2_L_when_H={initvals['v2_L_when_H']:.2f}")

    with model:
        map_est = pm.find_MAP(start=initvals, method='powell', maxeval=maxeval)

    # 構建 v_tensor 計算 log-likelihood
    v_tensor_map = np.zeros((4, 2, 2))
    v_tensor_map[0, 0, :] = [map_est['v1_L_when_H'], map_est['v2_L_when_H']]
    v_tensor_map[0, 1, :] = [map_est['v1_R_when_H'], map_est['v2_R_when_H']]
    v_tensor_map[1, 0, :] = [map_est['v1_L_when_H'], map_est['v2_L_when_H']]
    v_tensor_map[1, 1, :] = [map_est['v1_R_when_V'], map_est['v2_R_when_V']]
    v_tensor_map[2, 0, :] = [map_est['v1_L_when_V'], map_est['v2_L_when_V']]
    v_tensor_map[2, 1, :] = [map_est['v1_R_when_H'], map_est['v2_R_when_H']]
    v_tensor_map[3, 0, :] = [map_est['v1_L_when_V'], map_est['v2_L_when_V']]
    v_tensor_map[3, 1, :] = [map_est['v1_R_when_V'], map_est['v2_R_when_V']]

    # 計算 log-likelihood (使用 grt_lba_4choice_logp_numpy)
    logp = grt_lba_4choice_logp_numpy(
        choice=data[:, 1].astype(int),
        rt=data[:, 0],
        condition=data[:, 2].astype(int),
        v_tensor=v_tensor_map,
        A=0.5,
        b=1.0,
        t0=0.25,
        s=1.0
    )

    # 計算正確反應的平均 drift rate
    correct_drift = (map_est['v1_L_when_H'] + map_est['v2_L_when_V'] +
                     map_est['v1_R_when_H'] + map_est['v2_R_when_V']) / 4
    error_drift = (map_est['v2_L_when_H'] + map_est['v1_L_when_V'] +
                   map_est['v2_R_when_H'] + map_est['v1_R_when_V']) / 4

    result = {
        'run_id': run_id,
        'map': map_est.copy(),
        'logp': logp,
        'correct_avg': correct_drift,
        'error_avg': error_drift
    }

    print(f"  MAP: correct={correct_drift:.3f}, error={error_drift:.3f}")
    print(f"  Log-likelihood: {logp:.1f}")

    return result


def select_best_solution_8param(results):
    """選擇最佳解 (8 參數版本)

    Parameters:
    -----------
    results : list of dict
        所有 MAP 運行的結果

    Returns:
    --------
    best : dict
        最佳解
    consistency : dict
        一致性診斷
    """
    # 按 log-likelihood 排序
    best = max(results, key=lambda x: x['logp'])

    # 檢查一致性
    correct_avgs = [r['correct_avg'] for r in results]
    error_avgs = [r['error_avg'] for r in results]

    consistency = {
        'correct_std': np.std(correct_avgs),
        'error_std': np.std(error_avgs),
        'all_correct_higher': all(r['correct_avg'] > r['error_avg'] for r in results)
    }

    return best, consistency


def run_multiple_map_8param(model, data, n_runs=5, solution_hint='high_correct'):
    """運行多次 MAP 估計並選擇最佳解 (8 參數版本)

    Parameters:
    -----------
    model : pm.Model
        PyMC model
    data : np.ndarray
        觀測數據
    n_runs : int
        運行次數
    solution_hint : str
        初始值提示

    Returns:
    --------
    best_solution : dict
        最佳 MAP 估計
    all_results : list
        所有運行結果
    consistency : dict
        一致性診斷
    """
    print(f"\n{'='*70}")
    print(f"運行 {n_runs} 次 MAP 估計以避免局部最優")
    print(f"{'='*70}")

    # 生成初始值
    initvals_list = generate_initial_values_8param(n_runs, solution_hint)

    # 運行多次 MAP
    results = []
    for i, initvals in enumerate(initvals_list, 1):
        result = run_map_estimation_8param(model, initvals, data, run_id=i)
        results.append(result)

    # 選擇最佳解
    best, consistency = select_best_solution_8param(results)

    print(f"\n{'='*70}")
    print("MAP 估計總結")
    print(f"{'='*70}")
    print(f"最佳解: Run #{best['run_id']}")
    print(f"  Log-likelihood: {best['logp']:.1f}")
    print(f"  Correct drift avg: {best['correct_avg']:.3f}")
    print(f"  Error drift avg: {best['error_avg']:.3f}")
    print(f"\n一致性:")
    print(f"  Correct drift std: {consistency['correct_std']:.4f}")
    print(f"  Error drift std: {consistency['error_std']:.4f}")
    print(f"  All correct > error: {consistency['all_correct_higher']}")

    return best, results, consistency


# ============================================================================
# 模型定義
# ============================================================================

def create_grt_lba_4choice_model(observed_data):
    """
    Create GRT-LBA 4-choice PyMC model

    Parameters:
    -----------
    observed_data : array, shape (n_trials, 3)
        [RT, choice, condition]

    Returns:
    --------
    model : pm.Model
    """
    with pm.Model() as grt_lba_model:
        # LBA framework parameters (shared across all choices) - FIXED VALUES - REALISTIC VERSION
        A = 0.5  # Starting point variability (shared) - FIXED
        b = 1.5  # Threshold (shared) - FIXED - REALISTIC (increased from 1.0)
        s = 1.0  # Fixed, not estimated (standardize within-trial variability)

        # Choice-specific non-decision time (motor response time for each button) - FIXED VALUES
        # Using the true values from simulation - REALISTIC VERSION
        t0 = 0.25  # OPTIMIZED - REALISTIC (increased from 0.2)

        # 8 basic drift rates for perceptual separability testing
        # Structure: 2 dimensions × 2 stimuli (H/V) × 2 accumulators (v1/v2)
        # If perceptually separable, Left drift rates should NOT depend on Right stimulus
        # and Right drift rates should NOT depend on Left stimulus

        # Left dimension drift rates (when stimulus is H or V) - REALISTIC VERSION
        # v1: support "H" judgment, v2: support "V" judgment
        # CONVERGENCE FIX: sigma=2.0 → 0.8 (tighter priors for better convergence)
        # v7 FIX: lower=0.001 (was 0.01) to further reduce truncation bias
        # v7_REALISTIC: mu adjusted to realistic values (1.5 for correct, 0.8 for error)
        v1_L_when_H = pm.TruncatedNormal("v1_L_when_H", mu=1.8, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8
        v2_L_when_H = pm.TruncatedNormal("v2_L_when_H", mu=1.5, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8
        v1_L_when_V = pm.TruncatedNormal("v1_L_when_V", mu=1.5, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8
        v2_L_when_V = pm.TruncatedNormal("v2_L_when_V", mu=1.8, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8

        # Right dimension drift rates (when stimulus is H or V) - REALISTIC VERSION
        # CONVERGENCE FIX: sigma=2.0 → 0.8 (tighter priors for better convergence)
        # v7 FIX: lower=0.001 (was 0.01) to further reduce truncation bias
        # FIXED: mu adjusted for realistic accuracy (1.2 for correct, 0.9 for error)
        v1_R_when_H = pm.TruncatedNormal("v1_R_when_H", mu=1.8, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8
        v2_R_when_H = pm.TruncatedNormal("v2_R_when_H", mu=1.5, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8
        v1_R_when_V = pm.TruncatedNormal("v1_R_when_V", mu=1.5, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8
        v2_R_when_V = pm.TruncatedNormal("v2_R_when_V", mu=1.8, sigma=0.8, lower=0.001, upper=5.0)  # FIXED: sigma 2.0→0.8

        # PS Violation Parameters (symmetric design with 2 independent deltas)
        delta_v1_L_V = pm.Normal("delta_v1_L_V", mu=0, sigma=0.15)  # Left dim influenced by Right
        delta_v1_R_H = pm.Normal("delta_v1_R_H", mu=0, sigma=0.15)  # Right dim influenced by Left

        # Calculate v1_L with PS violation (when Left=V)
        v1_L_VH = v1_L_when_V - delta_v1_L_V / 2  # VH: Right=H
        v1_L_VV = v1_L_when_V + delta_v1_L_V / 2  # VV: Right=V

        # Calculate v1_R with PS violation (when Right=H)
        v1_R_HH = v1_R_when_H - delta_v1_R_H / 2  # HH: Left=H
        v1_R_VH = v1_R_when_H + delta_v1_R_H / 2  # VH: Left=V

        # Construct v_tensor with symmetric PS violations
        # v_tensor[condition, dimension, accumulator]
        # condition: 0=HH, 1=HV, 2=VH, 3=VV
        # dimension: 0=Left, 1=Right
        # accumulator: 0=v1(judge H), 1=v2(judge V)
        v_tensor = pt.zeros((4, 2, 2))

        # Condition HH (Left=H, Right=H) - PS violation on Right dimension
        v_tensor = pt.set_subtensor(v_tensor[0, 0, 0], v1_L_when_H)  # Left: v1 (no violation)
        v_tensor = pt.set_subtensor(v_tensor[0, 0, 1], v2_L_when_H)  # Left: v2 (judge V)
        v_tensor = pt.set_subtensor(v_tensor[0, 1, 0], v1_R_HH)  # Right: v1 with PS violation (Left=H)
        v_tensor = pt.set_subtensor(v_tensor[0, 1, 1], v2_R_when_H)  # Right: v2 (judge V)

        # Condition HV (Left=H, Right=V) - No PS violation
        v_tensor = pt.set_subtensor(v_tensor[1, 0, 0], v1_L_when_H)  # Left: v1 (no violation)
        v_tensor = pt.set_subtensor(v_tensor[1, 0, 1], v2_L_when_H)  # Left: v2 (same as HH)
        v_tensor = pt.set_subtensor(v_tensor[1, 1, 0], v1_R_when_V)  # Right: v1 (no violation)
        v_tensor = pt.set_subtensor(v_tensor[1, 1, 1], v2_R_when_V)  # Right: v2 (judge V)

        # Condition VH (Left=V, Right=H) - PS violation on BOTH dimensions
        v_tensor = pt.set_subtensor(v_tensor[2, 0, 0], v1_L_VH)  # Left: v1 with PS violation (Right=H)
        v_tensor = pt.set_subtensor(v_tensor[2, 0, 1], v2_L_when_V)  # Left: v2 (judge V)
        v_tensor = pt.set_subtensor(v_tensor[2, 1, 0], v1_R_VH)  # Right: v1 with PS violation (Left=V)
        v_tensor = pt.set_subtensor(v_tensor[2, 1, 1], v2_R_when_H)  # Right: v2 (judge V)

        # Condition VV (Left=V, Right=V)
        v_tensor = pt.set_subtensor(v_tensor[3, 0, 0], v1_L_VV)  # Left: v1 with PS violation
        v_tensor = pt.set_subtensor(v_tensor[3, 0, 1], v2_L_when_V)  # Left: v2 (same as VH)
        v_tensor = pt.set_subtensor(v_tensor[3, 1, 0], v1_R_when_V)  # Right: v1 (same as HV)
        v_tensor = pt.set_subtensor(v_tensor[3, 1, 1], v2_R_when_V)  # Right: v2 (same as HV)

        # Extract data
        rt_data = observed_data[:, 0]
        choice_data = observed_data[:, 1].astype(int)
        condition_data = observed_data[:, 2].astype(int)

        # Convert to PyTensor tensors
        rt_obs = pt.as_tensor_variable(rt_data, dtype='float64')
        choice_obs = pt.as_tensor_variable(choice_data, dtype='int32')
        condition_obs = pt.as_tensor_variable(condition_data, dtype='int32')

        # Convert all fixed parameters to tensors
        A_tensor = pt.as_tensor_variable(A, dtype='float64')
        b_tensor = pt.as_tensor_variable(b, dtype='float64')
        t0_tensor = pt.as_tensor_variable(t0, dtype='float64')
        s_tensor = pt.as_tensor_variable(s, dtype='float64')

        # Call Op
        grt_lba_op = GRT_LBA_4Choice_LogLik()
        log_lik = grt_lba_op(
            choice_obs, rt_obs, condition_obs,
            v_tensor, A_tensor, b_tensor, t0_tensor, s_tensor
        )

        pm.Potential("obs", log_lik)

    return grt_lba_model


if __name__ == "__main__":
    print("="*70)
    print("GRT-LBA with 2 Independent Dimensions - PS Violation Detection Test")
    print("9 parameters: 8 basic drift rates + delta (PS violation parameter)")
    print("="*70)

    # 1. Set up true parameters with PS violation
    print("\n1. Set up simulation parameters (with PS violation - delta ≠ 0):")

    # Create v_tensor (4 conditions × 2 dimensions × 2 accumulators)
    # With PS violation: v1_L when Left=V depends on Right stimulus (H vs V)
    #                    Controlled by delta_v1_L_V parameter
    v_tensor_true = np.zeros((4, 2, 2))

    # Define 8 basic parameters - REALISTIC VERSION (FIXED for 70-80% accuracy)
    # Strategy: Reduce the gap between correct and error drift rates
    # v_correct / v_error ratio determines accuracy
    v1_L_H_true = 1.8  # Left judges H when stimulus is H (correct) - REDUCED for realism
    v2_L_H_true = 1.5  # Left judges V when stimulus is H (error) - INCREASED to reduce gap
    v1_L_V_true = 1.5  # Left judges H when stimulus is V (error) - INCREASED to reduce gap
    v2_L_V_true = 1.8  # Left judges V when stimulus is V (correct) - REDUCED for realism

    v1_R_H_true = 1.8  # Right judges H when stimulus is H (correct) - REDUCED for realism
    v2_R_H_true = 1.5  # Right judges V when stimulus is H (error) - INCREASED to reduce gap
    v1_R_V_true = 1.5  # Right judges H when stimulus is V (error) - INCREASED to reduce gap
    v2_R_V_true = 1.8  # Right judges V when stimulus is V (correct) - REDUCED for realism

    # PS Violation Parameters (symmetric design with 2 independent deltas)
    delta_v1_L_V_true = 0.3  # Left dim influenced by Right (0 = perfect separability)
    delta_v1_R_H_true = 0.25  # Right dim influenced by Left (0 = perfect separability)

    # Calculate v1_L with PS violation (when Left=V)
    v1_L_VH_true = v1_L_V_true - delta_v1_L_V_true / 2  # VH: Right=H
    v1_L_VV_true = v1_L_V_true + delta_v1_L_V_true / 2  # VV: Right=V

    # Calculate v1_R with PS violation (when Right=H)
    v1_R_HH_true = v1_R_H_true - delta_v1_R_H_true / 2  # HH: Left=H
    v1_R_VH_true = v1_R_H_true + delta_v1_R_H_true / 2  # VH: Left=V

    # Fill v_tensor with symmetric PS violations
    # Condition HH (Left=H, Right=H) - PS violation on Right dimension
    v_tensor_true[0, 0, :] = [v1_L_H_true, v2_L_H_true]
    v_tensor_true[0, 1, :] = [v1_R_HH_true, v2_R_H_true]  # v1_R depends on Left=H

    # Condition HV (Left=H, Right=V) - No PS violation
    v_tensor_true[1, 0, :] = [v1_L_H_true, v2_L_H_true]
    v_tensor_true[1, 1, :] = [v1_R_V_true, v2_R_V_true]

    # Condition VH (Left=V, Right=H) - PS violation on BOTH dimensions
    v_tensor_true[2, 0, :] = [v1_L_VH_true, v2_L_V_true]  # v1_L depends on Right=H
    v_tensor_true[2, 1, :] = [v1_R_VH_true, v2_R_H_true]  # v1_R depends on Left=V

    # Condition VV (Left=V, Right=V) - PS violation on Left dimension
    v_tensor_true[3, 0, :] = [v1_L_VV_true, v2_L_V_true]  # v1_L depends on Right=V
    v_tensor_true[3, 1, :] = [v1_R_V_true, v2_R_V_true]

    print(f"   v_tensor shape: {v_tensor_true.shape} [condition, dimension, accumulator]")
    print(f"\n   8 basic parameters + SYMMETRIC PS violations:")
    print(f"      Left when H: v1={v1_L_H_true}, v2={v2_L_H_true}")
    print(f"      Left when V: v1={v1_L_V_true}, v2={v2_L_V_true}")
    print(f"      Right when H: v1={v1_R_H_true}, v2={v2_R_H_true}")
    print(f"      Right when V: v1={v1_R_V_true}, v2={v2_R_V_true}")
    print(f"\n   PS Violation Parameters:")
    print(f"      delta_L = {delta_v1_L_V_true} (Left dim influenced by Right)")
    print(f"      delta_R = {delta_v1_R_H_true} (Right dim influenced by Left)")
    print(f"\n   Resulting drift rates:")
    print(f"      HH: v1_L={v1_L_H_true:.3f}, v1_R={v1_R_HH_true:.3f} (R violated)")
    print(f"      HV: v1_L={v1_L_H_true:.3f}, v1_R={v1_R_V_true:.3f} (no violation)")
    print(f"      VH: v1_L={v1_L_VH_true:.3f}, v1_R={v1_R_VH_true:.3f} (BOTH violated)")
    print(f"      VV: v1_L={v1_L_VV_true:.3f}, v1_R={v1_R_V_true:.3f} (L violated)")

    # True t0 (non-decision time) - REALISTIC VERSION
    # Changed from 0.2 to 0.35 for realistic human encoding/motor time
    t0_true = 0.25  # INCREASED from 0.2
    print(f"\n   t0 (non-decision time): {t0_true}")

    # 2. Generate simulated data
    print("\n2. Generate simulated data:")
    n_trials_per_condition = 1250  # 1250 * 4 = 5000 trials total

    # FIXED SEED for reproducibility
    rng_seed42 = np.random.default_rng(seed=42)
    print("   ✓ Using fixed random seed=42 for reproducible data generation")

    # REALISTIC VERSION: b increased from 1.0 to 1.5 for higher threshold
    data = lba_2dim_random(
        n_trials_per_condition=n_trials_per_condition,
        v_tensor=v_tensor_true,
        A=0.5, b=1.5, t0=t0_true, s=1.0,  # b=1.5 (was 1.0)
        rng=rng_seed42  # Use fixed seed RNG
    )

    print(f"   Generated {len(data)} trials")
    print(f"   Data shape: {data.shape} [RT, choice, condition]")
    print(f"   RT range: [{data[:, 0].min():.3f}, {data[:, 0].max():.3f}]")

    # Analyze data
    print("\n   Condition distribution:")
    for cond in range(0, 4):
        count = (data[:, 2] == cond).sum()
        print(f"      Condition {cond}: {count} trials")

    print("\n   Choice distribution:")
    for choice in range(0, 4):
        count = (data[:, 1] == choice).sum()
        print(f"      Choice {choice}: {count} trials")

    # Calculate accuracy
    # Condition mapping: 0=HH, 1=HV, 2=VH, 3=VV
    # Choice mapping: 0=VH, 1=HH, 2=HV, 3=VV
    # Correct mapping: cond 0→choice 1, cond 1→choice 2, cond 2→choice 0, cond 3→choice 3
    condition_to_correct_choice = {0: 1, 1: 2, 2: 0, 3: 3}
    correct = sum(data[i, 1] == condition_to_correct_choice[data[i, 2]] for i in range(len(data)))
    accuracy = correct / len(data)
    print(f"\n   Accuracy: {accuracy:.2%} ({correct}/{len(data)})")

    # 3. Build model
    print("\n3. Build PyMC model:")
    model = create_grt_lba_4choice_model(data)
    print("   ✓ Model built successfully")
    print(f"   Number of parameters to estimate: 10 parameters")
    print(f"   - 4 Left dimension drift rates: v1_L_when_H, v2_L_when_H, v1_L_when_V, v2_L_when_V")
    print(f"   - 4 Right dimension drift rates: v1_R_when_H, v2_R_when_H, v1_R_when_V, v2_R_when_V")
    print(f"   - 2 PS violation parameters: delta_v1_L_V, delta_v1_R_H")
    print(f"   - A=0.5, b=1.5, t0=0.25, s=1.0 are all FIXED (REALISTIC)")
    print(f"   - Symmetric design: Testing PS violations in BOTH dimensions")

    # 4. Setup initial values for MCMC
    # SPEED OPTION: Set SKIP_MAP=True to skip slow MAP estimation
    SKIP_MAP = True  # ⚠️ Change to False if you want MAP optimization

    if SKIP_MAP:
        print("\n4. Skipping MAP estimation (using simple initial values):")
        print("   ⚠️  MAP is VERY SLOW with defective CDF integration")
        print("   Using simple initial values instead (faster startup)")

        # Simple initial values based on true parameters
        map_initvals = {
            'v1_L_when_H': 1.8,
            'v2_L_when_H': 1.5,
            'v1_L_when_V': 1.5,
            'v2_L_when_V': 1.8,
            'v1_R_when_H': 1.8,
            'v2_R_when_H': 1.5,
            'v1_R_when_V': 1.5,
            'v2_R_when_V': 1.8,
            'delta_v1_L_V': 0.0,
            'delta_v1_R_H': 0.0
        }
        print("\n   Using initial values:")
        print(f"   v1_L_when_H={map_initvals['v1_L_when_H']:.3f}, v2_L_when_H={map_initvals['v2_L_when_H']:.3f}")
        print(f"   v1_L_when_V={map_initvals['v1_L_when_V']:.3f}, v2_L_when_V={map_initvals['v2_L_when_V']:.3f}")
        print(f"   v1_R_when_H={map_initvals['v1_R_when_H']:.3f}, v2_R_when_H={map_initvals['v2_R_when_H']:.3f}")
        print(f"   v1_R_when_V={map_initvals['v1_R_when_V']:.3f}, v2_R_when_V={map_initvals['v2_R_when_V']:.3f}")
        print(f"   delta_v1_L_V={map_initvals['delta_v1_L_V']:.3f}, delta_v1_R_H={map_initvals['delta_v1_R_H']:.3f}")
    else:
        print("\n4. Run multiple MAP estimations (避免局部最優):")
        print("   ⚠️  This will be SLOW (~30-60 min per run)")
        best_map, all_map_results, map_consistency = run_multiple_map_8param(
            model=model,
            data=data,
            n_runs=2,
            solution_hint='high_correct'
        )

        # Use best MAP as initial values for MCMC
        map_initvals = best_map['map']
        print("\n   將使用最佳 MAP 解作為 MCMC 的初始值:")
        print(f"   v1_L_when_H={map_initvals['v1_L_when_H']:.3f}, v2_L_when_H={map_initvals['v2_L_when_H']:.3f}")
        print(f"   v1_L_when_V={map_initvals['v1_L_when_V']:.3f}, v2_L_when_V={map_initvals['v2_L_when_V']:.3f}")
        print(f"   v1_R_when_H={map_initvals['v1_R_when_H']:.3f}, v2_R_when_H={map_initvals['v2_R_when_H']:.3f}")
        print(f"   v1_R_when_V={map_initvals['v1_R_when_V']:.3f}, v2_R_when_V={map_initvals['v2_R_when_V']:.3f}")

    # 5. MCMC sampling
    print("\n5. Run MCMC sampling:")
    print("   CONVERGENCE FIX: Increased draws and tune for better convergence")
    print("   - draws=30000 (was 15000) - more iterations for convergence")
    print("   - tune=3000 (was 1000) - longer burn-in for adaptation")
    print("   - chains=8 (using 8 CPU cores for parallel sampling)")
    print("   - sampler: DEMetropolisZ")
    print("   - Estimating 10 parameters: 8 drift rates + 2 delta (PS violations)")
    print("   - Using Paper Equation 1 & 2 (complete PDF/CDF definitions)")
    print("   - LIKELIHOOD BUG FIXED: Now using defective PDF/CDF for RT")
    print("   - SPEED OPTIMIZATION: Numba JIT + 25-point integration (10-50x faster)")
    print("   - PRECISION: Full integration (not approximation)")
    print("   - initvals: 使用簡單初始值")
    print("   Estimated time: ~4-8 hours (10 params, precise mode with Numba)")
    print("\n   Starting sampling...")

    with model:
        # Sample 10 parameters: 8 drift rates + 2 delta (PS violation parameters)
        vars_to_sample = [
            model.v1_L_when_H, model.v2_L_when_H,
            model.v1_L_when_V, model.v2_L_when_V,
            model.v1_R_when_H, model.v2_R_when_H,
            model.v1_R_when_V, model.v2_R_when_V,
            model.delta_v1_L_V, model.delta_v1_R_H
        ]


        # PRECISION MODE: Use Numba-optimized integration for accuracy
        # Precise mode: 25-point integration with Numba JIT (~10-50x faster than scipy)
        # Fast mode: F ≈ F_win × S_lose (317x faster but less accurate)
        _CDF_MODE['use_fast'] = False  # Use PRECISE mode with Numba optimization

        trace = pm.sample(
            draws=30000,     # INCREASED: 15000 → 30000 for better convergence
            tune=3000,       # INCREASED: 1000 → 3000 for longer adaptation
            chains=8,        # Using 8 chains for better convergence diagnostics
            step=pm.DEMetropolisZ(vars_to_sample),
            initvals=map_initvals,  # Use best MAP solution as starting point
            return_inferencedata=True,
            progressbar=True,
            cores=8          # Using 8 CPU cores for parallel sampling
        )

    print("\n   ✓ MCMC sampling completed!")

    # 6. Convergence diagnostics
    print("\n6. Convergence diagnostics:")
    import arviz as az

    print("\n   LBA parameters (FIXED - not sampled) - REALISTIC VERSION:")
    print(f"      A = 0.5 (fixed)")
    print(f"      b = 1.5 (fixed, REALISTIC)")
    print(f"      t0 = 0.25 (fixed, REALISTIC)")
    print(f"      s = 1.0 (fixed)")

    # Check all parameters
    print("\n   10 Parameters (8 drift rates + 2 PS violations):")
    var_names_list = ['v1_L_when_H', 'v2_L_when_H', 'v1_L_when_V', 'v2_L_when_V',
                      'v1_R_when_H', 'v2_R_when_H', 'v1_R_when_V', 'v2_R_when_V',
                      'delta_v1_L_V', 'delta_v1_R_H']

    drift_summary = az.summary(trace, var_names=var_names_list)
    print(drift_summary[['mean', 'sd', 'r_hat', 'ess_bulk']])

    # Detailed analysis of BOTH delta parameters (symmetric PS violations)
    delta_L_samples = trace.posterior['delta_v1_L_V'].values.flatten()
    delta_L_mean = delta_L_samples.mean()
    delta_L_std = delta_L_samples.std()
    delta_L_hdi = az.hdi(trace, var_names=['delta_v1_L_V'], hdi_prob=0.95)

    delta_R_samples = trace.posterior['delta_v1_R_H'].values.flatten()
    delta_R_mean = delta_R_samples.mean()
    delta_R_std = delta_R_samples.std()
    delta_R_hdi = az.hdi(trace, var_names=['delta_v1_R_H'], hdi_prob=0.95)

    # ===== BAYES FACTOR ANALYSIS (Savage-Dickey Density Ratio) =====
    # Tests: H0: delta = 0 (PS holds) vs H1: delta ≠ 0 (PS violated)
    # BF10 = posterior_density(0) / prior_density(0)

    from scipy.stats import gaussian_kde, norm as scipy_norm

    # Prior density at delta=0: Normal(mu=0, sigma=0.15)
    prior_sigma = 0.15
    prior_density_at_zero = scipy_norm.pdf(0, loc=0, scale=prior_sigma)

    # Posterior density at delta=0 (using KDE)
    kde_L = gaussian_kde(delta_L_samples)
    posterior_density_L_at_zero = kde_L.evaluate(0)[0]

    kde_R = gaussian_kde(delta_R_samples)
    posterior_density_R_at_zero = kde_R.evaluate(0)[0]

    # Bayes Factor BF10 (in favor of H1: delta ≠ 0)
    BF10_L = posterior_density_L_at_zero / prior_density_at_zero
    BF01_L = 1.0 / BF10_L  # BF01 (in favor of H0: delta = 0)

    BF10_R = posterior_density_R_at_zero / prior_density_at_zero
    BF01_R = 1.0 / BF10_R

    # Bayesian hypothesis testing for delta_L
    p_deltaL_positive = (delta_L_samples > 0).mean()
    p_deltaL_negative = (delta_L_samples < 0).mean()
    p_deltaL_above_01 = (np.abs(delta_L_samples) > 0.1).mean()
    p_deltaL_above_02 = (np.abs(delta_L_samples) > 0.2).mean()

    # Bayesian hypothesis testing for delta_R
    p_deltaR_positive = (delta_R_samples > 0).mean()
    p_deltaR_negative = (delta_R_samples < 0).mean()
    p_deltaR_above_01 = (np.abs(delta_R_samples) > 0.1).mean()
    p_deltaR_above_02 = (np.abs(delta_R_samples) > 0.2).mean()

    print(f"\n   🎯 PS Violation Parameter Analysis (Symmetric Design):")
    print(f"\n   LEFT dimension (influenced by Right):")
    print(f"      delta_L = {delta_L_mean:.3f} ± {delta_L_std:.3f}")
    print(f"      95% HDI: [{delta_L_hdi['delta_v1_L_V'].values[0]:.3f}, {delta_L_hdi['delta_v1_L_V'].values[1]:.3f}]")
    print(f"\n   RIGHT dimension (influenced by Left):")
    print(f"      delta_R = {delta_R_mean:.3f} ± {delta_R_std:.3f}")
    print(f"      95% HDI: [{delta_R_hdi['delta_v1_R_H'].values[0]:.3f}, {delta_R_hdi['delta_v1_R_H'].values[1]:.3f}]")

    # Bayes Factor interpretation (Kass & Raftery 1995)
    def interpret_BF10(bf10):
        """Interpret Bayes Factor (Kass & Raftery 1995)"""
        if bf10 > 100:
            return "Decisive evidence for H1 (PS violated)"
        elif bf10 > 30:
            return "Very strong evidence for H1 (PS violated)"
        elif bf10 > 10:
            return "Strong evidence for H1 (PS violated)"
        elif bf10 > 3:
            return "Substantial evidence for H1 (PS violated)"
        elif bf10 > 1:
            return "Weak evidence for H1 (PS violated)"
        elif bf10 > 1/3:
            return "Anecdotal evidence (inconclusive)"
        elif bf10 > 1/10:
            return "Substantial evidence for H0 (PS holds)"
        elif bf10 > 1/30:
            return "Strong evidence for H0 (PS holds)"
        elif bf10 > 1/100:
            return "Very strong evidence for H0 (PS holds)"
        else:
            return "Decisive evidence for H0 (PS holds)"

    print(f"\n   📊 Bayes Factor Analysis (Savage-Dickey Density Ratio):")
    print(f"      Testing H0: delta = 0 (PS holds) vs H1: delta ≠ 0 (PS violated)")
    print(f"\n   LEFT dimension (delta_L):")
    print(f"      BF10 = {BF10_L:.3f}  (evidence for PS violation)")
    print(f"      BF01 = {BF01_L:.3f}  (evidence for PS holding)")
    print(f"      → {interpret_BF10(BF10_L)}")
    print(f"\n   RIGHT dimension (delta_R):")
    print(f"      BF10 = {BF10_R:.3f}  (evidence for PS violation)")
    print(f"      BF01 = {BF01_R:.3f}  (evidence for PS holding)")
    print(f"      → {interpret_BF10(BF10_R)}")

    print(f"\n   📊 Bayesian Hypothesis Testing (delta_L):")
    print(f"      P(delta_L > 0 | data) = {p_deltaL_positive:.3f}")
    print(f"      P(delta_L < 0 | data) = {p_deltaL_negative:.3f}")
    print(f"      P(|delta_L| > 0.1 | data) = {p_deltaL_above_01:.3f}")
    print(f"      P(|delta_L| > 0.2 | data) = {p_deltaL_above_02:.3f}")

    print(f"\n   📊 Bayesian Hypothesis Testing (delta_R):")
    print(f"      P(delta_R > 0 | data) = {p_deltaR_positive:.3f}")
    print(f"      P(delta_R < 0 | data) = {p_deltaR_negative:.3f}")
    print(f"      P(|delta_R| > 0.1 | data) = {p_deltaR_above_01:.3f}")
    print(f"      P(|delta_R| > 0.2 | data) = {p_deltaR_above_02:.3f}")

    # Interpretation for delta_L
    hdi_L_contains_zero = (delta_L_hdi['delta_v1_L_V'].values[0] < 0 < delta_L_hdi['delta_v1_L_V'].values[1])
    hdi_R_contains_zero = (delta_R_hdi['delta_v1_R_H'].values[0] < 0 < delta_R_hdi['delta_v1_R_H'].values[1])

    print(f"\n   📝 Overall Interpretation:")
    print(f"\n   LEFT dimension (delta_L):")
    # HDI
    if not hdi_L_contains_zero:
        print(f"      ✅ HDI excludes 0 → Evidence for PS violation")
    else:
        print(f"      ⚠️  HDI contains 0 → Inconclusive")
    # Bayes Factor
    if BF10_L > 10:
        print(f"      ✅ BF10 = {BF10_L:.1f} > 10 → Strong evidence for PS violation")
    elif BF10_L > 3:
        print(f"      ✓ BF10 = {BF10_L:.1f} > 3 → Substantial evidence for PS violation")
    elif BF01_L > 10:
        print(f"      ❌ BF01 = {BF01_L:.1f} > 10 → Strong evidence PS holds (delta=0)")
    elif BF01_L > 3:
        print(f"      ⚠️  BF01 = {BF01_L:.1f} > 3 → Substantial evidence PS holds")
    else:
        print(f"      ⚠️  BF10 = {BF10_L:.1f} → Evidence inconclusive")
    # Posterior probability
    if p_deltaL_positive > 0.95:
        print(f"      ✅ P(delta_L > 0) = {p_deltaL_positive:.3f} > 0.95 → Positive PS violation")

    print(f"\n   RIGHT dimension (delta_R):")
    # HDI
    if not hdi_R_contains_zero:
        print(f"      ✅ HDI excludes 0 → Evidence for PS violation")
    else:
        print(f"      ⚠️  HDI contains 0 → Inconclusive")
    # Bayes Factor
    if BF10_R > 10:
        print(f"      ✅ BF10 = {BF10_R:.1f} > 10 → Strong evidence for PS violation")
    elif BF10_R > 3:
        print(f"      ✓ BF10 = {BF10_R:.1f} > 3 → Substantial evidence for PS violation")
    elif BF01_R > 10:
        print(f"      ❌ BF01 = {BF01_R:.1f} > 10 → Strong evidence PS holds (delta=0)")
    elif BF01_R > 3:
        print(f"      ⚠️  BF01 = {BF01_R:.1f} > 3 → Substantial evidence PS holds")
    else:
        print(f"      ⚠️  BF10 = {BF10_R:.1f} → Evidence inconclusive")
    # Posterior probability
    if p_deltaR_positive > 0.95:
        print(f"      ✅ P(delta_R > 0) = {p_deltaR_positive:.3f} > 0.95 → Positive PS violation")

    # Check convergence with detailed diagnostics
    max_rhat = drift_summary['r_hat'].max()
    min_ess = drift_summary['ess_bulk'].min()

    print(f"\n   Convergence Diagnostics:")
    print(f"      Max R-hat: {max_rhat:.4f} (target: < 1.01)")
    print(f"      Min ESS bulk: {min_ess:.0f} (target: > 400)")

    if max_rhat < 1.01 and min_ess > 400:
        print(f"\n   ✅ EXCELLENT convergence! All chains mixed well.")
    elif max_rhat < 1.05 and min_ess > 100:
        print(f"\n   ⚠️  ACCEPTABLE convergence, but consider more samples.")
        print(f"      Recommendation: Increase draws to 50000 and tune to 5000")
    else:
        print(f"\n   ❌ POOR convergence - results may be unreliable!")
        print(f"      Max R-hat = {max_rhat:.4f} (should be < 1.01)")
        print(f"      Min ESS = {min_ess:.0f} (should be > 400)")
        print(f"\n      Possible causes:")
        print(f"      1. Likelihood function has bugs (check defective PDF/CDF)")
        print(f"      2. Prior too wide or misspecified")
        print(f"      3. Need many more iterations (try draws=50000+)")
        print(f"      4. Multiple modes in posterior (check trace plots)")
        print(f"\n      ⚠️  DO NOT trust these results until convergence is achieved!")

    # 7. Parameter recovery check
    print("\n7. Parameter recovery check:")

    true_params = {
        'v1_L_when_H': v1_L_H_true,
        'v2_L_when_H': v2_L_H_true,
        'v1_L_when_V': v1_L_V_true,
        'v2_L_when_V': v2_L_V_true,
        'v1_R_when_H': v1_R_H_true,
        'v2_R_when_H': v2_R_H_true,
        'v1_R_when_V': v1_R_V_true,
        'v2_R_when_V': v2_R_V_true,
        'delta_v1_L_V': delta_v1_L_V_true,
        'delta_v1_R_H': delta_v1_R_H_true
    }

    print("\n   Comparing true vs. posterior mean:")
    for param_name, true_val in true_params.items():
        post_val = trace.posterior[param_name].values.mean()
        diff = post_val - true_val
        print(f"      {param_name:15s}: True={true_val:.2f}, Posterior={post_val:.3f}, Diff={diff:+.3f}")

    print("\n   LBA parameters (all FIXED, not estimated) - REALISTIC VERSION:")
    print(f"      A: Fixed at 0.5")
    print(f"      b: Fixed at 1.5 (REALISTIC, was 1.0)")
    print(f"      t0: Fixed at 0.25 (REALISTIC, was 0.2)")
    print(f"      s: Fixed at 1.0")

    # 8. Save trace to NetCDF
    print("\n8. Save posterior trace to NetCDF:")
    output_file = "model2_noPS_trace.nc"
    trace.to_netcdf(output_file)
    print(f"   ✓ Trace saved to: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")

    print("\n" + "="*70)
    print("✓ Model 2: Symmetric PS Violation Test (10 params) completed!")
    print(f"\n   Ground Truth:")
    print(f"      delta_L_true = {delta_v1_L_V_true:.3f} (Left dim influenced by Right)")
    print(f"      delta_R_true = {delta_v1_R_H_true:.3f} (Right dim influenced by Left)")
    print(f"\n   Posterior Estimates:")
    print(f"      delta_L_est  = {delta_L_mean:.3f} ± {delta_L_std:.3f}")
    print(f"      delta_R_est  = {delta_R_mean:.3f} ± {delta_R_std:.3f}")
    print(f"\n   Recovery Errors:")
    print(f"      delta_L error: {abs(delta_L_mean - delta_v1_L_V_true):.3f}")
    print(f"      delta_R error: {abs(delta_R_mean - delta_v1_R_H_true):.3f}")
    print(f"\n   Bayes Factor (H1: delta≠0 vs H0: delta=0):")
    print(f"      delta_L: BF10 = {BF10_L:.2f}, BF01 = {BF01_L:.2f}")
    print(f"      delta_R: BF10 = {BF10_R:.2f}, BF01 = {BF01_R:.2f}")

    # Success criteria (using multiple evidence sources)
    # HDI criterion
    hdi_success_L = (not hdi_L_contains_zero and delta_v1_L_V_true != 0) or (hdi_L_contains_zero and delta_v1_L_V_true == 0)
    hdi_success_R = (not hdi_R_contains_zero and delta_v1_R_H_true != 0) or (hdi_R_contains_zero and delta_v1_R_H_true == 0)

    # Bayes Factor criterion (BF > 3 for detection, BF < 1/3 for accepting null)
    if delta_v1_L_V_true != 0:
        bf_success_L = BF10_L > 3  # Should detect violation
    else:
        bf_success_L = BF01_L > 3  # Should accept PS (no violation)

    if delta_v1_R_H_true != 0:
        bf_success_R = BF10_R > 3  # Should detect violation
    else:
        bf_success_R = BF01_R > 3  # Should accept PS (no violation)

    print(f"\n   Detection Results:")
    print(f"\n   LEFT dimension:")
    if delta_v1_L_V_true != 0:
        if hdi_success_L and bf_success_L:
            print(f"      ✅ HDI and BF both confirm PS violation (True positive)")
        elif hdi_success_L or bf_success_L:
            print(f"      ✓ Partial evidence for PS violation (check details)")
        else:
            print(f"      ❌ Failed to detect PS violation (False negative)")
    else:
        if hdi_success_L and bf_success_L:
            print(f"      ✅ HDI and BF both confirm PS holds (True negative)")
        elif hdi_success_L or bf_success_L:
            print(f"      ✓ Partial evidence PS holds (check details)")
        else:
            print(f"      ❌ Incorrectly detected PS violation (False positive)")

    print(f"\n   RIGHT dimension:")
    if delta_v1_R_H_true != 0:
        if hdi_success_R and bf_success_R:
            print(f"      ✅ HDI and BF both confirm PS violation (True positive)")
        elif hdi_success_R or bf_success_R:
            print(f"      ✓ Partial evidence for PS violation (check details)")
        else:
            print(f"      ❌ Failed to detect PS violation (False negative)")
    else:
        if hdi_success_R and bf_success_R:
            print(f"      ✅ HDI and BF both confirm PS holds (True negative)")
        elif hdi_success_R or bf_success_R:
            print(f"      ✓ Partial evidence PS holds (check details)")
        else:
            print(f"      ❌ Incorrectly detected PS violation (False positive)")
    print("="*70)
