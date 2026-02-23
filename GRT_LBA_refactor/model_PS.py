# 替換原有的編譯目錄，加入隨機字串避免多個 Job 互相衝突
import os
import uuid
unique_dir = f"pytensor_compile_{uuid.uuid4()}"
os.environ["PYTENSOR_FLAGS"] = f"base_compiledir=/tmp/{unique_dir}"

import numpy as np
import pytensor.tensor as pt
import pymc as pm
import arviz as az
from numba import jit
import math
import warnings
import xarray as xr
import pandas as pd

warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
# 1. NUMBA High-Efficiency Computation: Preserving 2D-LBA Independent Decision Logic (Defective PDF/CDF)
# ============================================================================

@jit(nopython=True, fastmath=True)
def fast_norm_pdf(x):
    return 0.3989422804014327 * np.exp(-0.5 * x * x)

@jit(nopython=True, fastmath=True)
def fast_norm_cdf(x):
    if x < -8.0: return 0.0
    if x > 8.0: return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    y = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    res = 1.0 - 0.3989422804014327 * np.exp(-0.5 * x * x) * y
    return res if x >= 0 else 1.0 - res

@jit(nopython=True, fastmath=True)
def lba_pdf(t, v, A, b, s):
    if t <= 0: return 1e-15
    ts = t * s
    z1 = (b-A-t*v)/ts; z2 = (b-t*v)/ts
    A_safe = A if A > 1e-5 else 1e-5
    pdf = (1.0/A_safe) * (-v*fast_norm_cdf(z1) + v*fast_norm_cdf(z2) + s*fast_norm_pdf(z1) - s*fast_norm_pdf(z2))
    return pdf if pdf > 1e-15 else 1e-15

@jit(nopython=True, fastmath=True)
def lba_cdf(t, v, A, b, s):
    if t <= 0: return 0.0
    ts = t * s
    z1 = (b-A-t*v)/ts; z2 = (b-t*v)/ts
    A_safe = A if A > 1e-5 else 1e-5
    cdf = 1.0 + ((b-A-t*v)/A_safe)*fast_norm_cdf(z1) - ((b-t*v)/A_safe)*fast_norm_cdf(z2) + \
                (ts/A_safe)*fast_norm_pdf(z1) - (ts/A_safe)*fast_norm_pdf(z2)
    if cdf < 0.0:
        cdf = 0.0
    elif cdf > 1.0 - 1e-15:
        cdf = 1.0 - 1e-15
    return cdf

@jit(nopython=True, fastmath=True)
def lba_def_pdf(t, v_win, v_lose, A, b, s):
    return lba_pdf(t, v_win, A, b, s) * (1.0 - lba_cdf(t, v_lose, A, b, s))

@jit(nopython=True, fastmath=True)
def lba_def_cdf(t, v_win, v_lose, A, b, s, n_pts=50):
    """使用 Simpson's Rule 積分，比 Riemann sum 更精確"""
    if t <= 0: return 0.0
    # Ensure n_pts is even for Simpson's Rule
    if n_pts % 2 == 1:
        n_pts += 1
    h = t / n_pts
    # Simpson's Rule: integral ≈ (h/3) * [f(x0) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + f(xn)]
    integral = lba_def_pdf(1e-10, v_win, v_lose, A, b, s)  # f(x0)
    integral += lba_def_pdf(t, v_win, v_lose, A, b, s)      # f(xn)
    for i in range(1, n_pts):
        tau = i * h
        if i % 2 == 1:
            integral += 4.0 * lba_def_pdf(tau, v_win, v_lose, A, b, s)
        else:
            integral += 2.0 * lba_def_pdf(tau, v_win, v_lose, A, b, s)
    return min(max(integral * h / 3.0, 0.0), 1.0)

# ============================================================================
# 2. Data Generation for PS Model (seed=42, 5000 trials)
# ============================================================================

def lba_2dim_random(n_trials_per_condition, v_tensor, A, b, t0, s=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    all_data = []

    # Generate data for each condition
    # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    for cond in range(4):
        v_left = v_tensor[cond, 0, :]  # Left dimension [v1_L, v2_L]
        v_right = v_tensor[cond, 1, :]  # Right dimension [v1_R, v2_R]

        trials_generated = 0
        while trials_generated < n_trials_per_condition:
            # Left dimension decision
            k1_L = rng.uniform(0, A)
            k2_L = rng.uniform(0, A)
            # Between-trial drift rate variability: sample from Normal(v_mean, s)
            v1_L_trial = max(0.01, rng.normal(v_left[0], s))
            v2_L_trial = max(0.01, rng.normal(v_left[1], s))
            t1_L = (b - k1_L) / v1_L_trial
            t2_L = (b - k2_L) / v2_L_trial

            # Left judgment: 0=D(/) (if v1 wins), 1=S(|) (if v2 wins)
            if t1_L < t2_L:
                left_judgment = 0  # Judge D (/)
            else:
                left_judgment = 1  # Judge S (|)

            # Right dimension decision
            k1_R = rng.uniform(0, A)
            k2_R = rng.uniform(0, A)
            v1_R_trial = max(0.01, rng.normal(v_right[0], s))
            v2_R_trial = max(0.01, rng.normal(v_right[1], s))
            t1_R = (b - k1_R) / v1_R_trial
            t2_R = (b - k2_R) / v2_R_trial

            # Right judgment: 0=D(/) (if v1 wins), 1=S(|) (if v2 wins)
            if t1_R < t2_R:
                right_judgment = 0  # Judge D (/)
            else:
                right_judgment = 1  # Judge S (|)

            # Combined choice: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
            # left_judgment: 0=D(/), 1=S(|)
            # right_judgment: 0=D(/), 1=S(|)
            choice_map_reverse = {
                (0, 1): 0,  # DS (/|): Left=D(/), Right=S(|)
                (1, 1): 1,  # SS (||): Left=S(|), Right=S(|)
                (1, 0): 2,  # SD (|/): Left=S(|), Right=D(/)
                (0, 0): 3   # DD (//): Left=D(/), Right=D(/)
            }
            final_choice = choice_map_reverse[(left_judgment, right_judgment)]

            # Decision time = max of both dimensions (slower one determines RT)
            decision_time = max(min(t1_L, t2_L), min(t1_R, t2_R))

            # Add choice-specific motor time
            rt = decision_time + t0

            # Filter: RT > 0, RT <= 5s, and valid (regenerate if RT > 5s)
            if rt > 0 and rt <= 5.0 and np.isfinite(rt):
                all_data.append([rt, final_choice, cond])
                trials_generated += 1

    return np.array(all_data)


# ============================================================================
# 3. PyMC Model and Pointwise Likelihood Op
# ============================================================================

class GRT_LBA_2D_PointwiseOp(pt.Op):
    itypes = [pt.ivector, pt.dvector, pt.ivector, pt.dtensor3, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        choice, rt, cond, v_tensor, A, b, t0, s = inputs
        log_liks = np.zeros(len(rt))
        # j_map: choice -> (left_correct, right_correct)
        # 0=DS: L=D(0), R=S(1); 1=SS: L=S(1), R=S(1); 2=SD: L=S(1), R=D(0); 3=DD: L=D(0), R=D(0)
        j_map = {0:(0,1), 1:(1,1), 2:(1,0), 3:(0,0)}

        # Convert scalar to Python float
        A_val = float(A)
        b_val = float(b)
        t0_val = float(t0)
        s_val = float(s)

        for i in range(len(rt)):
            t = max(float(rt[i]) - t0_val, 1e-4)  # 稍微提高下限，避免 PDF 在零點附近的奇異值
            l_c, r_c = j_map[int(choice[i])]
            c_idx = int(cond[i])

            # 1. 左維度 (Left Dimension)
            v_win_L = float(v_tensor[c_idx, 0, l_c])
            v_lose_L = float(v_tensor[c_idx, 0, 1-l_c])
            fL = lba_def_pdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)
            FL = lba_def_cdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)

            # 2. 右維度 (Right Dimension)
            v_win_R = float(v_tensor[c_idx, 1, r_c])
            v_lose_R = float(v_tensor[c_idx, 1, 1-r_c])
            fR = lba_def_pdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)
            FR = lba_def_cdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)

            # 3. AND 邏輯 (Parallel-AND)
            # 公式：f_joint(t) = fL(t)*FR(t) + fR(t)*FL(t)
            joint_pdf = fL * FR + fR * FL
            if joint_pdf < 1e-20:
                log_liks[i] = -50.0  # 給予較大懲罰項而非直接 log(1e-20)
            else:
                log_liks[i] = np.log(joint_pdf)

        outputs[0][0] = log_liks


def build_model_PS(observed_data, tune=10000, draws=6000, chains=12):
    """
    PS Model with Stimulus Dimension separation (4 drift parameters)
    - v_D_match, v_D_mismatch: D stimulus dimension (對 D/斜線 刺激的反應)
    - v_S_match, v_S_mismatch: S stimulus dimension (對 S/垂直 刺激的反應)
    - b, t0: fixed values (not estimated)

    PS 假設：D 和 S 刺激的處理是獨立的（不受另一維度影響）
    """

    with pm.Model() as model:
        # 固定 A 和 s 作為比例尺
        A, s = 0.5, 1.0

        # --- 共享參數 (固定值) ---
        b = 1.1   # Fixed threshold
        t0 = 0.1  # Fixed non-decision time

        # --- PS 模型：依刺激維度分離 (D vs S) ---
        # D 刺激維度 (Diagonal/斜線/)
        vt_D = pm.TruncatedNormal("vt_D", mu=3.0, sigma=1.0, lower=0.1, upper=8.0)
        vb_D = pm.Beta("vb_D", 2, 2)
        v_D_match = pm.Deterministic("v_D_match", vt_D * vb_D)        # D(/) stim → D resp (correct)
        v_D_mismatch = pm.Deterministic("v_D_mismatch", vt_D * (1 - vb_D))  # D(/) stim → S resp (error)

        # S 刺激維度 (Straight/垂直|)
        vt_S = pm.TruncatedNormal("vt_S", mu=3.0, sigma=1.0, lower=0.1, upper=8.0)
        vb_S = pm.Beta("vb_S", 2, 2)
        v_S_match = pm.Deterministic("v_S_match", vt_S * vb_S)        # S(|) stim → S resp (correct)
        v_S_mismatch = pm.Deterministic("v_S_mismatch", vt_S * (1 - vb_S))  # S(|) stim → D resp (error)

        # Build v_tensor [cond, dimension, accumulator]
        # accumulator: 0=D response(/), 1=S response(|)
        # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
        v_tensor = pt.zeros((4, 2, 2))
        # Cond 0 (DS): Left=D(/), Right=S(|)
        v_tensor = pt.set_subtensor(v_tensor[0,0,:], [v_D_match, v_D_mismatch])  # D stim
        v_tensor = pt.set_subtensor(v_tensor[0,1,:], [v_S_mismatch, v_S_match])  # S stim
        # Cond 1 (SS): Left=S(|), Right=S(|)
        v_tensor = pt.set_subtensor(v_tensor[1,0,:], [v_S_mismatch, v_S_match])  # S stim
        v_tensor = pt.set_subtensor(v_tensor[1,1,:], [v_S_mismatch, v_S_match])  # S stim
        # Cond 2 (SD): Left=S(|), Right=D(/)
        v_tensor = pt.set_subtensor(v_tensor[2,0,:], [v_S_mismatch, v_S_match])  # S stim
        v_tensor = pt.set_subtensor(v_tensor[2,1,:], [v_D_match, v_D_mismatch])  # D stim
        # Cond 3 (DD): Left=D(/), Right=D(/)
        v_tensor = pt.set_subtensor(v_tensor[3,0,:], [v_D_match, v_D_mismatch])  # D stim
        v_tensor = pt.set_subtensor(v_tensor[3,1,:], [v_D_match, v_D_mismatch])  # D stim

        # ============================================================================
        # Likelihood
        # ============================================================================
        log_lik_vec = GRT_LBA_2D_PointwiseOp()(
            pt.as_tensor_variable(observed_data[:, 1].astype('int32')),
            pt.as_tensor_variable(observed_data[:, 0]),
            pt.as_tensor_variable(observed_data[:, 2].astype('int32')),
            v_tensor,
            pt.as_tensor_variable(A, dtype='float64'),
            pt.as_tensor_variable(b, dtype='float64'),
            pt.as_tensor_variable(t0, dtype='float64'),
            pt.as_tensor_variable(s, dtype='float64')
        )

        pm.Deterministic("log_likelihood", log_lik_vec)
        pm.Potential("obs", pt.sum(log_lik_vec))

        init_vals = {
            "vt_D": 3.0, "vb_D": 0.7, "vt_S": 3.0, "vb_S": 0.7,
        }
        trace_PS = pm.sample(draws=draws, tune=tune, chains=chains,
                         step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)

    return trace_PS


# ============================================================================
# 4. Execution and Diagnostics
# ============================================================================
if __name__ == "__main__":
    print("="*70, flush=True)
    print("1. Generating 5000 2D-LBA data points (PS structure)...", flush=True)

    # Set parameters
    rng = np.random.default_rng(42)

    A = 0.5
    s = 1.0
    b = 1.1   # Threshold (fixed for data generation)
    t0 = 0.1  # Non-decision time (fixed for data generation)

    # --- Random data generation with PS assumption ---
    # Randomly sample v_match and v_mismatch for 2 stimulus dimensions (H and V)
    v_m_random = rng.uniform(1.5, 3.5, size=3)     # 3 groups: [0]=SS, [1]=DD, [2]=Mixed
    v_ms_random = rng.uniform(0.5, 1.5, size=3)    # 3 groups: [0]=SS, [1]=DD, [2]=Mixed

    print("\nRandomly generated true parameters (PS model - 2 dimensions):", flush=True)
    print(f"  H dimension (D stim): v_match={v_m_random[1]:.2f}, v_mismatch={v_ms_random[1]:.2f}", flush=True)
    print(f"  V dimension (S stim): v_match={v_m_random[0]:.2f}, v_mismatch={v_ms_random[0]:.2f}", flush=True)

    # Construct a v_tensor_PS with "PS pattern"
    # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    # accumulator: 0=D(/), 1=S(|)
    # PS model: H dimension (D stim) uses [1], V dimension (S stim) uses [0]
    v_tensor_PS = np.array([
        [[v_m_random[1], v_ms_random[1]], [v_ms_random[0], v_m_random[0]]],  # DS: L=D(H), R=S(V)
        [[v_ms_random[0], v_m_random[0]], [v_ms_random[0], v_m_random[0]]],  # SS: L=S(V), R=S(V)
        [[v_ms_random[0], v_m_random[0]], [v_m_random[1], v_ms_random[1]]],  # SD: L=S(V), R=D(H)
        [[v_m_random[1], v_ms_random[1]], [v_m_random[1], v_ms_random[1]]]   # DD: L=D(H), R=D(H)
    ])

    # Generate data
    data_PS = lba_2dim_random(
        n_trials_per_condition=10000 // 4,
        v_tensor=v_tensor_PS,
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )

    print(f"\nGenerated {len(data_PS)} trials", flush=True)
    print(f"  Min RT: {data_PS[:, 0].min():.3f}s", flush=True)
    print(f"  Max RT: {data_PS[:, 0].max():.3f}s", flush=True)
    print(f"  Mean RT: {data_PS[:, 0].mean():.3f}s", flush=True)

    # Run PS model sampling
    print("\n" + "="*70, flush=True)
    print("2. Running PS Model Sampling...", flush=True)
    print("="*70, flush=True)
    trace_PS = build_model_PS(data_PS, tune=10000, draws=6000, chains=12)

    # Save PS trace
    ps_nc_file = "ps_recovery_results.nc"
    trace_PS.to_netcdf(ps_nc_file)
    print(f"PS trace saved to: {ps_nc_file}", flush=True)

    # Print summary
    print("\n" + "="*70, flush=True)
    print("3. PS Model Results", flush=True)
    print("="*70, flush=True)

    ps_params = ['v_D_match', 'v_D_mismatch', 'v_S_match', 'v_S_mismatch']
    ps_summary = az.summary(trace_PS, var_names=ps_params)

    print("\nParameter Estimates (94% HDI):")
    print(f"  {'Parameter':<18} {'Mean':>8} {'SD':>8} {'HDI_3%':>8} {'HDI_97%':>8} {'R-hat':>8} {'ESS':>8}")
    print(f"  {'-'*72}")

    for param in ps_params:
        row = ps_summary.loc[param]
        print(f"  {param:<18} {row['mean']:>8.3f} {row['sd']:>8.3f} "
              f"{row['hdi_3%']:>8.3f} {row['hdi_97%']:>8.3f} "
              f"{row['r_hat']:>8.4f} {row['ess_bulk']:>8.0f}", flush=True)

    # Convergence check
    ps_max_rhat = ps_summary['r_hat'].max()
    ps_min_ess = ps_summary['ess_bulk'].min()
    print(f"\n  Convergence: max(R-hat)={ps_max_rhat:.4f}, min(ESS)={ps_min_ess:.0f}")
    print(f"  Status: {'OK' if ps_max_rhat < 1.01 and ps_min_ess > 400 else 'WARNING'}")

    # True vs Estimated comparison
    print("\n" + "="*70)
    print("4. Parameter Recovery")
    print("="*70)
    print(f"\n  {'Parameter':<18} {'True':>10} {'Estimated':>10} {'Bias':>10} {'In HDI':>8}")
    print(f"  {'-'*60}")

    ps_true = {
        'v_D_match': v_m_random[1],       # H stim -> group[1]=DD
        'v_D_mismatch': v_ms_random[1],
        'v_S_match': v_m_random[0],       # V stim -> group[0]=SS
        'v_S_mismatch': v_ms_random[0],
    }

    for param in ps_params:
        true_val = ps_true[param]
        est_val = ps_summary.loc[param, 'mean']
        hdi_low = ps_summary.loc[param, 'hdi_3%']
        hdi_high = ps_summary.loc[param, 'hdi_97%']
        bias = est_val - true_val
        in_hdi = "Yes" if hdi_low <= true_val <= hdi_high else "No"
        print(f"  {param:<18} {true_val:>10.3f} {est_val:>10.3f} {bias:>+10.3f} {in_hdi:>8}")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)
