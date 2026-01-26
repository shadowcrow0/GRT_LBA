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

# ============================================================================
# 1. NUMBA High-Efficiency Computation Core (PDF/CDF Logic Unchanged)
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
    return min(max(cdf, 0.0), 1.0 - 1e-15)

@jit(nopython=True, fastmath=True)
def lba_def_pdf(t, v_win, v_lose, A, b, s):
    return lba_pdf(t, v_win, A, b, s) * (1.0 - lba_cdf(t, v_lose, A, b, s))

@jit(nopython=True, fastmath=True)
def lba_def_cdf(t, v_win, v_lose, A, b, s, n_pts=30):
    if t <= 0: return 0.0
    tau = np.linspace(1e-10, t, n_pts)
    dt = t / n_pts
    integral = 0.0
    for i in range(n_pts):
        integral += lba_def_pdf(tau[i], v_win, v_lose, A, b, s)
    return min(max(integral * dt, 0.0), 1.0)

# ============================================================================
# 2. PyTensor Op: Compute Likelihood
# ============================================================================

class GRT_LBA_noPS_Op(pt.Op):
    itypes = [pt.ivector, pt.dvector, pt.ivector, pt.dtensor3, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        choice, rt, cond, v_tensor, A, b, t0, s = inputs
        log_liks = np.zeros(len(rt))

        # Convert scalar to Python float
        A_val = float(A)
        b_val = float(b)
        t0_val = float(t0)
        s_val = float(s)

        # Mapping: Choice 0=VH, 1=HH, 2=HV, 3=VV
        # j_map determines which dimension wins for Left/Right (L_win_idx, R_win_idx) -> 0=H, 1=V
        j_map = {0:(1,0), 1:(0,0), 2:(0,1), 3:(1,1)}

        for i in range(len(rt)):
            t = max(float(rt[i]) - t0_val, 1e-5)
            l_win, r_win = j_map[int(choice[i])]
            c_idx = int(cond[i])

            # Extract corresponding drift rates
            vwL = float(v_tensor[c_idx, 0, l_win])
            vlL = float(v_tensor[c_idx, 0, 1-l_win])
            vwR = float(v_tensor[c_idx, 1, r_win])
            vlR = float(v_tensor[c_idx, 1, 1-r_win])

            fL = lba_def_pdf(t, vwL, vlL, A_val, b_val, s_val)
            FL = lba_def_cdf(t, vwL, vlL, A_val, b_val, s_val)
            fR = lba_def_pdf(t, vwR, vlR, A_val, b_val, s_val)
            FR = lba_def_cdf(t, vwR, vlR, A_val, b_val, s_val)

            # 2D-LBA: Final RT is the joint probability density of max(tL, tR)
            log_liks[i] = np.log(fL * FR + fR * FL + 1e-25)
        outputs[0][0] = log_liks

# ============================================================================
# 3. Build NoPS Model (3 Parameter Sets Version)
# ============================================================================

def build_no_ps_model_v2(observed_data):
    """
    NoPS model with 3 parameter sets
    - Set 0: HH condition (||)
    - Set 1: VV condition (\\)
    - Set 2: Mixed (VH and HV shared)
    """
    with pm.Model() as model:
        A, b, s, t0 = 0.5, 1.5, 1.0, 0.25

        # Define 3 independent parameter sets: [0:HH, 1:VV, 2:Mixed]
        vt_raw = pm.HalfNormal("vt_raw", sigma=2.0, shape=3)
        vb_raw = pm.Beta("vb_raw", 2, 2, shape=3)

        v_m_raw = vt_raw * vb_raw
        v_ms_raw = vt_raw * (1 - vb_raw)

        # Map to 4 data conditions: 0=VH, 1=HH, 2=HV, 3=VV
        v_m = pt.stack([v_m_raw[2], v_m_raw[0], v_m_raw[2], v_m_raw[1]])
        v_ms = pt.stack([v_ms_raw[2], v_ms_raw[0], v_ms_raw[2], v_ms_raw[1]])

        pm.Deterministic("v_match_est", v_m_raw)  # 3 parameter set estimates
        pm.Deterministic("v_mismatch_est", v_ms_raw)
        pm.Deterministic("v_match", v_m)  # Match drift for 4 conditions
        pm.Deterministic("v_mismatch", v_ms)  # Mismatch drift for 4 conditions

        v_tensor = pt.zeros((4, 2, 2))
        for i in range(4):
            # Accumulator index: 0=H(|), 1=V(/)
            if i == 0:  # VH (/|): Left V, Right H
                v_tensor = pt.set_subtensor(v_tensor[i, 0, :], [v_ms[i], v_m[i]])
                v_tensor = pt.set_subtensor(v_tensor[i, 1, :], [v_m[i], v_ms[i]])
            elif i == 1:  # HH (||): Left H, Right H
                v_tensor = pt.set_subtensor(v_tensor[i, 0, :], [v_m[i], v_ms[i]])
                v_tensor = pt.set_subtensor(v_tensor[i, 1, :], [v_m[i], v_ms[i]])
            elif i == 2:  # HV (|/): Left H, Right V
                v_tensor = pt.set_subtensor(v_tensor[i, 0, :], [v_m[i], v_ms[i]])
                v_tensor = pt.set_subtensor(v_tensor[i, 1, :], [v_ms[i], v_m[i]])
            elif i == 3:  # VV (//): Left V, Right V
                v_tensor = pt.set_subtensor(v_tensor[i, 0, :], [v_ms[i], v_m[i]])
                v_tensor = pt.set_subtensor(v_tensor[i, 1, :], [v_ms[i], v_m[i]])

        log_lik_vec = GRT_LBA_noPS_Op()(
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
        trace = pm.sample(draws=2000, tune=1000, chains=24, step=pm.DEMetropolisZ(), random_seed=42)
        return trace

# ============================================================================
# 4. Data Generation Function
# ============================================================================

def lba_2dim_random(n_trials_per_condition, v_tensor, A, b, t0, s=1.0, rng=None):
    """
    Generate GRT-LBA simulated data with 2 independent dimension decisions
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
            v1_L_trial = max(0.01, rng.normal(v_left[0], s))
            v2_L_trial = max(0.01, rng.normal(v_left[1], s))
            t1_L = (b - k1_L) / v1_L_trial
            t2_L = (b - k2_L) / v2_L_trial

            # Left judgment: 0=H (if v1 wins), 1=V (if v2 wins)
            if t1_L < t2_L:
                left_judgment = 0  # Judge H
            else:
                left_judgment = 1  # Judge V

            # Right dimension decision
            k1_R = rng.uniform(0, A)
            k2_R = rng.uniform(0, A)
            v1_R_trial = max(0.01, rng.normal(v_right[0], s))
            v2_R_trial = max(0.01, rng.normal(v_right[1], s))
            t1_R = (b - k1_R) / v1_R_trial
            t2_R = (b - k2_R) / v2_R_trial

            # Right judgment: 0=H (if v1 wins), 1=V (if v2 wins)
            if t1_R < t2_R:
                right_judgment = 0  # Judge H
            else:
                right_judgment = 1  # Judge V

            # Combined choice: 0=VH(/|), 1=HH(||), 2=HV(|/), 3=VV(//)
            choice_map_reverse = {
                (1, 0): 0,  # VH (/|): Left=/, Right=|
                (0, 0): 1,  # HH (||): Left=|, Right=|
                (0, 1): 2,  # HV (|/): Left=|, Right=/
                (1, 1): 3   # VV (//): Left=/, Right=/
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

# ============================================================================
# 5. Execution and Analysis
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("1. Generating 5000 2D-LBA data points...")
    print("="*70)
    print("Using random non-PS data (4 completely independent parameter sets)")

    # Set parameters
    rng = np.random.default_rng(42)
    A, b, t0, s = 0.5, 1.5, 0.25, 1.0

    # --- Random data generation without PS assumption (same as model_tilted_matter_WAIC.py) ---
    # Randomly sample v_match and v_mismatch for each of the 4 conditions
    v_m_random = rng.uniform(1.5, 3.5, size=4)     # Randomly generate 4 different correct processing rates
    v_ms_random = rng.uniform(0.5, 1.5, size=4)    # Randomly generate 4 different incorrect processing rates

    print("\nRandomly generated true parameters:")
    cond_names_order = ['VH', 'HH', 'HV', 'VV']
    for i, name in enumerate(cond_names_order):
        print(f"  {name}: v_match={v_m_random[i]:.2f}, v_mismatch={v_ms_random[i]:.2f}")

    # Construct a v_tensor with "no PS pattern"
    # condition: 0=VH, 1=HH, 2=HV, 3=VV
    # accumulator: 0=H(|), 1=V(/)
    v_tensor = np.array([
        [[v_ms_random[0], v_m_random[0]], [v_m_random[0], v_ms_random[0]]],  # VH: L sees V, R sees H
        [[v_m_random[1], v_ms_random[1]], [v_m_random[1], v_ms_random[1]]],  # HH: L sees H, R sees H
        [[v_m_random[2], v_ms_random[2]], [v_ms_random[2], v_m_random[2]]],  # HV: L sees H, R sees V
        [[v_ms_random[3], v_m_random[3]], [v_ms_random[3], v_m_random[3]]]   # VV: L sees V, R sees V
    ])

    # Generate data
    data = lba_2dim_random(
        n_trials_per_condition=5000 // 4,
        v_tensor=v_tensor,
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )

    # Parse data
    rt = data[:, 0]
    choice = data[:, 1].astype(int)
    cond = data[:, 2].astype(int)

    # Condition labels (condition: 0=VH, 1=HH, 2=HV, 3=VV)
    cond_labels = {0: 'VH(/|)', 1: 'HH(||)', 2: 'HV(|/)', 3: 'VV(//)'}
    choice_labels = {0: 'VH', 1: 'HH', 2: 'HV', 3: 'VV'}

    print(f"\nTotal data points: {len(data)}")

    # Display drift rate settings for each condition
    print("\nDrift Rate Settings for Each Condition:")
    drift_configs = v_tensor.tolist()
    for c in range(4):
        vs = drift_configs[c]
        print(f"  {cond_labels[c]}:")
        print(f"    Left(L): H accumulator={vs[0][0]:.2f}, V accumulator={vs[0][1]:.2f}")
        print(f"    Right(R): H accumulator={vs[1][0]:.2f}, V accumulator={vs[1][1]:.2f}")

    # Condition distribution
    print("\nCondition Distribution:")
    for c in range(4):
        count = np.sum(cond == c)
        print(f"  {cond_labels[c]}: {count} trials ({count/len(cond)*100:.1f}%)")

    # Response time statistics
    print(f"\nResponse Time Statistics:")
    print(f"  Overall: Mean={rt.mean():.3f}s, SD={rt.std():.3f}s, Range=[{rt.min():.3f}, {rt.max():.3f}]")
    for c in range(4):
        rt_c = rt[cond == c]
        print(f"  {cond_labels[c]}: Mean={rt_c.mean():.3f}s, SD={rt_c.std():.3f}s")

    # Calculate accuracy (proportion choosing dominant option)
    print(f"\nAccuracy by Condition (Choosing Dominant Option):")
    correct_counts = []
    for c in range(4):
        mask = cond == c
        dominant_choice_map = {0: 0, 1: 1, 2: 2, 3: 3}  # cond -> expected choice
        expected = dominant_choice_map[c]
        n_correct = np.sum((cond == c) & (choice == expected))
        n_total = np.sum(mask)
        acc = n_correct / n_total * 100
        correct_counts.append(acc)
        print(f"  {cond_labels[c]}: {n_correct}/{n_total} ({acc:.1f}%) chose {choice_labels[expected]}")

    print(f"\nOverall Average Accuracy: {np.mean(correct_counts):.1f}%")

    print("\n" + "="*70)
    print("2. Running noPS Model Sampling (24 chains, 3 parameter sets)...")
    print("="*70)
    print("Parameter Configuration:")
    print("  - Set 0: HH condition (||)")
    print("  - Set 1: VV condition (\\\\)")
    print("  - Set 2: Mixed (VH and HV shared)")

    trace = build_no_ps_model_v2(data)

    print("\n" + "="*70)
    print("3. Diagnostic Report (R-hat & ESS):")
    print("="*70)
    summary = az.summary(trace, var_names=["v_match_est", "v_mismatch_est"])
    print("\n3 Parameter Set Estimates:")
    print(summary[['mean', 'r_hat', 'ess_bulk']])

    print("\nEstimates for Each Condition (After Mapping):")
    cond_summary = az.summary(trace, var_names=["v_match", "v_mismatch"])
    print(cond_summary[['mean', 'r_hat', 'ess_bulk']])

    print(f"\n4. True Value Comparison:")
    print(f"   [Note] Data uses 4 random parameter sets (no PS pattern), noPS model uses 3 parameter sets")
    print(f"\n   True Values (4 independent sets):")
    for i, name in enumerate(['VH', 'HH', 'HV', 'VV']):
        print(f"     {name}: v_match={v_m_random[i]:.3f}, v_mismatch={v_ms_random[i]:.3f}")

    print(f"\n   noPS 3 Parameter Set Estimates:")
    print(f"     Set 0 (HH): v_match={summary.loc['v_match_est[0]','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_est[0]','mean']:.3f}")
    print(f"     Set 1 (VV): v_match={summary.loc['v_match_est[1]','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_est[1]','mean']:.3f}")
    print(f"     Set 2 (Mixed VH/HV): v_match={summary.loc['v_match_est[2]','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_est[2]','mean']:.3f}")

    print(f"\n   Mapped Estimates for Each Condition:")
    for i, name in enumerate(['VH', 'HH', 'HV', 'VV']):
        vm_est = cond_summary.loc[f'v_match[{i}]','mean']
        vms_est = cond_summary.loc[f'v_mismatch[{i}]','mean']
        print(f"     {name}: v_match={vm_est:.3f}, v_mismatch={vms_est:.3f}")

    print("\n" + "="*70)
    print("5. Saving to NetCDF and Preparing for WAIC Comparison...")
    print("="*70)
    az.to_netcdf(trace, "noPS_3param_trace.nc")
    print("Saved to: noPS_3param_trace.nc")

    print("\n" + "="*70)
    print("All analyses complete!")
    print("="*70)
