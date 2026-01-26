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
def lba_def_cdf(t, v_win, v_lose, A, b, s, n_pts=30):
    if t <= 0: return 0.0
    tau = np.linspace(1e-10, t, n_pts)
    dt = t / n_pts
    integral = 0.0
    for i in range(n_pts):
        integral += lba_def_pdf(tau[i], v_win, v_lose, A, b, s)
    return min(max(integral * dt, 0.0), 1.0)

# ============================================================================
# 2. Data Generation and PS Structure (seed=42, 5000 trials)
# ============================================================================

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
        Between-trial drift rate variability (standard deviation of N(v, s) distribution)

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
            # Between-trial drift rate variability: sample from Normal(v_mean, s)
            # Standard LBA: drift rates vary across trials according to N(v, s)
            v1_L_trial = max(0.01, rng.normal(v_left[0], s))  # Ensure positive
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
            # Between-trial drift rate variability
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
            # left_judgment: 0=H(|), 1=V(/)
            # right_judgment: 0=H(|), 1=V(/)
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
# 3. Extract Drift Rates for Each Trial
# ============================================================================

def extract_trial_drift_rates(v_match, v_mismatch, data):
    """
    Extract drift rates for each trial from posterior estimates and observed data

    Parameters:
    -----------
    v_match : float
        Posterior mean of v_match
    v_mismatch : float
        Posterior mean of v_mismatch
    data : array, shape (n_trials, 3)
        Observed data [RT, choice, condition]

    Returns:
    --------
    trial_df : DataFrame
        Contains detailed information and drift rates for each trial
    """
    # Parse data
    rt = data[:, 0]
    choice = data[:, 1].astype(int)
    cond = data[:, 2].astype(int)

    # Prepare results
    results = []

    # Condition labels (condition: 0=VH, 1=HH, 2=HV, 3=VV)
    cond_labels = {0: 'VH(/|)', 1: 'HH(||)', 2: 'HV(|/)', 3: 'VV(//)'}
    choice_labels = {0: 'VH', 1: 'HH', 2: 'HV', 3: 'VV'}

    # Drift rate configuration for each condition
    # v_tensor[cond, dimension, accumulator]
    # dimension: 0=Left, 1=Right
    # accumulator: 0=judge H, 1=judge V
    drift_configs = {
        0: {  # VH: Left sees V, Right sees H
            'left_stim': 'V(/)',
            'right_stim': 'H(|)',
            'left_drift': [v_mismatch, v_match],  # [judge H, judge V]
            'right_drift': [v_match, v_mismatch]
        },
        1: {  # HH: Left sees H, Right sees H
            'left_stim': 'H(|)',
            'right_stim': 'H(|)',
            'left_drift': [v_match, v_mismatch],
            'right_drift': [v_match, v_mismatch]
        },
        2: {  # HV: Left sees H, Right sees V
            'left_stim': 'H(|)',
            'right_stim': 'V(/)',
            'left_drift': [v_match, v_mismatch],
            'right_drift': [v_mismatch, v_match]
        },
        3: {  # VV: Left sees V, Right sees V
            'left_stim': 'V(/)',
            'right_stim': 'V(/)',
            'left_drift': [v_mismatch, v_match],
            'right_drift': [v_mismatch, v_match]
        }
    }

    # Extract information for each trial
    for i in range(len(data)):
        c = cond[i]
        ch = choice[i]
        config = drift_configs[c]

        # Parse choice (0=VH, 1=HH, 2=HV, 3=VV)
        # Left judgment: 0=H(|), 1=V(/)
        # Right judgment: 0=H(|), 1=V(/)
        choice_map = {0: (1, 0), 1: (0, 0), 2: (0, 1), 3: (1, 1)}
        left_judgment, right_judgment = choice_map[ch]

        trial_info = {
            'trial': i,
            'condition': cond_labels[c],
            'left_stim': config['left_stim'],
            'right_stim': config['right_stim'],
            'choice': choice_labels[ch],
            'left_judgment': 'V(/)' if left_judgment == 1 else 'H(|)',
            'right_judgment': 'V(/)' if right_judgment == 1 else 'H(|)',
            'rt': rt[i],

            # Left drift rates
            'left_drift_judge_H': config['left_drift'][0],
            'left_drift_judge_V': config['left_drift'][1],

            # Right drift rates
            'right_drift_judge_H': config['right_drift'][0],
            'right_drift_judge_V': config['right_drift'][1],

            # Winning accumulator drift rate
            'left_winner_drift': config['left_drift'][left_judgment],
            'right_winner_drift': config['right_drift'][right_judgment],
        }

        results.append(trial_info)

    trial_df = pd.DataFrame(results)
    return trial_df

# ============================================================================
# 4. PyMC Model and Pointwise Likelihood Op
# ============================================================================

class GRT_LBA_2D_PointwiseOp(pt.Op):
        # Modified itypes definition
    itypes = [pt.ivector, pt.dvector, pt.ivector, pt.dtensor3, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        choice, rt, cond, v_tensor, A, b, t0, s = inputs
        log_liks = np.zeros(len(rt))
        j_map = {0:(1,0), 1:(0,0), 2:(0,1), 3:(1,1)}

        # Convert scalar to Python float
        A_val = float(A)
        b_val = float(b)
        t0_val = float(t0)
        s_val = float(s)

        for i in range(len(rt)):
            t = max(float(rt[i]) - t0_val, 1e-5)
            l_c, r_c = j_map[int(choice[i])]
            c_idx = int(cond[i])

            # Left dimension Defective PDF/CDF
            v_win_L = float(v_tensor[c_idx,0,l_c])
            v_lose_L = float(v_tensor[c_idx,0,1-l_c])
            fL = lba_def_pdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)
            FL = lba_def_cdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)
            # Right dimension Defective PDF/CDF
            v_win_R = float(v_tensor[c_idx,1,r_c])
            v_lose_R = float(v_tensor[c_idx,1,1-r_c])
            fR = lba_def_pdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)
            FR = lba_def_cdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)

            # Two-dimensional parallel processing likelihood: f_joint = fL*FR + fR*FL
            # Here we add the P(choice) term (in your original likelihood logic)
            log_liks[i] = np.log(fL * FR + fR * FL + 1e-20)

        outputs[0][0] = log_liks

def build_model(observed_data):
    with pm.Model() as model:
        A, b, s, t0 = 0.5, 1.5, 1.0, 0.25
        
        # 4-parameter reparameterization
        vt = pm.HalfNormal("vt", sigma=2.0)
        vb = pm.Beta("vb", 2, 2)
        v_m = pm.Deterministic("v_match", vt * vb) #vt=5 vb<5
        v_ms = pm.Deterministic("v_mismatch", vt * (1 - vb))
#vb->match% 1-vb->mismatch%
        v_tensor = pt.zeros((4, 2, 2))
        # Accumulator index: 0=H(|), 1=V(/)
        # Condition 0: VH (L=V, R=H) -> L:[v_ms, v_m], R:[v_m, v_ms]
        v_tensor = pt.set_subtensor(v_tensor[0,0,:], [v_ms, v_m])
        v_tensor = pt.set_subtensor(v_tensor[0,1,:], [v_m, v_ms])
        # Condition 1: HH (L=H, R=H) -> L:[v_m, v_ms], R:[v_m, v_ms]
        v_tensor = pt.set_subtensor(v_tensor[1,0,:], [v_m, v_ms])
        v_tensor = pt.set_subtensor(v_tensor[1,1,:], [v_m, v_ms])
        # Condition 2: HV (L=H, R=V) -> L:[v_m, v_ms], R:[v_ms, v_m]
        v_tensor = pt.set_subtensor(v_tensor[2,0,:], [v_m, v_ms])
        v_tensor = pt.set_subtensor(v_tensor[2,1,:], [v_ms, v_m])
        # Condition 3: VV (L=V, R=V) -> L:[v_ms, v_m], R:[v_ms, v_m]
        v_tensor = pt.set_subtensor(v_tensor[3,0,:], [v_ms, v_m])
        v_tensor = pt.set_subtensor(v_tensor[3,1,:], [v_ms, v_m])

        # Save V and H drift rates for each condition
        # Condition 0: VH(/|) - Left sees V, Right sees H
        pm.Deterministic("VH_L_V", v_m)   # Left V accumulator (match)
        pm.Deterministic("VH_L_H", v_ms)  # Left H accumulator (mismatch)
        pm.Deterministic("VH_R_V", v_ms)  # Right V accumulator (mismatch)
        pm.Deterministic("VH_R_H", v_m)   # Right H accumulator (match)

        # Condition 1: HH(||) - Left sees H, Right sees H
        pm.Deterministic("HH_L_V", v_ms)  # Left V accumulator (mismatch)
        pm.Deterministic("HH_L_H", v_m)   # Left H accumulator (match)
        pm.Deterministic("HH_R_V", v_ms)  # Right V accumulator (mismatch)
        pm.Deterministic("HH_R_H", v_m)   # Right H accumulator (match)

        # Condition 2: HV(|/) - Left sees H, Right sees V
        pm.Deterministic("HV_L_V", v_ms)  # Left V accumulator (mismatch)
        pm.Deterministic("HV_L_H", v_m)   # Left H accumulator (match)
        pm.Deterministic("HV_R_V", v_m)   # Right V accumulator (match)
        pm.Deterministic("HV_R_H", v_ms)  # Right H accumulator (mismatch)

        # Condition 3: VV(//) - Left sees V, Right sees V
        pm.Deterministic("VV_L_V", v_m)   # Left V accumulator (match)
        pm.Deterministic("VV_L_H", v_ms)  # Left H accumulator (mismatch)
        pm.Deterministic("VV_R_V", v_m)   # Right V accumulator (match)
        pm.Deterministic("VV_R_H", v_ms)  # Right H accumulator (mismatch)

        log_lik_vec = GRT_LBA_2D_PointwiseOp()(pt.as_tensor_variable(observed_data[:, 1].astype('int32')),
                                               pt.as_tensor_variable(observed_data[:, 0]),
                                               pt.as_tensor_variable(observed_data[:, 2].astype('int32')),
                                               v_tensor,
                                               pt.as_tensor_variable(A, dtype='float64'),
                                               pt.as_tensor_variable(b, dtype='float64'),
                                               pt.as_tensor_variable(t0, dtype='float64'),
                                               pt.as_tensor_variable(s, dtype='float64'))
        
        pm.Deterministic("log_likelihood", log_lik_vec)
        pm.Potential("obs", pt.sum(log_lik_vec))
        
        trace = pm.sample(draws=2000, tune=1000, chains=24, step=pm.DEMetropolisZ(), random_seed=42)
    return trace

# ============================================================================
# 5. Execution and Diagnostics
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("1. Generating 5000 2D-LBA data points...")
    print("="*70)
    print("Using random non-PS data (4 completely independent parameter sets)")

    # Set parameters
    rng = np.random.default_rng(42)
    A, b, t0, s = 0.5, 1.5, 0.25, 1.0

    # --- Random data generation without PS assumption (same as model4_noPS.py) ---
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
    v_tensor_PS = np.array([
        [[v_ms_random[2], v_m_random[2]], [v_m_random[0], v_ms_random[0]]],  # VH: L sees V, R sees H
        [[v_m_random[0], v_ms_random[0]], [v_m_random[0], v_ms_random[0]]],  # HH: L sees H, R sees H
        [[v_m_random[0], v_ms_random[0]], [v_ms_random[2], v_m_random[2]]],  # HV: L sees H, R sees V
        [[v_ms_random[2], v_m_random[2]], [v_ms_random[2], v_m_random[2]]]   # VV: L sees V, R sees V
    ])
    v_tensor_noPS = np.array([
        [[v_ms_random[2], v_m_random[2]], [v_m_random[2], v_ms_random[2]]],  # VH: L sees V, R sees H
        [[v_m_random[1], v_ms_random[1]], [v_m_random[1], v_ms_random[1]]],  # HH: L sees H, R sees H
        [[v_m_random[2], v_ms_random[2]], [v_ms_random[2], v_m_random[2]]],  # HV: L sees H, R sees V
        [[v_ms_random[0], v_m_random[0]], [v_ms_random[0], v_m_random[0]]]   # VV: L sees V, R sees V
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
    # drift_configs same as v_tensor, just for display
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
        # Based on condition mapping, dominant choice should be: VH→VH, HH→HH, HV→HV, VV→VV
        # condition: 0=VH, 1=HH, 2=HV, 3=VV
        # choice: 0=VH, 1=HH, 2=HV, 3=VV
        # Therefore it's identity mapping (verified through testing)
        dominant_choice_map = {0: 0, 1: 1, 2: 2, 3: 3}  # cond -> expected choice
        expected = dominant_choice_map[c]
        n_correct = np.sum((cond == c) & (choice == expected))
        n_total = np.sum(mask)
        acc = n_correct / n_total * 100
        correct_counts.append(acc)
        print(f"  {cond_labels[c]}: {n_correct}/{n_total} ({acc:.1f}%) chose {choice_labels[expected]}")

    print(f"\nOverall Average Accuracy: {np.mean(correct_counts):.1f}%")

    print("\n" + "="*70)
    print("2. Running 24-chain sampling...")
    print("="*70)
    trace = build_model(data)

    print("\n" + "="*70)
    print("3. Diagnostic Report (R-hat & ESS):")
    print("="*70)

    # Basic parameters
    summary = az.summary(trace, var_names=["v_match", "v_mismatch"])
    print("\nBasic Parameters:")
    print(summary[['mean', 'r_hat', 'ess_bulk']])

    # V and H drift rates for each condition
    drift_vars = ["HV_L_V", "HV_L_H", "HV_R_V", "HV_R_H",
                  "HH_L_V", "HH_L_H", "HH_R_V", "HH_R_H",
                  "VH_L_V", "VH_L_H", "VH_R_V", "VH_R_H",
                  "VV_L_V", "VV_L_H", "VV_R_V", "VV_R_H"]
    drift_summary = az.summary(trace, var_names=drift_vars)
    print("\nV and H Drift Rates for Each Condition:")
    print(drift_summary[['mean', 'r_hat', 'ess_bulk']])

    print(f"\n4. True Value Comparison:")
    print(f"   [Note] Data uses 4 random parameter sets (no PS pattern), but PS model assumes all conditions share parameters")
    print(f"   PS Model Estimates (Forced Sharing):")
    print(f"     v_match    = {summary.loc['v_match','mean']:.3f}")
    print(f"     v_mismatch = {summary.loc['v_mismatch','mean']:.3f}")
    print(f"\n   True Values (4 independent sets):")
    for i, name in enumerate(['VH', 'HH', 'HV', 'VV']):
        print(f"     {name}: v_match={v_m_random[i]:.3f}, v_mismatch={v_ms_random[i]:.3f}")

    # Display posterior drift rate estimates for each condition
    print(f"\n   Posterior Estimates for Each Condition's Drift Rate Configuration:")
    print(f"   VH(/|):")
    print(f"     Left(L): V accumulator={drift_summary.loc['VH_L_V','mean']:.3f}, H accumulator={drift_summary.loc['VH_L_H','mean']:.3f}")
    print(f"     Right(R): V accumulator={drift_summary.loc['VH_R_V','mean']:.3f}, H accumulator={drift_summary.loc['VH_R_H','mean']:.3f}")
    print(f"   HH(||):")
    print(f"     Left(L): V accumulator={drift_summary.loc['HH_L_V','mean']:.3f}, H accumulator={drift_summary.loc['HH_L_H','mean']:.3f}")
    print(f"     Right(R): V accumulator={drift_summary.loc['HH_R_V','mean']:.3f}, H accumulator={drift_summary.loc['HH_R_H','mean']:.3f}")
    print(f"   HV(|/):")
    print(f"     Left(L): V accumulator={drift_summary.loc['HV_L_V','mean']:.3f}, H accumulator={drift_summary.loc['HV_L_H','mean']:.3f}")
    print(f"     Right(R): V accumulator={drift_summary.loc['HV_R_V','mean']:.3f}, H accumulator={drift_summary.loc['HV_R_H','mean']:.3f}")
    print(f"   VV(//):")
    print(f"     Left(L): V accumulator={drift_summary.loc['VV_L_V','mean']:.3f}, H accumulator={drift_summary.loc['VV_L_H','mean']:.3f}")
    print(f"     Right(R): V accumulator={drift_summary.loc['VV_R_V','mean']:.3f}, H accumulator={drift_summary.loc['VV_R_H','mean']:.3f}")

    print("\n" + "="*70)
    print("5. Saving to NetCDF and Preparing for WAIC Comparison...")
    print("="*70)
    az.to_netcdf(trace, "final_4param_ps_results.nc")
    print("Saved to: final_4param_ps_results.nc")

    print("\n" + "="*70)
    print("6. Extracting Drift Rates for Each Trial...")
    print("="*70)

    # Extract posterior mean
    v_match_est = summary.loc['v_match', 'mean']
    v_mismatch_est = summary.loc['v_mismatch', 'mean']

    # Extract trial-level drift rates
    trial_df = extract_trial_drift_rates(v_match_est, v_mismatch_est, data)

    print(f"\nTotal {len(trial_df)} trials")
    print("\nFirst 10 trials information:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(trial_df.head(10).to_string(index=False))

    print("\n\nMean drift rates grouped by Condition:")
    print("-"*70)
    grouped = trial_df.groupby('condition').agg({
        'left_drift_judge_H': 'mean',
        'left_drift_judge_V': 'mean',
        'right_drift_judge_H': 'mean',
        'right_drift_judge_V': 'mean',
        'rt': 'mean'
    })
    print(grouped)

    print("\nMean drift rate of winning accumulator for each Condition:")
    print("-"*70)
    winner_stats = trial_df.groupby('condition').agg({
        'left_winner_drift': 'mean',
        'right_winner_drift': 'mean',
    })
    print(winner_stats)

    # Save results
    output_file = "trial_drift_rates.csv"
    trial_df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    print("\n" + "="*70)
    print("All analyses complete!")
    print("="*70)
