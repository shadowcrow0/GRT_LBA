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
    cond_labels = {0: 'HV(/|)', 1: 'HH(||)', 2: 'VH(|/)', 3: 'VV(//)'}
    choice_labels = {0: 'HV', 1: 'HH', 2: 'VH', 3: 'VV'}

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

def extract_trial_drift_rates_noPS(v_match_per_cond, v_mismatch_per_cond, data):
    """
    Extract drift rates for each trial from noPS model posterior estimates

    Parameters:
    -----------
    v_match_per_cond : array, shape (4,)
        Posterior mean of v_match for each condition [VH, HH, HV, VV]
    v_mismatch_per_cond : array, shape (4,)
        Posterior mean of v_mismatch for each condition [VH, HH, HV, VV]
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
    cond_labels = {0: 'HV(/|)', 1: 'HH(||)', 2: 'VH(|/)', 3: 'VV(//)'}
    choice_labels = {0: 'HV', 1: 'HH', 2: 'VH', 3: 'VV'}

    # Drift rate configuration for each condition (condition-specific)
    # v_tensor[cond, dimension, accumulator]
    # dimension: 0=Left, 1=Right
    # accumulator: 0=judge H, 1=judge V
    drift_configs = {
        0: {  # VH: Left sees V, Right sees H
            'left_stim': 'V(/)',
            'right_stim': 'H(|)',
            'left_drift': [v_mismatch_per_cond[0], v_match_per_cond[0]],  # [judge H, judge V]
            'right_drift': [v_match_per_cond[0], v_mismatch_per_cond[0]]
        },
        1: {  # HH: Left sees H, Right sees H
            'left_stim': 'H(|)',
            'right_stim': 'H(|)',
            'left_drift': [v_match_per_cond[1], v_mismatch_per_cond[1]],
            'right_drift': [v_match_per_cond[1], v_mismatch_per_cond[1]]
        },
        2: {  # HV: Left sees H, Right sees V
            'left_stim': 'H(|)',
            'right_stim': 'V(/)',
            'left_drift': [v_match_per_cond[2], v_mismatch_per_cond[2]],
            'right_drift': [v_mismatch_per_cond[2], v_match_per_cond[2]]
        },
        3: {  # VV: Left sees V, Right sees V
            'left_stim': 'V(/)',
            'right_stim': 'V(/)',
            'left_drift': [v_mismatch_per_cond[3], v_match_per_cond[3]],
            'right_drift': [v_mismatch_per_cond[3], v_match_per_cond[3]]
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
            'condition_idx': c,
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

            # Condition-specific parameters
            'cond_v_match': v_match_per_cond[c],
            'cond_v_mismatch': v_mismatch_per_cond[c],
        }

        results.append(trial_info)

    trial_df = pd.DataFrame(results)
    return trial_df

def display_data_summary(data, v_tensor):
    """
    Display comprehensive summary statistics for the generated data

    Parameters:
    -----------
    data : array, shape (n_trials, 3)
        Observed data [RT, choice, condition]
    v_tensor : array, shape (4, 2, 2)
        Drift rate tensor [condition, dimension, accumulator]

    Returns:
    --------
    summary_stats : dict
        Dictionary containing key statistics
    """
    # Parse data
    rt = data[:, 0]
    choice = data[:, 1].astype(int)
    cond = data[:, 2].astype(int)

    # Condition labels (condition: 0=VH, 1=HH, 2=HV, 3=VV)
    cond_labels = {0: 'HV(/|)', 1: 'HH(||)', 2: 'VH(|/)', 3: 'VV(//)'}
    choice_labels = {0: 'HV', 1: 'HH', 2: 'VH', 3: 'VV'}

    print(f"\nTotal data points: {len(data, flush=True)}", flush=True)

    # Display drift rate settings for each condition
    print("\nDrift Rate Settings for Each Condition:", flush=True)
    print("(Drift rate tensor structure: [condition, dimension, accumulator], flush=True)", flush=True)
    print("  dimension: 0=Left, 1=Right", flush=True)
    print("  accumulator: 0=judge H(|, flush=True), 1=judge V(/)", flush=True)

    drift_configs = v_tensor.tolist()
    for c in range(4):
        vs = drift_configs[c]
        print(f"\n  {cond_labels[c]}:", flush=True)
        print(f"    Left(L, flush=True):  H accumulator={vs[0][0]:.2f}, V accumulator={vs[0][1]:.2f}", flush=True)
        print(f"    Right(R, flush=True): H accumulator={vs[1][0]:.2f}, V accumulator={vs[1][1]:.2f}", flush=True)

    # Condition distribution
    print("\n" + "="*70, flush=True)
    print("Condition Distribution:", flush=True)
    print("="*70, flush=True)
    for c in range(4):
        count = np.sum(cond == c)
        print(f"  {cond_labels[c]}: {count} trials ({count/len(cond, flush=True)*100:.1f}%)", flush=True)

    # Response time statistics
    print("\n" + "="*70, flush=True)
    print("Response Time Statistics:", flush=True)
    print("="*70, flush=True)
    print(f"  Overall: Mean={rt.mean():.3f}s, SD={rt.std():.3f}s, Range=[{rt.min():.3f}, {rt.max():.3f}]", flush=True)
    print(flush=True)
    for c in range(4):
        rt_c = rt[cond == c]
        print(f"  {cond_labels[c]}: Mean={rt_c.mean():.3f}s, SD={rt_c.std():.3f}s", flush=True)

    # Calculate accuracy (proportion choosing dominant option)
    print("\n" + "="*70, flush=True)
    print("Accuracy by Condition (Choosing Dominant Option, flush=True):", flush=True)
    print("="*70, flush=True)
    correct_counts = []
    for c in range(4):
        mask = cond == c
        # Based on condition mapping, dominant choice should be: HV→HV, HH→HH, VH→VH, VV→VV
        # condition: 0=HV, 1=HH, 2=VH, 3=VV
        # choice: 0=HV, 1=HH, 2=VH, 3=VV
        # Therefore it's identity mapping (verified through testing)
        dominant_choice_map = {0: 0, 1: 1, 2: 2, 3: 3}  # cond -> expected choice
        expected = dominant_choice_map[c]
        n_correct = np.sum((cond == c) & (choice == expected))
        n_total = np.sum(mask)
        acc = n_correct / n_total * 100
        correct_counts.append(acc)
        print(f"  {cond_labels[c]}: {n_correct}/{n_total} ({acc:.1f}%, flush=True) chose {choice_labels[expected]}", flush=True)

    overall_accuracy = np.mean(correct_counts)
    print(f"\n  Overall Average Accuracy: {overall_accuracy:.1f}%", flush=True)

    # Prepare summary statistics dictionary
    summary_stats = {
        'n_trials': len(data),
        'overall_accuracy': overall_accuracy,
        'condition_accuracies': {cond_labels[i]: correct_counts[i] for i in range(4)},
        'mean_rt': rt.mean(),
        'sd_rt': rt.std(),
        'condition_mean_rts': {cond_labels[i]: rt[cond == i].mean() for i in range(4)}
    }

    return summary_stats

def analyze_and_save_results(trace, data, v_tensor, model_name="PS",
                              v_m_random=None, v_ms_random=None,
                              save_netcdf=True, save_trial_csv=True):
    """
    Analyze model results, compare with true values, and save outputs

    Parameters:
    -----------
    trace : InferenceData
        PyMC sampling trace/posterior
    data : array, shape (n_trials, 3)
        Observed data [RT, choice, condition]
    v_tensor : array, shape (4, 2, 2)
        Drift rate tensor used to generate data
    model_name : str
        Name of the model ("PS" or "noPS") for file naming
    v_m_random : array, optional
        True v_match values for each condition (for comparison)
    v_ms_random : array, optional
        True v_mismatch values for each condition (for comparison)
    save_netcdf : bool
        Whether to save trace to NetCDF file
    save_trial_csv : bool
        Whether to save trial-level drift rates to CSV

    Returns:
    --------
    results : dict
        Dictionary containing summary, drift_summary, and trial_df
    """

    print("\n" + "="*70, flush=True)
    print("Diagnostic Report (R-hat & ESS, flush=True):", flush=True)
    print("="*70, flush=True)

    # Basic parameters - different for PS and noPS models
    if model_name == "PS":
        summary = az.summary(trace, var_names=["v_match", "v_mismatch"])
        print("\nBasic Parameters (PS Model - Shared, flush=True):", flush=True)
        print(summary[['mean', 'r_hat', 'ess_bulk']], flush=True)
    else:  # noPS model
        param_vars = ["v_match_VH", "v_mismatch_VH",
                      "v_match_HH", "v_mismatch_HH",
                      "v_match_HV", "v_mismatch_HV",
                      "v_match_VV", "v_mismatch_VV"]
        summary = az.summary(trace, var_names=param_vars)
        print("\nBasic Parameters (noPS Model - Condition-Specific, flush=True):", flush=True)
        print(summary[['mean', 'r_hat', 'ess_bulk']], flush=True)

    # V and H drift rates for each condition
    drift_vars = ["HV_L_V", "HV_L_H", "HV_R_V", "HV_R_H",
                  "HH_L_V", "HH_L_H", "HH_R_V", "HH_R_H",
                  "VH_L_V", "VH_L_H", "VH_R_V", "VH_R_H",
                  "VV_L_V", "VV_L_H", "VV_R_V", "VV_R_H"]
    drift_summary = az.summary(trace, var_names=drift_vars)
    print("\nV and H Drift Rates for Each Condition:", flush=True)
    print(drift_summary[['mean', 'r_hat', 'ess_bulk']], flush=True)

    # True value comparison (if provided)
    print(f"\n" + "="*70, flush=True)
    print("True Value Comparison:", flush=True)
    print("="*70, flush=True)

    if model_name == "PS":
        print(f"   [PS Model] Posterior Estimates (Shared Parameters, flush=True):", flush=True)
        print(f"     v_match    = {summary.loc['v_match','mean']:.3f}", flush=True)
        print(f"     v_mismatch = {summary.loc['v_mismatch','mean']:.3f}", flush=True)

        if v_m_random is not None and v_ms_random is not None:
            print(f"\n   True Values Used to Generate Data:", flush=True)
            print(f"     H dimension: v_match={v_m_random[0]:.3f}, v_mismatch={v_ms_random[0]:.3f}", flush=True)
            print(f"     V dimension: v_match={v_m_random[2]:.3f}, v_mismatch={v_ms_random[2]:.3f}", flush=True)
    else:  # noPS model
        print(f"   [noPS Model] Posterior Estimates (Condition-Specific, flush=True):", flush=True)
        print(f"     VH: v_match={summary.loc['v_match_VH','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_VH','mean']:.3f}", flush=True)
        print(f"     HH: v_match={summary.loc['v_match_HH','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_HH','mean']:.3f}", flush=True)
        print(f"     HV: v_match={summary.loc['v_match_HV','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_HV','mean']:.3f}", flush=True)
        print(f"     VV: v_match={summary.loc['v_match_VV','mean']:.3f}, v_mismatch={summary.loc['v_mismatch_VV','mean']:.3f}", flush=True)

        if v_m_random is not None and v_ms_random is not None:
            print(f"\n   True Values Used to Generate Data:", flush=True)
            print(f"     VH (mixed, flush=True): v_match={v_m_random[2]:.3f}, v_mismatch={v_ms_random[2]:.3f}", flush=True)
            print(f"     HH (純H, flush=True):   v_match={v_m_random[1]:.3f}, v_mismatch={v_ms_random[1]:.3f}", flush=True)
            print(f"     HV (mixed, flush=True): v_match={v_m_random[2]:.3f}, v_mismatch={v_ms_random[2]:.3f}", flush=True)
            print(f"     VV (純V, flush=True):   v_match={v_m_random[0]:.3f}, v_mismatch={v_ms_random[0]:.3f}", flush=True)

    # Display posterior drift rate estimates for each condition
    print(f"\n   Posterior Drift Rate Configuration by Condition:", flush=True)
    print(f"   VH(/|, flush=True):", flush=True)
    print(f"     Left(L, flush=True):  V accumulator={drift_summary.loc['VH_L_V','mean']:.3f}, H accumulator={drift_summary.loc['VH_L_H','mean']:.3f}", flush=True)
    print(f"     Right(R, flush=True): V accumulator={drift_summary.loc['VH_R_V','mean']:.3f}, H accumulator={drift_summary.loc['VH_R_H','mean']:.3f}", flush=True)
    print(f"   HH(||, flush=True):", flush=True)
    print(f"     Left(L, flush=True):  V accumulator={drift_summary.loc['HH_L_V','mean']:.3f}, H accumulator={drift_summary.loc['HH_L_H','mean']:.3f}", flush=True)
    print(f"     Right(R, flush=True): V accumulator={drift_summary.loc['HH_R_V','mean']:.3f}, H accumulator={drift_summary.loc['HH_R_H','mean']:.3f}", flush=True)
    print(f"   HV(|/, flush=True):", flush=True)
    print(f"     Left(L, flush=True):  V accumulator={drift_summary.loc['HV_L_V','mean']:.3f}, H accumulator={drift_summary.loc['HV_L_H','mean']:.3f}", flush=True)
    print(f"     Right(R, flush=True): V accumulator={drift_summary.loc['HV_R_V','mean']:.3f}, H accumulator={drift_summary.loc['HV_R_H','mean']:.3f}", flush=True)
    print(f"   VV(//, flush=True):", flush=True)
    print(f"     Left(L, flush=True):  V accumulator={drift_summary.loc['VV_L_V','mean']:.3f}, H accumulator={drift_summary.loc['VV_L_H','mean']:.3f}", flush=True)
    print(f"     Right(R, flush=True): V accumulator={drift_summary.loc['VV_R_V','mean']:.3f}, H accumulator={drift_summary.loc['VV_R_H','mean']:.3f}", flush=True)

    # Save to NetCDF
    if save_netcdf:
        print("\n" + "="*70, flush=True)
        print("Saving Results to NetCDF...", flush=True)
        print("="*70, flush=True)
        netcdf_filename = f"{model_name.lower()}_results.nc"
        # Save complete trace including all groups (posterior, log_likelihood, sample_stats)
        # This ensures log_likelihood is saved for WAIC/LOO model comparison
        az.to_netcdf(trace, netcdf_filename)
        print(f"✓ Saved to: {netcdf_filename}", flush=True)
        print(f"  (Complete trace saved with log_likelihood for WAIC calculation)", flush=True)

    # Extract trial-level drift rates (only for PS model)
    if model_name == "PS":
        print("\n" + "="*70, flush=True)
        print("Extracting Trial-Level Drift Rates...", flush=True)
        print("="*70, flush=True)

        # Extract posterior mean
        v_match_est = summary.loc['v_match', 'mean']
        v_mismatch_est = summary.loc['v_mismatch', 'mean']

        # Extract trial-level drift rates
        trial_df = extract_trial_drift_rates(v_match_est, v_mismatch_est, data)

        print(f"\n✓ Extracted {len(trial_df, flush=True)} trials", flush=True)
        print("\nFirst 10 trials:", flush=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        print(trial_df.head(10, flush=True).to_string(index=False), flush=True)

        print("\n" + "="*70, flush=True)
        print("Trial Statistics by Condition:", flush=True)
        print("="*70, flush=True)
        grouped = trial_df.groupby('condition').agg({
            'left_drift_judge_H': 'mean',
            'left_drift_judge_V': 'mean',
            'right_drift_judge_H': 'mean',
            'right_drift_judge_V': 'mean',
            'rt': 'mean'
        })
        print("\nMean drift rates by condition:", flush=True)
        print(grouped, flush=True)

        print("\nMean winning accumulator drift rates:", flush=True)
        winner_stats = trial_df.groupby('condition').agg({
            'left_winner_drift': 'mean',
            'right_winner_drift': 'mean',
        })
        print(winner_stats, flush=True)

        # Save trial-level data
        if save_trial_csv:
            csv_filename = f"{model_name.lower()}_trial_drift_rates.csv"
            trial_df.to_csv(csv_filename, index=False)
            print(f"\n✓ Trial data saved to: {csv_filename}", flush=True)
    else:  # noPS model
        print("\n" + "="*70, flush=True)
        print("Extracting Trial-Level Drift Rates (noPS Model, flush=True)...", flush=True)
        print("="*70, flush=True)

        # Extract posterior means for each condition
        v_match_est = np.array([
            summary.loc['v_match_VH', 'mean'],
            summary.loc['v_match_HH', 'mean'],
            summary.loc['v_match_HV', 'mean'],
            summary.loc['v_match_VV', 'mean']
        ])
        v_mismatch_est = np.array([
            summary.loc['v_mismatch_VH', 'mean'],
            summary.loc['v_mismatch_HH', 'mean'],
            summary.loc['v_mismatch_HV', 'mean'],
            summary.loc['v_mismatch_VV', 'mean']
        ])

        # Extract trial-level drift rates
        trial_df = extract_trial_drift_rates_noPS(v_match_est, v_mismatch_est, data)

        print(f"\n✓ Extracted {len(trial_df, flush=True)} trials", flush=True)
        print("\nFirst 10 trials:", flush=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print(trial_df.head(10, flush=True).to_string(index=False), flush=True)

        print("\n" + "="*70, flush=True)
        print("Trial Statistics by Condition:", flush=True)
        print("="*70, flush=True)
        grouped = trial_df.groupby('condition').agg({
            'left_drift_judge_H': 'mean',
            'left_drift_judge_V': 'mean',
            'right_drift_judge_H': 'mean',
            'right_drift_judge_V': 'mean',
            'rt': 'mean',
            'cond_v_match': 'mean',
            'cond_v_mismatch': 'mean'
        })
        print("\nMean drift rates by condition:", flush=True)
        print(grouped, flush=True)

        print("\nMean winning accumulator drift rates:", flush=True)
        winner_stats = trial_df.groupby('condition').agg({
            'left_winner_drift': 'mean',
            'right_winner_drift': 'mean',
        })
        print(winner_stats, flush=True)

        # Save trial-level data
        if save_trial_csv:
            csv_filename = f"{model_name.lower()}_trial_drift_rates.csv"
            trial_df.to_csv(csv_filename, index=False)
            print(f"\n✓ Trial data saved to: {csv_filename}", flush=True)

    # Return results
    results = {
        'summary': summary,
        'drift_summary': drift_summary,
        'trial_df': trial_df,
        'v_match_est': v_match_est,
        'v_mismatch_est': v_mismatch_est,
        'grouped_stats': grouped,
        'winner_stats': winner_stats
    }

    return results

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

def build_model_PS(observed_data):
    with pm.Model() as model:
        A, b, s, t0 = 0.5, 1.5, 1.0, 0.25
        
        # 4-parameter reparameterization
        vt = pm.HalfNormal("vt", sigma=2.0)
        vb = pm.Beta("vb", 2, 2)
        v_m = pm.Deterministic("v_match", vt * vb) #vt=5 vb<5
        v_ms = pm.Deterministic("v_mismatch", vt * (1 - vb))

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
        init_vals = {"vt": 4.5, "vb": 0.7}
        trace_PS = pm.sample(draws=4000, tune=3000, chains=24, step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)
    return trace_PS
def build_model_noPS(observed_data):
    """
    Build noPS model: Each condition has independent v_match and v_mismatch parameters
    Total parameters: 8 (4 conditions × 2 parameters each)
    """
    with pm.Model() as model:
        A, b, s, t0 = 0.5, 1.5, 1.0, 0.25

        # ============================================================================
        # 重參數化 noPS：為 4 個情境分別定義 4 組 Total Speed 與 Bias
        # ============================================================================
        # 使用向量化定義 (shape=4)，一次定義所有情境
        vt = pm.HalfNormal("vt", sigma=2.0, shape=4)
        vb = pm.Beta("vb", 2, 2, shape=4)

        # 映射回各情境的 v_match 與 v_mismatch
        v_m = pm.Deterministic("v_match", vt * vb)
        v_ms = pm.Deterministic("v_mismatch", vt * (1 - vb))
        
        # 為了維持你原本 print 出來的變數名稱，可以手動指定 Deterministic
        v_m_VH = pm.Deterministic("v_match_VH", v_m[0])
        v_ms_VH = pm.Deterministic("v_mismatch_VH", v_ms[0])
        v_m_HH = pm.Deterministic("v_match_HH", v_m[1])
        v_ms_HH = pm.Deterministic("v_mismatch_HH", v_ms[1])
        v_m_HV = pm.Deterministic("v_match_HV", v_m[2])
        v_ms_HV = pm.Deterministic("v_mismatch_HV", v_ms[2])
        v_m_VV = pm.Deterministic("v_match_VV", v_m[3])
        v_ms_VV = pm.Deterministic("v_mismatch_VV", v_ms[3])
        # ============================================================================
        # Build v_tensor with condition-specific parameters
        # ============================================================================
        v_tensor = pt.zeros((4, 2, 2))
        
        # Cond 0: VH (| \) -> L:V, R:H
        v_tensor = pt.set_subtensor(v_tensor[0,0,:], [v_ms[0], v_m[0]])
        v_tensor = pt.set_subtensor(v_tensor[0,1,:], [v_m[0], v_ms[0]])
        # Cond 1: HH (| |) -> L:H, R:H
        v_tensor = pt.set_subtensor(v_tensor[1,0,:], [v_m[1], v_ms[1]])
        v_tensor = pt.set_subtensor(v_tensor[1,1,:], [v_m[1], v_ms[1]])
        # Cond 2: HV (\ |) -> L:H, R:V
        v_tensor = pt.set_subtensor(v_tensor[2,0,:], [v_m[2], v_ms[2]])
        v_tensor = pt.set_subtensor(v_tensor[2,1,:], [v_ms[2], v_m[2]])
        # Cond 3: VV (\ \) -> L:V, R:V
        v_tensor = pt.set_subtensor(v_tensor[3,0,:], [v_ms[3], v_m[3]])
        v_tensor = pt.set_subtensor(v_tensor[3,1,:], [v_ms[3], v_m[3]])
        # ============================================================================
        # Save drift rates for each condition (condition-specific values)
        # ============================================================================

        # Condition 0: VH(/|) - Left sees V, Right sees H
        pm.Deterministic("VH_L_V", v_m_VH)    # Left V accumulator (match for VH)
        pm.Deterministic("VH_L_H", v_ms_VH)   # Left H accumulator (mismatch for VH)
        pm.Deterministic("VH_R_V", v_ms_VH)   # Right V accumulator (mismatch for VH)
        pm.Deterministic("VH_R_H", v_m_VH)    # Right H accumulator (match for VH)

        # Condition 1: HH(||) - Left sees H, Right sees H
        pm.Deterministic("HH_L_V", v_ms_HH)   # Left V accumulator (mismatch for HH)
        pm.Deterministic("HH_L_H", v_m_HH)    # Left H accumulator (match for HH)
        pm.Deterministic("HH_R_V", v_ms_HH)   # Right V accumulator (mismatch for HH)
        pm.Deterministic("HH_R_H", v_m_HH)    # Right H accumulator (match for HH)

        # Condition 2: HV(|/) - Left sees H, Right sees V
        pm.Deterministic("HV_L_V", v_ms_HV)   # Left V accumulator (mismatch for HV)
        pm.Deterministic("HV_L_H", v_m_HV)    # Left H accumulator (match for HV)
        pm.Deterministic("HV_R_V", v_m_HV)    # Right V accumulator (match for HV)
        pm.Deterministic("HV_R_H", v_ms_HV)   # Right H accumulator (mismatch for HV)

        # Condition 3: VV(//) - Left sees V, Right sees V
        pm.Deterministic("VV_L_V", v_m_VV)    # Left V accumulator (match for VV)
        pm.Deterministic("VV_L_H", v_ms_VV)   # Left H accumulator (mismatch for VV)
        pm.Deterministic("VV_R_V", v_m_VV)    # Right V accumulator (match for VV)
        pm.Deterministic("VV_R_H", v_ms_VV)   # Right H accumulator (mismatch for VV)

        # ============================================================================
        # Likelihood
        # ============================================================================
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
        init_vals = {
            "vt": np.array([4.5, 3.0, 4.5, 4.0]), # 根據真值總和預估
            "vb": np.array([0.7, 0.8, 0.7, 0.7])  # 根據正確率預估
        }
        trace_noPS = pm.sample(
            draws=4000,    # 增加樣本數
            tune=3000,     # 給予更長的熱身期，讓 24 條鏈有時間匯合
            chains=24, 
            step=pm.DEMetropolisZ(), 
            random_seed=42, initvals=init_vals
        )
    return trace_noPS
# ============================================================================
# 5. Execution and Diagnostics
# ============================================================================
if __name__ == "__main__":
    print("="*70, flush=True)
    print("1. Generating 5000 2D-LBA data points...", flush=True)
   
    # Set parameters
    rng = np.random.default_rng(42)
    A, b, t0, s = 0.5, 1.5, 0.25, 1.0

    # --- Random data generation without PS assumption (same as model4_noPS.py) ---
    # Randomly sample v_match and v_mismatch for 3 parameter groups
    v_m_random = rng.uniform(1.5, 3.5, size=3)     # 3 groups: [0]=VV, [1]=HH, [2]=mixed(HV/VH)
    v_ms_random = rng.uniform(0.5, 1.5, size=3)    # 3 groups: [0]=VV, [1]=HH, [2]=mixed(HV/VH)

    print("\nRandomly generated true parameters (3 groups, flush=True):", flush=True)
    print(f"  Group [0]: v_match={v_m_random[0]:.2f}, v_mismatch={v_ms_random[0]:.2f}", flush=True)
    print(f"  Group [1]: v_match={v_m_random[1]:.2f}, v_mismatch={v_ms_random[1]:.2f}", flush=True)
    print(f"  Group [2]: v_match={v_m_random[2]:.2f}, v_mismatch={v_ms_random[2]:.2f}", flush=True)

    print("\nData Generation - PS Model Structure (shared across dimensions, flush=True):", flush=True)
    print(f"  H dimension: v_match={v_m_random[0]:.2f}, v_mismatch={v_ms_random[0]:.2f}", flush=True)
    print(f"  V dimension: v_match={v_m_random[2]:.2f}, v_mismatch={v_ms_random[2]:.2f}", flush=True)

    print("\nData Generation - noPS Model Structure (condition-specific, flush=True):", flush=True)
    print(f"  VH (mixed, flush=True): v_match={v_m_random[2]:.2f}, v_mismatch={v_ms_random[2]:.2f}", flush=True)
    print(f"  HH (純H, flush=True):   v_match={v_m_random[1]:.2f}, v_mismatch={v_ms_random[1]:.2f}", flush=True)
    print(f"  HV (mixed, flush=True): v_match={v_m_random[2]:.2f}, v_mismatch={v_ms_random[2]:.2f}", flush=True)
    print(f"  VV (純V, flush=True):   v_match={v_m_random[0]:.2f}, v_mismatch={v_ms_random[0]:.2f}", flush=True)

    # Construct a v_tensor_PS with " PS pattern"
    # condition: 0=VH, 1=HH, 2=HV, 3=VV
    # accumulator: 0=H(|), 1=V(/)
    v_tensor_PS = np.array([
        [[v_ms_random[2], v_m_random[2]], [v_m_random[0], v_ms_random[0]]],  # VH: L sees V, R sees H
        [[v_m_random[0], v_ms_random[0]], [v_m_random[0], v_ms_random[0]]],  # HH: L sees H, R sees H
        [[v_m_random[0], v_ms_random[0]], [v_ms_random[2], v_m_random[2]]],  # HV: L sees H, R sees V
        [[v_ms_random[2], v_m_random[2]], [v_ms_random[2], v_m_random[2]]]   # VV: L sees V, R sees V
    ])
    # Construct a v_tensor_noPS with " no PS pattern"
    v_tensor_noPS = np.array([
        [[v_ms_random[2], v_m_random[2]], [v_m_random[2], v_ms_random[2]]],  # VH: L sees V, R sees H
        [[v_m_random[1], v_ms_random[1]], [v_m_random[1], v_ms_random[1]]],  # HH: L sees H, R sees H
        [[v_m_random[2], v_ms_random[2]], [v_ms_random[2], v_m_random[2]]],  # HV: L sees H, R sees V
        [[v_ms_random[0], v_m_random[0]], [v_ms_random[0], v_m_random[0]]]   # VV: L sees V, R sees V
    ])
    # Generate data
    data_PS = lba_2dim_random(
        n_trials_per_condition=5000 // 4,
        v_tensor=v_tensor_PS,
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )
    data_noPS = lba_2dim_random(
        n_trials_per_condition=5000 // 4,
        v_tensor=v_tensor_noPS,
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )

    # Display data summary
    #summary_stats_PS = display_data_summary(data_PS, v_tensor_PS)
    
    
    # Run PS model sampling
    print("\n" + "="*70, flush=True)
    print("2.1 Running PS Model Sampling...", flush=True)
    print("="*70, flush=True)
    trace_PS = build_model_PS(data_PS)

    # Analyze and save PS model results
    results_PS = analyze_and_save_results(
        trace=trace_PS,
        data=data_PS,
        v_tensor=v_tensor_PS,
        model_name="PS",
        v_m_random=v_m_random,
        v_ms_random=v_ms_random,
        save_netcdf=True,
        save_trial_csv=True
    )
    summary_stats_noPS = display_data_summary(data_noPS, v_tensor_noPS)

    # Run noPS model sampling
    print("\n" + "="*70, flush=True)
    print("2.2 Running noPS Model Sampling...", flush=True)
    print("="*70, flush=True)
    trace_noPS = build_model_noPS(data_noPS)

    # Analyze and save PS model results
    results_noPS = analyze_and_save_results(
        trace=trace_noPS,
        data=data_noPS,
        v_tensor=v_tensor_noPS,
        model_name="noPS",
        v_m_random=v_m_random,
        v_ms_random=v_ms_random,
        save_netcdf=True,
        save_trial_csv=True
    )

    print("\n" + "="*70, flush=True)
    print("All analyses complete!", flush=True)
    print("="*70, flush=True)
