import os
import uuid

# 替換原有的編譯目錄，加入隨機字串避免多個 Job 互相衝突
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
import sys

warnings.filterwarnings('ignore')
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
# 2. Data Generation and PS Structure (seed=42, 5000 trials)
# ============================================================================

def lba_2dim_random(n_trials_per_condition, v_tensor, A, b, t0, s=1.0, rng=None, max_rt=5.0):
    """
    Race (Independent) LBA simulation for PS model.
    RT = min(RT_L, RT_R) - faster dimension determines overall RT.

    Args:
        max_rt: Maximum allowed RT in seconds (default 5.0). Trials exceeding this are resampled.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_data = []

    # Generate data for each condition
    # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    for cond in range(4):
        v_left = v_tensor[cond, 0, :]  # Left dimension [v1_L, v2_L]
        v_right = v_tensor[cond, 1, :]  # Right dimension [v1_R, v2_R]

        trials_generated = 0
        max_attempts = n_trials_per_condition * 10  # Prevent infinite loop
        attempts = 0

        while trials_generated < n_trials_per_condition and attempts < max_attempts:
            attempts += 1

            # Left dimension decision
            k1_L = rng.uniform(0, A)
            k2_L = rng.uniform(0, A)
            # Between-trial drift rate variability: sample from Normal(v_mean, s)
            # Standard LBA: drift rates vary across trials according to N(v, s)
            v1_L_trial = max(0.01, rng.normal(v_left[0], s))  # Ensure positive
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
            # Between-trial drift rate variability
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

            # Decision time = max of both dimensions (both must reach threshold)
            decision_time = max(min(t1_L, t2_L), min(t1_R, t2_R))

            # Add choice-specific motor time
            rt = decision_time + t0

            # Filter: RT > 0, finite, and within max_rt limit
            if rt > 0 and np.isfinite(rt) and rt <= max_rt:
                all_data.append([rt, final_choice, cond])
                trials_generated += 1

    return np.array(all_data)


def lba_4acc_random(n_trials_per_condition, v_m_groups, v_ms_groups, A, b, t0, s=1.0, rng=None, max_rt=5.0):
    """
    True 4-Accumulator Race for noPS model.

    4 response options (DS, SS, SD, DD) compete independently.
    Each accumulator races to threshold, first one wins.

    Args:
        n_trials_per_condition: Number of trials per condition
        v_m_groups: [4] array of v_match for [SS, DD, SD, DS] groups
        v_ms_groups: [4] array of v_mismatch for [SS, DD, SD, DS] groups
        A, b, t0, s: LBA parameters
        max_rt: Maximum allowed RT in seconds

    Groups mapping (4 separate groups - no merging):
        - Cond 0 (DS): DS group [3]
        - Cond 1 (SS): SS group [0]
        - Cond 2 (SD): SD group [2]
        - Cond 3 (DD): DD group [1]
    """
    if rng is None:
        rng = np.random.default_rng()

    all_data = []

    # 4 groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS (不合併 Mixed)
    cond_to_group = {0: 3, 1: 0, 2: 2, 3: 1}  # DS→DS, SS→SS, SD→SD, DD→DD

    # Generate data for each condition
    # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    for cond in range(4):
        group = cond_to_group[cond]
        v_m = v_m_groups[group]
        v_ms = v_ms_groups[group]

        # Correct response = condition index
        correct = cond

        trials_generated = 0
        max_attempts = n_trials_per_condition * 10

        while trials_generated < n_trials_per_condition and max_attempts > 0:
            max_attempts -= 1

            # 4 accumulators: DS(0), SS(1), SD(2), DD(3)
            # Correct accumulator gets v_match, others get v_mismatch
            times = []
            for j in range(4):
                v_j = v_m if j == correct else v_ms
                k = rng.uniform(0, A)
                d = max(0.01, rng.normal(v_j, s))
                t = (b - k) / d
                times.append(t)

            # Winner is the first to finish (min)
            winner = int(np.argmin(times))
            decision_time = times[winner]
            rt = decision_time + t0

            # Filter: RT > 0, finite, and within max_rt limit
            if rt > 0 and np.isfinite(rt) and rt <= max_rt:
                all_data.append([rt, winner, cond])
                trials_generated += 1

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

    # Condition labels: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    cond_labels = {0: 'DS(/|)', 1: 'SS(||)', 2: 'SD(|/)', 3: 'DD(//)'}
    choice_labels = {0: 'DS', 1: 'SS', 2: 'SD', 3: 'DD'}

    # Drift rate configuration for each condition
    # v_tensor[cond, dimension, accumulator]
    # dimension: 0=Left, 1=Right
    # accumulator: 0=judge D(/), 1=judge S(|)
    drift_configs = {
        0: {  # DS: Left sees D(/), Right sees S(|)
            'left_stim': 'D(/)',
            'right_stim': 'S(|)',
            'left_drift': [v_match, v_mismatch],  # [judge D, judge S]
            'right_drift': [v_mismatch, v_match]
        },
        1: {  # SS: Left sees S(|), Right sees S(|)
            'left_stim': 'S(|)',
            'right_stim': 'S(|)',
            'left_drift': [v_mismatch, v_match],
            'right_drift': [v_mismatch, v_match]
        },
        2: {  # SD: Left sees S(|), Right sees D(/)
            'left_stim': 'S(|)',
            'right_stim': 'D(/)',
            'left_drift': [v_mismatch, v_match],
            'right_drift': [v_match, v_mismatch]
        },
        3: {  # DD: Left sees D(/), Right sees D(/)
            'left_stim': 'D(/)',
            'right_stim': 'D(/)',
            'left_drift': [v_match, v_mismatch],
            'right_drift': [v_match, v_mismatch]
        }
    }

    # Extract information for each trial
    for i in range(len(data)):
        c = cond[i]
        ch = choice[i]
        config = drift_configs[c]

        # Parse choice: 0=DS, 1=SS, 2=SD, 3=DD
        # Left judgment: 0=D(/), 1=S(|)
        # Right judgment: 0=D(/), 1=S(|)
        choice_map = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
        left_judgment, right_judgment = choice_map[ch]

        trial_info = {
            'trial': i,
            'condition': cond_labels[c],
            'left_stim': config['left_stim'],
            'right_stim': config['right_stim'],
            'choice': choice_labels[ch],
            'judgment_L': 'S(|)' if left_judgment == 1 else 'D(/)',
            'judgment_R': 'S(|)' if right_judgment == 1 else 'D(/)',
            'rt': rt[i],

            # Left drift rates
            'left_drift_judge_D': config['left_drift'][0],
            'left_drift_judge_S': config['left_drift'][1],

            # Right drift rates
            'right_drift_judge_D': config['right_drift'][0],
            'right_drift_judge_S': config['right_drift'][1],

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
        Posterior mean of v_match for each condition [SD, DD, DS, SS]
    v_mismatch_per_cond : array, shape (4,)
        Posterior mean of v_mismatch for each condition [SD, DD, DS, SS]
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

    # Condition labels: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    cond_labels = {0: 'DS(/|)', 1: 'SS(||)', 2: 'SD(|/)', 3: 'DD(//)'}
    choice_labels = {0: 'DS', 1: 'SS', 2: 'SD', 3: 'DD'}

    # Drift rate configuration for each condition (condition-specific)
    # v_tensor[cond, dimension, accumulator]
    # dimension: 0=Left, 1=Right
    # accumulator: 0=judge D(/), 1=judge S(|)
    drift_configs = {
        0: {  # DS: Left sees D(/), Right sees S(|)
            'left_stim': 'D(/)',
            'right_stim': 'S(|)',
            'left_drift': [v_match_per_cond[0], v_mismatch_per_cond[0]],  # [judge D, judge S]
            'right_drift': [v_mismatch_per_cond[0], v_match_per_cond[0]]
        },
        1: {  # SS: Left sees S(|), Right sees S(|)
            'left_stim': 'S(|)',
            'right_stim': 'S(|)',
            'left_drift': [v_mismatch_per_cond[1], v_match_per_cond[1]],
            'right_drift': [v_mismatch_per_cond[1], v_match_per_cond[1]]
        },
        2: {  # SD: Left sees S(|), Right sees D(/)
            'left_stim': 'S(|)',
            'right_stim': 'D(/)',
            'left_drift': [v_mismatch_per_cond[2], v_match_per_cond[2]],
            'right_drift': [v_match_per_cond[2], v_mismatch_per_cond[2]]
        },
        3: {  # DD: Left sees D(/), Right sees D(/)
            'left_stim': 'D(/)',
            'right_stim': 'D(/)',
            'left_drift': [v_match_per_cond[3], v_mismatch_per_cond[3]],
            'right_drift': [v_match_per_cond[3], v_mismatch_per_cond[3]]
        }
    }

    # Extract information for each trial
    for i in range(len(data)):
        c = cond[i]
        ch = choice[i]
        config = drift_configs[c]

        # Parse choice: 0=DS, 1=SS, 2=SD, 3=DD
        # Left judgment: 0=D(/), 1=S(|)
        # Right judgment: 0=D(/), 1=S(|)
        choice_map = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
        left_judgment, right_judgment = choice_map[ch]

        trial_info = {
            'trial': i,
            'condition': cond_labels[c],
            'condition_idx': c,
            'left_stim': config['left_stim'],
            'right_stim': config['right_stim'],
            'choice': choice_labels[ch],
            'left_judgment': 'S(|)' if left_judgment == 1 else 'D(/)',
            'right_judgment': 'S(|)' if right_judgment == 1 else 'D(/)',
            'rt': rt[i],

            # Left drift rates
            'left_drift_judge_D': config['left_drift'][0],
            'left_drift_judge_S': config['left_drift'][1],

            # Right drift rates
            'right_drift_judge_D': config['right_drift'][0],
            'right_drift_judge_S': config['right_drift'][1],

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

    Naming Convention:
        Cond 0: SD (Same/Different - /|)
        Cond 1: DD (Different/Different - ||)
        Cond 2: DS (Different/Same - |/)
        Cond 3: SS (Same/Same - //)

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

    # Condition labels: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    cond_labels = {0: 'DS(/|)', 1: 'SS(||)', 2: 'SD(|/)', 3: 'DD(//)'}
    choice_labels = {0: 'DS', 1: 'SS', 2: 'SD', 3: 'DD'}

    print(f"\nTotal data points: {len(data)}", flush=True)

    # Display drift rate settings for each condition
    print("\nDrift Rate Settings for Each Condition:", flush=True)
    print("(Drift rate tensor structure: [condition, dimension, accumulator])", flush=True)
    print("  dimension: 0=Left, 1=Right", flush=True)
    print("  accumulator: 0=judge D(/), 1=judge S(|)", flush=True)

    drift_configs = v_tensor.tolist()
    for c in range(4):
        vs = drift_configs[c]
        print(f"\n  {cond_labels[c]}:", flush=True)
        print(f"    Left(L):  D accumulator={vs[0][0]:.2f}, S accumulator={vs[0][1]:.2f}", flush=True)
        print(f"    Right(R): D accumulator={vs[1][0]:.2f}, S accumulator={vs[1][1]:.2f}", flush=True)

    # Condition distribution
    print("\n" + "="*70, flush=True)
    print("Condition Distribution:", flush=True)
    print("="*70, flush=True)
    for c in range(4):
        count = np.sum(cond == c)
        print(f"  {cond_labels[c]}: {count} trials ({count/len(cond)*100:.1f}%)", flush=True)

    # Response time statistics
    print("\n" + "="*70, flush=True)
    print("Response Time Statistics:", flush=True)
    print("="*70, flush=True)
    print(f"  Overall: Mean={rt.mean():.3f}s, SD={rt.std():.3f}s, Range=[{rt.min():.3f}, {rt.max():.3f}]", flush=True)
    print("", flush=True)
    for c in range(4):
        rt_c = rt[cond == c]
        print(f"  {cond_labels[c]}: Mean={rt_c.mean():.3f}s, SD={rt_c.std():.3f}s", flush=True)

    # Calculate accuracy (proportion choosing dominant option)
    print("\n" + "="*70, flush=True)
    print("Accuracy by Condition (Choosing Dominant Option):", flush=True)
    print("="*70, flush=True)
    correct_counts = []
    for c in range(4):
        mask = cond == c
        # Based on condition mapping, dominant choice should be: DS→DS, SS→SS, SD→SD, DD→DD
        # condition: 0=DS, 1=SS, 2=SD, 3=DD
        # choice: 0=DS, 1=SS, 2=SD, 3=DD
        # Therefore it's identity mapping (verified through testing)
        dominant_choice_map = {0: 0, 1: 1, 2: 2, 3: 3}  # cond -> expected choice
        expected = dominant_choice_map[c]
        n_correct = np.sum((cond == c) & (choice == expected))
        n_total = np.sum(mask)
        acc = n_correct / n_total * 100
        correct_counts.append(acc)
        print(f"  {cond_labels[c]}: {n_correct}/{n_total} ({acc:.1f}%) chose {choice_labels[expected]}", flush=True)

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

def analyze_and_save_results(trace_PS, trace_noPS, data_PS, data_noPS,
                              v_tensor_PS, v_tensor_noPS,
                              v_m_random=None, v_ms_random=None,
                              ps_true_params=None,
                              log_file="model_recovery_results.log",
                              save_netcdf=True):
    """
    Analyze PS and noPS model results, save to .log file

    Naming Convention:
        DD (Different/Different - ||)
        SS (Same/Same - //)
        SD (Same/Different - /|)
        DS (Different/Same - |/)

    Parameters:
    -----------
    trace_PS : InferenceData
        PS model posterior trace
    trace_noPS : InferenceData
        noPS model posterior trace
    data_PS, data_noPS : array
        Observed data [RT, choice, condition]
    v_tensor_PS, v_tensor_noPS : array
        Drift rate tensors used to generate data
    v_m_random, v_ms_random : array, optional
        True parameter values for comparison
    log_file : str
        Output log file name
    save_netcdf : bool
        Whether to save traces to NetCDF

    Returns:
    --------
    results : dict
    """

    # Condition mapping: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    COND_MAP = {0: 'DS', 1: 'SS', 2: 'SD', 3: 'DD'}
    COND_LABELS = ['DS(/|)', 'SS(||)', 'SD(|/)', 'DD(//)']

    # Open log file
    with open(log_file, 'w') as f:
        def log(msg=""):
            f.write(msg + "\n")
            print(msg, flush=True)

        log("=" * 70)
        log(f"GRT-LBA Model Recovery Analysis")
        log("=" * 70)

        # ====================================================================
        # True Parameters
        # ====================================================================
        log("\n" + "=" * 70)
        log("TRUE PARAMETERS (Used to Generate Data)")
        log("=" * 70)

        if v_m_random is not None and v_ms_random is not None:
            log("\n4 Groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS (不合併 Mixed)")
            log(f"  SS (//): v_match={v_m_random[0]:.3f}, v_mismatch={v_ms_random[0]:.3f}")
            log(f"  DD (||): v_match={v_m_random[1]:.3f}, v_mismatch={v_ms_random[1]:.3f}")
            log(f"  SD (|/): v_match={v_m_random[2]:.3f}, v_mismatch={v_ms_random[2]:.3f}")
            log(f"  DS (/|): v_match={v_m_random[3]:.3f}, v_mismatch={v_ms_random[3]:.3f}")

        # ====================================================================
        # PS Model Results
        # ====================================================================
        log("\n" + "=" * 70)
        log("PS MODEL RESULTS")
        log("=" * 70)

        # PS parameters (t0 estimated, A/b/s fixed)
        ps_params = ['v_D_match', 'v_D_mismatch', 'v_S_match', 'v_S_mismatch', 't0']
        ps_summary = az.summary(trace_PS, var_names=ps_params)

        log("\nParameter Estimates (94% HDI):")
        log(f"  {'Parameter':<18} {'Mean':>8} {'SD':>8} {'HDI_3%':>8} {'HDI_97%':>8} {'R-hat':>8} {'ESS':>8}")
        log(f"  {'-'*72}")

        for param in ps_params:
            row = ps_summary.loc[param]
            log(f"  {param:<18} {row['mean']:>8.3f} {row['sd']:>8.3f} "
                f"{row['hdi_3%']:>8.3f} {row['hdi_97%']:>8.3f} "
                f"{row['r_hat']:>8.4f} {row['ess_bulk']:>8.0f}")

        # Convergence check
        ps_max_rhat = ps_summary['r_hat'].max()
        ps_min_ess = ps_summary['ess_bulk'].min()
        log(f"\n  Convergence: max(R-hat)={ps_max_rhat:.4f}, min(ESS)={ps_min_ess:.0f}")
        log(f"  Status: {'OK' if ps_max_rhat < 1.01 and ps_min_ess > 400 else 'WARNING'}")

        # Data accuracy by condition
        log("\nData Accuracy by Condition:")
        rt_PS = data_PS[:, 0]
        choice_PS = data_PS[:, 1].astype(int)
        cond_PS = data_PS[:, 2].astype(int)

        log(f"  {'Condition':<10} {'N':>6} {'Accuracy':>10} {'Mean RT':>10}")
        log(f"  {'-'*40}")
        for c in range(4):
            mask = cond_PS == c
            n = np.sum(mask)
            acc = np.mean(choice_PS[mask] == c) * 100
            mean_rt = np.mean(rt_PS[mask]) * 1000
            log(f"  {COND_LABELS[c]:<10} {n:>6} {acc:>9.1f}% {mean_rt:>9.0f}ms")

        # ====================================================================
        # noPS Model Results
        # ====================================================================
        log("\n" + "=" * 70)
        log("noPS MODEL RESULTS")
        log("=" * 70)

        # noPS parameters (t0 estimated, A/b/s fixed) - 4 groups
        nops_params = ['v_match_SS', 'v_mismatch_SS', 'v_match_DD', 'v_mismatch_DD',
                       'v_match_SD', 'v_mismatch_SD', 'v_match_DS', 'v_mismatch_DS', 't0']
        # Map to new names for display
        nops_display_names = {
            'v_match_SS': 'v_match_SS', 'v_mismatch_SS': 'v_mismatch_SS',
            'v_match_DD': 'v_match_DD', 'v_mismatch_DD': 'v_mismatch_DD',
            'v_match_SD': 'v_match_SD', 'v_mismatch_SD': 'v_mismatch_SD',
            'v_match_DS': 'v_match_DS', 'v_mismatch_DS': 'v_mismatch_DS',
            't0': 't0'
        }

        nops_summary = az.summary(trace_noPS, var_names=nops_params)

        log("\nParameter Estimates (94% HDI):")
        log(f"  {'Parameter':<18} {'Mean':>8} {'SD':>8} {'HDI_3%':>8} {'HDI_97%':>8} {'R-hat':>8} {'ESS':>8}")
        log(f"  {'-'*72}")

        for param in nops_params:
            row = nops_summary.loc[param]
            display_name = nops_display_names.get(param, param)
            log(f"  {display_name:<18} {row['mean']:>8.3f} {row['sd']:>8.3f} "
                f"{row['hdi_3%']:>8.3f} {row['hdi_97%']:>8.3f} "
                f"{row['r_hat']:>8.4f} {row['ess_bulk']:>8.0f}")

        # Convergence check
        nops_max_rhat = nops_summary['r_hat'].max()
        nops_min_ess = nops_summary['ess_bulk'].min()
        log(f"\n  Convergence: max(R-hat)={nops_max_rhat:.4f}, min(ESS)={nops_min_ess:.0f}")
        log(f"  Status: {'OK' if nops_max_rhat < 1.01 and nops_min_ess > 400 else 'WARNING'}")

        # Data accuracy by condition
        log("\nData Accuracy by Condition:")
        rt_noPS = data_noPS[:, 0]
        choice_noPS = data_noPS[:, 1].astype(int)
        cond_noPS = data_noPS[:, 2].astype(int)

        log(f"  {'Condition':<10} {'N':>6} {'Accuracy':>10} {'Mean RT':>10}")
        log(f"  {'-'*40}")
        for c in range(4):
            mask = cond_noPS == c
            n = np.sum(mask)
            acc = np.mean(choice_noPS[mask] == c) * 100
            mean_rt = np.mean(rt_noPS[mask]) * 1000
            log(f"  {COND_LABELS[c]:<10} {n:>6} {acc:>9.1f}% {mean_rt:>9.0f}ms")

        # ====================================================================
        # Parameter Recovery Comparison
        # ====================================================================
        log("\n" + "=" * 70)
        log("PARAMETER RECOVERY COMPARISON")
        log("=" * 70)

        if ps_true_params is not None or (v_m_random is not None and v_ms_random is not None):
            log("\nPS Model Recovery:")
            log(f"  {'Parameter':<18} {'True':>10} {'Estimated':>10} {'Bias':>10} {'In HDI':>8}")
            log(f"  {'-'*60}")

            # PS 使用明確定義的真實參數
            if ps_true_params is not None:
                ps_true = ps_true_params
            else:
                # Fallback: 使用舊的對應方式 (不推薦)
                ps_true = {
                    'v_D_match': v_m_random[0],
                    'v_D_mismatch': v_ms_random[0],
                    'v_S_match': v_m_random[2],
                    'v_S_mismatch': v_ms_random[2],
                    't0': 0.1,
                }

            for param in ['v_D_match', 'v_D_mismatch', 'v_S_match', 'v_S_mismatch', 't0']:
                true_val = ps_true.get(param, 1.0)
                est_val = ps_summary.loc[param, 'mean']
                hdi_low = ps_summary.loc[param, 'hdi_3%']
                hdi_high = ps_summary.loc[param, 'hdi_97%']
                bias = est_val - true_val
                in_hdi = "Yes" if hdi_low <= true_val <= hdi_high else "No"
                log(f"  {param:<18} {true_val:>10.3f} {est_val:>10.3f} {bias:>+10.3f} {in_hdi:>8}")

            log("\nnoPS Model Recovery (4 groups: SS, DD, SD, DS):")
            log(f"  {'Parameter':<18} {'True':>10} {'Estimated':>10} {'Bias':>10} {'In HDI':>8}")
            log(f"  {'-'*60}")

            nops_true = {
                'v_match_SS': v_m_random[0],      # SS
                'v_mismatch_SS': v_ms_random[0],
                'v_match_DD': v_m_random[1],      # DD
                'v_mismatch_DD': v_ms_random[1],
                'v_match_SD': v_m_random[2],      # SD
                'v_mismatch_SD': v_ms_random[2],
                'v_match_DS': v_m_random[3],      # DS
                'v_mismatch_DS': v_ms_random[3],
                't0': 0.1,                         # True non-decision time
            }

            for param in ['v_match_SS', 'v_mismatch_SS', 'v_match_DD', 'v_mismatch_DD',
                          'v_match_SD', 'v_mismatch_SD', 'v_match_DS', 'v_mismatch_DS', 't0']:
                true_val = nops_true.get(param, 1.0)
                est_val = nops_summary.loc[param, 'mean']
                hdi_low = nops_summary.loc[param, 'hdi_3%']
                hdi_high = nops_summary.loc[param, 'hdi_97%']
                bias = est_val - true_val
                in_hdi = "Yes" if hdi_low <= true_val <= hdi_high else "No"
                display_name = nops_display_names.get(param, param)
                log(f"  {display_name:<18} {true_val:>10.3f} {est_val:>10.3f} {bias:>+10.3f} {in_hdi:>8}")

        # ====================================================================
        # Summary
        # ====================================================================
        log("\n" + "=" * 70)
        log("SUMMARY")
        log("=" * 70)

        log(f"\nPS Model:   max(R-hat)={ps_max_rhat:.4f}, min(ESS)={ps_min_ess:.0f}")
        log(f"noPS Model: max(R-hat)={nops_max_rhat:.4f}, min(ESS)={nops_min_ess:.0f}")

        overall_ok = (ps_max_rhat < 1.01 and nops_max_rhat < 1.01 and
                      ps_min_ess > 400 and nops_min_ess > 400)
        log(f"\nOverall Convergence: {'GOOD' if overall_ok else 'NEEDS ATTENTION'}")

        log("\n" + "=" * 70)
        log(f"Results saved to: {log_file}")
        log("=" * 70)

    # Save NetCDF files
    if save_netcdf:
        az.to_netcdf(trace_PS, "ps_recovery_results.nc")
        az.to_netcdf(trace_noPS, "nops_recovery_results.nc")
        print(f"Saved: ps_recovery_results.nc, nops_recovery_results.nc")

    return {
        'ps_summary': ps_summary,
        'nops_summary': nops_summary,
        'log_file': log_file
    }

# ============================================================================
# 4. PyMC Likelihood Op - Race (for PS model)
# ============================================================================

class GRT_LBA_2D_PointwiseOp_Race(pt.Op):
    """
    Race (Independent) LBA for PS Model: 左右位置獨立處理，各自跑 LBA race
    RT = max(RT_L, RT_R) - 兩邊都要達到 threshold 才反應
    Likelihood: f_joint(t) = f_L(t)·F_R(t) + f_R(t)·F_L(t)
    這是 max(RT_L, RT_R) 的 PDF：其中一個在時間 t 完成，另一個已經完成
    """
    itypes = [pt.ivector, pt.dvector, pt.ivector, pt.dtensor3, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        choice, rt, cond, v_tensor, A, b, t0, s = inputs
        log_liks = np.zeros(len(rt))
        j_map = {0:(0,1), 1:(1,1), 2:(1,0), 3:(0,0)}

        A_val = float(A)
        b_val = float(b)
        t0_val = float(t0)
        s_val = float(s)

        for i in range(len(rt)):
            t = max(float(rt[i]) - t0_val, 1e-4)
            l_c, r_c = j_map[int(choice[i])]
            c_idx = int(cond[i])

            # 1. 左位置 (Left Position) - 獨立 LBA race
            v_win_L = float(v_tensor[c_idx, 0, l_c])
            v_lose_L = float(v_tensor[c_idx, 0, 1-l_c])
            fL = lba_def_pdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)
            FL = lba_def_cdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)

            # 2. 右位置 (Right Position) - 獨立 LBA race
            v_win_R = float(v_tensor[c_idx, 1, r_c])
            v_lose_R = float(v_tensor[c_idx, 1, 1-r_c])
            fR = lba_def_pdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)
            FR = lba_def_cdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)

            # 3. Race 邏輯: RT = max(RT_L, RT_R)
            # PDF of max: f_L·F_R + f_R·F_L (兩邊都要完成)
            joint_pdf = fL * FR + fR * FL

            if joint_pdf < 1e-20:
                log_liks[i] = -50.0
            else:
                log_liks[i] = np.log(joint_pdf)

        outputs[0][0] = log_liks


# ============================================================================
# PyMC Likelihood Op - 4-Accumulator Race (for noPS model)
# ============================================================================

class GRT_LBA_4AccRace_PointwiseOp(pt.Op):
    """
    4-Accumulator Race for noPS Model:
    - 4 個反應選項 (DS=0, SS=1, SD=2, DD=3) 各有獨立累加器
    - 正確選項有 v_match，其他 3 個有 v_mismatch
    - Likelihood: f[winner] × (1-F[loser1]) × (1-F[loser2]) × (1-F[loser3])

    Inputs:
        choice: 受試者的選擇 (0=DS, 1=SS, 2=SD, 3=DD)
        rt: 反應時間
        cond: 刺激條件 (0=DS, 1=SS, 2=SD, 3=DD)
        v_match: 各條件組的 v_match [SS, DD, SD, DS]
        v_mismatch: 各條件組的 v_mismatch [SS, DD, SD, DS]
        A, b, t0, s: LBA 參數
    """
    itypes = [pt.ivector, pt.dvector, pt.ivector, pt.dvector, pt.dvector,
              pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        choice, rt, cond, v_match, v_mismatch, A, b, t0, s = inputs
        log_liks = np.zeros(len(rt))

        # 4 groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS (不合併 Mixed)
        # Cond 0 (DS): DS group [3]
        # Cond 1 (SS): SS group [0]
        # Cond 2 (SD): SD group [2]
        # Cond 3 (DD): DD group [1]
        cond_to_group = {0: 3, 1: 0, 2: 2, 3: 1}

        A_val = float(A)
        b_val = float(b)
        t0_val = float(t0)
        s_val = float(s)

        for i in range(len(rt)):
            t = max(float(rt[i]) - t0_val, 1e-4)
            c_idx = int(cond[i])
            ch = int(choice[i])

            # Get parameter group for this condition
            group = cond_to_group[c_idx]
            v_m = float(v_match[group])
            v_ms = float(v_mismatch[group])

            # 4 accumulators: DS(0), SS(1), SD(2), DD(3)
            # Correct response = condition index (cond 0 → DS, cond 1 → SS, etc.)
            correct = c_idx

            # Set drift rates: correct gets v_match, others get v_mismatch
            v_all = [v_ms, v_ms, v_ms, v_ms]
            v_all[correct] = v_m

            # Compute single-accumulator PDF and CDF for all 4 accumulators
            f_all = [lba_pdf(t, v_all[j], A_val, b_val, s_val) for j in range(4)]
            F_all = [lba_cdf(t, v_all[j], A_val, b_val, s_val) for j in range(4)]

            # 4-way race likelihood: L = f[winner] × ∏(1 - F[losers])
            joint_pdf = f_all[ch]
            for j in range(4):
                if j != ch:
                    joint_pdf *= (1.0 - F_all[j])

            if joint_pdf < 1e-20:
                log_liks[i] = -50.0
            else:
                log_liks[i] = np.log(joint_pdf)

        outputs[0][0] = log_liks
def setup_universal_priors(observed_data):
    """
    自動根據輸入數據，生成適合該實驗數據尺度的先驗建議
    """
    min_rt = np.min(observed_data[:, 0])
    # 自動將 t0 上限設為最小反應時間的 90%
    suggested_t0_upper = min(0.25, min_rt * 0.9)
    return suggested_t0_upper
def build_model_PS(observed_data, tune=10000, draws=6000, chains=12):
    """
    PS Model with Stimulus Dimension separation (4 drift parameters + t0)
    - v_D_match, v_D_mismatch: D stimulus dimension (對 D/斜線 刺激的反應)
    - v_S_match, v_S_mismatch: S stimulus dimension (對 S/垂直 刺激的反應)
    - t0: non-decision time (estimated)
    - A, b, s: fixed for identifiability

    PS 假設：D 和 S 刺激的處理是獨立的（不受另一維度影響）
    """

    with pm.Model() as model:
        # 固定 A, b, s 作為比例尺
        A, b, s = 0.5, 1.1, 1.0

        # --- t0 (估計) ---
        t0 = pm.TruncatedNormal("t0", mu=0.1, sigma=0.02, lower=0.05, upper=0.2)

        # --- PS 模型：依刺激維度分離 (D vs S) ---
        # D 刺激維度 (Different/斜線/)
        vt_D = pm.TruncatedNormal("vt_D", mu=3.0, sigma=1.0, lower=0.1, upper=8.0)
        vb_D = pm.Beta("vb_D", 2, 2)
        v_D_match = pm.Deterministic("v_D_match", vt_D * vb_D)        # D(/) stim → D resp (correct)
        v_D_mismatch = pm.Deterministic("v_D_mismatch", vt_D * (1 - vb_D))  # D(/) stim → S resp (error)

        # S 刺激維度 (Same/垂直|)
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

        # PS model uses Race logic
        log_lik_vec = GRT_LBA_2D_PointwiseOp_Race()(
            pt.as_tensor_variable(observed_data[:, 1].astype('int32')),
            pt.as_tensor_variable(observed_data[:, 0]),
            pt.as_tensor_variable(observed_data[:, 2].astype('int32')),
            v_tensor,
            pt.as_tensor_variable(A, dtype='float64'),
            pt.as_tensor_variable(b, dtype='float64'),
            t0,
            pt.as_tensor_variable(s, dtype='float64')
        )

        pm.Deterministic("log_likelihood", log_lik_vec)
        pm.Potential("obs", pt.sum(log_lik_vec))

        init_vals = {
            "vt_D": 3.0, "vb_D": 0.7, "vt_S": 3.0, "vb_S": 0.7,
            "t0": 0.12,
        }
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                         step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)
    return trace

def build_model_noPS(observed_data, tune=10000, draws=6000, chains=12):
    """
    noPS Model with 4 groups (SS, DD, SD, DS) - 8 drift parameters + t0
    使用 4-Accumulator Race: 4 個反應選項各有獨立累加器競爭
    - v_match/mismatch_SS: SS condition group
    - v_match/mismatch_DD: DD condition group
    - v_match/mismatch_SD: SD condition group (不合併為 Mixed)
    - v_match/mismatch_DS: DS condition group (不合併為 Mixed)
    - t0: non-decision time (estimated)
    - A, b, s: fixed for identifiability

    Likelihood: f[winner] × (1-F[loser1]) × (1-F[loser2]) × (1-F[loser3])

    這樣可以看出 SD 和 DS 是否真的對稱
    """
    with pm.Model() as model:
        # 固定 A, b, s 作為比例尺
        A, b, s = 0.5, 1.1, 1.0

        # --- t0 (估計) ---
        t0 = pm.TruncatedNormal("t0", mu=0.1, sigma=0.02, lower=0.05, upper=0.2)

        # --- noPS 模型特定參數：違反知覺分離性 (4 groups: SS, DD, SD, DS) ---
        # 獨立參數化，不使用 vt × vb
        v_m = pm.TruncatedNormal("v_match", mu=3.0, sigma=1.0, lower=0.1, upper=8.0, shape=4)
        v_ms = pm.TruncatedNormal("v_mismatch", mu=1.0, sigma=1.0, lower=0.1, upper=8.0, shape=4)

        # Named parameters for each group: [0]=SS, [1]=DD, [2]=SD, [3]=DS
        v_m_SS = pm.Deterministic("v_match_SS", v_m[0])
        v_ms_SS = pm.Deterministic("v_mismatch_SS", v_ms[0])
        v_m_DD = pm.Deterministic("v_match_DD", v_m[1])
        v_ms_DD = pm.Deterministic("v_mismatch_DD", v_ms[1])
        v_m_SD = pm.Deterministic("v_match_SD", v_m[2])
        v_ms_SD = pm.Deterministic("v_mismatch_SD", v_ms[2])
        v_m_DS = pm.Deterministic("v_match_DS", v_m[3])
        v_ms_DS = pm.Deterministic("v_mismatch_DS", v_ms[3])

        # noPS model uses 4-Accumulator Race
        # 4 response options (DS, SS, SD, DD) compete independently
        # Correct response gets v_match, wrong responses get v_mismatch
        log_lik_vec = GRT_LBA_4AccRace_PointwiseOp()(
            pt.as_tensor_variable(observed_data[:, 1].astype('int32')),  # choice
            pt.as_tensor_variable(observed_data[:, 0]),                   # rt
            pt.as_tensor_variable(observed_data[:, 2].astype('int32')),  # cond
            v_m,   # v_match vector [SS, DD, SD, DS]
            v_ms,  # v_mismatch vector [SS, DD, SD, DS]
            pt.as_tensor_variable(A, dtype='float64'),
            pt.as_tensor_variable(b, dtype='float64'),
            t0,
            pt.as_tensor_variable(s, dtype='float64')
        )

        pm.Deterministic("log_likelihood", log_lik_vec)
        pm.Potential("obs", pt.sum(log_lik_vec))
        init_vals = {
            "v_match": np.array([2.5, 2.5, 2.5, 2.5]),     # 4 groups: SS, DD, SD, DS
            "v_mismatch": np.array([1.0, 1.0, 1.0, 1.0]),  # 4 groups: SS, DD, SD, DS
            "t0": 0.12,
        }
        trace_noPS = pm.sample(draws=draws, tune=tune, chains=chains,
                         step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)

    return trace_noPS
# ============================================================================
# 5. Execution and Diagnostics
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run PS and/or noPS model recovery')
    parser.add_argument('--model', type=str, default='both', choices=['ps', 'nops', 'both'],
                        help='Which model to run: ps, nops, or both (default: both)')
    args = parser.parse_args()

    run_ps = args.model in ['ps', 'both']
    run_nops = args.model in ['nops', 'both']

    print("="*70, flush=True)
    print(f"1. Generating 5000 2D-LBA data points... (mode: {args.model})", flush=True)
   
    # Set parameters
    rng = np.random.default_rng(42)
    
    A = 0.5
    s = 1.0
    b = 1.1   # Threshold (fixed for data generation)
    t0 = 0.1  # Non-decision time (fixed for data generation)

    # --- Random data generation without PS assumption ---
    # Randomly sample v_match and v_mismatch for 4 parameter groups (不合併 Mixed)
    v_m_random = rng.uniform(1.5, 3.5, size=4)     # 4 groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS
    v_ms_random = rng.uniform(0.5, 1.5, size=4)    # 4 groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS

    print("\nRandomly generated true parameters (4 groups - SD/DS 分開):", flush=True)
    print(f"  Group [0] SS: v_match={v_m_random[0]:.2f}, v_mismatch={v_ms_random[0]:.2f}", flush=True)
    print(f"  Group [1] DD: v_match={v_m_random[1]:.2f}, v_mismatch={v_ms_random[1]:.2f}", flush=True)
    print(f"  Group [2] SD: v_match={v_m_random[2]:.2f}, v_mismatch={v_ms_random[2]:.2f}", flush=True)
    print(f"  Group [3] DS: v_match={v_m_random[3]:.2f}, v_mismatch={v_ms_random[3]:.2f}", flush=True)
    print("  注意: SD 和 DS 現在是獨立估計，可以檢查是否對稱", flush=True)

    # --- PS Model: 明確定義獨立的真實參數 ---
    # H 維度 (D/斜線刺激) 和 V 維度 (S/垂直刺激) 的處理獨立
    true_v_D_match = rng.uniform(2.5, 3.5)      # D(/) 刺激 → D 判斷 (正確)
    true_v_D_mismatch = rng.uniform(0.5, 1.2)   # D(/) 刺激 → S 判斷 (錯誤)
    true_v_S_match = rng.uniform(2.0, 3.0)      # S(|) 刺激 → S 判斷 (正確)
    true_v_S_mismatch = rng.uniform(0.5, 1.2)   # S(|) 刺激 → D 判斷 (錯誤)

    print("\nData Generation - PS Model Structure (Perceptual Separability):", flush=True)
    print(f"  H/D(/) dimension: v_match={true_v_D_match:.2f}, v_mismatch={true_v_D_mismatch:.2f}", flush=True)
    print(f"  V/S(|) dimension: v_match={true_v_S_match:.2f}, v_mismatch={true_v_S_mismatch:.2f}", flush=True)

    # Construct v_tensor_PS with PS pattern
    # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    # accumulator: 0=D(/), 1=S(|)
    # H/D stim: [v_D_match, v_D_mismatch] = [judge_D_correct, judge_S_error]
    # V/S stim: [v_S_mismatch, v_S_match] = [judge_D_error, judge_S_correct]
    v_tensor_PS = np.array([
        [[true_v_D_match, true_v_D_mismatch], [true_v_S_mismatch, true_v_S_match]],  # DS: L=D(/), R=S(|)
        [[true_v_S_mismatch, true_v_S_match], [true_v_S_mismatch, true_v_S_match]],  # SS: L=S(|), R=S(|)
        [[true_v_S_mismatch, true_v_S_match], [true_v_D_match, true_v_D_mismatch]],  # SD: L=S(|), R=D(/)
        [[true_v_D_match, true_v_D_mismatch], [true_v_D_match, true_v_D_mismatch]]   # DD: L=D(/), R=D(/)
    ])
    # Generate PS data using 2D structure (max of two positions)
    data_PS = lba_2dim_random(
        n_trials_per_condition=5000 // 4,
        v_tensor=v_tensor_PS,
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )

    # Generate noPS data using true 4-Accumulator Race
    # Groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS (不合併 Mixed)
    data_noPS = lba_4acc_random(
        n_trials_per_condition=5000 // 4,
        v_m_groups=v_m_random,    # [SS, DD, SD, DS]
        v_ms_groups=v_ms_random,  # [SS, DD, SD, DS]
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )

    # v_tensor_noPS for analysis/plotting (kept for compatibility)
    # 4 groups: [0]=SS, [1]=DD, [2]=SD, [3]=DS
    v_tensor_noPS = np.array([
        [[v_m_random[3], v_ms_random[3]], [v_ms_random[3], v_m_random[3]]],  # DS: DS group [3]
        [[v_ms_random[0], v_m_random[0]], [v_ms_random[0], v_m_random[0]]],  # SS: SS group [0]
        [[v_ms_random[2], v_m_random[2]], [v_m_random[2], v_ms_random[2]]],  # SD: SD group [2]
        [[v_m_random[1], v_ms_random[1]], [v_m_random[1], v_ms_random[1]]]   # DD: DD group [1]
    ])


    
    
    trace_PS = None
    trace_noPS = None

    # Run PS model sampling
    if run_ps:
        print("\n" + "="*70, flush=True)
        print("2.1 Running PS Model Sampling...", flush=True)
        print("="*70, flush=True)
        trace_PS = build_model_PS(data_PS, tune=10000, draws=6000, chains=12)

        # Save PS trace immediately
        ps_nc_file = "trace_PS.nc"
        trace_PS.to_netcdf(ps_nc_file)
        print(f"PS trace saved to: {ps_nc_file}", flush=True)

    # Run noPS model sampling
    if run_nops:
        print("\n" + "="*70, flush=True)
        print("2.2 Running noPS Model Sampling...", flush=True)
        print("="*70, flush=True)
        trace_noPS = build_model_noPS(data_noPS, tune=10000, draws=6000, chains=12)

        # Save noPS trace immediately
        nops_nc_file = "trace_noPS.nc"
        trace_noPS.to_netcdf(nops_nc_file)
        print(f"noPS trace saved to: {nops_nc_file}", flush=True)

    # Analyze and save both models to .log file
    print("\n" + "="*70, flush=True)
    print("3. Analyzing and Saving Results...", flush=True)
    print("="*70, flush=True)

    # Analyze results
    if run_ps and run_nops:
        # PS model 的真實參數
        ps_true_params = {
            'v_D_match': true_v_D_match,
            'v_D_mismatch': true_v_D_mismatch,
            'v_S_match': true_v_S_match,
            'v_S_mismatch': true_v_S_mismatch,
            'b': b,
            't0': t0,
        }
        results = analyze_and_save_results(
            trace_PS=trace_PS,
            trace_noPS=trace_noPS,
            data_PS=data_PS,
            data_noPS=data_noPS,
            v_tensor_PS=v_tensor_PS,
            v_tensor_noPS=v_tensor_noPS,
            v_m_random=v_m_random,
            v_ms_random=v_ms_random,
            ps_true_params=ps_true_params,
            log_file="model_recovery_both_results.log",
            save_netcdf=False
        )
        print(f"\nResults saved to: {results['log_file']}", flush=True)

    elif run_nops:
        # Only noPS - simple summary
        print("\n" + "="*70, flush=True)
        print("noPS Model Summary (4 groups: SS, DD, SD, DS)", flush=True)
        print("="*70, flush=True)
        nops_params = ['v_match_SS', 'v_mismatch_SS', 'v_match_DD', 'v_mismatch_DD',
                       'v_match_SD', 'v_mismatch_SD', 'v_match_DS', 'v_mismatch_DS', 't0']
        nops_summary = az.summary(trace_noPS, var_names=nops_params)
        print(nops_summary)

        # Parameter recovery check
        print("\nParameter Recovery:")
        true_vals = {
            'v_match_SS': v_m_random[0], 'v_mismatch_SS': v_ms_random[0],
            'v_match_DD': v_m_random[1], 'v_mismatch_DD': v_ms_random[1],
            'v_match_SD': v_m_random[2], 'v_mismatch_SD': v_ms_random[2],
            'v_match_DS': v_m_random[3], 'v_mismatch_DS': v_ms_random[3],
            't0': t0
        }
        for param in nops_params:
            true_val = true_vals[param]
            est_val = nops_summary.loc[param, 'mean']
            hdi_low = nops_summary.loc[param, 'hdi_3%']
            hdi_high = nops_summary.loc[param, 'hdi_97%']
            in_hdi = "Yes" if hdi_low <= true_val <= hdi_high else "No"
            bias = est_val - true_val
            print(f"  {param:<18} True={true_val:.3f} Est={est_val:.3f} Bias={bias:+.3f} InHDI={in_hdi}")

        # Check symmetry between SD and DS
        print("\n  --- SD vs DS 對稱性檢查 ---")
        print(f"  v_match:    SD={nops_summary.loc['v_match_SD', 'mean']:.3f} vs DS={nops_summary.loc['v_match_DS', 'mean']:.3f}")
        print(f"  v_mismatch: SD={nops_summary.loc['v_mismatch_SD', 'mean']:.3f} vs DS={nops_summary.loc['v_mismatch_DS', 'mean']:.3f}")

    elif run_ps:
        # Only PS - simple summary
        print("\n" + "="*70, flush=True)
        print("PS Model Summary", flush=True)
        print("="*70, flush=True)
        ps_params = ['v_D_match', 'v_D_mismatch', 'v_S_match', 'v_S_mismatch', 't0']
        ps_summary = az.summary(trace_PS, var_names=ps_params)
        print(ps_summary)

    print("\n" + "="*70, flush=True)
    print("Analysis complete!", flush=True)
    print("="*70, flush=True)
