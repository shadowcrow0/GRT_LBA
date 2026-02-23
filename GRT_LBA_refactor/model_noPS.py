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
# 2. Data Generation and PS Structure (seed=42, 5000 trials)
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
# 3. Extract Drift Rates for Each Trial
# ============================================================================

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

    # Condition labels: 0=SD, 1=DD, 2=DS, 3=SS
    cond_labels = {0: 'SD(/|)', 1: 'DD(||)', 2: 'DS(|/)', 3: 'SS(//)'}
    choice_labels = {0: 'SD', 1: 'DD', 2: 'DS', 3: 'SS'}

    # Drift rate configuration for each condition (condition-specific)
    # v_tensor[cond, dimension, accumulator]
    # dimension: 0=Left, 1=Right
    # accumulator: 0=judge D(|), 1=judge S(/)
    drift_configs = {
        0: {  # SD: Left sees S(/), Right sees D(|)
            'left_stim': 'S(/)',
            'right_stim': 'D(|)',
            'left_drift': [v_mismatch_per_cond[0], v_match_per_cond[0]],  # [judge D, judge S]
            'right_drift': [v_match_per_cond[0], v_mismatch_per_cond[0]]
        },
        1: {  # DD: Left sees D(|), Right sees D(|)
            'left_stim': 'D(|)',
            'right_stim': 'D(|)',
            'left_drift': [v_match_per_cond[1], v_mismatch_per_cond[1]],
            'right_drift': [v_match_per_cond[1], v_mismatch_per_cond[1]]
        },
        2: {  # DS: Left sees D(|), Right sees S(/)
            'left_stim': 'D(|)',
            'right_stim': 'S(/)',
            'left_drift': [v_match_per_cond[2], v_mismatch_per_cond[2]],
            'right_drift': [v_mismatch_per_cond[2], v_match_per_cond[2]]
        },
        3: {  # SS: Left sees S(/), Right sees S(/)
            'left_stim': 'S(/)',
            'right_stim': 'S(/)',
            'left_drift': [v_mismatch_per_cond[3], v_match_per_cond[3]],
            'right_drift': [v_mismatch_per_cond[3], v_match_per_cond[3]]
        }
    }

    # Extract information for each trial
    for i in range(len(data)):
        c = cond[i]
        ch = choice[i]
        config = drift_configs[c]

        # Parse choice: 0=SD, 1=DD, 2=DS, 3=SS
        # Left judgment: 0=D(|), 1=S(/)
        # Right judgment: 0=D(|), 1=S(/)
        choice_map = {0: (1, 0), 1: (0, 0), 2: (0, 1), 3: (1, 1)}
        left_judgment, right_judgment = choice_map[ch]

        trial_info = {
            'trial': i,
            'condition': cond_labels[c],
            'condition_idx': c,
            'left_stim': config['left_stim'],
            'right_stim': config['right_stim'],
            'choice': choice_labels[ch],
            'left_judgment': 'S(/)' if left_judgment == 1 else 'D(|)',
            'right_judgment': 'S(/)' if right_judgment == 1 else 'D(|)',
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

    # Condition labels: 0=SD, 1=DD, 2=DS, 3=SS
    cond_labels = {0: 'SD(/|)', 1: 'DD(||)', 2: 'DS(|/)', 3: 'SS(//)'}
    choice_labels = {0: 'SD', 1: 'DD', 2: 'DS', 3: 'SS'}

    print(f"\nTotal data points: {len(data)}", flush=True)

    # Display drift rate settings for each condition
    print("\nDrift Rate Settings for Each Condition:", flush=True)
    print("(Drift rate tensor structure: [condition, dimension, accumulator])", flush=True)
    print("  dimension: 0=Left, 1=Right", flush=True)
    print("  accumulator: 0=judge D(|), 1=judge S(/)", flush=True)

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
        # Based on condition mapping, dominant choice should be: SD→SD, DD→DD, DS→DS, SS→SS
        # condition: 0=SD, 1=DD, 2=DS, 3=SS
        # choice: 0=SD, 1=DD, 2=DS, 3=SS
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

def analyze_and_save_results( trace_noPS, data_noPS,
                              v_m_random=None, v_ms_random=None,
                              log_file="model_recovery_results.log",
                              save_netcdf=True):
    """
    Analyze PS and noPS model results, save to .log file

    Parameters:
    -----------
    trace_noPS : InferenceData
        noPS model posterior trace
    data_noPS : array
        Observed data [RT, choice, condition]
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
    import datetime

    # Condition mapping: 0=SD(/|), 1=DD(||), 2=DS(|/), 3=SS(//)
    COND_MAP = {0: 'SD', 1: 'DD', 2: 'DS', 3: 'SS'}
    COND_LABELS = ['SD(/|)', 'DD(||)', 'DS(|/)', 'SS(//)']

    # Open log file
    with open(log_file, 'w') as f:
        def log(msg=""):
            f.write(msg + "\n")
            print(msg, flush=True)

        log("=" * 70)
        log(f"GRT-LBA Model Recovery Analysis")
        log(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("=" * 70)

        # ====================================================================
        # True Parameters
        # ====================================================================
        log("\n" + "=" * 70)
        log("TRUE PARAMETERS (Used to Generate Data)")
        log("=" * 70)

        if v_m_random is not None and v_ms_random is not None:
            log("\n3 Groups: [0]=SS, [1]=DD, [2]=Mixed(SD/DS)")
            log(f"  SS (//):     v_match={v_m_random[0]:.3f}, v_mismatch={v_ms_random[0]:.3f}")
            log(f"  DD (||):     v_match={v_m_random[1]:.3f}, v_mismatch={v_ms_random[1]:.3f}")
            log(f"  Mixed:       v_match={v_m_random[2]:.3f}, v_mismatch={v_ms_random[2]:.3f}")




        # ====================================================================
        # noPS Model Results
        # ====================================================================
        log("\n" + "=" * 70)
        log("noPS MODEL RESULTS")
        log("=" * 70)

        # noPS parameters (using new naming) - b and t0 are fixed, not estimated
        nops_params = ['v_match_SS', 'v_mismatch_SS', 'v_match_DD', 'v_mismatch_DD',
                       'v_match_Mixed', 'v_mismatch_Mixed']
        # Map to new names for display
        nops_display_names = {
            'v_match_SS': 'v_match_SS', 'v_mismatch_SS': 'v_mismatch_SS',
            'v_match_DD': 'v_match_DD', 'v_mismatch_DD': 'v_mismatch_DD',
            'v_match_Mixed': 'v_match_Mixed', 'v_mismatch_Mixed': 'v_mismatch_Mixed'
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

        if v_m_random is not None and v_ms_random is not None:
            nops_true = {
                'v_match_SS': v_m_random[0],      # SS
                'v_mismatch_SS': v_ms_random[0],
                'v_match_DD': v_m_random[1],      # DD
                'v_mismatch_DD': v_ms_random[1],
                'v_match_Mixed': v_m_random[2],   # Mixed
                'v_mismatch_Mixed': v_ms_random[2],
            }

            for param in ['v_match_SS', 'v_mismatch_SS', 'v_match_DD', 'v_mismatch_DD',
                          'v_match_Mixed', 'v_mismatch_Mixed']:
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

        log(f"noPS Model: max(R-hat)={nops_max_rhat:.4f}, min(ESS)={nops_min_ess:.0f}")

        log("\n" + "=" * 70)
        log(f"Results saved to: {log_file}")
        log("=" * 70)

    # Save NetCDF files
    if save_netcdf:
        az.to_netcdf(trace_noPS, "nops_recovery_results.nc")
        print(f"Saved: nops_recovery_results.nc")

    return {
        'nops_summary': nops_summary,
        'log_file': log_file
    }

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


def setup_universal_priors(observed_data):
    """
    自動根據輸入數據，生成適合該實驗數據尺度的先驗建議
    """
    min_rt = np.min(observed_data[:, 0])
    # 解除 t0 的人為上限，只保留 min_rt - 0.02 的緩衝
    suggested_t0_upper = min_rt - 0.02
    return suggested_t0_upper

def build_model_noPS(observed_data, tune=6000, draws=10000, chains=12):
    """
    noPS Model with 3 groups (SS, DD, Mixed) - 6 drift parameters
    b and t0 are fixed (b=1.1, t0=0.1)
    """
    with pm.Model() as model:
        # 固定 A 和 s 作為比例尺
        A, s = 0.5, 1.0

        # --- 共享參數 (固定值) ---
        b = 1.1   # Fixed threshold
        t0 = 0.1  # Fixed non-decision time

        # --- noPS 模型特定參數：刺激交互 (3 groups) ---
        vt = pm.TruncatedNormal("vt", mu=3.0, sigma=1.0, lower=0.1, upper=8.0, shape=3)
        vb = pm.Beta("vb", 2, 2, shape=3)

        v_m = pm.Deterministic("v_match", vt * vb)
        v_ms = pm.Deterministic("v_mismatch", vt * (1 - vb))

        # Named parameters for each group
        v_m_SS = pm.Deterministic("v_match_SS", v_m[0])
        v_ms_SS = pm.Deterministic("v_mismatch_SS", v_ms[0])
        v_m_DD = pm.Deterministic("v_match_DD", v_m[1])
        v_ms_DD = pm.Deterministic("v_mismatch_DD", v_ms[1])
        v_m_Mixed = pm.Deterministic("v_match_Mixed", v_m[2])
        v_ms_Mixed = pm.Deterministic("v_mismatch_Mixed", v_ms[2])

        # Build v_tensor
        # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
        # Groups: [0]=SS, [1]=DD, [2]=Mixed(DS/SD)
        v_tensor = pt.zeros((4, 2, 2))
        # Cond 0: DS -> Mixed [2], Left=D(/), Right=S(|)
        v_tensor = pt.set_subtensor(v_tensor[0,0,:], [v_m[2], v_ms[2]])    # D stim
        v_tensor = pt.set_subtensor(v_tensor[0,1,:], [v_ms[2], v_m[2]])    # S stim
        # Cond 1: SS -> SS [0], Left=S(|), Right=S(|)
        v_tensor = pt.set_subtensor(v_tensor[1,0,:], [v_ms[0], v_m[0]])    # S stim
        v_tensor = pt.set_subtensor(v_tensor[1,1,:], [v_ms[0], v_m[0]])    # S stim
        # Cond 2: SD -> Mixed [2], Left=S(|), Right=D(/)
        v_tensor = pt.set_subtensor(v_tensor[2,0,:], [v_ms[2], v_m[2]])    # S stim
        v_tensor = pt.set_subtensor(v_tensor[2,1,:], [v_m[2], v_ms[2]])    # D stim
        # Cond 3: DD -> DD [1], Left=D(/), Right=D(/)
        v_tensor = pt.set_subtensor(v_tensor[3,0,:], [v_m[1], v_ms[1]])    # D stim
        v_tensor = pt.set_subtensor(v_tensor[3,1,:], [v_m[1], v_ms[1]])    # D stim

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
            "vt": np.array([4.5, 3.0, 4.0]),  # 3 groups: [0]=SS, [1]=DD, [2]=Mixed
            "vb": np.array([0.7, 0.8, 0.7]),  # 3 groups: [0]=SS, [1]=DD, [2]=Mixed
        }
        trace_noPS = pm.sample(draws=draws, tune=tune, chains=chains,
                         step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)
        
    return trace_noPS
# ============================================================================
# 5. Execution and Diagnostics
# ============================================================================
if __name__ == "__main__":
    print("="*70, flush=True)
    print("1. Generating 10000 2D-LBA data points...", flush=True)
   
    # Set parameters
    rng = np.random.default_rng(42)
    
    A = 0.5
    s = 1.0
    b = 1.1   # Threshold (fixed for data generation)
    t0 = 0.1  # Non-decision time (fixed for data generation)

    # --- Random data generation without PS assumption (same as model4_noPS.py) ---
    # Randomly sample v_match and v_mismatch for 3 parameter groups
    v_m_random = rng.uniform(1.5, 3.5, size=3)     # 3 groups: [0]=SS, [1]=DD, [2]=Mixed(DS/SD)
    v_ms_random = rng.uniform(0.5, 1.5, size=3)    # 3 groups: [0]=SS, [1]=DD, [2]=Mixed(DS/SD)

    print("\nRandomly generated true parameters (3 groups):", flush=True)
    print(f"  Group [0] SS:    v_match={v_m_random[0]:.2f}, v_mismatch={v_ms_random[0]:.2f}", flush=True)
    print(f"  Group [1] DD:    v_match={v_m_random[1]:.2f}, v_mismatch={v_ms_random[1]:.2f}", flush=True)
    print(f"  Group [2] Mixed: v_match={v_m_random[2]:.2f}, v_mismatch={v_ms_random[2]:.2f}", flush=True)

    # Construct a v_tensor_noPS with "noPS pattern"
    # Naming: 0=DS(/|), 1=SS(||), 2=SD(|/), 3=DD(//)
    # Groups: [0]=SS, [1]=DD, [2]=Mixed(DS/SD)
    v_tensor_noPS = np.array([
        [[v_m_random[2], v_ms_random[2]], [v_ms_random[2], v_m_random[2]]],  # DS: Mixed, L=D(/), R=S(|)
        [[v_ms_random[0], v_m_random[0]], [v_ms_random[0], v_m_random[0]]],  # SS: SS group, L=S(|), R=S(|)
        [[v_ms_random[2], v_m_random[2]], [v_m_random[2], v_ms_random[2]]],  # SD: Mixed, L=S(|), R=D(/)
        [[v_m_random[1], v_ms_random[1]], [v_m_random[1], v_ms_random[1]]]   # DD: DD group, L=D(/), R=D(/)
    ])

    data_noPS = lba_2dim_random(
        n_trials_per_condition=10000 // 4,
        v_tensor=v_tensor_noPS,
        A=A, b=b, t0=t0, s=s,
        rng=rng
    )


    
    
   # Run noPS model sampling
    print("\n" + "="*70, flush=True)
    print("2.2 Running noPS Model Sampling...", flush=True)
    print("="*70, flush=True)
    trace_noPS = build_model_noPS(data_noPS, tune=6000, draws=10000, chains=12)

    # Analyze and save both models to .log file
    print("\n" + "="*70, flush=True)
    print("3. Analyzing and Saving Results...", flush=True)
    print("="*70, flush=True)

    results = analyze_and_save_results(
        trace_noPS=trace_noPS,
        data_noPS=data_noPS,
        v_m_random=v_m_random,
        v_ms_random=v_ms_random,
        log_file="model_recovery_noPS_results.log",
        save_netcdf=True
    )

    print("\n" + "="*70, flush=True)
    print("All analyses complete!", flush=True)
    print(f"Results saved to: {results['log_file']}", flush=True)
    print("="*70, flush=True)
