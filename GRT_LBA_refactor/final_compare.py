"""
PS vs noPS Model Comparison Analysis
比較 Perceptual Separability 和 No Perceptual Separability 兩個模型

Usage:
  python final_compare.py run <participant_id>   # Run models and compare
  python final_compare.py compare <participant_id>  # Compare existing results
  python final_compare.py 33   # Shorthand for 'run 33'
"""

import sys
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pytensor.tensor as pt
import pymc as pm
from numba import jit
import math
import xarray as xr

sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# NUMBA High-Efficiency Computation
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
    if cdf < 0.0: cdf = 0.0
    elif cdf > 1.0 - 1e-15: cdf = 1.0 - 1e-15
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
# PyMC Model Components
# ============================================================================

class GRT_LBA_2D_PointwiseOp(pt.Op):
    itypes = [pt.ivector, pt.dvector, pt.ivector, pt.dtensor3, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        choice, rt, cond, v_tensor, A, b, t0, s = inputs
        log_liks = np.zeros(len(rt))
        j_map = {0:(1,0), 1:(0,0), 2:(0,1), 3:(1,1)}

        A_val, b_val, t0_val, s_val = float(A), float(b), float(t0), float(s)

        for i in range(len(rt)):
            t = max(float(rt[i]) - t0_val, 1e-5)
            l_c, r_c = j_map[int(choice[i])]
            c_idx = int(cond[i])

            v_win_L = float(v_tensor[c_idx,0,l_c])
            v_lose_L = float(v_tensor[c_idx,0,1-l_c])
            fL = lba_def_pdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)
            FL = lba_def_cdf(t, v_win_L, v_lose_L, A_val, b_val, s_val)

            v_win_R = float(v_tensor[c_idx,1,r_c])
            v_lose_R = float(v_tensor[c_idx,1,1-r_c])
            fR = lba_def_pdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)
            FR = lba_def_cdf(t, v_win_R, v_lose_R, A_val, b_val, s_val)

            log_liks[i] = np.log(fL * FR + fR * FL + 1e-20)

        outputs[0][0] = log_liks

# ============================================================================
# Data Loading
# ============================================================================

def load_grt_lba_data(csv_path, rt_min=0.2, rt_max=2.0, participant_id=None, min_accuracy=0.70):
    """Load GRT_LBA.csv and convert to model format"""
    print(f"Loading data from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)

    if participant_id is not None:
        df = df[df['participant'] == participant_id].copy()
        print(f"Using participant {participant_id} only", flush=True)

        correct = (df['Response'] == df['stim_condition']).sum()
        acc = correct / len(df)
        print(f"Participant {participant_id} accuracy: {acc*100:.1f}%", flush=True)

        if acc < min_accuracy:
            raise ValueError(f"Participant {participant_id} accuracy ({acc*100:.1f}%) below threshold ({min_accuracy*100:.0f}%)")

    df = df[(df['RT'] >= rt_min) & (df['RT'] <= rt_max)].copy()
    df['Response'] = df['Response'].astype(int)

    data = np.column_stack([df['RT'].values, df['Response'].values, df['stim_condition'].values])

    print(f"Loaded {len(data)} trials (RT range: {rt_min}-{rt_max}s)", flush=True)

    cond_labels = {0: 'VH(/|)', 1: 'HH(||)', 2: 'HV(|/)', 3: 'VV(//)'}
    print("\nCondition distribution:", flush=True)
    for c in range(4):
        count = np.sum(data[:, 2] == c)
        print(f"  {cond_labels[c]}: {count} trials ({count/len(data)*100:.1f}%)", flush=True)

    return data, df

# ============================================================================
# Model Building Functions (NEW STRUCTURE)
# ============================================================================

def setup_universal_priors(observed_data):
    """
    自動根據輸入數據，生成適合該實驗數據尺度的先驗建議
    """
    min_rt = np.min(observed_data[:, 0])
    # 自動將 t0 上限設為最小反應時間的 90%
    suggested_t0_upper = min(0.25, min_rt * 0.9)
    return suggested_t0_upper


def build_model_PS(observed_data, chains=24, draws=20000, tune=40000):
    """
    PS Model with Left/Right separation (4 drift parameters + b, t0)
    - v_match_L, v_mismatch_L: Left dimension
    - v_match_R, v_mismatch_R: Right dimension
    - b: threshold (estimated)
    - t0: non-decision time (estimated, dynamic upper bound)

    修正版：動態計算 t0 上限，確保 t0 < min(RT)
    """
    # 根據該位受試者的最小 RT 動態計算 t0 上限
    min_rt = np.min(observed_data[:, 0])
    t0_limit = min(0.25, min_rt * 0.9)  # 確保 t0 永遠小於最小反應時間

    with pm.Model() as model:
        # 固定 A 和 s 作為比例尺
        A, s = 0.5, 1.0

        # --- 共享參數 (PS 與 noPS 邏輯一致) ---
        # 門檻 b: 非常窄的 informed prior (0.8~1.2)
        b = pm.TruncatedNormal("b", mu=1.0, sigma=0.1, lower=0.8, upper=1.2)
        # 非決策時間 t0: 動態上限，解決收斂問題
        t0 = pm.Uniform("t0", lower=0.01, upper=t0_limit)

        # --- PS 模型特定參數：維度分離 ---
        # 限制 v 的上限為 8.0，防止 Chain 噴發
        vt_L = pm.TruncatedNormal("vt_L", mu=3.0, sigma=1.0, lower=0.1, upper=8.0)
        vb_L = pm.Beta("vb_L", 2, 2)
        v_m_L = pm.Deterministic("v_match_L", vt_L * vb_L)
        v_ms_L = pm.Deterministic("v_mismatch_L", vt_L * (1 - vb_L))

        vt_R = pm.TruncatedNormal("vt_R", mu=3.0, sigma=1.0, lower=0.1, upper=8.0)
        vb_R = pm.Beta("vb_R", 2, 2)
        v_m_R = pm.Deterministic("v_match_R", vt_R * vb_R)
        v_ms_R = pm.Deterministic("v_mismatch_R", vt_R * (1 - vb_R))

        # Build v_tensor (維度分離邏輯)
        v_tensor = pt.zeros((4, 2, 2))
        # Condition 0: VH (L=V, R=H)
        v_tensor = pt.set_subtensor(v_tensor[0,0,:], [v_ms_L, v_m_L])
        v_tensor = pt.set_subtensor(v_tensor[0,1,:], [v_m_R, v_ms_R])
        # Condition 1: HH (L=H, R=H)
        v_tensor = pt.set_subtensor(v_tensor[1,0,:], [v_m_L, v_ms_L])
        v_tensor = pt.set_subtensor(v_tensor[1,1,:], [v_m_R, v_ms_R])
        # Condition 2: HV (L=H, R=V)
        v_tensor = pt.set_subtensor(v_tensor[2,0,:], [v_m_L, v_ms_L])
        v_tensor = pt.set_subtensor(v_tensor[2,1,:], [v_ms_R, v_m_R])
        # Condition 3: VV (L=V, R=V)
        v_tensor = pt.set_subtensor(v_tensor[3,0,:], [v_ms_L, v_m_L])
        v_tensor = pt.set_subtensor(v_tensor[3,1,:], [v_ms_R, v_m_R])

        log_lik_vec = GRT_LBA_2D_PointwiseOp()(
            pt.as_tensor_variable(observed_data[:, 1].astype('int32')),
            pt.as_tensor_variable(observed_data[:, 0]),
            pt.as_tensor_variable(observed_data[:, 2].astype('int32')),
            v_tensor,
            pt.as_tensor_variable(A, dtype='float64'),
            b, t0,
            pt.as_tensor_variable(s, dtype='float64')
        )

        pm.Deterministic("log_likelihood", log_lik_vec)
        pm.Potential("obs", pt.sum(log_lik_vec))

        # 動態調整初始值
        init_vals = {
            "vt_L": 3.0, "vb_L": 0.7, "vt_R": 3.0, "vb_R": 0.7,
            "t0": t0_limit * 0.8, "b": 1.0
        }
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                         step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)
    return trace


def build_model_noPS(observed_data, chains=24, draws=20000, tune=40000):
    """
    noPS Model with 3 groups (6 drift parameters + b, t0)
    - VV: v_match_VV, v_mismatch_VV
    - HH: v_match_HH, v_mismatch_HH
    - Mixed (VH & HV): v_match_Mixed, v_mismatch_Mixed
    - b: threshold (estimated)
    - t0: non-decision time (estimated, dynamic upper bound)

    修正版：動態計算 t0 上限，確保與 PS 模型完全對等
    """
    # 根據該位受試者的最小 RT 動態計算 t0 上限
    min_rt = np.min(observed_data[:, 0])
    t0_limit = min(0.25, min_rt * 0.9)

    with pm.Model() as model:
        # 固定 A 和 s 作為比例尺
        A, s = 0.5, 1.0

        # --- 共享參數 (確保與 PS 完全對等) ---
        # 門檻 b: 非常窄的 informed prior (0.8~1.2)
        b = pm.TruncatedNormal("b", mu=1.0, sigma=0.1, lower=0.8, upper=1.2)
        t0 = pm.Uniform("t0", lower=0.01, upper=t0_limit)

        # --- noPS 模型特定參數：刺激交互 (3 groups) ---
        vt = pm.TruncatedNormal("vt", mu=3.0, sigma=1.0, lower=0.1, upper=8.0, shape=3)
        vb = pm.Beta("vb", 2, 2, shape=3)

        v_m = pm.Deterministic("v_match", vt * vb)
        v_ms = pm.Deterministic("v_mismatch", vt * (1 - vb))

        # Named parameters for each group
        v_m_VV = pm.Deterministic("v_match_VV", v_m[0])
        v_ms_VV = pm.Deterministic("v_mismatch_VV", v_ms[0])
        v_m_HH = pm.Deterministic("v_match_HH", v_m[1])
        v_ms_HH = pm.Deterministic("v_mismatch_HH", v_ms[1])
        v_m_Mixed = pm.Deterministic("v_match_Mixed", v_m[2])
        v_ms_Mixed = pm.Deterministic("v_mismatch_Mixed", v_ms[2])

        # Build v_tensor
        v_tensor = pt.zeros((4, 2, 2))
        # Cond 0: VH -> Mixed [2]
        v_tensor = pt.set_subtensor(v_tensor[0,0,:], [v_ms[2], v_m[2]])
        v_tensor = pt.set_subtensor(v_tensor[0,1,:], [v_m[2], v_ms[2]])
        # Cond 1: HH -> HH [1]
        v_tensor = pt.set_subtensor(v_tensor[1,0,:], [v_m[1], v_ms[1]])
        v_tensor = pt.set_subtensor(v_tensor[1,1,:], [v_m[1], v_ms[1]])
        # Cond 2: HV -> Mixed [2]
        v_tensor = pt.set_subtensor(v_tensor[2,0,:], [v_m[2], v_ms[2]])
        v_tensor = pt.set_subtensor(v_tensor[2,1,:], [v_ms[2], v_m[2]])
        # Cond 3: VV -> VV [0]
        v_tensor = pt.set_subtensor(v_tensor[3,0,:], [v_ms[0], v_m[0]])
        v_tensor = pt.set_subtensor(v_tensor[3,1,:], [v_ms[0], v_m[0]])

        log_lik_vec = GRT_LBA_2D_PointwiseOp()(
            pt.as_tensor_variable(observed_data[:, 1].astype('int32')),
            pt.as_tensor_variable(observed_data[:, 0]),
            pt.as_tensor_variable(observed_data[:, 2].astype('int32')),
            v_tensor,
            pt.as_tensor_variable(A, dtype='float64'),
            b,
            t0,
            pt.as_tensor_variable(s, dtype='float64')
        )

        pm.Deterministic("log_likelihood", log_lik_vec)
        pm.Potential("obs", pt.sum(log_lik_vec))

        # 動態調整初始值
        init_vals = {
            "vt": np.array([3.0, 3.0, 3.0]),  # VV, HH, Mixed
            "vb": np.array([0.7, 0.7, 0.7]),
            "t0": t0_limit * 0.8, "b": 1.0
        }
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                         step=pm.DEMetropolisZ(), random_seed=42, initvals=init_vals)
    return trace


def get_valid_participants(csv_path="GRT_LBA.csv", min_accuracy=0.70, rt_min=0.2, rt_max=2.0):
    """
    找出所有正確率 >= min_accuracy 的受試者

    Returns:
    --------
    valid_participants : list of dict
        每個 dict 包含 participant_id, accuracy, n_trials
    """
    df = pd.read_csv(csv_path)

    # 先篩選 RT 範圍
    df_filtered = df[(df['RT'] >= rt_min) & (df['RT'] <= rt_max)].copy()

    participants = df_filtered['participant'].unique()
    valid_participants = []

    for pid in participants:
        p_data = df_filtered[df_filtered['participant'] == pid]
        correct = (p_data['Response'] == p_data['stim_condition']).sum()
        acc = correct / len(p_data)

        if acc >= min_accuracy:
            valid_participants.append({
                'participant_id': pid,
                'accuracy': acc,
                'n_trials': len(p_data)
            })

    # 按 participant_id 排序
    valid_participants.sort(key=lambda x: x['participant_id'])

    return valid_participants


def run_and_save_models(participant_id, csv_path="GRT_LBA.csv",
                        chains=24, draws=20000, tune=40000):
    """
    Run both PS and noPS models on real data and save as v3.nc

    增加 draws 和 tune 以改善 ESS

    Returns trace_ps, trace_nops, data
    """
    print("="*70, flush=True)
    print(f"Running Models for Participant {participant_id}", flush=True)
    print("="*70, flush=True)

    # Load data
    data, df = load_grt_lba_data(csv_path, participant_id=participant_id)

    # Run PS model
    print("\n" + "="*70, flush=True)
    print(f"Running PS Model (chains={chains}, draws={draws}, tune={tune})...", flush=True)
    print("="*70, flush=True)
    trace_ps = build_model_PS(data, chains=chains, draws=draws, tune=tune)

    # Save PS trace
    ps_file = f"ps_real_data_p{participant_id}_v3.nc"
    az.to_netcdf(trace_ps, ps_file)
    print(f"✓ Saved: {ps_file}", flush=True)

    # Print PS summary
    ps_params = ["v_match_L", "v_mismatch_L", "v_match_R", "v_mismatch_R", "b", "t0"]
    summary_ps = az.summary(trace_ps, var_names=ps_params)
    print("\nPS Model Parameter Estimates:", flush=True)
    print(summary_ps[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']], flush=True)

    # Run noPS model
    print("\n" + "="*70, flush=True)
    print(f"Running noPS Model (chains={chains}, draws={draws}, tune={tune})...", flush=True)
    print("="*70, flush=True)
    trace_nops = build_model_noPS(data, chains=chains, draws=draws, tune=tune)

    # Save noPS trace
    nops_file = f"nops_real_data_p{participant_id}_v3.nc"
    az.to_netcdf(trace_nops, nops_file)
    print(f"✓ Saved: {nops_file}", flush=True)

    # Print noPS summary
    nops_params = ["v_match_VV", "v_mismatch_VV", "v_match_HH", "v_mismatch_HH",
                   "v_match_Mixed", "v_mismatch_Mixed", "b", "t0"]
    summary_nops = az.summary(trace_nops, var_names=nops_params)
    print("\nnoPS Model Parameter Estimates:", flush=True)
    print(summary_nops[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']], flush=True)

    return trace_ps, trace_nops, data


def run_all_valid_participants(csv_path="GRT_LBA.csv", min_accuracy=0.70,
                                chains=24, draws=20000, tune=40000,
                                skip_existing=True):
    """
    依序執行所有正確率 >= 70% 的受試者

    Parameters:
    -----------
    csv_path : str
        CSV 檔案路徑
    min_accuracy : float
        最低正確率門檻 (預設 0.70)
    chains, draws, tune : int
        MCMC 參數
    skip_existing : bool
        是否跳過已有 v3.nc 檔案的受試者
    """
    print("="*70, flush=True)
    print("Batch Processing: All Valid Participants", flush=True)
    print("="*70, flush=True)

    # 找出所有符合條件的受試者
    valid_participants = get_valid_participants(csv_path, min_accuracy)

    print(f"\nFound {len(valid_participants)} participants with accuracy >= {min_accuracy*100:.0f}%:\n")
    for p in valid_participants:
        print(f"  P{p['participant_id']:3d}: {p['accuracy']*100:5.1f}% ({p['n_trials']} trials)")

    # 檢查哪些已經完成
    completed = []
    to_run = []

    for p in valid_participants:
        pid = p['participant_id']
        ps_file = f"ps_real_data_p{pid}_v3.nc"
        nops_file = f"nops_real_data_p{pid}_v3.nc"

        if skip_existing and Path(ps_file).exists() and Path(nops_file).exists():
            completed.append(pid)
        else:
            to_run.append(p)

    if completed:
        print(f"\n已完成 (skip): {completed}")

    if not to_run:
        print("\n所有受試者都已完成！")
        return

    print(f"\n待執行: {[p['participant_id'] for p in to_run]}")
    print(f"總共 {len(to_run)} 位受試者")
    print(f"\nMCMC 設定: chains={chains}, draws={draws}, tune={tune}")
    print("\n" + "="*70, flush=True)

    # 執行結果追蹤
    results = []
    failed = []

    for i, p in enumerate(to_run):
        pid = p['participant_id']
        print(f"\n{'#'*70}", flush=True)
        print(f"# Processing {i+1}/{len(to_run)}: Participant {pid}", flush=True)
        print(f"# Accuracy: {p['accuracy']*100:.1f}%, Trials: {p['n_trials']}", flush=True)
        print(f"{'#'*70}", flush=True)

        try:
            trace_ps, trace_nops, data = run_and_save_models(
                pid, csv_path=csv_path,
                chains=chains, draws=draws, tune=tune
            )

            # 檢查收斂性
            ps_params = ["v_match_L", "v_mismatch_L", "v_match_R", "v_mismatch_R", "b", "t0"]
            nops_params = ["v_match_VV", "v_mismatch_VV", "v_match_HH", "v_mismatch_HH",
                          "v_match_Mixed", "v_mismatch_Mixed", "b", "t0"]

            summary_ps = az.summary(trace_ps, var_names=ps_params)
            summary_nops = az.summary(trace_nops, var_names=nops_params)

            result = {
                'participant_id': pid,
                'accuracy': p['accuracy'],
                'ps_max_rhat': summary_ps['r_hat'].max(),
                'ps_min_ess': summary_ps['ess_bulk'].min(),
                'nops_max_rhat': summary_nops['r_hat'].max(),
                'nops_min_ess': summary_nops['ess_bulk'].min(),
                'status': 'completed'
            }

            # 檢查是否收斂良好
            if result['ps_max_rhat'] > 1.05 or result['ps_min_ess'] < 400:
                result['ps_converged'] = False
            else:
                result['ps_converged'] = True

            if result['nops_max_rhat'] > 1.05 or result['nops_min_ess'] < 400:
                result['nops_converged'] = False
            else:
                result['nops_converged'] = True

            results.append(result)

            print(f"\n✓ P{pid} 完成", flush=True)
            print(f"  PS:   r_hat={result['ps_max_rhat']:.3f}, ESS={result['ps_min_ess']:.0f} {'✓' if result['ps_converged'] else '⚠'}")
            print(f"  noPS: r_hat={result['nops_max_rhat']:.3f}, ESS={result['nops_min_ess']:.0f} {'✓' if result['nops_converged'] else '⚠'}")

        except Exception as e:
            print(f"\n✗ P{pid} 失敗: {e}", flush=True)
            failed.append({'participant_id': pid, 'error': str(e)})

    # 最終報告
    print("\n" + "="*70, flush=True)
    print("BATCH PROCESSING COMPLETE", flush=True)
    print("="*70, flush=True)

    print(f"\n完成: {len(results)} 位受試者")
    if failed:
        print(f"失敗: {len(failed)} 位受試者 - {[f['participant_id'] for f in failed]}")

    # 收斂統計
    ps_converged = sum(1 for r in results if r['ps_converged'])
    nops_converged = sum(1 for r in results if r['nops_converged'])

    print(f"\n收斂統計:")
    print(f"  PS model:   {ps_converged}/{len(results)} 收斂良好")
    print(f"  noPS model: {nops_converged}/{len(results)} 收斂良好")

    # 儲存結果摘要
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("batch_results_summary.csv", index=False)
        print(f"\n✓ 結果摘要已儲存: batch_results_summary.csv")

    # 列出需要重新執行的受試者
    need_rerun = [r['participant_id'] for r in results
                  if not r['ps_converged'] or not r['nops_converged']]
    if need_rerun:
        print(f"\n⚠ 收斂不良，可能需要重新執行: {need_rerun}")

    return results, failed


# 設定中文字型（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 載入模型結果
# ============================================================================

def load_model_traces(ps_file, nops_file):
    """
    載入 PS 和 noPS 模型的 trace 檔案

    Parameters:
    -----------
    ps_file : str
        PS model NetCDF 檔案路徑
    nops_file : str
        noPS model NetCDF 檔案路徑

    Returns:
    --------
    trace_ps, trace_nops : InferenceData objects
    """
    print("="*70)
    print("Loading Model Traces")
    print("="*70)

    print(f"\nLoading PS model from: {ps_file}")
    trace_ps = az.from_netcdf(ps_file)
    print(f"✓ PS model loaded")
    print(f"  Chains: {trace_ps.posterior.dims['chain']}")
    print(f"  Draws: {trace_ps.posterior.dims['draw']}")

    print(f"\nLoading noPS model from: {nops_file}")
    trace_nops = az.from_netcdf(nops_file)
    print(f"✓ noPS model loaded")
    print(f"  Chains: {trace_nops.posterior.dims['chain']}")
    print(f"  Draws: {trace_nops.posterior.dims['draw']}")

    return trace_ps, trace_nops

# ============================================================================
# 2. 模型比較：WAIC/LOO
# ============================================================================

def has_log_likelihood(trace):
    """檢查 trace 是否有 log_likelihood (在 log_likelihood group 或 posterior 中)"""
    # 檢查標準位置
    if hasattr(trace, 'log_likelihood') and trace.log_likelihood is not None:
        return True
    # 檢查是否在 posterior 中
    if hasattr(trace, 'posterior') and 'log_likelihood' in trace.posterior.data_vars:
        return True
    return False


def prepare_trace_for_waic(trace):
    """
    準備 trace 以供 WAIC/LOO 計算
    如果 log_likelihood 在 posterior 中，將其移到正確的 group
    """
    import xarray as xr

    # 如果已經有 log_likelihood group，直接返回
    if hasattr(trace, 'log_likelihood') and trace.log_likelihood is not None:
        return trace

    # 如果 log_likelihood 在 posterior 中，需要重建 InferenceData
    if hasattr(trace, 'posterior') and 'log_likelihood' in trace.posterior.data_vars:
        print("  (Moving log_likelihood from posterior to log_likelihood group)")

        # 取出 log_likelihood
        ll_data = trace.posterior['log_likelihood']

        # 創建新的 posterior（不含 log_likelihood）
        new_posterior = trace.posterior.drop_vars('log_likelihood')

        # 創建 log_likelihood dataset
        log_likelihood_ds = xr.Dataset({'log_likelihood': ll_data})

        # 重建 InferenceData
        new_trace = az.InferenceData(
            posterior=new_posterior,
            log_likelihood=log_likelihood_ds
        )

        # 複製其他 groups
        if hasattr(trace, 'sample_stats') and trace.sample_stats is not None:
            new_trace.add_groups({'sample_stats': trace.sample_stats})

        return new_trace

    return trace


def compare_models(trace_ps, trace_nops):
    """
    使用 WAIC 和 LOO 比較兩個模型

    Returns:
    --------
    comparison_waic, comparison_loo : DataFrame or None
        模型比較結果，如果沒有 log_likelihood 則返回 None
    """
    print("\n" + "="*70)
    print("Model Comparison: WAIC and LOO")
    print("="*70)

    # 檢查是否有 log_likelihood
    if not has_log_likelihood(trace_ps) or not has_log_likelihood(trace_nops):
        print("\n⚠ log_likelihood not found in trace files")
        print("  Skipping WAIC/LOO comparison")
        print("  (To enable WAIC/LOO, save log_likelihood when running the model)")
        return None, None

    # 準備 trace（如果 log_likelihood 在 posterior 中，移到正確位置）
    print("\nPreparing traces for WAIC/LOO calculation...")
    trace_ps_prepared = prepare_trace_for_waic(trace_ps)
    trace_nops_prepared = prepare_trace_for_waic(trace_nops)

    # 計算 WAIC
    print("\nCalculating WAIC...")
    waic_ps = az.waic(trace_ps_prepared, var_name="log_likelihood")
    waic_nops = az.waic(trace_nops_prepared, var_name="log_likelihood")

    print(f"  PS model WAIC: {waic_ps.elpd_waic:.2f} (SE: {waic_ps.se:.2f})")
    print(f"  noPS model WAIC: {waic_nops.elpd_waic:.2f} (SE: {waic_nops.se:.2f})")

    # 計算 LOO
    print("\nCalculating LOO-CV...")
    loo_ps = az.loo(trace_ps_prepared, var_name="log_likelihood")
    loo_nops = az.loo(trace_nops_prepared, var_name="log_likelihood")

    print(f"  PS model LOO: {loo_ps.elpd_loo:.2f} (SE: {loo_ps.se:.2f})")
    print(f"  noPS model LOO: {loo_nops.elpd_loo:.2f} (SE: {loo_nops.se:.2f})")

    # 模型比較
    print("\n" + "-"*70)
    print("Model Comparison Table (WAIC):")
    print("-"*70)
    compare_dict = {"PS": trace_ps_prepared, "noPS": trace_nops_prepared}
    comparison_waic = az.compare(compare_dict, ic="waic", var_name="log_likelihood")
    print(comparison_waic)

    print("\n" + "-"*70)
    print("Model Comparison Table (LOO):")
    print("-"*70)
    comparison_loo = az.compare(compare_dict, ic="loo", var_name="log_likelihood")
    print(comparison_loo)

    # 解釋結果
    print("\n" + "="*70)
    print("Interpretation:")
    print("="*70)

    best_model_waic = comparison_waic.index[0]
    best_model_loo = comparison_loo.index[0]

    print(f"\nBest model by WAIC: {best_model_waic}")
    print(f"Best model by LOO: {best_model_loo}")

    if best_model_waic == best_model_loo:
        print(f"\n✓ Both criteria agree: {best_model_waic} model is preferred")
    else:
        print(f"\n⚠ Criteria disagree - further investigation recommended")

    # WAIC 差異
    if len(comparison_waic) > 1:
        waic_diff = comparison_waic.loc[comparison_waic.index[1], 'elpd_diff']
        waic_se = comparison_waic.loc[comparison_waic.index[1], 'dse']
        print(f"\nWAIC difference: {abs(waic_diff):.2f} ± {waic_se:.2f}")
        if abs(waic_diff) > 2 * waic_se:
            print("  → Substantial evidence for model preference")
        else:
            print("  → Models are comparable (difference < 2 SE)")

    return comparison_waic, comparison_loo

# ============================================================================
# 3. 參數估計比較
# ============================================================================

def compare_parameters(trace_ps, trace_nops):
    """
    比較兩個模型的參數估計
    """
    print("\n" + "="*70)
    print("Parameter Estimates Comparison")
    print("="*70)

    # PS model 參數 (Left/Right separation: 4 parameters)
    print("\n" + "-"*70)
    print("PS Model Parameters (Left/Right separation):")
    print("-"*70)
    ps_params = ["v_match_L", "v_mismatch_L", "v_match_R", "v_mismatch_R", "b", "t0"]
    summary_ps = az.summary(trace_ps, var_names=ps_params)
    print(summary_ps[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])

    # noPS model 參數 (3 groups: VV, HH, Mixed)
    print("\n" + "-"*70)
    print("noPS Model Parameters (Group-specific: VV, HH, Mixed):")
    print("-"*70)
    nops_params = ["v_match_VV", "v_mismatch_VV",
                   "v_match_HH", "v_mismatch_HH",
                   "v_match_Mixed", "v_mismatch_Mixed", "b", "t0"]
    summary_nops = az.summary(trace_nops, var_names=nops_params)
    print(summary_nops[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])

    # 檢查收斂性
    print("\n" + "="*70)
    print("Convergence Diagnostics:")
    print("="*70)

    # PS model
    rhat_ps = summary_ps['r_hat'].max()
    ess_ps = summary_ps['ess_bulk'].min()
    print(f"\nPS Model:")
    print(f"  Max R-hat: {rhat_ps:.4f} {'✓' if rhat_ps < 1.01 else '⚠'}")
    print(f"  Min ESS: {ess_ps:.0f} {'✓' if ess_ps > 400 else '⚠'}")

    # noPS model
    rhat_nops = summary_nops['r_hat'].max()
    ess_nops = summary_nops['ess_bulk'].min()
    print(f"\nnoPS Model:")
    print(f"  Max R-hat: {rhat_nops:.4f} {'✓' if rhat_nops < 1.01 else '⚠'}")
    print(f"  Min ESS: {ess_nops:.0f} {'✓' if ess_nops > 400 else '⚠'}")

    return summary_ps, summary_nops

# ============================================================================
# 4. 視覺化比較
# ============================================================================

def plot_posterior_comparison(trace_ps, trace_nops, output_dir="./figures"):
    """
    視覺化後驗分布比較
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Generating Posterior Distribution Plots")
    print("="*70)

    # 1. PS model 參數分布 (Left/Right separation: 4 parameters)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ps_params = ["v_match_L", "v_mismatch_L", "v_match_R", "v_mismatch_R", "b", "t0"]
    ps_titles = ["PS: v_match_L (Left)", "PS: v_mismatch_L (Left)",
                 "PS: v_match_R (Right)", "PS: v_mismatch_R (Right)"]

    for i, (param, title) in enumerate(zip(ps_params, ps_titles)):
        row, col = i // 2, i % 2
        az.plot_posterior(trace_ps, var_names=[param], ax=axes[row, col],
                          hdi_prob=0.95, point_estimate='mean')
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')

    plt.suptitle("PS Model: Left/Right Drift Rate Parameters",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = f"{output_dir}/ps_model_posteriors.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

    # 2. noPS model 參數分布（3 groups: VV, HH, Mixed）
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    groups = ["VV", "HH", "Mixed"]
    for i, grp in enumerate(groups):
        az.plot_posterior(trace_nops, var_names=[f"v_match_{grp}"],
                         ax=axes[0, i], hdi_prob=0.95, point_estimate='mean')
        axes[0, i].set_title(f"{grp}: v_match", fontsize=12, fontweight='bold')

        az.plot_posterior(trace_nops, var_names=[f"v_mismatch_{grp}"],
                         ax=axes[1, i], hdi_prob=0.95, point_estimate='mean')
        axes[1, i].set_title(f"{grp}: v_mismatch", fontsize=12, fontweight='bold')

    plt.suptitle("noPS Model: Group-Specific Parameters (VV, HH, Mixed)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = f"{output_dir}/nops_model_posteriors.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

    # 3. Trace plots 檢查收斂
    print("\nGenerating trace plots...")

    # PS model traces
    az.plot_trace(trace_ps, var_names=["v_match_L", "v_mismatch_L", "v_match_R", "v_mismatch_R"],
                  compact=True, figsize=(12, 8))
    plt.suptitle("PS Model: Trace Plots (Left/Right Parameters)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = f"{output_dir}/ps_model_traces.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

    # 4. 參數比較圖（forest plot）
    print("\nGenerating forest plot...")

    # 提取 noPS 的 group 特定估計
    nops_params = ["v_match_VV", "v_match_HH", "v_match_Mixed",
                   "v_mismatch_VV", "v_mismatch_HH", "v_mismatch_Mixed"]
    summary_nops = az.summary(trace_nops, var_names=nops_params)

    ps_params = ["v_match_L", "v_mismatch_L", "v_match_R", "v_mismatch_R", "b", "t0"]
    summary_ps = az.summary(trace_ps, var_names=ps_params)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # v_match 比較
    groups = ["VV", "HH", "Mixed"]
    v_match_means = [summary_nops.loc[f"v_match_{g}", "mean"] for g in groups]
    v_match_sds = [summary_nops.loc[f"v_match_{g}", "sd"] for g in groups]

    y_pos = np.arange(len(groups))
    axes[0].errorbar(v_match_means, y_pos, xerr=v_match_sds,
                     fmt='o', capsize=5, label='noPS (group-specific)', markersize=8, color='blue')
    # PS model has L/R - draw both as horizontal lines
    axes[0].axvline(summary_ps.loc["v_match_L", "mean"],
                    color='red', linestyle='--', linewidth=2, label='PS v_match_L')
    axes[0].axvline(summary_ps.loc["v_match_R", "mean"],
                    color='orange', linestyle='--', linewidth=2, label='PS v_match_R')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(groups)
    axes[0].set_xlabel('v_match', fontsize=12, fontweight='bold')
    axes[0].set_title('v_match: PS (L/R) vs noPS (groups)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # v_mismatch 比較
    v_mismatch_means = [summary_nops.loc[f"v_mismatch_{g}", "mean"] for g in groups]
    v_mismatch_sds = [summary_nops.loc[f"v_mismatch_{g}", "sd"] for g in groups]

    axes[1].errorbar(v_mismatch_means, y_pos, xerr=v_mismatch_sds,
                     fmt='o', capsize=5, label='noPS (group-specific)', markersize=8, color='blue')
    axes[1].axvline(summary_ps.loc["v_mismatch_L", "mean"],
                    color='red', linestyle='--', linewidth=2, label='PS v_mismatch_L')
    axes[1].axvline(summary_ps.loc["v_mismatch_R", "mean"],
                    color='orange', linestyle='--', linewidth=2, label='PS v_mismatch_R')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(groups)
    axes[1].set_xlabel('v_mismatch', fontsize=12, fontweight='bold')
    axes[1].set_title('v_mismatch: PS (L/R) vs noPS (groups)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = f"{output_dir}/parameter_comparison_forest.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

# ============================================================================
# 5. 生成分析報告
# ============================================================================

def generate_report(comparison_waic, comparison_loo, summary_ps, summary_nops,
                    output_file="model_comparison_report.txt"):
    """
    生成文字格式的分析報告
    """
    print(f"\n" + "="*70)
    print(f"Generating Analysis Report")
    print(f"="*70)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PS vs noPS Model Comparison Report\n")
        f.write("="*70 + "\n\n")

        # 模型比較結果 (if available)
        if comparison_waic is not None:
            f.write("1. MODEL COMPARISON (WAIC)\n")
            f.write("-"*70 + "\n")
            f.write(comparison_waic.to_string())
            f.write("\n\n")

            f.write("2. MODEL COMPARISON (LOO)\n")
            f.write("-"*70 + "\n")
            f.write(comparison_loo.to_string())
            f.write("\n\n")
        else:
            f.write("1. MODEL COMPARISON (WAIC/LOO)\n")
            f.write("-"*70 + "\n")
            f.write("Not available (log_likelihood not found in trace files)\n\n")

        # PS model 參數
        f.write("3. PS MODEL PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(summary_ps.to_string())
        f.write("\n\n")

        # noPS model 參數
        f.write("4. noPS MODEL PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(summary_nops.to_string())
        f.write("\n\n")

        # 解釋
        f.write("5. INTERPRETATION\n")
        f.write("-"*70 + "\n")

        if comparison_waic is not None:
            best_model = comparison_waic.index[0]
            f.write(f"Best fitting model: {best_model}\n\n")

            if best_model == "PS":
                f.write("The Perceptual Separability (PS) model provides a better fit.\n")
                f.write("This model uses 4 parameters (v_match_L, v_mismatch_L, v_match_R, v_mismatch_R)\n")
                f.write("to separate Left and Right dimensions, suggesting that drift rates\n")
                f.write("depend primarily on which dimension (L/R) is being processed.\n")
            else:
                f.write("The no-PS model provides a better fit.\n")
                f.write("This model uses 6 parameters (VV, HH, Mixed groups) to capture\n")
                f.write("stimulus-dependent interactions between dimensions.\n")
                f.write("This suggests that drift rates depend on the specific stimulus\n")
                f.write("combination rather than just the processing dimension.\n")
        else:
            f.write("Cannot determine best model without WAIC/LOO comparison.\n")
            f.write("Please check parameter estimates and posterior predictive checks.\n")

        f.write("\n")

        # 收斂診斷
        f.write("6. CONVERGENCE DIAGNOSTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"PS Model:\n")
        f.write(f"  Max R-hat: {summary_ps['r_hat'].max():.4f}\n")
        f.write(f"  Min ESS: {summary_ps['ess_bulk'].min():.0f}\n\n")
        f.write(f"noPS Model:\n")
        f.write(f"  Max R-hat: {summary_nops['r_hat'].max():.4f}\n")
        f.write(f"  Min ESS: {summary_nops['ess_bulk'].min():.0f}\n")

    print(f"✓ Report saved to: {output_file}")

# ============================================================================
# 6. Main Execution
# ============================================================================

class TeeLogger:
    """同時輸出到 console 和 log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main(ps_file, nops_file, output_dir="./analysis_results"):
    """
    主要分析流程

    Parameters:
    -----------
    ps_file : str
        PS model NetCDF 檔案路徑
    nops_file : str
        noPS model NetCDF 檔案路徑
    output_dir : str
        輸出目錄
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 設定 log file
    log_file = os.path.join(output_dir, "analysis.log")
    logger = TeeLogger(log_file)
    sys.stdout = logger

    print("\n" + "="*70)
    print("PS vs noPS Model Comparison Analysis")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Log file: {log_file}")

    # 1. 載入模型
    trace_ps, trace_nops = load_model_traces(ps_file, nops_file)

    # 2. 模型比較
    comparison_waic, comparison_loo = compare_models(trace_ps, trace_nops)

    # 3. 參數比較
    summary_ps, summary_nops = compare_parameters(trace_ps, trace_nops)

    # 4. 視覺化
    plot_posterior_comparison(trace_ps, trace_nops,
                              output_dir=os.path.join(output_dir, "figures"))

    # 5. 生成報告
    report_file = os.path.join(output_dir, "model_comparison_report.txt")
    generate_report(comparison_waic, comparison_loo, summary_ps, summary_nops,
                    output_file=report_file)

    # 6. 儲存比較結果為 CSV (if available)
    if comparison_waic is not None:
        csv_file = os.path.join(output_dir, "model_comparison_waic.csv")
        comparison_waic.to_csv(csv_file)
        print(f"\n✓ WAIC comparison saved to: {csv_file}")

    if comparison_loo is not None:
        csv_file = os.path.join(output_dir, "model_comparison_loo.csv")
        comparison_loo.to_csv(csv_file)
        print(f"✓ LOO comparison saved to: {csv_file}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - analysis.log")
    print(f"  - figures/ (posterior plots, traces, forest plots)")
    print(f"  - model_comparison_report.txt")
    if comparison_waic is not None:
        print(f"  - model_comparison_waic.csv")
        print(f"  - model_comparison_loo.csv")

    # 關閉 logger
    sys.stdout = logger.terminal
    logger.close()
    print(f"\n✓ Log saved to: {log_file}")


if __name__ == "__main__":
    import glob

    # ========== USAGE ==========
    # python final_compare.py run <participant_id>    # Run models and compare (saves v3.nc)
    # python final_compare.py compare <participant_id> # Compare existing v3.nc files
    # python final_compare.py <participant_id>         # Shorthand for 'run'
    # ===========================

    def find_completed_participants(version="v3"):
        """Find all participants with both PS and noPS traces"""
        ps_files = glob.glob(f'ps_real_data_p*_{version}.nc')
        completed = []
        for ps_file in ps_files:
            # Extract participant ID from filename
            import re
            match = re.search(r'ps_real_data_p(\d+)_', ps_file)
            if match:
                pid = int(match.group(1))
                nops_file = f'nops_real_data_p{pid}_{version}.nc'
                if Path(nops_file).exists():
                    completed.append(pid)
        return sorted(completed)

    def print_usage():
        print("PS vs noPS Model Comparison Analysis (v3 - New Structure)")
        print("=" * 70)
        print("\nUsage:")
        print("  python final_compare.py batch                    # Run ALL valid participants (acc>=70%)")
        print("  python final_compare.py run <participant_id>     # Run models + compare")
        print("  python final_compare.py compare <participant_id> # Compare existing results")
        print("  python final_compare.py list                     # List valid participants")
        print("  python final_compare.py <participant_id>         # Same as 'run'")
        print("\nExamples:")
        print("  python final_compare.py batch       # Run all participants with acc >= 70%")
        print("  python final_compare.py run 33      # Run both models for P33, save v3.nc")
        print("  python final_compare.py compare 33  # Compare existing P33 v3.nc files")
        print("  python final_compare.py list        # Show all valid participants")
        print("\nModel Structure (v3):")
        print("  PS Model:   6 params (v_match_L, v_mismatch_L, v_match_R, v_mismatch_R, b, t0)")
        print("  noPS Model: 8 params (VV, HH, Mixed groups, b, t0)")
        print("\nMCMC Settings (improved for convergence):")
        print("  chains=24, draws=20000, tune=40000")
        print("  b and t0 are now estimated (dynamic t0 upper bound)")

    if len(sys.argv) < 2:
        print_usage()
        completed = find_completed_participants("v3")
        if completed:
            print(f"\nExisting v3 results: {completed}")
        sys.exit(1)

    # Parse arguments
    if sys.argv[1].lower() == 'batch':
        mode = 'batch'
        pid = None
    elif sys.argv[1].lower() == 'list':
        mode = 'list'
        pid = None
    elif sys.argv[1].lower() == 'run':
        mode = 'run'
        if len(sys.argv) < 3:
            print("Error: Please specify participant ID")
            print("Usage: python final_compare.py run <participant_id>")
            sys.exit(1)
        pid = int(sys.argv[2])
    elif sys.argv[1].lower() == 'compare':
        mode = 'compare'
        if len(sys.argv) < 3:
            print("Error: Please specify participant ID")
            print("Usage: python final_compare.py compare <participant_id>")
            sys.exit(1)
        pid = int(sys.argv[2])
    else:
        # Default: assume it's a participant ID and run mode
        mode = 'run'
        try:
            pid = int(sys.argv[1])
        except ValueError:
            print(f"Error: Unknown command '{sys.argv[1]}'")
            print_usage()
            sys.exit(1)

    # Execute based on mode
    if mode == 'list':
        # List all valid participants
        print(f"\n{'='*70}")
        print("Valid Participants (accuracy >= 70%)")
        print(f"{'='*70}")

        if not Path("GRT_LBA.csv").exists():
            print("Error: GRT_LBA.csv not found!")
            sys.exit(1)

        valid_participants = get_valid_participants("GRT_LBA.csv", min_accuracy=0.70)

        print(f"\nFound {len(valid_participants)} valid participants:\n")
        print(f"{'ID':>5} {'Accuracy':>10} {'Trials':>8} {'PS v3':>10} {'noPS v3':>10}")
        print("-" * 50)

        for p in valid_participants:
            pid = p['participant_id']
            ps_exists = "✓" if Path(f"ps_real_data_p{pid}_v3.nc").exists() else "-"
            nops_exists = "✓" if Path(f"nops_real_data_p{pid}_v3.nc").exists() else "-"
            print(f"{pid:>5} {p['accuracy']*100:>9.1f}% {p['n_trials']:>8} {ps_exists:>10} {nops_exists:>10}")

        completed = [p['participant_id'] for p in valid_participants
                    if Path(f"ps_real_data_p{p['participant_id']}_v3.nc").exists()
                    and Path(f"nops_real_data_p{p['participant_id']}_v3.nc").exists()]
        pending = [p['participant_id'] for p in valid_participants
                  if p['participant_id'] not in completed]

        print(f"\n已完成: {len(completed)} 位")
        print(f"待執行: {len(pending)} 位")
        if pending:
            print(f"  {pending}")

    elif mode == 'batch':
        # Batch process all valid participants
        if not Path("GRT_LBA.csv").exists():
            print("Error: GRT_LBA.csv not found!")
            sys.exit(1)

        results, failed = run_all_valid_participants(
            csv_path="GRT_LBA.csv",
            min_accuracy=0.70,
            chains=24,
            draws=20000,
            tune=40000,
            skip_existing=True
        )

    elif mode == 'run':
        print(f"\n{'='*70}")
        print(f"MODE: Run Models for Participant {pid}")
        print(f"{'='*70}")

        # Check if GRT_LBA.csv exists
        if not Path("GRT_LBA.csv").exists():
            print("Error: GRT_LBA.csv not found!")
            sys.exit(1)

        # Run models and save as v3.nc
        trace_ps, trace_nops, data = run_and_save_models(pid)

        # Now compare the models
        print(f"\n{'='*70}")
        print(f"Comparing Models...")
        print(f"{'='*70}")

        OUTPUT_DIR = f"./comparison_analysis_p{pid}_v3"
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Set up logger
        log_file = os.path.join(OUTPUT_DIR, "analysis.log")
        logger = TeeLogger(log_file)
        sys.stdout = logger

        # Compare models
        comparison_waic, comparison_loo = compare_models(trace_ps, trace_nops)

        # Parameter comparison
        summary_ps, summary_nops = compare_parameters(trace_ps, trace_nops)

        # Visualization
        plot_posterior_comparison(trace_ps, trace_nops,
                                  output_dir=os.path.join(OUTPUT_DIR, "figures"))

        # Generate report
        report_file = os.path.join(OUTPUT_DIR, "model_comparison_report.txt")
        generate_report(comparison_waic, comparison_loo, summary_ps, summary_nops,
                        output_file=report_file)

        # Save comparison results
        if comparison_waic is not None:
            comparison_waic.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_waic.csv"))
        if comparison_loo is not None:
            comparison_loo.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_loo.csv"))

        # Close logger
        sys.stdout = logger.terminal
        logger.close()

        print(f"\n{'='*70}")
        print("Analysis Complete!")
        print(f"{'='*70}")
        print(f"\nOutput files:")
        print(f"  - ps_real_data_p{pid}_v3.nc")
        print(f"  - nops_real_data_p{pid}_v3.nc")
        print(f"  - {OUTPUT_DIR}/ (figures, reports, logs)")

    elif mode == 'compare':
        # Compare existing v3.nc files
        PS_FILE = f"ps_real_data_p{pid}_v3.nc"
        NOPS_FILE = f"nops_real_data_p{pid}_v3.nc"
        OUTPUT_DIR = f"./comparison_analysis_p{pid}_v3"

        if not Path(PS_FILE).exists():
            print(f"Error: {PS_FILE} not found!")
            print("Run 'python final_compare.py run {pid}' first to generate the files.")
            completed = find_completed_participants("v3")
            if completed:
                print(f"\nExisting v3 results: {completed}")
            sys.exit(1)

        if not Path(NOPS_FILE).exists():
            print(f"Error: {NOPS_FILE} not found!")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"MODE: Compare Existing Results for Participant {pid}")
        print(f"{'='*70}")
        print(f"  PS file:   {PS_FILE}")
        print(f"  noPS file: {NOPS_FILE}")
        print(f"  Output:    {OUTPUT_DIR}")

        main(PS_FILE, NOPS_FILE, OUTPUT_DIR)
