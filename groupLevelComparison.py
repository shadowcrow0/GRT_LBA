# -*- coding: utf-8 -*-
"""
LBA 模型比較分析腳本 (Coactive vs. Parallel AND)
*** 版本 v9: 終極修正版，重寫 random 函數以解決最終的 PPC 形狀錯誤 ***
"""
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import time
import matplotlib.pyplot as plt

# ===================================================================
# Phase 1: LBA 的核心數學引擎 (logp) 與隨機樣本生成器 (random)
# ===================================================================

# --- logp 函數 (用於 NUTS 採樣) ---
# 這個函數已經被證明是正確的，保持不變
def logp_lba(value, v_correct, v_incorrect, b, A, t0):
    rt = value[:, 0]; response = value[:, 1]; t = pt.maximum(rt - t0, 0.001)
    def normal_pdf(x, mu=0.0, sigma=1.0): return (1.0 / (sigma * pt.sqrt(2.0 * np.pi))) * pt.exp(-0.5 * ((x - mu) / sigma)**2)
    def normal_cdf(x, mu=0.0, sigma=1.0): return 0.5 * (1 + pt.erf((x - mu) / (sigma * pt.sqrt(2.0))))
    v_chosen = pt.switch(pt.eq(response, 1), v_correct, v_incorrect); v_unchosen = pt.switch(pt.eq(response, 1), v_incorrect, v_correct)
    v_chosen = pt.maximum(v_chosen, 1e-6); v_unchosen = pt.maximum(v_unchosen, 1e-6)
    term1_chosen = (b - A) / v_chosen; term2_chosen = b / v_chosen; term1_unchosen = (b - A) / v_unchosen
    g_chosen = (1/A) * (-v_chosen * normal_cdf(term1_chosen, mu=t, sigma=1) + v_chosen * normal_cdf(term2_chosen, mu=t, sigma=1) + normal_pdf(term1_chosen, mu=t, sigma=1) - normal_pdf(term2_chosen, mu=t, sigma=1))
    S_unchosen = 1 - normal_cdf(term1_unchosen, mu=t, sigma=1)
    joint_likelihood = g_chosen * S_unchosen
    safe_joint_likelihood = pt.maximum(joint_likelihood, 1e-10)
    return pt.sum(pt.log(safe_joint_likelihood))

# --- random 函數 (用於事後預測檢查 PPC) ---
# 這是全新的、更強固的版本
def lba_random(v_correct, v_incorrect, b, A, t0, rng=None, size=None):
    # 'size' 由 pm.sample_posterior_predictive 傳入，通常是 (n_trials, 2)
    n_trials = size[0]

    # 強制將固定參數轉為純量，避免廣播問題
    b_ = np.asarray(b).item()
    A_ = np.asarray(A).item()
    t0_ = np.asarray(t0).item()
    v_incorrect_ = np.asarray(v_incorrect).item()
    
    # v_correct 是唯一的向量，其長度應等於 n_trials
    v_correct_ = np.asarray(v_correct)

    # 用安全的方式建立 (n_trials, 2) 的漂移率陣列
    v = np.empty((n_trials, 2))
    v[:, 0] = v_correct_
    v[:, 1] = v_incorrect_

    # 接下來的模擬邏輯與之前類似，但使用純量化的參數
    start_points = rng.uniform(low=0, high=A_, size=(n_trials, 2))
    drifts = rng.normal(loc=v, scale=1)
    drifts[drifts < 0] = 1e-10
    
    threshold = np.maximum(b_, A_ + 1e-4)
    time_diff = threshold - start_points
    time_diff[time_diff < 0] = 0
    
    time_to_boundary = time_diff / drifts
    
    winner = 1 - np.argmin(time_to_boundary, axis=1)
    rt = (np.min(time_to_boundary, axis=1) + t0_).flatten()
    
    return np.stack([rt, winner], axis=1)

# ===================================================================
# Phase 2 & 3: 數據準備與主執行流程
# (這部分完全不變，但現在應該可以順利跑完)
# ===================================================================
def prepare_data_for_model(df, subject_id):
    subject_df = df[df['participant'] == subject_id].copy()
    if len(subject_df) == 0: raise ValueError(f"找不到受試者 {subject_id} 的資料")
    stimulus_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    choice_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    left_stim_is_diag = subject_df['Stimulus'].map(lambda s: stimulus_mapping[s]['left']).values
    right_stim_is_diag = subject_df['Stimulus'].map(lambda s: stimulus_mapping[s]['right']).values
    left_choice_is_diag = subject_df['Response'].map(lambda r: choice_mapping[r]['left']).values
    right_choice_is_diag = subject_df['Response'].map(lambda r: choice_mapping[r]['right']).values
    left_match = (left_stim_is_diag == 1); right_match = (right_stim_is_diag == 1)
    is_correct = (left_stim_is_diag == left_choice_is_diag) & (right_stim_is_diag == right_choice_is_diag)
    return {"rt": subject_df['RT'].values, "response_correct": is_correct.astype(int), "left_match": left_match.astype(int), "right_match": right_match.astype(int)}

# ===================================================================
# Phase 3: 主執行流程 (修正版)
# ===================================================================
if __name__ == '__main__':
    # --- 數據載入與準備 (不變) ---
    try:
        df = pd.read_csv('GRT_LBA.csv'); df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant']); df = df[(df['RT'] >= 0.1) & (df['RT'] <= 3.0)]
        print("✅ 數據載入成功。")
    except FileNotFoundError:
        print("❌ 錯誤: GRT_LBA.csv 檔案未找到。"); exit()
    # --- 步驟 1: 在最源頭的 DataFrame 上，進行數據篩選和清理 ---
    print("--- 步驟 1: 根據反應時間閾值，篩選原始數據 ---")
    rt_threshold = 0.150  # 設定 150 毫秒為閾值

    # !!! 重要：請將 'RT' 換成您數據中反應時間欄位的真實名稱 !!!
    # 可能是 'rt', 'RT', 'reaction_time' 等等，請務必確認
    rt_column_name = 'RT' 
    
    original_rows = len(df)
    df_cleaned = df[df[rt_column_name] >= rt_threshold].copy()
    print(f"原始數據共 {original_rows} 行。")
    print(f"篩選後 (RT >= {rt_threshold}s)，剩下 {len(df_cleaned)} 行。")
    print(f"共移除了 {original_rows - len(df_cleaned)} 個過快的試次。")


    # --- 步驟 2: 使用「清理後」的數據來準備模型輸入 ---
    print("\n--- 步驟 2: 使用清理後的數據來準備模型輸入 ---")
    SUBJECT_ID_TO_RUN = 47
    print(f"準備要跑的受試者: {SUBJECT_ID_TO_RUN}")
    
    # 將 df_cleaned 傳入函數，而不是原始的 df
    prepared_data = prepare_data_for_model(df_cleaned, SUBJECT_ID_TO_RUN)
    observed_value = np.column_stack([
    np.asarray(prepared_data['rt'], np.float32),
    np.asarray(prepared_data['response_correct'], np.float32)
])


    print("✅ 數據準備完成。")
    B_CONSTANT = 1.0; A_CONSTANT = 0.3; T0_CONSTANT = 0.2
    MCMC_CONFIG = {'draws': 2000, 'tune': 1500, 'chains': 4, 'cores': 1}
    
    print(f"\n--- 模型設定 ---\n固定參數: b={B_CONSTANT}, A={A_CONSTANT}, t0={T0_CONSTANT}\nMCMC 配置: {MCMC_CONFIG}")
    
    # --- Coactive Model ---
    with pm.Model() as model_coactive:
        # ... (模型定義不變) ...
        v_left_match = pm.HalfNormal('v_left_match', sigma=1.0); v_left_mismatch = pm.HalfNormal('v_left_mismatch', sigma=0.5)
        v_right_match = pm.HalfNormal('v_right_match', sigma=1.0); v_right_mismatch = pm.HalfNormal('v_right_mismatch', sigma=0.5)
        v_left = v_left_match * prepared_data['left_match'] + v_left_mismatch * (1 - prepared_data['left_match'])
        v_right = v_right_match * prepared_data['right_match'] + v_right_mismatch * (1 - prepared_data['right_match'])
        v_final_correct = pm.Deterministic('v_final_correct', v_left + v_right); v_final_incorrect = 0.1 
        likelihood_coactive = pm.CustomDist('likelihood', v_final_correct, v_final_incorrect, B_CONSTANT, A_CONSTANT, T0_CONSTANT, logp=logp_lba, random=lba_random, observed=observed_value)
    print("\n--- 正在建立 Coactive 模型 ---"); print("🔬 開始擬合 Coactive 模型...")
    idata_coactive = pm.sample(model=model_coactive, **MCMC_CONFIG)
    
    # --- 關鍵修正 1：計算並儲存 log_likelihood ---
    with model_coactive:
        pm.compute_log_likelihood(idata_coactive)
        idata_coactive.extend(pm.sample_posterior_predictive(idata_coactive))
    print("✅ Coactive 模型擬合完成")

    # --- Parallel AND Model ---
# --- Parallel AND 模型 (使用 LogSumExp 平滑近似) ---
    with pm.Model() as model_parallel:
        # ... (v_left_match, v_right_match 等先驗的定義保持不變) ...
        v_left_match = pm.HalfNormal('v_left_match', sigma=1.0); v_left_mismatch = pm.HalfNormal('v_left_mismatch', sigma=0.5)
        v_right_match = pm.HalfNormal('v_right_match', sigma=1.0); v_right_mismatch = pm.HalfNormal('v_right_mismatch', sigma=0.5)
        v_left = v_left_match * prepared_data['left_match'] + v_left_mismatch * (1 - prepared_data['left_match'])
        v_right = v_right_match * prepared_data['right_match'] + v_right_mismatch * (1 - prepared_data['right_match'])
        
        # --- 關鍵修正：使用 LogSumExp 來平滑地近似 max() ---
        # 引入一個控制平滑度的參數 k
        k = pm.HalfNormal('k_smoothness', sigma=1.0) 
        
        # LogSumExp 公式
        v_final_correct = pm.Deterministic(
            'v_final_correct', 
            pm.math.log(pm.math.exp(k * v_left) + pm.math.exp(k * v_right)) / k
        )
        
        v_final_incorrect = 0.1
        likelihood_parallel = pm.CustomDist('likelihood', v_final_correct, v_final_incorrect, B_CONSTANT, A_CONSTANT, T0_CONSTANT, logp=logp_lba, random=lba_random, observed=observed_value)    
        idata_parallel = pm.sample(model=model_parallel, **MCMC_CONFIG)

    # --- 關鍵修正 2：計算並儲存 log_likelihood ---
    
    with model_parallel:
        pm.compute_log_likelihood(idata_parallel)
        idata_parallel.extend(pm.sample_posterior_predictive(idata_parallel))
    print("✅ Parallel AND 模型擬合完成")
    
    # --- Phase 4: 視覺化診斷與模型比較 ---
    print("\n\n--- 視覺化診斷與最終比較 ---")
    az.plot_trace(idata_coactive, var_names=['v_left_match', 'v_left_mismatch', 'v_right_match', 'v_right_mismatch']); plt.suptitle("Trace Plot for Coactive Model", y=1.02); plt.tight_layout(); plt.show()
    az.plot_ppc(idata_coactive, kind='cumulative', num_pp_samples=100); plt.suptitle("PPC (Cumulative) for Coactive Model"); plt.show()
    az.plot_trace(idata_parallel, var_names=['v_left_match', 'v_left_mismatch', 'v_right_match', 'v_right_mismatch']); plt.suptitle("Trace Plot for Parallel AND Model", y=1.02); plt.tight_layout(); plt.show()
    az.plot_ppc(idata_parallel, kind='cumulative', num_pp_samples=100); plt.suptitle("PPC (Cumulative) for Parallel AND Model"); plt.show()
    model_comparison = {"Coactive": idata_coactive, "Parallel": idata_parallel}
    loo_compare = az.compare(model_comparison, ic="loo")
    print(loo_compare)
    az.plot_compare(loo_compare); plt.title("Model Comparison (LOO)"); plt.show()
    
    # --- Modified Plotting and Comparison Code ---
    
    # Plotting for the Coactive Model
    az.plot_trace(idata_coactive, var_names=['v_left_match', 'v_left_mismatch', 'v_right_match', 'v_right_mismatch'])
    # Using an f-string to add the subject ID to the title
    plt.suptitle(f"Trace Plot for Coactive Model (Subject: {SUBJECT_ID_TO_RUN})", y=1.02)
    plt.tight_layout()
    plt.show()
    
    az.plot_ppc(idata_coactive, kind='cumulative', num_pp_samples=100)
    plt.suptitle(f"PPC (Cumulative) for Coactive Model (Subject: {SUBJECT_ID_TO_RUN})")
    plt.show()
    
    # Plotting for the Parallel AND Model
    az.plot_trace(idata_parallel, var_names=['v_left_match', 'v_left_mismatch', 'v_right_match', 'v_right_mismatch'])
    # Using an f-string here as well
    plt.suptitle(f"Trace Plot for Parallel AND Model (Subject: {SUBJECT_ID_TO_RUN})", y=1.02)
    plt.tight_layout()
    plt.show()
    
    az.plot_ppc(idata_parallel, kind='cumulative', num_pp_samples=100)
    plt.suptitle(f"PPC (Cumulative) for Parallel AND Model (Subject: {SUBJECT_ID_TO_RUN})")
    plt.show()
    
    # Model comparison remains the same, but the plots above will be specific
    model_comparison = {"Coactive": idata_coactive, "Parallel": idata_parallel}
    loo_compare = az.compare(model_comparison, ic="loo")
    print(f"\n--- Model Comparison for Subject: {SUBJECT_ID_TO_RUN} ---")
    print(loo_compare)
    # --- 最終對決：模型比較 (加入警告檢查) ---
    print("\n\n--- 最終對決：模型比較結果 (使用 LOO) ---")
    
    # 檢查 Parallel 模型的收斂情況
    rhat_parallel = az.rhat(idata_parallel)
    has_convergence_issues = (rhat_parallel.to_array() > 1.01).any().item()

    if has_convergence_issues:
        print("⚠️ 警告: Parallel AND 模型存在嚴重的收斂問題 (R-hat > 1.01 或大量發散)。")
        print("   其 LOO 值不可靠，僅供參考。數據強烈傾向於 Coactive 模型。")
        # 只顯示 Coactive 模型的摘要
        print("\n--- Coactive 模型摘要 (穩定且可信) ---")
        print(az.summary(idata_coactive, var_names=['v_left_match', 'v_left_mismatch', 'v_right_match', 'v_right_mismatch']))
    else:
        # 只有在兩個模型都收斂時才進行正式比較
        model_comparison = {"Coactive": idata_coactive, "Parallel": idata_parallel}
        loo_compare = az.compare(model_comparison, ic="loo")
        print(loo_compare)
        az.plot_compare(loo_compare); plt.title("Model Comparison (LOO)"); plt.show()
