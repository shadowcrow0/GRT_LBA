
# -*- coding: utf-8 -*-
"""
檔案1: 數據處理與 Coactive 模型擬合
輸出: coactive_model_results.nc
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import time
import matplotlib.pyplot as plt

# ===================================================================
# LBA 核心函數
# ===================================================================
def logp_lba(value, v_correct, v_incorrect, b, A, t0):
    """LBA 的對數機率函數"""
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
    return pt.log(safe_joint_likelihood)

def lba_random(v_correct, v_incorrect, b, A, t0, rng=None, size=None):
    """LBA 隨機抽樣函數"""
    n_trials = size[0]
    v_c = np.asarray(v_correct)
    v_i = np.asarray(v_incorrect)
    b_ = np.asarray(b).item(); A_ = np.asarray(A).item(); t0_ = np.asarray(t0).item()
    v = np.empty((n_trials, 2)); v[:, 0] = v_c
    if v_i.ndim == 0:
        v[:, 1] = np.full(n_trials, v_i)
    else:
        v[:, 1] = v_i
    start_points = rng.uniform(low=0, high=A_, size=(n_trials, 2))
    drifts = rng.normal(loc=v, scale=1, size=(n_trials, 2))
    drifts[drifts < 0] = 1e-10
    threshold = np.maximum(b_, A_ + 1e-4)
    time_diff = threshold - start_points
    time_diff[time_diff < 0] = 0
    time_to_boundary = time_diff / (drifts + 1e-8)
    winner = 1 - np.argmin(time_to_boundary, axis=1)
    rt = (np.min(time_to_boundary, axis=1) + t0_).flatten()
    return np.stack([rt, winner], axis=1)

def prepare_data_for_model(df, subject_id_list):
    """準備模型數據"""
    filtered_df = df[df['participant'].isin(subject_id_list)].copy()
    if len(filtered_df) == 0: raise ValueError("找不到任何指定受試者的資料")
    stimulus_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    choice_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    left_stim_is_diag = filtered_df['Stimulus'].map(lambda s: stimulus_mapping.get(s, {}).get('left')).values
    right_stim_is_diag = filtered_df['Stimulus'].map(lambda s: stimulus_mapping.get(s, {}).get('right')).values
    left_choice_is_diag = filtered_df['Response'].map(lambda r: choice_mapping.get(r, {}).get('left')).values
    right_choice_is_diag = filtered_df['Response'].map(lambda r: choice_mapping.get(r, {}).get('right')).values
    left_match = (left_stim_is_diag == 1)
    right_match = (right_stim_is_diag == 1)
    is_correct = (left_stim_is_diag == left_choice_is_diag) & (right_stim_is_diag == right_choice_is_diag)
    return {"rt": filtered_df['RT'].values, "response_correct": is_correct.astype(int), "left_match": left_match.astype(int), "right_match": right_match.astype(int)}

if __name__ == '__main__':
    print("=" * 60)
    print("檔案1: 數據處理與 Coactive 模型擬合")
    print("=" * 60)
    
    # --- 數據載入與預處理 ---
    try:
        df = pd.read_csv('GRT_LBA.csv')
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        prepared_full_data = prepare_data_for_model(df, df['participant'].unique())
        df['is_correct'] = prepared_full_data['response_correct']
        print("✅ 數據載入並計算正確率成功。")
    except Exception as e:
        print(f"❌ 數據載入或準備錯誤: {e}")
        exit()

    accuracy_per_subject = df.groupby('participant')['is_correct'].mean()
    high_accuracy_subjects = accuracy_per_subject[accuracy_per_subject > 0.80].index.tolist()
    df_filtered = df[df['participant'].isin(high_accuracy_subjects)].copy()
    rt_threshold = 0.150
    df_cleaned = df_filtered[df_filtered['RT'] >= rt_threshold].copy()
    print(f"✅ 數據預處理完成，共 {len(high_accuracy_subjects)} 位高正確率受試者，共 {len(df_cleaned)} 筆有效試次。")

    # 準備模型輸入數據
    participant_ids = df_cleaned['participant'].unique()
    participant_idx, _ = pd.factorize(df_cleaned['participant'])
    coords = {"participant": participant_ids, "obs_id": np.arange(len(participant_idx))}
    model_input_data = prepare_data_for_model(df_cleaned, participant_ids)
    observed_value = np.column_stack([
        np.asarray(model_input_data['rt'], np.float32),
        np.asarray(model_input_data['response_correct'], np.float32)
    ])

    # 保存數據供其他檔案使用
    data_for_models = {
        'observed_value': observed_value,
        'participant_idx': participant_idx,
        'model_input_data': model_input_data,
        'coords': coords
    }
    np.savez('model_data.npz', **data_for_models)
    print("✅ 模型數據已保存至 model_data.npz")

    # --- 模型設定 ---
    B_CONSTANT = 1.0
    A_CONSTANT = 0.1
    T0_CONSTANT = 0.1
    MCMC_CONFIG = {'draws': 2000, 'tune': 1500, 'chains': 4, 'cores': 1, 'target_accept': 0.9}

    print(f"\n開始 Coactive 模型擬合，觀測數據形狀: {observed_value.shape}")

    # --- 建立並擬合 Coactive 模型 ---
    print("\n--- 擬合 Coactive 模型 ---")
    with pm.Model(coords=coords) as hierarchical_coactive_model:
        # 階層參數定義
        mu_v_lm = pm.Normal('mu_v_lm', mu=1.0, sigma=0.5)
        sigma_v_lm = pm.HalfNormal('sigma_v_lm', sigma=0.5)
        offset_v_lm = pm.Normal('offset_v_lm', mu=0, sigma=1, dims="participant")
        v_left_match = pm.Deterministic('v_left_match', mu_v_lm + offset_v_lm * sigma_v_lm)
        
        mu_v_lmm = pm.Normal('mu_v_lmm', mu=0.5, sigma=0.5)
        sigma_v_lmm = pm.HalfNormal('sigma_v_lmm', sigma=0.5)
        offset_v_lmm = pm.Normal('offset_v_lmm', mu=0, sigma=1, dims="participant")
        v_left_mismatch = pm.Deterministic('v_left_mismatch', mu_v_lmm + offset_v_lmm * sigma_v_lmm)
        
        mu_v_rm = pm.Normal('mu_v_rm', mu=1.0, sigma=0.5)
        sigma_v_rm = pm.HalfNormal('sigma_v_rm', sigma=0.5)
        offset_v_rm = pm.Normal('offset_v_rm', mu=0, sigma=1, dims="participant")
        v_right_match = pm.Deterministic('v_right_match', mu_v_rm + offset_v_rm * sigma_v_rm)
        
        mu_v_rmm = pm.Normal('mu_v_rmm', mu=0.5, sigma=0.5)
        sigma_v_rmm = pm.HalfNormal('sigma_v_rmm', sigma=0.5)
        offset_v_rmm = pm.Normal('offset_v_rmm', mu=0, sigma=1, dims="participant")
        v_right_mismatch = pm.Deterministic('v_right_mismatch', mu_v_rmm + offset_v_rmm * sigma_v_rmm)
        
        # 計算個體層級的漂移率
        v_l = v_left_match[participant_idx] * model_input_data['left_match'] + v_left_mismatch[participant_idx] * (1 - model_input_data['left_match'])
        v_r = v_right_match[participant_idx] * model_input_data['right_match'] + v_right_mismatch[participant_idx] * (1 - model_input_data['right_match'])
        
        # Coactive 模型：簡單相加
        v_final_correct = pm.Deterministic('v_final_correct', v_l + v_r)
        v_final_incorrect = 0.1
        
        likelihood = pm.CustomDist('likelihood', v_final_correct, v_final_incorrect, B_CONSTANT, A_CONSTANT, T0_CONSTANT, logp=logp_lba, random=lba_random, observed=observed_value)
    
    # 擬合模型
    with hierarchical_coactive_model:
        start_time = time.time()
        print("開始 MCMC 採樣...")
        idata_coactive = pm.sample(**MCMC_CONFIG)
        print("計算對數似然值...")
        pm.compute_log_likelihood(idata_coactive)
        print("生成後驗預測...")
        idata_coactive.extend(pm.sample_posterior_predictive(idata_coactive, progressbar=True))
        end_time = time.time()
        print(f"✅ Coactive 模型擬合完成，耗時 {end_time - start_time:.1f} 秒")

    # 保存結果
    idata_coactive.to_netcdf('coactive_model_results.nc')
    print("✅ Coactive 模型結果已保存至 coactive_model_results.nc")
    
    # 快速檢查收斂性
    print("\n--- Coactive 模型收斂性檢查 ---")
    try:
        summary = az.summary(idata_coactive)
        max_rhat = summary['r_hat'].max()
        bad_rhat_count = (summary['r_hat'] > 1.01).sum()
        
        print(f"最大 r-hat: {max_rhat:.4f}")
        print(f"r-hat > 1.01 的參數數量: {bad_rhat_count}")
        
        if max_rhat < 1.01:
            print("✅ 模型收斂良好")
        elif max_rhat < 1.05:
            print("⚠️  模型收斂可接受")
        else:
            print("❌ 模型收斂有問題，建議增加採樣數或調整參數")
            
    except Exception as e:
        print(f"收斂性檢查失敗: {e}")
    
    print("\n" + "=" * 60)
    print("Coactive 模型處理完成！")
    print("下一步：運行檔案2處理 Parallel 模型")
    print("=" * 60)
