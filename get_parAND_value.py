import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import time
import matplotlib.pyplot as plt

# ===================================================================
# Phase 1: LBA 核心函數 (這部分維持不變)
# ===================================================================
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

# ===================================================================
# Phase 2: 數據準備函數 (這部分維持不變)
# ===================================================================
def prepare_data_for_model(df, subject_id_list):
    # 修改函數以處理一個受試者列表
    filtered_df = df[df['participant'].isin(subject_id_list)].copy()
    if len(filtered_df) == 0: raise ValueError("找不到任何指定受試者的資料")
    stimulus_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    choice_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    # 確保返回的字典包含 'rt', 'response_correct', 'left_match', 'right_match'
    stimulus_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    choice_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    left_stim_is_diag = filtered_df['Stimulus'].map(lambda s: stimulus_mapping[s]['left']).values
    right_stim_is_diag = filtered_df['Stimulus'].map(lambda s: stimulus_mapping[s]['right']).values
    left_choice_is_diag = filtered_df['Response'].map(lambda r: choice_mapping[r]['left']).values
    right_choice_is_diag = filtered_df['Response'].map(lambda r: choice_mapping[r]['right']).values
    left_match = (left_stim_is_diag == 1); right_match = (right_stim_is_diag == 1)
    is_correct = (left_stim_is_diag == left_choice_is_diag) & (right_stim_is_diag == right_choice_is_diag)
    return {"rt": filtered_df['RT'].values, "response_correct": is_correct.astype(int), "left_match": left_match.astype(int), "right_match": right_match.astype(int)}
if __name__ == '__main__':
    # --- 數據載入與篩選 ---
    try:
        df = pd.read_csv('GRT_LBA.csv')
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        # 為了計算正確率，我們需要先準備 is_correct 欄位
        prepared_full_data = prepare_data_for_model(df, df['participant'].unique())
        df['is_correct'] = prepared_full_data['response_correct']
        print("✅ 數據載入並計算正確率成功。")
    except Exception as e:
        print(f"❌ 數據載入錯誤: {e}"); exit()

    # 1. 根據正確率篩選受試者
    accuracy_per_subject = df.groupby('participant')['is_correct'].mean()
    high_accuracy_subjects = accuracy_per_subject[accuracy_per_subject > 0.80].index.tolist()
    df_filtered = df[df['participant'].isin(high_accuracy_subjects)].copy()
    print(f"✅ 根據正確率 > 80%，篩選出 {len(high_accuracy_subjects)} 位受試者。")

    # 2. 清理過快的反應時間
    rt_threshold = 0.150
    df_cleaned = df_filtered[df_filtered['RT'] >= rt_threshold].copy()
    print(f"✅ 清理 RT < {rt_threshold}s 的數據完成。")

    # 3. 準備 PyMC 需要的座標和索引
    participant_ids = df_cleaned['participant'].unique()
    participant_idx, _ = pd.factorize(df_cleaned['participant'])
    coords = {"participant": participant_ids, "obs_id": np.arange(len(participant_idx))}
    
    # 4. 準備最終給模型的數據
    model_input_data = prepare_data_for_model(df_cleaned, participant_ids)
    observed_value = np.column_stack([
        np.asarray(model_input_data['rt'], np.float32),
        np.asarray(model_input_data['response_correct'], np.float32)
    ])
    
    # --- 模型建立與設定 ---
    B_CONSTANT = 1.0; A_CONSTANT = 0.1; T0_CONSTANT = 0.1 # 使用更安全的固定值
    MCMC_CONFIG = {'draws': 2000, 'tune': 1500, 'chains': 4, 'cores': 1}

    with pm.Model(coords=coords) as parallel_hierarchical_model:
        print("\n--- 正在建立階層式 Parallel AND 模型 ---")

        # --- 為每一個 v 參數建立階層結構 (非中心化) ---
        # 1. v_left_match
        mu_v_lm = pm.Normal('mu_v_lm', mu=1.0, sigma=0.5)
        sigma_v_lm = pm.HalfNormal('sigma_v_lm', sigma=0.5)
        offset_v_lm = pm.Normal('offset_v_lm', mu=0.0, sigma=1.0, dims="participant")
        v_left_match = pm.Deterministic('v_left_match', mu_v_lm + offset_v_lm * sigma_v_lm)

        # 2. v_left_mismatch
        mu_v_lmm = pm.Normal('mu_v_lmm', mu=0.5, sigma=0.5)
        sigma_v_lmm = pm.HalfNormal('sigma_v_lmm', sigma=0.5)
        offset_v_lmm = pm.Normal('offset_v_lmm', mu=0.0, sigma=1.0, dims="participant")
        v_left_mismatch = pm.Deterministic('v_left_mismatch', mu_v_lmm + offset_v_lmm * sigma_v_lmm)

        # 3. v_right_match
        mu_v_rm = pm.Normal('mu_v_rm', mu=1.0, sigma=0.5)
        sigma_v_rm = pm.HalfNormal('sigma_v_rm', sigma=0.5)
        offset_v_rm = pm.Normal('offset_v_rm', mu=0.0, sigma=1.0, dims="participant")
        v_right_match = pm.Deterministic('v_right_match', mu_v_rm + offset_v_rm * sigma_v_rm)
        
        # 4. v_right_mismatch
        mu_v_rmm = pm.Normal('mu_v_rmm', mu=0.5, sigma=0.5)
        sigma_v_rmm = pm.HalfNormal('sigma_v_rmm', sigma=0.5)
        offset_v_rmm = pm.Normal('offset_v_rmm', mu=0.0, sigma=1.0, dims="participant")
        v_right_mismatch = pm.Deterministic('v_right_mismatch', mu_v_rmm + offset_v_rmm * sigma_v_rmm)

        # 5. (可選) 為 LogSumExp 的平滑參數 k 也建立階層結構
        mu_log_k = pm.Normal('mu_log_k', mu=np.log(2.0), sigma=1.0)
        sigma_log_k = pm.HalfNormal('sigma_log_k', sigma=1.0)
        offset_log_k = pm.Normal('offset_log_k', mu=0, sigma=1, dims="participant")
        # 使用 exp 確保 k 永遠為正
        k = pm.Deterministic('k', pt.exp(mu_log_k + offset_log_k * sigma_log_k))

        # --- 根據每筆觀測數據，匹配對應受試者的參數 ---
        v_l = v_left_match[participant_idx] * model_input_data['left_match'] + \
              v_left_mismatch[participant_idx] * (1 - model_input_data['left_match'])
              
        v_r = v_right_match[participant_idx] * model_input_data['right_match'] + \
              v_right_mismatch[participant_idx] * (1 - model_input_data['right_match'])
        
        k_obs = k[participant_idx]

        # --- LogSumExp 計算 v_final_correct ---
        v_final_correct = pm.Deterministic(
            'v_final_correct',
            (pt.log(pt.exp(k_obs * v_l) + pt.exp(k_obs * v_r))) / k_obs
        )
        v_final_incorrect = 0.1 # 假設錯誤反應的漂移率為一個小常數

        # --- Likelihood ---
        likelihood = pm.CustomDist(
            'likelihood',
            v_final_correct, v_final_incorrect, B_CONSTANT, A_CONSTANT, T0_CONSTANT,
            logp=logp_lba,
            observed=observed_value
        )
        
        # --- 執行採樣 ---
        print("\n🔬 開始擬合階層式 Parallel AND 模型...")
        idata_parallel_hierarchical = pm.sample(**MCMC_CONFIG)

    # --- 後續分析 ---
    print("\n✅ 階層模型擬合完成，顯示摘要...")
    # 查看群體層級的參數，它們是我們最感興趣的
    summary = az.summary(idata_parallel_hierarchical, var_names=['mu_v_lm', 'sigma_v_lm', 'mu_v_lmm', 'sigma_v_lmm', 'mu_v_rm', 'sigma_v_rm', 'mu_v_rmm', 'sigma_v_rmm', 'k'])
    print(summary)

    az.plot_trace(idata_parallel_hierarchical, var_names=['mu_v_lm', 'sigma_v_lm', 'mu_v_rm', 'sigma_v_rm'])
    plt.show()
