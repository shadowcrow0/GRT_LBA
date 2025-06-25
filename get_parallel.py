# -*- coding: utf-8 -*-
"""
檔案2: Parallel AND 模型擬合
需要先運行檔案1生成 model_data.npz
輸出: parallel_model_results.nc
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import time

# ===================================================================
# LBA 核心函數 (與檔案1相同)
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

def stable_logsumexp_mean(v1, v2, k):
    """數值穩定的 log-sum-exp 平均"""
    max_v = pm.math.maximum(k * v1, k * v2)
    return max_v/k + pm.math.log(pm.math.exp(k * v1 - max_v) + pm.math.exp(k * v2 - max_v))/k

if __name__ == '__main__':
    print("=" * 60)
    print("檔案2: Parallel AND 模型擬合")
    print("=" * 60)
    
    # --- 載入數據 ---
    try:
        data = np.load('model_data.npz', allow_pickle=True)
        observed_value = data['observed_value']
        participant_idx = data['participant_idx']
        model_input_data = data['model_input_data'].item()  # 轉回字典
        coords = data['coords'].item()  # 轉回字典
        print("✅ 成功載入模型數據")
        print(f"觀測數據形狀: {observed_value.shape}")
        print(f"參與者數量: {len(coords['participant'])}")
    except FileNotFoundError:
        print("❌ 找不到 model_data.npz，請先運行檔案1")
        exit()
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        exit()

    # --- 模型設定 ---
    B_CONSTANT = 1.0
    A_CONSTANT = 0.1
    T0_CONSTANT = 0.1
    MCMC_CONFIG = {'draws': 2000, 'tune': 1500, 'chains': 4, 'cores': 1, 'target_accept': 0.9}

    print("\n--- 擬合修正的 Parallel AND 模型 ---")
    
    # --- 建立並擬合 Parallel AND 模型 ---
    with pm.Model(coords=coords) as hierarchical_parallel_model:
        # 相同的階層參數定義
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
        
        # *** 修正的 k 參數和 Parallel AND 計算 ***
        # 選項1: 使用 Gamma 分布
        k = pm.Gamma('k_smoothness', alpha=2.0, beta=1.0)  # 平均值為2
        
        # 選項2: 如果還是有問題，可以改用固定值測試
        # k = 2.0  # 固定值
        
        # 選項3: 使用更保守的先驗
        # k = pm.Gamma('k_smoothness', alpha=1.5, beta=0.5)  # 平均值為3，但變異較小
        
        # 使用數值穩定的計算
        v_final_correct = pm.Deterministic('v_final_correct', stable_logsumexp_mean(v_l, v_r, k))
        v_final_incorrect = 0.1
        
        likelihood = pm.CustomDist('likelihood', v_final_correct, v_final_incorrect, B_CONSTANT, A_CONSTANT, T0_CONSTANT, logp=logp_lba, random=lba_random, observed=observed_value)
    
    # 擬合模型
    with hierarchical_parallel_model:
        start_time = time.time()
        print("開始 MCMC 採樣...")
        idata_parallel = pm.sample(**MCMC_CONFIG)
        print("計算對數似然值...")
        pm.compute_log_likelihood(idata_parallel)
        print("生成後驗預測...")
        idata_parallel.extend(pm.sample_posterior_predictive(idata_parallel, progressbar=True))
        end_time = time.time()
        print(f"✅ Parallel AND 模型擬合完成，耗時 {end_time - start_time:.1f} 秒")

    # 保存結果
    idata_parallel.to_netcdf('parallel_model_results.nc')
    print("✅ Parallel AND 模型結果已保存至 parallel_model_results.nc")
    
    # 快速檢查收斂性和 k 參數
    print("\n--- Parallel AND 模型檢查 ---")
    try:
        summary = az.summary(idata_parallel)
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
        
        # 檢查 k 參數
        if 'k_smoothness' in summary.index:
            k_summary = summary.loc['k_smoothness']
            print(f"\nk 參數統計:")
            print(f"  平均值: {k_summary['mean']:.3f}")
            print(f"  標準差: {k_summary['sd']:.3f}")
            print(f"  95% HDI: [{k_summary['hdi_2.5%']:.3f}, {k_summary['hdi_97.5%']:.3f}]")
            print(f"  r-hat: {k_summary['r_hat']:.4f}")
            
            if k_summary['mean'] > 10:
                print("⚠️  警告：k 值偏大，可能導致模型退化")
            elif k_summary['mean'] < 0.5:
                print("⚠️  警告：k 值偏小，可能效果不明顯")
            else:
                print("✅ k 值在合理範圍內")
        
    except Exception as e:
        print(f"模型檢查失敗: {e}")
    
    print("\n" + "=" * 60)
    print("Parallel AND 模型處理完成！")
    print("下一步：運行檔案3進行模型比較")
    print("=" * 60)
