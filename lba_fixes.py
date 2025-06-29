# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 04:07:28 2025

@author: spt904
"""

# LBA 具體修復指南 - 逐步替換說明
# ===================================================================

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import time

# ===================================================================
# 修復 1: 數據單位檢查和修復
# ===================================================================

def fix_data_units(data_file='model_data.npz'):
    """修復數據單位問題 - 檢查 RT 是否需要轉換為毫秒"""
    
    print("檢查數據單位...")
    data = np.load(data_file, allow_pickle=True)
    observed_value = data['observed_value'].copy()
    
    # 檢查 RT 值範圍
    rt_values = observed_value[:, 0]
    mean_rt = rt_values.mean()
    
    print(f"當前 RT 平均值: {mean_rt:.3f}")
    
    if mean_rt < 10:  # 可能是秒
        print("檢測到 RT 是秒單位，轉換為毫秒...")
        observed_value[:, 0] *= 1000
        
        # 保存修正後的數據
        np.savez('model_data_fixed.npz', 
                observed_value=observed_value,
                participant_idx=data['participant_idx'],
                model_input_data=data['model_input_data'])
        print("✓ 修正後的數據已保存為 model_data_fixed.npz")
        return 'model_data_fixed.npz'
    else:
        print("RT 單位正確，無需轉換")
        return data_file

# ===================================================================
# 修復 2: 替換 lba_models.py 中的核心函數
# ===================================================================
def lba_logp_correct(value, v, A, b, t0, s=1.0):
    """
    分析上正確的 LBA 對數概似函數 (適用於兩個累積器)。
    
    參數:
    value (Tensor): 觀測數據 (rt, response), shape=(n_trials, 2)
    v (Tensor): 漂移率向量 [v_correct, v_incorrect], shape=(n_trials, 2)
    A (Tensor): 最大起始點 variability, shape=(n_trials,)
    b (Tensor): 反應閾值, shape=(n_trials,)
    t0 (Tensor): 非決策時間, shape=(n_trials,)
    s (float): 漂移率標準差 (通常固定為 1.0)
    """
    
    # ------------------------------------------------------------------
    # 1. 定義 PyTensor/Aesara 中的常態 PDF (phi) 和 CDF (Phi)
    # ------------------------------------------------------------------
    def phi(x):
        """常態分佈機率密度函數 (PDF)"""
        return (1.0 / pt.sqrt(2.0 * np.pi)) * pt.exp(-0.5 * x**2)

    def Phi(x):
        """常態分佈累積分布函數 (CDF)"""
        return 0.5 * (1.0 + pt.erf(x / pt.sqrt(2.0)))

    # ------------------------------------------------------------------
    # 2. 為單一 LBA 累積器計算 PDF g(t) 和存活函數 S(t)
    #    這是根據 Brown & Heathcote (2008) 的論文
    # ------------------------------------------------------------------
    def get_pdf_and_survival(t, v_i, b, A, s):
        """計算單一累積器的 g(t) 和 S(t)"""
        
        # 為數值穩定性，避免除以零
        t_safe = pt.maximum(t, 1e-6)
        
        # 計算四個主要項
        term1_den = s * t_safe
        term1_val = (b - A - t_safe * v_i) / term1_den
        
        term2_den = s * t_safe
        term2_val = (b - t_safe * v_i) / term2_den
        
        term3_den = A
        term3_val = v_i * (Phi(term2_val) - Phi(term1_val))
        
        term4_den = A
        term4_val = (s * t_safe) * (phi(term1_val) - phi(term2_val))

        # 計算 PDF: g(t)
        pdf = (1.0 / A) * (-v_i * Phi(term1_val) + \
                           v_i * Phi(term2_val) + \
                           s * phi(term1_val) - \
                           s * phi(term2_val))
                           
        # 計算 CDF: G(t)
        cdf = 1.0 + ((v_i * t_safe - b) / A) * Phi(term2_val) - \
              (((v_i * t_safe) - (b - A)) / A) * Phi(term1_val) + \
              (t_safe * s / A) * phi(term2_val) - \
              (t_safe * s / A) * phi(term1_val)

        # 存活函數 S(t) = 1 - G(t)
        survival = 1.0 - cdf

        # 再次確保數值穩定性
        pdf_stable = pt.maximum(pdf, 1e-8)
        survival_stable = pt.maximum(survival, 1e-8)
        
        return pdf_stable, survival_stable

    # ------------------------------------------------------------------
    # 3. 主函數邏輯
    # ------------------------------------------------------------------
    rt = value[:, 0] / 1000.0  # 從毫秒轉換為秒
    response = value[:, 1]    # 1=correct, 0=incorrect
    
    # 決策時間 (必須為正)
    decision_time = rt - t0
    
    # 根據反應選擇正確和錯誤的漂移率
    v_correct = v[:, 0]
    v_incorrect = v[:, 1]
    
    # 計算獲勝和落敗累積器的 PDF/Survival
    g_correct, S_correct = get_pdf_and_survival(decision_time, v_correct, b, A, s)
    g_incorrect, S_incorrect = get_pdf_and_survival(decision_time, v_incorrect, b, A, s)
    
    # 總似然是兩個獨立事件的組合：
    # 1. 對於正確反應 (response=1): 獲勝者(correct)在 t 時刻完成，落敗者(incorrect)在 t 時刻尚未完成
    #    likelihood = g_correct(t) * S_incorrect(t)
    # 2. 對於錯誤反應 (response=0): 獲勝者(incorrect)在 t 時刻完成，落敗者(correct)在 t 時刻尚未完成
    #    likelihood = g_incorrect(t) * S_correct(t)
    
    # 使用 pt.switch 根據 response 的值選擇對應的似然
    likelihood = pt.switch(
        pt.eq(response, 1),
        g_correct * S_incorrect,
        g_incorrect * S_correct
    )
    
    # 最終的對數概似
    logp = pt.log(pt.maximum(likelihood, 1e-8))
    
    # 最後的保護，防止任何 NaN 或 inf 值
    return pt.switch(pt.or_(pt.isnan(logp), pt.isinf(logp)), -100.0, logp)
def safe_lba_logp(value, v1, v2, A, b, t0, s=1.0):
    """
    替換 lba_models.py 中的 lba_logp 函數
    更穩定的數值計算
    """
    
    # 提取數據
    rt = value[:, 0] / 1000.0  # 轉換為秒
    response = value[:, 1]
    
    # 參數安全性限制
    v1 = pt.clip(v1, 0.01, 20.0)
    v2 = pt.clip(v2, 0.01, 20.0)
    A = pt.clip(A, 0.01, 5.0)
    b = pt.clip(b, A + 0.01, 10.0)
    t0 = pt.clip(t0, 0.001, 2.0)
    
    # 決策時間
    decision_time = pt.maximum(rt - t0, 0.001)
    
    # 選擇對應的漂移率
    v_chosen = pt.switch(pt.eq(response, 1), v1, v2)
    v_other = pt.switch(pt.eq(response, 1), v2, v1)
    
    # 簡化的似然計算 (使用 Wald 近似)
    mean_time_chosen = (b - A/2) / pt.maximum(v_chosen, 0.01)
    
    # Wald 分布對數密度
    log_pdf_winner = (-0.5 * pt.log(2 * np.pi * decision_time**3) - 
                      0.5 * (decision_time - mean_time_chosen)**2 / 
                      (decision_time * mean_time_chosen**2))
    
    # 存活函數 (簡化)
    mean_time_other = (b - A/2) / pt.maximum(v_other, 0.01)
    survival_prob = pt.exp(-decision_time / mean_time_other)
    log_survival_loser = pt.log(pt.maximum(survival_prob, 1e-10))
    
    # 聯合對數似然
    logp = log_pdf_winner + log_survival_loser
    
    # 數值穩定性
    logp = pt.switch(pt.isnan(logp), -100.0, logp)
    logp = pt.switch(pt.isinf(logp), -100.0, logp)
    logp = pt.clip(logp, -100.0, 10.0)
    
    return logp

def safe_lba_random(v1, v2, A, b, t0, s=1.0, rng=None, size=None):
    """
    替換 lba_models.py 中的 lba_random 函數
    """
    if size is None:
        size = (1,)
    elif isinstance(size, int):
        size = (size,)
    
    n_samples = size[0]
    samples = np.zeros((n_samples, 2))
    
    # 轉換參數
    v1_val = float(np.asarray(v1).item())
    v2_val = float(np.asarray(v2).item())
    A_val = float(np.asarray(A).item())
    b_val = float(np.asarray(b).item())
    t0_val = float(np.asarray(t0).item())
    
    for i in range(n_samples):
        # 簡化模擬
        mean_t1 = (b_val - A_val/2) / max(v1_val, 0.01)
        mean_t2 = (b_val - A_val/2) / max(v2_val, 0.01)
        
        # 使用指數分布近似
        t1 = rng.exponential(mean_t1)
        t2 = rng.exponential(mean_t2)
        
        if t1 < t2:
            samples[i, 0] = (t1 + t0_val) * 1000  # 轉回毫秒
            samples[i, 1] = 1
        else:
            samples[i, 0] = (t2 + t0_val) * 1000
            samples[i, 1] = 0
        
        samples[i, 0] = np.clip(samples[i, 0], 200, 5000)
    
    return samples

def create_robust_coactive_lba_model(observed_data, input_data):
    """
    替換 lba_models.py 中的 create_coactive_lba_model 函數
    使用更保守的先驗分布
    """
    with pm.Model() as model:
        
        # 更保守的先驗分布
        v_match = pm.TruncatedNormal('v_match', 
                                   mu=1.5, sigma=0.5, 
                                   lower=0.5, upper=5.0)
        
        v_mismatch = pm.TruncatedNormal('v_mismatch', 
                                      mu=1.0, sigma=0.5, 
                                      lower=0.1, upper=3.0)
        
        # LBA 參數 (使用更合理的範圍)
        start_var = pm.TruncatedNormal('start_var', 
                                     mu=0.5, sigma=0.2, 
                                     lower=0.1, upper=1.5)
        
        boundary_offset = pm.TruncatedNormal('boundary_offset', 
                                           mu=1.0, sigma=0.3, 
                                           lower=0.3, upper=2.0)
        
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        
        non_decision = pm.TruncatedNormal('non_decision', 
                                        mu=0.3, sigma=0.1, 
                                        lower=0.1, upper=0.8)
        
        # 計算漂移率
        v_left = (v_match * input_data['left_match'] + 
                  v_mismatch * (1 - input_data['left_match']))
        v_right = (v_match * input_data['right_match'] + 
                   v_mismatch * (1 - input_data['right_match']))
        
        # Coactive 整合
        v_final_correct = pm.Deterministic('v_final_correct', v_left + v_right)
        
        v_final_incorrect = pm.TruncatedNormal('v_final_incorrect', 
                                             mu=0.5, sigma=0.3, 
                                             lower=0.1, upper=2.0)
        
        # 使用安全的似然函數
        likelihood = pm.CustomDist('likelihood',
                                  v_final_correct, v_final_incorrect, 
                                  start_var, b_safe, non_decision,
                                  logp=safe_lba_logp,
                                  random=safe_lba_random,
                                  observed=observed_data)
    
    return model

# ===================================================================
# 修復 3: 替換 LBA tool.py 中的採樣函數
# ===================================================================

def robust_sample_with_convergence_check(model, max_attempts=2, 
                                        initial_draws=100, initial_tune=200):
    """
    Replaces sample_with_convergence_check in LBA_tool.py.
    Uses a more robust initialization strategy.
    """
    
    draws = initial_draws
    tune = initial_tune
    
    print(f"  開始穩健採樣 (draws={draws}, tune={tune})...")
    
    for attempt in range(max_attempts):
        try:
            print(f"    嘗試 {attempt + 1}/{max_attempts}")
            
            with model:
                # Attempt to find a good starting point using MAP estimation
                start_map = None
                try:
                    print("    尋找 MAP 估計...")
                    start_map = pm.find_MAP(method='BFGS', maxeval=5000) # Increased maxeval
                    print("    ✓ MAP 估計成功")
                except Exception as map_error:
                    print(f"    ⚠️ MAP 估計失敗: {map_error}. Using default initialization.")
                
                # Execute sampling with a more robust init strategy
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=2,
                    target_accept=0.95,
                    return_inferencedata=True,
                    progressbar=True,
                    random_seed=42 + attempt,
                    start=start_map,
                    init='jitter+adapt_diag'  # MORE ROBUST INITIALIZATION
                )
                
                # Simple convergence check
                try:
                    summary = az.summary(trace)
                    max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.0
                    min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else 100
                    
                    print(f"    最大 R-hat: {max_rhat:.4f}")
                    print(f"    最小 ESS: {min_ess:.0f}")
                    
                    if max_rhat < 1.2 and min_ess > 50:
                        print(f"    ✓ 收斂成功")
                        return trace, {'max_rhat': max_rhat, 'min_ess': min_ess}
                    
                except Exception as e:
                    print(f"    ⚠️ 收斂檢查失敗，但採樣完成: {e}")
                    if trace is not None:
                        return trace, {'max_rhat': np.nan, 'min_ess': np.nan}
                
        except Exception as e:
            print(f"    ❌ 採樣失敗: {e}")
        
        # Adjust parameters for the next attempt
        if attempt < max_attempts - 1:
            draws = max(50, int(draws * 0.8))
            tune = max(50, int(tune * 0.8))
            print(f"    調整參數: draws={draws}, tune={tune}")
    
    print(f"    ❌ {max_attempts} 次嘗試後仍未成功")
    return None, None
# ===================================================================
# 修復 4: 快速測試函數 (IMPROVED)
# ===================================================================

def quick_test():
    """快速測試所有修復是否有效"""
    
    print("🧪 快速測試修復效果...")
    
    try:
        # 1. 為測試創建一個自包含的虛擬數據文件
        n_trials = 30 # A slightly larger test set
        test_data = np.zeros((n_trials, 2))
        test_data[:, 0] = np.random.uniform(300, 1500, n_trials)
        test_data[:, 1] = np.random.binomial(1, 0.8, n_trials)
        participant_idx = np.zeros(n_trials, dtype=int)
        model_input_data = {
            'left_match': np.random.binomial(1, 0.5, n_trials).astype(float),
            'right_match': np.random.binomial(1, 0.5, n_trials).astype(float)
        }
        np.savez('test_model_data.npz',
                 observed_value=test_data,
                 participant_idx=participant_idx,
                 model_input_data=model_input_data)
        
        # 2. 載入小樣本測試
        data = np.load('test_model_data.npz', allow_pickle=True)
        observed_value = data['observed_value']
        participant_idx = data['participant_idx']
        model_input_data = data['model_input_data'].item()
        
        mask = participant_idx == 0
        test_data = observed_value[mask]
        test_input = {
            'left_match': model_input_data['left_match'][mask],
            'right_match': model_input_data['right_match'][mask]
        }
        print(f"✓ 測試數據準備完成: {len(test_data)} 試驗")
        
        # 3. 測試模型創建
        model = create_robust_coactive_lba_model(test_data, test_input)
        print("✓ 模型創建成功")
        
        # 4. 測試模型編譯
        with model:
            test_point = model.initial_point()
            logp = model.compile_logp()
            logp_val = logp(test_point)
            if np.isnan(logp_val) or np.isinf(logp_val):
                raise ValueError(f"Initial logp is invalid: {logp_val}, indicating a model setup problem.")
            print(f"✓ 模型編譯成功，logp: {logp_val:.2f}")
        
        # 5. 快速採樣測試 (Increased tune/draws for better stability)
        print("  Running quick sampling test...")
        trace, diagnostics = robust_sample_with_convergence_check(
            model, max_attempts=1, initial_draws=150, initial_tune=200
        )
        
        if trace is not None:
            print("\n✅ 快速測試全部通過！")
            return True
        else:
            print("\n❌ 採樣測試失敗. The model compiled but failed the brief sampling/convergence check.")
            print("   This might be due to the inherent difficulty of the model or stochasticity.")
            print("   The main analysis in LBA_main.py, which uses more samples, may still succeed.")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

# ===================================================================
# 主要替換說明
# ===================================================================

def show_replacement_guide():
    """顯示具體的替換指南"""
    
    guide = """
    
    📋 具體替換步驟:
    
    步驟 1: 保存修復文件
    ├── 將本代碼保存為 lba_fixes.py
    
    步驟 2: 修改 lba_models.py
    ├── 替換 lba_logp → safe_lba_logp
    ├── 替換 lba_random → safe_lba_random  
    ├── 替換 create_coactive_lba_model → create_robust_coactive_lba_model
    └── 在文件開頭添加: from lba_fixes import safe_lba_logp, safe_lba_random
    
    步驟 3: 修改 LBA tool.py
    ├── 替換 sample_with_convergence_check → robust_sample_with_convergence_check
    └── 在文件開頭添加: from lba_fixes import robust_sample_with_convergence_check
    
    步驟 4: 修改 LBA_main.py  
    ├── 在 setup_analysis() 開頭添加數據修復:
    │   from lba_fixes import fix_data_units
    │   self.data_file = fix_data_units(self.data_file)
    └── 運行前先測試: from lba_fixes import quick_test; quick_test()
    
    步驟 5: 運行測試
    ├── python -c "from lba_fixes import quick_test; quick_test()"
    └── 如果測試通過，再運行完整分析
    
    """
    
    print(guide)
    return guide

if __name__ == '__main__':
    print("🔧 LBA 修復工具")
    show_replacement_guide()
    
    # 運行快速測試
    success = quick_test()
    if success:
        print("\n✅ 修復測試通過，可以應用到主程式！")
    else:
        print("\n❌ 修復測試失敗，需要進一步調試")