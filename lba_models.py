# -*- coding: utf-8 -*-
"""
LBA Models Module - Based on HDDM and original papers (FIXED & COMPATIBLE)
lba_models.py
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az
from lba_fixes import lba_logp_correct, safe_lba_random # <-- 添加這行，使用新函數
# ===================================================================
# Standardized & Robust Model Definitions
# Using tighter, more informed priors to improve sampling stability.
# ===================================================================

def create_coactive_lba_model(observed_data, input_data):
    """
    Robust Coactive LBA Model - drift rates are summed (v_left + v_right).
    """
    with pm.Model() as model:
        # Priors with reasonable constraints to improve stability
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=5.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.1, upper=3.0)
        
        # Standardized LBA parameters
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.5)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.3, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        
        # --- 修正後的程式碼 (lba_models.py) ---
        # 1. 從傳入的數據中動態計算最小反應時間 (單位：秒)
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        
        # 2. 為了數值穩定性，將上界設為略小於最小RT，並加上一個合理的絕對上限
        #    使用 pt.minimum 確保上界不會意外地變得過大
        upper_bound = pt.minimum(min_rt_seconds * 0.99, 0.5)
        
        # 3. 使用這個動態計算出的上界來定義先驗
        #    這是防止 "Bad initial energy" 錯誤的關鍵
        non_decision = pm.TruncatedNormal('non_decision',
                                        mu=0.15,
                                        sigma=0.05,
                                        lower=0.05,
                                        upper=upper_bound) # <-- 使用動態上界
        # Calculate drift rates based on input data
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        
        # Coactive integration rule
        v_final_correct = pm.Deterministic('v_final_correct', v_left + v_right)
        
        # --- 新的修正程式碼 ---
        # Define the mean drift rate for the incorrect accumulator as a free parameter
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
        
        # Create a deterministic variable that has the same shape as v_final_correct.
        # This ensures its shape is (n_trials,), making it compatible for stacking.
        v_final_incorrect = pm.Deterministic('v_final_incorrect', pt.full_like(v_final_correct, v_incorrect_base))
        
        # Now both tensors have the shape (n_trials,), so stacking along axis=1 will work.
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
                
        likelihood = pm.CustomDist('likelihood',
                                  v_rates,              # <--- 使用堆疊後的漂移率
                                  start_var, b_safe, non_decision,
                                  logp=lba_logp_correct, # <--- 指向新的 logp 函數
                                  random=safe_lba_random,
                                  observed=observed_data)
    return model

def create_parallel_and_lba_model(observed_data, input_data):
    """
    Robust Parallel AND LBA Model - takes the maximum drift rate.
    """
    with pm.Model() as model:
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=5.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.1, upper=3.0)
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.5)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.3, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        # 動態計算最小反應時間 (以秒為單位)
        # --- 修正後的程式碼 (lba_models.py) ---
        # 1. 從傳入的數據中動態計算最小反應時間 (單位：秒)
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        
        # 2. 為了數值穩定性，將上界設為略小於最小RT，並加上一個合理的絕對上限
        #    使用 pt.minimum 確保上界不會意外地變得過大
        upper_bound = pt.minimum(min_rt_seconds * 0.99, 0.5)
        
        # 3. 使用這個動態計算出的上界來定義先驗
        #    這是防止 "Bad initial energy" 錯誤的關鍵
        non_decision = pm.TruncatedNormal('non_decision',
                                        mu=0.15,
                                        sigma=0.05,
                                        lower=0.05,
                                        upper=upper_bound) # <-- 使用動態上界
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        v_final_correct = pm.Deterministic('v_final_correct', pt.maximum(v_left, v_right))
        # --- 新的修正程式碼 ---
        # Define the mean drift rate for the incorrect accumulator as a free parameter
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
        
        # Create a deterministic variable that has the same shape as v_final_correct.
        # This ensures its shape is (n_trials,), making it compatible for stacking.
        v_final_incorrect = pm.Deterministic('v_final_incorrect', pt.full_like(v_final_correct, v_incorrect_base))
        
        # Now both tensors have the shape (n_trials,), so stacking along axis=1 will work.
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
                
        
        likelihood = pm.CustomDist('likelihood',
                                  v_rates,              # <--- 使用堆疊後的漂移率
                                  start_var, b_safe, non_decision,
                                  logp=lba_logp_correct, # <--- 指向新的 logp 函數
                                  random=safe_lba_random,
                                  observed=observed_data)
    return model

def create_parallel_or_lba_model(observed_data, input_data):
    """
    Robust Parallel OR LBA Model - takes the minimum drift rate.
    """
    with pm.Model() as model:
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=5.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.1, upper=3.0)
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.5)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.3, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        # --- 修正後的程式碼 (lba_models.py) ---
        # 1. 從傳入的數據中動態計算最小反應時間 (單位：秒)
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        
        # 2. 為了數值穩定性，將上界設為略小於最小RT，並加上一個合理的絕對上限
        #    使用 pt.minimum 確保上界不會意外地變得過大
        upper_bound = pt.minimum(min_rt_seconds * 0.99, 0.5)
        
        # 3. 使用這個動態計算出的上界來定義先驗
        #    這是防止 "Bad initial energy" 錯誤的關鍵
        non_decision = pm.TruncatedNormal('non_decision',
                                        mu=0.15,
                                        sigma=0.05,
                                        lower=0.05,
                                        upper=upper_bound) # <-- 使用動態上界        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        v_final_correct = pm.Deterministic('v_final_correct', pt.minimum(v_left, v_right))
        # --- 新的修正程式碼 ---
        # Define the mean drift rate for the incorrect accumulator as a free parameter
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
        
        # Create a deterministic variable that has the same shape as v_final_correct.
        # This ensures its shape is (n_trials,), making it compatible for stacking.
        v_final_incorrect = pm.Deterministic('v_final_incorrect', pt.full_like(v_final_correct, v_incorrect_base))
        
        # Now both tensors have the shape (n_trials,), so stacking along axis=1 will work.
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
                
        likelihood = pm.CustomDist('likelihood',
                                  v_rates,              # <--- 使用堆疊後的漂移率
                                  start_var, b_safe, non_decision,
                                  logp=lba_logp_correct, # <--- 指向新的 logp 函數
                                  random=safe_lba_random,
                                  observed=observed_data)
    return model

def create_weighted_average_lba_model(observed_data, input_data):
    """
    Robust Weighted Average LBA Model.
    """
    with pm.Model() as model:
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=5.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.1, upper=3.0)
        w_left = pm.Beta('w_left', alpha=1.0, beta=1.0)
        w_right = pm.Deterministic('w_right', 1.0 - w_left)
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.5)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.3, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        # 動態計算最小反應時間 (以秒為單位)
        # --- 修正後的程式碼 (lba_models.py) ---
        # 1. 從傳入的數據中動態計算最小反應時間 (單位：秒)
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        
        # 2. 為了數值穩定性，將上界設為略小於最小RT，並加上一個合理的絕對上限
        #    使用 pt.minimum 確保上界不會意外地變得過大
        upper_bound = pt.minimum(min_rt_seconds * 0.99, 0.5)
        
        # 3. 使用這個動態計算出的上界來定義先驗
        #    這是防止 "Bad initial energy" 錯誤的關鍵
        non_decision = pm.TruncatedNormal('non_decision',
                                        mu=0.15,
                                        sigma=0.05,
                                        lower=0.05,
                                        upper=upper_bound) # <-- 使用動態上界
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        v_final_correct = pm.Deterministic('v_final_correct', w_left * v_left + w_right * v_right)
        # --- 新的修正程式碼 ---
        # Define the mean drift rate for the incorrect accumulator as a free parameter
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
        
        # Create a deterministic variable that has the same shape as v_final_correct.
        # This ensures its shape is (n_trials,), making it compatible for stacking.
        v_final_incorrect = pm.Deterministic('v_final_incorrect', pt.full_like(v_final_correct, v_incorrect_base))
        
        # Now both tensors have the shape (n_trials,), so stacking along axis=1 will work.
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
                
        
        likelihood = pm.CustomDist('likelihood',
                                  v_rates,              # <--- 使用堆疊後的漂移率
                                  start_var, b_safe, non_decision,
                                  logp=lba_logp_correct, # <--- 指向新的 logp 函數
                                  random=safe_lba_random,
                                  observed=observed_data)
        return model

MODEL_REGISTRY = {
    'Coactive_Addition': {
        'function': create_coactive_lba_model,
        'description': 'Coactive LBA: drift rates are summed (v_left + v_right)',
        'integration_type': 'Addition'
    },
    'Parallel_AND_Maximum': {
        'function': create_parallel_and_lba_model, 
        'description': 'Parallel AND LBA: maximum drift rate (max(v_left, v_right))',
        'integration_type': 'Maximum'
    },
    'Parallel_OR_Minimum': {
        'function': create_parallel_or_lba_model,
        'description': 'Parallel OR LBA: minimum drift rate (min(v_left, v_right))', 
        'integration_type': 'Minimum'
    },
    'Weighted_Average': {
        'function': create_weighted_average_lba_model,
        'description': 'Weighted Average LBA: weighted combination of drift rates',
        'integration_type': 'Weighted_Average'
    }
}

def get_available_models():
    return list(MODEL_REGISTRY.keys())

def create_model_by_name(model_name, observed_data, input_data):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {get_available_models()}")
    
    model_func = MODEL_REGISTRY[model_name]['function']
    print(f"Creating model: {model_name}")
    return model_func(observed_data, input_data)

# 在 lba_models.py 中修復 CustomDist 以確保 log_likelihood 正確生成

import pymc as pm
import pytensor.tensor as pt
import numpy as np
from lba_fixes import lba_logp_correct, safe_lba_random

def create_coactive_lba_model_fixed(observed_data, input_data):
    """
    修復版 Coactive LBA Model - 確保 log_likelihood 正確生成
    """
    with pm.Model() as model:
        # 參數定義保持不變
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=5.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.1, upper=3.0)
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.5)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.3, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        
        # 動態計算 non_decision 上界
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        upper_bound = pt.minimum(min_rt_seconds * 0.99, 0.5)
        non_decision = pm.TruncatedNormal('non_decision', mu=0.15, sigma=0.05, 
                                        lower=0.05, upper=upper_bound)
        
        # 計算漂移率
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        v_final_correct = pm.Deterministic('v_final_correct', v_left + v_right)
        
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
        v_final_incorrect = pm.Deterministic('v_final_incorrect', pt.full_like(v_final_correct, v_incorrect_base))
        
        # 堆疊漂移率
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
        
        # *** 關鍵修復：確保 log_likelihood 正確生成 ***
        # 方法 1: 使用 pm.Potential 直接指定 log_likelihood
        def lba_logp_wrapper(value, v_rates, start_var, b_safe, non_decision):
            """包裝函數確保正確的 log_likelihood 計算"""
            logp = lba_logp_correct(value, v_rates, start_var, b_safe, non_decision)
            # 確保返回的是標量（所有試驗的總 log_likelihood）
            return pt.sum(logp)
        
        # 使用 pm.Potential 而不是 CustomDist
        pm.Potential('likelihood_potential', 
                    lba_logp_wrapper(observed_data, v_rates, start_var, b_safe, non_decision))
        
        # 同時使用 CustomDist 來處理觀測數據，但專注於 random 功能
        likelihood = pm.CustomDist('likelihood',
                                  v_rates, start_var, b_safe, non_decision,
                                  logp=lba_logp_correct,
                                  random=safe_lba_random,
                                  observed=observed_data)
        
        # *** 方法 2: 手動計算 log_likelihood ***
        # 計算每個觀測的 log_likelihood
        individual_logp = lba_logp_correct(observed_data, v_rates, start_var, b_safe, non_decision)
        
        # 將其存儲為確定性變量，這樣 ArviZ 可以訪問它
        pm.Deterministic('log_likelihood_manual', individual_logp)
    
    return model

def create_parallel_and_lba_model_fixed(observed_data, input_data):
    """
    修復版 Parallel AND LBA Model
    """
    with pm.Model() as model:
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=5.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.1, upper=3.0)
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.5)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.3, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        upper_bound = pt.minimum(min_rt_seconds * 0.99, 0.5)
        non_decision = pm.TruncatedNormal('non_decision', mu=0.15, sigma=0.05,
                                        lower=0.05, upper=upper_bound)
        
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        v_final_correct = pm.Deterministic('v_final_correct', pt.maximum(v_left, v_right))
        
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
        v_final_incorrect = pm.Deterministic('v_final_incorrect', pt.full_like(v_final_correct, v_incorrect_base))
        
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
        
        # 同樣的修復
        def lba_logp_wrapper(value, v_rates, start_var, b_safe, non_decision):
            logp = lba_logp_correct(value, v_rates, start_var, b_safe, non_decision)
            return pt.sum(logp)
        
        pm.Potential('likelihood_potential',
                    lba_logp_wrapper(observed_data, v_rates, start_var, b_safe, non_decision))
        
        likelihood = pm.CustomDist('likelihood',
                                  v_rates, start_var, b_safe, non_decision,
                                  logp=lba_logp_correct,
                                  random=safe_lba_random,
                                  observed=observed_data)
        
        individual_logp = lba_logp_correct(observed_data, v_rates, start_var, b_safe, non_decision)
        pm.Deterministic('log_likelihood_manual', individual_logp)
    
    return model

# 更新模型註冊表
MODEL_REGISTRY_FIXED = {
    'Coactive_Addition': {
        'function': create_coactive_lba_model_fixed,
        'description': 'Coactive LBA: drift rates are summed (v_left + v_right) - LOG_LIKELIHOOD FIXED',
        'integration_type': 'Addition'
    },
    'Parallel_AND_Maximum': {
        'function': create_parallel_and_lba_model_fixed,
        'description': 'Parallel AND LBA: maximum drift rate (max(v_left, v_right)) - LOG_LIKELIHOOD FIXED',
        'integration_type': 'Maximum'
    }
}

def create_model_by_name_fixed(model_name, observed_data, input_data):
    """
    修復版模型創建函數
    """
    if model_name not in MODEL_REGISTRY_FIXED:
        # 回退到原始註冊表
        if model_name in MODEL_REGISTRY:
            print(f"⚠️ 使用原始版本的 {model_name}（可能有 log_likelihood 問題）")
            return MODEL_REGISTRY[model_name]['function'](observed_data, input_data)
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY_FIXED.keys())}")
    
    model_func = MODEL_REGISTRY_FIXED[model_name]['function']
    print(f"✅ 創建修復版模型: {model_name}")
    return model_func(observed_data, input_data)

# 增強版採樣函數，確保 log_likelihood 被正確計算
def sample_with_log_likelihood_fix(model, **kwargs):
    """
    採樣後修復 log_likelihood 問題
    """
    print("🔧 使用 log_likelihood 修復版採樣...")
    
    # 先進行正常採樣
    from LBA_tool import sample_with_convergence_check
    trace, diagnostics = sample_with_convergence_check(model, **kwargs)
    
    if trace is None:
        return None, None
    
    # 檢查 log_likelihood 是否存在
    if not hasattr(trace, 'log_likelihood'):
        print("⚠️ trace 缺少 log_likelihood，嘗試手動計算...")
        
        try:
            # 使用手動計算的 log_likelihood
            if 'log_likelihood_manual' in trace.posterior:
                print("✅ 找到手動計算的 log_likelihood")
                
                # 創建一個包含 log_likelihood 的新 trace
                # 這是一個簡化版本，實際實現可能需要更複雜的處理
                import xarray as xr
                
                # 將手動 log_likelihood 重命名
                ll_data = trace.posterior['log_likelihood_manual']
                
                # 創建新的 log_likelihood 數據集
                log_likelihood = xr.Dataset({
                    'likelihood': ll_data
                })
                
                # 將其添加到 trace
                trace = trace.assign(log_likelihood=log_likelihood)
                print("✅ log_likelihood 修復成功")
            else:
                print("❌ 無法找到手動計算的 log_likelihood")
        except Exception as e:
            print(f"❌ log_likelihood 修復失敗: {e}")
    else:
        print("✅ trace 已包含 log_likelihood")
    
    return trace, diagnostics

# 測試函數
def test_log_likelihood_fix():
    """
    測試 log_likelihood 修復是否有效
    """
    print("🧪 測試 log_likelihood 修復...")
    
    # 創建測試數據
    n_trials = 50
    test_data = np.zeros((n_trials, 2))
    test_data[:, 0] = np.random.uniform(300, 1500, n_trials)
    test_data[:, 1] = np.random.binomial(1, 0.8, n_trials)
    
    test_input = {
        'left_match': np.random.binomial(1, 0.5, n_trials).astype(float),
        'right_match': np.random.binomial(1, 0.5, n_trials).astype(float)
    }
    
    try:
        # 測試修復版模型
        model = create_model_by_name_fixed('Coactive_Addition', test_data, test_input)
        print("✅ 修復版模型創建成功")
        
        # 測試模型編譯
        with model:
            test_point = model.initial_point()
            logp = model.compile_logp()
            logp_val = logp(test_point)
            print(f"✅ 模型編譯成功，logp: {logp_val:.2f}")
        
        # 快速採樣測試
        trace, diagnostics = sample_with_log_likelihood_fix(
            model, max_attempts=1, draws=100, tune=200
        )
        
        if trace is not None:
            # 檢查 log_likelihood
            if hasattr(trace, 'log_likelihood'):
                print("✅ log_likelihood 存在")
                
                # 測試 WAIC 計算
                try:
                    waic_result = az.waic(trace)
                    print(f"✅ WAIC 計算成功: {waic_result.elpd_waic:.2f}")
                    return True
                except Exception as e:
                    print(f"❌ WAIC 計算失敗: {e}")
                    return False
            else:
                print("❌ log_likelihood 仍然缺失")
                return False
        else:
            print("❌ 採樣失敗")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False
def manual_model_comparison(models):
    """手動模型比較當 ArviZ 失敗時"""
    
    results = {}
    
    for name, trace in models.items():
        try:
            # 嘗試手動 WAIC 計算
            if hasattr(trace, 'log_likelihood'):
                ll = trace.log_likelihood.likelihood.values
                
                # 清理異常值
                ll_clean = ll[np.isfinite(ll)]
                
                if len(ll_clean) > 0:
                    # 簡化的 WAIC 近似
                    lppd = np.sum(np.log(np.mean(np.exp(ll_clean), axis=0)))
                    p_waic = np.sum(np.var(ll_clean, axis=0))
                    waic = -2 * (lppd - p_waic)
                    
                    results[name] = {
                        'waic': waic,
                        'lppd': lppd,
                        'p_waic': p_waic,
                        'valid': True
                    }
                else:
                    results[name] = {'valid': False, 'reason': 'No valid log_likelihood'}
            else:
                results[name] = {'valid': False, 'reason': 'No log_likelihood'}
                
        except Exception as e:
            results[name] = {'valid': False, 'reason': str(e)}
    
    # 找到最佳模型（最低 WAIC）
    valid_results = {k: v for k, v in results.items() if v.get('valid', False)}
    
    if valid_results:
        winner = min(valid_results, key=lambda k: valid_results[k]['waic'])
        return {
            'winner': winner,
            'method': 'Manual_WAIC',
            'results': results,
            'success': True
        }
    else:
        return {
            'winner': list(models.keys())[0],
            'method': 'Default',
            'results': results,
            'success': False
        }
    
if __name__ == "__main__":
    print("🔧 LBA log_likelihood 修復工具")
    
    # 運行測試
    success = test_log_likelihood_fix()
    if success:
        print("\n✅ log_likelihood 修復測試通過！")
        print("現在可以在 lba_models.py 中使用 create_model_by_name_fixed")
    else:
        print("\n❌ log_likelihood 修復測試失敗")