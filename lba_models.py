# -*- coding: utf-8 -*-
"""
LBA Models Module - Based on HDDM and original papers (FIXED & COMPATIBLE)
lba_models.py
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np

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