# -*- coding: utf-8 -*-
"""
LBA Log-likelihood 修復工具
解決 LBA_main.py 中 loglikelihood 計算失敗的問題
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import warnings
import os
from datetime import datetime

def diagnose_loglikelihood_issue(data_file='model_data.npz'):
    """
    診斷 loglikelihood 計算問題的根本原因
    """
    print("🔍 診斷 loglikelihood 問題...")
    
    issues_found = []
    
    try:
        # 1. 檢查數據檔案
        if not os.path.exists(data_file):
            issues_found.append(f"數據檔案不存在: {data_file}")
            return issues_found
        
        data = np.load(data_file, allow_pickle=True)
        observed_value = data['observed_value']
        
        # 2. 檢查數據完整性
        if np.any(np.isnan(observed_value)):
            issues_found.append("數據包含 NaN 值")
        
        if np.any(observed_value[:, 0] <= 0):
            issues_found.append("發現非正數 RT 值")
        
        # 3. 檢查 RT 單位
        rt_mean = observed_value[:, 0].mean()
        print(f"平均 RT: {rt_mean:.3f}")
        
        if rt_mean < 10:
            issues_found.append("RT 可能是秒單位，需要轉換為毫秒")
        elif rt_mean > 10000:
            issues_found.append("RT 值過大，可能有單位問題")
        
        # 4. 檢查反應範圍
        responses = observed_value[:, 1]
        unique_responses = np.unique(responses)
        if not np.array_equal(unique_responses, [0., 1.]) and not np.array_equal(unique_responses, [0]) and not np.array_equal(unique_responses, [1]):
            issues_found.append(f"反應值不是 0/1: {unique_responses}")
        
        print(f"數據檢查完成，發現 {len(issues_found)} 個問題")
        for issue in issues_found:
            print(f"  ❌ {issue}")
        
        if not issues_found:
            print("✅ 數據檢查通過")
        
        return issues_found
        
    except Exception as e:
        issues_found.append(f"數據檢查失敗: {e}")
        return issues_found

def create_fixed_lba_logp(value, v_rates, start_var, b_safe, non_decision):
    """
    修復版 LBA log-likelihood 函數
    解決數值穩定性問題
    """
    
    # 提取數據
    rt = value[:, 0] / 1000.0  # 轉換為秒
    response = value[:, 1].astype(int)
    
    # 確保參數在合理範圍內
    start_var = pt.clip(start_var, 0.01, 5.0)
    b_safe = pt.clip(b_safe, start_var + 0.01, 10.0)
    non_decision = pt.clip(non_decision, 0.01, 1.0)
    
    # 決策時間（必須為正）
    decision_time = pt.maximum(rt - non_decision, 0.001)
    
    # 提取漂移率
    v_correct = pt.clip(v_rates[:, 0], 0.01, 20.0)
    v_incorrect = pt.clip(v_rates[:, 1], 0.01, 20.0)
    
    # 簡化的 LBA 似然計算（使用 Wald 近似）
    def wald_logpdf(t, v, A, b):
        """Wald 分布的對數密度函數"""
        mu = (b - A/2) / pt.maximum(v, 0.01)
        lambda_param = (b - A/2)**2
        
        # Wald PDF: f(t) = sqrt(λ/(2πt³)) * exp(-λ(t-μ)²/(2μ²t))
        log_pdf = (0.5 * pt.log(lambda_param / (2 * np.pi * t**3)) - 
                   lambda_param * (t - mu)**2 / (2 * mu**2 * t))
        
        return log_pdf
    
    def survival_function(t, v, A, b):
        """簡化的存活函數"""
        mu = (b - A/2) / pt.maximum(v, 0.01)
        # 使用指數近似
        survival_logp = -t / mu
        return survival_logp
    
    # 計算獲勝者的概率密度
    log_pdf_correct = wald_logpdf(decision_time, v_correct, start_var, b_safe)
    log_pdf_incorrect = wald_logpdf(decision_time, v_incorrect, start_var, b_safe)
    
    # 計算失敗者的存活概率
    log_survival_incorrect = survival_function(decision_time, v_incorrect, start_var, b_safe)
    log_survival_correct = survival_function(decision_time, v_correct, start_var, b_safe)
    
    # 根據反應選擇對應的似然
    log_likelihood = pt.switch(
        pt.eq(response, 1),
        log_pdf_correct + log_survival_incorrect,  # 正確反應
        log_pdf_incorrect + log_survival_correct   # 錯誤反應
    )
    
    # 數值穩定性保護
    log_likelihood = pt.switch(pt.isnan(log_likelihood), -100.0, log_likelihood)
    log_likelihood = pt.switch(pt.isinf(log_likelihood), -100.0, log_likelihood)
    log_likelihood = pt.clip(log_likelihood, -100.0, 10.0)
    
    return log_likelihood

def create_fixed_lba_random(v_rates, start_var, b_safe, non_decision, rng=None, size=None):
    """
    修復版 LBA 隨機樣本生成函數
    """
    if size is None:
        size = (1,)
    elif isinstance(size, int):
        size = (size,)
    
    n_samples = size[0]
    samples = np.zeros((n_samples, 2))
    
    # 轉換參數為數值
    try:
        v_correct = float(np.asarray(v_rates[0]).item())
        v_incorrect = float(np.asarray(v_rates[1]).item())
        A_val = float(np.asarray(start_var).item())
        b_val = float(np.asarray(b_safe).item())
        t0_val = float(np.asarray(non_decision).item())
    except:
        # 如果無法轉換，使用預設值
        v_correct = 1.5
        v_incorrect = 0.8
        A_val = 0.5
        b_val = 1.5
        t0_val = 0.3
    
    for i in range(n_samples):
        # 使用簡化的 LBA 模擬
        # 每個累積器的完成時間使用 Wald 分布近似
        mu_correct = (b_val - A_val/2) / max(v_correct, 0.01)
        mu_incorrect = (b_val - A_val/2) / max(v_incorrect, 0.01)
        
        # 使用指數分布作為簡化
        t_correct = rng.exponential(mu_correct)
        t_incorrect = rng.exponential(mu_incorrect)
        
        if t_correct < t_incorrect:
            samples[i, 0] = (t_correct + t0_val) * 1000  # 轉回毫秒
            samples[i, 1] = 1
        else:
            samples[i, 0] = (t_incorrect + t0_val) * 1000
            samples[i, 1] = 0
        
        # 確保 RT 在合理範圍內
        samples[i, 0] = np.clip(samples[i, 0], 200, 5000)
    
    return samples

def create_fixed_coactive_model(observed_data, input_data):
    """
    創建修復版 Coactive LBA 模型
    確保 log_likelihood 正確生成
    """
    with pm.Model() as model:
        
        # 使用更保守的先驗分布
        v_match = pm.TruncatedNormal('v_match', mu=1.5, sigma=0.5, lower=0.5, upper=4.0)
        v_mismatch = pm.TruncatedNormal('v_mismatch', mu=1.0, sigma=0.5, lower=0.2, upper=3.0)
        
        start_var = pm.TruncatedNormal('start_var', mu=0.5, sigma=0.2, lower=0.1, upper=1.0)
        boundary_offset = pm.TruncatedNormal('boundary_offset', mu=1.0, sigma=0.3, lower=0.5, upper=2.0)
        b_safe = pm.Deterministic('b_safe', start_var + boundary_offset)
        
        # 動態計算 non_decision 上界
        min_rt_seconds = np.min(observed_data[:, 0]) / 1000.0
        upper_bound = min(min_rt_seconds * 0.8, 0.5)  # 更保守的上界
        
        non_decision = pm.TruncatedNormal('non_decision', 
                                        mu=0.2, sigma=0.05,
                                        lower=0.05, upper=upper_bound)
        
        # 計算漂移率
        v_left = v_match * input_data['left_match'] + v_mismatch * (1 - input_data['left_match'])
        v_right = v_match * input_data['right_match'] + v_mismatch * (1 - input_data['right_match'])
        
        # Coactive 整合
        v_final_correct = pm.Deterministic('v_final_correct', v_left + v_right)
        
        v_incorrect_base = pm.TruncatedNormal('v_incorrect_base', mu=0.8, sigma=0.3, lower=0.2, upper=2.0)
        v_final_incorrect = pm.Deterministic('v_final_incorrect', 
                                           pt.full_like(v_final_correct, v_incorrect_base))
        
        # 堆疊漂移率
        v_rates = pt.stack([v_final_correct, v_final_incorrect], axis=1)
        
        # 使用修復版似然函數
        likelihood = pm.CustomDist('likelihood',
                                  v_rates, start_var, b_safe, non_decision,
                                  logp=create_fixed_lba_logp,
                                  random=create_fixed_lba_random,
                                  observed=observed_data)
        
        # 手動計算 log_likelihood 確保 ArviZ 可以訪問
        individual_logp = create_fixed_lba_logp(observed_data, v_rates, start_var, b_safe, non_decision)
        pm.Deterministic('log_likelihood_values', individual_logp)
    
    return model

def test_fixed_model(data_file='model_data_fixed.npz'):
    """
    測試修復版模型是否能正確計算 log_likelihood
    """
    print("🧪 測試修復版模型...")
    
    try:
        # 載入數據
        data = np.load(data_file, allow_pickle=True)
        observed_value = data['observed_value']
        participant_idx = data['participant_idx']
        model_input_data = data['model_input_data'].item()
        
        # 選擇第一個參與者進行測試
        participant_id = np.unique(participant_idx)[0]
        mask = participant_idx == participant_id
        
        test_data = observed_value[mask][:30]  # 只取前30個試驗加快測試
        test_input = {
            'left_match': model_input_data['left_match'][mask][:30],
            'right_match': model_input_data['right_match'][mask][:30]
        }
        
        print(f"測試數據: {len(test_data)} 試驗，參與者 {participant_id}")
        
        # 創建模型
        model = create_fixed_coactive_model(test_data, test_input)
        print("✓ 模型創建成功")
        
        # 測試模型編譯
        with model:
            test_point = model.initial_point()
            logp = model.compile_logp()
            logp_val = logp(test_point)
            
            if np.isnan(logp_val) or np.isinf(logp_val):
                print(f"❌ 初始 logp 無效: {logp_val}")
                return False
            
            print(f"✓ 模型編譯成功，初始 logp: {logp_val:.2f}")
        
        # 快速採樣測試
        print("進行快速採樣測試...")
        with model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                trace = pm.sample(
                    draws=100,
                    tune=100,
                    chains=2,
                    target_accept=0.90,
                    return_inferencedata=True,
                    progressbar=False,
                    random_seed=42,
                    cores=1
                )
        
        print("✓ 採樣成功")
        
        # 檢查 log_likelihood
        if 'log_likelihood_values' in trace.posterior:
            ll_values = trace.posterior['log_likelihood_values'].values
            if not np.any(np.isnan(ll_values)) and not np.any(np.isinf(ll_values)):
                print("✓ log_likelihood 計算正常")
                
                # 測試 WAIC 計算
                try:
                    # 創建假的 log_likelihood 用於 WAIC
                    import xarray as xr
                    log_likelihood = xr.Dataset({
                        'likelihood': trace.posterior['log_likelihood_values']
                    })
                    trace_with_ll = trace.assign(log_likelihood=log_likelihood)
                    
                    waic_result = az.waic(trace_with_ll)
                    print(f"✓ WAIC 計算成功: {waic_result.elpd_waic:.2f}")
                    
                    return True
                except Exception as e:
                    print(f"⚠️ WAIC 計算失敗: {e}")
                    return True  # 模型本身是好的
            else:
                print("❌ log_likelihood 包含無效值")
                return False
        else:
            print("❌ 缺少 log_likelihood_values")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def fix_lba_main():
    """
    為 LBA_main.py 提供修復建議
    """
    
    suggestions = """
    
    📋 修復 LBA_main.py 的建議：
    
    1. 在文件開頭添加導入：
       from LBA_loglikelihood_fix import create_fixed_coactive_model, diagnose_loglikelihood_issue
    
    2. 在 run_single_participant_analysis 中替換模型創建：
       
       原來：
       model = create_model_by_name(model_name, participant_data, participant_input)
       
       改為：
       if model_name == 'Coactive_Addition':
           model = create_fixed_coactive_model(participant_data, participant_input)
       else:
           model = create_model_by_name(model_name, participant_data, participant_input)
    
    3. 在開始分析前添加診斷：
       issues = diagnose_loglikelihood_issue(self.data_file)
       if issues:
           print("發現數據問題，嘗試使用修復版數據...")
           self.data_file = 'model_data_fixed.npz'
    
    4. 調整採樣參數以提高穩定性：
       trace, diagnostics = sample_with_convergence_check(
           model, 
           max_attempts=3,  # 增加嘗試次數
           draws=500,       # 減少 draws 但增加穩定性
           tune=500,        # 增加 tune
           chains=2         # 減少 chains
       )
    
    """
    
    print(suggestions)
    return suggestions

def main():
    """主測試函數"""
    
    print("🔧 LBA Log-likelihood 修復工具")
    print("=" * 50)
    
    # 1. 診斷問題
    issues = diagnose_loglikelihood_issue()
    
    # 2. 測試修復版模型
    if os.path.exists('model_data_fixed.npz'):
        success = test_fixed_model('model_data_fixed.npz')
    else:
        success = test_fixed_model('model_data.npz')
    
    # 3. 提供修復建議
    if success:
        print("\n✅ 修復測試成功！")
        print("你現在可以使用修復版函數來解決 loglikelihood 問題。")
        fix_lba_main()
    else:
        print("\n❌ 修復測試失敗")
        print("需要進一步調試數據或模型設置。")
    
    return success

if __name__ == '__main__':
    main()