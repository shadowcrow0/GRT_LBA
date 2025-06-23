# 修正 MCMC 收斂問題的配置和建議

"""
當前問題分析：
1. R̂_max = 2.254 >> 1.05 (收斂失敗)
2. ESS_min = 3 << 100 (有效樣本數不足)
3. 17 個發散樣本 (模型參數化問題)
4. 達到最大樹深度 (幾何問題)
"""

import numpy as np
import pymc as pm

# ==========================================
# 1. 改進的 MCMC 配置
# ==========================================

def get_robust_mcmc_config():
    """
    獲取更穩健的 MCMC 配置
    """
    return {
        'draws': 1000,           # 增加樣本數
        'tune': 1000,            # 增加調優步數
        'chains': 4,             # 增加鏈數 (建議至少4條)
        'cores': 1,              # 序列採樣避免並行問題
        'target_accept': 0.95,   # 提高接受率 (從0.85→0.95)
        'max_treedepth': 12,     # 增加最大樹深度 (從8→12)
        'random_seed': [42, 43, 44, 45],  # 每條鏈不同種子
        'progressbar': True,
        'return_inferencedata': True,
        'init': 'adapt_diag',    # 更好的初始化
        'nuts_sampler': 'numpyro'  # 使用 NumPyro 後端 (更穩定)
    }

def get_conservative_mcmc_config():
    """
    獲取保守的 MCMC 配置 (用於困難模型)
    """
    return {
        'draws': 500,
        'tune': 1500,            # 更長的調優期
        'chains': 4,
        'cores': 1,
        'target_accept': 0.99,   # 極高接受率
        'max_treedepth': 15,     # 更深的樹
        'random_seed': [42, 43, 44, 45],
        'progressbar': True,
        'return_inferencedata': True,
        'init': 'jitter+adapt_diag',  # 抖動初始化
        'step_scale': 0.25       # 更小的步長
    }

# ==========================================
# 2. 模型重參數化建議
# ==========================================

def get_improved_priors():
    """
    改進的先驗分布 (更保守，幫助收斂)
    """
    
    improved_priors = {
        # 更保守的漂移率先驗
        'drift_priors': {
            'alpha_correct': 2.0,     # 降低 (原本2.5)
            'beta_correct': 2.0,      # 提高 (原本1.5)
            'alpha_incorrect': 1.5,   # 降低 (原本2.0)
            'beta_incorrect': 4.0,    # 提高 (原本3.0)
        },
        
        # 更保守的閾值先驗
        'threshold_priors': {
            'alpha': 2.0,            # 降低 (原本3.0)
            'beta': 2.0,             # 降低 (原本3.5)
        },
        
        # 更緊的變異性先驗
        'start_var_priors': {
            'lower': 0.05,           # 提高下限 (原本0.1)
            'upper': 0.5,            # 降低上限 (原本0.7)
        },
        
        # 更緊的非決策時間先驗
        'ndt_priors': {
            'lower': 0.05,
            'upper': 0.3,            # 降低上限 (原本0.4)
        }
    }
    
    return improved_priors

def apply_parameter_centering(params):
    """
    參數中心化 (幫助收斂)
    """
    
    centered_params = {}
    
    # 中心化變換
    for key, param in params.items():
        if 'drift' in key:
            # 對數變換漂移率
            centered_params[f'{key}_log'] = pm.Normal(f'{key}_log', mu=0, sigma=1)
            centered_params[key] = pm.Deterministic(key, pm.math.exp(centered_params[f'{key}_log']))
        elif 'threshold' in key:
            # 對數變換閾值
            centered_params[f'{key}_log'] = pm.Normal(f'{key}_log', mu=0, sigma=0.5)
            centered_params[key] = pm.Deterministic(key, pm.math.exp(centered_params[f'{key}_log']) + 0.1)
        else:
            centered_params[key] = param
    
    return centered_params

# ==========================================
# 3. 模型診斷和修復
# ==========================================

def diagnose_sampling_issues(trace, verbose=True):
    """
    診斷採樣問題
    """
    
    issues = []
    
    try:
        # 1. 檢查發散樣本
        if hasattr(trace, 'sample_stats'):
            divergences = trace.sample_stats.diverging.sum().values
            if divergences > 0:
                issues.append(f"發散樣本: {divergences}")
        
        # 2. 檢查能量問題
        if hasattr(trace, 'sample_stats') and 'energy' in trace.sample_stats:
            energy = trace.sample_stats.energy.values
            if np.var(energy) < 0.1:
                issues.append("能量變異性過低")
        
        # 3. 檢查 R-hat
        rhat = az.rhat(trace)
        max_rhat = float(rhat.to_array().max())
        if max_rhat > 1.1:
            issues.append(f"R-hat 過高: {max_rhat:.3f}")
        
        # 4. 檢查有效樣本數
        ess = az.ess(trace)
        min_ess = float(ess.to_array().min())
        if min_ess < 400:  # 對於複雜模型需要更多樣本
            issues.append(f"ESS 過低: {min_ess:.0f}")
        
        if verbose:
            if issues:
                print("⚠️ 發現採樣問題:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ 採樣診斷通過")
        
        return issues
        
    except Exception as e:
        if verbose:
            print(f"❌ 診斷失敗: {e}")
        return [f"診斷失敗: {e}"]

def suggest_fixes(issues):
    """
    根據問題提供修復建議
    """
    
    suggestions = []
    
    for issue in issues:
        if "發散" in issue:
            suggestions.extend([
                "提高 target_accept 到 0.95+",
                "增加 tune 步數到 1500+",
                "檢查先驗分布是否合理",
                "考慮重參數化模型"
            ])
        
        if "R-hat" in issue:
            suggestions.extend([
                "增加採樣鏈數到 4+",
                "增加 draws 到 1000+",
                "檢查初始值設定",
                "使用不同的隨機種子"
            ])
        
        if "ESS" in issue:
            suggestions.extend([
                "增加總採樣數",
                "檢查參數相關性",
                "考慮層級模型結構"
            ])
        
        if "能量" in issue:
            suggestions.extend([
                "增加 max_treedepth",
                "調整步長參數",
                "檢查後驗幾何結構"
            ])
    
    # 去重
    unique_suggestions = list(set(suggestions))
    
    print("\n💡 修復建議:")
    for i, suggestion in enumerate(unique_suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    return unique_suggestions

# ==========================================
# 4. 簡化模型用於測試
# ==========================================

def create_simplified_test_model(subject_data):
    """
    創建簡化的測試模型 (用於驗證數據和基本設定)
    """
    
    with pm.Model() as simple_model:
        
        # 僅包含最基本的參數
        drift_0 = pm.Gamma('drift_0', alpha=2, beta=2)
        drift_1 = pm.Gamma('drift_1', alpha=2, beta=2)
        drift_2 = pm.Gamma('drift_2', alpha=2, beta=2)
        drift_3 = pm.Gamma('drift_3', alpha=2, beta=2)
        
        threshold = pm.Gamma('threshold', alpha=2, beta=2)
        ndt = pm.Uniform('ndt', lower=0.1, upper=0.3)
        
        # 簡化的似然計算
        choices = subject_data['choices']
        rt = subject_data['rt']
        
        # 基本的選擇概率模型
        drifts = pm.math.stack([drift_0, drift_1, drift_2, drift_3])
        
        # 軟最大選擇概率
        choice_probs = pm.math.softmax(drifts)
        
        # 觀察模型
        pm.Categorical('observed_choices', p=choice_probs, observed=choices)
        
        # 簡化的 RT 模型
        expected_rt = ndt + threshold / drifts[choices]
        pm.Normal('observed_rt', mu=expected_rt, sigma=0.1, observed=rt)
    
    return simple_model

# ==========================================
# 使用建議
# ==========================================

def run_improved_sampling(model, config_type='robust'):
    """
    使用改進配置運行採樣
    """
    
    if config_type == 'robust':
        config = get_robust_mcmc_config()
    elif config_type == 'conservative':
        config = get_conservative_mcmc_config()
    else:
        raise ValueError("config_type must be 'robust' or 'conservative'")
    
    print(f"🎲 使用 {config_type} 配置進行採樣...")
    print(f"   Target accept: {config['target_accept']}")
    print(f"   Max treedepth: {config['max_treedepth']}")
    print(f"   Chains: {config['chains']}")
    
    with model:
        try:
            trace = pm.sample(**config)
            
            # 診斷採樣結果
            issues = diagnose_sampling_issues(trace)
            
            if issues:
                suggest_fixes(issues)
                return trace, False  # 採樣有問題
            else:
                print("✅ 採樣成功完成!")
                return trace, True   # 採樣成功
                
        except Exception as e:
            print(f"❌ 採樣失敗: {e}")
            return None, False

if __name__ == "__main__":
    print("MCMC 收斂問題診斷和修復建議")
    print("="*50)
    
    print("\n推薦的修復步驟:")
    print("1. 使用 get_robust_mcmc_config() 或 get_conservative_mcmc_config()")
    print("2. 檢查並改進先驗分布")
    print("3. 考慮模型重參數化") 
    print("4. 如果仍有問題，先測試簡化模型")
    print("5. 逐步增加模型複雜度")
    
    print("\n具體的配置建議:")
    robust_config = get_robust_mcmc_config()
    for key, value in robust_config.items():
        print(f"   {key}: {value}")
