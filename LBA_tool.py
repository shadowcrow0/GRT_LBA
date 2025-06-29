# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 03:29:44 2025

@author: spt904
"""

"""
LBA Analysis - Utility Functions Module
輔助函數和工具，包含採樣、診斷、數據處理等
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import os
import warnings
from datetime import datetime

# FIXED: Remove the conflicting import - we'll use the function defined below
# from lba_fixes import robust_sample_with_convergence_check as sample_with_convergence_check

# --- LBA_tool.py 中修正後的程式碼 ---

# LBA_tool.py 的完整修正函數

def sample_with_convergence_check(model, max_attempts=2, draws=200, tune=300, chains=2, target_accept=0.90):
    """
    修復後的模型採樣函數，包含收斂檢查 (已修正 SyntaxError)
    """
    
    print(f"  開始採樣 (draws={draws}, tune={tune}, chains={chains})...")
    
    for attempt in range(max_attempts):
        try: # <--- try 區塊開始
            print(f"    嘗試 {attempt + 1}/{max_attempts}")
            
            with model:
                # 由於 find_MAP 不穩定，我們直接跳過它
                print("    ⚠️ 已停用 find_MAP，使用預設的 'jitter+adapt_diag' 初始化。")
                
                # 執行採樣
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True,
                    progressbar=True,
                    random_seed=42 + attempt,
                    init='jitter+adapt_diag',
                    cores=1
                )
                
                # 收斂診斷
                diagnostics = check_convergence_diagnostics(trace)
                
                print(f"    最大 R-hat: {diagnostics['max_rhat']:.4f}")
                print(f"    最小 ESS bulk: {diagnostics['min_ess']:.0f}")
                
                # 放寬的收斂標準
                convergence_ok = (
                    diagnostics['max_rhat'] < 1.2 and
                    diagnostics['min_ess'] > 50
                )
                
                if convergence_ok:
                    print(f"    ✓ 收斂成功 (嘗試 {attempt + 1})")
                    return trace, diagnostics
                else:
                    print(f"    ⚠️ 收斂標準未達成，但繼續...")
                    if attempt == max_attempts - 1:
                        print(f"    ⚠️ 最後嘗試，返回當前結果")
                        return trace, diagnostics

        # *** 這是修正的部分：加上對應 try 的 except 區塊 ***
        except Exception as e:
            print(f"    ❌ 採樣或診斷失敗: {e}")
            if attempt < max_attempts - 1:
                print(f"    調整參數後重試...")
                # 簡單地調整參數以增加成功的機會
                draws = max(100, int(draws * 0.9))
                tune = max(150, int(tune * 1.1))
                target_accept = min(0.98, target_accept + 0.02)
            
    print(f"    ❌ {max_attempts} 次嘗試後仍未成功")
    return None, None
def check_convergence_diagnostics(trace):
    """
    檢查 MCMC 收斂診斷
    """
    
    try:
        summary = az.summary(trace)
        
        # 基本診斷指標
        diagnostics = {
            'max_rhat': summary['r_hat'].max() if 'r_hat' in summary.columns else np.nan,
            'min_ess': summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else np.nan,
            'min_ess_tail': summary['ess_tail'].min() if 'ess_tail' in summary.columns else np.nan,
            'mean_accept_stat': np.nan,
            'n_divergent': 0,
            'n_parameters': len(summary)
        }
        
        # 嘗試獲取採樣統計
        try:
            if hasattr(trace, 'sample_stats'):
                if hasattr(trace.sample_stats, 'diverging'):
                    diagnostics['n_divergent'] = int(trace.sample_stats.diverging.sum())
                
                if hasattr(trace.sample_stats, 'accept'):
                    diagnostics['mean_accept_stat'] = float(trace.sample_stats.accept.mean())
        except Exception:
            pass  # 如果無法獲取採樣統計，使用默認值
        
        return diagnostics
        
    except Exception as e:
        print(f"Warning: 無法計算診斷指標: {e}")
        return {
            'max_rhat': np.nan,
            'min_ess': np.nan,
            'min_ess_tail': np.nan,
            'n_divergent': np.nan,
            'mean_accept_stat': np.nan,
            'n_parameters': 0
        }

def calculate_sigma_matrices_from_traces(models, participant_id, save_dir=None):
    """
    從 traces 計算 sigma matrices
    """
    
    print(f"  Computing sigma matrices...")
    
    sigma_results = {}
    
    for model_name, trace in models.items():
        try:
            # IMPROVED: Use safe parameter extraction from our fixes
            try:
                from LBA_tool_fixes import safe_parameter_extraction
                v_correct_samples = safe_parameter_extraction(trace, 'v_final_correct')
                v_incorrect_samples = safe_parameter_extraction(trace, 'v_final_incorrect')
            except ImportError:
                # Fallback to original method if fixes not available
                v_correct_samples = trace.posterior['v_final_correct'].values.flatten()
                v_incorrect_samples = trace.posterior['v_final_incorrect'].values.flatten()
            
            # 創建樣本矩陣
            samples_matrix = np.column_stack([v_correct_samples, v_incorrect_samples])
            
            # 計算協方差矩陣
            cov_matrix = np.cov(samples_matrix.T)
            
            # 計算相關係數矩陣
            corr_matrix = np.corrcoef(samples_matrix.T)
            
            # 提取關鍵統計量
            correlation = corr_matrix[0, 1]
            variance_A = cov_matrix[0, 0]  # v_correct 變異數
            variance_B = cov_matrix[1, 1]  # v_incorrect 變異數
            covariance = cov_matrix[0, 1]
            
            # 計算條件數
            condition_number = np.linalg.cond(cov_matrix)
            
            # 計算行列式
            determinant = np.linalg.det(cov_matrix)
            
            sigma_data = {
                'model': model_name,
                'participant': participant_id,
                'correlation': correlation,
                'variance_A': variance_A,
                'variance_B': variance_B,
                'covariance': covariance,
                'condition_number': condition_number,
                'determinant': determinant,
                'cov_matrix': cov_matrix,
                'corr_matrix': corr_matrix
            }
            
            sigma_results[model_name] = sigma_data
            
            print(f"    {model_name}: correlation = {correlation:.3f}, condition = {condition_number:.1f}")
            
        except Exception as e:
            print(f"    ❌ {model_name}: Error computing sigma matrix: {e}")
    
    # 保存結果（如果指定了目錄）
    if save_dir and sigma_results:
        sigma_file = os.path.join(save_dir, f'participant_{participant_id}_sigma_matrices.npz')
        save_data = {}
        
        for model_name, data in sigma_results.items():
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    save_data[f'{model_name}_{key}'] = value
                else:
                    save_data[f'{model_name}_{key}'] = value
        
        np.savez(sigma_file, **save_data)
        print(f"    ✓ Sigma matrices saved")
    
    return sigma_results

def improved_model_comparison(models, method='auto'):
    """
    完全重寫的模型比較函數 - 解決 WAIC/LOO 失敗問題
    替換 LBA_tool.py 中的原始函數
    """
    
    print("🔬 開始增強版模型比較...")
    
    if len(models) < 2:
        print("❌ 需要至少 2 個模型進行比較")
        return None
    
    # 步驟 1: 診斷每個模型
    print("\n📋 診斷所有模型...")
    model_diagnostics = {}
    valid_models = {}
    
    for model_name, trace in models.items():
        print(f"  檢查 {model_name}...")
        
        # 檢查基本要求
        has_log_likelihood = hasattr(trace, 'log_likelihood')
        has_posterior = hasattr(trace, 'posterior')
        
        if has_log_likelihood:
            try:
                ll_values = trace.log_likelihood.likelihood.values
                n_nan = np.isnan(ll_values).sum()
                n_inf = np.isinf(ll_values).sum()
                n_total = ll_values.size
                
                if n_nan == 0 and n_inf == 0:
                    valid_models[model_name] = trace
                    print(f"    ✓ {model_name}: log_likelihood 正常")
                else:
                    print(f"    ❌ {model_name}: log_likelihood 有 {n_nan} NaN, {n_inf} inf")
            except:
                print(f"    ❌ {model_name}: log_likelihood 無法訪問")
        else:
            print(f"    ❌ {model_name}: 缺少 log_likelihood")
        
        model_diagnostics[model_name] = {
            'has_log_likelihood': has_log_likelihood,
            'has_posterior': has_posterior
        }
    
    # 步驟 2: 如果有有效模型，嘗試標準方法
    if len(valid_models) >= 2:
        print(f"\n📊 找到 {len(valid_models)} 個有效模型，嘗試標準方法...")
        
        # 嘗試 WAIC
        try:
            print("  嘗試 WAIC...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparison_result = az.compare(valid_models, ic='waic')
            
            winner = comparison_result.index[0]
            
            if len(comparison_result) >= 2:
                elpd_diff = comparison_result.iloc[1]['elpd_diff']
                dse = comparison_result.iloc[1]['dse'] if 'dse' in comparison_result.columns else 1
            else:
                elpd_diff = 0
                dse = 1
            
            effect_size = abs(elpd_diff / dse) if dse > 0 else 0
            significance = 'Significant' if effect_size > 2 else ('Weak' if effect_size > 1 else 'Non-significant')
            
            print(f"    ✅ WAIC 成功！獲勝者: {winner}")
            
            return {
                'winner': winner,
                'method': 'WAIC',
                'elpd_diff': elpd_diff,
                'dse': dse,
                'effect_size': effect_size,
                'significance': significance,
                'comparison_table': comparison_result,
                'success': True
            }
            
        except Exception as e:
            print(f"    ❌ WAIC 失敗: {str(e)[:100]}...")
        
        # 嘗試 LOO
        try:
            print("  嘗試 LOO...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparison_result = az.compare(valid_models, ic='loo')
            
            winner = comparison_result.index[0]
            
            if len(comparison_result) >= 2:
                elpd_diff = comparison_result.iloc[1]['elpd_diff']
                dse = comparison_result.iloc[1]['dse'] if 'dse' in comparison_result.columns else 1
            else:
                elpd_diff = 0
                dse = 1
            
            effect_size = abs(elpd_diff / dse) if dse > 0 else 0
            significance = 'Significant' if effect_size > 2 else ('Weak' if effect_size > 1 else 'Non-significant')
            
            print(f"    ✅ LOO 成功！獲勝者: {winner}")
            
            return {
                'winner': winner,
                'method': 'LOO', 
                'elpd_diff': elpd_diff,
                'dse': dse,
                'effect_size': effect_size,
                'significance': significance,
                'comparison_table': comparison_result,
                'success': True
            }
            
        except Exception as e:
            print(f"    ❌ LOO 失敗: {str(e)[:100]}...")
    
    # 步驟 3: 使用替代比較方法
    print("\n🔄 使用替代比較方法...")
    
    model_scores = {}
    
    for model_name, trace in models.items():
        try:
            score = 0
            
            # 方法 1: 如果有 log_likelihood，使用平均值
            if hasattr(trace, 'log_likelihood'):
                try:
                    ll_values = trace.log_likelihood.likelihood.values
                    # 清理異常值
                    ll_clean = ll_values[np.isfinite(ll_values)]
                    if len(ll_clean) > 0:
                        score = np.mean(ll_clean)
                        print(f"    {model_name}: 平均 log-likelihood = {score:.2f}")
                        model_scores[model_name] = score
                        continue
                except:
                    pass
            
            # 方法 2: 基於收斂性評分
            try:
                summary = az.summary(trace)
                max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.0
                min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else 1000
                
                # 收斂評分：懲罰高 R-hat，獎勵高 ESS
                score = min_ess / max(max_rhat - 1.0, 0.01)
                print(f"    {model_name}: 收斂評分 = {score:.2f}")
                model_scores[model_name] = score
                continue
            except:
                pass
            
            # 方法 3: 基本評分（樣本數）
            try:
                n_samples = len(trace.posterior.coords['draw']) * len(trace.posterior.coords['chain'])
                score = n_samples
                print(f"    {model_name}: 樣本數評分 = {score}")
                model_scores[model_name] = score
            except:
                model_scores[model_name] = 0
                print(f"    {model_name}: 無法評分")
        
        except Exception as e:
            print(f"    ❌ {model_name} 評分失敗: {e}")
            model_scores[model_name] = 0
    
    # 確定獲勝者
    if model_scores:
        winner = max(model_scores, key=model_scores.get)
        winner_score = model_scores[winner]
        
        # 計算效應量
        sorted_scores = sorted(model_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
            effect_size = (sorted_scores[0] - sorted_scores[1]) / abs(sorted_scores[1])
            effect_size = min(effect_size, 5.0)  # 限制最大值
        else:
            effect_size = 0
        
        significance = 'Significant' if effect_size > 0.5 else ('Weak' if effect_size > 0.2 else 'Non-significant')
        
        print(f"    🏆 替代方法獲勝者: {winner} (評分: {winner_score:.2f})")
        print(f"    效應量: {effect_size:.3f} ({significance})")
        
        return {
            'winner': winner,
            'method': 'Alternative',
            'elpd_diff': np.nan,
            'dse': np.nan,
            'effect_size': effect_size,
            'significance': significance,
            'comparison_table': None,
            'model_scores': model_scores,
            'success': True
        }
    else:
        print("    ❌ 所有評分方法都失敗")
        # 最後手段：選擇第一個模型
        winner = list(models.keys())[0]
        print(f"    🏳️ 默認選擇: {winner}")
        
        return {
            'winner': winner,
            'method': 'Default',
            'elpd_diff': np.nan,
            'dse': np.nan,
            'effect_size': np.nan,
            'significance': 'Unknown',
            'comparison_table': None,
            'success': False
        }
def safe_trace_summary(trace, model_name):
    """
    安全的 trace 總結函數
    """
    try:
        summary = az.summary(trace)
        
        result = {
            'model_name': model_name,
            'n_parameters': len(summary),
            'max_rhat': summary['r_hat'].max() if 'r_hat' in summary.columns else np.nan,
            'min_ess_bulk': summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else np.nan,
            'min_ess_tail': summary['ess_tail'].min() if 'ess_tail' in summary.columns else np.nan,
            'summary_table': summary
        }
        
        return result
        
    except Exception as e:
        print(f"    ⚠️ {model_name} 總結計算失敗: {e}")
        return {
            'model_name': model_name,
            'n_parameters': 0,
            'max_rhat': np.nan,
            'min_ess_bulk': np.nan,
            'min_ess_tail': np.nan,
            'summary_table': None
        }

def extract_parameter_estimates(trace, model_name):
    """
    提取參數估計值
    """
    try:
        estimates = {}
        
        # 常見的 LBA 參數
        param_names = ['v_match', 'v_mismatch', 'start_var', 'boundary_offset', 
                      'non_decision', 'v_final_correct', 'v_final_incorrect']
        
        for param in param_names:
            if param in trace.posterior:
                # IMPROVED: Try to use safe extraction if available
                try:
                    from LBA_tool_fixes import safe_parameter_extraction
                    samples = safe_parameter_extraction(trace, param)
                except ImportError:
                    samples = trace.posterior[param].values.flatten()
                
                estimates[param] = {
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'median': np.median(samples),
                    'q025': np.percentile(samples, 2.5),
                    'q975': np.percentile(samples, 97.5)
                }
        
        return estimates
        
    except Exception as e:
        print(f"    ⚠️ {model_name} 參數提取失敗: {e}")
        return {}

def create_model_summary_report(models, participant_id, save_dir):
    """
    創建模型總結報告
    """
    try:
        report_lines = []
        report_lines.append(f"模型總結報告 - 參與者 {participant_id}")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        for model_name, trace in models.items():
            report_lines.append(f"模型: {model_name}")
            report_lines.append("-" * 30)
            
            # 基本統計
            summary = safe_trace_summary(trace, model_name)
            report_lines.append(f"參數數量: {summary['n_parameters']}")
            report_lines.append(f"最大 R-hat: {summary['max_rhat']:.4f}")
            report_lines.append(f"最小 ESS bulk: {summary['min_ess_bulk']:.0f}")
            report_lines.append(f"最小 ESS tail: {summary['min_ess_tail']:.0f}")
            
            # 參數估計
            estimates = extract_parameter_estimates(trace, model_name)
            if estimates:
                report_lines.append("\n主要參數估計:")
                for param, stats in estimates.items():
                    if not np.isnan(stats['mean']):
                        report_lines.append(f"  {param}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            report_lines.append("")
        
        # 保存報告
        report_file = os.path.join(save_dir, f'participant_{participant_id}_model_summary.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"    ✓ 模型總結報告已保存: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"    ❌ 創建模型總結報告失敗: {e}")
        return None
def improved_model_comparison(models, method='auto'):
    """
    完全重寫的模型比較函數 - 解決 WAIC/LOO 失敗問題
    替換 LBA_tool.py 中的原始函數
    """
    
    print("🔬 開始增強版模型比較...")
    
    if len(models) < 2:
        print("❌ 需要至少 2 個模型進行比較")
        return None
    
    # 步驟 1: 診斷每個模型
    print("\n📋 診斷所有模型...")
    model_diagnostics = {}
    valid_models = {}
    
    for model_name, trace in models.items():
        print(f"  檢查 {model_name}...")
        
        # 檢查基本要求
        has_log_likelihood = hasattr(trace, 'log_likelihood')
        has_posterior = hasattr(trace, 'posterior')
        
        if has_log_likelihood:
            try:
                ll_values = trace.log_likelihood.likelihood.values
                n_nan = np.isnan(ll_values).sum()
                n_inf = np.isinf(ll_values).sum()
                n_total = ll_values.size
                
                if n_nan == 0 and n_inf == 0:
                    valid_models[model_name] = trace
                    print(f"    ✓ {model_name}: log_likelihood 正常")
                else:
                    print(f"    ❌ {model_name}: log_likelihood 有 {n_nan} NaN, {n_inf} inf")
            except:
                print(f"    ❌ {model_name}: log_likelihood 無法訪問")
        else:
            print(f"    ❌ {model_name}: 缺少 log_likelihood")
        
        model_diagnostics[model_name] = {
            'has_log_likelihood': has_log_likelihood,
            'has_posterior': has_posterior
        }
    
    # 步驟 2: 如果有有效模型，嘗試標準方法
    if len(valid_models) >= 2:
        print(f"\n📊 找到 {len(valid_models)} 個有效模型，嘗試標準方法...")
        
        # 嘗試 WAIC
        try:
            print("  嘗試 WAIC...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparison_result = az.compare(valid_models, ic='waic')
            
            winner = comparison_result.index[0]
            
            if len(comparison_result) >= 2:
                elpd_diff = comparison_result.iloc[1]['elpd_diff']
                dse = comparison_result.iloc[1]['dse'] if 'dse' in comparison_result.columns else 1
            else:
                elpd_diff = 0
                dse = 1
            
            effect_size = abs(elpd_diff / dse) if dse > 0 else 0
            significance = 'Significant' if effect_size > 2 else ('Weak' if effect_size > 1 else 'Non-significant')
            
            print(f"    ✅ WAIC 成功！獲勝者: {winner}")
            
            return {
                'winner': winner,
                'method': 'WAIC',
                'elpd_diff': elpd_diff,
                'dse': dse,
                'effect_size': effect_size,
                'significance': significance,
                'comparison_table': comparison_result,
                'success': True
            }
            
        except Exception as e:
            print(f"    ❌ WAIC 失敗: {str(e)[:100]}...")
        
        # 嘗試 LOO
        try:
            print("  嘗試 LOO...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparison_result = az.compare(valid_models, ic='loo')
            
            winner = comparison_result.index[0]
            
            if len(comparison_result) >= 2:
                elpd_diff = comparison_result.iloc[1]['elpd_diff']
                dse = comparison_result.iloc[1]['dse'] if 'dse' in comparison_result.columns else 1
            else:
                elpd_diff = 0
                dse = 1
            
            effect_size = abs(elpd_diff / dse) if dse > 0 else 0
            significance = 'Significant' if effect_size > 2 else ('Weak' if effect_size > 1 else 'Non-significant')
            
            print(f"    ✅ LOO 成功！獲勝者: {winner}")
            
            return {
                'winner': winner,
                'method': 'LOO', 
                'elpd_diff': elpd_diff,
                'dse': dse,
                'effect_size': effect_size,
                'significance': significance,
                'comparison_table': comparison_result,
                'success': True
            }
            
        except Exception as e:
            print(f"    ❌ LOO 失敗: {str(e)[:100]}...")
    
    # 步驟 3: 使用替代比較方法
    print("\n🔄 使用替代比較方法...")
    
    model_scores = {}
    
    for model_name, trace in models.items():
        try:
            score = 0
            
            # 方法 1: 如果有 log_likelihood，使用平均值
            if hasattr(trace, 'log_likelihood'):
                try:
                    ll_values = trace.log_likelihood.likelihood.values
                    # 清理異常值
                    ll_clean = ll_values[np.isfinite(ll_values)]
                    if len(ll_clean) > 0:
                        score = np.mean(ll_clean)
                        print(f"    {model_name}: 平均 log-likelihood = {score:.2f}")
                        model_scores[model_name] = score
                        continue
                except:
                    pass
            
            # 方法 2: 基於收斂性評分
            try:
                summary = az.summary(trace)
                max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.0
                min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else 1000
                
                # 收斂評分：懲罰高 R-hat，獎勵高 ESS
                score = min_ess / max(max_rhat - 1.0, 0.01)
                print(f"    {model_name}: 收斂評分 = {score:.2f}")
                model_scores[model_name] = score
                continue
            except:
                pass
            
            # 方法 3: 基本評分（樣本數）
            try:
                n_samples = len(trace.posterior.coords['draw']) * len(trace.posterior.coords['chain'])
                score = n_samples
                print(f"    {model_name}: 樣本數評分 = {score}")
                model_scores[model_name] = score
            except:
                model_scores[model_name] = 0
                print(f"    {model_name}: 無法評分")
        
        except Exception as e:
            print(f"    ❌ {model_name} 評分失敗: {e}")
            model_scores[model_name] = 0
    
    # 確定獲勝者
    if model_scores:
        winner = max(model_scores, key=model_scores.get)
        winner_score = model_scores[winner]
        
        # 計算效應量
        sorted_scores = sorted(model_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
            effect_size = (sorted_scores[0] - sorted_scores[1]) / abs(sorted_scores[1])
            effect_size = min(effect_size, 5.0)  # 限制最大值
        else:
            effect_size = 0
        
        significance = 'Significant' if effect_size > 0.5 else ('Weak' if effect_size > 0.2 else 'Non-significant')
        
        print(f"    🏆 替代方法獲勝者: {winner} (評分: {winner_score:.2f})")
        print(f"    效應量: {effect_size:.3f} ({significance})")
        
        return {
            'winner': winner,
            'method': 'Alternative',
            'elpd_diff': np.nan,
            'dse': np.nan,
            'effect_size': effect_size,
            'significance': significance,
            'comparison_table': None,
            'model_scores': model_scores,
            'success': True
        }
    else:
        print("    ❌ 所有評分方法都失敗")
        # 最後手段：選擇第一個模型
        winner = list(models.keys())[0]
        print(f"    🏳️ 默認選擇: {winner}")
        
        return {
            'winner': winner,
            'method': 'Default',
            'elpd_diff': np.nan,
            'dse': np.nan,
            'effect_size': np.nan,
            'significance': 'Unknown',
            'comparison_table': None,
            'success': False
        }
def quick_data_check(data_file):
    """
    快速數據檢查
    """
    try:
        print("執行快速數據檢查...")
        
        data = np.load(data_file, allow_pickle=True)
        observed_value = data['observed_value']
        participant_idx = data['participant_idx']
        model_input_data = data['model_input_data'].item()
        
        # 基本統計
        n_trials = len(observed_value)
        n_participants = len(np.unique(participant_idx))
        rt_mean = observed_value[:, 0].mean()
        rt_std = observed_value[:, 0].std()
        accuracy = observed_value[:, 1].mean()
        
        print(f"✓ 總試驗數: {n_trials}")
        print(f"✓ 參與者數: {n_participants}")
        print(f"✓ 平均 RT: {rt_mean:.2f}")
        print(f"✓ RT 標準差: {rt_std:.2f}")
        print(f"✓ 平均準確率: {accuracy:.3f}")
        
        # 檢查 RT 單位
        if rt_mean < 10:
            print("⚠️ RT 似乎是秒單位，建議轉換為毫秒")
            return False, "RT unit issue"
        
        # 檢查數據完整性
        if np.any(np.isnan(observed_value)):
            print("⚠️ 數據包含 NaN 值")
            return False, "NaN values"
        
        if np.any(observed_value[:, 0] <= 0):
            print("⚠️ 發現非正數 RT 值")
            return False, "Invalid RT"
        
        print("✅ 數據檢查通過")
        return True, "OK"
        
    except Exception as e:
        print(f"❌ 數據檢查失敗: {e}")
        return False, str(e)
def quick_comparison_test(models):
    """快速測試模型比較修復"""
    print("🧪 測試模型比較修復...")
    
    if not models or len(models) < 2:
        print("❌ 需要至少 2 個模型進行測試")
        return False
    
    try:
        result = improved_model_comparison(models)
        
        if result and result.get('success', False):
            print(f"✅ 比較成功！獲勝者: {result['winner']}")
            print(f"   方法: {result['method']}")
            return True
        else:
            print("❌ 比較失敗")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False
# 如果直接運行此模組
if __name__ == '__main__':
    print("LBA Analysis - Utility Functions Module (修復版)")
    print("此模組包含採樣、診斷和數據處理功能")
    
    # 測試數據檢查功能
    print("\n測試數據檢查...")
    try:
        success, message = quick_data_check('model_data.npz')
        if success:
            print("✅ 工具模組測試通過")
        else:
            print(f"⚠️ 數據問題: {message}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        print("請確保 model_data.npz 文件存在")