"""
Additional fixes for LBA_tool.py - Model Comparison and Utility Functions
"""

import numpy as np
import pandas as pd
import arviz as az
import warnings
import os

def robust_model_comparison(models, method='auto'):
    """
    FIXED: More robust model comparison function that handles ELPD computation errors.
    """
    
    print("  執行模型比較...")
    
    if len(models) < 2:
        print("    需要至少2個模型進行比較")
        return None
    
    try:
        # First attempt: Use ArviZ compare with WAIC
        print("    嘗試使用 WAIC 進行比較...")
        model_dict = {name: trace for name, trace in models.items()}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison_result = az.compare(model_dict, ic='waic')
        
        if comparison_result is not None and len(comparison_result) > 0:
            # Extract results
            winner = comparison_result.index[0]
            
            # Calculate ELPD difference if available
            if len(comparison_result) >= 2:
                elpd_diff = comparison_result.iloc[1]['elpd_diff'] if 'elpd_diff' in comparison_result.columns else 0
                dse = comparison_result.iloc[1]['dse'] if 'dse' in comparison_result.columns else 1
            else:
                elpd_diff = 0
                dse = 1
            
            # Calculate effect size
            effect_size = abs(elpd_diff / dse) if dse > 0 else 0
            
            # Determine significance
            if effect_size > 2:
                significance = 'Significant'
            elif effect_size > 1:
                significance = 'Weak'
            else:
                significance = 'Non-significant'
            
            result = {
                'winner': winner,
                'elpd_diff': elpd_diff,
                'dse': dse,
                'effect_size': effect_size,
                'significance': significance,
                'method': 'WAIC',
                'comparison_table': comparison_result
            }
            
            print(f"    ✓ WAIC 比較成功")
            print(f"    獲勝者: {winner}")
            print(f"    ELPD 差異: {elpd_diff:.3f} ± {dse:.3f}")
            print(f"    效應量: {effect_size:.3f} ({significance})")
            
            return result
            
    except Exception as e:
        print(f"    ❌ WAIC 比較失敗: {e}")
    
    # Second attempt: Try LOO
    try:
        print("    嘗試使用 LOO 進行比較...")
        model_dict = {name: trace for name, trace in models.items()}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison_result = az.compare(model_dict, ic='loo')
        
        if comparison_result is not None and len(comparison_result) > 0:
            winner = comparison_result.index[0]
            
            if len(comparison_result) >= 2:
                elpd_diff = comparison_result.iloc[1]['elpd_diff'] if 'elpd_diff' in comparison_result.columns else 0
                dse = comparison_result.iloc[1]['dse'] if 'dse' in comparison_result.columns else 1
            else:
                elpd_diff = 0
                dse = 1
            
            effect_size = abs(elpd_diff / dse) if dse > 0 else 0
            
            if effect_size > 2:
                significance = 'Significant'
            elif effect_size > 1:
                significance = 'Weak'
            else:
                significance = 'Non-significant'
            
            result = {
                'winner': winner,
                'elpd_diff': elpd_diff,
                'dse': dse,
                'effect_size': effect_size,
                'significance': significance,
                'method': 'LOO',
                'comparison_table': comparison_result
            }
            
            print(f"    ✓ LOO 比較成功")
            print(f"    獲勝者: {winner}")
            
            return result
            
    except Exception as e:
        print(f"    ❌ LOO 比較也失敗: {e}")
    
    # Fallback: Simple comparison based on available metrics
    print("    使用簡化比較方法...")
    try:
        model_scores = {}
        model_info = {}
        
        for name, trace in models.items():
            try:
                # Calculate simple metrics
                n_samples = len(trace.posterior.coords['draw']) * len(trace.posterior.coords['chain'])
                
                # Try to get convergence metrics
                try:
                    summary = az.summary(trace)
                    max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.0
                    min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else n_samples
                except:
                    max_rhat = 1.0
                    min_ess = n_samples
                
                # Simple scoring: penalize high R-hat, reward high ESS
                score = min_ess / max(max_rhat - 1.0, 0.01)
                
                model_scores[name] = score
                model_info[name] = {
                    'n_samples': n_samples,
                    'max_rhat': max_rhat,
                    'min_ess': min_ess,
                    'score': score
                }
                
            except Exception as model_error:
                print(f"      ❌ 無法評估模型 {name}: {model_error}")
                model_scores[name] = 0
                model_info[name] = {'score': 0, 'error': str(model_error)}
        
        if model_scores:
            # Find the best model
            winner = max(model_scores, key=model_scores.get)
            best_score = model_scores[winner]
            
            # Calculate a rough effect size based on score differences
            sorted_scores = sorted(model_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                score_diff = sorted_scores[0] - sorted_scores[1]
                relative_diff = score_diff / sorted_scores[1] if sorted_scores[1] > 0 else 1.0
                effect_size = min(relative_diff, 5.0)  # Cap at 5.0
            else:
                effect_size = 0.0
            
            if effect_size > 0.5:
                significance = 'Significant'
            elif effect_size > 0.2:
                significance = 'Weak'
            else:
                significance = 'Non-significant'
            
            result = {
                'winner': winner,
                'elpd_diff': np.nan,
                'dse': np.nan,
                'effect_size': effect_size,
                'significance': significance,
                'method': 'Simple',
                'comparison_table': None,
                'model_scores': model_scores,
                'model_info': model_info
            }
            
            print(f"    ✓ 簡化比較完成")
            print(f"    獲勝者: {winner} (得分: {best_score:.1f})")
            print(f"    效應量: {effect_size:.3f} ({significance})")
            
            return result
            
    except Exception as e:
        print(f"    ❌ 簡化比較也失敗: {e}")
    
    # Final fallback: Just pick the first model
    model_names = list(models.keys())
    winner = model_names[0]
    
    print(f"    ⚠️ 所有比較方法都失敗，默認選擇: {winner}")
    
    return {
        'winner': winner,
        'elpd_diff': np.nan,
        'dse': np.nan,
        'effect_size': np.nan,
        'significance': 'Unknown',
        'method': 'Default',
        'comparison_table': None,
        'error': 'All comparison methods failed'
    }

def safe_parameter_extraction(trace, param_name, default_value=None):
    """
    Safely extract parameters from trace with proper error handling.
    """
    try:
        if param_name not in trace.posterior:
            if default_value is not None:
                return default_value
            else:
                raise KeyError(f"Parameter '{param_name}' not found in trace")
        
        # Get parameter values
        param_data = trace.posterior[param_name]
        values = param_data.values
        
        # Handle different shapes
        if len(values.shape) > 2:
            # If there's a trial dimension, average across it
            values = np.mean(values, axis=-1)
        
        # Flatten across chains and draws
        values = values.flatten()
        
        # Remove any NaN or infinite values
        values = values[np.isfinite(values)]
        
        if len(values) == 0:
            if default_value is not None:
                return default_value
            else:
                raise ValueError(f"No valid values found for parameter '{param_name}'")
        
        return values
        
    except Exception as e:
        if default_value is not None:
            print(f"    Warning: Could not extract '{param_name}', using default: {e}")
            return default_value
        else:
            raise e

def diagnose_trace_issues(trace, model_name):
    """
    Diagnose common issues with MCMC traces.
    """
    print(f"    診斷 {model_name} 的 trace 問題...")
    
    issues = []
    
    try:
        # Check basic structure
        if not hasattr(trace, 'posterior'):
            issues.append("Missing posterior samples")
            return issues
        
        # Check dimensions
        coords = trace.posterior.coords
        n_chains = len(coords.get('chain', [0]))
        n_draws = len(coords.get('draw', [0]))
        
        if n_chains < 2:
            issues.append(f"Too few chains: {n_chains}")
        
        if n_draws < 100:
            issues.append(f"Too few draws: {n_draws}")
        
        # Check for required parameters
        required_params = ['v_final_correct', 'v_final_incorrect', 'b_safe', 'start_var', 'non_decision']
        missing_params = [p for p in required_params if p not in trace.posterior]
        
        if missing_params:
            issues.append(f"Missing parameters: {missing_params}")
        
        # Check parameter shapes
        for param in required_params:
            if param in trace.posterior:
                param_shape = trace.posterior[param].shape
                expected_shape = (n_chains, n_draws)
                
                if len(param_shape) < 2:
                    issues.append(f"Parameter '{param}' has insufficient dimensions: {param_shape}")
                elif param_shape[:2] != expected_shape:
                    if len(param_shape) > 2:
                        # This might be trial-level parameter, which is OK
                        pass
                    else:
                        issues.append(f"Parameter '{param}' has unexpected shape: {param_shape} vs expected {expected_shape}")
        
        # Check for convergence issues
        try:
            summary = az.summary(trace)
            if 'r_hat' in summary.columns:
                max_rhat = summary['r_hat'].max()
                if max_rhat > 1.2:
                    issues.append(f"Poor convergence: max R-hat = {max_rhat:.3f}")
            
            if 'ess_bulk' in summary.columns:
                min_ess = summary['ess_bulk'].min()
                if min_ess < 100:
                    issues.append(f"Low effective sample size: min ESS = {min_ess:.0f}")
        except:
            issues.append("Could not compute convergence diagnostics")
        
        if not issues:
            print(f"      ✓ No major issues detected")
        else:
            for issue in issues:
                print(f"      ⚠️ {issue}")
        
    except Exception as e:
        issues.append(f"Error during diagnosis: {e}")
        print(f"      ❌ 診斷失敗: {e}")
    
    return issues

def create_robust_summary_table(models, participant_id, save_dir):
    """
    Create a robust summary table that handles extraction errors gracefully.
    """
    print(f"    創建參數摘要表...")
    
    summary_data = []
    
    for model_name, trace in models.items():
        try:
            # Diagnose any issues first
            issues = diagnose_trace_issues(trace, model_name)
            
            # Try to extract parameters safely
            row = {'Model': model_name, 'Participant': participant_id}
            
            # Extract each parameter with fallbacks
            param_extractions = {
                'v_final_correct': 'V_Correct_Mean',
                'v_final_incorrect': 'V_Incorrect_Mean', 
                'b_safe': 'Boundary_Mean',
                'start_var': 'StartVar_Mean',
                'non_decision': 'NonDecision_Mean'
            }
            
            for param_name, col_name in param_extractions.items():
                try:
                    values = safe_parameter_extraction(trace, param_name, default_value=np.array([np.nan]))
                    row[col_name] = np.mean(values)
                    row[col_name.replace('_Mean', '_Std')] = np.std(values)
                except:
                    row[col_name] = np.nan
                    row[col_name.replace('_Mean', '_Std')] = np.nan
            
            # Add diagnostic info
            row['N_Issues'] = len(issues)
            row['Issues'] = '; '.join(issues) if issues else 'None'
            
            # Add convergence info if available
            try:
                summary = az.summary(trace)
                row['Max_Rhat'] = summary['r_hat'].max() if 'r_hat' in summary.columns else np.nan
                row['Min_ESS'] = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else np.nan
            except:
                row['Max_Rhat'] = np.nan
                row['Min_ESS'] = np.nan
            
            summary_data.append(row)
            
        except Exception as e:
            print(f"      ❌ 無法處理模型 {model_name}: {e}")
            # Add a minimal row to indicate failure
            summary_data.append({
                'Model': model_name,
                'Participant': participant_id,
                'Error': str(e)
            })
    
    if summary_data:
        try:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(save_dir, f'participant_{participant_id}_model_summary.csv')
            df.to_csv(csv_file, index=False)
            print(f"      ✓ 摘要表已保存: {csv_file}")
            return df
        except Exception as e:
            print(f"      ❌ 保存摘要表失敗: {e}")
    
    return None

def improved_model_comparison(models):
    """
    Wrapper function that uses the robust model comparison.
    This replaces the original function in LBA_tool.py.
    """
    return robust_model_comparison(models)

# Test function to verify the fixes
def test_parameter_extraction(trace, model_name):
    """
    Test function to verify parameter extraction works properly.
    """
    print(f"  測試 {model_name} 的參數提取...")
    
    required_params = ['v_final_correct', 'v_final_incorrect', 'b_safe', 'start_var', 'non_decision']
    
    for param in required_params:
        try:
            values = safe_parameter_extraction(trace, param)
            print(f"    ✓ {param}: 提取了 {len(values)} 個樣本，範圍 [{values.min():.3f}, {values.max():.3f}]")
        except Exception as e:
            print(f"    ❌ {param}: 提取失敗 - {e}")
    
    return True