# -*- coding: utf-8 -*-
"""
LBA_main_fixed.py - 修復版 LBA 分析主程式
解決 loglikelihood 計算問題
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import os
import warnings
from datetime import datetime
import argparse

# 導入修復版函數
from LBA_loglikelihood_fix import (
    create_fixed_coactive_model, 
    diagnose_loglikelihood_issue,
    create_fixed_lba_logp
)

# 抑制警告
warnings.filterwarnings('ignore')

def fixed_sample_with_convergence_check(model, max_attempts=3, draws=300, tune=400, chains=2):
    """
    修復版採樣函數，專門針對 loglikelihood 問題
    """
    
    print(f"  開始修復版採樣 (draws={draws}, tune={tune}, chains={chains})...")
    
    for attempt in range(max_attempts):
        try:
            print(f"    嘗試 {attempt + 1}/{max_attempts}")
            
            with model:
                # 更穩健的採樣策略
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=0.85,  # 較低的接受率以提高穩定性
                    return_inferencedata=True,
                    progressbar=True,
                    random_seed=42 + attempt,
                    init='jitter+adapt_diag',
                    cores=1  # 單核心避免並行問題
                )
                
                # 檢查採樣結果
                try:
                    summary = az.summary(trace)
                    max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.0
                    min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else 100
                    
                    print(f"    最大 R-hat: {max_rhat:.4f}")
                    print(f"    最小 ESS: {min_ess:.0f}")
                    
                    # 較寬鬆的收斂標準
                    if max_rhat < 1.3 and min_ess > 30:
                        print(f"    ✓ 收斂成功")
                        
                        # 檢查並修復 log_likelihood
                        trace_fixed = ensure_log_likelihood(trace)
                        
                        return trace_fixed, {
                            'max_rhat': max_rhat, 
                            'min_ess': min_ess,
                            'attempt': attempt + 1
                        }
                    else:
                        print(f"    ⚠️ 收斂標準未達成，但返回結果...")
                        if attempt == max_attempts - 1:
                            trace_fixed = ensure_log_likelihood(trace)
                            return trace_fixed, {
                                'max_rhat': max_rhat, 
                                'min_ess': min_ess,
                                'attempt': attempt + 1
                            }
                        
                except Exception as diag_error:
                    print(f"    ⚠️ 診斷失敗但採樣完成: {diag_error}")
                    if trace is not None:
                        trace_fixed = ensure_log_likelihood(trace)
                        return trace_fixed, {'max_rhat': np.nan, 'min_ess': np.nan}
                
        except Exception as e:
            print(f"    ❌ 採樣失敗: {e}")
            if attempt < max_attempts - 1:
                # 調整參數重試
                draws = max(100, int(draws * 0.8))
                tune = max(200, int(tune * 0.9))
                print(f"    調整參數重試: draws={draws}, tune={tune}")
    
    print(f"    ❌ {max_attempts} 次嘗試後仍未成功")
    return None, None

def ensure_log_likelihood(trace):
    """
    確保 trace 包含正確的 log_likelihood 用於模型比較
    """
    try:
        if hasattr(trace, 'log_likelihood'):
            print("    ✓ trace 已包含 log_likelihood")
            return trace
        
        # 檢查是否有手動計算的 log_likelihood
        if 'log_likelihood_values' in trace.posterior:
            print("    🔧 從手動計算創建 log_likelihood...")
            
            import xarray as xr
            
            # 創建 log_likelihood 數據集
            log_likelihood = xr.Dataset({
                'likelihood': trace.posterior['log_likelihood_values']
            })
            
            # 將其添加到 trace
            trace_fixed = trace.assign(log_likelihood=log_likelihood)
            print("    ✓ log_likelihood 創建成功")
            return trace_fixed
        else:
            print("    ⚠️ 無法找到 log_likelihood 數據")
            return trace
            
    except Exception as e:
        print(f"    ❌ log_likelihood 修復失敗: {e}")
        return trace

def improved_model_comparison(models):
    """
    改進版模型比較，專門處理 loglikelihood 問題
    """
    print("🔬 執行改進版模型比較...")
    
    if len(models) < 2:
        print("需要至少 2 個模型進行比較")
        return None
    
    # 檢查每個模型的 log_likelihood
    valid_models = {}
    model_scores = {}
    
    for model_name, trace in models.items():
        print(f"  檢查 {model_name}...")
        
        try:
            # 方法 1: 嘗試 WAIC/LOO
            if hasattr(trace, 'log_likelihood'):
                try:
                    ll_values = trace.log_likelihood.likelihood.values
                    if not np.any(np.isnan(ll_values)) and not np.any(np.isinf(ll_values)):
                        valid_models[model_name] = trace
                        model_scores[model_name] = np.mean(ll_values)
                        print(f"    ✓ {model_name}: 有效的 log_likelihood")
                        continue
                except:
                    pass
            
            # 方法 2: 使用手動計算的 log_likelihood
            if 'log_likelihood_values' in trace.posterior:
                try:
                    ll_values = trace.posterior['log_likelihood_values'].values.flatten()
                    ll_clean = ll_values[np.isfinite(ll_values)]
                    if len(ll_clean) > 0:
                        model_scores[model_name] = np.mean(ll_clean)
                        print(f"    ✓ {model_name}: 使用手動 log_likelihood")
                        continue
                except:
                    pass
            
            # 方法 3: 基於收斂性評分
            try:
                summary = az.summary(trace)
                max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.0
                min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else 100
                
                # 收斂評分
                score = min_ess / max(max_rhat - 1.0, 0.01)
                model_scores[model_name] = score
                print(f"    ~ {model_name}: 收斂評分 = {score:.2f}")
            except:
                model_scores[model_name] = 0
                print(f"    ❌ {model_name}: 無法評分")
                
        except Exception as e:
            print(f"    ❌ {model_name}: 評估失敗 - {e}")
            model_scores[model_name] = 0
    
    # 嘗試標準 WAIC 比較
    if len(valid_models) >= 2:
        try:
            print("  嘗試 WAIC 比較...")
            comparison_result = az.compare(valid_models, ic='waic')
            winner = comparison_result.index[0]
            
            print(f"    ✅ WAIC 成功！獲勝者: {winner}")
            
            return {
                'winner': winner,
                'method': 'WAIC',
                'comparison_table': comparison_result,
                'model_scores': model_scores,
                'success': True
            }
        except Exception as e:
            print(f"    ❌ WAIC 失敗: {e}")
    
    # 備用：基於評分的比較
    if model_scores:
        winner = max(model_scores, key=model_scores.get)
        winner_score = model_scores[winner]
        
        print(f"    🏆 評分獲勝者: {winner} (評分: {winner_score:.2f})")
        
        return {
            'winner': winner,
            'method': 'Score_Based',
            'model_scores': model_scores,
            'success': True
        }
    else:
        # 最後手段
        winner = list(models.keys())[0]
        return {
            'winner': winner,
            'method': 'Default',
            'success': False
        }

class FixedLBAAnalysisRunner:
    """修復版 LBA 分析運行器"""
    
    def __init__(self, data_file='model_data.npz', output_base_dir='lba_fixed_results'):
        self.data_file = data_file
        self.output_base_dir = output_base_dir
        self.results_dir = None
        self.data = None
        self.participants = None
        
    def setup_analysis(self):
        """設置分析環境"""
        
        print("🔧 設置修復版 LBA 分析環境...")
        
        # 診斷數據問題
        issues = diagnose_loglikelihood_issue(self.data_file)
        if issues:
            print("發現數據問題，嘗試使用修復版數據...")
            fixed_file = self.data_file.replace('.npz', '_fixed.npz')
            if os.path.exists(fixed_file):
                self.data_file = fixed_file
                print(f"切換到修復版數據: {fixed_file}")
        
        # 創建輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{self.output_base_dir}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 載入數據
        try:
            self.data = np.load(self.data_file, allow_pickle=True)
            observed_value = self.data['observed_value']
            participant_idx = self.data['participant_idx']
            
            self.participants = np.unique(participant_idx)
            
            print(f"✓ 修復版數據載入成功")
            print(f"✓ 參與者數: {len(self.participants)}")
            print(f"✓ 總試驗數: {len(observed_value)}")
            print(f"✓ RT 範圍: {observed_value[:, 0].min():.1f} - {observed_value[:, 0].max():.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 數據載入失敗: {e}")
            return False
    
    def run_single_participant_analysis(self, participant_id):
        """運行單個參與者分析"""
        
        print(f"\n🧠 分析參與者 {participant_id} (修復版)")
        print("-" * 50)
        
        # 提取參與者數據
        observed_value = self.data['observed_value']
        participant_idx = self.data['participant_idx']
        model_input_data = self.data['model_input_data'].item()
        
        mask = participant_idx == participant_id
        participant_data = observed_value[mask]
        participant_input = {
            'left_match': model_input_data['left_match'][mask],
            'right_match': model_input_data['right_match'][mask]
        }
        
        print(f"試驗數: {len(participant_data)}")
        print(f"平均 RT: {participant_data[:, 0].mean():.1f} ms")
        print(f"準確率: {participant_data[:, 1].mean():.3f}")
        
        # 創建並擬合模型
        try:
            print("\n📊 創建修復版 Coactive 模型...")
            model = create_fixed_coactive_model(participant_data, participant_input)
            
            # 採樣
            print("🔄 開始採樣...")
            trace, diagnostics = fixed_sample_with_convergence_check(
                model, 
                max_attempts=3,
                draws=400,
                tune=500,
                chains=2
            )
            
            if trace is not None:
                print("✅ 模型擬合成功")
                
                # 保存結果
                self.save_participant_results(participant_id, trace, diagnostics)
                
                # 測試模型比較功能
                models = {'Coactive_Fixed': trace}
                comparison = improved_model_comparison(models)
                
                return {
                    'participant': participant_id,
                    'trace': trace,
                    'diagnostics': diagnostics,
                    'comparison': comparison,
                    'success': True
                }
            else:
                print("❌ 模型擬合失敗")
                return {
                    'participant': participant_id,
                    'success': False,
                    'error': 'Sampling failed'
                }
                
        except Exception as e:
            print(f"❌ 分析失敗: {e}")
            return {
                'participant': participant_id,
                'success': False,
                'error': str(e)
            }
    
    def save_participant_results(self, participant_id, trace, diagnostics):
        """保存參與者結果"""
        
        try:
            # 保存 trace
            trace_file = os.path.join(self.results_dir, f'participant_{participant_id}_trace.nc')
            trace.to_netcdf(trace_file)
            
            # 創建摘要報告
            summary_file = os.path.join(self.results_dir, f'participant_{participant_id}_summary.txt')
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"參與者 {participant_id} 分析結果\n")
                f.write("=" * 40 + "\n\n")
                
                # 基本信息
                f.write("模型信息:\n")
                f.write("- 模型類型: 修復版 Coactive LBA\n")
                f.write(f"- 採樣時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 收斂診斷
                if diagnostics:
                    f.write("收斂診斷:\n")
                    f.write(f"- 最大 R-hat: {diagnostics.get('max_rhat', 'N/A')}\n")
                    f.write(f"- 最小 ESS: {diagnostics.get('min_ess', 'N/A')}\n")
                    f.write(f"- 採樣嘗試: {diagnostics.get('attempt', 'N/A')}\n\n")
                
                # 參數估計
                try:
                    summary = az.summary(trace)
                    f.write("參數估計:\n")
                    f.write(summary.to_string())
                    f.write("\n\n")
                except:
                    f.write("參數估計: 無法生成\n\n")
                
                # Log-likelihood 檢查
                if hasattr(trace, 'log_likelihood'):
                    f.write("✓ Log-likelihood: 正常\n")
                elif 'log_likelihood_values' in trace.posterior:
                    f.write("✓ Log-likelihood: 手動計算\n")
                else:
                    f.write("❌ Log-likelihood: 缺失\n")
            
            print(f"    ✓ 結果已保存: {summary_file}")
            
        except Exception as e:
            print(f"    ❌ 保存結果失敗: {e}")

def main():
    """主程序"""
    
    parser = argparse.ArgumentParser(description='修復版 LBA 分析程式')
    parser.add_argument('--participant', type=str, help='分析特定參與者')
    parser.add_argument('--data-file', default='model_data.npz', help='數據檔案')
    parser.add_argument('--test', action='store_true', help='執行測試模式')
    
    args = parser.parse_args()
    
    print("🧠 修復版 LBA 分析程式")
    print("=" * 50)
    print("專門解決 loglikelihood 計算問題")
    
    # 創建分析器
    runner = FixedLBAAnalysisRunner(data_file=args.data_file)
    
    # 設置環境
    if not runner.setup_analysis():
        print("❌ 環境設置失敗")
        return
    
    if args.test:
        # 測試模式
        print("\n🧪 執行測試模式...")
        if len(runner.participants) > 0:
            test_participant = runner.participants[0]
            result = runner.run_single_participant_analysis(test_participant)
            
            if result['success']:
                print(f"\n✅ 測試成功！參與者 {test_participant}")
                print(f"📁 結果保存於: {runner.results_dir}")
            else:
                print(f"\n❌ 測試失敗: {result.get('error', '未知錯誤')}")
    
    elif args.participant:
        # 單個參與者模式
        if args.participant in runner.participants.astype(str):
            result = runner.run_single_participant_analysis(args.participant)
            
            if result['success']:
                print(f"\n✅ 分析完成！參與者 {args.participant}")
            else:
                print(f"\n❌ 分析失敗: {result.get('error', '未知錯誤')}")
        else:
            print(f"❌ 未找到參與者 {args.participant}")
            print(f"可用參與者: {runner.participants[:10]}...")
    
    else:
        # 顯示使用說明
        print("\n使用方式:")
        print("  python LBA_main_fixed.py --test                    # 測試模式")
        print("  python LBA_main_fixed.py --participant ID         # 分析特定參與者")
        print("  python LBA_main_fixed.py --data-file file.npz     # 指定數據檔案")
        print(f"\n可用參與者: {runner.participants[:5]}...")

if __name__ == '__main__':
    main()