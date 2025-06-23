# -*- coding: utf-8 -*-
"""
model_fitting.py - 序列LBA模型擬合器
Sequential Processing LBA - Model Fitting Module

重新設計的完整功能：
- 單一和批次受試者擬合
- 完整的收斂診斷
- 參數估計和模型評估
- 結果儲存和管理
- 錯誤處理和恢復
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from sequential_model import SequentialLBA

warnings.filterwarnings('ignore')

class SequentialModelFitter:
    """序列LBA模型擬合器"""
    
    def __init__(self, first_side='left', time_split_ratio=0.6, mcmc_config=None):
        """
        初始化序列模型擬合器
        
        Args:
            first_side: 首先處理的通道 ('left' 或 'right')
            time_split_ratio: 時間分割比例 (0-1)
            mcmc_config: MCMC配置字典
        """
        
        # 初始化序列LBA模型
        self.model = SequentialLBA(first_side, time_split_ratio)
        
        # 設定MCMC配置
        self.mcmc_config = self._setup_mcmc_config(mcmc_config)
        
        # 設定收斂標準
        self.convergence_thresholds = self._setup_convergence_thresholds()
        
        print(f"✅ 序列LBA擬合器初始化完成")
        print(f"   處理順序: {self.model.first_side} 先處理")
        print(f"   時間分割: {self.model.time_split_ratio:.1%}")
        print(f"   MCMC設定: {self.mcmc_config['draws']} draws × {self.mcmc_config['chains']} chains")
        print(f"   總參數數: {len(self.model.all_param_names)}")
    
    def _setup_mcmc_config(self, user_config):
        """設定MCMC配置"""
        
        default_config = {
            'draws': 400,
            'tune': 400,
            'chains': 2,
            'cores': 1,
            'target_accept': 0.85,
            'max_treedepth': 8,
            'init': 'jitter+adapt_diag',
            'random_seed': 42,
            'progressbar': True,
            'return_inferencedata': True
        }
        
        if user_config:
            default_config.update(user_config)
        
        return default_config
    
    def _setup_convergence_thresholds(self):
        """設定收斂診斷標準"""
        
        return {
            'rhat_good': 1.01,
            'rhat_acceptable': 1.05,
            'rhat_problematic': 1.10,
            'ess_minimum': 100,
            'ess_good': 400,
            'max_divergent': 0,
            'energy_threshold': 0.3
        }
    
    def fit_subject(self, subject_data, verbose=True):
        """
        擬合單一受試者
        
        Args:
            subject_data: 受試者資料字典
            verbose: 是否顯示詳細訊息
            
        Returns:
            dict: 完整擬合結果
        """
        
        subject_id = subject_data['subject_id']
        start_time = time.time()
        
        if verbose:
            print(f"\n🔧 擬合受試者 {subject_id}")
            print(f"   試驗數: {subject_data['n_trials']}")
            print(f"   準確率: {subject_data['accuracy']:.1%}")
            print(f"   平均RT: {np.mean(subject_data['rt']):.3f}s")
        
        # 基本結果結構
        result = {
            'subject_id': subject_id,
            'success': False,
            'converged': False,
            'error': None,
            'sampling_time_minutes': 0.0,
            'n_trials': subject_data['n_trials'],
            'model_config': self.model.get_model_info(),
            'mcmc_config': self.mcmc_config
        }
        
        try:
            # 1. 資料驗證
            if not self._validate_subject_data(subject_data, verbose):
                result['error'] = 'Data validation failed'
                return result
            
            # 2. 模型建構
            if verbose:
                print("   🔧 建構PyMC模型...")
            
            pymc_model = self.model.build_model(subject_data)
            
            # 3. 模型驗證
            if not self._validate_model(pymc_model, verbose):
                result['error'] = 'Model validation failed'
                return result
            
            # 4. MCMC採樣
            if verbose:
                print("   🎲 執行MCMC採樣...")
            
            trace = self._run_mcmc_sampling(pymc_model, verbose)
            
            if trace is None:
                result['error'] = 'MCMC sampling failed'
                return result
            
            # 5. 結果處理
            sampling_time = time.time() - start_time
            result['sampling_time_minutes'] = sampling_time / 60
            result['trace'] = trace
            
            # 6. 參數估計
            parameter_estimates = self._extract_parameter_estimates(trace, verbose)
            result.update(parameter_estimates)
            
            # 7. 收斂診斷
            convergence_diagnostics = self._diagnose_convergence(trace, verbose)
            result['convergence_diagnostics'] = convergence_diagnostics
            result['converged'] = convergence_diagnostics['converged']
            
            # 8. 模型評估
            model_evaluation = self._evaluate_model_fit(trace, subject_data, verbose)
            result['model_evaluation'] = model_evaluation
            
            # 9. 觀察資料統計
            result.update(self._extract_observed_statistics(subject_data))
            
            result['success'] = True
            
            if verbose:
                status = "✅ 收斂" if result['converged'] else "⚠️ 收斂警告"
                print(f"   {status} (耗時 {sampling_time/60:.1f} 分鐘)")
                print(f"   R̂_max = {convergence_diagnostics['rhat_max']:.3f}")
                print(f"   ESS_min = {convergence_diagnostics['ess_bulk_min']:.0f}")
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['sampling_time_minutes'] = (time.time() - start_time) / 60
            
            if verbose:
                print(f"   ❌ 擬合失敗: {e}")
            
            return result
    
    def _validate_subject_data(self, subject_data, verbose=True):
        """驗證受試者資料"""
        
        required_fields = [
            'subject_id', 'n_trials', 'choices', 'rt',
            'left_stimuli', 'left_choices', 'right_stimuli', 'right_choices'
        ]
        
        # 檢查必要欄位
        for field in required_fields:
            if field not in subject_data:
                if verbose:
                    print(f"   ❌ 缺少必要欄位: {field}")
                return False
        
        # 檢查資料充足性
        if subject_data['n_trials'] < 50:
            if verbose:
                print(f"   ❌ 資料不足: {subject_data['n_trials']} < 50 trials")
            return False
        
        # 檢查資料長度一致性
        n_trials = subject_data['n_trials']
        for field in ['choices', 'rt', 'left_stimuli', 'left_choices', 'right_stimuli', 'right_choices']:
            if len(subject_data[field]) != n_trials:
                if verbose:
                    print(f"   ❌ 資料長度不一致: {field}")
                return False
        
        # 檢查資料範圍
        if not np.all(np.isin(subject_data['choices'], [0, 1, 2, 3])):
            if verbose:
                print("   ❌ 選擇資料範圍錯誤")
            return False
        
        if np.any(subject_data['rt'] <= 0):
            if verbose:
                print("   ❌ 反應時間包含非正值")
            return False
        
        return True
    
    def _validate_model(self, pymc_model, verbose=True):
        """驗證PyMC模型"""
        
        try:
            with pymc_model:
                test_point = pymc_model.initial_point()
                log_prob = pymc_model.compile_logp()(test_point)
                
                if not np.isfinite(log_prob):
                    if verbose:
                        print(f"   ❌ 無效的初始對數機率: {log_prob}")
                    return False
                
                if verbose:
                    print(f"   ✅ 模型驗證通過 (log_prob = {log_prob:.2f})")
                
                return True
                
        except Exception as e:
            if verbose:
                print(f"   ❌ 模型驗證失敗: {e}")
            return False
    
    def _run_mcmc_sampling(self, pymc_model, verbose=True):
        """執行MCMC採樣"""
        
        try:
            with pymc_model:
                # MAP估計（可選）
                map_estimate = None
                try:
                    if verbose:
                        print("   🎯 MAP估計...")
                    map_estimate = pm.find_MAP(method='BFGS', maxeval=800)
                    if verbose:
                        print("   ✅ MAP估計完成")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ MAP估計失敗: {e}")
                
                # NUTS採樣
                trace = pm.sample(
                    draws=self.mcmc_config['draws'],
                    tune=self.mcmc_config['tune'],
                    chains=self.mcmc_config['chains'],
                    cores=self.mcmc_config['cores'],
                    target_accept=self.mcmc_config['target_accept'],
                    max_treedepth=self.mcmc_config['max_treedepth'],
                    init=self.mcmc_config['init'],
                    initvals=map_estimate,
                    random_seed=self.mcmc_config['random_seed'],
                    progressbar=self.mcmc_config['progressbar'] and verbose,
                    return_inferencedata=self.mcmc_config['return_inferencedata']
                )
                
                return trace
                
        except Exception as e:
            if verbose:
                print(f"   ❌ MCMC採樣失敗: {e}")
            return None
    
    def _extract_parameter_estimates(self, trace, verbose=True):
        """提取參數估計"""
        
        try:
            summary = az.summary(trace, round_to=4)
            
            posterior_means = {}
            posterior_stds = {}
            credible_intervals = {}
            
            for param_name in self.model.all_param_names:
                if param_name in summary.index:
                    posterior_means[param_name] = float(summary.loc[param_name, 'mean'])
                    posterior_stds[param_name] = float(summary.loc[param_name, 'sd'])
                    
                    # 95%可信區間
                    if 'hdi_2.5%' in summary.columns and 'hdi_97.5%' in summary.columns:
                        credible_intervals[param_name] = [
                            float(summary.loc[param_name, 'hdi_2.5%']),
                            float(summary.loc[param_name, 'hdi_97.5%'])
                        ]
                else:
                    posterior_means[param_name] = np.nan
                    posterior_stds[param_name] = np.nan
                    credible_intervals[param_name] = [np.nan, np.nan]
            
            if verbose:
                n_valid_params = sum(1 for v in posterior_means.values() if not np.isnan(v))
                print(f"   📊 提取參數: {n_valid_params}/{len(self.model.all_param_names)}")
            
            return {
                'posterior_means': posterior_means,
                'posterior_stds': posterior_stds,
                'credible_intervals': credible_intervals,
                'parameter_summary': summary
            }
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 參數提取失敗: {e}")
            
            nan_dict = {name: np.nan for name in self.model.all_param_names}
            return {
                'posterior_means': nan_dict,
                'posterior_stds': nan_dict,
                'credible_intervals': {name: [np.nan, np.nan] for name in self.model.all_param_names},
                'parameter_summary': None
            }
    
    def _diagnose_convergence(self, trace, verbose=True):
        """完整的收斂診斷"""
        
        if verbose:
            print("   🔍 收斂診斷...")
        
        diagnostics = {
            'converged': False,
            'overall_status': 'failed',
            'rhat_max': np.nan,
            'rhat_mean': np.nan,
            'ess_bulk_min': np.nan,
            'ess_bulk_mean': np.nan,
            'ess_tail_min': np.nan,
            'ess_tail_mean': np.nan,
            'n_divergent': 0,
            'max_tree_depth': 0,
            'n_problematic_params': 0,
            'problematic_params': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            summary = az.summary(trace, round_to=4)
            
            # R̂ 統計
            if 'r_hat' in summary.columns:
                rhat_values = summary['r_hat'].dropna()
                if len(rhat_values) > 0:
                    diagnostics['rhat_max'] = float(rhat_values.max())
                    diagnostics['rhat_mean'] = float(rhat_values.mean())
            
            # ESS 統計
            if 'ess_bulk' in summary.columns:
                ess_bulk = summary['ess_bulk'].dropna()
                if len(ess_bulk) > 0:
                    diagnostics['ess_bulk_min'] = float(ess_bulk.min())
                    diagnostics['ess_bulk_mean'] = float(ess_bulk.mean())
            
            if 'ess_tail' in summary.columns:
                ess_tail = summary['ess_tail'].dropna()
                if len(ess_tail) > 0:
                    diagnostics['ess_tail_min'] = float(ess_tail.min())
                    diagnostics['ess_tail_mean'] = float(ess_tail.mean())
            
            # NUTS診斷
            try:
                if hasattr(trace, 'sample_stats'):
                    if 'diverging' in trace.sample_stats:
                        diagnostics['n_divergent'] = int(trace.sample_stats['diverging'].sum())
                    
                    if 'tree_depth' in trace.sample_stats:
                        diagnostics['max_tree_depth'] = int(trace.sample_stats['tree_depth'].max())
            except:
                pass
            
            # 問題參數識別
            problematic_params = []
            for param_name in summary.index:
                issues = []
                
                if 'r_hat' in summary.columns:
                    rhat = summary.loc[param_name, 'r_hat']
                    if not pd.isna(rhat) and rhat > self.convergence_thresholds['rhat_good']:
                        issues.append(f"R̂={rhat:.3f}")
                
                if 'ess_bulk' in summary.columns:
                    ess = summary.loc[param_name, 'ess_bulk']
                    if not pd.isna(ess) and ess < self.convergence_thresholds['ess_minimum']:
                        issues.append(f"ESS={ess:.0f}")
                
                if issues:
                    problematic_params.append({
                        'parameter': param_name,
                        'issues': issues,
                        'rhat': float(summary.loc[param_name, 'r_hat']) if 'r_hat' in summary.columns else np.nan,
                        'ess_bulk': float(summary.loc[param_name, 'ess_bulk']) if 'ess_bulk' in summary.columns else np.nan
                    })
            
            diagnostics['problematic_params'] = problematic_params
            diagnostics['n_problematic_params'] = len(problematic_params)
            
            # 整體收斂判斷
            rhat_ok = diagnostics['rhat_max'] <= self.convergence_thresholds['rhat_good']
            ess_ok = (diagnostics['ess_bulk_min'] >= self.convergence_thresholds['ess_minimum'] and
                     diagnostics['ess_tail_min'] >= self.convergence_thresholds['ess_minimum'])
            no_divergence = diagnostics['n_divergent'] == 0
            
            if rhat_ok and ess_ok and no_divergence:
                diagnostics['converged'] = True
                diagnostics['overall_status'] = 'converged'
            elif diagnostics['rhat_max'] <= self.convergence_thresholds['rhat_acceptable']:
                diagnostics['overall_status'] = 'warning'
            else:
                diagnostics['overall_status'] = 'failed'
            
            # 生成建議
            if not diagnostics['converged']:
                if diagnostics['rhat_max'] > self.convergence_thresholds['rhat_good']:
                    diagnostics['recommendations'].append("增加採樣數或鏈數")
                if diagnostics['ess_bulk_min'] < self.convergence_thresholds['ess_minimum']:
                    diagnostics['recommendations'].append("增加draws數量")
                if diagnostics['n_divergent'] > 0:
                    diagnostics['recommendations'].append("降低target_accept或增加adapt_delta")
            
            if verbose:
                if diagnostics['converged']:
                    print(f"      ✅ 收斂良好")
                else:
                    print(f"      ⚠️ 收斂問題: {len(problematic_params)} 個參數")
            
            return diagnostics
            
        except Exception as e:
            if verbose:
                print(f"   ❌ 收斂診斷失敗: {e}")
            diagnostics['error'] = str(e)
            return diagnostics
    
    def _evaluate_model_fit(self, trace, subject_data, verbose=True):
        """評估模型擬合品質"""
        
        evaluation = {
            'waic': np.nan,
            'waic_se': np.nan,
            'loo': np.nan,
            'loo_se': np.nan,
            'mean_ess': np.nan,
            'mean_rhat': np.nan
        }
        
        try:
            # WAIC
            try:
                waic_result = az.waic(trace)
                evaluation['waic'] = float(waic_result.waic)
                evaluation['waic_se'] = float(waic_result.se)
            except:
                pass
            
            # LOO
            try:
                loo_result = az.loo(trace)
                evaluation['loo'] = float(loo_result.loo)
                evaluation['loo_se'] = float(loo_result.se)
            except:
                pass
            
            # 平均統計
            try:
                ess_stats = az.ess(trace)
                evaluation['mean_ess'] = float(ess_stats.to_array().mean())
                
                rhat_stats = az.rhat(trace)
                evaluation['mean_rhat'] = float(rhat_stats.to_array().mean())
            except:
                pass
            
            if verbose:
                print(f"   📈 模型評估完成")
                if not np.isnan(evaluation['waic']):
                    print(f"      WAIC = {evaluation['waic']:.1f}")
            
            return evaluation
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 模型評估失敗: {e}")
            evaluation['error'] = str(e)
            return evaluation
    
    def _extract_observed_statistics(self, subject_data):
        """提取觀察資料統計"""
        
        return {
            'observed_accuracy': subject_data['accuracy'],
            'observed_mean_rt': float(np.mean(subject_data['rt'])),
            'observed_std_rt': float(np.std(subject_data['rt'])),
            'observed_left_accuracy': subject_data.get('left_accuracy', np.nan),
            'observed_right_accuracy': subject_data.get('right_accuracy', np.nan)
        }
    
    def fit_multiple_subjects(self, subjects_data, max_subjects=None, 
                            continue_on_failure=True, verbose=True):
        """
        批次擬合多個受試者
        
        Args:
            subjects_data: 受試者資料列表
            max_subjects: 最大受試者數限制
            continue_on_failure: 是否在失敗時繼續
            verbose: 是否顯示詳細訊息
            
        Returns:
            list: 所有受試者的擬合結果
        """
        
        if max_subjects:
            subjects_data = subjects_data[:max_subjects]
        
        n_subjects = len(subjects_data)
        
        if verbose:
            print(f"\n🎯 批次擬合開始")
            print(f"   受試者數: {n_subjects}")
            print(f"   模型: {self.model.first_side} 先處理")
            print(f"   MCMC: {self.mcmc_config['draws']} draws × {self.mcmc_config['chains']} chains")
            print("="*60)
        
        results = []
        batch_start_time = time.time()
        successful_count = 0
        converged_count = 0
        
        for i, subject_data in enumerate(subjects_data, 1):
            if verbose:
                print(f"\n📍 進度: {i}/{n_subjects} ({i/n_subjects*100:.1f}%)")
            
            # 擬合單一受試者
            result = self.fit_subject(subject_data, verbose=verbose)
            results.append(result)
            
            # 更新統計
            if result['success']:
                successful_count += 1
                if result['converged']:
                    converged_count += 1
                
                if verbose:
                    status = "✅ 收斂" if result['converged'] else "⚠️ 警告"
                    print(f"   {status} 受試者 {result['subject_id']}")
            else:
                if verbose:
                    print(f"   ❌ 失敗 受試者 {result['subject_id']}: {result['error']}")
                
                # 檢查是否繼續
                if not continue_on_failure:
                    print("   🛑 停止批次處理")
                    break
            
            # 早期失敗檢測
            if i >= 3:
                recent_failures = sum(1 for r in results[-3:] if not r['success'])
                if recent_failures >= 3:
                    if verbose:
                        print(f"\n⚠️ 警告: 連續3個受試者失敗")
                        user_input = input("是否繼續? (y/n): ")
                        if user_input.lower() != 'y':
                            print("批次處理終止")
                            break
        
        # 批次摘要
        batch_time = time.time() - batch_start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎉 批次擬合完成")
            print(f"⏱️ 總時間: {batch_time/60:.1f} 分鐘")
            print(f"✅ 成功: {successful_count}/{len(results)} ({successful_count/len(results)*100:.1f}%)")
            if successful_count > 0:
                print(f"🔄 收斂: {converged_count}/{successful_count} ({converged_count/successful_count*100:.1f}%)")
                avg_time = np.mean([r['sampling_time_minutes'] for r in results if r['success']])
                print(f"⏱️ 平均時間: {avg_time:.1f} 分鐘/受試者")
        
        return results
    
    def save_results(self, results, output_prefix="sequential_lba_results"):
        """
        儲存擬合結果到多個檔案
        
        Args:
            results: 擬合結果列表
            output_prefix: 輸出檔名前綴
            
        Returns:
            dict: 儲存的檔案路徑字典
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            print(f"\n💾 儲存結果...")
            
            # 1. 主要結果檔案
            main_data = []
            for result in results:
                main_row = {
                    'subject_id': result['subject_id'],
                    'success': result['success'],
                    'converged': result['converged'],
                    'error': result.get('error', ''),
                    'n_trials': result['n_trials'],
                    'sampling_time_minutes': result['sampling_time_minutes'],
                    'rhat_max': result.get('convergence_diagnostics', {}).get('rhat_max', np.nan),
                    'ess_bulk_min': result.get('convergence_diagnostics', {}).get('ess_bulk_min', np.nan),
                    'n_divergent': result.get('convergence_diagnostics', {}).get('n_divergent', 0),
                    'waic': result.get('model_evaluation', {}).get('waic', np.nan),
                    'observed_accuracy': result.get('observed_accuracy', np.nan),
                    'observed_mean_rt': result.get('observed_mean_rt', np.nan)
                }
                main_data.append(main_row)
            
            main_df = pd.DataFrame(main_data)
            main_filename = f"{output_prefix}_main_{timestamp}.csv"
            main_df.to_csv(main_filename, index=False, encoding='utf-8-sig')
            saved_files['main_results'] = main_filename
            
            # 2. 參數估計檔案
            successful_results = [r for r in results if r['success']]
            if successful_results:
                params_data = []
                for result in successful_results:
                    param_row = {'subject_id': result['subject_id']}
                    
                    # 後驗均值
                    param_row.update(result.get('posterior_means', {}))
                    
                    # 後驗標準差 (加上_std後綴)
                    for param, std_val in result.get('posterior_stds', {}).items():
                        param_row[f"{param}_std"] = std_val
                    
                    params_data.append(param_row)
                
                params_df = pd.DataFrame(params_data)
                params_filename = f"{output_prefix}_parameters_{timestamp}.csv"
                params_df.to_csv(params_filename, index=False, encoding='utf-8-sig')
                saved_files['parameters'] = params_filename
                
                # 3. 收斂診斷檔案
                convergence_data = []
                for result in successful_results:
                    conv_diag = result.get('convergence_diagnostics', {})
                    conv_row = {
                        'subject_id': result['subject_id'],
                        'converged': conv_diag.get('converged', False),
                        'overall_status': conv_diag.get('overall_status', 'unknown'),
                        'rhat_max': conv_diag.get('rhat_max', np.nan),
                        'rhat_mean': conv_diag.get('rhat_mean', np.nan),
                        'ess_bulk_min': conv_diag.get('ess_bulk_min', np.nan),
                        'ess_bulk_mean': conv_diag.get('ess_bulk_mean', np.nan),
                        'ess_tail_min': conv_diag.get('ess_tail_min', np.nan),
                        'ess_tail_mean': conv_diag.get('ess_tail_mean', np.nan),
                        'n_divergent': conv_diag.get('n_divergent', 0),
                        'n_problematic_params': conv_diag.get('n_problematic_params', 0)
                    }
                    convergence_data.append(conv_row)
                
                conv_df = pd.DataFrame(convergence_data)
                conv_filename = f"{output_prefix}_convergence_{timestamp}.csv"
                conv_df.to_csv(conv_filename, index=False, encoding='utf-8-sig')
                saved_files['convergence'] = conv_filename
                
                # 4. 模型評估檔案
                evaluation_data = []
                for result in successful_results:
                    eval_data = result.get('model_evaluation', {})
                    eval_row = {
                        'subject_id': result['subject_id'],
                        'waic': eval_data.get('waic', np.nan),
                        'waic_se': eval_data.get('waic_se', np.nan),
                        'loo': eval_data.get('loo', np.nan),
                        'loo_se': eval_data.get('loo_se', np.nan),
                        'mean_ess': eval_data.get('mean_ess', np.nan),
                        'mean_rhat': eval_data.get('mean_rhat', np.nan)
                    }
                    # 添加觀察統計
                    eval_row.update({
                        'observed_accuracy': result.get('observed_accuracy', np.nan),
                        'observed_mean_rt': result.get('observed_mean_rt', np.nan),
                        'observed_std_rt': result.get('observed_std_rt', np.nan),
                        'observed_left_accuracy': result.get('observed_left_accuracy', np.nan),
                        'observed_right_accuracy': result.get('observed_right_accuracy', np.nan)
                    })
                    evaluation_data.append(eval_row)
                
                eval_df = pd.DataFrame(evaluation_data)
                eval_filename = f"{output_prefix}_evaluation_{timestamp}.csv"
                eval_df.to_csv(eval_filename, index=False, encoding='utf-8-sig')
                saved_files['evaluation'] = eval_filename
            
            print(f"   ✅ 結果已儲存:")
            for file_type, filename in saved_files.items():
                print(f"      {file_type}: {filename}")
            
            return saved_files
            
        except Exception as e:
            print(f"   ❌ 儲存結果失敗: {e}")
            return {}
    
    def get_batch_summary(self, results):
        """獲得批次結果摘要"""
        
        total_subjects = len(results)
        successful_results = [r for r in results if r['success']]
        converged_results = [r for r in successful_results if r['converged']]
        
        summary = {
            'total_subjects': total_subjects,
            'successful_subjects': len(successful_results),
            'converged_subjects': len(converged_results),
            'success_rate': len(successful_results) / total_subjects if total_subjects > 0 else 0,
            'convergence_rate': len(converged_results) / len(successful_results) if successful_results else 0
        }
        
        if successful_results:
            # 時間統計
            times = [r['sampling_time_minutes'] for r in successful_results]
            summary.update({
                'mean_sampling_time': np.mean(times),
                'total_sampling_time': np.sum(times)
            })
            
            # 收斂統計
            if converged_results:
                rhat_values = [r['convergence_diagnostics']['rhat_max'] for r in converged_results]
                ess_values = [r['convergence_diagnostics']['ess_bulk_min'] for r in converged_results]
                
                summary.update({
                    'mean_rhat': np.mean(rhat_values),
                    'max_rhat': np.max(rhat_values),
                    'min_ess': np.min(ess_values),
                    'mean_ess': np.mean(ess_values)
                })
        
        return summary

# 便利函數
def create_fitter(first_side='left', time_split_ratio=0.6, **mcmc_kwargs):
    """快速創建擬合器"""
    return SequentialModelFitter(first_side, time_split_ratio, mcmc_kwargs)

def quick_fit_test(subject_data, first_side='left'):
    """快速擬合測試"""
    quick_config = {
        'draws': 100,
        'tune': 100,
        'chains': 1,
        'target_accept': 0.80,
        'progressbar': True
    }
    
    fitter = SequentialModelFitter(first_side, 0.6, quick_config)
    return fitter.fit_subject(subject_data)

def test_model_fitting():
    """測試模型擬合功能"""
    
    print("🧪 測試序列LBA擬合器...")
    
    try:
        # 創建測試資料
        n_trials = 100
        np.random.seed(42)
        
        test_data = {
            'subject_id': 999,
            'n_trials': n_trials,
            'accuracy': 0.75,
            'choices': np.random.choice([0, 1, 2, 3], size=n_trials),
            'rt': np.random.uniform(0.3, 1.5, size=n_trials),
            'left_stimuli': np.random.choice([0, 1], size=n_trials),
            'left_choices': np.random.choice([0, 1], size=n_trials),
            'right_stimuli': np.random.choice([0, 1], size=n_trials),
            'right_choices': np.random.choice([0, 1], size=n_trials),
            'left_accuracy': 0.8,
            'right_accuracy': 0.7
        }
        
        # 測試快速擬合
        print("   執行快速擬合測試...")
        result = quick_fit_test(test_data)
        
        if result['success']:
            print("   ✅ 擬合成功!")
            print(f"   收斂: {'是' if result['converged'] else '否'}")
            print(f"   參數數: {len([v for v in result['posterior_means'].values() if not np.isnan(v)])}")
            print(f"   時間: {result['sampling_time_minutes']:.1f} 分鐘")
        else:
            print(f"   ❌ 擬合失敗: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

if __name__ == "__main__":
    test_model_fitting()
