# drift_rate_evidence_integration.py - 修正錯誤版本

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def diagnose_sampling_issues(trace, verbose=True):
    """診斷採樣問題"""
    
    issues = []
    
    try:
        # 檢查發散樣本
        if hasattr(trace, 'sample_stats'):
            divergences = trace.sample_stats.diverging.sum().values
            if divergences > 0:
                issues.append(f"發散樣本: {divergences}")
        
        # 檢查 R-hat
        try:
            rhat = az.rhat(trace)
            max_rhat = float(rhat.to_array().max())
            if max_rhat > 1.1:
                issues.append(f"R-hat 過高: {max_rhat:.3f}")
        except:
            issues.append("R-hat 計算失敗")
        
        # 檢查有效樣本數
        try:
            ess = az.ess(trace)
            min_ess = float(ess.to_array().min())
            if min_ess < 100:
                issues.append(f"ESS 過低: {min_ess:.0f}")
        except:
            issues.append("ESS 計算失敗")
        
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

class EvidenceIntegrationComparison:
    """基於Single LBA的證據整合模型比較器 - 修正版本"""
    
    def __init__(self, mcmc_config=None):
        """初始化證據整合比較器"""
        
        self.mcmc_config = self._setup_mcmc_config(mcmc_config)
        
        # 固定的共享參數
        self.FIXED_PARAMS = {
            'shared_start_var': 0.35,
            'shared_threshold': 0.60,
            'shared_ndt': 0.22,
            'shared_noise': 0.25
        }
        
        print("✅ 初始化證據整合模型比較器 (修正版)")
        print("   固定參數:")
        for param, value in self.FIXED_PARAMS.items():
            print(f"     {param}: {value}")
    
    def _setup_mcmc_config(self, user_config):
        """設定改進的MCMC配置"""
        
        # 更保守的MCMC設定以避免發散
        default_config = {
            'draws': 600,               # 增加draws
            'tune': 800,                # 增加tune
            'chains': 4,                # 增加chains
            'cores': 1,                 # 序列採樣避免並行問題
            'target_accept': 0.95,      # 提高target_accept
            'max_treedepth': 12,        # 增加max_treedepth
            'init': 'adapt_diag',       # 更好的初始化
            'random_seed': [42, 43, 44, 45],  # 每條鏈不同種子
            'progressbar': True,
            'return_inferencedata': True
        }
        
        if user_config:
            default_config.update(user_config)
        
        return default_config
    
    def prepare_subject_data(self, df, subject_id):
        """準備受試者數據"""
        
        # 過濾受試者資料
        subject_df = df[df['participant'] == subject_id].copy()
        
        if len(subject_df) == 0:
            raise ValueError(f"找不到受試者 {subject_id} 的資料")
        
        # 刺激映射
        stimulus_mapping = {
            0: {'left': 1, 'right': 0},  # 左對角，右垂直
            1: {'left': 1, 'right': 1},  # 左對角，右對角
            2: {'left': 0, 'right': 0},  # 左垂直，右垂直
            3: {'left': 0, 'right': 1}   # 左垂直，右對角
        }
        
        # 選擇映射
        choice_mapping = {
            0: {'left': 1, 'right': 0},  # 選擇 \|
            1: {'left': 1, 'right': 1},  # 選擇 \/
            2: {'left': 0, 'right': 0},  # 選擇 ||
            3: {'left': 0, 'right': 1}   # 選擇 |/
        }
        
        # 分解刺激和選擇
        left_stimuli = []
        right_stimuli = []
        left_choices = []
        right_choices = []
        
        for _, row in subject_df.iterrows():
            stimulus = int(row['Stimulus'])
            choice = int(row['Response'])
            
            left_stimuli.append(stimulus_mapping[stimulus]['left'])
            right_stimuli.append(stimulus_mapping[stimulus]['right'])
            left_choices.append(choice_mapping[choice]['left'])
            right_choices.append(choice_mapping[choice]['right'])
        
        # 計算準確率
        left_correct = np.array(left_choices) == np.array(left_stimuli)
        right_correct = np.array(right_choices) == np.array(right_stimuli)
        both_correct = left_correct & right_correct
        
        return {
            'subject_id': subject_id,
            'n_trials': len(subject_df),
            'choices': subject_df['Response'].values,
            'rt': subject_df['RT'].values,
            'stimuli': subject_df['Stimulus'].values,
            'left_stimuli': np.array(left_stimuli),
            'right_stimuli': np.array(right_stimuli),
            'left_choices': np.array(left_choices),
            'right_choices': np.array(right_choices),
            'left_correct': left_correct,
            'right_correct': right_correct,
            'accuracy': np.mean(both_correct),
            'left_accuracy': np.mean(left_correct),
            'right_accuracy': np.mean(right_correct)
        }
    
    def step1_estimate_single_lba(self, subject_data):
        """步驟1: 使用Single LBA估計左右通道的 match/mismatch drift rates"""
        
        print("\n📍 步驟1: Single LBA估計左右通道 match/mismatch drift rates")
        print("-" * 60)
        print("   使用固定參數:")
        for param, value in self.FIXED_PARAMS.items():
            print(f"     {param}: {value}")
        
        with pm.Model() as single_lba_model:
            
            # === 固定的共享參數 ===
            shared_threshold = self.FIXED_PARAMS['shared_threshold']
            shared_start_var = self.FIXED_PARAMS['shared_start_var']
            shared_ndt = self.FIXED_PARAMS['shared_ndt']
            shared_noise = self.FIXED_PARAMS['shared_noise']
            
            # === 改進的先驗設定 ===
            # 添加RT範圍約束
            min_drift = shared_threshold / (1.5 - shared_ndt)  # ≈ 0.47
            max_drift = shared_threshold / (0.3 - shared_ndt)  # ≈ 7.5
            
            # 左通道的 match/mismatch drift rates
            left_drift_match = pm.TruncatedNormal(
                'left_drift_match', 
                mu=2.0, sigma=0.8, 
                lower=min_drift, upper=max_drift
            )
            left_drift_mismatch = pm.TruncatedNormal(
                'left_drift_mismatch', 
                mu=0.4, sigma=0.3, 
                lower=0.05, upper=2.0
            )
            
            # 右通道的 match/mismatch drift rates
            right_drift_match = pm.TruncatedNormal(
                'right_drift_match', 
                mu=2.0, sigma=0.8, 
                lower=min_drift, upper=max_drift
            )
            right_drift_mismatch = pm.TruncatedNormal(
                'right_drift_mismatch', 
                mu=0.4, sigma=0.3, 
                lower=0.05, upper=2.0
            )
            
            # === 軟約束 ===
            # 偏好 match > mismatch
            pm.Potential(
                'left_match_advantage', 
                pm.math.log(1 + pm.math.exp(left_drift_match - left_drift_mismatch - 0.1))
            )
            pm.Potential(
                'right_match_advantage', 
                pm.math.log(1 + pm.math.exp(right_drift_match - right_drift_mismatch - 0.1))
            )
            
            # === 數據準備 ===
            left_stimuli = subject_data['left_stimuli']
            left_choices = subject_data['left_choices']
            right_stimuli = subject_data['right_stimuli']
            right_choices = subject_data['right_choices']
            rt = subject_data['rt']
            
            # === 計算likelihood ===
            left_likelihood = self._compute_side_likelihood_match_mismatch(
                left_choices, left_stimuli, rt,
                left_drift_match, left_drift_mismatch, 
                shared_threshold, shared_start_var, shared_ndt, shared_noise, 'left'
            )
            
            right_likelihood = self._compute_side_likelihood_match_mismatch(
                right_choices, right_stimuli, rt,
                right_drift_match, right_drift_mismatch,
                shared_threshold, shared_start_var, shared_ndt, shared_noise, 'right'
            )
            
            # === 添加到模型 ===
            pm.Potential('left_likelihood', left_likelihood)
            pm.Potential('right_likelihood', right_likelihood)
        
        # 執行MCMC採樣
        print("   🎲 執行Single LBA採樣...")
        print("   使用改進的MCMC設定：高target_accept, 更多chains, 更多iterations")
        
        with single_lba_model:
            single_trace = pm.sample(**self.mcmc_config)
        
        # 檢查收斂
        issues = diagnose_sampling_issues(single_trace)
        if issues:
            print(f"   ⚠️ Single LBA採樣有問題: {issues}")
            print("   💡 建議：檢查數據品質或進一步調整MCMC設定")
        else:
            print("   ✅ Single LBA採樣成功")
        
        # 提取drift rate後驗分布
        drift_estimates = self._extract_drift_estimates(single_trace)
        
        return single_lba_model, single_trace, drift_estimates
    
    def _compute_side_likelihood_match_mismatch(self, decisions, stimuli, rt, 
                                              drift_match, drift_mismatch, 
                                              threshold, start_var, ndt, noise, side_name):
        """計算單邊LBA likelihood - 使用 match/mismatch 設計"""
        
        from pytensor.tensor import erf
        
        # 更嚴格的參數約束
        drift_match = pm.math.maximum(drift_match, 0.1)
        drift_mismatch = pm.math.maximum(drift_mismatch, 0.05)
        # 確保 match > mismatch
        drift_match = pm.math.maximum(drift_match, drift_mismatch + 0.05)
        
        # 計算決策時間
        decision_time = pm.math.maximum(rt - ndt, 0.01)
        
        # 判斷匹配性
        stimulus_match = pm.math.eq(decisions, stimuli)
        
        # 設定drift rates
        v_chosen = pm.math.where(stimulus_match, drift_match, drift_mismatch)
        v_unchosen = pm.math.where(stimulus_match, drift_mismatch, drift_match)
        
        # 使用更穩定的LBA計算
        sqrt_t = pm.math.sqrt(decision_time)
        
        # Chosen累積器的z-scores (更保守的裁剪)
        z1_chosen = pm.math.clip(
            (v_chosen * decision_time - threshold) / (noise * sqrt_t), 
            -4.0, 4.0  # 更保守的裁剪範圍
        )
        z2_chosen = pm.math.clip(
            (v_chosen * decision_time - start_var) / (noise * sqrt_t), 
            -4.0, 4.0
        )
        
        # Unchosen累積器的z-score
        z1_unchosen = pm.math.clip(
            (v_unchosen * decision_time - threshold) / (noise * sqrt_t), 
            -4.0, 4.0
        )
        
        def safe_normal_cdf(x):
            """安全的正態CDF函數"""
            x_safe = pm.math.clip(x, -4.0, 4.0)
            return 0.5 * (1 + erf(x_safe / pm.math.sqrt(2)))
        
        def safe_normal_pdf(x):
            """安全的正態PDF函數"""
            x_safe = pm.math.clip(x, -4.0, 4.0)
            return pm.math.exp(-0.5 * x_safe**2) / pm.math.sqrt(2 * np.pi)
        
        # Chosen的似然計算
        chosen_cdf_term = safe_normal_cdf(z1_chosen) - safe_normal_cdf(z2_chosen)
        chosen_pdf_term = (safe_normal_pdf(z1_chosen) - safe_normal_pdf(z2_chosen)) / (noise * sqrt_t)
        
        # 確保CDF項為正
        chosen_cdf_term = pm.math.maximum(chosen_cdf_term, 1e-10)
        
        # 完整的chosen似然
        chosen_likelihood = pm.math.maximum(
            (v_chosen / start_var) * chosen_cdf_term + chosen_pdf_term / start_var,
            1e-10
        )
        
        # Unchosen的存活機率
        unchosen_survival = pm.math.maximum(1 - safe_normal_cdf(z1_unchosen), 1e-10)
        
        # 聯合似然
        joint_likelihood = chosen_likelihood * unchosen_survival
        joint_likelihood = pm.math.maximum(joint_likelihood, 1e-12)
        
        # 轉為對數似然
        log_likelihood = pm.math.log(joint_likelihood)
        
        # 處理無效值 - 更保守的裁剪
        log_likelihood_safe = pm.math.clip(log_likelihood, -50.0, 10.0)
        
        return pm.math.sum(log_likelihood_safe)
    
    def _extract_drift_estimates(self, trace):
        """提取drift rate的後驗估計 - 修正版"""
        
        summary = az.summary(trace)
        
        estimates = {
            # 原始drift rates
            'left_drift_match': summary.loc['left_drift_match', 'mean'],
            'left_drift_mismatch': summary.loc['left_drift_mismatch', 'mean'],
            'right_drift_match': summary.loc['right_drift_match', 'mean'],
            'right_drift_mismatch': summary.loc['right_drift_mismatch', 'mean'],
            
            # 有意義的指標
            'left_processing_speed': summary.loc['left_drift_match', 'mean'],
            'left_noise_level': summary.loc['left_drift_mismatch', 'mean'],
            'left_discrimination': (summary.loc['left_drift_match', 'mean'] - 
                                   summary.loc['left_drift_mismatch', 'mean']),
            'left_efficiency_ratio': (summary.loc['left_drift_match', 'mean'] / 
                                     summary.loc['left_drift_mismatch', 'mean']),
            
            'right_processing_speed': summary.loc['right_drift_match', 'mean'],
            'right_noise_level': summary.loc['right_drift_mismatch', 'mean'], 
            'right_discrimination': (summary.loc['right_drift_match', 'mean'] - 
                                    summary.loc['right_drift_mismatch', 'mean']),
            'right_efficiency_ratio': (summary.loc['right_drift_match', 'mean'] / 
                                      summary.loc['right_drift_mismatch', 'mean']),
            
            # 跨通道比較
            'discrimination_asymmetry': abs(
                (summary.loc['left_drift_match', 'mean'] - summary.loc['left_drift_mismatch', 'mean']) -
                (summary.loc['right_drift_match', 'mean'] - summary.loc['right_drift_mismatch', 'mean'])
            ),
            'processing_asymmetry': abs(
                summary.loc['left_drift_match', 'mean'] - summary.loc['right_drift_match', 'mean']
            )
        }
        
        # 添加固定參數
        estimates.update(self.FIXED_PARAMS)
        
        print(f"   📊 估計結果:")
        print(f"     左通道 - 處理速度: {estimates['left_processing_speed']:.3f}, 噪音: {estimates['left_noise_level']:.3f}")
        print(f"     右通道 - 處理速度: {estimates['right_processing_speed']:.3f}, 噪音: {estimates['right_noise_level']:.3f}")
        print(f"     左通道辨別能力: {estimates['left_discrimination']:.3f} (效率比: {estimates['left_efficiency_ratio']:.2f})")
        print(f"     右通道辨別能力: {estimates['right_discrimination']:.3f} (效率比: {estimates['right_efficiency_ratio']:.2f})")
        print(f"     通道不對稱性: 辨別={estimates['discrimination_asymmetry']:.3f}, 處理={estimates['processing_asymmetry']:.3f}")
        
        # 對稱性檢查
        self._check_symmetry_assumption(estimates)
        
        return estimates
    
    def _check_symmetry_assumption(self, estimates):
        """檢查對稱性假設"""
        
        print(f"\n   🔍 對稱性假設檢查:")
        
        # 處理速度差異
        processing_diff = abs(estimates['left_processing_speed'] - estimates['right_processing_speed'])
        discrimination_diff = estimates['discrimination_asymmetry']
        
        print(f"     處理速度差異: {processing_diff:.3f}")
        print(f"     辨別能力差異: {discrimination_diff:.3f}")
        
        # 判斷對稱性
        if processing_diff < 0.2 and discrimination_diff < 0.3:
            print(f"     ✅ 支持對稱性假設 (差異小)")
            symmetry_support = True
        elif processing_diff < 0.5 and discrimination_diff < 0.6:
            print(f"     ⚠️ 弱支持對稱性假設 (差異中等)")
            symmetry_support = False
        else:
            print(f"     ❌ 不支持對稱性假設 (差異大)")
            symmetry_support = False
        
        estimates['symmetry_supported'] = symmetry_support
        
        return symmetry_support

def quick_symmetry_validation(csv_file='GRT_LBA.csv', n_subjects=5):
    """快速對稱性驗證"""
    
    print("🔍 快速對稱性驗證")
    print("=" * 30)
    
    # 簡化的MCMC設定用於快速驗證
    quick_mcmc = {
        'draws': 300,
        'tune': 400,
        'chains': 4,
        'target_accept': 0.90,
        'progressbar': True
    }
    
    analyzer = EvidenceIntegrationComparison(mcmc_config=quick_mcmc)
    
    try:
        # 載入資料
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.1) & (df['RT'] <= 3.0)]
        
        # 選擇受試者
        subjects = df['participant'].unique()[:n_subjects]
        
        symmetry_results = []
        
        for subject_id in subjects:
            print(f"\n📍 驗證受試者 {subject_id}")
            
            try:
                subject_data = analyzer.prepare_subject_data(df, subject_id)
                
                if subject_data['accuracy'] < 0.5:
                    print(f"   ⚠️ 跳過：準確率過低 ({subject_data['accuracy']:.1%})")
                    continue
                
                # 只進行步驟1的估計
                _, _, drift_estimates = analyzer.step1_estimate_single_lba(subject_data)
                
                symmetry_results.append({
                    'subject_id': subject_id,
                    'processing_diff': estimates['processing_asymmetry'],
                    'discrimination_diff': estimates['discrimination_asymmetry'],
                    'symmetry_supported': estimates['symmetry_supported']
                })
                
            except Exception as e:
                print(f"   ❌ 受試者 {subject_id} 分析失敗: {e}")
                continue
        
        # 總結對稱性結果
        if symmetry_results:
            support_count = sum(1 for r in symmetry_results if r['symmetry_supported'])
            print(f"\n🎯 對稱性驗證結果:")
            print(f"   成功分析: {len(symmetry_results)}/{len(subjects)} 受試者")
            print(f"   支持對稱性: {support_count}/{len(symmetry_results)} 受試者")
            print(f"   支持比例: {support_count/len(symmetry_results)*100:.1f}%")
            
            if support_count / len(symmetry_results) >= 0.6:
                print(f"   ✅ 整體支持對稱性假設，可考慮使用共享先驗")
            else:
                print(f"   ❌ 整體不支持對稱性假設，建議保持獨立參數")
        
        return symmetry_results
        
    except Exception as e:
        print(f"❌ 對稱性驗證失敗: {e}")
        return None

def run_improved_single_lba_only(csv_file='GRT_LBA.csv', subject_id=None):
    """運行改進的單步驟LBA估計 (不進行證據整合)"""
    
    print("🚀 改進的Single LBA估計")
    print("=" * 40)
    
    try:
        # 載入資料
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.1) & (df['RT'] <= 3.0)]
        
        analyzer = EvidenceIntegrationComparison()
        
        # 選擇受試者
        if subject_id is None:
            # 自動選擇準確率最高的受試者
            best_subject = None
            best_accuracy = 0
            
            for sid in df['participant'].unique()[:10]:  # 檢查前10個
                temp_data = analyzer.prepare_subject_data(df, sid)
                if temp_data['accuracy'] > best_accuracy and temp_data['n_trials'] >= 50:
                    best_accuracy = temp_data['accuracy']
                    best_subject = sid
            
            if best_subject is None:
                print("❌ 找不到合適的受試者")
                return None
            
            subject_id = best_subject
            print(f"✅ 自動選擇受試者 {subject_id} (準確率: {best_accuracy:.1%})")
        
        # 準備數據並執行估計
        subject_data = analyzer.prepare_subject_data(df, subject_id)
        print(f"📊 受試者資料: {subject_data['n_trials']} trials, 準確率 {subject_data['accuracy']:.1%}")
        
        # 執行Single LBA估計
        model, trace, estimates = analyzer.step1_estimate_single_lba(subject_data)
        
        print(f"\n✅ Single LBA估計完成!")
        return {
            'success': True,
            'subject_id': subject_id,
            'estimates': estimates,
            'trace': trace,
            'model': model
        }
        
    except Exception as e:
        print(f"❌ 估計失敗: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🎯 修正版本選項:")
    print("1. 測試改進的Single LBA估計")
    print("2. 快速對稱性驗證 (5個受試者)")
    print("3. 單一受試者完整估計")
    
    try:
        choice = input("\n請選擇 (1-3): ").strip()
        
        if choice == '1':
            print("\n🧪 測試改進的Single LBA估計...")
            result = run_improved_single_lba_only()
            
        elif choice == '2':
            print("\n🔍 執行快速對稱性驗證...")
            results = quick_symmetry_validation()
            
        elif choice == '3':
            subject_id = input("請輸入受試者ID (或按Enter自動選擇): ").strip()
            if not subject_id:
                subject_id = None
            else:
                subject_id = int(subject_id)
            
            print(f"\n🚀 開始受試者分析...")
            result = run_improved_single_lba_only(subject_id=subject_id)
            
        else:
            print("無效選擇")
            
    except KeyboardInterrupt:
        print("\n⏹️ 分析被中斷")
    except Exception as e:
        print(f"\n💥 錯誤: {e}")
