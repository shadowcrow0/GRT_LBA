def diagnose_sampling_issues(trace, verbose=True):
    """
    診斷採樣問題 (內建版本，不依賴外部模組)
    """
    
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
        return [f"診斷失敗: {e}"]# drift_rate_evidence_integration.py - 基於Single LBA的證據整合模型比較
# 使用Bayes Factor比較Coactive vs Parallel AND假設

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class EvidenceIntegrationComparison:
    """基於Single LBA的證據整合模型比較器"""
    
    def __init__(self, mcmc_config=None):
        """
        初始化證據整合比較器
        
        Args:
            mcmc_config: MCMC配置字典
        """
        
        self.mcmc_config = self._setup_mcmc_config(mcmc_config)
        
        print("✅ 初始化證據整合模型比較器")
        print("   分析流程:")
        print("     1. Single LBA: 估計左右通道各自的drift rate")
        print("     2. Coactive: 證據相加 (drift_left + drift_right)")
        print("     3. Parallel AND: 證據取最大值 (max(drift_left, drift_right))")
        print("     4. Bayes Factor: 比較兩種整合假設")
    
    def _setup_mcmc_config(self, user_config):
        """設定MCMC配置"""
        
        default_config = {
            'draws': 200,           # 減少採樣數
            'tune': 300,            # 減少調整期
            'chains': 2,
            'cores': 1,
            'target_accept': 0.95,
            'max_treedepth': 12,    # 增加樹深度
            'init': 'jitter+adapt_diag',
            'random_seed': 42,
            'progressbar': True,
            'return_inferencedata': True
        }
        
        if user_config:
            default_config.update(user_config)
        
        return default_config
    
    def prepare_subject_data(self, df, subject_id):
        """
        準備受試者數據 (根據你的GRT_LBA.csv欄位結構)
        
        你的數據欄位:
        - Response: 最終選擇 (0-3)
        - RT: 反應時間
        - participant: 受試者ID  
        - Stimulus: 刺激類型 (0-3)
        - Chanel1, Chanel2: 左右通道信息
        """
        
        # 過濾受試者資料
        subject_df = df[df['participant'] == subject_id].copy()
        
        if len(subject_df) == 0:
            raise ValueError(f"找不到受試者 {subject_id} 的資料")
        
        # 刺激映射：將刺激編號轉為左右通道的線條類型
        stimulus_mapping = {
            0: {'left': 1, 'right': 0},  # 左對角，右垂直
            1: {'left': 1, 'right': 1},  # 左對角，右對角
            2: {'left': 0, 'right': 0},  # 左垂直，右垂直
            3: {'left': 0, 'right': 1}   # 左垂直，右對角
        }
        
        # 選擇映射：將選擇編號轉為左右通道的判斷
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
            
            # 分解刺激
            left_stimuli.append(stimulus_mapping[stimulus]['left'])
            right_stimuli.append(stimulus_mapping[stimulus]['right'])
            
            # 分解選擇
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
        """
        步驟1: 使用Single LBA估計左右通道的drift rate
        共享 threshold, start_var, ndt, noise 參數
        """
        
        print("\n📍 步驟1: Single LBA估計左右通道drift rate")
        print("-" * 50)
        
        with pm.Model() as single_lba_model:
            
            # === 共享參數 (左右通道共用) ===
            shared_threshold = pm.Gamma('shared_threshold', alpha=3.0, beta=3.5)
            shared_start_var = pm.Uniform('shared_start_var', lower=0.1, upper=0.7)
            shared_ndt = pm.Uniform('shared_ndt', lower=0.05, upper=0.6)
            shared_noise = pm.Gamma('shared_noise', alpha=2.5, beta=8.0)
            
            # === 左通道獨立的drift rate參數 ===
            left_drift_correct = pm.Gamma('left_drift_correct', alpha=2.5, beta=1.2)
            left_drift_incorrect = pm.Gamma('left_drift_incorrect', alpha=2.0, beta=3.0)
            
            # === 右通道獨立的drift rate參數 ===
            right_drift_correct = pm.Gamma('right_drift_correct', alpha=2.5, beta=1.2)
            right_drift_incorrect = pm.Gamma('right_drift_incorrect', alpha=2.0, beta=3.0)
            
            # === 數據準備 ===
            left_stimuli = subject_data['left_stimuli']
            left_choices = subject_data['left_choices']
            right_stimuli = subject_data['right_stimuli']
            right_choices = subject_data['right_choices']
            rt = subject_data['rt']
            
            # === 計算left通道likelihood (使用共享參數) ===
            left_likelihood = self._compute_side_likelihood(
                left_choices, left_stimuli, rt,
                left_drift_correct, left_drift_incorrect, shared_threshold,
                shared_start_var, shared_ndt, shared_noise, 'left'
            )
            
            # === 計算right通道likelihood (使用共享參數) ===
            right_likelihood = self._compute_side_likelihood(
                right_choices, right_stimuli, rt,
                right_drift_correct, right_drift_incorrect, shared_threshold,
                shared_start_var, shared_ndt, shared_noise, 'right'
            )
            
            # === 添加到模型 ===
            pm.Potential('left_likelihood', left_likelihood)
            pm.Potential('right_likelihood', right_likelihood)
        
        # 執行MCMC採樣
        print("   🎲 執行Single LBA採樣...")
        with single_lba_model:
            single_trace = pm.sample(**self.mcmc_config)
        
        # 檢查收斂
        issues = diagnose_sampling_issues(single_trace)
        if issues:
            print(f"   ⚠️ Single LBA採樣有問題: {issues}")
        else:
            print("   ✅ Single LBA採樣成功")
        
        # 提取drift rate後驗分布
        drift_estimates = self._extract_drift_estimates(single_trace)
        
        return single_lba_model, single_trace, drift_estimates
    
    def step2_test_evidence_integration(self, subject_data, drift_estimates):
        """
        步驟2: 使用估計的drift rate測試兩種證據整合假設
        """
        
        print("\n📍 步驟2: 測試證據整合假設")
        print("-" * 50)
        
        # 首先從數據估計視覺特徵的處理難度係數
        difficulty_coefficients = self._estimate_visual_difficulty_coefficients(subject_data, drift_estimates)
        
        # 2A. Coactive模型
        print("   🔬 測試 Coactive 假設 (證據相加)...")
        coactive_model, coactive_trace = self._create_coactive_integration_model(
            subject_data, drift_estimates, difficulty_coefficients)
        
        # 2B. Parallel AND模型  
        print("   🔬 測試 Parallel AND 假設 (證據取最大值)...")
        parallel_and_model, parallel_and_trace = self._create_parallel_and_integration_model(
            subject_data, drift_estimates, difficulty_coefficients)
        
        return {
            'coactive_model': coactive_model,
            'coactive_trace': coactive_trace,
            'parallel_and_model': parallel_and_model,
            'parallel_and_trace': parallel_and_trace,
            'difficulty_coefficients': difficulty_coefficients
        }
    
    def _create_coactive_integration_model(self, subject_data, drift_estimates, difficulty_coefficients):
        """創建Coactive證據整合模型"""
        
        with pm.Model() as coactive_model:
            
            # === 使用估計的左右通道drift rate (固定值) ===
            left_drift = drift_estimates['left_drift_mean']   # 左通道的drift rate
            right_drift = drift_estimates['right_drift_mean'] # 右通道的drift rate
            
            print(f"     使用估計的drift rate: 左={left_drift:.3f}, 右={right_drift:.3f}")
            
            # === 固定的四選一決策參數 ===
            choice_threshold = 1.0    # 固定閾值
            choice_ndt = 0.2          # 固定非決策時間
            choice_noise = 0.3        # 固定噪音
            
            # === 計算Coactive假設下的四選一drift rates ===
            choices = subject_data['choices']
            rt = subject_data['rt']
            
            # 根據Coactive假設和數據驅動的難度係數計算每個選項的drift rate
            coactive_drift_rates = self._compute_coactive_choice_drifts(
                left_drift, right_drift, difficulty_coefficients)
            
            print(f"     Coactive drift rates: {coactive_drift_rates}")
            
            # === 計算四選一選擇的likelihood ===
            coactive_likelihood = self._compute_four_choice_likelihood(
                choices, rt, coactive_drift_rates, choice_threshold, choice_ndt, choice_noise
            )
            
            pm.Potential('coactive_likelihood', coactive_likelihood)
            
            # 添加觀察模型以便計算WAIC
            trial_drift_rates = coactive_drift_rates[choices]  # 每個trial對應的drift rate
            predicted_rt = choice_ndt + choice_threshold / trial_drift_rates
            pm.Normal('coactive_obs_rt', mu=predicted_rt, sigma=choice_noise, observed=rt)
        
        # 執行採樣
        with coactive_model:
            coactive_trace = pm.sample(**self.mcmc_config)
        
        issues = diagnose_sampling_issues(coactive_trace)
        if not issues:
            print("     ✅ Coactive模型採樣成功")
        else:
            print(f"     ⚠️ Coactive模型採樣問題: {issues}")
        
        return coactive_model, coactive_trace
    
    def _create_parallel_and_integration_model(self, subject_data, drift_estimates, difficulty_coefficients):
        """創建Parallel AND證據整合模型"""
        
        with pm.Model() as parallel_and_model:
            
            # === 使用估計的左右通道drift rate (固定值) ===
            left_drift = drift_estimates['left_drift_mean']   # 左通道的drift rate
            right_drift = drift_estimates['right_drift_mean'] # 右通道的drift rate
            
            print(f"     使用估計的drift rate: 左={left_drift:.3f}, 右={right_drift:.3f}")
            
            # === 固定的四選一決策參數 ===
            choice_threshold = 1.0    # 固定閾值
            choice_ndt = 0.2          # 固定非決策時間
            choice_noise = 0.3        # 固定噪音
            
            # === 計算Parallel AND假設下的四選一drift rates ===
            choices = subject_data['choices']
            rt = subject_data['rt']
            
            # 根據Parallel AND假設和數據驅動的難度係數計算每個選項的drift rate
            parallel_drift_rates = self._compute_parallel_and_choice_drifts(
                left_drift, right_drift, difficulty_coefficients)
            
            print(f"     Parallel AND drift rates: {parallel_drift_rates}")
            
            # === 計算四選一選擇的likelihood ===
            parallel_likelihood = self._compute_four_choice_likelihood(
                choices, rt, parallel_drift_rates, choice_threshold, choice_ndt, choice_noise
            )
            
            pm.Potential('parallel_likelihood', parallel_likelihood)
            
            # 添加觀察模型以便計算WAIC
            trial_drift_rates = parallel_drift_rates[choices]  # 每個trial對應的drift rate
            predicted_rt = choice_ndt + choice_threshold / trial_drift_rates
            pm.Normal('parallel_obs_rt', mu=predicted_rt, sigma=choice_noise, observed=rt)
        
        # 執行採樣
        with parallel_and_model:
            parallel_trace = pm.sample(**self.mcmc_config)
        
        issues = diagnose_sampling_issues(parallel_trace)
        if not issues:
            print("     ✅ Parallel AND模型採樣成功")
        else:
            print(f"     ⚠️ Parallel AND模型採樣問題: {issues}")
        
        return parallel_and_model, parallel_trace
    
    def _compute_coactive_choice_drifts(self, left_drift, right_drift, difficulty_coefficients):
        """
        計算Coactive假設下四個選項的drift rates
        使用從數據估計的視覺特徵處理難度係數
        
        選項對應:
        0: 左\右| (左對角 + 右垂直)
        1: 左\右/ (左對角 + 右對角)  
        2: 左|右| (左垂直 + 右垂直)
        3: 左|右/ (左垂直 + 右對角)
        
        Coactive假設: 兩個通道的處理能力相加
        """
        
        # 提取數據驅動的難度係數
        left_vertical_coeff = difficulty_coefficients['left_vertical']
        left_diagonal_coeff = difficulty_coefficients['left_diagonal']
        right_vertical_coeff = difficulty_coefficients['right_vertical']
        right_diagonal_coeff = difficulty_coefficients['right_diagonal']
        
        # 計算每個通道對不同視覺特徵的有效處理能力
        left_vertical_strength = left_drift * left_vertical_coeff
        left_diagonal_strength = left_drift * left_diagonal_coeff
        right_vertical_strength = right_drift * right_vertical_coeff
        right_diagonal_strength = right_drift * right_diagonal_coeff
        
        # Coactive: 相加 (兩個通道協同工作)
        drift_rates = np.array([
            left_diagonal_strength + right_vertical_strength,   # 選項0: 左\右|
            left_diagonal_strength + right_diagonal_strength,   # 選項1: 左\右/
            left_vertical_strength + right_vertical_strength,   # 選項2: 左|右|
            left_vertical_strength + right_diagonal_strength    # 選項3: 左|右/
        ])
        
        return drift_rates
    
    def _compute_parallel_and_choice_drifts(self, left_drift, right_drift, difficulty_coefficients):
        """
        計算Parallel AND假設下四個選項的drift rates
        使用從數據估計的視覺特徵處理難度係數
        
        Parallel AND假設: 取最大值 (較快的通道決定整體速度)
        """
        
        # 提取數據驅動的難度係數
        left_vertical_coeff = difficulty_coefficients['left_vertical']
        left_diagonal_coeff = difficulty_coefficients['left_diagonal']
        right_vertical_coeff = difficulty_coefficients['right_vertical']
        right_diagonal_coeff = difficulty_coefficients['right_diagonal']
        
        # 計算每個通道對不同視覺特徵的有效處理能力
        left_vertical_strength = left_drift * left_vertical_coeff
        left_diagonal_strength = left_drift * left_diagonal_coeff
        right_vertical_strength = right_drift * right_vertical_coeff
        right_diagonal_strength = right_drift * right_diagonal_coeff
        
        # Parallel AND: 取最大值 (最快的通道決定)
        drift_rates = np.array([
            max(left_diagonal_strength, right_vertical_strength),   # 選項0: 左\右|
            max(left_diagonal_strength, right_diagonal_strength),   # 選項1: 左\右/
            max(left_vertical_strength, right_vertical_strength),   # 選項2: 左|右|
            max(left_vertical_strength, right_diagonal_strength)    # 選項3: 左|右/
        ])
        
        return drift_rates
    
    def _compute_four_choice_likelihood(self, choices, rt, drift_rates, threshold, ndt, noise):
        """
        計算四選一選擇的likelihood
        
        Args:
            choices: 選擇陣列 (0-3)
            rt: 反應時間陣列
            drift_rates: 四個選項的drift rates [drift_0, drift_1, drift_2, drift_3]
            threshold, ndt, noise: LBA參數 (固定值)
        """
        
        # 為每個trial分配對應的drift rate
        trial_drift_rates = drift_rates[choices]
        
        # 計算決策時間
        decision_time = np.maximum(rt - ndt, 0.01)
        
        # 簡化的LBA likelihood計算
        # 假設每個選項都有相同的參數，只有drift rate不同
        predicted_rt = ndt + threshold / trial_drift_rates
        
        # 計算RT likelihood (使用正態分佈近似)
        rt_likelihood = np.sum(
            -0.5 * ((rt - predicted_rt) / noise) ** 2 - np.log(noise * np.sqrt(2 * np.pi))
        )
        
        return rt_likelihood
    
    def _extract_drift_estimates(self, trace):
        """提取drift rate的後驗估計 (配合共享參數設計)"""
        
        summary = az.summary(trace)
        
        # 提取基本drift rate參數
        left_correct = summary.loc['left_drift_correct', 'mean']
        left_incorrect = summary.loc['left_drift_incorrect', 'mean']
        right_correct = summary.loc['right_drift_correct', 'mean']
        right_incorrect = summary.loc['right_drift_incorrect', 'mean']
        
        return {
            'left_drift_mean': (left_correct + left_incorrect) / 2,
            'right_drift_mean': (right_correct + right_incorrect) / 2,
            'left_drift_correct': left_correct,
            'left_drift_incorrect': left_incorrect,
            'right_drift_correct': right_correct,
            'right_drift_incorrect': right_incorrect,
            # 添加共享參數
            'shared_threshold': summary.loc['shared_threshold', 'mean'],
            'shared_start_var': summary.loc['shared_start_var', 'mean'],
            'shared_ndt': summary.loc['shared_ndt', 'mean'],
            'shared_noise': summary.loc['shared_noise', 'mean']
        }
    
    def _estimate_visual_difficulty_coefficients(self, subject_data, drift_estimates):
        """
        從數據估計垂直線 vs 對角線的相對處理難度係數
        
        方法: 分析單通道對不同視覺特徵的表現差異
        """
        
        print("     🔍 從數據估計視覺特徵處理難度...")
        
        # 分析左通道對垂直線 vs 對角線的表現
        left_stimuli = subject_data['left_stimuli']
        left_choices = subject_data['left_choices']
        left_correct = subject_data['left_correct']
        
        # 左通道：垂直線 (0) vs 對角線 (1) 的準確率
        left_vertical_trials = left_stimuli == 0
        left_diagonal_trials = left_stimuli == 1
        
        if np.sum(left_vertical_trials) > 0:
            left_vertical_acc = np.mean(left_correct[left_vertical_trials])
        else:
            left_vertical_acc = 0.5
            
        if np.sum(left_diagonal_trials) > 0:
            left_diagonal_acc = np.mean(left_correct[left_diagonal_trials])
        else:
            left_diagonal_acc = 0.5
        
        # 分析右通道對垂直線 vs 對角線的表現
        right_stimuli = subject_data['right_stimuli']
        right_choices = subject_data['right_choices']
        right_correct = subject_data['right_correct']
        
        right_vertical_trials = right_stimuli == 0
        right_diagonal_trials = right_stimuli == 1
        
        if np.sum(right_vertical_trials) > 0:
            right_vertical_acc = np.mean(right_correct[right_vertical_trials])
        else:
            right_vertical_acc = 0.5
            
        if np.sum(right_diagonal_trials) > 0:
            right_diagonal_acc = np.mean(right_correct[right_diagonal_trials])
        else:
            right_diagonal_acc = 0.5
        
        # 計算相對難度係數 (以垂直線為基準 = 1.0)
        # 係數 = 該特徵準確率 / 垂直線準確率
        
        # 左通道係數
        if left_vertical_acc > 0:
            left_vertical_coeff = 1.0  # 基準
            left_diagonal_coeff = left_diagonal_acc / left_vertical_acc
        else:
            left_vertical_coeff = 1.0
            left_diagonal_coeff = 1.0
        
        # 右通道係數
        if right_vertical_acc > 0:
            right_vertical_coeff = 1.0  # 基準
            right_diagonal_coeff = right_diagonal_acc / right_vertical_acc
        else:
            right_vertical_coeff = 1.0
            right_diagonal_coeff = 1.0
        
        # 限制係數範圍，避免極端值
        left_diagonal_coeff = np.clip(left_diagonal_coeff, 0.5, 1.5)
        right_diagonal_coeff = np.clip(right_diagonal_coeff, 0.5, 1.5)
        
        coefficients = {
            'left_vertical': left_vertical_coeff,
            'left_diagonal': left_diagonal_coeff,
            'right_vertical': right_vertical_coeff,
            'right_diagonal': right_diagonal_coeff
        }
        
        print(f"     📊 估計的難度係數:")
        print(f"       左通道 - 垂直線: {left_vertical_coeff:.3f}, 對角線: {left_diagonal_coeff:.3f}")
        print(f"       右通道 - 垂直線: {right_vertical_coeff:.3f}, 對角線: {right_diagonal_coeff:.3f}")
        print(f"       左通道準確率 - 垂直線: {left_vertical_acc:.1%}, 對角線: {left_diagonal_acc:.1%}")
        print(f"       右通道準確率 - 垂直線: {right_vertical_acc:.1%}, 對角線: {right_diagonal_acc:.1%}")
        
        return coefficients
    
    def _compute_side_likelihood(self, decisions, stimuli, rt, drift_correct, drift_incorrect, 
                               threshold, start_var, ndt, noise, side_name):
        """
        計算單邊LBA likelihood (使用完整的LBA公式)
        這個函數計算2選擇LBA的對數似然，用於左右通道各自的垂直線vs對角線判斷
        
        Args:
            decisions: 決策陣列 (0=垂直, 1=對角)  
            stimuli: 刺激陣列 (0=垂直, 1=對角)
            rt: 反應時間陣列
            drift_correct, drift_incorrect: 正確和錯誤的drift rates
            threshold, start_var, ndt, noise: LBA參數
            side_name: 通道名稱 (用於調試)
        """
        
        from pytensor.tensor import erf
        
        # 參數約束
        drift_correct = pm.math.maximum(drift_correct, 0.1)
        drift_incorrect = pm.math.maximum(drift_incorrect, 0.05)
        drift_correct = pm.math.maximum(drift_correct, drift_incorrect + 0.05)
        threshold = pm.math.maximum(threshold, 0.1)
        start_var = pm.math.maximum(start_var, 0.05)
        ndt = pm.math.maximum(ndt, 0.05)
        noise = pm.math.maximum(noise, 0.1)
        
        # 計算決策時間
        decision_time = pm.math.maximum(rt - ndt, 0.01)
        
        # 判斷正確性
        stimulus_correct = pm.math.eq(decisions, stimuli)
        
        # 設定winner和loser的漂移率
        v_winner = pm.math.where(stimulus_correct, drift_correct, drift_incorrect)
        v_loser = pm.math.where(stimulus_correct, drift_incorrect, drift_correct)
        
        # 確保drift rates有最小值
        v_winner = pm.math.maximum(v_winner, 0.1)
        v_loser = pm.math.maximum(v_loser, 0.05)
        
        # 使用完整的LBA公式計算2選擇似然
        sqrt_t = pm.math.sqrt(decision_time)
        
        # Winner累積器的z-scores
        z1_winner = pm.math.clip(
            (v_winner * decision_time - threshold) / (noise * sqrt_t), 
            -4.5, 4.5
        )
        z2_winner = pm.math.clip(
            (v_winner * decision_time - start_var) / (noise * sqrt_t), 
            -4.5, 4.5
        )
        
        # Loser累積器的z-score
        z1_loser = pm.math.clip(
            (v_loser * decision_time - threshold) / (noise * sqrt_t), 
            -4.5, 4.5
        )
        
        def safe_normal_cdf(x):
            """安全的正態CDF函數"""
            x_safe = pm.math.clip(x, -4.5, 4.5)
            return 0.5 * (1 + erf(x_safe / pm.math.sqrt(2)))
        
        def safe_normal_pdf(x):
            """安全的正態PDF函數"""
            x_safe = pm.math.clip(x, -4.5, 4.5)
            return pm.math.exp(-0.5 * x_safe**2) / pm.math.sqrt(2 * np.pi)
        
        # Winner的似然計算
        winner_cdf_term = safe_normal_cdf(z1_winner) - safe_normal_cdf(z2_winner)
        winner_pdf_term = (safe_normal_pdf(z1_winner) - safe_normal_pdf(z2_winner)) / (noise * sqrt_t)
        
        # 確保CDF項為正
        winner_cdf_term = pm.math.maximum(winner_cdf_term, 1e-10)
        
        # 完整的winner似然
        winner_likelihood = pm.math.maximum(
            (v_winner / start_var) * winner_cdf_term + winner_pdf_term / start_var,
            1e-10
        )
        
        # Loser的存活機率
        loser_survival = pm.math.maximum(1 - safe_normal_cdf(z1_loser), 1e-10)
        
        # 聯合似然：winner的PDF × loser的survival
        joint_likelihood = winner_likelihood * loser_survival
        joint_likelihood = pm.math.maximum(joint_likelihood, 1e-12)
        
        # 轉為對數似然
        log_likelihood = pm.math.log(joint_likelihood)
        
        # 處理無效值 - 直接裁剪極端值
        log_likelihood_safe = pm.math.clip(log_likelihood, -100.0, 10.0)
        
        # 裁剪極端值並求和
        return pm.math.sum(pm.math.clip(log_likelihood_safe, -100.0, 10.0))
    
    def step3_compute_bayes_factors(self, integration_results):
        """
        步驟3: 計算Bayes Factor進行模型比較
        """
        
        print("\n📍 步驟3: Bayes Factor模型比較")
        print("-" * 50)
        
        coactive_trace = integration_results['coactive_trace']
        parallel_trace = integration_results['parallel_and_trace']
        
        try:
            # 方法1: 嘗試計算WAIC
            try:
                coactive_waic = az.waic(coactive_trace)
                parallel_waic = az.waic(parallel_trace)
                
                waic_diff = coactive_waic.waic - parallel_waic.waic
                waic_available = True
                
            except Exception as e:
                print(f"   ⚠️ WAIC計算失敗: {e}")
                waic_available = False
            
            # 方法2: 嘗試計算LOO
            try:
                coactive_loo = az.loo(coactive_trace)
                parallel_loo = az.loo(parallel_trace)
                
                loo_diff = coactive_loo.loo - parallel_loo.loo
                loo_available = True
                
            except Exception as e:
                print(f"   ⚠️ LOO計算失敗: {e}")
                loo_available = False
            
            # 方法3: 備用計算 - 使用似然估計
            if not waic_available and not loo_available:
                print("   🔄 使用備用方法計算模型比較...")
                
                # 計算平均對數似然
                try:
                    coactive_logp = coactive_trace.log_likelihood['coactive_obs_rt'].mean()
                    parallel_logp = parallel_trace.log_likelihood['parallel_obs_rt'].mean()
                    
                    likelihood_diff = float(coactive_logp.sum() - parallel_logp.sum())
                    
                    if likelihood_diff < -10:
                        conclusion = "強烈支持 Coactive 假設"
                    elif likelihood_diff < 0:
                        conclusion = "傾向支持 Coactive 假設"
                    elif likelihood_diff > 10:
                        conclusion = "強烈支持 Parallel AND 假設"
                    else:
                        conclusion = "傾向支持 Parallel AND 假設"
                    
                    comparison_results = {
                        'method': 'likelihood_comparison',
                        'likelihood_diff': likelihood_diff,
                        'conclusion': conclusion
                    }
                    
                except Exception as e:
                    print(f"   ❌ 備用方法也失敗: {e}")
                    return None
            
            else:
                # 使用WAIC或LOO結果
                if waic_available:
                    if waic_diff < -2:
                        waic_conclusion = "強烈支持 Coactive 假設"
                    elif waic_diff < 0:
                        waic_conclusion = "傾向支持 Coactive 假設"
                    elif waic_diff > 2:
                        waic_conclusion = "強烈支持 Parallel AND 假設"
                    else:
                        waic_conclusion = "傾向支持 Parallel AND 假設"
                else:
                    waic_diff = None
                    waic_conclusion = "WAIC無法計算"
                
                if loo_available:
                    if loo_diff < -2:
                        loo_conclusion = "強烈支持 Coactive 假設"
                    elif loo_diff < 0:
                        loo_conclusion = "傾向支持 Coactive 假設"
                    elif loo_diff > 2:
                        loo_conclusion = "強烈支持 Parallel AND 假設"
                    else:
                        loo_conclusion = "傾向支持 Parallel AND 假設"
                else:
                    loo_diff = None
                    loo_conclusion = "LOO無法計算"
                
                comparison_results = {
                    'method': 'information_criteria',
                    'coactive_waic': coactive_waic.waic if waic_available else None,
                    'parallel_waic': parallel_waic.waic if waic_available else None,
                    'waic_diff': waic_diff,
                    'waic_conclusion': waic_conclusion,
                    'coactive_loo': coactive_loo.loo if loo_available else None,
                    'parallel_loo': parallel_loo.loo if loo_available else None,
                    'loo_diff': loo_diff,
                    'loo_conclusion': loo_conclusion
                }
            
            print(f"   📊 模型比較結果:")
            if comparison_results['method'] == 'information_criteria':
                if waic_available:
                    print(f"      Coactive WAIC:    {coactive_waic.waic:.2f}")
                    print(f"      Parallel WAIC:    {parallel_waic.waic:.2f}")
                    print(f"      WAIC 差異:        {waic_diff:.2f}")
                    print(f"      WAIC 結論:        {waic_conclusion}")
                if loo_available:
                    print(f"      LOO 差異:         {loo_diff:.2f}")
                    print(f"      LOO 結論:         {loo_conclusion}")
            else:
                print(f"      似然差異:         {comparison_results['likelihood_diff']:.2f}")
                print(f"      結論:            {comparison_results['conclusion']}")
            
            return comparison_results
            
        except Exception as e:
            print(f"   ❌ Bayes Factor計算失敗: {e}")
            return None

def _get_final_conclusion(bayes_results):
    """獲取最終結論"""
    if not bayes_results:
        return "無法判斷"
    
    if bayes_results['method'] == 'information_criteria':
        if 'waic_conclusion' in bayes_results and bayes_results['waic_conclusion'] != "WAIC無法計算":
            return bayes_results['waic_conclusion']
        elif 'loo_conclusion' in bayes_results and bayes_results['loo_conclusion'] != "LOO無法計算":
            return bayes_results['loo_conclusion']
        else:
            return "無法判斷"
    else:
        return bayes_results.get('conclusion', '無法判斷')

def run_evidence_integration_analysis(csv_file='GRT_LBA.csv', subject_id=None, min_accuracy=0.5):
    """
    執行完整的證據整合分析
    
    Args:
        csv_file: 數據檔案路徑
        subject_id: 受試者ID，如果None則自動選擇符合條件的受試者
        min_accuracy: 最低準確率要求 (預設50%)
    """
    
    print("🚀 證據整合假設檢驗分析")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 載入資料 - 直接使用pandas，不依賴DataProcessor
        print("📂 載入資料...")
        df = pd.read_csv(csv_file)
        print(f"✅ 載入 {len(df)} 個試驗")
        
        # 基本資料清理
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.1) & (df['RT'] <= 3.0)]
        df = df[df['Response'].isin([0, 1, 2, 3])]
        df = df[df['Stimulus'].isin([0, 1, 2, 3])]
        
        print(f"✅ 清理後: {len(df)} 個試驗")
        print(f"   受試者數: {df['participant'].nunique()}")
        
        # 創建分析器
        analyzer = EvidenceIntegrationComparison()
        
        # 選擇受試者並檢查準確率
        if subject_id is None:
            # 自動選擇符合條件的受試者
            suitable_subjects = []
            
            print(f"\n🔍 尋找準確率 ≥ {min_accuracy:.0%} 的受試者...")
            
            for sid in df['participant'].unique():
                temp_data = analyzer.prepare_subject_data(df, sid)
                if temp_data['accuracy'] >= min_accuracy and temp_data['n_trials'] >= 50:
                    suitable_subjects.append({
                        'id': sid,
                        'accuracy': temp_data['accuracy'],
                        'n_trials': temp_data['n_trials']
                    })
            
            if not suitable_subjects:
                print(f"❌ 找不到準確率 ≥ {min_accuracy:.0%} 且試驗數 ≥ 50 的受試者")
                print("   建議降低準確率要求或檢查數據品質")
                return {
                    'success': False,
                    'error': f'No subjects with accuracy >= {min_accuracy:.0%}',
                    'total_time': time.time() - start_time
                }
            
            # 選擇準確率最高的受試者
            best_subject = max(suitable_subjects, key=lambda x: x['accuracy'])
            subject_id = best_subject['id']
            
            print(f"✅ 找到 {len(suitable_subjects)} 位符合條件的受試者")
            print(f"   自動選擇受試者 {subject_id} (準確率: {best_subject['accuracy']:.1%}, 試驗數: {best_subject['n_trials']})")
            
        else:
            # 檢查指定受試者是否符合條件
            temp_data = analyzer.prepare_subject_data(df, subject_id)
            if temp_data['accuracy'] < min_accuracy:
                print(f"❌ 受試者 {subject_id} 準確率 {temp_data['accuracy']:.1%} < {min_accuracy:.0%}")
                print("   跳過分析，建議選擇其他受試者")
                return {
                    'success': False,
                    'error': f'Subject {subject_id} accuracy {temp_data["accuracy"]:.1%} below threshold',
                    'total_time': time.time() - start_time
                }
        
        # 準備數據
        subject_data = analyzer.prepare_subject_data(df, subject_id)
        print(f"\n📊 受試者 {subject_id} 數據分析:")
        print(f"   試驗數: {subject_data['n_trials']}")
        print(f"   整體準確率: {subject_data['accuracy']:.1%}")
        print(f"   左通道準確率: {subject_data['left_accuracy']:.1%}")
        print(f"   右通道準確率: {subject_data['right_accuracy']:.1%}")
        
        # 檢查數據分布
        print(f"   刺激分布: {np.bincount(subject_data['stimuli'])}")
        print(f"   選擇分布: {np.bincount(subject_data['choices'])}")
        
        # 步驟1: Single LBA估計
        print(f"\n📍 開始三步驟分析...")
        single_model, single_trace, drift_estimates = analyzer.step1_estimate_single_lba(subject_data)
        
        # 步驟2: 證據整合測試
        integration_results = analyzer.step2_test_evidence_integration(subject_data, drift_estimates)
        
        # 步驟3: Bayes Factor比較
        bayes_results = analyzer.step3_compute_bayes_factors(integration_results)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 證據整合分析完成!")
        print(f"⏱️ 總時間: {total_time/60:.1f} 分鐘")
        print(f"🏆 最終結論: {_get_final_conclusion(bayes_results)}")
        print("="*60)
        
        return {
            'subject_id': subject_id,
            'subject_accuracy': subject_data['accuracy'],
            'n_trials': subject_data['n_trials'],
            'drift_estimates': drift_estimates,
            'integration_results': integration_results,
            'bayes_results': bayes_results,
            'total_time': total_time,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }

def run_batch_analysis(csv_file='GRT_LBA.csv', max_subjects=5, min_accuracy=0.5):
    """
    批次分析多個受試者
    
    Args:
        csv_file: 數據檔案路徑
        max_subjects: 最大分析受試者數
        min_accuracy: 最低準確率要求
    """
    
    print("🚀 批次證據整合分析")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    try:
        # 載入資料
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.1) & (df['RT'] <= 3.0)]
        df = df[df['Response'].isin([0, 1, 2, 3])]
        df = df[df['Stimulus'].isin([0, 1, 2, 3])]
        
        # 篩選符合條件的受試者
        analyzer = EvidenceIntegrationComparison()
        suitable_subjects = []
        
        for sid in df['participant'].unique():
            temp_data = analyzer.prepare_subject_data(df, sid)
            if temp_data['accuracy'] >= min_accuracy and temp_data['n_trials'] >= 50:
                suitable_subjects.append({
                    'id': sid,
                    'accuracy': temp_data['accuracy'],
                    'n_trials': temp_data['n_trials']
                })
        
        # 按準確率排序，選擇最好的
        suitable_subjects.sort(key=lambda x: x['accuracy'], reverse=True)
        selected_subjects = suitable_subjects[:max_subjects]
        
        print(f"📊 符合條件的受試者: {len(suitable_subjects)}")
        print(f"   選擇分析: {len(selected_subjects)} 位")
        
        # 逐一分析
        for i, subject_info in enumerate(selected_subjects, 1):
            print(f"\n{'='*40}")
            print(f"📍 分析 {i}/{len(selected_subjects)}: 受試者 {subject_info['id']}")
            print(f"   預期準確率: {subject_info['accuracy']:.1%}")
            
            result = run_evidence_integration_analysis(
                csv_file, 
                subject_id=subject_info['id'], 
                min_accuracy=min_accuracy
            )
            
            results.append(result)
            
            if result['success']:
                print(f"   ✅ 完成: {result['bayes_results']['waic_conclusion'] if result['bayes_results'] else '無法判斷'}")
            else:
                print(f"   ❌ 失敗: {result['error']}")
        
        # 統計結果
        successful_results = [r for r in results if r['success']]
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 批次分析完成!")
        print(f"⏱️ 總時間: {total_time/60:.1f} 分鐘")
        print(f"✅ 成功率: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        
        if successful_results:
            # 統計結論
            coactive_count = sum(1 for r in successful_results 
                               if r['bayes_results'] and 'Coactive' in r['bayes_results']['waic_conclusion'])
            parallel_count = sum(1 for r in successful_results 
                               if r['bayes_results'] and 'Parallel' in r['bayes_results']['waic_conclusion'])
            
            print(f"🏆 結論統計:")
            print(f"   支持 Coactive: {coactive_count} 位")
            print(f"   支持 Parallel AND: {parallel_count} 位")
            
        print("="*60)
        
        return {
            'results': results,
            'successful_count': len(successful_results),
            'total_time': total_time,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ 批次分析失敗: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }

if __name__ == "__main__":
    print("🎯 證據整合假設檢驗:")
    print("=" * 40)
    print("這個分析將會:")
    print("1. 用Single LBA估計左右通道drift rate")
    print("2. 測試Coactive (相加) vs Parallel AND (最大值) 假設")
    print("3. 用Bayes Factor判斷哪個假設更符合數據")
    print()
    print("選項:")
    print("1. 單一受試者分析 (自動選擇)")
    print("2. 指定受試者分析")
    print("3. 批次分析 (多個受試者)")
    
    try:
        choice = input("\n請選擇 (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 開始單一受試者分析 (自動選擇準確率最高者)...")
            result = run_evidence_integration_analysis()
            
            if result['success']:
                print("\n✅ 分析成功完成!")
            else:
                print("\n❌ 分析失敗")
                
        elif choice == '2':
            subject_id = int(input("請輸入受試者ID: "))
            print(f"\n🚀 開始受試者 {subject_id} 分析...")
            result = run_evidence_integration_analysis(subject_id=subject_id)
            
            if result['success']:
                print("\n✅ 分析成功完成!")
            else:
                print("\n❌ 分析失敗")
                
        elif choice == '3':
            max_subjects = int(input("請輸入最大分析受試者數 (建議3-5): ") or "3")
            print(f"\n🚀 開始批次分析 (最多{max_subjects}位受試者)...")
            result = run_batch_analysis(max_subjects=max_subjects)
            
            if result['success']:
                print("\n✅ 批次分析成功完成!")
            else:
                print("\n❌ 批次分析失敗")
        else:
            print("無效選擇，分析取消")
            
    except KeyboardInterrupt:
        print("\n⏹️ 分析被使用者中斷")
    except Exception as e:
        print(f"\n💥 未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
