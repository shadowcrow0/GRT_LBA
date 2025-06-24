# drift_rate_evidence_integration.py - 基於Single LBA的證據整合模型比較
# 使用Bayes Factor比較Coactive vs Parallel AND假設

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data_utils import DataProcessor
from grt_model_comparison import get_robust_mcmc_config, diagnose_sampling_issues

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
            'draws': 500,
            'tune': 500,
            'chains': 2,
            'cores': 1,
            'target_accept': 0.90,
            'max_treedepth': 10,
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
        """
        
        print("\n📍 步驟1: Single LBA估計左右通道drift rate")
        print("-" * 50)
        
        with pm.Model() as single_lba_model:
            
            # === 左通道LBA模型 ===
            left_drift_correct = pm.Gamma('left_drift_correct', alpha=2.5, beta=1.2)
            left_drift_incorrect = pm.Gamma('left_drift_incorrect', alpha=2.0, beta=3.0)
            left_threshold = pm.Gamma('left_threshold', alpha=3.0, beta=3.5)
            left_start_var = pm.Uniform('left_start_var', lower=0.1, upper=0.7)
            left_ndt = pm.Uniform('left_ndt', lower=0.05, upper=0.6)
            left_noise = pm.Gamma('left_noise', alpha=2.5, beta=8.0)
            
            # === 右通道LBA模型 ===
            right_drift_correct = pm.Gamma('right_drift_correct', alpha=2.5, beta=1.2)
            right_drift_incorrect = pm.Gamma('right_drift_incorrect', alpha=2.0, beta=3.0)
            right_threshold = pm.Gamma('right_threshold', alpha=3.0, beta=3.5)
            right_start_var = pm.Uniform('right_start_var', lower=0.1, upper=0.7)
            right_ndt = pm.Uniform('right_ndt', lower=0.05, upper=0.6)
            right_noise = pm.Gamma('right_noise', alpha=2.5, beta=8.0)
            
            # === 數據準備 ===
            left_stimuli = subject_data['left_stimuli']
            left_choices = subject_data['left_choices']
            right_stimuli = subject_data['right_stimuli']
            right_choices = subject_data['right_choices']
            rt = subject_data['rt']
            
            # === 計算left通道likelihood ===
            left_likelihood = self._compute_side_likelihood(
                left_choices, left_stimuli, rt,
                left_drift_correct, left_drift_incorrect, left_threshold,
                left_start_var, left_ndt, left_noise, 'left'
            )
            
            # === 計算right通道likelihood ===
            right_likelihood = self._compute_side_likelihood(
                right_choices, right_stimuli, rt,
                right_drift_correct, right_drift_incorrect, right_threshold,
                right_start_var, right_ndt, right_noise, 'right'
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
        
        # 2A. Coactive模型
        print("   🔬 測試 Coactive 假設 (證據相加)...")
        coactive_model, coactive_trace = self._create_coactive_integration_model(subject_data, drift_estimates)
        
        # 2B. Parallel AND模型  
        print("   🔬 測試 Parallel AND 假設 (證據取最大值)...")
        parallel_and_model, parallel_and_trace = self._create_parallel_and_integration_model(subject_data, drift_estimates)
        
        return {
            'coactive_model': coactive_model,
            'coactive_trace': coactive_trace,
            'parallel_and_model': parallel_and_model,
            'parallel_and_trace': parallel_and_trace
        }
    
    def _create_coactive_integration_model(self, subject_data, drift_estimates):
        """創建Coactive證據整合模型"""
        
        with pm.Model() as coactive_model:
            
            # === 使用估計的drift rate作為固定值或先驗 ===
            # 這裡我們使用估計的均值作為先驗的中心
            left_drift_mean = drift_estimates['left_drift_mean']
            right_drift_mean = drift_estimates['right_drift_mean']
            
            # 四選一決策的參數
            choice_threshold = pm.Gamma('coactive_choice_threshold', alpha=2.0, beta=2.0)
            choice_ndt = pm.Uniform('coactive_choice_ndt', lower=0.05, upper=0.3)
            choice_noise = pm.Gamma('coactive_choice_noise', alpha=2.0, beta=5.0)
            
            # === 根據四選一選擇計算組合drift rate ===
            choices = subject_data['choices']
            rt = subject_data['rt']
            
            # 對每個選擇，計算對應的Coactive drift rate
            # 選擇 0: 左對角右垂直 -> 左diagonal + 右vertical
            # 選擇 1: 左對角右對角 -> 左diagonal + 右diagonal  
            # 選擇 2: 左垂直右垂直 -> 左vertical + 右vertical
            # 選擇 3: 左垂直右對角 -> 左vertical + 右diagonal
            
            coactive_drifts = self._compute_coactive_drift_rates(
                choices, left_drift_mean, right_drift_mean
            )
            
            # === 計算四選一LBA likelihood ===
            coactive_likelihood = self._compute_choice_likelihood(
                choices, rt, coactive_drifts, choice_threshold, choice_ndt, choice_noise
            )
            
            pm.Potential('coactive_likelihood', coactive_likelihood)
        
        # 執行採樣
        with coactive_model:
            coactive_trace = pm.sample(**self.mcmc_config)
        
        issues = diagnose_sampling_issues(coactive_trace)
        if not issues:
            print("     ✅ Coactive模型採樣成功")
        else:
            print(f"     ⚠️ Coactive模型採樣問題: {issues}")
        
        return coactive_model, coactive_trace
    
    def _create_parallel_and_integration_model(self, subject_data, drift_estimates):
        """創建Parallel AND證據整合模型"""
        
        with pm.Model() as parallel_and_model:
            
            # === 使用估計的drift rate ===
            left_drift_mean = drift_estimates['left_drift_mean']
            right_drift_mean = drift_estimates['right_drift_mean']
            
            # 四選一決策的參數
            choice_threshold = pm.Gamma('parallel_choice_threshold', alpha=2.0, beta=2.0)
            choice_ndt = pm.Uniform('parallel_choice_ndt', lower=0.05, upper=0.3)
            choice_noise = pm.Gamma('parallel_choice_noise', alpha=2.0, beta=5.0)
            
            # === 根據四選一選擇計算組合drift rate ===
            choices = subject_data['choices']
            rt = subject_data['rt']
            
            # Parallel AND: 取最大值
            parallel_drifts = self._compute_parallel_and_drift_rates(
                choices, left_drift_mean, right_drift_mean
            )
            
            # === 計算四選一LBA likelihood ===
            parallel_likelihood = self._compute_choice_likelihood(
                choices, rt, parallel_drifts, choice_threshold, choice_ndt, choice_noise
            )
            
            pm.Potential('parallel_likelihood', parallel_likelihood)
        
        # 執行採樣
        with parallel_and_model:
            parallel_trace = pm.sample(**self.mcmc_config)
        
        issues = diagnose_sampling_issues(parallel_trace)
        if not issues:
            print("     ✅ Parallel AND模型採樣成功")
        else:
            print(f"     ⚠️ Parallel AND模型採樣問題: {issues}")
        
        return parallel_and_model, parallel_trace
    
    def _compute_coactive_drift_rates(self, choices, left_drift_mean, right_drift_mean):
        """
        計算Coactive假設下的drift rate (相加)
        """
        
        # 簡化假設：每個通道對垂直線和對角線有不同的敏感度
        left_vertical_strength = left_drift_mean * 0.8    # 左通道對垂直線的強度
        left_diagonal_strength = left_drift_mean * 1.2    # 左通道對對角線的強度
        right_vertical_strength = right_drift_mean * 0.8  # 右通道對垂直線的強度
        right_diagonal_strength = right_drift_mean * 1.2  # 右通道對對角線的強度
        
        # 為每個選擇計算Coactive drift rate
        drift_choice_0 = left_diagonal_strength + right_vertical_strength    # 左\右|
        drift_choice_1 = left_diagonal_strength + right_diagonal_strength   # 左\右/
        drift_choice_2 = left_vertical_strength + right_vertical_strength   # 左|右|
        drift_choice_3 = left_vertical_strength + right_diagonal_strength   # 左|右/
        
        # 根據實際選擇分配drift rate
        coactive_drifts = pm.math.switch(
            pm.math.eq(choices, 0), drift_choice_0,
            pm.math.switch(
                pm.math.eq(choices, 1), drift_choice_1,
                pm.math.switch(
                    pm.math.eq(choices, 2), drift_choice_2,
                    drift_choice_3
                )
            )
        )
        
        return pm.math.maximum(coactive_drifts, 0.1)  # 確保正值
    
    def _compute_parallel_and_drift_rates(self, choices, left_drift_mean, right_drift_mean):
        """
        計算Parallel AND假設下的drift rate (取最大值)
        """
        
        # 每個通道的強度
        left_vertical_strength = left_drift_mean * 0.8
        left_diagonal_strength = left_drift_mean * 1.2
        right_vertical_strength = right_drift_mean * 0.8
        right_diagonal_strength = right_drift_mean * 1.2
        
        # 為每個選擇計算Parallel AND drift rate (取最大值)
        drift_choice_0 = pm.math.maximum(left_diagonal_strength, right_vertical_strength)    # 左\右|
        drift_choice_1 = pm.math.maximum(left_diagonal_strength, right_diagonal_strength)   # 左\右/
        drift_choice_2 = pm.math.maximum(left_vertical_strength, right_vertical_strength)   # 左|右|
        drift_choice_3 = pm.math.maximum(left_vertical_strength, right_diagonal_strength)   # 左|右/
        
        # 根據實際選擇分配drift rate
        parallel_drifts = pm.math.switch(
            pm.math.eq(choices, 0), drift_choice_0,
            pm.math.switch(
                pm.math.eq(choices, 1), drift_choice_1,
                pm.math.switch(
                    pm.math.eq(choices, 2), drift_choice_2,
                    drift_choice_3
                )
            )
        )
        
        return pm.math.maximum(parallel_drifts, 0.1)  # 確保正值
    
    def _compute_choice_likelihood(self, choices, rt, drift_rates, threshold, ndt, noise):
        """
        計算四選一選擇的likelihood
        """
        
        # 應用參數約束
        drift_rates = pm.math.maximum(drift_rates, 0.1)
        threshold = pm.math.maximum(threshold, 0.1)
        ndt = pm.math.maximum(ndt, 0.05)
        noise = pm.math.maximum(noise, 0.1)
        
        # 計算決策時間
        decision_time = pm.math.maximum(rt - ndt, 0.01)
        
        # 簡化的LBA likelihood計算
        # 這裡使用簡化的正態分佈近似
        predicted_rt = threshold / drift_rates + ndt
        
        # RT likelihood
        rt_likelihood = pm.math.sum(
            -0.5 * ((rt - predicted_rt) / noise) ** 2 - pm.math.log(noise * pm.math.sqrt(2 * np.pi))
        )
        
        return rt_likelihood
    
    def _extract_drift_estimates(self, trace):
        """提取drift rate的後驗估計"""
        
        summary = az.summary(trace)
        
        return {
            'left_drift_mean': (summary.loc['left_drift_correct', 'mean'] + 
                              summary.loc['left_drift_incorrect', 'mean']) / 2,
            'right_drift_mean': (summary.loc['right_drift_correct', 'mean'] + 
                               summary.loc['right_drift_incorrect', 'mean']) / 2,
            'left_drift_correct': summary.loc['left_drift_correct', 'mean'],
            'left_drift_incorrect': summary.loc['left_drift_incorrect', 'mean'],
            'right_drift_correct': summary.loc['right_drift_correct', 'mean'],
            'right_drift_incorrect': summary.loc['right_drift_incorrect', 'mean']
        }
    
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
        
        # 處理無效值
        is_invalid = (
            pm.math.isnan(log_likelihood) | 
            pm.math.eq(log_likelihood, -np.inf) | 
            pm.math.eq(log_likelihood, np.inf)
        )
        log_likelihood_safe = pm.math.where(is_invalid, -100.0, log_likelihood)
        
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
            # 計算WAIC
            coactive_waic = az.waic(coactive_trace)
            parallel_waic = az.waic(parallel_trace)
            
            # 計算LOO
            coactive_loo = az.loo(coactive_trace)
            parallel_loo = az.loo(parallel_trace)
            
            # WAIC差異 (近似Bayes Factor)
            waic_diff = coactive_waic.waic - parallel_waic.waic
            loo_diff = coactive_loo.loo - parallel_loo.loo
            
            # 解釋結果
            if waic_diff < -2:
                waic_conclusion = "強烈支持 Coactive 假設"
            elif waic_diff < 0:
                waic_conclusion = "傾向支持 Coactive 假設"
            elif waic_diff > 2:
                waic_conclusion = "強烈支持 Parallel AND 假設"
            else:
                waic_conclusion = "傾向支持 Parallel AND 假設"
            
            if loo_diff < -2:
                loo_conclusion = "強烈支持 Coactive 假設"
            elif loo_diff < 0:
                loo_conclusion = "傾向支持 Coactive 假設"
            elif loo_diff > 2:
                loo_conclusion = "強烈支持 Parallel AND 假設"
            else:
                loo_conclusion = "傾向支持 Parallel AND 假設"
            
            comparison_results = {
                'coactive_waic': coactive_waic.waic,
                'parallel_waic': parallel_waic.waic,
                'waic_diff': waic_diff,
                'waic_conclusion': waic_conclusion,
                'coactive_loo': coactive_loo.loo,
                'parallel_loo': parallel_loo.loo,
                'loo_diff': loo_diff,
                'loo_conclusion': loo_conclusion
            }
            
            print(f"   📊 模型比較結果:")
            print(f"      Coactive WAIC:    {coactive_waic.waic:.2f}")
            print(f"      Parallel WAIC:    {parallel_waic.waic:.2f}")
            print(f"      WAIC 差異:        {waic_diff:.2f}")
            print(f"      WAIC 結論:        {waic_conclusion}")
            print(f"      LOO 差異:         {loo_diff:.2f}")
            print(f"      LOO 結論:         {loo_conclusion}")
            
            return comparison_results
            
        except Exception as e:
            print(f"   ❌ Bayes Factor計算失敗: {e}")
            return None

def run_evidence_integration_analysis(csv_file='GRT_LBA.csv', subject_id=None):
    """
    執行完整的證據整合分析
    """
    
    print("🚀 證據整合假設檢驗分析")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 載入資料
        processor = DataProcessor()
        df = processor.load_and_clean_data(csv_file)
        
        # 選擇受試者
        if subject_id is None:
            subject_id = df['participant'].iloc[0]
            print(f"自動選擇受試者: {subject_id}")
        
        # 創建分析器
        analyzer = EvidenceIntegrationComparison()
        
        # 準備數據
        subject_data = analyzer.prepare_subject_data(df, subject_id)
        print(f"受試者 {subject_id}: {subject_data['n_trials']} trials, 準確率 {subject_data['accuracy']:.1%}")
        
        # 步驟1: Single LBA估計
        single_model, single_trace, drift_estimates = analyzer.step1_estimate_single_lba(subject_data)
        
        # 步驟2: 證據整合測試
        integration_results = analyzer.step2_test_evidence_integration(subject_data, drift_estimates)
        
        # 步驟3: Bayes Factor比較
        bayes_results = analyzer.step3_compute_bayes_factors(integration_results)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 證據整合分析完成!")
        print(f"⏱️ 總時間: {total_time/60:.1f} 分鐘")
        print(f"🏆 最終結論: {bayes_results['waic_conclusion'] if bayes_results else '無法判斷'}")
        print("="*60)
        
        return {
            'subject_id': subject_id,
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

if __name__ == "__main__":
    print("🎯 證據整合假設檢驗:")
    print("=" * 40)
    print("這個分析將會:")
    print("1. 用Single LBA估計左右通道drift rate")
    print("2. 測試Coactive (相加) vs Parallel AND (最大值) 假設")
    print("3. 用Bayes Factor判斷哪個假設更符合數據")
    
    try:
        choice = input("\n是否開始分析? (y/n): ").strip().lower()
        
        if choice == 'y':
            print("\n🚀 開始證據整合分析...")
            result = run_evidence_integration_analysis()
            
            if result['success']:
                print("\n✅ 分析成功完成!")
            else:
                print("\n❌ 分析失敗")
        else:
            print("分析取消")
            
    except KeyboardInterrupt:
        print("\n⏹️ 分析被使用者中斷")
    except Exception as e:
        print(f"\n💥 未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
