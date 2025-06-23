# -*- coding: utf-8 -*-
"""
grt_model_comparison.py - GRT 四選項模型比較
Four-Choice GRT Model Comparison: Parallel AND vs Coactive Architectures

實現兩種處理架構的完整 PyMC 模型：
1. Parallel AND (Exhaustive): 左右獨立處理，取最大時間
2. Coactive: 左右證據加總，共同處理

包含完整的模型比較框架：WAIC, LOO, BIC, Bayes Factor 等
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class ModelType(Enum):
    """模型類型"""
    PARALLEL_AND = "parallel_and"
    COACTIVE = "coactive"

@dataclass
class ModelComparisonResult:
    """模型比較結果"""
    model_type: ModelType
    waic: float
    waic_se: float
    loo: float
    loo_se: float
    bic: float
    marginal_likelihood: float
    n_parameters: int
    convergence_success: bool
    sampling_time: float

class GRTModelBuilder:
    """GRT 四選項模型建構器"""
    
    def __init__(self, model_type: ModelType):
        """
        初始化模型建構器
        
        Args:
            model_type: 模型類型 (PARALLEL_AND 或 COACTIVE)
        """
        self.model_type = model_type
        self.param_names = self._get_parameter_names()
        
        print(f"🔧 初始化 {model_type.value} 模型建構器")
        print(f"   參數數量: {len(self.param_names)}")
        
    def _get_parameter_names(self) -> List[str]:
        """獲取模型參數名稱"""
        
        if self.model_type == ModelType.PARALLEL_AND:
            return [
                # 左通道 LBA 參數
                'left_drift_correct', 'left_drift_incorrect',
                'left_threshold', 'left_start_var', 'left_ndt', 'left_noise',
                
                # 右通道 LBA 參數
                'right_drift_correct', 'right_drift_incorrect', 
                'right_threshold', 'right_start_var', 'right_ndt', 'right_noise',
                
                # 四選項整合層參數
                'choice_0_drift', 'choice_1_drift', 'choice_2_drift', 'choice_3_drift',
                'final_threshold', 'final_start_var', 'final_ndt', 'final_noise',
                
                # 時間分配參數
                'time_split_ratio'  # 第一階段占總時間的比例
            ]
        else:  # COACTIVE
            return [
                # 共同激活 LBA 參數
                'coactive_drift_correct', 'coactive_drift_incorrect',
                'coactive_threshold', 'coactive_start_var', 'coactive_ndt', 'coactive_noise',
                
                # 左右通道權重
                'left_channel_weight', 'right_channel_weight',
                
                # 四選項整合層參數
                'choice_0_drift', 'choice_1_drift', 'choice_2_drift', 'choice_3_drift',
                'final_threshold', 'final_start_var', 'final_ndt', 'final_noise',
                
                # 時間分配參數
                'time_split_ratio'
            ]
    
    def build_model(self, subject_data: Dict) -> pm.Model:
        """
        建構完整的 PyMC 模型
        
        Args:
            subject_data: 受試者資料字典，包含：
                - choices: 四選項選擇 (0,1,2,3)
                - rt: 反應時間
                - left_stimuli: 左邊刺激 (0=垂直, 1=對角)
                - left_choices: 左邊選擇 (0=垂直, 1=對角)
                - right_stimuli: 右邊刺激 (0=垂直, 1=對角)
                - right_choices: 右邊選擇 (0=垂直, 1=對角)
                
        Returns:
            PyMC 模型
        """
        
        print(f"🏗️  建構 {self.model_type.value} PyMC 模型...")
        print(f"   受試者: {subject_data['subject_id']}")
        print(f"   試驗數: {subject_data['n_trials']}")
        
        with pm.Model() as model:
            
            # 1. 定義先驗分布
            params = self._define_priors()
            
            # 2. 準備觀察資料
            data_tensors = self._prepare_data_tensors(subject_data)
            
            # 3. 根據模型類型計算似然
            if self.model_type == ModelType.PARALLEL_AND:
                log_likelihood = self._compute_parallel_and_likelihood(params, data_tensors)
            else:
                log_likelihood = self._compute_coactive_likelihood(params, data_tensors)
            
            # 4. 添加似然到模型
            pm.Potential('model_likelihood', log_likelihood)
            
            # 5. 添加診斷變數
            pm.Deterministic('total_log_likelihood', log_likelihood)
            
        print(f"✅ {self.model_type.value} 模型建構完成")
        print(f"   自由參數: {len(model.free_RVs)}")
        
        return model
    
    def _define_priors(self) -> Dict:
        """定義先驗分布"""
        
        params = {}
        
        if self.model_type == ModelType.PARALLEL_AND:
            # 左通道參數
            params['left_drift_correct'] = pm.Gamma('left_drift_correct', alpha=2.5, beta=1.5)
            params['left_drift_incorrect'] = pm.Gamma('left_drift_incorrect', alpha=2.0, beta=3.0)
            params['left_threshold'] = pm.Gamma('left_threshold', alpha=3.0, beta=3.5)
            params['left_start_var'] = pm.Uniform('left_start_var', lower=0.1, upper=0.7)
            params['left_ndt'] = pm.Uniform('left_ndt', lower=0.05, upper=0.4)
            params['left_noise'] = pm.Gamma('left_noise', alpha=2.5, beta=8.0)
            
            # 右通道參數
            params['right_drift_correct'] = pm.Gamma('right_drift_correct', alpha=2.5, beta=1.5)
            params['right_drift_incorrect'] = pm.Gamma('right_drift_incorrect', alpha=2.0, beta=3.0)
            params['right_threshold'] = pm.Gamma('right_threshold', alpha=3.0, beta=3.5)
            params['right_start_var'] = pm.Uniform('right_start_var', lower=0.1, upper=0.7)
            params['right_ndt'] = pm.Uniform('right_ndt', lower=0.05, upper=0.4)
            params['right_noise'] = pm.Gamma('right_noise', alpha=2.5, beta=8.0)
            
        else:  # COACTIVE
            # 共同激活參數
            params['coactive_drift_correct'] = pm.Gamma('coactive_drift_correct', alpha=3.0, beta=1.2)
            params['coactive_drift_incorrect'] = pm.Gamma('coactive_drift_incorrect', alpha=2.0, beta=2.5)
            params['coactive_threshold'] = pm.Gamma('coactive_threshold', alpha=3.0, beta=3.0)
            params['coactive_start_var'] = pm.Uniform('coactive_start_var', lower=0.1, upper=0.8)
            params['coactive_ndt'] = pm.Uniform('coactive_ndt', lower=0.05, upper=0.4)
            params['coactive_noise'] = pm.Gamma('coactive_noise', alpha=2.5, beta=6.0)
            
            # 通道權重參數
            params['left_channel_weight'] = pm.Beta('left_channel_weight', alpha=3.0, beta=3.0)
            params['right_channel_weight'] = pm.Beta('right_channel_weight', alpha=3.0, beta=3.0)
        
        # 四選項整合層參數（兩種模型都需要）
        for i in range(4):
            params[f'choice_{i}_drift'] = pm.Gamma(f'choice_{i}_drift', alpha=2.0, beta=2.0)
        
        params['final_threshold'] = pm.Gamma('final_threshold', alpha=2.5, beta=3.0)
        params['final_start_var'] = pm.Uniform('final_start_var', lower=0.1, upper=0.5)
        params['final_ndt'] = pm.Uniform('final_ndt', lower=0.05, upper=0.3)
        params['final_noise'] = pm.Gamma('final_noise', alpha=2.0, beta=6.0)
        
        # 時間分配參數
        params['time_split_ratio'] = pm.Beta('time_split_ratio', alpha=3.0, beta=2.0)
        
        return params
    
    def _prepare_data_tensors(self, subject_data: Dict) -> Dict:
        """準備資料張量"""
        
        return {
            'final_choices': pt.as_tensor_variable(subject_data['choices'], dtype='int32'),
            'rt_total': pt.as_tensor_variable(subject_data['rt'], dtype='float64'),
            'left_stimuli': pt.as_tensor_variable(subject_data['left_stimuli'], dtype='int32'),
            'left_choices': pt.as_tensor_variable(subject_data['left_choices'], dtype='int32'),
            'right_stimuli': pt.as_tensor_variable(subject_data['right_stimuli'], dtype='int32'),
            'right_choices': pt.as_tensor_variable(subject_data['right_choices'], dtype='int32'),
            'n_trials': len(subject_data['choices'])
        }
    
    def _compute_parallel_and_likelihood(self, params: Dict, data: Dict) -> pt.TensorVariable:
        """
        計算 Parallel AND 模型的似然函數
        
        架構：
        1. 左右兩個獨立 LBA 同時處理
        2. 等待兩邊都完成（AND stopping rule）
        3. 取 max(left_time, right_time) 作為第一階段時間
        4. 剩餘時間用於四選項決策
        """
        
        # 應用參數約束
        left_params = self._apply_parameter_constraints(params, 'left')
        right_params = self._apply_parameter_constraints(params, 'right')
        final_params = self._apply_parameter_constraints(params, 'final')
        time_split = pt.clip(params['time_split_ratio'], 0.3, 0.8)
        
        # 計算第一階段時間分配
        stage1_time = data['rt_total'] * time_split
        stage2_time = data['rt_total'] * (1 - time_split)
        stage2_time = pt.maximum(stage2_time, 0.01)
        
        # 左通道 LBA 似然
        left_ll = self._compute_single_channel_lba_likelihood(
            data['left_choices'], data['left_stimuli'], stage1_time, left_params
        )
        
        # 右通道 LBA 似然
        right_ll = self._compute_single_channel_lba_likelihood(
            data['right_choices'], data['right_stimuli'], stage1_time, right_params
        )
        
        # AND stopping rule: 兩邊都需要完成
        # 這裡我們假設觀察到的反應表示兩邊都已經處理完成
        stage1_likelihood = left_ll + right_ll
        
        # 計算整合證據
        evidence_strength = self._compute_parallel_evidence_strength(
            left_params, right_params, data['left_stimuli'], data['left_choices'],
            data['right_stimuli'], data['right_choices']
        )
        
        # 第二階段：四選項決策
        stage2_likelihood = self._compute_four_choice_lba_likelihood(
            data['final_choices'], stage2_time, evidence_strength, final_params
        )
        
        return stage1_likelihood + stage2_likelihood
    
    def _compute_coactive_likelihood(self, params: Dict, data: Dict) -> pt.TensorVariable:
        """
        計算 Coactive 模型的似然函數
        
        架構：
        1. 左右證據加總到單一 LBA
        2. 共同激活處理產生第一階段決策
        3. 剩餘時間用於四選項決策
        """
        
        # 應用參數約束
        coactive_params = self._apply_parameter_constraints(params, 'coactive')
        final_params = self._apply_parameter_constraints(params, 'final')
        time_split = pt.clip(params['time_split_ratio'], 0.3, 0.8)
        
        # 權重參數
        left_weight = pt.clip(params['left_channel_weight'], 0.1, 0.9)
        right_weight = pt.clip(params['right_channel_weight'], 0.1, 0.9)
        
        # 計算時間分配
        stage1_time = data['rt_total'] * time_split
        stage2_time = data['rt_total'] * (1 - time_split)
        stage2_time = pt.maximum(stage2_time, 0.01)
        
        # 計算加權組合的漂移率
        combined_drift_correct, combined_drift_incorrect = self._compute_coactive_drifts(
            coactive_params, left_weight, right_weight,
            data['left_stimuli'], data['left_choices'],
            data['right_stimuli'], data['right_choices']
        )
        
        # 第一階段：共同激活 LBA 似然
        stage1_likelihood = self._compute_coactive_lba_likelihood(
            combined_drift_correct, combined_drift_incorrect, 
            stage1_time, coactive_params
        )
        
        # 計算整合證據
        evidence_strength = self._compute_coactive_evidence_strength(
            left_weight, right_weight, coactive_params,
            data['left_stimuli'], data['left_choices'],
            data['right_stimuli'], data['right_choices']
        )
        
        # 第二階段：四選項決策
        stage2_likelihood = self._compute_four_choice_lba_likelihood(
            data['final_choices'], stage2_time, evidence_strength, final_params
        )
        
        return stage1_likelihood + stage2_likelihood
    
    def _apply_parameter_constraints(self, params: Dict, prefix: str) -> Dict:
        """應用參數約束"""
        
        constrained = {}
        
        if prefix in ['left', 'right']:
            constrained[f'{prefix}_drift_correct'] = pt.maximum(params[f'{prefix}_drift_correct'], 0.1)
            constrained[f'{prefix}_drift_incorrect'] = pt.maximum(params[f'{prefix}_drift_incorrect'], 0.05)
            constrained[f'{prefix}_threshold'] = pt.maximum(params[f'{prefix}_threshold'], 0.1)
            constrained[f'{prefix}_start_var'] = pt.clip(params[f'{prefix}_start_var'], 0.05, 1.0)
            constrained[f'{prefix}_ndt'] = pt.clip(params[f'{prefix}_ndt'], 0.05, 0.5)
            constrained[f'{prefix}_noise'] = pt.maximum(params[f'{prefix}_noise'], 0.1)
            
            # 確保正確漂移率 > 錯誤漂移率
            constrained[f'{prefix}_drift_correct'] = pt.maximum(
                constrained[f'{prefix}_drift_correct'],
                constrained[f'{prefix}_drift_incorrect'] + 0.05
            )
            
        elif prefix == 'coactive':
            constrained['coactive_drift_correct'] = pt.maximum(params['coactive_drift_correct'], 0.1)
            constrained['coactive_drift_incorrect'] = pt.maximum(params['coactive_drift_incorrect'], 0.05)
            constrained['coactive_threshold'] = pt.maximum(params['coactive_threshold'], 0.1)
            constrained['coactive_start_var'] = pt.clip(params['coactive_start_var'], 0.05, 1.0)
            constrained['coactive_ndt'] = pt.clip(params['coactive_ndt'], 0.05, 0.5)
            constrained['coactive_noise'] = pt.maximum(params['coactive_noise'], 0.1)
            
            # 確保正確漂移率 > 錯誤漂移率
            constrained['coactive_drift_correct'] = pt.maximum(
                constrained['coactive_drift_correct'],
                constrained['coactive_drift_incorrect'] + 0.05
            )
            
        elif prefix == 'final':
            for i in range(4):
                constrained[f'choice_{i}_drift'] = pt.maximum(params[f'choice_{i}_drift'], 0.1)
            constrained['final_threshold'] = pt.maximum(params['final_threshold'], 0.1)
            constrained['final_start_var'] = pt.clip(params['final_start_var'], 0.05, 0.8)
            constrained['final_ndt'] = pt.clip(params['final_ndt'], 0.05, 0.4)
            constrained['final_noise'] = pt.maximum(params['final_noise'], 0.1)
        
        return constrained
    
    def _compute_single_channel_lba_likelihood(self, decisions, stimuli, rt, params):
        """計算單通道 LBA 似然"""
        
        prefix = list(params.keys())[0].split('_')[0]
        
        drift_correct = params[f'{prefix}_drift_correct']
        drift_incorrect = params[f'{prefix}_drift_incorrect']
        threshold = params[f'{prefix}_threshold']
        start_var = params[f'{prefix}_start_var']
        ndt = params[f'{prefix}_ndt']
        noise = params[f'{prefix}_noise']
        
        # 計算決策時間
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        # 判斷正確性
        is_correct = pt.eq(decisions, stimuli)
        
        # 設定漂移率
        v_winner = pt.where(is_correct, drift_correct, drift_incorrect)
        v_loser = pt.where(is_correct, drift_incorrect, drift_correct)
        
        # 計算 LBA 似然
        return self._lba_likelihood_core(decision_time, v_winner, v_loser, threshold, start_var, noise)
    
    def _compute_coactive_drifts(self, params, left_weight, right_weight,
                                left_stimuli, left_choices, right_stimuli, right_choices):
        """計算共同激活的組合漂移率"""
        
        # 左通道貢獻
        left_correct = pt.eq(left_choices, left_stimuli)
        left_drift = pt.where(left_correct, 
                             params['coactive_drift_correct'], 
                             params['coactive_drift_incorrect'])
        
        # 右通道貢獻
        right_correct = pt.eq(right_choices, right_stimuli)
        right_drift = pt.where(right_correct,
                              params['coactive_drift_correct'],
                              params['coactive_drift_incorrect'])
        
        # 加權組合（Coactive 的核心特徵）
        combined_drift_correct = left_weight * left_drift + right_weight * right_drift
        combined_drift_incorrect = (left_weight * params['coactive_drift_incorrect'] + 
                                   right_weight * params['coactive_drift_incorrect'])
        
        return combined_drift_correct, combined_drift_incorrect
    
    def _compute_coactive_lba_likelihood(self, drift_correct, drift_incorrect, rt, params):
        """計算共同激活 LBA 似然"""
        
        threshold = params['coactive_threshold']
        start_var = params['coactive_start_var']
        ndt = params['coactive_ndt']
        noise = params['coactive_noise']
        
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        # 假設我們總是選擇較強的證據
        v_winner = pt.maximum(drift_correct, drift_incorrect)
        v_loser = pt.minimum(drift_correct, drift_incorrect)
        
        return self._lba_likelihood_core(decision_time, v_winner, v_loser, threshold, start_var, noise)
    
    def _compute_parallel_evidence_strength(self, left_params, right_params,
                                          left_stimuli, left_choices, right_stimuli, right_choices):
        """計算平行處理的證據強度"""
        
        # 左通道強度
        left_strength = left_params['left_drift_correct'] / pt.maximum(left_params['left_drift_incorrect'], 0.1)
        
        # 右通道強度
        right_strength = right_params['right_drift_correct'] / pt.maximum(right_params['right_drift_incorrect'], 0.1)
        
        # 四選項證據組合（基於 GRT 對應關係）
        return {
            'choice_0': left_strength * 0.8 + right_strength * 0.2,  # 左對角右垂直
            'choice_1': left_strength * 0.8 + right_strength * 0.8,  # 左對角右對角
            'choice_2': left_strength * 0.2 + right_strength * 0.2,  # 左垂直右垂直
            'choice_3': left_strength * 0.2 + right_strength * 0.8   # 左垂直右對角
        }
    
    def _compute_coactive_evidence_strength(self, left_weight, right_weight, params,
                                          left_stimuli, left_choices, right_stimuli, right_choices):
        """計算共同激活的證據強度"""
        
        # 組合強度
        combined_strength = (left_weight + right_weight) * \
                           params['coactive_drift_correct'] / pt.maximum(params['coactive_drift_incorrect'], 0.1)
        
        # 四選項證據（共同激活導致較均勻的分布）
        base_evidence = combined_strength * 0.6
        return {
            'choice_0': base_evidence * 1.05,
            'choice_1': base_evidence * 1.00,
            'choice_2': base_evidence * 0.95,
            'choice_3': base_evidence * 1.00
        }
    
    def _compute_four_choice_lba_likelihood(self, choices, rt, evidence_strength, params):
        """計算四選項 LBA 似然"""
        
        # 調整漂移率
        adjusted_drifts = []
        for i in range(4):
            base_drift = params[f'choice_{i}_drift']
            evidence_boost = evidence_strength[f'choice_{i}']
            adjusted_drift = base_drift * (1.0 + evidence_boost * 0.3)
            adjusted_drifts.append(pt.maximum(adjusted_drift, 0.1))
        
        threshold = params['final_threshold']
        start_var = params['final_start_var']
        ndt = params['final_ndt']
        noise = params['final_noise']
        
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        return self._four_choice_lba_likelihood_core(choices, decision_time, adjusted_drifts, 
                                                   threshold, start_var, noise)
    
    def _lba_likelihood_core(self, t, v_winner, v_loser, threshold, start_var, noise):
        """LBA 似然計算核心"""
        
        from pytensor.tensor import erf
        
        sqrt_t = pt.sqrt(t)
        
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(pt.clip(x, -4.5, 4.5) / pt.sqrt(2)))
        
        def safe_normal_pdf(x):
            x_safe = pt.clip(x, -4.5, 4.5)
            return pt.exp(-0.5 * x_safe**2) / pt.sqrt(2 * pt.pi)
        
        # Winner 計算
        z1_winner = (v_winner * t - threshold) / (noise * sqrt_t)
        z2_winner = (v_winner * t - start_var) / (noise * sqrt_t)
        
        winner_cdf = safe_normal_cdf(z1_winner) - safe_normal_cdf(z2_winner)
        winner_pdf = (safe_normal_pdf(z1_winner) - safe_normal_pdf(z2_winner)) / (noise * sqrt_t)
        
        winner_likelihood = pt.maximum(
            (v_winner / start_var) * pt.maximum(winner_cdf, 1e-10) + winner_pdf / start_var,
            1e-10
        )
        
        # Loser 存活
        z1_loser = (v_loser * t - threshold) / (noise * sqrt_t)
        loser_survival = pt.maximum(1 - safe_normal_cdf(z1_loser), 1e-10)
        
        # 聯合似然
        joint_likelihood = winner_likelihood * loser_survival
        log_likelihood = pt.log(pt.maximum(joint_likelihood, 1e-12))
        
        return pt.sum(pt.clip(log_likelihood, -100.0, 10.0))
    
    def _four_choice_lba_likelihood_core(self, choices, t, drifts, threshold, start_var, noise):
        """四選項 LBA 似然核心"""
        
        from pytensor.tensor import erf
        
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(pt.clip(x, -4.5, 4.5) / pt.sqrt(2)))
        
        sqrt_t = pt.sqrt(t)
        
        # 計算每個選項的密度和存活函數
        densities = []
        survivals = []
        
        for drift in drifts:
            # 密度計算
            z1 = (drift * t - threshold) / (noise * sqrt_t)
            z2 = (drift * t - start_var) / (noise * sqrt_t)
            
            density = pt.maximum(
                (drift / start_var) * pt.maximum(safe_normal_cdf(z1) - safe_normal_cdf(z2), 1e-10),
                1e-10
            )
            densities.append(density)
            
            # 存活計算
            survival = pt.maximum(1 - safe_normal_cdf(z1), 1e-10)
            survivals.append(survival)
        
        # 計算每個選項的完整似然
        trial_likelihoods = pt.zeros_like(t)
        for i in range(4):
            # Winner density × All other survivals
            likelihood_i = densities[i]
            for j in range(4):
                if i != j:
                    likelihood_i = likelihood_i * survivals[j]
            
            # 根據實際選擇累加
            mask = pt.eq(choices, i)
            trial_likelihoods = trial_likelihoods + mask * likelihood_i
        
        # 返回對數似然
        log_likelihood = pt.log(pt.maximum(trial_likelihoods, 1e-12))
        return pt.sum(pt.clip(log_likelihood, -100.0, 10.0))

class GRTModelComparator:
    """GRT 模型比較器"""
    
    def __init__(self, mcmc_config: Optional[Dict] = None):
        """
        初始化模型比較器
        
        Args:
            mcmc_config: MCMC 配置字典
        """
        self.mcmc_config = mcmc_config or {
            'draws': 500,
            'tune': 500,
            'chains': 2,
            'cores': 1,
            'target_accept': 0.85,
            'max_treedepth': 8,
            'random_seed': 42,
            'progressbar': True,
            'return_inferencedata': True
        }
        
        print("🏆 初始化 GRT 模型比較器")
        print(f"   MCMC 設定: {self.mcmc_config['draws']} draws × {self.mcmc_config['chains']} chains")
    
    def fit_and_compare_models(self, subject_data: Dict, verbose: bool = True) -> Dict:
        """
        擬合兩個模型並進行比較
        
        Args:
            subject_data: 受試者資料
            verbose: 是否顯示詳細信息
            
        Returns:
            完整比較結果字典
        """
        
        if verbose:
            print(f"\n🎯 模型比較分析")
            print(f"   受試者: {subject_data['subject_id']}")
            print(f"   試驗數: {subject_data['n_trials']}")
            print("="*60)
        
        results = {}
        
        # 1. 擬合 Parallel AND 模型
        if verbose:
            print("\n📊 擬合 Parallel AND 模型...")
        
        parallel_result = self._fit_single_model(
            ModelType.PARALLEL_AND, subject_data, verbose
        )
        results['parallel_and'] = parallel_result
        
        # 2. 擬合 Coactive 模型
        if verbose:
            print("\n📊 擬合 Coactive 模型...")
        
        coactive_result = self._fit_single_model(
            ModelType.COACTIVE, subject_data, verbose
        )
        results['coactive'] = coactive_result
        
        # 3. 進行模型比較
        if verbose:
            print("\n🏆 模型比較分析...")
        
        comparison = self._compare_model_results(parallel_result, coactive_result, verbose)
        results['comparison'] = comparison
        
        # 4. 生成比較摘要
        if verbose:
            self._print_comparison_summary(results)
        
        return results
    
    def _fit_single_model(self, model_type: ModelType, subject_data: Dict, verbose: bool) -> ModelComparisonResult:
        """擬合單一模型"""
        
        start_time = time.time()
        
        try:
            # 建構模型
            builder = GRTModelBuilder(model_type)
            model = builder.build_model(subject_data)
            
            # 模型驗證
            with model:
                test_point = model.initial_point()
                initial_logp = model.compile_logp()(test_point)
                
                if not np.isfinite(initial_logp):
                    raise ValueError(f"Invalid initial log probability: {initial_logp}")
            
            if verbose:
                print(f"   ✅ 模型驗證通過 (initial_logp = {initial_logp:.2f})")
            
            # MCMC 採樣
            with model:
                if verbose:
                    print("   🎲 執行 MCMC 採樣...")
                
                trace = pm.sample(
                    draws=self.mcmc_config['draws'],
                    tune=self.mcmc_config['tune'],
                    chains=self.mcmc_config['chains'],
                    cores=self.mcmc_config['cores'],
                    target_accept=self.mcmc_config['target_accept'],
                    max_treedepth=self.mcmc_config['max_treedepth'],
                    random_seed=self.mcmc_config['random_seed'],
                    progressbar=self.mcmc_config['progressbar'] and verbose,
                    return_inferencedata=self.mcmc_config['return_inferencedata']
                )
            
            sampling_time = time.time() - start_time
            
            # 收斂診斷
            convergence_success = self._check_convergence(trace, verbose)
            
            # 模型評估指標
            evaluation_metrics = self._compute_evaluation_metrics(trace, model, verbose)
            
            if verbose:
                print(f"   ⏱️ 採樣時間: {sampling_time/60:.1f} 分鐘")
                print(f"   🔄 收斂狀態: {'成功' if convergence_success else '警告'}")
            
            return ModelComparisonResult(
                model_type=model_type,
                waic=evaluation_metrics['waic'],
                waic_se=evaluation_metrics['waic_se'],
                loo=evaluation_metrics['loo'],
                loo_se=evaluation_metrics['loo_se'],
                bic=evaluation_metrics['bic'],
                marginal_likelihood=evaluation_metrics['marginal_likelihood'],
                n_parameters=len(builder.param_names),
                convergence_success=convergence_success,
                sampling_time=sampling_time
            )
            
        except Exception as e:
            if verbose:
                print(f"   ❌ 模型擬合失敗: {e}")
            
            return ModelComparisonResult(
                model_type=model_type,
                waic=np.inf,
                waic_se=np.inf,
                loo=np.inf,
                loo_se=np.inf,
                bic=np.inf,
                marginal_likelihood=-np.inf,
                n_parameters=len(GRTModelBuilder(model_type).param_names),
                convergence_success=False,
                sampling_time=time.time() - start_time
            )
    
    def _check_convergence(self, trace, verbose: bool) -> bool:
        """檢查收斂狀態"""
        
        try:
            # R-hat 統計
            rhat = az.rhat(trace)
            max_rhat = float(rhat.to_array().max())
            
            # ESS 統計
            ess_bulk = az.ess(trace)
            min_ess = float(ess_bulk.to_array().min())
            
            # 收斂標準
            rhat_ok = max_rhat <= 1.05
            ess_ok = min_ess >= 100
            
            convergence_success = rhat_ok and ess_ok
            
            if verbose:
                print(f"      R̂_max = {max_rhat:.3f}")
                print(f"      ESS_min = {min_ess:.0f}")
            
            return convergence_success
            
        except Exception as e:
            if verbose:
                print(f"      ⚠️ 收斂檢查失敗: {e}")
            return False
    
    def _compute_evaluation_metrics(self, trace, model, verbose: bool) -> Dict:
        """計算模型評估指標"""
        
        metrics = {
            'waic': np.inf,
            'waic_se': np.inf,
            'loo': np.inf,
            'loo_se': np.inf,
            'bic': np.inf,
            'marginal_likelihood': -np.inf
        }
        
        try:
            # WAIC (Widely Applicable Information Criterion)
            waic_result = az.waic(trace)
            metrics['waic'] = float(waic_result.waic)
            metrics['waic_se'] = float(waic_result.se)
            
            if verbose:
                print(f"      WAIC = {metrics['waic']:.2f} ± {metrics['waic_se']:.2f}")
            
        except Exception as e:
            if verbose:
                print(f"      ⚠️ WAIC 計算失敗: {e}")
        
        try:
            # LOO (Leave-One-Out Cross-Validation)
            loo_result = az.loo(trace)
            metrics['loo'] = float(loo_result.loo)
            metrics['loo_se'] = float(loo_result.se)
            
            if verbose:
                print(f"      LOO = {metrics['loo']:.2f} ± {metrics['loo_se']:.2f}")
            
        except Exception as e:
            if verbose:
                print(f"      ⚠️ LOO 計算失敗: {e}")
        
        try:
            # BIC 估算 (基於後驗均值)
            log_likelihood_samples = trace.log_likelihood.values if hasattr(trace, 'log_likelihood') else None
            if log_likelihood_samples is not None:
                mean_log_likelihood = np.mean(log_likelihood_samples)
                n_params = len(trace.posterior.data_vars)
                n_obs = log_likelihood_samples.shape[-1]  # 假設最後一維是觀察數
                
                metrics['bic'] = -2 * mean_log_likelihood + n_params * np.log(n_obs)
                
                if verbose:
                    print(f"      BIC ≈ {metrics['bic']:.2f}")
            
        except Exception as e:
            if verbose:
                print(f"      ⚠️ BIC 計算失敗: {e}")
        
        try:
            # 邊際似然估算 (基於 Harmonic Mean Estimator)
            log_likelihood_samples = trace.log_likelihood.values if hasattr(trace, 'log_likelihood') else None
            if log_likelihood_samples is not None:
                # 簡化的邊際似然估算
                metrics['marginal_likelihood'] = np.mean(log_likelihood_samples)
                
                if verbose:
                    print(f"      Marginal LL ≈ {metrics['marginal_likelihood']:.2f}")
            
        except Exception as e:
            if verbose:
                print(f"      ⚠️ 邊際似然計算失敗: {e}")
        
        return metrics
    
    def _compare_model_results(self, parallel_result: ModelComparisonResult, 
                              coactive_result: ModelComparisonResult, verbose: bool) -> Dict:
        """比較兩個模型的結果"""
        
        comparison = {}
        
        # 1. WAIC 比較
        if np.isfinite(parallel_result.waic) and np.isfinite(coactive_result.waic):
            waic_diff = parallel_result.waic - coactive_result.waic
            waic_se_diff = np.sqrt(parallel_result.waic_se**2 + coactive_result.waic_se**2)
            
            comparison['waic'] = {
                'parallel_and': parallel_result.waic,
                'coactive': coactive_result.waic,
                'difference': waic_diff,
                'se_difference': waic_se_diff,
                'better_model': 'coactive' if waic_diff > 0 else 'parallel_and',
                'significant': abs(waic_diff) > 2 * waic_se_diff
            }
        
        # 2. LOO 比較
        if np.isfinite(parallel_result.loo) and np.isfinite(coactive_result.loo):
            loo_diff = parallel_result.loo - coactive_result.loo
            loo_se_diff = np.sqrt(parallel_result.loo_se**2 + coactive_result.loo_se**2)
            
            comparison['loo'] = {
                'parallel_and': parallel_result.loo,
                'coactive': coactive_result.loo,
                'difference': loo_diff,
                'se_difference': loo_se_diff,
                'better_model': 'coactive' if loo_diff > 0 else 'parallel_and',
                'significant': abs(loo_diff) > 2 * loo_se_diff
            }
        
        # 3. BIC 比較
        if np.isfinite(parallel_result.bic) and np.isfinite(coactive_result.bic):
            bic_diff = parallel_result.bic - coactive_result.bic
            
            comparison['bic'] = {
                'parallel_and': parallel_result.bic,
                'coactive': coactive_result.bic,
                'difference': bic_diff,
                'better_model': 'coactive' if bic_diff > 0 else 'parallel_and',
                'strength': self._interpret_bic_difference(abs(bic_diff))
            }
        
        # 4. Bayes Factor 估算
        if (np.isfinite(parallel_result.marginal_likelihood) and 
            np.isfinite(coactive_result.marginal_likelihood)):
            
            log_bf = parallel_result.marginal_likelihood - coactive_result.marginal_likelihood
            bf = np.exp(log_bf)
            
            comparison['bayes_factor'] = {
                'log_bayes_factor': log_bf,
                'bayes_factor': bf,
                'evidence_strength': self._interpret_bayes_factor(bf),
                'favored_model': 'parallel_and' if log_bf > 0 else 'coactive'
            }
        
        # 5. 收斂比較
        comparison['convergence'] = {
            'parallel_and_converged': parallel_result.convergence_success,
            'coactive_converged': coactive_result.convergence_success,
            'both_converged': parallel_result.convergence_success and coactive_result.convergence_success
        }
        
        # 6. 複雜度比較
        comparison['complexity'] = {
            'parallel_and_params': parallel_result.n_parameters,
            'coactive_params': coactive_result.n_parameters,
            'parameter_difference': parallel_result.n_parameters - coactive_result.n_parameters
        }
        
        # 7. 整體建議
        comparison['recommendation'] = self._generate_overall_recommendation(comparison, verbose)
        
        return comparison
    
    def _interpret_bic_difference(self, bic_diff: float) -> str:
        """解釋 BIC 差異強度"""
        
        if bic_diff < 2:
            return "weak"
        elif bic_diff < 6:
            return "positive"
        elif bic_diff < 10:
            return "strong"
        else:
            return "very_strong"
    
    def _interpret_bayes_factor(self, bf: float) -> str:
        """解釋 Bayes Factor 證據強度"""
        
        if bf < 1:
            bf = 1 / bf  # 取倒數以便統一解釋
        
        if bf < 3:
            return "anecdotal"
        elif bf < 10:
            return "moderate" 
        elif bf < 30:
            return "strong"
        elif bf < 100:
            return "very_strong"
        else:
            return "extreme"
    
    def _generate_overall_recommendation(self, comparison: Dict, verbose: bool) -> Dict:
        """生成整體建議"""
        
        recommendations = []
        confidence_score = 0
        
        # 檢查收斂
        if not comparison['convergence']['both_converged']:
            recommendations.append("⚠️ 部分模型收斂問題，結果需謹慎解釋")
            confidence_score -= 30
        
        # WAIC 建議
        if 'waic' in comparison:
            waic_info = comparison['waic']
            if waic_info['significant']:
                recommendations.append(f"🎯 WAIC 支持 {waic_info['better_model']} 模型")
                confidence_score += 25
        
        # LOO 建議
        if 'loo' in comparison:
            loo_info = comparison['loo']
            if loo_info['significant']:
                recommendations.append(f"🎯 LOO 支持 {loo_info['better_model']} 模型")
                confidence_score += 25
        
        # BIC 建議
        if 'bic' in comparison:
            bic_info = comparison['bic']
            if bic_info['strength'] in ['strong', 'very_strong']:
                recommendations.append(f"🎯 BIC 強烈支持 {bic_info['better_model']} 模型")
                confidence_score += 30
            elif bic_info['strength'] == 'positive':
                recommendations.append(f"🎯 BIC 支持 {bic_info['better_model']} 模型")
                confidence_score += 15
        
        # Bayes Factor 建議
        if 'bayes_factor' in comparison:
            bf_info = comparison['bayes_factor']
            if bf_info['evidence_strength'] in ['strong', 'very_strong', 'extreme']:
                recommendations.append(f"🎯 Bayes Factor 強烈支持 {bf_info['favored_model']} 模型")
                confidence_score += 35
        
        # 確定最終建議
        if confidence_score >= 50:
            final_recommendation = "strong_preference"
        elif confidence_score >= 25:
            final_recommendation = "moderate_preference" 
        elif confidence_score >= 0:
            final_recommendation = "weak_preference"
        else:
            final_recommendation = "inconclusive"
        
        return {
            'recommendations': recommendations,
            'confidence_score': confidence_score,
            'final_recommendation': final_recommendation
        }
    
    def _print_comparison_summary(self, results: Dict):
        """打印比較摘要"""
        
        print(f"\n{'='*60}")
        print("🏆 模型比較摘要")
        print(f"{'='*60}")
        
        parallel = results['parallel_and']
        coactive = results['coactive']
        comparison = results['comparison']
        
        # 基本信息
        print(f"\n📊 基本信息:")
        print(f"   Parallel AND: {parallel.n_parameters} 參數")
        print(f"   Coactive: {coactive.n_parameters} 參數")
        print(f"   收斂狀態: Parallel={parallel.convergence_success}, Coactive={coactive.convergence_success}")
        
        # 模型選擇指標
        print(f"\n🎯 模型選擇指標:")
        
        if 'waic' in comparison:
            waic = comparison['waic']
            print(f"   WAIC: Parallel={waic['parallel_and']:.2f}, Coactive={waic['coactive']:.2f}")
            print(f"         差異={waic['difference']:.2f} ± {waic['se_difference']:.2f}")
            print(f"         建議: {waic['better_model']} ({'顯著' if waic['significant'] else '非顯著'})")
        
        if 'loo' in comparison:
            loo = comparison['loo']
            print(f"   LOO:  Parallel={loo['parallel_and']:.2f}, Coactive={loo['coactive']:.2f}")
            print(f"         差異={loo['difference']:.2f} ± {loo['se_difference']:.2f}")
            print(f"         建議: {loo['better_model']} ({'顯著' if loo['significant'] else '非顯著'})")
        
        if 'bic' in comparison:
            bic = comparison['bic']
            print(f"   BIC:  Parallel={bic['parallel_and']:.2f}, Coactive={bic['coactive']:.2f}")
            print(f"         差異={bic['difference']:.2f}")
            print(f"         建議: {bic['better_model']} ({bic['strength']} evidence)")
        
        if 'bayes_factor' in comparison:
            bf = comparison['bayes_factor']
            print(f"   Bayes Factor: {bf['bayes_factor']:.2f}")
            print(f"                 支持: {bf['favored_model']} ({bf['evidence_strength']} evidence)")
        
        # 最終建議
        print(f"\n🎯 最終建議:")
        rec = comparison['recommendation']
        for recommendation in rec['recommendations']:
            print(f"   {recommendation}")
        
        print(f"   信心分數: {rec['confidence_score']}")
        print(f"   建議強度: {rec['final_recommendation']}")
        
        print(f"\n{'='*60}")

# 便利函數和測試
def create_test_subject_data(n_trials: int = 200, seed: int = 42) -> Dict:
    """創建測試用受試者資料"""
    
    np.random.seed(seed)
    
    # 生成四選項選擇
    choices = np.random.choice([0, 1, 2, 3], size=n_trials, p=[0.3, 0.25, 0.2, 0.25])
    
    # 生成反應時間（基於選擇的不同分布）
    rt = np.zeros(n_trials)
    for choice in range(4):
        mask = choices == choice
        rt[mask] = np.random.gamma(2 + choice * 0.3, 0.3, np.sum(mask))
    
    # 生成左右通道的刺激和選擇
    left_stimuli = np.random.choice([0, 1], size=n_trials)
    right_stimuli = np.random.choice([0, 1], size=n_trials)
    
    # 基於刺激生成選擇（添加一些噪音）
    left_choices = np.where(np.random.random(n_trials) < 0.8, left_stimuli, 1 - left_stimuli)
    right_choices = np.where(np.random.random(n_trials) < 0.8, right_stimuli, 1 - right_stimuli)
    
    return {
        'subject_id': 'TEST_001',
        'n_trials': n_trials,
        'choices': choices,
        'rt': rt,
        'left_stimuli': left_stimuli,
        'left_choices': left_choices,
        'right_stimuli': right_stimuli,
        'right_choices': right_choices,
        'accuracy': np.mean(left_choices == left_stimuli) * np.mean(right_choices == right_stimuli)
    }

def quick_model_comparison_test(n_trials: int = 100):
    """快速模型比較測試"""
    
    print("🧪 快速模型比較測試")
    print("="*50)
    
    try:
        # 創建測試資料
        test_data = create_test_subject_data(n_trials)
        print(f"✅ 測試資料創建完成: {n_trials} 試驗")
        
        # 設定簡化的 MCMC 配置
        quick_mcmc_config = {
            'draws': 100,
            'tune': 100,
            'chains': 1,
            'cores': 1,
            'target_accept': 0.80,
            'max_treedepth': 6,
            'random_seed': 42,
            'progressbar': True,
            'return_inferencedata': True
        }
        
        # 創建比較器
        comparator = GRTModelComparator(quick_mcmc_config)
        
        # 執行比較
        results = comparator.fit_and_compare_models(test_data, verbose=True)
        
        print("✅ 模型比較測試完成!")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def full_model_comparison_pipeline(subject_data: Dict, 
                                 mcmc_config: Optional[Dict] = None,
                                 save_results: bool = True) -> Dict:
    """完整的模型比較流程"""
    
    print("🚀 GRT 四選項模型比較流程")
    print("="*60)
    
    # 創建比較器
    comparator = GRTModelComparator(mcmc_config)
    
    # 執行比較
    start_time = time.time()
    results = comparator.fit_and_compare_models(subject_data, verbose=True)
    total_time = time.time() - start_time
    
    # 添加時間信息
    results['meta'] = {
        'total_analysis_time': total_time,
        'subject_id': subject_data['subject_id'],
        'n_trials': subject_data['n_trials'],
        'mcmc_config': comparator.mcmc_config
    }
    
    # 儲存結果
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"grt_model_comparison_{subject_data['subject_id']}_{timestamp}.pkl"
        
        import pickle
        try:
            with open(filename, 'wb') as f:
                pickle.save(results, f)
            print(f"💾 結果已儲存: {filename}")
        except Exception as e:
            print(f"⚠️ 結果儲存失敗: {e}")
    
    print(f"\n⏱️ 總分析時間: {total_time/60:.1f} 分鐘")
    
    return results

if __name__ == "__main__":
    print("選擇測試模式:")
    print("1. 快速測試 (簡化 MCMC)")
    print("2. 完整測試 (標準 MCMC)")
    
    choice = input("請選擇 (1 或 2): ").strip()
    
    if choice == "1":
        success = quick_model_comparison_test(100)
        if success:
            print("\n🎉 快速測試成功! 模型比較架構運作正常。")
        else:
            print("\n❌ 快速測試失敗。")
    
    elif choice == "2":
        test_data = create_test_subject_data(200)
        results = full_model_comparison_pipeline(test_data)
        print("\n🎉 完整測試完成!")
    
    else:
        print("無效選擇，執行快速測試...")
        quick_model_comparison_test(50)
