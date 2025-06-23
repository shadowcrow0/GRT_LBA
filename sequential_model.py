# -*- coding: utf-8 -*-
"""
sequential_model_improved.py - 改進的序列處理主模型
參考 Matlab 專案的參數處理方式
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Dict, Optional, Tuple
from single_side_lba import SingleSideLBA
from four_choice_lba import FourChoiceLBA

class SequentialLBA:
    """序列處理LBA主模型 - 改進版"""
    
    def __init__(self, first_side='left', time_split_ratio=0.6):
        """
        初始化序列處理LBA模型
        
        Args:
            first_side: 首先處理的通道 ('left' 或 'right')
            time_split_ratio: 第一階段佔總RT的比例 (0-1)
        """
        
        self.first_side = first_side
        self.second_side = 'right' if first_side == 'left' else 'left'
        self.time_split_ratio = time_split_ratio
        
        # 初始化子模組
        self.first_side_lba = SingleSideLBA(self.first_side)
        self.second_side_lba = SingleSideLBA(self.second_side)
        self.integration_lba = FourChoiceLBA()
        
        # 收集所有參數名稱
        self.all_param_names = (
            self.first_side_lba.param_names + 
            self.second_side_lba.param_names +
            self.integration_lba.param_names
        )
        
        # 設定參數轉換函數（參考 Matlab 的 transformSamples）
        self.param_transforms = self._setup_parameter_transforms()
        
        print(f"✅ 初始化序列處理LBA模型")
        print(f"   處理順序: {self.first_side} → {self.second_side}")
        print(f"   時間分割: {self.time_split_ratio:.1%} / {1-self.time_split_ratio:.1%}")
        print(f"   總參數數: {len(self.all_param_names)}")
    
    def _setup_parameter_transforms(self):
        """設定參數轉換函數（參考 Matlab loadParmSettings.m）"""
        
        transforms = {}
        
        # 漂移率參數 - 使用對數轉換確保正值
        for side in [self.first_side, self.second_side]:
            transforms[f'{side}_drift_correct'] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.1)
            }
            transforms[f'{side}_drift_incorrect'] = {
                'raw_to_natural': lambda x: pt.exp(x), 
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.05)
            }
            
            # 閾值參數
            transforms[f'{side}_threshold'] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.1)
            }
            
            # 起始點變異
            transforms[f'{side}_start_var'] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.05)
            }
            
            # 非決策時間
            transforms[f'{side}_ndt'] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.clip(x, 0.05, 0.8)
            }
            
            # 噪音參數
            transforms[f'{side}_noise'] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.1)
            }
        
        # 整合層參數
        for i in range(4):
            transforms[f'integration_drift_{i}'] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.1)
            }
            
        # 其他整合層參數
        for param in ['integration_threshold', 'integration_start_var', 
                     'integration_ndt', 'integration_noise']:
            transforms[param] = {
                'raw_to_natural': lambda x: pt.exp(x),
                'natural_to_raw': lambda x: pt.log(x),
                'constraint': lambda x: pt.maximum(x, 0.05)
            }
        
        return transforms
    
    def build_model(self, subject_data):
        """
        建構完整的序列處理PyMC模型 - 改進版
        """
        
        print(f"🔧 建構序列處理模型...")
        print(f"   受試者: {subject_data['subject_id']}")
        print(f"   試驗數: {subject_data['n_trials']}")
        
        with pm.Model() as sequential_model:
            
            # ========================================
            # 1. 定義原始參數先驗分布（在轉換後的空間）
            # ========================================
            
            raw_params = {}
            
            # 第一通道參數（在對數空間定義）
            raw_params[f'{self.first_side}_drift_correct_raw'] = pm.Normal(
                f'{self.first_side}_drift_correct_raw', mu=np.log(1.5), sigma=0.2
            )
            raw_params[f'{self.first_side}_drift_incorrect_raw'] = pm.Normal(
                f'{self.first_side}_drift_incorrect_raw', mu=np.log(0.8), sigma=0.2
            )
            raw_params[f'{self.first_side}_threshold_raw'] = pm.Normal(
                f'{self.first_side}_threshold_raw', mu=np.log(1.0), sigma=0.2
            )
            raw_params[f'{self.first_side}_start_var_raw'] = pm.Normal(
                f'{self.first_side}_start_var_raw', mu=np.log(0.3), sigma=0.3
            )
            raw_params[f'{self.first_side}_ndt_raw'] = pm.Normal(
                f'{self.first_side}_ndt_raw', mu=np.log(0.2), sigma=0.2
            )
            raw_params[f'{self.first_side}_noise_raw'] = pm.Normal(
                f'{self.first_side}_noise_raw', mu=np.log(0.3), sigma=0.3
            )
            
            # 第二通道參數
            raw_params[f'{self.second_side}_drift_correct_raw'] = pm.Normal(
                f'{self.second_side}_drift_correct_raw', mu=np.log(1.5), sigma=0.2
            )
            raw_params[f'{self.second_side}_drift_incorrect_raw'] = pm.Normal(
                f'{self.second_side}_drift_incorrect_raw', mu=np.log(0.8), sigma=0.2
            )
            raw_params[f'{self.second_side}_threshold_raw'] = pm.Normal(
                f'{self.second_side}_threshold_raw', mu=np.log(1.0), sigma=0.2
            )
            raw_params[f'{self.second_side}_start_var_raw'] = pm.Normal(
                f'{self.second_side}_start_var_raw', mu=np.log(0.3), sigma=0.3
            )
            raw_params[f'{self.second_side}_ndt_raw'] = pm.Normal(
                f'{self.second_side}_ndt_raw', mu=np.log(0.2), sigma=0.2
            )
            raw_params[f'{self.second_side}_noise_raw'] = pm.Normal(
                f'{self.second_side}_noise_raw', mu=np.log(0.3), sigma=0.3
            )
            
            # 整合層參數
            for i in range(4):
                raw_params[f'integration_drift_{i}_raw'] = pm.Normal(
                    f'integration_drift_{i}_raw', mu=np.log(1.0), sigma=0.2
                )
            
            raw_params['integration_threshold_raw'] = pm.Normal(
                'integration_threshold_raw', mu=np.log(0.8), sigma=0.2
            )
            raw_params['integration_start_var_raw'] = pm.Normal(
                'integration_start_var_raw', mu=np.log(0.2), sigma=0.3
            )
            raw_params['integration_ndt_raw'] = pm.Normal(
                'integration_ndt_raw', mu=np.log(0.15), sigma=0.2
            )
            raw_params['integration_noise_raw'] = pm.Normal(
                'integration_noise_raw', mu=np.log(0.25), sigma=0.3
            )
            
            # ========================================
            # 2. 轉換到自然參數空間並應用約束
            # ========================================
            
            natural_params = {}
            
            # 第一通道參數轉換
            first_side_params = self._transform_side_params(
                raw_params, self.first_side, natural_params
            )
            
            # 第二通道參數轉換
            second_side_params = self._transform_side_params(
                raw_params, self.second_side, natural_params
            )
            
            # 整合層參數轉換
            integration_params = self._transform_integration_params(
                raw_params, natural_params
            )
            
            # ========================================
            # 3. 準備資料張量
            # ========================================
            
            # 轉換為PyTensor張量
            final_choices = pt.as_tensor_variable(subject_data['choices'], dtype='int32')
            rt_total = pt.as_tensor_variable(subject_data['rt'], dtype='float64')
            
            first_stimuli = pt.as_tensor_variable(
                subject_data[f'{self.first_side}_stimuli'], dtype='int32'
            )
            first_choices = pt.as_tensor_variable(
                subject_data[f'{self.first_side}_choices'], dtype='int32'
            )
            
            second_stimuli = pt.as_tensor_variable(
                subject_data[f'{self.second_side}_stimuli'], dtype='int32'
            )
            second_choices = pt.as_tensor_variable(
                subject_data[f'{self.second_side}_choices'], dtype='int32'
            )
            
            # ========================================
            # 4. 時間分割
            # ========================================
            
            rt_first = rt_total * self.time_split_ratio
            rt_second = rt_total * (1 - self.time_split_ratio)
            
            # ========================================
            # 5. 計算似然函數
            # ========================================
            
            # 第一通道似然
            first_likelihood = self._compute_side_likelihood(
                first_choices, first_stimuli, rt_first, first_side_params
            )
            
            # 第二通道似然
            second_likelihood = self._compute_side_likelihood(
                second_choices, second_stimuli, rt_second, second_side_params
            )
            
            # 證據整合
            evidence_inputs = self._compute_evidence_combination_improved(
                first_side_params, second_side_params,
                first_stimuli, first_choices,
                second_stimuli, second_choices,
                subject_data['n_trials']
            )
            
            # 整合層似然
            integration_likelihood = self._compute_integration_likelihood(
                final_choices, evidence_inputs, rt_second, integration_params
            )
            
            # ========================================
            # 6. 添加似然到模型
            # ========================================
            
            pm.Potential('first_side_likelihood', first_likelihood)
            pm.Potential('second_side_likelihood', second_likelihood)
            pm.Potential('integration_likelihood', integration_likelihood)
            
            # 診斷變數
            pm.Deterministic('total_likelihood',
                           first_likelihood + second_likelihood + integration_likelihood)
        
        print(f"✅ 改進模型建構完成")
        print(f"   自由參數: {len(sequential_model.free_RVs)}")
        
        return sequential_model
    
    def _transform_side_params(self, raw_params, side_name, natural_params):
        """轉換單邊參數"""
        
        params = {}
        
        # 應用指數轉換和約束
        drift_correct_raw = raw_params[f'{side_name}_drift_correct_raw']
        drift_incorrect_raw = raw_params[f'{side_name}_drift_incorrect_raw']
        
        drift_correct = pt.maximum(pt.exp(drift_correct_raw), 0.1)
        drift_incorrect = pt.maximum(pt.exp(drift_incorrect_raw), 0.05)
        
        # 確保正確漂移率 > 錯誤漂移率
        drift_correct = pt.maximum(drift_correct, drift_incorrect + 0.05)
        
        params[f'{side_name}_drift_correct'] = drift_correct
        params[f'{side_name}_drift_incorrect'] = drift_incorrect
        
        # 其他參數
        params[f'{side_name}_threshold'] = pt.maximum(
            pt.exp(raw_params[f'{side_name}_threshold_raw']), 0.1
        )
        params[f'{side_name}_start_var'] = pt.maximum(
            pt.exp(raw_params[f'{side_name}_start_var_raw']), 0.05
        )
        params[f'{side_name}_ndt'] = pt.clip(
            pt.exp(raw_params[f'{side_name}_ndt_raw']), 0.05, 0.8
        )
        params[f'{side_name}_noise'] = pt.maximum(
            pt.exp(raw_params[f'{side_name}_noise_raw']), 0.1
        )
        
        return params
    
    def _transform_integration_params(self, raw_params, natural_params):
        """轉換整合層參數"""
        
        params = {}
        
        # 四個選項的漂移率
        for i in range(4):
            params[f'integration_drift_{i}'] = pt.maximum(
                pt.exp(raw_params[f'integration_drift_{i}_raw']), 0.1
            )
        
        # 其他參數
        params['integration_threshold'] = pt.maximum(
            pt.exp(raw_params['integration_threshold_raw']), 0.1
        )
        params['integration_start_var'] = pt.maximum(
            pt.exp(raw_params['integration_start_var_raw']), 0.05
        )
        params['integration_ndt'] = pt.clip(
            pt.exp(raw_params['integration_ndt_raw']), 0.05, 0.3
        )
        params['integration_noise'] = pt.maximum(
            pt.exp(raw_params['integration_noise_raw']), 0.1
        )
        
        return params
    
    def _compute_side_likelihood(self, decisions, stimuli, rt, params):
        """計算單邊似然函數 - 改進版"""
        
        side_name = list(params.keys())[0].split('_')[0]
        
        drift_correct = params[f'{side_name}_drift_correct']
        drift_incorrect = params[f'{side_name}_drift_incorrect']
        threshold = params[f'{side_name}_threshold']
        start_var = params[f'{side_name}_start_var']
        ndt = params[f'{side_name}_ndt']
        noise = params[f'{side_name}_noise']
        
        # 計算決策時間
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        # 判斷正確性
        is_correct = pt.eq(decisions, stimuli)
        
        # 設定winner和loser漂移率
        v_winner = pt.where(is_correct, drift_correct, drift_incorrect)
        v_loser = pt.where(is_correct, drift_incorrect, drift_correct)
        
        # 計算LBA密度
        return self._compute_lba_likelihood(
            decision_time, v_winner, v_loser, threshold, start_var, noise
        )
    
    def _compute_lba_likelihood(self, t, v_winner, v_loser, threshold, start_var, noise):
        """改進的LBA似然計算"""
        
        from pytensor.tensor import erf
        
        sqrt_t = pt.sqrt(t)
        
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(pt.clip(x, -4.5, 4.5) / pt.sqrt(2)))
        
        def safe_normal_pdf(x):
            x_clipped = pt.clip(x, -4.5, 4.5)
            return pt.exp(-0.5 * x_clipped**2) / pt.sqrt(2 * pt.pi)
        
        # Winner累積器
        z1_winner = (v_winner * t - threshold) / (noise * sqrt_t)
        z2_winner = (v_winner * t - start_var) / (noise * sqrt_t)
        
        winner_cdf = safe_normal_cdf(z1_winner) - safe_normal_cdf(z2_winner)
        winner_pdf = (safe_normal_pdf(z1_winner) - safe_normal_pdf(z2_winner)) / (noise * sqrt_t)
        
        winner_density = pt.maximum(
            (v_winner / start_var) * pt.maximum(winner_cdf, 1e-10) + winner_pdf / start_var,
            1e-10
        )
        
        # Loser存活機率
        z1_loser = (v_loser * t - threshold) / (noise * sqrt_t)
        loser_survival = pt.maximum(1 - safe_normal_cdf(z1_loser), 1e-10)
        
        # 聯合似然
        joint_likelihood = winner_density * loser_survival
        joint_likelihood = pt.maximum(joint_likelihood, 1e-12)
        
        log_likelihood = pt.log(joint_likelihood)
        log_likelihood = pt.clip(log_likelihood, -100.0, 10.0)
        
        return pt.sum(log_likelihood)
    
    def _compute_evidence_combination_improved(self, first_params, second_params,
                                             first_stimuli, first_choices,
                                             second_stimuli, second_choices,
                                             n_trials):
        """改進的證據組合計算"""
        
        # 使用參數的期望值來避免張量形狀問題
        first_correct_drift = first_params[f'{self.first_side}_drift_correct']
        first_incorrect_drift = first_params[f'{self.first_side}_drift_incorrect']
        second_correct_drift = second_params[f'{self.second_side}_drift_correct']
        second_incorrect_drift = second_params[f'{self.second_side}_drift_incorrect']
        
        # 計算平均證據強度
        first_evidence_base = (first_correct_drift + first_incorrect_drift) / 2
        second_evidence_base = (second_correct_drift + second_incorrect_drift) / 2
        
        # 處理順序權重
        if self.first_side == 'left':
            left_evidence = first_evidence_base * 1.1
            right_evidence = second_evidence_base * 1.0
        else:
            left_evidence = second_evidence_base * 1.0
            right_evidence = first_evidence_base * 1.1
        
        # 計算四個選項的證據強度
        evidence_inputs = {
            'choice_0': left_evidence * 0.8 + right_evidence * 0.2,  # 左對角右垂直
            'choice_1': left_evidence * 0.8 + right_evidence * 0.8,  # 左對角右對角
            'choice_2': left_evidence * 0.2 + right_evidence * 0.2,  # 左垂直右垂直
            'choice_3': left_evidence * 0.2 + right_evidence * 0.8   # 左垂直右對角
        }
        
        return evidence_inputs
    
    def _compute_integration_likelihood(self, choices, evidence_inputs, rt, params):
        """計算整合層似然"""
        
        # 基礎漂移率
        base_drifts = [
            params[f'integration_drift_{i}'] for i in range(4)
        ]
        
        threshold = params['integration_threshold']
        start_var = params['integration_start_var']
        ndt = params['integration_ndt']
        noise = params['integration_noise']
        
        # 調整漂移率
        adjusted_drifts = []
        for i, base_drift in enumerate(base_drifts):
            evidence_boost = evidence_inputs[f'choice_{i}']
            adjusted_drift = base_drift * (1.0 + evidence_boost * 0.3)
            adjusted_drifts.append(pt.maximum(adjusted_drift, 0.1))
        
        # 計算決策時間
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        # 計算四選一LBA似然
        return self._compute_4choice_lba_likelihood(
            choices, decision_time, adjusted_drifts, threshold, start_var, noise
        )
    
    def _compute_4choice_lba_likelihood(self, choices, t, drifts, threshold, start_var, noise):
        """四選一LBA似然計算"""
        
        # 計算每個選項的密度和存活函數
        densities = []
        survivals = []
        
        for drift in drifts:
            density = self._compute_single_lba_density(t, drift, threshold, start_var, noise)
            survival = self._compute_single_lba_survival(t, drift, threshold, start_var, noise)
            densities.append(density)
            survivals.append(survival)
        
        # 計算每個選項的完整似然
        likelihoods = []
        for i in range(4):
            other_survivals = [survivals[j] for j in range(4) if j != i]
            likelihood = densities[i]
            for survival in other_survivals:
                likelihood = likelihood * survival
            likelihoods.append(likelihood)
        
        # 根據實際選擇選取對應的似然
        trial_likelihoods = pt.zeros_like(t)
        for i in range(4):
            mask = pt.eq(choices, i)
            trial_likelihoods = trial_likelihoods + mask * likelihoods[i]
        
        # 確保正值並取對數
        trial_likelihoods = pt.maximum(trial_likelihoods, 1e-12)
        log_likelihoods = pt.log(trial_likelihoods)
        
        return pt.sum(log_likelihoods)
    
    def _compute_single_lba_density(self, t, drift, threshold, start_var, noise):
        """單一累積器LBA密度"""
        
        from pytensor.tensor import erf
        
        sqrt_t = pt.sqrt(t)
        
        z1 = pt.clip((drift * t - threshold) / (noise * sqrt_t), -4.5, 4.5)
        z2 = pt.clip((drift * t - start_var) / (noise * sqrt_t), -4.5, 4.5)
        
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(x / pt.sqrt(2)))
        
        def safe_normal_pdf(x):
            return pt.exp(-0.5 * x**2) / pt.sqrt(2 * pt.pi)
        
        cdf_term = safe_normal_cdf(z1) - safe_normal_cdf(z2)
        pdf_term = (safe_normal_pdf(z1) - safe_normal_pdf(z2)) / (noise * sqrt_t)
        
        cdf_term = pt.maximum(cdf_term, 1e-10)
        
        density = pt.maximum(
            (drift / start_var) * cdf_term + pdf_term / start_var,
            1e-10
        )
        
        return density
    
    def _compute_single_lba_survival(self, t, drift, threshold, start_var, noise):
        """單一累積器LBA存活函數"""
        
        from pytensor.tensor import erf
        
        sqrt_t = pt.sqrt(t)
        z1 = pt.clip((drift * t - threshold) / (noise * sqrt_t), -4.5, 4.5)
        
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(x / pt.sqrt(2)))
        
        survival = pt.maximum(1 - safe_normal_cdf(z1), 1e-10)
        return survival
    
    def get_model_info(self):
        """獲得模型資訊摘要"""
        
        return {
            'model_type': 'sequential_lba_improved',
            'first_side': self.first_side,
            'second_side': self.second_side,
            'time_split_ratio': self.time_split_ratio,
            'total_parameters': len(self.all_param_names),
            'parameter_names': self.all_param_names,
            'has_parameter_transforms': True,
            'transform_functions': list(self.param_transforms.keys())
        }

# 便利函數
def create_improved_sequential_model(first_side='left', time_split_ratio=0.6):
    """創建改進的序列處理LBA模型"""
    return SequentialLBA(first_side, time_split_ratio)

def test_improved_sequential_model():
    """測試改進的序列模型"""
    
    print("🧪 測試改進的序列處理模型...")
    
    try:
        # 創建測試資料
        n_trials = 50
        np.random.seed(42)
        
        test_subject_data = {
            'subject_id': 999,
            'n_trials': n_trials,
            'choices': np.random.choice([0, 1, 2, 3], size=n_trials),
            'rt': np.random.uniform(0.3, 1.5, size=n_trials),
            'left_stimuli': np.random.choice([0, 1], size=n_trials),
            'left_choices': np.random.choice([0, 1], size=n_trials),
            'right_stimuli': np.random.choice([0, 1], size=n_trials),
            'right_choices': np.random.choice([0, 1], size=n_trials),
            'accuracy': 0.75
        }
        
        # 創建改進的序列模型
        seq_model = SequentialLBA(first_side='left', time_split_ratio=0.6)
        
        # 獲得模型資訊
        model_info = seq_model.get_model_info()
        print(f"   模型類型: {model_info['model_type']}")
        print(f"   總參數數: {model_info['total_parameters']}")
        print(f"   參數轉換: {model_info['has_parameter_transforms']}")
        
        # 嘗試建構PyMC模型
        print("   測試改進的PyMC模型建構...")
        pymc_model = seq_model.build_model(test_subject_data)
        
        # 檢查模型基本性質
        print(f"   自由參數數量: {len(pymc_model.free_RVs)}")
        
        # 測試模型編譯
        with pymc_model:
            test_point = pymc_model.initial_point()
            log_prob = pymc_model.compile_logp()(test_point)
            print(f"   測試對數機率: {log_prob:.2f}")
            
            if np.isfinite(log_prob):
                print("   ✅ 改進模型編譯成功")
            else:
                print("   ⚠️ 警告: 模型初始對數機率無效")
        
        print("✅ 改進的序列模型測試成功!")
        return True
        
    except Exception as e:
        print(f"❌ 改進的序列模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_improved_sequential_model()
