def _compute_lba_density(self, t, drift, threshold, start_var, noise):
        """
        計算單一累積器的LBA密度函數 - 支援向量化
        
        Args:
            t: 決策時間（可以是向量）
            drift: 漂移率
            threshold: 閾值
            start_var: 起始點變異
            noise: 噪音參數
            
        Returns:
            density: 密度值（與t相同形狀）
        """
        
        from pytensor.tensor import erf
        
        sqrt_t = pt.sqrt(t)
        
        # 計算z-scores（向量化）
        z1 = pt.clip((drift * t - threshold) / (noise * sqrt_t), -4.5, 4.5)
        z2 = pt.clip((drift * t - start_var) / (noise * sqrt_t), -4.5, 4.5)
        
        # PyTensor兼容的正態函數
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(x / pt.sqrt(2)))
        
        def safe_normal_pdf(x):
            return pt.exp(-0.5 * x**2) / pt.sqrt(2 * pt.pi)
        
        # CDF項和PDF項（向量化）
        cdf_term = safe_normal_cdf(z1) - safe_normal_cdf(z2)
        pdf_term = (safe_normal_pdf(z1) - safe_normal_pdf(z2)) / (noise * sqrt_t)
        
        # 確保CDF項為正
        cdf_term = pt.maximum(cdf_term, 1e-10)
        
        # 完整密度計算（向量化）
        density = pt.maximum(
            (drift / start_var) * cdf_term + pdf_term / start_var,
            1e-10
        )# -*- coding: utf-8 -*-
"""
sequential_model.py - 序列處理主模型
Sequential Processing LBA - Main Sequential Model

功能：
- 整合單邊LBA和四選一LBA
- 實現序列處理架構
- 建構完整的PyMC模型
- 支援不同的處理順序和時間分割
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Dict, Optional, Tuple
from single_side_lba import SingleSideLBA
from four_choice_lba import FourChoiceLBA

class SequentialLBA:
    """序列處理LBA主模型"""
    
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
        
        print(f"✅ 初始化序列處理LBA模型")
        print(f"   處理順序: {self.first_side} → {self.second_side}")
        print(f"   時間分割: {self.time_split_ratio:.1%} / {1-self.time_split_ratio:.1%}")
        print(f"   總參數數: {len(self.all_param_names)}")
        print(f"     第一通道: {len(self.first_side_lba.param_names)}")
        print(f"     第二通道: {len(self.second_side_lba.param_names)}")  
        print(f"     整合層: {len(self.integration_lba.param_names)}")
    
    def build_model(self, subject_data):
        """
        建構完整的序列處理PyMC模型
        
        Args:
            subject_data: 受試者資料字典
            
        Returns:
            pymc.Model: 完整的PyMC模型
        """
        
        print(f"🔧 建構序列處理模型...")
        print(f"   受試者: {subject_data['subject_id']}")
        print(f"   試驗數: {subject_data['n_trials']}")
        
        with pm.Model() as sequential_model:
            
            # ========================================
            # 1. 定義參數先驗分布
            # ========================================
            
            # 第一通道參數
            first_side_params = self._define_side_params(self.first_side)
            
            # 第二通道參數  
            second_side_params = self._define_side_params(self.second_side)
            
            # 整合層參數
            integration_params = self._define_integration_params()
            
            # ========================================
            # 2. 準備資料張量
            # ========================================
            
            # 原始資料
            final_choices = pt.as_tensor_variable(subject_data['choices'])
            rt_total = pt.as_tensor_variable(subject_data['rt'])
            
            # 第一通道資料
            first_stimuli = pt.as_tensor_variable(subject_data[f'{self.first_side}_stimuli'])
            first_choices = pt.as_tensor_variable(subject_data[f'{self.first_side}_choices'])
            
            # 第二通道資料
            second_stimuli = pt.as_tensor_variable(subject_data[f'{self.second_side}_stimuli'])
            second_choices = pt.as_tensor_variable(subject_data[f'{self.second_side}_choices'])
            
            # ========================================
            # 3. 時間分割
            # ========================================
            
            rt_first = rt_total * self.time_split_ratio
            rt_second = rt_total * (1 - self.time_split_ratio)
            
            # ========================================
            # 4. 第一通道似然
            # ========================================
            
            first_likelihood = self.first_side_lba.compute_likelihood(
                first_choices, first_stimuli, rt_first, first_side_params
            )
            
            # ========================================
            # 5. 第二通道似然
            # ========================================
            
            second_likelihood = self.second_side_lba.compute_likelihood(
                second_choices, second_stimuli, rt_second, second_side_params
            )
            
            # ========================================
            # 6. 證據整合和四選一競爭
            # ========================================
            
            # 計算證據組合（簡化版）
            evidence_inputs = self._compute_evidence_combination(
                first_side_params, second_side_params, 
                first_stimuli, first_choices, 
                second_stimuli, second_choices
            )
            
            # 整合層似然
            integration_likelihood = self.integration_lba.compute_likelihood(
                final_choices, evidence_inputs, rt_second, integration_params
            )
            
            # ========================================
            # 7. 添加似然到模型
            # ========================================
            
            pm.Potential('first_side_likelihood', first_likelihood)
            pm.Potential('second_side_likelihood', second_likelihood)
            pm.Potential('integration_likelihood', integration_likelihood)
            
            # ========================================
            # 8. 模型診斷資訊
            # ========================================
            
            # 添加一些診斷變數（可選）
            pm.Deterministic('total_likelihood', 
                           first_likelihood + second_likelihood + integration_likelihood)
            
            # 計算理論準確率
            first_accuracy_theory = self._compute_theoretical_accuracy(first_side_params)
            second_accuracy_theory = self._compute_theoretical_accuracy(second_side_params)
            
            pm.Deterministic('first_side_accuracy_theory', first_accuracy_theory)
            pm.Deterministic('second_side_accuracy_theory', second_accuracy_theory)
        
        print(f"✅ 模型建構完成")
        print(f"   自由參數: {len(sequential_model.free_RVs)}")
        print(f"   觀察變數: {len(sequential_model.observed_RVs)}")
        
        return sequential_model
    
    def _define_side_params(self, side_name):
        """定義單邊通道的參數先驗分布"""
        
        # 獲得預設先驗設定
        if side_name == self.first_side:
            lba = self.first_side_lba
        else:
            lba = self.second_side_lba
            
        priors = lba.get_default_priors()
        
        params = {}
        
        # 漂移率參數
        params[f'{side_name}_drift_correct'] = pm.Gamma(
            f'{side_name}_drift_correct', 
            alpha=priors[f'{side_name}_drift_correct']['alpha'],
            beta=priors[f'{side_name}_drift_correct']['beta']
        )
        
        params[f'{side_name}_drift_incorrect'] = pm.Gamma(
            f'{side_name}_drift_incorrect',
            alpha=priors[f'{side_name}_drift_incorrect']['alpha'],
            beta=priors[f'{side_name}_drift_incorrect']['beta']
        )
        
        # 閾值參數
        params[f'{side_name}_threshold'] = pm.Gamma(
            f'{side_name}_threshold',
            alpha=priors[f'{side_name}_threshold']['alpha'],
            beta=priors[f'{side_name}_threshold']['beta']
        )
        
        # 起始點變異
        params[f'{side_name}_start_var'] = pm.Uniform(
            f'{side_name}_start_var',
            lower=priors[f'{side_name}_start_var']['lower'],
            upper=priors[f'{side_name}_start_var']['upper']
        )
        
        # 非決策時間
        params[f'{side_name}_ndt'] = pm.Uniform(
            f'{side_name}_ndt',
            lower=priors[f'{side_name}_ndt']['lower'],
            upper=priors[f'{side_name}_ndt']['upper']
        )
        
        # 噪音參數
        params[f'{side_name}_noise'] = pm.Gamma(
            f'{side_name}_noise',
            alpha=priors[f'{side_name}_noise']['alpha'],
            beta=priors[f'{side_name}_noise']['beta']
        )
        
        return params
    
    def _define_integration_params(self):
        """定義整合層參數先驗分布"""
        
        priors = self.integration_lba.get_default_priors()
        params = {}
        
        # 四個選項的漂移率
        for i in range(4):
            param_name = f'integration_drift_{i}'
            params[param_name] = pm.Gamma(
                param_name,
                alpha=priors[param_name]['alpha'],
                beta=priors[param_name]['beta']
            )
        
        # 其他整合層參數
        params['integration_threshold'] = pm.Gamma(
            'integration_threshold',
            alpha=priors['integration_threshold']['alpha'],
            beta=priors['integration_threshold']['beta']
        )
        
        params['integration_start_var'] = pm.Uniform(
            'integration_start_var',
            lower=priors['integration_start_var']['lower'],
            upper=priors['integration_start_var']['upper']
        )
        
        params['integration_ndt'] = pm.Uniform(
            'integration_ndt',
            lower=priors['integration_ndt']['lower'],
            upper=priors['integration_ndt']['upper']
        )
        
        params['integration_noise'] = pm.Gamma(
            'integration_noise',
            alpha=priors['integration_noise']['alpha'],
            beta=priors['integration_noise']['beta']
        )
        
        return params
    
    def _compute_evidence_combination(self, first_params, second_params, 
                                    first_stimuli, first_choices, 
                                    second_stimuli, second_choices):
        """
        計算證據組合（簡化版實現）- 修復PyTensor兼容性
        
        使用參數值作為證據強度的代理，避免形狀問題
        """
        
        # 提取漂移率作為證據強度
        first_correct = first_params[f'{self.first_side}_drift_correct']
        first_incorrect = first_params[f'{self.first_side}_drift_incorrect']
        second_correct = second_params[f'{self.second_side}_drift_correct']
        second_incorrect = second_params[f'{self.second_side}_drift_incorrect']
        
        # 計算每個通道的平均證據強度（避免逐個trial計算）
        # 使用期望值而非試驗特定值來避免張量形狀問題
        
        # 第一通道的期望證據
        first_vertical_prob = pt.mean(pt.eq(first_stimuli, 0).astype('float32'))
        first_diagonal_prob = 1.0 - first_vertical_prob
        
        first_evidence_vertical = first_vertical_prob * first_correct + (1 - first_vertical_prob) * first_incorrect
        first_evidence_diagonal = first_diagonal_prob * first_correct + (1 - first_diagonal_prob) * first_incorrect
        
        # 第二通道的期望證據
        second_vertical_prob = pt.mean(pt.eq(second_stimuli, 0).astype('float32'))
        second_diagonal_prob = 1.0 - second_vertical_prob
        
        second_evidence_vertical = second_vertical_prob * second_correct + (1 - second_vertical_prob) * second_incorrect
        second_evidence_diagonal = second_diagonal_prob * second_correct + (1 - second_diagonal_prob) * second_incorrect
        
        # 處理順序權重
        if self.first_side == 'left':
            left_weight = 1.1  # 先處理的通道有輕微優勢
            right_weight = 1.0
            left_vertical = first_evidence_vertical * left_weight
            left_diagonal = first_evidence_diagonal * left_weight
            right_vertical = second_evidence_vertical * right_weight
            right_diagonal = second_evidence_diagonal * right_weight
        else:
            left_weight = 1.0
            right_weight = 1.1
            left_vertical = second_evidence_vertical * left_weight
            left_diagonal = second_evidence_diagonal * left_weight
            right_vertical = first_evidence_vertical * right_weight
            right_diagonal = first_evidence_diagonal * right_weight
        
        # 組合成四個選項的證據（使用標量值）
        evidence_inputs = {
            'choice_0': left_diagonal + right_vertical,    # \|
            'choice_1': left_diagonal + right_diagonal,   # \/
            'choice_2': left_vertical + right_vertical,   # ||
            'choice_3': left_vertical + right_diagonal    # |/
        }
        
        return evidence_inputs
    
    def _compute_theoretical_accuracy(self, side_params):
        """計算理論準確率（用於模型診斷）"""
        
        side_name = list(side_params.keys())[0].split('_')[0]  # 提取side名稱
        
        drift_correct = side_params[f'{side_name}_drift_correct']
        drift_incorrect = side_params[f'{side_name}_drift_incorrect']
        
        # 簡化的準確率估計
        evidence_ratio = drift_correct / (drift_correct + drift_incorrect)
        
        return evidence_ratio
    
    def validate_model_setup(self, subject_data):
        """
        驗證模型設定的合理性
        
        Args:
            subject_data: 受試者資料
            
        Returns:
            bool: 設定是否合理
            str: 驗證訊息
        """
        
        try:
            # 檢查必要的資料欄位
            required_fields = [
                'subject_id', 'n_trials', 'choices', 'rt',
                f'{self.first_side}_stimuli', f'{self.first_side}_choices',
                f'{self.second_side}_stimuli', f'{self.second_side}_choices'
            ]
            
            for field in required_fields:
                if field not in subject_data:
                    return False, f"缺少必要資料欄位: {field}"
            
            # 檢查資料長度一致性
            n_trials = subject_data['n_trials']
            for field in ['choices', 'rt', f'{self.first_side}_stimuli', 
                         f'{self.first_side}_choices', f'{self.second_side}_stimuli', 
                         f'{self.second_side}_choices']:
                if len(subject_data[field]) != n_trials:
                    return False, f"資料長度不一致: {field} 有 {len(subject_data[field])} 個元素，期待 {n_trials}"
            
            # 檢查時間分割比例
            if not 0.1 <= self.time_split_ratio <= 0.9:
                return False, f"時間分割比例不合理: {self.time_split_ratio}，應在 [0.1, 0.9] 範圍內"
            
            # 檢查RT範圍
            rt_array = subject_data['rt']
            if np.any(rt_array <= 0):
                return False, "發現非正值的反應時間"
            
            min_rt_required = 0.15  # 最小可能的RT
            if np.any(rt_array < min_rt_required):
                return False, f"發現過短的反應時間 (< {min_rt_required}s)"
            
            # 檢查選擇值範圍
            choices = subject_data['choices']
            if not np.all(np.isin(choices, [0, 1, 2, 3])):
                return False, "最終選擇包含無效值（應為0,1,2,3）"
            
            for side in [self.first_side, self.second_side]:
                side_choices = subject_data[f'{side}_choices']
                side_stimuli = subject_data[f'{side}_stimuli']
                
                if not np.all(np.isin(side_choices, [0, 1])):
                    return False, f"{side}通道選擇包含無效值（應為0,1）"
                
                if not np.all(np.isin(side_stimuli, [0, 1])):
                    return False, f"{side}通道刺激包含無效值（應為0,1）"
            
            return True, "模型設定驗證通過"
            
        except Exception as e:
            return False, f"驗證過程發生錯誤: {e}"
    
    def get_model_info(self):
        """獲得模型資訊摘要"""
        
        return {
            'model_type': 'sequential_lba',
            'first_side': self.first_side,
            'second_side': self.second_side,
            'time_split_ratio': self.time_split_ratio,
            'total_parameters': len(self.all_param_names),
            'first_side_parameters': len(self.first_side_lba.param_names),
            'second_side_parameters': len(self.second_side_lba.param_names),
            'integration_parameters': len(self.integration_lba.param_names),
            'parameter_names': self.all_param_names
        }

# 便利函數
def create_sequential_model(first_side='left', time_split_ratio=0.6):
    """創建序列處理LBA模型"""
    return SequentialLBA(first_side, time_split_ratio)

def test_sequential_model():
    """測試序列模型功能"""
    
    print("🧪 測試序列處理模型...")
    
    try:
        # 創建測試資料
        n_trials = 100
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
        
        # 創建序列模型
        seq_model = SequentialLBA(first_side='left', time_split_ratio=0.6)
        
        # 驗證模型設定
        valid, message = seq_model.validate_model_setup(test_subject_data)
        print(f"   模型設定驗證: {message}")
        
        if not valid:
            print("❌ 模型設定驗證失敗")
            return False
        
        # 獲得模型資訊
        model_info = seq_model.get_model_info()
        print(f"   模型類型: {model_info['model_type']}")
        print(f"   總參數數: {model_info['total_parameters']}")
        print(f"   處理順序: {model_info['first_side']} → {model_info['second_side']}")
        
        # 嘗試建構PyMC模型（不進行採樣）
        print("   測試PyMC模型建構...")
        pymc_model = seq_model.build_model(test_subject_data)
        
        # 檢查模型基本性質
        print(f"   自由參數數量: {len(pymc_model.free_RVs)}")
        print(f"   觀察變數數量: {len(pymc_model.observed_RVs)}")
        
        # 測試模型編譯（基本檢查）
        with pymc_model:
            test_point = pymc_model.initial_point()
            log_prob = pymc_model.compile_logp()(test_point)
            print(f"   測試對數機率: {log_prob:.2f}")
            
            if not np.isfinite(log_prob):
                print("⚠️ 警告: 模型初始對數機率無效")
            else:
                print("   模型編譯成功")
        
        print("✅ 序列模型測試成功!")
        return True
        
    except Exception as e:
        print(f"❌ 序列模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 如果直接執行此檔案，進行測試
    test_sequential_model()
