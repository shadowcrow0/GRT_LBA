# -*- coding: utf-8 -*-
"""
four_choice_lba.py - 四選一LBA競爭器
Sequential Processing LBA - Four-Choice LBA Integration Module

功能：
- 實現四選一LBA競爭機制
- 整合左右通道證據輸出
- 計算最終決策的似然函數
- 支援PyTensor和PyMC
"""

import numpy as np
import pytensor.tensor as pt
from typing import Dict, List, Optional, Tuple

class FourChoiceLBA:
    """四選一LBA競爭器"""
    
    def __init__(self):
        """初始化四選一LBA競爭器"""
        
        # 四個選項的對應關係
        self.choice_descriptions = {
            0: 'Left\\Right|',   # 左對角右垂直
            1: 'Left\\Right/',   # 左對角右對角  
            2: 'Left|Right|',    # 左垂直右垂直
            3: 'Left|Right/'     # 左垂直右對角
        }
        
        # 整合層參數名稱
        self.param_names = [
            'integration_drift_0',      # 選項0的基礎漂移率
            'integration_drift_1',      # 選項1的基礎漂移率  
            'integration_drift_2',      # 選項2的基礎漂移率
            'integration_drift_3',      # 選項3的基礎漂移率
            'integration_threshold',    # 整合層決策閾值
            'integration_start_var',    # 整合層起始點變異
            'integration_ndt',          # 整合層非決策時間
            'integration_noise'         # 整合層擴散噪音
        ]
        
        print("✅ 初始化四選一LBA競爭器")
        print(f"   參數數量: {len(self.param_names)}")
        print("   選項對應:")
        for choice, desc in self.choice_descriptions.items():
            print(f"     選項 {choice}: {desc}")
    
    def compute_likelihood(self, choices, evidence_inputs, rt_remaining, params):
        """
        計算四選一LBA競爭的似然函數
        
        Args:
            choices: 最終選擇陣列 (0, 1, 2, 3)
            evidence_inputs: 來自雙通道的證據輸入字典
            rt_remaining: 剩餘反應時間（用於整合層）
            params: 整合層參數字典
            
        Returns:
            log_likelihood: 對數似然值
        """
        
        # 解包基礎參數
        drift_0 = params['integration_drift_0']
        drift_1 = params['integration_drift_1'] 
        drift_2 = params['integration_drift_2']
        drift_3 = params['integration_drift_3']
        threshold = params['integration_threshold']
        start_var = params['integration_start_var']
        ndt = params['integration_ndt']
        noise = params['integration_noise']
        
        # 應用參數邊界約束
        drifts = [
            pt.maximum(drift_0, 0.1),
            pt.maximum(drift_1, 0.1),
            pt.maximum(drift_2, 0.1), 
            pt.maximum(drift_3, 0.1)
        ]
        threshold = pt.maximum(threshold, 0.1)
        start_var = pt.maximum(start_var, 0.05)
        ndt = pt.maximum(ndt, 0.05)
        noise = pt.maximum(noise, 0.1)
        
        # 計算整合層的決策時間
        decision_time = pt.maximum(rt_remaining - ndt, 0.01)
        
        # 根據證據輸入調整漂移率
        adjusted_drifts = self._adjust_drifts_with_evidence(drifts, evidence_inputs)
        
        # 計算四選一LBA競爭似然
        log_likelihood = self._compute_4choice_lba_density(
            choices, decision_time, adjusted_drifts, threshold, start_var, noise
        )
        
        return log_likelihood
    
    def _adjust_drifts_with_evidence(self, base_drifts, evidence_inputs):
        """
        根據雙通道證據調整各選項的漂移率
        
        Args:
            base_drifts: 基礎漂移率列表 [drift_0, drift_1, drift_2, drift_3]
            evidence_inputs: 證據輸入字典
            
        Returns:
            adjusted_drifts: 調整後的漂移率列表
        """
        
        adjusted_drifts = []
        
        for i, base_drift in enumerate(base_drifts):
            # 獲得對應選項的證據加成
            evidence_boost = evidence_inputs.get(f'choice_{i}', 0.0)
            
            # 組合基礎漂移率和證據加成
            # 使用乘法組合以保持正值
            adjusted_drift = base_drift * (1.0 + evidence_boost * 0.5)
            
            # 確保最小值
            adjusted_drift = pt.maximum(adjusted_drift, 0.1)
            
            adjusted_drifts.append(adjusted_drift)
        
        return adjusted_drifts
    
    def _compute_4choice_lba_density(self, choices, decision_time, drifts, 
                                   threshold, start_var, noise):
        """
        計算四選一LBA密度函數 - 完全向量化版本
        
        避免Python循環，使用完全向量化的PyTensor操作
        """
        
        # 預先計算所有選項的密度和存活函數（向量化）
        densities_0 = self._compute_lba_density(decision_time, drifts[0], threshold, start_var, noise)
        densities_1 = self._compute_lba_density(decision_time, drifts[1], threshold, start_var, noise)
        densities_2 = self._compute_lba_density(decision_time, drifts[2], threshold, start_var, noise)
        densities_3 = self._compute_lba_density(decision_time, drifts[3], threshold, start_var, noise)
        
        survivals_0 = self._compute_lba_survival(decision_time, drifts[0], threshold, start_var, noise)
        survivals_1 = self._compute_lba_survival(decision_time, drifts[1], threshold, start_var, noise)
        survivals_2 = self._compute_lba_survival(decision_time, drifts[2], threshold, start_var, noise)
        survivals_3 = self._compute_lba_survival(decision_time, drifts[3], threshold, start_var, noise)
        
        # 計算每個選項的完整似然（winner density × all loser survivals）
        likelihood_0 = densities_0 * survivals_1 * survivals_2 * survivals_3
        likelihood_1 = densities_1 * survivals_0 * survivals_2 * survivals_3
        likelihood_2 = densities_2 * survivals_0 * survivals_1 * survivals_3
        likelihood_3 = densities_3 * survivals_0 * survivals_1 * survivals_2
        
        # 根據實際選擇選取對應的似然（向量化方式）
        trial_likelihoods = (
            pt.eq(choices, 0) * likelihood_0 +
            pt.eq(choices, 1) * likelihood_1 +
            pt.eq(choices, 2) * likelihood_2 +
            pt.eq(choices, 3) * likelihood_3
        )
        
        # 確保正值並取對數
        trial_likelihoods = pt.maximum(trial_likelihoods, 1e-12)
        log_likelihoods = pt.log(trial_likelihoods)
        
        # 返回總對數似然
        return pt.sum(log_likelihoods)
    
    def _compute_lba_density(self, t, drift, threshold, start_var, noise):
        """
        計算單一累積器的LBA密度函數 - 支援向量化輸入
        
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
        )
        
        return density
    
    def _compute_lba_survival(self, t, drift, threshold, start_var, noise):
        """
        計算單一累積器的LBA存活函數 - 支援向量化輸入
        
        Args:
            t: 決策時間（可以是向量）
            drift: 漂移率
            threshold: 閾值
            start_var: 起始點變異
            noise: 噪音參數
            
        Returns:
            survival: 存活機率（與t相同形狀）
        """
        
        from pytensor.tensor import erf
        
        sqrt_t = pt.sqrt(t)
        
        # 計算z-score（向量化）
        z1 = pt.clip((drift * t - threshold) / (noise * sqrt_t), -4.5, 4.5)
        
        # 正態CDF
        def safe_normal_cdf(x):
            return 0.5 * (1 + erf(x / pt.sqrt(2)))
        
        # 存活機率 = 1 - CDF（向量化）
        survival = pt.maximum(1 - safe_normal_cdf(z1), 1e-10)
        
        return survival
    
    def combine_channel_evidence(self, left_evidence, right_evidence, 
                               first_side='left'):
        """
        組合雙通道證據形成四個選項的證據輸入
        
        Args:
            left_evidence: 左通道證據字典
            right_evidence: 右通道證據字典
            first_side: 首先處理的通道 ('left' 或 'right')
            
        Returns:
            evidence_inputs: 四個選項的證據字典
        """
        
        # 提取證據值
        left_vertical = left_evidence.get('evidence_vertical', 0.0)
        left_diagonal = left_evidence.get('evidence_diagonal', 0.0)
        right_vertical = right_evidence.get('evidence_vertical', 0.0)
        right_diagonal = right_evidence.get('evidence_diagonal', 0.0)
        
        # 考慮處理順序的影響
        if first_side == 'left':
            # 左邊先處理，可能有更高的權重
            left_weight = 1.1
            right_weight = 1.0
        else:
            # 右邊先處理
            left_weight = 1.0
            right_weight = 1.1
        
        # 組合證據形成四個選項
        evidence_inputs = {
            'choice_0': left_diagonal * left_weight + right_vertical * right_weight,    # \|
            'choice_1': left_diagonal * left_weight + right_diagonal * right_weight,   # \/  
            'choice_2': left_vertical * left_weight + right_vertical * right_weight,   # ||
            'choice_3': left_vertical * left_weight + right_diagonal * right_weight    # |/
        }
        
        # 正規化以防止過大的證據值
        max_evidence = max(evidence_inputs.values())
        if max_evidence > 0:
            scale_factor = min(2.0 / max_evidence, 1.0)  # 限制最大證據為2.0
            for key in evidence_inputs:
                evidence_inputs[key] *= scale_factor
        
        return evidence_inputs
    
    def compute_choice_probabilities(self, evidence_inputs, params, rt_mean=None):
        """
        計算四選一的選擇機率（用於模型預測）
        
        Args:
            evidence_inputs: 證據輸入字典
            params: 整合層參數字典
            rt_mean: 平均反應時間
            
        Returns:
            choice_probs: 四個選項的選擇機率
        """
        
        if rt_mean is None:
            rt_mean = 0.8
        
        # 解包參數
        base_drifts = [
            float(params['integration_drift_0']),
            float(params['integration_drift_1']),
            float(params['integration_drift_2']),
            float(params['integration_drift_3'])
        ]
        threshold = float(params['integration_threshold'])
        ndt = float(params['integration_ndt'])
        
        # 計算決策時間
        decision_time = max(rt_mean - ndt, 0.1)
        
        # 調整漂移率
        adjusted_drifts = []
        for i, base_drift in enumerate(base_drifts):
            evidence_boost = evidence_inputs.get(f'choice_{i}', 0.0)
            adjusted_drift = base_drift * (1.0 + evidence_boost * 0.5)
            adjusted_drifts.append(max(adjusted_drift, 0.1))
        
        # 計算相對強度（簡化版本）
        relative_strengths = []
        for drift in adjusted_drifts:
            # 使用簡化的指數函數計算相對強度
            strength = np.exp(drift * decision_time / threshold)
            relative_strengths.append(strength)
        
        # 正規化為機率
        total_strength = sum(relative_strengths)
        choice_probs = [s / total_strength for s in relative_strengths]
        
        return np.array(choice_probs)
    
    def validate_parameters(self, params):
        """
        驗證整合層參數的合理性
        
        Args:
            params: 參數字典
            
        Returns:
            bool: 參數是否合理
            str: 驗證訊息
        """
        
        try:
            # 檢查所有必要參數是否存在
            for param_name in self.param_names:
                if param_name not in params:
                    return False, f"缺少參數: {param_name}"
            
            # 檢查參數值範圍
            for i in range(4):
                drift = float(params[f'integration_drift_{i}'])
                if drift <= 0:
                    return False, f"integration_drift_{i}必須 > 0，得到: {drift}"
            
            threshold = float(params['integration_threshold'])
            if threshold <= 0:
                return False, f"integration_threshold必須 > 0，得到: {threshold}"
            
            start_var = float(params['integration_start_var'])
            if start_var <= 0 or start_var >= threshold:
                return False, f"integration_start_var必須在 (0, threshold) 範圍內，得到: {start_var}"
            
            ndt = float(params['integration_ndt'])
            if ndt < 0 or ndt > 0.8:
                return False, f"integration_ndt必須在 [0, 0.8] 範圍內，得到: {ndt}"
            
            noise = float(params['integration_noise'])
            if noise <= 0:
                return False, f"integration_noise必須 > 0，得到: {noise}"
            
            return True, "整合層參數驗證通過"
            
        except (ValueError, KeyError) as e:
            return False, f"參數驗證錯誤: {e}"
    
    def get_default_priors(self):
        """
        獲得整合層參數的預設先驗分布設定
        
        Returns:
            dict: 先驗分布設定
        """
        
        priors = {}
        
        # 四個選項的基礎漂移率
        for i in range(4):
            priors[f'integration_drift_{i}'] = {
                'distribution': 'Gamma',
                'alpha': 2.0,
                'beta': 2.0,
                'description': f'選項{i}的基礎漂移率'
            }
        
        # 整合層其他參數
        priors.update({
            'integration_threshold': {
                'distribution': 'Gamma',
                'alpha': 2.5,
                'beta': 3.0,
                'description': '整合層決策閾值'
            },
            'integration_start_var': {
                'distribution': 'Uniform',
                'lower': 0.1,
                'upper': 0.5,
                'description': '整合層起始點變異'
            },
            'integration_ndt': {
                'distribution': 'Uniform',
                'lower': 0.05,
                'upper': 0.3,
                'description': '整合層非決策時間'
            },
            'integration_noise': {
                'distribution': 'Gamma',
                'alpha': 2.0,
                'beta': 6.0,
                'description': '整合層擴散噪音'
            }
        })
        
        return priors

# 便利函數
def create_four_choice_lba():
    """創建四選一LBA競爭器"""
    return FourChoiceLBA()

def test_four_choice_lba():
    """測試四選一LBA功能"""
    
    print("🧪 測試四選一LBA競爭器...")
    
    try:
        # 創建測試資料
        n_trials = 50
        np.random.seed(42)
        
        choices = np.random.choice([0, 1, 2, 3], size=n_trials)
        rt_remaining = np.random.uniform(0.2, 0.8, size=n_trials)
        
        # 創建四選一LBA
        four_choice_lba = FourChoiceLBA()
        
        # 測試參數
        test_params = {
            'integration_drift_0': 1.2,
            'integration_drift_1': 1.1,
            'integration_drift_2': 1.0,
            'integration_drift_3': 1.3,
            'integration_threshold': 0.8,
            'integration_start_var': 0.2,
            'integration_ndt': 0.15,
            'integration_noise': 0.25
        }
        
        # 測試參數驗證
        valid, message = four_choice_lba.validate_parameters(test_params)
        print(f"   參數驗證: {message}")
        
        if not valid:
            print("❌ 參數驗證失敗")
            return False
        
        # 測試證據組合
        left_evidence = {'evidence_vertical': 0.8, 'evidence_diagonal': 1.2}
        right_evidence = {'evidence_vertical': 1.0, 'evidence_diagonal': 0.9}
        
        evidence_inputs = four_choice_lba.combine_channel_evidence(
            left_evidence, right_evidence, 'left'
        )
        print(f"   證據組合: {len(evidence_inputs)} 個選項")
        
        # 測試選擇機率計算
        choice_probs = four_choice_lba.compute_choice_probabilities(
            evidence_inputs, test_params
        )
        print(f"   選擇機率: {choice_probs}")
        print(f"   機率總和: {np.sum(choice_probs):.3f}")
        
        # 測試先驗設定
        priors = four_choice_lba.get_default_priors()
        print(f"   先驗分布數量: {len(priors)}")
        
        print("✅ 四選一LBA測試成功!")
        return True
        
    except Exception as e:
        print(f"❌ 四選一LBA測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 如果直接執行此檔案，進行測試
    test_four_choice_lba()
