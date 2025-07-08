# -*- coding: utf-8 -*-
"""
single_side_lba.py - 單邊LBA處理器
Sequential Processing LBA - Single Side LBA Processor

功能：
- 實現單邊(左或右)的2選擇LBA模型
- 計算垂直線 vs 對角線的2選擇似然函數
- 支援PyTensor和PyMC
- 提供證據輸出用於後續整合
"""

import numpy as np
import pytensor.tensor as pt
from typing import NamedTuple, Dict, Optional

class SingleSideResult(NamedTuple):
    """單邊處理結果"""
    evidence_vertical: float      # 垂直線證據強度
    evidence_diagonal: float     # 對角線證據強度
    processing_time: float       # 處理時間
    confidence: float           # 決策信心度
    choice_probability: float   # 選擇機率

class SingleSideLBA:
    """單邊LBA處理器"""
    
    def __init__(self, side_name: str = 'left'):
        """
        初始化單邊LBA處理器
        
        Args:
            side_name: 'left' 或 'right'
        """
        self.side_name = side_name
        
        # Enhanced parameter structure for dual stimulus types
        self.param_names = [
            # Drift rates for different stimulus-response combinations
            f'{side_name}_v_vertical',          # Drift for vertical response when stimulus is vertical
            f'{side_name}_v_nonvertical',       # Drift for nonvertical response when stimulus is nonvertical
            f'{side_name}_v_vertical_error',    # Drift for vertical response when stimulus is nonvertical
            f'{side_name}_v_nonvertical_error', # Drift for nonvertical response when stimulus is vertical
            
            # Common LBA parameters
            f'{side_name}_threshold',           # 決策閾值
            f'{side_name}_start_var',           # 起始點變異
            f'{side_name}_ndt',                 # 非決策時間
            f'{side_name}_noise'                # 擴散噪音
        ]
        
        print(f"✅ 初始化 {side_name} 通道LBA處理器")
        print(f"   參數數量: {len(self.param_names)}")
    
    def compute_likelihood(self, decisions, stimuli, rt, params):
        """
        計算增強版2選擇LBA似然函數
        
        Args:
            decisions: 決策陣列 (0=vertical response, 1=nonvertical response)
            stimuli: 刺激陣列 (0=vertical stimulus, 1=nonvertical stimulus)
            rt: 反應時間陣列
            params: 參數字典
            
        Returns:
            log_likelihood: 對數似然值
        """
        
        # 解包增強參數
        v_vertical = params[f'{self.side_name}_v_vertical']
        v_nonvertical = params[f'{self.side_name}_v_nonvertical']
        v_vertical_error = params[f'{self.side_name}_v_vertical_error']
        v_nonvertical_error = params[f'{self.side_name}_v_nonvertical_error']
        
        threshold = params[f'{self.side_name}_threshold']
        start_var = params[f'{self.side_name}_start_var']
        ndt = params[f'{self.side_name}_ndt']
        noise = params[f'{self.side_name}_noise']
        
        # 只應用數值穩定性約束，不強迫任何drift rate關係
        v_vertical = pt.maximum(v_vertical, 0.05)
        v_nonvertical = pt.maximum(v_nonvertical, 0.05)
        v_vertical_error = pt.maximum(v_vertical_error, 0.05)
        v_nonvertical_error = pt.maximum(v_nonvertical_error, 0.05)
        
        threshold = pt.maximum(threshold, 0.1)
        start_var = pt.maximum(start_var, 0.05)
        ndt = pt.maximum(ndt, 0.05)
        noise = pt.maximum(noise, 0.1)
        
        # 計算決策時間（總RT減去非決策時間）
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        # 增強的stimulus-response邏輯
        # 根據實際的刺激-反應組合確定drift rates
        is_vertical_stimulus = pt.eq(stimuli, 0)
        is_vertical_response = pt.eq(decisions, 0)
        
        # 為vertical stimulus trials確定drift rates
        v_winner_vertical_stim = pt.where(
            is_vertical_response,  # If responded "vertical"
            v_vertical,            # Use v_vertical (correct for vertical stimulus)
            v_nonvertical_error    # Use v_nonvertical_error (incorrect for vertical stimulus)
        )
        v_loser_vertical_stim = pt.where(
            is_vertical_response,  # If responded "vertical" 
            v_nonvertical_error,   # Loser gets v_nonvertical_error
            v_vertical             # Loser gets v_vertical
        )
        
        # 為nonvertical stimulus trials確定drift rates
        v_winner_nonvertical_stim = pt.where(
            is_vertical_response,  # If responded "vertical"
            v_vertical_error,      # Use v_vertical_error (incorrect for nonvertical stimulus)
            v_nonvertical          # Use v_nonvertical (correct for nonvertical stimulus)
        )
        v_loser_nonvertical_stim = pt.where(
            is_vertical_response,  # If responded "vertical"
            v_nonvertical,         # Loser gets v_nonvertical
            v_vertical_error       # Loser gets v_vertical_error
        )
        
        # 根據stimulus type選擇適當的drift rates
        v_winner = pt.where(
            is_vertical_stimulus,
            v_winner_vertical_stim,
            v_winner_nonvertical_stim
        )
        v_loser = pt.where(
            is_vertical_stimulus,
            v_loser_vertical_stim, 
            v_loser_nonvertical_stim
        )
        
        # 計算2選擇LBA密度
        log_likelihood = self._compute_2choice_lba_density(
            decision_time, v_winner, v_loser, threshold, start_var, noise
        )
        
        return log_likelihood
    
    def _compute_2choice_lba_density(self, decision_time, v_winner, v_loser, 
                                   threshold, start_var, noise):
        """
        計算2選擇LBA密度函數的核心實現
        
        使用標準的LBA公式計算winner的PDF和loser的survival function
        """
        
        sqrt_t = pt.sqrt(decision_time)
        
        # Winner累積器的z-scores
        z1_winner = pt.clip(
            (v_winner * decision_time - threshold) / (noise * sqrt_t), 
            -4.5, 4.5
        )
        z2_winner = pt.clip(
            (v_winner * decision_time - start_var) / (noise * sqrt_t), 
            -4.5, 4.5
        )
        
        # Loser累積器的z-score
        z1_loser = pt.clip(
            (v_loser * decision_time - threshold) / (noise * sqrt_t), 
            -4.5, 4.5
        )
        
        # PyTensor兼容的Normal CDF和PDF函數
        from pytensor.tensor import erf
        
        def safe_normal_cdf(x):
            """安全的正態CDF函數"""
            x_safe = pt.clip(x, -4.5, 4.5)
            return 0.5 * (1 + erf(x_safe / pt.sqrt(2)))
        
        def safe_normal_pdf(x):
            """安全的正態PDF函數"""
            x_safe = pt.clip(x, -4.5, 4.5)
            return pt.exp(-0.5 * x_safe**2) / pt.sqrt(2 * pt.pi)
        
        # Winner的似然計算
        winner_cdf_term = safe_normal_cdf(z1_winner) - safe_normal_cdf(z2_winner)
        winner_pdf_term = (safe_normal_pdf(z1_winner) - safe_normal_pdf(z2_winner)) / (noise * sqrt_t)
        
        # 確保CDF項為正
        winner_cdf_term = pt.maximum(winner_cdf_term, 1e-10)
        
        # 完整的winner似然
        winner_likelihood = pt.maximum(
            (v_winner / start_var) * winner_cdf_term + winner_pdf_term / start_var,
            1e-10
        )
        
        # Loser的存活機率
        loser_survival = pt.maximum(1 - safe_normal_cdf(z1_loser), 1e-10)
        
        # 聯合似然：winner的PDF × loser的survival
        joint_likelihood = winner_likelihood * loser_survival
        joint_likelihood = pt.maximum(joint_likelihood, 1e-12)
        
        # 轉為對數似然
        log_likelihood = pt.log(joint_likelihood)
        
        # 處理無效值
        is_invalid = (
            pt.isnan(log_likelihood) | 
            pt.eq(log_likelihood, -np.inf) | 
            pt.eq(log_likelihood, np.inf)
        )
        log_likelihood_safe = pt.where(is_invalid, -100.0, log_likelihood)
        
        # 裁剪極端值並求和
        return pt.sum(pt.clip(log_likelihood_safe, -100.0, 10.0))
    
    def compute_choice_probabilities(self, stimuli, params, rt_mean=None):
        """
        計算增強版選擇機率（用於模型預測和驗證）
        
        Args:
            stimuli: 刺激陣列 (0=vertical, 1=nonvertical)
            params: 參數字典
            rt_mean: 平均反應時間（用於積分）
            
        Returns:
            choice_probs: 選擇機率陣列 [P(vertical), P(nonvertical)]
        """
        
        if rt_mean is None:
            rt_mean = 0.8  # 預設值
        
        # 解包增強參數
        v_vertical = float(params[f'{self.side_name}_v_vertical'])
        v_nonvertical = float(params[f'{self.side_name}_v_nonvertical'])
        v_vertical_error = float(params[f'{self.side_name}_v_vertical_error'])
        v_nonvertical_error = float(params[f'{self.side_name}_v_nonvertical_error'])
        
        threshold = float(params[f'{self.side_name}_threshold'])
        start_var = float(params[f'{self.side_name}_start_var'])
        ndt = float(params[f'{self.side_name}_ndt'])
        noise = float(params[f'{self.side_name}_noise'])
        
        # 只應用數值穩定性約束
        v_vertical = max(v_vertical, 0.05)
        v_nonvertical = max(v_nonvertical, 0.05)
        v_vertical_error = max(v_vertical_error, 0.05)
        v_nonvertical_error = max(v_nonvertical_error, 0.05)
        
        choice_probs = []
        
        for stimulus in stimuli:
            if stimulus == 0:  # Vertical stimulus
                # Correct response: vertical (0), Incorrect response: nonvertical (1)
                prob_vertical = self._compute_single_choice_prob(
                    v_vertical, v_nonvertical_error, threshold, start_var, ndt, noise, rt_mean
                )
                choice_probs.append([prob_vertical, 1 - prob_vertical])
            else:  # Nonvertical stimulus
                # Correct response: nonvertical (1), Incorrect response: vertical (0)
                prob_nonvertical = self._compute_single_choice_prob(
                    v_nonvertical, v_vertical_error, threshold, start_var, ndt, noise, rt_mean
                )
                choice_probs.append([1 - prob_nonvertical, prob_nonvertical])
        
        return np.array(choice_probs)
    
    def _compute_single_choice_prob(self, drift_chosen, drift_other, 
                                  threshold, start_var, ndt, noise, rt_mean):
        """
        計算單一選擇的機率（基於實際drift rates，不假設哪個是"正確"）
        """
        
        # 使用簡化的LBA公式
        decision_time = rt_mean - ndt
        if decision_time <= 0:
            decision_time = 0.1
        
        # 計算選擇的相對證據強度（不假設正確性）
        total_evidence = drift_chosen + drift_other
        evidence_ratio = drift_chosen / total_evidence if total_evidence > 0 else 0.5
        
        # 添加噪音因子以反映決策不確定性
        noise_factor = noise / (noise + threshold * 0.1)
        
        # 結合證據比和噪音因子
        prob = evidence_ratio * (1 - noise_factor * 0.3)
        
        return np.clip(prob, 0.05, 0.95)
    
    def compute_evidence_output(self, params, stimuli, decisions, rt, confidence_threshold=0.7):
        """
        計算增強版證據輸出，用於傳遞給dual LBA整合階段
        
        Args:
            params: 參數字典
            stimuli: 刺激陣列 (0=vertical, 1=nonvertical)
            decisions: 決策陣列 (0=vertical response, 1=nonvertical response)
            rt: 反應時間陣列
            confidence_threshold: 信心閾值
            
        Returns:
            evidence_dict: 包含vertical和nonvertical證據的字典
        """
        
        # 解包增強參數
        v_vertical = float(params[f'{self.side_name}_v_vertical'])
        v_nonvertical = float(params[f'{self.side_name}_v_nonvertical'])
        v_vertical_error = float(params[f'{self.side_name}_v_vertical_error'])
        v_nonvertical_error = float(params[f'{self.side_name}_v_nonvertical_error'])
        threshold = float(params[f'{self.side_name}_threshold'])
        ndt = float(params[f'{self.side_name}_ndt'])
        
        # 計算平均證據強度
        mean_rt = np.mean(rt)
        decision_time = max(mean_rt - ndt, 0.1)
        
        # 分別計算vertical stimulus和nonvertical stimulus的處理情況
        vertical_trials = stimuli == 0
        nonvertical_trials = stimuli == 1
        
        # Vertical evidence: 基於所有trial中vertical response的情況
        if np.any(vertical_trials):
            # 當stimulus是vertical時，vertical response的accuracy
            vertical_to_vertical_accuracy = np.mean(decisions[vertical_trials] == 0)
            # 使用實際的drift rates計算evidence
            evidence_vertical_from_vertical_stim = (
                v_vertical * vertical_to_vertical_accuracy + 
                v_nonvertical_error * (1 - vertical_to_vertical_accuracy)
            )
        else:
            evidence_vertical_from_vertical_stim = (v_vertical + v_nonvertical_error) / 2
            
        if np.any(nonvertical_trials):
            # 當stimulus是nonvertical時，vertical response的accuracy (這是error)
            vertical_to_nonvertical_accuracy = np.mean(decisions[nonvertical_trials] == 0)
            # 使用實際的drift rates計算evidence
            evidence_vertical_from_nonvertical_stim = (
                v_vertical_error * vertical_to_nonvertical_accuracy +
                v_nonvertical * (1 - vertical_to_nonvertical_accuracy)
            )
        else:
            evidence_vertical_from_nonvertical_stim = (v_vertical_error + v_nonvertical) / 2
        
        # 計算綜合vertical evidence
        n_vertical_trials = np.sum(vertical_trials)
        n_nonvertical_trials = np.sum(nonvertical_trials)
        total_trials = n_vertical_trials + n_nonvertical_trials
        
        if total_trials > 0:
            evidence_vertical = (
                evidence_vertical_from_vertical_stim * n_vertical_trials +
                evidence_vertical_from_nonvertical_stim * n_nonvertical_trials
            ) / total_trials
        else:
            evidence_vertical = (v_vertical + v_vertical_error) / 2
        
        # Nonvertical evidence: 類似邏輯
        if np.any(nonvertical_trials):
            nonvertical_to_nonvertical_accuracy = np.mean(decisions[nonvertical_trials] == 1)
            evidence_nonvertical_from_nonvertical_stim = (
                v_nonvertical * nonvertical_to_nonvertical_accuracy +
                v_vertical_error * (1 - nonvertical_to_nonvertical_accuracy)
            )
        else:
            evidence_nonvertical_from_nonvertical_stim = (v_nonvertical + v_vertical_error) / 2
            
        if np.any(vertical_trials):
            nonvertical_to_vertical_accuracy = np.mean(decisions[vertical_trials] == 1)
            evidence_nonvertical_from_vertical_stim = (
                v_nonvertical_error * nonvertical_to_vertical_accuracy +
                v_vertical * (1 - nonvertical_to_vertical_accuracy)
            )
        else:
            evidence_nonvertical_from_vertical_stim = (v_nonvertical_error + v_vertical) / 2
        
        if total_trials > 0:
            evidence_nonvertical = (
                evidence_nonvertical_from_nonvertical_stim * n_nonvertical_trials +
                evidence_nonvertical_from_vertical_stim * n_vertical_trials
            ) / total_trials
        else:
            evidence_nonvertical = (v_nonvertical + v_nonvertical_error) / 2
        
        # 計算整體準確性和信心度
        overall_accuracy = np.mean(decisions == stimuli)
        confidence = max(overall_accuracy, 1 - overall_accuracy)
        
        # 正規化證據（防止過大值）
        max_evidence = threshold * 1.5
        evidence_vertical = min(evidence_vertical, max_evidence)
        evidence_nonvertical = min(evidence_nonvertical, max_evidence)
        
        return {
            'evidence_vertical': evidence_vertical,
            'evidence_nonvertical': evidence_nonvertical,
            'processing_time': decision_time,
            'confidence': confidence,
            'overall_accuracy': overall_accuracy,
            'n_trials': len(stimuli),
            'n_vertical_trials': n_vertical_trials,
            'n_nonvertical_trials': n_nonvertical_trials,
            'side_name': self.side_name,
            'drift_rates': {
                'v_vertical': v_vertical,
                'v_nonvertical': v_nonvertical,
                'v_vertical_error': v_vertical_error,
                'v_nonvertical_error': v_nonvertical_error
            }
        }
    
    def validate_parameters(self, params):
        """
        驗證增強版參數的合理性（移除了強制correct > incorrect的約束）
        
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
            v_vertical = float(params[f'{self.side_name}_v_vertical'])
            v_nonvertical = float(params[f'{self.side_name}_v_nonvertical'])
            v_vertical_error = float(params[f'{self.side_name}_v_vertical_error'])
            v_nonvertical_error = float(params[f'{self.side_name}_v_nonvertical_error'])
            
            threshold = float(params[f'{self.side_name}_threshold'])
            start_var = float(params[f'{self.side_name}_start_var'])
            ndt = float(params[f'{self.side_name}_ndt'])
            noise = float(params[f'{self.side_name}_noise'])
            
            # 只檢查基本的數值有效性，不強制任何drift rate關係
            if v_vertical <= 0:
                return False, f"v_vertical必須 > 0，得到: {v_vertical}"
            
            if v_nonvertical <= 0:
                return False, f"v_nonvertical必須 > 0，得到: {v_nonvertical}"
                
            if v_vertical_error <= 0:
                return False, f"v_vertical_error必須 > 0，得到: {v_vertical_error}"
                
            if v_nonvertical_error <= 0:
                return False, f"v_nonvertical_error必須 > 0，得到: {v_nonvertical_error}"
            
            if threshold <= 0:
                return False, f"threshold必須 > 0，得到: {threshold}"
            
            if start_var <= 0 or start_var >= threshold:
                return False, f"start_var必須在 (0, threshold) 範圍內，得到: {start_var} vs threshold {threshold}"
            
            if ndt < 0 or ndt > 1.0:
                return False, f"ndt必須在 [0, 1] 範圍內，得到: {ndt}"
            
            if noise <= 0:
                return False, f"noise必須 > 0，得到: {noise}"
            
            return True, "增強版參數驗證通過 - 允許任何drift rate關係"
            
        except (ValueError, KeyError) as e:
            return False, f"參數驗證錯誤: {e}"
    
    def get_default_priors(self):
        """
        獲得增強版參數的預設先驗分布設定
        
        Returns:
            dict: 增強版先驗分布設定
        """
        
        return {
            # Drift rates for specific stimulus-response combinations
            f'{self.side_name}_v_vertical': {
                'distribution': 'Gamma',
                'alpha': 2.5,
                'beta': 1.5,
                'description': 'Vertical response drift when stimulus is vertical'
            },
            f'{self.side_name}_v_nonvertical': {
                'distribution': 'Gamma',
                'alpha': 2.5,
                'beta': 1.5,
                'description': 'Nonvertical response drift when stimulus is nonvertical'
            },
            f'{self.side_name}_v_vertical_error': {
                'distribution': 'Gamma',
                'alpha': 2.0,
                'beta': 3.0,
                'description': 'Vertical response drift when stimulus is nonvertical'
            },
            f'{self.side_name}_v_nonvertical_error': {
                'distribution': 'Gamma',
                'alpha': 2.0,
                'beta': 3.0,
                'description': 'Nonvertical response drift when stimulus is vertical'
            },
            
            # Common LBA parameters
            f'{self.side_name}_threshold': {
                'distribution': 'Gamma',
                'alpha': 3.0,
                'beta': 3.5,
                'description': '決策閾值'
            },
            f'{self.side_name}_start_var': {
                'distribution': 'Uniform',
                'lower': 0.1,
                'upper': 0.7,
                'description': '起始點變異'
            },
            f'{self.side_name}_ndt': {
                'distribution': 'Uniform',
                'lower': 0.05,
                'upper': 0.6,
                'description': '非決策時間'
            },
            f'{self.side_name}_noise': {
                'distribution': 'Gamma',
                'alpha': 2.5,
                'beta': 8.0,
                'description': '擴散噪音'
            }
        }

# 便利函數
def create_left_lba():
    """創建左通道LBA處理器"""
    return SingleSideLBA('left')

def create_right_lba():
    """創建右通道LBA處理器"""
    return SingleSideLBA('right')

def test_single_side_lba():
    """測試單邊LBA功能"""
    
    print("🧪 測試單邊LBA處理器...")
    
    try:
        # 創建測試資料
        n_trials = 100
        np.random.seed(42)
        
        stimuli = np.random.choice([0, 1], size=n_trials)
        decisions = np.random.choice([0, 1], size=n_trials)
        rt = np.random.uniform(0.3, 1.5, size=n_trials)
        
        # 創建LBA處理器
        left_lba = SingleSideLBA('left')
        
        # 測試增強版參數驗證
        test_params = {
            'left_v_vertical': 1.5,
            'left_v_nonvertical': 1.3,
            'left_v_vertical_error': 0.8,
            'left_v_nonvertical_error': 0.6,
            'left_threshold': 1.0,
            'left_start_var': 0.3,
            'left_ndt': 0.2,
            'left_noise': 0.3
        }
        
        valid, message = left_lba.validate_parameters(test_params)
        print(f"   參數驗證: {message}")
        
        if not valid:
            print("❌ 參數驗證失敗")
            return False
        
        # 測試選擇機率計算
        choice_probs = left_lba.compute_choice_probabilities(stimuli[:10], test_params)
        print(f"   選擇機率計算: {choice_probs.shape}")
        
        # 測試證據輸出
        evidence = left_lba.compute_evidence_output(test_params, stimuli, decisions, rt)
        print(f"   證據輸出: vertical={evidence['evidence_vertical']:.3f}, diagonal={evidence['evidence_diagonal']:.3f}")
        
        # 測試先驗設定
        priors = left_lba.get_default_priors()
        print(f"   先驗分布數量: {len(priors)}")
        
        print("✅ 單邊LBA測試成功!")
        return True
        
    except Exception as e:
        print(f"❌ 單邊LBA測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 如果直接執行此檔案，進行測試
    test_single_side_lba()
