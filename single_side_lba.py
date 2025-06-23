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
        
        # 參數名稱列表
        self.param_names = [
            f'{side_name}_drift_correct',      # 正確漂移率
            f'{side_name}_drift_incorrect',    # 錯誤漂移率
            f'{side_name}_threshold',          # 決策閾值
            f'{side_name}_start_var',          # 起始點變異
            f'{side_name}_ndt',                # 非決策時間
            f'{side_name}_noise'               # 擴散噪音
        ]
        
        print(f"✅ 初始化 {side_name} 通道LBA處理器")
        print(f"   參數數量: {len(self.param_names)}")
    
    def compute_likelihood(self, decisions, stimuli, rt, params):
        """
        計算單邊2選擇LBA似然函數
        
        Args:
            decisions: 決策陣列 (0=垂直, 1=對角)
            stimuli: 刺激陣列 (0=垂直, 1=對角)
            rt: 反應時間陣列
            params: 參數字典
            
        Returns:
            log_likelihood: 對數似然值
        """
        
        # 解包參數
        drift_correct = params[f'{self.side_name}_drift_correct']
        drift_incorrect = params[f'{self.side_name}_drift_incorrect']
        threshold = params[f'{self.side_name}_threshold']
        start_var = params[f'{self.side_name}_start_var']
        ndt = params[f'{self.side_name}_ndt']
        noise = params[f'{self.side_name}_noise']
        
        # 應用參數邊界約束
        drift_correct = pt.maximum(drift_correct, 0.1)
        drift_incorrect = pt.maximum(drift_incorrect, 0.05)
        threshold = pt.maximum(threshold, 0.1)
        start_var = pt.maximum(start_var, 0.05)
        ndt = pt.maximum(ndt, 0.05)
        noise = pt.maximum(noise, 0.1)
        
        # 確保正確漂移率高於錯誤漂移率
        drift_correct = pt.maximum(drift_correct, drift_incorrect + 0.05)
        
        # 計算決策時間（總RT減去非決策時間）
        decision_time = pt.maximum(rt - ndt, 0.01)
        
        # 判斷正確vs錯誤反應
        stimulus_correct = pt.eq(decisions, stimuli)
        
        # 設定winner和loser的漂移率
        v_winner = pt.where(stimulus_correct, drift_correct, drift_incorrect)
        v_loser = pt.where(stimulus_correct, drift_incorrect, drift_correct)
        
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
        計算選擇機率（用於模型預測和驗證）
        
        Args:
            stimuli: 刺激陣列
            params: 參數字典
            rt_mean: 平均反應時間（用於積分）
            
        Returns:
            choice_probs: 選擇機率陣列
        """
        
        if rt_mean is None:
            rt_mean = 0.8  # 預設值
        
        # 解包參數
        drift_correct = float(params[f'{self.side_name}_drift_correct'])
        drift_incorrect = float(params[f'{self.side_name}_drift_incorrect'])
        threshold = float(params[f'{self.side_name}_threshold'])
        start_var = float(params[f'{self.side_name}_start_var'])
        ndt = float(params[f'{self.side_name}_ndt'])
        noise = float(params[f'{self.side_name}_noise'])
        
        # 應用邊界約束
        drift_correct = max(drift_correct, 0.1)
        drift_incorrect = max(drift_incorrect, 0.05)
        drift_correct = max(drift_correct, drift_incorrect + 0.05)
        
        choice_probs = []
        
        for stimulus in stimuli:
            if stimulus == 0:  # 垂直線刺激
                # 正確選擇是垂直(0)，錯誤選擇是對角(1)
                prob_correct = self._compute_single_choice_prob(
                    drift_correct, drift_incorrect, threshold, start_var, ndt, noise, rt_mean
                )
                choice_probs.append([prob_correct, 1 - prob_correct])
            else:  # 對角線刺激
                # 正確選擇是對角(1)，錯誤選擇是垂直(0)
                prob_correct = self._compute_single_choice_prob(
                    drift_correct, drift_incorrect, threshold, start_var, ndt, noise, rt_mean
                )
                choice_probs.append([1 - prob_correct, prob_correct])
        
        return np.array(choice_probs)
    
    def _compute_single_choice_prob(self, drift_correct, drift_incorrect, 
                                  threshold, start_var, ndt, noise, rt_mean):
        """
        計算單一選擇的機率（簡化版本，用於快速估計）
        """
        
        # 使用簡化的LBA公式
        decision_time = rt_mean - ndt
        if decision_time <= 0:
            decision_time = 0.1
        
        # 計算正確選擇的優勢
        evidence_ratio = drift_correct / (drift_correct + drift_incorrect)
        
        # 添加一些隨機性以避免過度確定
        noise_factor = noise / (noise + threshold * 0.1)
        
        # 結合證據比和噪音因子
        prob = evidence_ratio * (1 - noise_factor * 0.5)
        
        return np.clip(prob, 0.1, 0.9)
    
    def compute_evidence_output(self, params, stimuli, decisions, rt, confidence_threshold=0.7):
        """
        計算證據輸出，用於傳遞給四選一整合階段
        
        Args:
            params: 參數字典
            stimuli: 刺激陣列
            decisions: 決策陣列
            rt: 反應時間陣列
            confidence_threshold: 信心閾值
            
        Returns:
            evidence_dict: 包含垂直和對角證據的字典
        """
        
        # 解包參數
        drift_correct = float(params[f'{self.side_name}_drift_correct'])
        drift_incorrect = float(params[f'{self.side_name}_drift_incorrect'])
        threshold = float(params[f'{self.side_name}_threshold'])
        
        # 計算平均證據強度
        mean_rt = np.mean(rt)
        decision_time = max(mean_rt - float(params[f'{self.side_name}_ndt']), 0.1)
        
        # 垂直線證據（stimulus=0時的證據）
        vertical_trials = stimuli == 0
        if np.any(vertical_trials):
            vertical_accuracy = np.mean(decisions[vertical_trials] == stimuli[vertical_trials])
            evidence_vertical = drift_correct * vertical_accuracy + drift_incorrect * (1 - vertical_accuracy)
        else:
            evidence_vertical = (drift_correct + drift_incorrect) / 2
        
        # 對角線證據（stimulus=1時的證據）
        diagonal_trials = stimuli == 1
        if np.any(diagonal_trials):
            diagonal_accuracy = np.mean(decisions[diagonal_trials] == stimuli[diagonal_trials])
            evidence_diagonal = drift_correct * diagonal_accuracy + drift_incorrect * (1 - diagonal_accuracy)
        else:
            evidence_diagonal = (drift_correct + drift_incorrect) / 2
        
        # 計算整體信心度
        overall_accuracy = np.mean(decisions == stimuli)
        confidence = max(overall_accuracy, 1 - overall_accuracy)  # 取較高者
        
        # 正規化證據（防止過大值）
        max_evidence = threshold * 2  # 設定最大證據值
        evidence_vertical = min(evidence_vertical, max_evidence)
        evidence_diagonal = min(evidence_diagonal, max_evidence)
        
        return {
            'evidence_vertical': evidence_vertical,
            'evidence_diagonal': evidence_diagonal,
            'processing_time': decision_time,
            'confidence': confidence,
            'choice_probability': overall_accuracy,
            'n_trials': len(stimuli),
            'side_name': self.side_name
        }
    
    def validate_parameters(self, params):
        """
        驗證參數的合理性
        
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
            drift_correct = float(params[f'{self.side_name}_drift_correct'])
            drift_incorrect = float(params[f'{self.side_name}_drift_incorrect'])
            threshold = float(params[f'{self.side_name}_threshold'])
            start_var = float(params[f'{self.side_name}_start_var'])
            ndt = float(params[f'{self.side_name}_ndt'])
            noise = float(params[f'{self.side_name}_noise'])
            
            # 檢查參數邊界
            if drift_correct <= 0:
                return False, f"drift_correct必須 > 0，得到: {drift_correct}"
            
            if drift_incorrect <= 0:
                return False, f"drift_incorrect必須 > 0，得到: {drift_incorrect}"
            
            if drift_correct <= drift_incorrect:
                return False, f"drift_correct必須 > drift_incorrect，得到: {drift_correct} vs {drift_incorrect}"
            
            if threshold <= 0:
                return False, f"threshold必須 > 0，得到: {threshold}"
            
            if start_var <= 0 or start_var >= threshold:
                return False, f"start_var必須在 (0, threshold) 範圍內，得到: {start_var} vs threshold {threshold}"
            
            if ndt < 0 or ndt > 1.0:
                return False, f"ndt必須在 [0, 1] 範圍內，得到: {ndt}"
            
            if noise <= 0:
                return False, f"noise必須 > 0，得到: {noise}"
            
            return True, "參數驗證通過"
            
        except (ValueError, KeyError) as e:
            return False, f"參數驗證錯誤: {e}"
    
    def get_default_priors(self):
        """
        獲得參數的預設先驗分布設定
        
        Returns:
            dict: 先驗分布設定
        """
        
        return {
            f'{self.side_name}_drift_correct': {
                'distribution': 'Gamma',
                'alpha': 2.5,
                'beta': 1.2,
                'description': '正確漂移率 - 較高值'
            },
            f'{self.side_name}_drift_incorrect': {
                'distribution': 'Gamma',
                'alpha': 2.0,
                'beta': 3.0,
                'description': '錯誤漂移率 - 較低值'
            },
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
        
        # 測試參數驗證
        test_params = {
            'left_drift_correct': 1.5,
            'left_drift_incorrect': 0.8,
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
