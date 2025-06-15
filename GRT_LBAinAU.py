# -*- coding: utf-8 -*-
"""
線條傾斜判斷任務 - 雙通道LBA模型 (Line Tilt Judgment Task - Dual-Channel LBA Model)
Line Tilt Judgment Task - Dual-Channel LBA Model

目的 / Purpose:
模擬受試者判斷左右兩條線條傾斜方向的認知過程
Simulate the cognitive process of judging the tilt direction of left and right lines

實驗設計 / Experimental Design:
- 刺激：四個角落出現不同線條組合 (\|, \/, |\, //)
- 任務：判斷剛才看到的是哪種組合
- 認知模型：兩個平行的LBA通道分別處理左右線條傾斜
- Stimuli: Different line combinations appear in four corners (\|, \/, |\, //)
- Task: Judge which combination was just seen
- Cognitive model: Two parallel LBA channels process left and right line tilts
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
import time
from typing import Dict, Optional

# ============================================================================
# 第一部分：數據預處理和刺激編碼
# Part 1: Data Preprocessing and Stimulus Encoding
# ============================================================================

def prepare_line_tilt_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    準備線條傾斜判斷任務的數據
    Prepare data for line tilt judgment task
    
    目的 / Purpose:
    將原始刺激編碼轉換為左右線條傾斜特徵
    Convert raw stimulus codes to left and right line tilt features
    
    參數來源 / Parameter Sources:
    - df: 原始CSV數據，包含Stimulus欄位(1,2,3,4)
    - df: Raw CSV data containing Stimulus column (1,2,3,4)
    
    刺激映射 / Stimulus Mapping:
    1 → 左上角 → 左線條:\, 右線條:| → (0, 1)
    2 → 左下角 → 左線條:\, 右線條:/ → (0, 0) 
    3 → 右上角 → 左線條:|, 右線條:| → (1, 1)
    4 → 右下角 → 左線條:|, 右線條:/ → (1, 0)
    """
    
    print("開始數據預處理 / Starting data preprocessing...")
    
    # 刺激編碼映射表 / Stimulus encoding mapping
    # 目的：將1-4的刺激編號轉換為左右線條傾斜特徵
    # Purpose: Convert stimulus numbers 1-4 to left/right line tilt features
    stimulus_mapping = {
        1: {'left_tilt': 0, 'right_tilt': 1, 'description': '左\\右|'},  # 左斜右直
        2: {'left_tilt': 0, 'right_tilt': 0, 'description': '左\\右/'},  # 左斜右斜  
        3: {'left_tilt': 1, 'right_tilt': 1, 'description': '左|右|'},  # 左直右直
        4: {'left_tilt': 1, 'right_tilt': 0, 'description': '左|右/'}   # 左直右斜
    }
    
    # 創建新的特徵欄位 / Create new feature columns
    # 目的：為LBA模型準備獨立的左右通道輸入
    # Purpose: Prepare independent left/right channel inputs for LBA model
    
    # 左線條傾斜特徵 / Left line tilt feature
    # 來源：從Stimulus欄位映射而來
    # Source: Mapped from Stimulus column
    # 0 = 斜線(\), 1 = 直線(|)
    df['left_line_tilt'] = df['Stimulus'].map(
        lambda x: stimulus_mapping.get(x, {'left_tilt': 0})['left_tilt']
    )
    
    # 右線條傾斜特徵 / Right line tilt feature  
    # 來源：從Stimulus欄位映射而來
    # Source: Mapped from Stimulus column
    # 0 = 斜線(/ or \), 1 = 直線(|)
    df['right_line_tilt'] = df['Stimulus'].map(
        lambda x: stimulus_mapping.get(x, {'right_tilt': 0})['right_tilt']
    )
    
    # 四選項組合編碼 / Four-choice combination encoding
    # 目的：將Response欄位轉換為0-3編碼供LBA使用
    # Purpose: Convert Response column to 0-3 encoding for LBA use
    # 來源：CSV文件的Response欄位
    # Source: Response column from CSV file
    df['choice_response'] = df['Response'].astype(int)
    
    # 數據清理 / Data cleaning
    # 目的：移除無效的反應時間和選擇
    # Purpose: Remove invalid reaction times and choices
    
    # 反應時間過濾 / Reaction time filtering
    # 來源：CSV文件的RT欄位
    # Source: RT column from CSV file
    # 範圍：0.1-10秒，移除過快或過慢的反應
    # Range: 0.1-10 seconds, remove too fast or too slow responses
    valid_rt = (df['RT'] >= 0.1) & (df['RT'] <= 10.0)
    
    # 選擇有效性過濾 / Choice validity filtering
    # 目的：確保選擇在有效範圍內
    # Purpose: Ensure choices are within valid range
    valid_choice = df['choice_response'].isin([0, 1, 2, 3])
    
    # 應用過濾條件 / Apply filtering conditions
    df_clean = df[valid_rt & valid_choice].copy()
    
    # 移除缺失值 / Remove missing values
    # 目的：確保所有必要欄位都有值
    # Purpose: Ensure all necessary columns have values
    df_clean = df_clean.dropna(subset=['left_line_tilt', 'right_line_tilt', 'choice_response', 'RT'])
    
    print(f"數據預處理完成 / Data preprocessing completed:")
    print(f"  原始數據量 / Original data: {len(df)} trials")
    print(f"  清理後數據量 / Cleaned data: {len(df_clean)} trials")
    print(f"  刺激分佈 / Stimulus distribution:")
    
    # 顯示刺激分佈 / Show stimulus distribution
    for stim, info in stimulus_mapping.items():
        count = len(df_clean[df_clean['Stimulus'] == stim])
        print(f"    刺激{stim} ({info['description']}): {count} trials")
    
    return df_clean

# ============================================================================
# 第二部分：雙通道LBA似然函數
# Part 2: Dual-Channel LBA Likelihood Function
# ============================================================================

def compute_dual_lba_likelihood(left_tilt, right_tilt, choice, rt, 
                               left_bias, right_bias, 
                               left_drift, right_drift,
                               noise_left, noise_right):
    """
    計算雙通道LBA模型的似然函數
    Compute likelihood for dual-channel LBA model
    
    目的 / Purpose:
    模擬兩個獨立的LBA通道分別處理左右線條傾斜判斷
    Simulate two independent LBA channels processing left/right line tilt judgments
    
    參數說明 / Parameter Descriptions:
    
    輸入特徵 / Input Features:
    - left_tilt: 左線條傾斜 (0=斜線, 1=直線) / Left line tilt (0=diagonal, 1=vertical)
    - right_tilt: 右線條傾斜 (0=斜線, 1=直線) / Right line tilt (0=diagonal, 1=vertical)  
    - choice: 受試者選擇 (0-3) / Subject choice (0-3)
    - rt: 反應時間 / Reaction time
    
    模型參數 / Model Parameters:
    - left_bias: 左通道判斷閾值 / Left channel judgment threshold
    - right_bias: 右通道判斷閾值 / Right channel judgment threshold
    - left_drift: 左通道漂移率 / Left channel drift rate
    - right_drift: 右通道漂移率 / Right channel drift rate
    - noise_left: 左通道雜訊 / Left channel noise
    - noise_right: 右通道雜訊 / Right channel noise
    """
    
    try:
        # LBA模型固定參數 / Fixed LBA parameters
        # 目的：設定LBA模型的基本結構參數
        # Purpose: Set basic structural parameters for LBA model
        A = 0.4      # 起始點變異 / Start point variability
        s = 0.3      # 漂移率標準差 / Drift rate standard deviation
        t0 = 0.2     # 非決策時間 / Non-decision time
        b = A + 0.6  # 決策閾值 / Decision threshold
        
        # 計算決策時間 / Calculate decision time
        # 目的：從總反應時間中減去非決策時間
        # Purpose: Subtract non-decision time from total reaction time
        # 來源：rt參數 (從CSV的RT欄位)
        # Source: rt parameter (from RT column in CSV)
        decision_time = np.maximum(rt - t0, 0.001)
        
        # === 左通道LBA計算 / Left Channel LBA Calculation ===
        
        # 左通道證據累積方向 / Left channel evidence accumulation direction
        # 目的：根據實際刺激和偏好計算證據強度
        # Purpose: Calculate evidence strength based on actual stimulus and bias
        
        # 如果左線條是直線(1)，向"直線"累積證據；如果是斜線(0)，向"斜線"累積證據
        # If left line is vertical(1), accumulate evidence for "vertical"; if diagonal(0), for "diagonal"
        left_evidence_strength = left_drift * (1 if left_tilt > left_bias else -1)
        
        # 左通道漂移率計算 / Left channel drift rate calculation
        # 目的：結合刺激強度和個體偏好
        # Purpose: Combine stimulus strength and individual bias
        v_left_correct = np.maximum(np.abs(left_evidence_strength) + noise_left, 0.1)
        v_left_incorrect = np.maximum(0.5 * left_drift + noise_left, 0.1)
        
        # === 右通道LBA計算 / Right Channel LBA Calculation ===
        
        # 右通道證據累積 / Right channel evidence accumulation
        # 目的：獨立處理右線條傾斜判斷
        # Purpose: Independently process right line tilt judgment
        right_evidence_strength = right_drift * (1 if right_tilt > right_bias else -1)
        
        v_right_correct = np.maximum(np.abs(right_evidence_strength) + noise_right, 0.1)
        v_right_incorrect = np.maximum(0.5 * right_drift + noise_right, 0.1)
        
        # === 四選項組合判斷 / Four-choice combination judgment ===
        
        # 目的：將兩個獨立的LBA通道結果組合成四選項判斷
        # Purpose: Combine two independent LBA channel results into four-choice judgment
        
        # 判斷左通道傾向 / Determine left channel preference
        # 0: 傾向斜線, 1: 傾向直線
        # 0: prefer diagonal, 1: prefer vertical
        left_decision = 1 if left_tilt > left_bias else 0
        
        # 判斷右通道傾向 / Determine right channel preference  
        right_decision = 1 if right_tilt > right_bias else 0
        
        # 組合決策映射 / Combined decision mapping
        # 目的：將左右通道決策組合為最終選擇
        # Purpose: Combine left/right channel decisions into final choice
        predicted_choice = left_decision * 2 + right_decision
        
        # === LBA似然計算 / LBA Likelihood Calculation ===
        
        # 選擇獲勝者和失敗者的漂移率 / Select winner and loser drift rates
        # 目的：根據實際選擇確定哪個累加器獲勝
        # Purpose: Determine which accumulator wins based on actual choice
        
        if choice == predicted_choice:
            # 正確選擇的情況 / Correct choice case
            v_winner = (v_left_correct + v_right_correct) / 2
            v_loser1 = (v_left_incorrect + v_right_correct) / 2
            v_loser2 = (v_left_correct + v_right_incorrect) / 2  
            v_loser3 = (v_left_incorrect + v_right_incorrect) / 2
        else:
            # 錯誤選擇的情況 / Incorrect choice case
            v_winner = (v_left_incorrect + v_right_incorrect) / 2
            v_loser1 = (v_left_correct + v_right_incorrect) / 2
            v_loser2 = (v_left_incorrect + v_right_correct) / 2
            v_loser3 = (v_left_correct + v_right_correct) / 2
        
        # LBA密度函數計算 / LBA density function calculation
        # 目的：計算給定參數下觀察到此反應時間和選擇的機率
        # Purpose: Calculate probability of observing this RT and choice given parameters
        
        sqrt_t = np.sqrt(decision_time)
        
        # 獲勝累加器的似然 / Winner accumulator likelihood
        z1_win = np.clip((v_winner * decision_time - b) / sqrt_t, -6, 6)
        z2_win = np.clip((v_winner * decision_time - A) / sqrt_t, -6, 6)
        
        winner_cdf = stats.norm.cdf(z1_win) - stats.norm.cdf(z2_win)
        winner_pdf = (stats.norm.pdf(z1_win) - stats.norm.pdf(z2_win)) / sqrt_t
        winner_likelihood = np.maximum((v_winner / A) * winner_cdf + winner_pdf / A, 1e-10)
        
        # 失敗累加器的生存函數 / Loser accumulators survival function
        survival_prob = 1.0
        
        for v_loser in [v_loser1, v_loser2, v_loser3]:
            z1_lose = np.clip((v_loser * decision_time - b) / sqrt_t, -6, 6)
            z2_lose = np.clip((v_loser * decision_time - A) / sqrt_t, -6, 6)
            
            loser_cdf = stats.norm.cdf(z1_lose) - stats.norm.cdf(z2_lose)
            survival_prob *= np.maximum(1 - loser_cdf, 1e-6)
        
        # 總似然 / Total likelihood
        # 目的：計算完整的LBA模型似然
        # Purpose: Calculate complete LBA model likelihood
        total_likelihood = winner_likelihood * survival_prob
        
        return np.log(np.maximum(total_likelihood, 1e-12))
        
    except Exception as e:
        print(f"LBA似然計算錯誤 / LBA likelihood calculation error: {e}")
        return -1000.0

# ============================================================================
# 第三部分：受試者分析函數
# Part 3: Subject Analysis Function
# ============================================================================

def analyze_line_tilt_subject(subject_id: int, subject_data: pd.DataFrame) -> Optional[Dict]:
    """
    分析單一受試者的線條傾斜判斷行為
    Analyze individual subject's line tilt judgment behavior
    
    目的 / Purpose:
    使用雙通道LBA模型估計受試者的認知參數
    Use dual-channel LBA model to estimate subject's cognitive parameters
    
    參數來源 / Parameter Sources:
    - subject_id: 受試者編號 (來自CSV的participant欄位)
    - subject_data: 該受試者的所有試驗數據 (從總數據中過濾)
    - subject_id: Subject ID (from participant column in CSV)
    - subject_data: All trial data for this subject (filtered from total data)
    """
    
    try:
        print(f"開始分析受試者 {subject_id} / Starting analysis for Subject {subject_id}...")
        
        # === 數據提取和驗證 / Data Extraction and Validation ===
        
        # 提取刺激特徵 / Extract stimulus features
        # 來源：數據預處理階段創建的特徵
        # Source: Features created during data preprocessing
        left_tilt_data = subject_data['left_line_tilt'].values    # 左線條傾斜
        right_tilt_data = subject_data['right_line_tilt'].values  # 右線條傾斜
        
        # 提取反應數據 / Extract response data  
        # 來源：CSV文件的原始欄位
        # Source: Original columns from CSV file
        choice_data = subject_data['choice_response'].values  # 選擇反應 (0-3)
        rt_data = subject_data['RT'].values                   # 反應時間
        
        # 數據量檢查 / Data quantity check
        # 目的：確保有足夠數據進行可靠的參數估計
        # Purpose: Ensure sufficient data for reliable parameter estimation
        n_trials = len(rt_data)
        if n_trials < 50:
            print(f"   數據量不足 / Insufficient data: {n_trials} trials (minimum: 50)")
            return None
        
        print(f"   數據提取完成 / Data extraction completed: {n_trials} trials")
        
        # === PyMC貝葉斯模型定義 / PyMC Bayesian Model Definition ===
        
        with pm.Model() as dual_lba_model:
            
            # === 先驗分佈定義 / Prior Distribution Definition ===
            # 目的：設定認知參數的先驗信念
            # Purpose: Set prior beliefs about cognitive parameters
            
            # 左通道判斷偏好 / Left channel judgment bias
            # 意義：受試者判斷左線條為"直線"的傾向
            # Meaning: Subject's tendency to judge left line as "vertical"
            # 範圍：0-1，0.5表示無偏好
            # Range: 0-1, 0.5 indicates no bias
            left_bias = pm.Beta('left_bias', alpha=2, beta=2)
            
            # 右通道判斷偏好 / Right channel judgment bias  
            # 意義：受試者判斷右線條為"直線"的傾向
            # Meaning: Subject's tendency to judge right line as "vertical"
            right_bias = pm.Beta('right_bias', alpha=2, beta=2)
            
            # 左通道處理強度 / Left channel processing strength
            # 意義：左通道的證據累積速度
            # Meaning: Evidence accumulation speed for left channel
            # 範圍：正值，數值越大處理越快
            # Range: positive values, higher means faster processing
            left_drift = pm.Gamma('left_drift', alpha=3, beta=1)
            
            # 右通道處理強度 / Right channel processing strength
            # 意義：右通道的證據累積速度  
            # Meaning: Evidence accumulation speed for right channel
            right_drift = pm.Gamma('right_drift', alpha=3, beta=1)
            
            # 左通道雜訊水平 / Left channel noise level
            # 意義：左通道處理的變異性
            # Meaning: Variability in left channel processing
            noise_left = pm.Gamma('noise_left', alpha=2, beta=4)
            
            # 右通道雜訊水平 / Right channel noise level
            # 意義：右通道處理的變異性
            # Meaning: Variability in right channel processing  
            noise_right = pm.Gamma('noise_right', alpha=2, beta=4)
            
            # === 似然函數定義 / Likelihood Function Definition ===
            
            # 目的：連接觀察數據與認知模型
            # Purpose: Connect observed data with cognitive model
            
            # 為每個試驗計算似然 / Calculate likelihood for each trial
            likelihood_values = []
            
            for i in range(n_trials):
                # 計算單一試驗的LBA似然 / Calculate LBA likelihood for single trial
                # 輸入：當前試驗的刺激和反應數據
                # Input: Current trial's stimulus and response data
                trial_likelihood = compute_dual_lba_likelihood(
                    left_tilt=left_tilt_data[i],     # 左線條傾斜特徵
                    right_tilt=right_tilt_data[i],   # 右線條傾斜特徵
                    choice=choice_data[i],           # 受試者選擇
                    rt=rt_data[i],                   # 反應時間
                    left_bias=left_bias,             # 左通道偏好參數
                    right_bias=right_bias,           # 右通道偏好參數
                    left_drift=left_drift,           # 左通道漂移率
                    right_drift=right_drift,         # 右通道漂移率
                    noise_left=noise_left,           # 左通道雜訊
                    noise_right=noise_right          # 右通道雜訊
                )
                likelihood_values.append(trial_likelihood)
            
            # 總似然 / Total likelihood
            # 目的：將所有試驗的似然組合
            # Purpose: Combine likelihood across all trials
            total_log_likelihood = pm.math.sum(pt.stack(likelihood_values))
            
            # 將似然納入模型 / Include likelihood in model
            pm.Potential('lba_likelihood', total_log_likelihood)
            
        print(f"   貝葉斯模型建構完成 / Bayesian model construction completed")
        
        # === MCMC採樣 / MCMC Sampling ===
        
        # 目的：從後驗分佈中採樣參數
        # Purpose: Sample parameters from posterior distribution
        
        with dual_lba_model:
            trace = pm.sample(
                draws=1000,           # 採樣數量 / Number of samples
                tune=1000,            # 調整期 / Tuning period  
                chains=4,             # 馬可夫鏈數量 / Number of Markov chains
                target_accept=0.9,    # 目標接受率 / Target acceptance rate
                cores=1,              # 計算核心數 / Number of cores
                random_seed=42,       # 隨機種子 / Random seed
                progressbar=True,     # 顯示進度條 / Show progress bar
                return_inferencedata=True  # 返回推論數據格式 / Return inference data format
            )
        
        print(f"   MCMC採樣完成 / MCMC sampling completed")
        
        # === 收斂性診斷 / Convergence Diagnostics ===
        
        # 目的：檢查採樣是否收斂到穩定分佈
        # Purpose: Check if sampling converged to stable distribution
        
        try:
            summary = az.summary(trace)
            rhat_max = summary['r_hat'].max() if 'r_hat' in summary else 1.0
            ess_min = summary['ess_bulk'].min() if 'ess_bulk' in summary else 100
            
            # 收斂性警告 / Convergence warnings
            convergence_ok = True
            if rhat_max > 1.05:
                print(f"   ⚠️ 收斂警告 / Convergence warning: R-hat = {rhat_max:.3f}")
                convergence_ok = False
            if ess_min < 100:
                print(f"   ⚠️ 採樣警告 / Sampling warning: ESS = {ess_min:.0f}")
                convergence_ok = False
                
        except Exception as e:
            print(f"   診斷計算警告 / Diagnostic calculation warning: {e}")
            rhat_max, ess_min = 1.05, 100
            convergence_ok = False
        
        # === 結果整理 / Result Organization ===
        
        result = {
            'subject_id': subject_id,           # 受試者編號
            'trace': trace,                     # MCMC採樣軌跡  
            'convergence': {                    # 收斂性統計
                'rhat_max': float(rhat_max),    # 最大R-hat值
                'ess_min': float(ess_min),      # 最小有效樣本數
                'converged': convergence_ok     # 是否收斂
            },
            'data_info': {                      # 數據資訊
                'n_trials': n_trials,          # 試驗總數
                'choice_distribution': {        # 選擇分佈
                    f'choice_{i}': int(np.sum(choice_data == i)) 
                    for i in range(4)
                },
                'mean_rt': float(np.mean(rt_data))  # 平均反應時間
            },
            'model_type': 'dual_channel_lba',   # 模型類型
            'success': True                     # 成功標記
        }
        
        status = "✅ 收斂良好" if convergence_ok else "⚠️ 收斂問題"
        print(f"{status} 受試者 {subject_id} 分析完成 / Subject {subject_id} analysis completed "
              f"(R̂={rhat_max:.3f}, ESS={ess_min:.0f})")
        
        return result
        
    except Exception as e:
        print(f"❌ 受試者 {subject_id} 分析失敗 / Subject {subject_id} analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'subject_id': subject_id, 
            'success': False, 
            'error': str(e),
            'model_type': 'dual_channel_lba'
        }

# ============================================================================
# 第四部分：主要分析器類別
# Part 4: Main Analyzer Class  
# ============================================================================

class LineTiltGRTAnalyzer:
    """
    線條傾斜判斷任務的GRT-LBA分析器
    GRT-LBA Analyzer for Line Tilt Judgment Task
    
    目的 / Purpose:
    整合數據載入、預處理、模型分析和結果儲存功能
    Integrate data loading, preprocessing, model analysis, and result saving
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        """
        初始化分析器
        Initialize analyzer
        
        參數 / Parameters:
        - csv_file: CSV數據文件路徑 / Path to CSV data file
        """
        
        print("="*60)
        print("線條傾斜判斷任務 - 雙通道LBA分析器")  
        print("Line Tilt Judgment Task - Dual-Channel LBA Analyzer")
        print("="*60)
        
        # 載入原始數據 / Load raw data
        # 來源：實驗產生的CSV文件
        # Source: CSV file generated from experiment
        try:
            self.raw_df = pd.read_csv(csv_file)
            print(f"✅ 數據載入成功 / Data loaded successfully: {len(self.raw_df)} trials")
        except FileNotFoundError:
            print(f"❌ 找不到數據文件 / Data file not found: {csv_file}")
            raise
        except Exception as e:
            print(f"❌ 數據載入失敗 / Data loading failed: {e}")
            raise
        
        # 數據預處理 / Data preprocessing
        # 目的：轉換刺激編碼為線條傾斜特徵
        # Purpose: Convert stimulus codes to line tilt features
        self.df = prepare_line_tilt_data(self.raw_df)
        
        # 受試者列表 / Subject list
        # 來源：participant欄位的唯一值
        # Source: Unique values from participant column
        self.participants = sorted(self.df['participant'].unique())
        print(f"✅ 發現 {len(self.participants)} 位受試者 / Found {len(self.participants)} subjects")
        
        # 顯示數據摘要 / Show data summary
        self._show_data_summary()
    
    def _show_data_summary(self):
        """
        顯示數據摘要統計
        Show data
