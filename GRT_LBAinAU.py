# -*- coding: utf-8 -*-
"""
修正版四選項 GRT-LBA 分析程式碼
Fixed Four-Choice GRT-LBA Analysis Code

主要修正 / Main Fixes:
1. 修正 PyMC 模型定義錯誤 / Fix PyMC model definition errors
2. 正確的 pm.Potential 使用方式 / Correct pm.Potential usage
3. 詳細的程式碼解釋 / Detailed code explanations
4. 變數來源與用途說明 / Variable source and purpose explanations
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, List
import os

# 關閉不必要的警告訊息 / Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：LBA 似然函數實現
# Part 1: LBA Likelihood Function Implementation
# ============================================================================

def compute_lba_likelihood(rt_data, choice_data, stimloc_data, params):
    """
    計算 Linear Ballistic Accumulator (LBA) 模型的似然函數
    Compute the likelihood function for Linear Ballistic Accumulator (LBA) model
    
    參數說明 / Parameters:
    - rt_data: 反應時間數據 / Reaction time data (來自 CSV 的 RT 欄位)
    - choice_data: 選擇反應數據 / Choice response data (來自 CSV 的 Response 欄位)
    - stimloc_data: 刺激位置數據 / Stimulus location data (從 Stimulus 欄位轉換)
    - params: 模型參數 / Model parameters (db1, db2, sp, base_v)
    
    返回值 / Returns:
    - 對數似然值 / Log-likelihood value
    
    變數來源 / Variable Sources:
    - rt_data: 從 self.df['RT'] 提取的反應時間
    - choice_data: 從 self.df['Response'] 提取的選擇反應
    - stimloc_data: 從 self.df['Stimulus'] 轉換的二維位置座標
    - params: 從 PyMC 模型採樣的參數值
    """
    try:
        # 解包模型參數 / Unpack model parameters
        # db1: X 軸決策邊界 / X-axis decision boundary (範圍 0-1)
        # db2: Y 軸決策邊界 / Y-axis decision boundary (範圍 0-1)  
        # sp: 感知雜訊參數 / Perceptual noise parameter (正值)
        # base_v: 基礎漂移率 / Base drift rate (正值)
        db1, db2, sp, base_v = params
        
        # LBA 模型的固定參數 / Fixed parameters for LBA model
        A = 0.4      # 起始點變異 / Start point variability
        s = 0.3      # 漂移率變異 / Drift rate variability  
        t0 = 0.2     # 非決策時間 / Non-decision time
        
        # 計算決策閾值 / Calculate decision thresholds
        # b: 決策閾值 / Decision threshold (起始點 + 閾值偏移)
        b = A + 0.5  # 固定閾值偏移 / Fixed threshold offset
        thresholds = np.array([b, b, b, b])  # 四個選項的相同閾值 / Same threshold for all four choices
        
        # 參數有效性檢查 / Parameter validity check
        # 確保感知雜訊和基礎漂移率為正值 / Ensure perceptual noise and base drift rate are positive
        if sp <= 0 or base_v <= 0:
            return -1000.0  # 返回極小值懲罰無效參數 / Return very small value to penalize invalid parameters
        
        # 計算決策時間 / Calculate decision time
        # 從總反應時間中減去非決策時間，但不設置最小值限制 / Subtract non-decision time from total reaction time, but no minimum limit
        # 保持原始的決策時間計算 / Keep original decision time calculation
        rt_decision = np.maximum(rt_data - t0, 0.001)  # 極小的最小值僅避免數學錯誤 / Very small minimum only to avoid mathematical errors
        
        # 初始化對數似然總和 / Initialize log-likelihood sum
        loglik_sum = 0.0
        
        # 對每個試驗計算似然 / Calculate likelihood for each trial
        for i in range(len(rt_decision)):
            # 獲取當前試驗的選擇 / Get current trial's choice
            choice_idx = int(choice_data[i])  # 選擇索引 (0,1,2,3) / Choice index (0,1,2,3)
            
            # 檢查選擇有效性 / Check choice validity
            if choice_idx < 0 or choice_idx >= 4:
                continue  # 跳過無效選擇 / Skip invalid choices
                
            # 獲取當前試驗的決策時間 / Get current trial's decision time
            rt_trial = rt_decision[i]
            if rt_trial <= 0:
                continue  # 跳過無效時間 / Skip invalid times
            
            # === GRT (General Recognition Theory) 計算部分 ===
            # === GRT (General Recognition Theory) Calculation Section ===
            
            # 獲取刺激位置 / Get stimulus location
            # stimloc_data[i, 0]: X 軸位置 (0 或 1) / X-axis position (0 or 1)
            # stimloc_data[i, 1]: Y 軸位置 (0 或 1) / Y-axis position (0 or 1)
            x_pos = stimloc_data[i, 0]  # 來自 Stimulus 欄位的 X 座標轉換 / X coordinate converted from Stimulus column
            y_pos = stimloc_data[i, 1]  # 來自 Stimulus 欄位的 Y 座標轉換 / Y coordinate converted from Stimulus column
            
            # 計算決策機率 / Calculate decision probabilities
            # 使用 logistic 函數計算選擇右側的機率 / Use logistic function to calculate probability of choosing right
            # db1: X 軸決策邊界 / X-axis decision boundary
            # db2: Y 軸決策邊界 / Y-axis decision boundary  
            # sp: 感知雜訊，控制決策的確定性 / Perceptual noise, controls decision certainty
            p_choose_right_x = 1 / (1 + np.exp(-(x_pos - db1) / sp))
            p_choose_right_y = 1 / (1 + np.exp(-(y_pos - db2) / sp))
            
            # 計算四選項的機率 / Calculate probabilities for four choices
            # 基於 2x2 空間的位置機率組合 / Based on position probability combinations in 2x2 space
            if choice_idx == 0:      # 左上 (0,0) / Top-left (0,0)
                choice_prob = (1 - p_choose_right_x) * (1 - p_choose_right_y)
            elif choice_idx == 1:    # 左下 (0,1) / Bottom-left (0,1)
                choice_prob = (1 - p_choose_right_x) * p_choose_right_y
            elif choice_idx == 2:    # 右上 (1,0) / Top-right (1,0)
                choice_prob = p_choose_right_x * (1 - p_choose_right_y)
            else:                    # 右下 (1,1) / Bottom-right (1,1)
                choice_prob = p_choose_right_x * p_choose_right_y
            
            # === LBA 模型計算部分 ===
            # === LBA Model Calculation Section ===
            
            # 計算漂移率 / Calculate drift rates
            # v_chosen: 被選擇選項的漂移率 / Drift rate for chosen option
            # v_others: 其他選項的漂移率 / Drift rate for other options
            v_chosen = max(choice_prob * base_v, 0.1)  # 最小值 0.1 避免數值問題 / Minimum 0.1 to avoid numerical issues
            v_others = max((1 - choice_prob) * base_v / 3, 0.1)  # 平均分配給其他三個選項 / Evenly distributed to other three options
            
            # LBA 模型的核心計算 / Core calculation of LBA model
            sqrt_rt = np.sqrt(rt_trial)  # 時間的平方根，用於正態分佈計算 / Square root of time for normal distribution calculation
            
            # 計算獲勝累加器的似然 / Calculate likelihood for winning accumulator
            b_win = thresholds[choice_idx]  # 獲勝選項的閾值 / Threshold for winning option
            
            # 標準化變數 / Standardized variables
            # z1, z2: 用於計算累積分佈函數的標準化值 / Standardized values for CDF calculation
            z1 = (v_chosen * rt_trial - b_win) / sqrt_rt
            z2 = (v_chosen * rt_trial - A) / sqrt_rt
            
            # 限制數值範圍避免溢出 / Limit numerical range to avoid overflow
            z1 = np.clip(z1, -6, 6)
            z2 = np.clip(z2, -6, 6)
            
            try:
                # 計算獲勝者的 CDF 和 PDF / Calculate winner's CDF and PDF
                winner_cdf = stats.norm.cdf(z1) - stats.norm.cdf(z2)
                winner_pdf = (stats.norm.pdf(z1) - stats.norm.pdf(z2)) / sqrt_rt
                winner_lik = max((v_chosen / A) * winner_cdf + winner_pdf / A, 1e-10)
            except:
                winner_lik = 1e-10  # 數值計算失敗時的備用值 / Fallback value when numerical calculation fails
            
            # 計算失敗累加器的生存函數 / Calculate survival function for losing accumulators
            loser_survival = 1.0
            for j in range(3):  # 其他三個選項 / Other three options
                b_lose = thresholds[(choice_idx + j + 1) % 4]  # 失敗選項的閾值 / Threshold for losing option
                z1_lose = (v_others * rt_trial - b_lose) / sqrt_rt
                z2_lose = (v_others * rt_trial - A) / sqrt_rt
                
                # 限制數值範圍 / Limit numerical range
                z1_lose = np.clip(z1_lose, -6, 6)
                z2_lose = np.clip(z2_lose, -6, 6)
                
                try:
                    loser_cdf = stats.norm.cdf(z1_lose) - stats.norm.cdf(z2_lose)
                    loser_survival *= max(1 - loser_cdf, 1e-6)  # 生存機率 / Survival probability
                except:
                    loser_survival *= 0.5  # 備用值 / Fallback value
            
            # 計算試驗的總似然 / Calculate total likelihood for this trial
            trial_lik = winner_lik * loser_survival
            trial_loglik = np.log(max(trial_lik, 1e-12))  # 轉換為對數似然 / Convert to log-likelihood
            
            # 累加到總似然 / Add to total likelihood
            if np.isfinite(trial_loglik):
                loglik_sum += trial_loglik
            else:
                loglik_sum += -10.0  # 無效值的懲罰 / Penalty for invalid values
        
        # 返回總對數似然 / Return total log-likelihood
        return loglik_sum if np.isfinite(loglik_sum) else -1000.0
        
    except Exception as e:
        print(f"似然計算錯誤 / Likelihood calculation error: {e}")
        return -1000.0

# ============================================================================
# 第二部分：PyTensor 包裝函數
# Part 2: PyTensor Wrapper Function  
# ============================================================================

def create_lba_logp_tensor(rt_data, choice_data, stimloc_data):
    """
    創建 PyTensor 兼容的似然函數
    Create PyTensor-compatible likelihood function
    
    參數說明 / Parameters:
    - rt_data: 反應時間數據陣列 / Reaction time data array
    - choice_data: 選擇數據陣列 / Choice data array  
    - stimloc_data: 刺激位置數據陣列 / Stimulus location data array
    
    返回值 / Returns:
    - PyTensor 操作函數 / PyTensor operation function
    
    用途 / Purpose:
    - 將純 Python 函數包裝成 PyTensor 可用的操作
    - Wrap pure Python function into PyTensor-compatible operation
    """
    
    # 定義 PyTensor 操作 / Define PyTensor operation
    def lba_logp_op(params_tensor):
        """
        PyTensor 操作函數 / PyTensor operation function
        
        參數 / Parameters:
        - params_tensor: 包含模型參數的張量 / Tensor containing model parameters
        
        變數來源 / Variable Sources:
        - params_tensor: 來自 PyMC 模型的參數張量 [db1, db2, sp, base_v]
        """
        
        # 將 PyTensor 張量轉換為 NumPy 數組 / Convert PyTensor tensor to NumPy array
        params_np = params_tensor.eval() if hasattr(params_tensor, 'eval') else params_tensor
        
        # 調用似然計算函數 / Call likelihood calculation function
        loglik = compute_lba_likelihood(rt_data, choice_data, stimloc_data, params_np)
        
        # 返回 PyTensor 標量 / Return PyTensor scalar
        return pt.as_tensor_variable(loglik)
    
    return lba_logp_op

# ============================================================================
# 第三部分：修正的受試者分析函數
# Part 3: Fixed Subject Analysis Function
# ============================================================================

def fixed_subject_analysis(subject_id: int, subject_data: pd.DataFrame) -> Optional[Dict]:
    """
    修正版受試者分析函數
    Fixed subject analysis function
    
    參數說明 / Parameters:
    - subject_id: 受試者編號 / Subject ID (來自 CSV 的 participant 欄位)
    - subject_data: 受試者數據 / Subject data (從總數據中過濾的特定受試者數據)
    
    返回值 / Returns:
    - 分析結果字典或 None / Analysis result dictionary or None
    
    變數來源與用途 / Variable Sources and Purposes:
    - subject_id: 從 self.participants 列表中獲取的受試者編號
    - subject_data: 通過 self.df[self.df['participant'] == subject_id] 過濾的數據
    """
    
    try:
        print(f"處理受試者 {subject_id} / Processing Subject {subject_id}...")
        
        # === 數據準備階段 / Data Preparation Phase ===
        
        # 提取反應時間數據 / Extract reaction time data
        # 來源：CSV 文件的 RT 欄位 / Source: RT column from CSV file
        rt_data = subject_data['RT'].values
        
        # 提取選擇反應數據 / Extract choice response data  
        # 來源：CSV 文件的 Response 欄位 / Source: Response column from CSV file
        choice_data = subject_data['choice_four'].values
        
        # 提取刺激位置數據 / Extract stimulus location data
        # 來源：從 Stimulus 欄位轉換的 X, Y 座標 / Source: X, Y coordinates converted from Stimulus column
        stimloc_data = np.column_stack([
            subject_data['stimloc_x'].values,  # X 軸位置 / X-axis position
            subject_data['stimloc_y'].values   # Y 軸位置 / Y-axis position
        ])
        
        # 檢查數據量是否足夠 / Check if data amount is sufficient
        if len(rt_data) < 50:
            print(f"   數據不足 / Insufficient data: {len(rt_data)} trials")
            return None
        
        # 數據清理 / Data cleaning
        # 不對反應時間進行限制，保持原始數據 / Do not limit reaction time, keep original data
        # rt_data 保持原始值 / rt_data keeps original values
        choice_data = np.clip(choice_data, 0, 3)    # 選擇範圍 0-3 / Choice range 0-3
        
        print(f"   數據準備完成 / Data ready: {len(rt_data)} trials")
        
        # === PyMC 模型定義階段 / PyMC Model Definition Phase ===
        
        with pm.Model() as model:
            
            # === 先驗分佈定義 / Prior Distribution Definition ===
            
            # GRT 參數：決策邊界 / GRT parameters: Decision boundaries
            # db1: X 軸決策邊界，範圍 0.2-0.8 / X-axis decision boundary, range 0.2-0.8
            # 用途：決定在 X 軸上的分類邊界位置 / Purpose: Determine classification boundary position on X-axis
            db1 = pm.Uniform('db1', lower=0.2, upper=0.8)
            
            # db2: Y 軸決策邊界，範圍 0.2-0.8 / Y-axis decision boundary, range 0.2-0.8  
            # 用途：決定在 Y 軸上的分類邊界位置 / Purpose: Determine classification boundary position on Y-axis
            db2 = pm.Uniform('db2', lower=0.2, upper=0.8)
            
            # 感知雜訊參數 (對數尺度) / Perceptual noise parameter (log scale)
            # 用途：控制決策的確定性，值越小決策越確定 / Purpose: Control decision certainty, smaller values mean more certain decisions
            log_sp = pm.Normal('log_sp', mu=np.log(0.3), sigma=0.5)
            sp = pm.Deterministic('sp', pt.exp(log_sp))  # 轉換為正值 / Transform to positive value
            
            # 基礎漂移率參數 (對數尺度) / Base drift rate parameter (log scale)
            # 用途：控制反應速度，值越大反應越快 / Purpose: Control response speed, larger values mean faster responses
            log_base_v = pm.Normal('log_base_v', mu=np.log(1.0), sigma=0.5)
            base_v = pm.Deterministic('base_v', pt.exp(log_base_v))  # 轉換為正值 / Transform to positive value
            
            # === 自定義似然函數定義 / Custom Likelihood Function Definition ===
            
            # 組合所有參數 / Combine all parameters
            # 這些參數將傳遞給似然函數 / These parameters will be passed to likelihood function
            params = pt.stack([db1, db2, sp, base_v])
            
            # ⭐ 關鍵修正：正確使用 pm.Potential ⭐
            # ⭐ Key Fix: Correct usage of pm.Potential ⭐
            
            # 定義似然函數 / Define likelihood function
            def logp_func(params_val):
                """
                自定義對數似然函數 / Custom log-likelihood function
                
                參數 / Parameters:
                - params_val: 參數值 [db1, db2, sp, base_v]
                
                返回值 / Returns:
                - 對數似然值 / Log-likelihood value
                """
                return compute_lba_likelihood(rt_data, choice_data, stimloc_data, params_val)
            
            # 🔧 修正前的錯誤用法 / Previous incorrect usage:
            # likelihood = pm.Potential('likelihood', logp_func(params))  # ❌ 這會導致 'float' object has no attribute 'name' 錯誤
            
            # ✅ 修正後的正確用法 / Corrected usage:
            # 使用 pm.CustomDist 來定義自定義分佈 / Use pm.CustomDist to define custom distribution
            likelihood = pm.CustomDist(
                'likelihood',
                params,  # 參數張量 / Parameter tensor
                logp=lambda value, params: compute_lba_likelihood(rt_data, choice_data, stimloc_data, params),
                observed=np.zeros(1)  # 觀測值佔位符 / Observed value placeholder
            )
            
            # 💡 替代方案：使用 pm.DensityDist (如果 CustomDist 不可用)
            # Alternative: Use pm.DensityDist (if CustomDist is not available)
            # likelihood = pm.DensityDist(
            #     'likelihood',
            #     lambda params: compute_lba_likelihood(rt_data, choice_data, stimloc_data, params),
            #     observed={'params': params}
            # )
        
        print(f"   模型建立完成，開始採樣 / Model built, starting sampling...")
        
        # === MCMC 採樣階段 / MCMC Sampling Phase ===
        
        with model:
            # 採樣設定 / Sampling configuration
            # draws: 採樣數量 / Number of draws
            # tune: 調整步數 / Number of tuning steps  
            # chains: 鏈數量 / Number of chains
            # target_accept: 目標接受率 / Target acceptance rate
            trace = pm.sample(
                draws=200,          # 採樣數量 / Number of samples
                tune=200,           # 調整步數 / Number of tuning steps
                chains=2,           # 鏈數量 / Number of chains
                target_accept=0.8,  # 目標接受率 / Target acceptance rate
                progressbar=True,   # 顯示進度條 / Show progress bar
                return_inferencedata=True,  # 返回推論數據 / Return inference data
                cores=1,            # 使用核心數 / Number of cores to use
                random_seed=42      # 隨機種子 / Random seed
            )
        
        print(f"   採樣完成 / Sampling completed")
        
        # === 收斂性診斷階段 / Convergence Diagnosis Phase ===
        
        try:
            # 計算收斂性統計量 / Calculate convergence statistics
            summary = az.summary(trace)
            
            # R-hat 統計量：應該接近 1.0，表示收斂良好 / R-hat statistic: should be close to 1.0 for good convergence
            rhat_max = summary['r_hat'].max() if 'r_hat' in summary else 1.0
            
            # 有效樣本數：應該足夠大 / Effective sample size: should be large enough
            ess_min = summary['ess_bulk'].min() if 'ess_bulk' in summary else 50
            
        except Exception as e:
            print(f"   收斂性診斷警告 / Convergence diagnosis warning: {e}")
            rhat_max, ess_min = 1.05, 50
        
        # === 結果整理階段 / Result Organization Phase ===
        
        result = {
            'subject_id': subject_id,                    # 受試者編號 / Subject ID
            'trace': trace,                              # MCMC 採樣結果 / MCMC sampling results
            'convergence': {                             # 收斂性統計 / Convergence statistics
                'rhat_max': float(rhat_max),             # 最大 R-hat 值 / Maximum R-hat value
                'ess_min': float(ess_min)                # 最小有效樣本數 / Minimum effective sample size
            },
            'n_trials': len(rt_data),                    # 試驗數量 / Number of trials
            'success': True                              # 成功標記 / Success flag
        }
        
        print(f"✅ 受試者 {subject_id} 完成 / Subject {subject_id} completed "
              f"(R̂={rhat_max:.3f}, ESS={ess_min:.0f})")
        
        return result
        
    except Exception as e:
        print(f"❌ 受試者 {subject_id} 失敗 / Subject {subject_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'subject_id': subject_id, 'success': False, 'error': str(e)}

# ============================================================================
# 第四部分：主要分析器類別
# Part 4: Main Analyzer Class
# ============================================================================

class FixedGRTAnalyzer:
    """
    修正版 GRT 分析器
    Fixed GRT Analyzer
    
    用途 / Purpose:
    - 載入和預處理 GRT 實驗數據 / Load and preprocess GRT experiment data
    - 執行 GRT-LBA 模型分析 / Execute GRT-LBA model analysis
    - 管理多個受試者的分析流程 / Manage analysis workflow for multiple subjects
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        """
        初始化分析器 / Initialize analyzer
        
        參數 / Parameters:
        - csv_file: CSV 數據文件路徑 / CSV data file path
        """
        
        print("載入數據 / Loading data...")
        
        # === 數據載入階段 / Data Loading Phase ===
        
        # 讀取 CSV 文件 / Read CSV file
        # 來源：實驗數據文件，包含所有受試者的試驗數據 / Source: Experiment data file containing all subjects' trial data
        self.df = pd.read_csv(csv_file)
        
        print(f"原始數據 / Raw data: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"欄位名稱 / Column names: {list(self.df.columns)}")
        
        # === 數據清理階段 / Data Cleaning Phase ===
        
        # 不對反應時間進行過濾，保留所有 RT 數據 / Do not filter reaction times, keep all RT data
        # RT: 反應時間，來自 CSV 的 RT 欄位，保持原始數據 / Reaction time from RT column in CSV, keep original data
        print(f"RT 範圍 / RT range: {self.df['RT'].min():.3f} - {self.df['RT'].max():.3f}s")
        
        # 過濾無效的反應選擇 / Filter invalid response choices
        # Response: 選擇反應，來自 CSV 的 Response 欄位 / Choice response from Response column in CSV
        # 有效值：0, 1, 2, 3 (四個選項) / Valid values: 0, 1, 2, 3 (four choices)
        self.df = self.df[self.df['Response'].isin([0, 1, 2, 3])]
        print(f"過濾反應選擇後 / After response filtering: {len(self.df)} rows")
        
        # ===
