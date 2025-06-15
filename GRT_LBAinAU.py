# -*- coding: utf-8 -*-
"""
Staged GRT-LBA Analysis with Sigma Matrix Preservation
階段式 GRT-LBA 分析，保持 Sigma 矩陣計算能力

Purpose / 目的:
- Implement staged Bayesian modeling to improve MCMC convergence
- 實施階段式貝葉斯建模來改善 MCMC 收斂性
- Preserve GRT theoretical framework throughout all stages  
- 在所有階段中保持 GRT 理論架構
- Enable Sigma matrix computation for independence testing
- 能夠計算 Sigma 矩陣進行獨立性檢驗

Strategy / 策略:
Stage 1: Simplified GRT-LBA (4 parameters / 4個參數)
Stage 2: Separate perceptual variabilities (5 parameters / 5個參數) 
Stage 3: Separate LBA thresholds (7 parameters / 7個參數)
Stage 4: Full model with all parameters (9 parameters / 9個參數)
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
warnings.filterwarnings('ignore')

# ============================================================================
# STAGE 1: SIMPLIFIED GRT-LBA MODEL
# 階段1：簡化的 GRT-LBA 模型
# ============================================================================

def stage1_simplified_grt_lba(rt_data: np.ndarray, choice_data: np.ndarray, 
                             stimloc_data: np.ndarray) -> pm.Model:
    """
    Stage 1: Simplified GRT-LBA Model
    階段1：簡化的 GRT-LBA 模型
    
    Purpose / 目的:
    - Establish baseline convergence with core GRT parameters
    - 建立核心 GRT 參數的基線收斂性
    - Reduce dimensionality by constraining similar parameters
    - 通過約束相似參數來降低維度
    - Maintain complete GRT→LBA transformation pathway
    - 維持完整的 GRT→LBA 轉換路徑
    
    Parameters Estimated / 估計參數:
    - db1, db2: Decision boundaries / 決策邊界
    - sp: Shared perceptual variability / 共享感知變異性 (sp1 = sp2 = sp)
    - A: LBA start point variability / LBA 起始點變異性
    - b: Shared threshold offset / 共享閾值偏移 (b1 = b2 = A + b)
    
    Fixed Parameters / 固定參數:
    - s = 0.3: Drift rate variability / 漂移率變異性
    - t0 = 0.25: Non-decision time / 非決策時間
    
    Expected Outcome / 預期結果:
    - R̂ < 1.1 for all parameters / 所有參數 R̂ < 1.1
    - Convergence within 10-15 minutes / 10-15分鐘內收斂
    """
    
    print("Stage 1: Building simplified GRT-LBA model...")
    print("階段1：建立簡化的 GRT-LBA 模型...")
    print("Parameters / 參數: db1, db2, sp (shared), A, b (shared)")
    
    with pm.Model() as model:
        # ====================================================================
        # GRT Parameters / GRT 參數
        # ====================================================================
        
        # Decision boundaries / 決策邊界
        # Purpose: Define decision regions in perceptual space
        # 目的：定義感知空間中的決策區域
        db1 = pm.Normal('db1', mu=0.5, sigma=0.3, 
                       doc="Decision boundary dimension 1 / 決策邊界維度1")
        db2 = pm.Normal('db2', mu=0.5, sigma=0.3,
                       doc="Decision boundary dimension 2 / 決策邊界維度2")
        
        # Shared perceptual variability / 共享感知變異性
        # Purpose: Simplify by assuming equal variability across dimensions
        # 目的：假設各維度變異性相等以簡化模型
        sp_shared = pm.HalfNormal('sp', sigma=0.2,
                                 doc="Shared perceptual variability / 共享感知變異性")
        
        # Create deterministic copies for Sigma matrix computation
        # 為 Sigma 矩陣計算創建確定性副本
        sp1 = pm.Deterministic('sp1', sp_shared,
                              doc="Perceptual variability dim 1 / 感知變異性維度1")
        sp2 = pm.Deterministic('sp2', sp_shared,
                              doc="Perceptual variability dim 2 / 感知變異性維度2")
        
        # ====================================================================
        # LBA Parameters / LBA 參數
        # ====================================================================
        
        # Start point variability / 起始點變異性
        # Purpose: Controls initial evidence accumulation range
        # 目的：控制初始證據累積範圍
        A = pm.HalfNormal('A', sigma=0.3,
                         doc="LBA start point variability / LBA 起始點變異性")
        
        # Shared threshold offset / 共享閾值偏移
        # Purpose: Simplify by assuming equal decision thresholds
        # 目的：假設相等決策閾值以簡化模型
        b_shared = pm.HalfNormal('b', sigma=0.4,
                                doc="Shared threshold offset / 共享閾值偏移")
        
        # Decision thresholds / 決策閾值
        # Purpose: Define evidence required for decision
        # 目的：定義決策所需的證據量
        b1 = pm.Deterministic('b1', A + b_shared,
                             doc="Threshold accumulator 1 / 閾值累加器1")
        b2 = pm.Deterministic('b2', A + b_shared,
                             doc="Threshold accumulator 2 / 閾值累加器2")
        
        # Fixed parameters for stability / 為穩定性固定的參數
        s_fixed = 0.3    # Drift rate variability / 漂移率變異性
        t0_fixed = 0.25  # Non-decision time / 非決策時間
        
        # ====================================================================
        # GRT→LBA Transformation / GRT→LBA 轉換
        # ====================================================================
        
        # Convert GRT parameters to LBA drift rates
        # 將 GRT 參數轉換為 LBA 漂移率
        # This maintains the theoretical connection between GRT and LBA
        # 這維持了 GRT 和 LBA 之間的理論聯繫
        
        pm.Potential('grt_lba_likelihood',
                    enhanced_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, s_fixed, t0_fixed),
                    doc="GRT-LBA likelihood / GRT-LBA 似然函數")
    
    return model

# ============================================================================
# STAGE 2: SEPARATE PERCEPTUAL VARIABILITIES
# 階段2：分離感知變異性
# ============================================================================

def stage2_separate_perceptual_variabilities(rt_data: np.ndarray, choice_data: np.ndarray, 
                                           stimloc_data: np.ndarray, 
                                           stage1_trace: az.InferenceData) -> pm.Model:
    """
    Stage 2: Separate Perceptual Variabilities
    階段2：分離感知變異性
    
    Purpose / 目的:
    - Test GRT separability assumption (sp1 ≠ sp2)
    - 測試 GRT 可分離性假設 (sp1 ≠ sp2)
    - Use Stage 1 results as informative priors
    - 使用階段1結果作為資訊性先驗
    - Maintain all other simplifications for stability
    - 維持所有其他簡化以保持穩定性
    
    New Parameters / 新參數:
    - sp1, sp2: Separate perceptual variabilities / 分離的感知變異性
    
    Theoretical Significance / 理論意義:
    - Enables testing of perceptual independence
    - 能夠測試感知獨立性
    - Critical for GRT assumptions validation
    - 對 GRT 假設驗證至關重要
    """
    
    print("Stage 2: Separating perceptual variabilities...")
    print("階段2：分離感知變異性...")
    print("New parameters / 新參數: sp1, sp2 (separate)")
    
    # Extract Stage 1 posterior summaries / 提取階段1後驗摘要
    stage1_summary = az.summary(stage1_trace)
    
    with pm.Model() as model:
        # ====================================================================
        # GRT Parameters with Stage 1 Priors / 使用階段1先驗的 GRT 參數
        # ====================================================================
        
        # Decision boundaries using Stage 1 results as priors
        # 使用階段1結果作為先驗的決策邊界
        db1_mean = float(stage1_summary.loc['db1', 'mean'])
        db1 = pm.Normal('db1', mu=db1_mean, sigma=0.1,
                       doc="Decision boundary 1 with informed prior / 有信息先驗的決策邊界1")
        
        db2_mean = float(stage1_summary.loc['db2', 'mean'])
        db2 = pm.Normal('db2', mu=db2_mean, sigma=0.1,
                       doc="Decision boundary 2 with informed prior / 有信息先驗的決策邊界2")
        
        # NEW: Separate perceptual variabilities / 新增：分離的感知變異性
        # Purpose: Test independence assumption of GRT
        # 目的：測試 GRT 的獨立性假設
        sp_prior_mean = float(stage1_summary.loc['sp', 'mean'])
        
        sp1 = pm.HalfNormal('sp1', sigma=sp_prior_mean * 0.5,
                           doc="Perceptual variability dimension 1 / 感知變異性維度1")
        sp2 = pm.HalfNormal('sp2', sigma=sp_prior_mean * 0.5,
                           doc="Perceptual variability dimension 2 / 感知變異性維度2")
        
        # ====================================================================
        # LBA Parameters (constrained by Stage 1) / LBA 參數（受階段1約束）
        # ====================================================================
        
        A_mean = float(stage1_summary.loc['A', 'mean'])
        A = pm.Normal('A', mu=A_mean, sigma=0.05,
                     doc="LBA start point variability / LBA 起始點變異性")
        
        b_mean = float(stage1_summary.loc['b', 'mean'])
        b_shared = pm.Normal('b', mu=b_mean, sigma=0.05,
                            doc="Shared threshold offset / 共享閾值偏移")
        
        # Still using shared thresholds / 仍使用共享閾值
        b1 = pm.Deterministic('b1', A + b_shared)
        b2 = pm.Deterministic('b2', A + b_shared)
        
        # Fixed parameters / 固定參數
        s_fixed = 0.3
        t0_fixed = 0.25
        
        # ====================================================================
        # Enhanced GRT→LBA Transformation / 增強的 GRT→LBA 轉換
        # ====================================================================
        
        pm.Potential('grt_lba_likelihood',
                    enhanced_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, s_fixed, t0_fixed))
    
    return model

# ============================================================================
# STAGE 3: SEPARATE LBA THRESHOLDS
# 階段3：分離 LBA 閾值
# ============================================================================

def stage3_separate_lba_thresholds(rt_data: np.ndarray, choice_data: np.ndarray, 
                                  stimloc_data: np.ndarray, 
                                  stage2_trace: az.InferenceData) -> pm.Model:
    """
    Stage 3: Separate LBA Thresholds
    階段3：分離 LBA 閾值
    
    Purpose / 目的:
    - Allow for response bias in LBA accumulators
    - 允許 LBA 累加器中的反應偏誤
    - Test threshold independence (b1 ≠ b2)
    - 測試閾值獨立性 (b1 ≠ b2)
    - Maintain GRT structure while adding LBA flexibility
    - 在增加 LBA 靈活性的同時維持 GRT 結構
    
    New Parameters / 新參數:
    - bMa1, bMa2: Separate threshold offsets / 分離的閾值偏移
    
    Model Complexity / 模型複雜性:
    - 7 parameters total / 總共7個參數
    - Approaching full model complexity / 接近完整模型複雜性
    """
    
    print("Stage 3: Separating LBA thresholds...")
    print("階段3：分離 LBA 閾值...")
    print("New parameters / 新參數: bMa1, bMa2 (separate threshold offsets)")
    
    stage2_summary = az.summary(stage2_trace)
    
    with pm.Model() as model:
        # ====================================================================
        # GRT Parameters (well-informed priors) / GRT 參數（充分信息的先驗）
        # ====================================================================
        
        # Very tight priors based on Stage 2 convergence
        # 基於階段2收斂的非常緊的先驗
        db1_mean = float(stage2_summary.loc['db1', 'mean'])
        db1 = pm.Normal('db1', mu=db1_mean, sigma=0.05)
        
        db2_mean = float(stage2_summary.loc['db2', 'mean'])
        db2 = pm.Normal('db2', mu=db2_mean, sigma=0.05)
        
        sp1_mean = float(stage2_summary.loc['sp1', 'mean'])
        sp1 = pm.Normal('sp1', mu=sp1_mean, sigma=0.05)
        
        sp2_mean = float(stage2_summary.loc['sp2', 'mean'])
        sp2 = pm.Normal('sp2', mu=sp2_mean, sigma=0.05)
        
        # ====================================================================
        # LBA Parameters with Separate Thresholds / 分離閾值的 LBA 參數
        # ====================================================================
        
        A_mean = float(stage2_summary.loc['A', 'mean'])
        A = pm.Normal('A', mu=A_mean, sigma=0.05)
        
        # NEW: Separate threshold offsets / 新增：分離的閾值偏移
        # Purpose: Allow for response bias and threshold asymmetry
        # 目的：允許反應偏誤和閾值不對稱
        b_prior_mean = float(stage2_summary.loc['b', 'mean'])
        
        bMa1 = pm.HalfNormal('bMa1', sigma=b_prior_mean * 0.5,
                            doc="Threshold offset accumulator 1 / 閾值偏移累加器1")
        bMa2 = pm.HalfNormal('bMa2', sigma=b_prior_mean * 0.5,
                            doc="Threshold offset accumulator 2 / 閾值偏移累加器2")
        
        # Separate decision thresholds / 分離的決策閾值
        b1 = pm.Deterministic('b1', A + bMa1,
                             doc="Decision threshold accumulator 1 / 決策閾值累加器1")
        b2 = pm.Deterministic('b2', A + bMa2,
                             doc="Decision threshold accumulator 2 / 決策閾值累加器2")
        
        # Still fixed / 仍然固定
        s_fixed = 0.3
        t0_fixed = 0.25
        
        # ====================================================================
        # Full GRT→LBA with Threshold Flexibility / 具有閾值靈活性的完整 GRT→LBA
        # ====================================================================
        
        pm.Potential('grt_lba_likelihood',
                    enhanced_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, s_fixed, t0_fixed))
    
    return model

# ============================================================================
# STAGE 4: FULL GRT-LBA MODEL
# 階段4：完整 GRT-LBA 模型
# ============================================================================

def stage4_full_grt_lba(rt_data: np.ndarray, choice_data: np.ndarray, 
                       stimloc_data: np.ndarray, 
                       stage3_trace: az.InferenceData) -> pm.Model:
    """
    Stage 4: Full GRT-LBA Model
    階段4：完整 GRT-LBA 模型
    
    Purpose / 目的:
    - Estimate all parameters with maximum flexibility
    - 以最大靈活性估計所有參數
    - Complete the theoretical GRT-LBA integration
    - 完成理論上的 GRT-LBA 整合
    - Enable full Sigma matrix computation
    - 能夠進行完整的 Sigma 矩陣計算
    
    All Parameters Estimated / 估計所有參數:
    - GRT: db1, db2, sp1, sp2
    - LBA: A, b1, b2, s, t0
    
    Final Objectives / 最終目標:
    - Test complete GRT assumptions
    - 測試完整的 GRT 假設
    - Compute comprehensive Sigma matrix
    - 計算綜合 Sigma 矩陣
    - Validate independence and separability
    - 驗證獨立性和可分離性
    """
    
    print("Stage 4: Full GRT-LBA model with all parameters...")
    print("階段4：具有所有參數的完整 GRT-LBA 模型...")
    print("All parameters / 所有參數: db1, db2, sp1, sp2, A, bMa1, bMa2, s, t0")
    
    stage3_summary = az.summary(stage3_trace)
    
    with pm.Model() as model:
        # ====================================================================
        # GRT Parameters (final estimation) / GRT 參數（最終估計）
        # ====================================================================
        
        # Decision boundaries with refined priors
        # 具有精細先驗的決策邊界
        db1_mean = float(stage3_summary.loc['db1', 'mean'])
        db1 = pm.Normal('db1', mu=db1_mean, sigma=0.05,
                       doc="Final decision boundary 1 / 最終決策邊界1")
        
        db2_mean = float(stage3_summary.loc['db2', 'mean'])
        db2 = pm.Normal('db2', mu=db2_mean, sigma=0.05,
                       doc="Final decision boundary 2 / 最終決策邊界2")
        
        # Perceptual variabilities with refined priors
        # 具有精細先驗的感知變異性
        sp1_mean = float(stage3_summary.loc['sp1', 'mean'])
        sp1 = pm.Normal('sp1', mu=sp1_mean, sigma=0.05,
                       doc="Final perceptual variability 1 / 最終感知變異性1")
        
        sp2_mean = float(stage3_summary.loc['sp2', 'mean'])
        sp2 = pm.Normal('sp2', mu=sp2_mean, sigma=0.05,
                       doc="Final perceptual variability 2 / 最終感知變異性2")
        
        # ====================================================================
        # LBA Parameters (complete estimation) / LBA 參數（完整估計）
        # ====================================================================
        
        A_mean = float(stage3_summary.loc['A', 'mean'])
        A = pm.Normal('A', mu=A_mean, sigma=0.05,
                     doc="Final start point variability / 最終起始點變異性")
        
        bMa1_mean = float(stage3_summary.loc['bMa1', 'mean'])
        bMa1 = pm.Normal('bMa1', mu=bMa1_mean, sigma=0.05,
                        doc="Final threshold offset 1 / 最終閾值偏移1")
        
        bMa2_mean = float(stage3_summary.loc['bMa2', 'mean'])
        bMa2 = pm.Normal('bMa2', mu=bMa2_mean, sigma=0.05,
                        doc="Final threshold offset 2 / 最終閾值偏移2")
        
        b1 = pm.Deterministic('b1', A + bMa1)
        b2 = pm.Deterministic('b2', A + bMa2)
        
        # NEW: Now estimate drift rate variability and non-decision time
        # 新增：現在估計漂移率變異性和非決策時間
        s = pm.HalfNormal('s', sigma=0.2,
                         doc="Drift rate variability / 漂移率變異性")
        t0 = pm.HalfNormal('t0', sigma=0.1,
                          doc="Non-decision time / 非決策時間")
        
        # ====================================================================
        # Complete GRT→LBA Transformation / 完整的 GRT→LBA 轉換
        # ====================================================================
        
        pm.Potential('grt_lba_likelihood',
                    enhanced_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, s, t0))
    
    return model

# ============================================================================
# ENHANCED LIKELIHOOD FUNCTION
# 增強的似然函數
# ============================================================================

def enhanced_vectorized_lba_loglik(rt_data: np.ndarray, choice_data: np.ndarray, 
                                  stimloc: np.ndarray, db1, db2, sp1, sp2, 
                                  A, b1, b2, s, t0):
    """
    Enhanced Vectorized LBA Log-Likelihood with Improved Numerical Stability
    具有改進數值穩定性的增強向量化 LBA 對數似然函數
    
    Purpose / 目的:
    - Maintain theoretical accuracy while improving convergence
    - 在改善收斂性的同時保持理論準確性
    - Handle edge cases and numerical instabilities
    - 處理邊界情況和數值不穩定性
    - Preserve GRT→LBA transformation fidelity
    - 保持 GRT→LBA 轉換的保真度
    """
    
    # Enhanced numerical stability constraints
    # 增強的數值穩定性約束
    A = pt.maximum(A, 0.1)
    b1 = pt.maximum(b1, A + 0.15)
    b2 = pt.maximum(b2, A + 0.15)
    s = pt.maximum(s, 0.15)
    t0 = pt.maximum(t0, 0.05)
    sp1 = pt.maximum(sp1, 0.05)
    sp2 = pt.maximum(sp2, 0.05)
    
    # Decision time calculation / 決策時間計算
    rt_decision = pt.maximum(rt_data - t0, 0.05)
    
    # GRT→LBA transformation with enhanced stability
    # 具有增強穩定性的 GRT→LBA 轉換
    v1_prob = 0.5 * (1 + pt.tanh((db1 - stimloc[:, 0]) / (2 * sp1)))
    v2_prob = 0.5 * (1 + pt.tanh((db2 - stimloc[:, 1]) / (2 * sp2)))
    
    # Combine probabilities for drift rates
    # 結合漂移率的概率
    v1 = v1_prob * v2_prob
    v2 = 1 - v1
    
    # Ensure meaningful drift rates
    # 確保有意義的漂移率
    v1 = pt.maximum(v1, 0.15)
    v2 = pt.maximum(v2, 0.15)
    
    # Normalize and scale / 歸一化和縮放
    v_sum = v1 + v2
    v1 = (v1 / v_sum) * s
    v2 = (v2 / v_sum) * s
    
    # Vectorized parameter selection / 向量化參數選擇
    v_winner = pt.where(pt.eq(choice_data, 0), v1, v2)
    v_loser = pt.where(pt.eq(choice_data, 0), v2, v1)
    b_winner = pt.where(pt.eq(choice_data, 0), b1, b2)
    b_loser = pt.where(pt.eq(choice_data, 0), b2, b1)
    
    # Enhanced LBA likelihood calculation
    # 增強的 LBA 似然計算
    rt_decision = pt.maximum(rt_decision, 0.05)
    sqrt_t = pt.sqrt(rt_decision)
    
    # Winner PDF with enhanced numerical bounds
    # 具有增強數值邊界的獲勝者PDF
    z1_win = (v_winner * rt_decision - b_winner) / sqrt_t
    z2_win = (v_winner * rt_decision - A) / sqrt_t
    z1_win = pt.clip(z1_win, -8, 8)
    z2_win = pt.clip(z2_win, -8, 8)
    
    # Stable normal calculations / 穩定的正態計算
    Phi_z1_win = 0.5 * (1 + pt.erf(z1_win / pt.sqrt(2)))
    Phi_z2_win = 0.5 * (1 + pt.erf(z2_win / pt.sqrt(2)))
    Phi_z1_win = pt.clip(Phi_z1_win, 1e-8, 1 - 1e-8)
    Phi_z2_win = pt.clip(Phi_z2_win, 1e-8, 1 - 1e-8)
    
    phi_z1_win = pt.exp(-0.5 * z1_win**2) / pt.sqrt(2 * np.pi)
    phi_z2_win = pt.exp(-0.5 * z2_win**2) / pt.sqrt(2 * np.pi)
    
    # Winner PDF calculation / 獲勝者PDF計算
    cdf_diff = pt.maximum(Phi_z1_win - Phi_z2_win, 1e-10)
    pdf_diff = (phi_z1_win - phi_z2_win) / sqrt_t
    
    winner_pdf = (v_winner / A) * cdf_diff + pdf_diff / A
    winner_pdf = pt.maximum(winner_pdf, 1e-10)
    winner_logpdf = pt.log(winner_pdf)
    
    # Loser survival function / 失敗者生存函數
    z1_lose = (v_loser * rt_decision - b_loser) / sqrt_t
    z2_lose = (v_loser * rt_decision - A) / sqrt_t
    z1_lose = pt.clip(z1_lose, -8, 8)
    z2_lose = pt.clip(z2_lose, -8, 8)
    
    Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
    Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
    Phi_z1_lose = pt.clip(Phi_z1_lose, 1e-8, 1 - 1e-8)
    Phi_z2_lose = pt.clip(Phi_z2_lose, 1e-8, 1 - 1e-8)
    
    loser_cdf = pt.maximum(Phi_z1_lose - Phi_z2_lose, 1e-10)
    loser_survival = pt.maximum(1 - loser_cdf, 1e-10)
    loser_log_survival = pt.log(loser_survival)
    
    # Combine with safety checks / 結合安全檢查
    trial_loglik = winner_logpdf + loser_log_survival
    trial_loglik = pt.where(pt.isnan(trial_loglik), -1000.0, trial_loglik)
    trial_loglik = pt.where(pt.isinf(trial_loglik), -1000.0, trial_loglik)
    trial_loglik = pt.maximum(trial_loglik, -1000.0)
    
    return pt.sum(trial_loglik)

# ============================================================================
# STAGED ANALYSIS CONTROLLER
# 階段式分析控制器
# ============================================================================

class StagedGRTLBAAnalyzer:
    """
    Staged GRT-LBA Bayesian Analysis Controller
    階段式 GRT-LBA 貝葉斯分析控制器
    
    Purpose / 目的:
    - Orchestrate staged modeling approach for improved convergence
    - 編排階段式建模方法以改善收斂性
    - Preserve Sigma matrix computation throughout all stages
    - 在所有階段中保持 Sigma 矩陣計算
    - Enable comprehensive GRT assumption testing
    - 能夠進行綜合的 GRT 假設檢驗
    
    Features / 特色:
    - Automatic convergence checking between stages
    - 階段間自動收斂檢查
    - Fallback to previous stage if convergence fails
    - 如果收斂失敗則回退到前一階段
    - Comprehensive results reporting
    - 綜合結果報告
    - Sigma matrix computation and independence testing
    - Sigma 矩陣計算和獨立性測試
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        """
        Initialize Staged GRT-LBA Analyzer
        初始化階段式 GRT-LBA 分析器
        
        Parameters / 參數:
        csv_file: Path to data file / 數據文件路徑
        """
        self.csv_file = csv_file
        self.results_dir = Path('staged_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Analysis tracking / 分析追蹤
        self.stage_results = {}  # Results for each stage / 每個階段的結果
        self.stage_traces = {}   # MCMC traces for each stage / 每個階段的MCMC軌跡
        self.final_results = {}  # Final analysis results / 最終分析結果
        
        # Load and prepare data / 載入和準備數據
        self.load_and_prepare_data()
        
        print("Staged GRT-LBA Analyzer initialized")
        print("階段式 GRT-LBA 分析器已初始化")
        print(f"Data: {len(self.df)} trials, {self.n_participants} participants")
        print(f"數據：{len(self.df)} 次試驗，{self.n_participants} 名參與者")
    
    def load_and_prepare_data(self):
        """
        Load and prepare data for staged analysis
        載入和準備階段式分析的數據
        """
        print("Loading data for staged GRT-LBA analysis...")
        print("載入階段式 GRT-LBA 分析數據...")
        
        df = pd.read_csv(self.csv_file)
        print(f"Original data: {len(df)} trials / 原始數據：{len(df)} 次試驗")
        
        # Standard RT filtering / 標準RT過濾
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.0)]
        print(f"After RT filtering: {len(df)} trials / RT過濾後：{len(df)} 次試驗")
        
        # Binary choice conversion / 二元選擇轉換
        df['choice_binary'] = (df['Response'] >= 2).astype(int)
        
        # Stimulus location mapping / 刺激位置映射
        if 'Stimulus' in df.columns:
            unique_stimuli = sorted(df['Stimulus'].unique())
            n_stim = len(unique_stimuli)
            
            # Create 2D grid layout / 創建2D網格佈局
            if n_stim <= 4:
                stim_locs = [[i % 2, i // 2] for i in range(n_stim)]
            else:
                stim_locs = [[i % 3, i // 3] for i in range(n_stim)]
            
            self.stimloc = np.array(stim_locs)
            stim_to_loc = {stim: loc for stim, loc in zip(unique_stimuli, stim_locs)}
            
            df['stimloc_x'] = df['Stimulus'].map(lambda x: stim_to_loc[x][0])
            df['stimloc_y'] = df['Stimulus'].map(lambda x: stim_to_loc[x][1])
        else:
            # Default layout / 默認佈局
            self.stimloc = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
            df['stimloc_x'] = 0
            df['stimloc_y'] = 0
        
        self.df = df
        self.participants = sorted(df['participant'].unique())
        self.n_participants = len(self.participants)
        
        print(f"Participants: {self.participants}")
        print(f"參與者：{self.participants}")
    
    def analyze_single_subject_staged(self, subject_id: int, 
                                    draws: int = 500, tune: int = 300, 
                                    chains: int = 2) -> Dict[str, Any]:
        """
        Perform staged analysis for a single subject
        對單個受試者進行階段式分析
        
        Parameters / 參數:
        subject_id: Subject identifier / 受試者標識符
        draws: MCMC draws per stage / 每階段MCMC抽樣數
        tune: MCMC tuning steps per stage / 每階段MCMC調整步數
        chains: Number of MCMC chains / MCMC鏈數量
        
        Returns / 返回:
        Dict containing results from successful stage / 包含成功階段結果的字典
        """
        
        print(f"\n{'='*60}")
        print(f"STAGED ANALYSIS FOR SUBJECT {subject_id}")
        print(f"受試者 {subject_id} 的階段式分析")
        print(f"{'='*60}")
        
        # Prepare subject data / 準備受試者數據
        subject_data = self.df[self.df['participant'] == subject_id].copy()
        
        if len(subject_data) < 20:
            print(f"⚠️  Warning: Subject {subject_id} has only {len(subject_data)} trials")
            print(f"⚠️  警告：受試者 {subject_id} 只有 {len(subject_data)} 次試驗")
            return None
        
        # Data arrays / 數據陣列
        rt_data = subject_data['RT'].values.astype(np.float32)
        choice_data = subject_data['choice_binary'].values.astype(np.int32)
        stimloc_data = np.column_stack([
            subject_data['stimloc_x'].values,
            subject_data['stimloc_y'].values
        ]).astype(np.float32)
        
        print(f"Data prepared: {len(rt_data)} trials")
        print(f"數據準備完成：{len(rt_data)} 次試驗")
        print(f"Choice distribution: {np.bincount(choice_data)}")
        print(f"選擇分佈：{np.bincount(choice_data)}")
        
        # Initialize stage tracking / 初始化階段追蹤
        stage_traces = {}
        stage_results = {}
        successful_stage = 0
        
        # ====================================================================
        # STAGE 1: SIMPLIFIED GRT-LBA
        # 階段1：簡化的 GRT-LBA
        # ====================================================================
        
        print(f"\n{'-'*40}")
        print("STAGE 1: Simplified GRT-LBA")
        print("階段1：簡化的 GRT-LBA")
        print(f"{'-'*40}")
        
        try:
            model1 = stage1_simplified_grt_lba(rt_data, choice_data, stimloc_data)
            
            print("Starting MCMC sampling...")
            print("開始 MCMC 採樣...")
            
            with model1:
                trace1 = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.95, max_treedepth=12,
                    cores=1, random_seed=123 + subject_id
                )
            
            # Check convergence / 檢查收斂性
            rhat_max = float(az.rhat(trace1).max())
            ess_min = float(az.ess(trace1).min())
            
            print(f"Stage 1 Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            print(f"階段1收斂性：R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            
            if rhat_max < 1.15:  # Slightly relaxed for early stages
                print("✅ Stage 1 SUCCESSFUL")
                print("✅ 階段1 成功")
                stage_traces[1] = trace1
                stage_results[1] = self.analyze_stage_results(trace1, 1, subject_id)
                successful_stage = 1
            else:
                print("❌ Stage 1 FAILED - Poor convergence")
                print("❌ 階段1 失敗 - 收斂性差")
                return None
                
        except Exception as e:
            print(f"❌ Stage 1 FAILED - Sampling error: {e}")
            print(f"❌ 階段1 失敗 - 採樣錯誤：{e}")
            return None
        
        # ====================================================================
        # STAGE 2: SEPARATE PERCEPTUAL VARIABILITIES
        # 階段2：分離感知變異性
        # ====================================================================
        
        print(f"\n{'-'*40}")
        print("STAGE 2: Separate Perceptual Variabilities")
        print("階段2：分離感知變異性")
        print(f"{'-'*40}")
        
        try:
            model2 = stage2_separate_perceptual_variabilities(
                rt_data, choice_data, stimloc_data, trace1)
            
            with model2:
                trace2 = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.95, max_treedepth=12,
                    cores=1, random_seed=123 + subject_id
                )
            
            rhat_max = float(az.rhat(trace2).max())
            ess_min = float(az.ess(trace2).min())
            
            print(f"Stage 2 Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            print(f"階段2收斂性：R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            
            if rhat_max < 1.15:
                print("✅ Stage 2 SUCCESSFUL")
                print("✅ 階段2 成功")
                stage_traces[2] = trace2
                stage_results[2] = self.analyze_stage_results(trace2, 2, subject_id)
                successful_stage = 2
            else:
                print("⚠️  Stage 2 FAILED - Using Stage 1 results")
                print("⚠️  階段2 失敗 - 使用階段1結果")
                
        except Exception as e:
            print(f"⚠️  Stage 2 FAILED - Sampling error: {e}")
            print(f"⚠️  階段2 失敗 - 採樣錯誤：{e}")
        
        # ====================================================================
        # STAGE 3: SEPARATE LBA THRESHOLDS
        # 階段3：分離 LBA 閾值
        # ====================================================================
        
        if successful_stage >= 2:
            print(f"\n{'-'*40}")
            print("STAGE 3: Separate LBA Thresholds")
            print("階段3：分離 LBA 閾值")
            print(f"{'-'*40}")
            
            try:
                model3 = stage3_separate_lba_thresholds(
                    rt_data, choice_data, stimloc_data, trace2)
                
                with model3:
                    trace3 = pm.sample(
                        draws=draws, tune=tune, chains=chains,
                        progressbar=True, return_inferencedata=True,
                        target_accept=0.96, max_treedepth=14,
                        cores=1, random_seed=123 + subject_id
                    )
                
                rhat_max = float(az.rhat(trace3).max())
                ess_min = float(az.ess(trace3).min())
                
                print(f"Stage 3 Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
                print(f"階段3收斂性：R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
                
                if rhat_max < 1.15:
                    print("✅ Stage 3 SUCCESSFUL")
                    print("✅ 階段3 成功")
                    stage_traces[3] = trace3
                    stage_results[3] = self.analyze_stage_results(trace3, 3, subject_id)
                    successful_stage = 3
                else:
                    print("⚠️  Stage 3 FAILED - Using Stage 2 results")
                    print("⚠️  階段3 失敗 - 使用階段2結果")
                    
            except Exception as e:
                print(f"⚠️  Stage 3 FAILED - Sampling error: {e}")
                print(f"⚠️  階段3 失敗 - 採樣錯誤：{e}")
        
        # ====================================================================
        # STAGE 4: FULL GRT-LBA MODEL
        # 階段4：完整 GRT-LBA 模型
        # ====================================================================
        
        if successful_stage >= 3:
            print(f"\n{'-'*40}")
            print("STAGE 4: Full GRT-LBA Model")
            print("階段4：完整 GRT-LBA 模型")
            print(f"{'-'*40}")
            
            try:
                model4 = stage4_full_grt_lba(
                    rt_data, choice_data, stimloc_data, trace3)
                
                with model4:
                    trace4 = pm.sample(
                        draws=draws + 200, tune=tune + 100, chains=chains,  # More sampling for full model
                        progressbar=True, return_inferencedata=True,
                        target_accept=0.97, max_treedepth=15,
                        cores=1, random_seed=123 + subject_id
                    )
                
                rhat_max = float(az.rhat(trace4).max())
                ess_min = float(az.ess(trace4).min())
                
                print(f"Stage 4 Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
                print(f"階段4收斂性：R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
                
                if rhat_max < 1.1:  # Stricter for final model
                    print("✅ Stage 4 SUCCESSFUL - Full model converged!")
                    print("✅ 階段4 成功 - 完整模型收斂！")
                    stage_traces[4] = trace4
                    stage_results[4] = self.analyze_stage_results(trace4, 4, subject_id)
                    successful_stage = 4
                else:
                    print("⚠️  Stage 4 FAILED - Using Stage 3 results")
                    print("⚠️  階段4 失敗 - 使用階段3結果")
                    
            except Exception as e:
                print(f"⚠️  Stage 4 FAILED - Sampling error: {e}")
                print(f"⚠️  階段4 失敗 - 採樣錯誤：{e}")
        
        # ====================================================================
        # COMPILE FINAL RESULTS
        # 編譯最終結果
        # ====================================================================
        
        print(f"\n{'='*40}")
        print(f"FINAL RESULTS FOR SUBJECT {subject_id}")
        print(f"受試者 {subject_id} 的最終結果")
        print(f"{'='*40}")
        print(f"Successful stages: {list(stage_results.keys())}")
        print(f"成功階段：{list(stage_results.keys())}")
        print(f"Using results from Stage {successful_stage}")
        print(f"使用階段 {successful_stage} 的結果")
        
        # Store results for this subject
        # 存儲此受試者的結果
        final_result = {
            'subject_id': subject_id,
            'successful_stages': list(stage_results.keys()),
            'final_stage_used': successful_stage,
            'stage_traces': stage_traces,
            'stage_results': stage_results,
            'sigma_matrix': self.compute_sigma_matrix(stage_traces[successful_stage]),
            'convergence_summary': self.summarize_convergence(stage_traces[successful_stage])
        }
        
        # Save individual results
        # 保存個別結果
        self.save_subject_results(final_result, subject_id)
        
        return final_result
    
    def analyze_stage_results(self, trace: az.InferenceData, stage: int, 
                            subject_id: int) -> Dict[str, Any]:
        """
        Analyze results from a specific stage
        分析特定階段的結果
        
        Parameters / 參數:
        trace: ArviZ InferenceData object / ArviZ 推論數據對象
        stage: Stage number / 階段編號
        subject_id: Subject identifier / 受試者標識符
        
        Returns / 返回:
        Dictionary with stage-specific analysis / 包含階段特定分析的字典
        """
        
        posterior = trace.posterior
        summary = az.summary(trace)
        
        # Extract parameter estimates / 提取參數估計
        results = {
            'stage': stage,
            'subject_id': subject_id,
            'parameter_estimates': {},
            'convergence_diagnostics': {}
        }
        
        # Parameter estimates / 參數估計
        for param in posterior.data_vars:
            if param in summary.index:
                samples = posterior[param].values.flatten()
                results['parameter_estimates'][param] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'hdi_2.5': float(np.percentile(samples, 2.5)),
                    'hdi_97.5': float(np.percentile(samples, 97.5)),
                    'median': float(np.median(samples))
                }
        
        # Convergence diagnostics / 收斂診斷
        try:
            ess = az.ess(trace)
            rhat = az.rhat(trace)
            
            for param in posterior.data_vars:
                if param in ess.data_vars:
                    ess_val = float(ess[param]) if ess[param].ndim == 0 else float(ess[param].min())
                    rhat_val = float(rhat[param]) if rhat[param].ndim == 0 else float(rhat[param].max())
                    
                    results['convergence_diagnostics'][param] = {
                        'ess': ess_val,
                        'rhat': rhat_val,
                        'converged': rhat_val < 1.1
                    }
        except Exception as e:
            results['convergence_diagnostics'] = {'error': str(e)}
        
        return results
    
    def compute_sigma_matrix(self, trace: az.InferenceData) -> Dict[str, Any]:
        """
        Compute Sigma matrix for GRT independence testing
        計算用於 GRT 獨立性測試的 Sigma 矩陣
        
        Purpose / 目的:
        - Extract covariance matrix of GRT parameters
        - 提取 GRT 參數的共變異數矩陣
        - Test independence assumptions
        - 測試獨立性假設
        - Provide correlation analysis
        - 提供相關性分析
        """
        
        posterior = trace.posterior
        
        # GRT parameters for Sigma matrix / Sigma 矩陣的 GRT 參數
        grt_params = ['db1', 'db2', 'sp1', 'sp2']
        available_params = []
        param_samples = []
        
        # Extract available GRT parameters / 提取可用的 GRT 參數
        for param in grt_params:
            if param in posterior.data_vars:
                samples = posterior[param].values.flatten()
                param_samples.append(samples)
                available_params.append(param)
        
        if len(param_samples) < 2:
            return {'error': 'Insufficient GRT parameters for Sigma matrix computation'}
        
        # Compute matrices / 計算矩陣
        samples_matrix = np.column_stack(param_samples)
        covariance_matrix = np.cov(samples_matrix.T)
        correlation_matrix = np.corrcoef(samples_matrix.T)
        
        # Independence tests / 獨立性測試
        independence_tests = {}
        if len(available_params) >= 2:
            for i in range(len(available_params)):
                for j in range(i+1, len(available_params)):
                    param1, param2 = available_params[i], available_params[j]
                    correlation = correlation_matrix[i, j]
                    
                    independence_tests[f'{param1}_{param2}'] = {
                        'correlation': float(correlation),
                        'abs_correlation': float(abs(correlation)),
                        'independent': bool(abs(correlation) < 0.3),
                        'evidence_strength': (
                            'weak' if abs(correlation) < 0.3 else
                            'moderate' if abs(correlation) < 0.6 else 'strong'
                        )
                    }
        
        # Separability test (sp1 vs sp2) / 可分離性測試 (sp1 vs sp2)
        separability_test = None
        if 'sp1' in available_params and 'sp2' in available_params:
            sp1_samples = posterior['sp1'].values.flatten()
            sp2_samples = posterior['sp2'].values.flatten()
            sp_ratio = sp1_samples / sp2_samples
            ratio_hdi = np.percentile(sp_ratio, [2.5, 97.5])
            
            separability_test = {
                'sp1_sp2_ratio_mean': float(np.mean(sp_ratio)),
                'sp1_sp2_ratio_hdi': [float(ratio_hdi[0]), float(ratio_hdi[1])],
                'separability_supported': bool(ratio_hdi[0] < 1.0 < ratio_hdi[1])
            }
        
        return {
            'parameter_names': available_params,
            'covariance_matrix': covariance_matrix.tolist(),
            'correlation_matrix': correlation_matrix.tolist(),
            'independence_tests': independence_tests,
            'separability_test': separability_test,
            'n_parameters': len(available_params)
        }
    
    def summarize_convergence(self, trace: az.InferenceData) -> Dict[str, Any]:
        """
        Summarize convergence diagnostics
        總結收斂診斷
        """
        
        try:
            ess = az.ess(trace)
            rhat = az.rhat(trace)
            
            # Overall statistics / 整體統計
            ess_values = [float(ess[param]) if ess[param].ndim == 0 else float(ess[param].min()) 
                         for param in ess.data_vars]
            rhat_values = [float(rhat[param]) if rhat[param].ndim == 0 else float(rhat[param].max()) 
                          for param in rhat.data_vars]
            
            return {
                'mean_ess': float(np.mean(ess_values)),
                'min_ess': float(np.min(ess_values)),
                'max_rhat': float(np.max(rhat_values)),
                'mean_rhat': float(np.mean(rhat_values)),
                'all_converged': bool(np.max(rhat_values) < 1.1),
                'n_parameters': len(rhat_values)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_subject_results(self, results: Dict[str, Any], subject_id: int):
        """
        Save individual subject results
        保存個別受試者結果
        """
        
        # Convert numpy types for JSON serialization
        # 為 JSON 序列化轉換 numpy 類型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Prepare results for saving (exclude traces for JSON)
        # 準備保存結果（為 JSON 排除軌跡）
        save_results = results.copy()
        traces = save_results.pop('stage_traces', {})
        
        # Save JSON results / 保存 JSON 結果
        results_file = self.results_dir / f'subject_{subject_id}_staged_results.json'
        converted_results = convert_numpy_types(save_results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        # Save traces separately / 分別保存軌跡
        for stage, trace in traces.items():
            trace_file = self.results_dir / f'subject_{subject_id}_stage_{stage}_trace.nc'
            trace.to_netcdf(trace_file)
        
        print(f"✅ Results saved for Subject {subject_id}")
        print(f"✅ 受試者 {subject_id} 的結果已保存")

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# 主要執行函數
# ============================================================================

def run_staged_analysis(max_subjects: Optional[int] = None, 
                       draws: int = 500, tune: int = 300, 
                       chains: int = 2) -> StagedGRTLBAAnalyzer:
    """
    Run complete staged GRT-LBA analysis
    運行完整的階段式 GRT-LBA 分析
    
    Parameters / 參數:
    max_subjects: Maximum number of subjects to analyze / 要分析的最大受試者數
    draws: MCMC draws per stage / 每階段 MCMC 抽樣數
    tune: MCMC tuning steps / MCMC 調整步數
    chains: Number of chains / 鏈數量
    
    Returns / 返回:
    StagedGRTLBAAnalyzer: Complete analyzer with results / 包含結果的完整分析器
    """
    
    print("="*80)
    print("STAGED GRT-LBA BAYESIAN ANALYSIS")
    print("階段式 GRT-LBA 貝葉斯分析")
    print("="*80)
    
    print(f"Analysis parameters / 分析參數:")
    print(f"  Draws per stage / 每階段抽樣數: {draws}")
    print(f"  Tuning steps / 調整步數: {tune}")
    print(f"  Chains / 鏈數: {chains}")
    print(f"  Max subjects / 最大受試者數: {max_subjects if max_subjects else 'All'}")
    
    # Initialize analyzer / 初始化分析器
    analyzer = StagedGRTLBAAnalyzer()
    
    # Select subjects to analyze / 選擇要分析的受試者
    subjects_to_analyze = analyzer.participants
    if max_subjects is not None:
        subjects_to_analyze = subjects_to_analyze[:max_subjects]
        print(f"Analyzing first {max_subjects} subjects for testing")
        print(f"為測試分析前 {max_subjects} 名受試者")
    
    # Run staged analysis for each subject / 對每名受試者運行階段式分析
    successful_analyses = 0
    failed_analyses = 0
    
    for subject_id in subjects_to_analyze:
        try:
            print(f"\n{'='*20} Processing Subject {subject_id} {'='*20}")
            print(f"\n{'='*20} 處理受試者 {subject_id} {'='*20}")
            
            result = analyzer.analyze_single_subject_staged(
                subject_id, draws=draws, tune=tune, chains=chains)
            
            if result is not None:
                analyzer.final_results[subject_id] = result
                successful_analyses += 1
                print(f"✅ Subject {subject_id} analysis completed successfully")
                print(f"✅ 受試者 {subject_id} 分析成功完成")
            else:
                failed_analyses += 1
                print(f"❌ Subject {subject_id} analysis failed")
                print(f"❌ 受試者 {subject_id} 分析失敗")
                
        except Exception as e:
            failed_analyses += 1
            print(f"❌ Subject {subject_id} analysis failed with error: {e}")
            print(f"❌ 受試者 {subject_id}
