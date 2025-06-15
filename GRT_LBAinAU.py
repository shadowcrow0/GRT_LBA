# -*- coding: utf-8 -*-
"""
Four-Choice Staged GRT-LBA Analysis
四選項階段式 GRT-LBA 分析

Purpose / 目的:
- Preserve complete 4-choice structure without simplification
- 保持完整的4選項結構，不進行簡化
- Implement staged Bayesian modeling for better convergence  
- 實施階段式貝葉斯建模以獲得更好的收斂性
- Map 2D GRT perceptual space to 4 LBA accumulators
- 將2D GRT感知空間映射到4個LBA累加器

Theoretical Framework / 理論框架:
- Left line: Chanel1 (0=left-tilted, 1=right-tilted)
- 左線條：Chanel1 (0=左傾, 1=右傾)
- Right line: Chanel2 (0=left-tilted, 1=right-tilted)  
- 右線條：Chanel2 (0=左傾, 1=右傾)

Four Response Options / 四個響應選項:
- Response 0: Both lines left-tilted (Chanel1=0, Chanel2=0)
- Response 1: Left left-tilted, Right right-tilted (Chanel1=0, Chanel2=1)
- Response 2: Left right-tilted, Right left-tilted (Chanel1=1, Chanel2=0)
- Response 3: Both lines right-tilted (Chanel1=1, Chanel2=1)
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
# FOUR-CHOICE DATA PREPROCESSING
# 四選項數據預處理
# ============================================================================

class FourChoiceDataPreprocessor:
    """
    Four-choice data preprocessing without simplification
    四選項數據預處理，不進行簡化
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        self.csv_file = csv_file
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load data maintaining full 4-choice structure
        載入數據並保持完整的4選項結構
        """
        print("Loading four-choice GRT-LBA data...")
        print("載入四選項 GRT-LBA 數據...")
        
        df = pd.read_csv(self.csv_file)
        print(f"Original data: {len(df)} trials / 原始數據：{len(df)} 次試驗")
        
        # Standard RT filtering
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.0)]
        print(f"After RT filtering: {len(df)} trials / RT過濾後：{len(df)} 次試驗")
        
        # KEEP original 4-choice responses (NO SIMPLIFICATION)
        # 保持原始4選項響應（不進行簡化）
        df['choice_four'] = df['Response'].astype(int)
        
        # Verify response distribution
        response_counts = df['choice_four'].value_counts().sort_index()
        print("\n=== Four-Choice Response Distribution ===")
        print("=== 四選項響應分佈 ===")
        for resp, count in response_counts.items():
            percentage = count / len(df) * 100
            print(f"Response {resp}: {count} trials ({percentage:.1f}%)")
        
        # Create perceptual space coordinates based on Chanel1 and Chanel2
        # 基於Chanel1和Chanel2創建感知空間坐標
        df['perc_dim1'] = df['Chanel1'].astype(float)  # Left line orientation
        df['perc_dim2'] = df['Chanel2'].astype(float)  # Right line orientation
        
        # Map stimulus conditions to 2D perceptual space
        # 將刺激條件映射到2D感知空間
        stim_mapping = {
            0: [0.0, 0.0],  # Both left-tilted
            1: [0.0, 1.0],  # Left left-tilted, Right right-tilted  
            2: [1.0, 0.0],  # Left right-tilted, Right left-tilted
            3: [1.0, 1.0]   # Both right-tilted
        }
        
        df['stimloc_x'] = df['Stimulus'].map(lambda x: stim_mapping[x][0])
        df['stimloc_y'] = df['Stimulus'].map(lambda x: stim_mapping[x][1])
        
        self.df = df
        self.participants = sorted(df['participant'].unique())
        self.n_participants = len(self.participants)
        
        print(f"\nParticipants: {self.participants}")
        print(f"參與者：{self.participants}")
        print(f"Perceptual space: 2D grid with 4 stimulus locations")
        print(f"感知空間：具有4個刺激位置的2D網格")

# ============================================================================
# STAGE 1: SIMPLIFIED FOUR-CHOICE GRT-LBA
# 階段1：簡化的四選項 GRT-LBA
# ============================================================================

def stage1_simplified_four_choice_grt_lba(rt_data: np.ndarray, choice_data: np.ndarray, 
                                         stimloc_data: np.ndarray) -> pm.Model:
    """
    Stage 1: Simplified Four-Choice GRT-LBA Model
    階段1：簡化的四選項 GRT-LBA 模型
    
    Simplifications for initial convergence:
    初始收斂的簡化：
    - Shared perceptual variability: sp1 = sp2 = sp
    - 共享感知變異性：sp1 = sp2 = sp
    - Shared threshold offset: b1 = b2 = b3 = b4 = A + b
    - 共享閾值偏移：b1 = b2 = b3 = b4 = A + b
    - Fixed drift rate variability and non-decision time
    - 固定漂移率變異性和非決策時間
    
    Parameters (6 total):
    參數（總共6個）：
    - db1, db2: Decision boundaries
    - sp: Shared perceptual variability  
    - A: Start point variability
    - b: Shared threshold offset
    - base_v: Base drift rate
    """
    
    print("Stage 1: Building simplified four-choice GRT-LBA model...")
    print("階段1：建立簡化的四選項 GRT-LBA 模型...")
    print("Parameters: db1, db2, sp (shared), A, b (shared), base_v")
    
    with pm.Model() as model:
        # ====================================================================
        # GRT Parameters
        # ====================================================================
        
        # Decision boundaries for 2D perceptual space
        db1 = pm.Normal('db1', mu=0.5, sigma=0.3,
                       doc="Decision boundary dimension 1 (left line)")
        db2 = pm.Normal('db2', mu=0.5, sigma=0.3,
                       doc="Decision boundary dimension 2 (right line)")
        
        # Shared perceptual variability (simplified)
        sp_shared = pm.HalfNormal('sp', sigma=0.2,
                                 doc="Shared perceptual variability")
        sp1 = pm.Deterministic('sp1', sp_shared)
        sp2 = pm.Deterministic('sp2', sp_shared)
        
        # ====================================================================
        # LBA Parameters (4 accumulators)
        # ====================================================================
        
        # Start point variability
        A = pm.HalfNormal('A', sigma=0.3,
                         doc="LBA start point variability")
        
        # Shared threshold offset (simplified)
        b_shared = pm.HalfNormal('b', sigma=0.4,
                                doc="Shared threshold offset")
        
        # Four decision thresholds (initially constrained to be equal)
        b1 = pm.Deterministic('b1', A + b_shared)
        b2 = pm.Deterministic('b2', A + b_shared)
        b3 = pm.Deterministic('b3', A + b_shared)
        b4 = pm.Deterministic('b4', A + b_shared)
        
        # Base drift rate for scaling
        base_v = pm.HalfNormal('base_v', sigma=0.5,
                              doc="Base drift rate scaling factor")
        
        # Fixed parameters for stability
        s_fixed = 0.3    # Drift rate variability
        t0_fixed = 0.25  # Non-decision time
        
        # ====================================================================
        # Four-Choice GRT→LBA Transformation
        # ====================================================================
        
        pm.Potential('four_choice_grt_lba_likelihood',
                    four_choice_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, b3, b4, 
                        base_v, s_fixed, t0_fixed))
    
    return model

# ============================================================================
# FOUR-CHOICE LBA LIKELIHOOD FUNCTION
# 四選項 LBA 似然函數
# ============================================================================

def four_choice_vectorized_lba_loglik(rt_data: np.ndarray, choice_data: np.ndarray, 
                                     stimloc: np.ndarray, db1, db2, sp1, sp2, 
                                     A, b1, b2, b3, b4, base_v, s, t0):
    """
    Four-Choice Vectorized LBA Log-Likelihood
    四選項向量化 LBA 對數似然函數
    
    Core Innovation: Map 2D GRT perceptual space to 4 LBA accumulators
    核心創新：將2D GRT感知空間映射到4個LBA累加器
    
    Mapping Strategy:
    映射策略：
    - v1: Evidence for Response 0 (both lines left-tilted)
    - v2: Evidence for Response 1 (left left, right right)
    - v3: Evidence for Response 2 (left right, right left)
    - v4: Evidence for Response 3 (both lines right-tilted)
    """
    
    # Enhanced numerical stability
    A = pt.maximum(A, 0.1)
    b1 = pt.maximum(b1, A + 0.15)
    b2 = pt.maximum(b2, A + 0.15)
    b3 = pt.maximum(b3, A + 0.15)
    b4 = pt.maximum(b4, A + 0.15)
    s = pt.maximum(s, 0.15)
    t0 = pt.maximum(t0, 0.05)
    sp1 = pt.maximum(sp1, 0.05)
    sp2 = pt.maximum(sp2, 0.05)
    base_v = pt.maximum(base_v, 0.1)
    
    # Decision time calculation
    rt_decision = pt.maximum(rt_data - t0, 0.05)
    
    # ====================================================================
    # GRT→Four-Choice LBA Transformation
    # ====================================================================
    
    # Compute evidence for each perceptual dimension
    # Left line probability (dimension 1)
    p_left_left = 0.5 * (1 - pt.tanh((stimloc[:, 0] - db1) / (2 * sp1)))
    p_left_right = 1 - p_left_left
    
    # Right line probability (dimension 2)  
    p_right_left = 0.5 * (1 - pt.tanh((stimloc[:, 1] - db2) / (2 * sp2)))
    p_right_right = 1 - p_right_left
    
    # Map to four response options based on line orientations
    # Response 0: Both left-tilted
    v1_raw = p_left_left * p_right_left
    
    # Response 1: Left left-tilted, Right right-tilted
    v2_raw = p_left_left * p_right_right
    
    # Response 2: Left right-tilted, Right left-tilted  
    v3_raw = p_left_right * p_right_left
    
    # Response 3: Both right-tilted
    v4_raw = p_left_right * p_right_right
    
    # Ensure meaningful drift rates
    v1_raw = pt.maximum(v1_raw, 0.05)
    v2_raw = pt.maximum(v2_raw, 0.05)
    v3_raw = pt.maximum(v3_raw, 0.05)
    v4_raw = pt.maximum(v4_raw, 0.05)
    
    # Normalize to ensure they sum to 1
    v_sum = v1_raw + v2_raw + v3_raw + v4_raw
    v1 = (v1_raw / v_sum) * base_v
    v2 = (v2_raw / v_sum) * base_v
    v3 = (v3_raw / v_sum) * base_v
    v4 = (v4_raw / v_sum) * base_v
    
    # ====================================================================
    # Four-Choice LBA Likelihood Computation
    # ====================================================================
    
    # Stack all drift rates and thresholds
    v_all = pt.stack([v1, v2, v3, v4], axis=1)  # Shape: (n_trials, 4)
    b_all = pt.stack([b1, b2, b3, b4], axis=0)  # Shape: (4,)
    
    # Expand for broadcasting
    rt_decision_expanded = pt.expand_dims(rt_decision, axis=1)  # Shape: (n_trials, 1)
    sqrt_t_expanded = pt.sqrt(rt_decision_expanded)
    
    # Compute winner likelihood for chosen accumulator
    choice_indices = choice_data.astype('int32')
    
    # Winner accumulator parameters
    v_winner = v_all[pt.arange(v_all.shape[0]), choice_indices]
    b_winner = b_all[choice_indices]
    
    # Winner PDF calculation
    z1_win = (v_winner * rt_decision - b_winner) / pt.sqrt(rt_decision)
    z2_win = (v_winner * rt_decision - A) / pt.sqrt(rt_decision)
    z1_win = pt.clip(z1_win, -8, 8)
    z2_win = pt.clip(z2_win, -8, 8)
    
    Phi_z1_win = 0.5 * (1 + pt.erf(z1_win / pt.sqrt(2)))
    Phi_z2_win = 0.5 * (1 + pt.erf(z2_win / pt.sqrt(2)))
    Phi_z1_win = pt.clip(Phi_z1_win, 1e-8, 1 - 1e-8)
    Phi_z2_win = pt.clip(Phi_z2_win, 1e-8, 1 - 1e-8)
    
    phi_z1_win = pt.exp(-0.5 * z1_win**2) / pt.sqrt(2 * np.pi)
    phi_z2_win = pt.exp(-0.5 * z2_win**2) / pt.sqrt(2 * np.pi)
    
    cdf_diff = pt.maximum(Phi_z1_win - Phi_z2_win, 1e-10)
    pdf_diff = (phi_z1_win - phi_z2_win) / pt.sqrt(rt_decision)
    
    winner_pdf = (v_winner / A) * cdf_diff + pdf_diff / A
    winner_pdf = pt.maximum(winner_pdf, 1e-10)
    winner_logpdf = pt.log(winner_pdf)
    
    # Loser survival functions for all non-chosen accumulators
    loser_log_survivals = []
    
    for acc_idx in range(4):
        # Skip if this is the chosen accumulator
        is_chosen = pt.eq(choice_indices, acc_idx)
        
        v_loser = v_all[:, acc_idx]
        b_loser = b_all[acc_idx]
        
        z1_lose = (v_loser * rt_decision - b_loser) / pt.sqrt(rt_decision)
        z2_lose = (v_loser * rt_decision - A) / pt.sqrt(rt_decision)
        z1_lose = pt.clip(z1_lose, -8, 8)
        z2_lose = pt.clip(z2_lose, -8, 8)
        
        Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
        Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
        Phi_z1_lose = pt.clip(Phi_z1_lose, 1e-8, 1 - 1e-8)
        Phi_z2_lose = pt.clip(Phi_z2_lose, 1e-8, 1 - 1e-8)
        
        loser_cdf = pt.maximum(Phi_z1_lose - Phi_z2_lose, 1e-10)
        loser_survival = pt.maximum(1 - loser_cdf, 1e-10)
        loser_log_survival = pt.log(loser_survival)
        
        # Only include if this accumulator is NOT chosen
        masked_log_survival = pt.where(is_chosen, 0.0, loser_log_survival)
        loser_log_survivals.append(masked_log_survival)
    
    # Sum all loser log-survivals
    total_loser_log_survival = pt.sum(pt.stack(loser_log_survivals, axis=1), axis=1)
    
    # Combine winner PDF and loser survivals
    trial_loglik = winner_logpdf + total_loser_log_survival
    trial_loglik = pt.where(pt.isnan(trial_loglik), -1000.0, trial_loglik)
    trial_loglik = pt.where(pt.isinf(trial_loglik), -1000.0, trial_loglik)
    trial_loglik = pt.maximum(trial_loglik, -1000.0)
    
    return pt.sum(trial_loglik)

# ============================================================================
# SUBSEQUENT STAGES (Framework)
# 後續階段（框架）
# ============================================================================

def stage2_separate_perceptual_variabilities_four_choice(rt_data, choice_data, stimloc_data, stage1_trace):
    """
    Stage 2: Separate perceptual variabilities (sp1 ≠ sp2)
    階段2：分離感知變異性 (sp1 ≠ sp2)
    """
    print("Stage 2: Four-choice model with separate perceptual variabilities...")
    stage1_summary = az.summary(stage1_trace)
    
    with pm.Model() as model:
        # Use Stage 1 results as informed priors
        db1_mean = float(stage1_summary.loc['db1', 'mean'])
        db1 = pm.Normal('db1', mu=db1_mean, sigma=0.1)
        
        db2_mean = float(stage1_summary.loc['db2', 'mean'])
        db2 = pm.Normal('db2', mu=db2_mean, sigma=0.1)
        
        # NEW: Separate perceptual variabilities
        sp_prior = float(stage1_summary.loc['sp', 'mean'])
        sp1 = pm.HalfNormal('sp1', sigma=sp_prior * 0.5)
        sp2 = pm.HalfNormal('sp2', sigma=sp_prior * 0.5)
        
        # Other parameters with informed priors
        A_mean = float(stage1_summary.loc['A', 'mean'])
        A = pm.Normal('A', mu=A_mean, sigma=0.05)
        
        b_mean = float(stage1_summary.loc['b', 'mean'])
        b_shared = pm.Normal('b', mu=b_mean, sigma=0.05)
        
        b1 = pm.Deterministic('b1', A + b_shared)
        b2 = pm.Deterministic('b2', A + b_shared)
        b3 = pm.Deterministic('b3', A + b_shared)
        b4 = pm.Deterministic('b4', A + b_shared)
        
        base_v_mean = float(stage1_summary.loc['base_v', 'mean'])
        base_v = pm.Normal('base_v', mu=base_v_mean, sigma=0.05)
        
        s_fixed = 0.3
        t0_fixed = 0.25
        
        pm.Potential('four_choice_grt_lba_likelihood',
                    four_choice_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, b3, b4, 
                        base_v, s_fixed, t0_fixed))
    
    return model

def stage3_separate_lba_thresholds_four_choice(rt_data, choice_data, stimloc_data, stage2_trace):
    """
    Stage 3: Separate LBA thresholds for all four accumulators
    階段3：為所有四個累加器分離LBA閾值
    """
    print("Stage 3: Four-choice model with separate thresholds...")
    stage2_summary = az.summary(stage2_trace)
    
    with pm.Model() as model:
        # GRT parameters with tight priors
        for param in ['db1', 'db2', 'sp1', 'sp2', 'A', 'base_v']:
            param_mean = float(stage2_summary.loc[param, 'mean'])
            if param.startswith('sp') or param in ['A', 'base_v']:
                globals()[param] = pm.Normal(param, mu=param_mean, sigma=0.05)
            else:
                globals()[param] = pm.Normal(param, mu=param_mean, sigma=0.05)
        
        # NEW: Separate threshold offsets for all four accumulators
        b_prior = float(stage2_summary.loc['b', 'mean'])
        bMa1 = pm.HalfNormal('bMa1', sigma=b_prior * 0.5)
        bMa2 = pm.HalfNormal('bMa2', sigma=b_prior * 0.5)
        bMa3 = pm.HalfNormal('bMa3', sigma=b_prior * 0.5)
        bMa4 = pm.HalfNormal('bMa4', sigma=b_prior * 0.5)
        
        b1 = pm.Deterministic('b1', A + bMa1)
        b2 = pm.Deterministic('b2', A + bMa2)
        b3 = pm.Deterministic('b3', A + bMa3)
        b4 = pm.Deterministic('b4', A + bMa4)
        
        s_fixed = 0.3
        t0_fixed = 0.25
        
        pm.Potential('four_choice_grt_lba_likelihood',
                    four_choice_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, b3, b4, 
                        base_v, s_fixed, t0_fixed))
    
    return model

def stage4_full_four_choice_grt_lba(rt_data, choice_data, stimloc_data, stage3_trace):
    """
    Stage 4: Full four-choice GRT-LBA model with all parameters estimated
    階段4：估計所有參數的完整四選項GRT-LBA模型
    """
    print("Stage 4: Full four-choice GRT-LBA model...")
    stage3_summary = az.summary(stage3_trace)
    
    with pm.Model() as model:
        # All GRT and LBA parameters with refined priors
        for param in ['db1', 'db2', 'sp1', 'sp2', 'A', 'bMa1', 'bMa2', 'bMa3', 'bMa4', 'base_v']:
            param_mean = float(stage3_summary.loc[param, 'mean'])
            if param.startswith('sp') or param in ['A', 'base_v'] or param.startswith('bMa'):
                globals()[param] = pm.Normal(param, mu=param_mean, sigma=0.05)
            else:
                globals()[param] = pm.Normal(param, mu=param_mean, sigma=0.05)
        
        b1 = pm.Deterministic('b1', A + bMa1)
        b2 = pm.Deterministic('b2', A + bMa2)
        b3 = pm.Deterministic('b3', A + bMa3)
        b4 = pm.Deterministic('b4', A + bMa4)
        
        # NEW: Estimate drift rate variability and non-decision time
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        pm.Potential('four_choice_grt_lba_likelihood',
                    four_choice_vectorized_lba_loglik(
                        rt_data, choice_data, stimloc_data,
                        db1, db2, sp1, sp2, A, b1, b2, b3, b4, 
                        base_v, s, t0))
    
    return model

# ============================================================================
# FOUR-CHOICE STAGED ANALYZER CLASS
# 四選項階段式分析器類別
# ============================================================================

class FourChoiceStagedGRTLBAAnalyzer:
    """
    Four-Choice Staged GRT-LBA Analyzer
    四選項階段式 GRT-LBA 分析器
    
    Maintains complete 4-choice structure throughout all analysis stages
    在所有分析階段中保持完整的4選項結構
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        self.csv_file = csv_file
        self.results_dir = Path('four_choice_staged_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Use four-choice preprocessor
        self.preprocessor = FourChoiceDataPreprocessor(csv_file)
        self.df = self.preprocessor.df
        self.participants = self.preprocessor.participants
        self.n_participants = self.preprocessor.n_participants
        
        self.stage_results = {}
        self.stage_traces = {}
        self.final_results = {}
        
        print("Four-Choice Staged GRT-LBA Analyzer initialized")
        print("四選項階段式 GRT-LBA 分析器已初始化")
        print(f"Data: {len(self.df)} trials, {self.n_participants} participants")
        print(f"Response distribution maintained: 4 choices (0,1,2,3)")
    
    def analyze_single_subject_four_choice_staged(self, subject_id: int, 
                                                 draws: int = 500, tune: int = 300, 
                                                 chains: int = 2) -> Dict[str, Any]:
        """
        Perform staged four-choice analysis for a single subject
        對單個受試者進行階段式四選項分析
        """
        
        print(f"\n{'='*60}")
        print(f"FOUR-CHOICE STAGED ANALYSIS FOR SUBJECT {subject_id}")
        print(f"受試者 {subject_id} 的四選項階段式分析")
        print(f"{'='*60}")
        
        # Prepare subject data
        subject_data = self.df[self.df['participant'] == subject_id].copy()
        
        if len(subject_data) < 20:
            print(f"⚠️  Warning: Subject {subject_id} has only {len(subject_data)} trials")
            return None
        
        # Data arrays for four-choice model
        rt_data = subject_data['RT'].values.astype(np.float32)
        choice_data = subject_data['choice_four'].values.astype(np.int32)  # Keep 4 choices!
        stimloc_data = np.column_stack([
            subject_data['stimloc_x'].values,
            subject_data['stimloc_y'].values
        ]).astype(np.float32)
        
        print(f"Data prepared: {len(rt_data)} trials")
        print(f"Four-choice distribution: {np.bincount(choice_data, minlength=4)}")
        
        # Initialize stage tracking
        stage_traces = {}
        stage_results = {}
        successful_stage = 0
        
        # ================================================================
        # STAGE 1: SIMPLIFIED FOUR-CHOICE GRT-LBA
        # ================================================================
        
        print(f"\n{'-'*40}")
        print("STAGE 1: Simplified Four-Choice GRT-LBA")
        print("階段1：簡化的四選項 GRT-LBA")
        print(f"{'-'*40}")
        
        try:
            model1 = stage1_simplified_four_choice_grt_lba(rt_data, choice_data, stimloc_data)
            
            print("Starting MCMC sampling for 4-choice model...")
            
            with model1:
                trace1 = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.95, max_treedepth=12,
                    cores=1, random_seed=123 + subject_id
                )
            
            rhat_max = float(az.rhat(trace1).max())
            ess_min = float(az.ess(trace1).min())
            
            print(f"Stage 1 Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            
            if rhat_max < 1.15:
                print("✅ Stage 1 SUCCESSFUL")
                stage_traces[1] = trace1
                stage_results[1] = self.analyze_stage_results(trace1, 1, subject_id)
                successful_stage = 1
            else:
                print("❌ Stage 1 FAILED - Poor convergence")
                return None
                
        except Exception as e:
            print(f"❌ Stage 1 FAILED - Sampling error: {e}")
            return None
        
        # ================================================================
        # STAGE 2: SEPARATE PERCEPTUAL VARIABILITIES
        # ================================================================
        
        print(f"\n{'-'*40}")
        print("STAGE 2: Separate Perceptual Variabilities")
        print("階段2：分離感知變異性")
        print(f"{'-'*40}")
        
        try:
            model2 = stage2_separate_perceptual_variabilities_four_choice(
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
            
            if rhat_max < 1.15:
                print("✅ Stage 2 SUCCESSFUL")
                stage_traces[2] = trace2
                stage_results[2] = self.analyze_stage_results(trace2, 2, subject_id)
                successful_stage = 2
            else:
                print("⚠️  Stage 2 FAILED - Using Stage 1 results")
                
        except Exception as e:
            print(f"⚠️  Stage 2 FAILED - Sampling error: {e}")
        
        # ================================================================
        # STAGE 3: SEPARATE LBA THRESHOLDS
        # ================================================================
        
        if successful_stage >= 2:
            print(f"\n{'-'*40}")
            print("STAGE 3: Separate Four-Choice LBA Thresholds")
            print("階段3：分離四選項 LBA 閾值")
            print(f"{'-'*40}")
            
            try:
                model3 = stage3_separate_lba_thresholds_four_choice(
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
                
                if rhat_max < 1.15:
                    print("✅ Stage 3 SUCCESSFUL")
                    stage_traces[3] = trace3
                    stage_results[3] = self.analyze_stage_results(trace3, 3, subject_id)
                    successful_stage = 3
                else:
                    print("⚠️  Stage 3 FAILED - Using Stage 2 results")
                    
            except Exception as e:
                print(f"⚠️  Stage 3 FAILED - Sampling error: {e}")
        
        # ================================================================
        # STAGE 4: FULL FOUR-CHOICE GRT-LBA MODEL
        # ================================================================
        
        if successful_stage >= 3:
            print(f"\n{'-'*40}")
            print("STAGE 4: Full Four-Choice GRT-LBA Model")
            print("階段4：完整四選項 GRT-LBA 模型")
            print(f"{'-'*40}")
            
            try:
                model4 = stage4_full_four_choice_grt_lba(
                    rt_data, choice_data, stimloc_data, trace3)
                
                with model4:
                    trace4 = pm.sample(
                        draws=draws + 200, tune=tune + 100, chains=chains,
                        progressbar=True, return_inferencedata=True,
                        target_accept=0.97, max_treedepth=15,
                        cores=1, random_seed=123 + subject_id
                    )
                
                rhat_max = float(az.rhat(trace4).max())
                ess_min = float(az.ess(trace4).min())
                
                print(f"Stage 4 Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
                
                if rhat_max < 1.1:
                    print("✅ Stage 4 SUCCESSFUL - Full four-choice model converged!")
                    stage_traces[4] = trace4
                    stage_results[4] = self.analyze_stage_results(trace4, 4, subject_id)
                    successful_stage = 4
                else:
                    print("⚠️  Stage 4 FAILED - Using Stage 3 results")
                    
            except Exception as e:
                print(f"⚠️  Stage 4 FAILED - Sampling error: {e}")
        
        # ================================================================
        # COMPILE FINAL RESULTS
        # ================================================================
        
        print(f"\n{'='*40}")
        print(f"FINAL FOUR-CHOICE RESULTS FOR SUBJECT {subject_id}")
        print(f"受試者 {subject_id} 的最終四選項結果")
        print(f"{'='*40}")
        print(f"Successful stages: {list(stage_results.keys())}")
        print(f"Using results from Stage {successful_stage}")
        
        # Compute four-choice specific analyses
        final_result = {
            'subject_id': subject_id,
            'model_type': 'four_choice_grt_lba',
            'successful_stages': list(stage_results.keys()),
            'final_stage_used': successful_stage,
            'stage_traces': stage_traces,
            'stage_results': stage_results,
            'sigma_matrix': self.compute_four_choice_sigma_matrix(stage_traces[successful_stage]),
            'convergence_summary': self.summarize_convergence(stage_traces[successful_stage]),
            'four_choice_analysis': self.analyze_four_choice_specific_metrics(stage_traces[successful_stage], choice_data)
        }
        
        # Save results
        self.save_subject_results(final_result, subject_id)
        
        return final_result
    
    def compute_four_choice_sigma_matrix(self, trace: az.InferenceData) -> Dict[str, Any]:
        """
        Compute enhanced Sigma matrix for four-choice GRT model
        計算四選項 GRT 模型的增強 Sigma 矩陣
        """
        
        posterior = trace.posterior
        
        # GRT parameters
        grt_params = ['db1', 'db2', 'sp1', 'sp2']
        # LBA threshold parameters (four accumulators)
        lba_params = ['bMa1', 'bMa2', 'bMa3', 'bMa4'] if 'bMa1' in posterior.data_vars else []
        
        all_params = []
        param_samples = []
        
        # Extract all available parameters
        for param_list in [grt_params, lba_params]:
            for param in param_list:
                if param in posterior.data_vars:
                    samples = posterior[param].values.flatten()
                    param_samples.append(samples)
                    all_params.append(param)
        
        if len(param_samples) < 2:
            return {'error': 'Insufficient parameters for Sigma matrix computation'}
        
        # Compute matrices
        samples_matrix = np.column_stack(param_samples)
        covariance_matrix = np.cov(samples_matrix.T)
        correlation_matrix = np.corrcoef(samples_matrix.T)
        
        # Enhanced independence tests for four-choice model
        independence_tests = {}
        separability_tests = {}
        
        # GRT independence tests
        grt_available = [p for p in grt_params if p in all_params]
        if len(grt_available) >= 2:
            for i in range(len(grt_available)):
                for j in range(i+1, len(grt_available)):
                    param1, param2 = grt_available[i], grt_available[j]
                    param1_idx = all_params.index(param1)
                    param2_idx = all_params.index(param2)
                    correlation = correlation_matrix[param1_idx, param2_idx]
                    
                    independence_tests[f'grt_{param1}_{param2}'] = {
                        'correlation': float(correlation),
                        'abs_correlation': float(abs(correlation)),
                        'independent': bool(abs(correlation) < 0.3),
                        'evidence_strength': (
                            'weak' if abs(correlation) < 0.3 else
                            'moderate' if abs(correlation) < 0.6 else 'strong'
                        )
                    }
        
        # LBA threshold independence tests (four accumulators)
        lba_available = [p for p in lba_params if p in all_params]
        if len(lba_available) >= 2:
            for i in range(len(lba_available)):
                for j in range(i+1, len(lba_available)):
                    param1, param2 = lba_available[i], lba_available[j]
                    param1_idx = all_params.index(param1)
                    param2_idx = all_params.index(param2)
                    correlation = correlation_matrix[param1_idx, param2_idx]
                    
                    independence_tests[f'lba_{param1}_{param2}'] = {
                        'correlation': float(correlation),
                        'abs_correlation': float(abs(correlation)),
                        'independent': bool(abs(correlation) < 0.3),
                        'evidence_strength': (
                            'weak' if abs(correlation) < 0.3 else
                            'moderate' if abs(correlation) < 0.6 else 'strong'
                        )
                    }
        
        # Four-choice separability tests
        if 'sp1' in all_params and 'sp2' in all_params:
            sp1_samples = posterior['sp1'].values.flatten()
            sp2_samples = posterior['sp2'].values.flatten()
            sp_ratio = sp1_samples / sp2_samples
            ratio_hdi = np.percentile(sp_ratio, [2.5, 97.5])
            
            separability_tests['perceptual_separability'] = {
                'sp1_sp2_ratio_mean': float(np.mean(sp_ratio)),
                'sp1_sp2_ratio_hdi': [float(ratio_hdi[0]), float(ratio_hdi[1])],
                'separability_supported': bool(ratio_hdi[0] < 1.0 < ratio_hdi[1])
            }
        
        # Four-accumulator threshold separability
        if len(lba_available) == 4:
            threshold_ratios = {}
            for i in range(len(lba_available)):
                for j in range(i+1, len(lba_available)):
                    param1, param2 = lba_available[i], lba_available[j]
                    samples1 = posterior[param1].values.flatten()
                    samples2 = posterior[param2].values.flatten()
                    ratio = samples1 / samples2
                    ratio_hdi = np.percentile(ratio, [2.5, 97.5])
                    
                    threshold_ratios[f'{param1}_{param2}_ratio'] = {
                        'mean': float(np.mean(ratio)),
                        'hdi': [float(ratio_hdi[0]), float(ratio_hdi[1])],
                        'different_thresholds': bool(ratio_hdi[0] > 1.1 or ratio_hdi[1] < 0.9)
                    }
            
            separability_tests['threshold_separability'] = threshold_ratios
        
        return {
            'parameter_names': all_params,
            'covariance_matrix': covariance_matrix.tolist(),
            'correlation_matrix': correlation_matrix.tolist(),
            'independence_tests': independence_tests,
            'separability_tests': separability_tests,
            'n_parameters': len(all_params),
            'model_type': 'four_choice_grt_lba'
        }
    
    def analyze_four_choice_specific_metrics(self, trace: az.InferenceData, choice_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze metrics specific to four-choice model
        分析四選項模型特定的指標
        """
        
        posterior = trace.posterior
        
        # Choice frequency analysis
        choice_counts = np.bincount(choice_data, minlength=4)
        choice_proportions = choice_counts / len(choice_data)
        
        # Extract drift rate scaling if available
        drift_rate_analysis = {}
        if 'base_v' in posterior.data_vars:
            base_v_samples = posterior['base_v'].values.flatten()
            drift_rate_analysis = {
                'base_drift_rate_mean': float(np.mean(base_v_samples)),
                'base_drift_rate_std': float(np.std(base_v_samples)),
                'base_drift_rate_hdi': [float(np.percentile(base_v_samples, 2.5)),
                                       float(np.percentile(base_v_samples, 97.5))]
            }
        
        # Threshold analysis for four accumulators
        threshold_analysis = {}
        threshold_params = ['b1', 'b2', 'b3', 'b4']
        available_thresholds = []
        
        for param in threshold_params:
            if param in posterior.data_vars:
                samples = posterior[param].values.flatten()
                available_thresholds.append(samples)
                threshold_analysis[param] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'hdi': [float(np.percentile(samples, 2.5)),
                           float(np.percentile(samples, 97.5))]
                }
        
        # Threshold ordering analysis
        threshold_ordering = {}
        if len(available_thresholds) == 4:
            # Compute probability that thresholds are ordered
            threshold_matrix = np.column_stack(available_thresholds)
            
            for i in range(4):
                for j in range(i+1, 4):
                    prob_greater = np.mean(threshold_matrix[:, i] > threshold_matrix[:, j])
                    threshold_ordering[f'P(b{i+1} > b{j+1})'] = float(prob_greater)
        
        return {
            'choice_frequencies': choice_counts.tolist(),
            'choice_proportions': choice_proportions.tolist(),
            'drift_rate_analysis': drift_rate_analysis,
            'threshold_analysis': threshold_analysis,
            'threshold_ordering': threshold_ordering,
            'model_complexity': len([p for p in posterior.data_vars if not p.startswith('_')])
        }
    
    def analyze_stage_results(self, trace: az.InferenceData, stage: int, 
                            subject_id: int) -> Dict[str, Any]:
        """
        Analyze results from a specific stage (adapted for four-choice)
        分析特定階段的結果（適應四選項）
        """
        
        posterior = trace.posterior
        summary = az.summary(trace)
        
        results = {
            'stage': stage,
            'subject_id': subject_id,
            'model_type': 'four_choice_grt_lba',
            'parameter_estimates': {},
            'convergence_diagnostics': {}
        }
        
        # Parameter estimates
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
        
        # Convergence diagnostics
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
    
    def summarize_convergence(self, trace: az.InferenceData) -> Dict[str, Any]:
        """
        Summarize convergence diagnostics for four-choice model
        總結四選項模型的收斂診斷
        """
        
        try:
            ess = az.ess(trace)
            rhat = az.rhat(trace)
            
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
                'n_parameters': len(rhat_values),
                'model_type': 'four_choice_grt_lba'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_subject_results(self, results: Dict[str, Any], subject_id: int):
        """
        Save individual subject results for four-choice model
        保存四選項模型的個別受試者結果
        """
        
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
        
        # Prepare results for saving
        save_results = results.copy()
        traces = save_results.pop('stage_traces', {})
        
        # Save JSON results
        results_file = self.results_dir / f'subject_{subject_id}_four_choice_staged_results.json'
        converted_results = convert_numpy_types(save_results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        # Save traces separately
        for stage, trace in traces.items():
            trace_file = self.results_dir / f'subject_{subject_id}_four_choice_stage_{stage}_trace.nc'
            trace.to_netcdf(trace_file)
        
        print(f"✅ Four-choice results saved for Subject {subject_id}")

# ============================================================================
# MAIN EXECUTION FUNCTIONS FOR FOUR-CHOICE MODEL
# 四選項模型的主要執行函數
# ============================================================================

def run_four_choice_staged_analysis(max_subjects: Optional[int] = None, 
                                   draws: int = 500, tune: int = 300, 
                                   chains: int = 2) -> FourChoiceStagedGRTLBAAnalyzer:
    """
    Run complete four-choice staged GRT-LBA analysis
    運行完整的四選項階段式 GRT-LBA 分析
    
    Key Differences from Binary Model:
    與二元模型的主要差異：
    - Maintains full 4-choice response structure
    - 維持完整的4選項響應結構
    - Maps 2D GRT space to 4 LBA accumulators
    - 將2D GRT空間映射到4個LBA累加器
    - Enables complete choice pattern analysis
    - 能夠進行完整的選擇模式分析
    """
    
    print("="*80)
    print("FOUR-CHOICE STAGED GRT-LBA BAYESIAN ANALYSIS")
    print("四選項階段式 GRT-LBA 貝葉斯分析")
    print("="*80)
    
    print(f"Analysis parameters:")
    print(f"  Model type: Four-choice (no simplification)")
    print(f"  模型類型：四選項（無簡化）")
    print(f"  Draws per stage: {draws}")
    print(f"  Tuning steps: {tune}")
    print(f"  Chains: {chains}")
    print(f"  Max subjects: {max_subjects if max_subjects else 'All'}")
    
    # Initialize four-choice analyzer
    analyzer = FourChoiceStagedGRTLBAAnalyzer()
    
    # Select subjects to analyze
    subjects_to_analyze = analyzer.participants
    if max_subjects is not None:
        subjects_to_analyze = subjects_to_analyze[:max_subjects]
        print(f"Analyzing first {max_subjects} subjects for testing")
    
    # Run four-choice staged analysis for each subject
    successful_analyses = 0
    failed_analyses = 0
    
    for subject_id in subjects_to_analyze:
        try:
            print(f"\n{'='*20} Processing Subject {subject_id} (Four-Choice) {'='*20}")
            
            result = analyzer.analyze_single_subject_four_choice_staged(
                subject_id, draws=draws, tune=tune, chains=chains)
            
            if result is not None:
                analyzer.final_results[subject_id] = result
                successful_analyses += 1
                print(f"✅ Subject {subject_id} four-choice analysis completed")
            else:
                failed_analyses += 1
                print(f"❌ Subject {subject_id} four-choice analysis failed")
                
        except Exception as e:
            failed_analyses += 1
            print(f"❌ Subject {subject_id} analysis failed with error: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FOUR-CHOICE ANALYSIS SUMMARY")
    print(f"四選項分析總結")
    print(f"{'='*60}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Success rate: {successful_analyses/(successful_analyses+failed_analyses)*100:.1f}%")
    
    return analyzer

# ============================================================================
# EXAMPLE USAGE
# 使用範例
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of four-choice staged GRT-LBA analysis
    四選項階段式 GRT-LBA 分析的使用範例
    """
    
    print("Starting Four-Choice Staged GRT-LBA Analysis...")
    print("開始四選項階段式 GRT-LBA 分析...")
    
    # Run analysis with reduced parameters for testing
    analyzer = run_four_choice_staged_analysis(
        max_subjects=2,  # Test with first 2 subjects
        draws=300,       # Reduced for faster testing
        tune=200,        # Reduced for faster testing
        chains=2
    )
    
    print("\nAnalysis completed!")
    print("分析完成！")
    print(f"Results saved in: {analyzer.results_dir}")
    print(f"結果保存在：{analyzer.results_dir}")
