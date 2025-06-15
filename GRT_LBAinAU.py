# -*- coding: utf-8 -*-
"""
Optimized Four-Choice GRT-LBA Analysis with PyMC Features
使用 PyMC 特性優化的四選項 GRT-LBA 分析

Key Optimizations / 主要優化:
1. PyMC coordinates and dims for better memory management
2. Custom distributions for LBA likelihood
3. Vectorized operations with pytensor
4. Hierarchical priors with non-centered parameterization
5. Advanced sampling configurations
6. Automatic model selection and comparison
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pytensor.tensor.extra_ops import broadcast_arrays
from pytensor.compile.ops import as_op
import scipy.stats as stats
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM PYTENSOR OPERATIONS FOR LBA
# 自定義 Pytensor 操作用於 LBA
# ============================================================================

@as_op(itypes=[pt.dvector, pt.dvector, pt.dmatrix, pt.dscalar, pt.dscalar, 
               pt.dscalar, pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar], 
       otypes=[pt.dscalar])
def fast_lba_loglik(rt_data, choice_data, stimloc_data, db1, db2, sp1, sp2, 
                    A, thresholds, base_v, s, t0):
    """
    Optimized LBA log-likelihood computation using NumPy
    使用 NumPy 優化的 LBA 對數似然計算
    """
    try:
        # Ensure positive parameters
        A = max(A, 0.1)
        s = max(s, 0.15)
        t0 = max(t0, 0.05)
        sp1 = max(sp1, 0.05)
        sp2 = max(sp2, 0.05)
        base_v = max(base_v, 0.1)
        
        # Decision time
        rt_decision = np.maximum(rt_data - t0, 0.05)
        
        # GRT computations
        p_left_left = 0.5 * (1 - np.tanh((stimloc_data[:, 0] - db1) / (2 * sp1)))
        p_left_right = 1 - p_left_left
        p_right_left = 0.5 * (1 - np.tanh((stimloc_data[:, 1] - db2) / (2 * sp2)))
        p_right_right = 1 - p_right_left
        
        # Four-choice drift rates
        v1_raw = p_left_left * p_right_left
        v2_raw = p_left_left * p_right_right
        v3_raw = p_left_right * p_right_left
        v4_raw = p_left_right * p_right_right
        
        # Normalize
        v_sum = v1_raw + v2_raw + v3_raw + v4_raw + 1e-10
        v_all = np.column_stack([
            (v1_raw / v_sum) * base_v,
            (v2_raw / v_sum) * base_v,
            (v3_raw / v_sum) * base_v,
            (v4_raw / v_sum) * base_v
        ])
        
        # LBA likelihood
        choice_indices = choice_data.astype(int)
        loglik_sum = 0.0
        
        for i, (rt_trial, choice_idx) in enumerate(zip(rt_decision, choice_indices)):
            if choice_idx < 0 or choice_idx >= 4:
                continue
                
            # Winner accumulator
            v_win = v_all[i, choice_idx]
            b_win = thresholds[choice_idx]
            
            # Winner PDF
            z1_win = (v_win * rt_trial - b_win) / np.sqrt(rt_trial)
            z2_win = (v_win * rt_trial - A) / np.sqrt(rt_trial)
            z1_win = np.clip(z1_win, -8, 8)
            z2_win = np.clip(z2_win, -8, 8)
            
            cdf_diff = max(stats.norm.cdf(z1_win) - stats.norm.cdf(z2_win), 1e-10)
            pdf_diff = (stats.norm.pdf(z1_win) - stats.norm.pdf(z2_win)) / np.sqrt(rt_trial)
            
            winner_pdf = (v_win / A) * cdf_diff + pdf_diff / A
            winner_pdf = max(winner_pdf, 1e-10)
            
            # Loser survivals
            loser_survival_product = 1.0
            for acc_idx in range(4):
                if acc_idx == choice_idx:
                    continue
                    
                v_lose = v_all[i, acc_idx]
                b_lose = thresholds[acc_idx]
                
                z1_lose = (v_lose * rt_trial - b_lose) / np.sqrt(rt_trial)
                z2_lose = (v_lose * rt_trial - A) / np.sqrt(rt_trial)
                z1_lose = np.clip(z1_lose, -8, 8)
                z2_lose = np.clip(z2_lose, -8, 8)
                
                loser_cdf = max(stats.norm.cdf(z1_lose) - stats.norm.cdf(z2_lose), 1e-10)
                loser_survival = max(1 - loser_cdf, 1e-10)
                loser_survival_product *= loser_survival
            
            trial_lik = winner_pdf * loser_survival_product
            trial_loglik = np.log(max(trial_lik, 1e-15))
            
            if np.isfinite(trial_loglik):
                loglik_sum += trial_loglik
            else:
                loglik_sum += -15.0
        
        return loglik_sum
        
    except Exception:
        return -1000.0

# ============================================================================
# OPTIMIZED DATA PREPROCESSOR WITH PYMC COORDINATES
# 使用 PyMC 坐標系統的優化數據預處理器
# ============================================================================

class OptimizedFourChoiceDataPreprocessor:
    """
    Optimized data preprocessor with PyMC coordinate system
    使用 PyMC 坐標系統的優化數據預處理器
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        self.csv_file = csv_file
        self.coords = {}
        self.dims = {}
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare data with PyMC coordinates"""
        print("Loading optimized four-choice GRT-LBA data...")
        
        df = pd.read_csv(self.csv_file)
        
        # Enhanced data filtering
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.0)]
        df = df[df['Response'].isin([0, 1, 2, 3])]
        
        # Create PyMC coordinates
        self.participants = sorted(df['participant'].unique())
        self.n_participants = len(self.participants)
        
        # Set up coordinate system
        self.coords = {
            'participant': self.participants,
            'choice': [0, 1, 2, 3],
            'perceptual_dim': ['dim1', 'dim2'],
            'accumulator': ['acc1', 'acc2', 'acc3', 'acc4']
        }
        
        self.dims = {
            'participant_effect': ['participant'],
            'choice_probs': ['choice'],
            'perceptual_space': ['perceptual_dim'],
            'threshold_vector': ['accumulator']
        }
        
        # Prepare data structures
        df['choice_four'] = df['Response'].astype(int)
        df['perc_dim1'] = df['Chanel1'].astype(float)
        df['perc_dim2'] = df['Chanel2'].astype(float)
        
        # Create stimulus location mapping
        stim_mapping = {
            0: [0.0, 0.0], 1: [0.0, 1.0], 
            2: [1.0, 0.0], 3: [1.0, 1.0]
        }
        
        df['stimloc_x'] = df['Stimulus'].map(lambda x: stim_mapping[x][0])
        df['stimloc_y'] = df['Stimulus'].map(lambda x: stim_mapping[x][1])
        
        self.df = df
        
        print(f"Data loaded: {len(df)} trials, {self.n_participants} participants")
        print(f"PyMC coordinates configured: {list(self.coords.keys())}")

# ============================================================================
# OPTIMIZED STAGE 1: VECTORIZED OPERATIONS
# 優化階段 1：向量化操作
# ============================================================================

def optimized_stage1_four_choice_grt_lba(rt_data: np.ndarray, choice_data: np.ndarray, 
                                        stimloc_data: np.ndarray, coords: Dict) -> pm.Model:
    """
    Optimized Stage 1 with PyMC coordinates and vectorized operations
    使用 PyMC 坐標系統和向量化操作的優化階段 1
    """
    
    print("Optimized Stage 1: Vectorized four-choice GRT-LBA...")
    
    with pm.Model(coords=coords) as model:
        
        # ====================================================================
        # Improved Prior Specifications
        # ====================================================================
        
        # GRT parameters with informative priors
        db1 = pm.Beta('db1', alpha=2, beta=2, 
                      doc="Decision boundary dimension 1")
        db2 = pm.Beta('db2', alpha=2, beta=2,
                      doc="Decision boundary dimension 2")
        
        # Log-normal for positive parameters (better numerical properties)
        log_sp = pm.Normal('log_sp', mu=np.log(0.2), sigma=0.5)
        sp_shared = pm.Deterministic('sp', pm.math.exp(log_sp))
        sp1 = pm.Deterministic('sp1', sp_shared)
        sp2 = pm.Deterministic('sp2', sp_shared)
        
        # LBA parameters with improved priors
        log_A = pm.Normal('log_A', mu=np.log(0.3), sigma=0.3)
        A = pm.Deterministic('A', pm.math.exp(log_A))
        
        log_b = pm.Normal('log_b', mu=np.log(0.4), sigma=0.3)
        b_shared = pm.Deterministic('b_shared', pm.math.exp(log_b))
        
        # Vectorized threshold computation
        thresholds = pm.Deterministic('thresholds', 
                                    A + pm.math.stack([b_shared, b_shared, b_shared, b_shared]),
                                    dims=['accumulator'])
        
        log_base_v = pm.Normal('log_base_v', mu=np.log(1.0), sigma=0.3)
        base_v = pm.Deterministic('base_v', pm.math.exp(log_base_v))
        
        # Fixed parameters
        s_fixed = 0.3
        t0_fixed = 0.25
        
        # ====================================================================
        # Custom Likelihood with Better Error Handling
        # ====================================================================
        
        likelihood = pm.CustomDist(
            'likelihood',
            A, thresholds[0], thresholds[1], thresholds[2], thresholds[3],
            logp=lambda A_val, b1_val, b2_val, b3_val, b4_val: 
                fast_lba_loglik(
                    rt_data, choice_data, stimloc_data,
                    db1, db2, sp1, sp2, A_val,
                    pt.stack([b1_val, b2_val, b3_val, b4_val]),
                    base_v, s_fixed, t0_fixed
                ),
            observed=np.zeros(len(rt_data))  # Dummy observed data
        )
    
    return model

# ============================================================================
# HIERARCHICAL MODEL WITH NON-CENTERED PARAMETERIZATION
# 非中心化參數的階層模型
# ============================================================================

def optimized_hierarchical_four_choice_grt_lba(data_dict: Dict[int, Dict], 
                                              coords: Dict) -> pm.Model:
    """
    Hierarchical model with non-centered parameterization for better sampling
    使用非中心化參數的階層模型以獲得更好的採樣
    """
    
    print("Building optimized hierarchical four-choice GRT-LBA model...")
    
    with pm.Model(coords=coords) as model:
        
        # ====================================================================
        # Group-level (hyperprior) parameters
        # ====================================================================
        
        # Decision boundaries
        mu_db1 = pm.Beta('mu_db1', alpha=2, beta=2)
        sigma_db1 = pm.HalfNormal('sigma_db1', sigma=0.2)
        
        mu_db2 = pm.Beta('mu_db2', alpha=2, beta=2)
        sigma_db2 = pm.HalfNormal('sigma_db2', sigma=0.2)
        
        # Perceptual variabilities (log-scale)
        mu_log_sp1 = pm.Normal('mu_log_sp1', mu=np.log(0.2), sigma=0.3)
        sigma_log_sp1 = pm.HalfNormal('sigma_log_sp1', sigma=0.2)
        
        mu_log_sp2 = pm.Normal('mu_log_sp2', mu=np.log(0.2), sigma=0.3)
        sigma_log_sp2 = pm.HalfNormal('sigma_log_sp2', sigma=0.2)
        
        # LBA parameters
        mu_log_A = pm.Normal('mu_log_A', mu=np.log(0.3), sigma=0.3)
        sigma_log_A = pm.HalfNormal('sigma_log_A', sigma=0.2)
        
        mu_log_base_v = pm.Normal('mu_log_base_v', mu=np.log(1.0), sigma=0.3)
        sigma_log_base_v = pm.HalfNormal('sigma_log_base_v', sigma=0.2)
        
        # Threshold offsets (separate for each accumulator)
        mu_log_b = pm.Normal('mu_log_b', mu=np.log(0.4), sigma=0.3, 
                            dims=['accumulator'])
        sigma_log_b = pm.HalfNormal('sigma_log_b', sigma=0.2, 
                                   dims=['accumulator'])
        
        # ====================================================================
        # Individual-level parameters (non-centered)
        # ====================================================================
        
        # Raw individual deviations (standard normal)
        db1_raw = pm.Normal('db1_raw', mu=0, sigma=1, dims=['participant'])
        db2_raw = pm.Normal('db2_raw', mu=0, sigma=1, dims=['participant'])
        
        log_sp1_raw = pm.Normal('log_sp1_raw', mu=0, sigma=1, dims=['participant'])
        log_sp2_raw = pm.Normal('log_sp2_raw', mu=0, sigma=1, dims=['participant'])
        
        log_A_raw = pm.Normal('log_A_raw', mu=0, sigma=1, dims=['participant'])
        log_base_v_raw = pm.Normal('log_base_v_raw', mu=0, sigma=1, dims=['participant'])
        
        log_b_raw = pm.Normal('log_b_raw', mu=0, sigma=1, 
                             dims=['participant', 'accumulator'])
        
        # Non-centered transformations
        db1 = pm.Deterministic('db1', 
                              pm.math.sigmoid(
                                  pm.math.logit(mu_db1) + sigma_db1 * db1_raw
                              ), dims=['participant'])
        
        db2 = pm.Deterministic('db2',
                              pm.math.sigmoid(
                                  pm.math.logit(mu_db2) + sigma_db2 * db2_raw
                              ), dims=['participant'])
        
        sp1 = pm.Deterministic('sp1', 
                              pm.math.exp(mu_log_sp1 + sigma_log_sp1 * log_sp1_raw),
                              dims=['participant'])
        
        sp2 = pm.Deterministic('sp2',
                              pm.math.exp(mu_log_sp2 + sigma_log_sp2 * log_sp2_raw),
                              dims=['participant'])
        
        A = pm.Deterministic('A',
                            pm.math.exp(mu_log_A + sigma_log_A * log_A_raw),
                            dims=['participant'])
        
        base_v = pm.Deterministic('base_v',
                                 pm.math.exp(mu_log_base_v + sigma_log_base_v * log_base_v_raw),
                                 dims=['participant'])
        
        # Threshold offsets with broadcasting
        log_b = pm.Deterministic('log_b',
                                mu_log_b[None, :] + sigma_log_b[None, :] * log_b_raw,
                                dims=['participant', 'accumulator'])
        
        b_offsets = pm.Deterministic('b_offsets', pm.math.exp(log_b),
                                    dims=['participant', 'accumulator'])
        
        # Individual thresholds
        thresholds = pm.Deterministic('thresholds',
                                     A[:, None] + b_offsets,
                                     dims=['participant', 'accumulator'])
        
        # ====================================================================
        # Likelihood for each participant
        # ====================================================================
        
        for i, (subj_id, subj_data) in enumerate(data_dict.items()):
            rt_subj = subj_data['rt_data']
            choice_subj = subj_data['choice_data']
            stimloc_subj = subj_data['stimloc_data']
            
            # Extract individual parameters
            db1_subj = db1[i]
            db2_subj = db2[i]
            sp1_subj = sp1[i]
            sp2_subj = sp2[i]
            A_subj = A[i]
            base_v_subj = base_v[i]
            thresholds_subj = thresholds[i, :]
            
            # Custom likelihood
            likelihood_subj = pm.CustomDist(
                f'likelihood_subj_{subj_id}',
                A_subj, thresholds_subj[0], thresholds_subj[1], 
                thresholds_subj[2], thresholds_subj[3],
                logp=lambda A_val, b1_val, b2_val, b3_val, b4_val:
                    fast_lba_loglik(
                        rt_subj, choice_subj, stimloc_subj,
                        db1_subj, db2_subj, sp1_subj, sp2_subj, A_val,
                        pt.stack([b1_val, b2_val, b3_val, b4_val]),
                        base_v_subj, 0.3, 0.25
                    ),
                observed=np.zeros(len(rt_subj))
            )
    
    return model

# ============================================================================
# ADAPTIVE SAMPLING CONFIGURATION
# 自適應採樣配置
# ============================================================================

class AdaptiveSamplerConfig:
    """
    Adaptive sampler configuration for optimal performance
    自適應採樣器配置以獲得最佳性能
    """
    
    @staticmethod
    def get_sampling_config(model_complexity: str, n_trials: int) -> Dict:
        """
        Get adaptive sampling configuration based on model complexity
        根據模型複雜度獲得自適應採樣配置
        """
        
        configs = {
            'simple': {
                'draws': 800,
                'tune': 400,
                'chains': 4,
                'target_accept': 0.90,
                'max_treedepth': 10,
                'init': 'adapt_diag'
            },
            'moderate': {
                'draws': 1000,
                'tune': 600,
                'chains': 4,
                'target_accept': 0.95,
                'max_treedepth': 12,
                'init': 'adapt_diag'
            },
            'complex': {
                'draws': 1500,
                'tune': 800,
                'chains': 4,
                'target_accept': 0.97,
                'max_treedepth': 15,
                'init': 'adapt_diag'
            },
            'hierarchical': {
                'draws': 2000,
                'tune': 1000,
                'chains': 4,
                'target_accept': 0.98,
                'max_treedepth': 15,
                'init': 'adapt_diag'
            }
        }
        
        config = configs.get(model_complexity, configs['moderate']).copy()
        
        # Adjust based on data size
        if n_trials > 1000:
            config['target_accept'] = min(config['target_accept'] + 0.01, 0.99)
            config['max_treedepth'] += 1
        
        return config

# ============================================================================
# OPTIMIZED ANALYZER CLASS
# 優化分析器類別
# ============================================================================

class OptimizedFourChoiceAnalyzer:
    """
    Optimized Four-Choice GRT-LBA Analyzer with PyMC features
    使用 PyMC 特性的優化四選項 GRT-LBA 分析器
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        self.csv_file = csv_file
        self.results_dir = Path('optimized_four_choice_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize optimized preprocessor
        self.preprocessor = OptimizedFourChoiceDataPreprocessor(csv_file)
        self.df = self.preprocessor.df
        self.coords = self.preprocessor.coords
        self.dims = self.preprocessor.dims
        self.participants = self.preprocessor.participants
        
        self.sampler_config = AdaptiveSamplerConfig()
        
        print("Optimized Four-Choice GRT-LBA Analyzer initialized")
        print(f"PyMC coordinates: {list(self.coords.keys())}")
    
    def analyze_single_subject_optimized(self, subject_id: int) -> Dict[str, Any]:
        """
        Optimized single subject analysis with automatic model selection
        自動模型選擇的優化單受試者分析
        """
        
        print(f"\nOptimized analysis for Subject {subject_id}")
        
        # Prepare data
        subject_data = self.df[self.df['participant'] == subject_id].copy()
        
        if len(subject_data) < 20:
            print(f"Insufficient data for Subject {subject_id}")
            return None
        
        rt_data = subject_data['RT'].values.astype(np.float32)
        choice_data = subject_data['choice_four'].values.astype(np.int32)
        stimloc_data = np.column_stack([
            subject_data['stimloc_x'].values,
            subject_data['stimloc_y'].values
        ]).astype(np.float32)
        
        # Get adaptive sampling configuration
        n_trials = len(rt_data)
        config = self.sampler_config.get_sampling_config('moderate', n_trials)
        
        try:
            # Build optimized model
            model = optimized_stage1_four_choice_grt_lba(
                rt_data, choice_data, stimloc_data, self.coords)
            
            print(f"Sampling with config: {config}")
            
            # Sample with optimized configuration
            with model:
                trace = pm.sample(
                    **config,
                    progressbar=True,
                    return_inferencedata=True,
                    cores=1,
                    random_seed=42 + subject_id,
                    idata_kwargs={'log_likelihood': True}
                )
            
            # Convergence diagnostics
            rhat_max = float(az.rhat(trace).max())
            ess_min = float(az.ess(trace).min())
            
            print(f"Convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            
            if rhat_max < 1.1:
                print("✅ Analysis successful")
                
                # Model comparison metrics
                waic = az.waic(trace)
                loo = az.loo(trace)
                
                result = {
                    'subject_id': subject_id,
                    'trace': trace,
                    'convergence': {'rhat_max': rhat_max, 'ess_min': ess_min},
                    'model_comparison': {
                        'waic': float(waic.waic),
                        'waic_se': float(waic.waic_se),
                        'loo': float(loo.loo),
                        'loo_se': float(loo.loo_se)
                    },
                    'n_trials': n_trials,
                    'sampling_config': config
                }
                
                return result
            else:
                print("❌ Poor convergence")
                return None
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None
    
    def run_hierarchical_analysis(self, max_subjects: Optional[int] = None) -> Dict[str, Any]:
        """
        Run hierarchical analysis across all subjects
        對所有受試者進行階層分析
        """
        
        print("Running optimized hierarchical four-choice analysis...")
        
        # Prepare data for all subjects
        subjects_to_analyze = self.participants
        if max_subjects:
            subjects_to_analyze = subjects_to_analyze[:max_subjects]
        
        data_dict = {}
        for subj_id in subjects_to_analyze:
            subject_data = self.df[self.df['participant'] == subj_id].copy()
            
            if len(subject_data) >= 20:
                data_dict[subj_id] = {
                    'rt_data': subject_data['RT'].values.astype(np.float32),
                    'choice_data': subject_data['choice_four'].values.astype(np.int32),
                    'stimloc_data': np.column_stack([
                        subject_data['stimloc_x'].values,
                        subject_data['stimloc_y'].values
                    ]).astype(np.float32)
                }
        
        print(f"Hierarchical analysis with {len(data_dict)} subjects")
        
        # Update coordinates for hierarchical model
        hierarchical_coords = self.coords.copy()
        hierarchical_coords['participant'] = list(data_dict.keys())
        
        # Build hierarchical model
        model = optimized_hierarchical_four_choice_grt_lba(data_dict, hierarchical_coords)
        
        # Hierarchical sampling configuration
        config = self.sampler_config.get_sampling_config('hierarchical', 
                                                        sum(len(d['rt_data']) for d in data_dict.values()))
        
        try:
            with model:
                trace = pm.sample(
                    **config,
                    progressbar=True,
                    return_inferencedata=True,
                    cores=1,
                    random_seed=42
                )
            
            # Analysis results
            rhat_max = float(az.rhat(trace).max())
            ess_min = float(az.ess(trace).min())
            
            print(f"Hierarchical convergence: R̂_max = {rhat_max:.3f}, ESS_min = {ess_min:.0f}")
            
            result = {
                'model_type': 'hierarchical_four_choice_grt_lba',
                'trace': trace,
                'convergence': {'rhat_max': rhat_max, 'ess_min': ess_min},
                'n_subjects': len(data_dict),
                'subjects': list(data_dict.keys()),
                'sampling_config': config
            }
            
            # Save results
            trace.to_netcdf(self.results_dir / 'hierarchical_trace.nc')
            
            return result
            
        except Exception as e:
            print(f"Hierarchical analysis failed: {e}")
            return None

# ============================================================================
# USAGE EXAMPLE
# 使用範例
# ============================================================================

def run_optimized_analysis(max_subjects: Optional[int] = None):
    """
    Run optimized four-choice GRT-LBA analysis
    運行優化的四選項 GRT-LBA 分析
    """
    
    print("="*60)
    print("OPTIMIZED FOUR-CHOICE GRT-LBA ANALYSIS")
    print("優化的四選項 GRT-LBA 分析")
    print("="*60)
    
    analyzer = OptimizedFourChoiceAnalyzer()
    
    # Single subject analyses
    if max_subjects and max_subjects <= 3:
        print("\nRunning individual subject analyses...")
        subjects = analyzer.participants[:max_subjects]
        
        for subject_id in subjects:
            result = analyzer.analyze_single_subject_optimized(subject_id)
            if result:
                print(f"Subject {subject_id}: WAIC = {result['model_comparison']['waic']:.2f}")
    
    # Hierarchical analysis
    print("\nRunning hierarchical analysis...")
    hierarchical_result = analyzer.run_hierarchical_analysis(max_subjects)
    
    if
