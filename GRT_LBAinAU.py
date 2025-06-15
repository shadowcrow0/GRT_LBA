# -*- coding: utf-8 -*-
"""
Individual Subject GRT-LBA Model Analysis - FIXED VERSION
Each subject gets their own Bayesian LBA analysis, then results are integrated
Includes sigma matrix analysis for each individual and combined results

MAIN FIX: Vectorized LBA likelihood function to avoid tensor-to-int conversion issues
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
from scipy.special import logit, expit
import json
import time
import warnings
import pickle
import os
from pathlib import Path
warnings.filterwarnings('ignore')

def grt_dbt(stimloc, db, sp, coactive=False):
    """
    GRT decision boundary transformation function (ported from MATLAB grt_dbt.m)
    
    Purpose:
        Converts GRT decision boundaries and perceptual variabilities into LBA drift rates
        
    Parameters:
    -----------
    stimloc : array, shape (n_items, 2)
        Stimulus locations [dimx, dimy] in 2D perceptual space
    db : array, shape (2,)
        Decision boundaries [bndx, bndy] separating response regions
    sp : array, shape (2,)
        Perceptual variabilities [sigmax, sigmay] (standard deviations of perceptual noise)
    coactive : bool
        Whether to use coactive model with multivariate normal distribution
    
    Returns:
    --------
    p : array, shape (n_items,)
        Drift rate probabilities for accumulator A
    """
    stimloc = np.atleast_2d(stimloc)
    db = np.atleast_1d(db)
    sp = np.atleast_1d(sp)
    
    if not coactive:
        # Standard GRT model - independent processing of dimensions
        if stimloc.shape[0] > 1:
            db = np.tile(db, (stimloc.shape[0], 1))
            sp = np.tile(sp, (stimloc.shape[0], 1))
        
        # Standardized distance and convert to probabilities
        zx = (db - stimloc) / sp
        p = stats.norm.cdf(zx)
        
        # Avoid extreme values that cause numerical issues
        p = np.clip(p, 0.00001, 0.99999)
        
        # Calculate probability for accumulator A (first dimension)
        p_result = p[:, 0] if p.ndim > 1 else p[0]
        
    else:
        # Coactive model using multivariate normal distribution
        p_result = []
        for i in range(stimloc.shape[0]):
            mean = stimloc[i]
            cov = np.diag(sp**2)
            p_a = stats.multivariate_normal.cdf(db, mean=mean, cov=cov)
            p_result.append(1 - p_a)  # Invert for P(choose B)
            
        p_result = np.array(p_result)
        p_result = np.clip(p_result, 0.00001, 0.99999)
    
    return p_result

def logit_transform(x, inverse=False):
    """Logit transformation function"""
    if inverse:
        return expit(x)  # 1 / (1 + exp(-x))
    else:
        return logit(x)   # log(x / (1-x))

def transform_boundaries(db_logit, stimloc):
    """Transform logit-space boundaries back to real coordinates"""
    min_x, max_x = stimloc[:, 0].min(), stimloc[:, 0].max()
    min_y, max_y = stimloc[:, 1].min(), stimloc[:, 1].max()
    
    db1_real = (logit_transform(db_logit[0], inverse=True) * 
                (max_x - min_x)) + min_x
    
    db2_real = (logit_transform(db_logit[1], inverse=True) * 
                (max_y - min_y)) + min_y
    
    return np.array([db1_real, db2_real])

def vectorized_lba_loglik_single_subject(rt_data, choice_data, stimloc, 
                                        db1, db2, sp1, sp2, A, b1, b2, s, t0):
    """
    FIXED: Vectorized LBA log-likelihood for single subject with GRT drift rate conversion
    
    This version uses vectorized operations instead of explicit loops to avoid
    the tensor-to-int conversion issue.
    
    Purpose:
        Calculate log-likelihood for LBA model where drift rates are derived
        from GRT decision boundaries and perceptual variabilities
        
    Parameters:
    -----------
    rt_data : tensor, shape (n_trials,)
        Response times for all trials
    choice_data : tensor, shape (n_trials,)
        Binary choices (0 or 1) for all trials
    stimloc : tensor, shape (n_trials, 2)
        Stimulus locations for each trial [x, y coordinates]
    db1, db2 : tensor (scalar)
        Decision boundaries for two dimensions
    sp1, sp2 : tensor (scalar)
        Perceptual variabilities for two dimensions
    A : tensor (scalar)
        LBA start point variability (uniform distribution width)
    b1, b2 : tensor (scalar)
        LBA decision thresholds for each accumulator
    s : tensor (scalar)
        LBA drift rate variability (noise in evidence accumulation)
    t0 : tensor (scalar)
        Non-decision time (encoding + motor response)
        
    Returns:
    --------
    total_loglik : tensor (scalar)
        Total log-likelihood across all trials
    """
    # Ensure minimum parameter values for numerical stability
    A = pt.maximum(A, 0.05)
    b1 = pt.maximum(b1, A + 0.05)  # Threshold must exceed start point variability
    b2 = pt.maximum(b2, A + 0.05)
    s = pt.maximum(s, 0.1)
    t0 = pt.maximum(t0, 0.01)
    sp1 = pt.maximum(sp1, 0.01)
    sp2 = pt.maximum(sp2, 0.01)
    
    # Decision time (remove non-decision time from RT)
    rt_decision = pt.maximum(rt_data - t0, 0.01)
    
    # Convert GRT parameters to LBA drift rates for all trials (VECTORIZED)
    # v1_prob: probability that evidence favors accumulator 1
    v1_prob = pt.sigmoid((db1 - stimloc[:, 0]) / sp1)
    v2_prob = pt.sigmoid((db2 - stimloc[:, 1]) / sp2)
    
    # Combine probabilities for drift rates
    v1 = v1_prob * v2_prob  # Probability for accumulator 1
    v2 = 1 - v1             # Probability for accumulator 2
    
    # Ensure minimum drift rates for numerical stability
    v1 = pt.maximum(v1, 0.1)
    v2 = pt.maximum(v2, 0.1)
    
    # Select parameters for winning and losing accumulators based on choice (VECTORIZED)
    v_winner = pt.where(pt.eq(choice_data, 0), v1, v2)
    v_loser = pt.where(pt.eq(choice_data, 0), v2, v1)
    b_winner = pt.where(pt.eq(choice_data, 0), b1, b2)
    b_loser = pt.where(pt.eq(choice_data, 0), b2, b1)
    
    # LBA likelihood calculation (VECTORIZED)
    rt_decision = pt.maximum(rt_decision, 0.01)
    sqrt_t = pt.sqrt(rt_decision)
    
    # Winner PDF calculation
    z1_win = (v_winner * rt_decision - b_winner) / sqrt_t
    z2_win = (v_winner * rt_decision - A) / sqrt_t
    z1_win = pt.clip(z1_win, -10, 10)  # Prevent overflow
    z2_win = pt.clip(z2_win, -10, 10)
    
    # Normal CDF and PDF values
    Phi_z1_win = 0.5 * (1 + pt.erf(z1_win / pt.sqrt(2)))
    Phi_z2_win = 0.5 * (1 + pt.erf(z2_win / pt.sqrt(2)))
    phi_z1_win = pt.exp(-0.5 * z1_win**2) / pt.sqrt(2 * np.pi)
    phi_z2_win = pt.exp(-0.5 * z2_win**2) / pt.sqrt(2 * np.pi)
    
    # LBA winner PDF components
    cdf_diff = pt.maximum(Phi_z1_win - Phi_z2_win, 1e-12)
    pdf_diff = (phi_z1_win - phi_z2_win) / sqrt_t
    
    winner_pdf = (v_winner / A) * cdf_diff + pdf_diff / A
    winner_pdf = pt.maximum(winner_pdf, 1e-12)
    winner_logpdf = pt.log(winner_pdf)
    
    # Loser survival function (probability of not finishing by time rt_decision)
    z1_lose = (v_loser * rt_decision - b_loser) / sqrt_t
    z2_lose = (v_loser * rt_decision - A) / sqrt_t
    z1_lose = pt.clip(z1_lose, -10, 10)
    z2_lose = pt.clip(z2_lose, -10, 10)
    
    Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
    Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
    
    loser_cdf = pt.maximum(Phi_z1_lose - Phi_z2_lose, 1e-12)
    loser_survival = pt.maximum(1 - loser_cdf, 1e-12)
    loser_log_survival = pt.log(loser_survival)
    
    # Combine winner PDF and loser survival for all trials
    trial_loglik = winner_logpdf + loser_log_survival
    
    # Sum across all trials (this is the key - no explicit loop needed)
    total_loglik = pt.sum(trial_loglik)
    
    return total_loglik

def compute_sigma_matrix(trace, param_names=['db1', 'db2', 'sp1', 'sp2']):
    """Compute the sigma matrix (covariance matrix) from MCMC samples"""
    posterior = trace.posterior
    
    # Extract parameter samples
    param_samples = []
    available_params = []
    
    for param in param_names:
        if param in posterior.data_vars:
            samples = posterior[param].values.flatten()
            param_samples.append(samples)
            available_params.append(param)
    
    if len(param_samples) == 0:
        return None, None, available_params
    
    # Stack samples into matrix (samples × parameters)
    samples_matrix = np.column_stack(param_samples)
    
    # Compute covariance and correlation matrices
    sigma_matrix = np.cov(samples_matrix.T)  # Covariance matrix
    correlation_matrix = np.corrcoef(samples_matrix.T)  # Correlation matrix
    
    return sigma_matrix, correlation_matrix, available_params

class IndividualSubjectGRTLBA:
    """Individual Subject GRT-LBA Analysis with FIXED likelihood function"""
    
    def __init__(self, csv_file='GRT_LBA.csv'):
        """Initialize with data loading"""
        self.csv_file = csv_file
        self.results_dir = Path('individual_results')
        self.results_dir.mkdir(exist_ok=True)
        self.load_and_prepare_data()
        
        # Storage for individual and combined results
        self.individual_results = {}
        self.individual_traces = {}
        self.combined_results = {}
    
    def load_and_prepare_data(self):
        """Load and prepare data for individual subject analysis"""
        print("Loading data for individual subject GRT-LBA analysis...")
        
        df = pd.read_csv(self.csv_file)
        print(f"Original data: {len(df)} trials")
        
        # Filter extreme RTs (standard preprocessing in RT modeling)
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.0)]
        print(f"After RT filtering: {len(df)} trials")
        
        # Convert Response to binary choice
        df['choice_binary'] = (df['Response'] >= 2).astype(int)
        
        # Create stimulus locations based on Stimulus column
        if 'Stimulus' in df.columns:
            unique_stimuli = sorted(df['Stimulus'].unique())
            n_stim = len(unique_stimuli)
            
            # Create 2D grid layout for stimuli
            if n_stim <= 4:
                # 2x2 grid for up to 4 stimuli
                stim_locs = []
                for i, stim in enumerate(unique_stimuli):
                    row = i // 2
                    col = i % 2
                    stim_locs.append([col, row])
            else:
                # 3x3 grid for more stimuli
                stim_locs = []
                for i, stim in enumerate(unique_stimuli):
                    row = i // 3
                    col = i % 3
                    stim_locs.append([col, row])
            
            self.stimloc = np.array(stim_locs)
            
            # Map stimulus numbers to locations
            stim_to_loc = {stim: loc for stim, loc in 
                          zip(unique_stimuli, stim_locs)}
            
            df['stimloc_x'] = df['Stimulus'].map(lambda x: stim_to_loc[x][0])
            df['stimloc_y'] = df['Stimulus'].map(lambda x: stim_to_loc[x][1])
        else:
            # Default 2x2 grid if no stimulus info
            self.stimloc = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
            df['stimloc_x'] = 0
            df['stimloc_y'] = 0
        
        self.df = df
        self.participants = sorted(df['participant'].unique())
        self.n_participants = len(self.participants)
        
        print(f"Participants: {self.n_participants}")
        print(f"Participants: {self.participants}")
        print(f"Stimulus locations shape: {self.stimloc.shape}")
        
        # Check data distribution per participant
        for p in self.participants[:5]:  # Show first 5
            p_data = df[df['participant'] == p]
            print(f"  Participant {p}: {len(p_data)} trials, "
                  f"choice dist: {p_data['choice_binary'].value_counts().to_dict()}")
    
    def build_single_subject_model(self, subject_data, subject_id):
        """Build GRT-LBA model for single subject with FIXED likelihood"""
        # Prepare data arrays
        rt_obs = subject_data['RT'].values.astype(np.float32)
        choice_obs = subject_data['choice_binary'].values.astype(np.int32)
        stimloc_obs = np.column_stack([
            subject_data['stimloc_x'].values,
            subject_data['stimloc_y'].values
        ]).astype(np.float32)
        
        print(f"  Building model for subject {subject_id}: {len(subject_data)} trials")
        
        with pm.Model() as model:
            # Decision boundary parameters (in logit space for unconstrained sampling)
            db1_logit = pm.Normal('db1_logit', mu=0, sigma=0.5)
            db2_logit = pm.Normal('db2_logit', mu=0, sigma=0.5)
            
            # Perceptual variability parameters (in log space for positive constraint)
            sp1_log = pm.Normal('sp1_log', mu=np.log(0.2), sigma=0.3)
            sp2_log = pm.Normal('sp2_log', mu=np.log(0.2), sigma=0.3)
            
            # LBA parameters (all in log space for positive constraint)
            A_log = pm.Normal('A_log', mu=np.log(0.35), sigma=0.3)
            bMa1_log = pm.Normal('bMa1_log', mu=np.log(0.25), sigma=0.5)
            bMa2_log = pm.Normal('bMa2_log', mu=np.log(0.25), sigma=0.5)
            s_log = pm.Normal('s_log', mu=np.log(0.25), sigma=0.3)
            t0_log = pm.Normal('t0_log', mu=np.log(0.22), sigma=0.3)
            
            # Transform to interpretable parameter space
            sp1 = pm.Deterministic('sp1', pm.math.exp(sp1_log))
            sp2 = pm.Deterministic('sp2', pm.math.exp(sp2_log))
            A = pm.Deterministic('A', pm.math.exp(A_log))
            s = pm.Deterministic('s', pm.math.exp(s_log))
            t0 = pm.Deterministic('t0', pm.math.exp(t0_log))
            
            # Transform boundaries to real coordinate space
            stimloc_tensor = pt.as_tensor(self.stimloc, dtype='float32')
            min_x = pt.min(stimloc_tensor[:, 0])
            max_x = pt.max(stimloc_tensor[:, 0])
            min_y = pt.min(stimloc_tensor[:, 1])
            max_y = pt.max(stimloc_tensor[:, 1])
            
            # Sigmoid maps logit space to (0,1), then scale to coordinate range
            db1 = pm.Deterministic('db1', 
                                  pm.math.sigmoid(db1_logit) * (max_x - min_x) + min_x)
            db2 = pm.Deterministic('db2', 
                                  pm.math.sigmoid(db2_logit) * (max_y - min_y) + min_y)
            
            # Transform thresholds (must exceed start point variability)
            bMa1 = pm.Deterministic('bMa1', pm.math.exp(bMa1_log))
            bMa2 = pm.Deterministic('bMa2', pm.math.exp(bMa2_log))
            b1 = pm.Deterministic('b1', A + bMa1)
            b2 = pm.Deterministic('b2', A + bMa2)
            
            # FIXED: Use vectorized LBA likelihood with GRT conversion
            pm.Potential('lba_grt_likelihood',
                        vectorized_lba_loglik_single_subject(
                            pt.as_tensor(rt_obs),
                            pt.as_tensor(choice_obs),
                            pt.as_tensor(stimloc_obs),
                            db1, db2, sp1, sp2, A, b1, b2, s, t0))
        
        return model
    
    def analyze_single_subject(self, subject_id, draws=400, tune=200, chains=2):
        """Analyze single subject with Bayesian GRT-LBA"""
        print(f"\n{'='*50}")
        print(f"ANALYZING SUBJECT {subject_id}")
        print(f"{'='*50}")
        
        # Get subject data
        subject_data = self.df[self.df['participant'] == subject_id].copy()
        
        if len(subject_data) < 20:
            print(f"  Warning: Subject {subject_id} has only {len(subject_data)} trials")
            return None
        
        # Build model
        model = self.build_single_subject_model(subject_data, subject_id)
        
        # Sample
        print(f"  Starting MCMC sampling...")
        start_time = time.time()
        
        try:
            with model:
                trace = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.9, max_treedepth=10,
                    cores=1, random_seed=123 + subject_id
                )
            
            elapsed = time.time() - start_time
            print(f"  Sampling completed in {elapsed:.1f} seconds")
            
            # Analyze results for this subject
            results = self.analyze_subject_results(trace, subject_id)
            
            # Save individual results
            self.save_individual_results(trace, results, subject_id)
            
            return trace
            
        except Exception as e:
            print(f"  ✗ Sampling failed for subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_subject_results(self, trace, subject_id):
        """Analyze results for individual subject"""
        print(f"  Analyzing results for subject {subject_id}...")
        
        posterior = trace.posterior
        
        # Extract parameter estimates
        results = {
            'subject_id': subject_id,
            'model_type': 'Individual_GRT_LBA',
            'parameter_estimates': {}
        }
        
        # Key parameters with their interpretations
        key_params = ['db1', 'db2', 'sp1', 'sp2', 'A', 's', 't0', 'b1', 'b2']
        
        for param in key_params:
            if param in posterior.data_vars:
                samples = posterior[param].values.flatten()
                results['parameter_estimates'][param] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'hdi_2.5': float(np.percentile(samples, 2.5)),
                    'hdi_97.5': float(np.percentile(samples, 97.5)),
                    'median': float(np.median(samples))
                }
        
        # Compute sigma matrix for independence testing
        sigma_matrix, correlation_matrix, available_params = compute_sigma_matrix(
            trace, param_names=['db1', 'db2', 'sp1', 'sp2'])
        
        if sigma_matrix is not None:
            results['sigma_matrix'] = {
                'covariance_matrix': sigma_matrix.tolist(),
                'correlation_matrix': correlation_matrix.tolist(),
                'parameter_names': available_params
            }
            
            # Test independence assumptions
            results['independence_tests'] = self.test_independence(correlation_matrix, available_params)
        
        # Separability test (sp1 vs sp2)
        if 'sp1' in posterior.data_vars and 'sp2' in posterior.data_vars:
            sp1_samples = posterior['sp1'].values.flatten()
            sp2_samples = posterior['sp2'].values.flatten()
            sp_ratio = sp1_samples / sp2_samples
            sp_ratio_hdi = np.percentile(sp_ratio, [2.5, 97.5])
            
            results['separability_test'] = {
                'sp1_sp2_ratio_mean': float(np.mean(sp_ratio)),
                'sp1_sp2_ratio_hdi': [float(sp_ratio_hdi[0]), float(sp_ratio_hdi[1])],
                'separability_supported': bool(sp_ratio_hdi[0] < 1.0 < sp_ratio_hdi[1])
            }
        
        # Model diagnostics
        try:
            ess = az.ess(trace)
            rhat = az.rhat(trace)
            
            results['diagnostics'] = {}
            for param in key_params:
                if param in ess.data_vars:
                    ess_val = float(ess[param]) if ess[param].ndim == 0 else float(ess[param].min())
                    rhat_val = float(rhat[param]) if rhat[param].ndim == 0 else float(rhat[param].max())
                    results['diagnostics'][param] = {
                        'ess': ess_val,
                        'rhat': rhat_val
                    }
        except Exception as e:
            results['diagnostics'] = {'error': str(e)}
        
        # Store results
        self.individual_results[subject_id] = results
        self.individual_traces[subject_id] = trace
        
        return results
    
    def test_independence(self, correlation_matrix, param_names):
        """Test parameter independence using correlation matrix"""
        independence_tests = {}
        
        if correlation_matrix is not None and len(param_names) >= 2:
            # Test all pairwise correlations
            for i in range(len(param_names)):
                for j in range(i+1, len(param_names)):
                    param1, param2 = param_names[i], param_names[j]
                    correlation = correlation_matrix[i, j]
                    
                    # Independence test with threshold
                    abs_corr = abs(correlation)
                    independence_tests[f'{param1}_{param2}'] = {
                        'correlation': float(correlation),
                        'independent': bool(abs_corr < 0.3),
                        'evidence_strength': ('weak' if abs_corr < 0.3 else 
                                            'moderate' if abs_corr < 0.6 else 'strong')
                    }
        
        return independence_tests
    
    def save_individual_results(self, trace, results, subject_id):
        """Save individual subject results"""
        # Save JSON results
        results_file = self.results_dir / f'subject_{subject_id}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trace as pickle
        trace_file = self.results_dir / f'subject_{subject_id}_trace.pkl'
        with open(trace_file, 'wb') as f:
            pickle.dump(trace, f)
        
        print(f"  ✓ Results saved for subject {subject_id}")
    
    def analyze_all_subjects(self, max_subjects=None, draws=400, tune=200):
        """Analyze all subjects individually"""
        print(f"\n{'='*70}")
        print(f"INDIVIDUAL SUBJECT GRT-LBA ANALYSIS")
        print(f"Total subjects: {self.n_participants}")
        print(f"{'='*70}")
        
        subjects_to_analyze = self.participants
        if max_subjects is not None:
            subjects_to_analyze = subjects_to_analyze[:max_subjects]
            print(f"Analyzing first {max_subjects} subjects for testing")
        
        successful_analyses = 0
        failed_analyses = 0
        
        for subject_id in subjects_to_analyze:
            try:
                trace = self.analyze_single_subject(subject_id, draws=draws, tune=tune)
                if trace is not None:
                    successful_analyses += 1
                else:
                    failed_analyses += 1
            except Exception as e:
                print(f"  ✗ Failed to analyze subject {subject_id}: {e}")
                failed_analyses += 1
        
        print(f"\n{'='*50}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Failed analyses: {failed_analyses}")
        print(f"Success rate: {successful_analyses/(successful_analyses+failed_analyses)*100:.1f}%")
        
        # Combine results
        if successful_analyses > 0:
            self.combine_results()
    
    def combine_results(self):
        """Combine individual subject results into group analysis"""
        print(f"\n{'='*50}")
        print(f"COMBINING INDIVIDUAL RESULTS")
        print(f"{'='*50}")
        
        if not self.individual_results:
            print("No individual results to combine")
            return
        
        # Initialize combined results structure
        self.combined_results = {
            'analysis_type': 'Combined_Individual_GRT_LBA',
            'n_subjects': len(self.individual_results),
            'subject_ids': list(self.individual_results.keys()),
            'group_parameter_estimates': {},
            'group_sigma_matrices': {},
            'group_independence_tests': {},
            'group_separability_tests': {},
            'summary_statistics': {}
        }
        
        # Combine parameter estimates
        key_params = ['db1', 'db2', 'sp1', 'sp2', 'A', 's', 't0', 'b1', 'b2']
        
        for param in key_params:
            param_values = []
            for subject_id, results in self.individual_results.items():
                if param in results['parameter_estimates']:
                    param_values.append(results['parameter_estimates'][param]['mean'])
            
            if param_values:
                self.combined_results['group_parameter_estimates'][param] = {
                    'group_mean': float(np.mean(param_values)),
                    'group_std': float(np.std(param_values)),
                    'group_median': float(np.median(param_values)),
                    'individual_values': param_values,
                    'n_subjects': len(param_values)
                }
        
        # Combine sigma matrices
        sigma_matrices = []
        correlation_matrices = []
        
        for subject_id, results in self.individual_results.items():
            if 'sigma_matrix' in results:
                sigma_matrices.append(np.array(results['sigma_matrix']['covariance_matrix']))
                correlation_matrices.append(np.array(results['sigma_matrix']['correlation_matrix']))
        
        if sigma_matrices:
            # Average covariance and correlation matrices across subjects
            mean_sigma = np.mean(sigma_matrices, axis=0)
            mean_correlation = np.mean(correlation_matrices, axis=0)
            
            self.combined_results['group_sigma_matrices'] = {
                'mean_covariance_matrix': mean_sigma.tolist(),
                'mean_correlation_matrix': mean_correlation.tolist(),
                'individual_covariance_matrices': [s.tolist() for s in sigma_matrices],
                'individual_correlation_matrices': [c.tolist() for c in correlation_matrices],
                'parameter_names': self.individual_results[list(self.individual_results.keys())[0]]['sigma_matrix']['parameter_names']
            }
        
        # Combine independence tests
        independence_summary = {}
        for subject_id, results in self.individual_results.items():
            if 'independence_tests' in results:
                for test_name, test_result in results['independence_tests'].items():
                    if test_name not in independence_summary:
                        independence_summary[test_name] = {
                            'correlations': [],
                            'independent_count': 0,
                            'total_count': 0
                        }
                    
                    independence_summary[test_name]['correlations'].append(test_result['correlation'])
                    independence_summary[test_name]['total_count'] += 1
                    if test_result['independent']:
                        independence_summary[test_name]['independent_count'] += 1
        
        # Summarize independence tests
        for test_name, summary in independence_summary.items():
            if summary['total_count'] > 0:
                self.combined_results['group_independence_tests'][test_name] = {
                    'mean_correlation': float(np.mean(summary['correlations'])),
                    'std_correlation': float(np.std(summary['correlations'])),
                    'independence_rate': float(summary['independent_count'] / summary['total_count']),
                    'individual_correlations': summary['correlations'],
                    'group_independence_supported': summary['independent_count'] / summary['total_count'] > 0.5
                }
        
        # Combine separability tests
        separability_ratios = []
        separability_supported_count = 0
        
        for subject_id, results in self.individual_results.items():
            if 'separability_test' in results:
                separability_ratios.append(results['separability_test']['sp1_sp2_ratio_mean'])
                if results['separability_test']['separability_supported']:
                    separability_supported_count += 1
        
        if separability_ratios:
            self.combined_results['group_separability_tests'] = {
                'group_mean_sp_ratio': float(np.mean(separability_ratios)),
                'group_std_sp_ratio': float(np.std(separability_ratios)),
                'individual_sp_ratios': separability_ratios,
                'separability_support_rate': float(separability_supported_count / len(separability_ratios)),
                'group_separability_supported': separability_supported_count / len(separability_ratios) > 0.5
            }
        
        # Summary statistics
        self.combined_results['summary_statistics'] = {
            'total_subjects_analyzed': len(self.individual_results),
            'parameters_estimated_per_subject': len(key_params),
            'average_ess': self.compute_average_ess(),
            'average_rhat': self.compute_average_rhat(),
            'model_convergence_rate': self.compute_convergence_rate()
        }
        
        # Save combined results
        self.save_combined_results()
        
        # Print summary
        self.print_combined_summary()
    
    def compute_average_ess(self):
        """Compute average Effective Sample Size across subjects and parameters"""
        all_ess = []
        for subject_id, results in self.individual_results.items():
            if 'diagnostics' in results and isinstance(results['diagnostics'], dict):
                for param, diag in results['diagnostics'].items():
                    if isinstance(diag, dict) and 'ess' in diag:
                        all_ess.append(diag['ess'])
        
        return float(np.mean(all_ess)) if all_ess else None
    
    def compute_average_rhat(self):
        """Compute average R-hat across subjects and parameters"""
        all_rhat = []
        for subject_id, results in self.individual_results.items():
            if 'diagnostics' in results and isinstance(results['diagnostics'], dict):
                for param, diag in results['diagnostics'].items():
                    if isinstance(diag, dict) and 'rhat' in diag:
                        all_rhat.append(diag['rhat'])
        
        return float(np.mean(all_rhat)) if all_rhat else None
    
    def compute_convergence_rate(self):
        """Compute rate of successful convergence (R-hat < 1.1)"""
        convergent_count = 0
        total_count = 0
        
        for subject_id, results in self.individual_results.items():
            if 'diagnostics' in results and isinstance(results['diagnostics'], dict):
                for param, diag in results['diagnostics'].items():
                    if isinstance(diag, dict) and 'rhat' in diag:
                        total_count += 1
                        if diag['rhat'] < 1.1:
                            convergent_count += 1
        
        return float(convergent_count / total_count) if total_count > 0 else None
    
    def save_combined_results(self):
        """Save combined results to file"""
        combined_file = self.results_dir / 'combined_results.json'
        with open(combined_file, 'w') as f:
            json.dump(self.combined_results, f, indent=2)
        
        print(f"✓ Combined results saved to {combined_file}")
    
    def print_combined_summary(self):
        """Print summary of combined results"""
        print(f"\n{'='*60}")
        print(f"COMBINED GROUP ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"Subjects analyzed: {self.combined_results['n_subjects']}")
        print(f"Subject IDs: {self.combined_results['subject_ids']}")
        
        # Group parameter estimates
        print(f"\nGROUP PARAMETER ESTIMATES:")
        print(f"{'-'*40}")
        for param, estimates in self.combined_results['group_parameter_estimates'].items():
            print(f"{param:>6}: {estimates['group_mean']:.3f} ± {estimates['group_std']:.3f} "
                  f"(n={estimates['n_subjects']})")
        
        # Group independence tests
        if self.combined_results['group_independence_tests']:
            print(f"\nGROUP INDEPENDENCE TESTS:")
            print(f"{'-'*40}")
            for test_name, test_result in self.combined_results['group_independence_tests'].items():
                independence_rate = test_result['independence_rate']
                mean_corr = test_result['mean_correlation']
                support = "✓" if test_result['group_independence_supported'] else "✗"
                print(f"{test_name}: r={mean_corr:.3f}, independence rate={independence_rate:.1%} {support}")
        
        # Group separability tests
        if self.combined_results['group_separability_tests']:
            print(f"\nGROUP SEPARABILITY TEST:")
            print(f"{'-'*40}")
            sep_test = self.combined_results['group_separability_tests']
            support_rate = sep_test['separability_support_rate']
            mean_ratio = sep_test['group_mean_sp_ratio']
            support = "✓" if sep_test['group_separability_supported'] else "✗"
            print(f"sp1/sp2 ratio: {mean_ratio:.3f}, support rate: {support_rate:.1%} {support}")
        
        # Model quality
        if self.combined_results['summary_statistics']:
            stats = self.combined_results['summary_statistics']
            print(f"\nMODEL QUALITY:")
            print(f"{'-'*40}")
            if stats['average_ess']:
                print(f"Average ESS: {stats['average_ess']:.0f}")
            if stats['average_rhat']:
                print(f"Average R̂: {stats['average_rhat']:.3f}")
            if stats['model_convergence_rate']:
                print(f"Convergence rate: {stats['model_convergence_rate']:.1%}")
        
        # Sigma matrix summary
        if self.combined_results['group_sigma_matrices']:
            print(f"\nGROUP SIGMA MATRIX (Mean Correlation):")
            print(f"{'-'*40}")
            corr_matrix = np.array(self.combined_results['group_sigma_matrices']['mean_correlation_matrix'])
            param_names = self.combined_results['group_sigma_matrices']['parameter_names']
            
            # Print correlation matrix
            print("     ", end="")
            for name in param_names:
                print(f"{name:>6}", end="")
            print()
            
            for i, name in enumerate(param_names):
                print(f"{name:>4}:", end="")
                for j in range(len(param_names)):
                    print(f"{corr_matrix[i,j]:6.3f}", end="")
                print()
    
    def load_existing_results(self):
        """Load existing individual results from files"""
        print("Loading existing individual results...")
        
        loaded_count = 0
        for subject_id in self.participants:
            results_file = self.results_dir / f'subject_{subject_id}_results.json'
            trace_file = self.results_dir / f'subject_{subject_id}_trace.pkl'
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    self.individual_results[subject_id] = results
                    
                    if trace_file.exists():
                        with open(trace_file, 'rb') as f:
                            trace = pickle.load(f)
                        self.individual_traces[subject_id] = trace
                    
                    loaded_count += 1
                    print(f"  ✓ Loaded results for subject {subject_id}")
                except Exception as e:
                    print(f"  ✗ Failed to load subject {subject_id}: {e}")
        
        print(f"Loaded results for {loaded_count} subjects")
        
        if loaded_count > 0:
            self.combine_results()
        
        return loaded_count
    
    def generate_detailed_report(self):
        """Generate detailed analysis report"""
        print(f"\n{'='*70}")
        print(f"DETAILED INDIVIDUAL SUBJECT GRT-LBA REPORT")
        print(f"{'='*70}")
        
        if not self.individual_results:
            print("No results available for detailed report")
            return
        
        # Individual subject summaries
        print(f"\nINDIVIDUAL SUBJECT SUMMARIES:")
        print(f"{'='*50}")
        
        for subject_id in sorted(self.individual_results.keys()):
            results = self.individual_results[subject_id]
            
            print(f"\nSubject {subject_id}:")
            print(f"{'-'*20}")
            
            # Parameter estimates
            if 'parameter_estimates' in results:
                key_params = ['db1', 'db2', 'sp1', 'sp2']
                for param in key_params:
                    if param in results['parameter_estimates']:
                        est = results['parameter_estimates'][param]
                        print(f"  {param}: {est['mean']:.3f} [{est['hdi_2.5']:.3f}, {est['hdi_97.5']:.3f}]")
            
            # Separability test
            if 'separability_test' in results:
                sep_test = results['separability_test']
                support = "✓" if sep_test['separability_supported'] else "✗"
                print(f"  Separability: {sep_test['sp1_sp2_ratio_mean']:.3f} {support}")
            
            # Independence tests
            if 'independence_tests' in results:
                indep_count = sum(1 for test in results['independence_tests'].values() 
                                if test['independent'])
                total_count = len(results['independence_tests'])
                print(f"  Independence: {indep_count}/{total_count} tests passed")
            
            # Model quality
            if 'diagnostics' in results and isinstance(results['diagnostics'], dict):
                avg_rhat = np.mean([diag['rhat'] for diag in results['diagnostics'].values() 
                                  if isinstance(diag, dict) and 'rhat' in diag])
                print(f"  Model quality: R̂={avg_rhat:.3f}")
        
        # Group-level conclusions
        print(f"\n{'='*50}")
        print(f"GROUP-LEVEL CONCLUSIONS")
        print(f"{'='*50}")
        
        if self.combined_results:
            # Independence conclusion
            if self.combined_results['group_independence_tests']:
                independence_results = []
                for test_name, test_result in self.combined_results['group_independence_tests'].items():
                    independence_results.append(test_result['group_independence_supported'])
                
                overall_independence = sum(independence_results) / len(independence_results)
                print(f"Overall Independence Support: {overall_independence:.1%}")
                
                if overall_independence > 0.7:
                    print("✓ Strong evidence for parameter independence (GRT assumption supported)")
                elif overall_independence > 0.4:
                    print("~ Mixed evidence for parameter independence")
                else:
                    print("✗ Weak evidence for parameter independence (GRT assumption violated)")
            
            # Separability conclusion
            if self.combined_results['group_separability_tests']:
                sep_test = self.combined_results['group_separability_tests']
                support_rate = sep_test['separability_support_rate']
                
                print(f"Overall Separability Support: {support_rate:.1%}")
                
                if support_rate > 0.7:
                    print("✓ Strong evidence for perceptual separability (GRT assumption supported)")
                elif support_rate > 0.4:
                    print("~ Mixed evidence for perceptual separability")
                else:
                    print("✗ Weak evidence for perceptual separability (GRT assumption violated)")
            
            # Overall conclusion
            print(f"\nOVERALL GRT ASSUMPTION TESTING:")
            if (self.combined_results.get('group_independence_tests') and 
                self.combined_results.get('group_separability_tests')):
                
                independence_support = np.mean([test['group_independence_supported'] 
                                              for test in self.combined_results['group_independence_tests'].values()])
                separability_support = self.combined_results['group_separability_tests']['group_separability_supported']
                
                if independence_support and separability_support:
                    print("✓ GRT assumptions are generally SUPPORTED across subjects")
                elif independence_support or separability_support:
                    print("~ GRT assumptions have MIXED support across subjects")
                else:
                    print("✗ GRT assumptions are generally VIOLATED across subjects")

def run_individual_subject_analysis(max_subjects=None, draws=400, tune=200):
    """
    Main function to run individual subject GRT-LBA analysis
    
    Parameters:
    -----------
    max_subjects : int, optional
        Maximum number of subjects to analyze (for testing)
    draws : int
        MCMC draws per subject  
    tune : int
        MCMC tuning steps per subject
        
    Returns:
    --------
    analyzer : IndividualSubjectGRTLBA
        Analyzer object with all results
    """
    print("STARTING INDIVIDUAL SUBJECT GRT-LBA ANALYSIS")
    print("Each subject analyzed separately, then results combined")
    print("="*70)
    
    # Initialize analyzer
    analyzer = IndividualSubjectGRTLBA('GRT_LBA.csv')
    
    # Check for existing results
    existing_count = analyzer.load_existing_results()
    
    if existing_count > 0:
        print(f"\nFound existing results for {existing_count} subjects")
        choice = input("Do you want to (1) use existing results, (2) reanalyze all, or (3) analyze missing only? [1/2/3]: ")
        
        if choice == '1':
            print("Using existing results...")
            analyzer.generate_detailed_report()
            return analyzer
        elif choice == '2':
            print("Reanalyzing all subjects...")
            analyzer.individual_results = {}
            analyzer.individual_traces = {}
        elif choice == '3':
            print("Analyzing missing subjects only...")
            pass
    
    # Run analysis
    try:
        analyzer.analyze_all_subjects(max_subjects=max_subjects, draws=draws, tune=tune)
        
        # Generate detailed report
        analyzer.generate_detailed_report()
        
        print(f"\n{'='*70}")
        print(f"INDIVIDUAL SUBJECT GRT-LBA ANALYSIS COMPLETED")
        print(f"{'='*70}")
        print(f"Individual results saved in: {analyzer.results_dir}")
        print(f"Combined results saved as: combined_results.json")
        
        return analyzer
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_test_analysis():
    """Quick test with subset of subjects"""
    print("Running quick test with 3 subjects...")
    return run_individual_subject_analysis(max_subjects=3, draws=200, tune=100)

if __name__ == "__main__":
    # For quick testing - uncomment to test with small subset
    # analyzer = quick_test_analysis()
    
    # For full analysis
    analyzer = run_individual_subject_analysis()
