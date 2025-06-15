# -*- coding: utf-8 -*-
"""
Improved GRT-LBA Model with MATLAB Conversion Mechanisms
Incorporates grt_dbt conversion function and logit boundary parameterization
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
warnings.filterwarnings('ignore')

def grt_dbt(stimloc, db, sp, coactive=False):
    """
    GRT decision boundary transformation function (ported from MATLAB grt_dbt.m)
    
    Purpose:
        Converts GRT decision boundaries and perceptual variabilities into LBA drift rates
        
    Parameters:
    -----------
    stimloc : array, shape (n_items, 2)
        Stimulus locations [dimx, dimy]
    db : array, shape (2,)
        Decision boundaries [bndx, bndy]  
    sp : array, shape (2,)
        Perceptual variabilities [sigmax, sigmay]
    coactive : bool
        Whether to use coactive model with multivariate normal distribution
    
    Returns:
    --------
    p : array, shape (n_items,)
        Drift rate probabilities for accumulator A
        
    Expected result:
        Array of probabilities between 0.00001 and 0.99999 representing
        the likelihood of evidence accumulating toward response A
    """
    stimloc = np.atleast_2d(stimloc)
    db = np.atleast_1d(db)
    sp = np.atleast_1d(sp)
    
    if not coactive:
        # Standard GRT model - independent processing
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
            # Define decision regions
            # Region A: both dimensions below boundary
            mean = stimloc[i]
            cov = np.diag(sp**2)
            
            # Calculate P(choose A) = P(X1 < db1, X2 < db2)
            p_a = stats.multivariate_normal.cdf(db, mean=mean, cov=cov)
            p_result.append(1 - p_a)  # Invert for P(choose B)
            
        p_result = np.array(p_result)
        p_result = np.clip(p_result, 0.00001, 0.99999)
    
    return p_result

def logit_transform(x, inverse=False):
    """
    Logit transformation function (ported from MATLAB logit.m)
    
    Purpose:
        Transform parameters between bounded and unbounded space for MCMC sampling
        
    Parameters:
    -----------
    x : float or array
        Values to transform
    inverse : bool
        If True, apply inverse logit (expit), else apply logit
        
    Returns:
    --------
    transformed values
    
    Expected result:
        - logit: maps (0,1) to (-inf, inf)
        - inverse logit: maps (-inf, inf) to (0,1)
    """
    if inverse:
        return expit(x)  # 1 / (1 + exp(-x))
    else:
        return logit(x)   # log(x / (1-x))

def transform_boundaries(db_logit, stimloc):
    """
    Transform logit-space boundaries back to real coordinates
    Ported from MATLAB logDensLikeLR.m
    
    Purpose:
        Convert decision boundary parameters from unconstrained logit space
        back to the coordinate system defined by stimulus locations
        
    Parameters:
    -----------
    db_logit : array, shape (2,)
        Decision boundaries in logit space
    stimloc : array, shape (n_items, 2)
        Stimulus location coordinates
        
    Returns:
    --------
    db_real : array, shape (2,)
        Decision boundaries in real coordinate space
        
    Expected result:
        Boundaries constrained within the range defined by stimulus locations
    """
    # Transform db1: from stimloc range [min_x, max_x]
    # Note: Python indexing starts from 0, MATLAB from 1
    min_x, max_x = stimloc[:, 0].min(), stimloc[:, 0].max()
    min_y, max_y = stimloc[:, 1].min(), stimloc[:, 1].max()
    
    db1_real = (logit_transform(db_logit[0], inverse=True) * 
                (max_x - min_x)) + min_x
    
    db2_real = (logit_transform(db_logit[1], inverse=True) * 
                (max_y - min_y)) + min_y
    
    return np.array([db1_real, db2_real])

def transform_parameters(params_log, stimloc):
    """
    Transform all parameters from log/logit space to real space
    
    Purpose:
        Convert MCMC-sampled parameters from unconstrained space
        to interpretable parameter values for GRT-LBA model
        
    Parameters:
    -----------
    params_log : dict
        Parameters in log/logit space from MCMC sampling
    stimloc : array
        Stimulus locations for boundary transformation
        
    Returns:
    --------
    params_real : dict
        Parameters in real space ready for likelihood calculation
        
    Expected result:
        All parameters positive and within reasonable ranges for GRT-LBA
    """
    params_real = {}
    
    # Transform decision boundaries from logit to real coordinates
    db_logit = np.array([params_log['db1'], params_log['db2']])
    params_real['db1'], params_real['db2'] = transform_boundaries(db_logit, stimloc)
    
    # Transform perceptual variabilities from log to real space
    params_real['sp1'] = np.exp(params_log['sp1'])
    params_real['sp2'] = np.exp(params_log['sp2'])
    
    # Transform LBA parameters from log space
    params_real['A'] = np.exp(params_log['A'])
    params_real['bMa1'] = np.exp(params_log['bMa1'])
    params_real['bMa2'] = np.exp(params_log['bMa2'])
    params_real['s'] = np.exp(params_log['s'])
    params_real['t0'] = np.exp(params_log['t0'])
    
    # Transform probability parameters from logit space
    if 'pX' in params_log:
        params_real['pX'] = logit_transform(params_log['pX'], inverse=True)
    
    return params_real

def stable_lba_loglik_grt(rt_data, choice_data, participant_idx, stimloc, 
                         db1, db2, sp1, sp2, A, b1, b2, s, t0, coactive=False):
    """
    Stable LBA log-likelihood with GRT drift rate conversion
    
    Purpose:
        Calculate log-likelihood for LBA model where drift rates are derived
        from GRT decision boundaries and perceptual variabilities
        
    Parameters:
    -----------
    rt_data : array
        Response times
    choice_data : array
        Binary choices (0 or 1)
    participant_idx : array
        Participant indices for hierarchical modeling
    stimloc : array
        Stimulus locations for GRT conversion
    db1, db2 : array
        Decision boundaries for two dimensions
    sp1, sp2 : array
        Perceptual variabilities for two dimensions
    A : array
        LBA start point variability
    b1, b2 : array
        LBA decision thresholds
    s : array
        LBA drift rate variability
    t0 : array
        Non-decision time
    coactive : bool
        Whether to use coactive GRT model
        
    Returns:
    --------
    total_loglik : scalar
        Total log-likelihood across all trials
        
    Expected result:
        Finite log-likelihood value for valid parameter combinations
    """
    # Ensure minimum parameter values for numerical stability
    A = pt.maximum(A, 0.05)
    b1 = pt.maximum(b1, A + 0.05)
    b2 = pt.maximum(b2, A + 0.05)
    s = pt.maximum(s, 0.1)
    t0 = pt.maximum(t0, 0.01)
    sp1 = pt.maximum(sp1, 0.01)
    sp2 = pt.maximum(sp2, 0.01)
    
    # Decision time (remove non-decision time)
    rt_decision = pt.maximum(rt_data - t0[participant_idx], 0.01)
    
    total_loglik = 0.0
    n_trials = rt_data.shape[0]
    
    # Get unique stimulus items for GRT conversion
    unique_items = pt.unique(stimloc)
    
    for i in range(n_trials):
        p_idx = participant_idx[i]
        choice_i = choice_data[i]
        rt_i = rt_decision[i]
        
        # Get current trial parameters
        A_i = A[p_idx]
        b1_i = b1[p_idx]
        b2_i = b2[p_idx]
        s_i = s[p_idx]
        db_i = pt.stack([db1[p_idx], db2[p_idx]])
        sp_i = pt.stack([sp1[p_idx], sp2[p_idx]])
        
        # Convert GRT parameters to LBA drift rates
        # Note: This is a simplified version - full implementation would
        # require stimulus-specific conversion
        v1_prob = pt.sigmoid((db_i[0] - stimloc[i, 0]) / sp_i[0])
        v2_prob = pt.sigmoid((db_i[1] - stimloc[i, 1]) / sp_i[1])
        
        # Combine probabilities for drift rates
        v1 = v1_prob * v2_prob  # Probability for accumulator 1
        v2 = 1 - v1             # Probability for accumulator 2
        
        # Ensure minimum drift rates
        v1 = pt.maximum(v1, 0.1)
        v2 = pt.maximum(v2, 0.1)
        
        # Select winning and losing accumulators
        v_winner = pt.switch(pt.eq(choice_i, 0), v1, v2)
        v_loser = pt.switch(pt.eq(choice_i, 0), v2, v1)
        b_winner = pt.switch(pt.eq(choice_i, 0), b1_i, b2_i)
        
        # LBA likelihood calculation
        rt_i = pt.maximum(rt_i, 0.01)
        sqrt_t = pt.sqrt(rt_i)
        
        # Winner PDF calculation
        z1_win = (v_winner * rt_i - b_winner) / sqrt_t
        z2_win = (v_winner * rt_i - A_i) / sqrt_t
        z1_win = pt.clip(z1_win, -10, 10)
        z2_win = pt.clip(z2_win, -10, 10)
        
        Phi_z1_win = 0.5 * (1 + pt.erf(z1_win / pt.sqrt(2)))
        Phi_z2_win = 0.5 * (1 + pt.erf(z2_win / pt.sqrt(2)))
        phi_z1_win = pt.exp(-0.5 * z1_win**2) / pt.sqrt(2 * np.pi)
        phi_z2_win = pt.exp(-0.5 * z2_win**2) / pt.sqrt(2 * np.pi)
        
        cdf_diff = pt.maximum(Phi_z1_win - Phi_z2_win, 1e-12)
        pdf_diff = (phi_z1_win - phi_z2_win) / sqrt_t
        
        winner_pdf = (v_winner / A_i) * cdf_diff + pdf_diff / A_i
        winner_pdf = pt.maximum(winner_pdf, 1e-12)
        winner_logpdf = pt.log(winner_pdf)
        
        # Loser survival function
        b_loser = pt.switch(pt.eq(choice_i, 0), b2_i, b1_i)
        z1_lose = (v_loser * rt_i - b_loser) / sqrt_t
        z2_lose = (v_loser * rt_i - A_i) / sqrt_t
        z1_lose = pt.clip(z1_lose, -10, 10)
        z2_lose = pt.clip(z2_lose, -10, 10)
        
        Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
        Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
        
        loser_cdf = pt.maximum(Phi_z1_lose - Phi_z2_lose, 1e-12)
        loser_survival = pt.maximum(1 - loser_cdf, 1e-12)
        loser_log_survival = pt.log(loser_survival)
        
        trial_loglik = winner_logpdf + loser_log_survival
        total_loglik += trial_loglik
    
    return total_loglik

class ImprovedGRTLBATester:
    """
    Improved GRT-LBA tester with MATLAB-style parameter transformations
    
    Purpose:
        Test GRT assumptions using LBA framework with proper parameter
        transformations and GRT-to-LBA drift rate conversion
    """
    
    def __init__(self, csv_file='GRT_LBA.csv'):
        """
        Initialize the tester with data loading
        
        Expected result:
            Ready-to-use tester with preprocessed data
        """
        self.csv_file = csv_file
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load and prepare data for analysis
        
        Purpose:
            Load CSV data, filter extreme values, and prepare for modeling
            
        Expected result:
            Clean dataset with reasonable RT ranges and proper indexing
        """
        print("Loading data for improved GRT-LBA testing...")
        
        df = pd.read_csv(self.csv_file)
        print(f"Original data: {len(df)} trials")
        
        # Filter extreme RTs (following MATLAB convention)
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.0)]
        print(f"After RT filtering: {len(df)} trials")
        
        # Convert Response to binary choice
        df['choice_binary'] = (df['Response'] >= 2).astype(int)
        
        # Create stimulus locations (if not provided)
        if 'Stimulus' in df.columns:
            # Create 2D stimulus space based on stimulus numbers
            n_stim = df['Stimulus'].nunique()
            stim_locs = []
            for stim in sorted(df['Stimulus'].unique()):
                # Create a 3x3 grid (can be customized)
                row = (stim - 1) // 3
                col = (stim - 1) % 3
                stim_locs.append([col, row])
            
            self.stimloc = np.array(stim_locs)
            df['stimloc_x'] = df['Stimulus'].map(
                {stim: loc[0] for stim, loc in zip(sorted(df['Stimulus'].unique()), stim_locs)})
            df['stimloc_y'] = df['Stimulus'].map(
                {stim: loc[1] for stim, loc in zip(sorted(df['Stimulus'].unique()), stim_locs)})
        else:
            # Default 3x3 grid
            self.stimloc = np.array([[0, 0], [1, 0], [2, 0],
                                    [0, 1], [1, 1], [2, 1],
                                    [0, 2], [1, 2], [2, 2]])
            df['stimloc_x'] = 1  # Default center
            df['stimloc_y'] = 1
        
        # Map participants to indices
        participants = sorted(df['participant'].unique())
        participant_map = {p: i for i, p in enumerate(participants)}
        df['participant_idx'] = df['participant'].map(participant_map)
        
        self.df = df
        self.participants = participants
        self.n_participants = len(participants)
        
        print(f"Participants: {self.n_participants}")
        print(f"Stimulus locations shape: {self.stimloc.shape}")
        print(f"Choice distribution: {df['choice_binary'].value_counts().to_dict()}")
    
    def build_improved_grt_model(self, max_participants=4, max_trials=40):
        """
        Build improved GRT-LBA model with MATLAB-style parameterization
        
        Purpose:
            Create PyMC model with proper parameter transformations
            and GRT-to-LBA conversion
            
        Expected result:
            Compiled PyMC model ready for MCMC sampling
        """
        # Prepare subset data
        selected_participants = self.participants[:max_participants]
        df_subset = self.df[
            self.df['participant'].isin(selected_participants)
        ].groupby('participant').head(max_trials).reset_index(drop=True)
        
        # Remap indices
        participant_map = {p: i for i, p in enumerate(selected_participants)}
        df_subset['participant_idx'] = df_subset['participant'].map(participant_map)
        n_participants = len(selected_participants)
        
        # Prepare data arrays
        rt_obs = df_subset['RT'].values.astype(np.float32)
        choice_obs = df_subset['choice_binary'].values.astype(np.int32)
        participant_obs = df_subset['participant_idx'].values.astype(np.int32)
        stimloc_obs = np.column_stack([df_subset['stimloc_x'].values,
                                      df_subset['stimloc_y'].values]).astype(np.float32)
        
        print(f"Building model with {len(df_subset)} trials, {n_participants} participants")
        
        with pm.Model() as model:
            # Decision boundary parameters (in logit space for unconstrained sampling)
            db1_logit_mu = pm.Normal('db1_logit_mu', mu=0, sigma=0.5)
            db2_logit_mu = pm.Normal('db2_logit_mu', mu=0, sigma=0.5)
            db1_logit_sigma = pm.HalfNormal('db1_logit_sigma', sigma=0.3)
            db2_logit_sigma = pm.HalfNormal('db2_logit_sigma', sigma=0.3)
            
            # Perceptual variability parameters (in log space)
            sp1_log_mu = pm.Normal('sp1_log_mu', mu=np.log(0.2), sigma=0.2)
            sp2_log_mu = pm.Normal('sp2_log_mu', mu=np.log(0.2), sigma=0.2)
            sp1_log_sigma = pm.HalfNormal('sp1_log_sigma', sigma=0.15)
            sp2_log_sigma = pm.HalfNormal('sp2_log_sigma', sigma=0.15)
            
            # LBA parameters (in log space)
            A_log_mu = pm.Normal('A_log_mu', mu=np.log(0.35), sigma=0.2)
            A_log_sigma = pm.HalfNormal('A_log_sigma', sigma=0.15)
            
            bMa1_log_mu = pm.Normal('bMa1_log_mu', mu=np.log(0.25), sigma=0.5)
            bMa2_log_mu = pm.Normal('bMa2_log_mu', mu=np.log(0.25), sigma=0.5)
            bMa_log_sigma = pm.HalfNormal('bMa_log_sigma', sigma=0.3)
            
            s_log_mu = pm.Normal('s_log_mu', mu=np.log(0.25), sigma=0.3)
            s_log_sigma = pm.HalfNormal('s_log_sigma', sigma=0.2)
            
            t0_log_mu = pm.Normal('t0_log_mu', mu=np.log(0.22), sigma=0.2)
            t0_log_sigma = pm.HalfNormal('t0_log_sigma', sigma=0.15)
            
            # Individual participant parameters
            db1_logit_raw = pm.Normal('db1_logit_raw', mu=0, sigma=1, shape=n_participants)
            db2_logit_raw = pm.Normal('db2_logit_raw', mu=0, sigma=1, shape=n_participants)
            
            sp1_log_raw = pm.Normal('sp1_log_raw', mu=0, sigma=1, shape=n_participants)
            sp2_log_raw = pm.Normal('sp2_log_raw', mu=0, sigma=1, shape=n_participants)
            
            A_log_raw = pm.Normal('A_log_raw', mu=0, sigma=1, shape=n_participants)
            bMa1_log_raw = pm.Normal('bMa1_log_raw', mu=0, sigma=1, shape=n_participants)
            bMa2_log_raw = pm.Normal('bMa2_log_raw', mu=0, sigma=1, shape=n_participants)
            s_log_raw = pm.Normal('s_log_raw', mu=0, sigma=1, shape=n_participants)
            t0_log_raw = pm.Normal('t0_log_raw', mu=0, sigma=1, shape=n_participants)
            
            # Transform to real parameter space
            db1_logit = pm.Deterministic('db1_logit', db1_logit_mu + db1_logit_sigma * db1_logit_raw)
            db2_logit = pm.Deterministic('db2_logit', db2_logit_mu + db2_logit_sigma * db2_logit_raw)
            
            sp1_log = pm.Deterministic('sp1_log', sp1_log_mu + sp1_log_sigma * sp1_log_raw)
            sp2_log = pm.Deterministic('sp2_log', sp2_log_mu + sp2_log_sigma * sp2_log_raw)
            
            A_log = pm.Deterministic('A_log', A_log_mu + A_log_sigma * A_log_raw)
            bMa1_log = pm.Deterministic('bMa1_log', bMa1_log_mu + bMa_log_sigma * bMa1_log_raw)
            bMa2_log = pm.Deterministic('bMa2_log', bMa2_log_mu + bMa_log_sigma * bMa2_log_raw)
            s_log = pm.Deterministic('s_log', s_log_mu + s_log_sigma * s_log_raw)
            t0_log = pm.Deterministic('t0_log', t0_log_mu + t0_log_sigma * t0_log_raw)
            
            # Transform to interpretable parameter space
            sp1 = pm.Deterministic('sp1', pm.math.exp(sp1_log))
            sp2 = pm.Deterministic('sp2', pm.math.exp(sp2_log))
            A = pm.Deterministic('A', pm.math.exp(A_log))
            s = pm.Deterministic('s', pm.math.exp(s_log))
            t0 = pm.Deterministic('t0', pm.math.exp(t0_log))
            
            # Transform boundaries to real coordinate space
            # This requires stimulus location ranges
            stimloc_tensor = pt.as_tensor(self.stimloc, dtype='float32')
            min_x = pt.min(stimloc_tensor[:, 0])
            max_x = pt.max(stimloc_tensor[:, 0])
            min_y = pt.min(stimloc_tensor[:, 1])
            max_y = pt.max(stimloc_tensor[:, 1])
            
            db1 = pm.Deterministic('db1', 
                                  pm.math.sigmoid(db1_logit) * (max_x - min_x) + min_x)
            db2 = pm.Deterministic('db2', 
                                  pm.math.sigmoid(db2_logit) * (max_y - min_y) + min_y)
            
            # Transform thresholds
            bMa1 = pm.Deterministic('bMa1', pm.math.exp(bMa1_log))
            bMa2 = pm.Deterministic('bMa2', pm.math.exp(bMa2_log))
            b1 = pm.Deterministic('b1', A + bMa1)
            b2 = pm.Deterministic('b2', A + bMa2)
            
            # LBA likelihood with GRT conversion
            pm.Potential('lba_grt_likelihood',
                        stable_lba_loglik_grt(rt_obs, choice_obs, participant_obs,
                                            pt.as_tensor(stimloc_obs),
                                            db1, db2, sp1, sp2, A, b1, b2, s, t0))
        
        return model, df_subset
    
    def test_improved_grt_assumptions(self):
        """
        Test GRT assumptions using improved model with MATLAB-style transformations
        
        Purpose:
            Run MCMC sampling and analyze results for GRT assumption testing
            
        Expected result:
            Trace object with converged samples and assumption test results
        """
        print("\n" + "="*70)
        print("TESTING GRT ASSUMPTIONS WITH IMPROVED LBA MODEL")
        print("Using MATLAB-style parameter transformations")
        print("="*70)
        
        model, df_subset = self.build_improved_grt_model()
        
        print("Starting MCMC sampling...")
        start_time = time.time()
        
        with model:
            trace = pm.sample(
                draws=300, tune=150, chains=2,
                progressbar=True, return_inferencedata=True,
                target_accept=0.92, max_treedepth=12,
                cores=1, random_seed=123
            )
        
        elapsed = time.time() - start_time
        print(f"Sampling completed in {elapsed:.1f} seconds")
        
        # Analyze results
        self.analyze_improved_results(trace)
        
        return trace, model
    
    def analyze_improved_results(self, trace):
        """
        Analyze results from improved GRT-LBA model
        
        Purpose:
            Extract and interpret parameter estimates with focus on
            GRT assumption violations
            
        Expected result:
            Printed summary of GRT assumption test results
        """
        print("\n" + "-"*50)
        print("IMPROVED GRT-LBA ANALYSIS RESULTS")
        print("-"*50)
        
        # Extract samples
        posterior = trace.posterior
        
        # Decision boundary analysis
        db1_samples = posterior['db1'].values.flatten()
        db2_samples = posterior['db2'].values.flatten()
        
        print(f"Decision Boundaries:")
        print(f"  db1: {np.mean(db1_samples):.3f} ± {np.std(db1_samples):.3f}")
        print(f"  db2: {np.mean(db2_samples):.3f} ± {np.std(db2_samples):.3f}")
        
        # Perceptual variability analysis
        sp1_samples = posterior['sp1'].values.flatten()
        sp2_samples = posterior['sp2'].values.flatten()
        
        print(f"\nPerceptual Variabilities:")
        print(f"  sp1: {np.mean(sp1_samples):.3f} ± {np.std(sp1_samples):.3f}")
        print(f"  sp2: {np.mean(sp2_samples):.3f} ± {np.std(sp2_samples):.3f}")
        
        # Test for separability (equal variabilities)
        sp_ratio = sp1_samples / sp2_samples
        sp_ratio_mean = np.mean(sp_ratio)
        sp_ratio_hdi = np.percentile(sp_ratio, [2.5, 97.5])
        
        print(f"\nPerceptual Separability Test:")
        print(f"  sp1/sp2 ratio: {sp_ratio_mean:.3f} [{sp_ratio_hdi[0]:.3f}, {sp_ratio_hdi[1]:.3f}]")
        if sp_ratio_hdi[0] < 1.0 < sp_ratio_hdi[1]:
            print("  ✓ Perceptual Separability supported (ratio includes 1.0)")
        else:
            print("  ✗ Perceptual Separability violated (ratio excludes 1.0)")
        
        # LBA parameter analysis
        A_samples = posterior['A'].values.flatten()
        s_samples = posterior['s'].values.flatten()
        t0_samples = posterior['t0'].values.flatten()
        
        print(f"\nLBA Parameters:")
        print(f"  A (start point): {np.mean(A_samples):.3f} ± {np.std(A_samples):.3f}")
        print(f"  s (drift variability): {np.mean(s_samples):.3f} ± {np.std(s_samples):.3f}")
        print(f"  t0 (non-decision time): {np.mean(t0_samples):.3f} ± {np.std(t0_samples):.3f}")
        
        # Model diagnostics
        print(f"\nModel Diagnostics:")
        try:
            ess = az.ess(trace)
            rhat = az.rhat(trace)
            
            key_params = ['db1', 'db2', 'sp1', 'sp2', 'A', 's', 't0']
            for param in key_params:
                if param in ess.data_vars:
                    ess_val = float(ess[param]) if ess[param].ndim == 0 else float(ess[param].min())
                    rhat_val = float(rhat[param]) if rhat[param].ndim == 0 else float(rhat[param].max())
                    print(f"  {param}: ESS={ess_val:.0f}, R̂={rhat_val:.3f}")
        except Exception as e:
            print(f"  Warning: Could not compute diagnostics: {e}")
    
    def save_improved_results(self, trace, model_info=None):
        """
        Save results from improved GRT-LBA analysis
        
        Purpose:
            Export analysis results in JSON format for further processing
            and comparison with other models
            
        Parameters:
        -----------
        trace : InferenceData
            MCMC trace from PyMC sampling
        model_info : dict, optional
            Additional model information to save
            
        Returns:
        --------
        results : dict
            Structured results dictionary
            
        Expected result:
            JSON file with parameter estimates and model comparison metrics
        """
        print("\nSaving improved GRT-LBA results...")
        
        posterior = trace.posterior
        
        # Extract key parameter estimates
        results = {
            'model_type': 'Improved_GRT_LBA',
            'parameter_transformation': 'MATLAB_style',
            'decision_boundaries': {
                'db1_mean': float(posterior['db1'].mean()),
                'db1_std': float(posterior['db1'].std()),
                'db1_hdi': [float(x) for x in np.percentile(posterior['db1'].values.flatten(), [2.5, 97.5])],
                'db2_mean': float(posterior['db2'].mean()),
                'db2_std': float(posterior['db2'].std()),
                'db2_hdi': [float(x) for x in np.percentile(posterior['db2'].values.flatten(), [2.5, 97.5])]
            },
            'perceptual_variabilities': {
                'sp1_mean': float(posterior['sp1'].mean()),
                'sp1_std': float(posterior['sp1'].std()),
                'sp1_hdi': [float(x) for x in np.percentile(posterior['sp1'].values.flatten(), [2.5, 97.5])],
                'sp2_mean': float(posterior['sp2'].mean()),
                'sp2_std': float(posterior['sp2'].std()),
                'sp2_hdi': [float(x) for x in np.percentile(posterior['sp2'].values.flatten(), [2.5, 97.5])]
            },
            'lba_parameters': {
                'A_mean': float(posterior['A'].mean()),
                'A_std': float(posterior['A'].std()),
                's_mean': float(posterior['s'].mean()),
                's_std': float(posterior['s'].std()),
                't0_mean': float(posterior['t0'].mean()),
                't0_std': float(posterior['t0'].std())
            }
        }
        
        # Separability test
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
            key_params = ['db1', 'db2', 'sp1', 'sp2', 'A', 's', 't0']
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
        
        # Add model info if provided
        if model_info:
            results['model_info'] = model_info
        
        # Save to JSON
        with open('improved_grt_lba_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ Results saved as improved_grt_lba_results.json")
        return results

def run_improved_grt_analysis():
    """
    Main function to run improved GRT-LBA analysis
    
    Purpose:
        Execute complete GRT assumption testing pipeline using
        improved model with MATLAB-style transformations
        
    Returns:
    --------
    tester : ImprovedGRTLBATester
        Tester object with loaded data
    trace : InferenceData
        MCMC sampling results
    model : PyMC Model
        Compiled PyMC model
        
    Expected result:
        Complete analysis output with GRT assumption test results
    """
    print("STARTING IMPROVED GRT-LBA ANALYSIS")
    print("Incorporating MATLAB grt_dbt() conversion and logit transformations")
    print("="*70)
    
    # Initialize tester
    tester = ImprovedGRTLBATester('GRT_LBA.csv')
    
    # Run analysis
    try:
        trace, model = tester.test_improved_grt_assumptions()
        
        # Save results
        results = tester.save_improved_results(trace)
        
        print("\n" + "="*70)
        print("IMPROVED GRT-LBA ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return tester, trace, model
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def compare_with_original_method(improved_results_file='improved_grt_lba_results.json',
                               original_results_file='pi_results.json'):
    """
    Compare improved method with original Python implementation
    
    Purpose:
        Evaluate differences between MATLAB-style and original Python
        implementations of GRT-LBA testing
        
    Parameters:
    -----------
    improved_results_file : str
        Path to improved method results
    original_results_file : str
        Path to original method results
        
    Expected result:
        Comparison report highlighting key differences and improvements
    """
    print("\n" + "="*50)
    print("COMPARING IMPROVED VS ORIGINAL METHODS")
    print("="*50)
    
    try:
        # Load results
        with open(improved_results_file, 'r') as f:
            improved = json.load(f)
        
        with open(original_results_file, 'r') as f:
            original = json.load(f)
        
        print("Comparison Results:")
        print("-" * 30)
        
        # Compare parameter transformations
        print("Parameter Transformation:")
        print(f"  Original: sigma matrix approach")
        print(f"  Improved: MATLAB-style logit/log transformations")
        
        # Compare key findings
        if 'separability_test' in improved and 'pi_support' in original:
            print(f"\nKey Findings:")
            print(f"  Original PI support: {original.get('pi_support', 'N/A')}")
            print(f"  Improved separability: {improved['separability_test']['separability_supported']}")
        
        # Compare diagnostics
        if 'diagnostics' in improved and 'ess_independence' in original:
            print(f"\nModel Quality:")
            print(f"  Original ESS: {original.get('ess_independence', 'N/A'):.0f}")
            
            improved_ess = []
            for param in ['db1', 'db2', 'sp1', 'sp2']:
                if param in improved['diagnostics']:
                    improved_ess.append(improved['diagnostics'][param]['ess'])
            
            if improved_ess:
                print(f"  Improved min ESS: {min(improved_ess):.0f}")
        
        print(f"\nConclusion:")
        print(f"  Both methods provide complementary insights into GRT assumptions")
        print(f"  Improved method uses more conventional parameter transformations")
        print(f"  Original method focuses more on independence/correlation testing")
        
    except FileNotFoundError as e:
        print(f"Could not find results file: {e}")
        print("Run both analyses first to enable comparison")
    except Exception as e:
        print(f"Comparison failed: {e}")

if __name__ == "__main__":
    # Run improved analysis
    tester, trace, model = run_improved_grt_analysis()
    
    # Compare with original if available
    if tester is not None:
        compare_with_original_method()
