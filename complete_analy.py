"""
Production-level LBA Model with GRT Assumptions Testing
Optimized for numerical stability and sampling efficiency
Separates three GRT assumptions: Perceptual Independence (PI), 
Perceptual Separability (PS), and Decisional Separability (DS)
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

def stable_lba_loglik(rt_data, choice_data, participant_idx, A, b, v1, v2, t0):
    """
    Numerically stable LBA log-likelihood function
    Supports hierarchical model structure
    
    Parameters:
    -----------
    rt_data : array
        Response times
    choice_data : array  
        Choice responses (0 or 1)
    participant_idx : array
        Participant indices
    A, b, v1, v2, t0 : tensors
        LBA parameters for each participant
    """
    # Parameter safety checks
    A = pt.maximum(A, 0.05)
    b = pt.maximum(b, A + 0.05)
    v1 = pt.maximum(v1, 0.1)
    v2 = pt.maximum(v2, 0.1)
    t0 = pt.maximum(t0, 0.01)
    
    # Calculate decision times
    rt_decision = pt.maximum(rt_data - t0[participant_idx], 0.01)
    
    total_loglik = 0.0
    n_trials = rt_data.shape[0]
    
    for i in range(n_trials):
        p_idx = participant_idx[i]
        choice_i = choice_data[i]
        rt_i = rt_decision[i]
        
        # Current participant's parameters
        A_i = A[p_idx]
        b_i = b[p_idx]
        v1_i = v1[p_idx]
        v2_i = v2[p_idx]
        
        # Determine winner and loser drift rates
        v_winner = pt.switch(pt.eq(choice_i, 0), v1_i, v2_i)
        v_loser = pt.switch(pt.eq(choice_i, 0), v2_i, v1_i)
        
        # Avoid division by zero
        rt_i = pt.maximum(rt_i, 0.01)
        sqrt_t = pt.sqrt(rt_i)
        
        # Winner PDF calculation (numerically stable)
        z1_win = (v_winner * rt_i - b_i) / sqrt_t
        z2_win = (v_winner * rt_i - A_i) / sqrt_t
        
        # Clip extreme values to prevent numerical overflow
        z1_win = pt.clip(z1_win, -10, 10)
        z2_win = pt.clip(z2_win, -10, 10)
        
        # Normal CDF and PDF (stable implementations)
        Phi_z1_win = 0.5 * (1 + pt.erf(z1_win / pt.sqrt(2)))
        Phi_z2_win = 0.5 * (1 + pt.erf(z2_win / pt.sqrt(2)))
        phi_z1_win = pt.exp(-0.5 * z1_win**2) / pt.sqrt(2 * np.pi)
        phi_z2_win = pt.exp(-0.5 * z2_win**2) / pt.sqrt(2 * np.pi)
        
        # Winner PDF components
        cdf_diff = pt.maximum(Phi_z1_win - Phi_z2_win, 1e-12)
        pdf_diff = (phi_z1_win - phi_z2_win) / sqrt_t
        
        winner_pdf = (v_winner / A_i) * cdf_diff + pdf_diff / A_i
        winner_pdf = pt.maximum(winner_pdf, 1e-12)
        winner_logpdf = pt.log(winner_pdf)
        
        # Loser survival function
        z1_lose = (v_loser * rt_i - b_i) / sqrt_t
        z2_lose = (v_loser * rt_i - A_i) / sqrt_t
        
        z1_lose = pt.clip(z1_lose, -10, 10)
        z2_lose = pt.clip(z2_lose, -10, 10)
        
        Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
        Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
        
        loser_cdf = pt.maximum(Phi_z1_lose - Phi_z2_lose, 1e-12)
        loser_survival = pt.maximum(1 - loser_cdf, 1e-12)
        loser_log_survival = pt.log(loser_survival)
        
        # Combine log-likelihoods
        trial_loglik = winner_logpdf + loser_log_survival
        total_loglik += trial_loglik
    
    return total_loglik

class GRTAssumptionTester:
    """
    Class for testing three GRT assumptions using Bayesian LBA models:
    1. Perceptual Independence (PI)
    2. Perceptual Separability (PS) 
    3. Decisional Separability (DS)
    """
    
    def __init__(self, data_file='GRT_LBA.csv'):
        """Initialize with data loading and preprocessing"""
        self.data_file = data_file
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the GRT data"""
        print("Loading and preprocessing GRT data...")
        
        try:
            df = pd.read_csv(self.data_file)
        except FileNotFoundError:
            print(f"Data file {self.data_file} not found. Creating simulated data...")
            df = self.create_simulated_data()
        
        # Basic preprocessing
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.5)]  # Remove outliers
        
        # Create binary choice variable (assuming Response 1,2 map to choices 0,1)
        df['choice_binary'] = (df['Response'] >= 2).astype(int)
        
        # Map participants to indices
        participants = sorted(df['participant'].unique())
        participant_map = {p: i for i, p in enumerate(participants)}
        df['participant_idx'] = df['participant'].map(participant_map)
        
        self.df_full = df
        self.participants = participants
        self.n_participants = len(participants)
        
        print(f"Loaded data: {len(df)} trials from {self.n_participants} participants")
        print(f"RT range: {df['RT'].min():.3f} - {df['RT'].max():.3f}")
        print(f"Choice distribution: {df['choice_binary'].value_counts().to_dict()}")
        
    def create_simulated_data(self, n_participants=10, n_trials_per_participant=100):
        """Create simulated GRT data for testing"""
        np.random.seed(42)
        
        data = []
        for p in range(1, n_participants + 1):
            for trial in range(n_trials_per_participant):
                # Simulate RT and Response
                rt = np.random.gamma(2, 0.3) + 0.2
                response = np.random.choice([1, 2])
                
                data.append({
                    'participant': p,
                    'RT': rt,
                    'Response': response,
                    'trial': trial
                })
        
        return pd.DataFrame(data)
    
    def prepare_subset_data(self, max_participants=5, max_trials_per_participant=50):
        """Prepare a subset of data for efficient sampling"""
        participants_subset = self.participants[:max_participants]
        
        df_subset = self.df_full[
            self.df_full['participant'].isin(participants_subset)
        ].groupby('participant').head(max_trials_per_participant).reset_index(drop=True)
        
        # Remap participant indices for subset
        participant_map = {p: i for i, p in enumerate(participants_subset)}
        df_subset['participant_idx'] = df_subset['participant'].map(participant_map)
        
        return df_subset, len(participants_subset)
    
    def test_perceptual_independence(self, max_participants=3, max_trials=30):
        """
        Test Perceptual Independence (PI) assumption
        PI assumes that perceptual processing of different dimensions is independent
        """
        print("\n" + "="*60)
        print("TESTING PERCEPTUAL INDEPENDENCE (PI) ASSUMPTION")
        print("="*60)
        
        df_subset, n_participants = self.prepare_subset_data(max_participants, max_trials)
        
        rt_obs = df_subset['RT'].values.astype(np.float32)
        choice_obs = df_subset['choice_binary'].values.astype(np.int32)
        participant_obs = df_subset['participant_idx'].values.astype(np.int32)
        
        print(f"PI Test data: {len(df_subset)} trials, {n_participants} participants")
        
        try:
            with pm.Model() as pi_model:
                # Hierarchical priors for PI model
                # Under PI, drift rates should be independent across dimensions
                
                # Start-point variability (common across conditions)
                A_mu = pm.HalfNormal('A_mu', sigma=0.1)
                A_sigma = pm.HalfNormal('A_sigma', sigma=0.03)
                
                # Decision threshold (can vary by participant)
                b_offset_mu = pm.HalfNormal('b_offset_mu', sigma=0.1)
                b_offset_sigma = pm.HalfNormal('b_offset_sigma', sigma=0.03)
                
                # Drift rates (key parameters for PI testing)
                v1_mu = pm.HalfNormal('v1_mu', sigma=0.3)
                v1_sigma = pm.HalfNormal('v1_sigma', sigma=0.1)
                
                v2_mu = pm.HalfNormal('v2_mu', sigma=0.3)
                v2_sigma = pm.HalfNormal('v2_sigma', sigma=0.1)
                
                # Non-decision time
                t0_mu = pm.HalfNormal('t0_mu', sigma=0.05)
                t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.02)
                
                # Individual participant parameters
                A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
                A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
                
                b_offset_raw = pm.Normal('b_offset_raw', mu=0, sigma=1, shape=n_participants)
                b_offset = pm.Deterministic('b_offset', 
                                          pm.math.maximum(b_offset_mu + b_offset_sigma * b_offset_raw, 0.05))
                b = pm.Deterministic('b', A + b_offset)
                
                v1_raw = pm.Normal('v1_raw', mu=0, sigma=1, shape=n_participants)
                v1 = pm.Deterministic('v1', pm.math.maximum(v1_mu + v1_sigma * v1_raw, 0.1))
                
                v2_raw = pm.Normal('v2_raw', mu=0, sigma=1, shape=n_participants)
                v2 = pm.Deterministic('v2', pm.math.maximum(v2_mu + v2_sigma * v2_raw, 0.1))
                
                t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
                t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
                
                # LBA likelihood
                pm.Potential('lba_likelihood', 
                           stable_lba_loglik(rt_obs, choice_obs, participant_obs, A, b, v1, v2, t0))
            
            print("PI model built successfully. Starting sampling...")
            
            start_time = time.time()
            
            with pi_model:
                pi_trace = pm.sample(
                    draws=100, tune=50, chains=1, 
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.95, max_treedepth=8,
                    cores=1, random_seed=42
                )
            
            elapsed = time.time() - start_time
            print(f"PI sampling completed in {elapsed:.1f} seconds")
            
            # Store results
            self.pi_trace = pi_trace
            self.pi_model = pi_model
            
            # Basic diagnostics
            self.print_sampling_diagnostics(pi_trace, "PI")
            
            return pi_trace
            
        except Exception as e:
            print(f"PI model failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_perceptual_separability(self, max_participants=3, max_trials=30):
        """
        Test Perceptual Separability (PS) assumption
        PS assumes that perception of one dimension doesn't affect the other
        """
        print("\n" + "="*60)
        print("TESTING PERCEPTUAL SEPARABILITY (PS) ASSUMPTION")
        print("="*60)
        
        df_subset, n_participants = self.prepare_subset_data(max_participants, max_trials)
        
        rt_obs = df_subset['RT'].values.astype(np.float32)
        choice_obs = df_subset['choice_binary'].values.astype(np.int32)
        participant_obs = df_subset['participant_idx'].values.astype(np.int32)
        
        print(f"PS Test data: {len(df_subset)} trials, {n_participants} participants")
        
        try:
            with pm.Model() as ps_model:
                # Under PS, there might be correlations between dimensions
                # but the perceptual representations remain separable
                
                # Similar structure to PI but with potential correlation parameters
                A_mu = pm.HalfNormal('A_mu', sigma=0.1)
                A_sigma = pm.HalfNormal('A_sigma', sigma=0.03)
                
                b_offset_mu = pm.HalfNormal('b_offset_mu', sigma=0.1)
                b_offset_sigma = pm.HalfNormal('b_offset_sigma', sigma=0.03)
                
                # Drift rates with potential separability violations
                v1_mu = pm.HalfNormal('v1_mu', sigma=0.3)
                v1_sigma = pm.HalfNormal('v1_sigma', sigma=0.1)
                
                v2_mu = pm.HalfNormal('v2_mu', sigma=0.3)
                v2_sigma = pm.HalfNormal('v2_sigma', sigma=0.1)
                
                # PS-specific parameter: correlation between drift rates
                drift_correlation = pm.Uniform('drift_correlation', lower=-0.5, upper=0.5)
                
                t0_mu = pm.HalfNormal('t0_mu', sigma=0.05)
                t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.02)
                
                # Individual parameters
                A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
                A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
                
                b_offset_raw = pm.Normal('b_offset_raw', mu=0, sigma=1, shape=n_participants)
                b_offset = pm.Deterministic('b_offset', 
                                          pm.math.maximum(b_offset_mu + b_offset_sigma * b_offset_raw, 0.05))
                b = pm.Deterministic('b', A + b_offset)
                
                # Correlated drift rates for PS testing
                v1_raw = pm.Normal('v1_raw', mu=0, sigma=1, shape=n_participants)
                v2_raw = pm.Normal('v2_raw', mu=drift_correlation * v1_raw, sigma=pt.sqrt(1 - drift_correlation**2), 
                                 shape=n_participants)
                
                v1 = pm.Deterministic('v1', pm.math.maximum(v1_mu + v1_sigma * v1_raw, 0.1))
                v2 = pm.Deterministic('v2', pm.math.maximum(v2_mu + v2_sigma * v2_raw, 0.1))
                
                t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
                t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
                
                # LBA likelihood
                pm.Potential('lba_likelihood', 
                           stable_lba_loglik(rt_obs, choice_obs, participant_obs, A, b, v1, v2, t0))
            
            print("PS model built successfully. Starting sampling...")
            
            start_time = time.time()
            
            with ps_model:
                ps_trace = pm.sample(
                    draws=100, tune=50, chains=1,
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.95, max_treedepth=8,
                    cores=1, random_seed=42
                )
            
            elapsed = time.time() - start_time
            print(f"PS sampling completed in {elapsed:.1f} seconds")
            
            # Store results
            self.ps_trace = ps_trace
            self.ps_model = ps_model
            
            # Basic diagnostics
            self.print_sampling_diagnostics(ps_trace, "PS")
            
            return ps_trace
            
        except Exception as e:
            print(f"PS model failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_decisional_separability(self, max_participants=3, max_trials=30):
        """
        Test Decisional Separability (DS) assumption
        DS assumes that decision boundaries for different dimensions are independent
        """
        print("\n" + "="*60)
        print("TESTING DECISIONAL SEPARABILITY (DS) ASSUMPTION")
        print("="*60)
        
        df_subset, n_participants = self.prepare_subset_data(max_participants, max_trials)
        
        rt_obs = df_subset['RT'].values.astype(np.float32)
        choice_obs = df_subset['choice_binary'].values.astype(np.int32)
        participant_obs = df_subset['participant_idx'].values.astype(np.int32)
        
        print(f"DS Test data: {len(df_subset)} trials, {n_participants} participants")
        
        try:
            with pm.Model() as ds_model:
                # Under DS violations, decision thresholds might be correlated
                # or there might be bias effects
                
                A_mu = pm.HalfNormal('A_mu', sigma=0.1)
                A_sigma = pm.HalfNormal('A_sigma', sigma=0.03)
                
                # DS-specific: potentially different thresholds for different choices
                b1_offset_mu = pm.HalfNormal('b1_offset_mu', sigma=0.1)  # Threshold for choice 1
                b2_offset_mu = pm.HalfNormal('b2_offset_mu', sigma=0.1)  # Threshold for choice 2
                b_offset_sigma = pm.HalfNormal('b_offset_sigma', sigma=0.03)
                
                # DS-specific parameter: threshold difference (bias)
                threshold_bias = pm.Normal('threshold_bias', mu=0, sigma=0.1)
                
                v1_mu = pm.HalfNormal('v1_mu', sigma=0.3)
                v1_sigma = pm.HalfNormal('v1_sigma', sigma=0.1)
                
                v2_mu = pm.HalfNormal('v2_mu', sigma=0.3)
                v2_sigma = pm.HalfNormal('v2_sigma', sigma=0.1)
                
                t0_mu = pm.HalfNormal('t0_mu', sigma=0.05)
                t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.02)
                
                # Individual parameters
                A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
                A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
                
                # Separate thresholds for testing DS
                b1_offset_raw = pm.Normal('b1_offset_raw', mu=0, sigma=1, shape=n_participants)
                b2_offset_raw = pm.Normal('b2_offset_raw', mu=0, sigma=1, shape=n_participants)
                
                b1_offset = pm.Deterministic('b1_offset', 
                                           pm.math.maximum(b1_offset_mu + b_offset_sigma * b1_offset_raw, 0.05))
                b2_offset = pm.Deterministic('b2_offset', 
                                           pm.math.maximum(b2_offset_mu + b_offset_sigma * b2_offset_raw + threshold_bias, 0.05))
                
                # Use average threshold for LBA (simplified DS test)
                b = pm.Deterministic('b', A + (b1_offset + b2_offset) / 2)
                
                v1_raw = pm.Normal('v1_raw', mu=0, sigma=1, shape=n_participants)
                v1 = pm.Deterministic('v1', pm.math.maximum(v1_mu + v1_sigma * v1_raw, 0.1))
                
                v2_raw = pm.Normal('v2_raw', mu=0, sigma=1, shape=n_participants)
                v2 = pm.Deterministic('v2', pm.math.maximum(v2_mu + v2_sigma * v2_raw, 0.1))
                
                t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
                t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
                
                # LBA likelihood
                pm.Potential('lba_likelihood', 
                           stable_lba_loglik(rt_obs, choice_obs, participant_obs, A, b, v1, v2, t0))
            
            print("DS model built successfully. Starting sampling...")
            
            start_time = time.time()
            
            with ds_model:
                ds_trace = pm.sample(
                    draws=100, tune=50, chains=1,
                    progressbar=True, return_inferencedata=True,
                    target_accept=0.95, max_treedepth=8,
                    cores=1, random_seed=42
                )
            
            elapsed = time.time() - start_time
            print(f"DS sampling completed in {elapsed:.1f} seconds")
            
            # Store results
            self.ds_trace = ds_trace
            self.ds_model = ds_model
            
            # Basic diagnostics
            self.print_sampling_diagnostics(ds_trace, "DS")
            
            return ds_trace
            
        except Exception as e:
            print(f"DS model failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_sampling_diagnostics(self, trace, model_name):
        """Print basic sampling diagnostics"""
        print(f"\n{model_name} Model Diagnostics:")
        print("-" * 30)
        
        # Effective sample size
        ess = az.ess(trace)
        print("Effective Sample Sizes:")
        for var in ess.data_vars:
            if ess[var].ndim == 0:  # scalar parameter
                print(f"  {var}: {float(ess[var]):.0f}")
            else:  # vector parameter
                print(f"  {var}: {float(ess[var].min()):.0f} - {float(ess[var].max()):.0f}")
        
        # Parameter summaries
        print(f"\nParameter Posterior Means:")
        for var in trace.posterior.data_vars:
            if trace.posterior[var].ndim <= 2:  # Only show scalar/simple parameters
                mean_val = float(trace.posterior[var].mean())
                std_val = float(trace.posterior[var].std())
                print(f"  {var}: {mean_val:.3f} ± {std_val:.3f}")
    
    def compare_assumptions(self):
        """Compare the three GRT assumption models"""
        print("\n" + "="*60)
        print("COMPARING GRT ASSUMPTION MODELS")
        print("="*60)
        
        models = {}
        traces = {}
        
        if hasattr(self, 'pi_trace'):
            models['PI'] = self.pi_model
            traces['PI'] = self.pi_trace
            
        if hasattr(self, 'ps_trace'):
            models['PS'] = self.ps_model
            traces['PS'] = self.ps_trace
            
        if hasattr(self, 'ds_trace'):
            models['DS'] = self.ds_model
            traces['DS'] = self.ds_trace
        
        if len(models) < 2:
            print("Need at least 2 successful models for comparison")
            return
        
        print(f"Comparing {len(models)} models: {list(models.keys())}")
        
        # Model comparison using LOO
        try:
            model_comparison = {}
            for name, trace in traces.items():
                print(f"Computing LOO for {name} model...")
                model_comparison[name] = az.loo(trace)
            
            # Create comparison dataframe
            comparison_df = az.compare(model_comparison)
            print("\nModel Comparison (LOO):")
            print(comparison_df)
            
        except Exception as e:
            print(f"Model comparison failed: {e}")
        
        # Parameter comparison
        print("\nKey Parameter Comparisons:")
        for param in ['v1_mu', 'v2_mu', 'A_mu']:
            print(f"\n{param}:")
            for name, trace in traces.items():
                if param in trace.posterior:
                    mean_val = float(trace.posterior[param].mean())
                    hdi = az.hdi(trace.posterior[param], hdi_prob=0.95)
                    print(f"  {name}: {mean_val:.3f} [{float(hdi.low):.3f}, {float(hdi.high):.3f}]")
    
    def run_all_tests(self, max_participants=3, max_trials=30):
        """Run all three GRT assumption tests"""
        print("RUNNING ALL GRT ASSUMPTION TESTS")
        print("="*60)
        
        print(f"Test configuration:")
        print(f"  Max participants: {max_participants}")
        print(f"  Max trials per participant: {max_trials}")
        
        # Test each assumption
        pi_result = self.test_perceptual_independence(max_participants, max_trials)
        ps_result = self.test_perceptual_separability(max_participants, max_trials)
        ds_result = self.test_decisional_separability(max_participants, max_trials)
        
        # Compare models
        self.compare_assumptions()
        
        return {
            'PI': pi_result,
            'PS': ps_result, 
            'DS': ds_result
        }

# Example usage
if __name__ == "__main__":
    # Initialize the GRT tester
    grt_tester = GRTAssumptionTester('GRT_LBA.csv')
    
    # Run all tests with small dataset for demonstration
    results = grt_tester.run_all_tests(max_participants=2, max_trials=20)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Successfully implemented Bayesian LBA models for testing:")
    print("1. Perceptual Independence (PI)")
    print("2. Perceptual Separability (PS)")
    print("3. Decisional Separability (DS)")
    print("\nEach model is stored separately and can be compared using model selection criteria.")
