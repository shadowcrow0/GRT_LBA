# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:18:05 2025

@author: spt904
"""

"""
GRT Perceptual Independence (PI) Testing using Bayesian LBA Model
Part 1 of 3: Tests whether perceptual processing of different dimensions is independent
Uses sigma matrix for covariance structure assessment
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

def stable_lba_loglik(rt_data, choice_data, participant_idx, A, b, v1, v2, t0):
    """Numerically stable LBA log-likelihood function"""
    A = pt.maximum(A, 0.05)
    b = pt.maximum(b, A + 0.05)
    v1 = pt.maximum(v1, 0.1)
    v2 = pt.maximum(v2, 0.1)
    t0 = pt.maximum(t0, 0.01)
    
    rt_decision = pt.maximum(rt_data - t0[participant_idx], 0.01)
    total_loglik = 0.0
    n_trials = rt_data.shape[0]
    
    for i in range(n_trials):
        p_idx = participant_idx[i]
        choice_i = choice_data[i]
        rt_i = rt_decision[i]
        
        A_i = A[p_idx]
        b_i = b[p_idx]
        v1_i = v1[p_idx]
        v2_i = v2[p_idx]
        
        v_winner = pt.switch(pt.eq(choice_i, 0), v1_i, v2_i)
        v_loser = pt.switch(pt.eq(choice_i, 0), v2_i, v1_i)
        
        rt_i = pt.maximum(rt_i, 0.01)
        sqrt_t = pt.sqrt(rt_i)
        
        # Winner PDF calculation
        z1_win = (v_winner * rt_i - b_i) / sqrt_t
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
        z1_lose = (v_loser * rt_i - b_i) / sqrt_t
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

class PerceptualIndependenceTester:
    """Test Perceptual Independence assumption using Bayesian LBA with sigma matrix"""
    
    def __init__(self, csv_file='GRT_LBA.csv'):
        self.csv_file = csv_file
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare data for PI testing"""
        print("Loading data for Perceptual Independence testing...")
        
        df = pd.read_csv(self.csv_file)
        print(f"Original data: {len(df)} trials")
        
        # Filter extreme RTs
        df = df[(df['RT'] > 0.15) & (df['RT'] < 2.0)]
        print(f"After RT filtering: {len(df)} trials")
        
        # Convert Response to binary choice
        df['choice_binary'] = (df['Response'] >= 2).astype(int)
        
        # Map participants to indices
        participants = sorted(df['participant'].unique())
        participant_map = {p: i for i, p in enumerate(participants)}
        df['participant_idx'] = df['participant'].map(participant_map)
        
        self.df = df
        self.participants = participants
        self.n_participants = len(participants)
        
        print(f"Participants: {self.n_participants}")
        print(f"Choice distribution: {df['choice_binary'].value_counts().to_dict()}")
    
    def prepare_analysis_subset(self, max_participants=4, max_trials=40):
        """Prepare subset for analysis"""
        selected_participants = self.participants[:max_participants]
        
        df_subset = self.df[
            self.df['participant'].isin(selected_participants)
        ].groupby('participant').head(max_trials).reset_index(drop=True)
        
        # Remap indices
        participant_map = {p: i for i, p in enumerate(selected_participants)}
        df_subset['participant_idx'] = df_subset['participant'].map(participant_map)
        
        return df_subset, len(selected_participants)
    
    def test_perceptual_independence(self):
        """
        Test Perceptual Independence using Bayesian LBA with sigma matrix
        PI assumes independent perceptual processing across dimensions
        """
        print("\n" + "="*70)
        print("TESTING PERCEPTUAL INDEPENDENCE (PI) ASSUMPTION")
        print("PI: Perceptual processing of different dimensions is independent")
        print("="*70)
        
        # Prepare data
        df_subset, n_participants = self.prepare_analysis_subset()
        
        rt_obs = df_subset['RT'].values.astype(np.float32)
        choice_obs = df_subset['choice_binary'].values.astype(np.int32)
        participant_obs = df_subset['participant_idx'].values.astype(np.int32)
        
        print(f"Analysis data: {len(df_subset)} trials, {n_participants} participants")
        
        # Build Bayesian LBA model for PI
        with pm.Model() as pi_model:
            print("Building PI model with sigma matrix structure...")
            
            # Hierarchical priors for drift rates (key for PI testing)
            v1_mu = pm.HalfNormal('v1_mu', sigma=0.5)
            v2_mu = pm.HalfNormal('v2_mu', sigma=0.5)
            
            # PI-specific: Independent sigma matrix for drift rates
            # Under PI, covariance between v1 and v2 should be zero
            v1_sigma = pm.HalfNormal('v1_sigma', sigma=0.3)
            v2_sigma = pm.HalfNormal('v2_sigma', sigma=0.3)
            
            # PI assumption test: independence correlation
            # Should be close to zero under PI
            drift_independence = pm.Normal('drift_independence', mu=0, sigma=0.1)
            
            # Construct sigma matrix for drift rates
            # [[v1_sigma^2, independence*v1_sigma*v2_sigma],
            #  [independence*v1_sigma*v2_sigma, v2_sigma^2]]
            sigma_11 = v1_sigma**2
            sigma_22 = v2_sigma**2
            sigma_12 = drift_independence * v1_sigma * v2_sigma
            
            # Other LBA parameters
            A_mu = pm.HalfNormal('A_mu', sigma=0.15)
            A_sigma = pm.HalfNormal('A_sigma', sigma=0.08)
            
            b_offset_mu = pm.HalfNormal('b_offset_mu', sigma=0.15)
            b_offset_sigma = pm.HalfNormal('b_offset_sigma', sigma=0.08)
            
            t0_mu = pm.HalfNormal('t0_mu', sigma=0.1)
            t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.05)
            
            # Individual participant parameters
            A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
            A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
            
            b_offset_raw = pm.Normal('b_offset_raw', mu=0, sigma=1, shape=n_participants)
            b_offset = pm.Deterministic('b_offset', 
                                      pm.math.maximum(b_offset_mu + b_offset_sigma * b_offset_raw, 0.05))
            b = pm.Deterministic('b', A + b_offset)
            
            # Drift rates with sigma matrix structure
            # Multivariate normal for testing independence
            drift_raw = pm.MvNormal('drift_raw', 
                                  mu=pt.zeros(2), 
                                  cov=pt.eye(2), 
                                  shape=(n_participants, 2))
            
            # Transform using sigma matrix
            v1_transform = v1_mu + pt.sqrt(sigma_11) * drift_raw[:, 0] + \
                          (sigma_12 / pt.sqrt(sigma_11)) * drift_raw[:, 1]
            v2_transform = v2_mu + pt.sqrt(sigma_22 - sigma_12**2/sigma_11) * drift_raw[:, 1]
            
            v1 = pm.Deterministic('v1', pm.math.maximum(v1_transform, 0.1))
            v2 = pm.Deterministic('v2', pm.math.maximum(v2_transform, 0.1))
            
            t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
            t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
            
            # LBA likelihood
            pm.Potential('lba_likelihood', 
                       stable_lba_loglik(rt_obs, choice_obs, participant_obs, A, b, v1, v2, t0))
        
        print("Starting Bayesian sampling for PI model...")
        start_time = time.time()
        
        with pi_model:
            pi_trace = pm.sample(
                draws=300, tune=150, chains=2,
                progressbar=True, return_inferencedata=True,
                target_accept=0.92, max_treedepth=12,
                cores=1, random_seed=123
            )
        
        elapsed = time.time() - start_time
        print(f"PI sampling completed in {elapsed:.1f} seconds")
        
        # Analyze results
        self.analyze_pi_results(pi_trace)
        
        # Save results to temporary space
        self.save_pi_results(pi_trace, pi_model)
        
        return pi_trace, pi_model
    
    def analyze_pi_results(self, trace):
        """Analyze PI test results"""
        print("\n" + "-"*50)
        print("PERCEPTUAL INDEPENDENCE ANALYSIS RESULTS")
        print("-"*50)
        
        # Key PI parameter: drift_independence
        independence_samples = trace.posterior['drift_independence'].values.flatten()
        independence_mean = float(np.mean(independence_samples))
        independence_std = float(np.std(independence_samples))
        independence_hdi = np.percentile(independence_samples, [2.5, 97.5])
        
        print(f"Drift Independence Parameter:")
        print(f"  Mean: {independence_mean:.4f}")
        print(f"  Std: {independence_std:.4f}")
        print(f"  95% HDI: [{independence_hdi[0]:.4f}, {independence_hdi[1]:.4f}]")
        
        # Interpretation
        print(f"\nPI Assumption Interpretation:")
        if abs(independence_mean) < 0.1 and independence_hdi[0] < 0.1 and independence_hdi[1] > -0.1:
            print("  ✓ SUPPORTS PI: Independence parameter close to zero")
            print("    Perceptual dimensions appear to be processed independently")
        else:
            print("  ✗ VIOLATES PI: Independence parameter significantly different from zero")
            print("    Perceptual dimensions show dependence in processing")
        
        # Sigma matrix components
        v1_sigma_samples = trace.posterior['v1_sigma'].values.flatten()
        v2_sigma_samples = trace.posterior['v2_sigma'].values.flatten()
        
        print(f"\nSigma Matrix Components:")
        print(f"  v1_sigma: {np.mean(v1_sigma_samples):.4f} ± {np.std(v1_sigma_samples):.4f}")
        print(f"  v2_sigma: {np.mean(v2_sigma_samples):.4f} ± {np.std(v2_sigma_samples):.4f}")
        
        # Compute effective covariance
        covariance_samples = independence_samples * v1_sigma_samples * v2_sigma_samples
        covariance_mean = float(np.mean(covariance_samples))
        covariance_hdi = np.percentile(covariance_samples, [2.5, 97.5])
        
        print(f"  Effective Covariance: {covariance_mean:.4f} [{covariance_hdi[0]:.4f}, {covariance_hdi[1]:.4f}]")
        
        # Model diagnostics
        print(f"\nModel Diagnostics:")
        ess = az.ess(trace)
        rhat = az.rhat(trace)
        
        key_params = ['drift_independence', 'v1_mu', 'v2_mu', 'v1_sigma', 'v2_sigma']
        for param in key_params:
            if param in ess.data_vars:
                ess_val = float(ess[param]) if ess[param].ndim == 0 else float(ess[param].min())
                rhat_val = float(rhat[param]) if rhat[param].ndim == 0 else float(rhat[param].max())
                print(f"  {param}: ESS={ess_val:.0f}, R̂={rhat_val:.3f}")
    
    def save_pi_results(self, trace, model):
        """Save PI results to temporary space"""
        print("\nSaving PI results to temporary space...")
        
        # Extract key results
        independence_samples = trace.posterior['drift_independence'].values.flatten()
        results = {
            'model_type': 'Perceptual_Independence',
            'assumption_tested': 'PI',
            'independence_mean': float(np.mean(independence_samples)),
            'independence_std': float(np.std(independence_samples)),
            'independence_hdi': [float(x) for x in np.percentile(independence_samples, [2.5, 97.5])],
            'v1_mu_mean': float(trace.posterior['v1_mu'].mean()),
            'v2_mu_mean': float(trace.posterior['v2_mu'].mean()),
            'v1_sigma_mean': float(trace.posterior['v1_sigma'].mean()),
            'v2_sigma_mean': float(trace.posterior['v2_sigma'].mean()),
            'n_samples': len(independence_samples),
            'ess_independence': float(az.ess(trace)['drift_independence']),
            'rhat_independence': float(az.rhat(trace)['drift_independence']),
            'pi_support': abs(float(np.mean(independence_samples))) < 0.1
        }
        
        # Save as JSON (lightweight)
        with open('pi_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trace data (pickled)
        try:
            with open('pi_trace.pkl', 'wb') as f:
                pickle.dump(trace, f)
            print("✓ PI trace saved as pi_trace.pkl")
        except Exception as e:
            print(f"Warning: Could not save trace: {e}")
        
        print("✓ PI results saved as pi_results.json")
        
        return results

def run_pi_analysis():
    """Run the Perceptual Independence analysis"""
    print("STARTING PERCEPTUAL INDEPENDENCE (PI) ANALYSIS")
    print("="*60)
    
    tester = PerceptualIndependenceTester('GRT_LBA.csv')
    
    try:
        pi_trace, pi_model = tester.test_perceptual_independence()
        print("\n✓ Perceptual Independence analysis completed successfully")
        return tester, pi_trace, pi_model
    except Exception as e:
        print(f"\n✗ Perceptual Independence analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    tester, trace, model = run_pi_analysis()
    
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:33:10 2025

@author: spt904
"""

"""
GRT Combined Analysis and Results Integration
Loads and compares results from all three GRT assumption tests
Provides comprehensive summary and interpretation
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class GRTResultsIntegrator:
    """Integrate and analyze results from all three GRT assumption tests"""
    
    def __init__(self):
        self.pi_results = None
        self.ps_results = None
        self.ds_results = None
        self.load_all_results()
    
    def load_all_results(self):
        """Load results from all three analyses"""
        print("Loading GRT assumption test results...")
        
        # Load PI results
        try:
            with open('pi_results.json', 'r') as f:
                self.pi_results = json.load(f)
            print("✓ PI results loaded")
        except FileNotFoundError:
            print("⚠ PI results not found")
        
        # Load PS results
        try:
            with open('ps_results.json', 'r') as f:
                self.ps_results = json.load(f)
            print("✓ PS results loaded")
        except FileNotFoundError:
            print("⚠ PS results not found")
        
        # Load DS results
        try:
            with open('ds_results.json', 'r') as f:
                self.ds_results = json.load(f)
            print("✓ DS results loaded")
        except FileNotFoundError:
            print("⚠ DS results not found")
    
    def generate_comprehensive_summary(self):
        """Generate comprehensive summary of all GRT assumption tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE GRT ASSUMPTION TESTING SUMMARY")
        print("="*80)
        
        # Overview table
        print("\nGRT Assumption Test Results Overview:")
        print("-" * 60)
        print(f"{'Assumption':<25} {'Support':<15} {'Key Parameter':<20}")
        print("-" * 60)
        
        if self.pi_results:
            pi_support = "✓ SUPPORTED" if self.pi_results['pi_support'] else "✗ VIOLATED"
            pi_key = f"Independence: {self.pi_results['independence_mean']:.3f}"
            print(f"{'Perceptual Independence':<25} {pi_support:<15} {pi_key:<20}")
        
        if self.ps_results:
            ps_support = "✓ SUPPORTED" if self.ps_results['ps_support'] else "✗ VIOLATED"
            ps_key = f"Correlation: {self.ps_results['perceptual_correlation_mean']:.3f}"
            print(f"{'Perceptual Separability':<25} {ps_support:<15} {ps_key:<20}")
        
        if self.ds_results:
            ds_support = "✓ SUPPORTED" if self.ds_results['ds_support'] else "✗ VIOLATED"
            ds_key = f"Bias: {self.ds_results['decision_bias_mean']:.3f}"
            print(f"{'Decisional Separability':<25} {ds_support:<15} {ds_key:<20}")
        
        print("-" * 60)
        
        # Detailed analysis for each assumption
        if self.pi_results:
            self.summarize_pi_results()
        
        if self.ps_results:
            self.summarize_ps_results()
        
        if self.ds_results:
            self.summarize_ds_results()
        
        # Overall GRT interpretation
        self.interpret_overall_grt()
        
        # Model comparison
        self.compare_model_quality()
    
    def summarize_pi_results(self):
        """Summarize Perceptual Independence results"""
        print("\n" + "="*50)
        print("PERCEPTUAL INDEPENDENCE (PI) DETAILED RESULTS")
        print("="*50)
        
        independence = self.pi_results['independence_mean']
        independence_hdi = self.pi_results['independence_hdi']
        
        print(f"Independence Parameter: {independence:.4f} [{independence_hdi[0]:.4f}, {independence_hdi[1]:.4f}]")
        print(f"Support for PI: {'YES' if self.pi_results['pi_support'] else 'NO'}")
        
        print(f"\nDrift Rate Parameters:")
        print(f"  v1_mu: {self.pi_results['v1_mu_mean']:.3f}")
        print(f"  v2_mu: {self.pi_results['v2_mu_mean']:.3f}")
        print(f"  v1_sigma: {self.pi_results['v1_sigma_mean']:.3f}")
        print(f"  v2_sigma: {self.pi_results['v2_sigma_mean']:.3f}")
        
        print(f"\nModel Quality:")
        print(f"  Effective Sample Size: {self.pi_results['ess_independence']:.0f}")
        print(f"  R-hat: {self.pi_results['rhat_independence']:.3f}")
        
        print(f"\nInterpretation:")
        if self.pi_results['pi_support']:
            print("  The independence parameter is close to zero, suggesting that")
            print("  perceptual processing of the two dimensions is independent.")
            print("  This supports the PI assumption of GRT.")
        else:
            print("  The independence parameter is significantly different from zero,")
            print("  suggesting dependence between perceptual dimensions.")
            print("  This violates the PI assumption of GRT.")
    
    def summarize_ps_results(self):
        """Summarize Perceptual Separability results"""
        print("\n" + "="*50)
        print("PERCEPTUAL SEPARABILITY (PS) DETAILED RESULTS")
        print("="*50)
        
        interference_12 = self.ps_results['interference_12_mean']
        interference_21 = self.ps_results['interference_21_mean']
        correlation = self.ps_results['perceptual_correlation_mean']
        
        print(f"Cross-Dimensional Interference:")
        print(f"  Dimension 1 → 2: {interference_12:.4f} {self.ps_results['interference_12_hdi']}")
        print(f"  Dimension 2 → 1: {interference_21:.4f} {self.ps_results['interference_21_hdi']}")
        print(f"Perceptual Correlation: {correlation:.4f} {self.ps_results['perceptual_correlation_hdi']}")
        
        print(f"Support for PS: {'YES' if self.ps_results['ps_support'] else 'NO'}")
        
        print(f"\nSignificant Effects:")
        print(f"  Interference 1→2: {'YES' if self.ps_results['significant_interference_12'] else 'NO'}")
        print(f"  Interference 2→1: {'YES' if self.ps_results['significant_interference_21'] else 'NO'}")
        print(f"  Perceptual Correlation: {'YES' if self.ps_results['significant_correlation'] else 'NO'}")
        
        print(f"\nModel Quality:")
        print(f"  ESS (Interference 1→2): {self.ps_results['ess_interference_12']:.0f}")
        print(f"  ESS (Interference 2→1): {self.ps_results['ess_interference_21']:.0f}")
        print(f"  ESS (Correlation): {self.ps_results['ess_correlation']:.0f}")
        
        print(f"\nInterpretation:")
        if self.ps_results['ps_support']:
            print("  No significant cross-dimensional interference or correlation detected.")
            print("  Perceptual representations appear to be separable by dimension.")
            print("  This supports the PS assumption of GRT.")
        else:
            print("  Significant cross-dimensional effects detected:")
            if self.ps_results['significant_interference_12']:
                print("    - Dimension 1 interferes with dimension 2 processing")
            if self.ps_results['significant_interference_21']:
                print("    - Dimension 2 interferes with dimension 1 processing")
            if self.ps_results['significant_correlation']:
                print("    - Significant correlation between perceptual representations")
            print("  This violates the PS assumption of GRT.")
    
    def summarize_ds_results(self):
        """Summarize Decisional Separability results"""
        print("\n" + "="*50)
        print("DECISIONAL SEPARABILITY (DS) DETAILED RESULTS")
        print("="*50)
        
        decision_bias = self.ds_results['decision_bias_mean']
        boundary_interaction = self.ds_results['boundary_interaction_mean']
        boundary_correlation = self.ds_results['boundary_correlation_mean']
        boundary_diff = self.ds_results['boundary_diff_mean']
        
        print(f"Decision Boundary Parameters:")
        print(f"  Decision Bias: {decision_bias:.4f} {self.ds_results['decision_bias_hdi']}")
        print(f"  Boundary Interaction: {boundary_interaction:.4f} {self.ds_results['boundary_interaction_hdi']}")
        print(f"  Boundary Correlation: {boundary_correlation:.4f} {self.ds_results['boundary_correlation_hdi']}")
        print(f"  Boundary Difference: {boundary_diff:.4f} {self.ds_results['boundary_diff_hdi']}")
        
        print(f"Support for DS: {'YES' if self.ds_results['ds_support'] else 'NO'}")
        
        print(f"\nSignificant Effects:")
        print(f"  Decision Bias: {'YES' if self.ds_results['significant_bias'] else 'NO'}")
        print(f"  Boundary Interaction: {'YES' if self.ds_results['significant_interaction'] else 'NO'}")
        print(f"  Boundary Correlation: {'YES' if self.ds_results['significant_correlation'] else 'NO'}")
        
        print(f"\nThreshold Parameters:")
        print(f"  b1_offset_mu: {self.ds_results['b1_offset_mu_mean']:.3f}")
        print(f"  b2_offset_mu: {self.ds_results['b2_offset_mu_mean']:.3f}")
        print(f"  b_offset_sigma: {self.ds_results['b_offset_sigma_mean']:.3f}")
        
        print(f"\nModel Quality:")
        print(f"  ESS (Decision Bias): {self.ds_results['ess_decision_bias']:.0f}")
        print(f"  ESS (Boundary Interaction): {self.ds_results['ess_boundary_interaction']:.0f}")
        print(f"  ESS (Boundary Correlation): {self.ds_results['ess_boundary_correlation']:.0f}")
        
        print(f"\nInterpretation:")
        if self.ds_results['ds_support']:
            print("  No significant decision boundary dependence detected.")
            print("  Decision boundaries appear to be separable across dimensions.")
            print("  This supports the DS assumption of GRT.")
        else:
            print("  Significant decision boundary dependence detected:")
            if self.ds_results['significant_bias']:
                print("    - Significant decision bias between boundaries")
            if self.ds_results['significant_interaction']:
                print("    - Significant boundary interaction effects")
            if self.ds_results['significant_correlation']:
                print("    - Significant correlation between boundary parameters")
            print("  This violates the DS assumption of GRT.")
    
    def interpret_overall_grt(self):
        """Provide overall interpretation of GRT assumptions"""
        print("\n" + "="*50)
        print("OVERALL GRT INTERPRETATION")
        print("="*50)
        
        supported_assumptions = []
        violated_assumptions = []
        
        if self.pi_results:
            if self.pi_results['pi_support']:
                supported_assumptions.append("Perceptual Independence (PI)")
            else:
                violated_assumptions.append("Perceptual Independence (PI)")
        
        if self.ps_results:
            if self.ps_results['ps_support']:
                supported_assumptions.append("Perceptual Separability (PS)")
            else:
                violated_assumptions.append("Perceptual Separability (PS)")
        
        if self.ds_results:
            if self.ds_results['ds_support']:
                supported_assumptions.append("Decisional Separability (DS)")
            else:
                violated_assumptions.append("Decisional Separability (DS)")
        
        print(f"Supported Assumptions ({len(supported_assumptions)}):")
        for assumption in supported_assumptions:
            print(f"  ✓ {assumption}")
        
        print(f"\nViolated Assumptions ({len(violated_assumptions)}):")
        for assumption in violated_assumptions:
            print(f"  ✗ {assumption}")
        
        # Overall GRT interpretation
        print(f"\nGRT Model Validity:")
        if len(violated_assumptions) == 0:
            print("  ✓ FULL GRT VALIDITY: All assumptions are supported")
            print("    The General Recognition Theory model is appropriate for this data.")
        elif len(violated_assumptions) == 1:
            print("  ⚠ PARTIAL GRT VALIDITY: One assumption violated")
            print(f"    Consider alternative models that relax the {violated_assumptions[0]} assumption.")
        elif len(violated_assumptions) == 2:
            print("  ⚠ LIMITED GRT VALIDITY: Two assumptions violated")
            print("    Standard GRT may not be appropriate. Consider more flexible models.")
        else:
            print("  ✗ GRT NOT VALID: All assumptions violated")
            print("    The data violates fundamental GRT assumptions. Alternative models needed.")
        
        # Recommendations
        print(f"\nRecommendations:")
        if len(violated_assumptions) == 0:
            print("  - Proceed with standard GRT analyses")
            print("  - The data satisfies GRT assumptions")
        else:
            print("  - Consider violated assumptions when interpreting results")
            if "Perceptual Independence (PI)" in violated_assumptions:
                print("  - Use models that allow perceptual dependencies")
            if "Perceptual Separability (PS)" in violated_assumptions:
                print("  - Consider integral dimension models")
            if "Decisional Separability (DS)" in violated_assumptions:
                print("  - Use models with flexible decision boundaries")
    
    def compare_model_quality(self):
        """Compare quality metrics across models"""
        print("\n" + "="*50)
        print("MODEL QUALITY COMPARISON")
        print("="*50)
        
        print(f"{'Model':<15} {'Key Parameter':<20} {'ESS':<8} {'R-hat':<8}")
        print("-" * 55)
        
        if self.pi_results:
            print(f"{'PI Model':<15} {'Independence':<20} {self.pi_results['ess_independence']:<8.0f} {self.pi_results['rhat_independence']:<8.3f}")
        
        if self.ps_results:
            ess_min = min(self.ps_results['ess_interference_12'], 
                         self.ps_results['ess_interference_21'], 
                         self.ps_results['ess_correlation'])
            rhat_max = max(self.ps_results['rhat_interference_12'], 
                          self.ps_results['rhat_interference_21'], 
                          self.ps_results['rhat_correlation'])
            print(f"{'PS Model':<15} {'Min Interference':<20} {ess_min:<8.0f} {rhat_max:<8.3f}")
        
        if self.ds_results:
            ess_min = min(self.ds_results['ess_decision_bias'], 
                         self.ds_results['ess_boundary_interaction'], 
                         self.ds_results['ess_boundary_correlation'])
            rhat_max = max(self.ds_results['rhat_decision_bias'], 
                          self.ds_results['rhat_boundary_interaction'], 
                          self.ds_results['rhat_boundary_correlation'])
            print(f"{'DS Model':<15} {'Min Boundary':<20} {ess_min:<8.0f} {rhat_max:<8.3f}")
        
        print("-" * 55)
        print("Note: ESS > 400 and R̂ < 1.01 indicate good convergence")
    
    def save_combined_results(self):
        """Save combined results summary"""
        print("\nSaving combined results...")
        
        combined_results = {
            'grt_analysis_summary': {
                'pi_support': self.pi_results['pi_support'] if self.pi_results else None,
                'ps_support': self.ps_results['ps_support'] if self.ps_results else None,
                'ds_support': self.ds_results['ds_support'] if self.ds_results else None,
                'overall_grt_validity': 'full' if all([
                    self.pi_results and self.pi_results['pi_support'],
                    self.ps_results and self.ps_results['ps_support'],
                    self.ds_results and self.ds_results['ds_support']
                ]) else 'partial_or_violated'
            },
            'detailed_results': {
                'pi': self.pi_results,
                'ps': self.ps_results,
                'ds': self.ds_results
            }
        }
        
        with open('grt_combined_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print("✓ Combined results saved as grt_combined_results.json")
        return combined_results

def run_combined_analysis():
    """Run the combined analysis"""
    print("STARTING GRT COMBINED ANALYSIS")
    print("="*60)
    
    integrator = GRTResultsIntegrator()
    integrator.generate_comprehensive_summary()
    combined_results = integrator.save_combined_results()
    
    print("\n" + "="*60)
    print("GRT COMBINED ANALYSIS COMPLETED")
    print("="*60)
    
    return integrator, combined_results

if __name__ == "__main__":
    integrator, results = run_combined_analysis()
    
    
"""
GRT Combined Analysis and Results Integration
Loads and compares results from all three GRT assumption tests
Provides comprehensive summary and interpretation
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class GRTResultsIntegrator:
    """Integrate and analyze results from all three GRT assumption tests"""
    
    def __init__(self):
        self.pi_results = None
        self.ps_results = None
        self.ds_results = None
        self.load_all_results()
    
    def load_all_results(self):
        """Load results from all three analyses"""
        print("Loading GRT assumption test results...")
        
        # Load PI results
        try:
            with open('pi_results.json', 'r') as f:
                self.pi_results = json.load(f)
            print("✓ PI results loaded")
        except FileNotFoundError:
            print("⚠ PI results not found")
        
        # Load PS results
        try:
            with open('ps_results.json', 'r') as f:
                self.ps_results = json.load(f)
            print("✓ PS results loaded")
        except FileNotFoundError:
            print("⚠ PS results not found")
        
        # Load DS results
        try:
            with open('ds_results.json', 'r') as f:
                self.ds_results = json.load(f)
            print("✓ DS results loaded")
        except FileNotFoundError:
            print("⚠ DS results not found")
    
    def generate_comprehensive_summary(self):
        """Generate comprehensive summary of all GRT assumption tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE GRT ASSUMPTION TESTING SUMMARY")
        print("="*80)
        
        # Overview table
        print("\nGRT Assumption Test Results Overview:")
        print("-" * 60)
        print(f"{'Assumption':<25} {'Support':<15} {'Key Parameter':<20}")
        print("-" * 60)
        
        if self.pi_results:
            pi_support = "✓ SUPPORTED" if self.pi_results['pi_support'] else "✗ VIOLATED"
            pi_key = f"Independence: {self.pi_results['independence_mean']:.3f}"
            print(f"{'Perceptual Independence':<25} {pi_support:<15} {pi_key:<20}")
        
        if self.ps_results:
            ps_support = "✓ SUPPORTED" if self.ps_results['ps_support'] else "✗ VIOLATED"
            ps_key = f"Correlation: {self.ps_results['perceptual_correlation_mean']:.3f}"
            print(f"{'Perceptual Separability':<25} {ps_support:<15} {ps_key:<20}")
        
        if self.ds_results:
            ds_support = "✓ SUPPORTED" if self.ds_results['ds_support'] else "✗ VIOLATED"
            ds_key = f"Bias: {self.ds_results['decision_bias_mean']:.3f}"
            print(f"{'Decisional Separability':<25} {ds_support:<15} {ds_key:<20}")
        
        print("-" * 60)
        
        # Detailed analysis for each assumption
        if self.pi_results:
            self.summarize_pi_results()
        
        if self.ps_results:
            self.summarize_ps_results()
        
        if self.ds_results:
            self.summarize_ds_results()
        
        # Overall GRT interpretation
        self.interpret_overall_grt()
        
        # Model comparison
        self.compare_model_quality()
    
    def summarize_pi_results(self):
        """Summarize Perceptual Independence results"""
        print("\n" + "="*50)
        print("PERCEPTUAL INDEPENDENCE (PI) DETAILED RESULTS")
        print("="*50)
        
        independence = self.pi_results['independence_mean']
        independence_hdi = self.pi_results['independence_hdi']
        
        print(f"Independence Parameter: {independence:.4f} [{independence_hdi[0]:.4f}, {independence_hdi[1]:.4f}]")
        print(f"Support for PI: {'YES' if self.pi_results['pi_support'] else 'NO'}")
        
        print(f"\nDrift Rate Parameters:")
        print(f"  v1_mu: {self.pi_results['v1_mu_mean']:.3f}")
        print(f"  v2_mu: {self.pi_results['v2_mu_mean']:.3f}")
        print(f"  v1_sigma: {self.pi_results['v1_sigma_mean']:.3f}")
        print(f"  v2_sigma: {self.pi_results['v2_sigma_mean']:.3f}")
        
        print(f"\nModel Quality:")
        print(f"  Effective Sample Size: {self.pi_results['ess_independence']:.0f}")
        print(f"  R-hat: {self.pi_results['rhat_independence']:.3f}")
        
        print(f"\nInterpretation:")
        if self.pi_results['pi_support']:
            print("  The independence parameter is close to zero, suggesting that")
            print("  perceptual processing of the two dimensions is independent.")
            print("  This supports the PI assumption of GRT.")
        else:
            print("  The independence parameter is significantly different from zero,")
            print("  suggesting dependence between perceptual dimensions.")
            print("  This violates the PI assumption of GRT.")
    
    def summarize_ps_results(self):
        """Summarize Perceptual Separability results"""
        print("\n" + "="*50)
        print("PERCEPTUAL SEPARABILITY (PS) DETAILED RESULTS")
        print("="*50)
        
        interference_12 = self.ps_results['interference_12_mean']
        interference_21 = self.ps_results['interference_21_mean']
        correlation = self.ps_results['perceptual_correlation_mean']
        
        print(f"Cross-Dimensional Interference:")
        print(f"  Dimension 1 → 2: {interference_12:.4f} {self.ps_results['interference_12_hdi']}")
        print(f"  Dimension 2 → 1: {interference_21:.4f} {self.ps_results['interference_21_hdi']}")
        print(f"Perceptual Correlation: {correlation:.4f} {self.ps_results['perceptual_correlation_hdi']}")
        
        print(f"Support for PS: {'YES' if self.ps_results['ps_support'] else 'NO'}")
        
        print(f"\nSignificant Effects:")
        print(f"  Interference 1→2: {'YES' if self.ps_results['significant_interference_12'] else 'NO'}")
        print(f"  Interference 2→1: {'YES' if self.ps_results['significant_interference_21'] else 'NO'}")
        print(f"  Perceptual Correlation: {'YES' if self.ps_results['significant_correlation'] else 'NO'}")
        
        print(f"\nModel Quality:")
        print(f"  ESS (Interference 1→2): {self.ps_results['ess_interference_12']:.0f}")
        print(f"  ESS (Interference 2→1): {self.ps_results['ess_interference_21']:.0f}")
        print(f"  ESS (Correlation): {self.ps_results['ess_correlation']:.0f}")
        
        print(f"\nInterpretation:")
        if self.ps_results['ps_support']:
            print("  No significant cross-dimensional interference or correlation detected.")
            print("  Perceptual representations appear to be separable by dimension.")
            print("  This supports the PS assumption of GRT.")
        else:
            print("  Significant cross-dimensional effects detected:")
            if self.ps_results['significant_interference_12']:
                print("    - Dimension 1 interferes with dimension 2 processing")
            if self.ps_results['significant_interference_21']:
                print("    - Dimension 2 interferes with dimension 1 processing")
            if self.ps_results['significant_correlation']:
                print("    - Significant correlation between perceptual representations")
            print("  This violates the PS assumption of GRT.")
    
    def summarize_ds_results(self):
        """Summarize Decisional Separability results"""
        print("\n" + "="*50)
        print("DECISIONAL SEPARABILITY (DS) DETAILED RESULTS")
        print("="*50)
        
        decision_bias = self.ds_results['decision_bias_mean']
        boundary_interaction = self.ds_results['boundary_interaction_mean']
        boundary_correlation = self.ds_results['boundary_correlation_mean']
        boundary_diff = self.ds_results['boundary_diff_mean']
        
        print(f"Decision Boundary Parameters:")
        print(f"  Decision Bias: {decision_bias:.4f} {self.ds_results['decision_bias_hdi']}")
        print(f"  Boundary Interaction: {boundary_interaction:.4f} {self.ds_results['boundary_interaction_hdi']}")
        print(f"  Boundary Correlation: {boundary_correlation:.4f} {self.ds_results['boundary_correlation_hdi']}")
        print(f"  Boundary Difference: {boundary_diff:.4f} {self.ds_results['boundary_diff_hdi']}")
        
        print(f"Support for DS: {'YES' if self.ds_results['ds_support'] else 'NO'}")
        
        print(f"\nSignificant Effects:")
        print(f"  Decision Bias: {'YES' if self.ds_results['significant_bias'] else 'NO'}")
        print(f"  Boundary Interaction: {'YES' if self.ds_results['significant_interaction'] else 'NO'}")
        print(f"  Boundary Correlation: {'YES' if self.ds_results['significant_correlation'] else 'NO'}")
        
        print(f"\nThreshold Parameters:")
        print(f"  b1_offset_mu: {self.ds_results['b1_offset_mu_mean']:.3f}")
        print(f"  b2_offset_mu: {self.ds_results['b2_offset_mu_mean']:.3f}")
        print(f"  b_offset_sigma: {self.ds_results['b_offset_sigma_mean']:.3f}")
        
        print(f"\nModel Quality:")
        print(f"  ESS (Decision Bias): {self.ds_results['ess_decision_bias']:.0f}")
        print(f"  ESS (Boundary Interaction): {self.ds_results['ess_boundary_interaction']:.0f}")
        print(f"  ESS (Boundary Correlation): {self.ds_results['ess_boundary_correlation']:.0f}")
        
        print(f"\nInterpretation:")
        if self.ds_results['ds_support']:
            print("  No significant decision boundary dependence detected.")
            print("  Decision boundaries appear to be separable across dimensions.")
            print("  This supports the DS assumption of GRT.")
        else:
            print("  Significant decision boundary dependence detected:")
            if self.ds_results['significant_bias']:
                print("    - Significant decision bias between boundaries")
            if self.ds_results['significant_interaction']:
                print("    - Significant boundary interaction effects")
            if self.ds_results['significant_correlation']:
                print("    - Significant correlation between boundary parameters")
            print("  This violates the DS assumption of GRT.")
    
    def interpret_overall_grt(self):
        """Provide overall interpretation of GRT assumptions"""
        print("\n" + "="*50)
        print("OVERALL GRT INTERPRETATION")
        print("="*50)
        
        supported_assumptions = []
        violated_assumptions = []
        
        if self.pi_results:
            if self.pi_results['pi_support']:
                supported_assumptions.append("Perceptual Independence (PI)")
            else:
                violated_assumptions.append("Perceptual Independence (PI)")
        
        if self.ps_results:
            if self.ps_results['ps_support']:
                supported_assumptions.append("Perceptual Separability (PS)")
            else:
                violated_assumptions.append("Perceptual Separability (PS)")
        
        if self.ds_results:
            if self.ds_results['ds_support']:
                supported_assumptions.append("Decisional Separability (DS)")
            else:
                violated_assumptions.append("Decisional Separability (DS)")
        
        print(f"Supported Assumptions ({len(supported_assumptions)}):")
        for assumption in supported_assumptions:
            print(f"  ✓ {assumption}")
        
        print(f"\nViolated Assumptions ({len(violated_assumptions)}):")
        for assumption in violated_assumptions:
            print(f"  ✗ {assumption}")
        
        # Overall GRT interpretation
        print(f"\nGRT Model Validity:")
        if len(violated_assumptions) == 0:
            print("  ✓ FULL GRT VALIDITY: All assumptions are supported")
            print("    The General Recognition Theory model is appropriate for this data.")
        elif len(violated_assumptions) == 1:
            print("  ⚠ PARTIAL GRT VALIDITY: One assumption violated")
            print(f"    Consider alternative models that relax the {violated_assumptions[0]} assumption.")
        elif len(violated_assumptions) == 2:
            print("  ⚠ LIMITED GRT VALIDITY: Two assumptions violated")
            print("    Standard GRT may not be appropriate. Consider more flexible models.")
        else:
            print("  ✗ GRT NOT VALID: All assumptions violated")
            print("    The data violates fundamental GRT assumptions. Alternative models needed.")
        
        # Recommendations
        print(f"\nRecommendations:")
        if len(violated_assumptions) == 0:
            print("  - Proceed with standard GRT analyses")
            print("  - The data satisfies GRT assumptions")
        else:
            print("  - Consider violated assumptions when interpreting results")
            if "Perceptual Independence (PI)" in violated_assumptions:
                print("  - Use models that allow perceptual dependencies")
            if "Perceptual Separability (PS)" in violated_assumptions:
                print("  - Consider integral dimension models")
            if "Decisional Separability (DS)" in violated_assumptions:
                print("  - Use models with flexible decision boundaries")
    
    def compare_model_quality(self):
        """Compare quality metrics across models"""
        print("\n" + "="*50)
        print("MODEL QUALITY COMPARISON")
        print("="*50)
        
        print(f"{'Model':<15} {'Key Parameter':<20} {'ESS':<8} {'R-hat':<8}")
        print("-" * 55)
        
        if self.pi_results:
            print(f"{'PI Model':<15} {'Independence':<20} {self.pi_results['ess_independence']:<8.0f} {self.pi_results['rhat_independence']:<8.3f}")
        
        if self.ps_results:
            ess_min = min(self.ps_results['ess_interference_12'], 
                         self.ps_results['ess_interference_21'], 
                         self.ps_results['ess_correlation'])
            rhat_max = max(self.ps_results['rhat_interference_12'], 
                          self.ps_results['rhat_interference_21'], 
                          self.ps_results['rhat_correlation'])
            print(f"{'PS Model':<15} {'Min Interference':<20} {ess_min:<8.0f} {rhat_max:<8.3f}")
        
        if self.ds_results:
            ess_min = min(self.ds_results['ess_decision_bias'], 
                         self.ds_results['ess_boundary_interaction'], 
                         self.ds_results['ess_boundary_correlation'])
            rhat_max = max(self.ds_results['rhat_decision_bias'], 
                          self.ds_results['rhat_boundary_interaction'], 
                          self.ds_results['rhat_boundary_correlation'])
            print(f"{'DS Model':<15} {'Min Boundary':<20} {ess_min:<8.0f} {rhat_max:<8.3f}")
        
        print("-" * 55)
        print("Note: ESS > 400 and R̂ < 1.01 indicate good convergence")
    
    def save_combined_results(self):
        """Save combined results summary"""
        print("\nSaving combined results...")
        
        combined_results = {
            'grt_analysis_summary': {
                'pi_support': self.pi_results['pi_support'] if self.pi_results else None,
                'ps_support': self.ps_results['ps_support'] if self.ps_results else None,
                'ds_support': self.ds_results['ds_support'] if self.ds_results else None,
                'overall_grt_validity': 'full' if all([
                    self.pi_results and self.pi_results['pi_support'],
                    self.ps_results and self.ps_results['ps_support'],
                    self.ds_results and self.ds_results['ds_support']
                ]) else 'partial_or_violated'
            },
            'detailed_results': {
                'pi': self.pi_results,
                'ps': self.ps_results,
                'ds': self.ds_results
            }
        }
        
        with open('grt_combined_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print("✓ Combined results saved as grt_combined_results.json")
        return combined_results

def run_combined_analysis():
    """Run the combined analysis"""
    print("STARTING GRT COMBINED ANALYSIS")
    print("="*60)
    
    integrator = GRTResultsIntegrator()
    integrator.generate_comprehensive_summary()
    combined_results = integrator.save_combined_results()
    
    print("\n" + "="*60)
    print("GRT COMBINED ANALYSIS COMPLETED")
    print("="*60)
    
    return integrator, combined_results

if __name__ == "__main__":
    integrator, results = run_combined_analysis()
