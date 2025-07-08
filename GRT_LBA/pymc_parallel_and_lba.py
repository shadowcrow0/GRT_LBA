# -*- coding: utf-8 -*-
"""
pymc_parallel_and_lba.py - PyMC ParallelAND Dual LBA
Use Bayesian inference with PyMC to estimate parameters and compare predictions
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import arviz as az
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from single_side_lba import SingleSideLBA

class PyMCParallelANDLBA:
    """PyMC implementation of ParallelAND Dual LBA"""
    
    def __init__(self):
        """Initialize PyMC ParallelAND system"""
        self.left_lba = SingleSideLBA('left')
        self.right_lba = SingleSideLBA('right')
        self.model = None
        self.trace = None
        
        print("✅ PyMC ParallelAND Dual LBA initialized")
    
    def load_data(self, file_path: str = "GRT_LBA.csv", participant_id: int = 31) -> Dict:
        """Load and prepare data for PyMC analysis"""
        
        data = pd.read_csv(file_path)
        participant_data = data[data['participant'] == participant_id].copy()
        
        print(f"📊 Loaded {len(participant_data)} trials for participant {participant_id}")
        
        # Convert to stimulus format
        participant_data['left_stimulus'] = participant_data['Chanel1'].map({0: 'nonvertical', 1: 'vertical'})
        participant_data['right_stimulus'] = participant_data['Chanel2'].map({0: 'nonvertical', 1: 'vertical'})
        
        # Prepare arrays for PyMC
        responses = participant_data['Response'].astype(int).values
        rts = participant_data['RT'].astype(float).values
        left_stimuli = participant_data['Chanel1'].values  # 0=nonvertical, 1=vertical
        right_stimuli = participant_data['Chanel2'].values
        
        return {
            'responses': responses,
            'rts': rts,
            'left_stimuli': left_stimuli,
            'right_stimuli': right_stimuli,
            'n_trials': len(participant_data),
            'participant_id': participant_id,
            'data_df': participant_data
        }
    
    def create_pymc_model(self, data: Dict) -> pm.Model:
        """Create PyMC model for ParallelAND Dual LBA"""
        
        print("🔧 Building PyMC ParallelAND model...")
        
        with pm.Model() as model:
            # Prior distributions for drift rates
            left_v_vertical = pm.Gamma('left_v_vertical', alpha=2.5, beta=1.5)
            left_v_nonvertical = pm.Gamma('left_v_nonvertical', alpha=2.5, beta=1.5)
            left_v_vertical_error = pm.Gamma('left_v_vertical_error', alpha=2.0, beta=3.0)
            left_v_nonvertical_error = pm.Gamma('left_v_nonvertical_error', alpha=2.0, beta=3.0)
            
            right_v_vertical = pm.Gamma('right_v_vertical', alpha=2.5, beta=1.5)
            right_v_nonvertical = pm.Gamma('right_v_nonvertical', alpha=2.5, beta=1.5)
            right_v_vertical_error = pm.Gamma('right_v_vertical_error', alpha=2.0, beta=3.0)
            right_v_nonvertical_error = pm.Gamma('right_v_nonvertical_error', alpha=2.0, beta=3.0)
            
            # Common parameters
            threshold = pm.Gamma('threshold', alpha=3.0, beta=3.5)
            start_var = pm.Uniform('start_var', lower=0.1, upper=0.7)
            ndt = pm.Uniform('ndt', lower=0.05, upper=0.6)
            noise = pm.Gamma('noise', alpha=2.5, beta=8.0)
            
            # Convert data to PyTensor tensors
            responses_tensor = pt.as_tensor_variable(data['responses'])
            rts_tensor = pt.as_tensor_variable(data['rts'])
            left_stimuli_tensor = pt.as_tensor_variable(data['left_stimuli'])
            right_stimuli_tensor = pt.as_tensor_variable(data['right_stimuli'])
            
            # Calculate drift rates for each trial and response option
            predicted_rts = self._compute_parallel_and_likelihood(
                left_stimuli_tensor, right_stimuli_tensor, responses_tensor,
                left_v_vertical, left_v_nonvertical, left_v_vertical_error, left_v_nonvertical_error,
                right_v_vertical, right_v_nonvertical, right_v_vertical_error, right_v_nonvertical_error,
                threshold, ndt, noise
            )
            
            # Likelihood: RT predictions with Normal distribution around predicted RT
            rt_sigma = pm.HalfNormal('rt_sigma', sigma=0.3)
            rt_likelihood = pm.Normal('rt_obs', mu=predicted_rts, sigma=rt_sigma, observed=rts_tensor)
            
        self.model = model
        print(f"   Model created with {len(data['responses'])} trials")
        return model
    
    def _compute_parallel_and_likelihood(self, left_stimuli, right_stimuli, responses,
                                       left_v_v, left_v_nv, left_v_v_err, left_v_nv_err,
                                       right_v_v, right_v_nv, right_v_v_err, right_v_nv_err,
                                       threshold, ndt, noise):
        """Compute ParallelAND RT predictions for each trial"""
        
        # Determine drift rates based on stimulus and response
        # Left side drift rates
        left_is_vertical_stim = pt.eq(left_stimuli, 1)
        left_is_vertical_resp = pt.or_(pt.eq(responses, 1), pt.eq(responses, 2))  # Responses where left responds vertical
        
        left_correct_vertical = left_is_vertical_stim & left_is_vertical_resp
        left_error_vertical = (~left_is_vertical_stim) & left_is_vertical_resp
        left_correct_nonvertical = (~left_is_vertical_stim) & (~left_is_vertical_resp)
        left_error_nonvertical = left_is_vertical_stim & (~left_is_vertical_resp)
        
        left_drift = pt.where(left_correct_vertical, left_v_v,
                     pt.where(left_error_vertical, left_v_v_err,
                     pt.where(left_correct_nonvertical, left_v_nv, left_v_nv_err)))
        
        # Right side drift rates
        right_is_vertical_stim = pt.eq(right_stimuli, 1)
        right_is_vertical_resp = pt.or_(pt.eq(responses, 0), pt.eq(responses, 1))  # Responses where right responds vertical
        
        right_correct_vertical = right_is_vertical_stim & right_is_vertical_resp
        right_error_vertical = (~right_is_vertical_stim) & right_is_vertical_resp
        right_correct_nonvertical = (~right_is_vertical_stim) & (~right_is_vertical_resp)
        right_error_nonvertical = right_is_vertical_stim & (~right_is_vertical_resp)
        
        right_drift = pt.where(right_correct_vertical, right_v_v,
                      pt.where(right_error_vertical, right_v_v_err,
                      pt.where(right_correct_nonvertical, right_v_nv, right_v_nv_err)))
        
        # ParallelAND: Use minimum drift rate
        effective_drift = pt.minimum(left_drift, right_drift)
        effective_drift = pt.maximum(effective_drift, 0.05)
        
        # Compute RT prediction
        decision_time = threshold / effective_drift
        predicted_rt = decision_time + ndt
        
        return predicted_rt
    
    def run_mcmc_sampling(self, n_samples: int = 2000, n_tune: int = 1000) -> az.InferenceData:
        """Run MCMC sampling to estimate parameters"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_pymc_model first.")
        
        print(f"🔬 Running MCMC sampling ({n_tune} tune + {n_samples} samples)...")
        
        with self.model:
            # Use NUTS sampler
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                target_accept=0.95,
                chains=4,
                cores=2,
                random_seed=42,
                progressbar=True,
                return_inferencedata=True
            )
        
        self.trace = trace
        print("✅ MCMC sampling completed")
        return trace
    
    def analyze_results(self, data: Dict) -> Dict:
        """Analyze MCMC results and compare predictions"""
        
        if self.trace is None:
            raise ValueError("No MCMC trace available. Run sampling first.")
        
        print("📈 Analyzing MCMC results...")
        
        # Extract posterior means
        posterior_means = {}
        for var in self.trace.posterior.data_vars:
            posterior_means[var] = float(self.trace.posterior[var].mean())
        
        # Generate predictions using posterior means
        predictions = self._generate_predictions(data, posterior_means)
        
        # Calculate fit metrics
        actual_rts = data['rts']
        predicted_rts = predictions['predicted_rts']
        
        rt_correlation = np.corrcoef(actual_rts, predicted_rts)[0, 1]
        rt_rmse = np.sqrt(np.mean((actual_rts - predicted_rts)**2))
        
        results = {
            'posterior_means': posterior_means,
            'predictions': predictions,
            'rt_correlation': rt_correlation,
            'rt_rmse': rt_rmse,
            'mean_actual_rt': np.mean(actual_rts),
            'mean_predicted_rt': np.mean(predicted_rts),
            'trace': self.trace
        }
        
        print(f"   RT correlation: {rt_correlation:.3f}")
        print(f"   RT RMSE: {rt_rmse:.3f}")
        print(f"   Mean actual RT: {np.mean(actual_rts):.3f}")
        print(f"   Mean predicted RT: {np.mean(predicted_rts):.3f}")
        
        return results
    
    def _generate_predictions(self, data: Dict, params: Dict) -> Dict:
        """Generate predictions using estimated parameters"""
        
        predictions = []
        
        for i in range(data['n_trials']):
            left_stim = data['left_stimuli'][i]
            right_stim = data['right_stimuli'][i]
            response = data['responses'][i]
            
            # Determine drift rates
            if left_stim == 1:  # Vertical stimulus
                left_v_v = params['left_v_vertical']
                left_v_nv = params['left_v_nonvertical_error']
            else:  # Nonvertical stimulus
                left_v_v = params['left_v_vertical_error']
                left_v_nv = params['left_v_nonvertical']
            
            if right_stim == 1:  # Vertical stimulus
                right_v_v = params['right_v_vertical']
                right_v_nv = params['right_v_nonvertical_error']
            else:  # Nonvertical stimulus
                right_v_v = params['right_v_vertical_error']
                right_v_nv = params['right_v_nonvertical']
            
            # Determine which drift rates were used for this response
            if response in [1, 2]:  # Left vertical response
                left_drift = left_v_v
            else:  # Left nonvertical response
                left_drift = left_v_nv
                
            if response in [0, 1]:  # Right vertical response
                right_drift = right_v_v
            else:  # Right nonvertical response
                right_drift = right_v_nv
            
            # ParallelAND: Use minimum drift rate
            effective_drift = min(left_drift, right_drift)
            effective_drift = max(effective_drift, 0.05)
            
            # Predict RT
            decision_time = params['threshold'] / effective_drift
            predicted_rt = decision_time + params['ndt']
            
            predictions.append({
                'trial': i,
                'left_stimulus': left_stim,
                'right_stimulus': right_stim,
                'response': response,
                'left_drift': left_drift,
                'right_drift': right_drift,
                'effective_drift': effective_drift,
                'predicted_rt': predicted_rt,
                'actual_rt': data['rts'][i]
            })
        
        return {
            'trial_predictions': predictions,
            'predicted_rts': [p['predicted_rt'] for p in predictions]
        }
    
    def save_results(self, results: Dict, data: Dict, output_prefix: str = "pymc_parallel_and"):
        """Save PyMC results to files"""
        
        participant_id = data['participant_id']
        
        # Save trace summary
        summary = az.summary(self.trace)
        summary.to_csv(f"{output_prefix}_trace_summary_p{participant_id}.csv")
        print(f"📁 Trace summary saved: {output_prefix}_trace_summary_p{participant_id}.csv")
        
        # Save predictions
        predictions_df = pd.DataFrame(results['predictions']['trial_predictions'])
        predictions_df.to_csv(f"{output_prefix}_predictions_p{participant_id}.csv", index=False)
        print(f"📁 Predictions saved: {output_prefix}_predictions_p{participant_id}.csv")
        
        # Save posterior means
        posterior_df = pd.DataFrame([results['posterior_means']])
        posterior_df.to_csv(f"{output_prefix}_posterior_means_p{participant_id}.csv", index=False)
        print(f"📁 Posterior means saved: {output_prefix}_posterior_means_p{participant_id}.csv")
        
        # Save fit metrics
        fit_metrics = {
            'participant_id': participant_id,
            'n_trials': data['n_trials'],
            'rt_correlation': results['rt_correlation'],
            'rt_rmse': results['rt_rmse'],
            'mean_actual_rt': results['mean_actual_rt'],
            'mean_predicted_rt': results['mean_predicted_rt']
        }
        fit_df = pd.DataFrame([fit_metrics])
        fit_df.to_csv(f"{output_prefix}_fit_metrics_p{participant_id}.csv", index=False)
        print(f"📁 Fit metrics saved: {output_prefix}_fit_metrics_p{participant_id}.csv")
    
    def plot_results(self, results: Dict, data: Dict, save_path: str = "pymc_parallel_and_results.png"):
        """Plot PyMC results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Trace plots
        az.plot_trace(self.trace, var_names=['left_v_vertical', 'right_v_vertical', 'threshold'], 
                     axes=axes[0, :])
        
        # RT prediction
        actual_rts = data['rts']
        predicted_rts = results['predictions']['predicted_rts']
        
        axes[1, 0].scatter(actual_rts, predicted_rts, alpha=0.6)
        axes[1, 0].plot([min(actual_rts), max(actual_rts)], [min(actual_rts), max(actual_rts)], 'r--')
        axes[1, 0].set_xlabel('Actual RT')
        axes[1, 0].set_ylabel('Predicted RT')
        axes[1, 0].set_title(f'RT Prediction (r={results["rt_correlation"]:.3f})')
        
        # RT distribution
        axes[1, 1].hist(actual_rts, bins=30, alpha=0.7, label='Actual', density=True)
        axes[1, 1].hist(predicted_rts, bins=30, alpha=0.7, label='Predicted', density=True)
        axes[1, 1].set_xlabel('RT')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('RT Distribution')
        axes[1, 1].legend()
        
        # Parameter correlations
        posterior_df = self.trace.posterior.to_dataframe().reset_index()
        key_params = ['left_v_vertical', 'right_v_vertical', 'threshold', 'ndt']
        if len(key_params) >= 2:
            axes[1, 2].scatter(posterior_df[key_params[0]], posterior_df[key_params[1]], alpha=0.3)
            axes[1, 2].set_xlabel(key_params[0])
            axes[1, 2].set_ylabel(key_params[1])
            axes[1, 2].set_title('Parameter Correlation')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results plot saved: {save_path}")
        
        return fig

def run_pymc_parallel_and_analysis():
    """Run complete PyMC ParallelAND analysis"""
    
    print("🚀 Starting PyMC ParallelAND Analysis")
    
    try:
        # Initialize
        pymc_lba = PyMCParallelANDLBA()
        
        # Load data
        data = pymc_lba.load_data()
        
        # Create model
        model = pymc_lba.create_pymc_model(data)
        
        # Run sampling
        trace = pymc_lba.run_mcmc_sampling(n_samples=1000, n_tune=1000)
        
        # Analyze results
        results = pymc_lba.analyze_results(data)
        
        # Save results
        pymc_lba.save_results(results, data)
        
        # Plot results
        pymc_lba.plot_results(results, data)
        
        print("✅ PyMC ParallelAND analysis completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ PyMC analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_pymc_parallel_and_analysis()