"""
LBA Analysis - Information Accumulation Module (FULLY FIXED VERSION)
處理信息累積相關的模擬和可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_lba_accumulation(v_correct, v_incorrect, b, A, t0, n_trials=10, s_noise=0.1):
    """
    FIXED: Simulates the LBA accumulation process with linear paths.
    Variability comes from start points and drift rates between trials.
    """
    results = {
        'correct_paths': [],
        'incorrect_paths': [],
        'boundary': b,
        'start_var': A,
        't0': t0
    }
    max_time = 2.0  # Max time to plot in seconds

    for _ in range(n_trials):
        # 1. Draw start points from a uniform distribution
        start_correct = np.random.uniform(0, A)
        start_incorrect = np.random.uniform(0, A)

        # 2. Draw drift rates for this trial (add inter-trial noise)
        # The LBA model assumes drift rates are variable across trials.
        # We use the mean drift from posterior and add a small noise term.
        rate_correct = np.random.normal(v_correct, s_noise)
        rate_incorrect = np.random.normal(v_incorrect, s_noise)
        
        # Ensure rates are positive
        rate_correct = max(rate_correct, 0.01)
        rate_incorrect = max(rate_incorrect, 0.01)

        # 3. Calculate time to hit the boundary (time = distance / speed)
        time_to_boundary_c = (b - start_correct) / rate_correct
        time_to_boundary_i = (b - start_incorrect) / rate_incorrect

        # 4. Determine the winner and the decision time
        decision_time = min(time_to_boundary_c, time_to_boundary_i)
        
        # We only plot up to the decision time, or max_time
        plot_duration = min(decision_time, max_time)
        time_points = np.linspace(0, plot_duration, 100)

        # 5. Generate the linear paths
        path_correct = start_correct + rate_correct * time_points
        path_incorrect = start_incorrect + rate_incorrect * time_points
        
        results['correct_paths'].append((time_points, path_correct))
        results['incorrect_paths'].append((time_points, path_incorrect))

    return results
def safe_extract_parameter_samples(trace, param_name, flatten=True):
    """
    Safely extract parameter samples from trace, handling different shapes and dimensions.
    """
    try:
        if param_name not in trace.posterior:
            print(f"    Warning: Parameter '{param_name}' not found in trace")
            return None
        
        # Get the parameter data
        param_data = trace.posterior[param_name]
        
        # Get the values as numpy array
        values = param_data.values
        
        # Handle different scenarios for flattening
        if flatten:
            # For parameters that vary by trial, we typically want to average across trials first
            # Check if there's a trial dimension (usually the last dimension)
            if len(values.shape) > 2:  # (chain, draw, trial) or similar
                # Average across trials (last dimension)
                values = np.mean(values, axis=-1)
            
            # Now flatten across chains and draws
            values = values.flatten()
        
        return values
        
    except Exception as e:
        print(f"    Error extracting parameter '{param_name}': {e}")
        return None

def create_accumulator_plot(trace, model_name, participant_id, save_dir, n_simulations=4):
    """
    Creates an LBA information accumulation plot for a participant.
    FIXED VERSION: Uses the corrected linear simulation.
    """
    print(f"  Creating accumulator plot for {model_name}...")
    try:
        # Use safe extraction method to get posterior means for a clean plot
        v_correct_mean = np.mean(safe_extract_parameter_samples(trace, 'v_final_correct'))
        v_incorrect_mean = np.mean(safe_extract_parameter_samples(trace, 'v_final_incorrect'))
        b_mean = np.mean(safe_extract_parameter_samples(trace, 'b_safe'))
        A_mean = np.mean(safe_extract_parameter_samples(trace, 'start_var'))
        t0_mean = np.mean(safe_extract_parameter_samples(trace, 'non_decision'))

        if any(np.isnan(x) for x in [v_correct_mean, v_incorrect_mean, b_mean, A_mean, t0_mean]):
            print("    ❌ Failed to extract one or more parameters")
            return False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Participant {participant_id}\nCorrected LBA Information Accumulation',
                     fontsize=16, fontweight='bold')
        
        # Use a single parameter set (the mean) for all 4 subplots for consistency
        # but simulate different trial paths
        for i in range(n_simulations):
            ax = axes[i // 2, i % 2]
            
            # Simulate new trials each time
            simulation = simulate_lba_accumulation(v_correct_mean, v_incorrect_mean, b_mean, A_mean, t0_mean, n_trials=8)
            
            for (time_points, path) in simulation['correct_paths']:
                ax.plot(time_points, path, color='blue', alpha=0.6)
            for (time_points, path) in simulation['incorrect_paths']:
                ax.plot(time_points, path, color='red', alpha=0.6)

            ax.plot([], [], color='blue', label='Correct Accumulator')
            ax.plot([], [], color='red', label='Incorrect Accumulator')
            ax.axhline(y=simulation['boundary'], color='black', linestyle='--', linewidth=2, label='Decision Boundary')
            ax.axhspan(0, simulation['start_var'], alpha=0.2, color='gray', label='Start Point Variation')
            ax.set(xlabel='Time (seconds)', ylabel='Evidence', 
                   title=f'Simulation Set {i+1}\n(v_c={v_correct_mean:.2f}, v_i={v_incorrect_mean:.2f})', 
                   ylim=(0, simulation['boundary'] * 1.2), xlim=(0, 1.5))
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plot_file = os.path.join(save_dir, f'participant_{participant_id}_{model_name}_accumulation_FIXED.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ FIXED accumulator plot saved: {plot_file}")
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to create accumulator plot: {e}")
        import traceback
        print(f"    Full traceback: {traceback.format_exc()}")
        return False
def create_accumulation_comparison_plot(models, participant_id, save_dir):
    """
    Compares the average accumulation processes of different models.
    FIXED VERSION: Handles parameter extraction properly.
    """
    print(f"  Creating accumulation comparison plot...")
    if len(models) < 2:
        print("    Need at least 2 models for comparison.")
        return False
    
    fig, axes = plt.subplots(1, len(models), figsize=(8*len(models), 6), sharey=True)
    if len(models) == 1: 
        axes = [axes]
    fig.suptitle(f'Accumulation Process Comparison - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    for i, (model_name, trace) in enumerate(models.items()):
        ax = axes[i]
        try:
            # FIXED: Use safe extraction and get mean values
            v_correct_samples = safe_extract_parameter_samples(trace, 'v_final_correct', flatten=True)
            v_incorrect_samples = safe_extract_parameter_samples(trace, 'v_final_incorrect', flatten=True)
            b_samples = safe_extract_parameter_samples(trace, 'b_safe', flatten=True)
            A_samples = safe_extract_parameter_samples(trace, 'start_var', flatten=True)
            t0_samples = safe_extract_parameter_samples(trace, 'non_decision', flatten=True)
            
            if any(x is None for x in [v_correct_samples, v_incorrect_samples, b_samples, A_samples, t0_samples]):
                raise ValueError("Failed to extract parameters")
            
            # Use mean values for comparison
            v_c = float(np.mean(v_correct_samples))
            v_i = float(np.mean(v_incorrect_samples))
            b = float(np.mean(b_samples))
            A = float(np.mean(A_samples))
            t0 = float(np.mean(t0_samples))
            
            # Ensure parameters are reasonable
            v_c = np.clip(v_c, 0.1, 10.0)
            v_i = np.clip(v_i, 0.1, 10.0)
            b = np.clip(b, A + 0.1, 10.0)
            A = np.clip(A, 0.1, 3.0)
            t0 = np.clip(t0, 0.01, 1.0)
            
            simulation = simulate_lba_accumulation(v_c, v_i, b, A, t0, n_trials=10)
            time_points = np.arange(len(simulation['correct_traces'][0])) * simulation['dt']

            for trial_idx in range(len(simulation['correct_traces'])):
                ax.plot(time_points, simulation['correct_traces'][trial_idx], color='blue', alpha=0.3)
                ax.plot(time_points, simulation['incorrect_traces'][trial_idx], color='red', alpha=0.3)
            
            ax.plot([], [], color='blue', label='Correct (Avg. Params)')
            ax.plot([], [], color='red', label='Incorrect (Avg. Params)')
            ax.axhline(y=b, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
            ax.axhspan(0, A, alpha=0.2, color='gray', label='Start Variation')
            ax.set(xlabel='Time (seconds)', ylabel='Evidence' if i==0 else '', 
                   title=f'{model_name}\n(v_c={v_c:.2f}, v_i={v_i:.2f})', xlim=(0, 2.0))
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plot_file = os.path.join(save_dir, f'participant_{participant_id}_accumulation_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Accumulation comparison plot saved")
    return True

def analyze_accumulation_dynamics(trace, model_name):
    """
    Analyzes accumulation dynamics characteristics.
    FIXED VERSION: Handles parameter extraction properly.
    """
    print(f"  Analyzing accumulation dynamics for {model_name}...")
    try:
        # FIXED: Use safe extraction method
        v_correct_samples = safe_extract_parameter_samples(trace, 'v_final_correct', flatten=True)
        v_incorrect_samples = safe_extract_parameter_samples(trace, 'v_final_incorrect', flatten=True)
        b_samples = safe_extract_parameter_samples(trace, 'b_safe', flatten=True)
        A_samples = safe_extract_parameter_samples(trace, 'start_var', flatten=True)
        t0_samples = safe_extract_parameter_samples(trace, 'non_decision', flatten=True)
        
        if any(x is None for x in [v_correct_samples, v_incorrect_samples, b_samples, A_samples, t0_samples]):
            raise ValueError("Failed to extract one or more parameters")
        
        # FIXED: Ensure all arrays have the same length
        min_length = min(len(v_correct_samples), len(v_incorrect_samples), 
                        len(b_samples), len(A_samples), len(t0_samples))
        
        # Truncate all arrays to the same length
        v_correct_samples = v_correct_samples[:min_length]
        v_incorrect_samples = v_incorrect_samples[:min_length]
        b_samples = b_samples[:min_length]
        A_samples = A_samples[:min_length]
        t0_samples = t0_samples[:min_length]
        
        # Avoid division by zero
        v_incorrect_samples = np.clip(v_incorrect_samples, 0.01, np.inf)
        A_samples = np.clip(A_samples, 0.01, np.inf)
        
        dynamics = {
            'model_name': model_name,
            'drift_ratio_mean': np.mean(v_correct_samples / v_incorrect_samples),
            'drift_ratio_std': np.std(v_correct_samples / v_incorrect_samples),
            'boundary_height_mean': np.mean(b_samples),
            'boundary_height_std': np.std(b_samples),
            'start_variability_mean': np.mean(A_samples),
            'start_variability_std': np.std(A_samples),
            'non_decision_time_mean': np.mean(t0_samples),
            'non_decision_time_std': np.std(t0_samples),
            'relative_boundary_mean': np.mean(b_samples / A_samples),
            'relative_boundary_std': np.std(b_samples / A_samples),
            'n_samples': min_length
        }
        
        print(f"    ✓ Dynamics analysis completed with {min_length} samples")
        return dynamics
        
    except Exception as e:
        print(f"    ❌ Error analyzing dynamics: {e}")
        import traceback
        print(f"    Full traceback: {traceback.format_exc()}")
        return None

def create_dynamics_summary_plot(dynamics_results, participant_id, save_dir):
    """
    Creates a summary plot of accumulation dynamics.
    """
    if not dynamics_results or len(dynamics_results) < 1: 
        print("    No dynamics results to plot")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Accumulation Dynamics Summary - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    model_names = [d['model_name'] for d in dynamics_results]
    metrics = ['drift_ratio', 'boundary_height', 'start_variability', 'non_decision_time', 'relative_boundary']
    titles = ['Drift Rate Ratio (v_c/v_i)', 'Decision Boundary (b)', 'Start Variability (A)', 
              'Non-Decision Time (t0)', 'Caution (b/A)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes.flatten()[i]
        means = [d[f'{metric}_mean'] for d in dynamics_results]
        stds = [d[f'{metric}_std'] for d in dynamics_results]
        
        # Filter out any NaN values
        valid_data = [(m, s, n) for m, s, n in zip(means, stds, model_names) 
                     if not (np.isnan(m) or np.isnan(s))]
        
        if valid_data:
            valid_means, valid_stds, valid_names = zip(*valid_data)
            bars = ax.bar(valid_names, valid_means, yerr=valid_stds, capsize=5, alpha=0.7, color='skyblue')
            ax.set_title(title)
            ax.grid(True, axis='y', alpha=0.5)
            
            # Rotate x-axis labels if needed
            if len(valid_names) > 2:
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes.flatten())):
        axes.flatten()[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_file = os.path.join(save_dir, f'participant_{participant_id}_dynamics_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Dynamics summary plot saved")
    return plot_file

def run_accumulation_analysis(models, participant_id, save_dir):
    """
    Runs the complete accumulation analysis pipeline.
    FIXED VERSION: Better error handling and parameter extraction.
    """
    print(f"\n🔬 Running accumulation analysis for participant {participant_id}...")
    results = {}
    dynamics_list = []
    
    if not models:
        print("    ❌ No models provided for analysis")
        return results
    
    for model_name, trace in models.items():
        print(f"\n  Processing model: {model_name}")
        
        # 1. Create accumulator plot
        plot_success = create_accumulator_plot(trace, model_name, participant_id, save_dir)
        results[f'{model_name}_accumulation_plot'] = plot_success
        
        # 2. Analyze dynamics
        dynamics = analyze_accumulation_dynamics(trace, model_name)
        if dynamics:
            results[f'{model_name}_dynamics'] = dynamics
            dynamics_list.append(dynamics)
            print(f"    ✓ Dynamics analysis completed")
        else:
            print(f"    ❌ Dynamics analysis failed")
            
    # 3. Create comparison plot if multiple models
    if len(models) > 1:
        print(f"\n  Creating model comparison plot...")
        comparison_success = create_accumulation_comparison_plot(models, participant_id, save_dir)
        results['comparison_plot'] = comparison_success
    
    # 4. Create dynamics summary if we have any dynamics results
    if dynamics_list:
        print(f"\n  Creating dynamics summary plot...")
        summary_plot_success = create_dynamics_summary_plot(dynamics_list, participant_id, save_dir)
        results['dynamics_summary_plot'] = summary_plot_success
    else:
        print("    ❌ No dynamics results available for summary plot")
    
    print(f"✓ Accumulation analysis completed for participant {participant_id}")
    print(f"    Results: {len([k for k, v in results.items() if v])} successful, "
          f"{len([k for k, v in results.items() if not v])} failed")
    
    return results