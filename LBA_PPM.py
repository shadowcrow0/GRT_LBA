"""
LBA Analysis - Posterior Predictive Module (FIXED VERSION)
處理後驗預測檢查相關功能 - 修復預測生成問題
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def improved_lba_simulation(v_correct, v_incorrect, b, A, t0, n_predictions=1000):
    """
    Improved LBA simulation that generates more realistic predictions.
    Uses proper LBA mechanics with linear accumulation.
    """
    
    predictions = np.empty((n_predictions, 2))
    
    for i in range(n_predictions):
        # Starting points (uniform between 0 and A)
        start_correct = np.random.uniform(0, A)
        start_incorrect = np.random.uniform(0, A)
        
        # Remaining distance to boundary
        remaining_correct = b - start_correct
        remaining_incorrect = b - start_incorrect
        
        # Time to reach boundary (with some noise)
        # Add small random noise to prevent deterministic results
        noise_factor = 0.1  # 10% noise
        
        if v_correct > 0:
            time_correct = remaining_correct / v_correct
            time_correct *= np.random.normal(1.0, noise_factor)  # Add noise
        else:
            time_correct = np.inf
            
        if v_incorrect > 0:
            time_incorrect = remaining_incorrect / v_incorrect
            time_incorrect *= np.random.normal(1.0, noise_factor)  # Add noise
        else:
            time_incorrect = np.inf
        
        # Winner takes all
        if time_correct < time_incorrect:
            response_time = time_correct + t0
            response = 1  # Correct
        else:
            response_time = time_incorrect + t0
            response = 0  # Incorrect
        
        # Convert to milliseconds and ensure reasonable bounds
        response_time_ms = response_time * 1000
        response_time_ms = np.clip(response_time_ms, 200, 5000)  # Reasonable RT bounds
        
        predictions[i, 0] = response_time_ms
        predictions[i, 1] = response
    
    return predictions

def generate_predictions_from_trace_fixed(trace, n_predictions=1000):
    """
    FIXED version: Generate predictions from trace with better parameter handling.
    """
    
    try:
        # Try to use safe parameter extraction if available
        try:
            from LBA_tool_fixes import safe_parameter_extraction
            
            v_correct_samples = safe_parameter_extraction(trace, 'v_final_correct')
            v_incorrect_samples = safe_parameter_extraction(trace, 'v_final_incorrect')
            b_samples = safe_parameter_extraction(trace, 'b_safe')
            A_samples = safe_parameter_extraction(trace, 'start_var')
            t0_samples = safe_parameter_extraction(trace, 'non_decision')
            
            print(f"    ✓ 使用安全參數提取方法")
            
        except ImportError:
            print(f"    ⚠️ 使用基本參數提取方法")
            # Fallback to basic extraction
            v_correct_samples = trace.posterior['v_final_correct'].values.flatten()
            v_incorrect_samples = trace.posterior['v_final_incorrect'].values.flatten()
            b_samples = trace.posterior['b_safe'].values.flatten()
            A_samples = trace.posterior['start_var'].values.flatten()
            t0_samples = trace.posterior['non_decision'].values.flatten()
        
        # Ensure all arrays have the same length
        min_length = min(len(v_correct_samples), len(v_incorrect_samples), 
                        len(b_samples), len(A_samples), len(t0_samples))
        
        if min_length == 0:
            raise ValueError("No valid parameter samples found")
        
        # Truncate to minimum length
        v_correct_samples = v_correct_samples[:min_length]
        v_incorrect_samples = v_incorrect_samples[:min_length]
        b_samples = b_samples[:min_length]
        A_samples = A_samples[:min_length]
        t0_samples = t0_samples[:min_length]
        
        print(f"    參數樣本數: {min_length}")
        print(f"    v_correct 範圍: [{v_correct_samples.min():.2f}, {v_correct_samples.max():.2f}]")
        print(f"    v_incorrect 範圍: [{v_incorrect_samples.min():.2f}, {v_incorrect_samples.max():.2f}]")
        print(f"    boundary 範圍: [{b_samples.min():.2f}, {b_samples.max():.2f}]")
        
        # Generate predictions using improved simulation
        predictions = np.empty((n_predictions, 2))
        
        for i in range(n_predictions):
            # Randomly select parameter set
            idx = np.random.randint(0, min_length)
            
            v_c = float(v_correct_samples[idx])
            v_i = float(v_incorrect_samples[idx])
            b = float(b_samples[idx])
            A = float(A_samples[idx])
            t0 = float(t0_samples[idx])
            
            # Ensure reasonable parameter values
            v_c = np.clip(v_c, 0.1, 10.0)
            v_i = np.clip(v_i, 0.1, 10.0)
            b = np.clip(b, A + 0.1, 10.0)
            A = np.clip(A, 0.1, 3.0)
            t0 = np.clip(t0, 0.01, 1.0)
            
            # Generate single prediction
            single_pred = improved_lba_simulation(v_c, v_i, b, A, t0, n_predictions=1)
            predictions[i] = single_pred[0]
        
        print(f"    ✓ 生成了 {n_predictions} 個預測")
        print(f"    預測準確率: {predictions[:, 1].mean():.3f}")
        print(f"    預測RT範圍: [{predictions[:, 0].min():.1f}, {predictions[:, 0].max():.1f}] ms")
        
        return predictions
        
    except Exception as e:
        print(f"    ❌ 預測生成失敗: {e}")
        import traceback
        print(f"    詳細錯誤: {traceback.format_exc()}")
        return None

def create_linear_accumulation_plot(models, participant_id, save_dir):
    """
    Create accumulation plots that look more like the linear LBA accumulation.
    """
    
    print(f"  創建線性累積圖...")
    
    n_models = len(models)
    if n_models == 0:
        return None
    
    fig, axes = plt.subplots(2, n_models, figsize=(8*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Linear LBA Accumulation - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    for i, (model_name, trace) in enumerate(models.items()):
        try:
            # Extract parameters safely
            try:
                from LBA_tool_fixes import safe_parameter_extraction
                v_correct_samples = safe_parameter_extraction(trace, 'v_final_correct')
                v_incorrect_samples = safe_parameter_extraction(trace, 'v_final_incorrect')
                b_samples = safe_parameter_extraction(trace, 'b_safe')
                A_samples = safe_parameter_extraction(trace, 'start_var')
                t0_samples = safe_parameter_extraction(trace, 'non_decision')
            except ImportError:
                v_correct_samples = trace.posterior['v_final_correct'].values.flatten()
                v_incorrect_samples = trace.posterior['v_final_incorrect'].values.flatten()
                b_samples = trace.posterior['b_safe'].values.flatten()
                A_samples = trace.posterior['start_var'].values.flatten()
                t0_samples = trace.posterior['non_decision'].values.flatten()
            
            # Use mean parameter values for cleaner visualization
            v_c = np.mean(v_correct_samples)
            v_i = np.mean(v_incorrect_samples)
            b = np.mean(b_samples)
            A = np.mean(A_samples)
            t0 = np.mean(t0_samples)
            
            # Ensure reasonable values
            v_c = np.clip(v_c, 0.5, 5.0)
            v_i = np.clip(v_i, 0.1, 3.0)
            b = np.clip(b, A + 0.1, 5.0)
            A = np.clip(A, 0.1, 2.0)
            t0 = np.clip(t0, 0.05, 0.5)
            
            # TOP PLOT: Linear accumulation traces
            ax1 = axes[0, i]
            
            # Time vector
            max_time = 2.0  # seconds
            dt = 0.01
            time_points = np.arange(0, max_time, dt)
            
            # Generate multiple accumulation traces
            n_trials = 8
            for trial in range(n_trials):
                # Random starting points
                start_c = np.random.uniform(0, A)
                start_i = np.random.uniform(0, A)
                
                # Linear accumulation with small noise
                acc_correct = start_c + v_c * time_points + np.random.normal(0, 0.1, len(time_points))
                acc_incorrect = start_i + v_i * time_points + np.random.normal(0, 0.1, len(time_points))
                
                # Find decision time
                decision_times_c = np.where(acc_correct >= b)[0]
                decision_times_i = np.where(acc_incorrect >= b)[0]
                
                if len(decision_times_c) > 0:
                    decision_t_c = decision_times_c[0]
                else:
                    decision_t_c = len(time_points)
                    
                if len(decision_times_i) > 0:
                    decision_t_i = decision_times_i[0]
                else:
                    decision_t_i = len(time_points)
                
                # Determine winner
                if decision_t_c < decision_t_i:
                    winner = 'correct'
                    decision_t = decision_t_c
                else:
                    winner = 'incorrect'
                    decision_t = decision_t_i
                
                # Plot up to decision time
                ax1.plot(time_points[:decision_t+1], acc_correct[:decision_t+1], 
                        color='blue', alpha=0.6, linewidth=2)
                ax1.plot(time_points[:decision_t+1], acc_incorrect[:decision_t+1], 
                        color='red', alpha=0.6, linewidth=2)
                
                # Mark decision point
                if winner == 'correct':
                    ax1.scatter(time_points[decision_t], acc_correct[decision_t], 
                              color='blue', s=50, zorder=5)
                else:
                    ax1.scatter(time_points[decision_t], acc_incorrect[decision_t], 
                              color='red', s=50, zorder=5)
            
            # Add boundary and start region
            ax1.axhline(y=b, color='black', linestyle='--', linewidth=3, label='Decision Boundary')
            ax1.axhspan(0, A, alpha=0.2, color='gray', label='Start Point Variation')
            
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Evidence')
            ax1.set_title(f'{model_name}\nLinear Accumulation')
            ax1.set_ylim(0, b * 1.2)
            ax1.set_xlim(0, 1.5)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # BOTTOM PLOT: Parameter summary
            ax2 = axes[1, i]
            
            param_names = ['v_correct', 'v_incorrect', 'boundary', 'start_var', 'non_decision']
            param_values = [v_c, v_i, b, A, t0]
            
            bars = ax2.bar(param_names, param_values, alpha=0.7, 
                          color=['blue', 'red', 'black', 'gray', 'green'])
            
            ax2.set_title(f'{model_name}\nParameter Values')
            ax2.set_ylabel('Parameter Value')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, param_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                        
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'{model_name}\nError: {str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, 'See above', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f'participant_{participant_id}_linear_accumulation.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Linear accumulation plot saved: {plot_file}")
    return plot_file

def posterior_predictive_check_fixed(models, participant_data, participant_id, save_dir):
    """
    FIXED & HARDENED: Posterior predictive check with better prediction generation
    and robust plotting that handles empty data arrays.
    """
    
    print(f"  執行修復且強化的後驗預測檢查...")
    
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Posterior Predictive Check (Fixed) - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    # Observed data statistics
    obs_rt = participant_data[:, 0]
    obs_response = participant_data[:, 1]
    obs_rt_correct = obs_rt[obs_response == 1]
    obs_rt_incorrect = obs_rt[obs_response == 0]
    obs_accuracy = obs_response.mean()
    
    print(f"    觀察數據: 準確率={obs_accuracy:.3f}, RT範圍=[{obs_rt.min():.1f}, {obs_rt.max():.1f}]")
    
    fit_results = {}
    
    for i, (model_name, trace) in enumerate(models.items()):
        ax_rt = axes[0, i]
        ax_stats = axes[1, i]
        try:
            print(f"    處理模型: {model_name}")
            
            predictions = generate_predictions_from_trace_fixed(trace, n_predictions=1000)
            
            if predictions is not None and len(predictions) > 0:
                pred_rt = predictions[:, 0]
                pred_response = predictions[:, 1]
                pred_rt_correct = pred_rt[pred_response == 1]
                pred_rt_incorrect = pred_rt[pred_response == 0]
                pred_accuracy = pred_response.mean() if len(pred_response) > 0 else 0
                
                print(f"      預測結果: 準確率={pred_accuracy:.3f}")
                
                # TOP PLOT: RT distributions (with checks for empty data)
                if len(obs_rt_correct) > 0:
                    ax_rt.hist(obs_rt_correct, bins=15, alpha=0.7, color='blue', 
                              label=f'Observed Correct (n={len(obs_rt_correct)})', density=True)
                if len(pred_rt_correct) > 0:
                    ax_rt.hist(pred_rt_correct, bins=15, alpha=0.5, color='lightblue', 
                              label=f'Predicted Correct', density=True, histtype='step', linewidth=3)

                if len(obs_rt_incorrect) > 0:
                    ax_rt.hist(obs_rt_incorrect, bins=15, alpha=0.7, color='red', 
                              label=f'Observed Incorrect (n={len(obs_rt_incorrect)})', density=True)
                if len(pred_rt_incorrect) > 0:
                    ax_rt.hist(pred_rt_incorrect, bins=15, alpha=0.5, color='lightcoral', 
                              label=f'Predicted Incorrect', density=True, histtype='step', linewidth=3)
                else: # Add a note if no incorrect responses were predicted
                    ax_rt.text(0.95, 0.95, 'No incorrect\nresponses predicted', transform=ax_rt.transAxes,
                               ha='right', va='top', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

                ax_rt.set_xlabel('Reaction Time (ms)')
                ax_rt.set_ylabel('Density')
                ax_rt.set_title(f'{model_name}\nRT Distributions')
                ax_rt.legend()
                ax_rt.grid(True, alpha=0.3)
                
                # BOTTOM PLOT: Statistics comparison (with checks for empty data)
                stats_names = ['Accuracy', 'Mean RT\n(All)', 'Mean RT\n(Correct)', 'Mean RT\n(Incorrect)']
                obs_values = [
                    obs_accuracy,
                    np.mean(obs_rt) if len(obs_rt) > 0 else 0,
                    np.mean(obs_rt_correct) if len(obs_rt_correct) > 0 else 0,
                    np.mean(obs_rt_incorrect) if len(obs_rt_incorrect) > 0 else 0
                ]
                pred_values = [
                    pred_accuracy,
                    np.mean(pred_rt) if len(pred_rt) > 0 else 0,
                    np.mean(pred_rt_correct) if len(pred_rt_correct) > 0 else 0,
                    np.mean(pred_rt_incorrect) if len(pred_rt_incorrect) > 0 else 0
                ]
                
                x_pos = np.arange(len(stats_names))
                width = 0.35
                
                ax_stats.bar(x_pos - width/2, obs_values, width, label='Observed', alpha=0.8, color='darkblue')
                ax_stats.bar(x_pos + width/2, pred_values, width, label='Predicted', alpha=0.8, color='darkred')
                
                ax_stats.set_xlabel('Statistics')
                ax_stats.set_ylabel('Value')
                ax_stats.set_title(f'{model_name}\nStatistics Comparison')
                ax_stats.set_xticks(x_pos)
                ax_stats.set_xticklabels(stats_names)
                ax_stats.legend()
                ax_stats.grid(True, alpha=0.3)
                
                fit_results[model_name] = {'success': True}
            else:
                raise ValueError("Prediction generation returned None or empty")
                
        except Exception as e:
            print(f"      ❌ {model_name} 預測或繪圖失敗: {e}")
            ax_rt.text(0.5, 0.5, f'{model_name}\nPrediction/Plotting Failed', 
                       ha='center', va='center', transform=ax_rt.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round", facecolor="red", alpha=0.3))
            ax_stats.text(0.5, 0.5, 'See above', 
                       ha='center', va='center', transform=ax_stats.transAxes)
            fit_results[model_name] = {'success': False, 'error': str(e)}
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f'participant_{participant_id}_posterior_predictive_check_fixed.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Fixed posterior predictive check saved: {plot_file}")
    return fit_results
# Wrapper function to replace the original
def run_comprehensive_ppc(models, participant_data, participant_id, save_dir):
    """
    Run comprehensive PPC with fixes - wrapper for backward compatibility.
    """
    print(f"\n🔍 Running FIXED comprehensive posterior predictive check...")
    
    results = {}
    
    # 1. Fixed posterior predictive check
    fit_results = posterior_predictive_check_fixed(models, participant_data, participant_id, save_dir)
    results['fit_results'] = fit_results
    
    # 2. Create linear accumulation plot
    linear_plot = create_linear_accumulation_plot(models, participant_id, save_dir)
    results['linear_accumulation_plot'] = linear_plot
    
    # 3. Create summary
    successful_models = [name for name, result in fit_results.items() if result.get('success', False)]
    failed_models = [name for name, result in fit_results.items() if not result.get('success', False)]
    
    print(f"    成功模型: {successful_models}")
    print(f"    失敗模型: {failed_models}")
    
    results['summary'] = {
        'successful_models': successful_models,
        'failed_models': failed_models,
        'n_successful': len(successful_models),
        'n_failed': len(failed_models)
    }
    
    print(f"✓ Fixed comprehensive PPC completed for participant {participant_id}")
    return results