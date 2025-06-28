"""
Enhanced LBA Analysis with Information Accumulation Plots and Posterior Predictive Checks
修復語法錯誤並添加信息累積圖和後驗預測檢查功能
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import time
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
import seaborn as sns

# ===================================================================
# 修復的 Sigma Matrix 總結函數
# ===================================================================

def create_sigma_comparison_plots(sigma_df, save_dir):
    """創建 sigma matrix 的對比分析圖表 - 修復語法錯誤"""
    
    print("Creating sigma matrix comparison plots...")
    
    # 按模型分組
    models = sigma_df['model'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sigma Matrix Analysis Across All Participants', fontsize=16, fontweight='bold')
    
    # 1. 相關係數分布
    for i, model in enumerate(models):
        model_data = sigma_df[sigma_df['model'] == model]
        axes[0, 0].hist(model_data['correlation'], alpha=0.7, label=model, bins=15)
    axes[0, 0].set_xlabel('Correlation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Drift Rate Correlation Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 條件數分布
    for i, model in enumerate(models):
        model_data = sigma_df[sigma_df['model'] == model]
        axes[0, 1].hist(model_data['condition_number'], alpha=0.7, label=model, bins=15)
    axes[0, 1].set_xlabel('Condition Number')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Matrix Condition Number Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 變異數對比
    for i, model in enumerate(models):
        model_data = sigma_df[sigma_df['model'] == model]
        axes[0, 2].scatter(model_data['variance_A'], model_data['variance_B'], 
                          alpha=0.7, label=model, s=50)
    axes[0, 2].set_xlabel('Variance A (v_correct)')
    axes[0, 2].set_ylabel('Variance B (v_incorrect)')
    axes[0, 2].set_title('Variance Relationship')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 相關係數 vs 條件數
    for i, model in enumerate(models):
        model_data = sigma_df[sigma_df['model'] == model]
        axes[1, 0].scatter(model_data['correlation'], model_data['condition_number'], 
                          alpha=0.7, label=model, s=50)
    axes[1, 0].set_xlabel('Correlation')
    axes[1, 0].set_ylabel('Condition Number')
    axes[1, 0].set_title('Correlation vs Condition Number')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 模型間相關係數對比 - 修復語法錯誤
    if len(models) == 2:
        model1_data = sigma_df[sigma_df['model'] == models[0]]
        model2_data = sigma_df[sigma_df['model'] == models[1]]
        
        # 確保參與者順序一致
        common_participants = set(model1_data['participant']) & set(model2_data['participant'])
        model1_corr = model1_data[model1_data['participant'].isin(common_participants)].sort_values('participant')['correlation'].values
        model2_corr = model2_data[model2_data['participant'].isin(common_participants)].sort_values('participant')['correlation'].values
        
        axes[1, 1].scatter(model1_corr, model2_corr, alpha=0.7, s=50)
        axes[1, 1].plot([min(model1_corr.min(), model2_corr.min()), 
                        max(model1_corr.max(), model2_corr.max())], 
                       [min(model1_corr.min(), model2_corr.min()), 
                        max(model1_corr.max(), model2_corr.max())], 
                       'r--', alpha=0.8)
        axes[1, 1].set_xlabel(f'{models[0]} Correlation')
        axes[1, 1].set_ylabel(f'{models[1]} Correlation')
        axes[1, 1].set_title('Model Correlation Comparison')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need 2 models\nfor comparison', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # 6. 綜合統計摘要
    summary_text = []
    for model in models:
        model_data = sigma_df[sigma_df['model'] == model]
        summary_text.append(f"{model}:")
        summary_text.append(f"  Mean corr: {model_data['correlation'].mean():.3f}")
        summary_text.append(f"  Mean cond: {model_data['condition_number'].mean():.1f}")
        summary_text.append(f"  N participants: {len(model_data)}")
        summary_text.append("")
    
    axes[1, 2].text(0.05, 0.95, '\n'.join(summary_text), 
                   transform=axes[1, 2].transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # 保存圖表
    plot_file = os.path.join(save_dir, 'sigma_matrix_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sigma matrix comparison plot saved: {plot_file}")

# ===================================================================
# 新增：LBA 信息累積圖函數
# ===================================================================

def simulate_lba_accumulation(v_correct, v_incorrect, b, A, t0, n_trials=100, dt=0.001):
    """
    模擬 LBA 累積過程，返回累積軌跡
    
    Parameters:
    - v_correct, v_incorrect: 漂移率
    - b: 決策邊界
    - A: 起始點變異
    - t0: 非決策時間
    - n_trials: 模擬試驗數
    - dt: 時間步長
    
    Returns:
    - accumulation_traces: 累積軌跡字典
    """
    
    max_time = 3.0  # 最大模擬時間（秒）
    time_steps = int(max_time / dt)
    time_grid = np.arange(0, max_time, dt)
    
    # 存儲所有試驗的累積軌跡
    correct_traces = []
    incorrect_traces = []
    decision_times = []
    decisions = []
    
    for trial in range(min(n_trials, 50)):  # 限制數量以避免內存問題
        # 隨機起始點
        start_correct = np.random.uniform(0, A)
        start_incorrect = np.random.uniform(0, A)
        
        # 初始化累積器
        acc_correct = np.zeros(time_steps)
        acc_incorrect = np.zeros(time_steps)
        acc_correct[0] = start_correct
        acc_incorrect[0] = start_incorrect
        
        # 模擬累積過程
        decision_made = False
        decision_time = max_time
        winner = 0  # 0: correct, 1: incorrect
        
        for t in range(1, time_steps):
            # 添加隨機噪音（布朗運動）
            noise_correct = np.random.normal(0, np.sqrt(dt))
            noise_incorrect = np.random.normal(0, np.sqrt(dt))
            
            # 更新累積器
            acc_correct[t] = acc_correct[t-1] + v_correct * dt + noise_correct
            acc_incorrect[t] = acc_incorrect[t-1] + v_incorrect * dt + noise_incorrect
            
            # 檢查是否達到邊界
            if not decision_made:
                if acc_correct[t] >= b:
                    decision_made = True
                    decision_time = time_grid[t]
                    winner = 0
                    break
                elif acc_incorrect[t] >= b:
                    decision_made = True
                    decision_time = time_grid[t]
                    winner = 1
                    break
        
        # 截斷到決策時間
        if decision_made:
            decision_idx = int(decision_time / dt)
            correct_traces.append(acc_correct[:decision_idx+1])
            incorrect_traces.append(acc_incorrect[:decision_idx+1])
        else:
            correct_traces.append(acc_correct)
            incorrect_traces.append(acc_incorrect)
        
        decision_times.append(decision_time + t0)
        decisions.append(winner)
    
    return {
        'correct_traces': correct_traces,
        'incorrect_traces': incorrect_traces,
        'decision_times': decision_times,
        'decisions': decisions,
        'time_grid': time_grid,
        'dt': dt,
        'boundary': b,
        'start_var': A
    }

def create_accumulator_plot(trace, model_name, participant_id, save_dir, n_simulations=4):
    """
    為每個參與者創建 LBA 信息累積圖（四條線）
    """
    
    print(f"  Creating accumulator plot for {model_name}...")
    
    try:
        # 提取模型參數的後驗樣本
        v_correct_samples = trace.posterior['v_final_correct'].values.flatten()
        v_incorrect_samples = trace.posterior['v_final_incorrect'].values.flatten()
        b_samples = trace.posterior['b_safe'].values.flatten()
        A_samples = trace.posterior['start_var'].values.flatten()
        t0_samples = trace.posterior['non_decision'].values.flatten()
        
        # 選擇幾組參數進行模擬
        n_posterior_samples = len(v_correct_samples)
        sample_indices = np.random.choice(n_posterior_samples, size=n_simulations, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Participant {participant_id}\nInformation Accumulation Process', 
                     fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, idx in enumerate(sample_indices):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # 當前參數組
            v_correct = v_correct_samples[idx]
            v_incorrect = v_incorrect_samples[idx]
            b = b_samples[idx]
            A = A_samples[idx]
            t0 = t0_samples[idx]
            
            # 模擬累積過程
            simulation = simulate_lba_accumulation(v_correct, v_incorrect, b, A, t0, n_trials=10)
            
            # 繪製累積軌跡
            for trial_idx in range(min(5, len(simulation['correct_traces']))):  # 最多顯示5條軌跡
                correct_trace = simulation['correct_traces'][trial_idx]
                incorrect_trace = simulation['incorrect_traces'][trial_idx]
                
                time_points = np.arange(len(correct_trace)) * simulation['dt']
                
                alpha = 0.7 if trial_idx == 0 else 0.4
                linewidth = 2 if trial_idx == 0 else 1
                
                ax.plot(time_points, correct_trace, color='blue', alpha=alpha, 
                       linewidth=linewidth, label='Correct' if trial_idx == 0 else "")
                ax.plot(time_points, incorrect_trace, color='red', alpha=alpha, 
                       linewidth=linewidth, label='Incorrect' if trial_idx == 0 else "")
            
            # 添加決策邊界
            ax.axhline(y=b, color='black', linestyle='--', linewidth=2, label='Decision boundary')
            
            # 添加起始變異區域
            ax.axhspan(0, A, alpha=0.2, color='gray', label='Start variation')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Evidence')
            ax.set_title(f'Simulation {i+1}\nv_c={v_correct:.2f}, v_i={v_incorrect:.2f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 設置合理的 y 軸範圍
            ax.set_ylim(-0.1, b * 1.2)
            ax.set_xlim(0, 2.0)  # 顯示前2秒
        
        plt.tight_layout()
        
        # 保存圖表
        plot_file = os.path.join(save_dir, f'participant_{participant_id}_{model_name}_accumulation.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Accumulator plot saved: {plot_file}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to create accumulator plot: {e}")
        return False

# ===================================================================
# 新增：後驗預測檢查函數
# ===================================================================

def posterior_predictive_check(models, participant_data, participant_id, save_dir):
    """
    執行後驗預測檢查，比較觀察數據與模型預測
    """
    
    print(f"  Performing posterior predictive check for participant {participant_id}...")
    
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Posterior Predictive Check - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    # 觀察數據統計
    obs_rt = participant_data[:, 0]
    obs_response = participant_data[:, 1]
    obs_rt_correct = obs_rt[obs_response == 1]
    obs_rt_incorrect = obs_rt[obs_response == 0]
    obs_accuracy = obs_response.mean()
    
    for i, (model_name, trace) in enumerate(models.items()):
        try:
            # 生成後驗預測樣本
            with pm.Model() as temp_model:
                # 這裡需要重建模型來生成預測
                # 為簡化，我們使用參數來模擬數據
                pred_samples = generate_predictions_from_trace(trace, n_predictions=1000)
            
            if pred_samples is not None:
                pred_rt = pred_samples[:, 0]
                pred_response = pred_samples[:, 1]
                pred_rt_correct = pred_rt[pred_response == 1]
                pred_rt_incorrect = pred_rt[pred_response == 0]
                pred_accuracy = pred_response.mean()
                
                # 上圖：RT 分布比較
                ax_rt = axes[0, i]
                
                # 繪製觀察數據
                ax_rt.hist(obs_rt_correct, bins=20, alpha=0.7, color='blue', 
                          label=f'Observed Correct (n={len(obs_rt_correct)})', density=True)
                ax_rt.hist(obs_rt_incorrect, bins=20, alpha=0.7, color='red', 
                          label=f'Observed Incorrect (n={len(obs_rt_incorrect)})', density=True)
                
                # 繪製預測數據
                ax_rt.hist(pred_rt_correct, bins=20, alpha=0.5, color='lightblue', 
                          label=f'Predicted Correct', density=True, histtype='step', linewidth=2)
                ax_rt.hist(pred_rt_incorrect, bins=20, alpha=0.5, color='lightcoral', 
                          label=f'Predicted Incorrect', density=True, histtype='step', linewidth=2)
                
                ax_rt.set_xlabel('Reaction Time (ms)')
                ax_rt.set_ylabel('Density')
                ax_rt.set_title(f'{model_name}\nRT Distributions')
                ax_rt.legend()
                ax_rt.grid(True, alpha=0.3)
                
                # 下圖：統計量比較
                ax_stats = axes[1, i]
                
                # 準備比較統計量
                stats_names = ['Accuracy', 'Mean RT\n(Correct)', 'Mean RT\n(Incorrect)', 
                              'RT Std\n(Correct)', 'RT Std\n(Incorrect)']
                
                obs_stats = [
                    obs_accuracy,
                    obs_rt_correct.mean() if len(obs_rt_correct) > 0 else np.nan,
                    obs_rt_incorrect.mean() if len(obs_rt_incorrect) > 0 else np.nan,
                    obs_rt_correct.std() if len(obs_rt_correct) > 0 else np.nan,
                    obs_rt_incorrect.std() if len(obs_rt_incorrect) > 0 else np.nan
                ]
                
                pred_stats = [
                    pred_accuracy,
                    pred_rt_correct.mean() if len(pred_rt_correct) > 0 else np.nan,
                    pred_rt_incorrect.mean() if len(pred_rt_incorrect) > 0 else np.nan,
                    pred_rt_correct.std() if len(pred_rt_correct) > 0 else np.nan,
                    pred_rt_incorrect.std() if len(pred_rt_incorrect) > 0 else np.nan
                ]
                
                x_pos = np.arange(len(stats_names))
                width = 0.35
                
                ax_stats.bar(x_pos - width/2, obs_stats, width, label='Observed', alpha=0.8, color='darkblue')
                ax_stats.bar(x_pos + width/2, pred_stats, width, label='Predicted', alpha=0.8, color='darkred')
                
                ax_stats.set_xlabel('Statistics')
                ax_stats.set_ylabel('Value')
                ax_stats.set_title(f'{model_name}\nSummary Statistics Comparison')
                ax_stats.set_xticks(x_pos)
                ax_stats.set_xticklabels(stats_names, rotation=45)
                ax_stats.legend()
                ax_stats.grid(True, alpha=0.3)
                
            else:
                # 如果預測失敗，顯示錯誤信息
                axes[0, i].text(0.5, 0.5, f'{model_name}\nPrediction Failed', 
                               ha='center', va='center', transform=axes[0, i].transAxes)
                axes[1, i].text(0.5, 0.5, 'See above', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                
        except Exception as e:
            print(f"    ❌ PPC failed for {model_name}: {e}")
            axes[0, i].text(0.5, 0.5, f'{model_name}\nError: {str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, i].transAxes)
    
    plt.tight_layout()
    
    # 保存圖表
    plot_file = os.path.join(save_dir, f'participant_{participant_id}_posterior_predictive_check.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Posterior predictive check saved: {plot_file}")

def generate_predictions_from_trace(trace, n_predictions=1000):
    """
    從 trace 生成預測數據
    """
    
    try:
        # 提取參數樣本
        v_correct_samples = trace.posterior['v_final_correct'].values.flatten()
        v_incorrect_samples = trace.posterior['v_final_incorrect'].values.flatten()
        b_samples = trace.posterior['b_safe'].values.flatten()
        A_samples = trace.posterior['start_var'].values.flatten()
        t0_samples = trace.posterior['non_decision'].values.flatten()
        
        n_posterior_samples = len(v_correct_samples)
        
        # 預測數據存儲
        predictions = np.empty((n_predictions, 2))
        
        for i in range(n_predictions):
            # 隨機選擇一組後驗參數
            idx = np.random.randint(0, n_posterior_samples)
            
            v_correct = v_correct_samples[idx]
            v_incorrect = v_incorrect_samples[idx]
            b = b_samples[idx]
            A = A_samples[idx]
            t0 = t0_samples[idx]
            
            # 簡化的 LBA 模擬
            # 起始點
            start_correct = np.random.uniform(0, A)
            start_incorrect = np.random.uniform(0, A)
            
            # 到達時間（使用 Wald 分布近似）
            if v_correct > 0 and v_incorrect > 0:
                mean_time_correct = (b - start_correct) / v_correct
                mean_time_incorrect = (b - start_incorrect) / v_incorrect
                
                # 添加隨機性
                time_correct = np.random.exponential(mean_time_correct)
                time_incorrect = np.random.exponential(mean_time_incorrect)
                
                # 選擇最快的
                if time_correct < time_incorrect:
                    predictions[i, 0] = (time_correct + t0) * 1000  # 轉換為毫秒
                    predictions[i, 1] = 1.0  # 正確反應
                else:
                    predictions[i, 0] = (time_incorrect + t0) * 1000
                    predictions[i, 1] = 0.0  # 錯誤反應
                
                # 確保合理的 RT 範圍
                predictions[i, 0] = np.clip(predictions[i, 0], 200, 5000)
            else:
                # 如果參數異常，使用默認值
                predictions[i, 0] = 1000
                predictions[i, 1] = 1.0
        
        return predictions
        
    except Exception as e:
        print(f"Error in prediction generation: {e}")
        return None

# ===================================================================
# 修改的詳細參與者分析函數
# ===================================================================

def save_enhanced_participant_analysis(models, participant_id, participant_data, comparison_results, save_dir):
    """
    保存增強的參與者分析結果，包含累積圖和後驗預測檢查
    """
    
    print(f"\nSaving enhanced analysis for participant {participant_id}...")
    
    # 1. 基本分析（原有功能）
    basic_results = save_detailed_participant_analysis(models, participant_id, participant_data, 
                                                      comparison_results, save_dir)
    
    # 2. 為每個模型創建信息累積圖
    for model_name, trace in models.items():
        success = create_accumulator_plot(trace, model_name, participant_id, save_dir)
        if success:
            print(f"    ✓ Accumulator plot created for {model_name}")
    
    # 3. 執行後驗預測檢查
    posterior_predictive_check(models, participant_data, participant_id, save_dir)
    
    # 4. 創建整合摘要圖
    create_integrated_summary_plot(models, participant_data, participant_id, save_dir)
    
    return basic_results

def create_integrated_summary_plot(models, participant_data, participant_id, save_dir):
    """
    創建整合摘要圖，包含多種分析視角
    """
    
    n_models = len(models)
    fig = plt.figure(figsize=(16, 12))
    
    # 創建子圖布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Integrated Analysis Summary - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    # 觀察數據統計
    obs_rt = participant_data[:, 0]
    obs_response = participant_data[:, 1]
    
    # 左上：觀察數據分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(obs_rt[obs_response == 1], bins=15, alpha=0.7, color='blue', label='Correct')
    ax1.hist(obs_rt[obs_response == 0], bins=15, alpha=0.7, color='red', label='Incorrect')
    ax1.set_xlabel('RT (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Observed RT Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上：精度-RT 關係
    ax2 = fig.add_subplot(gs[0, 1])
    accuracy = obs_response.mean()
    mean_rt = obs_rt.mean()
    ax2.scatter(mean_rt, accuracy, s=200, color='purple', alpha=0.8)
    ax2.set_xlabel('Mean RT (ms)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Speed-Accuracy Trade-off')
    ax2.grid(True, alpha=0.3)
    
    # 中間：參數比較（如果有兩個模型）
    if n_models == 2:
        model_names = list(models.keys())
        
        # 提取關鍵參數
        params_to_compare = ['v_final_correct', 'v_final_incorrect', 'b_safe', 'error_ratio']
        
        ax3 = fig.add_subplot(gs[0, 2:])
        
        x_pos = np.arange(len(params_to_compare))
        width = 0.35
        
        for i, param in enumerate(params_to_compare):
            values = []
            for model_name in model_names:
                if param in models[model_name].posterior:
                    values.append(models[model_name].posterior[param].values.flatten().mean())
                else:
                    values.append(0)
            
            ax3.bar(x_pos + i * width - width/2, values, width, 
                   label=model_names[i], alpha=0.8)
        
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Mean Value')
        ax3.set_title('Parameter Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(params_to_compare, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 下方：模型擬合質量
    for i, (model_name, trace) in enumerate(models.items()):
        ax = fig.add_subplot(gs[1, i*2:(i+1)*2])
        
        try:
            # 計算一些擬合統計量
            if hasattr(trace, 'log_likelihood') and 'likelihood' in trace.log_likelihood:
                ll = trace.log_likelihood['likelihood'].values
                ll_mean = ll.mean(axis=(0, 1))  # 對鏈和樣本求平均
                
                ax.plot(ll_mean, alpha=0.8, linewidth=2)
                ax.set_xlabel('Trial')
                ax.set_ylabel('Log-likelihood')
                ax.set_title(f'{model_name}\nTrial-wise Fit Quality')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{model_name}\nNo likelihood data', 
                       ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'{model_name}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # 最下方：診斷信息
    ax_diag = fig.add_subplot(gs[2, :])
    
    # 創建診斷摘要文本
    diag_text = []
    diag_text.append(f"PARTICIPANT {participant_id} DIAGNOSTIC SUMMARY")
    diag_text.append("=" * 50)
    diag_text.append(f"Trials: {len(participant_data)}")
    diag_text.append(f"Accuracy: {obs_response.mean():.3f}")
    diag_text.append(f"Mean RT: {obs_rt.mean():.1f} ms")
    diag_text.append(f"RT Range: {obs_rt.min():.1f} - {obs_rt.max():.1f} ms")
    diag_text.append("")
    
    for model_name, trace in models.items():
        diag_text.append(f"{model_name}:")
        try:
            summary = az.summary(trace)
            max_rhat = summary['r_hat'].max()
            min_ess = summary['ess_bulk'].min()
            diag_text.append(f"  Max R-hat: {max_rhat:.3f}")
            diag_text.append(f"  Min ESS: {min_ess:.0f}")
            
            # 參數估計
            if 'v_final_correct' in trace.posterior:
                v_correct_mean = trace.posterior['v_final_correct'].values.flatten().mean()
                diag_text.append(f"  v_correct: {v_correct_mean:.3f}")
            if 'v_final_incorrect' in trace.posterior:
                v_incorrect_mean = trace.posterior['v_final_incorrect'].values.flatten().mean()
                diag_text.append(f"  v_incorrect: {v_incorrect_mean:.3f}")
                
        except Exception as e:
            diag_text.append(f"  Error: {str(e)[:30]}...")
        diag_text.append("")
    
    ax_diag.text(0.05, 0.95, '\n'.join(diag_text), 
                transform=ax_diag.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax_diag.axis('off')
    
    plt.savefig(os.path.join(save_dir, f'participant_{participant_id}_integrated_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Integrated summary plot saved")

# ===================================================================
# 增強的綜合分析函數
# ===================================================================

def run_comprehensive_lba_analysis():
    """運行完整的綜合分析 - 包含所有新功能"""
    
    print("=== RUNNING COMPREHENSIVE LBA ANALYSIS ===")
    print("This analysis includes:")
    print("  - Improved model fitting with convergence checks")
    print("  - Detailed parameter analysis and sigma matrices")
    print("  - Information accumulation plots for each participant")
    print("  - Posterior predictive checks")
    print("  - Integrated summary visualizations")
    print("  - Complete result archival")
    
    # 創建結果目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"comprehensive_lba_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")
    
    # 載入數據
    try:
        data = np.load('model_data.npz', allow_pickle=True)
        observed_value = data['observed_value'].copy()
        observed_value[:, 0] *= 1000
        participant_idx = data['participant_idx']
        model_input_data = data['model_input_data'].item()
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None
    
    unique_participants = np.unique(participant_idx)
    print(f"Analyzing {len(unique_participants)} participants")
    
    all_results = []
    all_sigma_results = {}
    failed_participants = []
    
    for i, pid in enumerate(unique_participants):
        print(f"\n{'='*70}")
        print(f"PARTICIPANT {pid} ({i+1}/{len(unique_participants)})")
        print('='*70)
        
        # 提取參與者數據
        mask = participant_idx == pid
        participant_data = observed_value[mask]
        participant_input = {
            'left_match': model_input_data['left_match'][mask],
            'right_match': model_input_data['right_match'][mask]
        }
        
        n_trials = len(participant_data)
        print(f"Trials: {n_trials}")
        print(f"RT range: {participant_data[:, 0].min():.1f} - {participant_data[:, 0].max():.1f} ms")
        print(f"Accuracy: {participant_data[:, 1].mean():.3f}")
        
        if n_trials < 80:
            print(f"⚠️  Skipping participant {pid} - insufficient trials ({n_trials} < 80)")
            failed_participants.append({'participant': pid, 'reason': 'insufficient_trials', 'n_trials': n_trials})
            continue
        
        models = {}
        
        # 擬合模型
        model_funcs = {
            'Improved_Coactive': create_improved_coactive_model,
            'Improved_Parallel_AND': create_improved_parallel_and_model
        }
        
        for name, func in model_funcs.items():
            print(f"\nFitting {name}...")
            try:
                model = func(participant_data, participant_input)
                
                # 使用改進的採樣 - 適中的設定平衡質量和速度
                trace, diagnostics = sample_with_convergence_check(
                    model,
                    max_attempts=3,
                    draws=400,      # 足夠的樣本
                    tune=500,       # 適當的調整期
                    chains=4,
                    target_accept=0.90
                )
                
                if trace is not None:
                    models[name] = trace
                    print(f"✓ {name} completed successfully")
                    print(f"  Max R-hat: {diagnostics['max_rhat']:.4f}")
                    print(f"  Min ESS: {diagnostics['min_ess']:.0f}")
                    
                    # 參數健全性檢查
                    check_parameter_sanity(trace, name)
                else:
                    print(f"✗ {name} failed after all attempts")
                    
            except Exception as e:
                print(f"✗ {name} failed with error: {e}")
        
        # 模型比較
        comparison_results = None
        if len(models) == 2:
            comparison_results = improved_model_comparison(models)
            
            if comparison_results:
                print(f"\n📊 MODEL COMPARISON:")
                print(f"  Winner: {comparison_results['winner']}")
                print(f"  Method: {comparison_results['method']}")
                print(f"  ELPD difference: {comparison_results['elpd_diff']:.3f}")
                if comparison_results['dse'] > 0:
                    print(f"  Standard error: {comparison_results['dse']:.3f}")
                    print(f"  Effect size: {comparison_results['effect_size']:.3f}")
                print(f"  Significance: {comparison_results['significance']}")
        
        # 保存增強的綜合分析結果
        if len(models) > 0:
            detailed_results = save_enhanced_participant_analysis(
                models, pid, participant_data, comparison_results, results_dir
            )
            
            # 收集 sigma matrix 結果
            if 'sigma_results' in detailed_results:
                all_sigma_results[pid] = detailed_results['sigma_results']
            
            # 添加到總結果
            if comparison_results:
                participant_result = {
                    'participant': pid,
                    'n_trials': n_trials,
                    'winner': comparison_results['winner'],
                    'elpd_diff': comparison_results['elpd_diff'],
                    'dse': comparison_results['dse'],
                    'effect_size': comparison_results['effect_size'],
                    'significance': comparison_results['significance'],
                    'comparison_method': comparison_results['method'],
                    'obs_rt_mean': participant_data[:, 0].mean(),
                    'obs_rt_std': participant_data[:, 0].std(),
                    'obs_accuracy': participant_data[:, 1].mean(),
                    'convergence_success': len(models) == 2
                }
                
                # 添加 sigma matrix 統計
                if pid in all_sigma_results:
                    for model_name, sigma_data in all_sigma_results[pid].items():
                        participant_result[f'{model_name}_correlation'] = sigma_data['correlation']
                        participant_result[f'{model_name}_condition_number'] = sigma_data['condition_number']
                        participant_result[f'{model_name}_variance_A'] = sigma_data['variance_A']
                        participant_result[f'{model_name}_variance_B'] = sigma_data['variance_B']
                
                all_results.append(participant_result)
            else:
                failed_participants.append({'participant': pid, 'reason': 'comparison_failed', 'n_models': len(models)})
        else:
            failed_participants.append({'participant': pid, 'reason': 'no_successful_models', 'n_models': 0})
        
        print(f"✓ Participant {pid} analysis completed")
    
    # 保存和分析總結果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(results_dir, "comprehensive_analysis_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # 保存失敗記錄
        if failed_participants:
            failed_df = pd.DataFrame(failed_participants)
            failed_file = os.path.join(results_dir, "failed_participants.csv")
            failed_df.to_csv(failed_file, index=False)
            print(f"Failed participants logged: {failed_file}")
        
        # 保存 sigma matrix 總結
        if all_sigma_results:
            sigma_summary = create_sigma_matrix_summary(all_sigma_results, results_dir)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print('='*80)
        print(f"Results saved to: {results_file}")
        print(f"Total participants analyzed: {len(results_df)}")
        print(f"Failed participants: {len(failed_participants)}")
        
        # 詳細統計報告
        print_comprehensive_summary(results_df, all_sigma_results)
        
        # 創建總結圖表
        create_comprehensive_summary_plots(results_df, all_sigma_results, results_dir)
        
        # 創建最終報告
        create_final_analysis_report(results_df, all_sigma_results, failed_participants, results_dir)
        
        return results_df, all_sigma_results
    else:
        print("❌ No successful analyses completed")
        return None, None

def print_comprehensive_summary(results_df, all_sigma_results):
    """打印綜合統計摘要"""
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 50)
    
    # 基本統計
    print(f"Participants successfully analyzed: {len(results_df)}")
    print(f"Average trials per participant: {results_df['n_trials'].mean():.1f}")
    print(f"Average accuracy: {results_df['obs_accuracy'].mean():.3f}")
    print(f"Average RT: {results_df['obs_rt_mean'].mean():.1f} ms")
    
    # 模型偏好
    print(f"\n🏆 MODEL PREFERENCES:")
    model_counts = results_df['winner'].value_counts()
    for model, count in model_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  {model}: {count} ({percentage:.1f}%)")
    
    # 顯著性分析
    print(f"\n📈 SIGNIFICANCE ANALYSIS:")
    sig_counts = results_df['significance'].value_counts()
    for sig_level, count in sig_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  {sig_level}: {count} ({percentage:.1f}%)")
    
    # 方法使用統計
    print(f"\n🔬 COMPARISON METHODS:")
    method_counts = results_df['comparison_method'].value_counts()
    for method, count in method_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  {method}: {count} ({percentage:.1f}%)")
    
    # Sigma matrix 分析
    if all_sigma_results:
        print(f"\n🔢 SIGMA MATRIX ANALYSIS:")
        all_correlations = []
        all_conditions = []
        
        for pid, participant_sigma in all_sigma_results.items():
            for model_name, sigma_data in participant_sigma.items():
                all_correlations.append(sigma_data['correlation'])
                all_conditions.append(sigma_data['condition_number'])
        
        if all_correlations:
            print(f"  Average correlation: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")
            print(f"  Average condition number: {np.mean(all_conditions):.1f} ± {np.std(all_conditions):.1f}")
            
            # 矩陣質量分析
            well_conditioned = sum(1 for c in all_conditions if c < 10)
            moderately_conditioned = sum(1 for c in all_conditions if 10 <= c < 100)
            ill_conditioned = sum(1 for c in all_conditions if c >= 100)
            
            print(f"  Matrix conditions:")
            print(f"    Well-conditioned (< 10): {well_conditioned}")
            print(f"    Moderately conditioned (10-100): {moderately_conditioned}")
            print(f"    Ill-conditioned (≥ 100): {ill_conditioned}")

def create_comprehensive_summary_plots(results_df, all_sigma_results, save_dir):
    """創建綜合摘要圖表"""
    
    print("\nCreating comprehensive summary plots...")
    
    # 主摘要圖
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Comprehensive LBA Model Analysis Results', fontsize=20, fontweight='bold')
    
    # 1. 模型偏好餅圖
    winner_counts = results_df['winner'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    axes[0,0].pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%', 
                  colors=colors[:len(winner_counts)], startangle=90)
    axes[0,0].set_title('Model Preferences', fontsize=14, fontweight='bold')
    
    # 2. 效應量分布
    axes[0,1].hist(results_df['effect_size'], bins=15, alpha=0.7, edgecolor='black', color='lightblue')
    axes[0,1].axvline(2, color='red', linestyle='--', label='Strong evidence', linewidth=2)
    axes[0,1].axvline(1, color='orange', linestyle='--', label='Moderate evidence', linewidth=2)
    axes[0,1].set_xlabel('Effect Size')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Effect Size Distribution', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 顯著性分布
    sig_counts = results_df['significance'].value_counts()
    colors_sig = {'Significant': 'green', 'Weak': 'orange', 'Non-significant': 'red'}
    bar_colors = [colors_sig.get(x, 'gray') for x in sig_counts.index]
    axes[0,2].bar(sig_counts.index, sig_counts.values, color=bar_colors, alpha=0.8)
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Significance Distribution', fontsize=14, fontweight='bold')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. ELPD 差異分布
    axes[1,0].hist(results_df['elpd_diff'], bins=12, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[1,0].set_xlabel('ELPD Difference')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('ELPD Difference Distribution', fontsize=14, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. 精度分布
    axes[1,1].hist(results_df['obs_accuracy'], bins=15, alpha=0.7, edgecolor='black', color='gold')
    axes[1,1].set_xlabel('Accuracy')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. RT 分布
    axes[1,2].hist(results_df['obs_rt_mean'], bins=15, alpha=0.7, edgecolor='black', color='coral')
    axes[1,2].set_xlabel('Mean RT (ms)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Mean RT Distribution', fontsize=14, fontweight='bold')
    axes[1,2].grid(True, alpha=0.3)
    
    # 7-9. Sigma matrix 分析（如果有數據）
    if all_sigma_results:
        # 收集所有相關係數
        correlations_by_model = {}
        condition_numbers_by_model = {}
        
        for pid, participant_sigma in all_sigma_results.items():
            for model_name, sigma_data in participant_sigma.items():
                if model_name not in correlations_by_model:
                    correlations_by_model[model_name] = []
                    condition_numbers_by_model[model_name] = []
                correlations_by_model[model_name].append(sigma_data['correlation'])
                condition_numbers_by_model[model_name].append(sigma_data['condition_number'])
        
        # 7. 相關係數比較
        for model_name, correlations in correlations_by_model.items():
            axes[2,0].hist(correlations, alpha=0.6, label=model_name, bins=10)
        axes[2,0].set_xlabel('Correlation')
        axes[2,0].set_ylabel('Frequency')
        axes[2,0].set_title('Drift Rate Correlations', fontsize=14, fontweight='bold')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # 8. 條件數比較
        for model_name, conditions in condition_numbers_by_model.items():
            axes[2,1].hist(conditions, alpha=0.6, label=model_name, bins=10)
        axes[2,1].set_xlabel('Condition Number')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('Matrix Condition Numbers', fontsize=14, fontweight='bold')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        
        # 9. 方法使用統計
        method_counts = results_df['comparison_method'].value_counts()
        axes[2,2].bar(range(len(method_counts)), method_counts.values, 
                     color=['skyblue', 'lightcoral', 'lightgreen'][:len(method_counts)])
        axes[2,2].set_xticks(range(len(method_counts)))
        axes[2,2].set_xticklabels(method_counts.index, rotation=45)
        axes[2,2].set_ylabel('Count')
        axes[2,2].set_title('Comparison Methods Used', fontsize=14, fontweight='bold')
        axes[2,2].grid(True, alpha=0.3)
    else:
        # 如果沒有 sigma 數據，顯示其他統計
        axes[2,0].text(0.5, 0.5, 'No Sigma Matrix\nData Available', 
                      ha='center', va='center', transform=axes[2,0].transAxes, fontsize=12)
        axes[2,1].text(0.5, 0.5, 'No Sigma Matrix\nData Available', 
                      ha='center', va='center', transform=axes[2,1].transAxes, fontsize=12)
        axes[2,2].text(0.5, 0.5, 'No Sigma Matrix\nData Available', 
                      ha='center', va='center', transform=axes[2,2].transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_analysis_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comprehensive summary plot saved")

def create_final_analysis_report(results_df, all_sigma_results, failed_participants, save_dir):
    """創建最終分析報告"""
    
    report_file = os.path.join(save_dir, "FINAL_ANALYSIS_REPORT.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE LBA MODEL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 執行摘要
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total participants processed: {len(results_df) + len(failed_participants)}\n")
        f.write(f"Successful analyses: {len(results_df)}\n")
        f.write(f"Failed analyses: {len(failed_participants)}\n")
        f.write(f"Success rate: {len(results_df)/(len(results_df) + len(failed_participants))*100:.1f}%\n\n")
        
        # 模型比較結果
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("-" * 25 + "\n")
        model_counts = results_df['winner'].value_counts()
        for model, count in model_counts.items():
            percentage = count / len(results_df) * 100
            f.write(f"{model}: {count} participants ({percentage:.1f}%)\n")
        
        # 統計顯著性
        f.write(f"\nSTATISTICAL SIGNIFICANCE\n")
        f.write("-" * 25 + "\n")
        sig_counts = results_df['significance'].value_counts()
        for sig_level, count in sig_counts.items():
            percentage = count / len(results_df) * 100
            f.write(f"{sig_level}: {count} participants ({percentage:.1f}%)\n")
        
        # 方法學統計
        f.write(f"\nMETHODOLOGICAL STATISTICS\n")
        f.write("-" * 25 + "\n")
        method_counts = results_df['comparison_method'].value_counts()
        for method, count in method_counts.items():
            percentage = count / len(results_df) * 100
            f.write(f"{method}: {count} participants ({percentage:.1f}%)\n")
        
        # 數據質量統計
        f.write(f"\nDATA QUALITY STATISTICS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Average trials per participant: {results_df['n_trials'].mean():.1f}\n")
        f.write(f"Trial range: {results_df['n_trials'].min()} - {results_df['n_trials'].max()}\n")
        f.write(f"Average accuracy: {results_df['obs_accuracy'].mean():.3f} ± {results_df['obs_accuracy'].std():.3f}\n")
        f.write(f"Average RT: {results_df['obs_rt_mean'].mean():.1f} ± {results_df['obs_rt_mean'].std():.1f} ms\n")
        
        # Sigma matrix 分析
        if all_sigma_results:
            f.write(f"\nSIGMA MATRIX ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            all_correlations = []
            all_conditions = []
            
            for pid, participant_sigma in all_sigma_results.items():
                for model_name, sigma_data in participant_sigma.items():
                    all_correlations.append(sigma_data['correlation'])
                    all_conditions.append(sigma_data['condition_number'])
            
            f.write(f"Average correlation: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}\n")
            f.write(f"Correlation range: {np.min(all_correlations):.3f} - {np.max(all_correlations):.3f}\n")
            f.write(f"Average condition number: {np.mean(all_conditions):.1f} ± {np.std(all_conditions):.1f}\n")
            
            # 矩陣質量分類
            well_conditioned = sum(1 for c in all_conditions if c < 10)
            moderately_conditioned = sum(1 for c in all_conditions if 10 <= c < 100)
            ill_conditioned = sum(1 for c in all_conditions if c >= 100)
            
            f.write(f"\nMatrix condition classification:\n")
            f.write(f"  Well-conditioned (< 10): {well_conditioned}\n")
            f.write(f"  Moderately conditioned (10-100): {moderately_conditioned}\n")
            f.write(f"  Ill-conditioned (≥ 100): {ill_conditioned}\n")
        
        # 失敗分析
        if failed_participants:
            f.write(f"\nFAILURE ANALYSIS\n")
            f.write("-" * 15 + "\n")
            failure_reasons = {}
            for fp in failed_participants:
                reason = fp['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in failure_reasons.items():
                f.write(f"{reason}: {count} participants\n")
        
        # 文件清單
        f.write(f"\nGENERATED FILES\n")
        f.write("-" * 15 + "\n")
        f.write("This analysis generated the following files:\n")
        f.write("- comprehensive_analysis_results.csv (main results)\n")
        f.write("- sigma_matrix_summary.csv (sigma matrix data)\n")
        f.write("- comprehensive_analysis_summary.png (summary plots)\n")
        f.write("- sigma_matrix_comparison.png (sigma matrix plots)\n")
        f.write("- Individual participant files:\n")
        f.write("  * participant_XX_comprehensive.npz (detailed data)\n")
        f.write("  * participant_XX_*_accumulation.png (accumulator plots)\n")
        f.write("  * participant_XX_posterior_predictive_check.png (PPC plots)\n")
        f.write("  * participant_XX_integrated_summary.png (summary plots)\n")
        f.write("  * participant_XX_report.txt (individual reports)\n")
        
        f.write(f"\nANALYSIS COMPLETE\n")
        f.write("=" * 20 + "\n")
        f.write("All results have been saved and are ready for further analysis.\n")
    
    print(f"✓ Final analysis report saved: {report_file}")

# ===================================================================
# 主程序修改
# ===================================================================

# 在現有的函數基礎上，這裡是修改後的主程序部分

def main_enhanced_analysis():
    """增強版主分析程序"""
    
    print("ENHANCED LBA MODEL ANALYSIS")
    print("=" * 50)
    print("Features included:")
    print("  ✓ Improved model fitting with convergence diagnostics")
    print("  ✓ Sigma matrix computation and visualization")
    print("  ✓ Information accumulation plots (4 traces per participant)")
    print("  ✓ Posterior predictive checks")
    print("  ✓ Integrated summary visualizations")
    print("  ✓ Comprehensive result archival")
    print("  ✓ Detailed failure analysis and reporting")
    
    # 檢查數據可用性
    if not os.path.exists('model_data.npz'):
        print("\n❌ Required data file 'model_data.npz' not found!")
        print("Please ensure you have the correct data file in the working directory.")
        return None
    
    # 確認運行
    print(f"\n{'='*50}")
    print("ANALYSIS OPTIONS:")
    print("1. Run comprehensive analysis (all participants, all features)")
    print("2. Run test analysis (single participant, quick test)")
    print("3. Load and examine existing results")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "3":
        # 載入現有結果
        available_results = list_available_results()
        if available_results:
            if len(available_results) == 1:
                selected_dir = available_results[0]
            else:
                print(f"\nSelect results directory:")
                for i, dir_name in enumerate(available_results, 1):
                    print(f"{i}. {dir_name}")
                
                try:
                    dir_choice = int(input(f"Enter number (1-{len(available_results)}): ")) - 1
                    selected_dir = available_results[dir_choice]
                except:
                    selected_dir = available_results[0]
                    print(f"Invalid choice, using: {selected_dir}")
            
            print(f"\nLoading results from: {selected_dir}")
            loaded_results = load_and_analyze_saved_results(selected_dir)
        else:
            print("No existing results found. Running new analysis...")
            choice = "1"
    
    if choice == "2":
        # 測試分析
        print("\n🧪 Running test analysis...")
        if test_improved_models():
            print("✓ Test completed successfully!")
            
            print("\nRunning detailed single participant analysis...")
            models = run_improved_analysis()
            
            if models:
                print("✓ Single participant analysis completed!")
                print("Check generated plots and diagnostics.")
        else:
            print("❌ Test failed - check your setup")
    
    elif choice == "1" or choice not in ["2", "3"]:
        # 完整分析
        print("\n🚀 Starting comprehensive analysis...")
        print("This may take considerable time depending on the number of participants.")
        
        # 最終確認
        confirm = input("\nProceed with full analysis? (y/n): ")
        if confirm.lower() == 'y':
            try:
                results_df, all_sigma_results = run_comprehensive_lba_analysis()
                
                if results_df is not None:
                    print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
                    print(f"📁 All results saved to the output directory")
                    print(f"📊 Generated comprehensive visualizations")
                    print(f"📋 Created detailed reports")
                    
                    # 顯示關鍵統計
                    print(f"\n📈 KEY FINDINGS:")
                    model_counts = results_df['winner'].value_counts()
                    for model, count in model_counts.items():
                        percentage = count / len(results_df) * 100
                        print(f"  {model}: {count} participants ({percentage:.1f}%)")
                    
                    print(f"\n💡 Next steps:")
                    print(f"  - Review individual participant plots")
                    print(f"  - Examine sigma matrix analyses")
                    print(f"  - Check posterior predictive validations")
                    print(f"  - Read the final analysis report")
                    
                else:
                    print(f"\n❌ Analysis failed - check error messages above")
                    
            except Exception as e:
                print(f"\n❌ Analysis failed with error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Analysis cancelled.")
    
    print(f"\n{'='*50}")
    print("ANALYSIS SESSION COMPLETE")
    print('='*50)

# ===================================================================
# 額外的實用函數
# ===================================================================

def create_participant_gallery(results_dir, max_participants=9):
    """創建參與者結果畫廊 - 展示多個參與者的關鍵圖表"""
    
    print(f"Creating participant gallery...")
    
    # 尋找所有參與者的整合摘要圖
    summary_files = []
    for file in os.listdir(results_dir):
        if file.startswith('participant_') and file.endswith('_integrated_summary.png'):
            summary_files.append(file)
    
    if not summary_files:
        print("No participant summary files found")
        return
    
    # 限制顯示數量
    summary_files = sorted(summary_files)[:max_participants]
    n_files = len(summary_files)
    
    # 計算網格大小
    cols = 3
    rows = (n_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Participant Analysis Gallery', fontsize=20, fontweight='bold')
    
    for i, file in enumerate(summary_files):
        row = i // cols
        col = i % cols
        
        # 提取參與者ID
        participant_id = file.split('_')[1]
        
        try:
            # 載入和顯示圖片
            img_path = os.path.join(results_dir, file)
            img = plt.imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Participant {participant_id}', fontsize=14)
            axes[row, col].axis('off')
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error loading\nParticipant {participant_id}', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # 隱藏多餘的子圖
    for i in range(n_files, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'participant_gallery.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Participant gallery saved")

def create_model_comparison_matrix(results_df, save_dir):
    """創建模型比較矩陣圖"""
    
    if len(results_df) == 0:
        return
    
    # 準備數據
    participants = results_df['participant'].values
    winners = results_df['winner'].values
    effect_sizes = results_df['effect_size'].values
    significance = results_df['significance'].values
    
    # 創建比較矩陣
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison Matrix Analysis', fontsize=16, fontweight='bold')
    
    # 1. 勝者模式圖
    unique_models = np.unique(winners)
    model_colors = {model: plt.cm.Set3(i) for i, model in enumerate(unique_models)}
    colors = [model_colors[winner] for winner in winners]
    
    axes[0,0].scatter(participants, effect_sizes, c=colors, alpha=0.7, s=60)
    axes[0,0].set_xlabel('Participant ID')
    axes[0,0].set_ylabel('Effect Size')
    axes[0,0].set_title('Model Winners by Effect Size')
    axes[0,0].grid(True, alpha=0.3)
    
    # 添加圖例
    for model, color in model_colors.items():
        axes[0,0].scatter([], [], c=[color], label=model, s=60)
    axes[0,0].legend()
    
    # 2. 顯著性熱圖
    sig_mapping = {'Significant': 2, 'Weak': 1, 'Non-significant': 0}
    sig_values = [sig_mapping[s] for s in significance]
    
    # 重塑為矩陣形式（如果參與者足夠多）
    n_participants = len(participants)
    if n_participants >= 16:
        # 創建大致正方形的排列
        side = int(np.sqrt(n_participants))
        matrix_data = np.array(sig_values[:side*side]).reshape(side, side)
        
        im = axes[0,1].imshow(matrix_data, cmap='RdYlGn', aspect='auto')
        axes[0,1].set_title('Significance Pattern Matrix')
        cbar = plt.colorbar(im, ax=axes[0,1])
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Non-sig', 'Weak', 'Significant'])
    else:
        # 簡單的條形圖
        axes[0,1].bar(range(len(sig_values)), sig_values, 
                     color=['red' if s==0 else 'orange' if s==1 else 'green' for s in sig_values])
        axes[0,1].set_xlabel('Participant Index')
        axes[0,1].set_ylabel('Significance Level')
        axes[0,1].set_title('Significance by Participant')
        axes[0,1].set_yticks([0, 1, 2])
        axes[0,1].set_yticklabels(['Non-sig', 'Weak', 'Significant'])
    
    # 3. 效應量分布 by 模型
    for model in unique_models:
        model_data = results_df[results_df['winner'] == model]
        axes[1,0].hist(model_data['effect_size'], alpha=0.6, label=model, bins=10)
    
    axes[1,0].axvline(1, color='orange', linestyle='--', alpha=0.8, label='Weak threshold')
    axes[1,0].axvline(2, color='red', linestyle='--', alpha=0.8, label='Strong threshold')
    axes[1,0].set_xlabel('Effect Size')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Effect Size Distribution by Model')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 相關性分析（如果有相關列）
    if 'obs_rt_mean' in results_df.columns and 'obs_accuracy' in results_df.columns:
        scatter = axes[1,1].scatter(results_df['obs_rt_mean'], results_df['obs_accuracy'], 
                                   c=effect_sizes, cmap='viridis', alpha=0.7, s=60)
        axes[1,1].set_xlabel('Mean RT (ms)')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('RT-Accuracy Relationship\n(colored by effect size)')
        cbar = plt.colorbar(scatter, ax=axes[1,1])
        cbar.set_label('Effect Size')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'RT-Accuracy data\nnot available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Model comparison matrix saved")

def generate_analysis_index(results_dir):
    """生成分析結果索引HTML文件"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LBA Model Analysis Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .file-list {{ background-color: #ffffff; padding: 15px; border: 1px solid #dee2e6; }}
            .participant-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
            .participant-card {{ border: 1px solid #dee2e6; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>LBA Model Analysis Results</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p>This directory contains comprehensive LBA model analysis results including:</p>
            <ul>
                <li>Individual participant analyses with accumulator plots</li>
                <li>Posterior predictive checks</li>
                <li>Sigma matrix computations</li>
                <li>Model comparison results</li>
                <li>Comprehensive summary visualizations</li>
            </ul>
        </div>
        
        <div class="file-list">
            <h2>Key Files</h2>
            <ul>
                <li><strong>comprehensive_analysis_results.csv</strong> - Main results table</li>
                <li><strong>FINAL_ANALYSIS_REPORT.txt</strong> - Detailed text report</li>
                <li><strong>comprehensive_analysis_summary.png</strong> - Summary plots</li>
                <li><strong>sigma_matrix_comparison.png</strong> - Sigma matrix analysis</li>
                <li><strong>participant_gallery.png</strong> - Participant overview</li>
            </ul>
        </div>
        
        <h2>Individual Participant Results</h2>
        <div class="participant-grid">
    """
    
    # 尋找參與者文件
    participant_files = {}
    for file in os.listdir(results_dir):
        if file.startswith('participant_') and '_' in file:
            try:
                parts = file.split('_')
                if len(parts) >= 2:
                    pid = parts[1]
                    if pid not in participant_files:
                        participant_files[pid] = []
                    participant_files[pid].append(file)
            except:
                continue
    
    # 為每個參與者添加卡片
    for pid in sorted(participant_files.keys()):
        files = participant_files[pid]
        html_content += f"""
            <div class="participant-card">
                <h3>Participant {pid}</h3>
                <ul>
        """
        
        for file in sorted(files):
            if file.endswith('.png'):
                html_content += f'<li><a href="{file}">{file}</a></li>\n'
            elif file.endswith('.txt'):
                html_content += f'<li><a href="{file}">{file}</a> (Report)</li>\n'
            elif file.endswith('.npz'):
                html_content += f'<li>{file} (Data)</li>\n'
        
        html_content += """
                </ul>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    html_file = os.path.join(results_dir, 'index.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Analysis index saved: {html_file}")

# ===================================================================
# 修改的主要函數以包含新功能
# ===================================================================

def save_detailed_participant_analysis(models, participant_id, participant_data, comparison_results, save_dir):
    """原有的詳細參與者分析函數（保持向後兼容）"""
    
    print(f"\nSaving detailed analysis for participant {participant_id}...")
    
    # 1. 基本參與者資訊
    participant_info = {
        'participant_id': participant_id,
        'n_trials': len(participant_data),
        'rt_mean': participant_data[:, 0].mean(),
        'rt_std': participant_data[:, 0].std(),
        'rt_min': participant_data[:, 0].min(),
        'rt_max': participant_data[:, 0].max(),
        'accuracy': participant_data[:, 1].mean(),
        'comparison_winner': comparison_results['winner'] if comparison_results else 'Unknown',
        'comparison_method': comparison_results['method'] if comparison_results else 'Unknown'
    }
    
    # 2. 提取每個模型的後驗樣本
    posterior_samples = {}
    parameter_summaries = {}
    
    for model_name, trace in models.items():
        print(f"  Processing {model_name} posterior samples...")
        
        # 提取所有參數
        samples = {}
        summary = {}
        
        param_names = ['v_match_pos', 'v_mismatch_pos', 'v_final_correct', 'v_final_incorrect',
                       'b_safe', 'start_var', 'non_decision', 'error_ratio']
        
        # Parallel AND 特有參數
        if 'integration_strength' in trace.posterior:
            param_names.append('integration_strength')
        
        for param in param_names:
            if param in trace.posterior:
                param_samples = trace.posterior[param].values.flatten()
                samples[param] = param_samples
                
                summary[param] = {
                    'mean': param_samples.mean(),
                    'std': param_samples.std(),
                    'median': np.median(param_samples),
                    'q025': np.percentile(param_samples, 2.5),
                    'q975': np.percentile(param_samples, 97.5),
                    'min': param_samples.min(),
                    'max': param_samples.max()
                }
        
        posterior_samples[model_name] = samples
        parameter_summaries[model_name] = summary
    
    # 3. 計算 sigma matrices
    sigma_results = calculate_sigma_matrices_from_traces(models, participant_id, save_dir)
    
    # 4. 創建診斷圖表
    create_participant_diagnostic_plots(models, participant_id, save_dir)
    
    # 5. 保存綜合結果文件
    comprehensive_file = os.path.join(save_dir, f"participant_{participant_id}_comprehensive.npz")
    
    save_data = {
        'participant_info': participant_info,
        'participant_data': participant_data,
        'comparison_results': comparison_results,
        'parameter_summaries': parameter_summaries,
        'sigma_results': sigma_results
    }
    
    # 添加後驗樣本
    for model_name, samples in posterior_samples.items():
        for param_name, param_samples in samples.items():
            save_data[f"{model_name}_{param_name}_samples"] = param_samples
    
    np.savez(comprehensive_file, **save_data)
    print(f"✓ Comprehensive analysis saved: {comprehensive_file}")
    
    # 6. 創建文字總結報告
    create_participant_summary_report(participant_info, parameter_summaries, sigma_results, 
                                     comparison_results, save_dir)
    
    return {
        'participant_info': participant_info,
        'parameter_summaries': parameter_summaries,
        'sigma_results': sigma_results,
        'posterior_samples': posterior_samples
    }

# 最後，確保所有必需的導入和函數都已定義
def check_parameter_sanity(trace, model_name):
    """檢查參數的合理性"""
    
    print(f"\n{model_name} Parameter Summary:")
    
    try:
        # 基本參數
        if 'v_match_pos' in trace.posterior:
            v_match = trace.posterior['v_match_pos'].values.flatten()
            print(f"  v_match: {v_match.mean():.3f} ± {v_match.std():.3f}")
            
        if 'v_mismatch_pos' in trace.posterior:
            v_mismatch = trace.posterior['v_mismatch_pos'].values.flatten()
            print(f"  v_mismatch: {v_mismatch.mean():.3f} ± {v_mismatch.std():.3f}")
            
        if 'v_final_correct' in trace.posterior:
            v_correct = trace.posterior['v_final_correct'].values.flatten()
            print(f"  v_correct: {v_correct.mean():.3f} ± {v_correct.std():.3f}")
            
        if 'v_final_incorrect' in trace.posterior:
            v_incorrect = trace.posterior['v_final_incorrect'].values.flatten()
            print(f"  v_incorrect: {v_incorrect.mean():.3f} ± {v_incorrect.std():.3f}")
            
        if 'b_safe' in trace.posterior:
            boundary = trace.posterior['b_safe'].values.flatten()
            print(f"  boundary: {boundary.mean():.3f} ± {boundary.std():.3f}")
            
        if 'error_ratio' in trace.posterior:
            error_ratio = trace.posterior['error_ratio'].values.flatten()
            print(f"  error_ratio: {error_ratio.mean():.3f} ± {error_ratio.std():.3f}")
            
        # 檢查參數合理性
        warnings = []
        
        if 'v_final_correct' in trace.posterior and 'v_final_incorrect' in trace.posterior:
            v_correct = trace.posterior['v_final_correct'].values.flatten()
            v_incorrect = trace.posterior['v_final_incorrect'].values.flatten()
            
            if (v_correct <= v_incorrect).mean() > 0.1:
                warnings.append("v_incorrect sometimes > v_correct")
                
        if warnings:
            print(f"  ⚠️  Warnings: {', '.join(warnings)}")
        else:
            print(f"  ✓ Parameters look reasonable")
            
    except Exception as e:
        print(f"  ❌ Error checking parameters: {e}")

# 運行主程序
if __name__ == '__main__':
    main_enhanced_analysis()
