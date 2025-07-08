# -*- coding: utf-8 -*-
"""
parallel_and_analysis.py - ParallelAND關鍵指標分析
Analyze winning drift rates and left-right correlations in ParallelAND model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

def analyze_winning_drift_rates():
    """分析最終勝利的drift rate類型"""
    
    print("🏆 分析最終勝利的drift rate...")
    
    # 讀取預測資料
    predictions_df = pd.read_csv("pymc_parallel_and_predictions_p31.csv")
    
    # 為每個trial確定勝利的drift rate
    winning_rates = []
    
    for _, row in predictions_df.iterrows():
        left_drift = row['left_drift']
        right_drift = row['right_drift']
        effective_drift = row['effective_drift']
        response = row['response']
        left_stim = row['left_stimulus']
        right_stim = row['right_stimulus']
        
        # 確定勝利的是左邊還是右邊
        if effective_drift == left_drift:
            winning_side = 'left'
            winning_drift_value = left_drift
            # 確定具體的drift rate類型
            if response in [1, 2]:  # Left vertical response
                if left_stim == 1:  # Vertical stimulus
                    winning_drift_name = 'left_v_vertical'
                else:  # Nonvertical stimulus
                    winning_drift_name = 'left_v_vertical_error'
            else:  # Left nonvertical response
                if left_stim == 0:  # Nonvertical stimulus
                    winning_drift_name = 'left_v_nonvertical'
                else:  # Vertical stimulus
                    winning_drift_name = 'left_v_nonvertical_error'
        else:
            winning_side = 'right'
            winning_drift_value = right_drift
            # 確定具體的drift rate類型
            if response in [0, 1]:  # Right vertical response
                if right_stim == 1:  # Vertical stimulus
                    winning_drift_name = 'right_v_vertical'
                else:  # Nonvertical stimulus
                    winning_drift_name = 'right_v_vertical_error'
            else:  # Right nonvertical response
                if right_stim == 0:  # Nonvertical stimulus
                    winning_drift_name = 'right_v_nonvertical'
                else:  # Vertical stimulus
                    winning_drift_name = 'right_v_nonvertical_error'
        
        winning_rates.append({
            'trial': row['trial'],
            'response': response,
            'left_stimulus': left_stim,
            'right_stimulus': right_stim,
            'winning_side': winning_side,
            'winning_drift_name': winning_drift_name,
            'winning_drift_value': winning_drift_value,
            'left_drift': left_drift,
            'right_drift': right_drift,
            'effective_drift': effective_drift
        })
    
    winning_df = pd.DataFrame(winning_rates)
    
    # 統計勝利的drift rate類型
    winning_counts = Counter(winning_df['winning_drift_name'])
    side_counts = Counter(winning_df['winning_side'])
    
    print(f"\n📊 勝利drift rate統計 (總計 {len(winning_df)} trials):")
    print("="*50)
    
    print(f"勝利側別統計:")
    for side, count in side_counts.items():
        percentage = count / len(winning_df) * 100
        print(f"  {side}: {count} trials ({percentage:.1f}%)")
    
    print(f"\n勝利drift rate類型統計:")
    for drift_name, count in winning_counts.most_common():
        percentage = count / len(winning_df) * 100
        print(f"  {drift_name}: {count} trials ({percentage:.1f}%)")
    
    # 儲存結果
    winning_df.to_csv("winning_drift_rates_analysis.csv", index=False)
    print(f"\n📁 勝利drift rate分析已儲存: winning_drift_rates_analysis.csv")
    
    return winning_df, winning_counts, side_counts

def analyze_left_right_correlations():
    """分析左右LBA drift rates的相關係數"""
    
    print("\n🔗 分析左右LBA drift rates相關係數...")
    
    predictions_df = pd.read_csv("pymc_parallel_and_predictions_p31.csv")
    
    left_drifts = predictions_df['left_drift'].values
    right_drifts = predictions_df['right_drift'].values
    
    # 計算相關係數
    correlation, p_value = stats.pearsonr(left_drifts, right_drifts)
    spearman_corr, spearman_p = stats.spearmanr(left_drifts, right_drifts)
    
    print(f"左右drift rates相關分析:")
    print(f"  Pearson相關係數: r = {correlation:.4f}, p = {p_value:.4f}")
    print(f"  Spearman相關係數: ρ = {spearman_corr:.4f}, p = {spearman_p:.4f}")
    
    # 分別分析不同stimulus condition下的相關性
    print(f"\n按stimulus condition分析相關性:")
    
    conditions = []
    for stim_combo in [(0,0), (0,1), (1,0), (1,1)]:
        left_stim, right_stim = stim_combo
        subset = predictions_df[
            (predictions_df['left_stimulus'] == left_stim) & 
            (predictions_df['right_stimulus'] == right_stim)
        ]
        
        if len(subset) > 10:  # 只分析有足夠資料的condition
            subset_corr, subset_p = stats.pearsonr(subset['left_drift'], subset['right_drift'])
            condition_name = f"L{left_stim}R{right_stim}"
            print(f"  {condition_name} (n={len(subset)}): r = {subset_corr:.4f}, p = {subset_p:.4f}")
            
            conditions.append({
                'condition': condition_name,
                'left_stim': left_stim,
                'right_stim': right_stim,
                'n_trials': len(subset),
                'correlation': subset_corr,
                'p_value': subset_p
            })
    
    # 分析不同response type的相關性
    print(f"\n按response type分析相關性:")
    
    for response in sorted(predictions_df['response'].unique()):
        subset = predictions_df[predictions_df['response'] == response]
        if len(subset) > 10:
            subset_corr, subset_p = stats.pearsonr(subset['left_drift'], subset['right_drift'])
            print(f"  Response {response} (n={len(subset)}): r = {subset_corr:.4f}, p = {subset_p:.4f}")
    
    # 儲存相關性分析結果
    correlation_results = {
        'overall_pearson_r': correlation,
        'overall_pearson_p': p_value,
        'overall_spearman_r': spearman_corr,
        'overall_spearman_p': spearman_p,
        'conditions': conditions
    }
    
    # 將結果儲存為DataFrame
    summary_data = {
        'analysis_type': ['overall_pearson', 'overall_spearman'],
        'correlation': [correlation, spearman_corr],
        'p_value': [p_value, spearman_p],
        'n_trials': [len(predictions_df), len(predictions_df)]
    }
    
    correlation_summary_df = pd.DataFrame(summary_data)
    correlation_summary_df.to_csv("left_right_correlations_analysis.csv", index=False)
    print(f"\n📁 左右相關性分析已儲存: left_right_correlations_analysis.csv")
    
    return correlation_results

def plot_drift_rate_analysis():
    """繪製drift rate分析圖表"""
    
    print("\n📊 繪製drift rate分析圖表...")
    
    predictions_df = pd.read_csv("pymc_parallel_and_predictions_p31.csv")
    winning_df = pd.read_csv("winning_drift_rates_analysis.csv")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 勝利drift rate類型分布
    winning_counts = winning_df['winning_drift_name'].value_counts()
    axes[0, 0].bar(range(len(winning_counts)), winning_counts.values, alpha=0.7)
    axes[0, 0].set_xticks(range(len(winning_counts)))
    axes[0, 0].set_xticklabels(winning_counts.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Winning Drift Rate Types')
    
    # 2. 左右側勝利比例
    side_counts = winning_df['winning_side'].value_counts()
    axes[0, 1].pie(side_counts.values, labels=side_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Left vs Right Side Wins')
    
    # 3. 左右drift rates散點圖
    left_drifts = predictions_df['left_drift']
    right_drifts = predictions_df['right_drift']
    correlation = np.corrcoef(left_drifts, right_drifts)[0, 1]
    
    axes[0, 2].scatter(left_drifts, right_drifts, alpha=0.6, s=20)
    axes[0, 2].plot([left_drifts.min(), left_drifts.max()], 
                   [left_drifts.min(), left_drifts.max()], 'r--', alpha=0.8)
    axes[0, 2].set_xlabel('Left Drift Rate')
    axes[0, 2].set_ylabel('Right Drift Rate')
    axes[0, 2].set_title(f'Left-Right Drift Correlation (r={correlation:.3f})')
    
    # 4. 有效drift rate vs 左右drift rates
    effective_drifts = predictions_df['effective_drift']
    axes[1, 0].scatter(left_drifts, effective_drifts, alpha=0.6, s=20, label='Left vs Effective')
    axes[1, 0].scatter(right_drifts, effective_drifts, alpha=0.6, s=20, label='Right vs Effective')
    axes[1, 0].plot([0, max(left_drifts.max(), right_drifts.max())], 
                   [0, max(left_drifts.max(), right_drifts.max())], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Left/Right Drift Rate')
    axes[1, 0].set_ylabel('Effective Drift Rate')
    axes[1, 0].set_title('Effective vs Individual Drift Rates')
    axes[1, 0].legend()
    
    # 5. 按response type的drift rate分布
    responses = sorted(predictions_df['response'].unique())
    for i, response in enumerate(responses):
        subset = predictions_df[predictions_df['response'] == response]
        axes[1, 1].hist(subset['effective_drift'], bins=15, alpha=0.7, 
                       label=f'Response {response}', density=True)
    axes[1, 1].set_xlabel('Effective Drift Rate')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Effective Drift Rate by Response Type')
    axes[1, 1].legend()
    
    # 6. 勝利side隨trial的變化
    trial_indices = winning_df['trial'].values
    winning_sides_numeric = [1 if side == 'right' else 0 for side in winning_df['winning_side']]
    
    # 計算滑動平均
    window_size = 50
    if len(trial_indices) >= window_size:
        moving_avg = np.convolve(winning_sides_numeric, np.ones(window_size)/window_size, mode='valid')
        axes[1, 2].plot(trial_indices[window_size-1:], moving_avg, 'b-', linewidth=2)
        axes[1, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.8)
        axes[1, 2].set_xlabel('Trial Number')
        axes[1, 2].set_ylabel('Proportion Right Wins')
        axes[1, 2].set_title(f'Right Side Win Rate (Moving Avg, window={window_size})')
        axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('drift_rate_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Drift rate分析圖已儲存: drift_rate_analysis.png")
    
    return fig

def comprehensive_parallel_and_analysis():
    """完整的ParallelAND分析"""
    
    print("🚀 開始完整的ParallelAND分析...")
    print("="*60)
    
    # 1. 分析勝利的drift rates
    winning_df, winning_counts, side_counts = analyze_winning_drift_rates()
    
    # 2. 分析左右相關性
    correlation_results = analyze_left_right_correlations()
    
    # 3. 繪製分析圖表
    plot_drift_rate_analysis()
    
    # 4. 總結報告
    print(f"\n📋 ParallelAND分析總結:")
    print("="*60)
    
    total_trials = len(winning_df)
    
    print(f"總trial數: {total_trials}")
    print(f"\n最常勝利的drift rate:")
    for drift_name, count in list(winning_counts.most_common(3)):
        percentage = count / total_trials * 100
        print(f"  {drift_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n左右側勝利比例:")
    for side, count in side_counts.items():
        percentage = count / total_trials * 100
        print(f"  {side}: {count} ({percentage:.1f}%)")
    
    print(f"\n左右drift rates相關性:")
    print(f"  整體相關係數: r = {correlation_results['overall_pearson_r']:.4f}")
    print(f"  顯著性: p = {correlation_results['overall_pearson_p']:.4f}")
    
    # 判斷相關性強度
    corr_strength = abs(correlation_results['overall_pearson_r'])
    if corr_strength > 0.7:
        strength_desc = "強"
    elif corr_strength > 0.3:
        strength_desc = "中等"
    else:
        strength_desc = "弱"
    
    print(f"  相關性強度: {strength_desc}")
    
    print(f"\n✅ 分析完成！所有結果已儲存。")
    
    return {
        'winning_analysis': (winning_df, winning_counts, side_counts),
        'correlation_analysis': correlation_results
    }

if __name__ == "__main__":
    comprehensive_parallel_and_analysis()