# -*- coding: utf-8 -*-
"""
compare_predictions_analysis.py - 比較PyMC預測與實際資料
Compare PyMC ParallelAND predictions with actual data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_prediction_accuracy():
    """分析PyMC預測準確度"""
    
    print("📊 分析PyMC ParallelAND預測結果...")
    
    # 讀取預測結果
    predictions_df = pd.read_csv("pymc_parallel_and_predictions_p31.csv")
    fit_metrics_df = pd.read_csv("pymc_parallel_and_fit_metrics_p31.csv")
    posterior_df = pd.read_csv("pymc_parallel_and_posterior_means_p31.csv")
    
    print(f"   總試驗數: {len(predictions_df)}")
    print(f"   RT相關係數: {fit_metrics_df['rt_correlation'].iloc[0]:.4f}")
    print(f"   RT RMSE: {fit_metrics_df['rt_rmse'].iloc[0]:.4f}")
    print(f"   平均實際RT: {fit_metrics_df['mean_actual_rt'].iloc[0]:.4f}")
    print(f"   平均預測RT: {fit_metrics_df['mean_predicted_rt'].iloc[0]:.4f}")
    
    # 詳細分析
    actual_rts = predictions_df['actual_rt'].values
    predicted_rts = predictions_df['predicted_rt'].values
    
    # 統計檢驗
    t_stat, t_pval = stats.ttest_rel(actual_rts, predicted_rts)
    correlation, corr_pval = stats.pearsonr(actual_rts, predicted_rts)
    
    print(f"\n📈 統計分析:")
    print(f"   配對t檢驗: t={t_stat:.4f}, p={t_pval:.4f}")
    print(f"   皮爾森相關: r={correlation:.4f}, p={corr_pval:.4f}")
    
    # 分析效果drift rates
    print(f"\n🧠 估計的drift rates:")
    for param in ['left_v_vertical', 'left_v_nonvertical', 'right_v_vertical', 'right_v_nonvertical']:
        value = posterior_df[param].iloc[0]
        print(f"   {param}: {value:.4f}")
    
    # 分析ParallelAND效果
    effective_drifts = predictions_df['effective_drift'].values
    left_drifts = predictions_df['left_drift'].values  
    right_drifts = predictions_df['right_drift'].values
    
    print(f"\n⚡ ParallelAND分析:")
    print(f"   平均有效drift rate: {np.mean(effective_drifts):.4f}")
    print(f"   平均左側drift rate: {np.mean(left_drifts):.4f}")
    print(f"   平均右側drift rate: {np.mean(right_drifts):.4f}")
    
    # 檢查最小值規則
    min_matches = np.sum(effective_drifts == np.minimum(left_drifts, right_drifts))
    print(f"   正確應用最小值規則: {min_matches}/{len(effective_drifts)} ({min_matches/len(effective_drifts)*100:.1f}%)")
    
    return {
        'predictions_df': predictions_df,
        'fit_metrics': fit_metrics_df.iloc[0].to_dict(),
        'posterior_means': posterior_df.iloc[0].to_dict(),
        'statistical_tests': {
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'correlation': correlation,
            'correlation_pvalue': corr_pval
        }
    }

def plot_detailed_comparison():
    """詳細比較圖表"""
    
    predictions_df = pd.read_csv("pymc_parallel_and_predictions_p31.csv")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    actual_rts = predictions_df['actual_rt']
    predicted_rts = predictions_df['predicted_rt']
    
    # RT scatter plot
    axes[0, 0].scatter(actual_rts, predicted_rts, alpha=0.6, s=20)
    axes[0, 0].plot([actual_rts.min(), actual_rts.max()], 
                   [actual_rts.min(), actual_rts.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('實際RT')
    axes[0, 0].set_ylabel('預測RT')
    axes[0, 0].set_title('RT預測準確度')
    
    # RT residuals
    residuals = predicted_rts - actual_rts
    axes[0, 1].scatter(actual_rts, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('實際RT')
    axes[0, 1].set_ylabel('殘差 (預測-實際)')
    axes[0, 1].set_title('預測殘差')
    
    # RT histograms
    axes[0, 2].hist(actual_rts, bins=30, alpha=0.7, label='實際RT', density=True)
    axes[0, 2].hist(predicted_rts, bins=30, alpha=0.7, label='預測RT', density=True)
    axes[0, 2].set_xlabel('RT')
    axes[0, 2].set_ylabel('密度')
    axes[0, 2].set_title('RT分布比較')
    axes[0, 2].legend()
    
    # Drift rate analysis
    left_drifts = predictions_df['left_drift']
    right_drifts = predictions_df['right_drift']
    effective_drifts = predictions_df['effective_drift']
    
    axes[1, 0].scatter(left_drifts, right_drifts, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('左側drift rate')
    axes[1, 0].set_ylabel('右側drift rate')
    axes[1, 0].set_title('左右drift rate關係')
    
    # Effective drift distribution
    axes[1, 1].hist(effective_drifts, bins=20, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('有效drift rate (最小值)')
    axes[1, 1].set_ylabel('頻率')
    axes[1, 1].set_title('有效drift rate分布')
    
    # Response pattern
    response_counts = predictions_df['response'].value_counts().sort_index()
    axes[1, 2].bar(response_counts.index, response_counts.values, alpha=0.7)
    axes[1, 2].set_xlabel('反應類型')
    axes[1, 2].set_ylabel('頻率')
    axes[1, 2].set_title('反應模式分布')
    
    plt.tight_layout()
    plt.savefig('pymc_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 詳細比較圖已保存: pymc_detailed_comparison.png")
    
    return fig

def summarize_model_performance():
    """總結模型性能"""
    
    predictions_df = pd.read_csv("pymc_parallel_and_predictions_p31.csv")
    fit_metrics_df = pd.read_csv("pymc_parallel_and_fit_metrics_p31.csv")
    
    print(f"\n📋 PyMC ParallelAND模型總結:")
    print(f"="*50)
    
    # 基本統計
    actual_rts = predictions_df['actual_rt']
    predicted_rts = predictions_df['predicted_rt']
    
    print(f"資料量: {len(predictions_df)} 個trial")
    print(f"RT相關係數: {fit_metrics_df['rt_correlation'].iloc[0]:.4f}")
    print(f"RT RMSE: {fit_metrics_df['rt_rmse'].iloc[0]:.4f} 秒")
    print(f"平均絕對誤差: {np.mean(np.abs(predicted_rts - actual_rts)):.4f} 秒")
    
    # RT範圍比較
    print(f"\nRT範圍比較:")
    print(f"實際RT: {actual_rts.min():.3f} - {actual_rts.max():.3f} 秒")
    print(f"預測RT: {predicted_rts.min():.3f} - {predicted_rts.max():.3f} 秒")
    
    # ParallelAND規則驗證
    left_drifts = predictions_df['left_drift']
    right_drifts = predictions_df['right_drift']
    effective_drifts = predictions_df['effective_drift']
    
    min_rule_correct = np.sum(effective_drifts == np.minimum(left_drifts, right_drifts))
    print(f"\nParallelAND規則執行:")
    print(f"正確應用最小值規則: {min_rule_correct}/{len(predictions_df)} ({min_rule_correct/len(predictions_df)*100:.1f}%)")
    
    # 按反應類型分析
    print(f"\n按反應類型分析:")
    for response in sorted(predictions_df['response'].unique()):
        subset = predictions_df[predictions_df['response'] == response]
        if len(subset) > 0:
            corr = np.corrcoef(subset['actual_rt'], subset['predicted_rt'])[0,1]
            rmse = np.sqrt(np.mean((subset['actual_rt'] - subset['predicted_rt'])**2))
            print(f"  反應 {response}: n={len(subset)}, r={corr:.3f}, RMSE={rmse:.3f}")
    
    return {
        'summary_stats': {
            'n_trials': len(predictions_df),
            'rt_correlation': fit_metrics_df['rt_correlation'].iloc[0],
            'rt_rmse': fit_metrics_df['rt_rmse'].iloc[0],
            'mean_absolute_error': np.mean(np.abs(predicted_rts - actual_rts)),
            'parallel_and_rule_accuracy': min_rule_correct/len(predictions_df)
        }
    }

if __name__ == "__main__":
    # 執行完整分析
    print("🚀 開始PyMC預測結果分析...")
    
    # 基本分析
    results = analyze_prediction_accuracy()
    
    # 詳細圖表
    plot_detailed_comparison()
    
    # 性能總結
    summary = summarize_model_performance()
    
    print(f"\n✅ 分析完成！")
    print(f"主要發現: RT相關性 = {results['statistical_tests']['correlation']:.4f}")
    print(f"ParallelAND規則準確性 = {summary['summary_stats']['parallel_and_rule_accuracy']:.2f}")