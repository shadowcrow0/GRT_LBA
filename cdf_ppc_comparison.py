# -*- coding: utf-8 -*-
"""
cdf_ppc_comparison.py - 專注於CDF比較的PPC圖表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from parallel_and_ppc import ParallelANDPPC

def create_cdf_comparison_plot(participant_id: int = 40):
    """創建CDF比較圖表"""
    
    print(f"📊 創建參與者 {participant_id} 的CDF比較圖...")
    
    # 初始化PPC
    ppc = ParallelANDPPC()
    
    # 載入資料和生成模擬
    if not ppc.load_fitted_results(participant_id):
        print("❌ 載入資料失敗")
        return None
    
    ppc.generate_posterior_samples(n_samples=500)
    ppc.simulate_data_from_posterior(n_sim_datasets=50)
    
    observed_rts = ppc.observed_data['rts']
    simulated_datasets = ppc.simulated_data
    
    # 創建CDF比較圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 整體RT CDF比較
    x_vals = np.linspace(0, max(observed_rts), 1000)
    
    # Observed CDF
    observed_cdf = [np.mean(observed_rts <= x) for x in x_vals]
    axes[0, 0].plot(x_vals, observed_cdf, 'r-', linewidth=3, label='Observed', alpha=0.8)
    
    # Simulated CDFs
    for i, sim_data in enumerate(simulated_datasets[:10]):  # 只畫前10個避免太亂
        sim_rts = sim_data['rts']
        sim_cdf = [np.mean(sim_rts <= x) for x in x_vals]
        axes[0, 0].plot(x_vals, sim_cdf, 'b-', alpha=0.2, linewidth=1, 
                       label='Simulated' if i == 0 else "")
    
    axes[0, 0].set_xlabel('RT')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title('Overall RT CDF Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 按response type分組的CDF: 右上R1, 左下R0&2, 右下R3
    responses = ppc.observed_data['responses']
    
    # Response 1: axes[0, 1] (右上)
    ax = axes[0, 1]
    obs_mask = responses == 1
    if np.sum(obs_mask) >= 5:
        obs_rt_subset = observed_rts[obs_mask]
        x_subset = np.linspace(0, max(obs_rt_subset) * 1.2, 500)
        obs_cdf_subset = [np.mean(obs_rt_subset <= x) for x in x_subset]
        ax.plot(x_subset, obs_cdf_subset, 'r-', linewidth=3, 
               label='Observed R1', alpha=0.8)
        
        # Simulated data for response 1
        for i, sim_data in enumerate(simulated_datasets[:8]):
            sim_responses = sim_data['responses']
            sim_rts = sim_data['rts']
            sim_mask = sim_responses == 1
            
            if np.sum(sim_mask) > 3:
                sim_rt_subset = sim_rts[sim_mask]
                sim_cdf_subset = [np.mean(sim_rt_subset <= x) for x in x_subset]
                ax.plot(x_subset, sim_cdf_subset, 'b-', alpha=0.2, linewidth=1,
                       label='Simulated R1' if i == 0 else "")
    
    ax.set_xlabel('RT')
    ax.set_ylabel('Probability')
    ax.set_title('Response 1 CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Response 0 & 2: axes[1, 0] (左下)
    ax = axes[1, 0]
    
    for response_type in [0, 2]:
        obs_mask = responses == response_type
        if np.sum(obs_mask) < 5:
            continue
            
        obs_rt_subset = observed_rts[obs_mask]
        
        if len(obs_rt_subset) > 0:
            x_subset = np.linspace(0, max(obs_rt_subset) * 1.2, 500)
            obs_cdf_subset = [np.mean(obs_rt_subset <= x) for x in x_subset]
            
            color = 'red' if response_type == 0 else 'darkred'
            ax.plot(x_subset, obs_cdf_subset, color=color, linewidth=3, 
                   label=f'Observed R{response_type}', alpha=0.8)
            
            # Simulated data
            sim_color = 'blue' if response_type == 0 else 'darkblue'
            for i, sim_data in enumerate(simulated_datasets[:5]):
                sim_responses = sim_data['responses']
                sim_rts = sim_data['rts']
                sim_mask = sim_responses == response_type
                
                if np.sum(sim_mask) > 3:
                    sim_rt_subset = sim_rts[sim_mask]
                    sim_cdf_subset = [np.mean(sim_rt_subset <= x) for x in x_subset]
                    ax.plot(x_subset, sim_cdf_subset, color=sim_color, alpha=0.2, linewidth=1,
                           label=f'Simulated R{response_type}' if i == 0 else "")
    
    ax.set_xlabel('RT')
    ax.set_ylabel('Probability')
    ax.set_title('Response 0 & 2 CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Response 3: axes[1, 1] (右下)
    ax = axes[1, 1]
    obs_mask = responses == 3
    if np.sum(obs_mask) >= 5:
        obs_rt_subset = observed_rts[obs_mask]
        x_subset = np.linspace(0, max(obs_rt_subset) * 1.2, 500)
        obs_cdf_subset = [np.mean(obs_rt_subset <= x) for x in x_subset]
        ax.plot(x_subset, obs_cdf_subset, 'r-', linewidth=3, 
               label='Observed R3', alpha=0.8)
        
        # Simulated data for response 3
        for i, sim_data in enumerate(simulated_datasets[:8]):
            sim_responses = sim_data['responses']
            sim_rts = sim_data['rts']
            sim_mask = sim_responses == 3
            
            if np.sum(sim_mask) > 3:
                sim_rt_subset = sim_rts[sim_mask]
                sim_cdf_subset = [np.mean(sim_rt_subset <= x) for x in x_subset]
                ax.plot(x_subset, sim_cdf_subset, 'b-', alpha=0.2, linewidth=1,
                       label='Simulated R3' if i == 0 else "")
    
    ax.set_xlabel('RT')
    ax.set_ylabel('Probability')
    ax.set_title('Response 3 CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 儲存圖表
    save_path = f"cdf_ppc_comparison_p{participant_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 CDF比較圖已儲存: {save_path}")
    
    return fig

def main():
    """主要執行函數"""
    
    print("📈 ParallelAND模型CDF PPC比較")
    print("="*40)
    
    # 檢查是否有fitted結果
    try:
        summary_df = pd.read_csv("high_accuracy_parallel_and_summary.csv")
        available_participants = summary_df['participant_id'].tolist()
        print(f"📊 發現 {len(available_participants)} 位已fitted參與者")
        
        # 選擇BIC最佳的參與者
        best_participant = summary_df.loc[summary_df['bic'].idxmin(), 'participant_id']
        print(f"🏆 選擇BIC最佳參與者: {best_participant}")
        
        # 創建CDF比較圖
        fig = create_cdf_comparison_plot(best_participant)
        
        if fig:
            print("✅ CDF比較圖創建完成!")
        else:
            print("❌ CDF比較圖創建失敗")
            
    except FileNotFoundError:
        print("❌ 找不到 high_accuracy_parallel_and_summary.csv")
        print("   請先運行 fit_all_participants_parallel_and.py")
        
        # 使用預設參與者
        print("   使用預設參與者 40 進行分析...")
        fig = create_cdf_comparison_plot(40)

if __name__ == "__main__":
    main()