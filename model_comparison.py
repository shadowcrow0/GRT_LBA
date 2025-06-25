# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 00:35:34 2025

@author: spt904
"""

# -*- coding: utf-8 -*-
"""
檔案3: 模型比較與診斷
需要先運行檔案1和檔案2，載入兩個模型結果進行比較
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import warnings

def diagnose_models(idata_coactive, idata_parallel):
    """診斷兩個模型的差異和質量"""
    print("=" * 50)
    print("模型診斷報告")
    print("=" * 50)
    
    # 1. 檢查收斂性 - 修正版本
    print("1. 收斂性檢查：")
    print("-" * 30)
    
    # Coactive 模型
    try:
        summary_coactive = az.summary(idata_coactive)
        max_rhat_coactive = summary_coactive['r_hat'].max()
        bad_rhat_coactive = (summary_coactive['r_hat'] > 1.01).sum()
        
        print(f"Coactive 模型：")
        print(f"  最大 r-hat: {max_rhat_coactive:.4f}")
        print(f"  r-hat > 1.01 的參數數量: {bad_rhat_coactive}")
        
        if max_rhat_coactive < 1.01:
            print("  ✅ 收斂良好")
        elif max_rhat_coactive < 1.05:
            print("  ⚠️  收斂可接受")
        else:
            print("  ❌ 收斂有問題")
            
    except Exception as e:
        print(f"  Coactive 模型收斂性檢查失敗: {e}")
    
    # Parallel 模型
    try:
        summary_parallel = az.summary(idata_parallel)
        max_rhat_parallel = summary_parallel['r_hat'].max()
        bad_rhat_parallel = (summary_parallel['r_hat'] > 1.01).sum()
        
        print(f"Parallel 模型：")
        print(f"  最大 r-hat: {max_rhat_parallel:.4f}")
        print(f"  r-hat > 1.01 的參數數量: {bad_rhat_parallel}")
        
        if max_rhat_parallel < 1.01:
            print("  ✅ 收斂良好")
        elif max_rhat_parallel < 1.05:
            print("  ⚠️  收斂可接受")
        else:
            print("  ❌ 收斂有問題")
            
    except Exception as e:
        print(f"  Parallel 模型收斂性檢查失敗: {e}")
    
    # 2. 檢查 v_final_correct 的差異
    print("\n2. v_final_correct 差異分析：")
    print("-" * 30)
    
    try:
        v_coactive = idata_coactive.posterior['v_final_correct'].values.flatten()
        v_parallel = idata_parallel.posterior['v_final_correct'].values.flatten()
        
        min_len = min(len(v_coactive), len(v_parallel))
        v_coactive_sub = v_coactive[:min_len]
        v_parallel_sub = v_parallel[:min_len]
        
        correlation = np.corrcoef(v_coactive_sub, v_parallel_sub)[0,1]
        
        print(f"Coactive v_final_correct 範圍: [{v_coactive.min():.3f}, {v_coactive.max():.3f}]")
        print(f"Parallel v_final_correct 範圍: [{v_parallel.min():.3f}, {v_parallel.max():.3f}]")
        print(f"兩模型 v_final_correct 的相關係數: {correlation:.4f}")
        
        # 計算差異統計
        diff_mean = np.mean(v_parallel_sub - v_coactive_sub)
        diff_std = np.std(v_parallel_sub - v_coactive_sub)
        print(f"平均差異 (Parallel - Coactive): {diff_mean:.4f} ± {diff_std:.4f}")
        
        if abs(correlation) > 0.99:
            print("⚠️  警告：兩模型的 v_final_correct 幾乎相同！")
        elif abs(correlation) > 0.95:
            print("⚠️  注意：兩模型的 v_final_correct 高度相關")
        else:
            print("✅ 兩模型有明顯差異")
            
    except Exception as e:
        print(f"v_final_correct 差異分析失敗: {e}")
    
    # 3. k 參數診斷（如果存在）
    print("\n3. k 參數診斷：")
    print("-" * 30)
    
    try:
        if 'k_smoothness' in idata_parallel.posterior:
            k_values = idata_parallel.posterior['k_smoothness'].values.flatten()
            k_summary = summary_parallel.loc['k_smoothness']
            
            print(f"k 參數統計：")
            print(f"  平均值: {k_values.mean():.3f}")
            print(f"  標準差: {k_values.std():.3f}")
            print(f"  範圍: [{k_values.min():.3f}, {k_values.max():.3f}]")
            print(f"  中位數: {np.median(k_values):.3f}")
            print(f"  95% HDI: [{k_summary['hdi_2.5%']:.3f}, {k_summary['hdi_97.5%']:.3f}]")
            
            # 檢查 k 是否合理
            if k_values.mean() > 10:
                print("⚠️  警告：k 值過大，可能導致模型退化為 max(v_l, v_r)")
            elif k_values.mean() < 0.5:
                print("⚠️  警告：k 值過小，可能導致模型退化為平均值")
            else:
                print("✅ k 值在合理範圍內")
                
            # 檢查 k 的變異
            if k_values.std() > k_values.mean():
                print("⚠️  警告：k 值變異很大，估計不穩定")
            else:
                print("✅ k 值估計穩定")
        else:
            print("未找到 k_smoothness 參數")
            
    except Exception as e:
        print(f"k 參數診斷失敗: {e}")

def robust_model_comparison(idata_coactive, idata_parallel, observed_value):
    """使用多種方法進行模型比較"""
    print("\n" + "=" * 50)
    print("穩健的模型比較")
    print("=" * 50)
    
    model_dict = {
        "Coactive": idata_coactive,
        "Parallel": idata_parallel
    }
    
    # 1. LOO 比較
    print("1. LOO 比較：")
    print("-" * 30)
    
    loo_reliable = True
    try:
        # 計算 LOO
        loo_coactive = az.loo(idata_coactive)
        loo_parallel = az.loo(idata_parallel)
        
        # 檢查 Pareto k 值
        pareto_k_coactive = loo_coactive.pareto_k.values
        pareto_k_parallel = loo_parallel.pareto_k.values
        
        bad_k_coactive = int((pareto_k_coactive > 0.7).sum())
        bad_k_parallel = int((pareto_k_parallel > 0.7).sum())
        total_obs = len(pareto_k_coactive)
        
        print(f"觀測總數: {total_obs}")
        print(f"Coactive 模型 Pareto k > 0.7 的觀測數: {bad_k_coactive} ({bad_k_coactive/total_obs*100:.1f}%)")
        print(f"Parallel 模型 Pareto k > 0.7 的觀測數: {bad_k_parallel} ({bad_k_parallel/total_obs*100:.1f}%)")
        
        # 如果過多觀測有問題，標記為不可靠
        if bad_k_coactive > total_obs * 0.1 or bad_k_parallel > total_obs * 0.1:
            print("⚠️  警告：超過 10% 的觀測有高 Pareto k 值，LOO 結果可能不可靠")
            loo_reliable = False
        else:
            print("✅ LOO 結果可靠")
        
        # 顯示 LOO 比較結果
        loo_compare = az.compare(model_dict, ic="loo")
        print("\nLOO 比較結果：")
        print(loo_compare)
        
    except Exception as e:
        print(f"LOO 計算失敗: {e}")
        loo_compare = None
        loo_reliable = False
    
    # 2. WAIC 比較
    print("\n2. WAIC 比較：")
    print("-" * 30)
    
    try:
        waic_compare = az.compare(model_dict, ic="waic")
        print("WAIC 比較結果：")
        print(waic_compare)
    except Exception as e:
        print(f"WAIC 計算失敗: {e}")
        waic_compare = None
    
    # 3. 後驗預測檢查
    print("\n3. 後驗預測檢查：")
    print("-" * 30)
    
    try:
        # 提取預測和觀測數據
        pred_coactive = idata_coactive.posterior_predictive['likelihood'].values
        pred_parallel = idata_parallel.posterior_predictive['likelihood'].values
        
        observed_responses = observed_value[:, 1]  # 實際回應
        observed_rt = observed_value[:, 0]  # 實際反應時間
        
        # 計算預測準確率
        pred_resp_coactive = pred_coactive[:, :, :, 1]  # [chains, draws, obs, response]
        pred_resp_parallel = pred_parallel[:, :, :, 1]
        
        # 平均跨鏈和抽樣
        mean_pred_resp_coactive = np.mean(pred_resp_coactive, axis=(0,1))
        mean_pred_resp_parallel = np.mean(pred_resp_parallel, axis=(0,1))
        
        # 計算準確率
        obs_accuracy = np.mean(observed_responses)
        pred_accuracy_coactive = np.mean(mean_pred_resp_coactive)
        pred_accuracy_parallel = np.mean(mean_pred_resp_parallel)
        
        print(f"觀測準確率: {obs_accuracy:.3f}")
        print(f"Coactive 預測準確率: {pred_accuracy_coactive:.3f}")
        print(f"Parallel 預測準確率: {pred_accuracy_parallel:.3f}")
        
        # 計算準確率預測誤差
        acc_error_coactive = abs(pred_accuracy_coactive - obs_accuracy)
        acc_error_parallel = abs(pred_accuracy_parallel - obs_accuracy)
        
        print(f"Coactive 準確率預測誤差: {acc_error_coactive:.3f}")
        print(f"Parallel 準確率預測誤差: {acc_error_parallel:.3f}")
        
        # 反應時間預測檢查
        pred_rt_coactive = pred_coactive[:, :, :, 0]
        pred_rt_parallel = pred_parallel[:, :, :, 0]
        
        mean_pred_rt_coactive = np.mean(pred_rt_coactive, axis=(0,1))
        mean_pred_rt_parallel = np.mean(pred_rt_parallel, axis=(0,1))
        
        obs_mean_rt = np.mean(observed_rt)
        pred_mean_rt_coactive = np.mean(mean_pred_rt_coactive)
        pred_mean_rt_parallel = np.mean(mean_pred_rt_parallel)
        
        print(f"\n反應時間比較：")
        print(f"觀測平均 RT: {obs_mean_rt:.3f}")
        print(f"Coactive 預測平均 RT: {pred_mean_rt_coactive:.3f}")
        print(f"Parallel 預測平均 RT: {pred_mean_rt_parallel:.3f}")
        
        rt_error_coactive = abs(pred_mean_rt_coactive - obs_mean_rt)
        rt_error_parallel = abs(pred_mean_rt_parallel - obs_mean_rt)
        
        print(f"Coactive RT 預測誤差: {rt_error_coactive:.3f}")
        print(f"Parallel RT 預測誤差: {rt_error_parallel:.3f}")
        
        # 總體預測表現
        print(f"\n總體預測表現：")
        total_error_coactive = acc_error_coactive + rt_error_coactive
        total_error_parallel = acc_error_parallel + rt_error_parallel
        print(f"Coactive 總預測誤差: {total_error_coactive:.3f}")
        print(f"Parallel 總預測誤差: {total_error_parallel:.3f}")
        
        if total_error_coactive < total_error_parallel:
            print("✅ Coactive 模型預測表現較好")
        elif total_error_parallel < total_error_coactive:
            print("✅ Parallel 模型預測表現較好")
        else:
            print("⚖️  兩模型預測表現相當")
        
    except Exception as e:
        print(f"後驗預測檢查失敗: {e}")
    
    # 4. 總結
    print("\n4. 模型比較總結：")
    print("-" * 30)
    
    results_summary = []
    
    # LOO 結果
    if loo_compare is not None and loo_reliable:
        loo_winner = loo_compare.index[0]
        loo_diff = loo_compare.loc[loo_compare.index[1], 'elpd_diff']
        loo_se = loo_compare.loc[loo_compare.index[1], 'dse']
        results_summary.append(f"LOO: {loo_winner} 勝出 (差異: {loo_diff:.1f} ± {loo_se:.1f})")
    elif loo_compare is not None:
        results_summary.append("LOO: 結果不可靠")
    
    # WAIC 結果
    if waic_compare is not None:
        waic_winner = waic_compare.index[0]
        waic_diff = waic_compare.loc[waic_compare.index[1], 'elpd_diff']
        waic_se = waic_compare.loc[waic_compare.index[1], 'dse']
        results_summary.append(f"WAIC: {waic_winner} 勝出 (差異: {waic_diff:.1f} ± {waic_se:.1f})")
    
    # 預測表現結果
    if 'total_error_coactive' in locals() and 'total_error_parallel' in locals():
        if total_error_coactive < total_error_parallel:
            results_summary.append("預測表現: Coactive 較好")
        elif total_error_parallel < total_error_coactive:
            results_summary.append("預測表現: Parallel 較好")
        else:
            results_summary.append("預測表現: 相當")
    
    print("綜合結果：")
    for result in results_summary:
        print(f"  • {result}")
    
    # 判斷一致性
    if len(results_summary) >= 2:
        winners = []
        for result in results_summary:
            if "Coactive" in result and "勝出" in result or "較好" in result:
                winners.append("Coactive")
            elif "Parallel" in result and "勝出" in result or "較好" in result:
                winners.append("Parallel")
        
        if len(set(winners)) == 1:
            print(f"\n🎯 一致結論: {winners[0]} 模型表現較好")
        elif len(winners) == 0:
            print(f"\n⚖️  結論: 兩模型表現相當")
        else:
            print(f"\n🤔 結論: 不同評估方法結果不一致，需要進一步分析")
    
    return loo_compare, waic_compare

def create_comparison_plots(idata_coactive, idata_parallel, loo_compare=None, waic_compare=None):
    """創建比較圖表"""
    
    try:
        fig = plt.figure(figsize=(15, 10))
        
        # 子圖1: LOO 比較 (如果可用)
        if loo_compare is not None:
            ax1 = plt.subplot(2, 3, 1)
            az.plot_compare(loo_compare, ax=ax1)
            ax1.set_title("LOO-CV Model Comparison")
        
        # 子圖2: WAIC 比較 (如果可用)
        if waic_compare is not None:
            ax2 = plt.subplot(2, 3, 2)
            az.plot_compare(waic_compare, ax=ax2)
            ax2.set_title("WAIC Model Comparison")
        
        # 子圖3: v_final_correct 比較
        ax3 = plt.subplot(2, 3, 3)
        v_coactive = idata_coactive.posterior['v_final_correct'].values.flatten()
        v_parallel = idata_parallel.posterior['v_final_correct'].values.flatten()
        
        # 取樣本進行散點圖 (避免太多點)
        n_sample = min(1000, len(v_coactive), len(v_parallel))
        idx = np.random.choice(len(v_coactive), n_sample, replace=False)
        
        ax3.scatter(v_coactive[idx], v_parallel[idx], alpha=0.5, s=1)
        ax3.plot([v_coactive.min(), v_coactive.max()], 
                [v_coactive.min(), v_coactive.max()], 'r--', alpha=0.8)
        ax3.set_xlabel('Coactive v_final_correct')
        ax3.set_ylabel('Parallel v_final_correct')
        ax3.set_title('v_final_correct Comparison')
        
        # 子圖4: k 參數分布 (如果存在)
        if 'k_smoothness' in idata_parallel.posterior:
            ax4 = plt.subplot(2, 3, 4)
            k_values = idata_parallel.posterior['k_smoothness'].values.flatten()
            ax4.hist(k_values, bins=50, alpha=0.7, density=True)
            ax4.axvline(k_values.mean(), color='red', linestyle='--', 
                       label=f'Mean: {k_values.mean():.2f}')
            ax4.set_xlabel('k_smoothness')
            ax4.set_ylabel('Density')
            ax4.set_title('k Parameter Distribution')
            ax4.legend()
        
        # 子圖5 & 6: 參數比較 (選擇幾個關鍵參數)
        key_params = ['mu_v_lm', 'mu_v_rm']
        for i, param in enumerate(key_params):
            ax = plt.subplot(2, 3, 5+i)
            if param in idata_coactive.posterior and param in idata_parallel.posterior:
                coactive_param = idata_coactive.posterior[param].values.flatten()
                parallel_param = idata_parallel.posterior[param].values.flatten()
                
                ax.hist(coactive_param, bins=30, alpha=0.6, label='Coactive', density=True)
                ax.hist(parallel_param, bins=30, alpha=0.6, label='Parallel', density=True)
                ax.set_xlabel(param)
                ax.set_ylabel('Density')
                ax.set_title(f'{param} Comparison')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"創建比較圖失敗: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("檔案3: 模型比較與診斷")
    print("=" * 60)
    
    # --- 載入模型結果 ---
    try:
        print("載入 Coactive 模型結果...")
        idata_coactive = az.from_netcdf('coactive_model_results.nc')
        print("✅ Coactive 模型載入成功")
        
        print("載入 Parallel 模型結果...")
        idata_parallel = az.from_netcdf('parallel_model_results.nc')
        print("✅ Parallel 模型載入成功")
        
        print("載入觀測數據...")
        data = np.load('model_data.npz', allow_pickle=True)
        observed_value = data['observed_value']
        print("✅ 觀測數據載入成功")
        
    except FileNotFoundError as e:
        print(f"❌ 找不到必要檔案: {e}")
        print("請確保已運行檔案1和檔案2")
        exit()
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        exit()
    
    # --- 執行診斷 ---
    diagnose_models(idata_coactive, idata_parallel)
    
    # --- 執行比較 ---
    loo_result, waic_result = robust_model_comparison(idata_coactive, idata_parallel, observed_value)
    
    # --- 創建視覺化 ---
    print(f"\n--- 創建比較圖表 ---")
    create_comparison_plots(idata_coactive, idata_parallel, loo_result, waic_result)
    
    print("\n" + "=" * 60)
    print("模型比較分析完成！")
    print("=" * 60)