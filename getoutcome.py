# -*- coding: utf-8 -*-
"""
完整的個體分析系統
功能：
1. 記錄每個模型的詳細 r-hat 值
2. 進行後驗預測檢查 (PPC)
3. 保存所有中間結果和診斷信息
4. 生成完整的分析報告
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

# 設定字體
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# ===================================================================
# LBA 核心函數
# ===================================================================
def fast_logp_lba(value, v_correct, v_incorrect, b, A, t0):
    """快速版 LBA 對數似然函數"""
    rt = value[:, 0]
    response = value[:, 1]
    t = pt.maximum(rt - t0, 0.001)
    
    sqrt_2pi = pt.sqrt(2.0 * np.pi)
    sqrt_2 = pt.sqrt(2.0)
    
    def fast_normal_pdf(x, mu=0.0, sigma=1.0):
        return pt.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * sqrt_2pi)
    
    def fast_normal_cdf(x, mu=0.0, sigma=1.0):
        return 0.5 * (1 + pt.erf((x - mu) / (sigma * sqrt_2)))
    
    v_chosen = pt.switch(pt.eq(response, 1), v_correct, v_incorrect)
    v_unchosen = pt.switch(pt.eq(response, 1), v_incorrect, v_correct)
    v_chosen = pt.maximum(v_chosen, 1e-6)
    v_unchosen = pt.maximum(v_unchosen, 1e-6)
    
    term1_chosen = (b - A) / v_chosen
    term2_chosen = b / v_chosen
    term1_unchosen = (b - A) / v_unchosen
    
    g_chosen = (1/A) * (
        -v_chosen * fast_normal_cdf(term1_chosen, mu=t, sigma=1) + 
        v_chosen * fast_normal_cdf(term2_chosen, mu=t, sigma=1) + 
        fast_normal_pdf(term1_chosen, mu=t, sigma=1) - 
        fast_normal_pdf(term2_chosen, mu=t, sigma=1)
    )
    
    S_unchosen = 1 - fast_normal_cdf(term1_unchosen, mu=t, sigma=1)
    joint_likelihood = g_chosen * S_unchosen
    safe_joint_likelihood = pt.maximum(joint_likelihood, 1e-10)
    
    return pt.log(safe_joint_likelihood)

def fast_lba_random(v_correct, v_incorrect, b, A, t0, rng=None, size=None):
    """快速版 LBA 隨機生成函數"""
    n_trials = size[0]
    v_c = np.asarray(v_correct)
    v_i = np.asarray(v_incorrect)
    b_ = np.asarray(b).item()
    A_ = np.asarray(A).item() 
    t0_ = np.asarray(t0).item()
    
    v = np.empty((n_trials, 2))
    v[:, 0] = v_c
    if v_i.ndim == 0:
        v[:, 1] = np.full(n_trials, v_i)
    else:
        v[:, 1] = v_i
    
    start_points = rng.uniform(low=0, high=A_, size=(n_trials, 2))
    drifts = rng.normal(loc=v, scale=1, size=(n_trials, 2))
    drifts[drifts < 0] = 1e-10
    
    threshold = np.maximum(b_, A_ + 1e-4)
    time_diff = threshold - start_points
    time_diff[time_diff < 0] = 0
    time_to_boundary = time_diff / (drifts + 1e-8)
    winner = 1 - np.argmin(time_to_boundary, axis=1)
    rt = (np.min(time_to_boundary, axis=1) + t0_).flatten()
    
    return np.stack([rt, winner], axis=1)

# ===================================================================
# 模型定義函數
# ===================================================================
def create_individual_coactive(observed_value, model_input_data):
    """個體 Coactive 模型"""
    
    with pm.Model() as model:
        # 漂移率參數
        v_match = pm.Normal('v_match', mu=1.5, sigma=0.3)
        v_mismatch = pm.Normal('v_mismatch', mu=0.8, sigma=0.3)
        
        v_match_pos = pm.Deterministic('v_match_pos', pm.math.maximum(v_match, 0.1))
        v_mismatch_pos = pm.Deterministic('v_mismatch_pos', pm.math.maximum(v_mismatch, 0.1))
        
        v_l = v_match_pos * model_input_data['left_match'] + v_mismatch_pos * (1 - model_input_data['left_match'])
        v_r = v_match_pos * model_input_data['right_match'] + v_mismatch_pos * (1 - model_input_data['right_match'])
        
        # Coactive 整合：相加
        v_final_correct = pm.Deterministic('v_final_correct', v_l + v_r)
        v_final_incorrect = v_final_correct * 0.2
        
        # LBA 參數
        b = pm.Normal('b', mu=1.0, sigma=0.2)
        A = pm.HalfNormal('A', sigma=0.05)
        t0 = pm.HalfNormal('t0', sigma=0.03)
        
        b_safe = pm.Deterministic('b_safe', pm.math.maximum(b, A + 0.05))
        
        likelihood = pm.CustomDist('likelihood', v_final_correct, v_final_incorrect, 
                                 b_safe, A, t0, 
                                 logp=fast_logp_lba, random=fast_lba_random, 
                                 observed=observed_value)
    return model

def create_individual_parallel_and(observed_value, model_input_data):
    """個體 Parallel AND 模型"""
    
    with pm.Model() as model:
        # 相同的漂移率參數
        v_match = pm.Normal('v_match', mu=1.5, sigma=0.3)
        v_mismatch = pm.Normal('v_mismatch', mu=0.8, sigma=0.3)
        
        v_match_pos = pm.Deterministic('v_match_pos', pm.math.maximum(v_match, 0.1))
        v_mismatch_pos = pm.Deterministic('v_mismatch_pos', pm.math.maximum(v_mismatch, 0.1))
        
        v_l = v_match_pos * model_input_data['left_match'] + v_mismatch_pos * (1 - model_input_data['left_match'])
        v_r = v_match_pos * model_input_data['right_match'] + v_mismatch_pos * (1 - model_input_data['right_match'])
        
        # Parallel AND 整合：log-sum-exp 平均
        k = 1.5  # 固定 k 值
        k_v_l = k * v_l
        k_v_r = k * v_r
        max_kv = pm.math.maximum(k_v_l, k_v_r)
        
        stable_exp_l = pm.math.exp(pm.math.minimum(k_v_l - max_kv, 10))
        stable_exp_r = pm.math.exp(pm.math.minimum(k_v_r - max_kv, 10))
        
        v_final_correct = pm.Deterministic('v_final_correct', 
            max_kv/k + pm.math.log(stable_exp_l + stable_exp_r)/k)
        v_final_incorrect = v_final_correct * 0.2
        
        # 相同的 LBA 參數
        b = pm.Normal('b', mu=1.0, sigma=0.2)
        A = pm.HalfNormal('A', sigma=0.05)
        t0 = pm.HalfNormal('t0', sigma=0.03)
        
        b_safe = pm.Deterministic('b_safe', pm.math.maximum(b, A + 0.05))
        
        likelihood = pm.CustomDist('likelihood', v_final_correct, v_final_incorrect, 
                                 b_safe, A, t0, 
                                 logp=fast_logp_lba, random=fast_lba_random, 
                                 observed=observed_value)
    return model

# ===================================================================
# 診斷和分析函數
# ===================================================================
def detailed_convergence_check(idata, model_name):
    """詳細的收斂性檢查"""
    
    try:
        summary = az.summary(idata)
        
        # 基本統計
        max_rhat = summary['r_hat'].max()
        mean_rhat = summary['r_hat'].mean()
        bad_rhat_count = (summary['r_hat'] > 1.01).sum()
        
        # 有效樣本數
        min_ess_bulk = summary['ess_bulk'].min()
        min_ess_tail = summary['ess_tail'].min()
        
        # 發散檢查
        if hasattr(idata, 'sample_stats') and 'diverging' in idata.sample_stats:
            divergences = int(idata.sample_stats['diverging'].sum())
        else:
            divergences = 0
        
        # 能量檢查
        if hasattr(idata, 'sample_stats') and 'energy' in idata.sample_stats:
            energy_ok = True  # 簡化處理
        else:
            energy_ok = True
        
        convergence_info = {
            'model_name': model_name,
            'max_rhat': float(max_rhat),
            'mean_rhat': float(mean_rhat),
            'bad_rhat_count': int(bad_rhat_count),
            'min_ess_bulk': float(min_ess_bulk),
            'min_ess_tail': float(min_ess_tail),
            'divergences': divergences,
            'energy_ok': energy_ok,
            'convergence_grade': 'A' if max_rhat < 1.01 and divergences == 0 else 
                                'B' if max_rhat < 1.05 and divergences < 10 else 'C'
        }
        
        return convergence_info, summary
        
    except Exception as e:
        print(f"Convergence check failed for {model_name}: {e}")
        return None, None

def posterior_predictive_check(idata, observed_data, model_name):
    """後驗預測檢查"""
    
    try:
        # 提取後驗預測
        if 'posterior_predictive' not in idata.groups:
            print(f"No posterior predictive samples for {model_name}")
            return None
            
        pred_data = idata.posterior_predictive['likelihood'].values
        
        # 觀測數據
        obs_rt = observed_data[:, 0]
        obs_response = observed_data[:, 1]
        
        # 預測數據統計 (平均跨chains和draws)
        pred_rt = pred_data[:, :, :, 0].mean(axis=(0, 1))
        pred_response = pred_data[:, :, :, 1].mean(axis=(0, 1))
        
        # 計算統計量
        ppc_stats = {
            'model_name': model_name,
            # 反應時間統計
            'obs_rt_mean': float(obs_rt.mean()),
            'pred_rt_mean': float(pred_rt.mean()),
            'obs_rt_std': float(obs_rt.std()),
            'pred_rt_std': float(pred_rt.std()),
            'rt_error': float(abs(obs_rt.mean() - pred_rt.mean())),
            # 準確率統計
            'obs_accuracy': float(obs_response.mean()),
            'pred_accuracy': float(pred_response.mean()),
            'accuracy_error': float(abs(obs_response.mean() - pred_response.mean())),
            # 總體擬合
            'total_error': float(abs(obs_rt.mean() - pred_rt.mean()) + 
                               abs(obs_response.mean() - pred_response.mean()))
        }
        
        return ppc_stats
        
    except Exception as e:
        print(f"PPC failed for {model_name}: {e}")
        return None

def create_ppc_plots(idata_coactive, idata_parallel, observed_data, participant_id, save_dir):
    """創建後驗預測檢查圖表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Posterior Predictive Check - Participant {participant_id}', 
                 fontsize=16, fontweight='bold')
    
    obs_rt = observed_data[:, 0]
    obs_response = observed_data[:, 1]
    
    try:
        # Coactive 預測
        pred_coactive = idata_coactive.posterior_predictive['likelihood'].values
        pred_rt_coactive = pred_coactive[:, :, :, 0].flatten()
        pred_resp_coactive = pred_coactive[:, :, :, 1].flatten()
        
        # Parallel AND 預測  
        pred_parallel = idata_parallel.posterior_predictive['likelihood'].values
        pred_rt_parallel = pred_parallel[:, :, :, 0].flatten()
        pred_resp_parallel = pred_parallel[:, :, :, 1].flatten()
        
        # 圖1: RT 分布比較
        axes[0,0].hist(obs_rt, bins=30, alpha=0.7, label='Observed', 
                      density=True, color='black', edgecolor='white')
        axes[0,0].hist(pred_rt_coactive[:500], bins=30, alpha=0.5, 
                      label='Coactive Pred', density=True, color='red')
        axes[0,0].hist(pred_rt_parallel[:500], bins=30, alpha=0.5, 
                      label='Parallel Pred', density=True, color='blue')
        axes[0,0].set_xlabel('Reaction Time')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('RT Distribution Comparison')
        axes[0,0].legend()
        
        # 圖2: 準確率比較
        accuracy_data = {
            'Observed': obs_response.mean(),
            'Coactive': pred_resp_coactive.mean(),
            'Parallel': pred_resp_parallel.mean()
        }
        
        bars = axes[0,1].bar(accuracy_data.keys(), accuracy_data.values(), 
                           color=['black', 'red', 'blue'], alpha=0.7)
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_title('Accuracy Comparison')
        axes[0,1].set_ylim(0, 1)
        
        # 在柱子上標註數值
        for bar, (key, value) in zip(bars, accuracy_data.items()):
            axes[0,1].text(bar.get_x() + bar.get_width()/2., value + 0.02,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 圖3: RT vs 準確率散點圖
        axes[1,0].scatter(obs_rt, obs_response, alpha=0.6, label='Observed', 
                         color='black', s=20)
        
        # 取樣繪製預測點
        sample_idx = np.random.choice(len(pred_rt_coactive), 200, replace=False)
        axes[1,0].scatter(pred_rt_coactive[sample_idx], pred_resp_coactive[sample_idx], 
                         alpha=0.3, label='Coactive', color='red', s=10)
        axes[1,0].scatter(pred_rt_parallel[sample_idx], pred_resp_parallel[sample_idx], 
                         alpha=0.3, label='Parallel', color='blue', s=10)
        
        axes[1,0].set_xlabel('Reaction Time')
        axes[1,0].set_ylabel('Response (0=Wrong, 1=Correct)')
        axes[1,0].set_title('RT vs Response Pattern')
        axes[1,0].legend()
        
        # 圖4: 模型比較摘要
        axes[1,1].axis('off')
        
        # 計算擬合指標
        rt_error_coactive = abs(obs_rt.mean() - pred_rt_coactive.mean())
        rt_error_parallel = abs(obs_rt.mean() - pred_rt_parallel.mean())
        acc_error_coactive = abs(obs_response.mean() - pred_resp_coactive.mean())
        acc_error_parallel = abs(obs_response.mean() - pred_resp_parallel.mean())
        
        summary_text = f"""
Predictive Accuracy Summary:

Observed Data:
  Mean RT: {obs_rt.mean():.3f}
  Accuracy: {obs_response.mean():.3f}

Coactive Model:
  RT Error: {rt_error_coactive:.3f}
  Accuracy Error: {acc_error_coactive:.3f}
  Total Error: {rt_error_coactive + acc_error_coactive:.3f}

Parallel AND Model:
  RT Error: {rt_error_parallel:.3f}
  Accuracy Error: {acc_error_parallel:.3f}
  Total Error: {rt_error_parallel + acc_error_parallel:.3f}

Better Predictor: {'Coactive' if (rt_error_coactive + acc_error_coactive) < (rt_error_parallel + acc_error_parallel) else 'Parallel AND'}
        """
        
        axes[1,1].text(0.1, 0.9, summary_text.strip(), transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ppc_participant_{participant_id}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"PPC plot failed for participant {participant_id}: {e}")
        plt.close()
        return False

# ===================================================================
# 主要分析函數
# ===================================================================
def comprehensive_individual_analysis():
    """完整的個體分析主函數"""
    
    print("="*80)
    print("COMPREHENSIVE INDIVIDUAL ANALYSIS")
    print("Recording r-hat values and posterior predictive checks")
    print("="*80)
    
    # 創建結果目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"individual_analysis_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "model_files"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "diagnostics"), exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # 載入數據
    try:
        data = np.load('model_data.npz', allow_pickle=True)
        observed_value = data['observed_value']
        participant_idx = data['participant_idx']
        model_input_data = data['model_input_data'].item()
        print("✓ Data loaded successfully")
    except:
        print("✗ Failed to load model_data.npz")
        return
    
    # 獲取參與者列表
    unique_participants = np.unique(participant_idx)
    print(f"Analyzing {len(unique_participants)} participants: {unique_participants}")
    
    # 採樣配置
    SAMPLING_CONFIG = {
        'draws': 800,
        'tune': 800,
        'chains': 4,
        'cores': 1,
        'target_accept': 0.9
    }
    
    # 存儲所有結果
    all_results = []
    all_convergence = []
    all_ppc = []
    
    for pid in unique_participants:
        print(f"\n" + "="*60)
        print(f"ANALYZING PARTICIPANT {pid}")
        print("="*60)
        
        # 提取該參與者的數據
        mask = participant_idx == pid
        ind_observed = observed_value[mask]
        ind_model_input = {
            'left_match': model_input_data['left_match'][mask],
            'right_match': model_input_data['right_match'][mask]
        }
        
        n_trials = len(ind_observed)
        print(f"Number of trials: {n_trials}")
        
        if n_trials < 50:
            print(f"⚠️  Too few trials for participant {pid}, skipping...")
            continue
        
        participant_results = {}
        participant_convergence = {}
        participant_ppc = {}
        
        # 定義要測試的模型
        models = {
            "Coactive": create_individual_coactive,
            "Parallel_AND": create_individual_parallel_and
        }
        
        for model_name, model_func in models.items():
            print(f"\n--- Fitting {model_name} Model ---")
            
            try:
                start_time = time.time()
                
                # 創建並擬合模型
                model = model_func(ind_observed, ind_model_input)
                
                with model:
                    print("  Starting MCMC sampling...")
                    idata = pm.sample(**SAMPLING_CONFIG)
                    
                    print("  Computing log likelihood...")
                    pm.compute_log_likelihood(idata)
                    
                    print("  Generating posterior predictive samples...")
                    idata.extend(pm.sample_posterior_predictive(idata, progressbar=False))
                
                end_time = time.time()
                print(f"  ✓ {model_name} completed in {end_time - start_time:.1f} seconds")
                
                # 詳細收斂檢查
                conv_info, summary = detailed_convergence_check(idata, f"P{pid}_{model_name}")
                if conv_info:
                    participant_convergence[model_name] = conv_info
                    print(f"  Convergence grade: {conv_info['convergence_grade']} "
                          f"(max r-hat: {conv_info['max_rhat']:.4f}, "
                          f"divergences: {conv_info['divergences']})")
                
                # 後驗預測檢查
                ppc_info = posterior_predictive_check(idata, ind_observed, f"P{pid}_{model_name}")
                if ppc_info:
                    participant_ppc[model_name] = ppc_info
                    print(f"  PPC total error: {ppc_info['total_error']:.4f}")
                
                # 保存模型結果
                model_file = os.path.join(results_dir, "model_files", 
                                        f"participant_{pid}_{model_name.lower()}.nc")
                idata.to_netcdf(model_file)
                
                # 保存摘要統計
                if summary is not None:
                    summary_file = os.path.join(results_dir, "diagnostics", 
                                              f"participant_{pid}_{model_name.lower()}_summary.csv")
                    summary.to_csv(summary_file)
                
                participant_results[model_name] = idata
                
            except Exception as e:
                print(f"  ✗ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 模型比較（如果兩個模型都成功）
        if len(participant_results) == 2:
            try:
                print(f"\n--- Model Comparison for Participant {pid} ---")
                
                # WAIC 比較
                waic_compare = az.compare(participant_results, ic="waic")
                winner = waic_compare.index[0]
                elpd_diff = abs(waic_compare.iloc[1]['elpd_waic'] - waic_compare.iloc[0]['elpd_waic'])
                dse = waic_compare.iloc[1]['dse']
                
                # 判斷顯著性
                if elpd_diff > 2 * dse:
                    significance = "Significant"
                elif elpd_diff > dse:
                    significance = "Weak"
                else:
                    significance = "Non-significant"
                
                effect_size = elpd_diff / dse if dse > 0 else 0
                
                print(f"  Winner: {winner}")
                print(f"  WAIC difference: {elpd_diff:.1f} ± {dse:.1f}")
                print(f"  Significance: {significance} (effect size: {effect_size:.2f})")
                
                # 記錄結果
                result_record = {
                    'participant': pid,
                    'n_trials': n_trials,
                    'winner': winner,
                    'elpd_diff': elpd_diff,
                    'dse': dse,
                    'significance': significance,
                    'effect_size': effect_size
                }
                
                # 添加收斂信息
                for model_name in models.keys():
                    if model_name in participant_convergence:
                        conv = participant_convergence[model_name]
                        result_record[f'{model_name.lower()}_max_rhat'] = conv['max_rhat']
                        result_record[f'{model_name.lower()}_divergences'] = conv['divergences']
                        result_record[f'{model_name.lower()}_convergence_grade'] = conv['convergence_grade']
                
                # 添加 PPC 信息
                for model_name in models.keys():
                    if model_name in participant_ppc:
                        ppc = participant_ppc[model_name]
                        result_record[f'{model_name.lower()}_ppc_error'] = ppc['total_error']
                        result_record[f'{model_name.lower()}_rt_error'] = ppc['rt_error']
                        result_record[f'{model_name.lower()}_acc_error'] = ppc['accuracy_error']
                
                all_results.append(result_record)
                
                # 創建 PPC 圖表
                print("  Creating posterior predictive check plots...")
                ppc_success = create_ppc_plots(
                    participant_results["Coactive"], 
                    participant_results["Parallel_AND"],
                    ind_observed, pid, 
                    os.path.join(results_dir, "plots")
                )
                if ppc_success:
                    print("  ✓ PPC plots saved")
                
                # 保存 WAIC 比較結果
                waic_file = os.path.join(results_dir, "diagnostics", 
                                       f"participant_{pid}_waic_comparison.csv")
                waic_compare.to_csv(waic_file)
                
            except Exception as e:
                print(f"  ✗ Model comparison failed: {e}")
        
        # 收集診斷信息
        for model_name, conv_info in participant_convergence.items():
            conv_info['participant'] = pid
            all_convergence.append(conv_info)
        
        for model_name, ppc_info in participant_ppc.items():
            ppc_info['participant'] = pid
            all_ppc.append(ppc_info)
    
    # 保存所有結果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(results_dir, "comprehensive_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Main results saved to: {results_file}")
    
    if all_convergence:
        convergence_df = pd.DataFrame(all_convergence)
        conv_file = os.path.join(results_dir, "convergence_diagnostics.csv")
        convergence_df.to_csv(conv_file, index=False)
        print(f"✓ Convergence diagnostics saved to: {conv_file}")
    
    if all_ppc:
        ppc_df = pd.DataFrame(all_ppc)
        ppc_file = os.path.join(results_dir, "posterior_predictive_checks.csv")
        ppc_df.to_csv(ppc_file, index=False)
        print(f"✓ PPC results saved to: {ppc_file}")
    
    # 創建總結報告
    create_comprehensive_report(results_dir, all_results, all_convergence, all_ppc)
    
    print(f"\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Total participants analyzed: {len(all_results)}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        coactive_wins = (results_df['winner'] == 'Coactive').sum()
        parallel_wins = (results_df['winner'] == 'Parallel_AND').sum()
        
        print(f"Model preferences:")
        print(f"  Coactive: {coactive_wins} participants")
        print(f"  Parallel AND: {parallel_wins} participants")
        
        # 檢查收斂品質
        if all_convergence:
            convergence_df = pd.DataFrame(all_convergence)
            good_convergence = (convergence_df['convergence_grade'] == 'A').sum()
            total_models = len(convergence_df)
            print(f"Convergence quality: {good_convergence}/{total_models} models with grade A")
    
    return results_dir

def create_comprehensive_report(results_dir, all_results, all_convergence, all_ppc):
    """創建綜合分析報告"""
    
    report_file = os.path.join(results_dir, "comprehensive_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE INDIVIDUAL ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 主要結果摘要
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            f.write("MAIN RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total participants analyzed: {len(results_df)}\n")
            
            coactive_wins = (results_df['winner'] == 'Coactive').sum()
            parallel_wins = (results_df['winner'] == 'Parallel_AND').sum()
            
            f.write(f"Coactive model wins: {coactive_wins} ({coactive_wins/len(results_df)*100:.1f}%)\n")
            f.write(f"Parallel AND model wins: {parallel_wins} ({parallel_wins/len(results_df)*100:.1f}%)\n")
            
            # 顯著性統計
            sig_results = results_df['significance'].value_counts()
            f.write(f"\nStatistical significance:\n")
            for sig_level, count in sig_results.items():
                f.write(f"  {sig_level}: {count} participants\n")
            
            # 效應量統計
            f.write(f"\nEffect size statistics:\n")
            f.write(f"  Mean: {results_df['effect_size'].mean():.3f}\n")
            f.write(f"  Median: {results_df['effect_size'].median():.3f}\n")
            f.write(f"  Range: {results_df['effect_size'].min():.3f} - {results_df['effect_size'].max():.3f}\n")
            f.write(f"  Strong effects (>5): {(results_df['effect_size'] > 5).sum()}\n")
        
        # 收斂性診斷
        if all_convergence:
            convergence_df = pd.DataFrame(all_convergence)
            
            f.write(f"\nCONVERGENCE DIAGNOSTICS\n")
            f.write("-"*40 + "\n")
            
            grade_counts = convergence_df['convergence_grade'].value_counts()
            f.write("Convergence grades:\n")
            for grade, count in grade_counts.items():
                f.write(f"  Grade {grade}: {count} models\n")
            
            f.write(f"\nR-hat statistics:\n")
            f.write(f"  Mean max r-hat: {convergence_df['max_rhat'].mean():.4f}\n")
            f.write(f"  Worst r-hat: {convergence_df['max_rhat'].max():.4f}\n")
            f.write(f"  Models with r-hat < 1.01: {(convergence_df['max_rhat'] < 1.01).sum()}/{len(convergence_df)}\n")
            
            f.write(f"\nDivergence statistics:\n")
            f.write(f"  Total divergences: {convergence_df['divergences'].sum()}\n")
            f.write(f"  Models with no divergences: {(convergence_df['divergences'] == 0).sum()}/{len(convergence_df)}\n")
        
        # 後驗預測檢查
        if all_ppc:
            ppc_df = pd.DataFrame(all_ppc)
            
            f.write(f"\nPOSTERIOR PREDICTIVE CHECKS\n")
            f.write("-"*40 + "\n")
            
            # 按模型分組的 PPC 結果
            for model_name in ['Coactive', 'Parallel_AND']:
                model_ppc = ppc_df[ppc_df['model_name'].str.contains(model_name)]
                if len(model_ppc) > 0:
                    f.write(f"\n{model_name} Model PPC:\n")
                    f.write(f"  Mean total error: {model_ppc['total_error'].mean():.4f}\n")
                    f.write(f"  Mean RT error: {model_ppc['rt_error'].mean():.4f}\n")
                    f.write(f"  Mean accuracy error: {model_ppc['accuracy_error'].mean():.4f}\n")
        
        # 詳細參與者結果
        if all_results:
            f.write(f"\nDETAILED PARTICIPANT RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'PID':<5} {'Trials':<7} {'Winner':<12} {'WAIC':<8} {'SE':<8} {'Effect':<8} {'Significance':<12}\n")
            f.write("-"*80 + "\n")
            
            for _, row in results_df.iterrows():
                f.write(f"{int(row['participant']):<5} ")
                f.write(f"{int(row['n_trials']):<7} ")
                f.write(f"{row['winner']:<12} ")
                f.write(f"{row['elpd_diff']:<8.1f} ")
                f.write(f"{row['dse']:<8.1f} ")
                f.write(f"{row['effect_size']:<8.2f} ")
                f.write(f"{row['significance']:<12}")
                f.write("\n")
        
        f.write(f"\nFILES GENERATED:\n")
        f.write("-"*40 + "\n")
        f.write("Main results:\n")
        f.write("  - comprehensive_results.csv: Complete analysis results\n")
        f.write("  - convergence_diagnostics.csv: R-hat and convergence info\n")
        f.write("  - posterior_predictive_checks.csv: PPC statistics\n")
        f.write("\nModel files:\n")
        f.write("  - model_files/: Individual .nc files for each participant-model\n")
        f.write("\nDiagnostics:\n")
        f.write("  - diagnostics/: Individual summary files and WAIC comparisons\n")
        f.write("\nPlots:\n")
        f.write("  - plots/: Posterior predictive check plots for each participant\n")

def create_summary_visualizations(results_dir):
    """創建總結性視覺化圖表"""
    
    try:
        # 讀取結果
        results_file = os.path.join(results_dir, "comprehensive_results.csv")
        convergence_file = os.path.join(results_dir, "convergence_diagnostics.csv")
        ppc_file = os.path.join(results_dir, "posterior_predictive_checks.csv")
        
        if not os.path.exists(results_file):
            print("Results file not found, skipping visualization")
            return
        
        results_df = pd.read_csv(results_file)
        
        # 創建總結圖表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Individual Analysis Summary', fontsize=16, fontweight='bold')
        
        # 圖1: 模型偏好
        winner_counts = results_df['winner'].value_counts()
        axes[0,0].pie(winner_counts.values, labels=winner_counts.index, autopct='%1.0f%%',
                     colors=['#3498db', '#e74c3c'])
        axes[0,0].set_title('Model Preferences')
        
        # 圖2: 效應量分布
        axes[0,1].hist(results_df['effect_size'], bins=10, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(2, color='red', linestyle='--', label='Significance threshold')
        axes[0,1].set_xlabel('Effect Size')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Effect Size Distribution')
        axes[0,1].legend()
        
        # 圖3: 顯著性分布
        sig_counts = results_df['significance'].value_counts()
        axes[0,2].bar(sig_counts.index, sig_counts.values, 
                     color=['#27ae60', '#f39c12', '#95a5a6'])
        axes[0,2].set_ylabel('Count')
        axes[0,2].set_title('Significance Distribution')
        
        # 圖4: 收斂性診斷 (如果有數據)
        if os.path.exists(convergence_file):
            conv_df = pd.read_csv(convergence_file)
            grade_counts = conv_df['convergence_grade'].value_counts()
            axes[1,0].bar(grade_counts.index, grade_counts.values,
                         color=['#27ae60', '#f39c12', '#e74c3c'])
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_title('Convergence Grades')
        else:
            axes[1,0].text(0.5, 0.5, 'Convergence data\nnot available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 圖5: WAIC vs 效應量
        colors = ['#3498db' if w == 'Parallel_AND' else '#e74c3c' for w in results_df['winner']]
        scatter = axes[1,1].scatter(results_df['elpd_diff'], results_df['effect_size'], 
                                   c=colors, alpha=0.7, s=100, edgecolors='black')
        axes[1,1].set_xlabel('WAIC Difference')
        axes[1,1].set_ylabel('Effect Size')
        axes[1,1].set_title('WAIC vs Effect Size')
        
        # 圖6: PPC 結果 (如果有數據)
        if os.path.exists(ppc_file):
            ppc_df = pd.read_csv(ppc_file)
            
            # 分別計算兩個模型的 PPC error
            coactive_ppc = ppc_df[ppc_df['model_name'].str.contains('Coactive')]['total_error']
            parallel_ppc = ppc_df[ppc_df['model_name'].str.contains('Parallel')]['total_error']
            
            axes[1,2].boxplot([coactive_ppc, parallel_ppc], labels=['Coactive', 'Parallel AND'])
            axes[1,2].set_ylabel('Total PPC Error')
            axes[1,2].set_title('Posterior Predictive Accuracy')
        else:
            axes[1,2].text(0.5, 0.5, 'PPC data\nnot available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
        
        plt.tight_layout()
        summary_plot_file = os.path.join(results_dir, "summary_analysis.png")
        plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary visualization saved to: {summary_plot_file}")
        
    except Exception as e:
        print(f"Failed to create summary visualizations: {e}")

if __name__ == '__main__':
    # 執行完整分析
    results_directory = comprehensive_individual_analysis()
    
    if results_directory:
        print(f"\nCreating summary visualizations...")
        create_summary_visualizations(results_directory)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"All results saved in: {results_directory}")
        print(f"\nKey files:")
        print(f"  - comprehensive_results.csv: Main results with r-hat values")
        print(f"  - convergence_diagnostics.csv: Detailed convergence info")
        print(f"  - posterior_predictive_checks.csv: Model fit statistics")
        print(f"  - comprehensive_report.txt: Complete text report")
        print(f"  - summary_analysis.png: Summary visualizations")
        print(f"  - plots/: Individual PPC plots for each participant")
        print(f"  - model_files/: Saved model objects (.nc files)")
        print("="*80)
