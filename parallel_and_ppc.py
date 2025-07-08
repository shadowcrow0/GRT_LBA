# -*- coding: utf-8 -*-
"""
parallel_and_ppc.py - Posterior Predictive Check for ParallelAND LBA model
對ParallelAND模型進行後驗預測檢驗
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ParallelANDPPC:
    """ParallelAND模型後驗預測檢驗"""
    
    def __init__(self):
        self.observed_data = None
        self.posterior_samples = None
        self.simulated_data = None
        
    def load_fitted_results(self, participant_id: int, 
                           summary_file: str = "high_accuracy_parallel_and_summary.csv",
                           params_file: str = "high_accuracy_parallel_and_parameters.csv",
                           original_data_file: str = "GRT_LBA.csv") -> bool:
        """載入fitted結果和原始資料"""
        
        print(f"📊 載入參與者 {participant_id} 的fitted結果...")
        
        try:
            # 載入fitted參數
            params_df = pd.read_csv(params_file)
            participant_params = params_df[params_df['participant_id'] == participant_id]
            
            if len(participant_params) == 0:
                print(f"❌ 找不到參與者 {participant_id} 的參數")
                return False
            
            self.fitted_params = participant_params.iloc[0].to_dict()
            
            # 載入原始資料
            original_data = pd.read_csv(original_data_file)
            participant_data = original_data[original_data['participant'] == participant_id].copy()
            
            if len(participant_data) == 0:
                print(f"❌ 找不到參與者 {participant_id} 的原始資料")
                return False
            
            # 準備observed data
            self.observed_data = {
                'responses': participant_data['Response'].astype(int).values,
                'rts': participant_data['RT'].astype(float).values,
                'left_stimuli': participant_data['Chanel1'].values,
                'right_stimuli': participant_data['Chanel2'].values,
                'n_trials': len(participant_data),
                'participant_id': participant_id
            }
            
            print(f"   ✅ 成功載入 {self.observed_data['n_trials']} trials")
            return True
            
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return False
    
    def generate_posterior_samples(self, n_samples: int = 1000) -> None:
        """生成後驗樣本（模擬不確定性）"""
        
        print(f"🎲 生成 {n_samples} 個後驗樣本...")
        
        # 為簡化，我們在fitted參數周圍加入少量噪音來模擬後驗不確定性
        # 在實際應用中，這些應該來自MCMC trace
        
        base_params = self.fitted_params.copy()
        samples = []
        
        for i in range(n_samples):
            sample = {}
            for param_name, value in base_params.items():
                if param_name == 'participant_id':
                    sample[param_name] = value
                elif 'v_' in param_name:  # drift rates
                    # 在原值周圍加入10%的變異
                    sample[param_name] = np.random.gamma(value * 9, 0.1)
                elif param_name == 'threshold':
                    sample[param_name] = np.random.gamma(value * 9, 0.1)
                elif param_name == 'ndt':
                    sample[param_name] = np.random.uniform(max(0.05, value - 0.1), 
                                                         min(0.6, value + 0.1))
                elif param_name == 'start_var':
                    sample[param_name] = np.random.uniform(max(0.1, value - 0.1), 
                                                         min(0.7, value + 0.1))
                elif param_name == 'noise':
                    sample[param_name] = np.random.gamma(value * 9, 0.1)
                elif param_name == 'rt_sigma':
                    sample[param_name] = np.random.gamma(value * 9, 0.1)
                else:
                    sample[param_name] = value
            
            samples.append(sample)
        
        self.posterior_samples = samples
        print(f"   ✅ 生成完成")
    
    def simulate_data_from_posterior(self, n_sim_datasets: int = 100) -> None:
        """從後驗樣本生成模擬資料"""
        
        print(f"🔬 從後驗生成 {n_sim_datasets} 個模擬資料集...")
        
        if self.posterior_samples is None:
            print("❌ 請先生成後驗樣本")
            return
        
        simulated_datasets = []
        
        # 隨機選擇後驗樣本
        selected_samples = np.random.choice(len(self.posterior_samples), 
                                          size=n_sim_datasets, replace=True)
        
        for i, sample_idx in enumerate(selected_samples):
            params = self.posterior_samples[sample_idx]
            simulated_data = self._simulate_single_dataset(params)
            simulated_datasets.append(simulated_data)
        
        self.simulated_data = simulated_datasets
        print(f"   ✅ 生成 {len(simulated_datasets)} 個模擬資料集")
    
    def _simulate_single_dataset(self, params: Dict) -> Dict:
        """使用給定參數生成單一模擬資料集"""
        
        n_trials = self.observed_data['n_trials']
        
        simulated_responses = []
        simulated_rts = []
        
        for i in range(n_trials):
            left_stim = self.observed_data['left_stimuli'][i]
            right_stim = self.observed_data['right_stimuli'][i]
            
            # 計算4種可能反應的RT
            response_rts = []
            
            for response in range(4):
                # 確定drift rates
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
                
                # 確定使用的drift rates
                if response in [1, 2]:  # Left vertical response
                    left_drift = left_v_v
                else:  # Left nonvertical response
                    left_drift = left_v_nv
                    
                if response in [0, 1]:  # Right vertical response
                    right_drift = right_v_v
                else:  # Right nonvertical response
                    right_drift = right_v_nv
                
                # ParallelAND: 最小drift rate
                effective_drift = min(left_drift, right_drift)
                effective_drift = max(effective_drift, 0.05)
                
                # 模擬RT（加入變異性）
                mean_rt = params['threshold'] / effective_drift + params['ndt']
                rt = np.random.normal(mean_rt, params.get('rt_sigma', 0.3))
                rt = max(rt, 0.1)  # 最小RT
                
                response_rts.append(rt)
            
            # 選擇最快的反應
            chosen_response = np.argmin(response_rts)
            chosen_rt = response_rts[chosen_response]
            
            simulated_responses.append(chosen_response)
            simulated_rts.append(chosen_rt)
        
        return {
            'responses': np.array(simulated_responses),
            'rts': np.array(simulated_rts)
        }
    
    def compute_ppc_statistics(self) -> Dict:
        """計算PPC統計量"""
        
        print("📈 計算PPC統計量...")
        
        if self.simulated_data is None:
            print("❌ 請先生成模擬資料")
            return {}
        
        observed = self.observed_data
        
        # 觀察資料統計量
        obs_stats = {
            'mean_rt': np.mean(observed['rts']),
            'std_rt': np.std(observed['rts']),
            'min_rt': np.min(observed['rts']),
            'max_rt': np.max(observed['rts']),
            'median_rt': np.median(observed['rts']),
            'response_proportions': np.bincount(observed['responses'], minlength=4) / len(observed['responses'])
        }
        
        # 模擬資料統計量
        sim_stats = {
            'mean_rt': [],
            'std_rt': [],
            'min_rt': [],
            'max_rt': [],
            'median_rt': [],
            'response_proportions': []
        }
        
        for sim_data in self.simulated_data:
            sim_stats['mean_rt'].append(np.mean(sim_data['rts']))
            sim_stats['std_rt'].append(np.std(sim_data['rts']))
            sim_stats['min_rt'].append(np.min(sim_data['rts']))
            sim_stats['max_rt'].append(np.max(sim_data['rts']))
            sim_stats['median_rt'].append(np.median(sim_data['rts']))
            
            resp_props = np.bincount(sim_data['responses'], minlength=4) / len(sim_data['responses'])
            sim_stats['response_proportions'].append(resp_props)
        
        # 計算p-values (Bayesian p-values)
        p_values = {}
        for stat_name in ['mean_rt', 'std_rt', 'min_rt', 'max_rt', 'median_rt']:
            sim_values = np.array(sim_stats[stat_name])
            obs_value = obs_stats[stat_name]
            p_values[stat_name] = np.mean(sim_values >= obs_value)
        
        # Response proportions p-values
        sim_resp_props = np.array(sim_stats['response_proportions'])
        obs_resp_props = obs_stats['response_proportions']
        p_values['response_proportions'] = []
        
        for i in range(4):
            p_val = np.mean(sim_resp_props[:, i] >= obs_resp_props[i])
            p_values['response_proportions'].append(p_val)
        
        results = {
            'observed': obs_stats,
            'simulated': sim_stats,
            'p_values': p_values
        }
        
        # 印出結果
        print(f"\n📊 PPC結果 (p-values接近0.5表示模型fit良好):")
        print(f"   Mean RT: {obs_stats['mean_rt']:.3f} (p={p_values['mean_rt']:.3f})")
        print(f"   Std RT: {obs_stats['std_rt']:.3f} (p={p_values['std_rt']:.3f})")
        print(f"   Median RT: {obs_stats['median_rt']:.3f} (p={p_values['median_rt']:.3f})")
        
        print(f"\n   Response proportions:")
        for i, (obs_prop, p_val) in enumerate(zip(obs_resp_props, p_values['response_proportions'])):
            print(f"     Response {i}: {obs_prop:.3f} (p={p_val:.3f})")
        
        return results
    
    def plot_ppc_results(self, save_path: str = None) -> plt.Figure:
        """繪製PPC結果"""
        
        print("🎨 繪製PPC結果...")
        
        if self.simulated_data is None:
            print("❌ 請先生成模擬資料")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        observed = self.observed_data
        
        # 1. RT分布比較
        axes[0, 0].hist(observed['rts'], bins=30, alpha=0.7, density=True, 
                       label='Observed', color='red')
        
        # 繪製幾個模擬資料集
        for i, sim_data in enumerate(self.simulated_data[:10]):
            axes[0, 0].hist(sim_data['rts'], bins=30, alpha=0.1, density=True, 
                           color='blue', label='Simulated' if i == 0 else "")
        
        axes[0, 0].set_xlabel('RT')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('RT Distribution')
        axes[0, 0].legend()
        
        # 2. Mean RT分布
        sim_mean_rts = [np.mean(sim_data['rts']) for sim_data in self.simulated_data]
        axes[0, 1].hist(sim_mean_rts, bins=20, alpha=0.7, color='blue', density=True)
        axes[0, 1].axvline(np.mean(observed['rts']), color='red', linestyle='--', 
                          linewidth=2, label='Observed')
        axes[0, 1].set_xlabel('Mean RT')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Mean RT Distribution')
        axes[0, 1].legend()
        
        # 3. Std RT分布
        sim_std_rts = [np.std(sim_data['rts']) for sim_data in self.simulated_data]
        axes[0, 2].hist(sim_std_rts, bins=20, alpha=0.7, color='blue', density=True)
        axes[0, 2].axvline(np.std(observed['rts']), color='red', linestyle='--', 
                          linewidth=2, label='Observed')
        axes[0, 2].set_xlabel('Std RT')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('RT Std Distribution')
        axes[0, 2].legend()
        
        # 4. Response proportions
        obs_resp_props = np.bincount(observed['responses'], minlength=4) / len(observed['responses'])
        sim_resp_props = np.array([
            np.bincount(sim_data['responses'], minlength=4) / len(sim_data['responses']) 
            for sim_data in self.simulated_data
        ])
        
        x = np.arange(4)
        width = 0.35
        
        axes[1, 0].bar(x - width/2, obs_resp_props, width, label='Observed', 
                      alpha=0.7, color='red')
        axes[1, 0].bar(x + width/2, np.mean(sim_resp_props, axis=0), width, 
                      label='Simulated (mean)', alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Response')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].set_title('Response Proportions')
        axes[1, 0].set_xticks(x)
        axes[1, 0].legend()
        
        # 5. RT vs Response scatter
        axes[1, 1].scatter(observed['responses'], observed['rts'], 
                          alpha=0.6, color='red', label='Observed', s=20)
        
        # 合併所有模擬資料
        all_sim_responses = np.concatenate([sim_data['responses'] for sim_data in self.simulated_data[:5]])
        all_sim_rts = np.concatenate([sim_data['rts'] for sim_data in self.simulated_data[:5]])
        
        axes[1, 1].scatter(all_sim_responses, all_sim_rts, 
                          alpha=0.1, color='blue', label='Simulated', s=10)
        axes[1, 1].set_xlabel('Response')
        axes[1, 1].set_ylabel('RT')
        axes[1, 1].set_title('RT vs Response')
        axes[1, 1].legend()
        
        # 6. Quantile-Quantile plot
        obs_rt_sorted = np.sort(observed['rts'])
        sim_rt_quantiles = []
        
        for q in np.linspace(0.01, 0.99, len(obs_rt_sorted)):
            sim_q_values = [np.quantile(sim_data['rts'], q) for sim_data in self.simulated_data]
            sim_rt_quantiles.append(np.mean(sim_q_values))
        
        axes[1, 2].scatter(obs_rt_sorted, sim_rt_quantiles, alpha=0.6, s=20)
        axes[1, 2].plot([obs_rt_sorted.min(), obs_rt_sorted.max()], 
                       [obs_rt_sorted.min(), obs_rt_sorted.max()], 'r--', alpha=0.8)
        axes[1, 2].set_xlabel('Observed RT Quantiles')
        axes[1, 2].set_ylabel('Simulated RT Quantiles')
        axes[1, 2].set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 PPC圖表已儲存: {save_path}")
        
        return fig
    
    def run_complete_ppc(self, participant_id: int, n_posterior_samples: int = 1000, 
                        n_sim_datasets: int = 100) -> Dict:
        """執行完整的PPC分析"""
        
        print(f"🎯 執行參與者 {participant_id} 的完整PPC分析")
        print("="*60)
        
        # 載入資料
        if not self.load_fitted_results(participant_id):
            return {}
        
        # 生成後驗樣本
        self.generate_posterior_samples(n_posterior_samples)
        
        # 生成模擬資料
        self.simulate_data_from_posterior(n_sim_datasets)
        
        # 計算統計量
        ppc_results = self.compute_ppc_statistics()
        
        # 繪製結果
        fig = self.plot_ppc_results(f"ppc_participant_{participant_id}.png")
        
        # 儲存結果
        ppc_summary = {
            'participant_id': participant_id,
            'n_posterior_samples': n_posterior_samples,
            'n_sim_datasets': n_sim_datasets,
            'ppc_statistics': ppc_results
        }
        
        print(f"\n✅ PPC分析完成!")
        return ppc_summary

def main():
    """主要執行函數"""
    
    print("🔍 ParallelAND模型後驗預測檢驗 (PPC)")
    print("="*50)
    
    # 選擇要分析的受試者（可以改為從命令行參數獲取）
    participant_id = 40  # 可以修改為其他受試者
    
    # 執行PPC
    ppc = ParallelANDPPC()
    results = ppc.run_complete_ppc(participant_id, n_posterior_samples=500, n_sim_datasets=50)
    
    if results:
        print(f"\n🎉 參與者 {participant_id} 的PPC分析完成!")
    else:
        print("❌ PPC分析失敗")

if __name__ == "__main__":
    main()