# -*- coding: utf-8 -*-
"""
fit_all_participants_parallel_and.py - Fit ParallelAND model for all participants
處理所有受試者資料，計算log-likelihood和BIC
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from typing import Dict
import warnings
warnings.filterwarnings('ignore')
import time

class ParallelANDModelFitter:
    """ParallelAND模型fitting類別"""
    
    def __init__(self):
        self.results = []
        self.failed_participants = []
        
    def load_all_participants(self, file_path: str = "GRT_LBA.csv", accuracy_threshold: float = 0.65) -> Dict:
        """載入所有受試者資料並過濾正確率"""
        
        print("📊 載入所有受試者資料...")
        data = pd.read_csv(file_path)
        
        # 轉換stimulus格式
        data['left_stimulus'] = data['Chanel1']  # 0=nonvertical, 1=vertical
        data['right_stimulus'] = data['Chanel2']
        
        participants = sorted(data['participant'].unique())
        print(f"   發現 {len(participants)} 位受試者: {participants}")
        
        participant_data = {}
        filtered_count = 0
        
        for pid in participants:
            p_data = data[data['participant'] == pid].copy()
            
            # 計算正確率
            accuracy = self._calculate_accuracy(p_data)
            
            if accuracy >= accuracy_threshold:
                participant_data[pid] = {
                    'responses': p_data['Response'].astype(int).values,
                    'rts': p_data['RT'].astype(float).values,
                    'left_stimuli': p_data['Chanel1'].values,
                    'right_stimuli': p_data['Chanel2'].values,
                    'n_trials': len(p_data),
                    'participant_id': pid,
                    'accuracy': accuracy
                }
                print(f"   ✅ 參與者 {pid}: {len(p_data)} trials, 正確率={accuracy:.3f}")
            else:
                filtered_count += 1
                print(f"   ❌ 參與者 {pid}: 正確率={accuracy:.3f} < {accuracy_threshold}, 已過濾")
        
        print(f"\n📋 過濾結果:")
        print(f"   保留: {len(participant_data)} 位受試者")
        print(f"   過濾: {filtered_count} 位受試者")
        print(f"   正確率閾值: {accuracy_threshold}")
        
        return participant_data
    
    def create_parallel_and_model(self, data: Dict) -> pm.Model:
        """創建ParallelAND PyMC模型"""
        
        with pm.Model() as model:
            # Prior distributions (same as single_side_lba)
            left_v_vertical = pm.Gamma('left_v_vertical', alpha=2.5, beta=1.5)
            left_v_nonvertical = pm.Gamma('left_v_nonvertical', alpha=2.5, beta=1.5)
            left_v_vertical_error = pm.Gamma('left_v_vertical_error', alpha=2.0, beta=3.0)
            left_v_nonvertical_error = pm.Gamma('left_v_nonvertical_error', alpha=2.0, beta=3.0)
            A = pm.HalfNormal("start_point_variability", sigma=0.5)

            right_v_vertical = pm.Gamma('right_v_vertical', alpha=2.5, beta=1.5)
            right_v_nonvertical = pm.Gamma('right_v_nonvertical', alpha=2.5, beta=1.5)
            right_v_vertical_error = pm.Gamma('right_v_vertical_error', alpha=2.0, beta=3.0)
            right_v_nonvertical_error = pm.Gamma('right_v_nonvertical_error', alpha=2.0, beta=3.0)
            
            threshold = pm.Gamma('threshold', alpha=3.0, beta=3.5)
            ndt = pm.Uniform('ndt', lower=0.05, upper=0.6)
            effective_boundary = threshold  - A / 2
            

            # 轉換資料為tensors
            responses_tensor = pt.as_tensor_variable(data['responses'])
            rts_tensor = pt.as_tensor_variable(data['rts'])
            left_stimuli_tensor = pt.as_tensor_variable(data['left_stimuli'])
            right_stimuli_tensor = pt.as_tensor_variable(data['right_stimuli'])
            
            # 計算ParallelAND RT predictions
            predicted_rts = self._compute_parallel_and_rt(
                left_stimuli_tensor, right_stimuli_tensor, responses_tensor,
                left_v_vertical, left_v_nonvertical, left_v_vertical_error, left_v_nonvertical_error,
                right_v_vertical, right_v_nonvertical, right_v_vertical_error, right_v_nonvertical_error,
                threshold, ndt
            )
            
            # Likelihood
            rt_sigma = pm.HalfNormal('rt_sigma', sigma=0.3)
            pm.Normal('rt_obs', mu=predicted_rts, sigma=rt_sigma, observed=rts_tensor)
            
        return model
    
    def _compute_parallel_and_rt(self, left_stimuli, right_stimuli, responses,
                                left_v_v, left_v_nv, left_v_v_err, left_v_nv_err,
                                right_v_v, right_v_nv, right_v_v_err, right_v_nv_err,
                                threshold, ndt):
        """計算ParallelAND RT predictions"""
        
        # 左側drift rates (基於stimulus-response組合)
        left_is_vertical_stim = pt.eq(left_stimuli, 1)
        left_is_vertical_resp = pt.or_(pt.eq(responses, 1), pt.eq(responses, 2))
        
        # 依據single_side_lba邏輯：stimulus決定是否使用error drift rate
        left_drift = pt.where(
            left_is_vertical_stim,  # Vertical stimulus
            pt.where(left_is_vertical_resp, left_v_v, left_v_nv_err),
            pt.where(left_is_vertical_resp, left_v_v_err, left_v_nv)
        )
        
        # 右側drift rates (基於stimulus-response組合)
        right_is_vertical_stim = pt.eq(right_stimuli, 1)
        right_is_vertical_resp = pt.or_(pt.eq(responses, 0), pt.eq(responses, 1))
        
        # 依據single_side_lba邏輯：stimulus決定是否使用error drift rate
        right_drift = pt.where(
            right_is_vertical_stim,  # Vertical stimulus
            pt.where(right_is_vertical_resp, right_v_v, right_v_nv_err),
            pt.where(right_is_vertical_resp, right_v_v_err, right_v_nv)
        )
        
        # ParallelAND: 最小drift rate
        effective_drift = pt.minimum(left_drift, right_drift)
        effective_drift = pt.maximum(effective_drift, 0.05)
        
        # RT prediction
        decision_time = threshold / effective_drift
        predicted_rt = decision_time + ndt
        
        return predicted_rt
    
    def fit_participant(self, participant_id: int, data: Dict, 
                       n_samples: int = 1000, n_tune: int = 1000) -> Dict:
        """Fit模型給單一受試者"""
        
        print(f"\n🔬 Fitting participant {participant_id} ({data['n_trials']} trials)...")
        start_time = time.time()
        
        try:
            # 創建模型
            model = self.create_parallel_and_model(data)
            
            # MCMC sampling
            with model:
                trace = pm.sample(
                    draws=n_samples,
                    tune=n_tune,
                    target_accept=0.95,
                    chains=4,
                    cores=2,  # 單core避免並行問題
                    random_seed=42 + participant_id,
                    progressbar=True,  # 關閉progress bar避免過多輸出
                    return_inferencedata=True
                )
            
            # 計算posterior means
            posterior_means = {}
            for var in trace.posterior.data_vars:
                posterior_means[var] = float(trace.posterior[var].mean())
            
            # 計算log-likelihood
            log_likelihood = self._calculate_log_likelihood(data, posterior_means)
            
            # 計算BIC
            n_params = len(posterior_means)
            n_trials = data['n_trials']
            bic = -2 * log_likelihood + n_params * np.log(n_trials)
            
            # 計算其他metrics
            aic = -2 * log_likelihood + 2 * n_params
            
            fit_time = time.time() - start_time
            
            result = {
                'participant_id': participant_id,
                'n_trials': n_trials,
                'n_parameters': n_params,
                'log_likelihood': log_likelihood,
                'bic': bic,
                'aic': aic,
                'fit_time_seconds': fit_time,
                'converged': True,
                'posterior_means': posterior_means,
                'trace': trace
            }
            
            print(f"   ✅ 成功! LL={log_likelihood:.2f}, BIC={bic:.2f}, 時間={fit_time:.1f}s")
            return result
            
        except Exception as e:
            fit_time = time.time() - start_time
            print(f"   ❌ 失敗: {str(e)}")
            
            self.failed_participants.append({
                'participant_id': participant_id,
                'error': str(e),
                'fit_time_seconds': fit_time
            })
            
            return None
    
    def _calculate_log_likelihood(self, data: Dict, params: Dict) -> float:
        """計算log-likelihood"""
        
        try:
            predictions = self._generate_rt_predictions(data, params)
            actual_rts = data['rts']
            predicted_rts = predictions['predicted_rts']
            
            # 使用Normal likelihood
            sigma = params.get('rt_sigma', 0.3)
            log_likelihood = np.sum(
                -0.5 * np.log(2 * np.pi * sigma**2) - 
                0.5 * ((actual_rts - predicted_rts) / sigma)**2
            )
            
            return log_likelihood
            
        except Exception as e:
            print(f"   警告: log-likelihood計算失敗: {e}")
            return -np.inf
    
    def _generate_rt_predictions(self, data: Dict, params: Dict) -> Dict:
        """生成RT predictions"""
        
        predicted_rts = []
        
        for i in range(data['n_trials']):
            left_stim = data['left_stimuli'][i]
            right_stim = data['right_stimuli'][i]
            response = data['responses'][i]
            
            # 計算drift rates
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
            
            # RT prediction
            decision_time = params['threshold'] / effective_drift
            predicted_rt = decision_time + params['ndt']
            
            predicted_rts.append(predicted_rt)
        
        return {'predicted_rts': predicted_rts}
    
    def fit_all_participants(self, file_path: str = "GRT_LBA.csv", accuracy_threshold: float = 0.65) -> pd.DataFrame:
        """Fit模型給所有受試者（過濾正確率）"""
        
        print("🚀 開始fitting所有受試者的ParallelAND模型...")
        print("="*60)
        
        # 載入資料並過濾
        all_data = self.load_all_participants(file_path, accuracy_threshold)
        
        if len(all_data) == 0:
            print(f"❌ 沒有受試者符合正確率閾值 {accuracy_threshold}")
            return pd.DataFrame()
        
        # Fit每個受試者
        for participant_id, data in all_data.items():
            result = self.fit_participant(participant_id, data)
            if result is not None:
                # 添加正確率信息
                result['accuracy'] = data['accuracy']
                self.results.append(result)
        
        # 轉換為DataFrame
        if self.results:
            results_df = pd.DataFrame([
                {
                    'participant_id': r['participant_id'],
                    'accuracy': r['accuracy'],
                    'n_trials': r['n_trials'],
                    'n_parameters': r['n_parameters'],
                    'log_likelihood': r['log_likelihood'],
                    'bic': r['bic'],
                    'aic': r['aic'],
                    'fit_time_seconds': r['fit_time_seconds'],
                    'converged': r['converged']
                }
                for r in self.results
            ])
        else:
            results_df = pd.DataFrame()
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_prefix: str = "all_participants_parallel_and"):
        """儲存結果"""
        
        print(f"\n💾 儲存結果...")
        
        # 儲存總結果
        results_df.to_csv(f"{output_prefix}_summary.csv", index=False)
        print(f"   📁 總結果: {output_prefix}_summary.csv")
        
        # 儲存詳細參數
        detailed_params = []
        for result in self.results:
            param_row = {'participant_id': result['participant_id']}
            param_row.update(result['posterior_means'])
            detailed_params.append(param_row)
        
        if detailed_params:
            params_df = pd.DataFrame(detailed_params)
            params_df.to_csv(f"{output_prefix}_parameters.csv", index=False)
            print(f"   📁 詳細參數: {output_prefix}_parameters.csv")
        
        # 儲存失敗的受試者
        if self.failed_participants:
            failed_df = pd.DataFrame(self.failed_participants)
            failed_df.to_csv(f"{output_prefix}_failed.csv", index=False)
            print(f"   📁 失敗紀錄: {output_prefix}_failed.csv")
        
        # 印出總結
        print(f"\n📊 總結:")
        print(f"   成功fitting: {len(self.results)} 位受試者")
        print(f"   失敗: {len(self.failed_participants)} 位受試者")
        
        if len(results_df) > 0:
            print(f"   平均log-likelihood: {results_df['log_likelihood'].mean():.2f}")
            print(f"   平均BIC: {results_df['bic'].mean():.2f}")
            print(f"   平均AIC: {results_df['aic'].mean():.2f}")
            print(f"   最佳BIC: {results_df['bic'].min():.2f} (參與者 {results_df.loc[results_df['bic'].idxmin(), 'participant_id']})")

    def _calculate_accuracy(self, p_data: pd.DataFrame) -> float:
        """計算單一受試者的正確率（基於Correct欄位）"""
        
        # 直接使用Correct欄位計算正確率
        return p_data['Correct'].mean()

def main():
    """主要執行函數"""
    
    print("🎯 ParallelAND模型 - 高正確率受試者fitting (>65%)")
    print("="*60)
    
    # 初始化fitter
    fitter = ParallelANDModelFitter()
    
    # Fit所有符合條件的受試者
    results_df = fitter.fit_all_participants(accuracy_threshold=0.65)
    
    # 儲存結果
    if len(results_df) > 0:
        fitter.save_results(results_df, output_prefix="high_accuracy_parallel_and")
        
        # 顯示統計信息
        print(f"\n📊 正確率統計:")
        print(f"   平均正確率: {results_df['accuracy'].mean():.3f}")
        print(f"   正確率範圍: {results_df['accuracy'].min():.3f} - {results_df['accuracy'].max():.3f}")
        
        # 顯示前幾名結果
        print(f"\n🏆 BIC前5名:")
        top_5 = results_df.nsmallest(5, 'bic')[['participant_id', 'accuracy', 'log_likelihood', 'bic', 'aic']]
        print(top_5.to_string(index=False))
        
    else:
        print("❌ 沒有成功的結果")
    
    print(f"\n✅ 完成!")

if __name__ == "__main__":
    main()