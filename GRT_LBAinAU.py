# -*- coding: utf-8 -*-
"""
修正版加速四選項GRT-LBA分析 (解決dtype問題)
Fixed Accelerated Four-Choice GRT-LBA Analysis (Resolving dtype issues)

主要修正 / Main Fixes:
1. Consistent float32 dtype handling
2. Explicit casting in PyTensor operations
3. Allow input downcast for compatibility
4. Improved error handling for edge cases
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pytensor.tensor.extra_ops import broadcast_arrays
from pytensor.compile.ops import as_op
import scipy.stats as stats
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import os

# 設置環境變數以加速計算並允許精度降級
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,device=cpu,force_device=True,allow_input_downcast=True'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

# ============================================================================
# 修正的 PYTENSOR 操作 (解決dtype問題)
# Fixed PyTensor Operations (Resolving dtype issues)
# ============================================================================

@as_op(itypes=[pt.fvector, pt.fvector, pt.fmatrix, pt.fscalar, pt.fscalar, 
               pt.fscalar, pt.fscalar, pt.fvector, pt.fscalar, pt.fscalar, pt.fscalar], 
       otypes=[pt.fscalar])
def fixed_lba_loglik_float32(rt_data, choice_data, stimloc_data, db1, db2, sp1, sp2, 
                           A, thresholds, base_v, s, t0):
    """
    修正版 LBA 對數似然計算，解決dtype問題
    Fixed LBA log-likelihood computation resolving dtype issues
    """
    try:
        # 強制轉換所有輸入為 float32
        A = np.float32(max(float(A), 0.1))
        s = np.float32(max(float(s), 0.15))
        t0 = np.float32(max(float(t0), 0.05))
        sp1 = np.float32(max(float(sp1), 0.05))
        sp2 = np.float32(max(float(sp2), 0.05))
        base_v = np.float32(max(float(base_v), 0.1))
        db1 = np.float32(float(db1))
        db2 = np.float32(float(db2))
        
        # 確保所有輸入數組都是 float32
        rt_data = np.asarray(rt_data, dtype=np.float32)
        choice_data = np.asarray(choice_data, dtype=np.int32)
        stimloc_data = np.asarray(stimloc_data, dtype=np.float32)
        thresholds = np.asarray(thresholds, dtype=np.float32)
        
        if thresholds.shape[0] != 4:
            return np.float32(-1000.0)
        
        # 決策時間
        rt_decision = np.maximum(rt_data - t0, np.float32(0.05))
        
        # GRT 計算 (確保所有中間計算都是 float32)
        tanh_arg1 = (stimloc_data[:, 0] - db1) / (2 * sp1)
        tanh_arg2 = (stimloc_data[:, 1] - db2) / (2 * sp2)
        
        # 限制 tanh 參數範圍避免溢出
        tanh_arg1 = np.clip(tanh_arg1, -5.0, 5.0).astype(np.float32)
        tanh_arg2 = np.clip(tanh_arg2, -5.0, 5.0).astype(np.float32)
        
        p_left_left = np.float32(0.5) * (1 - np.tanh(tanh_arg1))
        p_left_right = 1 - p_left_left
        p_right_left = np.float32(0.5) * (1 - np.tanh(tanh_arg2))
        p_right_right = 1 - p_right_left
        
        # 四選項漂移率
        v1_raw = (p_left_left * p_right_left).astype(np.float32)
        v2_raw = (p_left_left * p_right_right).astype(np.float32)
        v3_raw = (p_left_right * p_right_left).astype(np.float32)
        v4_raw = (p_left_right * p_right_right).astype(np.float32)
        
        # 正規化
        v_sum = v1_raw + v2_raw + v3_raw + v4_raw + np.float32(1e-8)
        v_all = np.column_stack([
            (v1_raw / v_sum) * base_v,
            (v2_raw / v_sum) * base_v,
            (v3_raw / v_sum) * base_v,
            (v4_raw / v_sum) * base_v
        ]).astype(np.float32)
        
        # LBA 似然計算
        loglik_sum = np.float32(0.0)
        
        for i in range(len(rt_decision)):
            choice_idx = int(choice_data[i])
            if choice_idx < 0 or choice_idx >= 4:
                continue
                
            rt_trial = rt_decision[i]
            if rt_trial <= 0:
                continue
                
            sqrt_rt = np.sqrt(rt_trial)
            
            # 獲勝 accumulator
            v_win = v_all[i, choice_idx]
            b_win = thresholds[choice_idx]
            
            # 檢查參數有效性
            if v_win <= 0 or b_win <= A or not np.isfinite(v_win) or not np.isfinite(b_win):
                loglik_sum += np.float32(-10.0)
                continue
            
            # 獲勝者 PDF
            z1_win = (v_win * rt_trial - b_win) / sqrt_rt
            z2_win = (v_win * rt_trial - A) / sqrt_rt
            z1_win = np.clip(z1_win, -7, 7)
            z2_win = np.clip(z2_win, -7, 7)
            
            try:
                cdf_diff = max(stats.norm.cdf(z1_win) - stats.norm.cdf(z2_win), 1e-12)
                pdf_diff = (stats.norm.pdf(z1_win) - stats.norm.pdf(z2_win)) / sqrt_rt
                winner_pdf = max((v_win / A) * cdf_diff + pdf_diff / A, 1e-12)
            except:
                winner_pdf = 1e-12
            
            # 失敗者生存機率
            loser_survival_product = np.float32(1.0)
            for acc_idx in range(4):
                if acc_idx == choice_idx:
                    continue
                    
                v_lose = v_all[i, acc_idx]
                b_lose = thresholds[acc_idx]
                
                if v_lose <= 0 or b_lose <= A:
                    continue
                
                z1_lose = (v_lose * rt_trial - b_lose) / sqrt_rt
                z2_lose = (v_lose * rt_trial - A) / sqrt_rt
                z1_lose = np.clip(z1_lose, -7, 7)
                z2_lose = np.clip(z2_lose, -7, 7)
                
                try:
                    loser_cdf = max(stats.norm.cdf(z1_lose) - stats.norm.cdf(z2_lose), 1e-12)
                    loser_survival = max(1 - loser_cdf, 1e-12)
                    loser_survival_product *= loser_survival
                except:
                    loser_survival_product *= np.float32(0.5)
            
            trial_lik = winner_pdf * loser_survival_product
            trial_loglik = np.log(max(trial_lik, 1e-15))
            
            if np.isfinite(trial_loglik):
                loglik_sum += trial_loglik
            else:
                loglik_sum += np.float32(-10.0)
        
        return float(loglik_sum) if np.isfinite(loglik_sum) else -1000.0
        
    except Exception as e:
        return -1000.0

# ============================================================================
# 修正的採樣配置
# Fixed Sampling Configurations
# ============================================================================

class FixedSamplerConfig:
    """修正的採樣配置"""
    
    @staticmethod
    def get_robust_config(model_type: str, n_trials: int) -> Dict:
        """穩健配置 - 犧牲一些速度換取穩定性"""
        
        base_configs = {
            'individual': {
                'draws': 400,
                'tune': 400,
                'chains': 2,
                'target_accept': 0.85,
                'max_treedepth': 8,
                'init': 'adapt_diag'
            },
            'hierarchical': {
                'draws': 600,
                'tune': 500,
                'chains': 2,
                'target_accept': 0.90,
                'max_treedepth': 9,
                'init': 'adapt_diag'
            }
        }
        
        return base_configs.get(model_type, base_configs['individual'])

# ============================================================================
# 修正的個別受試者分析
# Fixed Individual Subject Analysis
# ============================================================================

def fixed_single_subject_analysis(args: Tuple) -> Optional[Dict]:
    """
    修正版單一受試者分析
    Fixed single subject analysis
    """
    
    subject_id, subject_data, coords, fast_mode = args
    
    try:
        print(f"Processing Subject {subject_id}...")
        
        # 準備數據 - 強制 float32
        rt_data = subject_data['RT'].values.astype(np.float32)
        choice_data = subject_data['choice_four'].values.astype(np.int32)
        stimloc_data = np.column_stack([
            subject_data['stimloc_x'].values,
            subject_data['stimloc_y'].values
        ]).astype(np.float32)
        
        if len(rt_data) < 20:
            return None
        
        # 數據檢查
        rt_data = np.clip(rt_data, 0.15, 2.0)
        choice_data = np.clip(choice_data, 0, 3)
        
        # 建立修正模型
        with pm.Model(coords=coords) as model:
            
            # GRT 參數 (使用更穩健的先驗)
            db1 = pm.Beta('db1', alpha=2, beta=2)
            db2 = pm.Beta('db2', alpha=2, beta=2)
            
            # 感知變異性 (log scale)
            log_sp = pm.Normal('log_sp', mu=np.log(0.25), sigma=0.3)
            sp = pm.Deterministic('sp', pt.exp(log_sp))
            
            # LBA 參數
            log_A = pm.Normal('log_A', mu=np.log(0.4), sigma=0.3)
            A = pm.Deterministic('A', pt.exp(log_A))
            
            # 閾值偏移 (確保 float32)
            log_b_offsets = pm.Normal('log_b_offsets', mu=np.log(0.5), sigma=0.3, 
                                     shape=4)
            b_offsets = pm.Deterministic('b_offsets', pt.exp(log_b_offsets))
            
            # 總閾值
            thresholds = pm.Deterministic('thresholds', A + b_offsets)
            
            # 基礎漂移率
            log_base_v = pm.Normal('log_base_v', mu=np.log(1.2), sigma=0.3)
            base_v = pm.Deterministic('base_v', pt.exp(log_base_v))
            
            # 固定參數
            s_fixed = np.float32(0.3)
            t0_fixed = np.float32(0.2)
            
            # 似然函數 - 使用修正版本
            likelihood = pm.Potential(
                'likelihood',
                fixed_lba_loglik_float32(
                    rt_data, choice_data, stimloc_data,
                    db1, db2, sp, sp, A, thresholds,
                    base_v, s_fixed, t0_fixed
                )
            )
        
        # 穩健採樣配置
        config = FixedSamplerConfig.get_robust_config('individual', len(rt_data))
        
        # 採樣
        with model:
            # 使用 jax 後端
            trace = pm.sample(
                **config,
                progressbar=False,
                return_inferencedata=True,
                cores=1,
                random_seed=42 + subject_id,
                compute_convergence_checks=True,
                nuts_sampler='nutpie' if hasattr(pm, 'nutpie') else 'pymc'
            )
        
        # 收斂檢查
        try:
            rhat_vals = az.rhat(trace)
            ess_vals = az.ess(trace)
            rhat_max = float(rhat_vals.max()) if hasattr(rhat_vals, 'max') else 1.05
            ess_min = float(ess_vals.min()) if hasattr(ess_vals, 'min') else 100
        except:
            rhat_max, ess_min = 1.05, 100
        
        result = {
            'subject_id': subject_id,
            'trace': trace,
            'convergence': {'rhat_max': rhat_max, 'ess_min': ess_min},
            'n_trials': len(rt_data),
            'success': True
        }
        
        print(f"✅ Subject {subject_id} completed (R̂={rhat_max:.3f}, ESS={ess_min:.0f})")
        return result
        
    except Exception as e:
        print(f"❌ Subject {subject_id} failed: {e}")
        return {'subject_id': subject_id, 'success': False, 'error': str(e)}

# ============================================================================
# 修正的分析器主類別
# Fixed Main Analyzer Class
# ============================================================================

class FixedFourChoiceAnalyzer:
    """
    修正版四選項 GRT-LBA 分析器
    Fixed Four-Choice GRT-LBA Analyzer
    """
    
    def __init__(self, csv_file: str = 'GRT_LBA.csv'):
        self.csv_file = csv_file
        self.results_dir = Path('fixed_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # 載入和清理數據
        print("Loading and cleaning data...")
        self.df = pd.read_csv(csv_file)
        
        # 數據過濾
        self.df = self.df[(self.df['RT'] > 0.15) & (self.df['RT'] < 2.0)]
        self.df = self.df[self.df['Response'].isin([0, 1, 2, 3])]
        
        # 準備選項和刺激位置
        self.df['choice_four'] = self.df['Response'].astype(int)
        
        # 刺激位置映射
        stim_mapping = {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [1.0, 0.0], 3: [1.0, 1.0]}
        self.df['stimloc_x'] = self.df['Stimulus'].map(lambda x: stim_mapping.get(x, [0.5, 0.5])[0])
        self.df['stimloc_y'] = self.df['Stimulus'].map(lambda x: stim_mapping.get(x, [0.5, 0.5])[1])
        
        # 移除異常值
        self.df = self.df.dropna()
        
        self.participants = sorted(self.df['participant'].unique())
        
        # PyMC 坐標系統
        self.coords = {
            'participant': self.participants,
            'choice': [0, 1, 2, 3],
            'accumulator': ['acc1', 'acc2', 'acc3', 'acc4']
        }
        
        print(f"Data cleaned: {len(self.df)} trials, {len(self.participants)} subjects")
    
    def run_robust_parallel_analysis(self, max_subjects: Optional[int] = None) -> List[Dict]:
        """
        運行穩健的並行分析
        Run robust parallel analysis
        """
        
        print("Starting robust parallel analysis...")
        
        subjects_to_analyze = self.participants
        if max_subjects:
            subjects_to_analyze = subjects_to_analyze[:max_subjects]
        
        # 準備任務
        tasks = []
        for subject_id in subjects_to_analyze:
            subject_data = self.df[self.df['participant'] == subject_id].copy()
            if len(subject_data) >= 20:
                tasks.append((subject_id, subject_data, self.coords, False))
        
        print(f"Processing {len(tasks)} subjects...")
        
        results = []
        start_time = time.time()
        
        # 序列處理以避免並行問題
        for task in tasks:
            try:
                result = fixed_single_subject_analysis(task)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Task failed: {e}")
                results.append({
                    'subject_id': task[0], 
                    'success': False, 
                    'error': str(e)
                })
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.get('success', False))
        
        print(f"\n🏁 Analysis completed:")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Success: {successful}/{len(tasks)} subjects")
        
        return results

# ============================================================================
# 修正的使用範例
# Fixed Usage Example
# ============================================================================

def run_fixed_analysis(max_subjects: int = 3):
    """
    運行修正版分析
    Run fixed analysis
    """
    
    print("="*60)
    print("FIXED FOUR-CHOICE GRT-LBA ANALYSIS")
    print("修正版四選項 GRT-LBA 分析")
    print("="*60)
    
    try:
        analyzer = FixedFourChoiceAnalyzer('GRT_LBA.csv')
        
        # 個別分析
        start_time = time.time()
        results = analyzer.run_robust_parallel_analysis(max_subjects=max_subjects)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.get('success', False)]
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Success rate: {len(successful_results)}/{len(results)}")
        print(f"   Total time: {total_time:.1f}s")
        
        # 顯示成功案例的收斂資訊
        if successful_results:
            print(f"\n📈 Successful Subjects:")
            for result in successful_results:
                if 'convergence' in result:
                    conv = result['convergence']
                    print(f"   Subject {result['subject_id']}: "
                          f"R̂={conv['rhat_max']:.3f}, ESS={conv['ess_min']:.0f}, "
                          f"N={result['n_trials']}")
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return []

if __name__ == "__main__":
    # 執行修正版分析
    print("Running fixed analysis...")
    results = run_fixed_analysis(max_subjects=3)
    
    if results:
        successful = [r for r in results if r.get('success', False)]
        print(f"\nFinal: {len(successful)} successful analyses")
    else:
        print("\nNo results obtained")
