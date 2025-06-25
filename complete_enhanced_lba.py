# complete_enhanced_lba.py - 完整版增強LBA分析器，自動過濾低品質數據

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class LBAConfig:
    """LBA模型配置類"""
    # 固定參數
    threshold: float = 0.8
    start_var: float = 0.2
    ndt: float = 0.15
    noise: float = 0.3
    
    # MCMC配置
    draws: int = 600
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.95
    max_treedepth: int = 12
    
    # 數據過濾配置
    min_accuracy: float = 0.5      # 最低正確率門檻 50%
    min_trials: int = 50           # 最低試驗數
    max_rt: float = 2.5           # 最大反應時間
    min_rt: float = 0.2           # 最小反應時間
    
    def get_mcmc_config(self) -> Dict:
        return {
            'draws': self.draws,
            'tune': self.tune,
            'chains': self.chains,
            'cores': self.cores,
            'target_accept': self.target_accept,
            'max_treedepth': self.max_treedepth,
            'init': 'adapt_diag',
            'progressbar': True,
            'return_inferencedata': True,
            'random_seed': [42, 43, 44, 45]
        }

class DataFilter:
    """數據過濾器"""
    
    def __init__(self, config: LBAConfig):
        self.config = config
        self.filtered_subjects = []
    
    def filter_subjects(self, df: pd.DataFrame) -> Tuple[List[int], List[Dict]]:
        """過濾受試者，返回合格的受試者ID列表和過濾詳情"""
        
        print(f"🔍 開始數據過濾...")
        print(f"   過濾條件: 正確率≥{self.config.min_accuracy:.0%}, 試驗數≥{self.config.min_trials}")
        
        valid_subjects = []
        filtered_details = []
        
        for subject_id in df['participant'].unique():
            subject_df = df[df['participant'] == subject_id].copy()
            
            # 計算基本統計
            n_trials = len(subject_df)
            
            # 計算正確率
            accuracy = self._calculate_accuracy(subject_df)
            
            # 檢查RT範圍
            valid_rt_ratio = self._check_rt_range(subject_df)
            
            # 應用過濾規則
            filter_result = self._apply_filters(subject_id, n_trials, accuracy, valid_rt_ratio)
            
            if filter_result['passed']:
                valid_subjects.append(subject_id)
                print(f"   ✅ 受試者 {subject_id}: {n_trials} trials, 正確率 {accuracy:.1%}")
            else:
                filtered_details.append(filter_result)
                print(f"   ❌ 受試者 {subject_id}: {filter_result['reason']}")
        
        print(f"\n📊 過濾結果:")
        print(f"   總受試者: {df['participant'].nunique()}")
        print(f"   合格受試者: {len(valid_subjects)}")
        print(f"   過濾受試者: {len(filtered_details)}")
        
        return valid_subjects, filtered_details
    
    def _calculate_accuracy(self, subject_df: pd.DataFrame) -> float:
        """計算受試者整體正確率"""
        
        # 映射刺激和選擇
        stimulus_map = {0: [1, 0], 1: [1, 1], 2: [0, 0], 3: [0, 1]}
        choice_map = {0: [1, 0], 1: [1, 1], 2: [0, 0], 3: [0, 1]}
        
        correct_count = 0
        total_count = len(subject_df)
        
        for _, row in subject_df.iterrows():
            stimulus = int(row['Stimulus'])
            choice = int(row['Response'])
            
            stim_left, stim_right = stimulus_map[stimulus]
            choice_left, choice_right = choice_map[choice]
            
            # 兩邊都對才算正確
            if stim_left == choice_left and stim_right == choice_right:
                correct_count += 1
        
        return correct_count / total_count if total_count > 0 else 0.0
    
    def _check_rt_range(self, subject_df: pd.DataFrame) -> float:
        """檢查RT範圍的有效性"""
        
        rt_values = subject_df['RT'].values
        valid_rt = np.sum((rt_values >= self.config.min_rt) & (rt_values <= self.config.max_rt))
        return valid_rt / len(rt_values) if len(rt_values) > 0 else 0.0
    
    def _apply_filters(self, subject_id: int, n_trials: int, accuracy: float, valid_rt_ratio: float) -> Dict:
        """應用過濾規則"""
        
        # 檢查正確率
        if accuracy < self.config.min_accuracy:
            return {
                'subject_id': subject_id,
                'passed': False,
                'reason': f'正確率過低 ({accuracy:.1%} < {self.config.min_accuracy:.0%})',
                'accuracy': accuracy,
                'n_trials': n_trials
            }
        
        # 檢查試驗數
        if n_trials < self.config.min_trials:
            return {
                'subject_id': subject_id,
                'passed': False,
                'reason': f'試驗數不足 ({n_trials} < {self.config.min_trials})',
                'accuracy': accuracy,
                'n_trials': n_trials
            }
        
        # 檢查RT範圍
        if valid_rt_ratio < 0.9:  # 至少90%的RT在合理範圍內
            return {
                'subject_id': subject_id,
                'passed': False,
                'reason': f'RT範圍異常 (有效比例: {valid_rt_ratio:.1%})',
                'accuracy': accuracy,
                'n_trials': n_trials
            }
        
        return {
            'subject_id': subject_id,
            'passed': True,
            'reason': '通過所有過濾條件',
            'accuracy': accuracy,
            'n_trials': n_trials
        }

class SimpleLBAModel:
    """簡化的LBA模型，專注於穩定性"""
    
    def __init__(self, config: LBAConfig, model_type: str = 'minimal'):
        self.config = config
        self.model_type = model_type
        self._model = None
    
    def build_model(self, data: Dict) -> pm.Model:
        """構建PyMC模型"""
        
        if self._model is not None:
            return self._model
        
        print(f"🔧 構建 {self.model_type} 模型")
        
        with pm.Model() as model:
            if self.model_type == 'minimal':
                params = self._build_minimal_params()
            else:  # constrained
                params = self._build_constrained_params()
            
            # 計算似然
            left_ll = self._compute_lba_likelihood(
                data['left_choices'], data['left_stimuli'], data['rt'],
                params['left_drift_match'], params['left_drift_mismatch']
            )
            
            right_ll = self._compute_lba_likelihood(
                data['right_choices'], data['right_stimuli'], data['rt'],
                params['right_drift_match'], params['right_drift_mismatch']
            )
            
            pm.Potential('left_likelihood', left_ll)
            pm.Potential('right_likelihood', right_ll)
        
        self._model = model
        return model
    
    def _build_minimal_params(self) -> Dict:
        """構建最小參數集"""
        
        left_drift_match = pm.Gamma('left_drift_match', alpha=2.0, beta=1.5)
        left_drift_mismatch = pm.Gamma('left_drift_mismatch', alpha=1.5, beta=3.0)
        right_drift_match = pm.Gamma('right_drift_match', alpha=2.0, beta=1.5)
        right_drift_mismatch = pm.Gamma('right_drift_mismatch', alpha=1.5, beta=3.0)
        
        # 軟約束
        pm.Potential('left_ordering', 
            pm.math.log(1 + pm.math.exp(3.0 * (left_drift_match - left_drift_mismatch - 0.2))))
        pm.Potential('right_ordering',
            pm.math.log(1 + pm.math.exp(3.0 * (right_drift_match - right_drift_mismatch - 0.2))))
        
        return {
            'left_drift_match': left_drift_match,
            'left_drift_mismatch': left_drift_mismatch,
            'right_drift_match': right_drift_match,
            'right_drift_mismatch': right_drift_mismatch
        }
    
    def _build_constrained_params(self) -> Dict:
        """構建強約束參數"""
        
        # 對數空間參數
        log_left_match = pm.Normal('log_left_match', mu=0.4, sigma=0.4)
        log_left_mismatch = pm.Normal('log_left_mismatch', mu=-0.6, sigma=0.3)
        log_right_match = pm.Normal('log_right_match', mu=0.4, sigma=0.4)
        log_right_mismatch = pm.Normal('log_right_mismatch', mu=-0.6, sigma=0.3)
        
        # 變換到正值並強制順序
        left_drift_mismatch_base = pm.math.exp(log_left_mismatch)
        right_drift_mismatch_base = pm.math.exp(log_right_mismatch)
        
        left_drift_match_base = left_drift_mismatch_base + pm.math.exp(log_left_match) + 0.15
        right_drift_match_base = right_drift_mismatch_base + pm.math.exp(log_right_match) + 0.15
        
        # 對稱性約束
        symmetry_weight = 0.3
        mean_match = (left_drift_match_base + right_drift_match_base) / 2
        mean_mismatch = (left_drift_mismatch_base + right_drift_mismatch_base) / 2
        
        left_drift_match = pm.Deterministic('left_drift_match',
            symmetry_weight * mean_match + (1 - symmetry_weight) * left_drift_match_base)
        left_drift_mismatch = pm.Deterministic('left_drift_mismatch',
            symmetry_weight * mean_mismatch + (1 - symmetry_weight) * left_drift_mismatch_base)
        
        right_drift_match = pm.Deterministic('right_drift_match',
            symmetry_weight * mean_match + (1 - symmetry_weight) * right_drift_match_base)
        right_drift_mismatch = pm.Deterministic('right_drift_mismatch',
            symmetry_weight * mean_mismatch + (1 - symmetry_weight) * right_drift_mismatch_base)
        
        return {
            'left_drift_match': left_drift_match,
            'left_drift_mismatch': left_drift_mismatch,
            'right_drift_match': right_drift_match,
            'right_drift_mismatch': right_drift_mismatch
        }
    
    def _compute_lba_likelihood(self, decisions, stimuli, rt, drift_match, drift_mismatch):
        """計算LBA似然"""
        
        from pytensor.tensor import erf
        
        # 固定參數
        threshold = self.config.threshold
        start_var = self.config.start_var
        ndt = self.config.ndt
        noise = self.config.noise
        
        # 參數安全化
        drift_match_safe = pm.math.clip(drift_match, 0.12, 6.0)
        drift_mismatch_safe = pm.math.clip(drift_mismatch, 0.08, 4.0)
        
        # 決策時間
        decision_time = pm.math.clip(rt - ndt, 0.05, 3.0)
        
        # 匹配性判斷
        stimulus_match = pm.math.eq(decisions, stimuli)
        
        # 漂移率分配
        v_chosen = pm.math.where(stimulus_match, drift_match_safe, drift_mismatch_safe)
        v_unchosen = pm.math.where(stimulus_match, drift_mismatch_safe, drift_match_safe)
        
        # LBA密度計算
        sqrt_t = pm.math.sqrt(decision_time)
        
        z1_chosen = pm.math.clip(
            (v_chosen * decision_time - threshold) / (noise * sqrt_t), -4.0, 4.0)
        z2_chosen = pm.math.clip(
            (v_chosen * decision_time - start_var) / (noise * sqrt_t), -4.0, 4.0)
        z1_unchosen = pm.math.clip(
            (v_unchosen * decision_time - threshold) / (noise * sqrt_t), -4.0, 4.0)
        
        # 正態函數
        def safe_normal_cdf(x):
            x_safe = pm.math.clip(x, -4.0, 4.0)
            return 0.5 * (1 + erf(x_safe / pm.math.sqrt(2)))
        
        def safe_normal_pdf(x):
            x_safe = pm.math.clip(x, -4.0, 4.0)
            return pm.math.exp(-0.5 * x_safe**2) / pm.math.sqrt(2 * np.pi)
        
        # Winner密度
        chosen_cdf_term = safe_normal_cdf(z1_chosen) - safe_normal_cdf(z2_chosen)
        chosen_pdf_term = (safe_normal_pdf(z1_chosen) - safe_normal_pdf(z2_chosen)) / (noise * sqrt_t)
        chosen_cdf_term = pm.math.maximum(chosen_cdf_term, 1e-8)
        
        chosen_likelihood = pm.math.maximum(
            (v_chosen / start_var) * chosen_cdf_term + chosen_pdf_term / start_var, 1e-8)
        
        # Loser存活
        unchosen_survival = pm.math.maximum(1 - safe_normal_cdf(z1_unchosen), 1e-8)
        
        # 聯合似然
        joint_likelihood = chosen_likelihood * unchosen_survival
        joint_likelihood = pm.math.maximum(joint_likelihood, 1e-10)
        log_likelihood = pm.math.log(joint_likelihood)
        log_likelihood_safe = pm.math.clip(log_likelihood, -40.0, 8.0)
        
        return pm.math.sum(log_likelihood_safe)
    
    def sample(self, data: Dict, **mcmc_kwargs) -> az.InferenceData:
        """執行MCMC採樣"""
        
        model = self.build_model(data)
        
        mcmc_config = self.config.get_mcmc_config()
        mcmc_config.update(mcmc_kwargs)
        
        with model:
            trace = pm.sample(**mcmc_config)
        
        return trace

class EnhancedLBAAnalyzer:
    """增強版LBA分析器"""
    
    def __init__(self, config: LBAConfig = None):
        self.config = config or LBAConfig()
        self.data_filter = DataFilter(self.config)
        self.models = {}
        
        print(f"✅ 增強版LBA分析器初始化完成")
        print(f"   自動過濾: 正確率≥{self.config.min_accuracy:.0%}")
    
    def prepare_subject_data(self, df: pd.DataFrame, subject_id: int) -> Dict:
        """準備單個受試者數據"""
        
        subject_df = df[df['participant'] == subject_id].copy()
        
        # 數據映射
        stimulus_map = {0: [1, 0], 1: [1, 1], 2: [0, 0], 3: [0, 1]}
        choice_map = {0: [1, 0], 1: [1, 1], 2: [0, 0], 3: [0, 1]}
        
        left_stim, right_stim = [], []
        left_choice, right_choice = [], []
        
        for _, row in subject_df.iterrows():
            s, c = int(row['Stimulus']), int(row['Response'])
            left_stim.append(stimulus_map[s][0])
            right_stim.append(stimulus_map[s][1])
            left_choice.append(choice_map[c][0])
            right_choice.append(choice_map[c][1])
        
        return {
            'subject_id': subject_id,
            'n_trials': len(subject_df),
            'rt': subject_df['RT'].values,
            'left_stimuli': np.array(left_stim),
            'right_stimuli': np.array(right_stim),
            'left_choices': np.array(left_choice),
            'right_choices': np.array(right_choice)
        }
    
    def fit_subject(self, data: Dict, model_type: str = 'minimal') -> Dict:
        """擬合單個受試者"""
        
        print(f"\n🎯 擬合受試者 {data['subject_id']} - 模型: {model_type}")
        
        # 獲取或創建模型
        model_key = f"{model_type}_{data['subject_id']}"
        if model_key not in self.models:
            self.models[model_key] = SimpleLBAModel(self.config, model_type)
        
        model = self.models[model_key]
        
        start_time = time.time()
        
        try:
            # 執行採樣
            trace = model.sample(data)
            sampling_time = time.time() - start_time
            
            # 收斂診斷
            convergence = self._diagnose_convergence(trace)
            
            # 提取結果
            results = self._extract_results(trace, data, model_type, convergence, sampling_time)
            
            return results
            
        except Exception as e:
            print(f"   ❌ 擬合失敗: {e}")
            return {
                'success': False,
                'subject_id': data['subject_id'],
                'model_type': model_type,
                'error': str(e)
            }
    
    def batch_analysis(self, df: pd.DataFrame, max_subjects: int = 10, 
                      model_type: str = 'minimal') -> Dict:
        """批次分析 - 自動過濾低品質數據"""
        
        print(f"\n🚀 批次分析開始")
        print(f"   模型類型: {model_type}")
        print(f"   最大受試者: {max_subjects}")
        
        # 第一步：過濾數據
        valid_subjects, filtered_details = self.data_filter.filter_subjects(df)
        
        if len(valid_subjects) == 0:
            print("❌ 沒有符合條件的受試者")
            return {
                'success': False,
                'error': '沒有符合條件的受試者',
                'filtered_details': filtered_details
            }
        
        # 第二步：選擇要分析的受試者
        selected_subjects = valid_subjects[:max_subjects]
        
        print(f"\n📊 開始分析 {len(selected_subjects)} 個受試者:")
        print(f"   選擇的受試者: {selected_subjects}")
        
        # 第三步：批次擬合
        results = []
        successful = 0
        converged = 0
        
        for i, subject_id in enumerate(selected_subjects, 1):
            print(f"\n📍 進度 {i}/{len(selected_subjects)}: 受試者 {subject_id}")
            
            try:
                # 準備數據
                data = self.prepare_subject_data(df, subject_id)
                
                # 執行擬合
                result = self.fit_subject(data, model_type)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    if result['converged']:
                        converged += 1
                        print(f"   ✅ 成功收斂")
                    else:
                        print(f"   ⚠️ 成功但未完全收斂")
                
            except Exception as e:
                print(f"   ❌ 受試者 {subject_id} 失敗: {e}")
                results.append({
                    'success': False,
                    'subject_id': subject_id,
                    'error': str(e)
                })
        
        # 生成報告
        return self._generate_batch_report(results, successful, converged, filtered_details)
    
    def _diagnose_convergence(self, trace) -> Dict:
        """收斂診斷"""
        try:
            summary = az.summary(trace)
            max_rhat = summary['r_hat'].max() if 'r_hat' in summary.columns else np.nan
            min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else np.nan
            
            n_divergent = 0
            if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
                n_divergent = int(trace.sample_stats.diverging.sum())
            
            converged = (max_rhat < 1.05 and min_ess > 200 and n_divergent == 0)
            
            status = "✅ 收斂良好" if converged else "⚠️ 收斂警告"
            print(f"   {status}: R̂={max_rhat:.3f}, ESS={min_ess:.0f}, 發散={n_divergent}")
            
            return {
                'converged': converged,
                'max_rhat': max_rhat,
                'min_ess': min_ess,
                'n_divergent': n_divergent
            }
        except Exception as e:
            return {'converged': False, 'error': str(e)}
    
    def _extract_results(self, trace, data: Dict, model_type: str, 
                        convergence: Dict, sampling_time: float) -> Dict:
        """提取結果"""
        try:
            summary = az.summary(trace)
            
            # 參數估計
            param_estimates = {}
            for param in ['left_drift_match', 'left_drift_mismatch', 'right_drift_match', 'right_drift_mismatch']:
                if param in summary.index:
                    param_estimates[param] = float(summary.loc[param, 'mean'])
                else:
                    param_estimates[param] = np.nan
            
            # 衍生指標
            left_discrimination = param_estimates['left_drift_match'] - param_estimates['left_drift_mismatch']
            right_discrimination = param_estimates['right_drift_match'] - param_estimates['right_drift_mismatch']
            processing_asymmetry = abs(param_estimates['left_drift_match'] - param_estimates['right_drift_match'])
            discrimination_asymmetry = abs(left_discrimination - right_discrimination)
            
            results = {
                'success': True,
                'model_type': model_type,
                'subject_id': data['subject_id'],
                'converged': convergence['converged'],
                'convergence_diagnostics': convergence,
                'sampling_time_minutes': sampling_time / 60,
                'param_estimates': param_estimates,
                'left_discrimination': left_discrimination,
                'right_discrimination': right_discrimination,
                'processing_asymmetry': processing_asymmetry,
                'discrimination_asymmetry': discrimination_asymmetry,
                'symmetry_supported': (processing_asymmetry < 0.3 and discrimination_asymmetry < 0.4)
            }
            
            # 打印簡要結果
            status = "✅ 收斂" if results['converged'] else "⚠️ 警告"
            print(f"   {status}: 左辨別={left_discrimination:.3f}, 右辨別={right_discrimination:.3f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ 結果提取失敗: {e}")
            return {'success': False, 'extraction_error': str(e)}
    
    def _generate_batch_report(self, results: List[Dict], successful: int, 
                              converged: int, filtered_details: List[Dict]) -> Dict:
        """生成批次分析報告"""
        
        print(f"\n📊 批次分析報告")
        print("=" * 50)
        
        total_analyzed = len(results)
        
        print(f"分析統計:")
        print(f"   分析受試者: {total_analyzed}")
        print(f"   成功擬合: {successful} ({successful/total_analyzed*100:.1f}%)")
        if successful > 0:
            print(f"   完全收斂: {converged} ({converged/successful*100:.1f}%)")
        
        print(f"\n過濾統計:")
        print(f"   過濾受試者: {len(filtered_details)}")
        
        # 過濾原因統計
        filter_reasons = {}
        for detail in filtered_details:
            reason = detail['reason']
            if '正確率過低' in reason:
                filter_reasons['正確率過低'] = filter_reasons.get('正確率過低', 0) + 1
            elif '試驗數不足' in reason:
                filter_reasons['試驗數不足'] = filter_reasons.get('試驗數不足', 0) + 1
            elif 'RT範圍異常' in reason:
                filter_reasons['RT範圍異常'] = filter_reasons.get('RT範圍異常', 0) + 1
        
        for reason, count in filter_reasons.items():
            print(f"     {reason}: {count} 位")
        
        # 對稱性統計
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            symmetry_count = sum(1 for r in successful_results if r.get('symmetry_supported', False))
            print(f"\n對稱性分析:")
            print(f"   支持對稱性: {symmetry_count}/{len(successful_results)} ({symmetry_count/len(successful_results)*100:.1f}%)")
        
        return {
            'success': True,
            'total_analyzed': total_analyzed,
            'successful': successful,
            'converged': converged,
            'filtered_count': len(filtered_details),
            'filter_reasons': filter_reasons,
            'results': results,
            'filtered_details': filtered_details
        }

def run_enhanced_analysis():
    """運行增強版分析"""
    
    print("🚀 增強版LBA分析 - 自動過濾低品質數據")
    print("=" * 60)
    
    # 詢問參數
    csv_file = input("請輸入CSV檔案路徑 (或按Enter使用預設 'GRT_LBA.csv'): ").strip()
    if not csv_file:
        csv_file = 'GRT_LBA.csv'
    
    # 詢問過濾門檻
    min_acc_input = input("最低正確率門檻 (按Enter使用50%): ").strip()
    min_accuracy = float(min_acc_input) / 100 if min_acc_input else 0.5
    
    # 詢問受試者數量
    max_subjects_input = input("最大分析受試者數 (按Enter使用10): ").strip()
    max_subjects = int(max_subjects_input) if max_subjects_input else 10
    
    # 詢問模型類型
    print("\n選擇模型類型:")
    print("1. minimal - 最小參數集")
    print("2. constrained - 強約束參數化")
    model_choice = input("請選擇 (1-2): ").strip()
    model_type = 'minimal' if model_choice == '1' else 'constrained'
    
    try:
        # 載入數據
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.2) & (df['RT'] <= 2.5)]
        
        print(f"\n✅ 數據載入成功:")
        print(f"   總試驗: {len(df)}")
        print(f"   總受試者: {df['participant'].nunique()}")
        
        # 創建分析器
        config = LBAConfig(
            min_accuracy=min_accuracy,
            draws=500,
            tune=800,
            target_accept=0.93
        )
        analyzer = EnhancedLBAAnalyzer(config)
        
        # 執行批次分析
        batch_result = analyzer.batch_analysis(df, max_subjects, model_type)
        
        if batch_result['success']:
            print(f"\n🎉 分析完成!")
            print(f"   過濾門檻: 正確率≥{min_accuracy:.0%}")
            print(f"   使用模型: {model_type}")
            print(f"   分析結果: 成功 {batch_result['successful']}/{batch_result['total_analyzed']}")
        else:
            print(f"❌ 分析失敗: {batch_result.get('error', 'Unknown error')}")
        
        return batch_result
        
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {csv_file}")
        print("💡 請確保檔案路徑正確")
        return None
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_single_subject_analysis():
    """運行單一受試者分析"""
    
    print("🎯 單一受試者分析")
    print("=" * 40)
    
    # 詢問參數
    csv_file = input("請輸入CSV檔案路徑 (或按Enter使用預設): ").strip() or 'GRT_LBA.csv'
    subject_input = input("請輸入受試者ID (或按Enter自動選擇): ").strip()
    subject_id = int(subject_input) if subject_input else None
    
    # 詢問模型類型
    print("選擇模型類型:")
    print("1. minimal - 最小參數集")
    print("2. constrained - 強約束參數化")
    model_choice = input("請選擇 (1-2): ").strip()
    model_type = 'minimal' if model_choice == '1' else 'constrained'
    
    try:
        # 載入數據
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.2) & (df['RT'] <= 2.5)]
        
        # 創建分析器
        config = LBAConfig()
        analyzer = EnhancedLBAAnalyzer(config)
        
        # 如果沒有指定受試者，先過濾數據找合適的
        if subject_id is None:
            valid_subjects, _ = analyzer.data_filter.filter_subjects(df)
            if not valid_subjects:
                print("❌ 沒有符合條件的受試者")
                return None
            subject_id = valid_subjects[0]
            print(f"✅ 自動選擇受試者: {subject_id}")
        
        # 準備數據
        data = analyzer.prepare_subject_data(df, subject_id)
        
        # 檢查該受試者是否符合條件
        subject_df = df[df['participant'] == subject_id]
        accuracy = analyzer.data_filter._calculate_accuracy(subject_df)
        
        if accuracy < config.min_accuracy:
            print(f"❌ 受試者 {subject_id} 正確率過低: {accuracy:.1%} < {config.min_accuracy:.0%}")
            return None
        
        print(f"✅ 受試者 {subject_id} 符合條件: 正確率 {accuracy:.1%}")
        
        # 執行分析
        result = analyzer.fit_subject(data, model_type)
        
        if result['success']:
            print(f"\n🎉 單一受試者分析完成!")
            print(f"   模型類型: {result['model_type']}")
            print(f"   收斂狀態: {'✅ 收斂' if result['converged'] else '⚠️ 警告'}")
            print(f"   採樣時間: {result['sampling_time_minutes']:.1f} 分鐘")
        else:
            print(f"❌ 分析失敗: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        return None

def show_filter_demo():
    """演示過濾功能"""
    
    print("📊 數據過濾功能演示")
    print("=" * 40)
    
    csv_file = input("請輸入CSV檔案路徑 (或按Enter使用預設): ").strip() or 'GRT_LBA.csv'
    
    try:
        # 載入數據
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
        df = df[(df['RT'] >= 0.2) & (df['RT'] <= 2.5)]
        
        print(f"✅ 數據載入: {len(df)} trials, {df['participant'].nunique()} 受試者")
        
        # 創建過濾器
        config = LBAConfig(min_accuracy=0.5)  # 50%正確率門檻
        data_filter = DataFilter(config)
        
        # 執行過濾
        valid_subjects, filtered_details = data_filter.filter_subjects(df)
        
        print(f"\n📈 過濾結果摘要:")
        print(f"   符合條件: {len(valid_subjects)} 位受試者")
        print(f"   被過濾: {len(filtered_details)} 位受試者")
        
        # 顯示被過濾的受試者詳情
        if filtered_details:
            print(f"\n❌ 被過濾的受試者詳情:")
            for detail in filtered_details[:10]:  # 最多顯示10個
                print(f"   受試者 {detail['subject_id']}: {detail['reason']}")
            
            if len(filtered_details) > 10:
                print(f"   ... 還有 {len(filtered_details) - 10} 位")
        
        # 顯示符合條件的受試者
        if valid_subjects:
            print(f"\n✅ 符合條件的受試者 (前10位): {valid_subjects[:10]}")
            if len(valid_subjects) > 10:
                print(f"   ... 還有 {len(valid_subjects) - 10} 位")
        
        return {'valid_subjects': valid_subjects, 'filtered_details': filtered_details}
        
    except Exception as e:
        print(f"❌ 演示失敗: {e}")
        return None

if __name__ == "__main__":
    print("🎯 增強版LBA分析選項:")
    print("1. 批次分析 (自動過濾 + 混合模型)")
    print("2. 單一受試者分析")
    print("3. 數據過濾功能演示")
    
    try:
        choice = input("\n請選擇 (1-3): ").strip()
        
        if choice == '1':
            result = run_enhanced_analysis()
            
        elif choice == '2':
            result = run_single_subject_analysis()
            
        elif choice == '3':
            result = show_filter_demo()
            
        else:
            print("無效選擇")
            
    except KeyboardInterrupt:
        print("\n⏹️ 分析被中斷")
    except Exception as e:
        print(f"\n💥 錯誤: {e}")

# ============================================================================
# 使用說明
# ============================================================================

"""
🎯 增強版LBA分析器功能說明：

1. **自動過濾低品質數據** ✅
   - 正確率 < 50% 自動過濾
   - 試驗數 < 50 自動過濾
   - RT範圍異常自動過濾
   - 詳細記錄過濾原因

2. **兩種模型可選** ✅
   - minimal: 最小參數集 (4個漂移率參數)
   - constrained: 強約束參數化 (對數變換 + 對稱性約束)

3. **批次分析功能** ✅
   - 自動過濾並分析多個受試者
   - 詳細的分析報告
   - 收斂統計和對稱性分析

4. **單一受試者分析** ✅
   - 針對特定受試者的詳細分析
   - 自動檢查數據品質

使用流程：
1. 選擇選項1進行批次分析
2. 輸入CSV檔案路徑
3. 設定正確率門檻 (建議50%)
4. 選擇模型類型 (建議先試minimal)
5. 查看分析結果

預期改善：
- 自動排除低品質數據，提高整體分析品質
- 兩種模型策略應對不同收斂情況  
- 詳細報告幫助理解數據特性
"""
