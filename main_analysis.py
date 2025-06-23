# -*- coding: utf-8 -*-
"""
main_analysis.py - 主分析流程
Sequential Processing LBA - Main Analysis Pipeline

功能：
- 整合完整的分析流程
- 支援不同的分析選項
- 提供模型比較功能
- 生成分析報告
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional
from data_utils import DataProcessor
from model_fitting import SequentialModelFitter

def main_sequential_analysis(csv_file='GRT_LBA.csv', max_subjects=None, 
                           first_side='left', time_split_ratio=0.6,
                           mcmc_config=None, save_results=True):
    """
    主要序列分析流程
    
    Args:
        csv_file: 資料檔案路徑
        max_subjects: 最大受試者數限制
        first_side: 首先處理的通道 ('left' 或 'right')
        time_split_ratio: 時間分割比例
        mcmc_config: MCMC配置字典
        save_results: 是否儲存結果
        
    Returns:
        dict: 完整分析結果
    """
    
    print("🚀 序列處理LBA分析")
    print("=" * 60)
    print(f"📂 資料檔案: {csv_file}")
    print(f"🧠 處理順序: {first_side} 先處理")
    print(f"⏰ 時間分割: {time_split_ratio:.1%} / {1-time_split_ratio:.1%}")
    if max_subjects:
        print(f"👥 受試者限制: {max_subjects}")
    print("=" * 60)
    
    analysis_start_time = time.time()
    
    try:
        # ========================================
        # 1. 載入和預處理資料
        # ========================================
        
        print("\n📂 階段1: 資料載入和預處理")
        print("-" * 40)
        
        processor = DataProcessor()
        data_df = processor.load_and_clean_data(csv_file)
        
        # 資料品質檢查
        quality_report = processor.validate_data_quality(data_df)
        
        # 受試者選擇
        participants = sorted(data_df['participant'].unique())
        if max_subjects:
            participants = participants[:max_subjects]
        
        print(f"✅ 資料預處理完成")
        print(f"   總試驗數: {len(data_df)}")
        print(f"   分析受試者: {len(participants)}")
        
        # ========================================
        # 2. 準備受試者資料
        # ========================================
        
        print(f"\n🔄 階段2: 準備受試者資料")
        print("-" * 40)
        
        subjects_data = []
        excluded_subjects = []
        
        for subject_id in participants:
            try:
                subject_data = processor.extract_subject_data(data_df, subject_id)
                
                # 檢查資料充足性
                if subject_data['n_trials'] >= 50:
                    subjects_data.append(subject_data)
                else:
                    excluded_subjects.append({
                        'subject_id': subject_id,
                        'reason': f"資料不足 ({subject_data['n_trials']} < 50 trials)"
                    })
            except Exception as e:
                excluded_subjects.append({
                    'subject_id': subject_id,
                    'reason': f"資料提取失敗: {e}"
                })
        
        print(f"✅ 受試者資料準備完成")
        print(f"   有效受試者: {len(subjects_data)}")
        if excluded_subjects:
            print(f"   排除受試者: {len(excluded_subjects)}")
            for excluded in excluded_subjects[:3]:  # 只顯示前3個
                print(f"     受試者 {excluded['subject_id']}: {excluded['reason']}")
            if len(excluded_subjects) > 3:
                print(f"     ... 還有 {len(excluded_subjects)-3} 個")
        
        if len(subjects_data) == 0:
            raise ValueError("沒有有效的受試者資料可以分析")
        
        # ========================================
        # 3. 模型擬合
        # ========================================
        
        print(f"\n🎯 階段3: 序列LBA模型擬合")
        print("-" * 40)
        
        # 創建擬合器
        fitter = SequentialModelFitter(
            first_side=first_side, 
            time_split_ratio=time_split_ratio,
            mcmc_config=mcmc_config
        )
        
        # 批次擬合
        fitting_results = fitter.fit_multiple_subjects(
            subjects_data, 
            max_subjects=None,  # 已經在前面限制了
            continue_on_failure=True,
            verbose=True
        )
        
        # ========================================
        # 4. 結果分析和摘要
        # ========================================
        
        print(f"\n📊 階段4: 結果分析")
        print("-" * 40)
        
        analysis_summary = analyze_fitting_results(fitting_results)
        
        # ========================================
        # 5. 儲存結果
        # ========================================
        
        if save_results:
            print(f"\n💾 階段5: 儲存結果")
            print("-" * 40)
            
            # 生成檔名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"sequential_lba_{first_side}_first_{timestamp}"
            
            saved_files = fitter.save_results(fitting_results, output_prefix)
            
            # 儲存分析摘要
            summary_filename = f"{output_prefix}_summary.txt"
            save_analysis_summary(analysis_summary, summary_filename)
            saved_files['summary'] = summary_filename
        else:
            saved_files = {}
        
        # ========================================
        # 6. 最終摘要
        # ========================================
        
        total_time = time.time() - analysis_start_time
        
        final_summary = {
            'analysis_complete': True,
            'total_time_minutes': total_time / 60,
            'data_info': {
                'csv_file': csv_file,
                'total_trials': len(data_df),
                'total_subjects': len(participants),
                'analyzed_subjects': len(subjects_data),
                'excluded_subjects': len(excluded_subjects)
            },
            'model_config': {
                'first_side': first_side,
                'time_split_ratio': time_split_ratio,
                'mcmc_config': mcmc_config
            },
            'fitting_results': fitting_results,
            'analysis_summary': analysis_summary,
            'quality_report': quality_report,
            'saved_files': saved_files
        }
        
        print(f"\n{'='*60}")
        print(f"🎉 序列LBA分析完成!")
        print(f"⏱️ 總時間: {total_time/60:.1f} 分鐘")
        print(f"✅ 成功率: {analysis_summary['success_rate']:.1%}")
        print(f"🔄 收斂率: {analysis_summary['convergence_rate']:.1%}")
        print(f"📁 輸出檔案: {len(saved_files)} 個")
        print(f"{'='*60}")
        
        return final_summary
        
    except Exception as e:
        print(f"\n❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'analysis_complete': False,
            'error': str(e),
            'total_time_minutes': (time.time() - analysis_start_time) / 60
        }

def analyze_fitting_results(results):
    """分析擬合結果，生成摘要統計"""
    
    print("🔍 分析擬合結果...")
    
    total_subjects = len(results)
    successful_results = [r for r in results if r.get('success', False)]
    converged_results = [r for r in successful_results if r.get('converged', False)]
    
    success_rate = len(successful_results) / total_subjects if total_subjects > 0 else 0
    convergence_rate = len(converged_results) / len(successful_results) if successful_results else 0
    
    # 基本統計
    summary = {
        'total_subjects': total_subjects,
        'successful_subjects': len(successful_results),
        'converged_subjects': len(converged_results),
        'success_rate': success_rate,
        'convergence_rate': convergence_rate
    }
    
    if successful_results:
        # 時間統計
        sampling_times = [r['sampling_time_minutes'] for r in successful_results]
        summary.update({
            'mean_sampling_time': np.mean(sampling_times),
            'std_sampling_time': np.std(sampling_times),
            'total_sampling_time': np.sum(sampling_times)
        })
        
        # 收斂統計
        if converged_results:
            rhat_values = [r['convergence_diagnostics']['rhat_max'] for r in converged_results 
                          if 'convergence_diagnostics' in r]
            ess_values = [r['convergence_diagnostics']['ess_bulk_min'] for r in converged_results 
                         if 'convergence_diagnostics' in r]
            
            if rhat_values:
                summary.update({
                    'mean_rhat': np.mean(rhat_values),
                    'max_rhat': np.max(rhat_values),
                    'min_ess': np.min(ess_values) if ess_values else np.nan,
                    'mean_ess': np.mean(ess_values) if ess_values else np.nan
                })
        
        # 參數統計
        if converged_results:
            # 收集所有參數
            all_params = {}
            for result in converged_results:
                for param_name, param_value in result['posterior_means'].items():
                    if not np.isnan(param_value):
                        if param_name not in all_params:
                            all_params[param_name] = []
                        all_params[param_name].append(param_value)
            
            # 計算參數統計
            param_stats = {}
            for param_name, values in all_params.items():
                if len(values) > 0:
                    param_stats[param_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'n_subjects': len(values)
                    }
            
            summary['parameter_statistics'] = param_stats
        
        # 失敗原因統計
        failed_results = [r for r in results if not r.get('success', False)]
        if failed_results:
            failure_reasons = {}
            for result in failed_results:
                reason = result.get('error', 'Unknown error')
                # 簡化錯誤訊息
                if 'Insufficient data' in reason:
                    reason = 'Insufficient data'
                elif 'validation failed' in reason:
                    reason = 'Model validation failed'
                elif 'MCMC' in reason or 'sampling' in reason:
                    reason = 'MCMC sampling failed'
                else:
                    reason = 'Other error'
                
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            summary['failure_reasons'] = failure_reasons
    
    print(f"✅ 結果分析完成")
    print(f"   成功率: {success_rate:.1%}")
    print(f"   收斂率: {convergence_rate:.1%}")
    if successful_results:
        print(f"   平均採樣時間: {summary['mean_sampling_time']:.1f} 分鐘")
    
    return summary

def save_analysis_summary(summary, filename):
    """儲存分析摘要到文字檔"""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("序列處理LBA分析摘要報告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本統計
            f.write("基本統計:\n")
            f.write("-" * 20 + "\n")
            f.write(f"總受試者數: {summary['total_subjects']}\n")
            f.write(f"成功擬合: {summary['successful_subjects']}\n")
            f.write(f"收斂擬合: {summary['converged_subjects']}\n")
            f.write(f"成功率: {summary['success_rate']:.1%}\n")
            f.write(f"收斂率: {summary['convergence_rate']:.1%}\n\n")
            
            # 時間統計
            if 'mean_sampling_time' in summary:
                f.write("時間統計:\n")
                f.write("-" * 20 + "\n")
                f.write(f"平均採樣時間: {summary['mean_sampling_time']:.1f} 分鐘\n")
                f.write(f"標準差: {summary['std_sampling_time']:.1f} 分鐘\n")
                f.write(f"總採樣時間: {summary['total_sampling_time']:.1f} 分鐘\n\n")
            
            # 收斂統計
            if 'mean_rhat' in summary:
                f.write("收斂統計:\n")
                f.write("-" * 20 + "\n")
                f.write(f"平均 R̂: {summary['mean_rhat']:.3f}\n")
                f.write(f"最大 R̂: {summary['max_rhat']:.3f}\n")
                f.write(f"最小 ESS: {summary['min_ess']:.0f}\n")
                f.write(f"平均 ESS: {summary['mean_ess']:.0f}\n\n")
            
            # 參數統計
            if 'parameter_statistics' in summary:
                f.write("參數統計摘要:\n")
                f.write("-" * 20 + "\n")
                for param_name, stats in summary['parameter_statistics'].items():
                    f.write(f"{param_name}:\n")
                    f.write(f"  平均: {stats['mean']:.3f}\n")
                    f.write(f"  標準差: {stats['std']:.3f}\n")
                    f.write(f"  範圍: [{stats['min']:.3f}, {stats['max']:.3f}]\n")
                    f.write(f"  受試者數: {stats['n_subjects']}\n\n")
            
            # 失敗原因
            if 'failure_reasons' in summary:
                f.write("失敗原因:\n")
                f.write("-" * 20 + "\n")
                for reason, count in summary['failure_reasons'].items():
                    f.write(f"{reason}: {count}\n")
        
        print(f"✅ 摘要報告已儲存: {filename}")
        
    except Exception as e:
        print(f"⚠️ 摘要報告儲存失敗: {e}")

def compare_processing_orders(csv_file='GRT_LBA.csv', max_subjects=3, 
                            time_split_ratio=0.6, mcmc_config=None):
    """
    比較不同處理順序的分析結果
    
    Args:
        csv_file: 資料檔案路徑
        max_subjects: 最大受試者數限制
        time_split_ratio: 時間分割比例
        mcmc_config: MCMC配置
        
    Returns:
        dict: 比較結果
    """
    
    print("🏆 處理順序比較分析")
    print("=" * 60)
    
    comparison_results = {}
    total_start_time = time.time()
    
    # 左先處理
    print("\n📍 分析1: 左邊先處理")
    print("-" * 30)
    left_first_result = main_sequential_analysis(
        csv_file=csv_file,
        max_subjects=max_subjects,
        first_side='left',
        time_split_ratio=time_split_ratio,
        mcmc_config=mcmc_config,
        save_results=True
    )
    comparison_results['left_first'] = left_first_result
    
    # 右先處理
    print("\n📍 分析2: 右邊先處理")
    print("-" * 30)
    right_first_result = main_sequential_analysis(
        csv_file=csv_file,
        max_subjects=max_subjects,
        first_side='right',
        time_split_ratio=time_split_ratio,
        mcmc_config=mcmc_config,
        save_results=True
    )
    comparison_results['right_first'] = right_first_result
    
    # 比較分析
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print("🎯 處理順序比較結果")
    print("-" * 60)
    
    for order_name, result in comparison_results.items():
        if result['analysis_complete']:
            summary = result['analysis_summary']
            print(f"{order_name:15s}: 成功率={summary['success_rate']:5.1%}, "
                  f"收斂率={summary['convergence_rate']:5.1%}, "
                  f"時間={result['total_time_minutes']:5.1f}分")
        else:
            print(f"{order_name:15s}: 分析失敗 - {result.get('error', 'Unknown error')}")
    
    print(f"\n總比較時間: {total_time/60:.1f} 分鐘")
    print("="*60)
    
    comparison_results['comparison_summary'] = {
        'total_comparison_time': total_time / 60,
        'both_successful': (left_first_result['analysis_complete'] and 
                           right_first_result['analysis_complete'])
    }
    
    return comparison_results

def quick_test_sequential(csv_file='GRT_LBA.csv'):
    """快速測試序列模型"""
    
    print("🧪 快速測試: 序列處理LBA")
    print("=" * 40)
    
    # 使用簡化的MCMC設定進行快速測試
    quick_mcmc_config = {
        'draws': 100,
        'tune': 100,
        'chains': 1,
        'cores': 1,
        'target_accept': 0.80,
        'max_treedepth': 6,
        'progressbar': True,
        'return_inferencedata': True
    }
    
    try:
        result = main_sequential_analysis(
            csv_file=csv_file,
            max_subjects=1,
            first_side='left',
            time_split_ratio=0.6,
            mcmc_config=quick_mcmc_config,
            save_results=False
        )
        
        if result['analysis_complete'] and result['analysis_summary']['successful_subjects'] > 0:
            print("✅ 序列模型快速測試成功!")
            print("🎯 模組架構運作正常")
            print("🔬 準備進行完整分析")
            return True
        else:
            print("❌ 快速測試失敗")
            return False
            
    except Exception as e:
        print(f"❌ 測試錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_analysis_pipeline(csv_file='GRT_LBA.csv', analysis_type='medium'):
    """
    執行完整分析管線
    
    Args:
        csv_file: 資料檔案路徑
        analysis_type: 分析類型 ('quick', 'small', 'medium', 'large', 'full', 'compare')
    """
    
    # 根據分析類型設定參數
    analysis_configs = {
        'quick': {
            'max_subjects': 1,
            'mcmc_config': {'draws': 100, 'tune': 100, 'chains': 1}
        },
        'small': {
            'max_subjects': 3,
            'mcmc_config': {'draws': 200, 'tune': 200, 'chains': 2}
        },
        'medium': {
            'max_subjects': 5,
            'mcmc_config': {'draws': 400, 'tune': 400, 'chains': 2}
        },
        'large': {
            'max_subjects': 10,
            'mcmc_config': {'draws': 500, 'tune': 500, 'chains': 2}
        },
        'full': {
            'max_subjects': None,
            'mcmc_config': {'draws': 600, 'tune': 600, 'chains': 3}
        }
    }
    
    if analysis_type == 'compare':
        # 處理順序比較
        return compare_processing_orders(
            csv_file=csv_file,
            max_subjects=3,
            mcmc_config={'draws': 200, 'tune': 200, 'chains': 2}
        )
    else:
        # 單一分析
        config = analysis_configs.get(analysis_type, analysis_configs['medium'])
        return main_sequential_analysis(
            csv_file=csv_file,
            first_side='left',
            time_split_ratio=0.6,
            **config
        )

# ============================================================================
# 執行介面
# ============================================================================

if __name__ == "__main__":
    print("🎯 序列處理LBA分析選項:")
    print("=" * 40)
    print("1. 快速測試 (1受試者, 簡化MCMC)")
    print("2. 小型分析 (3受試者)")
    print("3. 中型分析 (5受試者)")
    print("4. 大型分析 (10受試者)")
    print("5. 完整分析 (所有受試者)")
    print("6. 處理順序比較 (左vs右先處理)")
    
    try:
        choice = input("\n請選擇 (1-6): ").strip()
        
        analysis_map = {
            '1': 'quick',
            '2': 'small', 
            '3': 'medium',
            '4': 'large',
            '5': 'full',
            '6': 'compare'
        }
        
        analysis_type = analysis_map.get(choice, 'quick')
        
        print(f"\n🚀 執行{analysis_type}分析...")
        
        if analysis_type == 'quick':
            # 快速測試使用特殊函數
            success = quick_test_sequential()
        else:
            # 其他分析使用完整管線
            result = run_full_analysis_pipeline(analysis_type=analysis_type)
            success = result.get('analysis_complete', False) if isinstance(result, dict) else False
        
        if success:
            print("\n🎉 分析成功完成!")
            print("✅ 序列LBA模組架構運作正常")
        else:
            print("\n❌ 分析失敗")
            
    except KeyboardInterrupt:
        print("\n⏹️ 分析被使用者中斷")
    except Exception as e:
        print(f"\n💥 未預期錯誤: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 使用範例和說明
# ============================================================================

"""
序列處理LBA分析使用範例:

# 1. 基本使用
from main_analysis import main_sequential_analysis
result = main_sequential_analysis('GRT_LBA.csv', max_subjects=5, first_side='left')

# 2. 快速測試
from main_analysis import quick_test_sequential
success = quick_test_sequential('GRT_LBA.csv')

# 3. 處理順序比較
from main_analysis import compare_processing_orders
comparison = compare_processing_orders('GRT_LBA.csv', max_subjects=3)

# 4. 自訂MCMC設定
custom_mcmc = {
    'draws': 800,
    'tune': 800,
    'chains': 3,
    'target_accept': 0.90
}
result = main_sequential_analysis('GRT_LBA.csv', mcmc_config=custom_mcmc)

# 5. 完整分析管線
from main_analysis import run_full_analysis_pipeline
result = run_full_analysis_pipeline('GRT_LBA.csv', 'large')

檔案結構:
- data_utils.py: 資料預處理
- single_side_lba.py: 單邊LBA處理器
- four_choice_lba.py: 四選一LBA競爭器
- sequential_model.py: 序列處理主模型
- model_fitting.py: 模型擬合器
- main_analysis.py: 主分析流程 (此檔案)

輸出檔案:
- *_main_*.csv: 主要結果
- *_params_*.csv: 參數估計
- *_convergence_*.csv: 收斂診斷
- *_summary.txt: 分析摘要報告
"""
