#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
運行剩餘受試者的完整LBA分析
基於complete_reanalysis.py但只跑指定的受試者
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 檢查哪些受試者已經完成
def check_completed_participants():
    """檢查已完成的受試者"""
    completed = []
    results_dir = Path("complete_lba_results")
    
    if results_dir.exists():
        for file in results_dir.glob("participant_*_complete_lba_trace.nc"):
            participant_id = int(file.name.split('_')[1])
            completed.append(participant_id)
    
    return sorted(completed)

def get_remaining_participants():
    """獲取剩餘需要分析的受試者"""
    # 載入更新的數據檢查所有受試者 (只包含正確率>=65%的參與者)
    data = np.load('model_data_updated.npz', allow_pickle=True)
    all_participants = np.unique(data['participant_idx'])
    
    # 檢查已完成的受試者
    completed = check_completed_participants()
    
    # 計算剩餘受試者
    remaining = [p for p in all_participants if p not in completed]
    
    # 顯示原始參與者ID對應
    if 'participant_mapping' in data:
        mapping = data['participant_mapping'].item()
        original_ids = {new_id: old_id for old_id, new_id in mapping.items()}
        print(f"📊 參與者ID映射: {mapping}")
        print(f"📊 剩餘參與者的原始ID: {[original_ids[p] for p in remaining]}")
    
    print(f"📊 總受試者數: {len(all_participants)} (正確率>=65%)")
    print(f"✅ 已完成: {completed}")
    print(f"⏳ 剩餘需要跑: {remaining}")
    
    return remaining

# 修改原始的CompleteLBAReanalysis類，使其可以指定受試者清單
class RemainingParticipantsAnalysis:
    """專門跑剩餘受試者的分析"""
    
    def __init__(self, target_participants=None, results_dir="complete_lba_results", data_file='model_data_fixed.npz'):
        self.results_dir = Path(results_dir)
        self.data_file = data_file
        self.data = None
        self.target_participants = target_participants  # 指定的受試者清單
        self.all_participants = None
        self.results = {}
        
        # 刺激條件
        self.stimulus_conditions = {
            0: {'left': 'vertical', 'right': 'nonvertical'},
            1: {'left': 'nonvertical', 'right': 'nonvertical'}, 
            2: {'left': 'nonvertical', 'right': 'vertical'},
            3: {'left': 'vertical', 'right': 'vertical'}
        }
        
    def setup(self):
        """初始設置"""
        print("🔧 設置剩餘參與者分析...")
        print("=" * 60)
        
        # 載入數據
        self.data = np.load(self.data_file, allow_pickle=True)
        participant_idx = self.data['participant_idx']
        all_participants = np.unique(participant_idx)
        
        # 如果沒有指定，則自動獲取剩餘受試者
        if self.target_participants is None:
            self.target_participants = get_remaining_participants()
        
        self.all_participants = [p for p in all_participants if p in self.target_participants]
        
        print(f"📊 目標受試者: {self.target_participants}")
        print(f"📊 實際要跑的受試者: {self.all_participants}")
        print(f"📊 總試驗數: {len(participant_idx)}")
        
        # 創建結果目錄
        self.results_dir.mkdir(exist_ok=True)
        
        return len(self.all_participants) > 0

# 導入原始的CompleteDualLBAModelFitter
import sys
import importlib.util

# 嘗試從現有程式中導入CompleteDualLBAModelFitter
try:
    # 如果存在getposterior_complete.py，從中導入
    if Path('getposterior_complete.py').exists():
        spec = importlib.util.spec_from_file_location("getposterior_complete", "getposterior_complete.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CompleteDualLBAModelFitter = module.CompleteDualLBAModelFitter
        print("✅ 從getposterior_complete.py導入LBA模型")
    else:
        raise ImportError("找不到getposterior_complete.py")
        
except ImportError as e:
    print(f"⚠️ 無法導入LBA模型: {e}")
    print("請確認getposterior_complete.py存在且包含CompleteDualLBAModelFitter類")
    sys.exit(1)

def fit_participant_models(participant_id, data_dict):
    """擬合單個受試者的LBA模型"""
    
    print(f"\n🎯 開始擬合參與者 {participant_id}")
    
    # 準備參與者數據 (使用更新的數據結構)
    participant_mask = data_dict['participant_idx'] == participant_id
    
    # 從更新的數據結構中直接提取數據
    model_input = data_dict['model_input_data'].item()
    
    # 確保參與者有數據
    if np.sum(participant_mask) == 0:
        raise ValueError(f"參與者 {participant_id} 沒有數據")
    
    participant_data = pd.DataFrame({
        'participant_id': [participant_id] * np.sum(participant_mask),
        'stimulus_condition': model_input['stimulus_condition'][participant_mask],
        'observed_rt': model_input['observed_rt'][participant_mask], 
        'response_correct': model_input['response_correct'][participant_mask]
    })
    
    # 添加response列（如果存在）
    if 'response' in model_input:
        participant_data['response'] = model_input['response'][participant_mask]
    
    print(f"   試驗數: {len(participant_data)}")
    print(f"   RT範圍: {participant_data['observed_rt'].min():.3f} - {participant_data['observed_rt'].max():.3f}")
    
    # 初始化擬合器
    fitter = CompleteDualLBAModelFitter(random_seed=42)
    
    try:
        # 擬合模型 (進一步優化速度)
        summary, trace, priors = fitter.fit_model_for_participant(
            participant_data, draws=1000, tune=1000, chains=4
        )
        
        # 提取drift rate參數
        drift_params = fitter.extract_drift_parameters(trace, participant_data)
        
        # 進行RT和反應正確性模擬
        print(f"🎲 進行後驗預測模擬...")
        simulation_df = fitter.simulate_rt_and_accuracy(trace, participant_data, n_simulations=1000)
        simulation_summary = fitter.create_simulation_summary(simulation_df)
        
        # 進行最終左右drift rate模擬
        print(f"🎯 進行最終左右drift rate模擬...")
        final_drift_results = fitter.simulate_final_left_right_drifts(trace, participant_data, n_simulations=1000)
        
        # 保存結果
        output_dir = Path("complete_lba_results")
        output_dir.mkdir(exist_ok=True)
        
        # 保存trace
        trace_file = output_dir / f"participant_{participant_id}_complete_lba_trace.nc"
        trace.to_netcdf(trace_file)
        print(f"   ✅ Trace已保存: {trace_file}")
        
        # 保存摘要
        summary_file = output_dir / f"participant_{participant_id}_complete_lba_summary.csv"
        summary.to_csv(summary_file)
        print(f"   ✅ 摘要已保存: {summary_file}")
        
        # 保存drift parameters
        drift_file = output_dir / f"participant_{participant_id}_drift_params.npz"
        np.savez(drift_file, **drift_params)
        print(f"   ✅ Drift參數已保存: {drift_file}")
        
        # 保存模擬結果
        simulation_file = output_dir / f"participant_{participant_id}_simulation_results.csv"
        simulation_df.to_csv(simulation_file, index=False)
        print(f"   ✅ RT和反應模擬結果已保存: {simulation_file}")
        
        # 保存模擬摘要
        summary_stats_file = output_dir / f"participant_{participant_id}_simulation_summary.json"
        import json
        summary_to_save = {
            'overall_mae': float(simulation_summary['overall_mae']),
            'overall_rmse': float(simulation_summary['overall_rmse']),
            'participant_accuracy': float(simulation_summary['participant_accuracy'].iloc[0]) if len(simulation_summary['participant_accuracy']) > 0 else 0.0
        }
        with open(summary_stats_file, 'w') as f:
            json.dump(summary_to_save, f, indent=2)
        print(f"   ✅ 模擬摘要已保存: {summary_stats_file}")
        
        # 保存最終左右drift rate模擬結果
        final_drift_file = output_dir / f"participant_{participant_id}_final_left_right_drifts.csv"
        final_drift_df = pd.DataFrame([
            {k: v for k, v in result.items() if k not in ['v_left_samples', 'v_right_samples']}
            for result in final_drift_results
        ])
        final_drift_df.to_csv(final_drift_file, index=False)
        print(f"   ✅ 最終左右drift rate模擬結果已保存: {final_drift_file}")
        
        # 保存詳細samples
        samples_file = output_dir / f"participant_{participant_id}_drift_samples.npz"
        samples_data = {}
        for i, result in enumerate(final_drift_results):
            samples_data[f'trial_{i}_left_samples'] = result['v_left_samples']
            samples_data[f'trial_{i}_right_samples'] = result['v_right_samples']
        np.savez(samples_file, **samples_data)
        print(f"   ✅ 詳細drift rate樣本已保存: {samples_file}")
        
        # 顯示結果摘要
        print(f"\n📊 參與者 {participant_id} 結果摘要:")
        main_params = ['v_vertical_left', 'v_nonvertical_left', 'v_vertical_right', 'v_nonvertical_right',
                      'boundary', 'non_decision', 'start_point_variability']
        for param in main_params:
            if param in summary.index:
                row = summary.loc[param]
                print(f"   {param}: μ={row['mean']:.3f} ± {row['sd']:.3f}")
        
        print(f"   RT預測誤差 (MAE): {simulation_summary['overall_mae']:.3f}")
        print(f"   RT預測誤差 (RMSE): {simulation_summary['overall_rmse']:.3f}")
        print(f"   模擬試驗總數: {len(simulation_df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 參與者 {participant_id} 擬合失敗: {e}")
        return False

def main():
    """主函數"""
    print("🚀 開始運行剩餘受試者的完整LBA分析")
    print("=" * 80)
    
    # 檢查剩餘受試者
    remaining_participants = get_remaining_participants()
    
    if not remaining_participants:
        print("✅ 所有受試者已完成分析！")
        return
    
    print(f"\n將分析 {len(remaining_participants)} 位剩餘受試者: {remaining_participants}")
    
    # 載入更新的數據 (包含 GRT_LBA.csv 的所有參數)
    data = np.load('model_data_updated.npz', allow_pickle=True)
    
    # 分析每位受試者
    success_count = 0
    failed_participants = []
    
    for i, participant_id in enumerate(remaining_participants):
        print(f"\n{'='*20} 處理參與者 {participant_id} ({i+1}/{len(remaining_participants)}) {'='*20}")
        
        try:
            success = fit_participant_models(participant_id, data)
            if success:
                success_count += 1
                print(f"✅ 參與者 {participant_id} 完成")
            else:
                failed_participants.append(participant_id)
                print(f"❌ 參與者 {participant_id} 失敗")
        except Exception as e:
            failed_participants.append(participant_id)
            print(f"❌ 參與者 {participant_id} 發生錯誤: {e}")
    
    # 最終摘要
    print(f"\n🎉 分析完成！")
    print(f"✅ 成功: {success_count}/{len(remaining_participants)} 位受試者")
    if failed_participants:
        print(f"❌ 失敗: {failed_participants}")
    
    print(f"\n📁 所有結果已保存至: complete_lba_results/")

if __name__ == '__main__':
    main()