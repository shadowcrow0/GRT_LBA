# -*- coding: utf-8 -*-
"""
data_utils.py - 資料預處理模組
Sequential Processing LBA - Data Utilities Module

功能：
- 載入和清理實驗資料
- 將4選擇資料分解為左右通道特徵
- 提取單一受試者資料
- 資料品質檢查
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class DataProcessor:
    """資料預處理器"""
    
    def __init__(self):
        # 刺激映射：刺激編號 -> 左右線條方向
        self.stimulus_mapping = {
            0: {'left': 1, 'right': 0},  # 左對角，右垂直
            1: {'left': 1, 'right': 1},  # 左對角，右對角
            2: {'left': 0, 'right': 0},  # 左垂直，右垂直
            3: {'left': 0, 'right': 1}   # 左垂直，右對角
        }
        
        # 選擇映射：選擇編號 -> 左右判斷
        self.choice_mapping = {
            0: {'left': 1, 'right': 0},  # 選擇 \|
            1: {'left': 1, 'right': 1},  # 選擇 \/
            2: {'left': 0, 'right': 0},  # 選擇 ||
            3: {'left': 0, 'right': 1}   # 選擇 |/
        }
        
        # 描述映射
        self.stimulus_descriptions = {
            0: 'Left\\Right|',   # 左對角右垂直
            1: 'Left\\Right/',   # 左對角右對角
            2: 'Left|Right|',    # 左垂直右垂直
            3: 'Left|Right/'     # 左垂直右對角
        }
    
    def load_and_clean_data(self, csv_file: str) -> pd.DataFrame:
        """載入並清理實驗資料"""
        
        print("📂 載入實驗資料...")
        
        try:
            # 載入原始資料
            raw_df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"✅ 載入 {len(raw_df)} 個試驗")
            
            # 基本資料清理
            print("🔄 執行資料清理...")
            
            # RT範圍檢查
            valid_rt = (raw_df['RT'] >= 0.1) & (raw_df['RT'] <= 3.0)
            
            # 選擇有效性檢查
            valid_choice = raw_df['Response'].isin([0, 1, 2, 3])
            
            # 刺激有效性檢查
            valid_stimulus = raw_df['Stimulus'].isin([0, 1, 2, 3])
            
            # 組合所有有效條件
            valid_trials = valid_rt & valid_choice & valid_stimulus
            
            # 過濾資料
            clean_df = raw_df[valid_trials].copy()
            
            # 移除缺失值
            clean_df = clean_df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
            
            print(f"✅ 資料清理完成:")
            print(f"   原始: {len(raw_df)} 試驗")
            print(f"   清理後: {len(clean_df)} 試驗")
            print(f"   保留率: {len(clean_df)/len(raw_df)*100:.1f}%")
            print(f"   受試者數: {clean_df['participant'].nunique()}")
            
            # 添加分解特徵
            clean_df = self.add_decomposed_features(clean_df)
            
            # 顯示刺激分布
            self.print_stimulus_distribution(clean_df)
            
            return clean_df
            
        except FileNotFoundError:
            print(f"❌ 找不到檔案: {csv_file}")
            raise
        except Exception as e:
            print(f"❌ 資料載入失敗: {e}")
            raise
    
    def add_decomposed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加分解的左右通道特徵"""
        
        print("🔄 添加左右通道特徵...")
        
        df = df.copy()
        
        # 刺激特徵分解
        df['left_stimulus'] = df['Stimulus'].map(
            lambda x: self.stimulus_mapping.get(x, {'left': 0})['left']
        )
        df['right_stimulus'] = df['Stimulus'].map(
            lambda x: self.stimulus_mapping.get(x, {'right': 0})['right']
        )
        
        # 選擇特徵分解
        df['left_choice'] = df['Response'].map(
            lambda x: self.choice_mapping.get(x, {'left': 0})['left']
        )
        df['right_choice'] = df['Response'].map(
            lambda x: self.choice_mapping.get(x, {'right': 0})['right']
        )
        
        # 計算左右通道正確性
        df['left_correct'] = (df['left_choice'] == df['left_stimulus']).astype(int)
        df['right_correct'] = (df['right_choice'] == df['right_stimulus']).astype(int)
        
        # 計算整體正確性（兩個通道都要對）
        df['both_correct'] = (df['left_correct'] & df['right_correct']).astype(int)
        
        print(f"✅ 通道特徵添加完成")
        print(f"   左通道準確率: {df['left_correct'].mean():.1%}")
        print(f"   右通道準確率: {df['right_correct'].mean():.1%}")
        print(f"   整體準確率: {df['both_correct'].mean():.1%}")
        
        return df
    
    def print_stimulus_distribution(self, df: pd.DataFrame):
        """顯示刺激分布"""
        
        print("\n📊 刺激分布:")
        print("-" * 50)
        
        for stimulus in [0, 1, 2, 3]:
            count = len(df[df['Stimulus'] == stimulus])
            percentage = count / len(df) * 100
            description = self.stimulus_descriptions.get(stimulus, f'Stimulus {stimulus}')
            print(f"   刺激 {stimulus} ({description}): {count:4d} 試驗 ({percentage:5.1f}%)")
        
        print("-" * 50)
    
    def extract_subject_data(self, df: pd.DataFrame, subject_id: int) -> Dict:
        """提取單一受試者資料"""
        
        # 過濾受試者資料
        subject_df = df[df['participant'] == subject_id].copy()
        
        if len(subject_df) == 0:
            raise ValueError(f"找不到受試者 {subject_id} 的資料")
        
        # 基本統計
        n_trials = len(subject_df)
        accuracy = subject_df['both_correct'].mean()
        mean_rt = subject_df['RT'].mean()
        std_rt = subject_df['RT'].std()
        
        # 左右通道統計
        left_accuracy = subject_df['left_correct'].mean()
        right_accuracy = subject_df['right_correct'].mean()
        
        return {
            'subject_id': subject_id,
            'n_trials': n_trials,
            'accuracy': accuracy,
            'mean_rt': mean_rt,
            'std_rt': std_rt,
            'left_accuracy': left_accuracy,
            'right_accuracy': right_accuracy,
            
            # 原始資料陣列
            'stimuli': subject_df['Stimulus'].values,
            'choices': subject_df['Response'].values,
            'rt': subject_df['RT'].values,
            'correct': subject_df['both_correct'].values,
            
            # 左右通道資料陣列
            'left_stimuli': subject_df['left_stimulus'].values,
            'right_stimuli': subject_df['right_stimulus'].values,
            'left_choices': subject_df['left_choice'].values,
            'right_choices': subject_df['right_choice'].values,
            'left_correct': subject_df['left_correct'].values,
            'right_correct': subject_df['right_correct'].values,
            
            # 完整DataFrame（供進階分析使用）
            'dataframe': subject_df
        }
    
    def get_subject_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """獲得所有受試者的摘要統計"""
        
        print("📊 計算受試者摘要統計...")
        
        summary_data = []
        
        for subject_id in sorted(df['participant'].unique()):
            try:
                subject_data = self.extract_subject_data(df, subject_id)
                
                summary_data.append({
                    'subject_id': subject_id,
                    'n_trials': subject_data['n_trials'],
                    'accuracy': subject_data['accuracy'],
                    'left_accuracy': subject_data['left_accuracy'],
                    'right_accuracy': subject_data['right_accuracy'],
                    'mean_rt': subject_data['mean_rt'],
                    'std_rt': subject_data['std_rt']
                })
                
            except Exception as e:
                print(f"⚠️ 受試者 {subject_id} 資料提取失敗: {e}")
                continue
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"✅ 摘要統計完成: {len(summary_df)} 位受試者")
        
        return summary_df
    
    def validate_data_quality(self, df: pd.DataFrame, min_trials_per_subject=50):
        """驗證資料品質"""
        
        print("🔍 驗證資料品質...")
        
        # 檢查受試者試驗數
        subject_trial_counts = df.groupby('participant').size()
        insufficient_data = subject_trial_counts[subject_trial_counts < min_trials_per_subject]
        
        if len(insufficient_data) > 0:
            print(f"⚠️ {len(insufficient_data)} 位受試者資料不足 (< {min_trials_per_subject} 試驗):")
            for subject_id, count in insufficient_data.items():
                print(f"   受試者 {subject_id}: {count} 試驗")
        
        # 檢查RT分布
        rt_stats = df['RT'].describe()
        print(f"\n📊 RT分布統計:")
        print(f"   平均: {rt_stats['mean']:.3f}s")
        print(f"   標準差: {rt_stats['std']:.3f}s")
        print(f"   範圍: {rt_stats['min']:.3f}s - {rt_stats['max']:.3f}s")
        
        # 檢查準確率分布
        accuracy_stats = df['both_correct'].describe()
        print(f"\n🎯 準確率統計:")
        print(f"   平均: {accuracy_stats['mean']:.1%}")
        print(f"   範圍: {accuracy_stats['min']:.1%} - {accuracy_stats['max']:.1%}")
        
        # 檢查極端值
        very_fast = df['RT'] < 0.2
        very_slow = df['RT'] > 2.0
        very_low_acc = df.groupby('participant')['both_correct'].mean() < 0.3
        
        if very_fast.sum() > 0:
            print(f"⚠️ {very_fast.sum()} 個極快反應 (< 0.2s)")
        
        if very_slow.sum() > 0:
            print(f"⚠️ {very_slow.sum()} 個極慢反應 (> 2.0s)")
        
        if very_low_acc.sum() > 0:
            print(f"⚠️ {very_low_acc.sum()} 位受試者準確率極低 (< 30%)")
        
        print("✅ 資料品質檢查完成")
        
        return {
            'total_trials': len(df),
            'total_subjects': df['participant'].nunique(),
            'insufficient_data_subjects': len(insufficient_data),
            'very_fast_trials': very_fast.sum(),
            'very_slow_trials': very_slow.sum(),
            'low_accuracy_subjects': very_low_acc.sum(),
            'mean_rt': rt_stats['mean'],
            'mean_accuracy': accuracy_stats['mean']
        }

# 便利函數
def load_data(csv_file='GRT_LBA.csv'):
    """快速載入資料的便利函數"""
    processor = DataProcessor()
    return processor.load_and_clean_data(csv_file)

def get_subject_data(df, subject_id):
    """快速提取受試者資料的便利函數"""
    processor = DataProcessor()
    return processor.extract_subject_data(df, subject_id)

# 測試函數
def test_data_processor(csv_file='GRT_LBA.csv'):
    """測試資料處理器功能"""
    
    print("🧪 測試資料處理器...")
    
    try:
        # 創建處理器
        processor = DataProcessor()
        
        # 載入資料
        df = processor.load_and_clean_data(csv_file)
        
        # 驗證資料品質
        quality_report = processor.validate_data_quality(df)
        
        # 獲得摘要統計
        summary = processor.get_subject_summary(df)
        
        # 測試單一受試者提取
        first_subject = df['participant'].iloc[0]
        subject_data = processor.extract_subject_data(df, first_subject)
        
        print(f"\n✅ 資料處理器測試成功!")
        print(f"   總試驗數: {len(df)}")
        print(f"   受試者數: {df['participant'].nunique()}")
        print(f"   測試受試者 {first_subject}: {subject_data['n_trials']} 試驗")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料處理器測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 如果直接執行此檔案，進行測試
    test_data_processor()
