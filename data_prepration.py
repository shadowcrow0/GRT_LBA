# ===================================================================
# 檔案1: data_preparation.py
# 目的：將 CSV 處理成模型可用的格式
# ===================================================================

import numpy as np
import pandas as pd

def prepare_data_for_model(df, subject_id_list):
    """從 DataFrame 準備模型數據"""
    filtered_df = df[df['participant'].isin(subject_id_list)].copy()
    if len(filtered_df) == 0: 
        raise ValueError("找不到任何指定受試者的資料")
    
    stimulus_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    choice_mapping = {0: {'left': 1, 'right': 0}, 1: {'left': 1, 'right': 1}, 2: {'left': 0, 'right': 0}, 3: {'left': 0, 'right': 1}}
    
    left_stim_is_diag = filtered_df['Stimulus'].map(lambda s: stimulus_mapping.get(s, {}).get('left')).values
    right_stim_is_diag = filtered_df['Stimulus'].map(lambda s: stimulus_mapping.get(s, {}).get('right')).values
    left_choice_is_diag = filtered_df['Response'].map(lambda r: choice_mapping.get(r, {}).get('left')).values
    right_choice_is_diag = filtered_df['Response'].map(lambda r: choice_mapping.get(r, {}).get('right')).values
    
    left_match = (left_stim_is_diag == 1)
    right_match = (right_stim_is_diag == 1)
    is_correct = (left_stim_is_diag == left_choice_is_diag) & (right_stim_is_diag == right_choice_is_diag)
    
    return {
        "rt": filtered_df['RT'].values, 
        "response_correct": is_correct.astype(int), 
        "left_match": left_match.astype(int), 
        "right_match": right_match.astype(int)
    }

if __name__ == '__main__':
    print("=== 數據準備階段 ===")
    
    # 讀取 CSV
    df = pd.read_csv('GRT_LBA.csv')
    df = df.dropna(subset=['Response', 'RT', 'Stimulus', 'participant'])
    
    # 計算正確率並篩選高正確率受試者
    prepared_full_data = prepare_data_for_model(df, df['participant'].unique())
    df['is_correct'] = prepared_full_data['response_correct']
    
    accuracy_per_subject = df.groupby('participant')['is_correct'].mean()
    high_accuracy_subjects = accuracy_per_subject[accuracy_per_subject > 0.7].index.tolist()
    df_filtered = df[df['participant'].isin(high_accuracy_subjects)].copy()
    
    # RT 閾值篩選
    rt_threshold = 0.150
    df_cleaned = df_filtered[df_filtered['RT'] >= rt_threshold].copy()
    
    print(f"處理完成：{len(high_accuracy_subjects)} 位受試者，{len(df_cleaned)} 筆試次")
    
    # 準備模型數據
    participant_ids = df_cleaned['participant'].unique()
    participant_idx, _ = pd.factorize(df_cleaned['participant'])
    coords = {"participant": participant_ids, "obs_id": np.arange(len(participant_idx))}
    model_input_data = prepare_data_for_model(df_cleaned, participant_ids)
    observed_value = np.column_stack([
        np.asarray(model_input_data['rt'], np.float32),
        np.asarray(model_input_data['response_correct'], np.float32)
    ])
    
    # 保存數據
    data_for_models = {
        'observed_value': observed_value,
        'participant_idx': participant_idx,
        'model_input_data': model_input_data,
        'coords': coords
    }
    np.savez('model_data.npz', **data_for_models)
    print("✅ 數據已保存至 model_data.npz")

# ===================================================================
# 檔案2: coactive_vs_parallel_analysis.py  
# 目的：完整的模型比較分析
# ===================================================================

# [這裡放入之前成功的完整模型比較代碼]
# 包含：LBA函數、兩個模型定義、採樣、比較、視覺化

# ===================================================================
# 檔案結構總結
# ===================================================================

"""
最終你需要的檔案：

📁 你的工作目錄/
├── 📄 GRT_LBA.csv                      # 原始數據
├── 📄 data_preparation.py              # 檔案1：數據處理
├── 📄 coactive_vs_parallel_analysis.py # 檔案2：模型分析
└── 📁 輸出檔案/
    ├── 📄 model_data.npz               # 處理後的數據
    ├── 📄 coactive_correct.nc          # Coactive 模型結果
    ├── 📄 parallel_and_correct.nc      # Parallel AND 模型結果
    └── 📊 comparison_plots.png         # 比較圖表

執行順序：
1. python data_preparation.py           # 3分鐘
2. python coactive_vs_parallel_analysis.py  # 30分鐘

或者：
1. python complete_analysis.py          # 35分鐘一次完成
"""
