#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all participants' final_left_right_drifts.csv files into one combined dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def merge_all_drift_data():
    """Load and merge all participants' drift data"""
    
    results_dir = Path("complete_lba_results")
    drift_files = list(results_dir.glob("*final_left_right_drifts.csv"))
    
    print(f"📊 Found {len(drift_files)} drift rate files")
    
    all_data = []
    total_trials = 0
    
    for file in sorted(drift_files):
        try:
            df = pd.read_csv(file)
            participant_id = int(file.name.split('_')[1])
            
            # Verify participant_id matches the data
            if 'participant_id' in df.columns:
                if df['participant_id'].iloc[0] != participant_id:
                    print(f"   ⚠️  Warning: File {file.name} has participant_id mismatch")
            
            all_data.append(df)
            total_trials += len(df)
            print(f"   ✅ {file.name}: {len(df)} trials (Participant {participant_id})")
            
        except Exception as e:
            print(f"   ❌ {file.name}: Failed to load - {e}")
    
    if not all_data:
        raise ValueError("No data files loaded successfully")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n📈 Combined dataset:")
    print(f"   Total trials: {len(combined_df)}")
    print(f"   Participants: {len(combined_df['participant_id'].unique())}")
    print(f"   Columns: {list(combined_df.columns)}")
    
    # Add symmetry type column
    combined_df['symmetry_type'] = combined_df['stimulus_condition'].apply(
        lambda x: 'asymmetric' if x in [0, 2] else 'symmetric'
    )
    
    # Summary statistics
    print(f"\n📊 Data summary:")
    print(f"   Asymmetric conditions (0,2): {len(combined_df[combined_df['symmetry_type'] == 'asymmetric'])} trials")
    print(f"   Symmetric conditions (1,3): {len(combined_df[combined_df['symmetry_type'] == 'symmetric'])} trials")
    
    # Per participant summary
    print(f"\n👥 Per participant summary:")
    participant_summary = combined_df.groupby('participant_id').agg({
        'trial_id': 'count',
        'stimulus_condition': lambda x: list(x.unique()),
        'v_left_mean': ['mean', 'std', 'min', 'max'],
        'v_right_mean': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    participant_summary.columns = ['n_trials', 'conditions', 
                                  'v_left_mean', 'v_left_std', 'v_left_min', 'v_left_max',
                                  'v_right_mean', 'v_right_std', 'v_right_min', 'v_right_max']
    
    print(participant_summary)
    
    return combined_df, participant_summary

def save_combined_data(combined_df, participant_summary):
    """Save combined dataset and summary"""
    
    print(f"\n💾 Saving combined dataset...")
    
    # Save main combined dataset
    combined_df.to_csv('all_participants_final_left_right_drifts.csv', index=False)
    print(f"   ✅ Combined dataset saved: all_participants_final_left_right_drifts.csv")
    
    # Save participant summary
    participant_summary.to_csv('participant_summary_drift_data.csv')
    print(f"   ✅ Participant summary saved: participant_summary_drift_data.csv")
    
    # Create additional summary files
    
    # 1. Condition-wise summary
    condition_summary = combined_df.groupby(['participant_id', 'stimulus_condition']).agg({
        'trial_id': 'count',
        'v_left_mean': ['mean', 'std'],
        'v_right_mean': ['mean', 'std'],
        'drift_difference': ['mean', 'std']
    }).round(4)
    
    condition_summary.columns = ['n_trials', 'v_left_mean', 'v_left_std', 
                                'v_right_mean', 'v_right_std', 'drift_diff_mean', 'drift_diff_std']
    condition_summary.to_csv('condition_wise_summary.csv')
    print(f"   ✅ Condition-wise summary saved: condition_wise_summary.csv")
    
    # 2. Symmetry-wise summary
    symmetry_summary = combined_df.groupby(['participant_id', 'symmetry_type']).agg({
        'trial_id': 'count',
        'v_left_mean': ['mean', 'std'],
        'v_right_mean': ['mean', 'std'],
        'drift_difference': ['mean', 'std']
    }).round(4)
    
    symmetry_summary.columns = ['n_trials', 'v_left_mean', 'v_left_std', 
                               'v_right_mean', 'v_right_std', 'drift_diff_mean', 'drift_diff_std']
    symmetry_summary.to_csv('symmetry_wise_summary.csv')
    print(f"   ✅ Symmetry-wise summary saved: symmetry_wise_summary.csv")
    
    # 3. Overall statistics
    overall_stats = {
        'total_trials': len(combined_df),
        'total_participants': len(combined_df['participant_id'].unique()),
        'conditions': {
            'asymmetric': len(combined_df[combined_df['symmetry_type'] == 'asymmetric']),
            'symmetric': len(combined_df[combined_df['symmetry_type'] == 'symmetric'])
        },
        'drift_rates': {
            'v_left_mean': {
                'overall_mean': float(combined_df['v_left_mean'].mean()),
                'overall_std': float(combined_df['v_left_mean'].std()),
                'min': float(combined_df['v_left_mean'].min()),
                'max': float(combined_df['v_left_mean'].max())
            },
            'v_right_mean': {
                'overall_mean': float(combined_df['v_right_mean'].mean()),
                'overall_std': float(combined_df['v_right_mean'].std()),
                'min': float(combined_df['v_right_mean'].min()),
                'max': float(combined_df['v_right_mean'].max())
            }
        }
    }
    
    import json
    with open('combined_drift_data_stats.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"   ✅ Overall statistics saved: combined_drift_data_stats.json")
    
    return overall_stats

def main():
    """Main function"""
    print("🚀 Merging all participants' final_left_right_drifts.csv files")
    print("=" * 70)
    
    try:
        # Load and merge data
        combined_df, participant_summary = merge_all_drift_data()
        
        # Save combined data
        overall_stats = save_combined_data(combined_df, participant_summary)
        
        print(f"\n🎉 Data merging completed!")
        print(f"   Final dataset: {len(combined_df)} trials from {len(combined_df['participant_id'].unique())} participants")
        print(f"   Main file: all_participants_final_left_right_drifts.csv")
        
        # Quick preview of the combined data
        print(f"\n📋 Combined dataset preview:")
        print(combined_df.head())
        
        print(f"\n📊 Dataset shape: {combined_df.shape}")
        print(f"   Columns: {list(combined_df.columns)}")
        
    except Exception as e:
        print(f"❌ Merging failed: {e}")
        raise

if __name__ == '__main__':
    main()