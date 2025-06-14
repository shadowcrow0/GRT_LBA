"""
Execute Covariance Matrix LBA Analysis - Optimized Analysis for Your GRT Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set font parameters for English
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_data():
    """
    Load and analyze your GRT data
    """
    print("🔄 Loading GRT_LBA.csv data...")
    
    # Read data
    data = pd.read_csv('GRT_LBA.csv')
    print(f"✅ Successfully loaded data: {data.shape[0]} records")
    print(f"📊 Columns: {list(data.columns)}")
    
    # Basic statistics
    print(f"\n📈 Data overview:")
    print(f"  Number of subjects: {data['Subject'].nunique()}")
    print(f"  Overall accuracy: {data['acc'].mean():.3f}")
    print(f"  Mean reaction time: {data['RT'].mean():.3f} seconds")
    print(f"  RT range: {data['RT'].min():.3f} - {data['RT'].max():.3f} seconds")
    
    # Check trial distribution per subject
    subject_counts = data['Subject'].value_counts().sort_values(ascending=False)
    print(f"\n👥 Subject trial distribution (top 5):")
    for i, (subject, count) in enumerate(subject_counts.head().items()):
        print(f"  Subject {subject}: {count} trials")
    
    return data, subject_counts

def analyze_subject_performance(data):
    """
    Analyze performance of each subject
    """
    print(f"\n🎯 Individual subject performance analysis:")
    
    subject_stats = []
    for subject in data['Subject'].unique():
        subj_data = data[data['Subject'] == subject]
        
        # Basic statistics
        accuracy = subj_data['acc'].mean()
        mean_rt = subj_data['RT'].mean()
        n_trials = len(subj_data)
        
        # Symmetry analysis
        symmetric_trials = subj_data[subj_data['Chanel1'] == subj_data['Chanel2']]
        asymmetric_trials = subj_data[subj_data['Chanel1'] != subj_data['Chanel2']]
        
        sym_acc = symmetric_trials['acc'].mean() if len(symmetric_trials) > 0 else 0
        asym_acc = asymmetric_trials['acc'].mean() if len(asymmetric_trials) > 0 else 0
        sym_rt = symmetric_trials['RT'].mean() if len(symmetric_trials) > 0 else 0
        asym_rt = asymmetric_trials['RT'].mean() if len(asymmetric_trials) > 0 else 0
        
        subject_stats.append({
            'Subject': subject,
            'N_trials': n_trials,
            'Overall_Acc': accuracy,
            'Overall_RT': mean_rt,
            'Sym_Acc': sym_acc,
            'Asym_Acc': asym_acc,
            'Sym_RT': sym_rt,
            'Asym_RT': asym_rt,
            'Sym_Count': len(symmetric_trials),
            'Asym_Count': len(asymmetric_trials),
            'Symmetry_Effect_Acc': sym_acc - asym_acc,
            'Symmetry_Effect_RT': sym_rt - asym_rt
        })
    
    stats_df = pd.DataFrame(subject_stats)
    stats_df = stats_df.sort_values('N_trials', ascending=False)
    
    # Display detailed statistics for top 10 subjects
    print(f"Detailed statistics for top 10 subjects:")
    for _, row in stats_df.head(10).iterrows():
        print(f"Subject {int(row['Subject'])}: {int(row['N_trials'])} trials, "
              f"Accuracy={row['Overall_Acc']:.3f}, RT={row['Overall_RT']:.3f}s, "
              f"Symmetry Effect (Acc)={row['Symmetry_Effect_Acc']:.3f}")
    
    return stats_df

def detailed_symmetry_analysis(data):
    """
    Detailed symmetry effect analysis
    """
    print(f"\n🔍 Detailed symmetry effect analysis:")
    
    # Overall symmetry effects
    symmetric_data = data[data['Chanel1'] == data['Chanel2']]
    asymmetric_data = data[data['Chanel1'] != data['Chanel2']]
    
    print(f"Overall analysis:")
    print(f"  Symmetric trials: {len(symmetric_data)} ({len(symmetric_data)/len(data)*100:.1f}%)")
    print(f"  Asymmetric trials: {len(asymmetric_data)} ({len(asymmetric_data)/len(data)*100:.1f}%)")
    
    print(f"\nAccuracy comparison:")
    sym_acc = symmetric_data['acc'].mean()
    asym_acc = asymmetric_data['acc'].mean()
    print(f"  Symmetric stimulus accuracy: {sym_acc:.3f}")
    print(f"  Asymmetric stimulus accuracy: {asym_acc:.3f}")
    print(f"  Difference: {sym_acc - asym_acc:.3f}")
    
    print(f"\nReaction time comparison:")
    sym_rt = symmetric_data['RT'].mean()
    asym_rt = asymmetric_data['RT'].mean()
    print(f"  Symmetric stimulus RT: {sym_rt:.3f} seconds")
    print(f"  Asymmetric stimulus RT: {asym_rt:.3f} seconds")
    print(f"  Difference: {sym_rt - asym_rt:.3f} seconds")
    
    # Statistical tests
    from scipy.stats import ttest_ind
    
    # Accuracy t-test
    t_stat_acc, p_val_acc = ttest_ind(symmetric_data['acc'], asymmetric_data['acc'])
    print(f"\nStatistical test results:")
    print(f"  Accuracy difference t-test: t={t_stat_acc:.3f}, p={p_val_acc:.3f}")
    
    # RT t-test
    t_stat_rt, p_val_rt = ttest_ind(symmetric_data['RT'], asymmetric_data['RT'])
    print(f"  Reaction time difference t-test: t={t_stat_rt:.3f}, p={p_val_rt:.3f}")
    
    return {
        'symmetric_data': symmetric_data,
        'asymmetric_data': asymmetric_data,
        'sym_acc': sym_acc,
        'asym_acc': asym_acc,
        'sym_rt': sym_rt,
        'asym_rt': asym_rt,
        'acc_effect': sym_acc - asym_acc,
        'rt_effect': sym_rt - asym_rt,
        'p_val_acc': p_val_acc,
        'p_val_rt': p_val_rt
    }

def create_comprehensive_visualization(data, symmetry_results, subject_stats):
    """
    Create comprehensive analysis visualization
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('GRT Data Covariance Analysis: Behavioral Results and Symmetry Effects', fontsize=16, fontweight='bold')
    
    # 1. Overall accuracy distribution
    axes[0, 0].hist(data['acc'], bins=2, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Accuracy (0=Incorrect, 1=Correct)')
    axes[0, 0].set_ylabel('Number of Trials')
    axes[0, 0].set_title(f'Overall Accuracy Distribution\nMean Accuracy: {data["acc"].mean():.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Reaction time distribution
    axes[0, 1].hist(data['RT'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Reaction Time (seconds)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Reaction Time Distribution\nMean RT: {data["RT"].mean():.3f}s')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Subject trial distribution
    subject_counts = data['Subject'].value_counts().sort_index()
    axes[0, 2].bar(range(len(subject_counts)), subject_counts.values, alpha=0.7, color='orange')
    axes[0, 2].set_xlabel('Subject ID')
    axes[0, 2].set_ylabel('Number of Trials')
    axes[0, 2].set_title('Trial Distribution by Subject')
    axes[0, 2].set_xticks(range(len(subject_counts)))
    axes[0, 2].set_xticklabels([f'S{s}' for s in subject_counts.index], rotation=45)
    axes[0, 2].grid(True, alpha=0
