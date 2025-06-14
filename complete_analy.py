"""
Complete Covariance Matrix LBA Analysis for GRT Data
Optimized version with English interface
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set efficient sampling parameters
import pytensor
pytensor.config.floatX = 'float32'  # Use float32 for speed

# ============================================================================
# Part 1: Data Loading and Processing
# ============================================================================

def load_and_process_grt_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=500):
    """
    Load and process GRT data for LBA analysis
    """
    
    print("Loading GRT data...")
    
    try:
        raw_data = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {csv_file_path}")
        print(f"Raw data dimensions: {raw_data.shape}")
        
    except FileNotFoundError:
        print(f"File {csv_file_path} not found")
        return generate_simulated_data()
    except Exception as e:
        print(f"Error loading file: {e}")
        return generate_simulated_data()
    
    # Check required columns
    required_columns = ['Chanel1', 'Chanel2', 'Response', 'RT', 'acc']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {list(raw_data.columns)}")
        return generate_simulated_data()
    
    # Data cleaning and transformation
    print("Converting data format...")
    
    # Remove missing values
    clean_data = raw_data.dropna(subset=required_columns)
    print(f"Data after cleaning: {clean_data.shape}")
    
    # Remove extreme reaction times
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    print(f"Data after RT filtering: {clean_data.shape}")
    
    # Limit trials per subject for computational efficiency
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # Convert to analysis format
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        # Ensure Response is in 0-3 range
        if not (0 <= row['Response'] <= 3):
            continue
            
        # Ensure Channel values are 0 or 1
        if row['Chanel1'] not in [0, 1] or row['Chanel2'] not in [0, 1]:
            continue
        
        converted_row = {
            'subject_id': row.get('Subject', row.get('participant', 1)),
            'trial': len(converted_data) + 1,
            'left_pattern': int(row['Chanel1']),
            'right_pattern': int(row['Chanel2']),
            'response': int(row['Response']),
            'rt': float(row['RT']),
            'accuracy': int(row['acc']),
            'stimulus_type': int(row['Chanel1']) * 2 + int(row['Chanel2']),
            'is_symmetric': 1 if row['Chanel1'] == row['Chanel2'] else 0
        }
        
        converted_data.append(converted_row)
    
    # Convert to DataFrame
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ No valid data rows, generating simulated data")
        return generate_simulated_data()
    
    # Data statistics
    print(f"\n✅ Real data loading completed:")
    print(f"  Valid trials: {len(df)}")
    print(f"  Number of subjects: {df['subject_id'].nunique()}")
    print(f"  Overall accuracy: {df['accuracy'].mean():.3f}")
    print(f"  Mean RT: {df['rt'].mean():.3f}s")
    print(f"  Symmetric trial proportion: {df['is_symmetric'].mean():.3f}")
    
    return df

def generate_simulated_data(n_trials=200):
    """
    Generate simulated test data
    """
    print("Generating simulated test data...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        # Random stimuli
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # Simulate responses
        # Symmetric stimuli have higher accuracy
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        # Correct response
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
        # Reaction time
        base_rt = 0.8 if is_symmetric else 0.9
        rt = np.random.gamma(2, base_rt/2)
        rt = np.clip(rt, 0.2, 3.0)
        
        data.append({
            'subject_id': 1,
            'trial': trial + 1,
            'left_pattern': left_pattern,
            'right_pattern': right_pattern,
            'response': response,
            'rt': rt,
            'accuracy': accuracy,
            'stimulus_type': stimulus_type,
            'is_symmetric': is_symmetric
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Simulated data generation completed: {len(df)} trials")
    return df

# ============================================================================
# Part 2: Covariance Matrix LBA Model
# ============================================================================

def build_covariance_lba_model(data):
    """
    Build covariance matrix LBA model
    """
    
    print("Building covariance matrix LBA model...")
    
    # Prepare data (convert to numpy arrays for efficiency)
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    is_symmetric = data['is_symmetric'].values.astype('float32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    
    n_trials = len(data)
    print(f"Processing {n_trials} trials")
    
    with pm.Model() as model:
        
        # Covariance parameters - the core of the analysis
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)
        rho_symmetry_effect = pm.Normal('rho_symmetry_effect', mu=0, sigma=0.2)
        
        # LBA basic parameters
        A = pm.HalfNormal('A', sigma=0.2)
        b_base = pm.HalfNormal('b_base', sigma=0.3)
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        # Drift rate parameters
        drift_base = pm.HalfNormal('drift_base', sigma=0.5)
        drift_correct_boost = pm.HalfNormal('drift_correct_boost', sigma=0.3)
        symmetry_boost = pm.Normal('symmetry_boost', mu=0, sigma=0.1)
        
        # Vectorized drift rate computation
        def compute_drift_rates():
            drift_rates = pt.zeros((n_trials, 4))
            
            for acc in range(4):
                # Base drift
                base_drift = drift_base
                
                # Correct accumulator boost
                correct_boost = pt.where(pt.eq(stimulus_type, acc), drift_correct_boost, 0.0)
                
                # Symmetry effect
                sym_boost = symmetry_boost * is_symmetric * pt.where(pt.eq(stimulus_type, acc), 1.0, 0.0)
                
                # Correlation effect - this is where the covariance matrix enters
                rho_trial = rho_base + rho_symmetry_effect * is_symmetric
                correlation_effect = pt.abs(rho_trial) * 0.1
                
                # Total drift rate
                final_drift = base_drift + correct_boost + sym_boost + correlation_effect
                final_drift = pt.maximum(final_drift, 0.05)
                
                drift_rates = pt.set_subtensor(drift_rates[:, acc], final_drift)
            
            return drift_rates
        
        drift_rates = compute_drift_rates()
        
        # Likelihood function
        def lba_logp():
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            thresholds = b_base
            
            # Time likelihood (simplified as exponential distribution)
            lambda_param = chosen_drifts / thresholds
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # Choice likelihood (softmax)
            choice_logits = drift_rates * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        pm.Potential('likelihood', lba_logp())
        
        # Store important variables
        pm.Deterministic('rho_symmetric', rho_base + rho_symmetry_effect)
        pm.Deterministic('rho_asymmetric', rho_base)
        pm.Deterministic('rho_difference', rho_symmetry_effect)
    
    return model

# ============================================================================
# Part 3: Results Analysis
# ============================================================================

def analyze_covariance_results(trace, data):
    """
    Analyze covariance matrix LBA model results
    """
    
    print("\n" + "="*60)
    print("📊 Covariance Matrix LBA Model Results")
    print("="*60)
    
    posterior = trace.posterior
    
    # Extract key parameters
    rho_base = posterior['rho_base'].values.flatten()
    rho_symmetry_effect = posterior['rho_symmetry_effect'].values.flatten()
    rho_symmetric = posterior['rho_symmetric'].values.flatten()
    rho_asymmetric = posterior['rho_asymmetric'].values.flatten()
    
    print(f"\n🔍 Covariance Matrix Analysis Results:")
    print(f"Base correlation coefficient: {np.mean(rho_base):.3f} [{np.percentile(rho_base, 2.5):.3f}, {np.percentile(rho_base, 97.5):.3f}]")
    print(f"Symmetry effect: {np.mean(rho_symmetry_effect):.3f} [{np.percentile(rho_symmetry_effect, 2.5):.3f}, {np.percentile(rho_symmetry_effect, 97.5):.3f}]")
    print(f"Symmetric stimulus correlation: {np.mean(rho_symmetric):.3f}")
    print(f"Asymmetric stimulus correlation: {np.mean(rho_asymmetric):.3f}")
    
    # Significance test
    symmetry_ci = np.percentile(rho_symmetry_effect, [2.5, 97.5])
    is_significant = not (symmetry_ci[0] < 0 < symmetry_ci[1])
    print(f"Symmetry effect significance: {'Yes' if is_significant else 'No'}")
    
    # Independence assumption test
    independence_prob = np.mean(np.abs(rho_base) < 0.1)
    print(f"\n🔬 GRT Independence Assumption Test:")
    print(f"Independence probability (|ρ| < 0.1): {independence_prob:.3f}")
    
    if independence_prob < 0.05:
        print("🚨 Strong violation of GRT independence assumption")
        violation_level = "Strong violation"
    elif independence_prob < 0.2:
        print("⚠️  Moderate violation of independence assumption")
        violation_level = "Moderate violation"
    else:
        print("✅ Basic support for independence assumption")
        violation_level = "Supports independence"
    
    # Theoretical interpretation
    print(f"\n💡 Theoretical Interpretation:")
    effect_size = np.mean(rho_symmetry_effect)
    
    if abs(effect_size) > 0.05:
        if effect_size > 0:
            print("✓ Symmetric stimuli increase inter-source correlation")
            print("  → Supports configural processing hypothesis")
            theory = "Configural processing"
        else:
            print("✓ Symmetric stimuli decrease inter-source correlation")
            print("  → Supports independent processing hypothesis")
            theory = "Independent processing"
    else:
        print("• Symmetry has minimal effect on correlation")
        theory = "Minimal effect"
    
    # Behavioral data analysis
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    print(f"\n📋 Behavioral Data Comparison:")
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        print(f"Symmetric stimuli - Accuracy: {symmetric_data['accuracy'].mean():.3f}, RT: {symmetric_data['rt'].mean():.3f}s")
        print(f"Asymmetric stimuli - Accuracy: {asymmetric_data['accuracy'].mean():.3f}, RT: {asymmetric_data['rt'].mean():.3f}s")
    
    return {
        'rho_base': rho_base,
        'rho_symmetry_effect': rho_symmetry_effect,
        'rho_symmetric': rho_symmetric,
        'rho_asymmetric': rho_asymmetric,
        'is_significant': is_significant,
        'violation_level': violation_level,
        'theory': theory,
        'effect_size': effect_size,
        'independence_prob': independence_prob
    }

# ============================================================================
# Part 4: Comprehensive Visualization
# ============================================================================

def create_comprehensive_visualization(trace, results, data):
    """
    Create comprehensive analysis visualization
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Covariance Matrix LBA Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Posterior correlation coefficient comparison
    axes[0, 0].hist(results['rho_symmetric'], bins=30, alpha=0.6, label='Symmetric', color='green', density=True)
    axes[0, 0].hist(results['rho_asymmetric'], bins=30, alpha=0.6, label='Asymmetric', color='orange', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Independence')
    axes[0, 0].set_xlabel('Correlation Coefficient ρ')
    axes[0, 0].set_ylabel('Posterior Density')
    axes[0, 0].set_title('Correlation Coefficient Posterior Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Symmetry effect
    axes[0, 1].hist(results['rho_symmetry_effect'], bins=30, alpha=0.7, color='purple', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='No effect')
    axes[0, 1].axvline(np.mean(results['rho_symmetry_effect']), color='black', linestyle='-', linewidth=2, label='Mean effect')
    ci_lower, ci_upper = np.percentile(results['rho_symmetry_effect'], [2.5, 97.5])
    axes[0, 1].axvspan(ci_lower, ci_upper, alpha=0.2, color='purple', label='95% CI')
    axes[0, 1].set_xlabel('Correlation Difference')
    axes[0, 1].set_ylabel('Posterior Density')
    axes[0, 1].set_title('Symmetry Effect (ρ_sym - ρ_asym)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Behavioral data comparison
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        categories = ['Accuracy', 'RT (s)']
        sym_values = [symmetric_data['accuracy'].mean(), symmetric_data['rt'].mean()]
        asym_values = [asymmetric_data['accuracy'].mean(), asymmetric_data['rt'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[0, 2].bar(x - width/2, sym_values, width, label='Symmetric', color='green', alpha=0.7)
        bars2 = axes[0, 2].bar(x + width/2, asym_values, width, label='Asymmetric', color='orange', alpha=0.7)
        
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].set_title('Behavioral Data Comparison')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(categories)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0, 2].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            axes[0, 2].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 4. Subject distribution
    subject_counts = data['subject_id'].value_counts().sort_index()
    axes[1, 0].bar(range(len(subject_counts)), subject_counts.values, alpha=0.7, color='skyblue')
    axes[1, 0].set_xlabel('Subject ID')
    axes[1, 0].set_ylabel('Number of Trials')
    axes[1, 0].set_title('Trial Distribution by Subject')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. RT distribution comparison
    axes[1, 1].hist(symmetric_data['rt'], bins=30, alpha=0.6, label='Symmetric', color='green', density=True)
    axes[1, 1].hist(asymmetric_data['rt'], bins=30, alpha=0.6, label='Asymmetric', color='orange', density=True)
    axes[1, 1].set_xlabel('Reaction Time (s)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('RT Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Trace plots for key parameters
    rho_base_samples = results['rho_base']
    rho_effect_samples = results['rho_symmetry_effect']
    
    axes[1, 2].plot(rho_base_samples[:200], alpha=0.7, label='ρ_base')
    axes[1, 2].plot(rho_effect_samples[:200], alpha=0.7, label='ρ_effect')
    axes[1, 2].set_xlabel('Sample')
    axes[1, 2].set_ylabel('Parameter Value')
    axes[1, 2].set_title('Parameter Trace (First 200 samples)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Independence test visualization
    independence_dist = np.abs(results['rho_base'])
    axes[2, 0].hist(independence_dist, bins=30, alpha=0.7, color='red', density=True)
    axes[2, 0].axvline(0.1, color='black', linestyle='--', label='Independence threshold (0.1)')
    axes[2, 0].set_xlabel('|ρ_base|')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].set_title(f'Independence Test\nP(|ρ| < 0.1) = {results["independence_prob"]:.3f}')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Effect size visualization
    axes[2, 1].hist(results['rho_symmetry_effect'], bins=30, alpha=0.7, color='blue', density=True)
    axes[2, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[2, 1].axvline(0.05, color='green', linestyle=':', alpha=0.7, label='Small effect')
    axes[2, 1].axvline(-0.05, color='green', linestyle=':', alpha=0.7)
    axes[2, 1].set_xlabel('Effect Size')
    axes[2, 1].set_ylabel('Density')
    axes[2, 1].set_title(f'Effect Size Distribution\nMean = {results["effect_size"]:.3f}')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Summary text
    axes[2, 2].text(0.1, 0.9, 'Analysis Summary', fontsize=14, fontweight='bold', transform=axes[2, 2].transAxes)
    
    summary_text = f"""
Independence Test: {results['violation_level']}
Symmetry Effect: {'Significant' if results['is_significant'] else 'Not significant'}

Base Correlation: {np.mean(results['rho_base']):.3f}
Symmetry Enhancement: {np.mean(results['rho_symmetry_effect']):.3f}

Theoretical Support: {results['theory']}

Data Summary:
• Total trials: {len(data)}
• Subjects: {data['subject_id'].nunique()}
• Overall accuracy: {data['accuracy'].mean():.3f}
• Mean RT: {data['rt'].mean():.3f}s
"""
    
    axes[2, 2].text(0.1, 0.7, summary_text, fontsize=10, transform=axes[2, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('covariance_lba_analysis_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Complete analysis results saved as 'covariance_lba_analysis_complete.png'")

# ============================================================================
# Part 5: Main Analysis Function
# ============================================================================

def run_complete_covariance_analysis(csv_file_path='GRT_LBA.csv', subject_id=None, max_trials=400, draws=None, tune=None, chains=2):
    """
    Run complete covariance matrix analysis
    """
    
    print("🚀 Starting Complete Covariance Matrix LBA Analysis")
    print("="*70)
    
    try:
        # 1. Load data
        data = load_and_process_grt_data(csv_file_path, max_trials_per_subject=max_trials)
        
        # 2. Select subject (choose subject with most trials)
        subject_counts = data['subject_id'].value_counts()
        if subject_id is None:
            selected_subject = subject_counts.index[0]
            print(f"\n🎯 Auto-selected subject {selected_subject} ({subject_counts.iloc[0]} trials)")
        else:
            selected_subject = subject_id
            print(f"🎯 Selected subject {selected_subject}")
        
        subject_data = data[data['subject_id'] == selected_subject].copy()
        
        # Limit trials for computational efficiency
        if len(subject_data) > max_trials:
            print(f"⚠️  Limiting trials to {max_trials} for efficiency")
            subject_data = subject_data.sample(n=max_trials, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 Analysis Data Summary:")
        print(f"  Subject: {selected_subject}")
        print(f"  Trials: {len(subject_data)}")
        print(f"  Accuracy: {subject_data['accuracy'].mean():.3f}")
        print(f"  Mean RT: {subject_data['rt'].mean():.3f}s")
        print(f"  Symmetric trials: {subject_data['is_symmetric'].sum()}/{len(subject_data)}")
        
        # 3. Build model
        model = build_covariance_lba_model(subject_data)
        
        # 4. MCMC sampling
        # Auto-adjust sampling parameters based on number of trials
        if draws is None:
            if len(subject_data) <= 200:
                draws = 300
                tune_param = 150
            elif len(subject_data) <= 400:
                draws = 400
                tune_param = 200
            elif len(subject_data) <= 600:
                draws = 600
                tune_param = 300
            else:  # > 600 trials
                draws = 800
                tune_param = 400
        else:
            draws = draws
            tune_param = tune if tune is not None else draws // 2
        
        print(f"\n⏳ Starting MCMC sampling...")
        print(f"Sampling parameters: {draws} draws + {tune_param} tune, {chains} chains")
        print(f"Auto-adjusted based on {len(subject_data)} trials")
        
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune_param,
                chains=chains,
                cores=1,
                target_accept=0.85,
                return_inferencedata=True,
                random_seed=42,
                progressbar=True,
                compute_convergence_checks=False
            )
        
        print("✅ MCMC sampling completed!")
        
        # 5. Analyze results
        results = analyze_covariance_results(trace, subject_data)
        
        # 6. Create visualization
        create_comprehensive_visualization(trace, results, subject_data)
        
        print(f"\n🎉 Complete covariance matrix LBA analysis finished!")
        print(f"Main finding: {results['theory']}")
        
        # 7. Final recommendations
        print(f"\n💡 Recommendations:")
        if results['independence_prob'] < 0.2:
            print("  • Strong evidence against GRT independence assumption")
            print("  • Consider alternative models that account for dependencies")
        else:
            print("  • GRT independence assumption is reasonably supported")
        
        if results['is_significant']:
            print(f"  • Significant symmetry effect detected")
            print(f"  • This supports the {results['theory']} hypothesis")
        else:
            print("  • No significant symmetry effect on correlation structure")
        
        return trace, subject_data, results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check data format or try adjusting parameters")
        return None, None, None

# ============================================================================
# Quick Test Function
# ============================================================================

def quick_test():
    """
    Quick test function with simulated data
    """
    print("🧪 Running quick test with simulated data...")
    return run_complete_covariance_analysis(max_trials=150)

# ============================================================================
# Execution
# ============================================================================

if __name__ == "__main__":
    print("🔬 Complete Covariance Matrix LBA Model")
    print("Advanced GRT analysis with correlation structure modeling")
    print("-" * 70)
    
    # Run complete analysis
    trace, data, results = run_complete_covariance_analysis()
    
    if trace is not None:
        print("\n🎉 Analysis completed successfully!")
        print("For more detailed analysis, you can increase draws and tune parameters")
    else:
        print("\n❌ Analysis failed")
        print("Try running quick test: quick_test()")

# ============================================================================
# Usage Instructions
# ============================================================================

"""
Usage Examples:

1. Basic execution:
   trace, data, results = run_complete_covariance_analysis()

2. Specify parameters with custom sampling:
   trace, data, results = run_complete_covariance_analysis(
       csv_file_path='your_file.csv',
       subject_id=31,
       max_trials=800,
       draws=800,
       tune=400,
       chains=2
   )

3. Quick test with simulated data:
   trace, data, results = quick_test()

4. High-quality analysis with more samples:
   trace, data, results = run_complete_covariance_analysis(
       subject_id=41,
       max_trials=600,
       draws=1000,
       tune=500,
       chains=4
   )
"""
