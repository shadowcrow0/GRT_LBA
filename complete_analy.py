"""
Complete GRT Three Assumptions Analysis
Tests all three core GRT assumptions: Perceptual Separability, Perceptual Independence, and Decision Boundaries
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set efficient sampling parameters
import pytensor
pytensor.config.floatX = 'float32'

# ============================================================================
# Data Loading (same as before)
# ============================================================================

def load_and_process_grt_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=500):
    """Load and process GRT data for three assumptions analysis"""
    
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
        return generate_simulated_data()
    
    # Data cleaning
    clean_data = raw_data.dropna(subset=required_columns)
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # Convert to analysis format
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        if not (0 <= row['Response'] <= 3):
            continue
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
    
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ No valid data rows, generating simulated data")
        return generate_simulated_data()
    
    print(f"\n✅ Data loading completed:")
    print(f"  Valid trials: {len(df)}")
    print(f"  Number of subjects: {df['subject_id'].nunique()}")
    print(f"  Overall accuracy: {df['accuracy'].mean():.3f}")
    
    return df

def generate_simulated_data(n_trials=200):
    """Generate simulated test data with known GRT violations"""
    print("Generating simulated test data...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # Simulate with some GRT violations
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
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
# Complete GRT Three Assumptions Model
# ============================================================================

def build_complete_grt_model(data):
    """
    Build complete GRT model testing all three assumptions:
    1. Perceptual Separability
    2. Perceptual Independence  
    3. Decision Boundaries
    """
    
    print("Building complete GRT three assumptions model...")
    
    # Prepare data
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    left_pattern = data['left_pattern'].values.astype('int32')
    right_pattern = data['right_pattern'].values.astype('int32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    n_trials = len(data)
    
    print(f"Processing {n_trials} trials")
    
    with pm.Model() as model:
        
        # ================================================================
        # ASSUMPTION 1: PERCEPTUAL SEPARABILITY
        # Test if processing of one dimension affects the other
        # ================================================================
        
        # Separability parameters - how much one dimension affects the other
        sep_left_on_right = pm.Normal('sep_left_on_right', mu=0, sigma=0.2)  # Left affects right processing
        sep_right_on_left = pm.Normal('sep_right_on_left', mu=0, sigma=0.2)  # Right affects left processing
        
        # Base perceptual strength for each dimension
        strength_left_0 = pm.HalfNormal('strength_left_0', sigma=0.5)   # Left dimension, pattern 0
        strength_left_1 = pm.HalfNormal('strength_left_1', sigma=0.5)   # Left dimension, pattern 1
        strength_right_0 = pm.HalfNormal('strength_right_0', sigma=0.5) # Right dimension, pattern 0
        strength_right_1 = pm.HalfNormal('strength_right_1', sigma=0.5) # Right dimension, pattern 1
        
        # ================================================================
        # ASSUMPTION 2: PERCEPTUAL INDEPENDENCE
        # Test correlation between processing of different dimensions
        # ================================================================
        
        # Independence parameters - correlation between dimensions
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)  # Base correlation
        rho_pattern_effect = pm.Normal('rho_pattern_effect', mu=0, sigma=0.2)  # Pattern-dependent correlation
        
        # ================================================================
        # ASSUMPTION 3: DECISION BOUNDARIES
        # Test if decision boundaries are consistent across conditions
        # ================================================================
        
        # Base decision boundaries
        boundary_base = pm.HalfNormal('boundary_base', sigma=0.3)
        
        # Boundary shifts for different conditions
        boundary_shift_left = pm.Normal('boundary_shift_left', mu=0, sigma=0.2)    # Shift based on left pattern
        boundary_shift_right = pm.Normal('boundary_shift_right', mu=0, sigma=0.2)  # Shift based on right pattern
        boundary_shift_interaction = pm.Normal('boundary_shift_interaction', mu=0, sigma=0.1)  # Interaction effect
        
        # ================================================================
        # LBA CORE PARAMETERS
        # ================================================================
        
        A = pm.HalfNormal('A', sigma=0.2)  # Start point variability
        s = pm.HalfNormal('s', sigma=0.2)  # Drift rate variability
        t0 = pm.HalfNormal('t0', sigma=0.1)  # Non-decision time
        
        # ================================================================
        # COMPUTE TRIAL-BY-TRIAL PARAMETERS
        # ================================================================
        
        def compute_trial_parameters():
            # Initialize arrays
            drift_rates = pt.zeros((n_trials, 4))
            boundaries = pt.zeros(n_trials)
            correlations = pt.zeros(n_trials)
            
            for trial in range(n_trials):
                left_val = left_pattern[trial]
                right_val = right_pattern[trial]
                stim_type = stimulus_type[trial]
                
                # 1. SEPARABILITY EFFECTS
                # Compute perceived strength with separability violations
                if left_val == 0:
                    left_strength = strength_left_0 + sep_right_on_left * right_val
                else:
                    left_strength = strength_left_1 + sep_right_on_left * (1 - right_val)
                
                if right_val == 0:
                    right_strength = strength_right_0 + sep_left_on_right * left_val
                else:
                    right_strength = strength_right_1 + sep_left_on_right * (1 - left_val)
                
                # 2. INDEPENDENCE EFFECTS  
                # Correlation depends on stimulus pattern
                trial_correlation = rho_base + rho_pattern_effect * (left_val * right_val)
                correlations = pt.set_subtensor(correlations[trial], trial_correlation)
                
                # 3. DECISION BOUNDARY EFFECTS
                # Boundary shifts based on stimulus properties
                trial_boundary = (boundary_base + 
                                boundary_shift_left * left_val + 
                                boundary_shift_right * right_val +
                                boundary_shift_interaction * left_val * right_val)
                boundaries = pt.set_subtensor(boundaries[trial], pt.maximum(trial_boundary, 0.1))
                
                # 4. COMPUTE DRIFT RATES FOR EACH ACCUMULATOR
                for acc in range(4):
                    acc_left = acc // 2
                    acc_right = acc % 2
                    
                    # Base drift from perceptual evidence
                    base_drift = 0.1
                    
                    # Evidence from left dimension
                    if acc_left == left_val:
                        base_drift += left_strength
                    else:
                        base_drift += left_strength * 0.2  # Reduced for incorrect
                    
                    # Evidence from right dimension  
                    if acc_right == right_val:
                        base_drift += right_strength
                    else:
                        base_drift += right_strength * 0.2  # Reduced for incorrect
                    
                    # Correlation effect
                    correlation_boost = pt.abs(trial_correlation) * 0.15
                    if acc == stim_type:
                        base_drift += correlation_boost
                    
                    # Ensure positive drift
                    final_drift = pt.maximum(base_drift, 0.05)
                    drift_rates = pt.set_subtensor(drift_rates[trial, acc], final_drift)
            
            return drift_rates, boundaries, correlations
        
        drift_rates, boundaries, correlations = compute_trial_parameters()
        
        # ================================================================
        # LIKELIHOOD COMPUTATION
        # ================================================================
        
        def lba_logp():
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            trial_boundaries = boundaries
            
            # Time likelihood (simplified exponential)
            lambda_param = chosen_drifts / trial_boundaries
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # Choice likelihood (softmax with boundaries)
            choice_logits = drift_rates / trial_boundaries.reshape((-1, 1)) * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        pm.Potential('likelihood', lba_logp())
        
        # ================================================================
        # DERIVED QUANTITIES FOR HYPOTHESIS TESTING
        # ================================================================
        
        # Separability tests
        pm.Deterministic('separability_violation', pt.sqrt(sep_left_on_right**2 + sep_right_on_left**2))
        pm.Deterministic('separability_asymmetry', pt.abs(sep_left_on_right - sep_right_on_left))
        
        # Independence tests
        pm.Deterministic('independence_violation', pt.abs(rho_base))
        pm.Deterministic('pattern_dependency', pt.abs(rho_pattern_effect))
        
        # Decision boundary tests
        pm.Deterministic('boundary_inconsistency', 
                        pt.sqrt(boundary_shift_left**2 + boundary_shift_right**2 + boundary_shift_interaction**2))
        pm.Deterministic('boundary_main_effects', pt.sqrt(boundary_shift_left**2 + boundary_shift_right**2))
        pm.Deterministic('boundary_interaction_effect', pt.abs(boundary_shift_interaction))
    
    return model

# ============================================================================
# Three Assumptions Analysis
# ============================================================================

def analyze_three_assumptions(trace, data):
    """
    Analyze results for all three GRT assumptions
    """
    
    print("\n" + "="*80)
    print("🔬 COMPLETE GRT THREE ASSUMPTIONS ANALYSIS")
    print("="*80)
    
    posterior = trace.posterior
    
    # Extract parameters
    sep_left_on_right = posterior['sep_left_on_right'].values.flatten()
    sep_right_on_left = posterior['sep_right_on_left'].values.flatten()
    separability_violation = posterior['separability_violation'].values.flatten()
    
    rho_base = posterior['rho_base'].values.flatten()
    rho_pattern_effect = posterior['rho_pattern_effect'].values.flatten()
    independence_violation = posterior['independence_violation'].values.flatten()
    
    boundary_shift_left = posterior['boundary_shift_left'].values.flatten()
    boundary_shift_right = posterior['boundary_shift_right'].values.flatten()
    boundary_shift_interaction = posterior['boundary_shift_interaction'].values.flatten()
    boundary_inconsistency = posterior['boundary_inconsistency'].values.flatten()
    
    results = {}
    
    print("\n" + "="*60)
    print("📊 ASSUMPTION 1: PERCEPTUAL SEPARABILITY")
    print("="*60)
    
    # Separability hypothesis tests
    sep_violation_mean = np.mean(separability_violation)
    sep_violation_ci = np.percentile(separability_violation, [2.5, 97.5])
    sep_prob_violation = np.mean(separability_violation > 0.1)
    
    print(f"Separability violation magnitude: {sep_violation_mean:.4f} [{sep_violation_ci[0]:.4f}, {sep_violation_ci[1]:.4f}]")
    print(f"Left→Right effect: {np.mean(sep_left_on_right):.4f} [{np.percentile(sep_left_on_right, [2.5, 97.5])[0]:.4f}, {np.percentile(sep_left_on_right, [2.5, 97.5])[1]:.4f}]")
    print(f"Right→Left effect: {np.mean(sep_right_on_left):.4f} [{np.percentile(sep_right_on_left, [2.5, 97.5])[0]:.4f}, {np.percentile(sep_right_on_left, [2.5, 97.5])[1]:.4f}]")
    print(f"Probability of separability violation (>0.1): {sep_prob_violation:.3f}")
    
    if sep_prob_violation > 0.95:
        sep_conclusion = "STRONG VIOLATION"
        print("🚨 STRONG VIOLATION of perceptual separability")
    elif sep_prob_violation > 0.8:
        sep_conclusion = "MODERATE VIOLATION"
        print("⚠️  MODERATE VIOLATION of perceptual separability")
    elif sep_prob_violation > 0.5:
        sep_conclusion = "WEAK VIOLATION"
        print("⚠️  WEAK VIOLATION of perceptual separability")
    else:
        sep_conclusion = "SUPPORTS SEPARABILITY"
        print("✅ SUPPORTS perceptual separability assumption")
    
    results['separability'] = {
        'violation_magnitude': sep_violation_mean,
        'violation_ci': sep_violation_ci,
        'prob_violation': sep_prob_violation,
        'conclusion': sep_conclusion,
        'left_on_right': np.mean(sep_left_on_right),
        'right_on_left': np.mean(sep_right_on_left)
    }
    
    print("\n" + "="*60)
    print("📊 ASSUMPTION 2: PERCEPTUAL INDEPENDENCE")
    print("="*60)
    
    # Independence hypothesis tests
    indep_violation_mean = np.mean(independence_violation)
    indep_violation_ci = np.percentile(independence_violation, [2.5, 97.5])
    indep_prob_violation = np.mean(independence_violation > 0.1)
    pattern_effect_significant = not (np.percentile(rho_pattern_effect, 2.5) < 0 < np.percentile(rho_pattern_effect, 97.5))
    
    print(f"Independence violation magnitude: {indep_violation_mean:.4f} [{indep_violation_ci[0]:.4f}, {indep_violation_ci[1]:.4f}]")
    print(f"Base correlation: {np.mean(rho_base):.4f} [{np.percentile(rho_base, [2.5, 97.5])[0]:.4f}, {np.percentile(rho_base, [2.5, 97.5])[1]:.4f}]")
    print(f"Pattern-dependent correlation effect: {np.mean(rho_pattern_effect):.4f} [{np.percentile(rho_pattern_effect, [2.5, 97.5])[0]:.4f}, {np.percentile(rho_pattern_effect, [2.5, 97.5])[1]:.4f}]")
    print(f"Probability of independence violation (|ρ| > 0.1): {indep_prob_violation:.3f}")
    print(f"Pattern effect significant: {'Yes' if pattern_effect_significant else 'No'}")
    
    if indep_prob_violation > 0.95:
        indep_conclusion = "STRONG VIOLATION"
        print("🚨 STRONG VIOLATION of perceptual independence")
    elif indep_prob_violation > 0.8:
        indep_conclusion = "MODERATE VIOLATION"  
        print("⚠️  MODERATE VIOLATION of perceptual independence")
    elif indep_prob_violation > 0.5:
        indep_conclusion = "WEAK VIOLATION"
        print("⚠️  WEAK VIOLATION of perceptual independence")
    else:
        indep_conclusion = "SUPPORTS INDEPENDENCE"
        print("✅ SUPPORTS perceptual independence assumption")
    
    results['independence'] = {
        'violation_magnitude': indep_violation_mean,
        'violation_ci': indep_violation_ci,
        'prob_violation': indep_prob_violation,
        'conclusion': indep_conclusion,
        'base_correlation': np.mean(rho_base),
        'pattern_effect': np.mean(rho_pattern_effect),
        'pattern_effect_significant': pattern_effect_significant
    }
    
    print("\n" + "="*60)
    print("📊 ASSUMPTION 3: DECISION BOUNDARIES")
    print("="*60)
    
    # Decision boundary hypothesis tests
    boundary_violation_mean = np.mean(boundary_inconsistency)
    boundary_violation_ci = np.percentile(boundary_inconsistency, [2.5, 97.5])
    boundary_prob_violation = np.mean(boundary_inconsistency > 0.1)
    
    left_effect_significant = not (np.percentile(boundary_shift_left, 2.5) < 0 < np.percentile(boundary_shift_left, 97.5))
    right_effect_significant = not (np.percentile(boundary_shift_right, 2.5) < 0 < np.percentile(boundary_shift_right, 97.5))
    interaction_significant = not (np.percentile(boundary_shift_interaction, 2.5) < 0 < np.percentile(boundary_shift_interaction, 97.5))
    
    print(f"Boundary inconsistency magnitude: {boundary_violation_mean:.4f} [{boundary_violation_ci[0]:.4f}, {boundary_violation_ci[1]:.4f}]")
    print(f"Left pattern boundary shift: {np.mean(boundary_shift_left):.4f} [{np.percentile
