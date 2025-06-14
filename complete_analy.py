"""
Proper Hierarchical GRT-LBA Model with Dimension-Based Architecture
Tests three GRT assumptions: Decision Boundary Independence, Perceptual Separability, Perceptual Independence
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_grt_data(file_path, max_participants=None, max_trials_per_participant=None):
    """
    Load and prepare GRT data with proper dimensional structure
    """
    print("Loading and preparing GRT data...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original data: {len(df)} trials from {df['participant'].nunique()} participants")
    
    # Data cleaning
    df = df[(df['RT'] > 0.1) & (df['RT'] < 3.0)]
    print(f"After RT filtering: {len(df)} trials")
    
    # Optional: limit for testing
    if max_participants:
        participants = sorted(df['participant'].unique())[:max_participants]
        df = df[df['participant'].isin(participants)]
        print(f"Limited to {max_participants} participants: {len(df)} trials")
    
    if max_trials_per_participant:
        df = df.groupby('participant').head(max_trials_per_participant).reset_index(drop=True)
        print(f"Limited to {max_trials_per_participant} trials per participant: {len(df)} trials")
    
    # Create participant indices
    participants = sorted(df['participant'].unique())
    participant_map = {p: i for i, p in enumerate(participants)}
    df['participant_idx'] = df['participant'].map(participant_map)
    
    # GRT dimensional structure
    # Channel1 = Left dimension (0=Low, 1=High)
    # Channel2 = Right dimension (0=Low, 1=High)
    df['left_dim'] = df['Chanel1'].astype(int)
    df['right_dim'] = df['Chanel2'].astype(int)
    df['stimulus'] = df['stim_condition'].astype(int)
    df['response'] = df['Response'].astype(int)
    
    # Verify GRT mapping
    print(f"\nGRT Stimulus Mapping Verification:")
    for stim in range(4):
        stim_data = df[df['stimulus'] == stim].iloc[0]
        print(f"Stimulus {stim}: Left={stim_data['left_dim']}, Right={stim_data['right_dim']}")
    
    # Convert to arrays for PyMC
    data = {
        'rt': df['RT'].values.astype(np.float32),
        'response': df['response'].values.astype(np.int32),
        'stimulus': df['stimulus'].values.astype(np.int32),
        'left_dim': df['left_dim'].values.astype(np.int32),
        'right_dim': df['right_dim'].values.astype(np.int32),
        'participant_idx': df['participant_idx'].values.astype(np.int32),
        'n_participants': len(participants),
        'n_trials': len(df),
        'participants': participants
    }
    
    # Calculate accuracy
    accuracy = np.mean(df['response'] == df['stimulus'])
    print(f"\nDataset summary:")
    print(f"  Participants: {data['n_participants']}")
    print(f"  Total trials: {data['n_trials']}")
    print(f"  Overall accuracy: {accuracy:.3f}")
    
    return data, df

def hierarchical_grt_lba_model(data, test_mode=True):
    """
    Hierarchical GRT-LBA model testing three GRT assumptions
    
    Architecture:
    1. Left dimension processing (2 accumulators: Low/High)
    2. Right dimension processing (2 accumulators: Low/High) 
    3. Combined 4-accumulator race
    
    GRT Assumptions tested:
    1. Decision Boundary Independence
    2. Perceptual Separability
    3. Perceptual Independence
    """
    print("Building hierarchical GRT-LBA model...")
    
    if test_mode:
        # Use subset for testing (3 participants, 100 trials each)
        n_test_participants = min(3, data['n_participants'])
        n_test_trials_per_p = 100
        
        # Select subset
        subset_mask = data['participant_idx'] < n_test_participants
        subset_indices = np.where(subset_mask)[0]
        
        # Further limit trials per participant
        final_indices = []
        for p in range(n_test_participants):
            p_indices = subset_indices[data['participant_idx'][subset_indices] == p]
            selected = np.random.choice(p_indices, 
                                      min(n_test_trials_per_p, len(p_indices)), 
                                      replace=False)
            final_indices.extend(selected)
        
        final_indices = np.array(final_indices)
        
        rt_obs = data['rt'][final_indices]
        response_obs = data['response'][final_indices]
        stimulus_obs = data['stimulus'][final_indices]
        left_dim_obs = data['left_dim'][final_indices]
        right_dim_obs = data['right_dim'][final_indices]
        participant_obs = data['participant_idx'][final_indices]
        
        n_trials = len(final_indices)
        n_participants = n_test_participants
        
        print(f"Test mode: {n_participants} participants, {n_trials} trials")
    else:
        rt_obs = data['rt']
        response_obs = data['response']
        stimulus_obs = data['stimulus']
        left_dim_obs = data['left_dim']
        right_dim_obs = data['right_dim']
        participant_obs = data['participant_idx']
        n_trials = data['n_trials']
        n_participants = data['n_participants']
        
        print(f"Full mode: {n_participants} participants, {n_trials} trials")

    with pm.Model() as model:
        
        # =============================================
        # HIERARCHICAL GROUP-LEVEL PARAMETERS
        # =============================================
        
        print("Setting up hierarchical parameters...")
        
        # Basic LBA parameters (group level)
        A_mu = pm.HalfNormal('A_mu', sigma=0.3)  # Start point upper bound
        A_sigma = pm.HalfNormal('A_sigma', sigma=0.1)
        
        b_excess_mu = pm.HalfNormal('b_excess_mu', sigma=0.4)  # Threshold above A
        b_excess_sigma = pm.HalfNormal('b_excess_sigma', sigma=0.2)
        
        t0_mu = pm.HalfNormal('t0_mu', sigma=0.2)  # Non-decision time
        t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.05)
        
        # Drift rate parameters (group level)
        drift_base_mu = pm.HalfNormal('drift_base_mu', sigma=1.0)  # Base drift rate
        drift_base_sigma = pm.HalfNormal('drift_base_sigma', sigma=0.3)
        
        drift_correct_boost_mu = pm.HalfNormal('drift_correct_boost_mu', sigma=0.8)  # Boost for correct response
        drift_correct_boost_sigma = pm.HalfNormal('drift_correct_boost_sigma', sigma=0.3)
        
        # =============================================
        # GRT VIOLATION PARAMETERS (GROUP LEVEL)
        # =============================================
        
        # 1. DECISION BOUNDARY INDEPENDENCE VIOLATIONS
        # Left boundary affected by right dimension
        boundary_left_by_right_mu = pm.Normal('boundary_left_by_right_mu', mu=0, sigma=0.2)
        boundary_left_by_right_sigma = pm.HalfNormal('boundary_left_by_right_sigma', sigma=0.1)
        
        # Right boundary affected by left dimension  
        boundary_right_by_left_mu = pm.Normal('boundary_right_by_left_mu', mu=0, sigma=0.2)
        boundary_right_by_left_sigma = pm.HalfNormal('boundary_right_by_left_sigma', sigma=0.1)
        
        # 2. PERCEPTUAL SEPARABILITY VIOLATIONS
        # Left perception affected by right stimulus
        separability_left_by_right_mu = pm.Normal('separability_left_by_right_mu', mu=0, sigma=0.3)
        separability_left_by_right_sigma = pm.HalfNormal('separability_left_by_right_sigma', sigma=0.2)
        
        # Right perception affected by left stimulus
        separability_right_by_left_mu = pm.Normal('separability_right_by_left_mu', mu=0, sigma=0.3)
        separability_right_by_left_sigma = pm.HalfNormal('separability_right_by_left_sigma', sigma=0.2)
        
        # 3. PERCEPTUAL INDEPENDENCE VIOLATIONS
        # Correlation between left and right evidence
        independence_correlation_mu = pm.Normal('independence_correlation_mu', mu=0, sigma=0.3)
        independence_correlation_sigma = pm.HalfNormal('independence_correlation_sigma', sigma=0.2)
        
        # =============================================
        # INDIVIDUAL PARTICIPANT PARAMETERS
        # =============================================
        
        print("Setting up individual parameters...")
        
        # Individual LBA parameters
        A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
        A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
        
        b_excess_raw = pm.Normal('b_excess_raw', mu=0, sigma=1, shape=n_participants)
        b_excess = pm.Deterministic('b_excess', pm.math.maximum(b_excess_mu + b_excess_sigma * b_excess_raw, 0.05))
        b = pm.Deterministic('b', A + b_excess)
        
        t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
        t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.05))
        
        drift_base_raw = pm.Normal('drift_base_raw', mu=0, sigma=1, shape=n_participants)
        drift_base = pm.Deterministic('drift_base', pm.math.maximum(drift_base_mu + drift_base_sigma * drift_base_raw, 0.1))
        
        drift_correct_boost_raw = pm.Normal('drift_correct_boost_raw', mu=0, sigma=1, shape=n_participants)
        drift_correct_boost = pm.Deterministic('drift_correct_boost', 
                                             pm.math.maximum(drift_correct_boost_mu + drift_correct_boost_sigma * drift_correct_boost_raw, 0.1))
        
        # Individual GRT violation parameters
        boundary_left_by_right_raw = pm.Normal('boundary_left_by_right_raw', mu=0, sigma=1, shape=n_participants)
        boundary_left_by_right = pm.Deterministic('boundary_left_by_right', 
                                                 boundary_left_by_right_mu + boundary_left_by_right_sigma * boundary_left_by_right_raw)
        
        boundary_right_by_left_raw = pm.Normal('boundary_right_by_left_raw', mu=0, sigma=1, shape=n_participants)
        boundary_right_by_left = pm.Deterministic('boundary_right_by_left',
                                                 boundary_right_by_left_mu + boundary_right_by_left_sigma * boundary_right_by_left_raw)
        
        separability_left_by_right_raw = pm.Normal('separability_left_by_right_raw', mu=0, sigma=1, shape=n_participants)
        separability_left_by_right = pm.Deterministic('separability_left_by_right',
                                                     separability_left_by_right_mu + separability_left_by_right_sigma * separability_left_by_right_raw)
        
        separability_right_by_left_raw = pm.Normal('separability_right_by_left_raw', mu=0, sigma=1, shape=n_participants)
        separability_right_by_left = pm.Deterministic('separability_right_by_left',
                                                     separability_right_by_left_mu + separability_right_by_left_sigma * separability_right_by_left_raw)
        
        independence_correlation_raw = pm.Normal('independence_correlation_raw', mu=0, sigma=1, shape=n_participants)
        independence_correlation = pm.Deterministic('independence_correlation',
                                                   independence_correlation_mu + independence_correlation_sigma * independence_correlation_raw)
        
        # =============================================
        # GRT-LBA LIKELIHOOD FUNCTION
        # =============================================
        
        print("Setting up GRT-LBA likelihood...")
        
        def grt_lba_likelihood():
            """
            Exact GRT-LBA likelihood with dimensional architecture
            """
            rt_decision = pt.maximum(rt_obs - t0[participant_obs], 0.01)
            total_loglik = 0.0
            
            # Process each trial
            for i in range(n_trials):
                rt_i = rt_decision[i]
                response_i = response_obs[i]
                stimulus_i = stimulus_obs[i]
                left_i = left_dim_obs[i]
                right_i = right_dim_obs[i]
                participant_i = participant_obs[i]
                
                # Get individual parameters for this participant
                A_i = A[participant_i]
                b_i = b[participant_i]
                drift_base_i = drift_base[participant_i]
                drift_boost_i = drift_correct_boost[participant_i]
                
                # GRT violation parameters for this participant
                bound_lr_i = boundary_left_by_right[participant_i]
                bound_rl_i = boundary_right_by_left[participant_i]
                sep_lr_i = separability_left_by_right[participant_i]
                sep_rl_i = separability_right_by_left[participant_i]
                indep_i = independence_correlation[participant_i]
                
                # Compute drift rates for 4 accumulators with GRT effects
                drift_rates = pt.zeros(4)
                
                for acc in range(4):
                    # Determine accumulator's dimensional preferences
                    acc_left = acc // 2  # 0,1 -> 0; 2,3 -> 1
                    acc_right = acc % 2  # 0,2 -> 0; 1,3 -> 1
                    
                    # Base drift rate
                    base_rate = pt.switch(pt.eq(acc, stimulus_i),
                                        drift_base_i + drift_boost_i,
                                        drift_base_i)
                    
                    # PERCEPTUAL SEPARABILITY violations
                    # Left perception affected by right stimulus
                    sep_left_effect = pt.switch(pt.eq(acc_left, left_i),
                                              sep_lr_i * right_i,  # Enhancement when matching
                                              -sep_lr_i * right_i)  # Interference when mismatching
                    
                    # Right perception affected by left stimulus  
                    sep_right_effect = pt.switch(pt.eq(acc_right, right_i),
                                               sep_rl_i * left_i,
                                               -sep_rl_i * left_i)
                    
                    # PERCEPTUAL INDEPENDENCE violations
                    # Correlation effect when both dimensions match
                    indep_effect = pt.switch(pt.and_(pt.eq(acc_left, left_i), pt.eq(acc_right, right_i)),
                                           indep_i,
                                           0.0)
                    
                    # Final drift rate
                    final_drift = pt.maximum(base_rate + sep_left_effect + sep_right_effect + indep_effect, 0.05)
                    drift_rates = pt.set_subtensor(drift_rates[acc], final_drift)
                
                # DECISION BOUNDARY INDEPENDENCE violations
                # Modify thresholds based on irrelevant dimension
                threshold_left = b_i + bound_lr_i * right_i  # Left threshold affected by right
                threshold_right = b_i + bound_rl_i * left_i  # Right threshold affected by left
                
                # Apply threshold modifications to appropriate accumulators
                thresholds = pt.stack([
                    threshold_left,   # Acc 0: Left-High + Right-Low
                    b_i,             # Acc 1: Left-Low + Right-Low (baseline)
                    threshold_right, # Acc 2: Left-Low + Right-High
                    threshold_left + threshold_right - b_i  # Acc 3: Both affected
                ])
                
                # Exact LBA likelihood computation
                v_winner = drift_rates[response_i]
                threshold_winner = thresholds[response_i]
                
                sqrt_t = pt.sqrt(rt_i)
                v_t = v_winner * rt_i
                
                # Winner PDF
                z1_win = (v_t - threshold_winner) / sqrt_t
                z2_win = (v_t - A_i) / sqrt_t
                
                Phi_z1_win = 0.5 * (1 + pt.erf(z1_win / pt.sqrt(2)))
                Phi_z2_win = 0.5 * (1 + pt.erf(z2_win / pt.sqrt(2)))
                phi_z1_win = pt.exp(-0.5 * z1_win**2) / pt.sqrt(2 * np.pi)
                phi_z2_win = pt.exp(-0.5 * z2_win**2) / pt.sqrt(2 * np.pi)
                
                term1 = v_winner * (Phi_z1_win - Phi_z2_win)
                term2 = (phi_z1_win - phi_z2_win) / sqrt_t
                winner_pdf = (1 / A_i) * (term1 + term2)
                winner_logpdf = pt.log(pt.maximum(winner_pdf, 1e-10))
                
                # Losers survival
                losers_log_survival = 0.0
                for acc in range(4):
                    if acc != response_i:
                        v_loser = drift_rates[acc]
                        threshold_loser = thresholds[acc]
                        v_t_loser = v_loser * rt_i
                        
                        z1_lose = (v_t_loser - threshold_loser) / sqrt_t
                        z2_lose = (v_t_loser - A_i) / sqrt_t
                        
                        Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
                        Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
                        
                        loser_cdf = Phi_z1_lose - Phi_z2_lose
                        loser_survival = 1 - loser_cdf
                        losers_log_survival += pt.log(pt.maximum(loser_survival, 1e-10))
                
                # Total trial likelihood
                trial_loglik = winner_logpdf + losers_log_survival
                total_loglik += trial_loglik
            
            return total_loglik
        
        # Add likelihood to model
        pm.Potential('grt_lba_likelihood', grt_lba_likelihood())
        
        # =============================================
        # DERIVED QUANTITIES FOR INTERPRETATION
        # =============================================
        
        # Group-level GRT assumption tests
        pm.Deterministic('boundary_independence_violation', 
                        pt.sqrt(boundary_left_by_right_mu**2 + boundary_right_by_left_mu**2))
        
        pm.Deterministic('perceptual_separability_violation',
                        pt.sqrt(separability_left_by_right_mu**2 + separability_right_by_left_mu**2))
        
        pm.Deterministic('perceptual_independence_violation',
                        pt.abs(independence_correlation_mu))
        
        print("Model setup complete!")
        
    return model

def run_grt_analysis(data_file, test_mode=True, n_samples=200, n_chains=2):
    """
    Run complete GRT-LBA analysis
    """
    print("=" * 60)
    print("HIERARCHICAL GRT-LBA ANALYSIS")
    print("=" * 60)
    
    # Load data
    data, df = load_and_prepare_grt_data(data_file, 
                                        max_participants=5 if test_mode else None,
                                        max_trials_per_participant=150 if test_mode else None)
    
    # Build model
    model = hierarchical_grt_lba_model(data, test_mode=test_mode)
    
    # Sample
    print(f"\nStarting MCMC sampling...")
    print(f"Samples: {n_samples}, Chains: {n_chains}")
    
    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_samples//2,
            chains=n_chains,
            cores=1,
            target_accept=0.85,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42
        )
    
    print("✅ Sampling completed!")
    
    # Analysis
    print("\n" + "=" * 60)
    print("GRT ASSUMPTION TEST RESULTS")
    print("=" * 60)
    
    # Extract key parameters
    posterior = trace.posterior
    
    # Test results
    boundary_violation = posterior['boundary_independence_violation'].values.mean()
    separability_violation = posterior['perceptual_separability_violation'].values.mean()
    independence_violation = posterior['perceptual_independence_violation'].values.mean()
    
    print(f"\n1. DECISION BOUNDARY INDEPENDENCE:")
    print(f"   Violation magnitude: {boundary_violation:.4f}")
    print(f"   {'❌ VIOLATED' if boundary_violation > 0.1 else '✅ SATISFIED'}")
    
    print(f"\n2. PERCEPTUAL SEPARABILITY:")
    print(f"   Violation magnitude: {separability_violation:.4f}")
    print(f"   {'❌ VIOLATED' if separability_violation > 0.1 else '✅ SATISFIED'}")
    
    print(f"\n3. PERCEPTUAL INDEPENDENCE:")
    print(f"   Violation magnitude: {independence_violation:.4f}")
    print(f"   {'❌ VIOLATED' if independence_violation > 0.1 else '✅ SATISFIED'}")
    
    return trace, model, data

# Example usage
if __name__ == "__main__":
    # Run analysis
    trace, model, data = run_grt_analysis('GRT_LBA.csv', test_mode=True)
