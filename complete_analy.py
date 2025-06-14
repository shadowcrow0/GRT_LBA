"""
Progressive GRT Modeling: Step-by-step from 2-choice to full GRT
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import warnings
warnings.filterwarnings('ignore')

def load_progressive_data(n_participants=3, n_trials_per_participant=100):
    """
    Load subset of data for progressive modeling
    """
    print(f"Loading data for progressive modeling...")
    
    df = pd.read_csv('GRT_LBA.csv')
    
    # Take subset for faster development
    participants = sorted(df['participant'].unique())[:n_participants]
    df_subset = df[df['participant'].isin(participants)]
    
    # Limit trials per participant
    df_subset = df_subset.groupby('participant').head(n_trials_per_participant).reset_index(drop=True)
    
    # Clean data
    df_subset = df_subset[(df_subset['RT'] > 0.1) & (df_subset['RT'] < 3.0)]
    
    # Create indices
    participant_map = {p: i for i, p in enumerate(participants)}
    df_subset['participant_idx'] = df_subset['participant'].map(participant_map)
    
    print(f"Progressive dataset: {len(df_subset)} trials from {len(participants)} participants")
    print(f"Average trials per participant: {len(df_subset)//len(participants)}")
    
    return df_subset, participants

# =============================================================================
# STEP 1: 2-CHOICE LBA MODEL
# =============================================================================

def step1_2choice_lba(df, participants):
    """
    Step 1: Simple 2-choice LBA model
    Convert 4-choice problem to 2-choice for initial validation
    """
    print("\n" + "="*60)
    print("STEP 1: 2-CHOICE LBA MODEL")
    print("="*60)
    
    # Convert to 2-choice problem
    # Choice 1: Responses 0,1 (Left dimension focused)
    # Choice 2: Responses 2,3 (Right dimension focused)
    df['choice_2'] = (df['Response'] >= 2).astype(int)
    df['stimulus_2'] = (df['stim_condition'] >= 2).astype(int)
    
    print(f"2-choice conversion:")
    print(f"Choice 0 (responses 0,1): {np.sum(df['choice_2']==0)} trials")
    print(f"Choice 1 (responses 2,3): {np.sum(df['choice_2']==1)} trials")
    
    # Prepare data
    rt_obs = df['RT'].values.astype(np.float32)
    choice_obs = df['choice_2'].values.astype(np.int32)
    stimulus_obs = df['stimulus_2'].values.astype(np.int32)
    participant_obs = df['participant_idx'].values.astype(np.int32)
    
    n_trials = len(df)
    n_participants = len(participants)
    
    with pm.Model() as model_2choice:
        
        # Group-level parameters
        A_mu = pm.HalfNormal('A_mu', sigma=0.3)
        A_sigma = pm.HalfNormal('A_sigma', sigma=0.1)
        
        b_excess_mu = pm.HalfNormal('b_excess_mu', sigma=0.4)
        b_excess_sigma = pm.HalfNormal('b_excess_sigma', sigma=0.2)
        
        t0_mu = pm.HalfNormal('t0_mu', sigma=0.2)
        t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.05)
        
        v1_mu = pm.HalfNormal('v1_mu', sigma=1.0)
        v1_sigma = pm.HalfNormal('v1_sigma', sigma=0.3)
        
        v2_mu = pm.HalfNormal('v2_mu', sigma=1.0)
        v2_sigma = pm.HalfNormal('v2_sigma', sigma=0.3)
        
        # Individual parameters
        A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
        A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
        
        b_excess_raw = pm.Normal('b_excess_raw', mu=0, sigma=1, shape=n_participants)
        b_excess = pm.Deterministic('b_excess', pm.math.maximum(b_excess_mu + b_excess_sigma * b_excess_raw, 0.05))
        b = pm.Deterministic('b', A + b_excess)
        
        t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
        t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
        
        v1_raw = pm.Normal('v1_raw', mu=0, sigma=1, shape=n_participants)
        v1 = pm.Deterministic('v1', pm.math.maximum(v1_mu + v1_sigma * v1_raw, 0.1))
        
        v2_raw = pm.Normal('v2_raw', mu=0, sigma=1, shape=n_participants)
        v2 = pm.Deterministic('v2', pm.math.maximum(v2_mu + v2_sigma * v2_raw, 0.1))
        
        # Simplified LBA likelihood
        def lba_2choice_likelihood():
            rt_decision = pt.maximum(rt_obs - t0[participant_obs], 0.01)
            
            total_loglik = 0.0
            
            for i in range(n_trials):
                rt_i = rt_decision[i]
                choice_i = choice_obs[i]
                participant_i = participant_obs[i]
                
                A_i = A[participant_i]
                b_i = b[participant_i]
                v1_i = v1[participant_i]
                v2_i = v2[participant_i]
                
                # Determine drift rates
                if choice_i == 0:
                    v_winner = v1_i
                    v_loser = v2_i
                else:
                    v_winner = v2_i
                    v_loser = v1_i
                
                # Simplified LBA PDF for winner
                sqrt_t = pt.sqrt(rt_i)
                z1 = (v_winner * rt_i - b_i) / sqrt_t
                z2 = (v_winner * rt_i - A_i) / sqrt_t
                
                Phi_z1 = 0.5 * (1 + pt.erf(z1 / pt.sqrt(2)))
                Phi_z2 = 0.5 * (1 + pt.erf(z2 / pt.sqrt(2)))
                phi_z1 = pt.exp(-0.5 * z1**2) / pt.sqrt(2 * np.pi)
                phi_z2 = pt.exp(-0.5 * z2**2) / pt.sqrt(2 * np.pi)
                
                winner_pdf = (1/A_i) * (v_winner * (Phi_z1 - Phi_z2) + (phi_z1 - phi_z2) / sqrt_t)
                winner_logpdf = pt.log(pt.maximum(winner_pdf, 1e-10))
                
                # Simplified survival for loser
                z1_lose = (v_loser * rt_i - b_i) / sqrt_t
                z2_lose = (v_loser * rt_i - A_i) / sqrt_t
                Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
                Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
                
                loser_survival = 1 - (Phi_z1_lose - Phi_z2_lose)
                loser_log_survival = pt.log(pt.maximum(loser_survival, 1e-10))
                
                total_loglik += winner_logpdf + loser_log_survival
            
            return total_loglik
        
        pm.Potential('lba_likelihood', lba_2choice_likelihood())
        
        # Derived quantities
        pm.Deterministic('drift_difference', v1_mu - v2_mu)
    
    print("2-choice model created successfully!")
    return model_2choice

# =============================================================================
# STEP 2: 4-CHOICE LBA MODEL
# =============================================================================

def step2_4choice_lba(df, participants):
    """
    Step 2: Expand to full 4-choice LBA model (no GRT violations yet)
    """
    print("\n" + "="*60)
    print("STEP 2: 4-CHOICE LBA MODEL")
    print("="*60)
    
    # Use original 4-choice structure
    rt_obs = df['RT'].values.astype(np.float32)
    response_obs = df['Response'].values.astype(np.int32)
    stimulus_obs = df['stim_condition'].values.astype(np.int32)
    participant_obs = df['participant_idx'].values.astype(np.int32)
    
    n_trials = len(df)
    n_participants = len(participants)
    
    print(f"4-choice data:")
    for resp in range(4):
        count = np.sum(response_obs == resp)
        print(f"Response {resp}: {count} trials")
    
    with pm.Model() as model_4choice:
        
        # Group-level parameters (same structure as 2-choice)
        A_mu = pm.HalfNormal('A_mu', sigma=0.3)
        A_sigma = pm.HalfNormal('A_sigma', sigma=0.1)
        
        b_excess_mu = pm.HalfNormal('b_excess_mu', sigma=0.4)
        b_excess_sigma = pm.HalfNormal('b_excess_sigma', sigma=0.2)
        
        t0_mu = pm.HalfNormal('t0_mu', sigma=0.2)
        t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.05)
        
        # Drift rates for 4 accumulators
        drift_base_mu = pm.HalfNormal('drift_base_mu', sigma=0.8)
        drift_base_sigma = pm.HalfNormal('drift_base_sigma', sigma=0.3)
        
        drift_boost_mu = pm.HalfNormal('drift_boost_mu', sigma=0.6)
        drift_boost_sigma = pm.HalfNormal('drift_boost_sigma', sigma=0.3)
        
        # Individual parameters
        A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
        A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
        
        b_excess_raw = pm.Normal('b_excess_raw', mu=0, sigma=1, shape=n_participants)
        b_excess = pm.Deterministic('b_excess', pm.math.maximum(b_excess_mu + b_excess_sigma * b_excess_raw, 0.05))
        b = pm.Deterministic('b', A + b_excess)
        
        t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
        t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
        
        drift_base_raw = pm.Normal('drift_base_raw', mu=0, sigma=1, shape=n_participants)
        drift_base = pm.Deterministic('drift_base', pm.math.maximum(drift_base_mu + drift_base_sigma * drift_base_raw, 0.1))
        
        drift_boost_raw = pm.Normal('drift_boost_raw', mu=0, sigma=1, shape=n_participants)
        drift_boost = pm.Deterministic('drift_boost', pm.math.maximum(drift_boost_mu + drift_boost_sigma * drift_boost_raw, 0.1))
        
        # 4-choice LBA likelihood (no GRT violations)
        def lba_4choice_likelihood():
            rt_decision = pt.maximum(rt_obs - t0[participant_obs], 0.01)
            
            total_loglik = 0.0
            
            for i in range(n_trials):
                rt_i = rt_decision[i]
                response_i = response_obs[i]
                stimulus_i = stimulus_obs[i]
                participant_i = participant_obs[i]
                
                A_i = A[participant_i]
                b_i = b[participant_i]
                drift_base_i = drift_base[participant_i]
                drift_boost_i = drift_boost[participant_i]
                
                # Compute drift rates for 4 accumulators
                drift_rates = pt.zeros(4)
                for acc in range(4):
                    # Correct accumulator gets boost
                    if acc == stimulus_i:
                        rate = drift_base_i + drift_boost_i
                    else:
                        rate = drift_base_i
                    
                    drift_rates = pt.set_subtensor(drift_rates[acc], rate)
                
                # Winner accumulator
                v_winner = drift_rates[response_i]
                
                # Winner PDF
                sqrt_t = pt.sqrt(rt_i)
                z1 = (v_winner * rt_i - b_i) / sqrt_t
                z2 = (v_winner * rt_i - A_i) / sqrt_t
                
                Phi_z1 = 0.5 * (1 + pt.erf(z1 / pt.sqrt(2)))
                Phi_z2 = 0.5 * (1 + pt.erf(z2 / pt.sqrt(2)))
                phi_z1 = pt.exp(-0.5 * z1**2) / pt.sqrt(2 * np.pi)
                phi_z2 = pt.exp(-0.5 * z2**2) / pt.sqrt(2 * np.pi)
                
                winner_pdf = (1/A_i) * (v_winner * (Phi_z1 - Phi_z2) + (phi_z1 - phi_z2) / sqrt_t)
                winner_logpdf = pt.log(pt.maximum(winner_pdf, 1e-10))
                
                # Losers survival
                losers_log_survival = 0.0
                for acc in range(4):
                    if acc != response_i:
                        v_loser = drift_rates[acc]
                        z1_lose = (v_loser * rt_i - b_i) / sqrt_t
                        z2_lose = (v_loser * rt_i - A_i) / sqrt_t
                        
                        Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
                        Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
                        
                        loser_survival = 1 - (Phi_z1_lose - Phi_z2_lose)
                        losers_log_survival += pt.log(pt.maximum(loser_survival, 1e-10))
                
                total_loglik += winner_logpdf + losers_log_survival
            
            return total_loglik
        
        pm.Potential('lba_likelihood', lba_4choice_likelihood())
        
        # Derived quantities
        pm.Deterministic('accuracy_effect', drift_boost_mu)
    
    print("4-choice model created successfully!")
    return model_4choice

# =============================================================================
# STEP 3: FULL GRT MODEL
# =============================================================================

def step3_full_grt(df, participants):
    """
    Step 3: Add GRT violation parameters to 4-choice LBA
    """
    print("\n" + "="*60)
    print("STEP 3: FULL GRT MODEL")
    print("="*60)
    
    # Add dimensional information
    df['left_dim'] = df['Chanel1'].astype(int)
    df['right_dim'] = df['Chanel2'].astype(int)
    
    rt_obs = df['RT'].values.astype(np.float32)
    response_obs = df['Response'].values.astype(np.int32)
    stimulus_obs = df['stim_condition'].values.astype(np.int32)
    left_dim_obs = df['left_dim'].values.astype(np.int32)
    right_dim_obs = df['right_dim'].values.astype(np.int32)
    participant_obs = df['participant_idx'].values.astype(np.int32)
    
    n_trials = len(df)
    n_participants = len(participants)
    
    with pm.Model() as model_grt:
        
        # Base LBA parameters (same as Step 2)
        A_mu = pm.HalfNormal('A_mu', sigma=0.3)
        A_sigma = pm.HalfNormal('A_sigma', sigma=0.1)
        
        b_excess_mu = pm.HalfNormal('b_excess_mu', sigma=0.4)
        b_excess_sigma = pm.HalfNormal('b_excess_sigma', sigma=0.2)
        
        t0_mu = pm.HalfNormal('t0_mu', sigma=0.2)
        t0_sigma = pm.HalfNormal('t0_sigma', sigma=0.05)
        
        drift_base_mu = pm.HalfNormal('drift_base_mu', sigma=0.8)
        drift_base_sigma = pm.HalfNormal('drift_base_sigma', sigma=0.3)
        
        drift_boost_mu = pm.HalfNormal('drift_boost_mu', sigma=0.6)
        drift_boost_sigma = pm.HalfNormal('drift_boost_sigma', sigma=0.3)
        
        # GRT violation parameters (group level)
        separability_lr_mu = pm.Normal('separability_lr_mu', mu=0, sigma=0.2)
        separability_lr_sigma = pm.HalfNormal('separability_lr_sigma', sigma=0.1)
        
        separability_rl_mu = pm.Normal('separability_rl_mu', mu=0, sigma=0.2)
        separability_rl_sigma = pm.HalfNormal('separability_rl_sigma', sigma=0.1)
        
        independence_mu = pm.Normal('independence_mu', mu=0, sigma=0.2)
        independence_sigma = pm.HalfNormal('independence_sigma', sigma=0.1)
        
        # Individual parameters
        A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_participants)
        A = pm.Deterministic('A', pm.math.maximum(A_mu + A_sigma * A_raw, 0.05))
        
        b_excess_raw = pm.Normal('b_excess_raw', mu=0, sigma=1, shape=n_participants)
        b_excess = pm.Deterministic('b_excess', pm.math.maximum(b_excess_mu + b_excess_sigma * b_excess_raw, 0.05))
        b = pm.Deterministic('b', A + b_excess)
        
        t0_raw = pm.Normal('t0_raw', mu=0, sigma=1, shape=n_participants)
        t0 = pm.Deterministic('t0', pm.math.maximum(t0_mu + t0_sigma * t0_raw, 0.01))
        
        drift_base_raw = pm.Normal('drift_base_raw', mu=0, sigma=1, shape=n_participants)
        drift_base = pm.Deterministic('drift_base', pm.math.maximum(drift_base_mu + drift_base_sigma * drift_base_raw, 0.1))
        
        drift_boost_raw = pm.Normal('drift_boost_raw', mu=0, sigma=1, shape=n_participants)
        drift_boost = pm.Deterministic('drift_boost', pm.math.maximum(drift_boost_mu + drift_boost_sigma * drift_boost_raw, 0.1))
        
        # GRT individual parameters
        separability_lr_raw = pm.Normal('separability_lr_raw', mu=0, sigma=1, shape=n_participants)
        separability_lr = pm.Deterministic('separability_lr', separability_lr_mu + separability_lr_sigma * separability_lr_raw)
        
        separability_rl_raw = pm.Normal('separability_rl_raw', mu=0, sigma=1, shape=n_participants)
        separability_rl = pm.Deterministic('separability_rl', separability_rl_mu + separability_rl_sigma * separability_rl_raw)
        
        independence_raw = pm.Normal('independence_raw', mu=0, sigma=1, shape=n_participants)
        independence = pm.Deterministic('independence', independence_mu + independence_sigma * independence_raw)
        
        # Full GRT-LBA likelihood
        def grt_lba_likelihood():
            rt_decision = pt.maximum(rt_obs - t0[participant_obs], 0.01)
            
            total_loglik = 0.0
            
            for i in range(n_trials):
                rt_i = rt_decision[i]
                response_i = response_obs[i]
                stimulus_i = stimulus_obs[i]
                left_i = left_dim_obs[i]
                right_i = right_dim_obs[i]
                participant_i = participant_obs[i]
                
                A_i = A[participant_i]
                b_i = b[participant_i]
                drift_base_i = drift_base[participant_i]
                drift_boost_i = drift_boost[participant_i]
                sep_lr_i = separability_lr[participant_i]
                sep_rl_i = separability_rl[participant_i]
                indep_i = independence[participant_i]
                
                # Compute drift rates with GRT effects
                drift_rates = pt.zeros(4)
                
                for acc in range(4):
                    acc_left = acc // 2
                    acc_right = acc % 2
                    
                    # Base drift
                    base_rate = pt.switch(pt.eq(acc, stimulus_i),
                                        drift_base_i + drift_boost_i,
                                        drift_base_i)
                    
                    # Separability violations
                    sep_lr_effect = pt.switch(pt.eq(acc_left, left_i),
                                            sep_lr_i * right_i,
                                            -sep_lr_i * right_i)
                    
                    sep_rl_effect = pt.switch(pt.eq(acc_right, right_i),
                                            sep_rl_i * left_i,
                                            -sep_rl_i * left_i)
                    
                    # Independence violation
                    indep_effect = pt.switch(pt.and_(pt.eq(acc_left, left_i), pt.eq(acc_right, right_i)),
                                           indep_i, 0.0)
                    
                    final_rate = pt.maximum(base_rate + sep_lr_effect + sep_rl_effect + indep_effect, 0.05)
                    drift_rates = pt.set_subtensor(drift_rates[acc], final_rate)
                
                # LBA likelihood computation (same as Step 2)
                v_winner = drift_rates[response_i]
                
                sqrt_t = pt.sqrt(rt_i)
                z1 = (v_winner * rt_i - b_i) / sqrt_t
                z2 = (v_winner * rt_i - A_i) / sqrt_t
                
                Phi_z1 = 0.5 * (1 + pt.erf(z1 / pt.sqrt(2)))
                Phi_z2 = 0.5 * (1 + pt.erf(z2 / pt.sqrt(2)))
                phi_z1 = pt.exp(-0.5 * z1**2) / pt.sqrt(2 * np.pi)
                phi_z2 = pt.exp(-0.5 * z2**2) / pt.sqrt(2 * np.pi)
                
                winner_pdf = (1/A_i) * (v_winner * (Phi_z1 - Phi_z2) + (phi_z1 - phi_z2) / sqrt_t)
                winner_logpdf = pt.log(pt.maximum(winner_pdf, 1e-10))
                
                losers_log_survival = 0.0
                for acc in range(4):
                    if acc != response_i:
                        v_loser = drift_rates[acc]
                        z1_lose = (v_loser * rt_i - b_i) / sqrt_t
                        z2_lose = (v_loser * rt_i - A_i) / sqrt_t
                        
                        Phi_z1_lose = 0.5 * (1 + pt.erf(z1_lose / pt.sqrt(2)))
                        Phi_z2_lose = 0.5 * (1 + pt.erf(z2_lose / pt.sqrt(2)))
                        
                        loser_survival = 1 - (Phi_z1_lose - Phi_z2_lose)
                        losers_log_survival += pt.log(pt.maximum(loser_survival, 1e-10))
                
                total_loglik += winner_logpdf + losers_log_survival
            
            return total_loglik
        
        pm.Potential('grt_lba_likelihood', grt_lba_likelihood())
        
        # GRT assumption tests
        pm.Deterministic('separability_violation', pt.sqrt(separability_lr_mu**2 + separability_rl_mu**2))
        pm.Deterministic('independence_violation', pt.abs(independence_mu))
    
    print("Full GRT model created successfully!")
    return model_grt

def run_progressive_analysis():
    """
    Run the complete progressive analysis
    """
    print("🚀 PROGRESSIVE GRT MODELING")
    print("=" * 60)
    
    # Load data
    df, participants = load_progressive_data(n_participants=3, n_trials_per_participant=80)
    
    results = {}
    
    # Step 1: 2-choice LBA
    print(f"\n⏱️ Step 1 estimated time: 3-5 minutes")
    model_2choice = step1_2choice_lba(df, participants)
    
    try:
        with model_2choice:
            trace_2choice = pm.sample(draws=100, tune=50, chains=2, cores=1, 
                                    progressbar=True, return_inferencedata=True, random_seed=42)
        
        print("✅ Step 1 completed successfully!")
        results['step1'] = {'trace': trace_2choice, 'model': model_2choice}
        
        # Step 2: 4-choice LBA
        print(f"\n⏱️ Step 2 estimated time: 5-8 minutes")
        model_4choice = step2_4choice_lba(df, participants)
        
        with model_4choice:
            trace_4choice = pm.sample(draws=100, tune=50, chains=2, cores=1,
                                    progressbar=True, return_inferencedata=True, random_seed=42)
        
        print("✅ Step 2 completed successfully!")
        results['step2'] = {'trace': trace_4choice, 'model': model_4choice}
        
        # Step 3: Full GRT
        print(f"\n⏱️ Step 3 estimated time: 8-12 minutes")
        model_grt = step3_full_grt(df, participants)
        
        with model_grt:
            trace_grt = pm.sample(draws=150, tune=75, chains=2, cores=1,
                                progressbar=True, return_inferencedata=True, random_seed=42)
        
        print("✅ Step 3 completed successfully!")
        results['step3'] = {'trace': trace_grt, 'model': model_grt}
        
        # Final results
        print("\n" + "=" * 60)
        print("🎉 PROGRESSIVE MODELING COMPLETED!")
        print("=" * 60)
        
        # GRT results
        posterior = trace_grt.posterior
        sep_violation = posterior['separability_violation'].values.mean()
        indep_violation = posterior['independence_violation'].values.mean()
        
        print(f"\n📊 FINAL GRT RESULTS:")
        print(f"Perceptual Separability violation: {sep_violation:.4f}")
        print(f"Perceptual Independence violation: {indep_violation:.4f}")
        
        print(f"\nGRT Assumptions:")
        print(f"Separability: {'❌ VIOLATED' if sep_violation > 0.1 else '✅ SATISFIED'}")
        print(f"Independence: {'❌ VIOLATED' if indep_violation > 0.1 else '✅ SATISFIED'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Progressive modeling failed at step: {e}")
        import traceback
        traceback.print_exc()
        return results

# Example usage
if __name__ == "__main__":
    results = run_progressive_analysis()
