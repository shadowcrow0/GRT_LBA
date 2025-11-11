"""
GRT-LBA Joint Parameter Estimation - REDUCED 4-PARAMETER VERSION
=================================================================

Key Innovation: Dimensionality Reduction
- Original: 8 parameters (v_L_H_correct, v_L_H_error, v_L_V_correct, v_L_V_error, ...)
- Reduced: 4 parameters (v_L_correct, v_L_error, v_R_correct, v_R_error)

Rationale:
Under PERFECT perceptual separability assumption:
- Left dimension's correct drift should be SAME regardless of stimulus (H or V)
- Left dimension's error drift should be SAME regardless of stimulus
- Same logic applies to right dimension

Benefits:
1. Fewer parameters → Better identifiability
2. 4 conditions, 4 parameters → much better ratio than 4 conditions, 8 parameters
3. Stronger statistical constraints per parameter
4. Should eliminate multimodal posterior issues

Trade-off:
- Assumes perfect perceptual separability (no stimulus-specific effects)
- Less flexible than 8-parameter model

Author: YYC & Claude
Date: 2025-11-10
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import logging
import json
from datetime import datetime

from grt_lba_4choice_correct import lba_2dim_random, GRT_LBA_4Choice_LogLik

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
LBA_FIXED_PARAMS = {'A': 0.5, 'b': 1.0, 's': 1.0}
T0_PARAMS = {'t0_left': 0.0, 't0_right': 0.0, 'motor_t0': 0.10}
MCMC_CONFIG = {
    'draws': 20000,
    'tune': 5000,
    'chains': 4,
    'cores': 4
}

TRUE_PARAMS = {
    'v_L_correct': 3.5,  # Shared by v_L_H_correct and v_L_V_correct
    'v_L_error': 0.8,    # Shared by v_L_H_error and v_L_V_error
    'v_R_correct': 3.5,  # Shared by v_R_H_correct and v_R_V_correct
    'v_R_error': 0.8,    # Shared by v_R_H_error and v_R_V_error
}

def generate_data(n_trials_per_condition=1000, seed=42):
    """Generate GRT-LBA data (same as before, but note the symmetry)"""
    logger.info("="*70)
    logger.info("Generating GRT-LBA Data (4000 trials)")
    logger.info("="*70)
    logger.info(f"  Note: Data generated with SYMMETRIC parameters")
    logger.info(f"  v_L_H_correct = v_L_V_correct = 3.5")
    logger.info(f"  v_R_H_correct = v_R_V_correct = 3.5")
    logger.info(f"  (Perfect perceptual separability)")

    # Build v_tensor with symmetric structure
    v_tensor_true = np.array([
        # Condition 0: VH
        [[TRUE_PARAMS['v_L_error'], TRUE_PARAMS['v_L_correct']],    # Left: V correct
         [TRUE_PARAMS['v_R_correct'], TRUE_PARAMS['v_R_error']]],   # Right: H correct
        # Condition 1: HH
        [[TRUE_PARAMS['v_L_correct'], TRUE_PARAMS['v_L_error']],    # Left: H correct
         [TRUE_PARAMS['v_R_correct'], TRUE_PARAMS['v_R_error']]],   # Right: H correct
        # Condition 2: HV
        [[TRUE_PARAMS['v_L_correct'], TRUE_PARAMS['v_L_error']],    # Left: H correct
         [TRUE_PARAMS['v_R_error'], TRUE_PARAMS['v_R_correct']]],   # Right: V correct
        # Condition 3: VV
        [[TRUE_PARAMS['v_L_error'], TRUE_PARAMS['v_L_correct']],    # Left: V correct
         [TRUE_PARAMS['v_R_error'], TRUE_PARAMS['v_R_correct']]]    # Right: V correct
    ], dtype=np.float64)

    rng = np.random.RandomState(seed)
    data_array = lba_2dim_random(
        n_trials_per_condition=n_trials_per_condition,
        v_tensor=v_tensor_true,
        A=LBA_FIXED_PARAMS['A'], b=LBA_FIXED_PARAMS['b'],
        t0_array=np.array([T0_PARAMS['motor_t0']] * 4),
        s=LBA_FIXED_PARAMS['s'], rng=rng
    )

    df = pd.DataFrame(data_array, columns=['rt', 'choice', 'condition'])
    accuracy = (df['choice'] == df['condition']).sum() / len(df) * 100
    logger.info(f"✓ Generated {len(df)} trials, Accuracy: {accuracy:.1f}%")
    return df

def build_reduced_model(data):
    """Build REDUCED 4-parameter model"""
    logger.info("\n" + "="*70)
    logger.info("REDUCED 4-PARAMETER MODEL")
    logger.info("="*70)
    logger.info("  Parameters: v_L_correct, v_L_error, v_R_correct, v_R_error")
    logger.info("  Priors: moderately informative")
    logger.info("    Correct: mu=3.5, sigma=0.5, bounds=[2.0, 5.0]")
    logger.info("    Error:   mu=0.8, sigma=0.3, bounds=[0.2, 1.5]")

    with pm.Model() as model:
        # ONLY 4 parameters (instead of 8)
        v_L_correct = pm.TruncatedNormal('v_L_correct', mu=3.5, sigma=0.5, lower=2.0, upper=5.0)
        v_L_error = pm.TruncatedNormal('v_L_error', mu=0.8, sigma=0.3, lower=0.2, upper=1.5)
        v_R_correct = pm.TruncatedNormal('v_R_correct', mu=3.5, sigma=0.5, lower=2.0, upper=5.0)
        v_R_error = pm.TruncatedNormal('v_R_error', mu=0.8, sigma=0.3, lower=0.2, upper=1.5)

        # Build v_tensor using the 4 parameters
        # Key: Same v_L_correct used for BOTH H and V stimuli on left
        v_tensor = pt.stack([
            # Condition 0: VH - Left sees V, Right sees H
            pt.stack([pt.stack([v_L_error, v_L_correct]),      # Left: V correct
                     pt.stack([v_R_correct, v_R_error])]),     # Right: H correct
            # Condition 1: HH - Left sees H, Right sees H
            pt.stack([pt.stack([v_L_correct, v_L_error]),      # Left: H correct
                     pt.stack([v_R_correct, v_R_error])]),     # Right: H correct
            # Condition 2: HV - Left sees H, Right sees V
            pt.stack([pt.stack([v_L_correct, v_L_error]),      # Left: H correct
                     pt.stack([v_R_error, v_R_correct])]),     # Right: V correct
            # Condition 3: VV - Left sees V, Right sees V
            pt.stack([pt.stack([v_L_error, v_L_correct]),      # Left: V correct
                     pt.stack([v_R_error, v_R_correct])])      # Right: V correct
        ])

        # Likelihood (same as before)
        choice_t = pt.as_tensor_variable(data['choice'].values.astype(np.int32))
        rt_t = pt.as_tensor_variable(data['rt'].values.astype(np.float64))
        condition_t = pt.as_tensor_variable(data['condition'].values.astype(np.int32))
        A_t = pt.as_tensor_variable(np.float64(LBA_FIXED_PARAMS['A']))
        b_t = pt.as_tensor_variable(np.float64(LBA_FIXED_PARAMS['b']))
        t0_t = pt.as_tensor_variable(np.array([T0_PARAMS['motor_t0']] * 4, dtype=np.float64))
        s_t = pt.as_tensor_variable(np.float64(LBA_FIXED_PARAMS['s']))

        loglik_op = GRT_LBA_4Choice_LogLik()
        log_lik_total = loglik_op(choice_t, rt_t, condition_t, v_tensor, A_t, b_t, t0_t, s_t)
        pm.Potential('logp', log_lik_total)

    logger.info("✓ Model built successfully (4 parameters only)")
    return model

def main():
    start_time = datetime.now()
    logger.info("\n" + "="*70)
    logger.info("GRT-LBA REDUCED 4-PARAMETER ESTIMATION")
    logger.info(f"Start: {start_time}")
    logger.info("="*70)

    data = generate_data(n_trials_per_condition=1000, seed=42)

    logger.info("\n" + "="*70)
    logger.info("MCMC SAMPLING CONFIGURATION")
    logger.info(f"  Sampler: DEMetropolisZ")
    logger.info(f"  Draws: {MCMC_CONFIG['draws']:,}")
    logger.info(f"  Tune: {MCMC_CONFIG['tune']:,}")
    logger.info(f"  Chains: {MCMC_CONFIG['chains']}")
    logger.info(f"  Estimated time: ~30-45 minutes (faster due to fewer params)")
    logger.info("="*70)

    model = build_reduced_model(data)

    with model:
        trace = pm.sample(
            draws=MCMC_CONFIG['draws'],
            tune=MCMC_CONFIG['tune'],
            chains=MCMC_CONFIG['chains'],
            cores=MCMC_CONFIG['cores'],
            step=pm.DEMetropolisZ(tune='scaling', scaling=0.01, proposal_dist=pm.NormalProposal),
            return_inferencedata=True,
            progressbar=True,
            discard_tuned_samples=True,
            idata_kwargs={'log_likelihood': True}
        )

    # Compute log-likelihood for diagnostics
    logger.info("\n" + "="*70)
    logger.info("COMPUTING LOG-LIKELIHOOD DIAGNOSTICS")
    logger.info("="*70)

    with model:
        # Extract posterior samples
        posterior_samples = trace.posterior
        n_chains = len(posterior_samples.chain)
        n_draws = len(posterior_samples.draw)

        # Compute log-likelihood for each sample
        loglik_values = []
        for chain_idx in range(n_chains):
            chain_logliks = []
            for draw_idx in range(min(n_draws, 100)):  # Sample 100 draws per chain
                v_L_c = float(posterior_samples['v_L_correct'].values[chain_idx, draw_idx])
                v_L_e = float(posterior_samples['v_L_error'].values[chain_idx, draw_idx])
                v_R_c = float(posterior_samples['v_R_correct'].values[chain_idx, draw_idx])
                v_R_e = float(posterior_samples['v_R_error'].values[chain_idx, draw_idx])

                # Build v_tensor for this sample
                v_tensor_np = np.array([
                    [[v_L_e, v_L_c], [v_R_c, v_R_e]],  # Condition 0: VH
                    [[v_L_c, v_L_e], [v_R_c, v_R_e]],  # Condition 1: HH
                    [[v_L_c, v_L_e], [v_R_e, v_R_c]],  # Condition 2: HV
                    [[v_L_e, v_L_c], [v_R_e, v_R_c]]   # Condition 3: VV
                ], dtype=np.float64)

                # Compute log-likelihood using the custom Op
                from grt_lba_4choice_correct import GRT_LBA_4Choice_LogLik
                loglik_op = GRT_LBA_4Choice_LogLik()
                choice_t = pt.as_tensor_variable(data['choice'].values.astype(np.int32))
                rt_t = pt.as_tensor_variable(data['rt'].values.astype(np.float64))
                condition_t = pt.as_tensor_variable(data['condition'].values.astype(np.int32))
                v_t = pt.as_tensor_variable(v_tensor_np)
                A_t = pt.as_tensor_variable(np.float64(LBA_FIXED_PARAMS['A']))
                b_t = pt.as_tensor_variable(np.float64(LBA_FIXED_PARAMS['b']))
                t0_t = pt.as_tensor_variable(np.array([T0_PARAMS['motor_t0']] * 4, dtype=np.float64))
                s_t = pt.as_tensor_variable(np.float64(LBA_FIXED_PARAMS['s']))

                loglik = loglik_op(choice_t, rt_t, condition_t, v_t, A_t, b_t, t0_t, s_t)
                loglik_val = loglik.eval()
                chain_logliks.append(loglik_val)

            loglik_values.append(chain_logliks)

        # Report log-likelihood statistics
        loglik_array = np.array(loglik_values)
        logger.info(f"  Log-likelihood per chain (mean ± sd):")
        for chain_idx in range(n_chains):
            chain_mean = np.mean(loglik_array[chain_idx])
            chain_sd = np.std(loglik_array[chain_idx])
            logger.info(f"    Chain {chain_idx}: {chain_mean:.2f} ± {chain_sd:.2f}")

        overall_mean = np.mean(loglik_array)
        overall_sd = np.std(loglik_array)
        between_chain_var = np.var([np.mean(loglik_array[i]) for i in range(n_chains)])

        logger.info(f"\n  Overall: {overall_mean:.2f} ± {overall_sd:.2f}")
        logger.info(f"  Between-chain variance: {between_chain_var:.2f}")
        logger.info(f"  Within-chain variance: {np.mean([np.var(loglik_array[i]) for i in range(n_chains)]):.2f}")

        if between_chain_var < 1.0:
            logger.info("  ✅ Chains have similar log-likelihoods (low between-chain variance)")
        else:
            logger.info("  ⚠️  Chains exploring different likelihood regions!")

    summary = az.summary(trace)
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    logger.info(f"\n{summary}")

    logger.info("\n" + "="*70)
    logger.info("PARAMETER RECOVERY ACCURACY")
    logger.info("="*70)

    for param_name, true_value in TRUE_PARAMS.items():
        est_mean = summary.loc[param_name, 'mean']
        est_sd = summary.loc[param_name, 'sd']
        error_pct = abs(est_mean - true_value) / true_value * 100
        rhat = summary.loc[param_name, 'r_hat']
        ess = summary.loc[param_name, 'ess_bulk']
        logger.info(f"{param_name:20s}: True={true_value:.3f}, Est={est_mean:.3f}±{est_sd:.3f}, "
                   f"Error={error_pct:5.1f}%, R-hat={rhat:.4f}, ESS={ess:.0f}")

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    results = {
        'reduced_4param': {'summary': summary.to_dict(), 'true_params': TRUE_PARAMS},
        'metadata': {
            'start_time': str(start_time), 'end_time': str(end_time),
            'elapsed_seconds': elapsed, 'mcmc_config': MCMC_CONFIG,
            'lba_params': LBA_FIXED_PARAMS, 't0_params': T0_PARAMS,
            'n_trials': len(data),
            'n_parameters': 4,
            'model_assumptions': [
                'Perfect perceptual separability',
                'v_L_H_correct = v_L_V_correct (same correct drift for left dim)',
                'v_L_H_error = v_L_V_error (same error drift for left dim)',
                'v_R_H_correct = v_R_V_correct (same correct drift for right dim)',
                'v_R_H_error = v_R_V_error (same error drift for right dim)'
            ]
        }
    }

    with open('flat_JOINT_4param_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n" + "="*70)
    logger.info(f"✓ Completed in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"✓ Results saved to flat_JOINT_4param_results.json")
    logger.info("="*70)
    logger.info("\n" + "="*70)
    logger.info("KEY ADVANTAGE: 4 parameters for 4 conditions")
    logger.info("  → Much better parameter:data ratio than 8 param version")
    logger.info("  → Should resolve multimodal posterior issues")
    logger.info("  → Expected: Higher ESS and lower R-hat values")
    logger.info("="*70)

if __name__ == '__main__':
    main()