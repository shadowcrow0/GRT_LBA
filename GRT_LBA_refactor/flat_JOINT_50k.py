"""
GRT-LBA Joint Parameter Estimation - High-Resolution Version
============================================================

Configurations:
- Data: 300 trials per condition (~1200 total)
- Sampler: DEMetropolisZ
- MCMC: 50,000 draws, 10,000 tune
- Goal: Maximum effective sample size for best parameter recovery

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
MCMC_CONFIG = {'draws': 50000, 'tune': 10000, 'chains': 4, 'cores': 4}

TRUE_PARAMS = {
    'v_L_H_correct': 3.5, 'v_L_H_error': 0.8,
    'v_L_V_correct': 3.5, 'v_L_V_error': 0.8,
    'v_R_H_correct': 3.5, 'v_R_H_error': 0.8,
    'v_R_V_correct': 3.5, 'v_R_V_error': 0.8,
}

def generate_data(n_trials_per_condition=300, seed=42):
    """Generate GRT-LBA data"""
    logger.info("="*70)
    logger.info("Generating GRT-LBA Data")
    logger.info("="*70)

    v_tensor_true = np.array([
        [[TRUE_PARAMS['v_L_V_error'], TRUE_PARAMS['v_L_V_correct']],
         [TRUE_PARAMS['v_R_H_correct'], TRUE_PARAMS['v_R_H_error']]],
        [[TRUE_PARAMS['v_L_H_correct'], TRUE_PARAMS['v_L_H_error']],
         [TRUE_PARAMS['v_R_H_correct'], TRUE_PARAMS['v_R_H_error']]],
        [[TRUE_PARAMS['v_L_H_correct'], TRUE_PARAMS['v_L_H_error']],
         [TRUE_PARAMS['v_R_V_error'], TRUE_PARAMS['v_R_V_correct']]],
        [[TRUE_PARAMS['v_L_V_error'], TRUE_PARAMS['v_L_V_correct']],
         [TRUE_PARAMS['v_R_V_error'], TRUE_PARAMS['v_R_V_correct']]]
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

def build_joint_model(data):
    """Build joint estimation model with moderately informative priors"""
    logger.info("\n" + "="*70)
    logger.info("JOINT ESTIMATION MODEL")
    logger.info("="*70)

    with pm.Model() as model:
        # Priors
        v_L_H_correct = pm.TruncatedNormal('v_L_H_correct', mu=3.5, sigma=0.5, lower=2.0, upper=5.0)
        v_L_H_error = pm.TruncatedNormal('v_L_H_error', mu=0.8, sigma=0.3, lower=0.1, upper=1.5)
        v_L_V_correct = pm.TruncatedNormal('v_L_V_correct', mu=3.5, sigma=0.5, lower=2.0, upper=5.0)
        v_L_V_error = pm.TruncatedNormal('v_L_V_error', mu=0.8, sigma=0.3, lower=0.1, upper=1.5)
        v_R_H_correct = pm.TruncatedNormal('v_R_H_correct', mu=3.5, sigma=0.5, lower=2.0, upper=5.0)
        v_R_H_error = pm.TruncatedNormal('v_R_H_error', mu=0.8, sigma=0.3, lower=0.1, upper=1.5)
        v_R_V_correct = pm.TruncatedNormal('v_R_V_correct', mu=3.5, sigma=0.5, lower=2.0, upper=5.0)
        v_R_V_error = pm.TruncatedNormal('v_R_V_error', mu=0.8, sigma=0.3, lower=0.1, upper=1.5)

        # Build v_tensor
        v_tensor = pt.stack([
            pt.stack([pt.stack([v_L_V_error, v_L_V_correct]),
                     pt.stack([v_R_H_correct, v_R_H_error])]),
            pt.stack([pt.stack([v_L_H_correct, v_L_H_error]),
                     pt.stack([v_R_H_correct, v_R_H_error])]),
            pt.stack([pt.stack([v_L_H_correct, v_L_H_error]),
                     pt.stack([v_R_V_error, v_R_V_correct])]),
            pt.stack([pt.stack([v_L_V_error, v_L_V_correct]),
                     pt.stack([v_R_V_error, v_R_V_correct])])
        ])

        # Likelihood
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

    logger.info("✓ Model built successfully")
    return model

def main():
    start_time = datetime.now()
    logger.info("\n" + "="*70)
    logger.info("GRT-LBA JOINT ESTIMATION - 50K DRAWS")
    logger.info(f"Start: {start_time}")
    logger.info("="*70)

    data = generate_data(n_trials_per_condition=300, seed=42)

    logger.info("\n" + "="*70)
    logger.info("MCMC SAMPLING CONFIGURATION")
    logger.info(f"  Sampler: DEMetropolisZ")
    logger.info(f"  Draws: {MCMC_CONFIG['draws']:,}")
    logger.info(f"  Tune: {MCMC_CONFIG['tune']:,}")
    logger.info(f"  Chains: {MCMC_CONFIG['chains']}")
    logger.info(f"  Estimated time: ~60 minutes")
    logger.info("="*70)

    model = build_joint_model(data)

    with model:
        trace = pm.sample(
            draws=MCMC_CONFIG['draws'],
            tune=MCMC_CONFIG['tune'],
            chains=MCMC_CONFIG['chains'],
            cores=MCMC_CONFIG['cores'],
            step=pm.DEMetropolisZ(tune='scaling', scaling=0.01, proposal_dist=pm.NormalProposal),
            return_inferencedata=True,
            progressbar=True,
            discard_tuned_samples=True
        )

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
        'joint_50k': {'summary': summary.to_dict(), 'true_params': TRUE_PARAMS},
        'metadata': {
            'start_time': str(start_time), 'end_time': str(end_time),
            'elapsed_seconds': elapsed, 'mcmc_config': MCMC_CONFIG,
            'lba_params': LBA_FIXED_PARAMS, 't0_params': T0_PARAMS
        }
    }

    with open('flat_JOINT_50k_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n" + "="*70)
    logger.info(f"✓ Completed in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"✓ Results saved to flat_JOINT_50k_results.json")
    logger.info("="*70)

if __name__ == '__main__':
    main()
