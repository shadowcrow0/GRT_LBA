"""
Flat 架構 - 聯合參數估計 (Joint Estimation)
==============================================

直接估計全部 8 個參數,不使用分階段方法

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

# ============================================================================
# 配置
# ============================================================================

LBA_FIXED_PARAMS = {
    'A': 0.5,
    'b': 1.0,  # FIXED: Must match grt_lba_4choice_correct.py
    's': 1.0
}

T0_PARAMS = {
    't0_left': 0.0,
    't0_right': 0.0,
    'motor_t0': 0.10
}

MCMC_CONFIG = {
    'draws': 10000,
    'tune': 5000,
    'chains': 4,
    'cores': 4
}

TRUE_PARAMS = {
    'v_L_H_correct': 3.5,
    'v_L_H_error': 0.8,
    'v_L_V_correct': 3.5,
    'v_L_V_error': 0.8,
    'v_R_H_correct': 3.5,
    'v_R_H_error': 0.8,
    'v_R_V_correct': 3.5,
    'v_R_V_error': 0.8,
}

# ============================================================================
# 數據生成
# ============================================================================

def generate_data(n_trials_per_condition=300, seed=42):
    """生成 GRT-LBA 數據（使用 PS 假設的真實參數）"""
    logger.info(f"="*70)
    logger.info(f"Generate GRT-LBA Data (Flat Architecture - JOINT ESTIMATION)")
    logger.info(f"="*70)

    v_tensor_true = np.array([
        # Condition 0: VH - Left sees V, Right sees H
        [[TRUE_PARAMS['v_L_V_error'], TRUE_PARAMS['v_L_V_correct']],
         [TRUE_PARAMS['v_R_H_correct'], TRUE_PARAMS['v_R_H_error']]],

        # Condition 1: HH - Left sees H, Right sees H
        [[TRUE_PARAMS['v_L_H_correct'], TRUE_PARAMS['v_L_H_error']],
         [TRUE_PARAMS['v_R_H_correct'], TRUE_PARAMS['v_R_H_error']]],

        # Condition 2: HV - Left sees H, Right sees V
        [[TRUE_PARAMS['v_L_H_correct'], TRUE_PARAMS['v_L_H_error']],
         [TRUE_PARAMS['v_R_V_error'], TRUE_PARAMS['v_R_V_correct']]],

        # Condition 3: VV - Left sees V, Right sees V
        [[TRUE_PARAMS['v_L_V_error'], TRUE_PARAMS['v_L_V_correct']],
         [TRUE_PARAMS['v_R_V_error'], TRUE_PARAMS['v_R_V_correct']]]
    ], dtype=np.float64)

    rng = np.random.RandomState(seed)
    data_array = lba_2dim_random(
        n_trials_per_condition=n_trials_per_condition,
        v_tensor=v_tensor_true,
        A=LBA_FIXED_PARAMS['A'],
        b=LBA_FIXED_PARAMS['b'],
        t0_array=np.array([T0_PARAMS['motor_t0']] * 4),
        s=LBA_FIXED_PARAMS['s'],
        rng=rng
    )

    df = pd.DataFrame(data_array, columns=['rt', 'choice', 'condition'])

    # 計算 accuracy
    n_correct = (df['choice'] == df['condition']).sum()
    accuracy = n_correct / len(df) * 100

    logger.info(f"✓ Generated {len(df)} trials, Accuracy: {accuracy:.1f}%")
    logger.info(f"  Sample data (first 10 rows):")
    logger.info(f"\n{df.head(10)}")

    # Index consistency verification
    logger.info(f"\n  Index consistency check:")
    logger.info(f"    Condition values: {sorted(df['condition'].unique())}")
    logger.info(f"    Choice values: {sorted(df['choice'].unique())}")
    logger.info(f"    Both should be 0-indexed: [0, 1, 2, 3]")

    assert df['condition'].min() >= 0 and df['condition'].max() <= 3, "Condition indices out of range!"
    assert df['choice'].min() >= 0 and df['choice'].max() <= 3, "Choice indices out of range!"
    logger.info(f"    ✓ Index consistency verified")

    return df

# ============================================================================
# 聯合估計模型
# ============================================================================

def build_joint_model(data):
    """
    聯合估計全部 8 個參數 (不分階段)
    使用 moderately informative priors
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"JOINT ESTIMATION: Estimate All 8 Parameters Together")
    logger.info(f"{'='*70}")
    logger.info(f"Estimating: v_L_H_correct, v_L_H_error, v_L_V_correct, v_L_V_error")
    logger.info(f"            v_R_H_correct, v_R_H_error, v_R_V_correct, v_R_V_error")

    with pm.Model() as model:
        # 左邊維度參數 (moderately informative priors)
        v_L_H_correct = pm.TruncatedNormal('v_L_H_correct', mu=3.5, sigma=0.8, lower=1.5, upper=5.5)
        v_L_H_error = pm.TruncatedNormal('v_L_H_error', mu=0.8, sigma=0.5, lower=0.2, upper=2.0)
        v_L_V_correct = pm.TruncatedNormal('v_L_V_correct', mu=3.5, sigma=0.8, lower=1.5, upper=5.5)
        v_L_V_error = pm.TruncatedNormal('v_L_V_error', mu=0.8, sigma=0.5, lower=0.2, upper=2.0)

        # 右邊維度參數 (moderately informative priors)
        v_R_H_correct = pm.TruncatedNormal('v_R_H_correct', mu=3.5, sigma=0.8, lower=1.5, upper=5.5)
        v_R_H_error = pm.TruncatedNormal('v_R_H_error', mu=0.8, sigma=0.5, lower=0.2, upper=2.0)
        v_R_V_correct = pm.TruncatedNormal('v_R_V_correct', mu=3.5, sigma=0.8, lower=1.5, upper=5.5)
        v_R_V_error = pm.TruncatedNormal('v_R_V_error', mu=0.8, sigma=0.5, lower=0.2, upper=2.0)

        # 構建 v_tensor
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

    logger.info("✓ Joint estimation model built")
    return model

# ============================================================================
# 主程序
# ============================================================================

def main():
    start_time = datetime.now()
    logger.info(f"\n{'='*70}")
    logger.info(f"Flat Architecture - JOINT Parameter Estimation")
    logger.info(f"Start: {start_time}")
    logger.info(f"{'='*70}")

    # 生成數據
    data = generate_data(n_trials_per_condition=300, seed=42)

    # 建立聯合估計模型
    logger.info(f"\n{'='*70}")
    logger.info(f"MCMC sampling (DEMetropolisZ - required for custom Op)")
    logger.info(f"  draws={MCMC_CONFIG['draws']}, tune={MCMC_CONFIG['tune']}")
    logger.info(f"  chains={MCMC_CONFIG['chains']}, cores={MCMC_CONFIG['cores']}")
    logger.info(f"  Note: NUTS cannot be used with non-differentiable likelihood")
    logger.info(f"{'='*70}")

    model = build_joint_model(data)

    # MCMC sampling
    with model:
        trace = pm.sample(
            draws=MCMC_CONFIG['draws'],
            tune=MCMC_CONFIG['tune'],
            chains=MCMC_CONFIG['chains'],
            cores=MCMC_CONFIG['cores'],
            # Must use DEMetropolisZ because custom Op has no gradient
            step=pm.DEMetropolisZ(tune='scaling', scaling=0.01, proposal_dist=pm.NormalProposal),
            return_inferencedata=True,
            progressbar=True,
            discard_tuned_samples=True
        )

    # 分析結果
    summary = az.summary(trace)
    logger.info(f"\n{'='*70}")
    logger.info(f"JOINT Estimation Results Summary")
    logger.info(f"{'='*70}")
    logger.info(f"\n{summary}")

    # 計算 parameter recovery accuracy
    logger.info(f"\n{'='*70}")
    logger.info(f"Parameter Recovery Accuracy")
    logger.info(f"{'='*70}")

    for param_name, true_value in TRUE_PARAMS.items():
        est_mean = summary.loc[param_name, 'mean']
        est_sd = summary.loc[param_name, 'sd']
        error_pct = abs(est_mean - true_value) / true_value * 100
        rhat = summary.loc[param_name, 'r_hat']
        ess = summary.loc[param_name, 'ess_bulk']

        logger.info(f"{param_name:20s}: True={true_value:.3f}, Est={est_mean:.3f}±{est_sd:.3f}, "
                   f"Error={error_pct:5.1f}%, R-hat={rhat:.4f}, ESS={ess:.0f}")

    # 保存結果
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    results = {
        'joint_estimation': {
            'summary': summary.to_dict(),
            'true_params': TRUE_PARAMS
        },
        'metadata': {
            'start_time': str(start_time),
            'end_time': str(end_time),
            'elapsed_seconds': elapsed,
            'mcmc_config': MCMC_CONFIG,
            'lba_params': LBA_FIXED_PARAMS,
            't0_params': T0_PARAMS
        }
    }

    output_file = 'flat_JOINT_estimation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Completed in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"✓ Results saved to {output_file}")
    logger.info(f"{'='*70}")

if __name__ == '__main__':
    main()
