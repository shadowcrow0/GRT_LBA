# GRT-LBA Model Documentation
## General Recognition Theory - Linear Ballistic Accumulator

**Authors**: YYC & Claude
**Date**: 2025-11-10

---

## Table of Contents
1. [Model Overview](#model-overview)
2. [Parameter Definitions](#parameter-definitions)
3. [Model Architecture](#model-architecture)
4. [Implementation Details](#implementation-details)
5. [Current Findings](#current-findings)
6. [Critical Bugs Fixed](#critical-bugs-fixed)
7. [Estimation Challenges](#estimation-challenges)

---

## Model Overview

The GRT-LBA model combines **General Recognition Theory (GRT)** with the **Linear Ballistic Accumulator (LBA)** framework to model two-dimensional perceptual decision-making tasks. This model assumes **perceptual separability**, meaning that processing of the left dimension is independent of the right stimulus, and vice versa.

### Task Structure
- **4 stimulus conditions**: VH, HH, HV, VV (where V=Vertical, H=Horizontal)
- **4 response choices**: corresponding to the 4 conditions
- **Decision process**: Two independent 2-choice accumulators (left and right dimensions) compete in parallel
- **Response time**: RT = max(RT_left, RT_right) - the slower dimension determines overall RT

---

## Parameter Definitions

### 1. Drift Rate Parameters (8 total)

These parameters represent the rate of evidence accumulation for each perceptual decision:

#### Left Dimension Parameters
- **`v_L_H_correct`** (3.5): Drift rate when left stimulus is **H** and accumulator for **H** is evaluated
- **`v_L_H_error`** (0.8): Drift rate when left stimulus is **H** but accumulator for **V** is evaluated
- **`v_L_V_correct`** (3.5): Drift rate when left stimulus is **V** and accumulator for **V** is evaluated
- **`v_L_V_error`** (0.8): Drift rate when left stimulus is **V** but accumulator for **H** is evaluated

#### Right Dimension Parameters
- **`v_R_H_correct`** (3.5): Drift rate when right stimulus is **H** and accumulator for **H** is evaluated
- **`v_R_H_error`** (0.8): Drift rate when right stimulus is **H** but accumulator for **V** is evaluated
- **`v_R_V_correct`** (3.5): Drift rate when right stimulus is **V** and accumulator for **V** is evaluated
- **`v_R_V_error`** (0.8): Drift rate when right stimulus is **V** but accumulator for **H** is evaluated

**Note**: Values in parentheses are the true parameters used for data generation. The model aims to recover these values from observed RT and choice data.

### 2. LBA Framework Parameters (Fixed)

These structural parameters are held constant during estimation:

- **`A`** (0.5): Starting point variability - maximum uniform distribution of initial evidence
- **`b`** (1.0): Decision threshold - amount of evidence needed to trigger a response
- **`s`** (1.0): Within-trial noise - standard deviation of drift rate variability
- **`t0`** (0.10): Non-decision time - motor execution time after decision is made

---

## Model Architecture

### Flat Two-Dimension Architecture

```
Stimulus: [Left: H or V] × [Right: H or V]
              ↓                    ↓
         LEFT DIM              RIGHT DIM
         ┌─────────┐          ┌─────────┐
         │  H-acc  │          │  H-acc  │
         │  v_L_H  │          │  v_R_H  │
         └─────────┘          └─────────┘
              vs                   vs
         ┌─────────┐          ┌─────────┐
         │  V-acc  │          │  V-acc  │
         │  v_L_V  │          │  v_R_V  │
         └─────────┘          └─────────┘
              ↓                    ↓
          RT_left              RT_right
              └────────┬────────┘
                       ↓
            RT = max(RT_left, RT_right)
            Choice = faster accumulator
```

### Perceptual Separability Assumption

**Key principle**: Left dimension drift rates depend ONLY on the left stimulus, and right dimension drift rates depend ONLY on the right stimulus.

Example for condition **VH** (Left sees V, Right sees H):
```
Left dimension:
  - H accumulator gets v_L_V_error (slow, because stimulus is V not H)
  - V accumulator gets v_L_V_correct (fast, matches stimulus)

Right dimension:
  - H accumulator gets v_R_H_correct (fast, matches stimulus)
  - V accumulator gets v_R_H_error (slow, because stimulus is H not V)
```

### v_tensor Structure

The model uses a 3D tensor to organize drift rates:

```python
v_tensor[condition, dimension, accumulator]
# Shape: (4 conditions, 2 dimensions, 2 accumulators)

v_tensor = [
    # Condition 0: VH - Left sees V, Right sees H
    [[v_L_V_error, v_L_V_correct],  # Left: V wins
     [v_R_H_correct, v_R_H_error]], # Right: H wins

    # Condition 1: HH - Left sees H, Right sees H
    [[v_L_H_correct, v_L_H_error],  # Left: H wins
     [v_R_H_correct, v_R_H_error]], # Right: H wins

    # Condition 2: HV - Left sees H, Right sees V
    [[v_L_H_correct, v_L_H_error],  # Left: H wins
     [v_R_V_error, v_R_V_correct]], # Right: V wins

    # Condition 3: VV - Left sees V, Right sees V
    [[v_L_V_error, v_L_V_correct],  # Left: V wins
     [v_R_V_error, v_R_V_correct]]  # Right: V wins
]
```

**Critical Note**: Conditions are **0-indexed** (0, 1, 2, 3), not 1-indexed!

---

## Implementation Details

### PyMC Model Structure

```python
with pm.Model() as model:
    # Priors (moderately informative)
    v_L_H_correct = pm.TruncatedNormal('v_L_H_correct', mu=3.5, sigma=0.5,
                                        lower=2.0, upper=5.0)
    v_L_H_error = pm.TruncatedNormal('v_L_H_error', mu=0.8, sigma=0.3,
                                      lower=0.1, upper=1.5)
    # ... (6 more parameters)

    # Likelihood (custom PyTensor Op)
    log_lik = GRT_LBA_4Choice_LogLik()(choice, rt, condition, v_tensor,
                                        A, b, t0, s)
    pm.Potential('logp', log_lik)
```

### MCMC Sampler

**DEMetropolisZ** is used instead of NUTS because:
- Custom PyTensor Op `GRT_LBA_4Choice_LogLik` has no gradient implementation
- NUTS requires differentiable log-probability
- DEMetropolisZ uses Differential Evolution for gradient-free sampling

```python
step = pm.DEMetropolisZ(
    tune='scaling',
    scaling=0.01,
    proposal_dist=pm.NormalProposal
)
```

### Data Generation

- **300 trials per condition** (1,200 total trials)
- True parameters: correct=3.5, error=0.8
- Seed=42 for reproducibility
- Generated accuracy: ~82-85% (consistent with high correct/error drift rate ratio)

---

## Current Findings

### Joint Estimation Results (10,000 draws, 7 minutes)

#### Left Dimension Parameters ✅
| Parameter | True | Estimated | Error % | R-hat | ESS |
|-----------|------|-----------|---------|-------|-----|
| v_L_H_correct | 3.50 | 3.364 | 3.9% | 1.48 | 8 |
| v_L_H_error | 0.80 | 0.647 | 19.1% | 1.70 | 6 |
| v_L_V_correct | 3.50 | 3.360 | 4.0% | 1.48 | 8 |
| v_L_V_error | 0.80 | 0.649 | 18.9% | 1.69 | 5 |

**Status**: Reasonably accurate recovery, though convergence is poor

#### Right Dimension Parameters ❌
| Parameter | True | Estimated | Error % | R-hat | ESS |
|-----------|------|-----------|---------|-------|-----|
| v_R_H_correct | 3.50 | 2.494 | 28.7% | 2.45 | 6 |
| v_R_H_error | 0.80 | 0.502 | 37.3% | 2.01 | 7 |
| v_R_V_correct | 3.50 | 2.699 | 22.9% | 1.98 | 6 |
| v_R_V_error | 0.80 | 0.501 | 37.4% | 2.04 | 6 |

**Status**: **Systematic underestimation**, poor convergence

### Key Observations

1. **Left vs Right Asymmetry**: Left dimension parameters show 4-19% error, while right dimension shows 23-37% error. This asymmetry is unexpected given the symmetric model structure.

2. **Convergence Failure**:
   - R-hat values range from 1.48 to 2.45 (should be <1.01 for convergence)
   - ESS (Effective Sample Size) = 5-8 (should be >400 for reliable inference)
   - Indicates chains are exploring different regions of parameter space

3. **Error Drift Rate Bias**: Error drift rates are consistently underestimated across both dimensions, suggesting possible:
   - Model identifiability issues
   - Sampler exploration problems
   - v_tensor construction errors (though code review suggests this is correct)

---

## Critical Bugs Fixed

### Bug 1: LBA Threshold Parameter Mismatch
**File**: All estimation scripts
**Problem**: Data generated with `b=1.0`, but models estimated with `b=1.5`
**Effect**: Error drift rates estimated at ~1.5 instead of true value 0.8 (86% error)
**Fix**: Changed `LBA_FIXED_PARAMS['b']` from 1.5 to 1.0 in all scripts
**Result**: Error drift rates improved to 0.5-0.7 range (9-37% error)

### Bug 2: Condition Indexing Error
**File**: `grt_lba_4choice_correct.py` (line 369)
**Problem**: Used `v_tensor[cond - 1]` assuming 1-indexed conditions, but data uses 0-indexed (0,1,2,3)
**Effect**: All condition data misinterpreted, causing systematic parameter bias
**Fix**:
```python
# BEFORE (WRONG)
v_left = v_tensor[cond - 1, 0, :]

# AFTER (CORRECT)
v_left = v_tensor[cond, 0, :]
```
**Status**: Identified but NOT yet applied and tested

### Bug 3: NUTS Sampler Incompatibility
**Problem**: Attempted to use NUTS sampler with `target_accept=0.95`
**Error**: `ValueError: Model can not be sampled with NUTS alone. It either has discrete variables or a non-differentiable log-probability.`
**Cause**: Custom PyTensor Op has no gradient implementation
**Fix**: Must use DEMetropolisZ (gradient-free sampler)
**Limitation**: DEMetropolisZ is less efficient for 8-dimensional parameter space

---

## Estimation Challenges

### 1. Sampler Efficiency
- **DEMetropolisZ**: Only viable option, but inefficient for 8D space
  - 10,000 draws → ESS of 5-8 (0.05-0.08% efficiency)
  - Requires 50,000+ draws for adequate ESS

- **NUTS**: Ideal sampler, but requires gradient implementation
  - Would need to implement custom gradient for PyTensor Op
  - Non-trivial engineering effort

- **SMC (Sequential Monte Carlo)**: Tested but too slow
  - 4,000 trials, 15,000 draws → ran for 42+ minutes with no output
  - Does not scale well to this problem size

### 2. Model Identifiability
- **Constraint**: Only 2 dimensions of information (RT_left, RT_right), but 8 parameters
- **Challenge**: Multiple parameter combinations may produce similar RT distributions
- **Mitigation**: Using moderately informative priors (mu at true values, sigma=0.3-0.5)

### 3. Left-Right Asymmetry (Unresolved)
**Observation**: Right dimension parameters systematically underestimated

**Possible causes**:
1. **v_tensor construction error**: Unlikely, code review shows correct structure
2. **Data generation artifact**: Left stimulus processed first?
3. **Sampler bias**: DEMetropolisZ exploring left parameters more effectively?
4. **Model assumption violation**: Perceptual separability may not hold perfectly in generated data

**Status**: Requires further investigation

### 4. Next Steps

#### Immediate (In Progress):
1. ✅ **Terminate all running processes** (completed)
2. 🔄 **Run 50,000 draw estimation** to improve ESS and convergence
3. ✅ **Create comprehensive documentation** (this document)

#### Follow-up:
1. **Apply condition indexing fix** and re-run estimation
2. **Investigate right dimension bias**:
   - Check v_tensor values during sampling
   - Compare left vs right likelihood contributions
   - Test with swapped left/right labels
3. **Explore alternative samplers**:
   - Implement gradient for NUTS
   - Try adaptive Metropolis variants
4. **Model diagnostics**:
   - Posterior predictive checks
   - Parameter correlation analysis
   - Prior sensitivity analysis

---

## File Reference

### Core Implementation
- [`grt_lba_4choice_correct.py`](grt_lba_4choice_correct.py): Likelihood function (has condition indexing bug)
- [`flat_staged_estimation.py`](flat_staged_estimation.py): Staged approach (abandoned)
- [`flat_JOINT_estimation.py`](flat_JOINT_estimation.py): Joint estimation (10k draws, completed)
- [`flat_JOINT_50k.py`](flat_JOINT_50k.py): High-resolution joint estimation (50k draws, not yet run)

### Results
- [`flat_staged_estimation_results.json`](flat_staged_estimation_results.json): Staged results (biased)
- [`flat_JOINT_estimation_results.json`](flat_JOINT_estimation_results.json): Joint results (10k draws)

---

## References

**Linear Ballistic Accumulator (LBA)**:
- Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. *Cognitive Psychology, 57*(3), 153-178.

**General Recognition Theory (GRT)**:
- Ashby, F. G., & Townsend, J. T. (1986). Varieties of perceptual independence. *Psychological Review, 93*(2), 154-179.

**PyMC Documentation**:
- https://www.pymc.io/

---

**END OF DOCUMENTATION**


多個局部最優解 Multiple Local Optima:
