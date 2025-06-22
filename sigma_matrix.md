# Sigma Matrix Analysis for LBA Dual-Channel Model - Complete Code Explanation

## Overview
This code implements a comprehensive sigma matrix analysis system for examining variance-covariance structures between left and right processing channels in Linear Ballistic Accumulator (LBA) models. The analysis tests fundamental assumptions about channel independence and provides evidence for different judgment mechanisms in cognitive processing.

---

## Part 1: Channel Parameter Extraction

### Function: `extract_channel_parameters(results_df: pd.DataFrame, original_df: pd.DataFrame)`

**Purpose**: Extract left and right channel-specific parameters from behavioral data for variance-covariance analysis

**Implementation**: 
- Processes successful model fits to extract channel-specific behavioral measures
- Calculates bias, sensitivity, consistency, and speed measures for each channel
- Creates structured dataset for sigma matrix calculations

### Key Variables

#### Input Data Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `results_df` | pandas.DataFrame | LBA model fitting results with success indicators |
| `original_df` | pandas.DataFrame | Original experimental data with trial-by-trial responses |
| `channel_data` | list | Collection of channel-specific measures across subjects |

#### Channel Parameter Structure
```python
channel_data.append({
    'subject_id': subject_id,
    'left_bias': left_measures['bias'],
    'left_sensitivity': left_measures['sensitivity'],
    'left_consistency': left_measures['consistency'],
    'left_speed': left_measures['speed'],
    'right_bias': right_measures['bias'],
    'right_sensitivity': right_measures['sensitivity'],
    'right_consistency': right_measures['consistency'],
    'right_speed': right_measures['speed'],
    'accuracy': subject_data['Correct'].mean()
})
```

| Parameter | Channel | Purpose | Range |
|-----------|---------|---------|-------|
| `left_bias` | Left | Tendency to report left diagonal orientations | -0.5 to +0.5 |
| `left_sensitivity` | Left | Accuracy in left channel discrimination | 0 to 1 |
| `left_consistency` | Left | Reliability of left channel responses | 0 to 1 |
| `left_speed` | Left | Processing speed for left channel | Positive values |
| `right_bias` | Right | Tendency to report right diagonal orientations | -0.5 to +0.5 |
| `right_sensitivity` | Right | Accuracy in right channel discrimination | 0 to 1 |
| `right_consistency` | Right | Reliability of right channel responses | 0 to 1 |
| `right_speed` | Right | Processing speed for right channel | Positive values |

---

## Part 2: Left Channel Measures Calculation

### Function: `calculate_left_channel_measures(choices, rts, stimuli)`

**Purpose**: Compute left channel-specific cognitive measures from behavioral responses

**Implementation**: Analyzes responses to stimuli with left diagonal vs vertical orientations

### Stimulus-Response Mapping

#### Left Channel Stimulus Categories
```python
left_diagonal_trials = np.isin(stimuli, [0, 1])  # Stimuli with left\ 
left_vertical_trials = np.isin(stimuli, [2, 3])   # Stimuli with left|
```

| Stimulus Code | Left Orientation | Description |
|---------------|------------------|-------------|
| 0 | Diagonal (\) | Top-left stimulus with left diagonal |
| 1 | Diagonal (\) | Bottom-left stimulus with left diagonal |
| 2 | Vertical (\|) | Top-right stimulus with left vertical |
| 3 | Vertical (\|) | Bottom-right stimulus with left vertical |

#### Left Channel Response Categories
```python
left_diagonal_responses = np.isin(choices, [0, 1])
```

| Response Code | Interpretation | Left Channel Perception |
|---------------|----------------|------------------------|
| 0, 1 | Left diagonal perceived | Subject saw left diagonal orientation |
| 2, 3 | Left vertical perceived | Subject saw left vertical orientation |

### Left Channel Measures

#### Bias Calculation
```python
bias = np.mean(left_diagonal_responses) - 0.5
```

**Purpose**: Quantify systematic tendency to perceive left diagonal orientations  
**Range**: -0.5 (strong vertical bias) to +0.5 (strong diagonal bias)  
**Interpretation**: 0 = no bias, positive = diagonal preference, negative = vertical preference

#### Sensitivity Calculation
```python
left_diag_correct = np.mean(left_diagonal_responses[left_diagonal_trials])
left_vert_correct = np.mean(~left_diagonal_responses[left_vertical_trials])
sensitivity = (left_diag_correct + left_vert_correct) / 2
```

**Purpose**: Measure accuracy in discriminating left diagonal vs vertical orientations  
**Range**: 0 (chance performance) to 1 (perfect discrimination)  
**Implementation**: Average of hit rate for diagonals and correct rejection rate for verticals

#### Consistency Calculation
```python
response_entropy = -np.sum([p * np.log(p + 1e-10) for p in [np.mean(left_diagonal_responses), 1 - np.mean(left_diagonal_responses)]])
consistency = 1 - (response_entropy / np.log(2))
```

**Purpose**: Quantify reliability and predictability of left channel responses  
**Range**: 0 (maximum inconsistency) to 1 (perfect consistency)  
**Implementation**: Inverse normalized entropy of response distribution

#### Speed Calculation
```python
speed = 1 / np.mean(rts)
```

**Purpose**: Measure processing efficiency for left channel decisions  
**Implementation**: Inverse of mean reaction time (higher values = faster processing)

---

## Part 3: Right Channel Measures Calculation

### Function: `calculate_right_channel_measures(choices, rts, stimuli)`

**Purpose**: Compute right channel-specific cognitive measures from behavioral responses

**Implementation**: Analyzes responses to stimuli with right vertical vs diagonal orientations

### Right Channel Stimulus-Response Mapping

#### Right Channel Stimulus Categories
```python
right_vertical_trials = np.isin(stimuli, [0, 2])  # Stimuli with right|
right_diagonal_trials = np.isin(stimuli, [1, 3])  # Stimuli with right/
```

| Stimulus Code | Right Orientation | Description |
|---------------|-------------------|-------------|
| 0 | Vertical (\|) | Top-left stimulus with right vertical |
| 2 | Vertical (\|) | Top-right stimulus with right vertical |
| 1 | Diagonal (/) | Bottom-left stimulus with right diagonal |
| 3 | Diagonal (/) | Bottom-right stimulus with right diagonal |

#### Right Channel Response Categories
```python
right_diagonal_responses = np.isin(choices, [1, 3])
```

| Response Code | Interpretation | Right Channel Perception |
|---------------|----------------|-------------------------|
| 1, 3 | Right diagonal perceived | Subject saw right diagonal orientation |
| 0, 2 | Right vertical perceived | Subject saw right vertical orientation |

### Right Channel Measures

The right channel measures follow identical computational principles to left channel measures:

| Measure | Purpose | Implementation |
|---------|---------|----------------|
| `bias` | Right channel orientation preference | Proportion of diagonal responses - 0.5 |
| `sensitivity` | Right channel discrimination accuracy | Average hit rate and correct rejection rate |
| `consistency` | Right channel response reliability | 1 - normalized entropy |
| `speed` | Right channel processing efficiency | Inverse mean reaction time |

---

## Part 4: Sigma Matrix Calculations

### Function: `calculate_sigma_matrices(channel_df: pd.DataFrame)`

**Purpose**: Compute comprehensive variance-covariance matrices between left and right processing channels

**Implementation**: Calculates within-channel, cross-channel, and bilateral covariance structures

### Matrix Definitions

#### Variable Organization
```python
left_vars = ['left_bias', 'left_sensitivity', 'left_consistency', 'left_speed']
right_vars = ['right_bias', 'right_sensitivity', 'right_consistency', 'right_speed']
all_vars = left_vars + right_vars
```

| Variable Set | Components | Purpose |
|--------------|------------|---------|
| `left_vars` | 4 left channel measures | Within-left-channel analysis |
| `right_vars` | 4 right channel measures | Within-right-channel analysis |
| `all_vars` | 8 bilateral measures | Full cross-channel analysis |

#### Individual Channel Matrices
```python
left_data = channel_df[left_vars].values
right_data = channel_df[right_vars].values

sigma_left = np.cov(left_data.T)
sigma_right = np.cov(right_data.T)
```

| Matrix | Dimensions | Purpose |
|--------|------------|---------|
| `sigma_left` | 4×4 | Covariance within left channel parameters |
| `sigma_right` | 4×4 | Covariance within right channel parameters |

#### Cross-Channel Matrix
```python
sigma_cross = np.zeros((len(left_vars), len(right_vars)))
for i, left_var in enumerate(left_vars):
    for j, right_var in enumerate(right_vars):
        sigma_cross[i, j] = np.cov(channel_df[left_var], channel_df[right_var])[0, 1]
```

| Matrix | Dimensions | Purpose |
|--------|------------|---------|
| `sigma_cross` | 4×4 | Covariance between left and right channel parameters |

**Implementation**: Pairwise covariances between all left-right parameter combinations

#### Bilateral Matrix
```python
bilateral_data = channel_df[all_vars].values
sigma_bilateral = np.cov(bilateral_data.T)
```

| Matrix | Dimensions | Purpose |
|--------|------------|---------|
| `sigma_bilateral` | 8×8 | Complete covariance structure across all parameters |

#### Correlation Matrices
```python
corr_left = np.corrcoef(left_data.T)
corr_right = np.corrcoef(right_data.T)
corr_bilateral = np.corrcoef(bilateral_data.T)
```

**Purpose**: Standardized covariances for easier interpretation  
**Range**: -1 (perfect negative correlation) to +1 (perfect positive correlation)

---

## Part 5: Independence Evidence Analysis

### Function: `analyze_independence_evidence(sigma_results: Dict, channel_df: pd.DataFrame)`

**Purpose**: Test statistical evidence for channel independence using correlation analysis

**Implementation**: Statistical significance testing of cross-channel correlations

### Statistical Testing Framework

#### Correlation Significance Testing
```python
n_subjects = len(channel_df)
for i, left_var in enumerate(left_vars):
    for j, right_var in enumerate(right_vars):
        r = corr_cross[i, j]
        t_stat = r * np.sqrt((n_subjects - 2) / (1 - r**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_subjects - 2))
```

| Variable | Purpose | Formula |
|----------|---------|---------|
| `r` | Cross-channel correlation coefficient | Pearson correlation |
| `t_stat` | Test statistic for correlation | r√((n-2)/(1-r²)) |
| `p_value` | Statistical significance | Two-tailed t-test |

#### Independence Criteria
```python
if p_value < 0.05:
    significant_corrs.append((left_var, right_var, r, p_value))
```

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Significance level | p < 0.05 | Statistically significant correlation |
| Independence verdict | No significant correlations | Channels operate independently |
| Violation evidence | Any significant correlations | Cross-channel dependence detected |

### Separability Analysis

#### Bias and Sensitivity Correlations
```python
bias_corr = corr_cross[0, 0]  # left_bias × right_bias
sens_corr = corr_cross[1, 1]  # left_sensitivity × right_sensitivity
```

| Correlation | Variables | Theoretical Implication |
|-------------|-----------|------------------------|
| `bias_corr` | Left bias × Right bias | Systematic response tendencies |
| `sens_corr` | Left sensitivity × Right sensitivity | Perceptual discrimination abilities |

#### Separability Criteria
```python
separability_supported = abs(bias_corr) <= 0.3 and abs(sens_corr) <= 0.3
```

| Threshold | Interpretation | Theoretical Support |
|-----------|----------------|-------------------|
| ≤ 0.3 | Weak correlation | Separable processing channels |
| > 0.3 | Strong correlation | Integrated processing system |

### Channel Dominance Analysis

#### Variance Comparison
```python
left_variance = np.mean([np.var(channel_df[var]) for var in left_vars])
right_variance = np.mean([np.var(channel_df[var]) for var in right_vars])
variance_ratio = left_variance / right_variance
```

| Measure | Purpose | Interpretation |
|---------|---------|----------------|
| `left_variance` | Average variability in left channel | Individual differences magnitude |
| `right_variance` | Average variability in right channel | Individual differences magnitude |
| `variance_ratio` | Relative channel variability | Processing asymmetry indicator |

#### Symmetry Criteria
| Ratio Range | Interpretation |
|-------------|----------------|
| 0.5 - 2.0 | Symmetric processing |
| < 0.5 or > 2.0 | Asymmetric processing |

---

## Part 6: Visualization System

### Function: `visualize_sigma_matrices(sigma_results: Dict, channel_df: pd.DataFrame)`

**Purpose**: Create comprehensive visual representations of all variance-covariance structures

**Implementation**: 3×3 grid of specialized heatmaps and scatter plots

### Visualization Components

#### Covariance Heatmaps (Row 1)
```python
# Left channel covariance
sns.heatmap(sigma_results['sigma_left'], 
            xticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
            yticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
            annot=True, fmt='.3f', cmap='RdBu_r', center=0)
```

| Plot Position | Matrix | Purpose |
|---------------|--------|---------|
| (0,0) | Left channel covariance | Within-left relationships |
| (0,1) | Right channel covariance | Within-right relationships |
| (0,2) | Cross-channel covariance | Between-channel relationships |

#### Correlation Heatmaps (Row 2)
```python
# Cross-channel correlation
sns.heatmap(sigma_results['corr_cross'],
            annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1)
```

| Plot Position | Matrix | Color Scale | Purpose |
|---------------|--------|-------------|---------|
| (1,0) | Left channel correlation | -1 to +1 | Standardized within-left relationships |
| (1,1) | Right channel correlation | -1 to +1 | Standardized within-right relationships |
| (1,2) | Cross-channel correlation | -1 to +1 | Standardized between-channel relationships |

#### Comprehensive Analysis (Row 3)
```python
# Full bilateral correlation matrix
sns.heatmap(sigma_results['corr_bilateral'],
            xticklabels=full_labels, yticklabels=full_labels,
            annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1)

# Bias relationship scatter plot
ax.scatter(channel_df['left_bias'], channel_df['right_bias'], alpha=0.7)
r = np.corrcoef(channel_df['left_bias'], channel_df['right_bias'])[0, 1]
```

| Plot Position | Content | Purpose |
|---------------|---------|---------|
| (2,0-1) | Full 8×8 correlation matrix | Complete bilateral structure |
| (2,2) | Left vs right bias scatter | Key relationship visualization |

---

## Part 7: Judgment Mechanism Report

### Function: `generate_judgment_mechanism_report(sigma_results: Dict, independence_results: Dict, channel_df: pd.DataFrame)`

**Purpose**: Generate comprehensive theoretical interpretation of sigma matrix findings

**Implementation**: Structured text report with cognitive implications

### Report Components

#### Independence Assessment
```python
if independence_results['independence_supported']:
    print("✅ CHANNEL INDEPENDENCE: Supported")
else:
    print("❌ CHANNEL INDEPENDENCE: Violated")
    print(f"   • {len(independence_results['significant_correlations'])} significant cross-correlations")
```

| Outcome | Evidence | Theoretical Implication |
|---------|----------|------------------------|
| Independence supported | No significant cross-correlations | Modular processing architecture |
| Independence violated | Significant cross-correlations | Integrated processing system |

#### Separability Assessment
```python
if independence_results['separability_supported']:
    print("✅ PERCEPTUAL SEPARABILITY: Supported")
else:
    print("❌ PERCEPTUAL SEPARABILITY: Violated")
```

| Outcome | Evidence | Cognitive Interpretation |
|---------|----------|-------------------------|
| Separability supported | Weak bias correlations | Independent perceptual channels |
| Separability violated | Strong bias correlations | Holistic perception |

#### Processing Symmetry Assessment
```python
variance_ratio = independence_results['variance_ratio']
if 0.5 < variance_ratio < 2.0:
    print("⚖️ PROCESSING SYMMETRY: Supported")
else:
    print("⚖️ PROCESSING SYMMETRY: Asymmetric")
```

| Variance Ratio | Interpretation | Neural Implication |
|----------------|----------------|-------------------|
| 0.5 - 2.0 | Symmetric processing | Balanced hemispheric function |
| < 0.5 or > 2.0 | Asymmetric processing | Hemispheric specialization |

### Theoretical Implications

#### Evidence Against Independence
| Finding | Implication |
|---------|-------------|
| Bilateral integration | Decision-making involves cross-channel communication |
| Cross-talk between streams | Information sharing during processing |
| Violation of independence | Standard LBA assumptions invalid |

#### Evidence Against Separability
| Finding | Implication |
|---------|-------------|
| Cross-channel perception influence | Non-modular perceptual architecture |
| Holistic processing | Gestalt-like integration mechanisms |
| Non-featural processing | Global rather than local analysis |

---

## Part 8: Main Analysis Pipeline

### Function: `main_sigma_analysis(results_file: str, original_file: str)`

**Purpose**: Coordinate complete sigma matrix analysis workflow

**Implementation**: Sequential execution of all analysis components

### Pipeline Sequence

#### Data Loading and Processing
```python
results_df = pd.read_csv(results_file)
original_df = pd.read_csv(original_file)
channel_df = extract_channel_parameters(results_df, original_df)
```

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| Load results | CSV file | DataFrame | Model fitting outcomes |
| Load original | CSV file | DataFrame | Behavioral trial data |
| Extract parameters | Both DataFrames | Channel measures | Sigma analysis input |

#### Statistical Analysis
```python
sigma_results = calculate_sigma_matrices(channel_df)
independence_results = analyze_independence_evidence(sigma_results, channel_df)
```

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| Calculate matrices | Channel parameters | Covariance/correlation matrices | Mathematical foundation |
| Test independence | Matrices and data | Statistical test results | Theoretical evaluation |

#### Visualization and Reporting
```python
visualize_sigma_matrices(sigma_results, channel_df)
generate_judgment_mechanism_report(sigma_results, independence_results, channel_df)
```

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| Create visualizations | All results | PNG file | Visual interpretation |
| Generate report | All results | Text summary | Theoretical conclusions |

### Return Structure
```python
return {
    'sigma_results': sigma_results,
    'independence_results': independence_results,
    'channel_df': channel_df
}
```

| Component | Content | Usage |
|-----------|---------|-------|
| `sigma_results` | All matrices and correlations | Mathematical analysis |
| `independence_results` | Statistical test outcomes | Theoretical interpretation |
| `channel_df` | Channel parameter data | Further analysis |

---

## Scientific Significance

### Theoretical Contributions

#### LBA Model Validation
- **Independence Testing**: Empirical evaluation of core LBA assumptions
- **Channel Architecture**: Evidence for dual vs integrated processing
- **Parameter Relationships**: Identification of systematic dependencies

#### Cognitive Architecture Insights
- **Modular vs Holistic**: Evidence for processing organization
- **Hemispheric Function**: Symmetry vs specialization patterns
- **Integration Mechanisms**: Cross-channel communication evidence

#### Methodological Advances
- **Sigma Matrix Framework**: Novel approach to channel independence testing
- **Behavioral Parameter Extraction**: Sophisticated measure derivation
- **Comprehensive Visualization**: Multi-level analysis presentation

This sigma matrix analysis system provides a rigorous framework for testing fundamental assumptions about cognitive processing architecture and offers empirical evidence for different theoretical models of perceptual decision-making.
