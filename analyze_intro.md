# LBA Results Analyzer and Visualizer - Complete Code Explanation

## Overview
This code implements a comprehensive analysis and visualization system for Linear Ballistic Accumulator (LBA) model results from 18 participants. It processes model outputs, calculates derived metrics, and creates detailed visualizations to understand individual differences and cognitive patterns.

---

## Part 1: Data Loading and Processing

### Function: `load_and_process_results(results_file: str, original_data_file: str)`

**Purpose**: Load LBA model results and original experimental data, then compute enhanced metrics for comprehensive analysis

**Implementation**: 
- Loads CSV files containing model results and original experimental data
- Parses choice distribution strings using AST literal evaluation
- Calculates additional behavioral and cognitive metrics for each subject
- Merges model parameters with original experimental performance

### Key Variables

#### Input Data Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `results_df` | pandas.DataFrame | LBA model fitting results for all subjects |
| `original_df` | pandas.DataFrame | Original experimental data with trial-by-trial responses |
| `results_file` | str | Path to CSV file containing model results |
| `original_data_file` | str | Path to CSV file containing original experimental data |

#### Choice Distribution Processing
```python
def parse_choice_dist(dist_str):
    try:
        return ast.literal_eval(dist_str)
    except:
        return {}

results_df['choice_dist_parsed'] = results_df['choice_distribution'].apply(parse_choice_dist)
```

**Purpose**: Convert string representations of choice distributions back to dictionary format  
**Implementation**: Uses AST literal evaluation for safe parsing of Python literals

#### Enhanced Metrics Calculation
```python
# Basic performance metrics
accuracy = subject_data['Correct'].mean()
choice_data = subject_data['Response'].values
rt_data = subject_data['RT'].values

# Choice pattern analysis
choice_counts = {i: np.sum(choice_data == i) for i in range(4)}
total_trials = len(choice_data)
```

| Variable | Purpose | Calculation |
|----------|---------|-------------|
| `accuracy` | Overall task performance | Mean of correct responses |
| `choice_data` | Array of participant responses | Raw response values (0-3) |
| `rt_data` | Array of reaction times | Raw RT values in seconds |
| `choice_counts` | Frequency of each choice type | Count of responses for choices 0-3 |
| `total_trials` | Total number of valid trials | Length of choice data array |

#### Symmetry and Bias Analysis
```python
# Cognitive bias calculations
left_diagonal = choice_counts[0] + choice_counts[1]  # \ choices
left_vertical = choice_counts[2] + choice_counts[3]   # | choices (left)
right_vertical = choice_counts[0] + choice_counts[2]  # | choices (right)
right_diagonal = choice_counts[1] + choice_counts[3]  # / choices

left_bias = (left_diagonal - left_vertical) / total_trials
right_bias = (right_diagonal - right_vertical) / total_trials
```

| Variable | Choice Types | Purpose |
|----------|--------------|---------|
| `left_diagonal` | Choices 0 + 1 (\ orientations) | Count of leftward diagonal line preferences |
| `left_vertical` | Choices 2 + 3 (\| orientations) | Count of vertical line preferences on left |
| `right_vertical` | Choices 0 + 2 (\| orientations) | Count of vertical line preferences on right |
| `right_diagonal` | Choices 1 + 3 (/ orientations) | Count of rightward diagonal line preferences |
| `left_bias` | Normalized bias score | Preference for diagonal vs vertical on left side |
| `right_bias` | Normalized bias score | Preference for diagonal vs vertical on right side |

#### Additional Derived Metrics
```python
enhanced_row.update({
    'absolute_left_bias': abs(left_bias),
    'absolute_right_bias': abs(right_bias),
    'bilateral_bias_diff': abs(left_bias - right_bias),
    'rt_std': np.std(rt_data),
    'choice_entropy': -sum([p * np.log(p) for p in [choice_counts[i]/total_trials for i in range(4)] if p > 0])
})
```

| Variable | Purpose | Range/Interpretation |
|----------|---------|---------------------|
| `absolute_left_bias` | Magnitude of left channel bias | 0-1, higher = stronger bias |
| `absolute_right_bias` | Magnitude of right channel bias | 0-1, higher = stronger bias |
| `bilateral_bias_diff` | Asymmetry between channels | 0-1, higher = more asymmetric |
| `rt_std` | Response time variability | Seconds, higher = more variable |
| `choice_entropy` | Response randomness | 0-log(4), higher = more random |

---

## Part 2: Summary Statistics Calculation

### Function: `calculate_summary_statistics(df: pd.DataFrame)`

**Purpose**: Compute comprehensive descriptive statistics across all subjects for model validation and interpretation

**Implementation**: Calculates means, standard deviations, ranges, and categorical counts for key metrics

### Key Statistical Variables

#### Sample and Success Metrics
```python
stats = {
    'n_subjects': len(df),
    'n_successful': len(successful_df),
    'success_rate': len(successful_df) / len(df),
}
```

| Variable | Purpose | Interpretation |
|----------|---------|----------------|
| `n_subjects` | Total sample size | Number of participants analyzed |
| `n_successful` | Valid model fits | Number of successful convergences |
| `success_rate` | Analysis reliability | Proportion of successful model fits |

#### Performance Statistics
```python
'accuracy_mean': successful_df['accuracy'].mean(),
'accuracy_std': successful_df['accuracy'].std(),
'accuracy_range': [successful_df['accuracy'].min(), successful_df['accuracy'].max()],
```

| Variable | Purpose | Interpretation |
|----------|---------|----------------|
| `accuracy_mean` | Average task performance | Overall group performance level |
| `accuracy_std` | Performance variability | Individual differences magnitude |
| `accuracy_range` | Performance spread | Minimum and maximum accuracy values |

#### Cognitive Bias Statistics
```python
'left_bias_mean': successful_df['left_bias'].mean(),
'right_bias_mean': successful_df['right_bias'].mean(),
'bilateral_bias_diff_mean': successful_df['bilateral_bias_diff'].mean(),
```

| Variable | Purpose | Interpretation |
|----------|---------|----------------|
| `left_bias_mean` | Average left channel bias | Group tendency for left diagonal vs vertical |
| `right_bias_mean` | Average right channel bias | Group tendency for right diagonal vs vertical |
| `bilateral_bias_diff_mean` | Average asymmetry | Group tendency for unequal channel processing |

#### Model Quality Statistics
```python
'rhat_mean': successful_df['rhat_max'].mean(),
'ess_mean': successful_df['ess_min'].mean(),
'sampling_time_mean': successful_df['sampling_time_minutes'].mean(),
```

| Variable | Purpose | Threshold | Interpretation |
|----------|---------|-----------|----------------|
| `rhat_mean` | Convergence quality | ≤ 1.05 | Chain mixing effectiveness |
| `ess_mean` | Sample adequacy | ≥ 100 | Effective posterior samples |
| `sampling_time_mean` | Computational efficiency | Lower better | MCMC computation time |

#### Individual Difference Categories
```python
'accuracy_below_chance': (successful_df['accuracy'] < 0.25).sum(),
'strong_left_bias': (successful_df['absolute_left_bias'] > 0.2).sum(),
'strong_right_bias': (successful_df['absolute_right_bias'] > 0.2).sum(),
'asymmetric_processing': (successful_df['bilateral_bias_diff'] > 0.15).sum()
```

| Variable | Threshold | Purpose |
|----------|-----------|---------|
| `accuracy_below_chance` | < 0.25 | Count of participants performing below random |
| `strong_left_bias` | > 0.2 | Count with pronounced left channel bias |
| `strong_right_bias` | > 0.2 | Count with pronounced right channel bias |
| `asymmetric_processing` | > 0.15 | Count with asymmetric bilateral processing |

---

## Part 3: Overview Visualizations

### Function: `create_overview_visualizations(data: Dict)`

**Purpose**: Generate comprehensive visual summary of LBA analysis results across all subjects

**Implementation**: Creates 16-panel figure with multiple chart types showing different aspects of the analysis

### Visualization Components

#### Analysis Quality Plots (Panels 1-2)
```python
# Panel 1: Success Rate Pie Chart
success_data = [df['success'].sum(), (~df['success']).sum()]
plt.pie(success_data, labels=['Success', 'Failed'], autopct='%1.1f%%', startangle=90)

# Panel 2: Convergence Rate Pie Chart
conv_data = [successful_df['converged'].sum(), (~successful_df['converged']).sum()]
plt.pie(conv_data, labels=['Converged', 'Not Converged'], autopct='%1.1f%%', startangle=90)
```

**Purpose**: Display model fitting success rates and MCMC convergence quality  
**Implementation**: Pie charts showing proportions of successful analyses

#### Performance Distribution Plots (Panels 3-4)
```python
# Panel 3: Accuracy Histogram
plt.hist(successful_df['accuracy'], bins=10, alpha=0.7, edgecolor='black')
plt.axvline(0.25, color='red', linestyle='--', label='Chance Level')

# Panel 4: Reaction Time Histogram
plt.hist(successful_df['mean_rt'], bins=10, alpha=0.7, edgecolor='black')
```

**Purpose**: Show distributions of task performance and response speed  
**Implementation**: Histograms with reference lines for chance performance

#### Model Quality Plots (Panels 5-7)
```python
# Panel 5: Sampling Time Distribution
plt.hist(successful_df['sampling_time_minutes'], bins=10, alpha=0.7, edgecolor='black')

# Panel 6: R-hat Distribution
plt.hist(successful_df['rhat_max'], bins=10, alpha=0.7, edgecolor='black')
plt.axvline(1.05, color='red', linestyle='--', label='Good Convergence')

# Panel 7: ESS Distribution
plt.hist(successful_df['ess_min'], bins=10, alpha=0.7, edgecolor='black')
plt.axvline(100, color='red', linestyle='--', label='Adequate ESS')
```

**Purpose**: Assess computational efficiency and model convergence quality  
**Implementation**: Histograms with threshold lines for acceptable convergence

#### Cognitive Pattern Analysis (Panels 8, 13-16)
```python
# Panel 8: Bilateral Bias Scatter Plot
plt.scatter(successful_df['left_bias'], successful_df['right_bias'], alpha=0.7)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

# Panel 13: Speed-Accuracy Relationship
plt.scatter(successful_df['mean_rt'], successful_df['accuracy'], alpha=0.7)

# Panel 16: Subject Performance Ranking
sorted_df = successful_df.sort_values('accuracy')
colors = ['red' if acc < 0.25 else 'orange' if acc < 0.5 else 'green' for acc in sorted_df['accuracy']]
plt.bar(range(len(sorted_df)), sorted_df['accuracy'], color=colors, alpha=0.7)
```

**Purpose**: Visualize cognitive biases, speed-accuracy tradeoffs, and individual differences  
**Implementation**: Scatter plots and color-coded bar charts

#### Choice Distribution Analysis (Panels 9-12)
```python
for i in range(4):
    choice_col = f'choice_{i}'
    plt.hist(successful_df[choice_col], bins=10, alpha=0.7, edgecolor='black')
```

**Purpose**: Display frequency distributions for each of the four choice options  
**Implementation**: Separate histograms for choices 0-3

---

## Part 4: Individual Differences Analysis

### Function: `create_individual_differences_analysis(data: Dict)`

**Purpose**: Create detailed analysis of individual differences in cognitive processing patterns

**Implementation**: 3x3 grid of specialized plots focusing on individual variation and clustering

### Advanced Analysis Components

#### Performance Categorization (Panel 1)
```python
perf_categories = []
for acc in successful_df['accuracy']:
    if acc < 0.25:
        perf_categories.append('Below Chance')
    elif acc < 0.5:
        perf_categories.append('Poor')
    elif acc < 0.75:
        perf_categories.append('Moderate')
    else:
        perf_categories.append('Good')
```

| Category | Accuracy Range | Interpretation |
|----------|----------------|----------------|
| Below Chance | < 0.25 | Systematic response bias or confusion |
| Poor | 0.25-0.50 | Limited task understanding |
| Moderate | 0.50-0.75 | Adequate performance with room for improvement |
| Good | > 0.75 | Strong task performance |

#### Bias Strength Analysis (Panel 2)
```python
bias_categories = []
for _, row in successful_df.iterrows():
    max_bias = max(row['absolute_left_bias'], row['absolute_right_bias'])
    if max_bias < 0.05:
        bias_categories.append('No Bias')
    elif max_bias < 0.15:
        bias_categories.append('Moderate Bias')
    else:
        bias_categories.append('Strong Bias')
```

| Category | Bias Threshold | Interpretation |
|----------|----------------|----------------|
| No Bias | < 0.05 | Balanced processing across channels |
| Moderate Bias | 0.05-0.15 | Slight preference for certain orientations |
| Strong Bias | > 0.15 | Pronounced systematic bias |

#### Processing Symmetry Analysis (Panel 3)
```python
symmetry_categories = []
for diff in successful_df['bilateral_bias_diff']:
    if diff < 0.05:
        symmetry_categories.append('Symmetric')
    elif diff < 0.15:
        symmetry_categories.append('Mildly Asymmetric')
    else:
        symmetry_categories.append('Strongly Asymmetric')
```

| Category | Asymmetry Level | Interpretation |
|----------|-----------------|----------------|
| Symmetric | < 0.05 | Equal processing between left/right channels |
| Mildly Asymmetric | 0.05-0.15 | Slight processing differences |
| Strongly Asymmetric | > 0.15 | Major hemispheric processing differences |

#### Advanced Visualization Techniques

##### Choice Pattern Heatmap (Panel 4)
```python
choice_matrix = successful_df[['choice_0', 'choice_1', 'choice_2', 'choice_3']].T
sns.heatmap(choice_matrix, annot=False, cmap='viridis', ax=ax)
```

**Purpose**: Display choice usage patterns across all subjects simultaneously  
**Implementation**: Transposed matrix with subjects as columns, choices as rows

##### Correlation Matrix (Panel 5)
```python
corr_vars = ['accuracy', 'mean_rt', 'left_bias', 'right_bias', 'bilateral_bias_diff', 'choice_entropy']
corr_matrix = successful_df[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
```

**Purpose**: Examine relationships between key cognitive and performance variables  
**Implementation**: Pearson correlation matrix with color-coding

##### Subject Clustering (Panel 6)
```python
scatter = ax.scatter(successful_df['mean_rt'], successful_df['accuracy'], 
                    c=successful_df['bilateral_bias_diff'], cmap='plasma', alpha=0.7)
```

**Purpose**: Identify subject clusters based on performance and processing asymmetry  
**Implementation**: 3D scatter plot with color-coded third dimension

---

## Part 5: Detailed Summary Report

### Function: `print_detailed_summary(data: Dict)`

**Purpose**: Generate comprehensive text-based summary of all analysis results

**Implementation**: Formatted console output with organized sections for different metric categories

### Summary Components

#### Overall Success Metrics
```python
print(f"   Total subjects analyzed: {stats['n_subjects']}")
print(f"   Successful analyses: {stats['n_successful']}")
print(f"   Success rate: {stats['success_rate']:.1%}")
```

**Purpose**: Report overall analysis pipeline effectiveness  
**Implementation**: Basic counts and percentages

#### Performance Summary
```python
print(f"   Mean accuracy: {stats['accuracy_mean']:.1%} (SD: {stats['accuracy_std']:.1%})")
print(f"   Accuracy range: {stats['accuracy_range'][0]:.1%} - {stats['accuracy_range'][1]:.1%}")
print(f"   Subjects below chance: {stats['accuracy_below_chance']}/{stats['n_successful']}")
```

**Purpose**: Summarize task performance across all subjects  
**Implementation**: Descriptive statistics with clinical significance indicators

#### Individual Subject Table
```python
print("Subject | Accuracy | Mean RT | Left Bias | Right Bias | Asymmetry")
print("-" * 70)

for _, row in successful_df.iterrows():
    print(f"{row['subject_id']:7d} | {row['accuracy']:8.1%} | {row['mean_rt']:7.3f} | "
          f"{row['left_bias']:9.3f} | {row['right_bias']:10.3f} | {row['bilateral_bias_diff']:9.3f}")
```

**Purpose**: Provide subject-by-subject detailed results  
**Implementation**: Formatted table with aligned columns

---

## Part 6: Main Analysis Pipeline

### Function: `main_analysis(results_file: str, original_file: str)`

**Purpose**: Coordinate complete analysis pipeline from data loading through visualization generation

**Implementation**: Sequential execution of all analysis components with progress reporting

### Pipeline Components

#### File Input Parameters
| Parameter | Default Value | Purpose |
|-----------|---------------|---------|
| `results_file` | 'dual_lba_results_20250615_122314.csv' | Path to LBA model results |
| `original_file` | 'GRT_LBA.csv' | Path to original experimental data |

#### Execution Sequence
```python
# 1. Data processing
data = load_and_process_results(results_file, original_file)

# 2. Summary statistics
print_detailed_summary(data)

# 3. Overview visualizations
create_overview_visualizations(data)

# 4. Individual differences analysis
create_individual_differences_analysis(data)
```

**Purpose**: Ensure systematic execution of all analysis components  
**Implementation**: Sequential function calls with shared data structure

### Output Files Generated

| File Name | Content | Purpose |
|-----------|---------|---------|
| `lba_overview_analysis.png` | 16-panel overview figure | Comprehensive results summary |
| `individual_differences_analysis.png` | 9-panel individual differences figure | Detailed pattern analysis |

---

## Key Scientific Applications

### Model Validation
- **Convergence Assessment**: R-hat and ESS metrics validate MCMC reliability
- **Fit Quality**: Success rates indicate model appropriateness for data
- **Parameter Interpretation**: Bias metrics reveal cognitive processing patterns

### Individual Differences Research
- **Cognitive Phenotyping**: Classification of subjects by processing patterns
- **Asymmetry Detection**: Identification of hemispheric processing differences
- **Performance Clustering**: Grouping subjects by similar cognitive profiles

### Theoretical Insights
- **Dual-Channel Validation**: Tests independence assumptions in LBA models
- **Bias Quantification**: Measures systematic deviations from optimal performance
- **Speed-Accuracy Relationships**: Examines fundamental cognitive trade-offs

This comprehensive analysis system transforms raw LBA model outputs into interpretable scientific insights about individual differences in cognitive processing.
