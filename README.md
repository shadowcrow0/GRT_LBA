# GRT_LBA
Implement GRT data on LBA with pymc 

Test result

✅ Real data loading completed:
  Valid trials: 14224
  Number of subjects: 18
  Overall accuracy: 0.630
  Mean RT: 0.718s
  Symmetric trial proportion: 0.498
🎯 Selected subject 32

📊 Analysis Data Summary:
  Subject: 32
  Trials: 800
  Accuracy: 0.785
  Mean RT: 0.656s
  Symmetric trials: 424/800
Building covariance matrix LBA model...
Processing 800 trials

ampling 2 chains for 200 tune and 400 draw iterations (400 + 800 draws total) took 377 seconds.
✅ MCMC sampling completed!

============================================================
📊 Covariance Matrix LBA Model Results
============================================================

🔍 Covariance Matrix Analysis Results:
Base correlation coefficient: -0.002 [-0.485, 0.480]
Symmetry effect: 0.004 [-0.374, 0.389]
Symmetric stimulus correlation: 0.001
Asymmetric stimulus correlation: -0.002
Symmetry effect significance: No

🔬 GRT Independence Assumption Test:
Independence probability (|ρ| < 0.1): 0.149
⚠️  Moderate violation of independence assumption

💡 Theoretical Interpretation:
• Symmetry has minimal effect on correlation

📋 Behavioral Data Comparison:
Symmetric stimuli - Accuracy: 0.717, RT: 0.694s
Asymmetric stimuli - Accuracy: 0.862, RT: 0.613s

📊 Complete analysis results saved as 'covariance_lba_analysis_complete.png'

🎉 Complete covariance matrix LBA analysis finished!
Main finding: Minimal effect

💡 Recommendations:
  • Strong evidence against GRT independence assumption
  • Consider alternative models that account for dependencies
  • No significant symmetry effect on correlation structure
