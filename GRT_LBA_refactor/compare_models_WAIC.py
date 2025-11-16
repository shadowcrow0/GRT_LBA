"""
Model Comparison Script: Compare PS vs Non-PS using WAIC
=========================================================
讀取兩個已儲存的 trace 檔案並進行 WAIC 比較

Author: YYC & Claude
Date: 2025-11-16
"""

import arviz as az
import numpy as np
import os

print("=" * 70)
print("Model Comparison: PS (8 params) vs Non-PS (16 params)")
print("Using WAIC (Widely Applicable Information Criterion)")
print("=" * 70)

# Check if trace files exist
model1_file = "model1_PS_trace.nc"
model2_file = "model2_NonPS_trace.nc"

if not os.path.exists(model1_file):
    print(f"\n✗ Error: {model1_file} not found!")
    print("Please run Model1_PS_8param.py first.")
    exit(1)

if not os.path.exists(model2_file):
    print(f"\n✗ Error: {model2_file} not found!")
    print("Please run Model2_NonPS_16param.py first.")
    exit(1)

# Load traces
print("\n1. Loading posterior traces...")
print(f"   Loading {model1_file}...")
trace1 = az.from_netcdf(model1_file)
print(f"   ✓ Model 1 (PS, 8 params) loaded")
print(f"   File size: {os.path.getsize(model1_file) / 1024:.1f} KB")

print(f"\n   Loading {model2_file}...")
trace2 = az.from_netcdf(model2_file)
print(f"   ✓ Model 2 (Non-PS, 16 params) loaded")
print(f"   File size: {os.path.getsize(model2_file) / 1024:.1f} KB")

# Check if log_likelihood is available
print("\n2. Checking log_likelihood availability...")

if 'log_likelihood' not in trace1.sample_stats:
    print("   ✗ Warning: log_likelihood not found in Model 1 trace")
    print("   WAIC calculation requires log_likelihood to be stored during sampling")
    print("   Please re-run Model1_PS_8param.py with idata_kwargs={'log_likelihood': True}")

if 'log_likelihood' not in trace2.sample_stats:
    print("   ✗ Warning: log_likelihood not found in Model 2 trace")
    print("   WAIC calculation requires log_likelihood to be stored during sampling")
    print("   Please re-run Model2_NonPS_16param.py with idata_kwargs={'log_likelihood': True}")

# Perform WAIC comparison
print("\n3. Calculating WAIC...")

try:
    comparison = az.compare({'Model1_PS_8param': trace1, 'Model2_NonPS_16param': trace2}, ic='waic')

    print("\n" + "=" * 70)
    print("WAIC Comparison Results:")
    print("=" * 70)
    print(comparison)

    # Save results
    output_file = "model_comparison_WAIC.log"
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("WAIC Model Comparison: PS (8 params) vs Non-PS (16 params)\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(comparison))
        f.write("\n\n")
        f.write("Interpretation:\n")
        f.write("  - Lower WAIC = Better model\n")
        f.write("  - dWAIC = WAIC difference from best model\n")
        f.write("  - weight = Akaike weight (probability model is best)\n")
        f.write("  - se = Standard error of WAIC estimate\n")
        f.write("  - dse = Standard error of dWAIC\n")
        f.write("\n")
        f.write("Expected:\n")
        f.write("  - If data follows PS → Model1_PS should have lower WAIC\n")
        f.write("  - If data follows Non-PS → Model2_NonPS should have lower WAIC\n")

    print(f"\n✓ Results saved to: {output_file}")

    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("  - Lower WAIC = Better model")
    print("  - dWAIC = WAIC difference from best model")
    print("  - weight = Akaike weight (probability model is best)")
    print("  - se = Standard error of WAIC estimate")
    print("  - dse = Standard error of dWAIC")
    print("\nExpected:")
    print("  - If data follows PS → Model1_PS should have lower WAIC")
    print("  - If data follows Non-PS → Model2_NonPS should have lower WAIC")

    # Winner
    winner = comparison.index[0]
    print(f"\n🏆 Winner: {winner}")

except Exception as e:
    print(f"\n✗ Error calculating WAIC: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure both models were run successfully")
    print("  2. Check that log_likelihood was computed during sampling")
    print("  3. Verify trace files are not corrupted")

print("\n" + "=" * 70)
print("✓ Model comparison completed!")
print("=" * 70)
