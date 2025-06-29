# quick_import_test.py - Test if all imports work
print("🧪 Testing imports...")

try:
    from lba_models import create_model_by_name, get_available_models
    print("✓ lba_models import successful")
except Exception as e:
    print(f"❌ lba_models import failed: {e}")

try:
    from LBA_tool import sample_with_convergence_check
    print("✓ LBA_tool.sample_with_convergence_check import successful")
except Exception as e:
    print(f"❌ LBA_tool.sample_with_convergence_check import failed: {e}")

try:
    from LBA_tool_fixes import robust_model_comparison
    print("✓ LBA_tool_fixes.robust_model_comparison import successful")
except Exception as e:
    print(f"❌ LBA_tool_fixes.robust_model_comparison import failed: {e}")

try:
    from LBA_IAM import run_accumulation_analysis
    print("✓ LBA_IAM.run_accumulation_analysis import successful")
except Exception as e:
    print(f"❌ LBA_IAM.run_accumulation_analysis import failed: {e}")

try:
    from LBA_PPM import run_comprehensive_ppc
    print("✓ LBA_PPM.run_comprehensive_ppc import successful")
except Exception as e:
    print(f"❌ LBA_PPM.run_comprehensive_ppc import failed: {e}")

try:
    from LBA_visualize import create_sigma_comparison_plots
    print("✓ LBA_visualize imports successful")
except Exception as e:
    print(f"❌ LBA_visualize imports failed: {e}")

print("\n📋 Import Summary:")
print("If all imports show ✓, you can proceed with the full analysis.")
print("If any imports show ❌, those modules need to be fixed first.")

# Quick data test
try:
    import numpy as np
    data = np.load('model_data.npz', allow_pickle=True)
    print("✓ Data file loads successfully")
    
    rt_mean = data['observed_value'][:, 0].mean()
    if rt_mean < 10:
        print(f"⚠️ RT units may need conversion (mean={rt_mean:.3f})")
    else:
        print(f"✓ RT units look correct (mean={rt_mean:.1f})")
        
except Exception as e:
    print(f"❌ Data loading failed: {e}")

print("\n🚀 If all tests pass, you can now run:")
print("   python LBA_main.py --mode test")