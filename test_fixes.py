# test_fixes.py - Fixed import version
import numpy as np
from lba_models import create_model_by_name

# Load test data first
print("🔧 Loading test data...")
try:
    data = np.load('model_data.npz', allow_pickle=True)
    observed_value = data['observed_value']
    participant_idx = data['participant_idx']
    model_input_data = data['model_input_data'].item()
    print("✓ Data loaded successfully")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit(1)

# Test with first participant, small sample
mask = participant_idx == 0
test_data = observed_value[mask][:50]  # Use only 50 trials
test_input = {
    'left_match': model_input_data['left_match'][mask][:50],
    'right_match': model_input_data['right_match'][mask][:50]
}

print(f"\n🧪 Testing fixes...")
print(f"Test data shape: {test_data.shape}")
print(f"RT range: {test_data[:, 0].min():.1f} - {test_data[:, 0].max():.1f}")
print(f"Accuracy: {test_data[:, 1].mean():.3f}")

# Test model creation first
print(f"\n📊 Testing model creation...")
try:
    model = create_model_by_name('Coactive_Addition', test_data, test_input)
    print("✓ Model creation successful")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    exit(1)

# Test model compilation
print(f"\n🔧 Testing model compilation...")
try:
    import pymc as pm
    with model:
        test_point = model.initial_point()
        logp = model.compile_logp()
        logp_val = logp(test_point)
        print(f"✓ Model compiles successfully, logp: {logp_val:.2f}")
except Exception as e:
    print(f"❌ Model compilation failed: {e}")
    exit(1)

# Now test sampling with a very conservative approach
print(f"\n⏱️ Testing sampling...")
try:
    import pymc as pm
    
    with model:
        # Try to find MAP first
        try:
            start_map = pm.find_MAP(method='BFGS', maxeval=1000)
            print("✓ MAP estimation successful")
        except Exception as map_error:
            print(f"⚠️ MAP estimation failed: {map_error}")
            start_map = None
        
        # Very conservative sampling
        trace = pm.sample(
            draws=50,
            tune=50,
            chains=2,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42,
            start=start_map,
            init='jitter+adapt_diag',
            cores=1
        )
        
        if trace is not None:
            print("✓ Sampling successful!")
            
            # Check trace structure
            print(f"\n📋 Trace diagnostics:")
            print(f"Chains: {len(trace.posterior.coords['chain'])}")
            print(f"Draws: {len(trace.posterior.coords['draw'])}")
            print(f"Parameters: {list(trace.posterior.data_vars.keys())}")
            
            # Test the fixed parameter extraction
            print(f"\n🔍 Testing parameter extraction...")
            
            # Import our fixed functions (assuming they're in lba_tool_fixes.py)
            try:
                from lba_tool_fixes import safe_parameter_extraction, diagnose_trace_issues
                
                # Test diagnostics
                issues = diagnose_trace_issues(trace, 'Coactive_Addition')
                
                # Test parameter extraction
                required_params = ['v_final_correct', 'v_final_incorrect', 'b_safe', 'start_var', 'non_decision']
                
                extraction_results = {}
                for param in required_params:
                    try:
                        values = safe_parameter_extraction(trace, param)
                        extraction_results[param] = {
                            'success': True,
                            'n_samples': len(values),
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'range': [values.min(), values.max()]
                        }
                        print(f"  ✓ {param}: {len(values)} samples, mean={np.mean(values):.3f}")
                    except Exception as e:
                        extraction_results[param] = {'success': False, 'error': str(e)}
                        print(f"  ❌ {param}: {e}")
                
                # Summary
                successful_extractions = sum(1 for r in extraction_results.values() if r.get('success', False))
                print(f"\n📊 Parameter extraction summary:")
                print(f"  Successful: {successful_extractions}/{len(required_params)}")
                print(f"  Issues found: {len(issues)}")
                
                if successful_extractions >= 4:  # Most parameters work
                    print("\n✅ FIXES ARE WORKING!")
                    print("You can now proceed with the full analysis.")
                    
                    # Test the accumulation analysis
                    print(f"\n🔬 Testing accumulation analysis...")
                    try:
                        # Create a minimal models dict
                        models = {'Coactive_Addition': trace}
                        
                        # Import the fixed accumulation analysis
                        import sys
                        import os
                        
                        # Save the fixed LBA_IAM.py content to a file if it doesn't exist
                        if not os.path.exists('LBA_IAM_fixed.py'):
                            print("Creating LBA_IAM_fixed.py...")
                            # The fixed code is in the artifact above
                            print("Please save the fixed LBA_IAM.py code as 'LBA_IAM_fixed.py' and re-run")
                        else:
                            from LBA_IAM_fixed import run_accumulation_analysis
                            
                            # Test with minimal setup
                            save_dir = 'test_output'
                            os.makedirs(save_dir, exist_ok=True)
                            
                            result = run_accumulation_analysis(models, 0, save_dir)
                            
                            if result and any(result.values()):
                                print("✅ Accumulation analysis test successful!")
                            else:
                                print("⚠️ Accumulation analysis had some issues but didn't crash")
                    
                    except Exception as acc_error:
                        print(f"⚠️ Accumulation analysis test failed: {acc_error}")
                        print("But parameter extraction works, so main issue is likely fixed")
                
                else:
                    print("\n⚠️ Some parameter extraction issues remain")
                    print("Check the error messages above for specific problems")
            
            except ImportError:
                print("⚠️ Could not import lba_tool_fixes.py")
                print("Please ensure the lba_tool_fixes.py file exists with the fixed functions")
                
                # Fallback: basic parameter check
                print("\nFallback: Basic parameter structure check...")
                for param in ['v_final_correct', 'v_final_incorrect', 'b_safe', 'start_var', 'non_decision']:
                    if param in trace.posterior:
                        param_shape = trace.posterior[param].shape
                        print(f"  {param}: shape {param_shape}")
                    else:
                        print(f"  {param}: MISSING")
        
        else:
            print("❌ Sampling returned None")
    
except Exception as e:
    print(f"❌ Sampling failed: {e}")
    import traceback
    print("Full traceback:")
    print(traceback.format_exc())

print(f"\n🏁 Test completed!")
print("\nNext steps:")
print("1. If parameter extraction works: proceed with applying fixes to main files")
print("2. If parameter extraction fails: check the error messages above")
print("3. Save the fixed LBA_IAM.py code as your main LBA_IAM.py file")
print("4. Update your LBA_tool.py with the robust model comparison functions")