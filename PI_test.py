"""
GRT Combined Analysis and Results Integration
Loads and compares results from all three GRT assumption tests
Provides comprehensive summary and interpretation
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class GRTResultsIntegrator:
    """Integrate and analyze results from all three GRT assumption tests"""
    
    def __init__(self):
        self.pi_results = None
        self.ps_results = None
        self.ds_results = None
        self.load_all_results()
    
    def load_all_results(self):
        """Load results from all three analyses"""
        print("Loading GRT assumption test results...")
        
        # Load PI results
        try:
            with open('pi_results.json', 'r') as f:
                self.pi_results = json.load(f)
            print("✓ PI results loaded")
        except FileNotFoundError:
            print("⚠ PI results not found")
        
        # Load PS results
        try:
            with open('ps_results.json', 'r') as f:
                self.ps_results = json.load(f)
            print("✓ PS results loaded")
        except FileNotFoundError:
            print("⚠ PS results not found")
        
        # Load DS results
        try:
            with open('ds_results.json', 'r') as f:
                self.ds_results = json.load(f)
            print("✓ DS results loaded")
        except FileNotFoundError:
            print("⚠ DS results not found")
    
    def generate_comprehensive_summary(self):
        """Generate comprehensive summary of all GRT assumption tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE GRT ASSUMPTION TESTING SUMMARY")
        print("="*80)
        
        # Overview table
        print("\nGRT Assumption Test Results Overview:")
        print("-" * 60)
        print(f"{'Assumption':<25} {'Support':<15} {'Key Parameter':<20}")
        print("-" * 60)
        
        if self.pi_results:
            pi_support = "✓ SUPPORTED" if self.pi_results['pi_support'] else "✗ VIOLATED"
            pi_key = f"Independence: {self.pi_results['independence_mean']:.3f}"
            print(f"{'Perceptual Independence':<25} {pi_support:<15} {pi_key:<20}")
        
        if self.ps_results:
            ps_support = "✓ SUPPORTED" if self.ps_results['ps_support'] else "✗ VIOLATED"
            ps_key = f"Correlation: {self.ps_results['perceptual_correlation_mean']:.3f}"
            print(f"{'Perceptual Separability':<25} {ps_support:<15} {ps_key:<20}")
        
        if self.ds_results:
            ds_support = "✓ SUPPORTED" if self.ds_results['ds_support'] else "✗ VIOLATED"
            ds_key = f"Bias: {self.ds_results['decision_bias_mean']:.3f}"
            print(f"{'Decisional Separability':<25} {ds_support:<15} {ds_key:<20}")
        
        print("-" * 60)
        
        # Detailed analysis for each assumption
        if self.pi_results:
            self.summarize_pi_results()
        
        if self.ps_results:
            self.summarize_ps_results()
        
        if self.ds_results:
            self.summarize_ds_results()
        
        # Overall GRT interpretation
        self.interpret_overall_grt()
        
        # Model comparison
        self.compare_model_quality()
    
    def summarize_pi_results(self):
        """Summarize Perceptual Independence results"""
        print("\n" + "="*50)
        print("PERCEPTUAL INDEPENDENCE (PI) DETAILED RESULTS")
        print("="*50)
        
        independence = self.pi_results['independence_mean']
        independence_hdi = self.pi_results['independence_hdi']
        
        print(f"Independence Parameter: {independence:.4f} [{independence_hdi[0]:.4f}, {independence_hdi[1]:.4f}]")
        print(f"Support for PI: {'YES' if self.pi_results['pi_support'] else 'NO'}")
        
        print(f"\nDrift Rate Parameters:")
        print(f"  v1_mu: {self.pi_results['v1_mu_mean']:.3f}")
        print(f"  v2_mu: {self.pi_results['v2_mu_mean']:.3f}")
        print(f"  v1_sigma: {self.pi_results['v1_sigma_mean']:.3f}")
        print(f"  v2_sigma: {self.pi_results['v2_sigma_mean']:.3f}")
        
        print(f"\nModel Quality:")
        print(f"  Effective Sample Size: {self.pi_results['ess_independence']:.0f}")
        print(f"  R-hat: {self.pi_results['rhat_independence']:.3f}")
        
        print(f"\nInterpretation:")
        if self.pi_results['pi_support']:
            print("  The independence parameter is close to zero, suggesting that")
            print("  perceptual processing of the two dimensions is independent.")
            print("  This supports the PI assumption of GRT.")
        else:
            print("  The independence parameter is significantly different from zero,")
            print("  suggesting dependence between perceptual dimensions.")
            print("  This violates the PI assumption of GRT.")
    
    def summarize_ps_results(self):
        """Summarize Perceptual Separability results"""
        print("\n" + "="*50)
        print("PERCEPTUAL SEPARABILITY (PS) DETAILED RESULTS")
        print("="*50)
        
        interference_12 = self.ps_results['interference_12_mean']
        interference_21 = self.ps_results['interference_21_mean']
        correlation = self.ps_results['perceptual_correlation_mean']
        
        print(f"Cross-Dimensional Interference:")
        print(f"  Dimension 1 → 2: {interference_12:.4f} {self.ps_results['interference_12_hdi']}")
        print(f"  Dimension 2 → 1: {interference_21:.4f} {self.ps_results['interference_21_hdi']}")
        print(f"Perceptual Correlation: {correlation:.4f} {self.ps_results['perceptual_correlation_hdi']}")
        
        print(f"Support for PS: {'YES' if self.ps_results['ps_support'] else 'NO'}")
        
        print(f"\nSignificant Effects:")
        print(f"  Interference 1→2: {'YES' if self.ps_results['significant_interference_12'] else 'NO'}")
        print(f"  Interference 2→1: {'YES' if self.ps_results['significant_interference_21'] else 'NO'}")
        print(f"  Perceptual Correlation: {'YES' if self.ps_results['significant_correlation'] else 'NO'}")
        
        print(f"\nModel Quality:")
        print(f"  ESS (Interference 1→2): {self.ps_results['ess_interference_12']:.0f}")
        print(f"  ESS (Interference 2→1): {self.ps_results['ess_interference_21']:.0f}")
        print(f"  ESS (Correlation): {self.ps_results['ess_correlation']:.0f}")
        
        print(f"\nInterpretation:")
        if self.ps_results['ps_support']:
            print("  No significant cross-dimensional interference or correlation detected.")
            print("  Perceptual representations appear to be separable by dimension.")
            print("  This supports the PS assumption of GRT.")
        else:
            print("  Significant cross-dimensional effects detected:")
            if self.ps_results['significant_interference_12']:
                print("    - Dimension 1 interferes with dimension 2 processing")
            if self.ps_results['significant_interference_21']:
                print("    - Dimension 2 interferes with dimension 1 processing")
            if self.ps_results['significant_correlation']:
                print("    - Significant correlation between perceptual representations")
            print("  This violates the PS assumption of GRT.")
    
    def summarize_ds_results(self):
        """Summarize Decisional Separability results"""
        print("\n" + "="*50)
        print("DECISIONAL SEPARABILITY (DS) DETAILED RESULTS")
        print("="*50)
        
        decision_bias = self.ds_results['decision_bias_mean']
        boundary_interaction = self.ds_results['boundary_interaction_mean']
        boundary_correlation = self.ds_results['boundary_correlation_mean']
        boundary_diff = self.ds_results['boundary_diff_mean']
        
        print(f"Decision Boundary Parameters:")
        print(f"  Decision Bias: {decision_bias:.4f} {self.ds_results['decision_bias_hdi']}")
        print(f"  Boundary Interaction: {boundary_interaction:.4f} {self.ds_results['boundary_interaction_hdi']}")
        print(f"  Boundary Correlation: {boundary_correlation:.4f} {self.ds_results['boundary_correlation_hdi']}")
        print(f"  Boundary Difference: {boundary_diff:.4f} {self.ds_results['boundary_diff_hdi']}")
        
        print(f"Support for DS: {'YES' if self.ds_results['ds_support'] else 'NO'}")
        
        print(f"\nSignificant Effects:")
        print(f"  Decision Bias: {'YES' if self.ds_results['significant_bias'] else 'NO'}")
        print(f"  Boundary Interaction: {'YES' if self.ds_results['significant_interaction'] else 'NO'}")
        print(f"  Boundary Correlation: {'YES' if self.ds_results['significant_correlation'] else 'NO'}")
        
        print(f"\nThreshold Parameters:")
        print(f"  b1_offset_mu: {self.ds_results['b1_offset_mu_mean']:.3f}")
        print(f"  b2_offset_mu: {self.ds_results['b2_offset_mu_mean']:.3f}")
        print(f"  b_offset_sigma: {self.ds_results['b_offset_sigma_mean']:.3f}")
        
        print(f"\nModel Quality:")
        print(f"  ESS (Decision Bias): {self.ds_results['ess_decision_bias']:.0f}")
        print(f"  ESS (Boundary Interaction): {self.ds_results['ess_boundary_interaction']:.0f}")
        print(f"  ESS (Boundary Correlation): {self.ds_results['ess_boundary_correlation']:.0f}")
        
        print(f"\nInterpretation:")
        if self.ds_results['ds_support']:
            print("  No significant decision boundary dependence detected.")
            print("  Decision boundaries appear to be separable across dimensions.")
            print("  This supports the DS assumption of GRT.")
        else:
            print("  Significant decision boundary dependence detected:")
            if self.ds_results['significant_bias']:
                print("    - Significant decision bias between boundaries")
            if self.ds_results['significant_interaction']:
                print("    - Significant boundary interaction effects")
            if self.ds_results['significant_correlation']:
                print("    - Significant correlation between boundary parameters")
            print("  This violates the DS assumption of GRT.")
    
    def interpret_overall_grt(self):
        """Provide overall interpretation of GRT assumptions"""
        print("\n" + "="*50)
        print("OVERALL GRT INTERPRETATION")
        print("="*50)
        
        supported_assumptions = []
        violated_assumptions = []
        
        if self.pi_results:
            if self.pi_results['pi_support']:
                supported_assumptions.append("Perceptual Independence (PI)")
            else:
                violated_assumptions.append("Perceptual Independence (PI)")
        
        if self.ps_results:
            if self.ps_results['ps_support']:
                supported_assumptions.append("Perceptual Separability (PS)")
            else:
                violated_assumptions.append("Perceptual Separability (PS)")
        
        if self.ds_results:
            if self.ds_results['ds_support']:
                supported_assumptions.append("Decisional Separability (DS)")
            else:
                violated_assumptions.append("Decisional Separability (DS)")
        
        print(f"Supported Assumptions ({len(supported_assumptions)}):")
        for assumption in supported_assumptions:
            print(f"  ✓ {assumption}")
        
        print(f"\nViolated Assumptions ({len(violated_assumptions)}):")
        for assumption in violated_assumptions:
            print(f"  ✗ {assumption}")
        
        # Overall GRT interpretation
        print(f"\nGRT Model Validity:")
        if len(violated_assumptions) == 0:
            print("  ✓ FULL GRT VALIDITY: All assumptions are supported")
            print("    The General Recognition Theory model is appropriate for this data.")
        elif len(violated_assumptions) == 1:
            print("  ⚠ PARTIAL GRT VALIDITY: One assumption violated")
            print(f"    Consider alternative models that relax the {violated_assumptions[0]} assumption.")
        elif len(violated_assumptions) == 2:
            print("  ⚠ LIMITED GRT VALIDITY: Two assumptions violated")
            print("    Standard GRT may not be appropriate. Consider more flexible models.")
        else:
            print("  ✗ GRT NOT VALID: All assumptions violated")
            print("    The data violates fundamental GRT assumptions. Alternative models needed.")
        
        # Recommendations
        print(f"\nRecommendations:")
        if len(violated_assumptions) == 0:
            print("  - Proceed with standard GRT analyses")
            print("  - The data satisfies GRT assumptions")
        else:
            print("  - Consider violated assumptions when interpreting results")
            if "Perceptual Independence (PI)" in violated_assumptions:
                print("  - Use models that allow perceptual dependencies")
            if "Perceptual Separability (PS)" in violated_assumptions:
                print("  - Consider integral dimension models")
            if "Decisional Separability (DS)" in violated_assumptions:
                print("  - Use models with flexible decision boundaries")
    
    def compare_model_quality(self):
        """Compare quality metrics across models"""
        print("\n" + "="*50)
        print("MODEL QUALITY COMPARISON")
        print("="*50)
        
        print(f"{'Model':<15} {'Key Parameter':<20} {'ESS':<8} {'R-hat':<8}")
        print("-" * 55)
        
        if self.pi_results:
            print(f"{'PI Model':<15} {'Independence':<20} {self.pi_results['ess_independence']:<8.0f} {self.pi_results['rhat_independence']:<8.3f}")
        
        if self.ps_results:
            ess_min = min(self.ps_results['ess_interference_12'], 
                         self.ps_results['ess_interference_21'], 
                         self.ps_results['ess_correlation'])
            rhat_max = max(self.ps_results['rhat_interference_12'], 
                          self.ps_results['rhat_interference_21'], 
                          self.ps_results['rhat_correlation'])
            print(f"{'PS Model':<15} {'Min Interference':<20} {ess_min:<8.0f} {rhat_max:<8.3f}")
        
        if self.ds_results:
            ess_min = min(self.ds_results['ess_decision_bias'], 
                         self.ds_results['ess_boundary_interaction'], 
                         self.ds_results['ess_boundary_correlation'])
            rhat_max = max(self.ds_results['rhat_decision_bias'], 
                          self.ds_results['rhat_boundary_interaction'], 
                          self.ds_results['rhat_boundary_correlation'])
            print(f"{'DS Model':<15} {'Min Boundary':<20} {ess_min:<8.0f} {rhat_max:<8.3f}")
        
        print("-" * 55)
        print("Note: ESS > 400 and R̂ < 1.01 indicate good convergence")
    
    def save_combined_results(self):
        """Save combined results summary"""
        print("\nSaving combined results...")
        
        combined_results = {
            'grt_analysis_summary': {
                'pi_support': self.pi_results['pi_support'] if self.pi_results else None,
                'ps_support': self.ps_results['ps_support'] if self.ps_results else None,
                'ds_support': self.ds_results['ds_support'] if self.ds_results else None,
                'overall_grt_validity': 'full' if all([
                    self.pi_results and self.pi_results['pi_support'],
                    self.ps_results and self.ps_results['ps_support'],
                    self.ds_results and self.ds_results['ds_support']
                ]) else 'partial_or_violated'
            },
            'detailed_results': {
                'pi': self.pi_results,
                'ps': self.ps_results,
                'ds': self.ds_results
            }
        }
        
        with open('grt_combined_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print("✓ Combined results saved as grt_combined_results.json")
        return combined_results

def run_combined_analysis():
    """Run the combined analysis"""
    print("STARTING GRT COMBINED ANALYSIS")
    print("="*60)
    
    integrator = GRTResultsIntegrator()
    integrator.generate_comprehensive_summary()
    combined_results = integrator.save_combined_results()
    
    print("\n" + "="*60)
    print("GRT COMBINED ANALYSIS COMPLETED")
    print("="*60)
    
    return integrator, combined_results

if __name__ == "__main__":
    integrator, results = run_combined_analysis()
