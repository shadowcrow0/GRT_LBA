# -*- coding: utf-8 -*-
"""
parallel_and_dual_lba_test.py - ParallelAND Dual LBA Test
Test dual LBA with ParallelAND rule using GRT_LBA.csv data

ParallelAND Rule: Take minimum of left and right LBA winner times
Test consistency between predicted response/RT and actual data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path

from single_side_lba import SingleSideLBA
from dual_lba_recorder import DualLBARecorder

class ParallelANDDualLBA:
    """Dual LBA with ParallelAND integration rule"""
    
    def __init__(self):
        """Initialize ParallelAND Dual LBA system"""
        self.left_lba = SingleSideLBA('left')
        self.right_lba = SingleSideLBA('right')
        self.recorder = DualLBARecorder()
        self.recorder.set_lba_processors(self.left_lba, self.right_lba)
        
        print("✅ ParallelAND Dual LBA initialized")
        print("   Rule: min(left_winner_time, right_winner_time)")
    
    def load_grt_data(self, file_path: str = "GRT_LBA.csv") -> pd.DataFrame:
        """Load and preprocess GRT_LBA.csv data"""
        
        data = pd.read_csv(file_path)
        print(f"📊 Loaded {len(data)} trials from {file_path}")
        
        # Convert channel data to stimulus types
        data['left_stimulus'] = data['Chanel1'].map({0: 'nonvertical', 1: 'vertical'})
        data['right_stimulus'] = data['Chanel2'].map({0: 'nonvertical', 1: 'vertical'})
        
        # Response mapping verification
        print("   Data preview:")
        print(f"   Stimulus conditions: {sorted(data['stim_condition'].unique())}")
        print(f"   Responses: {sorted(data['Response'].astype(int).unique())}")
        print(f"   Participants: {sorted(data['participant'].unique())}")
        
        return data
    
    def compute_parallel_and_rt(self, v_left: float, v_right: float, threshold: float, 
                               ndt: float, noise: float) -> float:
        """
        Compute RT using ParallelAND rule
        
        ParallelAND Logic:
        - Drift rate: min(v_left, v_right) - slowest accumulator determines speed
        - RT: Will be maximum (slower) because drift rate is minimum
        
        Args:
            v_left: Left side drift rate
            v_right: Right side drift rate
            threshold: Decision threshold
            ndt: Non-decision time
            noise: Noise parameter
            
        Returns:
            Predicted RT
        """
        
        # ParallelAND: Use minimum drift rate (slowest accumulator)
        effective_drift = min(v_left, v_right)
        effective_drift = max(effective_drift, 0.05)  # Prevent division by zero
        
        # Compute decision time using effective drift rate
        # Higher threshold/drift ratio = longer time
        mean_decision_time = threshold / effective_drift
        
        # Add some variability based on noise
        decision_time = mean_decision_time * (1 + np.random.normal(0, noise * 0.1))
        decision_time = max(decision_time, 0.1)  # Minimum decision time
        
        return decision_time + ndt
    
    def predict_response_rt(self, trial_data: Dict, params: Dict) -> Tuple[int, float]:
        """
        Predict response and RT using ParallelAND rule
        
        Args:
            trial_data: Single trial data
            params: LBA parameters
            
        Returns:
            Tuple of (predicted_response, predicted_rt)
        """
        
        left_stimulus = trial_data['left_stimulus']
        right_stimulus = trial_data['right_stimulus']
        
        # Get drift rates for left side
        if left_stimulus == 'vertical':
            v_left_vertical = params['left_v_vertical']
            v_left_nonvertical = params['left_v_nonvertical_error']
        else:
            v_left_vertical = params['left_v_vertical_error'] 
            v_left_nonvertical = params['left_v_nonvertical']
        
        # Get drift rates for right side
        if right_stimulus == 'vertical':
            v_right_vertical = params['right_v_vertical']
            v_right_nonvertical = params['right_v_nonvertical_error']
        else:
            v_right_vertical = params['right_v_vertical_error']
            v_right_nonvertical = params['right_v_nonvertical']
        
        # Calculate RT for all 4 possible responses using ParallelAND
        response_times = []
        
        for response in range(4):
            # Response mapping based on your design:
            # Response 0: Left nonvertical, Right vertical
            # Response 1: Left vertical, Right vertical  
            # Response 2: Left vertical, Right nonvertical
            # Response 3: Left nonvertical, Right nonvertical
            
            if response in [1, 2]:  # Left vertical
                left_drift = v_left_vertical
            else:  # Left nonvertical (responses 0, 3)
                left_drift = v_left_nonvertical
                
            if response in [0, 1]:  # Right vertical  
                right_drift = v_right_vertical
            else:  # Right nonvertical (responses 2, 3)
                right_drift = v_right_nonvertical
        
            # Compute RT using ParallelAND rule (min drift rate -> max RT)
            rt = self.compute_parallel_and_rt(
                left_drift, right_drift, 
                params['left_threshold'], params['left_ndt'], params['left_noise']
            )
            response_times.append(rt)
        
        # ParallelAND: Choose response with minimum time
        predicted_response = np.argmin(response_times)
        predicted_rt = response_times[predicted_response]
        
        return predicted_response, predicted_rt
    
    def test_participant_data(self, data: pd.DataFrame, participant_id: int, 
                            test_params: Dict = None) -> Dict:
        """Test ParallelAND dual LBA on one participant's data"""
        
        participant_data = data[data['participant'] == participant_id].copy()
        
        if len(participant_data) == 0:
            print(f"❌ No data found for participant {participant_id}")
            return {}
        
        print(f"\n🧪 Testing participant {participant_id} ({len(participant_data)} trials)")
        
        # Use default test parameters if none provided
        if test_params is None:
            test_params = {
                'left_v_vertical': 1.5,
                'left_v_nonvertical': 1.3,
                'left_v_vertical_error': 0.8,
                'left_v_nonvertical_error': 0.6,
                'left_threshold': 1.0,
                'left_start_var': 0.3,
                'left_ndt': 0.2,
                'left_noise': 0.3,
                
                'right_v_vertical': 1.4,
                'right_v_nonvertical': 1.2,
                'right_v_vertical_error': 0.7,
                'right_v_nonvertical_error': 0.5,
                'right_threshold': 1.1,
                'right_start_var': 0.35,
                'right_ndt': 0.25,
                'right_noise': 0.35
            }
        
        # Predict responses and RTs
        predictions = []
        actual_responses = []
        actual_rts = []
        
        for _, trial in participant_data.iterrows():
            trial_dict = {
                'left_stimulus': trial['left_stimulus'],
                'right_stimulus': trial['right_stimulus']
            }
            
            pred_response, pred_rt = self.predict_response_rt(trial_dict, test_params)
            
            predictions.append({
                'predicted_response': pred_response,
                'predicted_rt': pred_rt,
                'actual_response': int(trial['Response']),
                'actual_rt': float(trial['RT']),
                'stimulus_condition': trial['stim_condition']
            })
            
            actual_responses.append(int(trial['Response']))
            actual_rts.append(float(trial['RT']))
        
        # Calculate accuracy metrics
        pred_responses = [p['predicted_response'] for p in predictions]
        pred_rts = [p['predicted_rt'] for p in predictions]
        
        response_accuracy = np.mean(np.array(pred_responses) == np.array(actual_responses))
        rt_correlation = np.corrcoef(pred_rts, actual_rts)[0, 1]
        rt_rmse = np.sqrt(np.mean((np.array(pred_rts) - np.array(actual_rts))**2))
        
        results = {
            'participant_id': participant_id,
            'n_trials': len(participant_data),
            'response_accuracy': response_accuracy,
            'rt_correlation': rt_correlation,
            'rt_rmse': rt_rmse,
            'mean_actual_rt': np.mean(actual_rts),
            'mean_predicted_rt': np.mean(pred_rts),
            'predictions': predictions,
            'parameters': test_params
        }
        
        print(f"   Response accuracy: {response_accuracy:.3f}")
        print(f"   RT correlation: {rt_correlation:.3f}")
        print(f"   RT RMSE: {rt_rmse:.3f}")
        print(f"   Mean actual RT: {np.mean(actual_rts):.3f}")
        print(f"   Mean predicted RT: {np.mean(pred_rts):.3f}")
        
        return results
    
    def visualize_results(self, results: Dict, save_path: str = "parallel_and_results.png"):
        """Visualize ParallelAND test results"""
        
        if not results:
            print("❌ No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        predictions = results['predictions']
        actual_responses = [p['actual_response'] for p in predictions]
        pred_responses = [p['predicted_response'] for p in predictions]
        actual_rts = [p['actual_rt'] for p in predictions]
        pred_rts = [p['predicted_rt'] for p in predictions]
        
        # Response prediction accuracy
        axes[0, 0].scatter(actual_responses, pred_responses, alpha=0.6)
        axes[0, 0].plot([0, 3], [0, 3], 'r--', label='Perfect prediction')
        axes[0, 0].set_xlabel('Actual Response')
        axes[0, 0].set_ylabel('Predicted Response')
        axes[0, 0].set_title(f'Response Prediction (Acc: {results["response_accuracy"]:.3f})')
        axes[0, 0].legend()
        
        # RT prediction
        axes[0, 1].scatter(actual_rts, pred_rts, alpha=0.6)
        axes[0, 1].plot([min(actual_rts), max(actual_rts)], 
                       [min(actual_rts), max(actual_rts)], 'r--', label='Perfect prediction')
        axes[0, 1].set_xlabel('Actual RT')
        axes[0, 1].set_ylabel('Predicted RT')
        axes[0, 1].set_title(f'RT Prediction (r: {results["rt_correlation"]:.3f})')
        axes[0, 1].legend()
        
        # Response distribution
        unique_responses = sorted(set(actual_responses))
        actual_counts = [actual_responses.count(r) for r in unique_responses]
        pred_counts = [pred_responses.count(r) for r in unique_responses]
        
        x = np.arange(len(unique_responses))
        width = 0.35
        axes[1, 0].bar(x - width/2, actual_counts, width, label='Actual', alpha=0.7)
        axes[1, 0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Response')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Response Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(unique_responses)
        axes[1, 0].legend()
        
        # RT distribution
        axes[1, 1].hist(actual_rts, bins=20, alpha=0.7, label='Actual', density=True)
        axes[1, 1].hist(pred_rts, bins=20, alpha=0.7, label='Predicted', density=True)
        axes[1, 1].set_xlabel('RT')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('RT Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results visualization saved: {save_path}")
        
        return fig

def run_parallel_and_test():
    """Run complete ParallelAND dual LBA test"""
    
    print("🚀 Starting ParallelAND Dual LBA Test")
    
    try:
        # Initialize system
        parallel_and_lba = ParallelANDDualLBA()
        
        # Load data
        data = parallel_and_lba.load_grt_data()
        
        # Test on first participant
        participant_id = data['participant'].iloc[0]
        results = parallel_and_lba.test_participant_data(data, participant_id)
        
        if results:
            # Visualize results
            parallel_and_lba.visualize_results(results)
            
            # Save results
            results_summary = {
                'participant_id': results['participant_id'],
                'n_trials': results['n_trials'],
                'response_accuracy': results['response_accuracy'],
                'rt_correlation': results['rt_correlation'],
                'rt_rmse': results['rt_rmse'],
                'mean_actual_rt': results['mean_actual_rt'],
                'mean_predicted_rt': results['mean_predicted_rt']
            }
            
            print(f"\n📋 Results Summary:")
            for key, value in results_summary.items():
                print(f"   {key}: {value}")
            
            # Save detailed results to CSV
            predictions_df = pd.DataFrame(results['predictions'])
            predictions_df.to_csv(f"parallel_and_predictions_p{participant_id}.csv", index=False)
            print(f"📁 Detailed predictions saved: parallel_and_predictions_p{participant_id}.csv")
            
            # Save summary results
            summary_df = pd.DataFrame([results_summary])
            summary_df.to_csv(f"parallel_and_summary_p{participant_id}.csv", index=False)
            print(f"📁 Summary results saved: parallel_and_summary_p{participant_id}.csv")
            
            return results
        
        else:
            print("❌ Test failed - no results generated")
            return None
            
    except Exception as e:
        print(f"❌ ParallelAND test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_parallel_and_test()