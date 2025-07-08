# -*- coding: utf-8 -*-
"""
dual_lba_recorder.py - Comprehensive Recording System for Dual LBA
Records all evidence, parameters, and log-likelihoods for left/right processing
Enables detailed relationship analysis between bilateral processing
"""

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from typing import Dict, List, NamedTuple, Optional, Tuple
from pathlib import Path
import json

class TrialRecord(NamedTuple):
    """Complete record for a single trial"""
    trial_id: int
    participant_id: int
    stimulus_condition: int
    left_stimulus: str
    right_stimulus: str
    response: int
    rt: float
    
    # Left side processing - parameters
    left_v_vertical: float
    left_v_nonvertical: float
    left_v_vertical_error: float
    left_v_nonvertical_error: float
    
    # Left side processing - actual drift rates used for this trial
    left_actual_drift_rate: float      # The actual drift rate used for this trial
    left_drift_type: str              # Which drift type was used ('v_vertical', 'v_nonvertical', etc.)
    left_stimulus_type: str           # 'vertical' or 'nonvertical'
    left_response_type: str           # 'vertical' or 'nonvertical'
    left_is_correct: bool             # Whether this trial was correct for left side
    
    # Left side processing - evidence and probabilities
    left_evidence_vertical: float
    left_evidence_nonvertical: float
    left_loglikelihood: float
    left_choice_prob_vertical: float
    left_choice_prob_nonvertical: float
    
    # Right side processing - parameters
    right_v_vertical: float
    right_v_nonvertical: float
    right_v_vertical_error: float
    right_v_nonvertical_error: float
    
    # Right side processing - actual drift rates used for this trial
    right_actual_drift_rate: float     # The actual drift rate used for this trial
    right_drift_type: str             # Which drift type was used
    right_stimulus_type: str          # 'vertical' or 'nonvertical'
    right_response_type: str          # 'vertical' or 'nonvertical'
    right_is_correct: bool            # Whether this trial was correct for right side
    
    # Right side processing - evidence and probabilities
    right_evidence_vertical: float
    right_evidence_nonvertical: float
    right_loglikelihood: float
    right_choice_prob_vertical: float
    right_choice_prob_nonvertical: float
    
    # Combined processing
    combined_loglikelihood: float
    predicted_response: int
    prediction_confidence: float

class DualLBARecorder:
    """Comprehensive recording system for dual LBA analysis"""
    
    def __init__(self):
        """Initialize the dual LBA recorder"""
        self.trial_records: List[TrialRecord] = []
        self.left_lba = None
        self.right_lba = None
        self.current_params = {}
        
        print("✅ Dual LBA Recorder initialized")
        print("   Records: evidence, parameters, log-likelihoods for both sides")
    
    def set_lba_processors(self, left_lba, right_lba):
        """Set the left and right LBA processors"""
        self.left_lba = left_lba
        self.right_lba = right_lba
        print(f"   Left processor: {left_lba.side_name}")
        print(f"   Right processor: {right_lba.side_name}")
    
    def record_trial_batch(self, participant_id: int, trial_data: Dict, params: Dict) -> pd.DataFrame:
        """
        Record a complete batch of trials with comprehensive tracking
        
        Args:
            participant_id: Participant identifier
            trial_data: Dictionary with trial information
            params: Complete parameter set for both sides
            
        Returns:
            DataFrame with complete trial records
        """
        
        self.current_params = params.copy()
        
        # Extract trial data
        stimuli = trial_data['stimuli']
        decisions = trial_data['decisions'] 
        rt = trial_data['rt']
        stimulus_conditions = trial_data['stimulus_conditions']
        left_stimuli = trial_data['left_stimuli']
        right_stimuli = trial_data['right_stimuli']
        
        print(f"\n📊 Recording {len(stimuli)} trials for participant {participant_id}")
        
        # Calculate comprehensive evidence and likelihoods
        left_results = self._calculate_side_results('left', stimuli, decisions, rt, params)
        right_results = self._calculate_side_results('right', stimuli, decisions, rt, params)
        
        # Record each trial
        batch_records = []
        for i in range(len(stimuli)):
            
            # Calculate combined log-likelihood
            combined_ll = left_results['trial_loglikelihoods'][i] + right_results['trial_loglikelihoods'][i]
            
            # Determine actual drift rates used for each side
            left_drift_info = self._determine_actual_drift_rate('left', left_stimuli[i], decisions[i], params)
            right_drift_info = self._determine_actual_drift_rate('right', right_stimuli[i], decisions[i], params)
            
            # Predict response based on combined evidence
            left_vertical_strength = left_results['trial_evidence_vertical'][i]
            left_nonvertical_strength = left_results['trial_evidence_nonvertical'][i]
            right_vertical_strength = right_results['trial_evidence_vertical'][i]
            right_nonvertical_strength = right_results['trial_evidence_nonvertical'][i]
            
            # Four possible responses based on left-right combinations
            response_strengths = {
                0: left_nonvertical_strength + right_vertical_strength,   # Left nonvertical, right vertical
                1: left_vertical_strength + right_vertical_strength,      # Left vertical, right vertical
                2: left_vertical_strength + right_nonvertical_strength,   # Left vertical, right nonvertical
                3: left_nonvertical_strength + right_nonvertical_strength # Left nonvertical, right nonvertical
            }
            
            predicted_response = max(response_strengths, key=response_strengths.get)
            prediction_confidence = response_strengths[predicted_response] / sum(response_strengths.values())
            
            trial_record = TrialRecord(
                trial_id=i,
                participant_id=participant_id,
                stimulus_condition=stimulus_conditions[i],
                left_stimulus=left_stimuli[i],
                right_stimulus=right_stimuli[i],
                response=decisions[i],
                rt=rt[i],
                
                # Left side parameters
                left_v_vertical=params['left_v_vertical'],
                left_v_nonvertical=params['left_v_nonvertical'],
                left_v_vertical_error=params['left_v_vertical_error'],
                left_v_nonvertical_error=params['left_v_nonvertical_error'],
                
                # Left side actual drift rate info for this trial
                left_actual_drift_rate=left_drift_info['actual_drift_rate'],
                left_drift_type=left_drift_info['drift_type'],
                left_stimulus_type=left_drift_info['stimulus_type'],
                left_response_type=left_drift_info['response_type'],
                left_is_correct=left_drift_info['is_correct'],
                
                # Left side evidence and probabilities
                left_evidence_vertical=left_results['trial_evidence_vertical'][i],
                left_evidence_nonvertical=left_results['trial_evidence_nonvertical'][i],
                left_loglikelihood=left_results['trial_loglikelihoods'][i],
                left_choice_prob_vertical=left_results['trial_choice_probs'][i][0],
                left_choice_prob_nonvertical=left_results['trial_choice_probs'][i][1],
                
                # Right side parameters
                right_v_vertical=params['right_v_vertical'],
                right_v_nonvertical=params['right_v_nonvertical'], 
                right_v_vertical_error=params['right_v_vertical_error'],
                right_v_nonvertical_error=params['right_v_nonvertical_error'],
                
                # Right side actual drift rate info for this trial
                right_actual_drift_rate=right_drift_info['actual_drift_rate'],
                right_drift_type=right_drift_info['drift_type'],
                right_stimulus_type=right_drift_info['stimulus_type'],
                right_response_type=right_drift_info['response_type'],
                right_is_correct=right_drift_info['is_correct'],
                
                # Right side evidence and probabilities
                right_evidence_vertical=right_results['trial_evidence_vertical'][i],
                right_evidence_nonvertical=right_results['trial_evidence_nonvertical'][i],
                right_loglikelihood=right_results['trial_loglikelihoods'][i],
                right_choice_prob_vertical=right_results['trial_choice_probs'][i][0],
                right_choice_prob_nonvertical=right_results['trial_choice_probs'][i][1],
                
                # Combined results
                combined_loglikelihood=combined_ll,
                predicted_response=predicted_response,
                prediction_confidence=prediction_confidence
            )
            
            batch_records.append(trial_record)
            self.trial_records.append(trial_record)
        
        # Convert to DataFrame for analysis
        records_df = pd.DataFrame(batch_records)
        
        print(f"   ✅ Recorded {len(batch_records)} trials")
        print(f"   Combined log-likelihood: {records_df['combined_loglikelihood'].sum():.3f}")
        print(f"   Prediction accuracy: {np.mean(records_df['predicted_response'] == records_df['response']):.3f}")
        
        return records_df
    
    def _calculate_side_results(self, side: str, stimuli, decisions, rt, params) -> Dict:
        """Calculate comprehensive results for one side (left or right)"""
        
        lba_processor = self.left_lba if side == 'left' else self.right_lba
        
        # Extract side-specific parameters
        side_params = {
            f'{side}_v_vertical': params[f'{side}_v_vertical'],
            f'{side}_v_nonvertical': params[f'{side}_v_nonvertical'],
            f'{side}_v_vertical_error': params[f'{side}_v_vertical_error'],
            f'{side}_v_nonvertical_error': params[f'{side}_v_nonvertical_error'],
            f'{side}_threshold': params[f'{side}_threshold'],
            f'{side}_start_var': params[f'{side}_start_var'],
            f'{side}_ndt': params[f'{side}_ndt'],
            f'{side}_noise': params[f'{side}_noise']
        }
        
        # Calculate trial-by-trial log-likelihoods
        trial_loglikelihoods = self._calculate_trial_loglikelihoods(
            lba_processor, stimuli, decisions, rt, side_params
        )
        
        # Calculate trial-by-trial evidence
        trial_evidence_vertical, trial_evidence_nonvertical = self._calculate_trial_evidence(
            lba_processor, stimuli, decisions, rt, side_params
        )
        
        # Calculate trial-by-trial choice probabilities
        trial_choice_probs = lba_processor.compute_choice_probabilities(stimuli, side_params)
        
        return {
            'trial_loglikelihoods': trial_loglikelihoods,
            'trial_evidence_vertical': trial_evidence_vertical,
            'trial_evidence_nonvertical': trial_evidence_nonvertical,
            'trial_choice_probs': trial_choice_probs
        }
    
    def _calculate_trial_loglikelihoods(self, lba_processor, stimuli, decisions, rt, params) -> List[float]:
        """Calculate log-likelihood for each individual trial"""
        
        trial_lls = []
        
        for i in range(len(stimuli)):
            # Calculate likelihood for this single trial
            single_stimulus = np.array([stimuli[i]])
            single_decision = np.array([decisions[i]])
            single_rt = np.array([rt[i]])
            
            # Convert to PyTensor for computation
            single_stimulus_pt = pt.as_tensor_variable(single_stimulus)
            single_decision_pt = pt.as_tensor_variable(single_decision)
            single_rt_pt = pt.as_tensor_variable(single_rt)
            
            # Extract parameters as PyTensor variables
            pt_params = {}
            for key, value in params.items():
                pt_params[key] = pt.as_tensor_variable(float(value))
            
            # Calculate likelihood for this trial
            trial_ll = lba_processor.compute_likelihood(
                single_decision_pt, single_stimulus_pt, single_rt_pt, pt_params
            )
            
            # Evaluate the result
            trial_ll_value = float(trial_ll.eval())
            trial_lls.append(trial_ll_value)
        
        return trial_lls
    
    def _calculate_trial_evidence(self, lba_processor, stimuli, decisions, rt, params) -> Tuple[List[float], List[float]]:
        """Calculate evidence for each individual trial"""
        
        trial_evidence_vertical = []
        trial_evidence_nonvertical = []
        
        # Extract drift rates
        v_vertical = params[f'{lba_processor.side_name}_v_vertical']
        v_nonvertical = params[f'{lba_processor.side_name}_v_nonvertical']
        v_vertical_error = params[f'{lba_processor.side_name}_v_vertical_error']
        v_nonvertical_error = params[f'{lba_processor.side_name}_v_nonvertical_error']
        
        for i in range(len(stimuli)):
            stimulus = stimuli[i]
            decision = decisions[i]
            
            # Calculate evidence based on stimulus-response combination
            if stimulus == 0:  # Vertical stimulus
                if decision == 0:  # Correct vertical response
                    evidence_v = v_vertical
                    evidence_nv = v_nonvertical_error
                else:  # Incorrect nonvertical response  
                    evidence_v = v_vertical
                    evidence_nv = v_nonvertical_error
            else:  # Nonvertical stimulus
                if decision == 1:  # Correct nonvertical response
                    evidence_v = v_vertical_error
                    evidence_nv = v_nonvertical
                else:  # Incorrect vertical response
                    evidence_v = v_vertical_error
                    evidence_nv = v_nonvertical
            
            trial_evidence_vertical.append(evidence_v)
            trial_evidence_nonvertical.append(evidence_nv)
        
        return trial_evidence_vertical, trial_evidence_nonvertical
    
    def _determine_actual_drift_rate(self, side: str, stimulus: str, response: int, params: Dict) -> Dict:
        """
        Determine the actual drift rate used for a specific trial
        
        Based on user's design:
        Response 0: Left vertical, Right nonvertical
        Response 1: Left vertical, Right vertical  
        Response 2: Left nonvertical, Right vertical
        Response 3: Left nonvertical, Right nonvertical
        """
        
        # Determine what response was made for this side
        if side == 'left':
            # For left side: response 1,2 = vertical; response 0,3 = nonvertical
            side_response = 'vertical' if response in [1, 2] else 'nonvertical'
        else:  # right side
            # For right side: response 0,1 = vertical; response 2,3 = nonvertical
            side_response = 'vertical' if response in [0, 1] else 'nonvertical'
        
        # Determine if this was a correct response for this side
        is_correct = (stimulus == side_response)
        
        # Get the actual drift rate used
        if stimulus == 'vertical':
            if side_response == 'vertical':
                # Correct vertical response to vertical stimulus
                drift_type = f'{side}_v_vertical'
                actual_drift_rate = params[drift_type]
            else:
                # Incorrect nonvertical response to vertical stimulus
                drift_type = f'{side}_v_nonvertical_error'
                actual_drift_rate = params[drift_type]
        else:  # stimulus == 'nonvertical'
            if side_response == 'nonvertical':
                # Correct nonvertical response to nonvertical stimulus
                drift_type = f'{side}_v_nonvertical'
                actual_drift_rate = params[drift_type]
            else:
                # Incorrect vertical response to nonvertical stimulus
                drift_type = f'{side}_v_vertical_error'
                actual_drift_rate = params[drift_type]
        
        return {
            'actual_drift_rate': float(actual_drift_rate),
            'drift_type': drift_type.split('_', 1)[1],  # Remove side prefix
            'stimulus_type': stimulus,
            'response_type': side_response,
            'is_correct': is_correct
        }
    
    def analyze_left_right_relationships(self, records_df: pd.DataFrame) -> Dict:
        """Analyze relationships between left and right processing"""
        
        print("\n🔍 Analyzing left-right processing relationships...")
        
        # Correlation analyses
        correlations = {
            'evidence_correlations': {
                'vertical': np.corrcoef(records_df['left_evidence_vertical'], 
                                      records_df['right_evidence_vertical'])[0,1],
                'nonvertical': np.corrcoef(records_df['left_evidence_nonvertical'], 
                                         records_df['right_evidence_nonvertical'])[0,1],
                'cross_vertical_nonvertical': np.corrcoef(records_df['left_evidence_vertical'], 
                                                        records_df['right_evidence_nonvertical'])[0,1],
                'cross_nonvertical_vertical': np.corrcoef(records_df['left_evidence_nonvertical'], 
                                                        records_df['right_evidence_vertical'])[0,1]
            },
            'loglikelihood_correlations': {
                'left_right': np.corrcoef(records_df['left_loglikelihood'], 
                                        records_df['right_loglikelihood'])[0,1]
            },
            'choice_probability_correlations': {
                'vertical': np.corrcoef(records_df['left_choice_prob_vertical'], 
                                      records_df['right_choice_prob_vertical'])[0,1],
                'nonvertical': np.corrcoef(records_df['left_choice_prob_nonvertical'], 
                                         records_df['right_choice_prob_nonvertical'])[0,1]
            }
        }
        
        # Performance metrics
        performance = {
            'total_loglikelihood': records_df['combined_loglikelihood'].sum(),
            'mean_loglikelihood': records_df['combined_loglikelihood'].mean(),
            'prediction_accuracy': np.mean(records_df['predicted_response'] == records_df['response']),
            'mean_confidence': records_df['prediction_confidence'].mean()
        }
        
        # Side-specific metrics
        side_metrics = {
            'left': {
                'mean_evidence_vertical': records_df['left_evidence_vertical'].mean(),
                'mean_evidence_nonvertical': records_df['left_evidence_nonvertical'].mean(),
                'total_loglikelihood': records_df['left_loglikelihood'].sum(),
                'evidence_ratio': records_df['left_evidence_vertical'].mean() / records_df['left_evidence_nonvertical'].mean()
            },
            'right': {
                'mean_evidence_vertical': records_df['right_evidence_vertical'].mean(),
                'mean_evidence_nonvertical': records_df['right_evidence_nonvertical'].mean(),
                'total_loglikelihood': records_df['right_loglikelihood'].sum(),
                'evidence_ratio': records_df['right_evidence_vertical'].mean() / records_df['right_evidence_nonvertical'].mean()
            }
        }
        
        print(f"   Evidence correlations (V-V): {correlations['evidence_correlations']['vertical']:.4f}")
        print(f"   Evidence correlations (NV-NV): {correlations['evidence_correlations']['nonvertical']:.4f}")
        print(f"   Log-likelihood correlation: {correlations['loglikelihood_correlations']['left_right']:.4f}")
        print(f"   Prediction accuracy: {performance['prediction_accuracy']:.4f}")
        
        return {
            'correlations': correlations,
            'performance': performance,
            'side_metrics': side_metrics
        }
    
    def save_comprehensive_results(self, output_dir: str = "dual_lba_results"):
        """Save all recorded results and analyses"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert all records to DataFrame
        all_records_df = pd.DataFrame(self.trial_records)
        
        # Save detailed trial records
        all_records_df.to_csv(output_path / "comprehensive_trial_records.csv", index=False)
        print(f"   ✅ Detailed trial records saved: {output_path / 'comprehensive_trial_records.csv'}")
        
        # Analyze relationships
        relationships = self.analyze_left_right_relationships(all_records_df)
        
        # Save relationship analysis
        with open(output_path / "left_right_relationship_analysis.json", 'w') as f:
            json.dump(relationships, f, indent=2)
        print(f"   ✅ Relationship analysis saved: {output_path / 'left_right_relationship_analysis.json'}")
        
        # Save parameter summary
        param_summary = {
            'current_parameters': self.current_params,
            'total_trials_recorded': len(self.trial_records),
            'participants': list(set([r.participant_id for r in self.trial_records]))
        }
        
        with open(output_path / "parameter_summary.json", 'w') as f:
            json.dump(param_summary, f, indent=2)
        print(f"   ✅ Parameter summary saved: {output_path / 'parameter_summary.json'}")
        
        print(f"\n📁 All results saved to: {output_path}")
        return output_path
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all recorded data"""
        
        if not self.trial_records:
            return {"message": "No trial records available"}
        
        records_df = pd.DataFrame(self.trial_records)
        
        return {
            'total_trials': len(records_df),
            'participants': records_df['participant_id'].nunique(),
            'stimulus_conditions': records_df['stimulus_condition'].nunique(),
            'overall_loglikelihood': records_df['combined_loglikelihood'].sum(),
            'prediction_accuracy': np.mean(records_df['predicted_response'] == records_df['response']),
            'left_right_evidence_correlation': np.corrcoef(
                records_df['left_evidence_vertical'] + records_df['left_evidence_nonvertical'],
                records_df['right_evidence_vertical'] + records_df['right_evidence_nonvertical']
            )[0,1]
        }

# Convenience functions
def create_dual_lba_recorder():
    """Create a new dual LBA recorder"""
    return DualLBARecorder()

def test_dual_lba_recorder():
    """Test the dual LBA recorder functionality"""
    
    print("🧪 Testing Dual LBA Recorder...")
    
    try:
        from single_side_lba import SingleSideLBA
        
        # Create test data
        n_trials = 50
        np.random.seed(42)
        
        trial_data = {
            'stimuli': np.random.choice([0, 1], size=n_trials),
            'decisions': np.random.choice([0, 1, 2, 3], size=n_trials),
            'rt': np.random.uniform(0.3, 1.5, size=n_trials),
            'stimulus_conditions': np.random.choice([0, 1, 2, 3], size=n_trials),
            'left_stimuli': ['vertical' if x == 0 else 'nonvertical' for x in np.random.choice([0, 1], size=n_trials)],
            'right_stimuli': ['vertical' if x == 0 else 'nonvertical' for x in np.random.choice([0, 1], size=n_trials)]
        }
        
        # Test parameters
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
        
        # Create recorder and LBA processors
        recorder = DualLBARecorder()
        left_lba = SingleSideLBA('left')
        right_lba = SingleSideLBA('right')
        
        recorder.set_lba_processors(left_lba, right_lba)
        
        # Record trial batch
        records_df = recorder.record_trial_batch(1, trial_data, test_params)
        
        print(f"   Trial records shape: {records_df.shape}")
        print(f"   Columns: {len(records_df.columns)}")
        
        # Test analysis
        relationships = recorder.analyze_left_right_relationships(records_df)
        print(f"   Relationship analysis completed")
        
        # Test saving
        output_path = recorder.save_comprehensive_results("test_dual_lba_output")
        print(f"   Results saved to: {output_path}")
        
        # Test summary
        summary = recorder.get_summary_statistics()
        print(f"   Summary: {len(summary)} metrics")
        
        print("✅ Dual LBA Recorder test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Dual LBA Recorder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dual_lba_recorder()