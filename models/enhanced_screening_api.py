"""
Enhanced API integration for ensemble models.
This module provides a drop-in replacement for the original screening model
with automatic ensemble selection based on confidence requirements.
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, Optional

from dyscalculia_screening_model import DyscalculiaScreeningModel
from ensemble_dyscalculia_model import EnsembleDyscalculiaModel


class EnhancedScreeningModel:
    """
    Enhanced screening model that automatically selects between
    original and ensemble models based on requirements.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, use_ensemble: str = 'auto'):
        """
        Initialize the enhanced screening model.
        
        Parameters:
        -----------
        confidence_threshold : float
            Minimum confidence required for predictions
        use_ensemble : str
            'auto' - automatically select based on confidence
            'always' - always use ensemble
            'never' - never use ensemble (use original)
            'voting' - use voting ensemble
            'stacking' - use stacking ensemble
        """
        self.confidence_threshold = confidence_threshold
        self.use_ensemble = use_ensemble
        
        # Initialize models
        self.original_model = DyscalculiaScreeningModel()
        self.voting_ensemble = None
        self.stacking_ensemble = None
        
        # Try to load pre-trained models
        self._load_models()
        
        # Track model performance
        self.performance_stats = {
            'original': {'predictions': 0, 'avg_confidence': 0},
            'voting': {'predictions': 0, 'avg_confidence': 0},
            'stacking': {'predictions': 0, 'avg_confidence': 0}
        }
    
    def _load_models(self):
        """Load pre-trained models if available."""
        # Load original model
        if os.path.exists('../models/dyscalculia_screening_model.pkl'):
            self.original_model.load_model()
            print("Loaded original model")
        
        # Load ensemble models
        if os.path.exists('../models/ensemble_voting_model.pkl'):
            self.voting_ensemble = EnsembleDyscalculiaModel(ensemble_type='voting')
            self.voting_ensemble.load_model('../models/ensemble_voting_model.pkl')
            print("Loaded voting ensemble model")
        
        if os.path.exists('../models/ensemble_stacking_model.pkl'):
            self.stacking_ensemble = EnsembleDyscalculiaModel(ensemble_type='stacking')
            self.stacking_ensemble.load_model('../models/ensemble_stacking_model.pkl')
            print("Loaded stacking ensemble model")
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions with automatic model selection.
        
        Parameters:
        -----------
        data : dict or DataFrame
            Assessment data for prediction
            
        Returns:
        --------
        dict
            Enhanced prediction results with model selection info
        """
        # First, get original model prediction
        original_pred = self.original_model.predict(data)
        
        # Determine which model to use
        model_choice = self._select_model(original_pred)
        
        if model_choice == 'original':
            final_pred = original_pred
            model_used = 'original'
        elif model_choice == 'voting' and self.voting_ensemble:
            final_pred = self.voting_ensemble.predict(data)
            model_used = 'voting_ensemble'
        elif model_choice == 'stacking' and self.stacking_ensemble:
            final_pred = self.stacking_ensemble.predict(data)
            model_used = 'stacking_ensemble'
        else:
            # Fallback to original if ensemble not available
            final_pred = original_pred
            model_used = 'original (fallback)'
        
        # Update performance statistics
        self._update_stats(model_used, final_pred['confidence'])
        
        # Enhance the prediction result
        enhanced_result = {
            **final_pred,
            'model_used': model_used,
            'confidence_threshold': self.confidence_threshold,
            'meets_threshold': final_pred['confidence'] >= self.confidence_threshold
        }
        
        # Add recommendation based on confidence
        if final_pred['confidence'] < 0.5:
            enhanced_result['recommendation'] = 'Consider additional assessment'
        elif final_pred['confidence'] < self.confidence_threshold:
            enhanced_result['recommendation'] = 'Moderate confidence - monitor closely'
        else:
            enhanced_result['recommendation'] = 'High confidence prediction'
        
        # Add uncertainty quantification
        if 'probabilities' in final_pred:
            probs = list(final_pred['probabilities'].values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            max_entropy = -np.log(1.0 / len(probs))
            enhanced_result['uncertainty'] = entropy / max_entropy  # Normalized entropy
        
        return enhanced_result
    
    def _select_model(self, original_pred: Dict[str, Any]) -> str:
        """Select which model to use based on configuration and confidence."""
        if self.use_ensemble == 'never':
            return 'original'
        elif self.use_ensemble == 'always':
            return 'voting' if self.voting_ensemble else 'original'
        elif self.use_ensemble == 'voting':
            return 'voting' if self.voting_ensemble else 'original'
        elif self.use_ensemble == 'stacking':
            return 'stacking' if self.stacking_ensemble else 'original'
        elif self.use_ensemble == 'auto':
            # Use ensemble if original confidence is below threshold
            if original_pred['confidence'] < self.confidence_threshold:
                # Prefer voting ensemble for better interpretability
                if self.voting_ensemble:
                    return 'voting'
                elif self.stacking_ensemble:
                    return 'stacking'
            return 'original'
        
        return 'original'
    
    def _update_stats(self, model_used: str, confidence: float):
        """Update performance statistics."""
        model_key = model_used.split()[0].replace('_ensemble', '')
        if model_key in self.performance_stats:
            stats = self.performance_stats[model_key]
            n = stats['predictions']
            stats['avg_confidence'] = (stats['avg_confidence'] * n + confidence) / (n + 1)
            stats['predictions'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model usage and performance."""
        return {
            'performance_stats': self.performance_stats,
            'models_available': {
                'original': True,
                'voting_ensemble': self.voting_ensemble is not None,
                'stacking_ensemble': self.stacking_ensemble is not None
            },
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'use_ensemble': self.use_ensemble
            }
        }
    
    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple samples with model selection info.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with multiple assessment records
            
        Returns:
        --------
        pd.DataFrame
            Predictions with model selection and confidence info
        """
        results = []
        
        for idx, row in data.iterrows():
            pred = self.predict(row.to_dict())
            pred['index'] = idx
            results.append(pred)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add summary statistics
        print(f"\nBatch Prediction Summary:")
        print(f"Total predictions: {len(results_df)}")
        print(f"Model usage:")
        model_counts = results_df['model_used'].value_counts()
        for model, count in model_counts.items():
            print(f"  {model}: {count} ({count/len(results_df)*100:.1f}%)")
        
        print(f"\nConfidence statistics:")
        print(f"  Mean: {results_df['confidence'].mean():.3f}")
        print(f"  Std: {results_df['confidence'].std():.3f}")
        print(f"  Below threshold: {(~results_df['meets_threshold']).sum()} "
              f"({(~results_df['meets_threshold']).sum()/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def save_model(self, path: str = '../models/enhanced_screening_model.pkl'):
        """Save the enhanced model configuration."""
        config = {
            'confidence_threshold': self.confidence_threshold,
            'use_ensemble': self.use_ensemble,
            'performance_stats': self.performance_stats
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(config, path)
        print(f"Enhanced model configuration saved to {path}")
    
    def load_model(self, path: str = '../models/enhanced_screening_model.pkl'):
        """Load the enhanced model configuration."""
        if os.path.exists(path):
            config = joblib.load(path)
            self.confidence_threshold = config['confidence_threshold']
            self.use_ensemble = config['use_ensemble']
            self.performance_stats = config.get('performance_stats', self.performance_stats)
            print(f"Enhanced model configuration loaded from {path}")
            
            # Reload models
            self._load_models()


# Backward compatibility wrapper
def create_enhanced_model(original_model_instance=None, **kwargs):
    """
    Create an enhanced model that's backward compatible with the original API.
    
    This function allows easy integration into existing code by replacing:
    model = DyscalculiaScreeningModel()
    
    With:
    model = create_enhanced_model()
    """
    enhanced = EnhancedScreeningModel(**kwargs)
    
    # If an original model instance is provided, copy its configuration
    if original_model_instance:
        enhanced.features = original_model_instance.features
        enhanced.target = original_model_instance.target
    
    # Add compatibility methods
    enhanced.train = lambda X, y, optimize=True: enhanced.original_model.train(X, y, optimize)
    enhanced.evaluate = lambda X, y: enhanced.original_model.evaluate(X, y)
    enhanced.load_data = lambda path: enhanced.original_model.load_data(path)
    enhanced.preprocess_data = lambda data: enhanced.original_model.preprocess_data(data)
    
    return enhanced


if __name__ == "__main__":
    # Example usage
    print("Testing Enhanced Screening Model")
    print("="*50)
    
    # Create enhanced model
    model = EnhancedScreeningModel(confidence_threshold=0.75, use_ensemble='auto')
    
    # Get performance summary
    print("\nModel Status:")
    summary = model.get_performance_summary()
    print(f"Models available: {summary['models_available']}")
    print(f"Configuration: {summary['configuration']}")
    
    # Test prediction
    test_data = {
        'number_recognition': 3,
        'number_comparison': 2,
        'counting_skills': 4,
        'place_value': 2,
        'calculation_accuracy': 2,
        'calculation_fluency': 1,
        'arithmetic_facts_recall': 2,
        'word_problem_solving': 1,
        'working_memory_score': 1,
        'visual_spatial_score': 2
    }
    
    print("\nTest Prediction:")
    result = model.predict(test_data)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Model used: {result['model_used']}")
    print(f"Meets threshold: {result['meets_threshold']}")
    print(f"Recommendation: {result['recommendation']}")
    if 'uncertainty' in result:
        print(f"Uncertainty: {result['uncertainty']:.3f}")
    
    # Save configuration
    model.save_model()