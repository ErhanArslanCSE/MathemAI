import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble_dyscalculia_model import EnsembleDyscalculiaModel
from models.dyscalculia_screening_model import DyscalculiaScreeningModel

def generate_test_data():
    """Generate synthetic test data if real data is not available."""
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic data with realistic patterns
    data = {
        'number_recognition': np.random.randint(1, 6, n_samples),
        'number_comparison': np.random.randint(1, 6, n_samples),
        'counting_skills': np.random.randint(1, 6, n_samples),
        'place_value': np.random.randint(1, 6, n_samples),
        'calculation_accuracy': np.random.randint(1, 6, n_samples),
        'calculation_fluency': np.random.randint(1, 6, n_samples),
        'arithmetic_facts_recall': np.random.randint(1, 6, n_samples),
        'word_problem_solving': np.random.randint(1, 6, n_samples),
        'working_memory_score': np.random.randint(1, 4, n_samples),
        'visual_spatial_score': np.random.randint(1, 4, n_samples),
        'math_anxiety_level': np.random.choice(['low', 'medium', 'high', 'very_high'], n_samples),
        'attention_score': np.random.choice(['very_low', 'low', 'normal'], n_samples)
    }
    
    # Generate correlated diagnosis based on features
    diagnosis = []
    for i in range(n_samples):
        score_sum = sum([data[key][i] for key in ['number_recognition', 'number_comparison', 
                                                   'calculation_accuracy', 'calculation_fluency']])
        if score_sum <= 8:
            diagnosis.append('dyscalculia')
        elif score_sum <= 12:
            diagnosis.append('math_difficulty')
        else:
            diagnosis.append('typical')
    
    data['diagnosis'] = diagnosis
    
    return pd.DataFrame(data)

def compare_models():
    """Compare original model with ensemble model."""
    print("Comparing Original Model vs Ensemble Model")
    print("="*60)
    
    # Check if real data exists
    data_path = '../datasets/dyscalculia_assessment_data.csv'
    if os.path.exists(data_path):
        print("Loading real data...")
        data = pd.read_csv(data_path)
    else:
        print("Real data not found. Generating synthetic data for testing...")
        data = generate_test_data()
        # Save synthetic data for future use
        os.makedirs('../datasets', exist_ok=True)
        data.to_csv(data_path, index=False)
        print(f"Synthetic data saved to {data_path}")
    
    # Initialize models
    original_model = DyscalculiaScreeningModel()
    ensemble_voting = EnsembleDyscalculiaModel(ensemble_type='voting')
    ensemble_stacking = EnsembleDyscalculiaModel(ensemble_type='stacking')
    
    # Preprocess data (same for all models)
    X_train, X_test, y_train, y_test = original_model.preprocess_data(data)
    
    # Train and evaluate original model
    print("\n1. Training Original Random Forest Model...")
    original_model.train(X_train, y_train, optimize=False)
    orig_results = original_model.evaluate(X_test, y_test)
    
    # Train and evaluate voting ensemble
    print("\n2. Training Voting Ensemble Model...")
    ensemble_voting.train(X_train, y_train, optimize=False)
    voting_results = ensemble_voting.evaluate(X_test, y_test)
    
    # Train and evaluate stacking ensemble
    print("\n3. Training Stacking Ensemble Model...")
    ensemble_stacking.train(X_train, y_train, optimize=False)
    stacking_results = ensemble_stacking.evaluate(X_test, y_test)
    
    # Summary comparison
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Original Model Accuracy: {orig_results['accuracy']:.4f}")
    print(f"Voting Ensemble Accuracy: {voting_results['accuracy']:.4f}")
    print(f"Stacking Ensemble Accuracy: {stacking_results['accuracy']:.4f}")
    print(f"\nVoting Ensemble AUC: {voting_results['avg_auc']:.4f}")
    print(f"Stacking Ensemble AUC: {stacking_results['avg_auc']:.4f}")
    
    # Test prediction confidence
    print("\n" + "="*60)
    print("PREDICTION CONFIDENCE COMPARISON")
    print("="*60)
    
    # Take a sample for prediction
    sample = X_test.iloc[0].to_dict()
    true_label = y_test.iloc[0]
    
    print(f"\nTrue diagnosis: {true_label}")
    
    # Original model prediction
    orig_pred = original_model.predict(sample)
    print(f"\nOriginal Model:")
    print(f"  Prediction: {orig_pred['prediction']}")
    print(f"  Confidence: {orig_pred['confidence']:.3f}")
    
    # Voting ensemble prediction
    voting_pred = ensemble_voting.predict(sample)
    print(f"\nVoting Ensemble:")
    print(f"  Prediction: {voting_pred['prediction']}")
    print(f"  Confidence: {voting_pred['confidence']:.3f}")
    print(f"  Confidence Level: {voting_pred['confidence_level']}")
    if 'individual_predictions' in voting_pred:
        print("  Individual Model Predictions:")
        for model_name, pred_info in voting_pred['individual_predictions'].items():
            print(f"    {model_name}: {pred_info['prediction']} (conf: {pred_info['confidence']:.3f})")
    
    # Stacking ensemble prediction
    stacking_pred = ensemble_stacking.predict(sample)
    print(f"\nStacking Ensemble:")
    print(f"  Prediction: {stacking_pred['prediction']}")
    print(f"  Confidence: {stacking_pred['confidence']:.3f}")
    print(f"  Confidence Level: {stacking_pred['confidence_level']}")
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    ensemble_voting.save_model('../models/ensemble_voting_model.pkl')
    ensemble_stacking.save_model('../models/ensemble_stacking_model.pkl')
    
    print("\nModels saved successfully!")
    
    return {
        'original': orig_results,
        'voting': voting_results,
        'stacking': stacking_results
    }

if __name__ == "__main__":
    results = compare_models()
    print("\nTesting completed successfully!")