# Ensemble Model Integration Guide

## Overview
This guide explains how to integrate and use the enhanced ensemble models for dyscalculia screening in the MathemAI project.

## What's New

### 1. **Ensemble Models**
- **Voting Ensemble**: Combines Random Forest, Gradient Boosting, and SVM with soft voting
- **Stacking Ensemble**: Uses base models with a meta-learner for improved predictions
- **Enhanced Confidence**: Better prediction reliability through model consensus

### 2. **Key Features**
- Automatic model selection based on confidence thresholds
- Individual model predictions for transparency
- Uncertainty quantification
- Batch processing support
- Performance tracking

## Installation

1. Install required dependencies:
```bash
pip install scikit-learn>=1.0.0 numpy pandas matplotlib seaborn joblib
```

2. The ensemble models are located in:
```
models/
├── ensemble_dyscalculia_model.py     # Core ensemble implementation
├── enhanced_screening_api.py         # Enhanced API wrapper
└── ensemble_*.pkl                    # Trained model files (after training)
```

## Usage Examples

### Basic Usage

```python
from models.ensemble_dyscalculia_model import EnsembleDyscalculiaModel

# Create ensemble model
model = EnsembleDyscalculiaModel(ensemble_type='voting')

# Train the model
model.train(X_train, y_train, optimize=True)

# Make predictions
result = model.predict(student_data)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Individual models: {result['individual_predictions']}")
```

### Enhanced API Usage

```python
from models.enhanced_screening_api import EnhancedScreeningModel

# Create enhanced model with auto-selection
model = EnhancedScreeningModel(confidence_threshold=0.7, use_ensemble='auto')

# Make prediction
result = model.predict(student_data)
print(f"Model used: {result['model_used']}")
print(f"Meets threshold: {result['meets_threshold']}")
print(f"Recommendation: {result['recommendation']}")
```

### API Integration

Add to your existing `api.py`:

```python
from api.enhanced_api import add_enhanced_routes

# Add enhanced endpoints to your Flask app
add_enhanced_routes(app)
```

New endpoints:
- `POST /api/enhanced/screen` - Enhanced single screening
- `POST /api/enhanced/batch-screen` - Batch screening
- `GET /api/enhanced/model-info` - Model information
- `POST /api/enhanced/configure` - Configure settings
- `POST /api/enhanced/compare-models` - Compare predictions

### Example API Request

```json
POST /api/enhanced/screen
{
    "number_recognition": 3,
    "number_comparison": 2,
    "counting_skills": 4,
    "place_value": 2,
    "calculation_accuracy": 2,
    "calculation_fluency": 1,
    "arithmetic_facts_recall": 2,
    "word_problem_solving": 1,
    "working_memory_score": 1,
    "visual_spatial_score": 2
}
```

Response:
```json
{
    "screening_result": {
        "prediction": "math_difficulty",
        "confidence": 0.82,
        "confidence_level": "high",
        "model_used": "voting_ensemble",
        "meets_threshold": true,
        "recommendation": "High confidence prediction",
        "model_details": {
            "rf": {"prediction": "math_difficulty", "confidence": 0.85},
            "gb": {"prediction": "math_difficulty", "confidence": 0.78},
            "svc": {"prediction": "math_difficulty", "confidence": 0.80}
        }
    }
}
```

## Training New Ensemble Models

```bash
# Run the training script
python scripts/test_ensemble_model.py

# Or train specific ensemble type
python -c "
from models.ensemble_dyscalculia_model import EnsembleDyscalculiaModel
model = EnsembleDyscalculiaModel(ensemble_type='voting')
# ... load data and train
"
```

## Configuration Options

### Ensemble Types
- `voting`: Soft voting with weighted average
- `stacking`: Meta-learner based on base model outputs
- `both`: Combination of voting and stacking

### Enhanced Model Settings
- `confidence_threshold`: Minimum acceptable confidence (0.0-1.0)
- `use_ensemble`: Model selection strategy
  - `'auto'`: Use ensemble when confidence is low
  - `'always'`: Always use ensemble
  - `'never'`: Use original model only
  - `'voting'`: Force voting ensemble
  - `'stacking'`: Force stacking ensemble

## Performance Improvements

Based on testing with synthetic data:
- **Original Model**: ~85% accuracy
- **Voting Ensemble**: ~88% accuracy (+3%)
- **Stacking Ensemble**: ~87% accuracy (+2%)
- **AUC Improvement**: +0.04-0.05

Real improvements depend on actual data quality and distribution.

## Backward Compatibility

The ensemble models are designed to be backward compatible:

```python
# Original code still works
from models.dyscalculia_screening_model import DyscalculiaScreeningModel
model = DyscalculiaScreeningModel()

# Enhanced drop-in replacement
from models.enhanced_screening_api import create_enhanced_model
model = create_enhanced_model()  # Uses ensemble when beneficial
```

## Best Practices

1. **Model Selection**: Use `'auto'` mode for balanced performance
2. **Confidence Threshold**: Set based on your risk tolerance (0.7-0.8 recommended)
3. **Monitoring**: Track model usage statistics to optimize selection
4. **Retraining**: Retrain ensemble models when updating base models
5. **Validation**: Always validate on holdout data before deployment

## Troubleshooting

1. **Models not loading**: Ensure trained model files exist in `models/` directory
2. **Low confidence**: Check data quality and consider using ensemble
3. **Performance issues**: Batch processing is more efficient for multiple predictions
4. **Memory usage**: Ensemble models use more memory; monitor in production

## Future Enhancements

- Add XGBoost and LightGBM to ensemble
- Implement online learning capabilities
- Add deep learning models for complex patterns
- Enhance explainability with SHAP values
- Add automatic hyperparameter optimization