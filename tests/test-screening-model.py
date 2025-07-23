import pytest
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dyscalculia_screening_model import DyscalculiaScreeningModel

class TestDyscalculiaScreeningModel:
    """Tests for the dyscalculia screening model."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        model = DyscalculiaScreeningModel()
        try:
            model.load_model()
        except:
            pass  # Model might not exist yet, that's OK for testing
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample assessment data for testing."""
        return {
            "number_recognition": 3,
            "number_comparison": 2,
            "counting_skills": 4,
            "place_value": 2,
            "calculation_accuracy": 2,
            "calculation_fluency": 1,
            "arithmetic_facts_recall": 2,
            "word_problem_solving": 1,
            "working_memory_score": "low",
            "visual_spatial_score": "normal",
            "math_anxiety_level": "high",
            "attention_score": "normal"
        }
    
    def test_model_initialization(self, model):
        """Test that the model initializes correctly."""
        assert model.features is not None
        assert model.target == 'diagnosis'
        assert hasattr(model, 'pipeline')
    
    def test_prediction_format(self, model, sample_data):
        """Test the format of model predictions."""
        # This test will only run if the model file exists
        if os.path.exists(model.model_path):
            prediction = model.predict(sample_data)
            
            # Check if prediction has expected structure
            assert isinstance(prediction, dict)
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert "probabilities" in prediction
            
            # Check if prediction values are valid
            assert prediction["prediction"] in ["dyscalculia", "math_difficulty", "typical"]
            assert 0 <= prediction["confidence"] <= 1
            
            # Check if probabilities sum to approximately 1
            assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
        else:
            pytest.skip("Model file not found, skipping prediction test")
    
    def test_input_format_handling(self, model, sample_data):
        """Test that the model can handle different input formats."""
        if os.path.exists(model.model_path):
            # Test with dictionary input
            dict_prediction = model.predict(sample_data)
            
            # Test with DataFrame input
            df = pd.DataFrame([sample_data])
            df_prediction = model.predict(df)
            
            # The predictions should be identical
            assert dict_prediction == df_prediction
        else:
            pytest.skip("Model file not found, skipping format test")
    
    def test_categorical_mapping(self, model):
        """Test that categorical variables are mapped correctly."""
        # Create test data with different categorical values
        test_data = pd.DataFrame([{
            "number_recognition": 3,
            "number_comparison": 3,
            "counting_skills": 3,
            "place_value": 3,
            "calculation_accuracy": 3,
            "calculation_fluency": 3,
            "arithmetic_facts_recall": 3,
            "word_problem_solving": 3,
            "working_memory_score": "low",
            "visual_spatial_score": "normal",
            "math_anxiety_level": "high",
            "attention_score": "low"
        }])
        
        # Use the preprocess_data method to test categorical mapping
        # This is a whitebox test that assumes knowledge of the model's internals
        try:
            # We're not using the return value, just checking if it processes without errors
            model.preprocess_data(test_data)
            assert True  # If we got here, no exception was raised
        except Exception as e:
            pytest.fail(f"Categorical mapping failed with error: {e}")
    
    def test_save_load_model(self, model, tmp_path):
        """Test saving and loading the model."""
        # Save the model to a temporary path
        temp_model_path = tmp_path / "test_model.pkl"
        model.save_model(path=str(temp_model_path))
        
        # Check if the file was created
        assert temp_model_path.exists()
        
        # Create a new model and load from the temporary path
        new_model = DyscalculiaScreeningModel()
        new_model.load_model(path=str(temp_model_path))
        
        # Check if model was loaded successfully
        assert hasattr(new_model, 'pipeline')
    
    def test_handling_missing_features(self, model, sample_data):
        """Test that the model handles missing features gracefully."""
        if os.path.exists(model.model_path):
            # Create a copy with some features removed
            incomplete_data = sample_data.copy()
            del incomplete_data["calculation_accuracy"]
            del incomplete_data["calculation_fluency"]
            
            try:
                prediction = model.predict(incomplete_data)
                assert "prediction" in prediction  # Should still produce a prediction
            except Exception as e:
                pytest.fail(f"Model failed with missing features: {e}")
        else:
            pytest.skip("Model file not found, skipping missing features test")
