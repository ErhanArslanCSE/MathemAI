import pytest
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.intervention_recommender import InterventionRecommender

class TestInterventionRecommender:
    """Tests for the intervention recommender model."""
    
    @pytest.fixture
    def recommender(self):
        """Create a recommender instance for testing."""
        recommender = InterventionRecommender()
        try:
            recommender.load_model()
        except:
            pass  # Model might not exist yet, that's OK for testing
        return recommender
    
    @pytest.fixture
    def sample_data(self):
        """Create sample student data for testing."""
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
    
    def test_recommender_initialization(self, recommender):
        """Test that the recommender initializes correctly."""
        assert recommender.numeric_features is not None
        assert recommender.categorical_features is not None
        assert hasattr(recommender, 'model')
        assert hasattr(recommender, 'preprocessor')
    
    def test_recommendation_format(self, recommender, sample_data):
        """Test the format of intervention recommendations."""
        # This test will only run if the model file exists
        if os.path.exists(recommender.model_path):
            recommendation = recommender.recommend_intervention(sample_data)
            
            # Check if recommendation has expected structure
            assert isinstance(recommendation, dict)
            assert "cluster" in recommendation
            assert "recommended_interventions" in recommendation
            assert "description" in recommendation
            
            # Check if values are valid
            assert isinstance(recommendation["cluster"], int)
            assert isinstance(recommendation["recommended_interventions"], list)
            assert isinstance(recommendation["description"], str)
            assert recommendation["cluster"] >= 0
        else:
            pytest.skip("Model file not found, skipping recommendation test")
    
    def test_input_format_handling(self, recommender, sample_data):
        """Test that the recommender can handle different input formats."""
        if os.path.exists(recommender.model_path):
            # Test with dictionary input
            dict_recommendation = recommender.recommend_intervention(sample_data)
            
            # Test with DataFrame input
            df = pd.DataFrame([sample_data])
            df_recommendation = recommender.recommend_intervention(df)
            
            # The cluster assignments should be identical
            assert dict_recommendation["cluster"] == df_recommendation["cluster"]
        else:
            pytest.skip("Model file not found, skipping format test")
    
    def test_save_load_model(self, recommender, tmp_path):
        """Test saving and loading the model."""
        # Create test cluster interventions data
        recommender.cluster_interventions = {
            0: ['multisensory_approach', 'visual_aids'],
            1: ['game_based_learning', 'technology_assisted']
        }
        
        # Save the model to a temporary path
        temp_model_path = tmp_path / "test_recommender.pkl"
        recommender.save_model(path=str(temp_model_path))
        
        # Check if the file was created
        assert temp_model_path.exists()
        
        # Create a new recommender and load from the temporary path
        new_recommender = InterventionRecommender()
        new_recommender.load_model(path=str(temp_model_path))
        
        # Check if model and interventions were loaded successfully
        assert hasattr(new_recommender, 'model')
        assert hasattr(new_recommender, 'cluster_interventions')
        assert new_recommender.cluster_interventions == recommender.cluster_interventions
    
    def test_handling_missing_features(self, recommender, sample_data):
        """Test that the recommender handles missing features gracefully."""
        if os.path.exists(recommender.model_path):
            # Create a copy with some features removed
            incomplete_data = sample_data.copy()
            del incomplete_data["calculation_accuracy"]
            del incomplete_data["visual_spatial_score"]
            
            try:
                recommendation = recommender.recommend_intervention(incomplete_data)
                assert "cluster" in recommendation  # Should still produce a recommendation
            except Exception as e:
                pytest.fail(f"Recommender failed with missing features: {e}")
        else:
            pytest.skip("Model file not found, skipping missing features test")
    
    def test_find_optimal_clusters(self, recommender):
        """Test the functionality to find optimal clusters."""
        # Create a simple dataset for testing
        test_features = np.array([
            [1, 1, 1],
            [1, 2, 1],
            [8, 8, 8],
            [9, 8, 9],
            [1, 2, 3],
            [9, 7, 8]
        ])
        
        # Find optimal clusters (should be between 2 and 3 for this dataset)
        optimal_clusters = recommender.find_optimal_clusters(test_features, max_clusters=4)
        
        # Check if result is within expected range
        assert 2 <= optimal_clusters <= 4
    
    def test_cluster_intervention_descriptions(self, recommender):
        """Test that the cluster descriptions are appropriate."""
        # Set up test cluster interventions
        recommender.cluster_interventions = {
            0: ['multisensory_approach', 'visual_aids'],
            1: ['game_based_learning', 'structured_sequence']
        }
        
        # Get descriptions for each cluster
        for cluster, interventions in recommender.cluster_interventions.items():
            description = recommender._get_cluster_description(cluster, pd.DataFrame())
            
            # Check if description contains the intervention names
            for intervention in interventions[:2]:  # Only first two are included in description
                formatted_name = intervention.replace('_', ' ').title()
                assert formatted_name in description
