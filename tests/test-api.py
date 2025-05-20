import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from api.app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestAPIEndpoints:
    """Tests for the API endpoints."""
    
    @pytest.fixture
    def sample_assessment_data(self):
        """Create sample assessment data for API testing."""
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
    
    @pytest.fixture
    def sample_intervention_data(self):
        """Create sample intervention data for API testing."""
        return {
            "student_id": 12,
            "intervention_type": "multisensory_approach",
            "start_date": "2025-05-01",
            "end_date": "2025-06-12",
            "duration_weeks": 6,
            "sessions_completed": 10,
            "pre_assessment_score": 42,
            "post_assessment_score": 56,
            "number_recognition_improvement": 2,
            "number_comparison_improvement": 1,
            "counting_improvement": 1,
            "calculation_improvement": 2,
            "problem_solving_improvement": 1,
            "math_anxiety_change": "decreased",
            "teacher_feedback": "Student shows improved number recognition",
            "parent_feedback": "Child is less anxious about math homework"
        }
    
    @pytest.fixture
    def sample_error_data(self):
        """Create sample error pattern data for API testing."""
        return {
            "student_id": 15,
            "question_type": "addition",
            "question": "6 + 7 = ?",
            "student_answer": "12",
            "correct_answer": "13",
            "is_correct": 0,
            "response_time_seconds": 15.3,
            "attempt_count": 2
        }
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'models_loaded' in data
        assert 'timestamp' in data
    
    # The following tests use patching to mock model functionality
    # to avoid relying on actual trained models during testing
    
    @patch('api.app.screening_model')
    @patch('api.app.recommender')
    @patch('api.app.models_loaded', True)
    def test_screening_endpoint(self, mock_recommender, mock_model, client, sample_assessment_data):
        """Test the screening endpoint."""
        # Mock the model predict function
        mock_model.predict.return_value = {
            "prediction": "dyscalculia",
            "confidence": 0.85,
            "probabilities": {
                "dyscalculia": 0.85,
                "math_difficulty": 0.10,
                "typical": 0.05
            }
        }
        
        # Mock the recommender function
        mock_recommender.recommend_intervention.return_value = {
            "cluster": 2,
            "recommended_interventions": ["multisensory_approach", "visual_aids"],
            "description": "Based on the assessment profile..."
        }
        
        # Send a POST request to the screening endpoint
        response = client.post(
            '/api/screen',
            json=sample_assessment_data,
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check structure
        assert 'screening_result' in data
        assert 'intervention_recommendation' in data
        assert 'prediction' in data['screening_result']
        assert 'confidence' in data['screening_result'] 
        assert 'probabilities' in data['screening_result']
        assert 'recommended_interventions' in data['intervention_recommendation']
    
    @patch('api.app.recommender')
    @patch('api.app.models_loaded', True)
    def test_recommend_endpoint(self, mock_recommender, client, sample_assessment_data):
        """Test the recommend endpoint."""
        # Mock the recommender function
        mock_recommender.recommend_intervention.return_value = {
            "cluster": 2,
            "recommended_interventions": ["multisensory_approach", "visual_aids"],
            "description": "Based on the assessment profile..."
        }
        
        # Send a POST request to the recommend endpoint
        response = client.post(
            '/api/recommend',
            json=sample_assessment_data,
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check structure
        assert 'cluster' in data
        assert 'recommended_interventions' in data
        assert 'description' in data
        assert isinstance(data['cluster'], int)
        assert isinstance(data['recommended_interventions'], list)
    
    @patch('api.app.pd.read_csv')
    @patch('api.app.pd.DataFrame.to_csv')
    def test_save_assessment_endpoint(self, mock_to_csv, mock_read_csv, client, sample_assessment_data):
        """Test the save assessment endpoint."""
        # Mock pd.read_csv to return a DataFrame
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df['student_id'].max.return_value = 20
        mock_read_csv.return_value = mock_df
        
        # Send a POST request to the save assessment endpoint
        response = client.post(
            '/api/save-assessment',
            json=sample_assessment_data,
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check structure
        assert 'status' in data
        assert 'message' in data
        assert 'student_id' in data
        assert data['status'] == 'success'
    
    @patch('api.app.pd.read_csv')
    @patch('api.app.pd.DataFrame.to_csv')
    def test_save_intervention_endpoint(self, mock_to_csv, mock_read_csv, client, sample_intervention_data):
        """Test the save intervention endpoint."""
        # Mock pd.read_csv to return a DataFrame
        mock_df = MagicMock()
        mock_df.empty = False
        mock_read_csv.return_value = mock_df
        
        # Send a POST request to the save intervention endpoint
        response = client.post(
            '/api/save-intervention',
            json=sample_intervention_data,
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check structure
        assert 'status' in data
        assert 'message' in data
        assert 'intervention_id' in data
        assert data['status'] == 'success'
    
    @patch('api.app.pd.read_csv')
    @patch('api.app.pd.DataFrame.to_csv')
    def test_save_error_pattern_endpoint(self, mock_to_csv, mock_read_csv, client, sample_error_data):
        """Test the error pattern endpoint."""
        # Mock pd.read_csv to return a DataFrame
        mock_df = MagicMock()
        mock_df.empty = False
        mock_read_csv.return_value = mock_df
        
        # Send a POST request to the error patterns endpoint
        response = client.post(
            '/api/error-patterns',
            json=sample_error_data,
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check structure
        assert 'status' in data
        assert 'message' in data
        assert 'question_id' in data
        assert data['status'] == 'success'
    
    def test_log_error_endpoint(self, client):
        """Test the log error endpoint."""
        # Send a POST request to the log error endpoint
        response = client.post(
            '/api/log-error',
            json={"message": "Test error message"},
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check structure
        assert 'status' in data
        assert data['status'] == 'error logged'
