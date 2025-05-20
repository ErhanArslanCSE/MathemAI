import pytest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import needed modules
from models.dyscalculia_screening_model import DyscalculiaScreeningModel
from models.intervention_recommender import InterventionRecommender

@pytest.fixture(scope="session")
def create_test_data():
    """Create small test datasets for testing."""
    
    # Create test assessment data
    assessment_data = pd.DataFrame({
        'student_id': list(range(1, 11)),
        'age': np.random.randint(6, 13, 10),
        'grade': np.random.randint(1, 8, 10),
        'number_recognition': np.random.randint(1, 6, 10),
        'number_comparison': np.random.randint(1, 6, 10),
        'counting_skills': np.random.randint(1, 6, 10),
        'place_value': np.random.randint(1, 6, 10),
        'calculation_accuracy': np.random.randint(1, 6, 10),
        'calculation_fluency': np.random.randint(1, 6, 10),
        'arithmetic_facts_recall': np.random.randint(1, 6, 10),
        'word_problem_solving': np.random.randint(1, 6, 10),
        'math_anxiety_level': np.random.choice(['low', 'medium', 'high'], 10),
        'attention_score': np.random.choice(['normal', 'low', 'very_low'], 10),
        'working_memory_score': np.random.choice(['normal', 'low', 'very_low'], 10),
        'visual_spatial_score': np.random.choice(['normal', 'low', 'very_low'], 10),
        'error_patterns': np.random.choice(['transposition', 'reversal', 'miscounting', 'none'], 10),
        'response_time': np.random.choice(['fast', 'average', 'slow'], 10),
        'diagnosis': np.random.choice(['dyscalculia', 'math_difficulty', 'typical'], 10)
    })
    
    # Create test intervention data
    intervention_data = pd.DataFrame({
        'student_id': list(range(1, 6)),  # Only half the students have interventions
        'intervention_id': [f'INT{i:03d}' for i in range(1, 6)],
        'intervention_type': np.random.choice([
            'multisensory_approach', 'visual_aids', 'game_based_learning',
            'structured_sequence', 'technology_assisted'
        ], 5),
        'start_date': ['2025-01-15'] * 5,
        'end_date': ['2025-02-26'] * 5,
        'duration_weeks': [6] * 5,
        'sessions_completed': np.random.randint(8, 13, 5),
        'pre_assessment_score': np.random.randint(30, 71, 5),
        'post_assessment_score': np.random.randint(40, 81, 5),
        'number_recognition_improvement': np.random.randint(0, 3, 5),
        'number_comparison_improvement': np.random.randint(0, 3, 5),
        'counting_improvement': np.random.randint(0, 3, 5),
        'calculation_improvement': np.random.randint(0, 3, 5),
        'problem_solving_improvement': np.random.randint(0, 3, 5),
        'math_anxiety_change': np.random.choice(['decreased', 'no_change', 'increased'], 5),
        'teacher_feedback': ['Feedback from teacher'] * 5,
        'parent_feedback': ['Feedback from parent'] * 5
    })
    
    # Create test error analysis data
    error_data = pd.DataFrame({
        'student_id': np.random.choice(range(1, 11), 15),
        'question_id': [f'Q{i}' for i in range(1, 16)],
        'question_type': np.random.choice([
            'addition', 'subtraction', 'number_comparison', 'counting'
        ], 15),
        'question': ['Test question'] * 15,
        'student_answer': ['Student answer'] * 15,
        'correct_answer': ['Correct answer'] * 15,
        'is_correct': np.random.choice([0, 1], 15),
        'response_time_seconds': np.random.uniform(5, 20, 15),
        'attempt_count': np.random.randint(1, 4, 15),
        'session_date': ['2025-03-01'] * 15
    })
    
    return {
        'assessment_data': assessment_data,
        'intervention_data': intervention_data,
        'error_data': error_data
    }

@pytest.fixture
def sample_student():
    """Create a sample student data dictionary for testing."""
    return {
        'student_id': 100,
        'age': 8,
        'grade': 3,
        'number_recognition': 3,
        'number_comparison': 2,
        'counting_skills': 4,
        'place_value': 2,
        'calculation_accuracy': 2,
        'calculation_fluency': 1,
        'arithmetic_facts_recall': 2,
        'word_problem_solving': 1,
        'math_anxiety_level': 'high',
        'attention_score': 'normal',
        'working_memory_score': 'low',
        'visual_spatial_score': 'normal',
        'error_patterns': 'transposition',
        'response_time': 'slow',
        'diagnosis': 'dyscalculia'
    }

@pytest.fixture
def mock_screening_model():
    """Create a simple mock screening model for testing."""
    model = DyscalculiaScreeningModel()
    
    # Mock the predict method to avoid needing a trained model
    def mock_predict(data):
        # This is a very simple mock that returns a fixed prediction
        return {
            'prediction': 'dyscalculia',
            'confidence': 0.85,
            'probabilities': {
                'dyscalculia': 0.85,
                'math_difficulty': 0.10,
                'typical': 0.05
            }
        }
    
    model.predict = mock_predict
    return model

@pytest.fixture
def mock_recommender():
    """Create a simple mock recommender for testing."""
    recommender = InterventionRecommender()
    
    # Mock the recommend_intervention method
    def mock_recommend(data):
        # This is a very simple mock that returns fixed recommendations
        return {
            'cluster': 2,
            'recommended_interventions': ['multisensory_approach', 'visual_aids'],
            'description': 'Based on the assessment profile, this student would benefit from...'
        }
    
    recommender.recommend_intervention = mock_recommend
    return recommender
