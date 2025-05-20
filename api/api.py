from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dyscalculia_screening_model import DyscalculiaScreeningModel
from models.intervention_recommender import InterventionRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

# Load models
screening_model = DyscalculiaScreeningModel()
recommender = InterventionRecommender()

try:
    screening_model.load_model()
    recommender.load_model()
    models_loaded = True
    logger.info("Models loaded successfully")
except Exception as e:
    models_loaded = False
    logger.error(f"Failed to load models: {e}")
    print("Models not found. Please train the models first.")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/screen', methods=['POST'])
def screen_for_dyscalculia():
    """
    Screen a student for dyscalculia based on assessment data.
    
    Expected JSON payload:
    {
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
    """
    if not models_loaded:
        logger.error("Models not loaded for screening request")
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 500
    
    try:
        # Get data from request
        data = request.json
        logger.info(f"Received screening request with data: {data}")
        
        # Prepare data for prediction
        # Convert categorical variables if needed
        if 'math_anxiety_level' in data:
            anxiety_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
            if isinstance(data['math_anxiety_level'], str):
                data['math_anxiety_level'] = anxiety_map.get(data['math_anxiety_level'], 1)
        
        if 'attention_score' in data:
            attention_map = {'normal': 2, 'low': 1, 'very_low': 0}
            if isinstance(data['attention_score'], str):
                data['attention_score'] = attention_map.get(data['attention_score'], 1)
                
        if 'working_memory_score' in data:
            memory_map = {'normal': 2, 'low': 1, 'very_low': 0}
            if isinstance(data['working_memory_score'], str):
                data['working_memory_score'] = memory_map.get(data['working_memory_score'], 1)
                
        if 'visual_spatial_score' in data:
            spatial_map = {'normal': 2, 'low': 1, 'very_low': 0}
            if isinstance(data['visual_spatial_score'], str):
                data['visual_spatial_score'] = spatial_map.get(data['visual_spatial_score'], 1)
        
        # Convert to DataFrame
        student_data = pd.DataFrame([data])
        
        # Make prediction
        prediction_result = screening_model.predict(student_data)
        
        # Get intervention recommendation
        recommendation = recommender.recommend_intervention(student_data)
        
        # Combine results
        result = {
            'screening_result': prediction_result,
            'intervention_recommendation': recommendation
        }
        
        logger.info(f"Screening result: {json.dumps(result, default=str)}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in screening: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred during screening: {str(e)}'
        }), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_intervention():
    """
    Recommend interventions for a student based on assessment data.
    
    This endpoint can be used separately from screening if the user
    already has assessment data and just needs recommendations.
    """
    if not models_loaded:
        logger.error("Models not loaded for recommendation request")
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 500
    
    try:
        # Get data from request
        data = request.json
        logger.info(f"Received recommendation request with data: {data}")
        
        # Prepare data
        if 'math_anxiety_level' in data:
            anxiety_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
            if isinstance(data['math_anxiety_level'], str):
                data['math_anxiety_level'] = anxiety_map.get(data['math_anxiety_level'], 1)
        
        if 'attention_score' in data:
            attention_map = {'normal': 2, 'low': 1, 'very_low': 0}
            if isinstance(data['attention_score'], str):
                data['attention_score'] = attention_map.get(data['attention_score'], 1)
                
        if 'working_memory_score' in data:
            memory_map = {'normal': 2, 'low': 1, 'very_low': 0}
            if isinstance(data['working_memory_score'], str):
                data['working_memory_score'] = memory_map.get(data['working_memory_score'], 1)
                
        if 'visual_spatial_score' in data:
            spatial_map = {'normal': 2, 'low': 1, 'very_low': 0}
            if isinstance(data['visual_spatial_score'], str):
                data['visual_spatial_score'] = spatial_map.get(data['visual_spatial_score'], 1)
        
        # Convert to DataFrame
        student_data = pd.DataFrame([data])
        
        # Get recommendations
        recommendations = recommender.recommend_intervention(student_data)
        
        logger.info(f"Recommendation result: {json.dumps(recommendations, default=str)}")
        
        return jsonify(recommendations)
    
    except Exception as e:
        logger.error(f"Error in recommendation: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred during recommendation: {str(e)}'
        }), 500

@app.route('/api/log-error', methods=['POST'])
def log_error():
    """Log client-side errors"""
    try:
        data = request.json
        logger.error(f"Client-side error: {data.get('message')}")
        return jsonify({'status': 'error logged'})
    except Exception as e:
        logger.error(f"Error logging client error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-assessment', methods=['POST'])
def save_assessment():
    """
    Save a new assessment record to the dataset.
    This allows the system to collect more data over time.
    """
    try:
        data = request.json
        logger.info(f"Received new assessment data: {data}")
        
        # Load existing assessment data
        try:
            assessment_df = pd.read_csv('../datasets/dyscalculia_assessment_data.csv')
        except FileNotFoundError:
            # Create a new DataFrame if file doesn't exist
            assessment_df = pd.DataFrame(columns=[
                'student_id', 'age', 'grade', 'number_recognition', 'number_comparison',
                'counting_skills', 'place_value', 'calculation_accuracy', 'calculation_fluency',
                'arithmetic_facts_recall', 'word_problem_solving', 'math_anxiety_level',
                'attention_score', 'working_memory_score', 'visual_spatial_score',
                'error_patterns', 'response_time', 'diagnosis'
            ])
        
        # Generate a new student_id if not provided
        if 'student_id' not in data:
            # Use the max existing ID + 1, or 1 if no records exist
            data['student_id'] = int(assessment_df['student_id'].max() + 1) if not assessment_df.empty else 1
        
        # Add a row to the DataFrame
        new_row = pd.DataFrame([data])
        assessment_df = pd.concat([assessment_df, new_row], ignore_index=True)
        
        # Save the updated DataFrame
        assessment_df.to_csv('../datasets/dyscalculia_assessment_data.csv', index=False)
        
        logger.info(f"Saved new assessment data for student ID: {data['student_id']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Assessment data saved successfully',
            'student_id': data['student_id']
        })
    
    except Exception as e:
        logger.error(f"Error saving assessment data: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred saving assessment data: {str(e)}'
        }), 500

@app.route('/api/save-intervention', methods=['POST'])
def save_intervention():
    """
    Save a new intervention record to the dataset.
    This allows the system to track intervention effectiveness.
    """
    try:
        data = request.json
        logger.info(f"Received new intervention data: {data}")
        
        # Load existing intervention data
        try:
            intervention_df = pd.read_csv('../datasets/intervention_tracking_data.csv')
        except FileNotFoundError:
            # Create a new DataFrame if file doesn't exist
            intervention_df = pd.DataFrame(columns=[
                'student_id', 'intervention_id', 'intervention_type', 'start_date', 'end_date',
                'duration_weeks', 'sessions_completed', 'pre_assessment_score', 'post_assessment_score',
                'number_recognition_improvement', 'number_comparison_improvement', 'counting_improvement',
                'calculation_improvement', 'problem_solving_improvement', 'math_anxiety_change',
                'teacher_feedback', 'parent_feedback'
            ])
        
        # Generate a new intervention_id if not provided
        if 'intervention_id' not in data:
            # Create a new ID with format INT###
            last_id = 0
            if not intervention_df.empty and 'intervention_id' in intervention_df.columns:
                # Extract the numeric part of the last ID if it exists
                existing_ids = intervention_df['intervention_id'].str.extract(r'INT(\d+)')
                if not existing_ids.empty and not existing_ids.dropna().empty:
                    last_id = int(existing_ids.dropna().astype(int).max())
            
            # Create new ID
            data['intervention_id'] = f"INT{str(last_id + 1).zfill(3)}"
        
        # Add a row to the DataFrame
        new_row = pd.DataFrame([data])
        intervention_df = pd.concat([intervention_df, new_row], ignore_index=True)
        
        # Save the updated DataFrame
        intervention_df.to_csv('../datasets/intervention_tracking_data.csv', index=False)
        
        logger.info(f"Saved new intervention data with ID: {data['intervention_id']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Intervention data saved successfully',
            'intervention_id': data['intervention_id']
        })
    
    except Exception as e:
        logger.error(f"Error saving intervention data: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred saving intervention data: {str(e)}'
        }), 500

@app.route('/api/error-patterns', methods=['POST'])
def save_error_pattern():
    """
    Save a new error pattern record to the dataset.
    This helps track specific types of mathematical errors for research.
    """
    try:
        data = request.json
        logger.info(f"Received new error pattern data: {data}")
        
        # Load existing error pattern data
        try:
            error_df = pd.read_csv('../datasets/error_analysis_data.csv')
        except FileNotFoundError:
            # Create a new DataFrame if file doesn't exist
            error_df = pd.DataFrame(columns=[
                'student_id', 'question_id', 'question_type', 'question', 
                'student_answer', 'correct_answer', 'is_correct',
                'response_time_seconds', 'attempt_count', 'session_date'
            ])
        
        # Generate a new question_id if not provided
        if 'question_id' not in data:
            # Create a new ID with format Q###
            if not error_df.empty and 'question_id' in error_df.columns:
                # Extract the numeric part of the last ID
                existing_ids = error_df['question_id'].str.extract(r'Q(\d+)')
                if not existing_ids.empty and not existing_ids.dropna().empty:
                    last_id = int(existing_ids.dropna().astype(int).max())
                    data['question_id'] = f"Q{last_id + 1}"
                else:
                    data['question_id'] = "Q1"
            else:
                data['question_id'] = "Q1"
        
        # Add session date if not provided
        if 'session_date' not in data:
            data['session_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add is_correct if not provided
        if 'is_correct' not in data:
            # Compare student_answer and correct_answer
            data['is_correct'] = int(str(data.get('student_answer', '')).strip() == 
                                   str(data.get('correct_answer', '')).strip())
        
        # Add a row to the DataFrame
        new_row = pd.DataFrame([data])
        error_df = pd.concat([error_df, new_row], ignore_index=True)
        
        # Save the updated DataFrame
        error_df.to_csv('../datasets/error_analysis_data.csv', index=False)
        
        logger.info(f"Saved new error pattern data with ID: {data['question_id']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Error pattern data saved successfully',
            'question_id': data['question_id']
        })
    
    except Exception as e:
        logger.error(f"Error saving error pattern data: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred saving error pattern data: {str(e)}'
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_models():
    """
    Endpoint to trigger model retraining.
    This allows for periodic updates to the models as more data is collected.
    """
    try:
        # Check for authorization
        auth_key = request.headers.get('X-API-Key')
        if not auth_key or auth_key != os.environ.get('MATHEMAT_API_KEY', 'dev_key'):
            logger.warning("Unauthorized retrain attempt")
            return jsonify({
                'error': 'Unauthorized'
            }), 401
        
        logger.info("Starting model retraining process")
        
        # Import the training scripts
        from scripts.train_models import train_all_models
        
        # Trigger the training
        result = train_all_models()
        
        if result['success']:
            # Reload the models
            global screening_model, recommender, models_loaded
            
            try:
                screening_model = DyscalculiaScreeningModel()
                recommender = InterventionRecommender()
                
                screening_model.load_model()
                recommender.load_model()
                
                models_loaded = True
                logger.info("Models reloaded successfully after retraining")
            except Exception as e:
                logger.error(f"Failed to reload models after retraining: {e}", exc_info=True)
                models_loaded = False
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during model retraining: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'An error occurred during model retraining: {str(e)}'
        }), 500

@app.route('/api/export-data', methods=['GET'])
def export_data():
    """
    Export datasets in a format suitable for research or backup.
    """
    try:
        # Check for authorization
        auth_key = request.headers.get('X-API-Key')
        if not auth_key or auth_key != os.environ.get('MATHEMAT_API_KEY', 'dev_key'):
            logger.warning("Unauthorized data export attempt")
            return jsonify({
                'error': 'Unauthorized'
            }), 401
        
        dataset_type = request.args.get('type', 'all')
        format_type = request.args.get('format', 'json')
        
        data = {}
        
        # Load the requested datasets
        if dataset_type in ['all', 'assessment']:
            try:
                assessment_df = pd.read_csv('../datasets/dyscalculia_assessment_data.csv')
                if format_type == 'json':
                    data['assessment_data'] = assessment_df.to_dict(orient='records')
            except Exception as e:
                logger.error(f"Error loading assessment data: {e}")
                data['assessment_data'] = {'error': str(e)}
        
        if dataset_type in ['all', 'intervention']:
            try:
                intervention_df = pd.read_csv('../datasets/intervention_tracking_data.csv')
                if format_type == 'json':
                    data['intervention_data'] = intervention_df.to_dict(orient='records')
            except Exception as e:
                logger.error(f"Error loading intervention data: {e}")
                data['intervention_data'] = {'error': str(e)}
        
        if dataset_type in ['all', 'error']:
            try:
                error_df = pd.read_csv('../datasets/error_analysis_data.csv')
                if format_type == 'json':
                    data['error_data'] = error_df.to_dict(orient='records')
            except Exception as e:
                logger.error(f"Error loading error data: {e}")
                data['error_data'] = {'error': str(e)}
        
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred during data export: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """
    Get basic statistics about the collected data.
    """
    try:
        stats = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        # Assessment data stats
        try:
            assessment_df = pd.read_csv('../datasets/dyscalculia_assessment_data.csv')
            stats['datasets']['assessment'] = {
                'total_records': len(assessment_df),
                'dyscalculia_count': len(assessment_df[assessment_df['diagnosis'] == 'dyscalculia']),
                'math_difficulty_count': len(assessment_df[assessment_df['diagnosis'] == 'math_difficulty']),
                'typical_count': len(assessment_df[assessment_df['diagnosis'] == 'typical'])
            }
        except Exception as e:
            stats['datasets']['assessment'] = {'error': str(e)}
        
        # Intervention data stats
        try:
            intervention_df = pd.read_csv('../datasets/intervention_tracking_data.csv')
            stats['datasets']['intervention'] = {
                'total_records': len(intervention_df),
                'unique_students': len(intervention_df['student_id'].unique()),
                'intervention_types': intervention_df['intervention_type'].value_counts().to_dict(),
                'avg_improvement': (
                    intervention_df['post_assessment_score'] - 
                    intervention_df['pre_assessment_score']
                ).mean()
            }
        except Exception as e:
            stats['datasets']['intervention'] = {'error': str(e)}
        
        # Error analysis stats
        try:
            error_df = pd.read_csv('../datasets/error_analysis_data.csv')
            stats['datasets']['error_analysis'] = {
                'total_records': len(error_df),
                'unique_students': len(error_df['student_id'].unique()),
                'question_types': error_df['question_type'].value_counts().to_dict(),
                'correct_percentage': (error_df['is_correct'].sum() / len(error_df)) * 100,
                'avg_response_time': error_df['response_time_seconds'].mean()
            }
        except Exception as e:
            stats['datasets']['error_analysis'] = {'error': str(e)}
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error generating statistics: {e}", exc_info=True)
        return jsonify({
            'error': f'An error occurred generating statistics: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create log directory if it doesn't exist
    os.makedirs('../logs', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))