"""
Enhanced API endpoints with ensemble model support.
This can be integrated into the existing api.py or used as a standalone enhancement.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_screening_api import EnhancedScreeningModel

# Initialize enhanced model
enhanced_model = EnhancedScreeningModel(confidence_threshold=0.7, use_ensemble='auto')

def add_enhanced_routes(app):
    """Add enhanced API routes to existing Flask app."""
    
    @app.route('/api/enhanced/screen', methods=['POST'])
    def enhanced_screen():
        """
        Enhanced screening endpoint with ensemble models.
        
        Accepts same payload as original /api/screen but returns enhanced results.
        """
        try:
            data = request.json
            
            # Use enhanced model for prediction
            result = enhanced_model.predict(data)
            
            # Format response
            response = {
                'screening_result': {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result.get('probabilities', {}),
                    'confidence_level': result.get('confidence_level', 'unknown'),
                    'model_used': result['model_used'],
                    'meets_threshold': result['meets_threshold'],
                    'recommendation': result['recommendation']
                }
            }
            
            # Add individual model predictions if available
            if 'individual_predictions' in result:
                response['screening_result']['model_details'] = result['individual_predictions']
            
            # Add uncertainty measure if available
            if 'uncertainty' in result:
                response['screening_result']['uncertainty'] = result['uncertainty']
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced/batch-screen', methods=['POST'])
    def batch_screen():
        """
        Batch screening endpoint for multiple students.
        
        Expected payload:
        {
            "students": [
                {
                    "student_id": "123",
                    "number_recognition": 3,
                    ...
                },
                ...
            ]
        }
        """
        try:
            data = request.json
            students_data = data.get('students', [])
            
            if not students_data:
                return jsonify({'error': 'No student data provided'}), 400
            
            # Convert to DataFrame
            df = pd.DataFrame(students_data)
            
            # Get student IDs if provided
            student_ids = df.get('student_id', range(len(df)))
            
            # Remove non-feature columns
            feature_df = df.drop(['student_id'], axis=1, errors='ignore')
            
            # Batch predict
            results_df = enhanced_model.batch_predict(feature_df)
            
            # Format results
            results = []
            for idx, (_, row) in enumerate(results_df.iterrows()):
                result = {
                    'student_id': student_ids.iloc[idx] if hasattr(student_ids, 'iloc') else student_ids[idx],
                    'prediction': row['prediction'],
                    'confidence': row['confidence'],
                    'model_used': row['model_used'],
                    'meets_threshold': row['meets_threshold'],
                    'recommendation': row['recommendation']
                }
                results.append(result)
            
            # Add summary statistics
            summary = {
                'total_screened': len(results),
                'confidence_stats': {
                    'mean': results_df['confidence'].mean(),
                    'std': results_df['confidence'].std(),
                    'min': results_df['confidence'].min(),
                    'max': results_df['confidence'].max()
                },
                'model_usage': results_df['model_used'].value_counts().to_dict(),
                'below_threshold_count': (~results_df['meets_threshold']).sum()
            }
            
            return jsonify({
                'results': results,
                'summary': summary
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced/model-info', methods=['GET'])
    def model_info():
        """Get information about available models and performance."""
        try:
            summary = enhanced_model.get_performance_summary()
            
            return jsonify({
                'models_available': summary['models_available'],
                'configuration': summary['configuration'],
                'performance_stats': summary['performance_stats']
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced/configure', methods=['POST'])
    def configure_model():
        """
        Configure the enhanced model settings.
        
        Expected payload:
        {
            "confidence_threshold": 0.75,
            "use_ensemble": "auto"  // "auto", "always", "never", "voting", "stacking"
        }
        """
        try:
            data = request.json
            
            # Update configuration
            if 'confidence_threshold' in data:
                enhanced_model.confidence_threshold = float(data['confidence_threshold'])
            
            if 'use_ensemble' in data:
                enhanced_model.use_ensemble = data['use_ensemble']
            
            # Save configuration
            enhanced_model.save_model()
            
            return jsonify({
                'status': 'success',
                'configuration': {
                    'confidence_threshold': enhanced_model.confidence_threshold,
                    'use_ensemble': enhanced_model.use_ensemble
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced/compare-models', methods=['POST'])
    def compare_models():
        """
        Compare predictions from different models for the same input.
        
        Useful for understanding model behavior and debugging.
        """
        try:
            data = request.json
            
            comparisons = {}
            
            # Get original model prediction
            enhanced_model.use_ensemble = 'never'
            original_result = enhanced_model.predict(data)
            comparisons['original'] = {
                'prediction': original_result['prediction'],
                'confidence': original_result['confidence'],
                'probabilities': original_result.get('probabilities', {})
            }
            
            # Get voting ensemble prediction if available
            if enhanced_model.voting_ensemble:
                enhanced_model.use_ensemble = 'voting'
                voting_result = enhanced_model.predict(data)
                comparisons['voting_ensemble'] = {
                    'prediction': voting_result['prediction'],
                    'confidence': voting_result['confidence'],
                    'probabilities': voting_result.get('probabilities', {}),
                    'individual_models': voting_result.get('individual_predictions', {})
                }
            
            # Get stacking ensemble prediction if available
            if enhanced_model.stacking_ensemble:
                enhanced_model.use_ensemble = 'stacking'
                stacking_result = enhanced_model.predict(data)
                comparisons['stacking_ensemble'] = {
                    'prediction': stacking_result['prediction'],
                    'confidence': stacking_result['confidence'],
                    'probabilities': stacking_result.get('probabilities', {})
                }
            
            # Reset to auto mode
            enhanced_model.use_ensemble = 'auto'
            
            # Calculate agreement
            predictions = [comp['prediction'] for comp in comparisons.values()]
            agreement = len(set(predictions)) == 1
            
            return jsonify({
                'comparisons': comparisons,
                'agreement': agreement,
                'models_compared': list(comparisons.keys())
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app


# Example of integrating with existing api.py
def integrate_enhanced_endpoints():
    """
    Example of how to integrate enhanced endpoints into existing API.
    
    In api.py, add:
    from api.enhanced_api import add_enhanced_routes
    add_enhanced_routes(app)
    """
    pass


if __name__ == "__main__":
    # Standalone test server
    app = Flask(__name__)
    app = add_enhanced_routes(app)
    
    print("Enhanced API endpoints available:")
    print("  POST /api/enhanced/screen - Enhanced single screening")
    print("  POST /api/enhanced/batch-screen - Batch screening")
    print("  GET  /api/enhanced/model-info - Model information")
    print("  POST /api/enhanced/configure - Configure model settings")
    print("  POST /api/enhanced/compare-models - Compare model predictions")
    
    app.run(debug=True, port=5001)