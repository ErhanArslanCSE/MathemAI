import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dyscalculia_screening_model import DyscalculiaScreeningModel
from models.intervention_recommender import InterventionRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def train_screening_model(optimize=True):
    """
    Train the dyscalculia screening model.
    
    Parameters:
    -----------
    optimize : bool
        Whether to perform hyperparameter optimization
        
    Returns:
    --------
    dict
        Dictionary containing training results
    """
    try:
        logger.info("Initializing dyscalculia screening model training")
        
        # Initialize the model
        model = DyscalculiaScreeningModel()
        
        # Load data
        data = model.load_data()
        if data is None:
            logger.error("Failed to load data for screening model")
            return {
                'success': False,
                'model': 'screening',
                'error': 'Failed to load data'
            }
        
        # Preprocess data
        X_train, X_test, y_train, y_test = model.preprocess_data(data)
        
        # Train the model
        logger.info(f"Training screening model with optimization={optimize}")
        model.train(X_train, y_train, optimize=optimize)
        
        # Evaluate the model
        evaluation = model.evaluate(X_test, y_test)
        
        # Save the model
        model.save_model()
        
        logger.info(f"Screening model training completed with accuracy: {evaluation['accuracy']:.4f}")
        
        return {
            'success': True,
            'model': 'screening',
            'accuracy': float(evaluation['accuracy']),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error training screening model: {e}", exc_info=True)
        return {
            'success': False,
            'model': 'screening',
            'error': str(e)
        }

def train_intervention_recommender(optimize_clusters=True):
    """
    Train the intervention recommender model.
    
    Parameters:
    -----------
    optimize_clusters : bool
        Whether to find the optimal number of clusters
        
    Returns:
    --------
    dict
        Dictionary containing training results
    """
    try:
        logger.info("Initializing intervention recommender training")
        
        # Initialize the recommender
        recommender = InterventionRecommender()
        
        # Load data
        assessment_data, intervention_data = recommender.load_data()
        if assessment_data is None or intervention_data is None:
            logger.error("Failed to load data for intervention recommender")
            return {
                'success': False,
                'model': 'recommender',
                'error': 'Failed to load data'
            }
        
        # Preprocess data
        merged_data = recommender.preprocess_data(assessment_data, intervention_data)
        
        # Train the recommender
        logger.info(f"Training intervention recommender with optimize_clusters={optimize_clusters}")
        recommender.train(merged_data, optimize_clusters=optimize_clusters)
        
        # Save the model
        recommender.save_model()
        
        logger.info(f"Intervention recommender training completed with {recommender.n_clusters} clusters")
        
        return {
            'success': True,
            'model': 'recommender',
            'n_clusters': recommender.n_clusters,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error training intervention recommender: {e}", exc_info=True)
        return {
            'success': False,
            'model': 'recommender',
            'error': str(e)
        }

def train_all_models(optimize=True):
    """
    Train all models.
    
    Parameters:
    -----------
    optimize : bool
        Whether to perform optimization during training
        
    Returns:
    --------
    dict
        Dictionary containing training results for all models
    """
    logger.info("Starting training for all models")
    
    # Create directories if they don't exist
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
    # Train screening model
    screening_result = train_screening_model(optimize=optimize)
    
    # Train intervention recommender
    recommender_result = train_intervention_recommender(optimize_clusters=optimize)
    
    # Combine results
    result = {
        'success': screening_result['success'] and recommender_result['success'],
        'timestamp': datetime.datetime.now().isoformat(),
        'models': {
            'screening': screening_result,
            'recommender': recommender_result
        }
    }
    
    logger.info(f"All model training complete. Overall success: {result['success']}")
    
    return result

def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description='Train models for the MathemAI project')
    parser.add_argument('--no-optimize', action='store_true', help='Disable hyperparameter optimization')
    parser.add_argument('--model', type=str, choices=['all', 'screening', 'recommender'], 
                       default='all', help='Which model to train')
    args = parser.parse_args()
    
    optimize = not args.no_optimize
    
    try:
        if args.model == 'all':
            result = train_all_models(optimize=optimize)
        elif args.model == 'screening':
            result = train_screening_model(optimize=optimize)
        elif args.model == 'recommender':
            result = train_intervention_recommender(optimize_clusters=optimize)
        
        if result['success']:
            print("Training completed successfully!")
        else:
            print(f"Training failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()