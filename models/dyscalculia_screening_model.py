import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class DyscalculiaScreeningModel:
    def __init__(self):
        """Initialize the Dyscalculia Screening Model."""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        self.features = [
            'number_recognition', 'number_comparison', 'counting_skills', 
            'place_value', 'calculation_accuracy', 'calculation_fluency',
            'arithmetic_facts_recall', 'word_problem_solving', 
            'working_memory_score', 'visual_spatial_score'
        ]
        
        self.target = 'diagnosis'
        self.model_path = '../models/dyscalculia_screening_model.pkl'
        
    def load_data(self, file_path='../datasets/dyscalculia_assessment_data.csv'):
        """
        Load the assessment data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing assessment data
            
        Returns:
        --------
        pd.DataFrame
            Loaded data as a pandas DataFrame
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """
        Preprocess the data for model training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing the assessment data
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test split data
        """
        # Convert categorical variables if needed
        # Example: map math anxiety levels to numerical values
        anxiety_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
        if 'math_anxiety_level' in data.columns:
            data['math_anxiety_level'] = data['math_anxiety_level'].map(anxiety_map)
        
        # Map attention scores
        attention_map = {'normal': 2, 'low': 1, 'very_low': 0}
        if 'attention_score' in data.columns:
            data['attention_score'] = data['attention_score'].map(attention_map)
        
        # Select features and target
        X = data[self.features]
        y = data[self.target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, optimize=True):
        """
        Train the model with hyperparameter tuning if requested.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        optimize : bool
            Whether to perform hyperparameter optimization
            
        Returns:
        --------
        self
            Trained model instance
        """
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            self.pipeline = grid_search.best_estimator_
            
        else:
            print("Training model with default parameters...")
            self.pipeline.fit(X_train, y_train)
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing target
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Print evaluation results
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(y_test.unique()),
                   yticklabels=sorted(y_test.unique()))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix plot
        os.makedirs('../docs/figures', exist_ok=True)
        plt.savefig('../docs/figures/confusion_matrix.png')
        plt.close()
        
        # Feature importance
        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': self.pipeline.named_steps['classifier'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save the feature importance plot
            plt.savefig('../docs/figures/feature_importance.png')
            plt.close()
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, path=None):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        path : str, optional
            Path where the model will be saved
        """
        if path is None:
            path = self.model_path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        path : str, optional
            Path to the saved model
            
        Returns:
        --------
        self
            Model instance with loaded model
        """
        if path is None:
            path = self.model_path
            
        try:
            self.pipeline = joblib.load(path)
            print(f"Model loaded from {path}")
            return self
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, data):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        data : pd.DataFrame or dict
            Data to make predictions on
            
        Returns:
        --------
        dict
            Dictionary containing predictions and probabilities
        """
        # Convert to DataFrame if dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all features are present
        missing_features = set(self.features) - set(data.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                data[feature] = 0
        
        # Make predictions
        predictions = self.pipeline.predict(data[self.features])
        
        # Get probability scores
        probabilities = self.pipeline.predict_proba(data[self.features])
        
        # Create results dictionary
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': pred,
                'confidence': max(probabilities[i]),
                'probabilities': {
                    class_name: prob
                    for class_name, prob in zip(self.pipeline.classes_, probabilities[i])
                }
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    # Initialize the model
    model = DyscalculiaScreeningModel()
    
    # Load and preprocess data
    data = model.load_data()
    if data is not None:
        X_train, X_test, y_train, y_test = model.preprocess_data(data)
        
        # Train the model
        model.train(X_train, y_train, optimize=True)
        
        # Evaluate the model
        evaluation = model.evaluate(X_test, y_test)
        
        # Save the model
        model.save_model()
        
        # Example prediction
        sample = X_test.iloc[0].to_dict()
        prediction = model.predict(sample)
        print(f"\nSample prediction: {prediction}")
    else:
        print("Unable to proceed due to data loading error.")