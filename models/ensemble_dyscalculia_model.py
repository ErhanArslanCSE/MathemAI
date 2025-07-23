import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class EnsembleDyscalculiaModel:
    def __init__(self, ensemble_type='voting'):
        """
        Initialize the Enhanced Dyscalculia Screening Model with Ensemble Methods.
        
        Parameters:
        -----------
        ensemble_type : str
            Type of ensemble to use: 'voting', 'stacking', or 'both'
        """
        self.ensemble_type = ensemble_type
        self.features = [
            'number_recognition', 'number_comparison', 'counting_skills', 
            'place_value', 'calculation_accuracy', 'calculation_fluency',
            'arithmetic_facts_recall', 'word_problem_solving', 
            'working_memory_score', 'visual_spatial_score'
        ]
        self.target = 'diagnosis'
        self.model_path = '../models/ensemble_dyscalculia_model.pkl'
        
        # Initialize base models
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svc': SVC(probability=True, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Initialize ensemble models
        self._initialize_ensemble()
        
    def _initialize_ensemble(self):
        """Initialize the ensemble model based on the specified type."""
        if self.ensemble_type == 'voting':
            # Soft voting classifier for better probability estimates
            self.ensemble = VotingClassifier(
                estimators=list(self.base_models.items())[:3],  # Use RF, GB, and SVC
                voting='soft',
                weights=[2, 1.5, 1]  # Give more weight to RF and GB
            )
        
        elif self.ensemble_type == 'stacking':
            # Stacking classifier with LogisticRegression as meta-learner
            self.ensemble = StackingClassifier(
                estimators=list(self.base_models.items())[:3],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5  # 5-fold cross-validation for training meta-learner
            )
        
        elif self.ensemble_type == 'both':
            # Create both ensembles and use another voting classifier to combine them
            voting_ensemble = VotingClassifier(
                estimators=list(self.base_models.items())[:3],
                voting='soft',
                weights=[2, 1.5, 1]
            )
            
            stacking_ensemble = StackingClassifier(
                estimators=list(self.base_models.items())[:3],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            )
            
            self.ensemble = VotingClassifier(
                estimators=[('voting', voting_ensemble), ('stacking', stacking_ensemble)],
                voting='soft'
            )
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ensemble', self.ensemble)
        ])
    
    def load_data(self, file_path='../datasets/dyscalculia_assessment_data.csv'):
        """Load the assessment data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """Preprocess the data for model training."""
        # Convert categorical variables
        anxiety_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
        if 'math_anxiety_level' in data.columns:
            data['math_anxiety_level'] = data['math_anxiety_level'].map(anxiety_map)
        
        attention_map = {'normal': 2, 'low': 1, 'very_low': 0}
        if 'attention_score' in data.columns:
            data['attention_score'] = data['attention_score'].map(attention_map)
        
        # Select features and target
        X = data[self.features]
        y = data[self.target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, optimize=True):
        """Train the ensemble model with optional hyperparameter tuning."""
        print(f"Training {self.ensemble_type} ensemble model...")
        
        if optimize and self.ensemble_type == 'voting':
            # Optimize individual base models first
            print("Optimizing base models...")
            
            param_grids = {
                'rf': {
                    'ensemble__voting__rf__n_estimators': [100, 200],
                    'ensemble__voting__rf__max_depth': [10, 20, None],
                    'ensemble__voting__rf__min_samples_split': [2, 5]
                },
                'gb': {
                    'ensemble__voting__gb__n_estimators': [100, 200],
                    'ensemble__voting__gb__learning_rate': [0.05, 0.1, 0.15],
                    'ensemble__voting__gb__max_depth': [3, 5, 7]
                }
            }
            
            # Use a simplified grid search for demonstration
            param_grid = {
                'ensemble__weights': [[1, 1, 1], [2, 1, 1], [2, 1.5, 1], [3, 2, 1]]
            }
            
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            self.pipeline = grid_search.best_estimator_
            
        else:
            # Train without optimization
            self.pipeline.fit(X_train, y_train)
        
        # Calculate cross-validation scores for individual models
        print("\nCross-validation scores for base models:")
        for name, model in list(self.base_models.items())[:3]:
            base_pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            scores = cross_val_score(base_pipeline, X_train, y_train, cv=5, scoring='accuracy')
            print(f"{name.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Calculate ensemble cross-validation score
        ensemble_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print(f"\nEnsemble: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std() * 2:.4f})")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate the ensemble model performance with enhanced metrics."""
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate AUC for multi-class (One-vs-Rest)
        from sklearn.preprocessing import label_binarize
        classes = sorted(y_test.unique())
        y_test_binary = label_binarize(y_test, classes=classes)
        
        if len(classes) > 2:
            auc_scores = []
            for i in range(len(classes)):
                auc = roc_auc_score(y_test_binary[:, i], y_proba[:, i])
                auc_scores.append(auc)
            avg_auc = np.mean(auc_scores)
        else:
            avg_auc = roc_auc_score(y_test, y_proba[:, 1])
        
        # Print evaluation results
        print(f"Ensemble Model ({self.ensemble_type}) Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average AUC: {avg_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {self.ensemble_type.capitalize()} Ensemble')
        
        os.makedirs('../docs/figures', exist_ok=True)
        plt.savefig(f'../docs/figures/ensemble_{self.ensemble_type}_confusion_matrix.png')
        plt.close()
        
        # Feature importance (averaged across models if possible)
        self._plot_feature_importance(X_test)
        
        # Prediction confidence distribution
        self._plot_confidence_distribution(y_proba, y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'avg_auc': avg_auc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def _plot_feature_importance(self, X_test):
        """Plot feature importance for ensemble models."""
        importances = []
        
        if self.ensemble_type == 'voting':
            # Get feature importance from tree-based models
            ensemble = self.pipeline.named_steps['ensemble']
            for name, model in ensemble.estimators:
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
        
        if importances:
            # Average the importances
            avg_importance = np.mean(importances, axis=0)
            
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': avg_importance
            }).sort_values('Importance', ascending=False)
            
            print("\nAverage Feature Importance:")
            print(feature_importance)
            
            # Visualize
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title(f'Feature Importance - {self.ensemble_type.capitalize()} Ensemble')
            plt.tight_layout()
            plt.savefig(f'../docs/figures/ensemble_{self.ensemble_type}_feature_importance.png')
            plt.close()
    
    def _plot_confidence_distribution(self, y_proba, y_true, y_pred):
        """Plot the distribution of prediction confidence scores."""
        # Get maximum probability for each prediction
        confidences = np.max(y_proba, axis=1)
        correct_predictions = y_true == y_pred
        
        plt.figure(figsize=(10, 6))
        
        # Plot confidence distribution for correct and incorrect predictions
        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct_predictions], bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(confidences[~correct_predictions], bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Prediction Accuracy')
        plt.legend()
        
        # Plot average confidence by class
        plt.subplot(1, 2, 2)
        class_confidences = []
        classes = sorted(np.unique(y_true))
        
        for cls in classes:
            mask = y_true == cls
            avg_conf = np.mean(confidences[mask])
            class_confidences.append(avg_conf)
        
        plt.bar(classes, class_confidences)
        plt.xlabel('Class')
        plt.ylabel('Average Confidence')
        plt.title('Average Confidence by Class')
        
        plt.tight_layout()
        plt.savefig(f'../docs/figures/ensemble_{self.ensemble_type}_confidence_distribution.png')
        plt.close()
    
    def save_model(self, path=None):
        """Save the trained ensemble model."""
        if path is None:
            path = self.model_path
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the entire pipeline
        joblib.dump(self.pipeline, path)
        
        # Save ensemble configuration
        config = {
            'ensemble_type': self.ensemble_type,
            'features': self.features,
            'target': self.target
        }
        
        config_path = path.replace('.pkl', '_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f)
        
        print(f"Ensemble model saved to {path}")
        print(f"Configuration saved to {config_path}")
    
    def load_model(self, path=None):
        """Load a trained ensemble model."""
        if path is None:
            path = self.model_path
            
        try:
            self.pipeline = joblib.load(path)
            
            # Load configuration
            config_path = path.replace('.pkl', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
                    self.ensemble_type = config.get('ensemble_type', self.ensemble_type)
                    self.features = config.get('features', self.features)
                    self.target = config.get('target', self.target)
            
            print(f"Ensemble model loaded from {path}")
            return self
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, data):
        """
        Make predictions with enhanced confidence metrics.
        
        Returns prediction with individual model predictions for transparency.
        """
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
        probabilities = self.pipeline.predict_proba(data[self.features])
        
        # Get individual model predictions if using voting
        individual_predictions = {}
        if self.ensemble_type == 'voting' and hasattr(self.pipeline.named_steps['ensemble'], 'estimators_'):
            scaled_data = self.pipeline.named_steps['scaler'].transform(data[self.features])
            for name, model in self.pipeline.named_steps['ensemble'].estimators_:
                ind_pred = model.predict(scaled_data)
                ind_prob = model.predict_proba(scaled_data)
                individual_predictions[name] = {
                    'prediction': ind_pred[0],
                    'confidence': float(np.max(ind_prob[0]))
                }
        
        # Create detailed results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': pred,
                'confidence': float(max(probabilities[i])),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.pipeline.classes_, probabilities[i])
                },
                'ensemble_type': self.ensemble_type
            }
            
            if individual_predictions:
                result['individual_predictions'] = individual_predictions
            
            # Add confidence level interpretation
            confidence = float(max(probabilities[i]))
            if confidence >= 0.8:
                result['confidence_level'] = 'high'
            elif confidence >= 0.6:
                result['confidence_level'] = 'medium'
            else:
                result['confidence_level'] = 'low'
            
            results.append(result)
        
        return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    # Test different ensemble types
    for ensemble_type in ['voting', 'stacking']:
        print(f"\n{'='*50}")
        print(f"Testing {ensemble_type.upper()} Ensemble")
        print(f"{'='*50}\n")
        
        # Initialize the model
        model = EnsembleDyscalculiaModel(ensemble_type=ensemble_type)
        
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
            
            # Example prediction with detailed output
            sample = X_test.iloc[0].to_dict()
            prediction = model.predict(sample)
            print(f"\nSample prediction: {prediction}")
        else:
            print("Unable to proceed due to data loading error.")