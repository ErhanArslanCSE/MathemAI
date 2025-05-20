import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class InterventionRecommender:
    def __init__(self, n_clusters=4):
        """
        Initialize the Intervention Recommender model.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for the KMeans algorithm
        """
        self.n_clusters = n_clusters
        
        # Define feature groups
        self.numeric_features = [
            'number_recognition', 'number_comparison', 'counting_skills', 
            'place_value', 'calculation_accuracy', 'calculation_fluency',
            'arithmetic_facts_recall', 'word_problem_solving', 
            'working_memory_score', 'visual_spatial_score'
        ]
        
        self.categorical_features = ['math_anxiety_level', 'attention_score']
        
        # Initialize preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        
        # Initialize clustering model
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('cluster', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
        ])
        
        # Dictionary to store intervention effectiveness for each cluster
        self.cluster_interventions = {}
        self.model_path = '../models/intervention_recommender.pkl'
        
    def load_data(self, assessment_path='../datasets/dyscalculia_assessment_data.csv',
                 intervention_path='../datasets/intervention_tracking_data.csv'):
        """
        Load assessment and intervention data.
        
        Parameters:
        -----------
        assessment_path : str
            Path to the assessment data file
        intervention_path : str
            Path to the intervention data file
            
        Returns:
        --------
        tuple
            assessment_data, intervention_data as pandas DataFrames
        """
        try:
            assessment_data = pd.read_csv(assessment_path)
            intervention_data = pd.read_csv(intervention_path)
            
            print(f"Loaded assessment data with {assessment_data.shape[0]} rows")
            print(f"Loaded intervention data with {intervention_data.shape[0]} rows")
            
            return assessment_data, intervention_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def preprocess_data(self, assessment_data, intervention_data):
        """
        Preprocess and merge the assessment and intervention data.
        
        Parameters:
        -----------
        assessment_data : pd.DataFrame
            Assessment data
        intervention_data : pd.DataFrame
            Intervention data
            
        Returns:
        --------
        pd.DataFrame
            Merged and preprocessed data
        """
        # Convert categorical variables
        anxiety_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
        attention_map = {'normal': 2, 'low': 1, 'very_low': 0}
        
        if 'math_anxiety_level' in assessment_data.columns:
            assessment_data['math_anxiety_level'] = assessment_data['math_anxiety_level'].map(anxiety_map)
        
        if 'attention_score' in assessment_data.columns:
            assessment_data['attention_score'] = assessment_data['attention_score'].map(attention_map)
        
        # Calculate improvement metrics from intervention data
        intervention_summary = intervention_data.groupby(['student_id', 'intervention_type'])[
            'post_assessment_score', 'pre_assessment_score'
        ].agg({
            'post_assessment_score': 'max',
            'pre_assessment_score': 'min'
        }).reset_index()
        
        intervention_summary['improvement'] = (
            intervention_summary['post_assessment_score'] - 
            intervention_summary['pre_assessment_score']
        )
        
        # Get most effective intervention for each student
        best_interventions = intervention_summary.loc[
            intervention_summary.groupby('student_id')['improvement'].idxmax()
        ]
        
        # Merge with assessment data
        merged_data = assessment_data.merge(
            best_interventions[['student_id', 'intervention_type', 'improvement']],
            left_on='student_id',
            right_on='student_id',
            how='left'
        )
        
        # Handle missing values for students without interventions
        merged_data['intervention_type'] = merged_data['intervention_type'].fillna('none')
        merged_data['improvement'] = merged_data['improvement'].fillna(0)
        
        return merged_data
    
    def find_optimal_clusters(self, features, max_clusters=10):
        """
        Find the optimal number of clusters using silhouette score.
        
        Parameters:
        -----------
        features : array-like
            Feature matrix
        max_clusters : int
            Maximum number of clusters to try
            
        Returns:
        --------
        int
            Optimal number of clusters
        """
        silhouette_scores = []
        
        # Try different numbers of clusters
        for n in range(2, max_clusters + 1):
            # Create and fit KMeans
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {n}, the silhouette score is {silhouette_avg:.4f}")
        
        # Find optimal number of clusters
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method for Optimal Clusters')
        plt.axvline(x=optimal_clusters, color='r', linestyle='--')
        
        # Save the plot
        os.makedirs('../docs/figures', exist_ok=True)
        plt.savefig('../docs/figures/optimal_clusters.png')
        plt.close()
        
        return optimal_clusters
    
    def train(self, data, optimize_clusters=True, max_clusters=8):
        """
        Train the model to find student clusters and effective interventions.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed and merged data
        optimize_clusters : bool
            Whether to find optimal number of clusters
        max_clusters : int
            Maximum number of clusters to try if optimizing
            
        Returns:
        --------
        self
            Trained model instance
        """
        # Select features for clustering
        features = data[self.numeric_features + self.categorical_features].copy()
        
        # Find optimal number of clusters if requested
        if optimize_clusters:
            print("Finding optimal number of clusters...")
            # Transform features for silhouette analysis
            transformed_features = self.preprocessor.fit_transform(features)
            optimal_n = self.find_optimal_clusters(transformed_features, max_clusters)
            print(f"Optimal number of clusters: {optimal_n}")
            
            # Update model with optimal clusters
            self.n_clusters = optimal_n
            self.model.named_steps['cluster'] = KMeans(
                n_clusters=optimal_n, random_state=42, n_init=10
            )
        
        # Fit the model
        print("Training clustering model...")
        self.model.fit(features)
        
        # Assign clusters to data
        data['cluster'] = self.model.predict(features)
        
        # Analyze intervention effectiveness by cluster
        print("\nAnalyzing intervention effectiveness by cluster:")
        for cluster in range(self.n_clusters):
            cluster_data = data[data['cluster'] == cluster]
            
            # Skip if no students in cluster
            if len(cluster_data) == 0:
                print(f"Cluster {cluster}: No students")
                continue
            
            # Calculate mean improvement by intervention type
            intervention_effectiveness = cluster_data.groupby('intervention_type')['improvement'].agg([
                'mean', 'count'
            ]).sort_values('mean', ascending=False)
            
            print(f"\nCluster {cluster} ({len(cluster_data)} students):")
            print(intervention_effectiveness)
            
            # Store recommended interventions for this cluster
            # Only include interventions with at least 2 students
            effective_interventions = intervention_effectiveness[
                intervention_effectiveness['count'] >= 2
            ]
            
            if not effective_interventions.empty:
                self.cluster_interventions[cluster] = effective_interventions.index.tolist()
            else:
                # If no interventions have enough students, use all interventions
                self.cluster_interventions[cluster] = intervention_effectiveness.index.tolist()
        
        # Visualize clusters
        self._visualize_clusters(data)
        
        return self
    
    def _visualize_clusters(self, data):
        """
        Create visualizations of the clusters.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with cluster assignments
        """
        # Create directory for figures
        os.makedirs('../docs/figures', exist_ok=True)
        
        # PCA for dimensionality reduction is not used here as it would
        # require additional code to interpret in the context of the original features
        
        # Create a cluster profile visualization
        plt.figure(figsize=(14, 8))
        
        # Calculate mean of numeric features by cluster
        cluster_profiles = data.groupby('cluster')[self.numeric_features].mean()
        
        # Create a heatmap
        sns.heatmap(cluster_profiles, cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': 'Mean Value'})
        plt.title('Cluster Profiles: Mean Values of Numeric Features')
        plt.tight_layout()
        plt.savefig('../docs/figures/cluster_profiles.png')
        plt.close()
        
        # Visualize intervention effectiveness by cluster
        plt.figure(figsize=(12, 8))
        
        # Prepare data for intervention effectiveness visualization
        effectiveness_data = []
        
        for cluster, interventions in self.cluster_interventions.items():
            cluster_data = data[data['cluster'] == cluster]
            
            for intervention in interventions:
                intervention_data = cluster_data[cluster_data['intervention_type'] == intervention]
                
                if not intervention_data.empty:
                    mean_improvement = intervention_data['improvement'].mean()
                    
                    effectiveness_data.append({
                        'cluster': f'Cluster {cluster}',
                        'intervention': intervention,
                        'mean_improvement': mean_improvement,
                        'count': len(intervention_data)
                    })
        
        # Convert to DataFrame
        effectiveness_df = pd.DataFrame(effectiveness_data)
        
        if not effectiveness_df.empty:
            # Create a bubble chart
            plt.figure(figsize=(12, 8))
            
            # Define a color palette
            palette = sns.color_palette("husl", len(effectiveness_df['intervention'].unique()))
            
            # Create a dictionary mapping interventions to colors
            color_dict = dict(zip(effectiveness_df['intervention'].unique(), palette))
            
            # Assign colors based on intervention type
            effectiveness_df['color'] = effectiveness_df['intervention'].map(color_dict)
            
            # Create the scatter plot
            for i, row in effectiveness_df.iterrows():
                plt.scatter(
                    row['cluster'], 
                    row['mean_improvement'], 
                    s=row['count'] * 100,  # Size based on count
                    color=row['color'],
                    alpha=0.7,
                    edgecolors='black'
                )
                
                # Add text label
                plt.text(
                    row['cluster'],
                    row['mean_improvement'],
                    row['intervention'],
                    ha='center',
                    va='center',
                    fontsize=8
                )
            
            plt.xlabel('Cluster')
            plt.ylabel('Mean Improvement')
            plt.title('Intervention Effectiveness by Cluster')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Create a custom legend for intervention types
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, label=intervention, markersize=8)
                for intervention, color in color_dict.items()
            ]
            
            plt.legend(handles=legend_elements, title='Intervention Type', 
                     loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.savefig('../docs/figures/intervention_effectiveness.png')
            plt.close()
    
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
        
        # Save the model and intervention mappings
        model_data = {
            'model': self.model,
            'cluster_interventions': self.cluster_interventions,
            'n_clusters': self.n_clusters,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(model_data, path)
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
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.cluster_interventions = model_data['cluster_interventions']
            self.n_clusters = model_data['n_clusters']
            self.numeric_features = model_data['numeric_features']
            self.categorical_features = model_data['categorical_features']
            
            print(f"Model loaded from {path}")
            return self
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def recommend_intervention(self, student_data):
        """
        Recommend interventions for a student based on their assessment data.
        
        Parameters:
        -----------
        student_data : pd.DataFrame or dict
            Student assessment data
            
        Returns:
        --------
        dict
            Dictionary containing recommended interventions and rationale
        """
        # Convert to DataFrame if dictionary
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])
        
        # Ensure all features are present
        required_features = self.numeric_features + self.categorical_features
        missing_features = set(required_features) - set(student_data.columns)
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                student_data[feature] = 0
        
        # Predict cluster
        cluster = self.model.predict(student_data[required_features])[0]
        
        # Get recommended interventions for this cluster
        recommended_interventions = self.cluster_interventions.get(cluster, [])
        
        # Create result dictionary
        result = {
            'cluster': int(cluster),
            'recommended_interventions': recommended_interventions,
            'description': self._get_cluster_description(cluster, student_data)
        }
        
        return result
    
    def _get_cluster_description(self, cluster, student_data):
        """
        Generate a description of the cluster and why the interventions are recommended.
        
        Parameters:
        -----------
        cluster : int
            Cluster number
        student_data : pd.DataFrame
            Student data
            
        Returns:
        --------
        str
            Description of the cluster and intervention recommendations
        """
        # This is a simple description. In a real application, you would analyze
        # the cluster characteristics more thoroughly to provide better explanations.
        
        # Get the top interventions for this cluster
        top_interventions = self.cluster_interventions.get(cluster, [])[:2]
        
        if not top_interventions:
            return "Unable to provide intervention recommendations for this profile."
        
        # Simple description templates
        descriptions = {
            'multisensory_approach': (
                "Multisensory approaches use tactile and visual methods to help students "
                "understand mathematical concepts through multiple senses."
            ),
            'visual_aids': (
                "Visual aids like number lines and manipulatives help students visualize "
                "mathematical relationships and build stronger conceptual understanding."
            ),
            'game_based_learning': (
                "Game-based learning increases engagement and reduces math anxiety, "
                "making practice more enjoyable and effective."
            ),
            'structured_sequence': (
                "A highly structured, sequential approach breaks down concepts into "
                "manageable steps for students who need clear, consistent progression."
            ),
            'technology_assisted': (
                "Technology-assisted learning provides interactive, adaptive practice "
                "with immediate feedback and engages students through digital tools."
            )
        }
        
        # Generate description
        description = f"Based on the assessment profile, this student shows patterns similar to other students in Cluster {cluster}. "
        
        # Add intervention recommendations
        description += "The recommended interventions are:\n\n"
        
        for intervention in top_interventions:
            if intervention in descriptions:
                description += f"- {intervention.replace('_', ' ').title()}: {descriptions[intervention]}\n\n"
            else:
                description += f"- {intervention.replace('_', ' ').title()}\n\n"
        
        return description


if __name__ == "__main__":
    # Initialize the recommender
    recommender = InterventionRecommender()
    
    # Load data
    assessment_data, intervention_data = recommender.load_data()
    
    if assessment_data is not None and intervention_data is not None:
        # Preprocess data
        merged_data = recommender.preprocess_data(assessment_data, intervention_data)
        
        # Train the model
        recommender.train(merged_data, optimize_clusters=True)
        
        # Save the model
        recommender.save_model()
        
        # Example recommendation
        sample = merged_data.iloc[0][recommender.numeric_features + recommender.categorical_features].to_dict()
        recommendation = recommender.recommend_intervention(sample)
        print(f"\nSample recommendation: {recommendation}")
    else:
        print("Unable to proceed due to data loading error.")