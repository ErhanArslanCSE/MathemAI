# MathemAI Notebooks

This directory contains Jupyter notebooks for data analysis, model development, and visualization related to the MathemAI project. These notebooks provide interactive explorations of dyscalculia and math learning difficulties, serving as both development tools and educational resources.

## Overview

The notebooks in this directory are designed for:

1. **Data Exploration**: Analyzing assessment data to understand patterns in math learning difficulties
2. **Model Development**: Prototyping and testing machine learning models before implementation
3. **Visualization**: Creating interactive visualizations of math learning patterns
4. **Research**: Supporting ongoing research into dyscalculia and math learning difficulties
5. **Documentation**: Providing detailed explanations of algorithms and methodologies

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow (optional)

### Installation

1. Install Jupyter if you haven't already:
   ```bash
   pip install jupyter
   ```

2. Install required dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Notebook Descriptions

### Data Analysis

- **`dyscalculia_assessment_analysis.ipynb`**: Exploratory data analysis of dyscalculia assessment data
- **`error_pattern_analysis.ipynb`**: Analysis of common error patterns in mathematical problem-solving

### Model Development

- **`screening_model_development.ipynb`**: Development and evaluation of the dyscalculia screening model
- **`intervention_recommender_development.ipynb`**: Building and testing the intervention recommendation system

### Visualization

- **`learning_progress_visualization.ipynb`**: Interactive visualizations of student learning progress
- **`intervention_effectiveness_visualization.ipynb`**: Analysis of intervention effectiveness across different profiles

## Contributing

When contributing new notebooks or improving existing ones, please follow these guidelines:

1. **Organization**: Include clear section headings and a table of contents
2. **Documentation**: Document your code with comments and markdown explanations
3. **Reproducibility**: Ensure all results are reproducible with the provided data
4. **Dependencies**: List any additional dependencies not in the main requirements.txt
5. **Output**: Include output in the notebook for key cells to make it viewable on GitHub

## Best Practices

1. **Data Privacy**: Never include real student data in notebooks. Use anonymized or synthetic data.
2. **Version Control**: Clear outputs before committing to avoid large Git diffs.
3. **Long-running Cells**: Indicate expected runtime for cells that take more than a few seconds.
4. **Memory Usage**: Include memory optimization techniques for large datasets.
5. **References**: Cite relevant research papers or resources when applicable.

## Future Work

We plan to expand this collection with notebooks covering:

- Integration with neuroimaging data
- Natural language processing for analyzing math explanations
- Computer vision for analyzing handwritten math work
- Reinforcement learning for adaptive intervention strategies

## Resources

- [Jupyter Notebook Documentation](https://jupyter.org/documentation)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Dyscalculia Research Resources](https://www.understood.org/en/articles/what-is-dyscalculia)