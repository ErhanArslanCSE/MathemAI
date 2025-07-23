# Development Guide

This guide provides instructions for setting up your development environment and working with the MathemAI codebase.

## Setting Up Your Development Environment

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Git
- pip (Python package manager)
- A code editor of your choice (VS Code, PyCharm, etc.)

### Step 1: Clone the Repository

```bash
git clone https://github.com/openimpactai/OpenImpactAI.git
cd OpenImpactAI/AI-Education-Projects/MathemAI
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
# Using venv (built into Python 3)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Sample Datasets

For development and testing purposes, generate sample datasets:

```bash
python scripts/generate_datasets.py
```

### Step 5: Run the Tests

Ensure everything is working correctly by running the tests:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=.
```

## Project Structure

Here's a breakdown of the key components of the MathemAI project:

### Models

Located in the `models/` directory:

- `dyscalculia_screening_model.py`: Handles the classification of math learning difficulties
- `intervention_recommender.py`: Provides personalized intervention recommendations

### Data

The `datasets/` directory contains:

- `dyscalculia_assessment_data.csv`: Assessment data for screening
- `error_analysis_data.csv`: Detailed error patterns made by students
- `intervention_tracking_data.csv`: Intervention effectiveness data

### API

The API is implemented in `api/app.py` and provides endpoints for:

- `/api/screen`: Screening for dyscalculia and math difficulties
- `/api/recommend`: Getting personalized intervention recommendations
- `/api/save-assessment`: Saving new assessment data
- `/api/save-intervention`: Tracking intervention outcomes
- `/api/error-patterns`: Recording specific error patterns

### Scripts

Utility scripts in the `scripts/` directory:

- `generate_datasets.py`: Creates simulated datasets for development
- `train_models.py`: Trains the machine learning models

## Development Workflow

### 1. Creating a New Feature or Fixing a Bug

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. Make your changes, following the coding standards (see below)

3. Write or update tests for your changes

4. Run the tests to ensure everything works:
   ```bash
   pytest
   ```

5. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

### 2. Submitting a Pull Request

1. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to the repository on GitHub and create a pull request

3. Fill in the pull request template with details about your changes

4. Wait for code review and address any feedback

### 3. Code Review Process

- All pull requests require at least one review before merging
- Reviewers will check for:
  - Code quality and adherence to standards
  - Test coverage
  - Documentation
  - Alignment with project goals

## Coding Standards

### Python Code Style

We follow PEP 8 guidelines for Python code:

- Use 4 spaces for indentation (not tabs)
- Maximum line length of 79 characters
- Use clear, descriptive variable and function names
- Include docstrings for all functions, classes, and modules

Example function with proper docstring:

```python
def calculate_score(assessment_data):
    """
    Calculate a numerical score based on assessment data.
    
    Parameters:
    -----------
    assessment_data : dict
        Dictionary containing assessment metrics
        
    Returns:
    --------
    float
        Normalized score between 0 and 1
    """
    # Implementation here
    return score
```

### Testing

- Write tests for all new functionality
- Use pytest for testing
- Aim for at least 80% code coverage

Example test:

```python
def test_calculate_score():
    # Test with normal data
    data = {
        'number_recognition': 3,
        'calculation_accuracy': 4
    }
    score = calculate_score(data)
    assert 0 <= score <= 1
    
    # Test with edge cases
    empty_data = {}
    score = calculate_score(empty_data)
    assert score == 0
```

## Working with the Data

### Data Format

Assessment data includes:

- Numerical scores (1-5) for various math skills
- Categorical variables for factors like anxiety level
- Timestamps for tracking progress

### Adding New Features to Models

When adding new features to the models:

1. Update the relevant model class in the `models/` directory
2. Modify the preprocessing steps if needed
3. Update the API to expose the new functionality
4. Add tests for the new features

### Modifying the API

When modifying the API:

1. Update the endpoint in `api/app.py`
2. Add appropriate error handling
3. Include input validation
4. Update the API documentation

## Common Development Tasks

### Training Models with New Data

```bash
python scripts/train_models.py --model all
```

### Running the API Locally

```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000`

### Adding a New Model

1. Create a new file in the `models/` directory
2. Implement the model class with appropriate methods
3. Add a training function in `scripts/train_models.py`
4. Create tests for the new model
5. Update the API to expose the model's functionality

## Troubleshooting

### Common Issues

- **Missing dependencies**: Ensure you've installed all requirements with `pip install -r requirements.txt`
- **File not found errors**: Check that you've generated the datasets with `scripts/generate_datasets.py`
- **Model loading errors**: Make sure you've trained the models with `scripts/train_models.py`

### Getting Help

If you encounter issues:

1. Check the existing issues on GitHub to see if someone else has encountered the same problem
2. Join our community discussion forum
3. Ask for help in our Discord channel
4. Open a new issue with detailed information about the problem

## Documentation Guidelines

When adding or updating documentation:

- Use clear, concise language
- Include code examples where appropriate
- Update both inline comments and external documentation
- Follow Markdown formatting conventions

## Resources for Developers

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)