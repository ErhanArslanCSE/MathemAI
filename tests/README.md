# MathemAI Tests

This directory contains the test suite for the MathemAI project. These tests help ensure the reliability and correctness of the system as it evolves.

## Overview

The test suite covers the main components of the MathemAI project:

- Model tests (screening model and intervention recommender)
- API endpoint tests
- Data validation tests

## Test Files

```
tests/
├── conftest.py                 # Pytest configurations and fixtures
├── test_screening_model.py     # Tests for dyscalculia screening model
├── test_recommender.py         # Tests for intervention recommender
├── test_api.py                 # Tests for API endpoints
└── test_data.py                # Tests for data validation
```

This simple structure keeps all tests in a single directory without complex nesting, making it easier to navigate and maintain.

## Running Tests

To run the tests, make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Then, run the tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mathematai

# Run a specific test file
pytest tests/test_models/test_screening_model.py

# Run tests with verbose output
pytest -v
```

## Writing New Tests

When contributing to MathemAI, please add appropriate tests for your new features or bug fixes. Test files should follow these naming conventions:

- Test files should start with `test_`
- Test functions should also start with `test_`
- Group related tests in the appropriate subdirectory

Example test function:

```python
def test_screening_model_prediction():
    """Test that the screening model makes the expected predictions."""
    # Setup test data
    test_input = {...}
    expected_output = {...}
    
    # Run the function being tested
    actual_output = screening_model.predict(test_input)
    
    # Assert expected behavior
    assert actual_output == expected_output
```

## Test Coverage

We aim to maintain high test coverage for all components of MathemAI. When submitting changes, please ensure that your code is well-tested and that you haven't reduced the overall test coverage.

## Continuous Integration

Tests are automatically run on all pull requests through our CI pipeline. Pull requests with failing tests cannot be merged until the issues are resolved.

## Getting Help

If you need help with the test suite, please:
- Check the project documentation
- Ask questions in the community forum
- Open an issue with the "question" tag

Thank you for helping ensure MathemAI's quality through testing!
