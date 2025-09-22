# Tests

This directory contains unit tests and integration tests for the DUAL framework.

## Running Tests

Install development dependencies:
```bash
pip install -e .[dev]
```

Run all tests:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=dual --cov-report=html
```

Run specific test modules:
```bash
pytest tests/test_models.py
pytest tests/test_data.py
pytest tests/test_utils.py
```

## Test Structure

```
tests/
├── test_models.py      # Model architecture tests
├── test_data.py        # Data loading and preprocessing tests  
├── test_utils.py       # Utility function tests
├── test_training.py    # Training pipeline tests
├── test_evaluation.py  # Evaluation pipeline tests
├── conftest.py         # Shared test fixtures
└── fixtures/           # Test data and mock objects
```

## Test Guidelines

- Use pytest for all tests
- Include both unit tests and integration tests
- Mock external dependencies (datasets, models) when appropriate
- Ensure tests run quickly (< 1 second each for unit tests)
- Add tests for new features and bug fixes
- Maintain >90% test coverage