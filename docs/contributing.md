# Contributing to DUAL

We welcome contributions to the DUAL framework! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/DUAL.git
   cd DUAL
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 and use the following tools:

- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking

Format your code before submitting:
```bash
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/
```

### Code Structure

- **src/dual/**: Main package code
- **tests/**: Unit tests and integration tests
- **scripts/**: Training and evaluation scripts
- **configs/**: Configuration files
- **docs/**: Documentation

### Testing

- Write tests for new features and bug fixes
- Run tests before submitting: `pytest tests/`
- Maintain test coverage >90%: `pytest tests/ --cov=dual`
- Use meaningful test names and docstrings

### Documentation

- Update docstrings for new/modified functions
- Add type hints to all functions
- Update README.md if adding major features
- Add examples for new functionality

## 🔧 Types of Contributions

### Bug Reports

When filing bug reports, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Python version, dependencies, OS)
- **Error messages** and stack traces

### Feature Requests

For feature requests, please provide:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Proposed implementation** (if you have ideas)
- **Compatibility considerations**

### Code Contributions

#### Areas for Contribution

- **New backbone architectures** (e.g., Vision Transformers, EfficientNets)
- **Additional datasets** (e.g., Meta-Dataset, ORBIT)
- **Evaluation metrics** (e.g., confidence intervals, statistical tests)
- **Visualization tools** (e.g., t-SNE plots, attention maps)
- **Performance optimizations**
- **Documentation improvements**

#### Pull Request Process

1. **Create an issue** first to discuss major changes
2. **Write tests** for your changes
3. **Update documentation** as needed
4. **Follow code style** guidelines
5. **Write clear commit messages**
6. **Submit pull request** with description

#### Commit Message Format

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(models): add Vision Transformer backbone
fix(data): resolve miniImageNet loading issue  
docs(readme): update installation instructions
test(models): add unit tests for DUAL model
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=dual --cov-report=html

# Run integration tests only
pytest tests/ -m integration
```

### Writing Tests

Example test structure:
```python
import pytest
import torch
from dual.models import DualModel

class TestDualModel:
    def test_init(self):
        """Test model initialization."""
        model = DualModel(backbone='resnet12', n_way=5)
        assert model.backbone == 'resnet12'
        assert model.n_way == 5
    
    def test_forward(self):
        """Test forward pass."""
        model = DualModel(backbone='resnet12', n_way=5)
        # Test with dummy data
        support_x = torch.randn(5, 3, 84, 84)
        query_x = torch.randn(75, 3, 84, 84)
        logits = model(support_x, None, query_x)
        assert logits.shape == (75, 5)
    
    @pytest.mark.integration
    def test_training_step(self):
        """Integration test for training step."""
        # Test complete training step
        pass
```

## 📚 Documentation Style

### Docstring Format

Use Google-style docstrings:

```python
def align_features(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Align features using the DUAL alignment mechanism.
    
    Args:
        features: Input feature tensor of shape (N, D)
        targets: Target labels of shape (N,)
        
    Returns:
        Aligned feature tensor of shape (N, D)
        
    Raises:
        ValueError: If features and targets have incompatible shapes
        
    Example:
        >>> features = torch.randn(100, 512)
        >>> targets = torch.randint(0, 5, (100,))
        >>> aligned = align_features(features, targets)
        >>> print(aligned.shape)
        torch.Size([100, 512])
    """
    pass
```

## 🎯 Priority Areas

Current priority areas for contributions:

1. **Model improvements**: New architectures, attention mechanisms
2. **Dataset support**: Additional few-shot learning datasets
3. **Evaluation tools**: Better metrics, visualization, analysis
4. **Performance**: Speed optimizations, memory efficiency
5. **Robustness**: Better handling of edge cases, error messages

## 💬 Community

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, announcements
- **Code Reviews**: All PRs are reviewed by maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Acknowledged in paper acknowledgments (for research contributions)

Thank you for contributing to DUAL! 🚀