# Test Suite for Agentic Academic Review System

This directory contains the test suite for the literature-grounded review system.

## Test Structure

```
tests/
├── __init__.py
├── README.md
├── test_literature_searcher.py    # Tests for Semantic Scholar integration
├── test_agent_librarian.py         # Tests for Librarian agent
├── test_agent_reader.py            # Tests for Reader agent
├── test_agent_fact_checker.py      # Tests for Fact-Checker agent
└── test_agent_critic.py            # Tests for Critic agent
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_literature_searcher.py
```

### Run Specific Test Class

```bash
pytest tests/test_literature_searcher.py::TestSearchMethods
```

### Run Specific Test

```bash
pytest tests/test_literature_searcher.py::TestSearchMethods::test_search_by_keywords
```

### Run with Coverage Report

```bash
pytest --cov=core --cov=agents --cov-report=html
```

### Run Quick Tests (Skip Slow/Integration Tests)

```bash
pytest -m "not slow and not integration"
```

### Run Literature-Grounded Tests Only

```bash
pytest -m literature
```

## Test Markers

Tests are marked by type:

- `unit`: Fast, isolated unit tests
- `integration`: Slower tests that may use external APIs
- `slow`: Tests that take longer to run
- `literature`: Tests requiring literature grounding features
- `standard`: Tests for standard (non-literature) functionality

## Test Files Descriptions

### test_literature_searcher.py

Tests the Semantic Scholar API integration:
- Search by keywords
- Search by abstract similarity
- Get most cited papers
- Fetch paper details and abstracts
- Error handling and rate limiting

### test_agent_librarian.py

Tests the Librarian agent (Stage 1):
- Keyword extraction from papers
- Key findings extraction from baseline papers
- Baseline summary generation
- Baseline reference creation

### test_agent_reader.py

Tests the Reader agent (Stage 2):
- Literature context building
- Evidence extraction without baseline (standard mode)
- Evidence extraction with baseline (literature-grounded mode)
- Novelty ranking and assessment
- Batch processing

### test_agent_fact_checker.py

Tests the Fact-Checker agent (Stage 3):
- Trigger detection (first-study claims, contradictions, implausible claims)
- Verification query generation
- Claim verification against literature
- Batch fact checking
- Result summarization

### test_agent_critic.py

Tests the Critic agent (Stage 4):
- Novelty-adjusted score calculation
- Research trajectory section generation
- Grounded review synthesis
- Backward compatibility with standard Review

## Writing New Tests

When adding new features, follow these patterns:

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Tests for new feature."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return Mock(spec=Config)

    def test_basic_functionality(self, mock_config):
        """Test basic functionality."""
        # Arrange
        # Act
        # Assert
        assert True

    @patch('module.call_llm')
    def test_with_external_api(self, mock_llm, mock_config):
        """Test with mocked external API."""
        mock_llm.return_value = {
            'success': True,
            'content': '{"result": "value"}',
            'cost': 0.01
        }
        # Test implementation...

    def test_error_handling(self, mock_config):
        """Test error handling."""
        # Test error conditions...
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=core --cov=agents --cov-report=xml
```

## Troubleshooting

### Tests Failing Due to Missing API Keys

Some integration tests may require API keys. Set these as environment variables:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

Or run only unit tests:

```bash
pytest -m "not integration"
```

### Tests Timing Out

Increase timeout in `pytest.ini` or run tests excluding slow ones:

```bash
pytest -m "not slow"
```

### Coverage Report Missing

Install pytest-cov:

```bash
pip install pytest-cov
```

## Contributing

When contributing new tests:
1. Follow the existing naming conventions
2. Add appropriate docstrings
3. Use appropriate markers (unit, integration, etc.)
4. Mock external API calls
5. Test both success and failure cases
6. Update this README if adding new test files
