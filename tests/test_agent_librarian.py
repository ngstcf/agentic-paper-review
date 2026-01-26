"""
Unit tests for the Librarian Agent.

Tests baseline reference creation and keyword extraction.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.data_models import Paper, BaselineReference, RelatedPaperMetadata
from core.config_loader import Config
from agents.agent_librarian import (
    _extract_search_keywords,
    _extract_key_findings,
    _generate_baseline_summary,
    create_baseline_reference
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_paper():
    """Create a mock paper for testing."""
    return Paper(
        id="test-paper-001",
        filename="test_paper.pdf",
        original_path="/papers/test_paper.pdf",
        content_markdown="# Test Paper\n\nThis is a test paper about machine learning.",
        token_count=500,
        metadata=Mock(
            title="A Novel Approach to Machine Learning",
            authors=["Smith, J.", "Doe, A."],
            abstract="This paper presents a novel machine learning approach for text classification using neural networks.",
            year=2023,
            keywords=["machine learning", "neural networks", "classification"]
        )
    )


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.domain = "machine_learning"
    config.get_llm_config.return_value = {
        'extractor_provider': 'openai',
        'extractor_model': 'gpt-4',
        'temperature': 0.3,
        'max_retries': 2
    }
    config.get_prompt.return_value = "Test prompt"
    return config


@pytest.fixture
def mock_baseline_papers():
    """Create mock baseline papers."""
    return [
        RelatedPaperMetadata(
            paper_id="baseline1",
            title="Foundations of Machine Learning",
            authors=["Author, A.", "Author, B."],
            year=2020,
            citation_count=500,
            abstract="This paper establishes the foundations of machine learning.",
            key_findings=[
                "Supervised learning requires labeled data",
                "Neural networks can approximate any function"
            ]
        ),
        RelatedPaperMetadata(
            paper_id="baseline2",
            title="Text Classification Methods",
            authors=["Researcher, C."],
            year=2022,
            citation_count=200,
            abstract="A survey of text classification methods.",
            key_findings=[
                "Transformer models outperform traditional methods",
                "Pre-training improves performance on downstream tasks"
            ]
        )
    ]


# ============================================================================
# TESTS: Keyword Extraction
# ============================================================================

class TestKeywordExtraction:
    """Tests for keyword extraction functionality."""

    @patch('agents.agent_librarian.call_llm')
    def test_extract_keywords_from_paper(self, mock_llm, mock_paper, mock_config):
        """Test extracting keywords from paper title and abstract."""
        # Mock LLM response
        mock_llm.return_value = {
            'success': True,
            'content': '{"sub_topic": "neural text classification", "keywords": ["neural networks", "text classification", "deep learning", "NLP", "transformers"]}',
            'cost': 0.001
        }

        keywords = _extract_search_keywords(mock_paper, mock_config)

        assert len(keywords) == 5
        assert "neural networks" in keywords
        assert "text classification" in keywords

    @patch('agents.agent_librarian.call_llm')
    def test_extract_keywords_fallback(self, mock_llm, mock_paper, mock_config):
        """Test fallback keyword extraction when LLM fails."""
        # Mock LLM failure
        mock_llm.return_value = {
            'success': False,
            'error': 'API error'
        }

        keywords = _extract_search_keywords(mock_paper, mock_config)

        # Should fall back to title words
        assert len(keywords) > 0
        assert any("machine" in k.lower() for k in keywords)


# ============================================================================
# TESTS: Key Findings Extraction
# ============================================================================

class TestKeyFindingsExtraction:
    """Tests for extracting key findings from baseline papers."""

    @patch('agents.agent_librarian.call_llm')
    def test_extract_key_findings(self, mock_llm, mock_config):
        """Test extracting key findings from a paper abstract."""
        mock_paper = RelatedPaperMetadata(
            paper_id="test123",
            title="Test Paper",
            authors=["Author"],
            year=2023,
            abstract="This paper shows that neural networks outperform traditional methods in text classification tasks.",
            citation_count=100
        )

        mock_llm.return_value = {
            'success': True,
            'content': '{"key_findings": ["Neural networks outperform traditional methods", "Performance improvement is significant"]}',
            'cost': 0.001
        }

        findings = _extract_key_findings(mock_paper, mock_config)

        assert len(findings) == 2
        assert "outperform" in findings[0].lower()

    def test_extract_key_findings_no_abstract(self, mock_config):
        """Test handling paper without abstract."""
        mock_paper = RelatedPaperMetadata(
            paper_id="test123",
            title="Test Paper",
            authors=["Author"],
            year=2023,
            abstract=None,
            citation_count=100
        )

        with patch('agents.agent_librarian.call_llm') as mock_llm:
            findings = _extract_key_findings(mock_paper, mock_config)
            assert len(findings) == 1
            assert "not available" in findings[0].lower()


# ============================================================================
# TESTS: Baseline Summary Generation
# ============================================================================

class TestBaselineSummary:
    """Tests for baseline summary generation."""

    @patch('agents.agent_librarian.call_llm')
    def test_generate_baseline_summary(self, mock_llm, mock_config, mock_baseline_papers):
        """Test generating state of the art summary."""
        mock_llm.return_value = {
            'success': True,
            'content': "Current research in neural text classification focuses on transformer architectures and pre-training methods. Key findings show that neural networks outperform traditional methods, and pre-training significantly improves performance on downstream tasks.",
            'cost': 0.002
        }

        summary = _generate_baseline_summary(
            sub_topic="neural text classification",
            baseline_papers=mock_baseline_papers,
            config=mock_config
        )

        assert len(summary) > 0
        assert "neural" in summary.lower() or "transformer" in summary.lower()


# ============================================================================
# TESTS: Baseline Reference Creation
# ============================================================================

class TestBaselineReferenceCreation:
    """Tests for creating baseline references."""

    @patch('agents.agent_librarian._generate_baseline_summary')
    @patch('agents.agent_librarian._extract_key_findings')
    @patch('agents.agent_librarian.LiteratureSearcher.get_most_cited')
    @patch('agents.agent_librarian._extract_search_keywords')
    @patch('agents.agent_librarian.LiteratureSearcher')
    def test_create_baseline_reference_success(
        self, mock_searcher_class, mock_extract_keywords,
        mock_get_cited, mock_extract_findings, mock_generate_summary,
        mock_paper, mock_config, mock_baseline_papers
    ):
        """Test successful baseline reference creation."""
        # Setup mocks
        mock_extract_keywords.return_value = ["neural networks", "text classification"]
        mock_searcher_instance = Mock()
        mock_searcher_instance.get_most_cited.return_value = mock_baseline_papers
        mock_searcher_instance.get_api_call_count.return_value = 5
        mock_searcher_class.return_value = mock_searcher_instance

        mock_extract_findings.side_effect = [
            ["Finding 1", "Finding 2"],
            ["Finding 3", "Finding 4"]
        ]

        mock_generate_summary.return_value = "State of the art summary text."

        # Execute
        baseline = create_baseline_reference(mock_paper, mock_config)

        # Verify
        assert baseline is not None
        assert baseline.sub_topic == "neural networks"
        assert len(baseline.baseline_papers) == 2
        assert baseline.total_api_calls == 5
        assert len(baseline.query_keywords) > 0

    @patch('agents.agent_librarian._extract_search_keywords')
    @patch('agents.agent_librarian.LiteratureSearcher')
    def test_create_baseline_reference_no_papers_found(
        self, mock_searcher_class, mock_extract_keywords,
        mock_paper, mock_config
    ):
        """Test handling when no baseline papers are found."""
        mock_extract_keywords.return_value = ["neural networks"]
        mock_searcher_instance = Mock()
        mock_searcher_instance.get_most_cited.return_value = []
        mock_searcher_instance.get_api_call_count.return_value = 2
        mock_searcher_class.return_value = mock_searcher_instance

        baseline = create_baseline_reference(mock_paper, mock_config)

        # Should return None or baseline with empty papers
        assert baseline is not None
        assert len(baseline.baseline_papers) == 0


# ============================================================================
# TESTS: Configuration
# ============================================================================

class TestLibrarianConfiguration:
    """Tests for librarian-specific configuration."""

    def test_literature_config_loading(self):
        """Test loading literature configuration."""
        from utilities.helpers import load_yaml_config

        config = load_yaml_config("config/literature_sources.yaml")

        assert 'librarian' in config
        assert config.get('enabled', True) is not None
        assert config['librarian'].get('baseline_papers_count') is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
