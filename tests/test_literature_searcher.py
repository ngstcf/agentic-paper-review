"""
Unit tests for the Literature Searcher module.

Tests Semantic Scholar API integration and paper retrieval.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from core.literature_searcher import (
    LiteratureSearcher,
    SearchConfig,
    RelatedPaperMetadata
)
from core.data_models import RelatedPaperMetadata as RMD


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def search_config():
    """Create test search configuration."""
    return SearchConfig(
        api_key=None,  # Use free tier
        timeout=10,
        max_retries=2
    )


@pytest.fixture
def mock_paper_response():
    """Mock Semantic Scholar paper search response."""
    return {
        "data": [
            {
                "paperId": "test123",
                "title": "Test Paper on Machine Learning",
                "authors": [{"name": "Smith, J."}, {"name": "Doe, A."}],
                "year": 2023,
                "abstract": "This is a test abstract about machine learning.",
                "citationCount": 150,
                "venue": "Test Conference",
                "url": "https://example.com/paper123",
                "openAccessPdf": {
                    "url": "https://example.com/open.pdf"
                }
            },
            {
                "paperId": "test456",
                "title": "Another Test Paper",
                "authors": [{"name": "Johnson, B."}],
                "year": 2022,
                "abstract": "Another abstract for testing.",
                "citationCount": 75,
                "venue": "Test Journal"
            }
        ]
    }


@pytest.fixture
def mock_paper_detail_response():
    """Mock Semantic Scholar paper detail response."""
    return {
        "paperId": "test123",
        "title": "Test Paper on Machine Learning",
        "authors": [{"name": "Smith, J."}, {"name": "Doe, A."}],
        "year": 2023,
        "abstract": "This is a test abstract about machine learning.",
        "citationCount": 150,
        "venue": "Test Conference",
        "url": "https://example.com/paper123",
        "openAccessPdf": {
            "url": "https://example.com/open.pdf"
        },
        "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"]
    }


# ============================================================================
# TESTS: LiteratureSearcher Initialization
# ============================================================================

class TestLiteratureSearcherInit:
    """Tests for LiteratureSearcher initialization."""

    def test_init_default_config(self, search_config):
        """Test initialization with default configuration."""
        searcher = LiteratureSearcher(search_config)
        assert searcher.config is not None
        assert searcher._session is not None
        assert searcher._api_calls_today == 0

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        config = SearchConfig(api_key="test_key_123")
        searcher = LiteratureSearcher(config)
        assert "x-api-key" in searcher._session.headers
        assert searcher._session.headers["x-api-key"] == "test_key_123"


# ============================================================================
# TESTS: Search Methods
# ============================================================================

class TestSearchMethods:
    """Tests for literature search methods."""

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_search_by_keywords(self, mock_request, search_config, mock_paper_response):
        """Test searching papers by keywords."""
        # Setup mock
        mock_request.return_value = mock_paper_response

        searcher = LiteratureSearcher(search_config)
        results = searcher.search_by_keywords(
            keywords=["machine learning", "neural networks"],
            limit=10
        )

        assert len(results) == 2
        assert results[0].paper_id == "test123"
        assert results[0].title == "Test Paper on Machine Learning"
        assert results[0].citation_count == 150
        assert len(results[0].authors) == 2

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_search_by_abstract_similarity(self, mock_request, search_config, mock_paper_response):
        """Test searching papers by abstract similarity."""
        mock_request.return_value = mock_paper_response

        searcher = LiteratureSearcher(search_config)
        results = searcher.search_by_abstract_similarity(
            abstract="This paper discusses machine learning and neural networks.",
            limit=10
        )

        assert len(results) == 2
        assert results[0].paper_id == "test123"

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_get_most_cited(self, mock_request, search_config, mock_paper_response):
        """Test getting most cited papers."""
        mock_request.return_value = mock_paper_response

        searcher = LiteratureSearcher(search_config)
        results = searcher.get_most_cited(
            field_keywords=["machine learning"],
            years=5,
            limit=5
        )

        assert len(results) == 2
        # Should be sorted by citation count
        assert results[0].citation_count >= results[1].citation_count


# ============================================================================
# TESTS: Paper Details
# ============================================================================

class TestPaperDetails:
    """Tests for fetching individual paper details."""

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_get_paper_details(self, mock_request, search_config, mock_paper_detail_response):
        """Test fetching detailed paper information."""
        mock_request.return_value = mock_paper_detail_response

        searcher = LiteratureSearcher(search_config)
        paper = searcher.get_paper_details("test123")

        assert paper is not None
        assert paper.paper_id == "test123"
        assert paper.title == "Test Paper on Machine Learning"
        assert paper.citation_count == 150
        assert paper.open_access_pdf == "https://example.com/open.pdf"

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_get_paper_details_not_found(self, mock_request, search_config):
        """Test fetching details for non-existent paper."""
        mock_request.return_value = None

        searcher = LiteratureSearcher(search_config)
        paper = searcher.get_paper_details("nonexistent")

        assert paper is None

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_fetch_abstract(self, mock_request, search_config, mock_paper_detail_response):
        """Test fetching paper abstract."""
        mock_request.return_value = mock_paper_detail_response

        searcher = LiteratureSearcher(search_config)
        abstract = searcher.fetch_abstract("test123")

        assert abstract is not None
        assert "machine learning" in abstract.lower()

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_fetch_abstract_not_available(self, mock_request, search_config):
        """Test fetching abstract when not available."""
        mock_request.return_value = {"paperId": "test123", "abstract": None}

        searcher = LiteratureSearcher(search_config)
        abstract = searcher.fetch_abstract("test123")

        assert abstract is None


# ============================================================================
# TESTS: API Call Tracking
# ============================================================================

class TestAPICallTracking:
    """Tests for API call tracking functionality."""

    def test_get_api_call_count(self, search_config):
        """Test getting API call count."""
        searcher = LiteratureSearcher(search_config)
        assert searcher.get_api_call_count() == 0

    def test_reset_api_call_count(self, search_config):
        """Test resetting API call count."""
        searcher = LiteratureSearcher(search_config)
        searcher._api_calls_today = 42
        searcher.reset_api_call_count()
        assert searcher.get_api_call_count() == 0


# ============================================================================
# TESTS: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_empty_search_results(self, mock_request, search_config):
        """Test handling of empty search results."""
        mock_request.return_value = {"data": []}

        searcher = LiteratureSearcher(search_config)
        results = searcher.search_by_keywords(["nonexistent topic"])

        assert len(results) == 0

    @patch('core.literature_searcher.LiteratureSearcher._make_request')
    def test_malformed_paper_response(self, mock_request, search_config):
        """Test handling of malformed paper response."""
        mock_request.return_value = {
            "data": [
                {
                    "paperId": "test123",
                    # Missing required fields
                }
            ]
        }

        searcher = LiteratureSearcher(search_config)
        results = searcher.search_by_keywords(["test"])

        # Should return empty list for malformed data
        assert len(results) == 0


# ============================================================================
# TESTS: Configuration
# ============================================================================

class TestConfiguration:
    """Tests for configuration handling."""

    @patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': 'env_key_123'})
    def test_create_from_env_with_key(self):
        """Test creating searcher from environment with API key."""
        from core.literature_searcher import create_searcher_from_env
        searcher = create_searcher_from_env()
        assert "x-api-key" in searcher._session.headers

    def test_create_from_env_without_key(self):
        """Test creating searcher from environment without API key."""
        # Remove API key from environment if present
        os.environ.pop('SEMANTIC_SCHOLAR_API_KEY', None)

        from core.literature_searcher import create_searcher_from_env
        searcher = create_searcher_from_env()
        assert "x-api-key" not in searcher._session.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
