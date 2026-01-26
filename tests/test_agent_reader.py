"""
Unit tests for the Reader Agent.

Tests literature-optional extraction with novelty ranking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.data_models import (
    Paper,
    NoveltyRankedExtraction,
    BaselineReference,
    RelatedPaperMetadata
)
from core.config_loader import Config
from agents.agent_reader import (
    _build_literature_context,
    extract_criterion_evidence,
    process_paper_extractions
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_paper():
    """Create a mock paper."""
    return Paper(
        id="test-paper-001",
        filename="test_paper.pdf",
        original_path="/papers/test_paper.pdf",
        content_markdown="# Introduction\n\nThis paper introduces a new method for text classification.",
        token_count=1000,
        metadata=Mock(
            title="Novel Text Classification Method",
            authors=["Smith, J."],
            abstract="A novel approach to text classification."
        )
    )


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.domain = "nlp"
    config.get_criteria.return_value = [
        {'id': 'theoretical_contribution', 'name': 'Theoretical Contribution', 'weight': 12},
        {'id': 'empirical_rigor', 'name': 'Empirical Rigor', 'weight': 18}
    ]
    config.get_llm_config.return_value = {
        'extractor_provider': 'openai',
        'extractor_model': 'gpt-4',
        'temperature': 0.3,
        'max_retries': 2,
        'max_parallel': 2
    }
    config.get_prompt.return_value = (
        "Paper: {paper_markdown}\n"
        "Criterion: {criterion_name}\n"
        "Description: {criterion_description}\n"
        "Domain: {domain}"
    )
    config.get_criterion_by_id = Mock(side_effect=lambda cid: next(
        (c for c in config.get_criteria() if c['id'] == cid), None
    ))
    return config


@pytest.fixture
def mock_criterion():
    """Create a mock evaluation criterion."""
    return {
        'id': 'theoretical_contribution',
        'name': 'Theoretical Contribution',
        'description': 'Evaluates the theoretical contribution of the paper.',
        'weight': 12,
        'scale': {
            'type': 'numeric',
            'range': [1, 5],
            'labels': {
                1: "No contribution",
                2: "Minor contribution",
                3: "Moderate contribution",
                4: "Significant contribution",
                5: "Exceptional contribution"
            }
        }
    }


@pytest.fixture
def mock_baseline():
    """Create a mock baseline reference."""
    return BaselineReference(
        sub_topic="text classification",
        query_keywords=["text classification", "neural networks"],
        baseline_papers=[
            RelatedPaperMetadata(
                paper_id="baseline1",
                title="Standard Text Classification",
                authors=["Author, A."],
                year=2020,
                citation_count=300,
                abstract="Standard approaches to text classification.",
                key_findings=["Traditional methods work well", "Feature engineering is important"]
            )
        ],
        key_findings_summary="Current research focuses on neural network approaches.",
        total_api_calls=5
    )


# ============================================================================
# TESTS: Literature Context Building
# ============================================================================

class TestLiteratureContextBuilding:
    """Tests for building literature context strings."""

    def test_build_literature_context_with_baseline(self, mock_baseline):
        """Test building literature context with baseline."""
        context = _build_literature_context(mock_baseline)

        assert "text classification" in context.lower()
        assert "baseline reference" in context.lower()
        assert "standard text classification" in context.lower()
        assert "novelty ranking" in context.lower()

    def test_build_literature_context_empty_baseline(self):
        """Test building literature context with empty baseline."""
        baseline = BaselineReference(
            sub_topic="test",
            query_keywords=[],
            baseline_papers=[],
            key_findings_summary="No papers found.",
            total_api_calls=0
        )

        context = _build_literature_context(baseline)

        assert "test" in context.lower()
        assert "no papers found" in context.lower()


# ============================================================================
# TESTS: Evidence Extraction (Standard Mode)
# ============================================================================

class TestEvidenceExtractionStandard:
    """Tests for evidence extraction without literature grounding."""

    @patch('agents.agent_reader.call_llm')
    def test_extract_without_baseline(self, mock_llm, mock_paper, mock_criterion, mock_config):
        """Test extraction without baseline (standard mode)."""
        # Mock LLM response for standard extraction
        mock_llm.return_value = {
            'success': True,
            'content': '''{
                "score": 4,
                "score_justification": "Strong theoretical contribution",
                "evidence": [
                    {"quote": "Novel approach to classification", "page_reference": "p.1", "relevance": "Direct"}
                ],
                "strengths": ["New method proposed"],
                "weaknesses": ["Limited theoretical analysis"],
                "confidence": 0.8
            }''',
            'cost': 0.01
        }

        result = extract_criterion_evidence(
            paper=mock_paper,
            criterion=mock_criterion,
            config=mock_config,
            baseline=None  # No baseline
        )

        assert result is not None
        assert result.score == 4
        assert result.novelty_ranking == 3  # Default when no baseline
        assert result.contradicts_baseline is False
        assert result.extends_baseline is False


# ============================================================================
# TESTS: Evidence Extraction (Literature-Grounded Mode)
# ============================================================================

class TestEvidenceExtractionGrounded:
    """Tests for evidence extraction with literature grounding."""

    @patch('agents.agent_reader.call_llm')
    def test_extract_with_baseline(self, mock_llm, mock_paper, mock_criterion, mock_config, mock_baseline):
        """Test extraction with baseline (literature-grounded mode)."""
        # Mock LLM response with novelty assessment
        mock_llm.return_value = {
            'success': True,
            'content': '''{
                "score": 4,
                "score_justification": "Strong contribution over baseline",
                "evidence": [
                    {"quote": "Our method improves upon standard approaches", "page_reference": "p.2", "relevance": "Direct"}
                ],
                "strengths": ["Extends prior work"],
                "weaknesses": ["Limited comparison"],
                "confidence": 0.85,
                "novelty_ranking": 4,
                "novelty_rationale": "Significant improvement over baseline methods",
                "contradicts_baseline": false,
                "extends_baseline": true,
                "prior_work_gaps": ["Limited to English text", "No handling of multilingual data"]
            }''',
            'cost': 0.015
        }

        result = extract_criterion_evidence(
            paper=mock_paper,
            criterion=mock_criterion,
            config=mock_config,
            baseline=mock_baseline
        )

        assert result is not None
        assert result.score == 4
        assert result.novelty_ranking == 4
        assert result.contradicts_baseline is False
        assert result.extends_baseline is True
        assert len(result.prior_work_gaps) == 2

    @patch('agents.agent_reader.call_llm')
    def test_extract_with_contradiction(self, mock_llm, mock_paper, mock_criterion, mock_config, mock_baseline):
        """Test extraction that contradicts baseline."""
        mock_llm.return_value = {
            'success': True,
            'content': '''{
                "score": 3,
                "score_justification": "Contradicts established findings",
                "evidence": [{"quote": "Our results differ from prior work", "page_reference": "p.3", "relevance": "Direct"}],
                "strengths": ["Challenges assumptions"],
                "weaknesses": ["Needs replication"],
                "confidence": 0.6,
                "novelty_ranking": 2,
                "novelty_rationale": "Contradicts well-established results",
                "contradicts_baseline": true,
                "extends_baseline": false,
                "prior_work_gaps": []
            }''',
            'cost': 0.012
        }

        result = extract_criterion_evidence(
            paper=mock_paper,
            criterion=mock_criterion,
            config=mock_config,
            baseline=mock_baseline
        )

        assert result.contradicts_baseline is True
        assert result.extends_baseline is False
        assert result.novelty_ranking == 2


# ============================================================================
# TESTS: Batch Processing
# ============================================================================

class TestBatchProcessing:
    """Tests for processing multiple criteria."""

    @patch('agents.agent_reader.call_llm')
    def test_process_paper_extractions_without_baseline(self, mock_llm, mock_paper, mock_config):
        """Test processing extractions without baseline."""
        # Mock responses for both criteria
        mock_llm.side_effect = [
            {
                'success': True,
                'content': '{"score": 4, "score_justification": "Good", "evidence": [], "strengths": [], "weaknesses": [], "confidence": 0.8}',
                'cost': 0.01
            },
            {
                'success': True,
                'content': '{"score": 3, "score_justification": "Fair", "evidence": [], "strengths": [], "weaknesses": [], "confidence": 0.7}',
                'cost': 0.01
            }
        ]

        results = process_paper_extractions(mock_paper, mock_config, baseline=None)

        assert len(results) == 2
        assert all(isinstance(r, NoveltyRankedExtraction) for r in results)
        assert results[0].novelty_ranking == 3  # Default without baseline

    @patch('agents.agent_reader.call_llm')
    def test_process_paper_extractions_with_baseline(self, mock_llm, mock_paper, mock_config, mock_baseline):
        """Test processing extractions with baseline."""
        mock_llm.side_effect = [
            {
                'success': True,
                'content': '{"score": 4, "score_justification": "Good", "evidence": [], "strengths": [], "weaknesses": [], "confidence": 0.8, "novelty_ranking": 4, "contradicts_baseline": false, "extends_baseline": true, "prior_work_gaps": ["gap1"]}',
                'cost': 0.015
            },
            {
                'success': True,
                'content': '{"score": 3, "score_justification": "Fair", "evidence": [], "strengths": [], "weaknesses": [], "confidence": 0.7, "novelty_ranking": 3, "contradicts_baseline": false, "extends_baseline": false, "prior_work_gaps": []}',
                'cost': 0.012
            }
        ]

        results = process_paper_extractions(mock_paper, mock_config, baseline=mock_baseline)

        assert len(results) == 2
        assert results[0].novelty_ranking == 4
        assert results[1].novelty_ranking == 3


# ============================================================================
# TESTS: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in Reader agent."""

    @patch('agents.agent_reader.call_llm')
    def test_llm_failure(self, mock_llm, mock_paper, mock_criterion, mock_config):
        """Test handling of LLM failure."""
        mock_llm.return_value = {
            'success': False,
            'error': 'API error'
        }

        result = extract_criterion_evidence(
            paper=mock_paper,
            criterion=mock_criterion,
            config=mock_config
        )

        assert result is None

    @patch('agents.agent_reader.call_llm')
    def test_malformed_json_response(self, mock_llm, mock_paper, mock_criterion, mock_config):
        """Test handling of malformed JSON response."""
        mock_llm.return_value = {
            'success': True,
            'content': 'This is not valid JSON',
            'cost': 0.01
        }

        result = extract_criterion_evidence(
            paper=mock_paper,
            criterion=mock_criterion,
            config=mock_config
        )

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
