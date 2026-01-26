"""
Unit tests for the Fact-Checker Agent.

Tests just-in-time verification of suspicious claims.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.data_models import NoveltyRankedExtraction, FactCheckResult, RelatedPaperMetadata
from core.config_loader import Config
from agents.agent_fact_checker import (
    _analyze_for_triggers,
    _generate_verification_query,
    _verify_claim,
    run_fact_checks,
    summarize_fact_checks
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.get_criteria.return_value = [
        {
            'id': 'theoretical_contribution',
            'name': 'Theoretical Contribution',
            'weight': 12
        },
        {
            'id': 'empirical_rigor',
            'name': 'Empirical Rigor',
            'weight': 18
        }
    ]
    config.get_llm_config.return_value = {
        'extractor_provider': 'openai',
        'extractor_model': 'gpt-4',
        'temperature': 0.3,
        'max_retries': 2
    }
    return config


@pytest.fixture
def mock_literature_config():
    """Create mock literature configuration."""
    return {
        'enabled': True,
        'fact_checker': {
            'max_verifications_per_section': 3,
            'max_verifications_total': 10,
            'triggers': {
                'first_study_claim': {
                    'enabled': True,
                    'confidence_threshold': 0.7
                },
                'contradiction_high_citation': {
                    'enabled': True,
                    'citation_threshold': 100
                },
                'implausible_claim': {
                    'enabled': True,
                    'confidence_threshold': 0.6
                }
            }
        },
        'semantic_scholar': {
            'api_key': None,
            'base_url': 'https://api.semanticscholar.org/graph/v1',
            'timeout': 30,
            'max_retries': 3
        }
    }


@pytest.fixture
def mock_extraction_high_novelty():
    """Create extraction with high novelty claim."""
    return NoveltyRankedExtraction(
        paper_id="test-paper",
        criterion_id="theoretical_contribution",
        score=5,
        score_justification="This is the first study to apply this method to this domain, representing a novel and unprecedented contribution.",
        evidence=[],
        strengths=["Novel approach"],
        weaknesses=["Limited validation"],
        confidence=0.65,  # Below threshold
        model_used="openai/gpt-4",
        cost=0.01,
        novelty_ranking=5,  # Exceptional novelty
        contradicts_baseline=False,
        extends_baseline=True,
        prior_work_gaps=[]
    )


@pytest.fixture
def mock_extraction_contradiction():
    """Create extraction that contradicts baseline."""
    return NoveltyRankedExtraction(
        paper_id="test-paper",
        criterion_id="theoretical_contribution",
        score=4,
        score_justification="Contradicts prior findings by Smith et al., showing opposite results.",
        evidence=[],
        strengths=["Challenges assumptions"],
        weaknesses=["Needs replication"],
        confidence=0.7,
        model_used="openai/gpt-4",
        cost=0.01,
        novelty_ranking=4,
        contradicts_baseline=True,  # Contradicts baseline
        extends_baseline=False,
        prior_work_gaps=[]
    )


@pytest.fixture
def mock_extraction_normal():
    """Create normal extraction without triggers."""
    return NoveltyRankedExtraction(
        paper_id="test-paper",
        criterion_id="empirical_rigor",
        score=4,
        score_justification="Solid empirical analysis with good methodology.",
        evidence=[],
        strengths=["Good data"],
        weaknesses=["Small sample"],
        confidence=0.8,
        model_used="openai/gpt-4",
        cost=0.01,
        novelty_ranking=3,  # Moderate novelty
        contradicts_baseline=False,
        extends_baseline=False,
        prior_work_gaps=[]
    )


# ============================================================================
# TESTS: Trigger Analysis
# ============================================================================

class TestTriggerAnalysis:
    """Tests for analyzing extraction triggers."""

    def test_first_study_claim_trigger(self, mock_extraction_high_novelty, mock_config, mock_literature_config):
        """Test detection of 'first study' claim trigger."""
        triggers = _analyze_for_triggers(
            extraction=mock_extraction_high_novelty,
            criterion={'id': 'theoretical_contribution'},
            config=mock_config,
            literature_config=mock_literature_config
        )

        assert len(triggers) > 0
        assert any(t.trigger_type == 'first_study_claim' for t in triggers)
        assert triggers[0].should_trigger is True

    def test_contradiction_trigger(self, mock_extraction_contradiction, mock_config, mock_literature_config):
        """Test detection of contradiction trigger."""
        triggers = _analyze_for_triggers(
            extraction=mock_extraction_contradiction,
            criterion={'id': 'theoretical_contribution'},
            config=mock_config,
            literature_config=mock_literature_config
        )

        assert len(triggers) > 0
        assert any(t.trigger_type == 'contradiction' for t in triggers)

    def test_no_triggers(self, mock_extraction_normal, mock_config, mock_literature_config):
        """Test normal extraction with no triggers."""
        triggers = _analyze_for_triggers(
            extraction=mock_extraction_normal,
            criterion={'id': 'empirical_rigor'},
            config=mock_config,
            literature_config=mock_literature_config
        )

        assert len(triggers) == 0

    def test_trigger_disabled(self, mock_extraction_high_novelty, mock_config):
        """Test behavior when trigger is disabled."""
        config = mock_literature_config.copy()
        config['fact_checker']['triggers']['first_study_claim']['enabled'] = False

        triggers = _analyze_for_triggers(
            extraction=mock_extraction_high_novelty,
            criterion={'id': 'theoretical_contribution'},
            config=mock_config,
            literature_config=config
        )

        # Should not trigger first_study_claim when disabled
        assert not any(t.trigger_type == 'first_study_claim' and t.should_trigger for t in triggers)


# ============================================================================
# TESTS: Query Generation
# ============================================================================

class TestQueryGeneration:
    """Tests for verification query generation."""

    @patch('agents.agent_fact_checker.call_llm')
    def test_generate_verification_query(self, mock_llm, mock_extraction_high_novelty):
        """Test generating search query for verification."""
        mock_llm.return_value = {
            'success': True,
            'content': 'first study domain method application',
            'cost': 0.001
        }

        trigger = Mock(
            trigger_type='first_study_claim',
            claim_text="This is the first study to apply this method",
            should_trigger=True
        )

        query = _generate_verification_query(
            trigger=trigger,
            extraction=mock_extraction_high_novelty,
            criterion={'id': 'theoretical_contribution'}
        )

        assert len(query) > 0
        assert "domain" in query.lower() or "method" in query.lower()

    @patch('agents.agent_fact_checker.call_llm')
    def test_generate_query_fallback(self, mock_llm, mock_extraction_high_novelty):
        """Test fallback query generation when LLM fails."""
        mock_llm.return_value = {
            'success': False,
            'error': 'API error'
        }

        trigger = Mock(
            trigger_type='first_study_claim',
            claim_text="This is the first study to apply this method",
            should_trigger=True
        )

        query = _generate_verification_query(
            trigger=trigger,
            extraction=mock_extraction_high_novelty,
            criterion={'id': 'theoretical_contribution'}
        )

        # Should fallback to simple keyword extraction
        assert len(query) > 0


# ============================================================================
# TESTS: Claim Verification
# ============================================================================

class TestClaimVerification:
    """Tests for claim verification."""

    @patch('agents.agent_fact_checker.LiteratureSearcher.search_by_keywords')
    def test_verify_claim_found_prior_work(self, mock_search, mock_config, mock_literature_config):
        """Test verification when prior work is found."""
        # Mock search results
        mock_search.return_value = [
            RelatedPaperMetadata(
                paper_id="prior1",
                title="Prior Work on Similar Topic",
                authors=["Author"],
                year=2022,
                citation_count=50,
                abstract="This paper already applied similar methods."
            )
        ]

        trigger = Mock(
            trigger_type='first_study_claim',
            claim_text="First study to apply method",
            should_trigger=True
        )

        extraction = NoveltyRankedExtraction(
            paper_id="test",
            criterion_id="test",
            score=5,
            score_justification="First study claim",
            evidence=[],
            strengths=[],
            weaknesses=[],
            confidence=0.6,
            model_used="openai/gpt-4",
            cost=0.01,
            novelty_ranking=5,
            contradicts_baseline=False,
            extends_baseline=True,
            prior_work_gaps=[]
        )

        result = _verify_claim(
            trigger=trigger,
            search_query="similar method application",
            extraction=extraction,
            config=mock_config,
            literature_config=mock_literature_config
        )

        assert result.found_prior_work is True
        assert result.verification_status in ["disputed", "confirmed"]
        assert len(result.papers_found) > 0

    @patch('agents.agent_fact_checker.LiteratureSearcher.search_by_keywords')
    def test_verify_claim_no_prior_work(self, mock_search, mock_config, mock_literature_config):
        """Test verification when no prior work is found."""
        mock_search.return_value = []

        trigger = Mock(
            trigger_type='first_study_claim',
            claim_text="Truly novel approach",
            should_trigger=True
        )

        extraction = NoveltyRankedExtraction(
            paper_id="test",
            criterion_id="test",
            score=5,
            score_justification="Novel approach",
            evidence=[],
            strengths=[],
            weaknesses=[],
            confidence=0.8,
            model_used="openai/gpt-4",
            cost=0.01,
            novelty_ranking=5,
            contradicts_baseline=False,
            extends_baseline=True,
            prior_work_gaps=[]
        )

        result = _verify_claim(
            trigger=trigger,
            search_query="novel approach",
            extraction=extraction,
            config=mock_config,
            literature_config=mock_literature_config
        )

        assert result.found_prior_work is False
        assert result.verification_status == "confirmed" or result.verification_status == "novel"


# ============================================================================
# TESTS: Batch Fact Checking
# ============================================================================

class TestBatchFactChecking:
    """Tests for running fact checks on multiple extractions."""

    @patch('agents.agent_fact_checker._verify_claim')
    @patch('agents.agent_fact_checker._generate_verification_query')
    @patch('agents.agent_fact_checker._analyze_for_triggers')
    def test_run_fact_checks_with_triggers(
        self, mock_analyze, mock_generate_query, mock_verify,
        mock_config, mock_literature_config
    ):
        """Test running fact checks when triggers are detected."""
        # Setup mock trigger
        mock_trigger = Mock(
            trigger_type='first_study_claim',
            claim_text="Test claim",
            should_trigger=True
        )
        mock_analyze.return_value = [mock_trigger]
        mock_generate_query.return_value = "test query"

        # Setup mock verification result
        mock_verify.return_value = FactCheckResult(
            claim="Test claim",
            criterion_id="test",
            search_query="test query",
            found_prior_work=False,
            prior_work_summary="No prior work",
            verification_status="confirmed",
            recommendation="Claim appears valid",
            papers_found=[]
        )

        extractions = [mock_extraction_high_novelty()]

        results = run_fact_checks(
            extractions=extractions,
            criteria=mock_config.get_criteria(),
            config=mock_config,
            literature_config=mock_literature_config
        )

        assert len(results) > 0
        assert all(isinstance(r, FactCheckResult) for r in results)

    def test_run_fact_checks_disabled(self, mock_config, mock_extraction_normal):
        """Test that fact checks are skipped when disabled."""
        disabled_config = {'enabled': False}

        results = run_fact_checks(
            extractions=[mock_extraction_normal],
            criteria=mock_config.get_criteria(),
            config=mock_config,
            literature_config=disabled_config
        )

        assert len(results) == 0


# ============================================================================
# TESTS: Summarization
# ============================================================================

class TestSummarization:
    """Tests for fact-check result summarization."""

    def test_summarize_no_fact_checks(self):
        """Test summarization when no fact checks were performed."""
        summary = summarize_fact_checks([])

        assert "No fact-checks performed" in summary

    def test_summarize_with_disputed_claims(self):
        """Test summarization with disputed claims."""
        disputed = FactCheckResult(
            claim="First study to use method",
            criterion_id="theoretical_contribution",
            search_query="first study method",
            found_prior_work=True,
            prior_work_summary="Found prior work",
            verification_status="disputed",
            recommendation="Reduce novelty rating",
            papers_found=[
                RelatedPaperMetadata(
                    paper_id="prior1",
                    title="Prior Work",
                    authors=["Author"],
                    year=2020,
                    citation_count=100
                )
            ]
        )

        summary = summarize_fact_checks([disputed])

        assert "disputed" in summary.lower()
        assert "prior work" in summary.lower()
        assert "verification" in summary.lower()

    def test_summarize_mixed_results(self):
        """Test summarization with mixed verification statuses."""
        fact_checks = [
            FactCheckResult(
                claim="Claim 1",
                criterion_id="test",
                search_query="query1",
                found_prior_work=True,
                prior_work_summary="Found work",
                verification_status="disputed",
                recommendation="Review",
                papers_found=[]
            ),
            FactCheckResult(
                claim="Claim 2",
                criterion_id="test",
                search_query="query2",
                found_prior_work=False,
                prior_work_summary="No work",
                verification_status="confirmed",
                recommendation="Valid",
                papers_found=[]
            )
        ]

        summary = summarize_fact_checks(fact_checks)

        assert "disputed" in summary.lower()
        assert "confirmed" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
