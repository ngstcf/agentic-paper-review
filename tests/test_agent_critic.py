"""
Unit tests for the Critic Agent.

Tests literature-optional review synthesis with research trajectory.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.data_models import (
    Paper,
    PaperMetadata,
    GroundedReview,
    Review,
    WeightedBreakdown,
    NoveltyRankedExtraction,
    BaselineReference,
    FactCheckResult,
    RelatedPaperMetadata,
    DetailedAssessment
)
from core.config_loader import Config
from agents.agent_critic import (
    _calculate_novelty_adjusted_score,
    _generate_research_trajectory,
    synthesize_grounded_review
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
        content_markdown="# Test Paper\n\nContent here.",
        token_count=1000,
        metadata=PaperMetadata(
            title="Novel Machine Learning Approach",
            authors=["Smith, J."],
            abstract="A novel approach to ML."
        )
    )


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.domain = "machine_learning"
    config.get_criteria.return_value = [
        {'id': 'theoretical_contribution', 'name': 'Theoretical Contribution', 'weight': 12},
        {'id': 'empirical_rigor', 'name': 'Empirical Rigor', 'weight': 18}
    ]
    config.get_llm_config.return_value = {
        'synthesizer_provider': 'openai',
        'synthesizer_model': 'gpt-4',
        'temperature': 0.5,
        'max_retries': 2
    }
    config.get_agent_config.return_value = {
        'critic_temperature': 0.6,
        'critic_max_json_retries': 3,
        'librarian_temperature': 0.3,
        'librarian_summary_temperature': 0.5,
        'fact_checker_temperature': 0.3,
        'librarian_pre_search_delay': 1.0,
        'librarian_retry_delay': 2.0,
        'fact_checker_max_retries': 2,
    }
    config.get_novelty_config.return_value = {
        'base_factor': 0.025,
        'contradiction_penalty': 0.05,
        'extension_bonus': 0.03,
    }
    config.get_prompt.return_value = "Prompt: {paper_title}"
    return config


@pytest.fixture
def mock_extractions():
    """Create mock novelty-ranked extractions."""
    return [
        NoveltyRankedExtraction(
            paper_id="test-paper",
            criterion_id="theoretical_contribution",
            score=4,
            score_justification="Good contribution",
            evidence=[],
            strengths=["Novel approach"],
            weaknesses=["Limited theory"],
            confidence=0.8,
            model_used="openai/gpt-4",
            cost=0.01,
            novelty_ranking=4,
            contradicts_baseline=False,
            extends_baseline=True,
            prior_work_gaps=["Gap in handling edge cases"]
        ),
        NoveltyRankedExtraction(
            paper_id="test-paper",
            criterion_id="empirical_rigor",
            score=3,
            score_justification="Moderate rigor",
            evidence=[],
            strengths=["Good data"],
            weaknesses=["Small sample"],
            confidence=0.7,
            model_used="openai/gpt-4",
            cost=0.01,
            novelty_ranking=3,
            contradicts_baseline=False,
            extends_baseline=False,
            prior_work_gaps=[]
        )
    ]


@pytest.fixture
def mock_baseline():
    """Create a mock baseline reference."""
    return BaselineReference(
        sub_topic="machine learning approaches",
        query_keywords=["machine learning", "neural networks"],
        baseline_papers=[
            RelatedPaperMetadata(
                paper_id="baseline1",
                title="Standard ML Methods",
                authors=["Author, A."],
                year=2020,
                citation_count=500,
                abstract="Standard approaches to ML.",
                key_findings=["Traditional methods work well", "Deep learning improves performance"]
            ),
            RelatedPaperMetadata(
                paper_id="baseline2",
                title="Neural Network Applications",
                authors=["Researcher, B."],
                year=2022,
                citation_count=300,
                abstract="Applications of neural networks.",
                key_findings=["CNNs work for images", "RNNs for sequences"]
            )
        ],
        key_findings_summary="Current research focuses on deep learning approaches for various applications.",
        total_api_calls=5
    )


@pytest.fixture
def mock_fact_checks():
    """Create mock fact-check results."""
    return [
        FactCheckResult(
            claim="First study to apply this method",
            criterion_id="theoretical_contribution",
            search_query="first method application",
            found_prior_work=True,
            prior_work_summary="Found similar prior work",
            verification_status="disputed",
            recommendation="Consider reducing novelty claim",
            papers_found=[]
        )
    ]


# ============================================================================
# TESTS: Novelty Score Adjustment
# ============================================================================

class TestNoveltyScoreAdjustment:
    """Tests for novelty-adjusted score calculation."""

    def test_adjustment_high_novelty(self, mock_extractions, mock_config):
        """Test score adjustment with high average novelty."""
        # Extractions have avg novelty of 3.5 (4 + 3) / 2
        base_score = 75.0
        adjusted = _calculate_novelty_adjusted_score(base_score, mock_extractions, mock_config)

        # Should be higher than base (positive adjustment for high novelty)
        assert adjusted > base_score
        assert adjusted <= 100  # Cap at 100

    def test_adjustment_low_novelty(self, mock_config):
        """Test score adjustment with low average novelty."""
        low_novelty_extractions = [
            NoveltyRankedExtraction(
                paper_id="test",
                criterion_id="test",
                score=3,
                score_justification="Minor contribution",
                evidence=[],
                strengths=[],
                weaknesses=[],
                confidence=0.6,
                model_used="openai/gpt-4",
                cost=0.01,
                novelty_ranking=1,  # Very low novelty
                contradicts_baseline=False,
                extends_baseline=False,
                prior_work_gaps=[]
            )
        ]

        base_score = 75.0
        adjusted = _calculate_novelty_adjusted_score(base_score, low_novelty_extractions, mock_config)

        # Should be lower than base (negative adjustment for low novelty)
        assert adjusted < base_score
        assert adjusted >= 0  # Floor at 0

    def test_adjustment_with_contradiction(self, mock_config):
        """Test additional penalty for contradictions."""
        contradictory_extractions = [
            NoveltyRankedExtraction(
                paper_id="test",
                criterion_id="test",
                score=3,
                score_justification="Test",
                evidence=[],
                strengths=[],
                weaknesses=[],
                confidence=0.7,
                model_used="openai/gpt-4",
                cost=0.01,
                novelty_ranking=3,
                contradicts_baseline=True,  # Contradicts baseline
                extends_baseline=False,
                prior_work_gaps=[]
            )
        ]

        base_score = 75.0
        adjusted_contradiction = _calculate_novelty_adjusted_score(
            base_score, contradictory_extractions, mock_config
        )

        # Create same extraction without contradiction
        no_contradiction = contradictory_extractions.copy()
        no_contradiction[0].contradicts_baseline = False
        adjusted_no_contradiction = _calculate_novelty_adjusted_score(
            base_score, no_contradiction, mock_config
        )

        # Contradiction should result in lower score
        assert adjusted_contradiction < adjusted_no_contradiction

    def test_adjustment_with_extensions(self, mock_config):
        """Test bonus for extending baseline."""
        extension_extractions = [
            NoveltyRankedExtraction(
                paper_id="test",
                criterion_id="test",
                score=3,
                score_justification="Test",
                evidence=[],
                strengths=[],
                weaknesses=[],
                confidence=0.7,
                model_used="openai/gpt-4",
                cost=0.01,
                novelty_ranking=3,
                contradicts_baseline=False,
                extends_baseline=True,  # Extends baseline
                prior_work_gaps=["gap1", "gap2"]
            )
        ]

        base_score = 75.0
        adjusted_extension = _calculate_novelty_adjusted_score(
            base_score, extension_extractions, mock_config
        )

        # Create same extraction without extension
        no_extension = extension_extractions.copy()
        no_extension[0].extends_baseline = False
        adjusted_no_extension = _calculate_novelty_adjusted_score(
            base_score, no_extension, mock_config
        )

        # Extension should result in higher score
        assert adjusted_extension > adjusted_no_extension


# ============================================================================
# TESTS: Research Trajectory Generation
# ============================================================================

class TestResearchTrajectory:
    """Tests for research trajectory section generation."""

    @patch('agents.agent_critic.call_llm')
    def test_generate_research_trajectory(
        self, mock_llm, mock_paper, mock_baseline,
        mock_extractions, mock_fact_checks, mock_config
    ):
        """Test generation of research trajectory section."""
        mock_llm.return_value = {
            'success': True,
            'content': """
## Research Trajectory and Position

This paper extends the current line of research in machine learning approaches.
It builds upon the foundational work by Author et al. and Researcher et al.,
particularly in the area of neural network applications.

### Key Contributions
- Addresses gap in handling edge cases
- Provides novel optimization approach

### Novelty Assessment
The paper shows high novelty in theoretical contribution (4/5) and
moderate novelty in empirical rigor (3/5).

### Verification Notes
One claim regarding being the "first study" requires further review
as prior work was found.
""",
            'cost': 0.02
        }

        trajectory = _generate_research_trajectory(
            paper=mock_paper,
            baseline=mock_baseline,
            extractions=mock_extractions,
            fact_checks=mock_fact_checks,
            config=mock_config
        )

        assert len(trajectory) > 0
        assert "research trajectory" in trajectory.lower() or "position" in trajectory.lower()

    @patch('agents.agent_critic.call_llm')
    def test_generate_trajectory_fallback(self, mock_llm, mock_paper, mock_baseline):
        """Test fallback when LLM fails."""
        mock_llm.return_value = {
            'success': False,
            'error': 'API error'
        }

        trajectory = _generate_research_trajectory(
            paper=mock_paper,
            baseline=mock_baseline,
            extractions=[],
            fact_checks=[],
            config=mock_config
        )

        # Should return fallback template
        assert len(trajectory) > 0


# ============================================================================
# TESTS: Grounded Review Synthesis
# ============================================================================

class TestGroundedReviewSynthesis:
    """Tests for synthesizing grounded reviews."""

    @patch('agents.agent_critic._generate_research_trajectory')
    @patch('agents.agent_critic._calculate_novelty_adjusted_score')
    @patch('agents.agent_critic.calculate_recommendation')
    @patch('agents.agent_critic.call_llm')
    def test_synthesize_grounded_review_success(
        self, mock_llm, mock_calc_rec, mock_calc_adj, mock_gen_traj,
        mock_paper, mock_extractions, mock_config, mock_baseline, mock_fact_checks
    ):
        """Test successful grounded review synthesis."""
        # Setup mocks
        mock_calc_rec.return_value = ("Accept", "Good score", 85.0, {})
        mock_calc_adj.return_value = 87.5
        mock_gen_traj.return_value = "Research trajectory section..."

        mock_llm.return_value = {
            'success': True,
            'content': '''{
                "executive_summary": "Strong paper with good contribution",
                "detailed_assessment": {
                    "major_strengths": ["Novel approach"],
                    "major_concerns": [],
                    "minor_issues": ["Minor formatting"]
                },
                "revision_suggestions": ["Add more details"],
                "decision_confidence": 0.85,
                "flags": [],
                "criterion_narrative": {
                    "theoretical_contribution": "Good theory",
                    "empirical_rigor": "Solid methods"
                }
            }''',
            'cost': 0.03
        }

        review = synthesize_grounded_review(
            paper=mock_paper,
            extractions=mock_extractions,
            config=mock_config,
            baseline=mock_baseline,
            fact_checks=mock_fact_checks
        )

        assert review is not None
        assert isinstance(review, GroundedReview)
        assert review.novelty_adjusted_score == 87.5
        assert review.research_trajectory_section == "Research trajectory section..."
        assert len(review.literature_context.fact_checks) == 1

    @patch('agents.agent_critic.calculate_recommendation')
    @patch('agents.agent_critic.call_llm')
    def test_synthesize_review_without_baseline(
        self, mock_llm, mock_calc_rec,
        mock_paper, mock_extractions, mock_config
    ):
        """Test synthesis without baseline (standard mode)."""
        mock_calc_rec.return_value = ("Accept", "Good score", 80.0, {})
        mock_llm.return_value = {
            'success': True,
            'content': '''{
                "executive_summary": "Good paper",
                "detailed_assessment": {"major_strengths": [], "major_concerns": [], "minor_issues": []},
                "revision_suggestions": [],
                "decision_confidence": 0.8,
                "flags": [],
                "criterion_narrative": {}
            }''',
            'cost': 0.02
        }

        review = synthesize_grounded_review(
            paper=mock_paper,
            extractions=mock_extractions,
            config=mock_config,
            baseline=None,  # No baseline
            fact_checks=[]
        )

        assert review is not None
        assert isinstance(review, GroundedReview)
        assert review.novelty_adjusted_score is None  # No adjustment without baseline
        assert review.research_trajectory_section == ""  # No trajectory without baseline

    @patch('agents.agent_critic.calculate_recommendation')
    def test_synthesize_review_llm_failure(self, mock_calc_rec, mock_paper, mock_extractions, mock_config):
        """Test handling of LLM failure during synthesis."""
        mock_calc_rec.return_value = ("Accept", "Good score", 80.0, {})

        with patch('agents.agent_critic.call_llm') as mock_llm:
            mock_llm.return_value = {
                'success': False,
                'error': 'API error'
            }

            review = synthesize_grounded_review(
                paper=mock_paper,
                extractions=mock_extractions,
                config=mock_config
            )

            assert review is None


# ============================================================================
# TESTS: Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with standard Review."""

    def test_grounded_review_is_review(self):
        """Test that GroundedReview is a subclass of Review."""
        from core.data_models import Review

        # Create a minimal GroundedReview
        grounded = GroundedReview(
            paper_id="test",
            paper_title="Test",
            paper_filename="test.pdf",
            overall_score=80.0,
            weighted_breakdown={},
            recommendation="Accept",
            recommendation_rationale="Good",
            executive_summary="Summary",
            detailed_assessment=DetailedAssessment(
                major_strengths=[],
                major_concerns=[],
                minor_issues=[]
            ),
            criterion_narrative={},
            revision_suggestions=[],
            decision_confidence=0.8,
            synthesizer_model_used="openai/gpt-4",
            extractor_model_used="openai/gpt-4"
        )

        # Should be instance of Review
        assert isinstance(grounded, Review)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
