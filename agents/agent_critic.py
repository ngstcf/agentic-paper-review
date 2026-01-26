"""
The Critic Agent (Modified Synthesizer)

STAGE 4: Synthesizes the review with literature-grounded assessment.

Adds:
- Research Trajectory section showing where the paper fits in the literature
- Novelty-adjusted scoring
- Integration of fact-checker results
- Literature context in the final review
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError

from core.data_models import (
    Paper,
    Review,
    GroundedReview,
    WeightedBreakdown,
    NoveltyRankedExtraction,
    BaselineReference,
    FactCheckResult,
    LiteratureContext
)
from core.config_loader import Config
from core.llm_wrapper import call_llm
from utilities.helpers import get_weights_table


def _calculate_novelty_adjusted_score(
    base_score: float,
    extractions: List[NoveltyRankedExtraction],
    config: Config
) -> float:
    """
    Adjust the overall score based on novelty rankings.

    Args:
        base_score: The base calculated score
        extractions: List of novelty-ranked extractions
        config: System configuration

    Returns:
        Novelty-adjusted score (can be higher or lower)
    """
    # Calculate average novelty ranking
    novelty_scores = [e.novelty_ranking for e in extractions]
    avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 3

    # Novelty adjustment factor (1-5 scale maps to -5% to +5% adjustment)
    # 1 = -5%, 2 = -2.5%, 3 = 0%, 4 = +2.5%, 5 = +5%
    adjustment_factor = (avg_novelty - 3) * 0.025

    # Check for contradictions (penalty)
    has_contradictions = any(e.contradicts_baseline for e in extractions)
    if has_contradictions:
        adjustment_factor -= 0.05  # Additional 5% penalty

    # Check for significant extensions (bonus)
    has_extensions = any(e.extends_baseline for e in extractions)
    if has_extensions:
        adjustment_factor += 0.03  # Additional 3% bonus

    # Apply adjustment
    adjusted_score = base_score * (1 + adjustment_factor)

    # Clamp to valid range
    adjusted_score = max(0, min(100, adjusted_score))

    return adjusted_score


def _generate_research_trajectory(
    paper: Paper,
    baseline: BaselineReference,
    extractions: List[NoveltyRankedExtraction],
    fact_checks: List[FactCheckResult],
    config: Config
) -> str:
    """
    Generate the Research Trajectory section for the review.

    Args:
        paper: The target paper
        baseline: Baseline reference
        extractions: Novelty-ranked extractions
        fact_checks: Fact-check results
        config: System configuration

    Returns:
        Formatted research trajectory section
    """
    # Build summary of novelty findings
    novelty_summary = "### Novelty Assessment by Criterion\n\n"

    for extraction in extractions:
        novelty_labels = {
            1: "No novelty - replicates prior work",
            2: "Marginal novelty - minor extension",
            3: "Moderate novelty - builds on prior work",
            4: "High novelty - significant advance",
            5: "Exceptional novelty - groundbreaking"
        }

        novelty_summary += f"**{extraction.criterion_id}**: {novelty_labels.get(extraction.novelty_ranking, 'Unknown')}\n"

        if extraction.contradicts_baseline:
            novelty_summary += f" - ⚠️ Note: Contradicts established findings\n"

        if extraction.extends_baseline and extraction.prior_work_gaps:
            novelty_summary += f" - ✓ Addresses gaps: {', '.join(extraction.prior_work_gaps[:2])}\n"

        novelty_summary += "\n"

    # Build fact-check summary if available
    fact_check_summary = ""
    if fact_checks:
        disputed = [fc for fc in fact_checks if fc.verification_status == "disputed"]
        if disputed:
            fact_check_summary = "### ⚠️ Claims Requiring Further Review\n\n"
            for fc in disputed[:3]:  # Top 3
                fact_check_summary += f"**{fc.criterion_id}**: {fc.claim[:80]}...\n"
                fact_check_summary += f"- {fc.recommendation}\n\n"

    # Generate the trajectory using LLM
    prompt = f"""
Based on the following information, write a "Research Trajectory and Position" section
for an academic review.

TARGET PAPER:
Title: {paper.metadata.title}
Abstract: {paper.metadata.abstract}

BASELINE REFERENCE:
Sub-topic: {baseline.sub_topic}
State of the Art: {baseline.key_findings_summary}

Key Baseline Papers:
{json.dumps([{"title": p.title, "year": p.year, "citations": p.citation_count, "findings": p.key_findings}
              for p in baseline.baseline_papers[:3]], indent=2)}

NOVELTY ASSESSMENT:
{novelty_summary}

{fact_check_summary}

TASK:
Write a concise (200-300 words) "Research Trajectory and Position" section that:

1. **Position in Research Landscape**: Does this paper extend, challenge, or pivot
   from the existing line of research? Use phrases like "builds on", "extends",
   "challenges", "offers a new perspective on", etc.

2. **Key Contributions**: What specific questions does this paper answer that
   baseline papers do not? Be specific.

3. **Novelty Summary**: Overall assessment of the paper's novelty based on the
   criterion-by-criterion analysis above.

4. **Verification Notes** (if applicable): Note any claims that require further
   verification based on fact-checking.

Write in clear, academic prose suitable for inclusion in a peer review.
"""

    response = call_llm(
        prompt=prompt,
        system_prompt="You are an expert academic reviewer skilled at positioning research within the broader literature.",
        provider=config.get_llm_config()['synthesizer_provider'],
        model=config.get_llm_config()['synthesizer_model'],
        temperature=0.6,
        max_retries=2
    )

    if response['success']:
        return response['content'].strip()

    # Fallback: simple template
    return f"""
## Research Trajectory and Position

This paper addresses the topic of **{baseline.sub_topic}**.

### Position in the Literature

The target paper [position relative to baseline papers].

### Key Contributions

[Summary of contributions based on novelty assessment]

### Novelty Assessment

{novelty_summary}

{fact_check_summary}
"""


def synthesize_grounded_review(
    paper: Paper,
    extractions: List[NoveltyRankedExtraction],
    config: Config,
    baseline: Optional[BaselineReference] = None,
    fact_checks: List[FactCheckResult] = None
) -> Optional[GroundedReview]:
    """
    Critic Agent: Synthesize review with literature-grounded assessment.

    Args:
        paper: The target paper
        extractions: Novelty-ranked extractions
        config: System configuration
        baseline: Optional baseline reference
        fact_checks: Optional fact-check results

    Returns:
        GroundedReview with literature context and research trajectory
    """
    print(f"[Critic] Synthesizing grounded review for: {paper.filename}")

    if not extractions:
        print(f"[Critic Error] No extractions provided for {paper.filename}.")
        return None

    llm_config = config.get_llm_config()

    # Use critic-specific prompts if available, otherwise fall back to synthesizer
    try:
        prompt_template = config.get_prompt("critic_user")
        system_prompt = config.get_prompt("critic_system")
    except:
        prompt_template = config.get_prompt("synthesizer_user")
        system_prompt = config.get_prompt("synthesizer_system")

    # Calculate recommendation and score
    recommendation, rationale, base_score, breakdown = calculate_recommendation(extractions, config)

    # Calculate novelty-adjusted score if baseline is available
    novelty_adjusted_score = None
    if baseline:
        novelty_adjusted_score = _calculate_novelty_adjusted_score(base_score, extractions, config)
        print(f"[Critic] Base score: {base_score:.1f}, Novelty-adjusted: {novelty_adjusted_score:.1f}")

    # Generate research trajectory section if literature context is available
    research_trajectory = ""
    literature_context = LiteratureContext()

    if baseline:
        research_trajectory = _generate_research_trajectory(
            paper=paper,
            baseline=baseline,
            extractions=extractions,
            fact_checks=fact_checks or [],
            config=config
        )

        literature_context = LiteratureContext(
            baseline_reference=baseline,
            fact_checks=fact_checks or [],
            total_api_calls=baseline.total_api_calls
        )

    # Prepare extractions for prompt (convert to JSON)
    extractions_list = [e.model_dump(mode='json') for e in extractions]
    extractions_json = json.dumps(extractions_list, indent=2)
    weights_table = get_weights_table(config.get_criteria())

    # Build prompt with optional literature context
    literature_section = ""
    if baseline and research_trajectory:
        literature_section = f"""

## LITERATURE CONTEXT (For Your Reference)
{research_trajectory}

When writing your review, consider how the paper's claims relate to this literature context.
"""

    prompt = prompt_template.format(
        paper_title=paper.metadata.title,
        paper_abstract=paper.metadata.abstract,
        json_dump_of_extractions=extractions_json,
        weights_table=weights_table,
        calculated_score=f"{novelty_adjusted_score or base_score:.1f}",
        calculated_recommendation=f"{recommendation} (Rationale: {rationale})"
    )

    # Add literature context to prompt if available
    if literature_section:
        prompt += literature_section

    # Call LLM
    response = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=llm_config['synthesizer_provider'],
        model=llm_config['synthesizer_model'],
        temperature=llm_config['temperature'],
        max_retries=llm_config['max_retries']
    )

    if not response['success']:
        print(f"[Critic Error] LLM call failed for {paper.filename}: {response['error']}")
        return None

    json_text = ""
    try:
        raw_content = response['content']
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')

        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise json.JSONDecodeError("Could not find JSON object markers", raw_content, 0)

        json_text = raw_content[start_index:end_index + 1]
        review_data = json.loads(json_text)

        # Calculate costs
        total_extraction_cost = sum(e.cost for e in extractions)
        total_cost = total_extraction_cost + response['cost']

        if 'criterion_narrative' not in review_data or not isinstance(review_data['criterion_narrative'], dict):
            print(f"[Critic Warning] 'criterion_narrative' not found or not a dict. Setting to empty.")
            review_data['criterion_narrative'] = {}

        extractor_model_name = extractions[0].model_used if extractions else "unknown_extractor"
        synthesizer_model_name = f"{llm_config['synthesizer_provider']}/{llm_config['synthesizer_model']}"

        # Use the novelty-adjusted score if available, otherwise use base score
        final_score = novelty_adjusted_score or base_score

        review = GroundedReview(
            paper_id=paper.id,
            paper_title=paper.metadata.title,
            paper_filename=paper.filename,
            overall_score=final_score,
            weighted_breakdown=breakdown,
            synthesizer_model_used=synthesizer_model_name,
            extractor_model_used=extractor_model_name,
            total_cost=total_cost,
            literature_context=literature_context,
            research_trajectory_section=research_trajectory,
            novelty_adjusted_score=novelty_adjusted_score,
            **review_data
        )

        return review

    except json.JSONDecodeError as e:
        print(f"[Critic Error] Failed to parse JSON for {paper.filename}: {e}")
        return None
    except ValidationError as e:
        print(f"[Critic Error] Pydantic validation failed for {paper.filename}: {e}")
        return None


# Re-export calculate_recommendation for backward compatibility
from agents.agent_synthesizer import calculate_recommendation


def synthesize_review(
    paper: Paper,
    extractions: List[NoveltyRankedExtraction],
    config: Config
) -> Optional[Review]:
    """
    Standard synthesis without literature grounding (backward compatible).

    Delegates to the original synthesizer for backward compatibility.
    """
    from agents.agent_synthesizer import synthesize_review as _synthesize_review_standard

    # Convert to standard extractions if needed
    standard_extractions = []
    for e in extractions:
        # Create standard Extraction from NoveltyRankedExtraction
        standard_extractions.append(Extraction(
            paper_id=e.paper_id,
            criterion_id=e.criterion_id,
            score=e.score,
            score_justification=e.score_justification,
            evidence=e.evidence,
            strengths=e.strengths,
            weaknesses=e.weaknesses,
            confidence=e.confidence,
            model_used=e.model_used,
            extraction_timestamp=e.extraction_timestamp,
            cost=e.cost
        ))

    return _synthesize_review_standard(paper, standard_extractions, config)
