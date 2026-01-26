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


def _create_fallback_review(
    paper: Paper,
    extractions: List[NoveltyRankedExtraction],
    config: Config,
    base_score: float,
    breakdown: Dict[str, WeightedBreakdown],
    recommendation: str,
    rationale: str,
    novelty_adjusted_score: Optional[float] = None,
    literature_context: LiteratureContext = None,
    research_trajectory: str = ""
) -> GroundedReview:
    """
    Create a fallback review when LLM JSON parsing fails.

    This ensures the system always returns a valid review even when
    the LLM doesn't return properly formatted JSON.
    """
    from core.data_models import DetailedAssessment

    llm_config = config.get_llm_config()
    final_score = novelty_adjusted_score or base_score

    # Build criterion narratives from extractions
    criterion_narrative = {}
    for e in extractions:
        narrative = f"**Score: {e.score}/100**\n\n{e.score_justification}\n\n"
        if e.strengths:
            narrative += "**Strengths:**\n" + "\n".join(f"- {s}" for s in e.strengths[:3]) + "\n\n"
        if e.weaknesses:
            narrative += "**Weaknesses:**\n" + "\n".join(f"- {w}" for w in e.weaknesses[:3]) + "\n\n"
        criterion_narrative[e.criterion_id] = narrative

    # Build detailed assessment
    all_strengths = []
    all_weaknesses = []
    all_issues = []

    for e in extractions:
        all_strengths.extend(e.strengths[:2])
        all_weaknesses.extend(e.weaknesses[:2])
        if e.confidence < 0.7:
            all_issues.append(f"Low confidence in assessment for {e.criterion_id}")

    detailed_assessment = DetailedAssessment(
        major_strengths=all_strengths[:5],
        major_concerns=all_weaknesses[:5],
        minor_issues=all_issues[:3]
    )

    # Build revision suggestions
    revision_suggestions = []
    for e in extractions:
        if e.score < 70:
            revision_suggestions.append(f"Improve {e.criterion_id}: {e.weaknesses[0] if e.weaknesses else 'address concerns noted'}")
    if not revision_suggestions:
        revision_suggestions = ["Consider addressing minor reviewer comments"]

    # Calculate costs
    total_cost = sum(e.cost for e in extractions)

    extractor_model_name = extractions[0].model_used if extractions else "unknown_extractor"
    synthesizer_model_name = f"{llm_config['synthesizer_provider']}/{llm_config['synthesizer_model']}"

    # Build executive summary
    executive_summary = f"""This paper was evaluated across {len(extractions)} criteria. The overall score is {final_score:.1f}/100.

**Recommendation:** {recommendation}

**Rationale:** {rationale}

"""

    if research_trajectory:
        executive_summary += f"\n{research_trajectory}\n"

    # Create the review
    review = GroundedReview(
        paper_id=paper.id,
        paper_title=paper.metadata.title,
        paper_filename=paper.filename,
        overall_score=final_score,
        weighted_breakdown=breakdown,
        recommendation=recommendation,
        recommendation_rationale=rationale,
        executive_summary=executive_summary,
        detailed_assessment=detailed_assessment,
        criterion_narrative=criterion_narrative,
        revision_suggestions=revision_suggestions,
        decision_confidence=sum(e.confidence for e in extractions) / len(extractions) if extractions else 0.5,
        synthesizer_model_used=synthesizer_model_name,
        extractor_model_used=extractor_model_name,
        total_cost=total_cost,
        literature_context=literature_context or LiteratureContext(),
        research_trajectory_section=research_trajectory,
        novelty_adjusted_score=novelty_adjusted_score
    )

    print(f"[Critic] Created fallback review for {paper.filename}")
    return review


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

    # Call LLM with JSON parsing retry
    max_json_retries = 3
    review_data = None
    raw_response = None

    for json_attempt in range(max_json_retries):
        response = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            provider=llm_config['synthesizer_provider'],
            model=llm_config['synthesizer_model'],
            temperature=llm_config['temperature'],
            max_retries=llm_config['max_retries'],
            response_format="json"
        )

        if not response['success']:
            print(f"[Critic Error] LLM call failed for {paper.filename}: {response['error']}")
            return None

        raw_content = response['content']
        raw_response = raw_content
        response_len = len(raw_content)

        # Debug logging
        print(f"[Critic] Attempt {json_attempt + 1}/{max_json_retries}: Response length = {response_len} chars")
        print(f"[Critic] Model: {llm_config['synthesizer_provider']}/{llm_config['synthesizer_model']}")

        # Try to find JSON in the response
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')

        print(f"[Critic] JSON markers: start={start_index}, end={end_index}")

        if start_index == -1 or end_index == -1 or end_index < start_index:
            print(f"[Critic Warning] No valid JSON markers found")
            print(f"[Critic] Response preview (first 300 chars):\n{raw_content[:300]}")
            print(f"[Critic] Response preview (last 300 chars):\n{raw_content[-300:]}")

            # Check if response looks complete
            if len(raw_content) < 500:
                print(f"[Critic] Response suspiciously short - may be incomplete")

            if json_attempt < max_json_retries - 1:
                print(f"[Critic] Retrying with stricter instructions...")
                # Add stricter JSON requirement to prompt
                prompt += "\n\nIMPORTANT: You must respond with valid JSON only. No markdown, no text before/after the JSON."
                continue
            else:
                print(f"[Critic] All retries exhausted. Using fallback review.")
                # Create fallback review after all retries fail
                return _create_fallback_review(
                    paper=paper,
                    extractions=extractions,
                    config=config,
                    base_score=base_score,
                    breakdown=breakdown,
                    recommendation=recommendation,
                    rationale=rationale,
                    novelty_adjusted_score=novelty_adjusted_score,
                    literature_context=literature_context,
                    research_trajectory=research_trajectory
                )

        # Extract JSON
        json_text = raw_content[start_index:end_index + 1]
        json_len = len(json_text)
        print(f"[Critic] Extracted JSON: {json_len} chars")

        try:
            review_data = json.loads(json_text)
            print(f"[Critic] ✓ JSON parsed successfully on attempt {json_attempt + 1}")
            break  # Success - exit retry loop
        except json.JSONDecodeError as e:
            print(f"[Critic Warning] JSON parsing failed on attempt {json_attempt + 1}: {e}")
            print(f"[Critic] JSON preview (first 300 chars):\n{json_text[:300]}")

            if json_attempt < max_json_retries - 1:
                print(f"[Critic] Retrying...")
                prompt += "\n\nIMPORTANT: Ensure your JSON is properly formatted and complete."
                continue
            else:
                print(f"[Critic] All retries exhausted. Using fallback review.")
                return _create_fallback_review(
                    paper=paper,
                    extractions=extractions,
                    config=config,
                    base_score=base_score,
                    breakdown=breakdown,
                    recommendation=recommendation,
                    rationale=rationale,
                    novelty_adjusted_score=novelty_adjusted_score,
                    literature_context=literature_context,
                    research_trajectory=research_trajectory
                )

    # If we get here, JSON parsing succeeded
    if not review_data:
        print(f"[Critic Error] Unexpected: review_data is None after successful parsing")
        return None

    # Calculate costs
    total_extraction_cost = sum(e.cost for e in extractions)
    total_cost = total_extraction_cost + response.get('cost', 0.0)

    if 'criterion_narrative' not in review_data or not isinstance(review_data['criterion_narrative'], dict):
        print(f"[Critic Warning] 'criterion_narrative' not found or not a dict. Setting to empty.")
        review_data['criterion_narrative'] = {}

    extractor_model_name = extractions[0].model_used if extractions else "unknown_extractor"
    synthesizer_model_name = f"{llm_config['synthesizer_provider']}/{llm_config['synthesizer_model']}"

    # Use the novelty-adjusted score if available, otherwise use base score
    final_score = novelty_adjusted_score or base_score

    try:
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
    except ValidationError as e:
        print(f"[Critic Error] Pydantic validation failed for {paper.filename}: {e}")
        print(f"[Critic] Using fallback review due to validation error")
        return _create_fallback_review(
            paper=paper,
            extractions=extractions,
            config=config,
            base_score=base_score,
            breakdown=breakdown,
            recommendation=recommendation,
            rationale=rationale,
            novelty_adjusted_score=novelty_adjusted_score,
            literature_context=literature_context,
            research_trajectory=research_trajectory
        )


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
