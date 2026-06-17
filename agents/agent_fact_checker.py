"""
The Fact-Checker Agent

STAGE 3: Just-in-time verification of suspicious or bold claims.

Triggered when the Reader identifies claims that warrant verification:
- Claims of being "first" or "novel" without strong evidence
- Contradictions of highly-cited work
- Implausible methodological claims
- Missing citations for well-known results
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.data_models import (
    NoveltyRankedExtraction,
    FactCheckResult,
    RelatedPaperMetadata
)
from core.config_loader import Config
from core.llm_wrapper import call_llm
from core.literature_searcher import LiteratureSearcher, SearchConfig
from utilities.helpers import load_yaml_config


@dataclass
class FactCheckTrigger:
    """A condition that triggers a fact-check."""
    trigger_type: str
    description: str
    should_trigger: bool
    claim_text: str
    confidence: float = 0.0


def _analyze_for_triggers(
    extraction: NoveltyRankedExtraction,
    criterion: Dict[str, Any],
    config: Config,
    literature_config: Dict[str, Any]
) -> List[FactCheckTrigger]:
    """
    Analyze an extraction for conditions that should trigger fact-checking.

    Args:
        extraction: The novelty-ranked extraction to analyze
        criterion: The evaluation criterion
        config: System configuration
        literature_config: Literature grounding configuration

    Returns:
        List of triggers that should activate fact-checking
    """
    triggers = []
    fact_checker_config = literature_config.get('fact_checker', {})
    trigger_configs = fact_checker_config.get('triggers', {})

    # Check for "first study" claims
    if trigger_configs.get('first_study_claim', {}).get('enabled', False):
        confidence_threshold = trigger_configs['first_study_claim'].get('confidence_threshold', 0.7)

        # Look for first-study language in the extraction
        text = extraction.score_justification.lower()
        first_study_keywords = ['first', 'first study', 'novel', 'unprecedented', 'pioneering']

        has_first_claim = any(keyword in text for keyword in first_study_keywords)

        if has_first_claim and extraction.novelty_ranking >= 4 and extraction.confidence < confidence_threshold:
            triggers.append(FactCheckTrigger(
                trigger_type='first_study_claim',
                description='Claims to be first study with moderate confidence',
                should_trigger=True,
                claim_text=extraction.score_justification[:200],
                confidence=extraction.confidence
            ))

    # Check for contradictions with baseline
    if trigger_configs.get('contradiction_high_citation', {}).get('enabled', False):
        if extraction.contradicts_baseline:
            triggers.append(FactCheckTrigger(
                trigger_type='contradiction',
                description='Contradicts baseline findings',
                should_trigger=True,
                claim_text=extraction.score_justification[:200],
                confidence=extraction.confidence
            ))

    # Check for implausible claims (low confidence with high novelty claim)
    if trigger_configs.get('implausible_claim', {}).get('enabled', False):
        threshold = trigger_configs['implausible_claim'].get('confidence_threshold', 0.6)

        if extraction.novelty_ranking == 5 and extraction.confidence < threshold:
            triggers.append(FactCheckTrigger(
                trigger_type='implausible',
                description='Exceptional novelty claim with low confidence',
                should_trigger=True,
                claim_text=extraction.score_justification[:200],
                confidence=extraction.confidence
            ))

    return triggers


def _generate_verification_query(
    trigger: FactCheckTrigger,
    extraction: NoveltyRankedExtraction,
    criterion: Dict[str, Any]
) -> str:
    """
    Generate a search query to verify the triggered claim.

    Args:
        trigger: The fact-check trigger
        extraction: The extraction containing the claim
        criterion: The evaluation criterion

    Returns:
        Search query string
    """
    prompt = f"""
Generate a specific academic search query to verify this claim:

CLAIM: {trigger.claim_text}

CONTEXT:
- Criterion: {criterion['name']}
- Novelty Ranking: {extraction.novelty_ranking}/5
- Extraction Score: {extraction.score}/5

TASK:
Create a search query that would find prior work that:
1. Either supports this claim (confirming it's truly novel)
2. Or contradicts this claim (showing prior work exists)

The query should:
- Use academic terminology
- Be specific enough to return relevant papers
- Include key methodological or domain terms
- Be 3-8 words long

Return ONLY the search query, no explanation.
"""

    response = call_llm(
        prompt=prompt,
        system_prompt="You are an expert academic researcher skilled at crafting literature search queries.",
        provider=extraction.model_used.split('/')[0],
        model=extraction.model_used.split('/')[-1],
        temperature=0.3,
        max_retries=2,
        role="extraction"
    )

    if response['success']:
        # Clean up the query
        query = response['content'].strip()
        # Remove quotes if present
        query = query.strip('"\'')
        return query

    # Fallback: simple keyword extraction from the claim
    return " ".join(trigger.claim_text.split()[:6])


def _verify_claim(
    trigger: FactCheckTrigger,
    search_query: str,
    extraction: NoveltyRankedExtraction,
    config: Config,
    literature_config: Dict[str, Any]
) -> FactCheckResult:
    """
    Perform a verification search for a triggered claim.

    Args:
        trigger: The fact-check trigger
        search_query: The search query to use
        extraction: The extraction being verified
        config: System configuration
        literature_config: Literature configuration

    Returns:
        FactCheckResult with verification findings
    """
    semantic_config = literature_config.get('semantic_scholar', {})
    fact_checker_config = literature_config.get('fact_checker', {})

    # Initialize searcher
    search_config = SearchConfig(
        api_key=semantic_config.get('api_key'),
        base_url=semantic_config.get('base_url'),
        timeout=semantic_config.get('timeout', 30),
        max_retries=semantic_config.get('max_retries', 3)
    )
    searcher = LiteratureSearcher(search_config)

    # Search for prior work
    verification_config = fact_checker_config.get('verification_search', {})
    papers = searcher.search_by_keywords(
        keywords=search_query.split(),
        limit=verification_config.get('limit', 10),
        min_citation_count=verification_config.get('min_citation_count', 5)
    )

    # Analyze results
    found_prior_work = len(papers) > 0
    verification_status = "novel"  # Default

    if found_prior_work:
        # Check if any papers directly contradict the claim
        # For now, we'll mark as "disputed" if we found relevant papers
        verification_status = "disputed"
    else:
        verification_status = "confirmed"  # No prior work found, claim may be valid

    # Generate summary and recommendation
    if found_prior_work:
        prior_work_summary = f"Found {len(papers)} potentially relevant papers:\n"
        for p in papers[:5]:
            prior_work_summary += f"- {p.title} ({p.year or 'n.d.'}), {p.citation_count or 0} citations\n"
            if p.abstract:
                prior_work_summary += f"  Abstract: {p.abstract[:200]}...\n"

        recommendation = "Review these papers to determine if the claim is overstated. Consider reducing novelty ranking if prior work is highly relevant."
    else:
        prior_work_summary = "No prior work found matching this claim."
        recommendation = "No contradictory prior work found. The novelty claim may be valid."

    return FactCheckResult(
        claim=trigger.claim_text[:500],
        criterion_id=extraction.criterion_id,
        search_query=search_query,
        found_prior_work=found_prior_work,
        prior_work_summary=prior_work_summary[:1000],  # Limit length
        verification_status=verification_status,
        recommendation=recommendation,
        papers_found=papers[:5]  # Store top 5 for reference
    )


def run_fact_checks(
    extractions: List[NoveltyRankedExtraction],
    criteria: List[Dict[str, Any]],
    config: Config,
    literature_config: Optional[Dict[str, Any]] = None
) -> List[FactCheckResult]:
    """
    Run fact-checking on a list of extractions.

    Identifies triggers, performs verification searches, and returns findings.

    Args:
        extractions: List of novelty-ranked extractions
        criteria: List of evaluation criteria
        config: System configuration
        literature_config: Optional literature configuration

    Returns:
        List of FactCheckResult
    """
    if literature_config is None:
        literature_config = load_yaml_config("config/literature_sources.yaml")

    # Check if fact-checking is enabled
    if not literature_config.get('enabled', True):
        print("[Fact-Checker] Disabled by configuration")
        return []

    fact_checker_config = literature_config.get('fact_checker', {})
    max_per_section = fact_checker_config.get('max_verifications_per_section', 3)
    max_total = fact_checker_config.get('max_verifications_total', 10)

    results = []
    total_checks = 0

    # Create criteria map for easy lookup
    criteria_map = {c['id']: c for c in criteria}

    print("\n[Fact-Checker] Analyzing extractions for verification triggers...")

    for extraction in extractions:
        if total_checks >= max_total:
            print(f"[Fact-Checker] Reached maximum verification limit ({max_total})")
            break

        criterion = criteria_map.get(extraction.criterion_id)
        if not criterion:
            continue

        # Analyze for triggers
        triggers = _analyze_for_triggers(extraction, criterion, config, literature_config)

        # Process triggers (limit per criterion)
        for i, trigger in enumerate(triggers):
            if total_checks >= max_total or i >= max_per_section:
                break

            if not trigger.should_trigger:
                continue

            print(f"[Fact-Checker] Trigger: {trigger.trigger_type} for {extraction.criterion_id}")

            # Generate verification query
            search_query = _generate_verification_query(trigger, extraction, criterion)
            print(f"[Fact-Checker] Query: {search_query}")

            # Perform verification
            result = _verify_claim(
                trigger=trigger,
                search_query=search_query,
                extraction=extraction,
                config=config,
                literature_config=literature_config
            )

            results.append(result)
            total_checks += 1

            print(f"[Fact-Checker] Result: {result.verification_status} - {len(result.papers_found)} papers found")

    print(f"[Fact-Checker] Completed {len(results)} verification checks\n")

    return results


def summarize_fact_checks(fact_checks: List[FactCheckResult]) -> str:
    """
    Generate a summary of fact-check results for inclusion in the review.

    Args:
        fact_checks: List of FactCheckResult

    Returns:
        Formatted summary string
    """
    if not fact_checks:
        return "No fact-checks performed."

    summary = "## Literature Verification Summary\n\n"

    disputed = [fc for fc in fact_checks if fc.verification_status == "disputed"]
    confirmed = [fc for fc in fact_checks if fc.verification_status == "confirmed"]
    novel = [fc for fc in fact_checks if fc.verification_status == "novel"]

    if disputed:
        summary += f"### ⚠️ Claims Requiring Review ({len(disputed)})\n\n"
        for fc in disputed:
            summary += f"**{fc.criterion_id}**: {fc.claim[:100]}...\n\n"
            summary += f"- Search: \"{fc.search_query}\"\n"
            summary += f"- Finding: {fc.papers_found} relevant papers found\n"
            summary += f"- Recommendation: {fc.recommendation}\n\n"

    if confirmed:
        summary += f"### ✓ Verified Novel Claims ({len(confirmed)})\n\n"
        for fc in confirmed:
            summary += f"**{fc.criterion_id}**: {fc.claim[:100]}...\n"
            summary += f"- No contradictory prior work found\n\n"

    if novel:
        summary += f"### 🔍 Pending Verification ({len(novel)})\n\n"
        for fc in novel:
            summary += f"**{fc.criterion_id}**: {fc.claim[:100]}...\n\n"

    return summary
