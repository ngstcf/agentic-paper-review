"""
The Reader Agent (Literature-Optional Extractor)

STAGE 2: Extracts findings from the target paper.

When a BaselineReference is provided, this agent also ranks novelty against
the literature. Without a baseline, it behaves identically to the standard
Extractor agent.

This makes literature grounding completely optional - the system gracefully
degrades to standard review mode when baseline is None.
"""

import json
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from core.data_models import (
    Paper,
    Extraction,
    NoveltyRankedExtraction,
    BaselineReference
)
from core.config_loader import Config
from core.llm_wrapper import call_llm
from utilities.helpers import get_scale_definition


def _build_literature_context(baseline: BaselineReference) -> str:
    """
    Build literature context for injection into standard prompts.

    This is appended to the standard prompt when baseline is available.

    Args:
        baseline: The BaselineReference from the Librarian

    Returns:
        Formatted string for inclusion in prompts
    """
    context = f"""

## LITERATURE CONTEXT (For Novelty Assessment)

### Research Sub-Topic: {baseline.sub_topic}

### State of the Art
{baseline.key_findings_summary}

### Key Baseline Papers (sorted by citation count)
"""

    for i, paper in enumerate(baseline.baseline_papers, 1):
        context += f"\n{i}. **{paper.title}** ({paper.year or 'n.d.'})\n"
        context += f"   - Authors: {', '.join(paper.authors[:3])}"
        if len(paper.authors) > 3:
            context += " et al."
        context += f"\n   - Citations: {paper.citation_count or 0}\n"

        if paper.key_findings:
            context += f"   - Key Findings:\n"
            for finding in paper.key_findings:
                context += f"     • {finding}\n"

    context += f"""

### Additional Instructions

When scoring this criterion, also assess:
1. **Novelty Ranking (1-5)**: How novel is this paper's approach compared to the baseline above?
   - 1: No novelty - completely replicates prior work
   - 2: Marginal novelty - minor extension of prior work
   - 3: Moderate novelty - builds on prior work with some new insights
   - 4: High novelty - significant advance over prior work
   - 5: Exceptional novelty - groundbreaking, paradigm-shifting work

2. **Contradicts Baseline?** (true/false): Does this paper contradict well-established findings?

3. **Extends Baseline?** (true/false): Does this paper address gaps left by prior work?

4. **Prior Work Gaps** (list): What specific questions do baseline papers NOT answer that this paper does?

Include these fields in your JSON response:
- novelty_ranking (int 1-5)
- contradicts_baseline (boolean)
- extends_baseline (boolean)
- prior_work_gaps (list of strings)
"""

    return context


def extract_criterion_evidence(
    paper: Paper,
    criterion: Dict[str, Any],
    config: Config,
    baseline: Optional[BaselineReference] = None
) -> Optional[NoveltyRankedExtraction]:
    """
    Extract evidence for a single criterion.

    When baseline is provided, also assesses novelty against literature.
    Without baseline, behaves like standard extractor.

    Args:
        paper: The target paper
        criterion: The evaluation criterion
        config: System configuration
        baseline: Optional BaselineReference for novelty comparison

    Returns:
        NoveltyRankedExtraction with standard assessment plus optional novelty ranking
    """
    llm_config = config.get_llm_config()
    prompt_template = config.get_prompt("extractor_user")
    system_prompt = config.get_prompt("extractor_system").format(domain=config.domain)

    # Build standard prompt
    max_content_tokens = config.get_system_config()['max_content_tokens']
    paper_content = paper.content_markdown
    if len(paper_content) > max_content_tokens * 4:
        print(f"[Reader Warning] Truncating paper content for {criterion['id']}")
        paper_content = paper_content[:max_content_tokens * 4]

    prompt = prompt_template.format(
        paper_markdown=paper_content,
        criterion_name=criterion['name'],
        criterion_description=criterion['description'],
        sub_questions="\n".join(f"- {q}" for q in criterion.get('sub_questions', [])),
        scale_definition=get_scale_definition(criterion.get('scale', {})),
        domain=config.domain
    )

    # Inject literature context if baseline is available
    if baseline:
        prompt += _build_literature_context(baseline)

    response = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=llm_config['extractor_provider'],
        model=llm_config['extractor_model'],
        temperature=llm_config['temperature'],
        max_retries=llm_config['max_retries']
    )

    if not response['success']:
        print(f"[Reader Error] LLM call failed for {paper.filename} on {criterion['id']}: {response['error']}")
        return None

    json_text = ""
    try:
        raw_content = response['content']

        # Robust JSON parsing
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')

        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise json.JSONDecodeError("Could not find JSON object markers", raw_content, 0)

        json_text = raw_content[start_index:end_index + 1]
        extraction_data = json.loads(json_text)

        # Create NoveltyRankedExtraction
        extraction = NoveltyRankedExtraction(
            paper_id=paper.id,
            criterion_id=criterion['id'],
            model_used=f"{llm_config['extractor_provider']}/{llm_config['extractor_model']}",
            cost=response['cost'],
            **extraction_data
        )

        # Set default novelty values if not provided (backward compatibility)
        if baseline and 'novelty_ranking' not in extraction_data:
            extraction.novelty_ranking = 3  # Default to moderate
        if not baseline:
            # Reset novelty defaults when no baseline provided
            extraction.novelty_ranking = 3
            extraction.contradicts_baseline = False
            extraction.extends_baseline = False
            extraction.prior_work_gaps = []

        return extraction

    except json.JSONDecodeError as e:
        print(f"[Reader Error] Failed to parse JSON for {paper.filename} on {criterion['id']}: {e}")
        return None
    except ValidationError as e:
        print(f"[Reader Error] Pydantic validation failed for {paper.filename} on {criterion['id']}: {e}")
        return None


def process_paper_extractions(
    paper: Paper,
    config: Config,
    baseline: Optional[BaselineReference] = None
) -> List[NoveltyRankedExtraction]:
    """
    Run extraction on all criteria for a single paper.

    When baseline is provided, includes novelty assessment.
    Without baseline, returns standard extractions (with default novelty values).

    Args:
        paper: The target paper
        config: System configuration
        baseline: Optional BaselineReference for novelty comparison

    Returns:
        List of NoveltyRankedExtraction for all criteria
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    criteria = config.get_criteria()
    llm_config = config.get_llm_config()
    extractions = []

    mode = "LITERATURE-GROUNDED" if baseline else "STANDARD"
    print(f"[Reader] [{mode}] Starting extraction for: {paper.filename}")

    with ThreadPoolExecutor(max_workers=llm_config['max_parallel']) as executor:
        futures = {
            executor.submit(
                extract_criterion_evidence,
                paper,
                c,
                config,
                baseline
            ): c
            for c in criteria
        }

        for future in as_completed(futures):
            criterion = futures[future]
            try:
                result = future.result()
                if result:
                    extractions.append(result)
                    novelty_info = f" (novelty: {result.novelty_ranking}/5)" if baseline else ""
                    print(f"  > Completed: {criterion['id']}{novelty_info}")
                else:
                    print(f"  > FAILED: {criterion['id']}")
            except Exception as e:
                print(f"[Reader Error] Thread failed for {criterion['id']}: {e}")

    return sorted(extractions, key=lambda e: e.criterion_id)


# ============================================================================
# BACKWARD COMPATIBILITY: Import from original extractor
# ============================================================================

# Re-export for backward compatibility with existing code that uses agent_extractor
from agents.agent_extractor import (
    build_extraction_prompt,
    process_paper_extractions as process_paper_extractions_standard
)
