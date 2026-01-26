"""
The Librarian Agent

STAGE 1: Creates a Baseline Reference by searching for the most cited papers
in the target paper's sub-topic and extracting their key findings.
"""

import json
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from core.data_models import Paper, BaselineReference, RelatedPaperMetadata
from core.config_loader import Config
from core.literature_searcher import LiteratureSearcher, SearchConfig
from core.world_bank_searcher import WorldBankSearcher, WorldBankSearchConfig
from core.llm_wrapper import call_llm
from utilities.helpers import load_yaml_config


def _extract_search_keywords(
    paper: Paper,
    config: Config
) -> List[str]:
    """
    Extract search keywords from the target paper.

    Uses LLM to identify the specific sub-topic and generate search keywords.

    Args:
        paper: The target paper to analyze
        config: System configuration

    Returns:
        List of search keywords for finding related papers
    """
    prompt = f"""
Analyze this paper's title and abstract to extract 5-8 specific search keywords
that would help find the most relevant related work in this field.

PAPER TITLE: {paper.metadata.title}

ABSTRACT: {paper.metadata.abstract}

TASK:
1. Identify the main research sub-topic (e.g., "causal inference in development economics")
2. Extract 5-8 specific keywords that would return highly relevant papers
3. Focus on methodological terms and domain-specific concepts
4. Avoid generic terms like "study", "analysis", "research"

Return your answer as a JSON object:
{{
    "sub_topic": "brief description of the sub-topic",
    "keywords": ["keyword1", "keyword2", "keyword3", ...]
}}
"""

    response = call_llm(
        prompt=prompt,
        system_prompt="You are an expert academic librarian who specializes in identifying research sub-topics and generating effective search queries.",
        provider=config.get_llm_config()['extractor_provider'],
        model=config.get_llm_config()['extractor_model'],
        temperature=0.3,  # Lower temperature for consistent keyword extraction
        max_retries=config.get_llm_config()['max_retries']
    )

    if not response['success']:
        print(f"[Librarian Error] Failed to extract keywords: {response['error']}")
        # Fallback: simple keyword extraction from title
        return paper.metadata.title.lower().split()[:5]

    try:
        start_index = response['content'].find('{')
        end_index = response['content'].rfind('}')
        json_text = response['content'][start_index:end_index + 1]
        result = json.loads(json_text)
        return result.get('keywords', [])
    except Exception as e:
        print(f"[Librarian Error] Failed to parse keyword extraction: {e}")
        return paper.metadata.title.lower().split()[:5]


def _extract_key_findings(
    paper_metadata: RelatedPaperMetadata,
    config: Config
) -> List[str]:
    """
    Extract key findings from a baseline paper's abstract.

    Args:
        paper_metadata: The baseline paper metadata
        config: System configuration

    Returns:
        List of 2-3 key findings from the paper
    """
    if not paper_metadata.abstract:
        return ["Abstract not available"]

    prompt = f"""
Extract the 2-3 most important findings or contributions from this paper.

TITLE: {paper_metadata.title}
AUTHORS: {', '.join(paper_metadata.authors)}
ABSTRACT:
{paper_metadata.abstract}

TASK:
Identify the 2-3 most significant findings, contributions, or claims made in this paper.
Focus on substantive results, not methodology or context.

Return your answer as a JSON object:
{{
    "key_findings": [
        "First key finding...",
        "Second key finding...",
        "Third key finding (if applicable)..."
    ]
}}
"""

    response = call_llm(
        prompt=prompt,
        system_prompt="You are an expert at identifying the core contributions of academic papers.",
        provider=config.get_llm_config()['extractor_provider'],
        model=config.get_llm_config()['extractor_model'],
        temperature=0.3,
        max_retries=config.get_llm_config()['max_retries']
    )

    if not response['success']:
        return ["Failed to extract findings"]

    try:
        start_index = response['content'].find('{')
        end_index = response['content'].rfind('}')
        json_text = response['content'][start_index:end_index + 1]
        result = json.loads(json_text)
        return result.get('key_findings', ["No clear findings identified"])
    except Exception:
        return ["Failed to parse findings"]


def _get_active_sources(
    literature_config: Dict[str, Any]
) -> List[str]:
    """
    Get list of active literature sources.

    Users enable/disable sources in config/literature_sources.yaml:
    sources:
      semantic_scholar:
        enabled: true
      world_bank:
        enabled: false

    Args:
        literature_config: Literature configuration

    Returns:
        List of enabled source names
    """
    sources_config = literature_config.get("sources", {})
    active = []

    # Check Semantic Scholar
    if sources_config.get("semantic_scholar", {}).get("enabled", True):
        active.append("semantic_scholar")

    # Check World Bank
    if sources_config.get("world_bank", {}).get("enabled", False):
        active.append("world_bank")

    return active


def _deduplicate_papers(
    papers: List[RelatedPaperMetadata]
) -> List[RelatedPaperMetadata]:
    """
    Deduplicate papers by title (case-insensitive).

    Args:
        papers: List of papers to deduplicate

    Returns:
        Deduplicated list
    """
    seen = set()
    deduped = []
    for paper in papers:
        title_lower = paper.title.lower().strip()
        if title_lower not in seen:
            seen.add(title_lower)
            deduped.append(paper)
    return deduped


def _search_multiple_sources(
    keywords: List[str],
    recency_years: int,
    limit: int,
    literature_config: Dict[str, Any]
) -> List[RelatedPaperMetadata]:
    """
    Search multiple literature sources and combine results.

    Args:
        keywords: Search keywords
        recency_years: Years to look back
        limit: Total number of papers to return
        literature_config: Literature configuration

    Returns:
        Combined and deduplicated list of papers
    """
    all_papers = []
    active_sources = _get_active_sources(literature_config)
    print(f"[Librarian] Active sources: {', '.join(active_sources)}")

    # Search Semantic Scholar
    if "semantic_scholar" in active_sources:
        semantic_config = literature_config.get("sources", {}).get("semantic_scholar", {})
        if semantic_config.get("enabled", True):
            print("[Librarian] Searching Semantic Scholar...")
            search_config = SearchConfig(
                api_key=semantic_config.get('api_key'),
                base_url=semantic_config.get('base_url', "https://api.semanticscholar.org/graph/v1"),
                timeout=semantic_config.get('timeout', 30),
                max_retries=semantic_config.get('max_retries', 3)
            )
            searcher = LiteratureSearcher(search_config)

            semantic_papers = searcher.get_most_cited(
                field_keywords=keywords[:8],
                years=recency_years,
                limit=limit
            )
            all_papers.extend(semantic_papers)
            print(f"[Librarian]   Semantic Scholar: {len(semantic_papers)} papers")

    # Search World Bank
    if "world_bank" in active_sources:
        wb_config = literature_config.get("sources", {}).get("world_bank", {})
        if wb_config.get("enabled", False):
            import time as time_module
            time_module.sleep(1.0)  # Delay before World Bank search

            print("[Librarian] Searching World Bank...")
            wb_search_config = WorldBankSearchConfig(
                base_url=wb_config.get('base_url', "https://search.worldbank.org/api/v3/wds"),
                timeout=wb_config.get('timeout', 30),
                max_retries=wb_config.get('max_retries', 3)
            )
            wb_searcher = WorldBankSearcher(wb_search_config)

            # Calculate date range for World Bank
            from datetime import datetime, timezone
            current_year = datetime.now(timezone.utc).year
            start_date = f"{current_year - recency_years}-01-01"
            end_date = f"{current_year}-12-31"

            wb_papers = wb_searcher.search_by_keywords(
                keywords=keywords[:5],
                limit=limit // 2,  # Get fewer from World Bank
                start_date=start_date,
                end_date=end_date
            )
            all_papers.extend(wb_papers)
            print(f"[Librarian]   World Bank: {len(wb_papers)} papers")

    # Deduplicate by title
    all_papers = _deduplicate_papers(all_papers)
    print(f"[Librarian] Total after deduplication: {len(all_papers)} papers")

    return all_papers


def _generate_baseline_summary(
    sub_topic: str,
    baseline_papers: List[RelatedPaperMetadata],
    config: Config
) -> str:
    """
    Generate a summary of the state of the art from baseline papers.

    Args:
        sub_topic: The research sub-topic
        baseline_papers: List of baseline papers with key findings
        config: System configuration

    Returns:
        Summary of the current state of knowledge
    """
    # Build a summary of baseline papers
    papers_summary = ""
    for i, paper in enumerate(baseline_papers, 1):
        papers_summary += f"\n{i}. {paper.title} ({paper.year or 'n.d.'})\n"
        papers_summary += f"   Citations: {paper.citation_count or 0}\n"
        if paper.key_findings:
            papers_summary += f"   Key Findings:\n"
            for finding in paper.key_findings:
                papers_summary += f"   - {finding}\n"

    prompt = f"""
Synthesize the following baseline papers into a coherent summary of the current
state of knowledge in this research area.

RESEARCH SUB-TOPIC: {sub_topic}

BASELINE PAPERS (sorted by citation count):
{papers_summary}

TASK:
Write a concise paragraph (3-5 sentences) that:
1. Describes the current state of knowledge in this area
2. Identifies the main approaches or methodologies used
3. Highlights what is well-established vs. what is still being explored
4. Identifies any gaps or open questions

This summary will help readers understand where new papers fit in the research trajectory.
"""

    response = call_llm(
        prompt=prompt,
        system_prompt="You are an expert at synthesizing academic literature and identifying research trends.",
        provider=config.get_llm_config()['extractor_provider'],
        model=config.get_llm_config()['extractor_model'],
        temperature=0.5,
        max_retries=config.get_llm_config()['max_retries']
    )

    if not response['success']:
        return "Failed to generate baseline summary"

    return response['content'].strip()


def create_baseline_reference(
    paper: Paper,
    config: Config,
    literature_config: Optional[Dict[str, Any]] = None
) -> Optional[BaselineReference]:
    """
    The Librarian Agent: Create a Baseline Reference for literature comparison.

    Process:
    1. Extract search keywords from the target paper
    2. Search for the most cited papers in the sub-topic
    3. Extract key findings from each baseline paper
    4. Generate a summary of the state of the art

    Args:
        paper: The target paper
        config: System configuration
        literature_config: Optional literature-specific configuration

    Returns:
        BaselineReference with baseline papers and state of the art summary
    """
    print(f"\n[Librarian] Creating baseline reference for: {paper.filename}")

    # Load literature configuration
    if literature_config is None:
        literature_config = load_yaml_config("config/literature_sources.yaml")

    librarian_config = literature_config.get('librarian', {})

    # Step 1: Extract search keywords
    print("[Librarian] Step 1: Extracting search keywords...")
    keywords = _extract_search_keywords(paper, config)
    print(f"[Librarian]   Keywords: {', '.join(keywords[:5])}")

    # Also try to get keywords from metadata if available
    if paper.metadata.keywords:
        keywords.extend(paper.metadata.keywords)
        # Remove duplicates while preserving order
        seen = set()
        keywords = [x for x in keywords if not (x in seen or seen.add(x))]

    # Step 2: Search multiple sources for papers
    print("[Librarian] Step 2: Searching literature sources...")
    baseline_count = librarian_config.get('baseline_papers_count', 5)
    recency_years = librarian_config.get('recency_years', 5)

    # Add delay before search to avoid rate limits
    import time as time_module
    time_module.sleep(1.0)

    baseline_papers = _search_multiple_sources(
        keywords=keywords,
        recency_years=recency_years,
        limit=baseline_count * 2,  # Get more to filter
        literature_config=literature_config
    )

    if not baseline_papers:
        print("[Librarian] Warning: No baseline papers found. Using broader search...")
        # Add delay before retry
        time_module.sleep(2.0)
        # Try with just the first few keywords and broader year range
        baseline_papers = _search_multiple_sources(
            keywords=keywords[:3],
            recency_years=recency_years + 2,
            limit=baseline_count,
            literature_config=literature_config
        )

    # Sort by citation count and take top N
    baseline_papers = sorted(
        baseline_papers,
        key=lambda p: p.citation_count or 0,
        reverse=True
    )[:baseline_count]

    print(f"[Librarian]   Found {len(baseline_papers)} baseline papers")

    # Step 3: Extract key findings from baseline papers
    if librarian_config.get('extract_key_findings', True):
        print("[Librarian] Step 3: Extracting key findings...")
        for i, bp in enumerate(baseline_papers, 1):
            print(f"[Librarian]   Processing paper {i}/{len(baseline_papers)}: {bp.title[:50]}...")
            findings = _extract_key_findings(bp, config)
            bp.key_findings = findings

    # Step 4: Generate baseline summary
    print("[Librarian] Step 4: Generating baseline summary...")
    sub_topic = " ".join(keywords[:3])  # Simple sub-topic representation
    summary = _generate_baseline_summary(sub_topic, baseline_papers, config)

    # Create baseline reference
    baseline_ref = BaselineReference(
        sub_topic=sub_topic,
        query_keywords=keywords[:8],
        baseline_papers=baseline_papers,
        key_findings_summary=summary,
        total_api_calls=searcher.get_api_call_count()
    )

    print(f"[Librarian] Baseline reference created with {searcher.get_api_call_count()} API calls")

    return baseline_ref


def create_baseline_reference_batch(
    papers: List[Paper],
    config: Config
) -> Dict[str, BaselineReference]:
    """
    Create baseline references for multiple papers.

    Note: This can be optimized by caching and sharing baseline references
    for papers on similar topics.

    Args:
        papers: List of target papers
        config: System configuration

    Returns:
        Dictionary mapping paper_id to BaselineReference
    """
    results = {}

    for paper in papers:
        baseline = create_baseline_reference(paper, config)
        if baseline:
            results[paper.id] = baseline

    return results
