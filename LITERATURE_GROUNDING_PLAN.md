# Literature Grounding Module: Implementation Plan

## Overview

This document outlines the implementation of a multi-stage literature grounding system for the Agentic Academic Review System. The system adds four new agent types that work together to ground paper reviews in the existing literature.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PAPER INGESTION                                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: BASELINE REFERENCE                                         │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Agent: The Librarian                                          │   │
│ │ Task: Search for the 5 most cited papers in the sub-topic    │   │
│ │ Output: Baseline Reference (key findings, citation counts)    │   │
│ └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: NOVELTY-RANKED EXTRACTION                                  │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Agent: The Reader (modified Extractor)                       │   │
│ │ Task: Extract findings from target paper                     │   │
│ │       Use Baseline Reference to rank novelty (1-5)           │   │
│ │ Output: Evidence scores + Novelty Ranking                     │   │
│ └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Bold claim?     │
                    │   Suspicious?     │
                    └─────────┬─────────┘
                         Yes  │  No
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: JUST-IN-TIME VERIFICATION (Conditional)                    │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Agent: The Fact-Checker                                      │   │
│ │ Task: Triggered search for suspicious claims                │   │
│ │       Check if paper ignores prior work                     │   │
│ │ Limit: Max 3 external verification searches per section     │   │
│ │ Output: Verification reports (confirmed, disputed, novel)    │   │
│ └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: SYNTHESIS WITH RESEARCH TRAJECTORY                         │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Agent: The Critic (modified Synthesizer)                    │   │
│ │ Task: Compile final review                                   │   │
│ │       Highlight where paper fits in research trajectory      │   │
│ │       Incorporate novelty rankings and verifications         │   │
│ │ Output: Final review with literature-grounded assessment     │   │
│ └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Core Module: Literature Searcher

**File**: `core/literature_searcher.py`

**Responsibilities**:
- Interface with Semantic Scholar API
- Search by keywords, paper title, abstract similarity
- Fetch paper metadata (abstract, citation count, authors)
- Download full PDFs for open-access papers

**Key Classes**:
```python
class LiteratureSearcher:
    """Search academic databases for related papers."""

    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[PaperMetadata]
    def search_by_abstract_similarity(self, abstract: str, limit: int = 10) -> List[PaperMetadata]
    def get_most_cited(self, field: str, years: int = 5, limit: int = 5) -> List[PaperMetadata]
    def fetch_abstract(self, paper_id: str) -> str
    def fetch_full_text(self, paper_id: str) -> Optional[str]  # Open access only
```

**API Details**:
- **Semantic Scholar API**: Free tier, 100 requests/minute
- **Endpoint**: `https://api.semanticscholar.org/graph/v1`
- **Fields**: `paperId`, `title`, `abstract`, `citationCount`, `openAccessPdf`, `authors`
- **Rate Limiting**: Built-in retry with exponential backoff

### 2. Agent: The Librarian

**File**: `agents/agent_librarian.py`

**Responsibilities**:
- Extract keywords/sub-topic from paper title, abstract, and keywords section
- Search for the 5 most cited papers in the sub-topic
- Extract key findings from baseline papers
- Create a structured "Baseline Reference" document

**Input**:
- Paper metadata (title, abstract, keywords)
- Domain configuration

**Output**:
```python
class BaselineReference:
    sub_topic: str
    query_keywords: List[str]
    baseline_papers: List[BaselinePaper]
    key_findings_summary: str
    created_at: datetime

class BaselinePaper:
    paper_id: str
    title: str
    authors: List[str]
    year: int
    citation_count: int
    abstract: str
    key_findings: List[str]  # Extracted by LLM
    open_access: bool
```

**Prompt Strategy**:
```
Given the target paper's title and abstract:
1. Extract 3-5 specific sub-topic keywords
2. Use these to find the 5 most cited papers
3. For each baseline paper, extract the 2-3 key findings
4. Summarize the current state of knowledge in this sub-topic
```

### 3. Agent: The Reader (Modified Extractor)

**File**: `agents/agent_reader.py` (extends `agent_extractor.py`)

**Modifications to Existing Extractor**:
- Takes `BaselineReference` as additional input
- For each criterion extraction, includes a novelty ranking (1-5)
- Compares extracted evidence against baseline findings

**New Output Structure**:
```python
class NoveltyRankedExtraction(CriterionExtraction):
    # Existing fields...
    novelty_ranking: int  # 1-5, compared to baseline
    novelty_rationale: str  # Why this ranking
    contradicts_baseline: bool
    extends_baseline: bool
    prior_work_gaps: List[str]  # What baseline papers don't address
```

**Prompt Addition**:
```
BASELINE REFERENCE:
{baseline_reference}

When extracting evidence for this criterion, assess:
1. How does this paper's approach differ from the baseline? (novelty_ranking)
2. Does it contradict any established findings? (contradicts_baseline)
3. Does it address gaps left by prior work? (extends_baseline)
4. What questions do baseline papers NOT answer that this paper does? (prior_work_gaps)
```

### 4. Agent: The Fact-Checker

**File**: `agents/agent_fact_checker.py`

**Responsibilities**:
- Monitor extraction output for suspicious/bold claims
- Trigger targeted searches when novelty claim seems unjustified
- Verify if prior work already addressed the claim
- Limit: Max 3 verification searches per review section

**Trigger Conditions** (configurable):
- Novelty ranking = 5 with weak evidence
- Claims of "first", "novel", "unprecedented" without citations
- Contradiction of well-cited findings (>100 citations)
- Methodological claims that seem implausible

**Verification Process**:
```python
class FactCheckResult:
    claim: str
    search_query: str
    found_prior_work: bool
    prior_work_summary: str
    verification_status: Literal["confirmed", "disputed", "novel"]
    recommendation: str  # How to handle this in review
```

**Output Example**:
```
CLAIM: "This is the first study to apply causal inference to this topic"
SEARCH: "causal inference [topic]"
RESULT: DISPUTED
PRIOR WORK: "Smith et al. (2021) applied instrumental variables to similar
             research questions in Journal of Applied Economics"
RECOMMENDATION: "Reduce novelty ranking; request citation of Smith et al."
```

### 5. Agent: The Critic (Modified Synthesizer)

**File**: `agents/agent_critic.py` (extends `agent_synthesizer.py`)

**Modifications to Existing Synthesizer**:
- Incorporates Baseline Reference into review context
- Includes a "Research Trajectory" section
- Weights novelty rankings into final score
- Presents fact-checker findings with recommendations

**New Review Sections**:
```markdown
## Literature Context and Novelty Assessment

### State of the Art
[Summary from Baseline Reference]

### Position in Research Trajectory
This paper [extends/builds on/pivots away from] the established line of
research represented by [Baseline Paper 1], [Baseline Paper 2], and others.

Key distinctions:
- [Novel finding 1] vs [baseline finding]
- [Methodological advance]
- [Gap addressed]

### Verification Notes
[Fact-checker results, if any]
```

### 6. Configuration

**File**: `config/literature_sources.yaml`

```yaml
literature_search:
  primary_api: "semantic_scholar"
  semantic_scholar:
    base_url: "https://api.semanticscholar.org/graph/v1"
    api_key: null  # Free tier doesn't require key
    timeout: 30
    max_retries: 3

librarian:
  baseline_papers_count: 5
  citation_count_min: 10  # Minimum citations for baseline
  recency_years: 5        # Search within last N years
  extract_key_findings: true

fact_checker:
  max_verifications_per_section: 3
  trigger_on_novelty_claim: true
  trigger_on_first_study_claim: true
  trigger_on_contradiction_high_citation: true
  high_citation_threshold: 100
```

### 7. Data Models

**File**: `core/data_models.py` (extensions)

```python
@dataclass
class LiteratureContext:
    """All literature-related information for a review."""
    baseline_reference: BaselineReference
    novelty_rankings: Dict[str, NoveltyRankedExtraction]
    fact_checks: List[FactCheckResult]
    search_queries_made: List[str]
    total_api_calls: int

@dataclass
class GroundedReview(Review):
    """Extended review with literature grounding."""
    literature_context: LiteratureContext
    research_trajectory_section: str
    novelty_adjusted_score: float
```

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)
- [ ] Create `core/literature_searcher.py` with Semantic Scholar integration
- [ ] Add literature data models to `core/data_models.py`
- [ ] Create `config/literature_sources.yaml`
- [ ] Write tests for searcher module

### Phase 2: The Librarian (Stage 1)
- [ ] Implement `agents/agent_librarian.py`
- [ ] Create baseline reference generation logic
- [ ] Add Librarian to workflow pipeline
- [ ] Test with sample papers

### Phase 3: The Reader (Stage 2)
- [ ] Modify `agents/agent_extractor.py` → `agents/agent_reader.py`
- [ ] Add novelty ranking to extraction prompts
- [ ] Update data structures for ranked extractions
- [ ] Test novelty assessment accuracy

### Phase 4: The Fact-Checker (Stage 3)
- [ ] Implement `agents/agent_fact_checker.py`
- [ ] Add claim detection logic
- [ ] Implement verification workflow with rate limiting
- [ ] Test triggered search functionality

### Phase 5: The Critic (Stage 4)
- [ ] Modify `agents/agent_synthesizer.py` → `agents/agent_critic.py`
- [ ] Add research trajectory section to review
- [ ] Integrate fact-checker results
- [ ] Update output formatting

### Phase 6: Workflow Integration
- [ ] Update `run_review_with_dir.py` with 4-stage pipeline
- [ ] Add progress tracking for literature stages
- [ ] Update output generator for new review sections
- [ ] Add cost tracking for literature searches

### Phase 7: Testing and Validation
- [ ] Test end-to-end with sample papers
- [ ] Validate against known literature
- [ ] Calibrate novelty rankings
- [ ] Optimize API usage and caching

## API Usage Estimates

### Per Paper (Typical)

| Stage | API Calls | Cost (USD) |
|-------|-----------|------------|
| Librarian search | 1-2 | Free |
| Librarian abstracts | 5 | Free |
| Fact-checker (conditional) | 0-3 | Free |
| **Total** | **6-10** | **Free** |

### Batch Processing (100 papers)

| Metric | Estimate |
|--------|----------|
| Total API calls | 600-1000 |
| Processing time | ~30-60 min |
| Semantic Scholar free tier | 5000 calls/day (✓ sufficient) |

## Success Metrics

1. **Novelty Detection Accuracy**: Can the system correctly identify truly novel contributions?
2. **Citation Gap Detection**: Can it find missing prior work?
3. **Processing Efficiency**: Does it add <5 minutes per paper?
4. **Review Quality Improvement**: Do literature-grounded reviews catch more issues?

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| API rate limits | Implement caching, batch requests, fallback strategies |
| False novelty claims | Require multiple citations; human verification of 5-star ratings |
| Paywall papers | Focus on open access; use abstracts when full text unavailable |
| Computational cost | Cache literature search results per sub-topic |
| Domain specificity | Use domain-specific keywords; fine-tune search queries |

## Future Enhancements

1. **Multi-source integration**: Add arXiv, PubMed, OpenAlex
2. **Citation graph analysis**: Track influence between papers
3. **Temporal analysis**: Understand research evolution over time
4. **Author expertise matching**: Find reviewers based on citation networks
5. **Literature database**: Cache and reuse baseline references across reviews

## Dependencies

```python
# New requirements
requests>=2.31.0      # HTTP client for Semantic Scholar
tenacity>=8.2.3      # Retry logic with exponential backoff
aiohttp>=3.9.0       # Async HTTP (optional, for performance)
```

---

**Next Steps**: Review this plan and proceed with Phase 1 implementation upon approval.
