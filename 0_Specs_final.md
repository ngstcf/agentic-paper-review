# Academic Review System - Technical Specification

**Version:** 1.0  
**Date:** 2025-11-01  
**Primary Domain:** Development Economics (Generalizable to other disciplines)  
**Target Scale:** 500+ papers per review cycle

---

## 1. System Overview

### 1.1 Purpose
An automated academic paper review system that uses Large Language Models (LLMs) to evaluate research papers against customizable criteria, producing structured reviews with evidence-based scoring and recommendations.

### 1.2 Core Capabilities
- Batch processing of 500+ papers across multiple formats
- Multi-agent LLM architecture for evidence extraction and synthesis
- Customizable review criteria with weighted scoring
- Multiple LLM provider support with cost tracking - with the ability to rerun paper review once papers have been ingested, independent selection of LLM provider for extraction and synthesis
- Comprehensive output formats (Markdown, Word, LaTeX, CSV, HTML dashboard)

### 1.3 Architecture Pattern
```
Paper Ingestion (MarkItDown)
    ↓
Agent 1: Evidence Extractor (per criterion, parallelizable)
    ↓
Agent 2: Review Synthesizer (holistic evaluation)
    ↓
Output Generation (multiple formats)
```

---

## 2. Input Specifications

### 2.1 Paper Ingestion

#### Supported Formats
- **PDF** 
- **Microsoft Word** (.docx, .doc)
- **Markdown** (.md)
- **Plain Text** (.txt)
- **PowerPoint** (.pptx) - for presentation-based submissions

#### Ingestion Methods
1. **Directory Import**
   - Recursive scanning of folder structures
   - Subdirectory support (e.g., `/track-1/`, `/track-2/`)
   - Bulk selection/deselection via checkboxes
   - Status indicators: ✓ Ready | ⚠ Parse Warning | ✗ Failed

2. **Manual Upload**
   - Drag-and-drop interface
   - Individual file selection
   - Per-file inclusion toggle

3. **URL Import** (Optional Phase 2)
   - Direct download from arXiv, SSRN, ResearchGate
   - Batch URL list import via CSV

#### Paper Processing Pipeline
```python
# Pseudo-code structure
def ingest_paper(file_path):
    """
    Convert paper to normalized Markdown format
    """
    from markitdown import MarkItDown
    
    md = MarkItDown()
    result = md.convert(file_path)
    
    paper = {
        "id": generate_uuid(),
        "original_path": file_path,
        "filename": os.path.basename(file_path),
        "content_markdown": result.text_content,
        "token_count": estimate_tokens(result.text_content),
        "metadata": extract_metadata(result.text_content),
        "status": "ready" if result.text_content else "failed",
        "timestamp": datetime.utcnow()
    }
    
    return paper

def extract_metadata(markdown_text):
    """
    Parse paper metadata using LLM or regex patterns
    """
    return {
        "title": extract_title(markdown_text),
        "authors": extract_authors(markdown_text),
        "abstract": extract_abstract(markdown_text),
        "year": extract_year(markdown_text),
        "keywords": extract_keywords(markdown_text)
    }
```

#### Validation Rules
- **File size limit:** 50 MB per file
- **Batch size limit:** 1000 files per import
- **Token limit check:** Flag papers exceeding 80% of model context window
- **Text extraction validation:** Reject files with <100 extractable words
- **Duplicate detection (optional):** Flag papers with >85% similarity in title/abstract 

### 2.2 Configuration Files

#### Directory Structure
```
project_root/
├── config/
│   ├── .env                      # API keys and secrets
│   ├── criteria.yaml             # Review criteria definitions
│   ├── prompts/
│   │   ├── extractor_template.txt
│   │   └── synthesizer_template.txt
│   └── domain_presets/
│       ├── development_economics.yaml
│       ├── computer_science.yaml
│       └── medicine.yaml
├── papers/
│   └── [uploaded papers]
├── outputs/
│   ├── extractions/              # Agent 1 cached outputs
│   ├── reviews/                  # Agent 2 final reviews
│   └── reports/                  # Consolidated outputs

```

#### .env Configuration
```bash
# LLM Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
PERPLEXITY_API_KEY=...
OLLAMA_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434

# Default Settings
PROVIDER_EXTRACTION=deepseek
PROVIDER_SYNTHESIS=gemini
EXTRACTOR_MODEL=deepseek-chat 
SYNTHESIZER_MODEL=gemini-2.5-flash

# LLM Parameters
TEMPERATURE=0.3
MAX_TOKENS=4096
MAX_RETRIES=3
RETRY_DELAY=2  # seconds

# Cost Management
COST_LIMIT_USD=100.00
WARN_THRESHOLD_USD=80.00

# Performance
MAX_PARALLEL_EXTRACTIONS=5  # Concurrent API calls
RATE_LIMIT_RPM=50           # Requests per minute

# Logging
LOG_LEVEL=INFO
LOG_FILE=system.log
```

---

## 3. Review Criteria System

### 3.1 Criteria Structure

Each criterion is defined with the following schema:

```yaml
# criteria.yaml
criteria:
  - id: theoretical_contribution
    name: Theoretical Contribution
    description: |
      Evaluates the paper's advancement of theoretical understanding,
      conceptual frameworks, or formal models in the field.
    
    weight: 15  # Percentage (all weights must sum to 100)
    
    scale:
      type: numeric  # Options: numeric, qualitative, binary
      range: [1, 5]
      labels:
        1: "No contribution"
        2: "Minimal contribution"
        3: "Moderate contribution"
        4: "Strong contribution"
        5: "Exceptional contribution"
    
    sub_questions:
      - "Does the paper present a novel theoretical framework?"
      - "Are existing theories extended or refined in meaningful ways?"
      - "Is the theoretical contribution clearly articulated?"
    
    evidence_requirements:
      - "Explicit theoretical claims or propositions"
      - "Connection to existing theoretical literature"
      - "Logical development of theoretical arguments"
    
    threshold:
      min_acceptable: 3
      auto_reject_below: 1
    
    extraction_prompt: |
      Analyze the theoretical contribution of this paper.
      
      Focus on:
      1. Novel concepts, frameworks, or models introduced
      2. Extensions or refinements of existing theories
      3. Clarity of theoretical articulation
      
      Provide:
      - Direct quotes with page numbers supporting your assessment
      - Specific theoretical claims made by authors
      - Comparison to relevant theoretical literature (if mentioned)
      
      Output as structured JSON.
```

### 3.2 Default Criteria Set (Development Economics)

```yaml
domain: development_economics

criteria:
  1. theoretical_contribution:      weight: 15%
  2. empirical_rigor:               weight: 20%
  3. data_transparency:             weight: 10%
  4. methodological_appropriateness: weight: 15%
  5. policy_relevance:              weight: 10%
  6. clarity_structure:             weight: 10%
  7. originality_novelty:           weight: 10% 
  8. literature_engagement:         weight: 5%
  9. statistical_robustness:        weight: 5%

total_weight: 100%
```

### 3.3 Criteria Management Interface

#### Operations
- **Add Criterion:** Modal form with all schema fields
- **Delete Criterion:** Confirmation dialog, auto-adjust remaining weights
- **Reorder Criteria:** Drag-and-drop interface
- **Edit Weights:** Live validation ensuring sum = 100%
- **Import/Export:** YAML file upload/download
- **Presets:** One-click loading of domain-specific configurations

#### Validation Rules
- Total weight must equal 100% (±0.1% tolerance)
- Each criterion must have unique ID
- Scale ranges must be valid (min < max)
- Threshold values must be within scale range
- At least 3 criteria required (maximum 20 recommended)

---

## 4. LLM Provider Integration

### 4.1 Supported Providers

| Provider | Models | Context Window | Cost Tier |
|----------|--------|----------------|-----------|
| **OpenAI** | gpt-4o, gpt-4o-mini | 128K | $$$ |
| **Anthropic** | claude-opus-4-20250514, claude-sonnet-4-20250514, claude-3-7-sonnet-latest | 200K | $$$ |
| **Google** | gemini-2.0-flash, gemini-2.0-pro | 2M | $$ |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | 64K | $ |
| **Perplexity** | sonar, sonar-pro | 127K | $$ |
| **Ollama** | User-configured local models | Varies | Free |

### 4.2 Unified API Wrapper

Use **LiteLLM** library for unified interface:

```python
from litellm import completion
import os

def call_llm(prompt, system_prompt, provider, model, temperature=0.3):
    """
    Unified LLM API call with retry logic
    """
    model_string = f"{provider}/{model}"
    
    try:
        response = completion(
            model=model_string,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4096,
            timeout=60
        )
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": response.usage,
            "cost": calculate_cost(response.usage, model_string)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None
        }

def calculate_cost(usage, model_string):
    """
    Calculate API call cost based on token usage
    """
    # Cost lookup table (update with current pricing)
    pricing = {
        "openai/gpt-4o": {"input": 2.50, "output": 10.00},  # per 1M tokens
        "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
        # ... add all models
    }
    
    if model_string in pricing:
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing[model_string]["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing[model_string]["output"]
        return input_cost + output_cost
    
    return 0.0  # Unknown cost
```

### 4.3 Cost Estimation & Tracking

```python
def estimate_batch_cost(papers, criteria, provider, model):
    """
    Estimate total cost before running batch analysis
    """
    avg_paper_tokens = calculate_average_tokens(papers)
    
    # Agent 1: Evidence Extraction
    extraction_calls = len(papers) * len(criteria)
    extraction_input_tokens = extraction_calls * avg_paper_tokens
    extraction_output_tokens = extraction_calls * 1000  # ~1K tokens per extraction
    
    # Agent 2: Review Synthesis
    synthesis_calls = len(papers)
    synthesis_input_tokens = synthesis_calls * (len(criteria) * 500)  # Compact extractions
    synthesis_output_tokens = synthesis_calls * 2000  # ~2K tokens per review
    
    total_cost = calculate_cost_for_tokens(
        input_tokens=extraction_input_tokens + synthesis_input_tokens,
        output_tokens=extraction_output_tokens + synthesis_output_tokens,
        model=f"{provider}/{model}"
    )
    
    return {
        "total_papers": len(papers),
        "total_api_calls": extraction_calls + synthesis_calls,
        "estimated_cost_usd": total_cost,
        "cost_per_paper": total_cost / len(papers)
    }
```

---

## 5. Two-Agent Analysis Architecture

### 5.1 Agent 1: Evidence Extractor

#### Purpose
Extract criterion-specific evidence, quotes, and preliminary scores from papers.

#### Input Schema
```json
{
  "paper_id": "uuid-string",
  "paper_content": "Full markdown text from MarkItDown",
  "criterion": {
    "id": "empirical_rigor",
    "name": "Empirical Rigor",
    "description": "...",
    "sub_questions": [...],
    "scale": {...}
  }
}
```

#### Output Schema
```json
{
  "paper_id": "uuid-string",
  "criterion_id": "empirical_rigor",
  "score": 4,
  "score_justification": "The paper employs a difference-in-differences approach with robust standard errors and multiple placebo tests, demonstrating strong causal identification.",
  
  "evidence": [
    {
      "quote": "We employ a difference-in-differences approach with staggered treatment adoption across 12 districts",
      "page_reference": "Page 12, Methods section",
      "relevance": "Primary methodological approach"
    },
    {
      "quote": "Robustness checks include: (1) placebo tests using pre-treatment periods, (2) alternative control groups, (3) permutation tests",
      "page_reference": "Page 18, Results section",
      "relevance": "Demonstrates methodological rigor"
    }
  ],
  
  "strengths": [
    "Clear causal identification strategy",
    "Comprehensive robustness checks",
    "Appropriate use of panel data techniques"
  ],
  
  "weaknesses": [
    "Limited discussion of parallel trends assumption",
    "No formal power analysis reported"
  ],
  
  "missing_elements": [
    "Sensitivity analysis for specification choices",
    "Discussion of external validity"
  ],
  
  "confidence": 0.85,
  "extraction_timestamp": "2025-11-01T14:32:10Z",
  "model_used": "claude-sonnet-4-20250514"
}
```

#### Prompt Template

```
SYSTEM PROMPT:
You are an expert academic reviewer specializing in {domain}. Your task is to extract evidence for a specific evaluation criterion from a research paper. Be thorough, objective, and cite specific evidence.

USER PROMPT:
# Paper Content
{paper_markdown}

# Evaluation Criterion
**Name:** {criterion_name}
**Description:** {criterion_description}

**Sub-questions to consider:**
{sub_questions}

**Scoring Scale:**
{scale_definition}

# Task
Analyze this paper against the criterion "{criterion_name}". Provide:

1. **Score** (on the defined scale): Your assessment
2. **Score Justification** (2-3 sentences): Why this score?
3. **Evidence** (3-5 key quotes):
   - Direct quotes from the paper
   - Page/section references
   - Relevance explanation
4. **Strengths** (bullet points): What the paper does well
5. **Weaknesses** (bullet points): What could be improved
6. **Missing Elements** (if any): Expected elements not found
7. **Confidence** (0.0-1.0): How confident are you in this assessment?

Output as valid JSON matching this schema:
{output_schema}

IMPORTANT:
- Only use evidence DIRECTLY from the paper
- Include page numbers or section names for all quotes
- Be specific, not generic
- If evidence is insufficient, note this and lower confidence
```

#### Processing Logic

```python
def extract_criterion_evidence(paper, criterion, config):
    """
    Agent 1: Extract evidence for a single criterion
    """
    # Check cache first
    cache_key = f"{paper['id']}_{criterion['id']}"
    cached = load_from_cache(cache_key)
    if cached:
        return cached
    
    # Build prompt
    prompt = build_extraction_prompt(
        paper_content=paper['content_markdown'],
        criterion=criterion,
        domain=config['domain']
    )
    
    # Call LLM
    response = call_llm(
        prompt=prompt,
        system_prompt=get_system_prompt("extractor", config['domain']),
        provider=config['extractor_provider'],
        model=config['extractor_model'],
        temperature=config['temperature']
    )
    
    if not response['success']:
        return handle_extraction_error(response['error'], paper, criterion)
    
    # Parse JSON response
    try:
        extraction = json.loads(response['content'])
        extraction['paper_id'] = paper['id']
        extraction['criterion_id'] = criterion['id']
        extraction['model_used'] = config['extractor_model']
        extraction['extraction_timestamp'] = datetime.utcnow().isoformat()
        
        # Validate extraction schema
        validate_extraction_schema(extraction)
        
        # Cache result
        save_to_cache(cache_key, extraction)
        
        return extraction
    
    except json.JSONDecodeError as e:
        return handle_json_parse_error(e, response['content'], paper, criterion)

def process_paper_extractions(paper, criteria, config):
    """
    Run Agent 1 on all criteria for a single paper
    """
    extractions = []
    
    if config['parallel_extraction']:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=config['max_parallel']) as executor:
            futures = [
                executor.submit(extract_criterion_evidence, paper, criterion, config)
                for criterion in criteria
            ]
            extractions = [f.result() for f in as_completed(futures)]
    else:
        # Sequential processing (better for rate limits)
        for criterion in criteria:
            extraction = extract_criterion_evidence(paper, criterion, config)
            extractions.append(extraction)
            
            # Rate limiting
            time.sleep(config['rate_limit_delay'])
    
    return extractions
```

### 5.2 Agent 2: Review Synthesizer

#### Purpose
Synthesize extracted evidence into coherent narrative review with final recommendation.

#### Input Schema
```json
{
  "paper_id": "uuid-string",
  "paper_metadata": {
    "title": "...",
    "authors": "...",
    "abstract": "..."
  },
  "extractions": [
    {/* Agent 1 output for criterion 1 */},
    {/* Agent 1 output for criterion 2 */},
    // ...
  ],
  "criteria_config": {
    "weights": {...},
    "thresholds": {...}
  }
}
```

#### Output Schema
```json
{
  "paper_id": "uuid-string",
  "paper_title": "Impact of Microfinance on Rural Entrepreneurship",
  
  "overall_score": 82.5,
  "weighted_breakdown": {
    "theoretical_contribution": {"score": 4, "weight": 15, "weighted": 12.0},
    "empirical_rigor": {"score": 5, "weight": 20, "weighted": 20.0},
    // ...
  },
  
  "recommendation": "Accept",
  "recommendation_rationale": "This paper makes a strong empirical contribution to the microfinance literature with rigorous causal identification. While theoretical framing could be strengthened, the methodological quality and policy relevance justify acceptance.",
  
  "executive_summary": "This study examines the impact of microfinance access on rural entrepreneurship using a natural experiment in Bangladesh. The paper employs a difference-in-differences design leveraging staggered program rollout across 12 districts. Strengths include robust causal identification, comprehensive data transparency, and clear policy implications. The primary limitation is modest theoretical contribution.",
  
  "detailed_assessment": {
    "major_strengths": [
      "Exceptional empirical rigor with clear causal identification (Criterion 2: 5/5)",
      "Outstanding data transparency with replication materials (Criterion 3: 5/5)",
      "Strong policy relevance with actionable recommendations (Criterion 5: 4/5)"
    ],
    
    "major_concerns": [],
    
    "minor_issues": [
      "Theoretical contribution is adequate but not groundbreaking (Criterion 1: 3/5)",
      "Literature review could engage more deeply with recent RCT evidence (Criterion 8: 3/5)"
    ],
    
    "revision_suggestions": [
      "Strengthen theoretical framework by connecting findings to broader development theory",
      "Expand discussion of external validity and generalizability",
      "Add sensitivity analysis for key specification choices"
    ]
  },
  
  "criterion_narrative": {
    "theoretical_contribution": "The paper provides a moderate theoretical contribution. While it applies existing theories of credit constraints and entrepreneurship, it does not substantially extend theoretical frameworks. The contribution is primarily empirical.",
    // ... narratives for each criterion
  },
  
  "decision_confidence": 0.82,
  "flags": [],
  
  "synthesis_timestamp": "2025-11-01T14:45:22Z",
  "model_used": "claude-sonnet-4-20250514"
}
```

#### Recommendation Logic

```python
def calculate_recommendation(extractions, criteria_config):
    """
    Determine Accept/Reject/Maybe based on weighted scores and thresholds
    """
    weighted_scores = []
    total_weight = 0
    
    for extraction in extractions:
        criterion = criteria_config[extraction['criterion_id']]
        weighted_score = extraction['score'] * (criterion['weight'] / 100)
        weighted_scores.append(weighted_score)
        total_weight += criterion['weight']
    
    overall_score = sum(weighted_scores) / total_weight * 100
    
    # Check threshold violations
    critical_failures = []
    for extraction in extractions:
        criterion = criteria_config[extraction['criterion_id']]
        if 'threshold' in criterion:
            if extraction['score'] < criterion['threshold'].get('auto_reject_below', 0):
                critical_failures.append(extraction['criterion_id'])
    
    # Decision rules
    if critical_failures:
        return "Reject", f"Critical failure in: {', '.join(critical_failures)}"
    
    elif overall_score >= 85:
        return "Accept", "Strong overall performance across all criteria"
    
    elif overall_score >= 70:
        # Check for borderline concerns
        low_scores = [e for e in extractions if e['score'] < 3]
        if len(low_scores) >= 3:
            return "Maybe", "Mixed performance with several weak areas"
        else:
            return "Accept", "Good overall performance with minor improvements needed"
    
    elif overall_score >= 60:
        return "Maybe", "Borderline performance requiring careful consideration"
    
    else:
        return "Reject", "Below acceptance threshold with significant weaknesses"
```

#### Prompt Template

```
SYSTEM PROMPT:
You are a senior academic editor synthesizing multiple criterion-level assessments into a coherent review. Write in a professional, constructive tone suitable for author feedback.

USER PROMPT:
# Paper Information
**Title:** {paper_title}
**Authors:** {paper_authors}
**Abstract:** {paper_abstract}

# Extracted Evidence (from Agent 1)
{json_dump_of_extractions}

# Review Configuration
**Criteria Weights:**
{weights_table}

**Overall Weighted Score:** {calculated_score}/100

**Preliminary Recommendation:** {calculated_recommendation}

# Task
Synthesize these criterion-level assessments into a comprehensive review. Provide:

1. **Executive Summary** (3-4 sentences):
   - Research question and approach
   - Key strengths and limitations
   - Bottom-line assessment

2. **Detailed Assessment**:
   - Major Strengths (2-4 bullet points): Highlight exceptional aspects
   - Major Concerns (if any): Critical issues that impact recommendation
   - Minor Issues (2-4 bullet points): Areas for improvement

3. **Criterion Narratives**:
   - For each criterion, write 2-3 sentences synthesizing the evidence
   - Avoid simply repeating scores; explain the reasoning

4. **Recommendation & Rationale**:
   - Final recommendation: Accept | Maybe | Reject
   - 2-3 sentences justifying the decision
   - Reference specific criteria that drove the decision

5. **Revision Suggestions** (if Accept or Maybe):
   - 3-5 concrete, actionable suggestions for authors

6. **Decision Confidence** (0.0-1.0):
   - How confident are you in this recommendation?
   - Lower confidence if extractions had low confidence or conflicting evidence

Output as valid JSON matching this schema:
{output_schema}

GUIDELINES:
- Be constructive and professional
- Connect the recommendation clearly to the evidence
- Identify patterns across criteria (e.g., strong methods but weak theory)
- For "Maybe" recommendations, clearly state what would tip it to Accept/Reject
- Flag any inconsistencies you notice in the extractions
```

---

## 6. Output Specifications

### 6.1 Output File Naming Convention

```
{timestamp}_{provider}_{model}_{paper_id}_{output_type}.{extension}

Examples:
20251101_143022_claude_sonnet-4_abc123_review.md
20251101_143022_claude_sonnet-4_abc123_review.docx
20251101_143022_claude_sonnet-4_batch_consolidated.csv
```

### 6.2 Individual Paper Review (Markdown)

```markdown
# Review: {Paper Title}

**Paper ID:** {uuid}  
**Authors:** {author list}  
**Review Date:** {timestamp}  
**Reviewer:** LLM-Assisted Review ({model_name})  
**Overall Score:** {score}/100  
**Recommendation:** {Accept|Maybe|Reject}  

---

## Executive Summary

{2-3 paragraph summary}

---

## Overall Assessment

**Recommendation:** **{ACCEPT|MAYBE|REJECT}**

**Rationale:**
{Recommendation justification}

**Confidence:** {confidence_score}/1.0

---

## Detailed Evaluation

### Major Strengths
- {strength 1}
- {strength 2}
...

### Major Concerns
- {concern 1} (if any)
...

### Minor Issues
- {issue 1}
...

---

## Criterion-by-Criterion Analysis

### 1. Theoretical Contribution (Weight: 15%, Score: 4/5, Weighted: 12.0)

**Score Justification:**
{Agent 1 justification}

**Evidence:**
> "{quote 1}" (Page X)
> "{quote 2}" (Page Y)

**Strengths:**
- {strength 1}

**Weaknesses:**
- {weakness 1}

**Assessment:**
{Agent 2 narrative synthesis}

---

{Repeat for all criteria}

---

## Revision Suggestions

1. {Suggestion 1}
2. {Suggestion 2}
...

---

## Score Breakdown

| Criterion | Weight | Raw Score | Weighted Score |
|-----------|--------|-----------|----------------|
| Theoretical Contribution | 15% | 4/5 | 12.0 |
| Empirical Rigor | 20% | 5/5 | 20.0 |
| ... | ... | ... | ... |
| **TOTAL** | **100%** | — | **82.5** |

---

## Metadata

- **Analysis Date:** {timestamp}
- **LLM Provider:** {provider}
- **Extractor Model:** {extractor_model}
- **Synthesizer Model:** {synthesizer_model}
- **Processing Time:** {duration}
- **API Cost:** ${cost}

---

## Reviewer Notes

{Any flags, warnings, or manual annotations}
```

### 6.3 Consolidated CSV Output

```csv
paper_id,title,authors,overall_score,recommendation,confidence,theoretical_contribution_score,theoretical_contribution_text,empirical_rigor_score,empirical_rigor_text,...,processing_timestamp,model_used,api_cost_usd
abc123,"Impact of Microfinance","Author et al.",82.5,Accept,0.82,4,"Moderate theoretical contribution...",5,"Exceptional empirical rigor...",2025-11-01T14:45:22Z,claude-sonnet-4,0.23
def456,"Rural Health Interventions","Smith et al.",67.0,Maybe,0.75,3,"Adequate theoretical framing...",4,"Strong methodology with minor gaps...",2025-11-01T14:52:10Z,claude-sonnet-4,0.21
...
```

### 6.4 HTML Dashboard

Interactive dashboard with:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Review Dashboard - Batch {timestamp}</title>
    <!-- Include Chart.js, DataTables, Tailwind CSS from CDN -->
</head>
<body>
    <!-- Header -->
    <header>
        <h1>Academic Review Dashboard</h1>
        <div class="stats">
            <div>Total Papers: {count}</div>
            <div>Accept: {accept_count} ({accept_pct}%)</div>
            <div>Maybe: {maybe_count} ({maybe_pct}%)</div>
            <div>Reject: {reject_count} ({reject_pct}%)</div>
            <div>Total Cost: ${total_cost}</div>
        </div>
    </header>

    <!-- Filters -->
    <section class="filters">
        <select id="recommendation-filter">
            <option value="all">All Recommendations</option>
            <option value="accept">Accept</option>
            <option value="maybe">Maybe</option>
            <option value="reject">Reject</option>
        </select>
        
        <input type="range" id="score-threshold" min="0" max="100" value="0">
        <label>Min Score: <span id="score-value">0</span></label>
        
        <input type="text" id="search-box" placeholder="Search titles/authors...">
    </section>

    <!-- Visualizations -->
    <section class="visualizations">
        <div class="chart">
            <h3>Score Distribution</h3>
            <canvas id="score-histogram"></canvas>
        </div>
        
        <div class="chart">
            <h3>Criterion Performance</h3>
            <canvas id="criterion-radar"></canvas>
        </div>
        
        <div class="chart">
            <h3>Recommendation Breakdown</h3>
            <canvas id="recommendation-pie"></canvas>
        </div>
    </section>

    <!-- Data Table -->
    <section class="data-table">
        <table id="reviews-table" class="display">
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Authors</th>
                    <th>Score</th>
                    <th>Recommendation</th>
                    <th>Confidence</th>
                    <th>Flags</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <!-- Populated via JavaScript -->
            </tbody>
        </table>
    </section>

    <!-- Individual Paper Modal -->
    <div id="paper-modal" class="modal">
        <div class="modal-content">
            <h2 id="paper-title"></h2>
            <div id="paper-details">
                <!-- Full review rendered here -->
            </div>
            <button id="export-pdf">Export as PDF</button>
            <button id="edit-review">Edit Review</button>
            <button id="close-modal">Close</button>
        </div>
    </div>

    <script>
        // Dashboard interactivity code
        // Load data from JSON endpoint
        // Initialize DataTables, Chart.js visualizations
        // Handle filtering, sorting, modal display
    </script>
</body>
</html>
```

### 6.5 LaTeX Output (for submission to journals)

```latex
\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{Review Report: {Paper Title}}
\author{LLM-Assisted Review ({model\_name})}
\date{\today}

\begin{document}

\maketitle

\section*{Executive Summary}
{executive_summary}

\section{Overall Assessment}
\textbf{Recommendation:} \texttt{{RECOMMENDATION}}

\textbf{Overall Score:} {score}/100

\textbf{Confidence:} {confidence}

\textbf{Rationale:} {rationale}

\section{Detailed Evaluation}

\subsection{Major Strengths}
\begin{itemize}
    \item {strength_1}
    \item {strength_2}
\end{itemize}

\subsection{Major Concerns}
\begin{itemize}
    \item {concern_1}
\end{itemize}

\section{Criterion-by-Criterion Analysis}

{criterion_sections}

\section{Revision Suggestions}
\begin{enumerate}
    \item {suggestion_1}
    \item {suggestion_2}
\end{enumerate}

\section{Score Breakdown}

\begin{table}[h]
\centering
\begin{tabular}{@{}llll@{}}
\toprule
Criterion & Weight & Raw Score & Weighted Score \\
\midrule
{table_rows}
\midrule
\textbf{TOTAL} & \textbf{100\%} & — & \textbf{{overall_score}} \\
\bottomrule
\end{tabular}
\caption{Detailed score breakdown}
\end{table}

\end{document}
```

### 6.6 Word Document (.docx) Structure

Using `python-docx`:

```python
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def generate_word_review(review_data, output_path):
    """
    Generate formatted Word document review
    """
    doc = Document()
    
    # Title
    title = doc.add_heading(f"Review: {review_data['paper_title']}", level=1)
    
    # Metadata table
    metadata_table = doc.add_table(rows=5, cols=2)
    metadata_table.style = 'Light Grid Accent 1'
    
    metadata = [
        ("Paper ID", review_data['paper_id']),
        ("Authors", review_data['paper_metadata']['authors']),
        ("Review Date", review_data['synthesis_timestamp']),
        ("Overall Score", f"{review_data['overall_score']}/100"),
        ("Recommendation", review_data['recommendation'])
    ]
    
    for i, (key, value) in enumerate(metadata):
        metadata_table.rows[i].cells[0].text = key
        metadata_table.rows[i].cells[1].text = str(value)
    
    doc.add_paragraph()
    
    # Executive Summary
    doc.add_heading("Executive Summary", level=2)
    doc.add_paragraph(review_data['executive_summary'])
    
    # Overall Assessment
    doc.add_heading("Overall Assessment", level=2)
    
    rec_para = doc.add_paragraph()
    rec_para.add_run("Recommendation: ").bold = True
    rec_run = rec_para.add_run(review_data['recommendation'])
    rec_run.bold = True
    
    # Color code recommendation
    if review_data['recommendation'] == "Accept":
        rec_run.font.color.rgb = RGBColor(0, 128, 0)  # Green
    elif review_data['recommendation'] == "Reject":
        rec_run.font.color.rgb = RGBColor(255, 0, 0)  # Red
    else:
        rec_run.font.color.rgb = RGBColor(255, 165, 0)  # Orange
    
    doc.add_paragraph(f"Rationale: {review_data['recommendation_rationale']}")
    doc.add_paragraph(f"Confidence: {review_data['decision_confidence']:.2f}")
    
    # Detailed Assessment
    doc.add_heading("Detailed Evaluation", level=2)
    
    doc.add_heading("Major Strengths", level=3)
    for strength in review_data['detailed_assessment']['major_strengths']:
        doc.add_paragraph(strength, style='List Bullet')
    
    if review_data['detailed_assessment']['major_concerns']:
        doc.add_heading("Major Concerns", level=3)
        for concern in review_data['detailed_assessment']['major_concerns']:
            doc.add_paragraph(concern, style='List Bullet')
    
    doc.add_heading("Minor Issues", level=3)
    for issue in review_data['detailed_assessment']['minor_issues']:
        doc.add_paragraph(issue, style='List Bullet')
    
    # Criterion-by-Criterion
    doc.add_page_break()
    doc.add_heading("Criterion-by-Criterion Analysis", level=2)
    
    for criterion_id, narrative in review_data['criterion_narrative'].items():
        # Find corresponding extraction
        extraction = next(e for e in review_data['extractions'] 
                         if e['criterion_id'] == criterion_id)
        
        criterion_name = extraction['criterion_name']
        score = extraction['score']
        weight = extraction['weight']
        
        doc.add_heading(f"{criterion_name} (Score: {score}, Weight: {weight}%)", level=3)
        doc.add_paragraph(narrative)
        
        # Evidence quotes
        if extraction['evidence']:
            doc.add_paragraph("Key Evidence:", style='Intense Quote')
            for ev in extraction['evidence'][:3]:  # Top 3 quotes
                quote_para = doc.add_paragraph(style='Quote')
                quote_para.add_run(f'"{ev["quote"]}" ').italic = True
                quote_para.add_run(f"({ev['page_reference']})")
    
    # Score Breakdown Table
    doc.add_page_break()
    doc.add_heading("Score Breakdown", level=2)
    
    score_table = doc.add_table(rows=len(review_data['weighted_breakdown'])+2, cols=4)
    score_table.style = 'Medium Grid 1 Accent 1'
    
    # Header row
    header_cells = score_table.rows[0].cells
    header_cells[0].text = "Criterion"
    header_cells[1].text = "Weight"
    header_cells[2].text = "Raw Score"
    header_cells[3].text = "Weighted Score"
    
    # Data rows
    for i, (criterion_id, data) in enumerate(review_data['weighted_breakdown'].items(), 1):
        row_cells = score_table.rows[i].cells
        row_cells[0].text = criterion_id.replace('_', ' ').title()
        row_cells[1].text = f"{data['weight']}%"
        row_cells[2].text = f"{data['score']}"
        row_cells[3].text = f"{data['weighted']:.1f}"
    
    # Total row
    total_cells = score_table.rows[-1].cells
    total_cells[0].text = "TOTAL"
    total_cells[1].text = "100%"
    total_cells[2].text = "—"
    total_cells[3].text = f"{review_data['overall_score']:.1f}"
    
    # Make total row bold
    for cell in total_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
    
    # Save document
    doc.save(output_path)
```

---

## 7. Quality Assurance & Validation

### 7.1 Pre-Flight Checks

Before running batch analysis:

```python
def pre_flight_validation(papers, criteria, config):
    """
    Validate configuration before expensive API calls
    """
    issues = []
    warnings = []
    
    # Check 1: Papers validation
    if len(papers) == 0:
        issues.append("No papers selected for review")
    
    for paper in papers:
        if paper['token_count'] > config['model_context_limit'] * 0.8:
            warnings.append(f"Paper {paper['filename']} may exceed context window")
        
        if paper['token_count'] < 500:
            warnings.append(f"Paper {paper['filename']} seems too short (< 500 tokens)")
    
    # Check 2: Criteria validation
    total_weight = sum(c['weight'] for c in criteria)
    if abs(total_weight - 100) > 0.1:
        issues.append(f"Criteria weights sum to {total_weight}%, must equal 100%")
    
    if len(criteria) < 3:
        warnings.append("Fewer than 3 criteria may produce unreliable reviews")
    
    # Check 3: API credentials
    provider = config['provider']
    if not check_api_key(provider):
        issues.append(f"Missing API key for {provider}")
    
    # Check 4: Cost estimation
    cost_estimate = estimate_batch_cost(papers, criteria, provider, config['model'])
    if cost_estimate['estimated_cost_usd'] > config['cost_limit_usd']:
        issues.append(f"Estimated cost ${cost_estimate['estimated_cost_usd']:.2f} exceeds limit ${config['cost_limit_usd']:.2f}")
    elif cost_estimate['estimated_cost_usd'] > config['warn_threshold_usd']:
        warnings.append(f"Estimated cost ${cost_estimate['estimated_cost_usd']:.2f} is high")
    
    # Check 5: Rate limits
    total_api_calls = len(papers) * (len(criteria) + 1)  # Extractions + syntheses
    estimated_duration = total_api_calls / config['rate_limit_rpm'] * 60
    
    if estimated_duration > 3600:  # > 1 hour
        warnings.append(f"Estimated processing time: {estimated_duration/3600:.1f} hours")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "cost_estimate": cost_estimate,
        "estimated_duration_seconds": estimated_duration
    }
```

### 7.2 Calibration Papers

Include pre-scored papers to validate LLM accuracy:

```yaml
# calibration_set.yaml
calibration_papers:
  - paper_id: cal_001
    filename: "known_excellent_paper.pdf"
    ground_truth:
      overall_score: 92
      recommendation: Accept
      criterion_scores:
        theoretical_contribution: 5
        empirical_rigor: 5
        # ...
    
  - paper_id: cal_002
    filename: "known_weak_paper.pdf"
    ground_truth:
      overall_score: 58
      recommendation: Reject
      criterion_scores:
        theoretical_contribution: 2
        empirical_rigor: 3
        # ...

validation_thresholds:
  max_score_deviation: 10  # Points
  min_recommendation_agreement: 0.8  # 80%
```

```python
def run_calibration_check(calibration_papers, criteria, config):
    """
    Validate LLM performance against known-good reviews
    """
    results = []
    
    for cal_paper in calibration_papers:
        # Run normal review process
        llm_review = process_paper_full(cal_paper, criteria, config)
        
        # Compare to ground truth
        score_deviation = abs(llm_review['overall_score'] - cal_paper['ground_truth']['overall_score'])
        recommendation_match = llm_review['recommendation'] == cal_paper['ground_truth']['recommendation']
        
        criterion_deviations = {}
        for criterion_id in cal_paper['ground_truth']['criterion_scores']:
            llm_score = next(e['score'] for e in llm_review['extractions'] 
                           if e['criterion_id'] == criterion_id)
            truth_score = cal_paper['ground_truth']['criterion_scores'][criterion_id]
            criterion_deviations[criterion_id] = abs(llm_score - truth_score)
        
        results.append({
            "paper_id": cal_paper['paper_id'],
            "score_deviation": score_deviation,
            "recommendation_match": recommendation_match,
            "criterion_deviations": criterion_deviations,
            "passed": score_deviation <= config['validation_thresholds']['max_score_deviation'] 
                     and recommendation_match
        })
    
    # Aggregate results
    pass_rate = sum(r['passed'] for r in results) / len(results)
    avg_deviation = sum(r['score_deviation'] for r in results) / len(results)
    
    if pass_rate < config['validation_thresholds']['min_recommendation_agreement']:
        raise CalibrationFailedException(
            f"Calibration pass rate {pass_rate:.1%} below threshold. "
            f"Avg deviation: {avg_deviation:.1f} points. "
            f"Consider adjusting prompts or switching models."
        )
    
    return {
        "pass_rate": pass_rate,
        "avg_deviation": avg_deviation,
        "detailed_results": results
    }
```

### 7.3 Consistency Checks

Post-processing validation:

```python
def validate_review_consistency(review):
    """
    Check for logical inconsistencies in generated review
    """
    flags = []
    
    # Check 1: High overall score but "Reject" recommendation
    if review['overall_score'] >= 75 and review['recommendation'] == "Reject":
        flags.append("INCONSISTENT: High score but rejection recommended")
    
    # Check 2: Low overall score but "Accept" recommendation
    if review['overall_score'] < 65 and review['recommendation'] == "Accept":
        flags.append("INCONSISTENT: Low score but acceptance recommended")
    
    # Check 3: Mismatched criterion assessments
    high_scores = [e for e in review['extractions'] if e['score'] >= 4]
    if len(high_scores) >= 7 and review['recommendation'] == "Reject":
        flags.append("INCONSISTENT: Mostly high criterion scores but rejection")
    
    # Check 4: Confidence vs recommendation alignment
    if review['decision_confidence'] < 0.5 and review['recommendation'] in ["Accept", "Reject"]:
        flags.append("WARNING: Low confidence on strong recommendation")
    
    # Check 5: Theoretical contribution vs originality
    theory = next((e for e in review['extractions'] if e['criterion_id'] == 'theoretical_contribution'), None)
    originality = next((e for e in review['extractions'] if e['criterion_id'] == 'originality_novelty'), None)
    
    if theory and originality:
        if abs(theory['score'] - originality['score']) >= 3:
            flags.append("INCONSISTENT: Large gap between theoretical contribution and originality scores")
    
    # Check 6: Missing evidence in low-confidence extractions
    for extraction in review['extractions']:
        if extraction['confidence'] < 0.6 and len(extraction['evidence']) < 2:
            flags.append(f"WARNING: Low confidence on {extraction['criterion_id']} with limited evidence")
    
    return flags
```

### 7.4 Audit Trail

Log all LLM interactions for transparency:

```python
import logging
from datetime import datetime

def setup_audit_logging():
    """
    Configure comprehensive audit trail
    """
    logger = logging.getLogger('audit')
    logger.setLevel(logging.INFO)
    
    # File handler with detailed format
    handler = logging.FileHandler(f"audit_{datetime.now().strftime('%Y%m%d')}.log")
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def log_llm_interaction(logger, paper_id, agent, criterion_id, prompt, response, cost):
    """
    Log every LLM API call
    """
    logger.info(f"PAPER_ID={paper_id} | AGENT={agent} | CRITERION={criterion_id} | "
                f"PROMPT_TOKENS={response['usage'].prompt_tokens} | "
                f"COMPLETION_TOKENS={response['usage'].completion_tokens} | "
                f"COST=${cost:.4f} | "
                f"SUCCESS={response['success']}")
    
    # Store full prompt/response for debugging (optional, can be large)
    if os.getenv('STORE_FULL_INTERACTIONS') == 'true':
        interaction_file = f"interactions/{paper_id}_{agent}_{criterion_id}.json"
        with open(interaction_file, 'w') as f:
            json.dump({
                "prompt": prompt,
                "response": response['content'],
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)
```

---

## 8. Human-in-the-Loop Features

### 8.1 Review Override Interface

Allow human reviewers to modify LLM outputs:

```python
class ReviewOverride:
    """
    Track human modifications to LLM reviews
    """
    def __init__(self, review_id):
        self.review_id = review_id
        self.overrides = []
    
    def override_criterion_score(self, criterion_id, new_score, justification, reviewer_name):
        """
        Record manual score adjustment
        """
        override = {
            "type": "criterion_score",
            "criterion_id": criterion_id,
            "original_score": self.get_original_score(criterion_id),
            "new_score": new_score,
            "justification": justification,
            "reviewer": reviewer_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.overrides.append(override)
        self.apply_override(override)
    
    def override_recommendation(self, new_recommendation, justification, reviewer_name):
        """
        Record manual recommendation change
        """
        override = {
            "type": "recommendation",
            "original_recommendation": self.get_original_recommendation(),
            "new_recommendation": new_recommendation,
            "justification": justification,
            "reviewer": reviewer_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.overrides.append(override)
        self.apply_override(override)
    
    def add_reviewer_note(self, note_text, reviewer_name):
        """
        Add supplementary human commentary
        """
        note = {
            "type": "note",
            "text": note_text,
            "reviewer": reviewer_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.overrides.append(note)
    
    def generate_override_report(self):
        """
        Summarize all human modifications
        """
        return {
            "review_id": self.review_id,
            "total_overrides": len(self.overrides),
            "overrides_by_type": self.count_by_type(),
            "overrides": self.overrides
        }
```

### 8.2 Collaborative Review Workflow

Support multiple reviewers:

```yaml
# workflow_config.yaml
review_workflow:
  mode: collaborative  # Options: single, collaborative, blind
  
  stages:
    - name: llm_initial_review
      agent: llm
      output: draft_review
    
    - name: primary_reviewer
      agent: human
      role: primary_reviewer
      tasks:
        - validate_llm_scores
        - add_qualitative_feedback
        - adjust_recommendation
    
    - name: secondary_reviewer
      agent: human
      role: secondary_reviewer
      tasks:
        - review_primary_assessment
        - flag_disagreements
    
    - name: meta_reviewer
      agent: human
      role: meta_reviewer
      trigger: disagreement_detected
      tasks:
        - resolve_conflicts
        - final_decision

  disagreement_threshold: 10  # Points difference triggers meta-review
  
  blind_review: false  # If true, hide author names
```

### 8.3 Dispute Resolution

Track and resolve reviewer disagreements:

```python
def detect_disagreements(llm_review, human_review_1, human_review_2):
    """
    Identify significant disagreements between reviewers
    """
    disagreements = []
    
    # Check overall scores
    scores = [
        llm_review['overall_score'],
        human_review_1['overall_score'],
        human_review_2['overall_score']
    ]
    
    if max(scores) - min(scores) > 15:
        disagreements.append({
            "type": "overall_score",
            "values": scores,
            "severity": "high"
        })
    
    # Check recommendation conflicts
    recommendations = [
        llm_review['recommendation'],
        human_review_1['recommendation'],
        human_review_2['recommendation']
    ]
    
    if len(set(recommendations)) > 1:
        disagreements.append({
            "type": "recommendation",
            "values": recommendations,
            "severity": "critical"
        })
    
    # Check criterion-level disagreements
    for criterion_id in llm_review['criteria']:
        criterion_scores = [
            llm_review['criteria'][criterion_id]['score'],
            human_review_1['criteria'][criterion_id]['score'],
            human_review_2['criteria'][criterion_id]['score']
        ]
        
        if max(criterion_scores) - min(criterion_scores) >= 2:
            disagreements.append({
                "type": "criterion_score",
                "criterion": criterion_id,
                "values": criterion_scores,
                "severity": "medium"
            })
    
    return disagreements

def generate_dispute_resolution_view(paper_id, disagreements):
    """
    Create side-by-side comparison for meta-reviewer
    """
    return {
        "paper_id": paper_id,
        "disagreements": disagreements,
        "resolution_required": len([d for d in disagreements if d['severity'] in ['high', 'critical']]) > 0,
        "recommended_action": "meta_review" if resolution_required else "averaging"
    }
```

---

## 9. System Implementation

### 9.1 Technology Stack

#### Backend
```python
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
markitdown==0.0.1a2
litellm==1.0.0
python-docx==1.1.0
pandas==2.1.3
sqlalchemy==2.0.23
aiosqlite==0.19.0
python-multipart==0.0.6
httpx==0.25.1
tenacity==8.2.3  # For retry logic
pyyaml==6.0.1
jinja2==3.1.2
```

#### Frontend
```javascript
// package.json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-table": "^8.10.7",
    "recharts": "^2.10.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.3.5",
    "lucide-react": "^0.294.0",
    "react-dropzone": "^14.2.3"
  }
}
```

### 9.2 Database Schema

```sql
-- SQLite schema for metadata and tracking

CREATE TABLE papers (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_path TEXT,
    title TEXT,
    authors TEXT,
    abstract TEXT,
    year INTEGER,
    content_markdown TEXT,
    token_count INTEGER,
    status TEXT,  -- 'pending', 'processing', 'completed', 'failed'
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(filename)
);

CREATE TABLE reviews (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    overall_score REAL,
    recommendation TEXT,  -- 'Accept', 'Maybe', 'Reject'
    decision_confidence REAL,
    executive_summary TEXT,
    recommendation_rationale TEXT,
    extractor_model TEXT,
    synthesizer_model TEXT,
    processing_duration_seconds REAL,
    api_cost_usd REAL,
    review_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE extractions (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    criterion_id TEXT NOT NULL,
    criterion_name TEXT,
    score INTEGER,
    score_justification TEXT,
    evidence_json TEXT,  -- Serialized JSON array
    strengths_json TEXT,
    weaknesses_json TEXT,
    missing_elements_json TEXT,
    confidence REAL,
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE human_overrides (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    override_type TEXT,  -- 'criterion_score', 'recommendation', 'note'
    criterion_id TEXT,
    original_value TEXT,
    new_value TEXT,
    justification TEXT,
    reviewer_name TEXT,
    override_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews(id)
);

CREATE TABLE batch_jobs (
    id TEXT PRIMARY KEY,
    status TEXT,  -- 'queued', 'running', 'completed', 'failed'
    total_papers INTEGER,
    completed_papers INTEGER,
    failed_papers INTEGER,
    total_api_calls INTEGER,
    total_cost_usd REAL,
    start_timestamp TIMESTAMP,
    end_timestamp TIMESTAMP,
    config_json TEXT  -- Serialized configuration
);

CREATE TABLE api_logs (
    id TEXT PRIMARY KEY,
    paper_id TEXT,
    agent TEXT,  -- 'extractor', 'synthesizer'
    criterion_id TEXT,
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    success BOOLEAN,
    error_message TEXT,
    call_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_papers_status ON papers(status);
CREATE INDEX idx_reviews_paper_id ON reviews(paper_id);
CREATE INDEX idx_reviews_recommendation ON reviews(recommendation);
CREATE INDEX idx_extractions_review_id ON extractions(review_id);
CREATE INDEX idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX idx_api_logs_paper_id ON api_logs(paper_id);
```

### 9.3 API Endpoints

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import uuid

app = FastAPI(title="Academic Review System API")

# Paper Management
@app.post("/api/papers/upload")
async def upload_papers(files: List[UploadFile] = File(...)):
    """Upload one or more papers"""
    pass

@app.post("/api/papers/import-directory")
async def import_directory(directory_path: str, recursive: bool = True):
    """Import papers from local directory"""
    pass

@app.get("/api/papers")
async def list_papers(status: str = None, skip: int = 0, limit: int = 100):
    """List all papers with optional filtering"""
    pass

@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get paper details"""
    pass

@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Remove paper from system"""
    pass

# Criteria Management
@app.get("/api/criteria")
async def get_criteria():
    """Get current criteria configuration"""
    pass

@app.put("/api/criteria")
async def update_criteria(criteria: List[dict]):
    """Update criteria configuration"""
    pass

@app.post("/api/criteria/presets/{preset_name}")
async def load_preset(preset_name: str):
    """Load domain-specific preset"""
    pass

# Review Processing
@app.post("/api/reviews/batch")
async def start_batch_review(
    paper_ids: List[str],
    criteria: List[dict],
    config: dict,
    background_tasks: BackgroundTasks
):
    """Start batch review process"""
    batch_id = str(uuid.uuid4())
    background_tasks.add_task(process_batch, batch_id, paper_ids, criteria, config)
    return {"batch_id": batch_id, "status": "queued"}

@app.get("/api/reviews/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get batch processing status"""
    pass

@app.get("/api/reviews/{review_id}")
async def get_review(review_id: str):
    """Get complete review"""
    pass

@app.put("/api/reviews/{review_id}/override")
async def override_review(review_id: str, override_data: dict):
    """Apply human override to review"""
    pass

# Cost Management
@app.post("/api/reviews/estimate-cost")
async def estimate_cost(paper_ids: List[str], criteria: List[dict], config: dict):
    """Estimate cost before running"""
    pass

@app.get("/api/reviews/batch/{batch_id}/cost")
async def get_batch_cost(batch_id: str):
    """Get actual cost of completed batch"""
    pass

# Outputs
@app.get("/api/reviews/{review_id}/export/{format}")
async def export_review(review_id: str, format: str):
    """Export review in specified format (md, docx, latex, pdf)"""
    pass

@app.get("/api/reviews/batch/{batch_id}/export/csv")
async def export_batch_csv(batch_id: str):
    """Export consolidated CSV"""
    pass

@app.get("/api/reviews/batch/{batch_id}/dashboard")
async def get_dashboard_data(batch_id: str):
    """Get data for HTML dashboard"""
    pass

# Calibration
@app.post("/api/calibration/run")
async def run_calibration(calibration_papers: List[dict], config: dict):
    """Run calibration check"""
    pass

# Health & Monitoring
@app.get("/api/health")
async def health_check():
    """System health check"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/stats")
async def get_system_stats():
    """Get overall system statistics"""
    pass
```

### 9.4 Core Processing Logic

```python
# core/processor.py

import asyncio
from typing import List, Dict
from datetime import datetime
import json

class ReviewProcessor:
    """
    Main orchestrator for the two-agent review system
    """
    
    def __init__(self, config: Dict, db_session):
        self.config = config
        self.db = db_session
        self.audit_logger = setup_audit_logging()
    
    async def process_batch(self, batch_id: str, paper_ids: List[str], criteria: List[Dict]):
        """
        Process entire batch of papers
        """
        batch_job = self.create_batch_job(batch_id, paper_ids, criteria)
        
        try:
            # Pre-flight validation
            validation = pre_flight_validation(paper_ids, criteria, self.config)
            if not validation['valid']:
                raise ValidationError(validation['issues'])
            
            # Optional: Run calibration
            if self.config.get('run_calibration', False):
                calibration_result = await self.run_calibration(criteria)
                if not calibration_result['passed']:
                    self.audit_logger.warning(f"Calibration failed: {calibration_result}")
            
            # Process each paper
            results = []
            for i, paper_id in enumerate(paper_ids):
                try:
                    self.update_batch_progress(batch_id, i, len(paper_ids))
                    
                    paper = self.db.get_paper(paper_id)
                    review = await self.process_single_paper(paper, criteria)
                    
                    results.append(review)
                    batch_job.completed_papers += 1
                    
                except Exception as e:
                    self.audit_logger.error(f"Failed to process paper {paper_id}: {e}")
                    batch_job.failed_papers += 1
                    results.append(self.create_error_review(paper_id, str(e)))
            
            # Generate consolidated outputs
            self.generate_batch_outputs(batch_id, results)
            
            batch_job.status = 'completed'
            batch_job.end_timestamp = datetime.utcnow()
            
            return {
                "batch_id": batch_id,
                "status": "completed",
                "completed": batch_job.completed_papers,
                "failed": batch_job.failed_papers,
                "total_cost": batch_job.total_cost_usd
            }
        
        except Exception as e:
            batch_job.status = 'failed'
            batch_job.end_timestamp = datetime.utcnow()
            raise
    
    async def process_single_paper(self, paper: Dict, criteria: List[Dict]) -> Dict:
        """
        Complete review pipeline for one paper
        """
        start_time = datetime.utcnow()
        
        # Stage 1: Evidence Extraction (Agent 1)
        extractions = await self.extract_all_criteria(paper, criteria)
        
        # Stage 2: Review Synthesis (Agent 2)
        review = await self.synthesize_review(paper, extractions, criteria)
        
        # Stage 3: Quality Checks
        consistency_flags = validate_review_consistency(review)
        review['flags'] = consistency_flags
        
        # Calculate processing metrics
        processing_duration = (datetime.utcnow() - start_time).total_seconds()
        review['processing_duration_seconds'] = processing_duration
        
        # Save to database
        self.save_review_to_db(review)
        
        return review
    
    async def extract_all_criteria(self, paper: Dict, criteria: List[Dict]) -> List[Dict]:
        """
        Run Agent 1 on all criteria (parallel or sequential)
        """
        extractions = []
        
        if self.config.get('parallel_extraction', True):
            # Parallel extraction with semaphore for rate limiting
            sem = asyncio.Semaphore(self.config['max_parallel_extractions'])
            
            async def extract_with_limit(criterion):
                async with sem:
                    return await self.extract_criterion_evidence(paper, criterion)
            
            tasks = [extract_with_limit(c) for c in criteria]
            extractions = await asyncio.gather(*tasks)
        
        else:
            # Sequential extraction
            for criterion in criteria:
                extraction = await self.extract_criterion_evidence(paper, criterion)
                extractions.append(extraction)
                
                # Rate limiting delay
                await asyncio.sleep(self.config.get('rate_limit_delay', 1))
        
        return extractions
    
    async def extract_criterion_evidence(self, paper: Dict, criterion: Dict) -> Dict:
        """
        Agent 1: Extract evidence for single criterion
        """
        # Check cache
        cache_key = f"{paper['id']}_{criterion['id']}"
        cached = self.load_extraction_cache(cache_key)
        if cached:
            return cached
        
        # Build prompt
        prompt = self.build_extraction_prompt(paper, criterion)
        system_prompt = self.load_prompt_template('extractor_system')
        
        # Call LLM with retry logic
        response = await self.call_llm_with_retry(
            prompt=prompt,
            system_prompt=system_prompt,
            agent='extractor'
        )
        
        # Parse and validate response
        extraction = self.parse_extraction_response(response, paper['id'], criterion['id'])
        
        # Log to audit trail
        self.audit_logger.info(
            f"EXTRACTION | Paper={paper['id']} | Criterion={criterion['id']} | "
            f"Score={extraction['score']} | Confidence={extraction['confidence']:.2f} | "
            f"Cost=${response['cost']:.4f}"
        )
        
        # Cache result
        self.save_extraction_cache(cache_key, extraction)
        
        return extraction
    
    async def synthesize_review(self, paper: Dict, extractions: List[Dict], criteria: List[Dict]) -> Dict:
        """
        Agent 2: Synthesize final review from extractions
        """
        # Calculate weighted scores
        weighted_breakdown = self.calculate_weighted_scores(extractions, criteria)
        overall_score = sum(item['weighted'] for item in weighted_breakdown.values())
        
        # Determine preliminary recommendation
        preliminary_rec, rec_rationale = calculate_recommendation(extractions, criteria)
        
        # Build synthesis prompt
        prompt = self.build_synthesis_prompt(
            paper=paper,
            extractions=extractions,
            weighted_breakdown=weighted_breakdown,
            overall_score=overall_score,
            preliminary_rec=preliminary_rec
        )
        system_prompt = self.load_prompt_template('synthesizer_system')
        
        # Call LLM
        response = await self.call_llm_with_retry(
            prompt=prompt,
            system_prompt=system_prompt,
            agent='synthesizer'
        )
        
        # Parse response
        review = self.parse_synthesis_response(response, paper['id'])
        
        # Add computed data
        review['overall_score'] = overall_score
        review['weighted_breakdown'] = weighted_breakdown
        review['extractions'] = extractions
        review['paper_metadata'] = {
            'title': paper.get('title'),
            'authors': paper.get('authors'),
            'abstract': paper.get('abstract')
        }
        
        # Log to audit trail
        self.audit_logger.info(
            f"SYNTHESIS | Paper={paper['id']} | "
            f"Score={overall_score:.1f} | Rec={review['recommendation']} | "
            f"Confidence={review['decision_confidence']:.2f} | "
            f"Cost=${response['cost']:.4f}"
        )
        
        return review
    
    async def call_llm_with_retry(self, prompt: str, system_prompt: str, agent: str) -> Dict:
        """
        Call LLM with exponential backoff retry logic
        """
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(
            stop=stop_after_attempt(self.config['max_retries']),
            wait=wait_exponential(multiplier=self.config['retry_delay'], min=2, max=30)
        )
        async def _call():
            return call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                provider=self.config[f'{agent}_provider'],
                model=self.config[f'{agent}_model'],
                temperature=self.config['temperature']
            )
        
        try:
            response = await _call()
            
            if not response['success']:
                raise LLMCallException(response['error'])
            
            # Check cost limit
            self.check_cost_limit(response['cost'])
            
            return response
        
        except Exception as e:
            self.audit_logger.error(f"LLM call failed after retries: {e}")
            raise
    
    def generate_batch_outputs(self, batch_id: str, reviews: List[Dict]):
        """
        Generate all output formats for batch
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provider = self.config['provider']
        model = self.config['model'].replace('-', '_')
        
        output_dir = f"outputs/{batch_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Consolidated CSV
        csv_path = f"{output_dir}/{timestamp}_{provider}_{model}_consolidated.csv"
        self.generate_csv_output(reviews, csv_path)
        
        # 2. Individual reviews (Markdown)
        for review in reviews:
            paper_id = review['paper_id']
            md_path = f"{output_dir}/{timestamp}_{provider}_{model}_{paper_id}_review.md"
            self.generate_markdown_review(review, md_path)
        
        # 3. HTML Dashboard
        dashboard_path = f"{output_dir}/{timestamp}_{provider}_{model}_dashboard.html"
        self.generate_html_dashboard(reviews, dashboard_path)
        
        # 4. Batch summary report
        summary_path = f"{output_dir}/{timestamp}_{provider}_{model}_summary.md"
        self.generate_batch_summary(reviews, summary_path)
        
        self.audit_logger.info(f"Generated all outputs for batch {batch_id} in {output_dir}")
```

---

## 10. Deployment & Scaling

### 10.1 Deployment Options

#### Option 1: Local Desktop Application
```bash
# Using PyInstaller to create standalone executable
pyinstaller --onefile --windowed \
    --add-data "config:config" \
    --add-data "prompts:prompts" \
    --name "AcademicReviewSystem" \
    main.py
```

#### Option 2: Self-Hosted Server
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./reviews.db
    volumes:
      - ./papers:/app/papers
      - ./outputs:/app/outputs
      - ./config:/app/config
    command: uvicorn main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000

  # Optional: Redis for job queue (if scaling to multiple workers)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

#### Option 3: Cloud Deployment (AWS/GCP/Azure)
- Backend: AWS Lambda + API Gateway (serverless)
- Storage: S3 for papers and outputs
- Database: RDS (PostgreSQL) or DynamoDB
- Frontend: CloudFront + S3 (static hosting)
- Job Queue: SQS or Cloud Tasks

### 10.2 Performance Optimization

```python
# Optimizations for large batches (500+ papers)

# 1. Batch LLM calls where possible
async def batch_extract_criteria(papers: List[Dict], criterion: Dict):
    """
    Extract same criterion across multiple papers in one API call
    (if model supports batching)
    """
    pass

# 2. Smart caching strategy
class ExtractionCache:
    """
    Cache extractions with invalidation logic
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.memory_cache = {}  # LRU cache in memory
    
    def get(self, paper_id: str, criterion_id: str, criteria_version: str) -> Optional[Dict]:
        """
        Check cache, invalidate if criteria changed
        """
        cache_key = f"{paper_id}_{criterion_id}_{criteria_version}"
        
        # Check memory first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk
        cache_file = f"{self.cache_dir}/{cache_key}.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.memory_cache[cache_key] = data
                return data
        
        return None

# 3. Progressive output generation
async def stream_reviews_as_completed(batch_id: str):
    """
    Yield reviews as they complete rather than waiting for entire batch
    """
    async for review in review_generator:
        # Save immediately
        save_review_to_db(review)
        
        # Update dashboard in real-time
        update_dashboard(batch_id, review)
        
        # Yield for streaming response
        yield review

# 4. Database query optimization
def get_batch_reviews_optimized(batch_id: str):
    """
    Use SQL joins to fetch all related data in one query
    """
    query = """
    SELECT 
        r.id, r.overall_score, r.recommendation,
        p.title, p.authors,
        json_group_array(
            json_object(
                'criterion_id', e.criterion_id,
                'score', e.score,
                'justification', e.score_justification
            )
        ) as extractions
    FROM reviews r
    JOIN papers p ON r.paper_id = p.id
    LEFT JOIN extractions e ON r.id = e.review_id
    WHERE r.batch_id = ?
    GROUP BY r.id
    """
    return db.execute(query, [batch_id]).fetchall()
```

### 10.3 Monitoring & Alerting

```python
# monitoring.py

from dataclasses import dataclass
from typing import List
import time

@dataclass
class SystemMetrics:
    """Real-time system metrics"""
    active_jobs: int
    queue_length: int
    avg_paper_processing_time: float
    total_api_calls_last_hour: int
    total_cost_last_hour: float
    error_rate: float
    cache_hit_rate: float

class MonitoringService:
    """
    Monitor system health and performance
    """
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5%
            'cost_per_hour': 50.0,  # $50/hour
            'avg_processing_time': 300,  # 5 minutes per paper
        }
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current metrics"""
        return SystemMetrics(
            active_jobs=self.count_active_jobs(),
            queue_length=self.get_queue_length(),
            avg_paper_processing_time=self.calculate_avg_processing_time(),
            total_api_calls_last_hour=self.count_api_calls_last_hour(),
            total_cost_last_hour=self.calculate_cost_last_hour(),
            error_rate=self.calculate_error_rate(),
            cache_hit_rate=self.calculate_cache_hit_rate()
        )
    
    def check_alerts(self, metrics: SystemMetrics) -> List[str]:
        """Check if any thresholds exceeded"""
        alerts = []
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        
        if metrics.total_cost_last_hour > self.alert_thresholds['cost_per_hour']:
            alerts.append(f"High hourly cost: ${metrics.total_cost_last_hour:.2f}")
        
        if metrics.avg_paper_processing_time > self.alert_thresholds['avg_processing_time']:
            alerts.append(f"Slow processing: {metrics.avg_paper_processing_time:.0f}s per paper")
        
        return alerts
    
    def send_alert(self, alerts: List[str]):
        """Send alerts via configured channel (email, Slack, etc.)"""
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
            # TODO: Implement email/Slack notification
```

---

## 11. User Guide & Documentation

### 11.1 Quick Start Guide

```markdown
# Quick Start: Academic Review System

## 1. Installation

### Desktop Application
1. Download the installer for your platform (Windows/Mac/Linux)
2. Run the installer and follow prompts
3. Launch "Academic Review System"

### Self-Hosted
```bash
git clone https://github.com/your-org/academic-review-system
cd academic-review-system
docker-compose up -d
```
Access at http://localhost:3000

## 2. Initial Configuration

### Step 1: Set up API Keys
1. Navigate to Settings → API Configuration
2. Enter your API keys for desired LLM providers
3. Test connection with "Verify Keys" button

### Step 2: Configure Review Criteria
1. Go to Criteria Configuration
2. Choose a preset (e.g., "Development Economics") or start from scratch
3. Adjust weights to total 100%
4. Customize scoring scales and thresholds

### Step 3: Upload Papers
1. Click "Import Papers"
2. Select directory or drag-and-drop files
3. Review the paper list and deselect any files to exclude
4. Click "Confirm Selection"

## 3. Run Your First Review

1. Select papers to review (check boxes)
2. Click "Start Review"
3. Choose LLM provider and model
4. Review cost estimate
5. Click "Confirm and Process"
6. Monitor progress in real-time
7. Access results once complete

## 4. Review Outputs

- **Individual Reviews**: Click any paper to view full review
- **CSV Export**: Download consolidated spreadsheet
- **Dashboard**: View aggregate statistics and visualizations
- **Edit Reviews**: Click "Override" to adjust LLM assessments
```

### 11.2 Best Practices

```markdown
# Best Practices for Academic Review System

## Prompt Engineering

### Extraction Prompts
- **Be specific** about evidence requirements
- **Include examples** of good vs. poor evidence
- **Request page numbers** for all claims
- **Set confidence thresholds** to flag uncertain assessments

### Synthesis Prompts
- **Emphasize constructive tone** for author feedback
- **Request explicit reasoning** for recommendation
- **Ask for revision suggestions** (actionable, specific)

## Model Selection

### For Extraction (Agent 1)
- **High-context models** (Gemini 2.0 Pro, Claude) for long papers
- **Cost-effective options** (GPT-4o-mini, Deepseek) if budget-constrained
- **Consistent temperature** (0.2-0.3) for reproducibility

### For Synthesis (Agent 2)
- **Reasoning-focused models** (Claude Opus, GPT-4o) for nuanced synthesis
- **Higher context** not as critical (only processes extractions, not full papers)

## Quality Assurance

1. **Always run calibration** on first use with new criteria/model
2. **Manually review 10-20%** of LLM outputs initially
3. **Track human-LLM agreement** rates over time
4. **Flag low-confidence reviews** for priority human review
5. **Iterate on prompts** based on error patterns

## Cost Management

- **Start small**: Test with 5-10 papers before full batch
- **Use caching**: Don't re-extract if criteria unchanged
- **Mix models**: Cheap extraction, premium synthesis
- **Monitor spend**: Set hard limits in configuration
- **Compare providers**: Cost/quality tradeoffs vary by use case

## Ethical Considerations

1. **LLM reviews are tools, not replacements** for human judgment
2. **Always disclose** LLM assistance in review process
3. **Provide author right to human review** if requested
4. **Audit for bias** regularly (check score distributions across demographics)
5. **Protect confidentiality**: Use local deployment for sensitive papers
```

---

## 12. Testing & Validation

### 12.1 Unit Tests

```python
# tests/test_extraction.py

import pytest
from core.processor import ReviewProcessor

@pytest.fixture
def sample_paper():
    return {
        "id": "test_001",
        "content_markdown": "# Sample Paper\n\n## Introduction\n...",
        "token_count": 5000
    }

@pytest.fixture
def sample_criterion():
    return {
        "id": "empirical_rigor",
        "name": "Empirical Rigor",
        "weight": 20,
        "scale": {"type": "numeric", "range": [1, 5]},
        "threshold": {"min_acceptable": 3}
    }

@pytest.mark.asyncio
async def test_extraction_returns_valid_schema(sample_paper, sample_criterion):
    """Test that extraction outputs valid JSON schema"""
    processor = ReviewProcessor(test_config, test_db)
    
    extraction = await processor.extract_criterion_evidence(
        sample_paper, 
        sample_criterion
    )
    
    assert "score" in extraction
    assert 1 <= extraction["score"] <= 5
    assert "evidence" in extraction
    assert isinstance(extraction["evidence"], list)
    assert "confidence" in extraction
    assert 0.0 <= extraction["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_synthesis_respects_weights(sample_extractions, sample_criteria):
    """Test that weighted scoring is correct"""
    processor = ReviewProcessor(test_config, test_db)
    
    review = await processor.synthesize_review(
        sample_paper,
        sample_extractions,
        sample_criteria
    )
    
    # Manually calculate expected score
    expected_score = sum(
        e["score"] * (c["weight"] / 100) 
        for e, c in zip(sample_extractions, sample_criteria)
    )
    
    assert abs(review["overall_score"] - expected_score) < 0.1

def test_recommendation_logic_thresholds():
    """Test that recommendation thresholds work correctly"""
    assert calculate_recommendation(score=90, failures=[]) == ("Accept", "...")
    assert calculate_recommendation(score=55, failures=[]) == ("Reject", "...")
    assert calculate_recommendation(score=75, failures=["critical"]) == ("Reject", "...")
```

### 12.2 Integration Tests

```python
# tests/test_integration.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_review_pipeline():
    """Test complete pipeline from paper upload to output generation"""
    
    # Setup
    test_paper_path = "tests/fixtures/sample_paper.pdf"
    test_criteria = load_test_criteria()
    test_config = load_test_config()
    
    # Upload paper
    paper = await upload_paper(test_paper_path)
    assert paper["status"] == "ready"
    
    # Process review
    review = await process_single_paper(paper, test_criteria, test_config)
    
    # Validate outputs
    assert review["overall_score"] is not None
    assert review["recommendation"] in ["Accept", "Maybe", "Reject"]
    assert len(review["extractions"]) == len(test_criteria)
    
    # Check output files generated
    assert os.path.exists(f"outputs/{review['id']}_review.md")
    assert os.path.exists(f"outputs/{review['id']}_review.docx")

@pytest.mark.integration
def test_batch_processing_with_failures():
    """Test that batch continues after individual paper failures"""
    
    papers = [
        valid_paper_1,
        corrupted_paper,  # Should fail
        valid_paper_2
    ]
    
    batch_result = process_batch(batch_id, papers, criteria, config)
    
    assert batch_result["completed"] == 2
    assert batch_result["failed"] == 1
    assert len(batch_result["reviews"]) == 3  # Includes error placeholder
```

---

## 13. Appendix

### 13.1 Sample Prompt Templates

#### Extractor System Prompt
```
You are an expert academic reviewer specializing in {domain}. Your role is to objectively analyze research papers and extract evidence for specific evaluation criteria.

Key principles:
1. Base all assessments on explicit evidence from the paper
2. Cite specific quotes with page numbers
3. Distinguish between strong and weak evidence
4. Note when expected elements are missing
5. Provide calibrated confidence scores
6. Maintain objectivity and avoid personal bias

Output format: Always respond with valid JSON matching the provided schema.
```

#### Synthesizer System Prompt
```
You are a senior academic editor synthesizing multiple criterion-level assessments into a coherent, constructive review.

Writing guidelines:
1. Professional, collegial tone suitable for peer review
2. Balance critique with recognition of strengths
3. Provide actionable, specific revision suggestions
4. Justify recommendations with explicit reasoning
5. Identify patterns across criteria (e.g., methods strong but theory weak)
6. For borderline cases, clearly state what would change the decision

Output format: Always respond with valid JSON matching the provided schema.
```

### 13.2 Example Criterion Configuration (YAML)

```yaml
# Example: Computer Science conference review criteria

domain: computer_science
criteria:
  - id: technical_contribution
    name: Technical Contribution
    description: |
      Novelty and significance of the technical approach, algorithm, or system.
      Does the work advance the state-of-the-art?
    weight: 25
    scale:
      type: numeric
      range: [1, 5]
      labels:
        1: "No contribution / Incremental"
        2: "Minor contribution"
        3: "Moderate contribution"
        4: "Significant contribution"
        5: "Outstanding contribution"
    sub_questions:
      - "What is novel about the approach?"
      - "How does it compare to existing solutions?"
      - "Is the contribution clearly articulated?"
    threshold:
      min_acceptable: 3
      auto_reject_below: 2

  - id: experimental_evaluation
    name: Experimental Evaluation
    description: |
      Rigor and comprehensiveness of experimental methodology.
      Are claims supported by evidence?
    weight: 20
    scale:
      type: numeric
      range: [1, 5]
    sub_questions:
      - "Are baselines appropriate and well-chosen?"
      - "Are datasets standard/relevant?"
      - "Are experiments reproducible?"
      - "Are ablation studies included?"
    threshold:
      min_acceptable: 3

  - id: reproducibility
    name: Reproducibility
    description: |
      Availability of code, data, and sufficient implementation details.
    weight: 15
    scale:
      type: qualitative
      options: ["Excellent", "Good", "Adequate", "Poor", "Insufficient"]
    sub_questions:
      - "Is code publicly available?"
      - "Are hyperparameters specified?"
      - "Can results be reproduced from the paper alone?"

  - id: writing_quality
    name: Writing Quality
    description: |
      Clarity of presentation, organization, and language.
    weight: 10
    scale:
      type: numeric
      range: [1, 5]
    threshold:
      min_acceptable: 2  # Lower threshold - can improve with editing

  - id: related_work
    name: Related Work
    description: |
      Completeness and accuracy of literature survey.
      Proper positioning relative to prior work.
    weight: 10
    scale:
      type: numeric
      range: [1, 5]

  - id: significance
    name: Significance / Impact
    description: |
      Potential impact on the field. Is this work important?
    weight: 20
    scale:
      type: numeric
      range: [1, 5]
    sub_questions:
      - "Will this influence future research?"
      - "Does it solve an important problem?"
      - "Is the scope appropriate?"
    threshold:
      min_acceptable: 3
```

### 13.3 Troubleshooting Guide

```markdown
# Troubleshooting Common Issues

## Problem: Papers failing to parse

**Symptoms**: Status shows "Failed" after upload

**Solutions**:
1. Check if PDF is scanned image (needs OCR)
2. Try converting to text first using external tool
3. Ensure file isn't password-protected
4. Check file size < 50MB limit

## Problem: LLM returns invalid JSON

**Symptoms**: Parsing errors in logs, missing extractions

**Solutions**:
1. Lower temperature (try 0.1-0.2)
2. Add explicit JSON formatting instructions to prompt
3. Try different model (some parse JSON better)
4. Check for prompt injection from paper content

## Problem: Reviews seem inconsistent

**Symptoms**: Similar papers get very different scores

**Solutions**:
1. Run calibration check with known papers
2. Add few-shot examples to prompts
3. Lower temperature for more consistent scoring
4. Check if criteria descriptions are ambiguous

## Problem: Cost exceeds budget

**Symptoms**: API spending higher than estimated

**Solutions**:
1. Use cheaper models for extraction (GPT-4o-mini, Deepseek)
2. Enable caching to avoid re-processing
3. Reduce max_tokens in config
4. Process in smaller batches to monitor costs

## Problem: Slow processing speed

**Symptoms**: Taking >5 minutes per paper

**Solutions**:
1. Enable parallel extraction (max_parallel_extractions: 5)
2. Use faster models (Gemini Flash, Claude Sonnet vs Opus)
3. Check rate limit settings aren't too conservative
4. Consider batching API calls if provider supports it

## Problem: Reviews lack specificity

**Symptoms**: Generic feedback, no page references

**Solutions**:
1. Update extraction prompts to explicitly request quotes
2. Add examples of good vs poor evidence
3. Increase output token limit
4. Try model with better instruction-following (Claude, GPT-4)
```

---

## 14. Roadmap & Future Enhancements

### Phase 1 (MVP) - ✓ Current Specification
- MarkItDown ingestion
- Two-agent architecture
- Multi-provider LLM support
- Customizable criteria
- CSV/Markdown/Word outputs
- Basic dashboard

### Phase 2 (3-6 months)
- **Multi-language support**: Review papers in non-English languages
- **Citation analysis**: Integrate with Semantic Scholar API for citation context
- **Collaborative features**: Multi-reviewer workflows with conflict resolution
- **Fine-tuned models**: Train small models on domain-specific reviews
- **Mobile app**: iOS/Android for on-the-go review access

### Phase 3 (6-12 months)
- **Meta-review aggregation**: Combine multiple LLM reviews (ensemble)
- **Active learning**: Improve prompts based on human override patterns
- **Conference integration**: API for uploading directly from CMT/EasyChair
- **Bias detection**: Automated checks for demographic/