# Agentic Academic Review System

A multi-agent LLM system for structured academic paper review. Ingests PDFs (and other formats), evaluates each paper against custom criteria, and produces scored reviews with optional cross-model adjudication and literature grounding.

By [Ng Chong](https://c3.unu.edu/ng-chong-publications) | UNU Campus Computing Centre

**Full documentation:** https://c3.unu.edu/projects/ai/paperreview/userguide.html
**Blog article:** https://c3.unu.edu/blog/from-months-to-days-ai-assisted-peer-review-with-human-oversight

## How It Works

The system uses a team of AI agents, each with a distinct role:

| Agent | Role |
|-------|------|
| **Specialist** | Conducts detailed, criterion-by-criterion analysis of each paper |
| **Editor** | Synthesizes specialist findings into a polished final review |
| **Judge** | Resolves conflicts between reviews from different AI models |
| **Librarian** | Searches academic databases for related papers (optional) |
| **Fact-Checker** | Verifies suspicious claims like "first study" (optional) |
| **Critic** | Synthesizes reviews with research trajectory analysis (optional) |

### Pipeline

1. **Ingestion** — PDFs, Markdown, DOCX, and TXT files are read and converted to text.
2. **Extraction** — The Specialist reads the paper once per criterion, extracting scores with evidence and quotes.
3. **Synthesis** — The Editor gathers all specialist reports and writes a final weighted review.
4. **Output** — Individual `.md` reviews and a consolidated `.csv` spreadsheet.
5. **(Optional) Comparison + Judge** — Run with multiple models, compare results, and have an AI Judge adjudicate conflicts.

### Literature Grounding (Optional)

Enable with `--literature-grounding` to add a 4-stage literature analysis:

1. **Librarian** — Searches Semantic Scholar, Arxiv, and World Bank for related papers
2. **Reader** — Extracts evidence and ranks novelty (1-5) against baseline literature
3. **Fact-Checker** — Verifies suspicious claims through targeted searches
4. **Critic** — Synthesizes with research trajectory and novelty-adjusted scoring

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
CUSTOM_OPENAI_API_KEY=your-key
CUSTOM_OPENAI_API_BASE=http://your-server:port/v1
```

### Run Your First Review

```bash
# 1. Set up a run directory
python setup_run.py --run-dir my_review_run

# 2. Drop papers into my_review_run/papers/

# 3. Run the review
python run_with_custom_params.py \
  --run-dir my_review_run \
  --provider-extraction deepseek \
  --extractor-model deepseek-reasoner \
  --provider-synthesis openai \
  --synthesizer-model gpt-4o-mini
```

Results appear in `my_review_run/outputs/reviews/` (individual reviews) and `my_review_run/outputs/reports/` (consolidated CSV).

### Literature-Grounded Review

```bash
python setup_run_literature.py --run-dir my_literature_review
python run_review_with_dir_literature.py \
  --run-dir my_literature_review \
  --literature-grounding
```

### Compare Models + AI Judge

```bash
# Step 1: Run with different models (repeat with different flags)
python run_with_custom_params.py --run-dir my_review_run \
  --provider-extraction openai --extractor-model gpt-4o-mini \
  --provider-synthesis openai --synthesizer-model gpt-4o-mini

python run_with_custom_params.py --run-dir my_review_run \
  --provider-extraction deepseek --extractor-model deepseek-reasoner \
  --provider-synthesis deepseek --synthesizer-model deepseek-chat

# Step 2: Find conflicts
python compare_reports.py --run-dir my_review_run

# Step 3: Adjudicate
python judge_conflicts.py --run-dir my_review_run
```

## Customization

### Review Criteria

Edit `my_review_run/input/criteria.yaml` to define what the system evaluates. Each criterion has an `id`, `name`, `description`, `weight`, and scoring `scale`. All weights must sum to 100.

```yaml
criteria:
  - id: empirical_rigor
    name: Empirical Rigor
    description: |
      Assesses the quality of the empirical methods, data,
      and execution. Look for research design, causal
      identification, and statistical analysis.
    weight: 20
    scale:
      type: numeric
      range: [1, 5]
      labels:
        1: "Fundamentally flawed"
        2: "Significant weaknesses"
        3: "Adequate"
        4: "Strong and robust"
        5: "Exceptional / state-of-the-art"
```

### Agent Prompts

Edit the text files in `my_review_run/input/prompts/` to control agent behavior:

| File | Controls |
|------|----------|
| `extractor_system.txt` | Specialist's role and analytical style |
| `extractor_user.txt` | Extraction task and output schema |
| `synthesizer_system.txt` | Editor's tone and editorial perspective |
| `synthesizer_user.txt` | Synthesis task and review structure |

### Recommendation Thresholds

Default thresholds (configurable in `criteria.yaml`):

| Score | Recommendation |
|-------|---------------|
| 85+ | Accept |
| 70-84 | Accept with Revisions |
| 50-69 | Revise and Resubmit |
| <50 | Reject |

### Domain Specialization

Set the `domain` field in `criteria.yaml` to pivot the entire system to a new field (e.g., `machine_learning`, `clinical_psychology`).

## Batch Processing

Distribute large paper collections across multiple run directories:

```bash
python setup_batch_runs.py \
  --master-papers-dir papers_master \
  --base-run-dir run_dir \
  --num-runs 10 \
  --papers-per-run 50 \
  --create-batch-script

# Sequential
python run_batch.py

# Parallel
python run_batch.py --parallel --max-workers 4
```

The system tracks progress per directory and resumes automatically after interruptions.

## Supported LLM Providers

| Provider | Example Models |
|----------|---------------|
| OpenAI | GPT-5, GPT-4o, GPT-4o-mini |
| Gemini | gemini-2.5-flash, gemini-2.5-pro |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Claude | claude-opus-4, claude-sonnet-4, claude-haiku-4.5 |
| Perplexity | sonar, sonar-pro |
| Ollama | Any local model via API |

You can mix providers — e.g., DeepSeek for extraction, OpenAI for synthesis.

## Literature Sources

Configured in `config/literature_sources.yaml`:

| Source | Requires API Key | Citation Data |
|--------|-----------------|---------------|
| Semantic Scholar | Optional (free tier: 100 req/min) | Yes |
| Arxiv | No | No |
| World Bank | No | No |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FAILED criterion` / JSON error | Increase `max_tokens` in `core/llm_wrapper.py` (e.g., 4096 to 8192) |
| `Unsupported parameter: max_tokens` | Known issue with some endpoints; add a bypass rule in `core/llm_wrapper.py` |
| Re-parse papers | Delete `ingestion_cache.json` |
| Re-review papers | Delete `progress.json` |
| Re-adjudicate | Delete `judge_progress.json` |

## Cost Estimation

```
(Papers × Criteria × Extractor cost) + (Papers × Synthesizer cost) + (Conflicts × Judge cost)
```

A 500-paper batch with 8 criteria = ~4,000 extractor calls + 500 synthesizer calls. Always test with 5-10 papers first.

## License

[MIT](LICENSE)
