# Agentic Paper Review System

A domain-agnostic, multi-agent LLM system for structured academic paper review. Ingests PDFs (and other formats), evaluates each paper against custom criteria, and produces scored reviews with optional cross-model adjudication and literature grounding. While designed for academic papers, the fully configurable criteria and prompts make it applicable to any document that needs structured evaluation — grant proposals, project reports, policy briefs, or technical documentation.

**Human oversight is central to the design.** The AI produces structured reviews and scores — the final accept/reject decisions are always made by humans. Multi-model comparison and the AI Judge help reviewers focus where it matters by surfacing the papers where models disagree, rather than requiring manual review of every assessment.


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

PDFs are parsed with [Docling](https://github.com/docling-project/docling) (MIT). The first parse downloads its layout and table models from Hugging Face (about 1 GB, once), and parsing uses roughly 2 GB of RAM.

### Configure API Keys

Create a `.env` file at the project root (see `.env_example` for all options):

```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
CUSTOM_OPENAI_API_KEY=your-key
CUSTOM_OPENAI_API_BASE=http://your-server:port/v1

# Model selection (swap freely — any model supported by litellm)
PROVIDER_EXTRACTION=deepseek
EXTRACTOR_MODEL=deepseek-v4-flash
PROVIDER_SYNTHESIS=deepseek
SYNTHESIZER_MODEL=deepseek-v4-flash
JUDGE_PROVIDER=gemini
JUDGE_MODEL=gemini-3.1-flash-lite
```

### Option A: Web Dashboard

```bash
python web_app.py --port 8050
```

Open **http://localhost:8050** to configure models, edit criteria/prompts, run reviews, and view results — all from the browser. See [Web Dashboard](#web-dashboard) for details.

### Option B: Command Line

```bash
# 1. Set up a run directory
python setup_run.py --run-dir my_review_run

# 2. Review criteria and domain BEFORE running
#    Edit my_review_run/input/criteria.yaml — verify that `domain` matches
#    your papers (e.g., computer_science, development_economics).

# 3. Drop papers into my_review_run/papers/

# 4. Run the review
python run_with_custom_params.py \
  --run-dir my_review_run \
  --provider-extraction deepseek \
  --extractor-model deepseek-v4-flash \
  --provider-synthesis openai \
  --synthesizer-model gpt-5.4-mini
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
  --provider-extraction openai --extractor-model gpt-5.4-mini \
  --provider-synthesis openai --synthesizer-model gpt-5.4-mini

python run_with_custom_params.py --run-dir my_review_run \
  --provider-extraction deepseek --extractor-model deepseek-v4-flash \
  --provider-synthesis deepseek --synthesizer-model deepseek-v4-flash

# Step 2: Find conflicts
python compare_reports.py --run-dir my_review_run

# Step 3: Adjudicate
python judge_conflicts.py --run-dir my_review_run
```

## Web Dashboard

A built-in web UI for configuring and running reviews from the browser.

### Start the Dashboard

```bash
python web_app.py --port 8050
```

Open **http://localhost:8050**.

> **Note:** The web dashboard is designed for local use only. It has no authentication — do not expose it on a public network or the open internet. If you need remote access, use an SSH tunnel or VPN.

### Features

- **Start/Stop Runs** — Select a run directory, pick Standard or Literature-Grounded mode, and start reviews with one click
- **Real-time Progress** — Live SSE progress bar, stage indicators, cost tracking, and event log
- **Config Editor** — Inline-edit provider, model, temperature, and other settings per run (API keys remain masked and non-editable)
- **Criteria Editor** — Edit `criteria.yaml` directly in the browser with YAML validation
- **Prompt Editor** — Edit all 4 prompt templates (extractor/synthesizer system/user) with template variable hints
- **Literature Sources Editor** — Edit `literature_sources.yaml` with YAML validation
- **Model Costs Editor** — Maintain custom token pricing for models litellm doesn't know (e.g., new DeepSeek releases) with model cost lookup
- **Batch Processing** — "Batch Run" button to sequentially process all run directories with a progress overview
- **Results Table** — Sortable table with scores, recommendations, cost, and one-click review viewing
- **Review Viewer** — Modal viewer with rendered Markdown for individual paper reviews

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Config | Inline-editable `.env` settings (providers, models, temperatures, etc.) |
| Criteria | Full YAML editor for review criteria with save/reload and validation |
| Prompts | Select and edit prompt templates with template variable reference |
| Sources | YAML editor for literature source configuration |
| Costs | Custom model token pricing ($/million tokens) for accurate cost tracking |

### CLI vs Web Dashboard

The web dashboard covers common workflows. Some advanced features require the CLI:

| Capability | CLI | Web |
|------------|-----|-----|
| Single-directory review | Yes | Yes |
| Literature-grounded review | Yes | Yes |
| Sequential batch processing | Yes | Yes |
| Parallel batch (`--parallel --max-workers`) | Yes | No |
| Batch setup & paper distribution | Yes | No — use `setup_batch_runs.py` |
| No-cache mode (force re-processing) | Yes | No — delete `progress.json` |
| Provider override flags (`--provider-*`) | Yes | Partial — edit `.env` in Config tab |
| Automated conflict comparison | Yes | No — run `compare_reports.py` from CLI |
| Retry & concurrency control | Yes | No |

Use the dashboard for interactive reviews, configuration, and result browsing. Use the CLI for large-scale batch setup, parallel execution, and parameter sweeps.

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

All models are configurable via `.env` — no code changes needed when providers release new models.

| Provider | Example Models |
|----------|---------------|
| OpenAI | gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano |
| DeepSeek | deepseek-v4-pro, deepseek-v4-flash |
| Gemini | gemini-3.5-flash, gemini-3.1-flash-lite |
| Anthropic | claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5 |
| Perplexity | sonar, sonar-pro |
| Custom/Ollama | Any OpenAI-compatible endpoint via `CUSTOM_OPENAI_API_BASE` |

You can mix providers — e.g., DeepSeek for extraction, Gemini for judge, OpenAI for synthesis. Token limits are auto-detected via litellm; override per-role with `MAX_TOKENS_EXTRACTION`, `MAX_TOKENS_SYNTHESIS`, `MAX_TOKENS_JUDGE` in `.env`.

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
| `FAILED criterion` / JSON error | Set `MAX_TOKENS_EXTRACTION=32768` in `.env` to increase output limit |
| `Unsupported parameter: max_tokens` | Known issue with some endpoints; the custom_openai bypass handles this automatically |
| `model isn't mapped yet` | Add the model to `_EXTRA_MODELS` in `core/llm_wrapper.py` or set the provider's `*_API_BASE` env var |
| Re-parse papers | Delete `ingestion_cache.json` |
| Re-review papers | Delete `progress.json` |
| Re-adjudicate | Delete `judge_progress.json` |

## Cost Estimation

Each paper is evaluated per criterion (8 criteria = 8 extraction calls) plus one synthesis call, plus one call per conflict for the Judge.

**Prompt caching** significantly reduces extraction costs: the paper content is placed in the system message prefix and cached across all criterion calls for the same paper. The first criterion pays full price; the remaining 7 hit the cache at up to 90% discount (Anthropic) or are auto-cached (OpenAI, DeepSeek). Criteria are extracted in parallel for each paper.

Always test with 5-10 papers first to verify your criteria and estimate costs before committing to a full batch.

## Citation

If you use this software in your work, please cite it. GitHub's **Cite this repository** button (in the sidebar) gives the reference in APA and BibTeX, generated from [`CITATION.cff`](CITATION.cff). For background on the approach, see the [release announcement](https://c3.unu.edu/blog/agentic-paper-review-release) and [From Months to Days](https://c3.unu.edu/blog/from-months-to-days-ai-assisted-peer-review-with-human-oversight).

```bibtex
@misc{chong2026agentic,
  author = {Chong, Ng},
  title  = {Agentic Paper Review System},
  year   = {2026},
  note   = {Version 2.1.0},
  url    = {https://github.com/ngstcf/agentic-paper-review}
}
```

## License

[MIT](LICENSE)

Versions up to commit `c117aba` parsed PDFs with PyMuPDF4LLM, whose dependencies are licensed AGPL-3.0 (or commercially by Artifex) and, in some releases, PolyForm Noncommercial. If you run one of those versions, the licenses of those dependencies apply to you. Current versions parse PDFs with Docling (MIT) and have no AGPL or noncommercial dependencies.
