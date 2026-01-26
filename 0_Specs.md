Input: directory path to the store of academic papers, which can be in PDF, md, txt, or docx format. with the option to upload papers                                 
Output: Structured review (Markdown, Word, or LaTeX), including scores, comments, and recommendations (Accept | Reject | Maybe) with rationale
Consolidated into a csv file [title, criterion 1, criterion 2, ..., Recommendation (Accept | Reject | Maybe)] 
LLM Provider selection: OpenAI, Gemini, Deepseek, Claude, Perplexity, Ollama
LLM model selection: OpenAI(GPT-5, GPT-mini, GPT-4o-mini),Gemini (gemini-2.5-flash, gemini-2.5-pro), Deepseek (deepseek-chat, deepseek-reasoner), Claude (claude-opus-4-20250514,claude-sonnet-4-20250514, claude-3-7-sonnet-latest), perplexity (sonar, sonar-pro), Ollama [https://x.org/api/model]
LLM parameters: available in .env, ollama requires a key
Reviewer interface: Editable table or form for modifying criteria, weights, and scoring scales.


* Default criteria suggestions (editable):
    1. Theoretical contribution 
    2. Empirical rigor
    3. Data transparency
    4. Methodological appropriateness
    5. Policy relevance / practical implications
    6. Clarity and structure
    7. Originality and novelty [examining related work and contributions]
    8. Literature engagement
    9. Statistical robustness
  
* Editable parameters:
    * Add/delete criteria
    * Adjust weighting (e.g., total 100%)
    * Change evaluation scale (e.g., 1–5, 1–10, qualitative)

Phase 1: ingest papers [dir] (whole dir (de)selectable for inclusion in the scope) | [manual upload] [((individual papers are de)selectable for inclusion in the scope)]
Phase 2: Run analysis engine [evidence extraction and narrative review drafting] and can be rerun anytime 
Name output files with timestamp to distinguish and LLMprovider the different runs.




{
  "weights": {
    "impact": 0.35,
    "methodological_rigor": 0.25,
    "author_credibility": 0.15,
    "practicality": 0.15,
    "momentum": 0.10
  },
  "criteria": {
    "impact": {
      "description": "Potential to advance the field or shift thinking",
      "indicators": [
        "Novel methodologies, architectures, or theoretical frameworks",
        "Significant performance improvements (>10% on established benchmarks)",
        "Solves previously intractable problems or opens new research directions",
        "Cross-domain applicability or generalizable insights",
        "Challenges existing assumptions with strong evidence"
      ],
      "red_flags": [
        "Incremental improvements without novelty",
        "Overfitting to specific benchmarks",
        "Claims without proper baselines"
      ]
    },
    "methodological_rigor": {
      "description": "Scientific quality and reproducibility",
      "indicators": [
        "Comprehensive ablation studies showing what actually matters",
        "Multiple datasets/domains tested (not just one benchmark)",
        "Statistical significance testing and error bars",
        "Clear limitations section acknowledging weaknesses",
        "Reproducibility checklist completed (if available)",
        "Fair comparison with state-of-the-art baselines"
      ],
      "red_flags": [
        "Cherry-picked results or missing negative results",
        "Vague experimental setup",
        "No ablation studies"
      ]
    },
    "author_credibility": {
      "description": "Track record and institutional backing",
      "indicators": [
        "Leading AI labs (DeepMind, OpenAI, Anthropic, Tencent, Alibaba, DeepSeek, Meta AI, Google Research, etc.)",
        "Top academic institutions (Stanford, MIT, CMU, Berkeley, etc.)",
        "Authors with strong publication history in relevant area (h-index >20 for senior, >10 for junior)",
        "Collaboration between industry and academia",
        "Previously published influential papers in the domain"
      ],
      "notes": "Lower weight because good work can come from anywhere; don't filter out unknown authors with strong methodology"
    },
    "practicality": {
      "description": "Usability and accessibility for the community",
      "indicators": [
        "Code released (GitHub/HuggingFace) or committed release date",
        "Models/weights available or planned release",
        "Reasonable computational requirements (or efficient alternatives provided)",
        "Clear documentation and usage examples",
        "Permissive licensing (Apache, MIT, etc.)",
        "API access or demo available"
      ],
      "considerations": [
        "Theory papers may score lower here but still be valuable",
        "Hardware efficiency gains (memory, speed, parallelization)"
      ]
    },
    "momentum": {
      "description": "Timeliness and community interest",
      "indicators": [
        "Addresses current research priorities: reasoning, agents, multimodal, alignment, efficiency",
        "Builds on or challenges recent breakthrough work (<12 months old)",
        "Early social signals: Twitter/X engagement, community discussion",
        "Presented at major conferences (NeurIPS, ICML, ICLR, ACL, CVPR)",
        "Multiple follow-up papers already emerging"
      ],
      "notes": "Lowest weight to avoid recency bias; foundational work may have lasting value"
    }
  },
  "filtering_rules": {
    "must_have": [
      "At least 2 out of 3: impact, methodological rigor, or author credibility score high",
      "No critical red flags in methodology"
    ],
    "priority_boost": [
      "Preprints from established labs (fast-track review)",
      "Papers with exceptional open-source contributions",
      "Interdisciplinary work connecting multiple domains"
    ],
    "deprioritize": [
      "Minor incremental improvements (<5% gains)",
      "Papers with no code/data and vague methodology",
      "Pure application papers without methodological contribution"
    ]
  },
  "scoring_guide": {
    "scale": "0-10 for each criterion",
    "thresholds": {
      "must_read": "≥8.0 weighted average",
      "high_priority": "7.0-7.9",
      "moderate": "6.0-6.9",
      "low_priority": "<6.0"
    }
  }
}


academic_review_system/
├── config/
│   ├── .env
│   └── ...
├── papers/
│   └── ...
├── outputs/
│   └── ...
├── core/
│   ├── __init__.py         
│   ├── config_loader.py
│   ├── llm_wrapper.py
│   └── ...
├── agents/
│   ├── __init__.py        
│   ├── agent_extractor.py
│   └── agent_synthesizer.py
├── utilities/
│   ├── __init__.py       
│   ├── output_generator.py
│   └── helpers.py
└── run_review.py


==

That's an excellent question. It gets to the core logic of the system, and the answer highlights the different roles of the Python code versus the LLMs.

The two values are determined in completely different ways:

1.  **`overall_score`** is **mathematically calculated by our Python script.**
2.  **`decision_confidence`** is **determined by the Synthesizer LLM (Agent 2)** based on instructions in its prompt.

Here is a detailed breakdown of each.

-----

### 1\. How the `overall_score` is Determined (Calculated by Python)

The `overall_score` is a **weighted average** calculated by the `calculate_recommendation` function inside `agents/agent_synthesizer.py`.

The script does *not* just add up the scores. It follows these steps:

1.  **Get Agent 1's Score:** It takes the raw score from Agent 1 (e.g., `score: 4`).
2.  **Normalize the Score:** It finds the maximum possible score for that criterion (e.g., if the scale is 1-5, the `max_score` is 5). It then divides the raw score by the max score to get a normalized value (e.g., `4 / 5 = 0.8`).
3.  **Apply the Weight:** It multiplies this normalized score by the criterion's weight (defined in `criteria.yaml`). For example, if "Empirical Rigor" has a `weight: 20`, the calculation is `0.8 * 20 = 16`.
4.  **Sum All Scores:** It repeats this for all criteria and adds them together.
5.  **Final Score:** This final sum is the `overall_score` out of 100.

#### The Exact Code

This logic is in the `calculate_recommendation` function in `agents/agent_synthesizer.py`:

```python
def calculate_recommendation(
    extractions: List[Extraction],
    config: Config
) -> Tuple[str, str, float, Dict[str, WeightedBreakdown]]:
    
    # ... (setup) ...
    
    for ext in extractions:
        criterion = config.get_criterion_by_id(ext.criterion_id)
        if criterion:
            scale = criterion.get('scale', {})
            scale_labels = scale.get('labels', {})
            max_score = len(scale_labels)  # <--- Finds the max (e.g., 5)
            
            if max_score > 0:
                weight = criterion.get('weight', 0)
                
                # --- THIS IS THE KEY CALCULATION ---
                normalized_score = (ext.score / max_score) * weight
                total_score += normalized_score
                # -------------------------------------
                
                weighted_scores[ext.criterion_id] = WeightedBreakdown(
                    score=ext.score,
                    weight=weight,
                    weighted_score=normalized_score
                )

    # ... (setup) ...
    
    # --- FINAL SCORE (normalized to 100) ---
    total_weight = sum(c.get('weight', 0) for c in config.get_criteria())
    final_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
    # -----------------------------------------
    
    # ... (recommendation logic) ...
        
    return rec, f"Overall score: {final_score:.1f}", final_score, weighted_scores
```

-----

### 2\. How `decision_confidence` is Determined (Provided by the LLM)

The `decision_confidence` is **not calculated by Python at all.**

Instead, we **instruct the Synthesizer LLM (Agent 2) to provide this value** as part of its JSON response. The goal is to have the LLM evaluate its *own* synthesis and tell us how confident it is.

We do this by giving it explicit instructions in the prompt.

#### The Exact Prompt

This instruction is in `config/prompts/synthesizer_user.txt`:

```
# Task
Synthesize these criterion-level assessments into a comprehensive review.

Output as valid JSON matching this schema:
{{
  "executive_summary": "...",
  "detailed_assessment": {{ ... }},
  "criterion_narrative": {{ ... }},
  "recommendation": "...",
  "recommendation_rationale": "...",
  "revision_suggestions": ["..."],
  
  // --- THIS IS THE INSTRUCTION ---
  "decision_confidence": <float: 0.0-1.0>
  // -------------------------------
}}
```

*(Note: I'm showing the schema part of the prompt, as that's what the LLM uses to structure its output. The system prompt also tells it to be objective.)*

The LLM is *supposed* to look at the `confidence` scores from Agent 1 (which are passed to it in the `json_dump_of_extractions`) and use those to inform its final `decision_confidence`. If Agent 1 was very uncertain about several criteria, Agent 2 *should* report a lower overall confidence.

### Summary: Python vs. LLM

Here is a simple table to summarize the difference:

| Feature | `overall_score` | `decision_confidence` |
| :--- | :--- | :--- |
| **Who Calculates It?** | **Python Script** | **Agent 2 (LLM)** |
| **Source of Truth** | `agents/agent_synthesizer.py` | `config/prompts/synthesizer_user.txt` |
| **Method** | Mathematical weighted average | LLM's self-assessment of its own synthesis |

This distinction is a key part of the multi-agent design. Does this logic make sense, or would you like to explore how we could change or improve either of these calculations?


#### Criteria Modification ####

The system was built from the ground up to be **almost fully dynamic**. You do **not** need to change any Python code to add, remove, or edit criteria.

All the dependencies are handled automatically by the code's dynamic looping.

### The Single Source of Truth

The *only* file you need to edit is: **`config/criteria.yaml`**

-----

### Your 2-Step "To-Do" List

Here is the entire process for adding a new criterion:

#### Step 1: Add the New Criterion to `criteria.yaml`

1.  Open `config/criteria.yaml`.
2.  Go to the bottom of the `criteria:` list.
3.  Copy-paste an existing criterion block (like `originality_novelty`) and use it as a template.
4.  Edit the fields for your new criterion:

<!-- end list -->

```yaml
# ... (all your other criteria above) ...

  - id: originality_novelty
    name: Originality and Novelty
    description: "Evaluates whether the paper addresses a new question, uses a novel dataset, or applies a new method in a creative way."
    weight: 10 # <-- Pay attention to this
    scale:
      type: numeric
      range: [1, 5]
      labels:
        1: "Highly derivative"
        2: "Minor novelty"
        3: "Moderately original"
        4: "Highly original"
        5: "Groundbreaking"
        
  # --- YOUR NEW CRITERION ---
  - id: statistical_robustness # <-- Must be a unique ID
    name: Statistical Robustness
    description: "Evaluates the quality and appropriateness of statistical tests, power analysis, and sensitivity checks."
    weight: 15 # <-- NEW
    scale:
      type: numeric
      range: [1, 5]
      labels:
        1: "Statistically flawed"
        2: "Weak / Inappropriate tests"
        3: "Adequate"
        4: "Robust and appropriate"
        5: "Exceptional / State-of-the-art"
```

  * **`id`:** Must be a unique, one-word string. This is used as the key in the database and CSV header.
  * **`description`:** This is **very important**. Agent 1 (Extractor) injects this description into its prompt to know *what* to look for. A clear description gives you a better result.
  * **`weight`:** See Step 2.

#### Step 2: Adjust All Weights to Sum to 100

This is the **only manual dependency** you must manage.

After adding your new criterion's weight (e.g., `weight: 15`), you must go back and reduce the weights of the other criteria until the total sum of *all* weights is exactly **100**.

If the sum is not 100, the `overall_score` calculation will be incorrect.

-----

### The "Ripple Effect" (What Happens Automatically)

Once you've made those two changes, here is how all the dependencies are automatically handled by the system:

  * **Agent 1 (Extractor):**

      * The `process_paper_extractions` function will automatically see your new criterion.
      * It will dynamically create a **new API call** to the Extractor LLM, using the `description` you provided.
      * This will happen for *every single paper*.

  * **Agent 2 (Synthesizer):**

      * The `calculate_recommendation` function will automatically receive the extraction for your new criterion.
      * It will dynamically factor its score into the `overall_score` using the `weight` you provided.
      * The LLM will see the new criterion's data and automatically include it in the `executive_summary` and `criterion_narrative`.

  * **Markdown Report:**

      * The `save_review_markdown` function will automatically loop over the new criterion and add a new section for it to the final `.md` file.

  * **Consolidated CSV Report:**

      * The `save_consolidated_csv` function will automatically see the new criterion and **add a new column** to your CSV report (e.g., `statistical_robustness_score`).

### ⚠️ Important Implications to Consider

1.  **Cost and Time:** This is the most important dependency. For every 1 criterion you add, you are adding **1 new LLM API call *per paper***.

      * **500 papers x 1 new criterion = 500 additional API calls.**
      * This will increase your total cost and the total time it takes to run a batch.

2.  **Prompt Quality:** The system currently uses a *single* generic prompt (`config/prompts/extractor_user.txt`) for all criteria. This means the **quality of the `description` field** in your `criteria.yaml` is **critical** to getting a good extraction.

3.  **Context Window (Agent 2):** Each new criterion adds more data that must be "stuffed" into the prompt for Agent 2. This is fine for 5, 10, or 15 criteria. If you were to add 50 criteria, you might exceed the LLM's context window.