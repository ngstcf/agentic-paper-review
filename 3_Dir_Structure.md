academic_review_system/
├── .env                              # ← GLOBAL API KEYS (secure, not in git)
├── .gitignore                        # ← Ensure .env is ignored
├── config/                           # Default configuration templates
├── core/                             # Core modules
├── agents/                           # Agent modules
├── utilities/                        # Utility modules
├── run_review_with_dir.py            # Main script
├── setup_run.py                      # Setup script
├── setup_batch_runs.py               # Setup script: distribute papers from a master dir across multiple run dirs
├── run_with_custom_params.py         # Custom parameters script
├── compare_reports.py                # Report comparison
├── judge_conflicts.py                # Judge conflicts
└── my_review_run/                    # Example run directory
    ├── papers/                       # Papers go here
    ├── input/
    │   ├── .env                      # ← LLM PARAMETERS ONLY (no API keys)
    │   ├── criteria.yaml
    │   └── prompts/
    └── outputs/
        ├── reviews/
        └── reports/


At run time:
my_review_run/                           # Example run directory
├── papers/                              # Your PDF files go here
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── paper3.pdf
├── input/                               # Configuration for this run
│   ├── .env                             # LLM parameters
│   ├── criteria.yaml                    # Review criteria
│   └── prompts/                         # LLM prompts
│       ├── extractor_system.txt
│       ├── extractor_user.txt
│       ├── synthesizer_system.txt
│       └── synthesizer_user.txt
├── outputs/
│   ├── reviews/                         # Individual review files
│   │   ├── paper1_openai_gpt-4o_20230715_103000.md
│   │   ├── paper2_openai_gpt-4o_20230715_103500.md
│   │   └── paper3_openai_gpt-4o_20230715_104000.md
│   └── reports/                         # Consolidated reports
│       ├── report_consolidated_20230715_104500.csv
│       ├── HUMAN_REVIEW_discrepancies_20230715_105000.csv
│       └── JUDGE_VERDICTS_report_20230715_105500.csv
├── ingestion_cache.json                 # Paper parsing cache
├── progress.json                        # Review progress with config hash
└── judge_progress.json                  # Judge progress with config hash