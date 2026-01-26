#!/usr/bin/env python3
"""
setup_run_literature.py

Set up a new run directory with literature grounding configuration.
This creates a directory structure for literature-grounded paper reviews.
"""

import os
import shutil
import argparse
from pathlib import Path


def setup_run_directory_literature(
    run_dir: str,
    config_dir: str = "config",
    semantic_scholar_api_key: str = None
):
    """
    Set up a new run directory with literature grounding support.

    Args:
        run_dir: Directory for this run
        config_dir: Source config directory
        semantic_scholar_api_key: Optional Semantic Scholar API key
    """
    run_path = Path(run_dir)

    # Create directory structure
    papers_dir = run_path / "papers"
    input_dir = run_path / "input"
    outputs_dir = run_path / "outputs"
    reports_dir = outputs_dir / "reports"
    reviews_dir = outputs_dir / "reviews"

    for directory in [papers_dir, input_dir, outputs_dir, reports_dir, reviews_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copy config files
    if os.path.exists(config_dir):
        print(f"Copying config files from {config_dir} to {input_dir}")

        # Copy criteria.yaml
        shutil.copy(f"{config_dir}/criteria.yaml", input_dir / "criteria.yaml")

        # Copy prompts directory
        prompts_src = Path(config_dir) / "prompts"
        prompts_dst = input_dir / "prompts"
        if prompts_src.exists():
            if prompts_dst.exists():
                shutil.rmtree(prompts_dst)
            shutil.copytree(prompts_src, prompts_dst)

        # Copy literature_sources.yaml
        literature_src = f"{config_dir}/literature_sources.yaml"
        if os.path.exists(literature_src):
            shutil.copy(literature_src, input_dir / "literature_sources.yaml")
            print(f"Copied literature_sources.yaml")
        else:
            print(f"Warning: {literature_src} not found. Literature grounding may not work properly.")

    else:
        print(f"Warning: Config directory {config_dir} not found.")

    # Create .env file with LLM parameters and literature grounding settings
    env_file = input_dir / ".env"
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(f"""# LLM Configuration Parameters for {run_dir}
# ======================================
# API keys are loaded from the global .env file at the project root
# ======================================

# Extraction Configuration
PROVIDER_EXTRACTION=openai
EXTRACTOR_MODEL=gpt-4o-mini

# Synthesis Configuration
PROVIDER_SYNTHESIS=deepseek
SYNTHESIZER_MODEL=deepseek-reasoner

# General Parameters
TEMPERATURE=0.2
MAX_RETRIES=3
MAX_PARALLEL_EXTRACTIONS=5

# Judge Configuration
JUDGE_PROVIDER=google
JUDGE_MODEL=gemini-2.5-flash
JUDGE_TEMPERATURE=0.1

# Literature Grounding Configuration
# ======================================
# Enable/disable literature grounding for this run
LITERATURE_GROUNDING_ENABLED=true

# Semantic Scholar API (optional - free tier works without key)
# Get your API key from: https://www.semanticscholar.org/product/api#api-key
""")
            if semantic_scholar_api_key:
                f.write(f"SEMANTIC_SCHOLAR_API_KEY={semantic_scholar_api_key}\n")
            else:
                f.write("# SEMANTIC_SCHOLAR_API_KEY=your_api_key_here\n")

    print(f"\nRun directory set up at: {run_path}")
    print(f"Please place your papers in: {papers_dir}")
    print(f"\nConfiguration files:")
    print(f"  - {input_dir / 'criteria.yaml'}")
    print(f"  - {input_dir / 'literature_sources.yaml'}")
    print(f"\nLLM parameters in: {input_dir / '.env'}")
    print(f"\nTo run the review:")
    print(f"  python run_review_literature.py {run_dir}")
    print(f"\nTo disable literature grounding:")
    print(f"  python run_review_literature.py {run_dir} --no-literature")
    print(f"  or edit {input_dir / '.env'} and set LITERATURE_GROUNDING_ENABLED=false")


def main():
    parser = argparse.ArgumentParser(
        description="Set up a new run directory with literature grounding support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup
  python setup_run_literature.py --run-dir my_review

  # With Semantic Scholar API key
  python setup_run_literature.py --run-dir my_review --semantic-scholar-key YOUR_KEY

  # Custom config directory
  python setup_run_literature.py --run-dir my_review --config-dir config
        """
    )
    parser.add_argument("--run-dir", required=True, help="Directory for this run")
    parser.add_argument("--config-dir", default="config", help="Source config directory")
    parser.add_argument("--semantic-scholar-key", help="Semantic Scholar API key (optional)")

    args = parser.parse_args()

    setup_run_directory_literature(
        run_dir=args.run_dir,
        config_dir=args.config_dir,
        semantic_scholar_api_key=args.semantic_scholar_key
    )


if __name__ == "__main__":
    main()
