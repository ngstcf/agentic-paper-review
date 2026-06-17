# setup_literature_review.py
import os
import shutil
import argparse
from pathlib import Path
from core.config_loader import MODEL_DEFAULTS

def setup_literature_run_directory(run_dir: str, config_dir: str = "config"):
    """Set up a new literature-grounded review run directory with the required structure.

    This setup script is for the standalone literature review pipeline:
    - Uses literature_sources.yaml for configuration (not criteria.yaml)
    - Runs 4-stage pipeline: Librarian → Reader → Fact-Checker → Critic
    - Outputs GroundedReview with research trajectory and novelty analysis

    Usage:
        python setup_literature_review.py --run-dir my_literature_review
    """
    run_path = Path(run_dir)

    # Create directory structure
    papers_dir = run_path / "papers"
    input_dir = run_path / "input"
    outputs_dir = run_path / "outputs"
    literature_dir = run_path / "literature"  # For baseline reference papers
    reports_dir = outputs_dir / "reports"
    reviews_dir = outputs_dir / "reviews"

    for directory in [papers_dir, input_dir, outputs_dir, literature_dir, reports_dir, reviews_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copy literature-specific config files
    if os.path.exists(config_dir):
        print(f"Copying literature config files from {config_dir} to {input_dir}")

        # Copy literature_sources.yaml (required for literature pipeline)
        literature_sources_src = Path(config_dir) / "literature_sources.yaml"
        if literature_sources_src.exists():
            shutil.copy(literature_sources_src, input_dir / "literature_sources.yaml")
            print(f"  ✓ Copied literature_sources.yaml")
        else:
            print(f"  ⚠ Warning: literature_sources.yaml not found in {config_dir}")
            print(f"    Creating default literature_sources.yaml template...")
            create_default_literature_sources(input_dir / "literature_sources.yaml")

        # Copy prompts directory (if literature-specific prompts exist)
        prompts_src = Path(config_dir) / "prompts_literature"
        prompts_dst = input_dir / "prompts"
        if not prompts_src.exists():
            # Fallback to regular prompts
            prompts_src = Path(config_dir) / "prompts"

        if prompts_src.exists():
            if prompts_dst.exists():
                shutil.rmtree(prompts_dst)
            shutil.copytree(prompts_src, prompts_dst)
            print(f"  ✓ Copied prompts from {prompts_src}")

        # Create a .env file with LLM PARAMETERS for literature pipeline
        env_file = input_dir / ".env"
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write(f"""# LLM Configuration for Literature-Grounded Review
# ======================================
# Run Directory: {run_dir}
# API keys are loaded from the global .env file at the project root
# ======================================

# Stage 1: Librarian (Baseline Reference Creation)
PROVIDER_LIBRARIAN={MODEL_DEFAULTS["synthesizer_provider"]}
LIBRARIAN_MODEL={MODEL_DEFAULTS["synthesizer_model"]}
LIBRARIAN_TEMPERATURE=0.2

# Stage 2: Reader (Novelty Ranking & Extraction)
PROVIDER_READER={MODEL_DEFAULTS["extractor_provider"]}
READER_MODEL={MODEL_DEFAULTS["extractor_model"]}
READER_TEMPERATURE=0.3

# Stage 3: Fact-Checker (Claim Verification)
PROVIDER_FACT_CHECKER={MODEL_DEFAULTS["synthesizer_provider"]}
FACT_CHECKER_MODEL={MODEL_DEFAULTS["synthesizer_model"]}
FACT_CHECKER_TEMPERATURE=0.1

# Stage 4: Critic (Grounded Synthesis)
PROVIDER_CRITIC={MODEL_DEFAULTS["synthesizer_provider"]}
CRITIC_MODEL={MODEL_DEFAULTS["synthesizer_model"]}
CRITIC_TEMPERATURE=0.2

# General Parameters
TEMPERATURE=0.2
MAX_RETRIES=3
MAX_PARALLEL_EXTRACTIONS=5

# Literature Source Configuration
LITERATURE_SOURCES_PATH={input_dir / 'literature_sources.yaml'}
""")
            print(f"  ✓ Created .env file with literature pipeline configuration")
    else:
        print(f"⚠ Warning: Config directory {config_dir} not found.")
        print(f"  You'll need to create literature_sources.yaml manually.")

    print(f"\n{'='*60}")
    print(f"✅ Literature Review Directory Set Up")
    print(f"{'='*60}")
    print(f"📁 Run Directory: {run_path}")
    print(f"\n📄 Place papers to review in: {papers_dir}")
    print(f"📚 Place baseline literature in: {literature_dir}")
    print(f"⚙️  Modify LLM parameters in: {env_file}")
    print(f"📖 Configure literature sources in: {input_dir / 'literature_sources.yaml'}")
    print(f"\n🔑 API keys are loaded from the global .env file at the project root")
    print(f"\n🚀 To run the literature review:")
    print(f"   python run_review_literature.py --run-dir {run_dir}")

def create_default_literature_sources(output_path: Path):
    """Create a default literature_sources.yaml template."""
    default_content = """# Literature Sources Configuration
# ======================================
# This file defines the baseline literature sources for the literature-grounded review pipeline.
# These sources will be used by the Librarian agent to create a baseline reference,
# and by the Reader agent to rank novelty and identify research trajectory.

# Source categories
categories:
  - name: "Foundational Papers"
    description: "Key papers that established the field"
    topics:
      - "knowledge representation"
      - "hierarchical analysis"
      - "research synthesis"

  - name: "Recent Advances"
    description: "State-of-the-art developments (last 2-3 years)"
    topics:
      - "large language models"
      - "AI agents"
      - "automated review systems"

# Literature sources (baseline papers)
sources:
  - id: "baseline_001"
    title: "Example Baseline Paper 1"
    authors: ["Author One", "Author Two"]
    year: 2024
    venue: "Example Conference"
    topics: ["knowledge representation", "hierarchical analysis"]
    key_contributions:
      - "Established methodology for hierarchical knowledge organization"
      - "Introduced structural analysis framework"

  - id: "baseline_002"
    title: "Example Baseline Paper 2"
    authors: ["Author Three"]
    year: 2023
    venue: "Example Journal"
    topics: ["automated review", "AI agents"]
    key_contributions:
      - "Developed automated review pipeline"
      - "Proposed quality metrics for synthesis"

# Research trajectory definition
research_trajectory:
  starting_point: "earliest_work"
  progression:
    - stage: "initial_concepts"
      description: "Early work establishing foundational concepts"
    - stage: "methodological_advances"
      description: "Development of key methodologies"
    - stage: "current_state"
      description: "State-of-the-art approaches"
    - stage: "open_challenges"
      description: "Current limitations and future directions"

# Novelty dimensions for ranking
novelty_dimensions:
  - name: "methodological_novelty"
    weight: 0.3
    description: "New methods or approaches"

  - name: "conceptual_novelty"
    weight: 0.3
    description: "New concepts or frameworks"

  - name: "application_novelty"
    weight: 0.2
    description: "New applications or use cases"

  - name: "empirical_novelty"
    weight: 0.2
    description: "New empirical findings or data"
"""

    with open(output_path, "w") as f:
        f.write(default_content)
    print(f"  ✓ Created default literature_sources.yaml template")


def main():
    parser = argparse.ArgumentParser(
        description="Set up a new literature-grounded review run directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup
  python setup_literature_review.py --run-dir my_literature_review

  # Setup with custom config directory
  python setup_literature_review.py --run-dir my_literature_review --config-dir custom_config

The literature review pipeline uses:
  - literature_sources.yaml (not criteria.yaml)
  - 4-stage pipeline: Librarian → Reader → Fact-Checker → Critic
  - Outputs GroundedReview with research trajectory analysis
        """
    )
    parser.add_argument("--run-dir", required=True, help="Directory for this literature review run")
    parser.add_argument("--config-dir", default="config", help="Source config directory")
    args = parser.parse_args()

    setup_literature_run_directory(args.run_dir, args.config_dir)


if __name__ == "__main__":
    main()