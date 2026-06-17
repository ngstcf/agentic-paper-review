# setup_batch_literature_review.py
import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Dict
import math
from core.config_loader import MODEL_DEFAULTS


def create_default_literature_sources(output_path: Path):
    """Create a default literature_sources.yaml template."""
    default_content = """# Literature Sources Configuration
# ======================================
# This file defines the baseline literature sources for the literature-grounded review pipeline.

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


def setup_batch_literature_runs(
    master_papers_dir: str,
    master_literature_dir: str,
    base_run_dir: str,
    num_runs: int,
    papers_per_run: int = None,
    config_dir: str = "config",
    shuffle: bool = True,
    distribution: str = "sequential",
    even_distribution: bool = True
) -> List[str]:
    """
    Set up multiple literature-grounded review run directories.

    Args:
        master_papers_dir: Directory containing papers to review
        master_literature_dir: Directory containing baseline literature papers
        base_run_dir: Base name for run directories (e.g., "literature_run")
        num_runs: Number of run directories to create
        papers_per_run: Number of papers per run directory (None for even distribution)
        config_dir: Source config directory
        shuffle: Whether to shuffle papers before distribution
        distribution: How to distribute papers ("sequential", "random", "round_robin")
        even_distribution: Whether to distribute papers evenly when insufficient papers

    Returns:
        List of created run directory paths
    """
    # Validate inputs
    if not os.path.exists(master_papers_dir):
        raise ValueError(f"Master papers directory does not exist: {master_papers_dir}")

    if not os.path.exists(master_literature_dir):
        print(f"Warning: Master literature directory does not exist: {master_literature_dir}")
        print(f"  Literature directory will be created but left empty.")

    if not os.path.exists(config_dir):
        raise ValueError(f"Config directory does not exist: {config_dir}")

    # Get all papers from master directory
    all_papers = []
    supported_extensions = ('.pdf', '.docx', '.doc', '.md', '.txt', '.pptx')

    for root, _, files in os.walk(master_papers_dir):
        for file in files:
            if file.endswith(supported_extensions) and not file.startswith('~'):
                file_path = os.path.join(root, file)
                all_papers.append(file_path)

    total_papers = len(all_papers)

    # Get literature papers (for baseline reference)
    all_literature = []
    if os.path.exists(master_literature_dir):
        for root, _, files in os.walk(master_literature_dir):
            for file in files:
                if file.endswith(supported_extensions) and not file.startswith('~'):
                    file_path = os.path.join(root, file)
                    all_literature.append(file_path)

    # Handle papers_per_run calculation
    if papers_per_run is None:
        papers_per_run = math.ceil(total_papers / num_runs)
        max_papers_needed = total_papers
        print(f"Even distribution: Will distribute {total_papers} papers across {num_runs} directories")
    else:
        max_papers_needed = num_runs * papers_per_run

        if total_papers < max_papers_needed:
            if even_distribution:
                papers_per_run = math.ceil(total_papers / num_runs)
                max_papers_needed = total_papers
                print(f"Even distribution: Adjusted to use all {total_papers} papers across {num_runs} directories")

    # Shuffle papers if requested
    if shuffle:
        random.shuffle(all_papers)

    # Distribute papers
    if even_distribution and total_papers < num_runs * papers_per_run:
        paper_groups = distribute_papers_evenly(all_papers, num_runs)
    else:
        paper_groups = distribute_papers(all_papers[:max_papers_needed], num_runs, papers_per_run, distribution)

    # Create run directories
    created_dirs = []

    for i in range(num_runs):
        run_dir_name = f"{base_run_dir}{i+1}"
        run_dir_path = Path(run_dir_name)

        # Create directory structure (including literature directory)
        papers_dir = run_dir_path / "papers"
        input_dir = run_dir_path / "input"
        outputs_dir = run_dir_path / "outputs"
        literature_dir = run_dir_path / "literature"
        reports_dir = outputs_dir / "reports"
        reviews_dir = outputs_dir / "reviews"

        for directory in [papers_dir, input_dir, outputs_dir, literature_dir, reports_dir, reviews_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Copy literature-specific config files
        print(f"Copying literature config files to {input_dir}")

        # Copy literature_sources.yaml
        literature_sources_src = Path(config_dir) / "literature_sources.yaml"
        if literature_sources_src.exists():
            shutil.copy(literature_sources_src, input_dir / "literature_sources.yaml")
        else:
            print(f"  Creating default literature_sources.yaml...")
            create_default_literature_sources(input_dir / "literature_sources.yaml")

        # Copy prompts directory
        prompts_src = Path(config_dir) / "prompts_literature"
        prompts_dst = input_dir / "prompts"
        if not prompts_src.exists():
            prompts_src = Path(config_dir) / "prompts"

        if prompts_src.exists():
            if prompts_dst.exists():
                shutil.rmtree(prompts_dst)
            shutil.copytree(prompts_src, prompts_dst)

        # Create .env file with literature pipeline parameters
        env_file = input_dir / ".env"
        with open(env_file, "w") as f:
            f.write(f"""# LLM Configuration for Literature-Grounded Review
# ======================================
# Run Directory: {run_dir_name}
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

        # Copy papers to this run directory
        papers_to_copy = paper_groups[i]
        for paper_path in papers_to_copy:
            paper_name = os.path.basename(paper_path)
            dest_path = papers_dir / paper_name
            shutil.copy2(paper_path, dest_path)

        # Copy literature papers (baseline reference) - same for all runs
        for lit_path in all_literature:
            lit_name = os.path.basename(lit_path)
            dest_path = literature_dir / lit_name
            shutil.copy2(lit_path, dest_path)

        lit_info = f" with {len(all_literature)} literature papers" if all_literature else ""
        created_dirs.append(str(run_dir_path))
        print(f"Created {run_dir_name} with {len(papers_to_copy)} papers{lit_info}")

    return created_dirs


def distribute_papers_evenly(papers: List[str], num_runs: int) -> List[List[str]]:
    """Distribute papers as evenly as possible among run directories."""
    paper_groups = [[] for _ in range(num_runs)]
    base_papers_per_dir = len(papers) // num_runs
    remainder = len(papers) % num_runs

    paper_idx = 0
    for i in range(num_runs):
        papers_in_this_dir = base_papers_per_dir + (1 if i < remainder else 0)
        for j in range(papers_in_this_dir):
            if paper_idx < len(papers):
                paper_groups[i].append(papers[paper_idx])
                paper_idx += 1

    return paper_groups


def distribute_papers(
    papers: List[str],
    num_runs: int,
    papers_per_run: int,
    distribution: str
) -> List[List[str]]:
    """Distribute papers among run directories according to the specified method."""
    paper_groups = [[] for _ in range(num_runs)]

    if distribution == "sequential":
        for i in range(num_runs):
            start_idx = i * papers_per_run
            end_idx = min(start_idx + papers_per_run, len(papers))
            paper_groups[i] = papers[start_idx:end_idx]

    elif distribution == "random":
        random.shuffle(papers)
        for i in range(num_runs):
            start_idx = i * papers_per_run
            end_idx = min(start_idx + papers_per_run, len(papers))
            paper_groups[i] = papers[start_idx:end_idx]

    elif distribution == "round_robin":
        for i, paper in enumerate(papers):
            run_idx = i % num_runs
            if len(paper_groups[run_idx]) < papers_per_run:
                paper_groups[run_idx].append(paper)

    else:
        raise ValueError(f"Unknown distribution method: {distribution}")

    return paper_groups


def create_literature_batch_script(
    run_dirs: List[str],
    script_name: str = "run_batch_literature_review.py",
    custom_params: Dict[str, str] = None
):
    """Create a batch script to run all literature review directories."""

    with open(script_name, "w") as f:
        f.write("""#!/usr/bin/env python3
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging

# Configure logging to suppress LiteLLM messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

def stream_output(process, prefix=""):
    \"\"\"Stream output from a subprocess in real-time.\"\"\"
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"{prefix}{line.rstrip()}")
    process.stdout.close()
    return_code = process.wait()
    return return_code

def run_single_literature_review(run_dir):
    \"\"\"Run the literature review system for a single directory.\"\"\"
    cmd = ["python", "run_review_literature.py", "--run-dir", run_dir]

    print(f"\\n{'='*60}")
    print(f"🚀 Processing Literature Review: {run_dir}")
    print(f"{'='*60}")
    print(f"🔧 Executing: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        return_code = stream_output(process, f"[{run_dir}] ")

        if return_code == 0:
            print(f"\\n✅ Successfully completed {run_dir}")
            return True
        else:
            print(f"\\n❌ Error running {run_dir} (return code: {return_code})")
            return False

    except Exception as e:
        print(f"\\n❌ Exception processing {run_dir}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run batch literature review process")
    parser.add_argument("--parallel", action="store_true", help="Run directories in parallel")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    args = parser.parse_args()

    # List of run directories
    run_dirs = [
""")

        for run_dir in run_dirs:
            f.write(f'        "{run_dir}",\n')

        f.write("""    ]

    if not run_dirs:
        print("❌ No run directories found")
        sys.exit(1)

    print("=" * 80)
    print("🚀 Starting Batch Literature Review Process")
    print(f"📁 Processing {len(run_dirs)} directories")
    print("📚 Literature-Grounded Pipeline: Librarian → Reader → Fact-Checker → Critic")
    if args.parallel:
        print(f"⚡ Running in parallel with {args.max_workers} workers")
    else:
        print(f"🔄 Running sequentially")
    print("=" * 80)

    successful_runs = 0
    failed_runs = 0

    if args.parallel:
        print("\\n⚡ Starting parallel processing...")

        def process_with_prefix(run_dir):
            return run_single_literature_review(run_dir), run_dir

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_dir = {
                executor.submit(process_with_prefix, run_dir): run_dir
                for run_dir in run_dirs
            }

            completed = 0
            for future in as_completed(future_to_dir):
                run_dir = future_to_dir[future]
                completed += 1

                try:
                    success, _ = future.result()
                    if success:
                        successful_runs += 1
                    else:
                        failed_runs += 1
                except Exception as e:
                    print(f"\\n❌ Exception processing {run_dir}: {e}")
                    failed_runs += 1

                print(f"\\n📊 Batch Progress: {completed}/{len(run_dirs)} directories completed")

    else:
        for i, run_dir in enumerate(run_dirs, 1):
            print(f"\\n📊 Batch Progress: {i}/{len(run_dirs)} directories")

            success = run_single_literature_review(run_dir)
            if success:
                successful_runs += 1
            else:
                failed_runs += 1

    # Final summary
    print("\\n" + "=" * 80)
    print("🎉 Batch Literature Review Complete!")
    print("=" * 80)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📁 Total directories: {len(run_dirs)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
""")

    os.chmod(script_name, 0o755)
    print(f"Created batch script: {script_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Set up multiple literature-grounded review run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup with literature directory
  python setup_batch_literature_review.py \\
      --master-papers-dir papers_to_review \\
      --master-literature-dir baseline_literature \\
      --base-run-dir literature_run \\
      --num-runs 3

  # Setup with specific papers per run
  python setup_batch_literature_review.py \\
      --master-papers-dir papers_to_review \\
      --master-literature-dir baseline_literature \\
      --base-run-dir literature_run \\
      --num-runs 5 \\
      --papers-per-run 10 \\
      --create-batch-script

The literature review pipeline uses:
  - literature_sources.yaml (not criteria.yaml)
  - 4-stage pipeline: Librarian → Reader → Fact-Checker → Critic
  - Outputs GroundedReview with research trajectory analysis
        """
    )
    parser.add_argument("--master-papers-dir", required=True,
                        help="Directory containing papers to review")
    parser.add_argument("--master-literature-dir", required=True,
                        help="Directory containing baseline literature papers")
    parser.add_argument("--base-run-dir", required=True,
                        help="Base name for run directories (e.g., 'literature_run')")
    parser.add_argument("--num-runs", type=int, required=True,
                        help="Number of run directories to create")
    parser.add_argument("--papers-per-run", type=int,
                        help="Number of papers per run directory (None for even distribution)")
    parser.add_argument("--config-dir", default="config",
                        help="Source config directory")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle papers before distribution")
    parser.add_argument("--distribution", choices=["sequential", "random", "round_robin"],
                        default="sequential", help="How to distribute papers")
    parser.add_argument("--no-even-distribution", action="store_true",
                        help="Don't distribute papers evenly when insufficient papers")
    parser.add_argument("--create-batch-script", action="store_true",
                        help="Create a batch script to run all directories")
    parser.add_argument("--batch-script-name", default="run_batch_literature_review.py",
                        help="Name of the batch script to create")

    args = parser.parse_args()

    # Create run directories
    created_dirs = setup_batch_literature_runs(
        master_papers_dir=args.master_papers_dir,
        master_literature_dir=args.master_literature_dir,
        base_run_dir=args.base_run_dir,
        num_runs=args.num_runs,
        papers_per_run=args.papers_per_run,
        config_dir=args.config_dir,
        shuffle=not args.no_shuffle,
        distribution=args.distribution,
        even_distribution=not args.no_even_distribution
    )

    print(f"\nSuccessfully created {len(created_dirs)} literature review directories:")
    for run_dir in created_dirs:
        print(f"  - {run_dir}")

    # Create batch script if requested
    if args.create_batch_script:
        create_literature_batch_script(created_dirs, args.batch_script_name)
        print(f"\nCreated batch script: {args.batch_script_name}")
        print(f"Run it with: python {args.batch_script_name}")
        print(f"  Or with parallel processing: python {args.batch_script_name} --parallel")


if __name__ == "__main__":
    main()
