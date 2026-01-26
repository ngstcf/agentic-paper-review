#!/usr/bin/env python3
"""
setup_batch_runs_literature.py

Set up multiple run directories with literature grounding support.
Papers are distributed from a master directory across multiple runs.
"""

import os
import shutil
import argparse
import random
import math
from pathlib import Path
from typing import List, Dict


def setup_batch_runs_literature(
    master_papers_dir: str,
    base_run_dir: str,
    num_runs: int,
    papers_per_run: int = None,
    config_dir: str = "config",
    shuffle: bool = True,
    distribution: str = "sequential",
    even_distribution: bool = True,
    semantic_scholar_api_key: str = None,
    literature_enabled: bool = True
) -> List[str]:
    """
    Set up multiple run directories with literature grounding support.

    Args:
        master_papers_dir: Directory containing all papers
        base_run_dir: Base name for run directories (e.g., "run_dir")
        num_runs: Number of run directories to create
        papers_per_run: Number of papers per run directory (None for even distribution)
        config_dir: Source config directory
        shuffle: Whether to shuffle papers before distribution
        distribution: How to distribute papers ("sequential", "random", "round_robin")
        even_distribution: Whether to distribute papers evenly when insufficient papers
        semantic_scholar_api_key: Semantic Scholar API key
        literature_enabled: Whether literature grounding is enabled by default

    Returns:
        List of created run directory paths
    """
    # Validate inputs
    if not os.path.exists(master_papers_dir):
        raise ValueError(f"Master papers directory does not exist: {master_papers_dir}")

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
    print(f"Found {total_papers} papers in {master_papers_dir}")

    # Handle papers_per_run calculation
    if papers_per_run is None:
        # Even distribution mode
        papers_per_run = math.ceil(total_papers / num_runs)
        max_papers_needed = total_papers
        print(f"Even distribution: Will distribute {total_papers} papers across {num_runs} directories")
    else:
        max_papers_needed = num_runs * papers_per_run

        if total_papers < max_papers_needed:
            if even_distribution:
                # Adjust to use all papers evenly
                papers_per_run = math.ceil(total_papers / num_runs)
                max_papers_needed = total_papers
                print(f"Even distribution: Adjusted to use all {total_papers} papers across {num_runs} directories")
            else:
                print(f"Warning: Only {total_papers} papers available, but {max_papers_needed} needed")

    # Shuffle papers if requested
    if shuffle:
        random.shuffle(all_papers)
        print("Papers shuffled for distribution")

    # Distribute papers according to the specified method
    if even_distribution and total_papers < num_runs * papers_per_run:
        paper_groups = distribute_papers_evenly(all_papers, num_runs)
    else:
        paper_groups = distribute_papers(all_papers[:max_papers_needed], num_runs, papers_per_run, distribution)

    # Create run directories
    created_dirs = []

    for i in range(num_runs):
        run_dir_name = f"{base_run_dir}{i+1}"
        run_dir_path = Path(run_dir_name)

        # Create directory structure
        papers_dir = run_dir_path / "papers"
        input_dir = run_dir_path / "input"
        outputs_dir = run_dir_path / "outputs"
        reports_dir = outputs_dir / "reports"
        reviews_dir = outputs_dir / "reviews"

        for directory in [papers_dir, input_dir, outputs_dir, reports_dir, reviews_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Copy config files
        print(f"Copying config files to {input_dir}")
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
        else:
            print(f"Warning: {literature_src} not found")

        # Create .env file with LLM and literature parameters
        env_file = input_dir / ".env"
        with open(env_file, "w") as f:
            f.write(f"""# LLM Configuration Parameters for {run_dir_name}
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
LITERATURE_GROUNDING_ENABLED={"true" if literature_enabled else "false"}
""")
            if semantic_scholar_api_key:
                f.write(f"SEMANTIC_SCHOLAR_API_KEY={semantic_scholar_api_key}\n")

        # Copy papers to this run directory
        papers_to_copy = paper_groups[i]
        for paper_path in papers_to_copy:
            paper_name = os.path.basename(paper_path)
            dest_path = papers_dir / paper_name
            shutil.copy2(paper_path, dest_path)

        created_dirs.append(str(run_dir_path))
        print(f"Created {run_dir_name} with {len(papers_to_copy)} papers")

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
    script_name: str = "run_batch_literature.py",
    enable_literature: bool = True
):
    """Create a batch script to run all directories with literature grounding."""

    with open(script_name, "w") as f:
        f.write("""#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def stream_output(process, prefix=""):
    \"""Stream output from a subprocess in real-time.\"""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"{prefix}{line.rstrip()}")
    process.stdout.close()
    return_code = process.wait()
    return return_code


def run_single_directory(run_dir, enable_literature=True):
    \"""Run the literature-grounded review system for a single directory.\"""
    cmd = ["python", "run_review_literature.py", run_dir]
    if not enable_literature:
        cmd.append("--no-literature")

    print(f"\\n{'='*60}")
    print(f"Processing Directory: {run_dir}")
    if enable_literature:
        print(f"Mode: LITERATURE-GROUNDED")
    else:
        print(f"Mode: STANDARD")
    print(f"{'='*60}")

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
            print(f"\\nSuccessfully completed {run_dir}")
            return True
        else:
            print(f"\\nError running {run_dir} (return code: {return_code})")
            return False

    except Exception as e:
        print(f"\\nException processing {run_dir}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run literature-grounded batch review process")
    parser.add_argument("--parallel", action="store_true", help="Run directories in parallel")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--no-literature", action="store_true", help="Disable literature grounding")
    args = parser.parse_args()

    # List of run directories
    run_dirs = [
""")

        for run_dir in run_dirs:
            f.write(f'        "{run_dir}",\n')

        f.write("""    ]

    if not run_dirs:
        print("No run directories found")
        sys.exit(1)

    enable_literature = not args.no_literature

    print("=" * 80)
    if enable_literature:
        print(f"Starting LITERATURE-GROUNDED Batch Review Process")
    else:
        print(f"Starting STANDARD Batch Review Process")
    print(f"Processing {len(run_dirs)} directories")
    if args.parallel:
        print(f"Running in parallel with {args.max_workers} workers")
    else:
        print(f"Running sequentially")
    print("=" * 80)

    successful_runs = 0
    failed_runs = 0

    if args.parallel:
        print("\\nStarting parallel processing...")

        def process_with_prefix(run_dir):
            return run_single_directory(run_dir, enable_literature), run_dir

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
                    print(f"\\nException processing {run_dir}: {e}")
                    failed_runs += 1

                print(f"\\nBatch Progress: {completed}/{len(run_dirs)} directories completed")

    else:
        # Run sequentially
        for i, run_dir in enumerate(run_dirs, 1):
            print(f"\\nBatch Progress: {i}/{len(run_dirs)} directories")

            success = run_single_directory(run_dir, enable_literature)
            if success:
                successful_runs += 1
            else:
                failed_runs += 1

    # Final summary
    print("\\n" + "=" * 80)
    print("Batch Process Complete!")
    print("=" * 80)
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total directories: {len(run_dirs)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
""")

    os.chmod(script_name, 0o755)
    print(f"Created batch script: {script_name}")
    print(f"Run it with: python {script_name}")
    if not enable_literature:
        print(f"Add --no-literature flag to run in standard mode")


def main():
    parser = argparse.ArgumentParser(
        description="Set up multiple literature-grounded run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup with 5 runs
  python setup_batch_runs_literature.py --master-papers-dir papers_master --base-run-dir run_dir --num-runs 5

  # With Semantic Scholar API key
  python setup_batch_runs_literature.py --master-papers-dir papers_master --base-run-dir run_dir --num-runs 5 --semantic-scholar-key YOUR_KEY

  # With literature disabled by default
  python setup_batch_runs_literature.py --master-papers-dir papers_master --base-run-dir run_dir --num-runs 5 --no-literature

  # Round-robin distribution
  python setup_batch_runs_literature.py --master-papers-dir papers_master --base-run-dir run_dir --num-runs 5 --distribution round-robin

  # Create batch script automatically
  python setup_batch_runs_literature.py --master-papers-dir papers_master --base-run-dir run_dir --num-runs 5 --create-batch-script
        """
    )

    # Required arguments
    parser.add_argument("--master-papers-dir", required=True, help="Directory containing all papers")
    parser.add_argument("--base-run-dir", required=True, help="Base name for run directories (e.g., 'run_dir')")
    parser.add_argument("--num-runs", type=int, required=True, help="Number of run directories to create")

    # Optional arguments
    parser.add_argument("--papers-per-run", type=int, help="Number of papers per run directory (None for even distribution)")
    parser.add_argument("--config-dir", default="config", help="Source config directory")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle papers before distribution")
    parser.add_argument("--distribution", choices=["sequential", "random", "round_robin"], default="sequential",
                        help="How to distribute papers")
    parser.add_argument("--no-even-distribution", action="store_true",
                        help="Don't distribute papers evenly when insufficient papers")

    # Literature-specific options
    parser.add_argument("--semantic-scholar-key", help="Semantic Scholar API key (optional)")
    parser.add_argument("--no-literature", action="store_true",
                        help="Disable literature grounding by default in created runs")

    # Batch script options
    parser.add_argument("--create-batch-script", action="store_true",
                        help="Create a batch script to run all directories")
    parser.add_argument("--batch-script-name", default="run_batch_literature.py",
                        help="Name of the batch script to create")

    args = parser.parse_args()

    # Create run directories
    created_dirs = setup_batch_runs_literature(
        master_papers_dir=args.master_papers_dir,
        base_run_dir=args.base_run_dir,
        num_runs=args.num_runs,
        papers_per_run=args.papers_per_run,
        config_dir=args.config_dir,
        shuffle=not args.no_shuffle,
        distribution=args.distribution,
        even_distribution=not args.no_even_distribution,
        semantic_scholar_api_key=args.semantic_scholar_key,
        literature_enabled=not args.no_literature
    )

    print(f"\nSuccessfully created {len(created_dirs)} literature-grounded run directories:")
    for run_dir in created_dirs:
        print(f"  - {run_dir}")

    print(f"\nTo run reviews:")
    print(f"  python run_review_literature.py <run_dir>")

    # Create batch script if requested
    if args.create_batch_script:
        create_literature_batch_script(
            created_dirs,
            args.batch_script_name,
            enable_literature=not args.no_literature
        )


if __name__ == "__main__":
    main()
