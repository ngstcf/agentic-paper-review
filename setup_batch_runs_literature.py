# setup_batch_runs_literature.py
import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
import math
from core.config_loader import MODEL_DEFAULTS

def setup_batch_runs(
    master_papers_dir: str, 
    base_run_dir: str, 
    num_runs: int, 
    papers_per_run: int = None,
    config_dir: str = "config",
    shuffle: bool = True,
    distribution: str = "sequential",
    even_distribution: bool = True
) -> List[str]:
    """
    Set up multiple run directories with papers distributed from a master directory.
    
    Args:
        master_papers_dir: Directory containing all papers
        base_run_dir: Base name for run directories (e.g., "run_dir")
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
                print(f"Some directories will have {papers_per_run} papers, others will have {papers_per_run - 1}")
            else:
                print(f"Warning: Only {total_papers} papers available, but {max_papers_needed} needed ({papers_per_run} × {num_runs}).")
                print(f"Will use only {max_papers_needed} papers, leaving {total_papers - max_papers_needed} unused.")
    
    # Shuffle papers if requested
    if shuffle:
        random.shuffle(all_papers)
    
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
        
        # Create a .env file with LLM PARAMETERS ONLY (no API keys)
        env_file = input_dir / ".env"
        with open(env_file, "w") as f:
            f.write(f"""# LLM Configuration Parameters for {run_dir_name}
# ======================================
# API keys are loaded from the global .env file at the project root
# ======================================

# Extraction Configuration
PROVIDER_EXTRACTION={MODEL_DEFAULTS["extractor_provider"]}
EXTRACTOR_MODEL={MODEL_DEFAULTS["extractor_model"]}

# Synthesis Configuration
PROVIDER_SYNTHESIS={MODEL_DEFAULTS["synthesizer_provider"]}
SYNTHESIZER_MODEL={MODEL_DEFAULTS["synthesizer_model"]}

# General Parameters
TEMPERATURE=0.2
MAX_RETRIES=3
MAX_PARALLEL_EXTRACTIONS=5

# Judge Configuration
JUDGE_PROVIDER={MODEL_DEFAULTS["judge_provider"]}
JUDGE_MODEL={MODEL_DEFAULTS["judge_model"]}
JUDGE_TEMPERATURE=0.1
""")
        
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
    """
    Distribute papers as evenly as possible among run directories.
    
    Args:
        papers: List of paper file paths
        num_runs: Number of run directories
    
    Returns:
        List of paper groups, one for each run directory
    """
    paper_groups = [[] for _ in range(num_runs)]
    
    # Calculate base papers per directory and remainder
    base_papers_per_dir = len(papers) // num_runs
    remainder = len(papers) % num_runs
    
    # Distribute papers
    paper_idx = 0
    for i in range(num_runs):
        # First 'remainder' directories get one extra paper
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
    """
    Distribute papers among run directories according to the specified method.
    
    Args:
        papers: List of paper file paths
        num_runs: Number of run directories
        papers_per_run: Number of papers per run directory
        distribution: Distribution method ("sequential", "random", "round_robin")
    
    Returns:
        List of paper groups, one for each run directory
    """
    paper_groups = [[] for _ in range(num_runs)]
    
    if distribution == "sequential":
        # Distribute papers in sequential chunks
        for i in range(num_runs):
            start_idx = i * papers_per_run
            end_idx = min(start_idx + papers_per_run, len(papers))
            paper_groups[i] = papers[start_idx:end_idx]
    
    elif distribution == "random":
        # Randomly assign papers to each run
        random.shuffle(papers)
        for i in range(num_runs):
            start_idx = i * papers_per_run
            end_idx = min(start_idx + papers_per_run, len(papers))
            paper_groups[i] = papers[start_idx:end_idx]
    
    elif distribution == "round_robin":
        # Distribute papers in round-robin fashion
        for i, paper in enumerate(papers):
            run_idx = i % num_runs
            if len(paper_groups[run_idx]) < papers_per_run:
                paper_groups[run_idx].append(paper)
    
    else:
        raise ValueError(f"Unknown distribution method: {distribution}")
    
    return paper_groups

def create_batch_script(
    run_dirs: List[str], 
    script_name: str = "run_batch_literature.py",
    custom_params: Dict[str, str] = None
):
    """Create a batch script to run all directories with custom parameters."""
    
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

def run_single_directory(run_dir):
    \"\"\"Run the review system for a single directory.\"\"\"
    cmd = ["python", "run_review_with_dir_literature.py", "--run-dir", run_dir, "--literature-grounding"]
    
    print(f"\\n{'='*60}")
    print(f"🚀 Processing Directory: {run_dir}")
    print(f"{'='*60}")
    print(f"🔧 Executing: {' '.join(cmd)}")
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream the output
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
    parser = argparse.ArgumentParser(description="Run batch review process")
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
    print(f"🚀 Starting Batch Review Process")
    print(f"📁 Processing {len(run_dirs)} directories")
    if args.parallel:
        print(f"⚡ Running in parallel with {args.max_workers} workers")
    else:
        print(f"🔄 Running sequentially")
    print("=" * 80)
    
    successful_runs = 0
    failed_runs = 0
    
    if args.parallel:
        # Run in parallel
        print("\\n⚡ Starting parallel processing...")
        
        def process_with_prefix(run_dir):
            return run_single_directory(run_dir), run_dir
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all jobs
            future_to_dir = {
                executor.submit(process_with_prefix, run_dir): run_dir 
                for run_dir in run_dirs
            }
            
            # Process as they complete
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
        # Run sequentially
        for i, run_dir in enumerate(run_dirs, 1):
            print(f"\\n📊 Batch Progress: {i}/{len(run_dirs)} directories")
            
            success = run_single_directory(run_dir)
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
    
    # Final summary
    print("\\n" + "=" * 80)
    print("🎉 Batch Process Complete!")
    print("=" * 80)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📁 Total directories: {len(run_dirs)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
""")
    
    os.chmod(script_name, 0o755)  # Make executable
    print(f"Created batch script: {script_name}")

def main():
    parser = argparse.ArgumentParser(description="Set up batch run directories for literature-grounded reviews")
    parser.add_argument("--master-papers-dir", required=True, help="Directory containing all papers")
    parser.add_argument("--base-run-dir", required=True, help="Base name for run directories (e.g., 'run_dir')")
    parser.add_argument("--num-runs", type=int, required=True, help="Number of run directories to create")
    parser.add_argument("--papers-per-run", type=int, help="Number of papers per run directory (None for even distribution)")
    parser.add_argument("--config-dir", default="config", help="Source config directory")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle papers before distribution")
    parser.add_argument("--distribution", choices=["sequential", "random", "round_robin"], 
                        default="sequential", help="How to distribute papers")
    parser.add_argument("--no-even-distribution", action="store_true", 
                        help="Don't distribute papers evenly when insufficient papers")
    parser.add_argument("--create-batch-script", action="store_true", 
                        help="Create a batch script to run all directories")
    parser.add_argument("--batch-script-name", default="run_batch_literature.py",
                        help="Name of the batch script to create")
    
    # Custom parameters for batch script
    parser.add_argument("--provider-extraction", help="Extraction provider for all runs")
    parser.add_argument("--extractor-model", help="Extraction model for all runs")
    parser.add_argument("--provider-synthesis", help="Synthesis provider for all runs")
    parser.add_argument("--synthesizer-model", help="Synthesis model for all runs")
    parser.add_argument("--temperature", type=float, help="Temperature for all runs")
    parser.add_argument("--max-retries", type=int, help="Max retries for all runs")
    parser.add_argument("--max-parallel", type=int, help="Max parallel extractions for all runs")
    parser.add_argument("--judge-provider", help="Judge provider for all runs")
    parser.add_argument("--judge-model", help="Judge model for all runs")
    parser.add_argument("--judge-temperature", type=float, help="Judge temperature for all runs")
    
    args = parser.parse_args()
    
    # Collect custom parameters
    custom_params = {}
    if args.provider_extraction:
        custom_params["provider_extraction"] = args.provider_extraction
    if args.extractor_model:
        custom_params["extractor_model"] = args.extractor_model
    if args.provider_synthesis:
        custom_params["provider_synthesis"] = args.provider_synthesis
    if args.synthesizer_model:
        custom_params["synthesizer_model"] = args.synthesizer_model
    if args.temperature is not None:
        custom_params["temperature"] = str(args.temperature)
    if args.max_retries is not None:
        custom_params["max_retries"] = str(args.max_retries)
    if args.max_parallel is not None:
        custom_params["max_parallel"] = str(args.max_parallel)
    if args.judge_provider:
        custom_params["judge_provider"] = args.judge_provider
    if args.judge_model:
        custom_params["judge_model"] = args.judge_model
    if args.judge_temperature is not None:
        custom_params["judge_temperature"] = str(args.judge_temperature)
    
    # Create run directories
    created_dirs = setup_batch_runs(
        master_papers_dir=args.master_papers_dir,
        base_run_dir=args.base_run_dir,
        num_runs=args.num_runs,
        papers_per_run=args.papers_per_run,
        config_dir=args.config_dir,
        shuffle=not args.no_shuffle,
        distribution=args.distribution,
        even_distribution=not args.no_even_distribution
    )
    
    print(f"\nSuccessfully created {len(created_dirs)} run directories:")
    for run_dir in created_dirs:
        print(f"  - {run_dir}")
    
    # Create batch script if requested
    if args.create_batch_script:
        create_batch_script(created_dirs, args.batch_script_name, custom_params)
        print(f"\nCreated batch script: {args.batch_script_name}")
        print(f"Run it with: python {args.batch_script_name}")
        if custom_params:
            print("Custom parameters will be applied to all runs.")

if __name__ == "__main__":
    main()