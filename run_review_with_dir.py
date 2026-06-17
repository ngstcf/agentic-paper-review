# run_review_with_dir.py (cleaned up and typos fixed)
import time
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import logging

# Configure logging to suppress LiteLLM messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from core.config_loader import Config, MODEL_DEFAULTS
from core.paper_ingestor import (
    load_ingestion_cache, 
    save_ingestion_cache, 
    ingest_directory
)
from agents.agent_extractor import process_paper_extractions
from agents.agent_synthesizer import synthesize_review
from utilities.output_generator import save_review_markdown, save_consolidated_csv
from utilities.helpers import setup_logging, get_config_hash
from core.data_models import Review
from core.progress import (
    get_emitter, configure_emitter, SSEBackend,
    RunStarted, StageStarted, StageCompleted,
    PaperCompleted, RunProgress, CostUpdate, RunCompleted
)

def print_progress_bar(current, total, prefix="", suffix="", length=50):
    """Print a progress bar to the console."""
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()  # New line when complete
        sys.stdout.flush()  # Ensure it's displayed immediately

def load_progress(progress_file: str, current_config_hash: str) -> Dict[str, Dict]:
    """Load the progress file and check if configuration matches."""
    if not os.path.exists(progress_file):
        return {"papers": {}}
        
    try:
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
        
        # Check if configuration has changed
        stored_config_hash = progress_data.get("config_hash")
        if stored_config_hash != current_config_hash:
            print(f"\n[Progress] Configuration has changed (stored: {stored_config_hash[:8]}..., current: {current_config_hash[:8]}...)")
            print("[Progress] Resetting progress - all papers will be reprocessed with new configuration")
            return {"papers": {}}
        
        print(f"\n[Progress] Configuration matches (hash: {current_config_hash[:8]}...)")
        papers_count = len(progress_data.get('papers', {}))
        print(f"[Progress] Loaded {papers_count} papers from previous run")
        return progress_data
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\n[Progress] Error loading progress file: {e}")
        print("[Progress] Starting fresh")
        return {"papers": {}}

def save_progress(progress_file: str, progress: Dict[str, Dict], config_hash: str):
    """Save the progress file with configuration hash."""
    import json
    from datetime import datetime
    
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    progress_data = {
        "config_hash": config_hash,
        "papers": progress,
        "last_updated": datetime.now().isoformat()
    }
    
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2, default=datetime_handler)

def setup_run_directory(run_dir: str) -> Dict[str, str]:
    """Create and set up the run directory structure."""
    run_path = Path(run_dir)
    
    # Create directory structure
    papers_dir = run_path / "papers"
    input_dir = run_path / "input"
    outputs_dir = run_path / "outputs"
    reports_dir = outputs_dir / "reports"
    reviews_dir = outputs_dir / "reviews"
    
    for directory in [papers_dir, input_dir, outputs_dir, reports_dir, reviews_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create a default config if it doesn't exist
    criteria_file = input_dir / "criteria.yaml"
    if not criteria_file.exists():
        print(f"[Setup] Creating default config at {criteria_file}")
        # Copy default config from the original location
        import shutil
        shutil.copy("config/criteria.yaml", criteria_file)
        
        # Copy prompts directory
        prompts_src = Path("config/prompts")
        prompts_dst = input_dir / "prompts"
        if prompts_src.exists():
            if prompts_dst.exists():
                shutil.rmtree(prompts_dst)
            shutil.copytree(prompts_src, prompts_dst)
        
        # Create a .env file with default values
        env_file = input_dir / ".env"
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write(f"""# LLM Configuration
PROVIDER_EXTRACTION={MODEL_DEFAULTS["extractor_provider"]}
EXTRACTOR_MODEL={MODEL_DEFAULTS["extractor_model"]}
PROVIDER_SYNTHESIS={MODEL_DEFAULTS["synthesizer_provider"]}
SYNTHESIZER_MODEL={MODEL_DEFAULTS["synthesizer_model"]}
TEMPERATURE=0.2
MAX_RETRIES=3
MAX_PARALLEL_EXTRACTIONS=5

# Judge Configuration
JUDGE_PROVIDER={MODEL_DEFAULTS["judge_provider"]}
JUDGE_MODEL={MODEL_DEFAULTS["judge_model"]}
JUDGE_TEMPERATURE=0.1
""")
    
    return {
        "run_dir": str(run_path),
        "papers_dir": str(papers_dir),
        "input_dir": str(input_dir),
        "outputs_dir": str(outputs_dir),
        "reports_dir": str(reports_dir),
        "reviews_dir": str(reviews_dir),
        "cache_file": str(run_path / "ingestion_cache.json"),
        "progress_file": str(run_path / "progress.json")
    }

def main():
    parser = argparse.ArgumentParser(description="Run the academic review system with a specific directory")
    parser.add_argument("--run-dir", required=True, help="Directory for this run")
    parser.add_argument("--override-config", help="Path to a config file to override the default") # Typo 1 fixed
    parser.add_argument("--force-reload", action="store_true", help="Force reload of configuration")
    parser.add_argument("--web", action="store_true", help="Enable web dashboard SSE backend")
    args = parser.parse_args()
    
    start_time = time.time()
    setup_logging()

    # Configure progress emitter with optional SSE backend
    if args.web:
        emitter = configure_emitter(use_sse=True)
    else:
        emitter = get_emitter()
    
    print("=" * 80, flush=True)  
    print("🚀 Starting Academic Review System with Run Directory", flush=True)  
    print("=" * 80, flush=True)  
    
    # 1. Set up the run directory
    dirs = setup_run_directory(args.run_dir)
    print(f"\n[Setup] Using run directory: {dirs['run_dir']}", flush=True)
    
    # 2. Load Configuration
    print("\n[Config] Loading configuration...", flush=True)
    
    # Force reload environment variables if requested
    if args.force_reload: # Typo 2 fixed (was force-reload)
        import importlib
        import sys
        # Clear any cached environment variables
        if 'dotenv' in sys.modules:
            importlib.reload(sys.modules['dotenv'])
        
        # Force reload of the config
        config = Config(config_path=dirs["input_dir"])
    else:
        config = Config(config_path=dirs["input_dir"])
    
    # Override config if specified
    if args.override_config:
        print(f"[Config] Overriding config with {args.override_config}")
        override_config = Config(config_path=args.override_config)
        # Merge the configs
        config.env.update(override_config.env)

    # --- DEBUG PRINTS ---
   # print(f"\n[Config] DEBUG: Environment variables after loading:")
    #print(f"   PROVIDER_EXTRACTION: {os.environ.get('PROVIDER_EXTRACTION', 'not set')}")
    #print(f"   EXTRACTOR_MODEL: {os.environ.get('EXTRACTOR_MODEL', 'not set')}")
    #print(f"   PROVIDER_SYNTHESIS: {os.environ.get('PROVIDER_SYNTHESIS', 'not set')}")
    #print(f"   SYNTHESIZER_MODEL: {os.environ.get('SYNTHESIZER_MODEL', 'not set')}") 

    #llm_config_debug = config.get_llm_config()
    #print(f"\n[Config] DEBUG: get_llm_config() returns:")
    #print(f"   extractor_provider: {llm_config_debug['extractor_provider']}")
    #print(f"   extractor_model: {llm_config_debug['extractor_model']}")
    #print(f"   synthesizer_provider: {llm_config_debug['synthesizer_provider']}")
    #print(f"   synthesizer_model: {llm_config_debug['synthesizer_model']}") 
    # --- END OF DEBUG PRINTS ---

    # 3. Generate configuration hash
    config_hash = get_config_hash(config)
    print(f"[Config] Configuration hash: {config_hash[:8]}...", flush=True)
    
    # Display LLM configuration
    llm_config = config.get_llm_config() 
    print(f"[Config] Extraction: {llm_config['extractor_provider']}/{llm_config['extractor_model']}", flush=True)
    print(f"[Config] Synthesis: {llm_config['synthesizer_provider']}/{llm_config['synthesizer_model']}", flush=True)
    print(f"[Config] Temperature: {llm_config['temperature']}",flush=True)
    
    # 4. Load ingestion cache from the run directory
    print(f"\n[Cache] Loading ingestion cache...", flush=True)
    ingestion_cache = load_ingestion_cache(dirs["cache_file"])
    
    # 5. Ingest papers
    print(f"\n[Ingest] Ingesting papers from '{dirs['papers_dir']}' directory...", flush=True)
    ingest_start = time.time()
    papers, cache_was_updated = ingest_directory(dirs["papers_dir"], ingestion_cache, dirs["cache_file"])
    ingest_time = time.time() - ingest_start

    if cache_was_updated:
        save_ingestion_cache(ingestion_cache, dirs["cache_file"])

    if not papers:
        print("\n❌ [Main] No papers found or ingested. Exiting.", flush=True)
        return

    print(f"\n[Ingest] ✅ Ingestion completed in {ingest_time:.1f} seconds", flush=True)
    print(f"[Ingest] 📄 Loaded {len(papers)} papers for processing.", flush=True)

    # Emit RunStarted event
    emitter.emit(RunStarted(
        run_dir=dirs['run_dir'],
        mode="standard",
        paper_count=len(papers),
        config=llm_config
    ))
    
    # 6. Load progress to resume from where we left off
    progress_data = load_progress(dirs["progress_file"], config_hash)
    progress_papers = progress_data.get("papers", {})
    final_reviews = []
    
    # Initialize cost tracking variables
    total_batch_cost = 0.0
    new_processing_cost = 0.0  # Track cost of new processing only
    cached_cost = 0.0  # Track cost from cache
    
    # Calculate progress statistics
    completed_count = len(progress_papers)
    remaining_count = len(papers) - completed_count
    print(f"\n[Progress] {completed_count} papers already completed, {remaining_count} papers remaining")
    
    if remaining_count == 0:
        print("\n✅ [Progress] All papers have been processed with current configuration!")
        print(f"💰 [Cost] No new processing needed - all data loaded from cache")
    else:
        print(f"\n[Progress] Starting processing of {remaining_count} remaining papers...")
    
    # 7. Process Each Paper
    processed_count = 0
    for i, paper in enumerate(papers, 1):
        # Skip if already processed with current configuration
        if paper.filename in progress_papers:
            print(f"\n⏭️  [{i}/{len(papers)}] Skipping: {paper.filename} (already processed)", flush=True)
            # Convert the dict back to a Review object
            review_data = progress_papers[paper.filename]["review"]
            review = Review.model_validate(review_data)
            final_reviews.append(review)
            
            # Add to cached cost (not new processing cost)
            paper_cost = progress_papers[paper.filename]["cost"]
            cached_cost += paper_cost
            total_batch_cost += paper_cost
            continue
            
        print(f"\n📄 [{i}/{len(papers)}] Processing: {paper.filename}")
        paper_start_time = time.time()
        
        # 8. Agent 1: Extract Evidence (Parallel)
        print(f"   🔍 [Agent 1] Starting evidence extraction...", flush=True)
        emitter.emit(StageStarted(stage_name="extraction", paper_filename=paper.filename))
        extraction_start = time.time()
        extractions = process_paper_extractions(paper, config)
        extraction_time = time.time() - extraction_start

        if not extractions:
            print(f"   ❌ [Agent 1] Failed to get any extractions for {paper.filename}. Skipping.", flush=True)
            continue

        print(f"   ✅ [Agent 1] Completed {len(extractions)} extractions in {extraction_time:.1f}s", flush=True)
        emitter.emit(StageCompleted(stage_name="extraction", duration_s=extraction_time,
                                     result_summary=f"{len(extractions)} extractions"))

        # 9. Agent 2: Synthesize Review
        print(f"   📝 [Agent 2] Starting review synthesis...")
        emitter.emit(StageStarted(stage_name="synthesis", paper_filename=paper.filename))
        synthesis_start = time.time()
        review = synthesize_review(paper, extractions, config)
        synthesis_time = time.time() - synthesis_start
        
        if not review:
            print(f"   ❌ [Agent 2] Failed to synthesize review for {paper.filename}. Skipping.", flush=True)
            continue
        
        print(f"   ✅ [Agent 2] Completed synthesis in {synthesis_time:.1f}s", flush=True)
        emitter.emit(StageCompleted(stage_name="synthesis", duration_s=synthesis_time,
                                     result_summary=f"score={review.overall_score:.1f}"))
        
        # 10. Save Individual Output
        print(f"   💾 [Output] Saving review...", flush=True)
        save_review_markdown(review, paper, config, dirs["reviews_dir"])
        
        # 11. Update progress
        progress_papers[paper.filename] = {
            "review": review.model_dump(),
            "cost": review.total_cost,
            "timestamp": review.synthesis_timestamp.isoformat()
        }
        save_progress(dirs["progress_file"], progress_papers, config_hash)
        
        final_reviews.append(review)
        
        # Track costs separately
        paper_cost = review.total_cost
        total_batch_cost += paper_cost
        new_processing_cost += paper_cost
        
        paper_end_time = time.time()
        paper_total_time = paper_end_time - paper_start_time
        
        processed_count += 1

        # Emit progress events
        emitter.emit(PaperCompleted(
            paper_filename=paper.filename,
            score=review.overall_score,
            recommendation=review.recommendation,
            cost=paper_cost,
            duration_s=paper_total_time
        ))
        emitter.emit(CostUpdate(paper_cost=paper_cost, total_cost=total_batch_cost))
        elapsed = time.time() - start_time
        done = completed_count + processed_count
        remaining = len(papers) - done
        est_remaining = (elapsed / done * remaining) if done > 0 else 0
        emitter.emit(RunProgress(
            papers_done=done, papers_total=len(papers),
            elapsed_s=elapsed, estimated_remaining_s=est_remaining
        ))
        
        # Print summary for this paper
        print(f"   📊 [Summary] Score: {review.overall_score:.1f}/100, Recommendation: {review.recommendation}", flush=True)
        print(f"   💰 [Cost] Paper cost: ${paper_cost:.4f}, New total: ${new_processing_cost:.4f}", flush=True)
        print(f"   ⏱️  [Time] Paper time: {paper_total_time:.1f}s, Avg per paper: {new_processing_cost/max(processed_count, 1):.1f}s", flush=True)
        
        # Update overall progress
        overall_progress = completed_count + processed_count
        print_progress_bar(overall_progress, len(papers), 
                          prefix=f"[Progress] Overall", 
                          suffix=f"({overall_progress}/{len(papers)})")

    # 12. Save Consolidated Report
    if final_reviews:
        print(f"\n📊 [Report] Generating consolidated report...", flush=True)
        save_consolidated_csv(final_reviews, dirs["reports_dir"])
        print(f"   ✅ [Report] Saved to {dirs['reports_dir']}", flush=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80, flush=True)
    print("🎉 Batch Complete!", flush=True)
    print("=" * 80, flush=True)
    print(f"📊 [Summary] Total papers processed: {len(final_reviews)}", flush=True)
    print(f"💰 [Summary] Cost breakdown:", flush=True)
    print(f"   • New processing cost: ${new_processing_cost:.4f}", flush=True)
    print(f"   • Cached data cost: ${cached_cost:.4f}", flush=True)
    print(f"   • Total batch cost: ${total_batch_cost:.4f}", flush=True)
    
    if new_processing_cost == 0:
        print(f"   ✅ No API calls made - all data loaded from cache!", flush=True)
    elif cached_cost > 0:
        print(f"   📈 {completed_count} papers from cache, {processed_count} newly processed", flush=True)
    
    # Cost validation
    cost_warning_threshold = float(os.environ.get("COST_WARNING_PER_PAPER", 1.0))
    if new_processing_cost > 0 and processed_count > 0:
        avg_cost_per_paper = new_processing_cost / processed_count
        if avg_cost_per_paper > cost_warning_threshold:
            print(f"\n⚠️  [Warning] High average cost per paper: ${avg_cost_per_paper:.4f}", flush=True)
            print(f"   Consider using cheaper models for extraction to reduce costs", flush=True)
    
    print(f"⏱️  [Summary] Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)", flush=True)
    if len(papers) > 0:
        print(f"📈 [Summary] Average time per paper: {total_time/len(papers):.1f} seconds", flush=True)
    print(f"📁 [Summary] Results saved in: {dirs['outputs_dir']}", flush=True)
    print("=" * 80, flush=True)

    # Emit RunCompleted event
    emitter.emit(RunCompleted(
        total_papers=len(final_reviews),
        total_cost=total_batch_cost,
        total_time_s=total_time,
        output_dir=dirs['outputs_dir']
    ))

if __name__ == "__main__":
    main()