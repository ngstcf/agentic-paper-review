# run_review_with_dir_literature.py
# Enhanced version with optional literature grounding for standard review pipeline
import time
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging

# Configure logging to suppress LiteLLM messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from core.config_loader import Config
from core.paper_ingestor import (
    load_ingestion_cache,
    save_ingestion_cache,
    ingest_directory
)
from agents.agent_extractor import process_paper_extractions
from agents.agent_synthesizer import synthesize_review
from utilities.output_generator import save_review_markdown, save_consolidated_csv
from utilities.helpers import setup_logging, get_config_hash
from core.data_models import Review, GroundedReview, BaselineReference, FactCheckResult

# Literature grounding imports
_literature_agents_available = True
try:
    from agents.agent_librarian import create_baseline_reference
    from agents.agent_reader import process_paper_extractions as process_paper_extractions_literature
    from agents.agent_fact_checker import run_fact_checks
    from agents.agent_critic import synthesize_grounded_review
except ImportError as e:
    _literature_agents_available = False
    _literature_import_error = str(e)

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
                f.write("""# LLM Configuration
PROVIDER_EXTRACTION=openai
EXTRACTOR_MODEL=gpt-4o-mini
PROVIDER_SYNTHESIS=openai
SYNTHESIZER_MODEL=gpt-4o
TEMPERATURE=0.2
MAX_RETRIES=3
MAX_PARALLEL_EXTRACTIONS=5

# Judge Configuration
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o
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

def _calculate_novelty_adjusted_score(
    base_score: float,
    extractions: List,
    config: Config
) -> float:
    """
    Calculate novelty-adjusted score based on extraction novelty rankings.

    Args:
        base_score: The base calculated score
        extractions: List of novelty-ranked extractions (or standard extractions)
        config: System configuration

    Returns:
        Novelty-adjusted score
    """
    # Check if extractions have novelty rankings
    if not extractions or not hasattr(extractions[0], 'novelty_ranking'):
        # Standard extractions without novelty rankings - no adjustment
        return base_score

    # Calculate average novelty ranking
    novelty_scores = [e.novelty_ranking for e in extractions]
    avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 3

    # Novelty adjustment factor (1-5 scale maps to -5% to +5% adjustment)
    adjustment_factor = (avg_novelty - 3) * 0.025

    # Check for contradictions (penalty)
    has_contradictions = any(getattr(e, 'contradicts_baseline', False) for e in extractions)
    if has_contradictions:
        adjustment_factor -= 0.05  # Additional 5% penalty

    # Check for significant extensions (bonus)
    has_extensions = any(getattr(e, 'extends_baseline', False) for e in extractions)
    if has_extensions:
        adjustment_factor += 0.03  # Additional 3% bonus

    # Apply adjustment
    adjusted_score = base_score * (1 + adjustment_factor)

    # Clamp to valid range
    return max(0, min(100, adjusted_score))


def enhance_review_with_literature(
    base_review: Review,
    paper: 'Paper',
    baseline: Optional[BaselineReference],
    fact_checks: List[FactCheckResult],
    extractions: List,
    config: Config
) -> Optional[GroundedReview]:
    """
    Enhance a standard review with literature-grounded insights.

    This function takes a complete Review (with detailed criterion_narrative from the
    standard synthesizer) and adds literature context: Research Trajectory, novelty adjustment.

    Args:
        base_review: The standard Review with all details
        paper: The target paper
        baseline: Baseline reference from Librarian
        fact_checks: Fact-check results
        extractions: Novelty-ranked extractions
        config: System configuration

    Returns:
        GroundedReview with literature enhancement, or None if enhancement fails
    """
    from core.data_models import LiteratureContext

    try:
        # Generate research trajectory section from baseline and extractions
        print(f"   📚 [Librarian] Generating research trajectory section...", flush=True)

        # Check if librarian found any papers
        has_baseline_papers = baseline and len(baseline.baseline_papers) > 0

        # Build research trajectory narrative from extractions (always generated)
        research_trajectory = "### Research Trajectory and Position\n\n"

        # Add baseline context if available
        if has_baseline_papers:
            research_trajectory += f"This paper builds on and extends the existing line of research concerning {baseline.sub_topic or 'the topic'}. "
            research_trajectory += f"Based on {len(baseline.baseline_papers)} baseline papers, the target paper {'pivots decisively' if any(e.extends_baseline for e in extractions) else 'builds upon'} prior work.\n\n"

            # Add key findings from baseline
            if baseline.key_findings_summary:
                research_trajectory += f"**State of the Art:** {baseline.key_findings_summary}\n\n"
        else:
            research_trajectory += f"**⚠️ Note:** Literature grounding was enabled, but the Librarian agent was unable to retrieve baseline papers (possibly due to API rate limits or search constraints). The research trajectory below is based on novelty rankings assigned during extraction without direct comparison to prior work.\n\n"
            research_trajectory += f"This paper presents contributions in the field of {paper.metadata.title[:80] if paper.metadata.title else 'the topic'}...\n\n"

        # Add novelty context from extractions (only if literature-enhanced)
        has_novelty_rankings = extractions and hasattr(extractions[0], 'novelty_ranking')

        if has_novelty_rankings:
            high_novelty = [e for e in extractions if e.novelty_ranking >= 4]
            if high_novelty:
                topics = ", ".join([e.criterion_id.replace("_", " ") for e in high_novelty[:3]])
                research_trajectory += f"The paper's key contribution is addressing aspects of {topics}. "

            # Add novelty scores
            novelty_scores = [e.novelty_ranking for e in extractions]
            avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 3
            research_trajectory += f"\n\n**Novelty Assessment:**\n"
            research_trajectory += f"Across all evaluated criteria, the paper demonstrates {'high novelty' if avg_novelty >= 4 else 'moderate novelty' if avg_novelty >= 3 else 'limited novelty'} (average: {avg_novelty:.1f}/5). "
            research_trajectory += "Key novel aspects include:\n"
            for e in extractions:
                if e.novelty_ranking >= 4:
                    justification = e.score_justification[:100] + "..." if len(e.score_justification) > 100 else e.score_justification
                    research_trajectory += f"\n- **{e.criterion_id.replace('_', ' ').title()}**: {justification}"

            research_trajectory += f"\n\n**Verification Note:** While the review is based on comprehensive analysis of the target paper, claims about specific metrics or percentages should be verified against the full manuscript.\n"
        else:
            # No novelty rankings available (standard extraction)
            research_trajectory += f"\n\n**Novelty Assessment:**\n"
            research_trajectory += "Standard evaluation without literature-grounded novelty assessment.\n"

        print(f"   📊 [Novelty] Calculating novelty-adjusted score...", flush=True)
        novelty_adjusted_score = _calculate_novelty_adjusted_score(
            base_score=base_review.overall_score,
            extractions=extractions,
            config=config
        )

        # Add note about librarian contribution effectiveness
        novelty_adjustment = novelty_adjusted_score - base_review.overall_score
        if abs(novelty_adjustment) < 1.0 and has_baseline_papers:
            research_trajectory += f"\n\n**⚠️ Note on Librarian Contribution:** While {len(baseline.baseline_papers)} baseline papers were retrieved, the minimal novelty adjustment ({novelty_adjustment:+.1f}) suggests they may not have been directly relevant to the target paper's specific contributions. The novelty rankings above should be interpreted with this limitation in mind.\n"
        elif not has_baseline_papers:
            research_trajectory += f"\n\n**⚠️ Note on Librarian Contribution:** No baseline papers were retrieved by the Librarian agent (possibly due to API rate limits or search constraints). The novelty rankings above are based solely on the extraction agent's assessment without direct comparison to prior work.\n"

        # Build literature context
        literature_context = LiteratureContext(
            baseline_reference=baseline,
            fact_checks=fact_checks or [],
            total_papers_consulted=len(baseline.baseline_papers) if baseline else 0,
            search_queries_made=[]
        )

        # Convert Review to GroundedReview by adding literature fields
        review_data = base_review.model_dump()
        review_data['literature_context'] = literature_context
        review_data['research_trajectory_section'] = research_trajectory
        review_data['novelty_adjusted_score'] = novelty_adjusted_score
        review_data['llm_fallback_used'] = False

        enhanced_review = GroundedReview(**review_data)

        print(f"   ✅ [Literature] Base score: {base_review.overall_score:.1f}, Novelty-adjusted: {novelty_adjusted_score:.1f}", flush=True)
        return enhanced_review

    except Exception as e:
        print(f"   ❌ [Literature] Enhancement failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

        # Fall back to standard review converted to GroundedReview
        return GroundedReview(
            **base_review.model_dump(),
            literature_context=LiteratureContext(),
            research_trajectory_section="",
            novelty_adjusted_score=None,
            llm_fallback_used=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the academic review system (standard or literature-grounded)\n\n"
                    "Default: Standard review mode (no literature grounding)\n"
                    "Use --literature-grounding to enable literature enhancement (Librarian → Reader → Fact-Checker → Enhanced Synthesis)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, help="Directory for this run")
    parser.add_argument("--override-config", help="Path to a config file to override the default")
    parser.add_argument("--force-reload", action="store_true", help="Force reload of configuration")
    parser.add_argument("--literature-grounding", action="store_true", help="Enable literature grounding enhancement (Librarian → Reader → Fact-Checker → Enhanced Synthesis)")
    args = parser.parse_args()
    
    start_time = time.time()
    setup_logging()
    
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

    # Literature Grounding Mode
    use_literature_grounding = args.literature_grounding
    if use_literature_grounding:
        if not _literature_agents_available:
            print(f"\n⚠️  [Literature Grounding] ERROR: Literature agents not available!", flush=True)
            print(f"   Import error: {_literature_import_error}", flush=True)
            print(f"   Falling back to standard review mode.", flush=True)
            use_literature_grounding = False
        else:
            print(f"\n[Literature Grounding] ✅ ENABLED - Enhancement Mode", flush=True)
            print(f"[Literature Grounding] Will add literature insights to standard synthesis", flush=True)
    else:
        print(f"\n[Literature Grounding] DISABLED (standard review mode)", flush=True)
    
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

    # 5.5. Literature Grounding Enhancement (if enabled)
    literature_context = {}  # Will hold (baseline, fact_checks) per paper
    if use_literature_grounding and len(papers) > 0:
        print(f"\n" + "=" * 80, flush=True)
        print(f"LITERATURE GROUNDING ENHANCEMENT", flush=True)
        print("=" * 80, flush=True)
        literature_start = time.time()

        for idx, paper in enumerate(papers, 1):
            print(f"\n📚 [{idx}/{len(papers)}] Gathering literature context for: {paper.filename}", flush=True)

            try:
                # Stage 1: Librarian - Create baseline reference
                print(f"   [Stage 1/3] Librarian: Creating baseline reference...", flush=True, end=" ")
                baseline = create_baseline_reference(paper, config)
                if baseline and baseline.baseline_papers:
                    print(f"✅ Found {len(baseline.baseline_papers)} papers", flush=True)
                else:
                    print(f"⚠️  No baseline papers found", flush=True)
                    baseline = None

                # Stage 2: Reader - Would be done during extraction with novelty rankings
                # Stage 3: Fact-Checker - Would be done after extraction
                # For now, store the baseline reference
                literature_context[paper.filename] = {
                    "baseline": baseline,
                    "fact_checks": []
                }

            except Exception as e:
                print(f"   ❌ Error: {e}", flush=True)
                literature_context[paper.filename] = {"baseline": None, "fact_checks": []}

        literature_time = time.time() - literature_start
        print(f"\n✅ [Literature Grounding] Context gathered for {len(papers)} paper(s) in {literature_time:.1f}s", flush=True)
        print("=" * 80, flush=True)
    
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

        # Get literature context for this paper (if available)
        paper_literature = literature_context.get(paper.filename, {"baseline": None, "fact_checks": []})
        baseline = paper_literature["baseline"]

        # 8. Agent 1: Extract Evidence (with or without literature enhancement)
        if use_literature_grounding and baseline:
            print(f"   🔍 [Agent 1] Starting evidence extraction (LITERATURE-ENHANCED)...", flush=True)
            extraction_start = time.time()

            # Use literature-aware extraction (adds novelty rankings)
            # process_paper_extractions_literature is from agent_reader and accepts baseline parameter
            extractions = process_paper_extractions_literature(
                paper=paper,
                config=config,
                baseline=baseline
            )
            extraction_time = time.time() - extraction_start

            if not extractions:
                print(f"   ❌ [Agent 1] Failed to get extractions. Falling back to standard extraction.", flush=True)
                extractions = process_paper_extractions(paper, config)
                extraction_time = time.time() - extraction_start
        else:
            print(f"   🔍 [Agent 1] Starting evidence extraction...", flush=True)
            extraction_start = time.time()
            extractions = process_paper_extractions(paper, config)
            extraction_time = time.time() - extraction_start

        if not extractions:
            print(f"   ❌ [Agent 1] Failed to get any extractions for {paper.filename}. Skipping.", flush=True)
            continue

        print(f"   ✅ [Agent 1] Completed {len(extractions)} extractions in {extraction_time:.1f}s", flush=True)

        # 8.5. Fact-Checker (if literature grounding enabled)
        fact_checks = []
        if use_literature_grounding and baseline:
            print(f"   🔎 [Agent 1.5] Running fact-checking...", flush=True, end=" ")
            fact_check_start = time.time()

            try:
                fact_checks = run_fact_checks(paper, extractions, config) or []
                fact_check_time = time.time() - fact_check_start

                if fact_checks:
                    disputed = sum(1 for fc in fact_checks if fc.verification_status == "disputed")
                    print(f"✅ {len(fact_checks)} checks ({disputed} disputed) in {fact_check_time:.1f}s", flush=True)
                else:
                    print(f"✅ No verification triggers in {fact_check_time:.1f}s", flush=True)

            except Exception as e:
                print(f"⚠️  Skipped: {e}", flush=True)
                fact_checks = []

        # 9. Agent 2: Synthesize Review (with or without literature enhancement)
        if use_literature_grounding:
            print(f"   📝 [Agent 2] Starting synthesis (LITERATURE-ENHANCED)...", flush=True)
            synthesis_start = time.time()

            # First, get the standard review with all detailed criterion narratives
            print(f"   📝 [Agent 2a] Running standard synthesis for detailed review...", flush=True)
            base_review = synthesize_review(paper, extractions, config)

            if not base_review:
                print(f"   ❌ [Agent 2a] Standard synthesis failed. Using grounded synthesis directly.", flush=True)
                review = synthesize_grounded_review(
                    paper=paper,
                    extractions=extractions,
                    config=config,
                    baseline=baseline,
                    fact_checks=fact_checks
                )
            else:
                print(f"   ✅ [Agent 2a] Standard synthesis complete (score: {base_review.overall_score:.1f})", flush=True)
                # Then enhance with literature context (add Research Trajectory, novelty adjustment)
                print(f"   📚 [Agent 2b] Adding literature context...", flush=True)
                review = enhance_review_with_literature(
                    base_review=base_review,
                    paper=paper,
                    baseline=baseline,
                    fact_checks=fact_checks,
                    extractions=extractions,
                    config=config
                )
            synthesis_time = time.time() - synthesis_start
        else:
            print(f"   📝 [Agent 2] Starting review synthesis...", flush=True)
            synthesis_start = time.time()
            review = synthesize_review(paper, extractions, config)
            synthesis_time = time.time() - synthesis_start

        if not review:
            print(f"   ❌ [Agent 2] Failed to synthesize review for {paper.filename}. Skipping.", flush=True)
            continue

        print(f"   ✅ [Agent 2] Completed synthesis in {synthesis_time:.1f}s", flush=True)

        # 10. Save Individual Output
        print(f"   💾 [Output] Saving review...", flush=True)

        # Generate filename following the naming convention:
        # [paper_name]_[extractor_model]_[synthesizer_model]_[timestamp].md
        paper_base_name = os.path.splitext(paper.filename)[0]
        extractor_model_name = review.extractor_model_used.replace("/", "_")
        synthesizer_model_name = review.synthesizer_model_used.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        review_filename = f"{paper_base_name}_{extractor_model_name}_{synthesizer_model_name}_{timestamp}.md"
        output_path = os.path.join(dirs["reviews_dir"], review_filename)

        # Save with literature context indicator
        save_review_markdown(
            review=review,
            output_path=output_path,
            paper=paper,
            config=config,
            include_literature_context=use_literature_grounding
        )
        
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
    if new_processing_cost > 0 and processed_count > 0:
        avg_cost_per_paper = new_processing_cost / processed_count
        if avg_cost_per_paper > 1.0:  # Arbitrary threshold
            print(f"\n⚠️  [Warning] High average cost per paper: ${avg_cost_per_paper:.4f}", flush=True)
            print(f"   Consider using cheaper models for extraction to reduce costs", flush=True)
    
    print(f"⏱️  [Summary] Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)", flush=True)
    if len(papers) > 0:
        print(f"📈 [Summary] Average time per paper: {total_time/len(papers):.1f} seconds", flush=True)
    print(f"📁 [Summary] Results saved in: {dirs['outputs_dir']}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()