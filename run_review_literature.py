#!/usr/bin/env python3
"""
Run Review with Literature Grounding

Enhanced review workflow that adds literature grounding stages:
1. Librarian: Create baseline reference from most cited papers
2. Reader: Extract evidence with novelty ranking
3. Fact-Checker: Verify suspicious claims
4. Critic: Synthesize with research trajectory

Usage:
    python run_review_literature.py <run_directory> [--no-literature]

The --no-literature flag disables literature grounding and runs in standard mode.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_models import Paper, GroundedReview, BaselineReference, LiteratureContext
from core.config_loader import Config
from core.paper_ingestor import ingest_directory
from agents.agent_librarian import create_baseline_reference, create_baseline_reference_batch
from agents.agent_reader import process_paper_extractions_with_novelty
from agents.agent_fact_checker import run_fact_checks, summarize_fact_checks
from agents.agent_critic import synthesize_grounded_review
from utilities.output_generator import save_review_markdown, save_consolidated_csv
from utilities.helpers import load_yaml_config


def should_enable_literature_grounding(
    config_file: str = "config/literature_sources.yaml",
    env_override: bool = True
) -> bool:
    """
    Check if literature grounding is enabled.

    Args:
        config_file: Path to literature configuration
        env_override: Whether to allow environment variable override

    Returns:
        True if literature grounding should be enabled
    """
    # Check environment override first
    if env_override:
        env_value = os.getenv("LITERATURE_GROUNDING_ENABLED", "").lower()
        if env_value in ("false", "0", "no", "disabled"):
            return False
        elif env_value in ("true", "1", "yes", "enabled"):
            return True

    # Check configuration file
    try:
        lit_config = load_yaml_config(config_file)
        return lit_config.get("enabled", True)
    except:
        return True  # Default to enabled


def run_literature_grounded_review(
    papers: List[Paper],
    config: Config,
    literature_config: dict,
    progress_callback: Optional[callable] = None
) -> List[GroundedReview]:
    """
    Run the full literature-grounded review pipeline.

    Args:
        papers: List of papers to review
        config: System configuration
        literature_config: Literature grounding configuration
        progress_callback: Optional callback for progress updates

    Returns:
        List of grounded reviews
    """
    reviews = []

    total_papers = len(papers)
    for i, paper in enumerate(papers, 1):
        if progress_callback:
            progress_callback(i, total_papers, f"Processing: {paper.filename}")

        print(f"\n{'='*60}")
        print(f"LITERATURE-GROUNDED REVIEW [{i}/{total_papers}]: {paper.filename}")
        print(f"{'='*60}")

        try:
            # ====================================================================
            # STAGE 1: LIBRARIAN - Create Baseline Reference
            # ====================================================================
            print("\n[Stage 1/4] Librarian: Creating baseline reference...")
            baseline = create_baseline_reference(paper, config, literature_config)

            if not baseline:
                print("[Librarian] Failed to create baseline reference. Falling back to standard review.")
                # Fall back to standard review
                from agents.agent_reader import process_paper_extractions
                from agents.agent_synthesizer import synthesize_review

                extractions = process_paper_extractions(paper, config)
                review = synthesize_review(paper, extractions, config)

                if review:
                    reviews.append(review)
                continue

            print(f"[Librarian] ✓ Created baseline with {len(baseline.baseline_papers)} papers")

            # ====================================================================
            # STAGE 2: READER - Extract with Novelty Ranking
            # ====================================================================
            print("\n[Stage 2/4] Reader: Extracting evidence with novelty ranking...")
            extractions = process_paper_extractions_with_novelty(
                paper=paper,
                config=config,
                baseline=baseline
            )

            if not extractions:
                print("[Reader] Failed to extract evidence. Skipping paper.")
                continue

            print(f"[Reader] ✓ Completed {len(extractions)} criterion extractions")

            # ====================================================================
            # STAGE 3: FACT-CHECKER - Verify Suspicious Claims
            # ====================================================================
            print("\n[Stage 3/4] Fact-Checker: Running verification checks...")
            fact_checks = run_fact_checks(
                extractions=extractions,
                criteria=config.get_criteria(),
                config=config,
                literature_config=literature_config
            )

            if fact_checks:
                print(f"[Fact-Checker] ✓ Completed {len(fact_checks)} verification checks")
                # Print summary
                disputed = [fc for fc in fact_checks if fc.verification_status == "disputed"]
                if disputed:
                    print(f"[Fact-Checker] ⚠️  {len(disputed)} claims require further review")
            else:
                print("[Fact-Checker] ✓ No verification triggers detected")

            # ====================================================================
            # STAGE 4: CRITIC - Synthesize with Research Trajectory
            # ====================================================================
            print("\n[Stage 4/4] Critic: Synthesizing grounded review...")
            review = synthesize_grounded_review(
                paper=paper,
                extractions=extractions,
                config=config,
                baseline=baseline,
                fact_checks=fact_checks
            )

            if review:
                print(f"[Critic] ✓ Review synthesized (score: {review.overall_score:.1f})")
                reviews.append(review)

                # Print summary
                avg_novelty = sum(e.novelty_ranking for e in extractions) / len(extractions)
                print(f"\n[Summary]")
                print(f"  Overall Score: {review.overall_score:.1f}")
                print(f"  Recommendation: {review.recommendation}")
                print(f"  Avg Novelty: {avg_novelty:.2f}/5")
                if review.novelty_adjusted_score:
                    adjustment = review.novelty_adjusted_score - review.overall_score
                    print(f"  Novelty Adjustment: {adjustment:+.1f}")

            else:
                print(f"[Critic] ✗ Failed to synthesize review")

        except Exception as e:
            print(f"\n[Error] Failed to review {paper.filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return reviews


def run_standard_review(
    papers: List[Paper],
    config: Config,
    progress_callback: Optional[callable] = None
) -> List:
    """
    Run standard review without literature grounding.

    Args:
        papers: List of papers to review
        config: System configuration
        progress_callback: Optional callback for progress updates

    Returns:
        List of standard reviews
    """
    from agents.agent_reader import process_paper_extractions
    from agents.agent_synthesizer import synthesize_review

    reviews = []
    total_papers = len(papers)

    for i, paper in enumerate(papers, 1):
        if progress_callback:
            progress_callback(i, total_papers, f"Processing: {paper.filename}")

        print(f"\n{'='*60}")
        print(f"STANDARD REVIEW [{i}/{total_papers}]: {paper.filename}")
        print(f"{'='*60}")

        try:
            extractions = process_paper_extractions(paper, config)
            review = synthesize_review(paper, extractions, config)

            if review:
                reviews.append(review)
                print(f"✓ Review complete (score: {review.overall_score:.1f})")

        except Exception as e:
            print(f"✗ Failed to review {paper.filename}: {e}")
            continue

    return reviews


def main():
    parser = argparse.ArgumentParser(
        description="Run literature-grounded academic paper reviews"
    )
    parser.add_argument(
        "run_directory",
        help="Directory containing papers to review (or run directory with config)"
    )
    parser.add_argument(
        "--no-literature",
        action="store_true",
        help="Disable literature grounding (use standard review)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for reviews (default: <run_directory>/outputs)"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_directory)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return 1

    # ========================================================================
    # SETUP
    # ========================================================================
    print("\n" + "="*60)
    print("AGENTIC ACADEMIC REVIEW SYSTEM v2.0")
    print("Literature-Grounded Review Mode")
    print("="*60)

    # Determine if literature grounding is enabled
    use_literature = not args.no_literature and should_enable_literature_grounding()

    if use_literature:
        print("\n[Literature Grounding] ENABLED")
        print("  - Stage 1: Librarian (baseline reference)")
        print("  - Stage 2: Reader (novelty-ranking)")
        print("  - Stage 3: Fact-Checker (verification)")
        print("  - Stage 4: Critic (research trajectory)")
    else:
        print("\n[Literature Grounding] DISABLED")
        print("  - Running in standard review mode")

    # Load configuration
    print(f"\n[Config] Loading from: {run_dir}")
    config = Config(str(run_dir))
    print(f"[Config] Domain: {config.domain}")

    # Load literature configuration
    literature_config = {}
    if use_literature:
        try:
            literature_config = load_yaml_config("config/literature_sources.yaml")
            print(f"[Config] Literature config loaded")
        except Exception as e:
            print(f"[Warning] Failed to load literature config: {e}")
            use_literature = False

    # ========================================================================
    # INGESTION
    # ========================================================================
    print("\n[Ingestion] Processing papers...")
    papers = ingest_directory(str(run_dir / "papers"), config)

    if not papers:
        print("[Ingestion] No papers found to review")
        return 1

    print(f"[Ingestion] ✓ Processed {len(papers)} papers")

    # ========================================================================
    # REVIEW
    # ========================================================================
    print("\n[Review] Starting review pipeline...")

    if use_literature:
        reviews = run_literature_grounded_review(
            papers=papers,
            config=config,
            literature_config=literature_config
        )
    else:
        reviews = run_standard_review(
            papers=papers,
            config=config
        )

    if not reviews:
        print("\n[Review] No reviews generated")
        return 1

    # ========================================================================
    # OUTPUT
    # ========================================================================
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    print(f"\n[Output] Saving reviews to: {output_dir}")

    # Save individual reviews
    for review in reviews:
        filename = f"{review.paper_filename}_review.md"
        filepath = output_dir / filename

        save_review_markdown(
            review=review,
            output_path=str(filepath),
            include_literature_context=use_literature
        )
        print(f"  - {filename}")

    # Save consolidated CSV
    csv_path = output_dir / "consolidated_reviews.csv"
    save_consolidated_csv(
        reviews=reviews,
        output_path=str(csv_path),
        include_literature_metrics=use_literature
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("REVIEW COMPLETE")
    print("="*60)
    print(f"\nPapers Reviewed: {len(reviews)}")
    print(f"Output Directory: {output_dir}")

    # Calculate statistics
    scores = [r.overall_score for r in reviews]
    print(f"\nScore Statistics:")
    print(f"  Average: {sum(scores)/len(scores):.1f}")
    print(f"  Range: {min(scores):.1f} - {max(scores):.1f}")

    recommendations = {}
    for r in reviews:
        recommendations[r.recommendation] = recommendations.get(r.recommendation, 0) + 1

    print(f"\nRecommendations:")
    for rec, count in recommendations.items():
        print(f"  {rec}: {count}")

    total_cost = sum(r.total_cost for r in reviews)
    print(f"\nTotal Cost: ${total_cost:.4f}")

    if use_literature:
        # Calculate novelty statistics
        all_novelty = []
        for r in reviews:
            if hasattr(r, 'literature_context') and r.literature_context.baseline_reference:
                # This is a grounded review
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
