import os
import pandas as pd
from typing import List, Optional
from datetime import datetime
from core.data_models import Review, Paper, GroundedReview
from core.config_loader import Config
from utilities.helpers import sanitize_model_name

def save_review_markdown(review: Review, paper: Paper, config: Config, output_dir="outputs/reviews"):
    """
    Saves the final review as a detailed Markdown file
    using the new <paper>_<ext_model>_<syn_model>_<timestamp> format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paper_base_name = os.path.splitext(paper.filename)[0]
    # Format timestamp for filename (needs to be Windows-compatible)
    timestamp = review.synthesis_timestamp.strftime('%Y%m%d_%H%M%S')
    
    ext_model_name = sanitize_model_name(review.extractor_model_used)
    syn_model_name = sanitize_model_name(review.synthesizer_model_used)
    
    filename = f"{paper_base_name}_{ext_model_name}_{syn_model_name}_{timestamp}.md"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Review: {review.paper_title}\n\n")
        f.write(f"**Paper ID:** `{paper.id}`\n")
        f.write(f"**Original Filename:** {paper.filename}\n")
        f.write(f"**Review Date:** {review.synthesis_timestamp.strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write(f"## Overall Assessment\n\n")
        f.write(f"**Recommendation:** **{review.recommendation.upper()}**\n")
        f.write(f"**Overall Score:** **{review.overall_score:.1f} / 100**\n")
        f.write(f"**Rationale:** {review.recommendation_rationale}\n")
        f.write(f"**Confidence:** {review.decision_confidence:.0%}\n\n")
        
        f.write(f"## Executive Summary\n\n{review.executive_summary}\n\n")
        
        f.write(f"## Detailed Assessment\n\n")
        f.write("### Major Strengths\n")
        for item in review.detailed_assessment.major_strengths:
            f.write(f"- {item}\n")
        f.write("\n### Major Concerns\n")
        if not review.detailed_assessment.major_concerns:
            f.write("None\n")
        for item in review.detailed_assessment.major_concerns:
            f.write(f"- {item}\n")
        f.write("\n### Minor Issues\n")
        if not review.detailed_assessment.minor_issues:
            f.write("None\n")
        for item in review.detailed_assessment.minor_issues:
            f.write(f"- {item}\n")
        
        f.write(f"\n## Revision Suggestions\n\n")
        if not review.revision_suggestions:
            f.write("None\n")
        for i, item in enumerate(review.revision_suggestions, 1):
            f.write(f"{i}. {item}\n")
            
        f.write(f"\n## Criterion-by-Criterion Analysis\n\n")
        for crit_id, narrative in review.criterion_narrative.items():
            criterion = config.get_criterion_by_id(crit_id)
            breakdown = review.weighted_breakdown.get(crit_id)
            if criterion and breakdown:
                f.write(f"### {criterion.get('name', crit_id)} (Weight: {breakdown.weight}%, Score: {breakdown.score})\n")
                f.write(f"{narrative}\n\n")
            
        f.write(f"## Metadata\n\n")
        f.write(f"- **Extractor Model:** {review.extractor_model_used}\n")
        f.write(f"- **Synthesizer Model:** {review.synthesizer_model_used}\n")
        f.write(f"- **Total API Cost:** ${review.total_cost:.4f}\n")
        
    print(f"[Output] Saved Markdown review to {filepath}")

def save_consolidated_csv(reviews: List[Review], output_dir="outputs/reports"):
    """
    Saves a CSV summary of all reviews.
    - Adds 'paper_filename' (human-readable)
    - Fixes the 'title' column bug
    - Removes 'paper_id' (meaningless to user)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f"report_consolidated_{timestamp}.csv")
    
    report_data = []
    for review in reviews:
        
        paper_base_name = os.path.splitext(review.paper_filename)[0]
        
        title = review.paper_title
        if title.strip() == "No Title Found" or title.strip().startswith("Placeholder:"):
            title = paper_base_name

        data = {
            "paper_filename": paper_base_name,
            "title": title,
            "overall_score": review.overall_score,
            "recommendation": review.recommendation,
            "confidence": review.decision_confidence,
            "total_cost_usd": review.total_cost,
            "extractor_model_used": review.extractor_model_used,
            "synthesizer_model_used": review.synthesizer_model_used,
            "timestamp": review.synthesis_timestamp # Pass through the timestamp
        }
        
        for crit_id, breakdown in review.weighted_breakdown.items():
            data[f"{crit_id}_score"] = breakdown.score
        
        report_data.append(data)
        
    df = pd.DataFrame(report_data)
    
    all_cols = list(df.columns)
    core_cols = [
        "paper_filename", "title", "extractor_model_used", "synthesizer_model_used", 
        "overall_score", "recommendation", "confidence", "total_cost_usd", "timestamp"
    ]

    score_cols = [col for col in all_cols if col.endswith("_score")]
    other_cols = [c for c in all_cols if c not in core_cols and c not in score_cols]
    
    final_cols = [c for c in (core_cols + score_cols + other_cols) if c in df.columns]
    
    df = df[final_cols]
    
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"[Output] Saved consolidated CSV report to {filepath}")


# ============================================================================
# LITERATURE-GROUNDED REVIEW OUTPUT
# ============================================================================

def save_review_markdown(
    review: Review,
    output_path: str,
    paper: Optional[Paper] = None,
    config: Optional[Config] = None,
    include_literature_context: bool = False
):
    """
    Save a review as Markdown.

    This function handles both standard reviews and grounded reviews.

    Args:
        review: The review (Review or GroundedReview)
        output_path: Path to save the markdown file
        paper: Optional paper object (for backward compatibility)
        config: Optional config (for backward compatibility)
        include_literature_context: Whether to include literature context sections
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Review: {review.paper_title}\n\n")
        f.write(f"**Paper ID:** `{review.paper_id}`\n")
        f.write(f"**Filename:** {review.paper_filename}\n")
        f.write(f"**Review Date:** {review.synthesis_timestamp.strftime('%Y-%m-%d %H:%M')}\n\n")

        # Overall Assessment
        f.write(f"## Overall Assessment\n\n")
        f.write(f"**Recommendation:** **{review.recommendation.upper()}**\n")
        f.write(f"**Overall Score:** **{review.overall_score:.1f} / 100**\n")
        f.write(f"**Rationale:** {review.recommendation_rationale}\n")
        f.write(f"**Confidence:** {review.decision_confidence:.0%}\n\n")

        # Literature-Grounded Sections (if available and enabled)
        if include_literature_context and isinstance(review, GroundedReview):
            # Research Trajectory
            if review.research_trajectory_section:
                f.write(f"\n{review.research_trajectory_section}\n\n")

            # Novelty-Adjusted Score
            if review.novelty_adjusted_score is not None:
                adjustment = review.novelty_adjusted_score - review.overall_score
                f.write(f"### Novelty Impact on Score\n\n")
                f.write(f"Base Score: {review.overall_score:.1f}\n")
                f.write(f"Novelty Adjustment: {adjustment:+.1f}\n")
                f.write(f"**Novelty-Adjusted Score: {review.novelty_adjusted_score:.1f}**\n\n")

            # Fact-Check Summary
            if review.literature_context and review.literature_context.fact_checks:
                from agents.agent_fact_checker import summarize_fact_checks
                f.write("\n" + summarize_fact_checks(review.literature_context.fact_checks) + "\n")

        # Executive Summary
        f.write(f"## Executive Summary\n\n{review.executive_summary}\n\n")

        # Detailed Assessment
        f.write(f"## Detailed Assessment\n\n")
        f.write("### Major Strengths\n")
        for item in review.detailed_assessment.major_strengths:
            f.write(f"- {item}\n")
        f.write("\n### Major Concerns\n")
        if not review.detailed_assessment.major_concerns:
            f.write("None\n")
        for item in review.detailed_assessment.major_concerns:
            f.write(f"- {item}\n")
        f.write("\n### Minor Issues\n")
        if not review.detailed_assessment.minor_issues:
            f.write("None\n")
        for item in review.detailed_assessment.minor_issues:
            f.write(f"- {item}\n")

        # Revision Suggestions
        f.write(f"\n## Revision Suggestions\n\n")
        if not review.revision_suggestions:
            f.write("None\n")
        for i, item in enumerate(review.revision_suggestions, 1):
            f.write(f"{i}. {item}\n")

        # Criterion-by-Criterion Analysis
        f.write(f"\n## Criterion-by-Criterion Analysis\n\n")
        for crit_id, narrative in review.criterion_narrative.items():
            breakdown = review.weighted_breakdown.get(crit_id)
            if breakdown:
                f.write(f"### {crit_id} (Weight: {breakdown.weight}%, Score: {breakdown.score})\n")
                f.write(f"{narrative}\n\n")

        # Metadata
        f.write(f"## Metadata\n\n")
        f.write(f"- **Extractor Model:** {review.extractor_model_used}\n")
        f.write(f"- **Synthesizer Model:** {review.synthesizer_model_used}\n")
        f.write(f"- **Total API Cost:** ${review.total_cost:.4f}\n")

    print(f"[Output] Saved review to {output_path}")


def save_consolidated_csv(
    reviews: List[Review],
    output_path: str,
    include_literature_metrics: bool = False
):
    """
    Save a consolidated CSV of all reviews.

    Args:
        reviews: List of reviews (Review or GroundedReview)
        output_path: Path to save the CSV
        include_literature_metrics: Whether to include literature grounding columns
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    report_data = []
    for review in reviews:
        paper_base_name = os.path.splitext(review.paper_filename)[0]
        title = review.paper_title
        if title.strip() == "No Title Found":
            title = paper_base_name

        data = {
            "paper_filename": paper_base_name,
            "title": title,
            "overall_score": review.overall_score,
            "recommendation": review.recommendation,
            "confidence": review.decision_confidence,
            "total_cost_usd": review.total_cost,
            "extractor_model_used": review.extractor_model_used,
            "synthesizer_model_used": review.synthesizer_model_used,
            "timestamp": review.synthesis_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Add criterion scores
        for crit_id, breakdown in review.weighted_breakdown.items():
            data[f"{crit_id}_score"] = breakdown.score

        # Add literature metrics if available
        if include_literature_metrics and isinstance(review, GroundedReview):
            if review.novelty_adjusted_score is not None:
                data["novelty_adjusted_score"] = review.novelty_adjusted_score
                data["novelty_adjustment"] = review.novelty_adjusted_score - review.overall_score

            if review.literature_context:
                ctx = review.literature_context
                data["literature_api_calls"] = ctx.total_api_calls
                data["fact_checks_performed"] = len(ctx.fact_checks)

                disputed = sum(1 for fc in ctx.fact_checks if fc.verification_status == "disputed")
                data["fact_checks_disputed"] = disputed

                if ctx.baseline_reference:
                    data["baseline_papers_count"] = len(ctx.baseline_reference.baseline_papers)

        report_data.append(data)

    df = pd.DataFrame(report_data)

    # Organize columns
    all_cols = list(df.columns)
    core_cols = [
        "paper_filename", "title", "extractor_model_used", "synthesizer_model_used",
        "overall_score", "recommendation", "confidence", "total_cost_usd", "timestamp"
    ]

    if include_literature_metrics:
        literature_cols = [
            "novelty_adjusted_score", "novelty_adjustment",
            "baseline_papers_count", "literature_api_calls",
            "fact_checks_performed", "fact_checks_disputed"
        ]
        core_cols.extend(literature_cols)

    score_cols = [col for col in all_cols if col.endswith("_score")]
    other_cols = [c for c in all_cols if c not in core_cols and c not in score_cols]

    final_cols = [c for c in (core_cols + score_cols + other_cols) if c in df.columns]
    df = df[final_cols]

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[Output] Saved consolidated CSV to {output_path}")