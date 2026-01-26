import pandas as pd
import os
import glob
import json
from datetime import datetime
from core.llm_wrapper import call_llm
from core.config_loader import Config
from utilities.helpers import sanitize_model_name, get_judge_config_hash
from core.paper_ingestor import load_ingestion_cache

# --- Configuration ---
CONFLICT_COLUMN = "recommendation" # The column to check for conflicts

# --- LLM-as-a-Judge Settings ---
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "openai")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
JUDGE_TEMPERATURE = 0.1

# The prompt for the Judge LLM
JUDGE_SYSTEM_PROMPT = """
You are a senior academic editor and expert adjudicator. Your task is to 
resolve a conflict between two AI-generated reviews for the same paper.
You will be given the original paper's content and the two conflicting reviews.

Your goal is to:
1.  Read the original paper to understand its core claims and quality.
2.  Read both reviews.
3.  Determine which review (A or B) is more accurate, insightful, and better
    justified by the evidence.
4.  Provide a final, authoritative recommendation (Accept, Accept w/ Revisions, 
    Revise and Resubmit, Reject) for the paper.
5.  Explain your reasoning clearly.

Output your verdict *only* as a valid JSON object.
"""

JUDGE_USER_PROMPT = """
# ORIGINAL PAPER CONTENT (Markdown)
{paper_content}

---

# REVIEW A (Champion of Group A)
**Model:** {model_a}
**Recommendation:** {recommendation_a}
**Score:** {score_a}

{review_a_content}

---

# REVIEW B (Champion of Group B)
**Model:** {model_b}
**Recommendation:** {recommendation_b}
**Score:** {score_b}

{review_b_content}

---

# TASK
Act as the senior editor. Analyze the paper and the two conflicting reviews.
Provide your verdict in the following JSON format:

{{
  "judge_recommendation": "Your final recommendation for the paper (Accept, Accept with Revisions, Revise and Resubmit, Reject)",
  "judge_rationale": "A 2-3 sentence explanation for your recommendation, based on the original paper.",
  "winning_review": "A, B, or 'Neither'. Which review was more accurate and helpful?",
  "winning_review_rationale": "A 2-3 sentence explanation of why you chose A, B, or Neither. Be specific about the flaws or strengths of each review."
}}
"""

def find_latest_discrepancy_report(reports_dir: str) -> str:
    """Finds the most recent 'HUMAN_REVIEW_discrepancies' file."""
    files = glob.glob(os.path.join(reports_dir, "HUMAN_REVIEW_discrepancies_*.csv"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_paper_cache(cache_file: str) -> dict[str, str]:
    """Loads the ingestion cache and returns a dict of {filename: content}."""
    print("Loading paper ingestion cache...")
    cache = load_ingestion_cache(cache_file) # This returns a dict of {filename: Paper}
    return {filename: paper.content_markdown for filename, paper in cache.items()}

def find_review_file_path(row, reviews_dir: str) -> str:
    """Reconstructs the .md review filepath from a CSV row."""
    try:
        paper_base = row['paper_filename']
        ext_model = sanitize_model_name(row['extractor_model_used'])
        syn_model = sanitize_model_name(row['synthesizer_model_used'])
        
        # Convert pandas timestamp string back to datetime obj, then to filename format
        ts_obj = pd.to_datetime(row['timestamp'])
        ts_str = ts_obj.strftime('%Y%m%d_%H%M%S')
        
        filename = f"{paper_base}_{ext_model}_{syn_model}_{ts_str}.md"
        return os.path.join(reviews_dir, filename)
    except Exception as e:
        print(f"Error reconstructing filename for {row['paper_filename']}: {e}")
        return None

def read_review_content(filepath: str) -> str:
    """Reads the content of a single .md review file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"  [Judge Error] Review file not found: {filepath}")
        return f"ERROR: Review file not found at {filepath}"
    except Exception as e:
        print(f"  [Judge Error] Could not read file {filepath}: {e}")
        return f"ERROR: Could not read file: {e}"

# --- FIX 1: Added 'reviews_dir' as an argument ---
def run_judge(paper_content: str, review_a_row: pd.Series, review_b_row: pd.Series, reviews_dir: str) -> dict[str, any]:
    """Calls the Judge LLM to adjudicate a single conflict."""
    
    # --- FIX 2: Pass 'reviews_dir' to the function ---
    review_a_path = find_review_file_path(review_a_row, reviews_dir)
    review_b_path = find_review_file_path(review_b_row, reviews_dir)
    # --- END FIX 2 ---
    
    review_a_content = read_review_content(review_a_path)
    review_b_content = read_review_content(review_b_path)
    
    max_paper_tokens = 100000 
    if len(paper_content) > max_paper_tokens * 4:
        print("  [Judge] Truncating paper content for judge prompt...")
        paper_content = paper_content[:max_paper_tokens * 4]

    prompt = JUDGE_USER_PROMPT.format(
        paper_content=paper_content,
        model_a=f"{review_a_row['extractor_model_used']} / {review_a_row['synthesizer_model_used']}",
        recommendation_a=review_a_row['recommendation'],
        score_a=f"{review_a_row['overall_score']:.1f}",
        review_a_content=review_a_content,
        model_b=f"{review_b_row['extractor_model_used']} / {review_b_row['synthesizer_model_used']}",
        recommendation_b=review_b_row['recommendation'],
        score_b=f"{review_b_row['overall_score']:.1f}",
        review_b_content=review_b_content
    )
    
    config = Config()
    llm_config = config.get_llm_config()

    response = call_llm(
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        provider=JUDGE_PROVIDER,
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_retries=llm_config['max_retries']
    )
    
    if not response['success']:
        print(f"  [Judge Error] LLM call failed: {response['error']}")
        return {"error": response['error']}

    try:
        raw_content = response['content']
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')
        json_text = raw_content[start_index : end_index + 1]
        verdict = json.loads(json_text)
        verdict['cost'] = response['cost']
        return verdict
    except Exception as e:
        print(f"  [Judge Error] Failed to parse judge's JSON response: {e}")
        print(f"  [Judge Error] Raw output: {response['content']}")
        return {"error": f"Failed to parse JSON: {e}"}

def load_judge_progress(progress_file: str, current_config_hash: str) -> dict[str, dict]:
    """Load the judge progress file and check if configuration matches."""
    if not os.path.exists(progress_file):
        return {}
        
    try:
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
        
        # Check if configuration has changed
        stored_config_hash = progress_data.get("config_hash")
        if stored_config_hash != current_config_hash:
            print(f"[Judge Progress] Judge configuration has changed (stored: {stored_config_hash[:8]}..., current: {current_config_hash[:8]}...)")
            print("[Judge Progress] Resetting judge progress - all conflicts will be readjudicated with new configuration")
            return {}
        
        print(f"[Judge Progress] Judge configuration matches (hash: {current_config_hash[:8]}...)")
        print(f"[Judge Progress] Loaded {len(progress_data.get('papers', {}))} adjudicated papers from previous run")
        return progress_data.get('papers', {})
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Judge Progress] Error loading progress file: {e}")
        print("[Judge Progress] Starting fresh")
        return {}

def save_judge_progress(progress_file: str, progress: dict[str, dict], config_hash: str):
    """Save the judge progress file with configuration hash."""
    import json
    from datetime import datetime
    
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    progress_data = {
        "config_hash": config_hash,
        "papers": progress
    }
    
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2, default=datetime_handler)

def adjudicate_conflicts(run_dir: str = None):
    """Main function to run the LLM-as-a-Judge system."""
    
    print("--- Starting LLM-as-a-Judge Adjudication ---")
    
    # Use custom directories if provided
    if run_dir:
        reports_dir = os.path.join(run_dir, "outputs", "reports")
        reviews_dir = os.path.join(run_dir, "outputs", "reviews")
        cache_file = os.path.join(run_dir, "ingestion_cache.json")
        judge_progress_file = os.path.join(run_dir, "judge_progress.json")
    else:
        # Fallback to default structure
        reports_dir = "outputs/reports"
        reviews_dir = "outputs/reviews"
        cache_file = "outputs/ingestion_cache.json" # Assumes cache is in outputs
        if not os.path.exists(cache_file):
             cache_file = "ingestion_cache.json" # Tries root
        judge_progress_file = "outputs/judge_progress.json"
    
    # Generate judge configuration hash
    judge_config_hash = get_judge_config_hash()
    print(f"[Judge] Judge configuration hash: {judge_config_hash[:8]}...")
    
    # Load judge progress
    judge_progress = load_judge_progress(judge_progress_file, judge_config_hash)
    
    discrepancy_file = find_latest_discrepancy_report(reports_dir)
    if not discrepancy_file:
        print(f"Error: No 'HUMAN_REVIEW_discrepancies_*.csv' file found in {reports_dir}.")
        print("Please run 'compare_reports.py' first.")
        return
        
    print(f"Loading discrepancy report: {discrepancy_file}")
    df = pd.read_csv(discrepancy_file)
    
    paper_content_cache = load_paper_cache(cache_file)
    if not paper_content_cache:
        print(f"Error: Could not load '{cache_file}'.")
        return
        
    grouped = df.groupby('paper_filename')
    final_verdicts = []
    total_cost = 0.0
    
    print(f"Found {len(grouped)} papers with conflicts to adjudicate...")
    
    for i, (filename, group) in enumerate(grouped):
        # Skip if already adjudicated with current judge configuration
        if filename in judge_progress:
            print(f"\n--- Skipping Adjudication {i+1}/{len(grouped)}: {filename} (already adjudicated) ---")
            verdict_data = judge_progress[filename]
            final_verdicts.append(verdict_data)
            total_cost += verdict_data.get('judge_cost', 0.0)
            continue
            
        print(f"\n--- Adjudicating Paper {i+1}/{len(grouped)}: {filename} ---")
        
        # --- NEW "CHAMPION vs. CHAMPION" LOGIC ---
        if len(group) < 2:
            print(f"  [Judge Error] Group for {filename} has < 2 rows. Skipping.")
            continue
            
        # 1. Get counts of each recommendation
        counts = group[CONFLICT_COLUMN].value_counts()
        
        if len(counts) < 2:
            # This shouldn't happen if compare_reports.py ran correctly
            print(f"  [Judge Error] Group for {filename} has no conflict. Skipping.")
            continue
            
        # 2. Get the first two conflicting recommendation types
        rec_a_name = counts.index[0]
        rec_b_name = counts.index[1]
        
        # 3. Get all reviews for Group A and find its champion (highest score)
        group_a_reviews = group[group[CONFLICT_COLUMN] == rec_a_name]
        review_a_row = group_a_reviews.loc[group_a_reviews['overall_score'].idxmax()]
        
        # 4. Get all reviews for Group B and find its champion (highest score)
        group_b_reviews = group[group[CONFLICT_COLUMN] == rec_b_name]
        review_b_row = group_b_reviews.loc[group_b_reviews['overall_score'].idxmax()]
        
        print(f"  [Judge] Found conflict between '{rec_a_name}' ({len(group_a_reviews)}) and '{rec_b_name}' ({len(group_b_reviews)})")
        print(f"  [Judge] Review A (Champion '{rec_a_name}'): {review_a_row['extractor_model_used']}/{review_a_row['synthesizer_model_used']} (Score: {review_a_row['overall_score']:.1f})")
        print(f"  [Judge] Review B (Champion '{rec_b_name}'): {review_b_row['extractor_model_used']}/{review_b_row['synthesizer_model_used']} (Score: {review_b_row['overall_score']:.1f})")
        # --- END OF NEW LOGIC ---
        
        # --- Robust Cache Lookup ---
        filename_clean = filename.strip()
        paper_key = next((k for k in paper_content_cache if k.startswith(filename_clean)), None)
        
        if not paper_key:
            print(f"  [Judge Error] Paper content not found in cache for key '{filename_clean}'. Skipping.")
            continue
        
        paper_content = paper_content_cache[paper_key]
        # --- End Cache Lookup ---

        # --- FIX 3: Pass 'reviews_dir' to the function ---
        verdict = run_judge(paper_content, review_a_row, review_b_row, reviews_dir)
        # --- END FIX 3 ---
        
        total_cost += verdict.get('cost', 0.0)
        verdict_data = {
            "paper_filename": filename,
            "title": review_a_row['title'],
            "judge_recommendation": verdict.get('judge_recommendation'),
            "judge_rationale": verdict.get('judge_rationale'),
            "winning_review": verdict.get('winning_review'),
            "winning_review_rationale": verdict.get('winning_review_rationale'),
            "model_a_recommendation": review_a_row['recommendation'],
            "model_b_recommendation": review_b_row['recommendation'],
            "model_a": f"{review_a_row['extractor_model_used']}/{review_a_row['synthesizer_model_used']}",
            # --- BONUS FIX: Corrected typo 'synthesaizer' -> 'synthesizer' ---
            "model_b": f"{review_b_row['extractor_model_used']}/{review_b_row['synthesizer_model_used']}",
            "judge_cost": verdict.get('cost', 0.0),
            "error": verdict.get('error')
        }
        
        # Update judge progress
        judge_progress[filename] = verdict_data
        save_judge_progress(judge_progress_file, judge_progress, judge_config_hash)
        
        final_verdicts.append(verdict_data)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(reports_dir, f"JUDGE_VERDICTS_report_{timestamp}.csv")
    
    if not final_verdicts:
        print("\n--- Adjudication Complete ---")
        print("No new verdicts to save (all conflicts may have been previously adjudicated).")
        return

    verdict_df = pd.DataFrame(final_verdicts)
    verdict_df.to_csv(output_path, index=False)
    
    print("\n--- Adjudication Complete ---")
    print(f"Total judging cost for this run: ${total_cost:.4f}")
    print(f"✅ Success! Saved Judge's verdicts to:")
    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the judge conflicts system with a specific directory")
    parser.add_argument("--run-dir", help="Directory for this run (e.g., 'run_dir2')")
    args = parser.parse_args()
    
    if not args.run_dir:
        print("No --run-dir specified, running in default 'outputs/' mode.")
    
    adjudicate_conflicts(args.run_dir)