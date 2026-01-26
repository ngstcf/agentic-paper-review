import time
from core.config_loader import Config
# --- MODIFIED IMPORTS ---
from core.paper_ingestor import (
    load_ingestion_cache, 
    save_ingestion_cache, 
    ingest_directory
)
# --- (End of modified imports) ---
from agents.agent_extractor import process_paper_extractions
from agents.agent_synthesizer import synthesize_review
from utilities.output_generator import save_review_markdown, save_consolidated_csv
from utilities.helpers import setup_logging

def main():
    start_time = time.time()
    setup_logging()
    
    print("--- Starting Academic Review System ---")
    
    # 1. Load Configuration
    print("[Main] Loading configuration...")
    config = Config(config_path="config")
    
    # --- MODIFIED INGESTION LOGIC ---
    print("[Main] Loading ingestion cache...")
    ingestion_cache = load_ingestion_cache()

    print("[Main] Ingesting papers from 'papers/' directory...")
    # Pass the cache to the ingestor
    papers, cache_was_updated = ingest_directory("papers", ingestion_cache)
    
    if cache_was_updated:
        save_ingestion_cache(ingestion_cache)
    # --- (End of modified logic) ---

    if not papers:
        print("[Main] No papers found or ingested. Exiting.")
        return
        
    print(f"[Main] Loaded {len(papers)} papers for processing.")
    
    final_reviews = []
    total_batch_cost = 0.0
    
    # 3. Process Each Paper
    for i, paper in enumerate(papers, 1):
        print(f"\n--- Processing Paper {i}/{len(papers)}: {paper.filename} ---")
        paper_start_time = time.time()
        
        # 4. Agent 1: Extract Evidence (Parallel)
        extractions = process_paper_extractions(paper, config)
        if not extractions:
            print(f"[Main] Failed to get any extractions for {paper.filename}. Skipping.")
            continue
            
        # 5. Agent 2: Synthesize Review
        review = synthesize_review(paper, extractions, config)
        if not review:
            print(f"[Main] Failed to synthesize review for {paper.filename}. Skipping.")
            continue
            
        # 6. Save Individual Output
        save_review_markdown(review, paper, config)
        
        final_reviews.append(review)
        total_batch_cost += review.total_cost
        paper_end_time = time.time()
        
        print(f"[Main] Finished processing {paper.filename} in {paper_end_time - paper_start_time:.2f}s")
        print(f"[Main] Cost for this paper: ${review.total_cost:.4f}")

    # 7. Save Consolidated Report
    if final_reviews:
        print("\n--- Batch Complete ---")
        save_consolidated_csv(final_reviews)
    
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    print(f"Total batch cost: ${total_batch_cost:.4f}")
    print("--- Academic Review System Finished ---")

if __name__ == "__main__":
    main()