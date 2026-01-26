import os
import json
from markitdown import MarkItDown
from core.data_models import Paper, PaperMetadata
from litellm import token_counter
import logging
from typing import List, Optional, Dict, Tuple





# --- NEW: Define the cache file path ---
CACHE_FILE = "outputs/ingestion_cache.json"

def estimate_tokens(text: str) -> int:
    """Estimate token count using litellm's counter."""
    try:
        return token_counter(model="gpt-4", text=text)
    except Exception:
        # Fallback to a rough estimate
        return len(text) // 4

def extract_metadata(markdown_text: str) -> PaperMetadata:
    """
    Rudimentary metadata extraction.
    This should be replaced by a more robust LLM-based call.
    """
    title = markdown_text.split('\n')[0].lstrip('# ').strip()
    return PaperMetadata(
        title=title if title else "No Title Found",
        abstract="Placeholder: Abstract extraction not fully implemented."
    )

def ingest_paper(file_path: str) -> Optional[Paper]:
    """
    Convert a single paper to normalized Markdown using MarkItDown.
    (This is the slow part we want to avoid)
    """
    print(f"[Ingestor] Parsing (Slow): {os.path.basename(file_path)}")
    try:
        md = MarkItDown()

        result = md.convert(file_path)
       
        
        if not result.text_content or len(result.text_content) < 100:
            logging.warning(f"Failed to extract sufficient text from {file_path}")
            return None
            
        token_count = estimate_tokens(result.text_content)
        metadata = extract_metadata(result.text_content)
        
        return Paper(
            filename=os.path.basename(file_path),
            original_path=file_path,
            content_markdown=result.text_content,
            token_count=token_count,
            metadata=metadata
        )

    except Exception as e:
        logging.error(f"Error ingesting {file_path}: {e}")
        return None

# --- NEW: Function to load the cache ---
def load_ingestion_cache() -> Dict[str, Paper]:
    """Loads the ingestion cache from disk if it exists."""
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(CACHE_FILE):
        return {}
        
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            # Use filename as the key for easy lookup
            cache = {
                data['filename']: Paper.model_validate(data)
                for data in cache_data
            }
            print(f"[Cache] Loaded {len(cache)} papers from cache.")
            return cache
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[Cache] Could not load cache, will re-ingest all. Error: {e}")
        return {}

# --- NEW: Function to save the cache ---
def save_ingestion_cache(cache: Dict[str, Paper]):
    """Saves the ingestion cache to disk."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            # Convert list of Paper objects to a JSON-serializable list
            cache_data = [paper.model_dump() for paper in cache.values()]
            json.dump(cache_data, f, indent=2, default=str) # default=str for datetimes
        print(f"[Cache] Saved {len(cache)} papers to cache.")
    except Exception as e:
        print(f"[Cache] Error saving cache: {e}")

# --- MODIFIED: This function is now cache-aware ---
def ingest_directory(directory_path: str, cache: Dict[str, Paper]) -> Tuple[List[Paper], bool]:
    """
    Recursively scan a directory and ingest supported papers,
    using the cache to avoid re-parsing.
    """
    supported_extensions = ('.pdf', '.docx', '.doc', '.md', '.txt', '.pptx')
    papers_to_process = []
    found_in_cache = 0
    newly_ingested = 0
    cache_updated = False

    # --- Step 1: Find all current paper files on disk ---
    current_files_on_disk = {} # {filename: filepath}
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(supported_extensions) and not file.startswith('~'):
                file_path = os.path.join(root, file)
                current_files_on_disk[file] = file_path
    
    # --- Step 2: Prune cache ---
    # Remove any papers from the cache that are no longer on disk
    cached_filenames = list(cache.keys())
    for filename in cached_filenames:
        if filename not in current_files_on_disk:
            del cache[filename]
            cache_updated = True
            print(f"[Cache] Pruned '{filename}' from cache (file deleted).")

    # --- Step 3: Ingest files ---
    # Check cache for existing files, ingest new ones
    for filename, file_path in current_files_on_disk.items():
        if filename in cache:
            # Load from cache (fast)
            papers_to_process.append(cache[filename])
            found_in_cache += 1
        else:
            # Ingest from disk (slow)
            paper = ingest_paper(file_path)
            if paper:
                papers_to_process.append(paper)
                cache[filename] = paper # Add to cache
                newly_ingested += 1
                cache_updated = True
    
    print(f"[Ingestor] Ingestion complete: {found_in_cache} from cache, {newly_ingested} newly parsed.")
    return papers_to_process, cache_updated