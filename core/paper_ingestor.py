# core/paper_ingestor.py (enhanced version)
import os
import json
from markitdown import MarkItDown
from core.data_models import Paper, PaperMetadata
from litellm import token_counter
import logging
from typing import List, Optional, Dict, Tuple
import hashlib
import time
from datetime import datetime

import pymupdf4llm

def estimate_tokens(text: str) -> int:
    """Estimate token count using litellm's counter."""
    try:
        return token_counter(model="gpt-4", text=text)
    except Exception:
        # Fallback to a rough estimate
        return len(text) // 4

def extract_metadata(markdown_text: str) -> PaperMetadata:
    """
    Extract metadata from markdown text including title, abstract, authors.
    """
    lines = markdown_text.split('\n')

    # Extract title (first non-empty line, removing markdown #)
    title = "No Title Found"
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Remove leading # symbols
            title = stripped.lstrip('#').strip()
            break

    # Extract abstract
    # Look for abstract in the first part of the document
    abstract = None
    abstract_started = False
    abstract_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for "Abstract" heading or similar patterns
        if not abstract_started:
            if (stripped.lower().startswith('abstract') or
                stripped.lower() in ['summary', 'introduction'] or
                '## abstract' in stripped.lower() or
                '# abstract' in stripped.lower()):
                abstract_started = True
                # Skip the heading line itself, start from next
                continue

        # If we're in the abstract section
        elif abstract_started:
            # Stop at next major heading or end of reasonable abstract length
            if (stripped.startswith('#') and not stripped.startswith('###')) or \
               len(abstract_lines) > 20:  # Reasonable abstract length
                break

            # Skip empty lines at start
            if not abstract_lines and not stripped:
                continue

            abstract_lines.append(stripped)

    # If we found abstract content, join it
    if abstract_lines:
        abstract = '\n'.join(abstract_lines)
    else:
        # Try to find abstract as the first paragraph after title
        in_content = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not in_content and stripped:
                continue
            in_content = True
            # Skip title line
            if i == 0 or stripped.startswith('#'):
                continue
            # First paragraph could be abstract
            if stripped and len(stripped) > 50:  # Minimum reasonable length
                abstract = stripped
                break
            if abstract_lines:
                break

    return PaperMetadata(
        title=title if title else "No Title Found",
        abstract=abstract if abstract else "No abstract available."
    )

def get_file_hash(file_path: str) -> str:
    """Generate a hash of the file to detect changes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def print_ingestion_progress(current, total, filename="", stage=""):
    """Print ingestion progress with a simple progress bar."""
    if total > 0:
        percent = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Format: [Ingest] |████████-----| 60% (3/5) Parsing: paper.pdf
        print(f'\r[Ingest] |{bar}| {percent:.0f}% ({current}/{total}) {stage}: {filename[:30]}...', 
              end='', flush=True)
        
        if current == total:
            print()  # New line when complete

def ingest_paper(file_path: str, show_progress=True) -> Optional[Paper]:
    """
    Convert a single paper to normalized Markdown using MarkItDown.
    (This is the slow part we want to avoid)
    """
    filename = os.path.basename(file_path)
    
    if show_progress:
        print(f"[Ingestor] Parsing: {filename}")
    
    try:
        start_time = time.time()
        
        # Get file size for progress tracking
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        result = pymupdf4llm.to_markdown(file_path)
        
        if not result or len(result) < 100:
            logging.warning(f"Failed to extract sufficient text from {filename}")
            if show_progress:
                print(f"[Ingestor] ⚠️  Warning: Insufficient text extracted from {filename}")
            return None
        
        parse_time = time.time() - start_time
        
        token_count = estimate_tokens(result)
        metadata = extract_metadata(result)
        file_hash = get_file_hash(file_path)
        
        if show_progress:
            print(f"[Ingestor] ✅ Parsed {filename} ({file_size_mb:.1f}MB, {token_count:,} tokens, {parse_time:.1f}s)")
        
        return Paper(
            filename=filename,
            original_path=file_path,
            content_markdown=result,
            token_count=token_count,
            metadata=metadata,
            file_hash=file_hash  # Store the file hash
        )

    except Exception as e:
        logging.error(f"Error ingesting {file_path}: {e}")
        if show_progress:
            print(f"[Ingestor] ❌ Error parsing {filename}: {str(e)[:50]}...")
        return None

def load_ingestion_cache(cache_file: str = "outputs/ingestion_cache.json") -> Dict[str, Paper]:
    """Loads the ingestion cache from disk if it exists."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if not os.path.exists(cache_file):
        return {}
        
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
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

def save_ingestion_cache(cache: Dict[str, Paper], cache_file: str = "outputs/ingestion_cache.json"):
    """Saves the ingestion cache to disk."""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            # Convert list of Paper objects to a JSON-serializable list
            cache_data = [paper.model_dump() for paper in cache.values()]
            json.dump(cache_data, f, indent=2, default=str) # default=str for datetimes
        print(f"[Cache] Saved {len(cache)} papers to cache.")
    except Exception as e:
        print(f"[Cache] Error saving cache: {e}")

def ingest_directory(directory_path: str, cache: Dict[str, Paper], cache_file: str = "outputs/ingestion_cache.json") -> Tuple[List[Paper], bool]:
    """
    Recursively scan a directory and ingest supported papers,
    using the cache to avoid re-parsing.
    """
    supported_extensions = ('.pdf', '.docx', '.doc', '.md', '.txt', '.pptx')
    papers_to_process = []
    found_in_cache = 0
    newly_ingested = 0
    cache_updated = False
    
    print(f"[Ingest] Scanning directory: {directory_path}")
    
    # --- Step 1: Find all current paper files on disk ---
    current_files_on_disk = {} # {filename: filepath}
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(supported_extensions) and not file.startswith('~'):
                file_path = os.path.join(root, file)
                current_files_on_disk[file] = file_path
    
    total_files = len(current_files_on_disk)
    
    if total_files == 0:
        print("[Ingest] ⚠️  No supported files found in directory")
        return [], False
    
    print(f"[Ingest] Found {total_files} files to process")
    
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
    processed_count = 0
    
    for filename, file_path in current_files_on_disk.items():
        processed_count += 1
        
        # Show overall progress
        print_ingestion_progress(processed_count, total_files, filename, "Checking")
        
        if filename in cache:
            # Check if the file has changed by comparing hashes
            current_hash = get_file_hash(file_path)
            cached_paper = cache[filename]
            
            # If the file has a hash and it matches, use the cached version
            if hasattr(cached_paper, 'file_hash') and cached_paper.file_hash == current_hash:
                papers_to_process.append(cached_paper)
                found_in_cache += 1
                print(f"\r[Ingest] |{'█' * 30}| 100% ({processed_count}/{total_files}) Using cache: {filename[:30]}...{' ' * 20}", flush=True)
            else:
                # File has changed or doesn't have a hash, re-ingest
                print(f"\n[Ingest] File '{filename}' has changed, re-ingesting...")
                paper = ingest_paper(file_path, show_progress=True)
                if paper:
                    papers_to_process.append(paper)
                    cache[filename] = paper
                    newly_ingested += 1
                    cache_updated = True
        else:
            # Ingest from disk (slow)
            print()  # New line for new ingestion
            paper = ingest_paper(file_path, show_progress=True)
            if paper:
                papers_to_process.append(paper)
                cache[filename] = paper # Add to cache
                newly_ingested += 1
                cache_updated = True
    
    # Final summary
    print(f"\n[Ingest] Ingestion complete!")
    print(f"[Ingest] 📊 Summary:")
    print(f"[Ingest]   • Total files found: {total_files}")
    print(f"[Ingest]   • From cache: {found_in_cache}")
    print(f"[Ingest]   • Newly parsed: {newly_ingested}")
    print(f"[Ingest]   • Total to process: {len(papers_to_process)}")
    
    # Calculate total tokens and file sizes
    total_tokens = sum(p.token_count for p in papers_to_process)
    total_size_mb = sum(os.path.getsize(p.original_path) for p in papers_to_process) / (1024 * 1024)
    
    if total_tokens > 0:
        print(f"[Ingest] 📈 Stats:")
        print(f"[Ingest]   • Total tokens: {total_tokens:,}")
        print(f"[Ingest]   • Total size: {total_size_mb:.1f}MB")
        print(f"[Ingest]   • Avg tokens/paper: {total_tokens // len(papers_to_process):,}")
    
    return papers_to_process, cache_updated