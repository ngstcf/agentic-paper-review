import os
from markitdown import MarkItDown
from core.data_models import Paper, PaperMetadata
from litellm import token_counter
import logging
from typing import List, Optional  # <--- THIS LINE IS ADDED

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
    # This is a placeholder. A real implementation would use regex or a small
    # LLM call to find abstract, authors, etc.
    return PaperMetadata(
        title=title if title else "No Title Found",
        abstract="Placeholder: Abstract extraction not fully implemented."
    )

def ingest_paper(file_path: str) -> Optional[Paper]:
    """
    Convert a single paper to normalized Markdown using MarkItDown.
    """
    print(f"[Ingestor] Processing: {os.path.basename(file_path)}")
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

def ingest_directory(directory_path: str) -> List[Paper]:
    """
    Recursively scan a directory and ingest all supported papers.
    """
    supported_extensions = ('.pdf', '.docx', '.doc', '.md', '.txt', '.pptx')
    papers = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(supported_extensions) and not file.startswith('~'):
                file_path = os.path.join(root, file)
                paper = ingest_paper(file_path)
                if paper:
                    papers.append(paper)
                    
    return papers