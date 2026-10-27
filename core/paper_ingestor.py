# core/paper_ingestor.py (enhanced version)
import os
import re
import html
import json
import threading
from core.data_models import Paper, PaperMetadata
from litellm import token_counter
import logging
from typing import Any, List, Optional, Dict, Tuple
import hashlib
import time
from datetime import datetime

# Extracted text shorter than this (placeholders excluded) counts as a failed parse.
MIN_TEXT_CHARS = 100

# Docling takes several seconds to load its models, so converters are built once per process
# and shared. Conversions are serialized: web_app.py runs pipelines in background threads
# and Docling's thread-safety has not been verified.
_converters: Dict[bool, Any] = {}
_convert_lock = threading.Lock()

def estimate_tokens(text: str) -> int:
    """Estimate token count using litellm's counter."""
    try:
        return token_counter(model="gpt-4", text=text)
    except Exception:
        # Fallback to a rough estimate
        return len(text) // 4

# Title/abstract extraction works on Docling's Markdown: elements are separated by blank lines,
# headings start with '#', figures show up as <!-- image --> placeholders, and one paragraph can
# arrive split into one-line fragments that end mid-sentence (often at a hyphen).
MAX_TITLE_CHARS = 300
MAX_ABSTRACT_CHARS = 3000
MIN_ABSTRACT_CHARS = 60

_HEADING_RE = re.compile(r'^#{1,6}\s+(.*)$')
_IMAGE_RE = re.compile(r'^(<!--\s*image\s*-->|!\[[^\]]*\]\([^)]*\))$', re.IGNORECASE)
# Opens the body of a document: "1 Introduction", "2.1 Method", "II. Methods", "Introduction"
_BODY_HEADING_RE = re.compile(
    r'^(?:\d{1,2}(?:\.\d+)*[.)]?\s+\S|(?-i:[IVXLC]+)[.)]\s+\S|introduction\b)', re.IGNORECASE)
# "Abstract: text", "Abstract—text" (IEEE), "**Abstract.** text", "ABSTRACT text"
_INLINE_ABSTRACT_RE = re.compile(
    r'^(?:[*_\s]*abstract[*_\s]*(?:[:.–—]+|-(?=[A-Z]))[*_\s]*|'
    r'(?:\*\*|__)abstract(?:\*\*|__)\s*|(?-i:ABSTRACT)\s+)(\S.*)$',
    re.IGNORECASE | re.DOTALL)
# Section labels, compared with spaces and punctuation removed (so "A B S T R A C T" and "Abstract:" both match)
_LABELS = {'abstract': 'abstract', 'summary': 'summary', 'executivesummary': 'summary'}
# Generic template headings that precede the real title, compared with spaces and punctuation removed
_BANNERS = frozenset(
    'articletype researcharticle originalarticle originalresearch originalpaper researchpaper reviewarticle '
    'shortcommunication briefreport fullpaper regularpaper technicalreport workingpaper preprint manuscript '
    'acceptedmanuscript underreview positionpaper casestudy systematicreview article letter review perspective '
    'commentary editorial survey tutorial draft contentslistsavailableatsciencedirect'.split())
# Headings that are never a title, compared with spaces, digits and punctuation removed
_NON_TITLE_HEADINGS = frozenset(
    'keywords keyword keyterms indexterms ccsconcepts acmreferenceformat categoriesandsubjectdescriptors '
    'generalterms additionalkeywordsandphrases articleinfo articlehistory contents tableofcontents references '
    'acknowledgments acknowledgements acknowledgment authors authorinformation correspondence funding'.split())
# A short submission stamp that carries a date or version: "Preprint version, February 16, 2024"
_STAMP_RE = re.compile(
    r'^(?:preprint|submitted|accepted|manuscript|arxiv|under review|working paper)\b.*?'
    r'(?:\b\d{4}\b|\bversion\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)', re.IGNORECASE)
# Words that mark an author/affiliation line
_AFFILIATION_RE = re.compile(
    r'universit|institut|college|school|department|faculty|laborator|academy|hospital|cent(?:er|re)\b|'
    r'\binc\b|\bltd\b|gmbh|@', re.IGNORECASE)
# Blocks that end an abstract: keyword lists and figure/table captions
_STOP_RE = re.compile(
    r'^[*_\s]*(?:key\s*-?words?\b|index terms?\b|ccs concepts?\b|additional key words\b|'
    r'categories and subject|general terms\b|subject headings?\b|acm reference format\b|'
    r'jel\b|mathematics subject|msc\b|pacs\b|word count\b|(?:figure|fig\.|table|algorithm|listing)\s*\d)',
    re.IGNORECASE)
# Boilerplate that can follow an abstract (only checked after the first paragraph, so a real abstract can't be cut)
_BOILERPLATE_RE = re.compile(
    r'^[*_\s]*(?:corresponding author|correspondence|disclaimer|acknowledg|funding\b|received\b|submitted\b|'
    r'accepted\b|published\b|copyright\b|©|preprint\b|under review|conflicts? of interest|declarations?\b|'
    r'the views expressed)', re.IGNORECASE)
# Keywords or a corresponding-author note merged into the end of the abstract paragraph
_INLINE_END_RE = re.compile(
    r'\s(?:key\s*-?words?|index terms?|ccs concepts?)\s*[:—–\-]|'
    r'\s[∗†‡§¶*]?\s*corresponding author', re.IGNORECASE)
# A title cannot end on one of these words, so the next block continues it
_DANGLING = frozenset(
    'and or of for with in on to the a an via by from at as using into between through towards toward vs versus '
    'than under over across about against among within without'.split())

def _ends_with_split_hyphen(text: str) -> bool:
    """True when the text ends in a hyphen attached to a letter or digit, i.e. a word split at a line break."""
    return len(text) > 1 and text[-1] == '-' and text[-2].isalnum()

def _join(first: str, second: str) -> str:
    """Join two fragments of one paragraph; a hyphen split at a line break is kept, without a space."""
    if _ends_with_split_hyphen(first) and second[:1].islower():
        return first + second
    return first + ' ' + second

def _join_lines(lines: List[str]) -> str:
    """Join the lines of one paragraph like _join does, in time linear in the paragraph's length."""
    groups: List[List[str]] = []
    for line in lines:
        if groups and _ends_with_split_hyphen(groups[-1][-1]) and line[:1].islower():
            groups[-1].append(line)
        else:
            groups.append([line])
    return ' '.join(''.join(group) for group in groups)

def _strip_comments(text: str) -> str:
    """Remove <!-- ... --> comments in one linear scan; a comment that never closes is left as it is."""
    parts, pos = [], 0
    while True:
        start = text.find('<!--', pos)
        end = text.find('-->', start + 4) if start != -1 else -1
        if end == -1:
            break
        parts.append(text[pos:start])
        parts.append(' ')
        pos = end + 3
    parts.append(text[pos:])
    return ''.join(parts)

def _clean(text: str) -> str:
    """Strip HTML entities, comments, links and bold markers (the patterns are bounded to stay linear)."""
    text = _strip_comments(html.unescape(text))
    text = re.sub(r'\[([^\]]{0,300})\]\([^)]{0,500}\)', r'\1', text)
    text = re.sub(r'(\*\*|__)(.{1,300}?)\1', r'\2', text)
    return ' '.join(text.split())

def _truncate(text: str, limit: int) -> str:
    """Cut text to at most `limit` characters, at a sentence end when one is close enough."""
    if len(text) <= limit:
        return text
    cut = text.rfind('. ', 0, limit)
    return text[:cut + 1] if cut > limit * 0.6 else text[:limit].rstrip()

def _parse_blocks(markdown_text: str) -> List[Tuple[str, str]]:
    """Split Markdown into ('heading' | 'text' | 'image', text) blocks, joining hard-wrapped lines."""
    blocks: List[Tuple[str, str]] = []
    lines: List[str] = []  # the lines of the paragraph being read

    def flush():
        if lines:
            blocks.append(('text', _join_lines(lines)))
            lines.clear()

    for raw_line in markdown_text.split('\n'):
        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        if not line:
            flush()
        elif _IMAGE_RE.match(line):
            flush()
            blocks.append(('image', ''))
        elif line.startswith('<!--') and not _strip_comments(line).strip():
            flush()  # other placeholders (e.g. undecoded formulas) carry no text
        elif heading:
            flush()
            blocks.append(('heading', re.sub(r'\s+#+$', '', heading.group(1)).strip()))
        else:
            lines.append(line)
    flush()
    return blocks

def _label_of(text: str) -> Optional[str]:
    """'abstract' or 'summary' when the text is just such a label (numbered, letter-spaced or with a colon)."""
    core = re.sub(r'^\d+[.)]?\s+', '', text.strip(' *_#'))
    return _LABELS.get(re.sub(r'[\s:.\-–—]+', '', core).lower()) if len(core) <= 40 else None

def _is_section_label(text: str) -> bool:
    return _label_of(text) is not None

def _is_non_title_heading(text: str) -> bool:
    return _is_section_label(text) or re.sub(r'[\W\d_]+', '', text).lower() in _NON_TITLE_HEADINGS

def _is_prose(text: str, min_words: int = 20) -> bool:
    """Running text, as opposed to author lists, affiliations and footnotes (mostly capitalised words)."""
    words = text.split()
    if len(words) < min_words or not re.search(r'[.!?]["\')\]]*(?:\s|$)', text):
        return False
    return sum(word[0].islower() for word in words) / len(words) >= 0.55

def _is_stop(text: str) -> bool:
    """Blocks that end an abstract: keyword lists, captions, tables, and short contact/e-mail lines."""
    return bool(_STOP_RE.match(text)) or text.startswith('|') or ('@' in text and len(text.split()) < 30)

def _is_footnote(text: str) -> bool:
    """A footnote line: starts with a footnote symbol, or with a lone asterisk (a pair would be Markdown emphasis)."""
    return text[:1] in '∗†‡§¶' or (text[:1] == '*' and text.count('*') == 1)

def _ends_sentence(text: str) -> bool:
    return bool(re.search(r'[.!?]["\'”’)\]]*$', text.rstrip()))

def _is_banner(text: str) -> bool:
    """A generic template heading ("Research Article") or a short dated preprint/submission stamp."""
    return re.sub(r'[\W_]+', '', text).lower() in _BANNERS or (len(text) <= 60 and bool(_STAMP_RE.match(text)))

def _is_subtitle(title: str, head: List[Tuple[str, str]], end: int) -> bool:
    """
    True when head[end] continues a short title that has no colon: a heading that is not a label,
    banner, date or author line, followed by an author/affiliation block.
    """
    if end + 1 >= len(head) or head[end][0] != 'heading' or len(title.split()) > 8:
        return False
    candidate = head[end][1]
    if not 3 <= len(candidate.split()) <= 25 or _AFFILIATION_RE.search(candidate) or re.search(r'\d{4}', candidate):
        return False
    if _is_section_label(candidate) or _BODY_HEADING_RE.match(candidate) or _is_banner(candidate):
        return False
    return head[end + 1][0] in ('heading', 'text') and bool(_AFFILIATION_RE.search(head[end + 1][1]))

def _clean_title(title: str) -> str:
    """Clean a title and drop a trailing footnote symbol ("... in Japan?†")."""
    title = _clean(title[:MAX_TITLE_CHARS])
    return re.sub(r'(?:\s+\*|[∗†‡§¶])+$', '', title).rstrip()

def _continues_title(title: str, head: List[Tuple[str, str]], end: int) -> bool:
    """
    True when head[end] continues the title: a block after a title that ends mid-phrase (colon, semicolon,
    comma, dash or a dangling word such as "and"), or a subtitle heading between a short title and the authors.
    """
    if end >= len(head) or head[end][0] not in ('heading', 'text'):
        return False
    candidate = head[end][1]
    if len(candidate) > 200 or '@' in candidate or _is_section_label(candidate) or _is_banner(candidate):
        return False
    words = title.split()
    if title.endswith((':', ';', ',', '-', '–', '—')):
        return True
    if words and words[-1].lower().strip('.,;:') in _DANGLING:
        return not _AFFILIATION_RE.search(candidate) and not _BODY_HEADING_RE.match(candidate)
    return _is_subtitle(title, head, end)

def _title_from(head: List[Tuple[str, str]], i: int) -> Tuple[str, int]:
    """The title heading at head[i], with any subtitle blocks that continue it; returns (title, next index)."""
    title, end = head[i][1], i + 1
    while end < min(i + 3, len(head)) and _continues_title(title, head, end):
        title, end = title + ' ' + head[end][1], end + 1
    return _clean_title(title), end

def _find_title(blocks: List[Tuple[str, str]]) -> Tuple[Optional[str], int]:
    """
    Return (title, index of the first block after it). Docling marks the title as a heading, so it is the
    first heading that is not a banner, label or body heading. If the first heading is one of those, the
    title is the text before it; failing that, the first later heading (reading order can put the title
    after the abstract and keywords). Without any headings, it is the first line of text.
    """
    head = blocks[:40]
    headings = [i for i, (kind, text) in enumerate(head) if kind == 'heading' and not _is_banner(text)]
    for rank, i in enumerate(headings):
        text = head[i][1]
        body = bool(_BODY_HEADING_RE.match(text)) and not _is_section_label(text)
        if body or _is_non_title_heading(text):
            if rank == 0:
                earlier = next((j for j in range(i) if head[j][0] == 'text' and not _is_banner(head[j][1])), None)
                if earlier is not None:
                    return _clean_title(head[earlier][1]), earlier + 1
            if body:
                break  # reached the body without seeing a title heading
            continue
        return _title_from(head, i)
    if not headings:
        for i, (kind, text) in enumerate(head):
            if kind == 'text' and not _is_banner(text):
                return _clean_title(text), i + 1
    return None, 0

def _front_matter_end(blocks: List[Tuple[str, str]], start: int) -> int:
    """Index of the first body heading after `start`, or about 20k characters in."""
    chars = 0
    for i in range(start, len(blocks)):
        kind, text = blocks[i]
        if kind == 'heading' and _BODY_HEADING_RE.match(text) and not _is_section_label(text):
            return i
        chars += len(text)
        if chars > 20000:
            return i
    return len(blocks)

def _cut_inline_end(text: str) -> str:
    """Drop keywords or a corresponding-author note that were merged into the end of the paragraph."""
    match = _INLINE_END_RE.search(text, 100)
    return text[:match.start()].rstrip() if match else text

def _index_after(blocks: List[Tuple[str, str]], start: int, chars: int) -> int:
    """Index of the first block that lies more than `chars` characters past `start`."""
    total = 0
    for i in range(start, len(blocks)):
        total += len(blocks[i][1])
        if total > chars:
            return i
    return len(blocks)

def _final_paragraph_follows(blocks: List[Tuple[str, str]], i: int) -> bool:
    """True when blocks[i] is a paragraph of running text that is directly followed by a heading."""
    return (i + 1 < len(blocks) and blocks[i][0] == 'text' and _is_prose(blocks[i][1], 12)
            and not _is_stop(blocks[i][1]) and blocks[i + 1][0] == 'heading')

def _collect_abstract(blocks: List[Tuple[str, str]], start: int, first: str, strict: bool) -> str:
    """
    Gather an abstract's paragraphs from blocks[start:], after `first` (text already known to belong to it).
    A block that continues an unfinished sentence is joined to the previous paragraph. With `strict`
    (no explicit label to anchor on) only such continuations are added, so that a missing Introduction
    heading cannot pull the first body paragraph into the abstract.
    """
    parts = [first] if first else []
    size = len(first)
    i = start
    while i < len(blocks):
        kind, text = blocks[i]
        if kind != 'text' or _is_stop(text) or size + len(text) > MAX_ABSTRACT_CHARS:
            break
        if parts and (_is_footnote(text) or _BOILERPLATE_RE.match(text)):
            # In a two-column layout an author footnote can sit between the paragraphs of the abstract, so
            # skip it when the abstract's last paragraph follows (one block of running text, then a heading)
            if _is_footnote(text) and _final_paragraph_follows(blocks, i + 1):
                i += 1
                continue
            break
        unfinished = bool(parts) and not _ends_sentence(parts[-1])
        # A new paragraph must be real running text: stray footnote marks, dates and "Preprint" lines end the abstract
        if parts and not unfinished and (strict or len(text.split()) < 6):
            break
        if unfinished:
            parts[-1] = _join(parts[-1], text)
        else:
            parts.append(text)
        size += len(text)
        i += 1
    abstract = '\n\n'.join(_clean(_truncate(_cut_inline_end(part), MAX_ABSTRACT_CHARS)) for part in parts)
    # A footnote mark can trail the last sentence ("... availability. 1")
    return re.sub(r'(?<=[.!?])\s*(?:\d{1,2}|[∗†‡§¶*]+)$', '', abstract)

def _find_abstract(blocks: List[Tuple[str, str]], start: int) -> Optional[str]:
    """
    Find the abstract. Attempts, in order: an "Abstract" label (a heading, or inline as "Abstract: ..."),
    a "Summary" heading, the first running-text paragraph of the front matter, and finally the first
    running-text paragraph anywhere (documents without an abstract).
    """
    end = _front_matter_end(blocks, start)
    # Reading order can put an "Abstract" heading after the first body headings, so for that label
    # headings (not inline text) are also accepted up to 40k characters past the front matter
    late = _index_after(blocks, end, 40000)
    for label_name, limit in (('abstract', late), ('summary', end)):
        for i in range(limit):  # from the very top: the abstract can come before the title
            kind, text = blocks[i]
            inline = None
            if _label_of(text) == label_name:
                inline = ''
            elif label_name == 'abstract' and i < end:
                match = _INLINE_ABSTRACT_RE.match(text)
                if match:
                    inline = match.group(1)
            if inline is None or (i >= end and kind != 'heading'):
                continue
            abstract = _collect_abstract(blocks, i + 1, inline, strict=False)
            if len(abstract) >= MIN_ABSTRACT_CHARS:
                return abstract
    for low, high in ((start, end), (end, len(blocks))):
        for i in range(low, high):
            kind, text = blocks[i]
            if kind == 'text' and _is_prose(text) and not _is_stop(text):
                return _collect_abstract(blocks, i + 1, text, strict=True)
    return None

def extract_metadata(markdown_text: str) -> PaperMetadata:
    """
    Extract the title and abstract from parsed Markdown.
    Best-effort: a missing title or abstract keeps its "not found" placeholder, and a failure here
    never costs the paper its ingestion.
    """
    title = abstract = None
    try:
        blocks = _parse_blocks(markdown_text)
        title, start = _find_title(blocks)
        abstract = _find_abstract(blocks, start)
    except Exception as e:
        logging.warning(f"Could not extract title/abstract: {e}")

    return PaperMetadata(
        title=title or "No Title Found",
        abstract=abstract or "No abstract available."
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

def _get_converter(ocr: bool) -> Any:
    """Return the shared Docling converter for the given OCR setting, creating it on first use."""
    if ocr not in _converters:
        # Imported here because Docling pulls in torch, which scripts that only read the cache don't need
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = ocr
        _converters[ocr] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        )
    return _converters[ocr]

def _text_length(markdown: str) -> int:
    """Length of the extracted text, ignoring Docling's <!-- image --> style placeholders."""
    return len(re.sub(r'<!--.*?-->', '', markdown).strip())

def convert_to_markdown(file_path: str) -> str:
    """
    Convert a file to Markdown with Docling.
    OCR is off on the first pass because it roughly doubles parse time on PDFs that already
    have a text layer; a PDF that yields almost no text is retried once with OCR (scans).
    """
    with _convert_lock:
        markdown = _get_converter(ocr=False).convert(file_path).document.export_to_markdown()
        if _text_length(markdown) < MIN_TEXT_CHARS and file_path.lower().endswith('.pdf'):
            markdown = _get_converter(ocr=True).convert(file_path).document.export_to_markdown()
    return markdown

def ingest_paper(file_path: str, show_progress=True) -> Optional[Paper]:
    """
    Convert a single paper to normalized Markdown using Docling.
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
        
        result = convert_to_markdown(file_path)

        if _text_length(result) < MIN_TEXT_CHARS:
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