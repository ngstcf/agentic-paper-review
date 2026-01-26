import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Ingestion Models ---

class PaperMetadata(BaseModel):
    title: Optional[str] = "No Title Found"
    authors: List[str] = []
    abstract: Optional[str] = "No Abstract Found"
    year: Optional[int] = None
    keywords: List[str] = []

class Paper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    original_path: str
    content_markdown: str
    token_count: int
    metadata: PaperMetadata
    file_hash: Optional[str] = None  # Add file hash to detect changes
    status: str = "ready"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# --- Agent 1 (Extractor) Models ---

class EvidenceItem(BaseModel):
    quote: str
    page_reference: str
    relevance: str

class Extraction(BaseModel):
    paper_id: str
    criterion_id: str
    score: int
    score_justification: str
    evidence: List[EvidenceItem]
    strengths: List[str]
    weaknesses: List[str]
    confidence: float
    model_used: str  # This stores the extractor model
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    cost: float = 0.0

# --- Agent 2 (Synthesizer) Models ---

class WeightedBreakdown(BaseModel):
    score: int
    weight: int
    weighted_score: float

class DetailedAssessment(BaseModel):
    major_strengths: List[str]
    major_concerns: List[str]
    minor_issues: List[str]

class Review(BaseModel):
    paper_id: str
    paper_title: str
    
    # --- NEW FIELD ---
    paper_filename: str
    # --- (End of new field) ---
    
    overall_score: float
    weighted_breakdown: Dict[str, WeightedBreakdown]
    recommendation: str
    recommendation_rationale: str
    executive_summary: str
    detailed_assessment: DetailedAssessment
    criterion_narrative: Dict[str, str]
    revision_suggestions: List[str]
    decision_confidence: float
    flags: List[str] = []
    synthesis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    extractor_model_used: str
    synthesizer_model_used: str
    total_cost: float = 0.0