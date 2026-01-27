"""
World Bank Documents & Reports Searcher Module

Integrates with World Bank API to search for and retrieve development-related
publications, reports, and working papers.

World Bank API is free and open - no API key required.
Documentation: https://documents.worldbank.org/en/publication/documents-reports/api
"""

import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from core.data_models import RelatedPaperMetadata


@dataclass
class WorldBankSearchConfig:
    """Configuration for World Bank search."""
    base_url: str = "https://search.worldbank.org/api/v3/wds"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class WorldBankSearcher:
    """
    Search World Bank Documents & Reports for development-related publications.

    Features:
    - Keyword-based search
    - Date range filtering
    - Field-specific queries
    - Abstract and PDF retrieval
    - Rate limiting and retry logic

    Note: World Bank does NOT provide citation counts.
    Results are sorted by relevance or date, not by citations.
    """

    def __init__(self, config: Optional[WorldBankSearchConfig] = None):
        """
        Initialize the World Bank searcher.

        Args:
            config: Search configuration
        """
        self.config = config or WorldBankSearchConfig()
        self._session = requests.Session()
        self._api_calls_today = 0

    def search_by_keywords(
        self,
        keywords: List[str],
        limit: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        document_types: Optional[List[str]] = None
    ) -> List[RelatedPaperMetadata]:
        """
        Search for World Bank documents by keywords.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)
            document_types: Optional list of document types to filter (e.g., "Working Paper")

        Returns:
            List of related paper metadata
        """
        query = " ".join(keywords)
        return self._search_papers(
            query=query,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            document_types=document_types
        )

    def _search_papers(
        self,
        query: str,
        limit: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        document_types: Optional[List[str]] = None
    ) -> List[RelatedPaperMetadata]:
        """
        Internal search method.

        Args:
            query: Search query string
            limit: Maximum number of results
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            document_types: Optional document type filters

        Returns:
            List of paper metadata
        """
        # Fields to retrieve from World Bank API
        # Available fields: https://documents.worldbank.org/en/publication/documents-reports/api
        fields = [
            "display_title",    # Title
            "authr",            # Authors
            "abstracts",        # Abstract
            "docdt",            # Document date
            "docty",            # Document type
            "url",              # URL
            "pdfurl",           # PDF URL
            "count",            # Country/region
            "keywd",            # Keywords
            "repnb",            # Report number
        ]

        params = {
            "format": "json",
            "qterm": query,
            "fl": ",".join(fields),
            "rows": limit,
            "os": 0,  # Offset (start from beginning)
        }

        # Add date range if specified
        if start_date:
            params["strdate"] = start_date
        if end_date:
            params["enddate"] = end_date

        # Add document type filter if specified
        if document_types:
            # World Bank API uses docty_exact for exact matching
            # Multiple values can be separated by "^"
            params["docty_exact"] = "^".join(document_types)

        data = self._make_request(self.config.base_url, params=params)
        if not data:
            return []

        papers = []
        for item in data.get("docs", {}).get("doc", []):
            paper = self._parse_paper_metadata(item)
            if paper:
                papers.append(paper)

        self._api_calls_today += 1
        return papers

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic.
        """
        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"[World Bank] Request failed: {e}")

            if retry_count < self.config.max_retries:
                wait_time = self.config.retry_delay * (2 ** retry_count)
                print(f"[World Bank] Retrying in {wait_time}s...")
                import time
                time.sleep(wait_time)
                return self._make_request(url, params, retry_count + 1)

            return None

    def _parse_paper_metadata(self, data: Dict[str, Any]) -> Optional[RelatedPaperMetadata]:
        """
        Parse World Bank API response into RelatedPaperMetadata.

        World Bank API response structure:
        - display_title: Paper title
        - authr: Authors (comma-separated string)
        - abstracts: Abstract text
        - docdt: Document date
        - docty: Document type
        - pdfurl: PDF URL
        - url: Document URL
        """
        try:
            # Extract authors - World Bank returns as comma-separated string or list
            authors = []
            if "authr" in data and data["authr"]:
                authr = data["authr"]
                if isinstance(authr, str):
                    authors = [a.strip() for a in authr.split(";") if a.strip()]
                elif isinstance(authr, list):
                    authors = [str(a) for a in authr if a]

            # Extract year from document date
            year = None
            if "docdt" in data and data["docdt"]:
                try:
                    # World Bank dates are typically in YYYY-MM-DD format
                    date_str = str(data["docdt"])
                    year = int(date_str[:4]) if len(date_str) >= 4 else None
                except (ValueError, TypeError):
                    pass

            # Extract abstract
            abstract = None
            if "abstracts" in data and data["abstracts"]:
                abstract = str(data["abstracts"])

            # Extract PDF URL
            pdf_url = None
            if "pdfurl" in data and data["pdfurl"]:
                pdf_url = data["pdfurl"]

            # Extract document URL
            url = None
            if "url" in data and data["url"]:
                url = data["url"]

            # Get document type for venue
            venue = None
            if "docty" in data and data["docty"]:
                venue = str(data["docty"])

            # Generate a paper ID (World Bank uses guid or we can create one)
            paper_id = data.get("guid") or f"wb_{hash(str(data))}"

            return RelatedPaperMetadata(
                paper_id=paper_id,
                title=str(data.get("display_title", "Unknown Title")),
                authors=authors,
                year=year,
                abstract=abstract,
                citation_count=None,  # World Bank doesn't provide citation counts
                venue=venue,
                url=url,
                open_access_pdf=pdf_url,
                key_findings=[],  # Will be populated by LLM if needed
                relevance_score=0.0,  # Could implement relevance scoring
                source="world_bank",
            )

        except Exception as e:
            print(f"[World Bank] Error parsing paper metadata: {e}")
            return None

    def get_api_call_count(self) -> int:
        """Return the number of API calls made."""
        return self._api_calls_today

    def reset_api_call_count(self):
        """Reset the API call counter."""
        self._api_calls_today = 0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_world_bank_searcher() -> WorldBankSearcher:
    """
    Create a WorldBankSearcher instance with default configuration.

    The World Bank API is free and requires no authentication.
    """
    return WorldBankSearcher()
