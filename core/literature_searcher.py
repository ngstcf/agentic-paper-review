"""
Literature Searcher Module

Integrates with Semantic Scholar API to search for and retrieve academic papers.
Provides the foundation for literature-grounded reviews.
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from core.data_models import RelatedPaperMetadata


@dataclass
class SearchConfig:
    """Configuration for literature search."""
    api_key: Optional[str] = None
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    # Request limits
    max_papers_per_search: int = 100
    min_citation_count: int = 10
    recency_years: int = 5


class LiteratureSearcher:
    """
    Search academic databases for related papers using Semantic Scholar API.

    Features:
    - Keyword-based search
    - Abstract similarity search
    - Citation-based ranking
    - Open-access PDF retrieval
    - Rate limiting and retry logic
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the literature searcher.

        Args:
            config: Search configuration. If None, uses environment variables.
        """
        self.config = config or self._default_config()
        self._session = requests.Session()
        self._api_calls_today = 0

        if self.config.api_key:
            self._session.headers.update({
                "x-api-key": self.config.api_key
            })

    def _default_config(self) -> SearchConfig:
        """Create default configuration from environment variables."""
        return SearchConfig(
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            base_url=os.getenv("SEMANTIC_SCHOLAR_API_URL",
                              "https://api.semanticscholar.org/graph/v1"),
            timeout=int(os.getenv("SEMANTIC_SCHOLAR_TIMEOUT", "30")),
        )

    # ========================================================================
    # SEARCH METHODS
    # ========================================================================

    def search_by_keywords(
        self,
        keywords: List[str],
        limit: int = 20,
        min_citation_count: int = 10,
        year_range: Optional[tuple[int, int]] = None,
        fields: Optional[List[str]] = None
    ) -> List[RelatedPaperMetadata]:
        """
        Search for papers by keywords.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            min_citation_count: Minimum citation count filter
            year_range: Optional (min_year, max_year) tuple
            fields: Additional fields to retrieve from API

        Returns:
            List of related paper metadata
        """
        query = " ".join(keywords)
        return self._search_papers(
            query=query,
            limit=limit,
            min_citation_count=min_citation_count,
            year_range=year_range,
            fields=fields
        )

    def search_by_abstract_similarity(
        self,
        abstract: str,
        limit: int = 20,
        min_citation_count: int = 10
    ) -> List[RelatedPaperMetadata]:
        """
        Search for papers similar to the given abstract.

        Uses Semantic Scholar's relevance search with the abstract text.

        Args:
            abstract: The abstract text to find similar papers
            limit: Maximum number of results
            min_citation_count: Minimum citation count filter

        Returns:
            List of related paper metadata
        """
        # Extract key phrases from abstract for search
        # Use first ~200 words as search query
        query = " ".join(abstract.split()[:200]) if abstract else ""

        return self._search_papers(
            query=query,
            limit=limit,
            min_citation_count=min_citation_count,
            search_type="relevance"
        )

    def get_most_cited(
        self,
        field_keywords: List[str],
        years: int = 5,
        limit: int = 5
    ) -> List[RelatedPaperMetadata]:
        """
        Get the most cited papers in a specific field/sub-topic.

        Args:
            field_keywords: Keywords defining the field/sub-topic
            years: Number of years to look back (default: 5)
            limit: Number of papers to return (default: 5)

        Returns:
            List of most cited papers, sorted by citation count
        """
        current_year = datetime.now(timezone.utc).year
        year_range = (current_year - years, current_year)

        results = self.search_by_keywords(
            keywords=field_keywords,
            limit=limit * 3,  # Get more to filter for quality
            min_citation_count=self.config.min_citation_count,
            year_range=year_range
        )

        # Sort by citation count and return top N
        return sorted(results, key=lambda p: p.citation_count or 0, reverse=True)[:limit]

    def get_paper_details(
        self,
        paper_id: str,
        include_full_text: bool = False
    ) -> Optional[RelatedPaperMetadata]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID
            include_full_text: Whether to fetch full text if available

        Returns:
            Paper metadata or None if not found
        """
        fields = self._default_fields(include_full_text)
        url = f"{self.config.base_url}/paper/{paper_id}"

        params = {"fields": ",".join(fields)}

        data = self._make_request(url, params=params)
        if not data:
            return None

        return self._parse_paper_metadata(data)

    def fetch_abstract(self, paper_id: str) -> Optional[str]:
        """
        Fetch the abstract for a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Abstract text or None if unavailable
        """
        paper = self.get_paper_details(paper_id)
        return paper.abstract if paper else None

    def fetch_full_text(self, paper_id: str) -> Optional[str]:
        """
        Fetch full text PDF content for an open-access paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Full text content or None if not available
        """
        paper = self.get_paper_details(paper_id, include_full_text=True)

        if not paper or not paper.open_access_pdf:
            return None

        try:
            response = self._session.get(
                paper.open_access_pdf,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            # Return PDF content (would need PDF parsing in practice)
            # For now, return the URL for the caller to handle
            return paper.open_access_pdf

        except Exception as e:
            print(f"Error fetching full text: {e}")
            return None

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    def _search_papers(
        self,
        query: str,
        limit: int = 20,
        min_citation_count: int = 10,
        year_range: Optional[tuple[int, int]] = None,
        search_type: str = "keyword",
        fields: Optional[List[str]] = None
    ) -> List[RelatedPaperMetadata]:
        """
        Internal search method with common parameters.
        """
        url = f"{self.config.base_url}/paper/search"

        if fields is None:
            fields = self._default_fields()

        params = {
            "query": query,
            "limit": min(limit, self.config.max_papers_per_search),
            "fields": ",".join(fields),
            "minCitationCount": min_citation_count,
        }

        # Add year filter if specified
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        # Use relevance search for abstract similarity
        if search_type == "relevance":
            # Semantic Scholar uses the same endpoint but ranks by relevance
            pass

        data = self._make_request(url, params=params)
        if not data:
            return []

        papers = []
        for item in data.get("data", []):
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
        Make HTTP request with retry logic and rate limiting.
        """
        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            # Rate limiting - wait and retry
            if e.response.status_code == 429:
                if retry_count < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** retry_count)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    return self._make_request(url, params, retry_count + 1)
                else:
                    print(f"Max retries exceeded for URL: {url}")

            print(f"HTTP error: {e}")
            return None

        except Exception as e:
            print(f"Request failed: {e}")
            if retry_count < self.config.max_retries:
                time.sleep(self.config.retry_delay)
                return self._make_request(url, params, retry_count + 1)
            return None

    def _parse_paper_metadata(self, data: Dict[str, Any]) -> Optional[RelatedPaperMetadata]:
        """
        Parse API response into RelatedPaperMetadata.
        """
        try:
            # Extract authors
            authors = []
            if "authors" in data:
                authors = [
                    author.get("name", "Unknown")
                    for author in data["authors"]
                ]

            # Extract open access PDF URL
            open_access_pdf = None
            if "openAccessPdf" in data and data["openAccessPdf"]:
                open_access_pdf = data["openAccessPdf"].get("url")

            return RelatedPaperMetadata(
                paper_id=data.get("paperId", ""),
                title=data.get("title", ""),
                authors=authors,
                year=data.get("year"),
                abstract=data.get("abstract"),
                citation_count=data.get("citationCount", 0),
                venue=data.get("venue"),
                url=data.get("url"),
                open_access_pdf=open_access_pdf,
            )

        except Exception as e:
            print(f"Error parsing paper metadata: {e}")
            return None

    def _default_fields(self, include_full_text: bool = False) -> List[str]:
        """
        Default fields to request from Semantic Scholar API.
        """
        fields = [
            "paperId",
            "title",
            "abstract",
            "authors",
            "year",
            "citationCount",
            "venue",
            "url",
            "openAccessPdf",
            "fieldsOfStudy",
            "publicationDate",
        ]

        if include_full_text:
            fields.append("openAccessPdf")

        return fields

    def get_api_call_count(self) -> int:
        """Return the number of API calls made today."""
        return self._api_calls_today

    def reset_api_call_count(self):
        """Reset the API call counter."""
        self._api_calls_today = 0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_searcher_from_env() -> LiteratureSearcher:
    """
    Create a LiteratureSearcher instance using environment variables.

    Environment variables:
    - SEMANTIC_SCHOLAR_API_KEY: Optional API key for higher rate limits
    - SEMANTIC_SCHOLAR_API_URL: API base URL (default: official URL)
    - SEMANTIC_SCHOLAR_TIMEOUT: Request timeout in seconds (default: 30)
    """
    return LiteratureSearcher()
