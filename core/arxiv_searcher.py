"""
Arxiv Searcher Module

Integrates with Arxiv API to search for and retrieve academic papers.
Arxiv API is free and open - no API key required.

Documentation: http://export.arxiv.org/api_help/docs/user-manual.html
"""

import requests
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from xml.etree import ElementTree

from core.data_models import RelatedPaperMetadata


@dataclass
class ArxivSearchConfig:
    """Configuration for Arxiv search."""
    base_url: str = "http://export.arxiv.org/api/query"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    # Arxiv requests one request per 3 seconds, but allows short bursts
    min_request_interval: float = 3.0


class ArxivSearcher:
    """
    Search Arxiv for academic papers.

    Features:
    - Keyword-based search
    - Category filtering (cs.AI, math.NA, etc.)
    - Date range filtering
    - Abstract and PDF retrieval
    - Rate limiting and retry logic

    Note: Arxiv does NOT provide citation counts.
    Results are sorted by relevance, recency, or lastUpdatedDate.
    """

    def __init__(self, config: Optional[ArxivSearchConfig] = None):
        """
        Initialize the Arxiv searcher.

        Args:
            config: Search configuration
        """
        self.config = config or ArxivSearchConfig()
        self._session = requests.Session()
        self._api_calls_today = 0
        self._last_request_time = 0

    def search_by_keywords(
        self,
        keywords: List[str],
        limit: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> List[RelatedPaperMetadata]:
        """
        Search Arxiv by keywords.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)
            categories: Optional list of Arxiv categories (e.g., ["cs.AI", "cs.LG"])

        Returns:
            List of related paper metadata
        """
        # Build Arxiv query - use AND logic for keywords
        # Arxiv query syntax: http://export.arxiv.org/api_query_args
        query_parts = []

        # Add keyword search (all fields)
        if keywords:
            keyword_query = " AND ".join([f"all:{kw}" for kw in keywords])
            query_parts.append(f"({keyword_query})")

        # Add category filter if specified
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query_parts.append(f"({cat_query})")

        # Combine all parts
        query = " AND ".join(query_parts) if query_parts else "all:*"

        # Add date range filter
        if start_date or end_date:
            date_filter = ""
            if start_date:
                # Convert to YYYYMMDDHHMM format for Arxiv
                date_filter += f"[{start_date.replace('-', '')}0000"
            else:
                date_filter += "[199101010000"
            if end_date:
                date_filter += f" TO {end_date.replace('-', '')}0000"
            else:
                date_filter += " TO 203001010000"
            date_filter += "]"  # Add closing bracket
            query += f" submittedDate:{date_filter}"

        return self._search_papers(
            query=query,
            limit=limit
        )

    def _search_papers(
        self,
        query: str,
        limit: int = 20,
        sort_by: str = "relevance"
    ) -> List[RelatedPaperMetadata]:
        """
        Internal search method.

        Args:
            query: Arxiv query string (following Arxiv query syntax)
            limit: Maximum number of results (max 2000 per Arxiv API)
            sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate"

        Returns:
            List of paper metadata
        """
        # Rate limiting - Arxiv requests 1 request per 3 seconds
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self.config.min_request_interval:
            sleep_time = self.config.min_request_interval - time_since_last_request
            print(f"[Arxiv] Rate limiting: sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(limit, 2000),  # Arxiv max is 2000
            "sortBy": sort_by,
            "sortOrder": "descending"
        }

        data = self._make_request(self.config.base_url, params=params)
        if not data:
            return []

        papers = []
        for entry in data.get("entries", []):
            paper = self._parse_paper_metadata(entry)
            if paper:
                papers.append(paper)

        self._api_calls_today += 1
        self._last_request_time = time.time()
        return papers

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic. Returns parsed JSON from Atom feed.
        """
        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            # Parse Atom XML response
            return self._parse_atom_response(response.text)

        except requests.exceptions.RequestException as e:
            print(f"[Arxiv] Request failed: {e}")

            if retry_count < self.config.max_retries:
                wait_time = self.config.retry_delay * (2 ** retry_count)
                print(f"[Arxiv] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self._make_request(url, params, retry_count + 1)

            return None

    def _parse_atom_response(self, xml_text: str) -> Dict[str, Any]:
        """
        Parse Arxiv Atom XML response into dictionary.

        Arxiv returns results in Atom XML format.
        """
        result = {"entries": []}

        try:
            # Parse XML
            root = ElementTree.fromstring(xml_text)

            # Define namespace
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }

            # Extract entries
            for entry in root.findall("atom:entry", ns):
                entry_data = {}

                # Title
                title_elem = entry.find("atom:title", ns)
                if title_elem is not None:
                    entry_data["title"] = title_elem.text.strip().replace("\n", " ")

                # Authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                entry_data["authors"] = authors

                # Summary (abstract)
                summary_elem = entry.find("atom:summary", ns)
                if summary_elem is not None:
                    entry_data["summary"] = summary_elem.text.strip().replace("\n", " ")

                # Published date
                published_elem = entry.find("atom:published", ns)
                if published_elem is not None:
                    entry_data["published"] = published_elem.text

                # Arxiv ID
                id_elem = entry.find("atom:id", ns)
                if id_elem is not None:
                    entry_data["id"] = id_elem.text

                # Primary category
                primary_cat = entry.find("arxiv:primary_category", ns)
                if primary_cat is not None:
                    entry_data["primary_category"] = primary_cat.get("term")

                # Categories (list)
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)
                entry_data["categories"] = categories

                # PDF link
                pdf_url = None
                for link in entry.findall("atom:link", ns):
                    if link.get("type") == "application/pdf":
                        pdf_url = link.get("href")
                        break
                entry_data["pdf_url"] = pdf_url

                # Abstract URL (landing page)
                for link in entry.findall("atom:link", ns):
                    if link.get("type") == "text/html":
                        entry_data["url"] = link.get("href")
                        break

                # Arxiv identifier (from URL)
                arxiv_id = None
                if "id" in entry_data:
                    # Extract ID from URL like http://arxiv.org/abs/2301.12345v1
                    parts = entry_data["id"].split("/")
                    if parts:
                        arxiv_id = parts[-1]
                entry_data["arxiv_id"] = arxiv_id

                result["entries"].append(entry_data)

        except ElementTree.ParseError as e:
            print(f"[Arxiv] Error parsing XML response: {e}")
        except Exception as e:
            print(f"[Arxiv] Error processing response: {e}")

        return result

    def _parse_paper_metadata(self, data: Dict[str, Any]) -> Optional[RelatedPaperMetadata]:
        """
        Parse Arxiv API response into RelatedPaperMetadata.
        """
        try:
            # Extract authors
            authors = data.get("authors", [])

            # Extract year from published date
            year = None
            published = data.get("published")
            if published:
                try:
                    # Arxiv dates are in ISO 8601 format: 2023-01-15T12:34:56Z
                    date_obj = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    year = date_obj.year
                except (ValueError, TypeError):
                    pass

            # Extract abstract
            abstract = data.get("summary")

            # Extract PDF URL
            pdf_url = data.get("pdf_url")

            # Extract landing page URL
            url = data.get("url")

            # Use primary category as venue
            venue = data.get("primary_category")

            # Generate paper ID
            paper_id = data.get("arxiv_id", f"arxiv_{hash(str(data))}")

            return RelatedPaperMetadata(
                paper_id=paper_id,
                title=data.get("title", "Unknown Title"),
                authors=authors,
                year=year,
                abstract=abstract,
                citation_count=None,  # Arxiv doesn't provide citation counts
                venue=venue,
                url=url,
                open_access_pdf=pdf_url,
                key_findings=[],  # Will be populated by LLM if needed
                relevance_score=0.0,  # Could implement relevance scoring
                source="arxiv",
            )

        except Exception as e:
            print(f"[Arxiv] Error parsing paper metadata: {e}")
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

def create_arxiv_searcher() -> ArxivSearcher:
    """
    Create an ArxivSearcher instance with default configuration.

    The Arxiv API is free and requires no authentication.
    """
    return ArxivSearcher()
