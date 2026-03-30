"""
Progress Event System

Thread-safe event emitter for real-time progress tracking.
Supports multiple backends: SSE (web), Print (CLI backward compat).
"""

import threading
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Base event."""
    event_type: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class RunStarted(Event):
    event_type: str = "run_started"
    run_dir: str = ""
    mode: str = "standard"
    paper_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageStarted(Event):
    event_type: str = "stage_started"
    stage_name: str = ""
    paper_filename: str = ""


@dataclass
class StageProgress(Event):
    event_type: str = "stage_progress"
    stage_name: str = ""
    current: int = 0
    total: int = 0
    detail: str = ""


@dataclass
class StageCompleted(Event):
    event_type: str = "stage_completed"
    stage_name: str = ""
    duration_s: float = 0.0
    result_summary: str = ""


@dataclass
class PaperCompleted(Event):
    event_type: str = "paper_completed"
    paper_filename: str = ""
    score: float = 0.0
    recommendation: str = ""
    cost: float = 0.0
    duration_s: float = 0.0


@dataclass
class RunProgress(Event):
    event_type: str = "run_progress"
    papers_done: int = 0
    papers_total: int = 0
    elapsed_s: float = 0.0
    estimated_remaining_s: float = 0.0


@dataclass
class Error(Event):
    event_type: str = "error"
    stage_name: str = ""
    message: str = ""
    recoverable: bool = True


@dataclass
class CostUpdate(Event):
    event_type: str = "cost_update"
    paper_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class RunCompleted(Event):
    event_type: str = "run_completed"
    total_papers: int = 0
    total_cost: float = 0.0
    total_time_s: float = 0.0
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend(ABC):
    @abstractmethod
    def emit(self, event: Event):
        pass


class PrintBackend(Backend):
    """CLI print backend - backward compatible with existing output."""

    def emit(self, event: Event):
        if isinstance(event, RunStarted):
            print(f"\n{'='*60}", flush=True)
            print(f"Run started: {event.mode} mode, {event.paper_count} papers", flush=True)
            print(f"{'='*60}", flush=True)
        elif isinstance(event, StageStarted):
            print(f"   [{event.stage_name}] Starting for {event.paper_filename}...", flush=True)
        elif isinstance(event, StageProgress):
            if event.total > 0:
                pct = 100 * event.current / event.total
                print(f"   [{event.stage_name}] {event.current}/{event.total} ({pct:.0f}%) {event.detail}", flush=True)
        elif isinstance(event, StageCompleted):
            print(f"   [{event.stage_name}] Completed in {event.duration_s:.1f}s - {event.result_summary}", flush=True)
        elif isinstance(event, PaperCompleted):
            print(f"   Paper done: {event.paper_filename} | Score: {event.score:.1f} | {event.recommendation} | ${event.cost:.4f}", flush=True)
        elif isinstance(event, RunProgress):
            pct = 100 * event.papers_done / max(event.papers_total, 1)
            eta = f"{event.estimated_remaining_s:.0f}s" if event.estimated_remaining_s > 0 else "N/A"
            print(f"   Progress: {event.papers_done}/{event.papers_total} ({pct:.0f}%) ETA: {eta}", flush=True)
        elif isinstance(event, Error):
            prefix = "WARNING" if event.recoverable else "ERROR"
            print(f"   [{prefix}] [{event.stage_name}] {event.message}", flush=True)
        elif isinstance(event, CostUpdate):
            print(f"   Cost: paper=${event.paper_cost:.4f}, total=${event.total_cost:.4f}", flush=True)
        elif isinstance(event, RunCompleted):
            print(f"\n{'='*60}", flush=True)
            print(f"Run completed: {event.total_papers} papers, ${event.total_cost:.4f}, {event.total_time_s:.1f}s", flush=True)
            print(f"Output: {event.output_dir}", flush=True)
            print(f"{'='*60}", flush=True)


class SSEBackend(Backend):
    """Server-Sent Events backend for the web dashboard."""

    def __init__(self):
        self._listeners: List[Any] = []  # List of asyncio.Queue
        self._lock = threading.Lock()

    def add_listener(self, queue):
        with self._lock:
            self._listeners.append(queue)

    def remove_listener(self, queue):
        with self._lock:
            try:
                self._listeners.remove(queue)
            except ValueError:
                pass

    def emit(self, event: Event):
        data = event.to_json()
        with self._lock:
            dead = []
            for q in self._listeners:
                try:
                    q.put_nowait(data)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._listeners.remove(q)


# ---------------------------------------------------------------------------
# Emitter (singleton)
# ---------------------------------------------------------------------------

_emitter_instance: Optional["ProgressEmitter"] = None
_emitter_lock = threading.Lock()


class ProgressEmitter:
    """Thread-safe progress event emitter."""

    def __init__(self):
        self._backends: List[Backend] = []
        self._lock = threading.Lock()

    def add_backend(self, backend: Backend):
        with self._lock:
            self._backends.append(backend)

    def remove_backend(self, backend: Backend):
        with self._lock:
            try:
                self._backends.remove(backend)
            except ValueError:
                pass

    def emit(self, event: Event):
        with self._lock:
            for backend in self._backends:
                try:
                    backend.emit(event)
                except Exception as e:
                    print(f"[ProgressEmitter] Backend error: {e}")

    @property
    def has_sse_backend(self) -> bool:
        with self._lock:
            return any(isinstance(b, SSEBackend) for b in self._backends)


def get_emitter() -> ProgressEmitter:
    """Get or create the singleton emitter."""
    global _emitter_instance
    with _emitter_lock:
        if _emitter_instance is None:
            _emitter_instance = ProgressEmitter()
            _emitter_instance.add_backend(PrintBackend())
        return _emitter_instance


def configure_emitter(backends: Optional[List[Backend]] = None, use_sse: bool = False):
    """Configure the singleton emitter with specified backends."""
    global _emitter_instance
    with _emitter_lock:
        _emitter_instance = ProgressEmitter()
        if backends:
            for b in backends:
                _emitter_instance.add_backend(b)
        else:
            _emitter_instance.add_backend(PrintBackend())
        if use_sse:
            _emitter_instance.add_backend(SSEBackend())
    return _emitter_instance


def get_sse_backend() -> Optional[SSEBackend]:
    """Get the SSE backend from the current emitter, if present."""
    emitter = get_emitter()
    with emitter._lock:
        for b in emitter._backends:
            if isinstance(b, SSEBackend):
                return b
    return None
