"""
FastAPI Web Application

Serves the dashboard UI and provides REST/SSE API endpoints
for managing and monitoring review runs.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import threading
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.progress import (
    get_emitter, get_sse_backend, SSEBackend,
    Event, RunStarted, RunCompleted, PaperCompleted, RunProgress,
    configure_emitter,
)

app = FastAPI(title="Academic Review System")

# Static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Run state tracking
# ---------------------------------------------------------------------------

_active_runs: Dict[str, Dict[str, Any]] = {}
_run_history: List[Dict[str, Any]] = []
_cancel_flags: Dict[str, bool] = {}


def _find_run_dirs() -> List[Dict[str, str]]:
    """Scan for run directories at the project root."""
    root = os.path.dirname(os.path.abspath(__file__))
    dirs = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "papers")):
            # Count papers
            papers = (
                glob.glob(os.path.join(path, "papers", "*.pdf"))
                + glob.glob(os.path.join(path, "papers", "*.md"))
            )
            # Check for outputs
            has_output = os.path.isdir(os.path.join(path, "outputs", "reviews"))
            dirs.append({
                "name": entry,
                "path": path,
                "paper_count": len(papers),
                "has_output": has_output,
            })
    return dirs


def _load_config_safe(run_dir: str) -> Dict[str, Any]:
    """Load config for a run directory, masking API keys."""
    env_path = os.path.join(run_dir, "input", ".env")
    config_data: Dict[str, Any] = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if "KEY" in key.upper() or "SECRET" in key.upper():
                        config_data[key] = "***masked***"
                    else:
                        config_data[key] = value
    return config_data


def _load_criteria(run_dir: str) -> Dict[str, Any]:
    """Load criteria.yaml for a run directory."""
    criteria_path = os.path.join(run_dir, "input", "criteria.yaml")
    if not os.path.exists(criteria_path):
        # Try default
        criteria_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "criteria.yaml")
    if not os.path.exists(criteria_path):
        return {}
    import yaml
    with open(criteria_path) as f:
        return yaml.safe_load(f) or {}


def _get_reviews(run_dir: str) -> List[Dict[str, Any]]:
    """Get list of review files for a run directory."""
    reviews_dir = os.path.join(run_dir, "outputs", "reviews")
    if not os.path.isdir(reviews_dir):
        return []
    reviews = []
    for f in sorted(glob.glob(os.path.join(reviews_dir, "*.md")), reverse=True):
        reviews.append({
            "filename": os.path.basename(f),
            "path": f,
            "size": os.path.getsize(f),
            "modified": datetime.fromtimestamp(os.path.getmtime(f)).isoformat(),
        })
    return reviews


def _parse_csv_results(run_dir: str) -> List[Dict[str, Any]]:
    """Parse the latest consolidated CSV into JSON."""
    reports_dir = os.path.join(run_dir, "outputs", "reports")
    if not os.path.isdir(reports_dir):
        return []
    csv_files = sorted(glob.glob(os.path.join(reports_dir, "*.csv")), reverse=True)
    if not csv_files:
        return []
    try:
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        return df.to_dict(orient="records")
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pipeline runner (background thread)
# ---------------------------------------------------------------------------

def _run_pipeline(run_dir: str, run_id: str, mode: str, config_overrides: Dict[str, Any]):
    """Execute the review pipeline in a background thread."""
    from core.config_loader import Config
    from core.paper_ingestor import load_ingestion_cache, save_ingestion_cache, ingest_directory
    from utilities.helpers import setup_logging, get_config_hash

    emitter = get_emitter()
    setup_logging()

    start_time = time.time()
    _cancel_flags[run_id] = False

    try:
        dirs = {
            "run_dir": run_dir,
            "papers_dir": os.path.join(run_dir, "papers"),
            "input_dir": os.path.join(run_dir, "input"),
            "outputs_dir": os.path.join(run_dir, "outputs"),
            "reports_dir": os.path.join(run_dir, "outputs", "reports"),
            "reviews_dir": os.path.join(run_dir, "outputs", "reviews"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # Apply config overrides to .env
        if config_overrides:
            env_path = os.path.join(dirs["input_dir"], ".env")
            env_vars = {}
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k] = v
            env_vars.update(config_overrides)
            with open(env_path, "w") as f:
                for k, v in env_vars.items():
                    if not k.endswith("_API_KEY"):
                        f.write(f"{k}={v}\n")

        config = Config(config_path=dirs["input_dir"])
        config_hash = get_config_hash(config)

        # Ingest papers
        cache_file = os.path.join(run_dir, "ingestion_cache.json")
        ingestion_cache = load_ingestion_cache(cache_file)
        papers, cache_was_updated = ingest_directory(dirs["papers_dir"], ingestion_cache, cache_file)
        if cache_was_updated:
            save_ingestion_cache(ingestion_cache, cache_file)

        if not papers:
            emitter.emit(Event(event_type="error", stage_name="ingest", message="No papers found", recoverable=False))
            return

        emitter.emit(RunStarted(run_dir=run_dir, mode=mode, paper_count=len(papers)))

        # Select pipeline
        if mode == "literature":
            from run_review_with_dir_literature import main as _lit_main
            # Import needed agents
            from agents.agent_librarian import create_baseline_reference
            from agents.agent_reader import process_paper_extractions as process_lit
            from agents.agent_fact_checker import run_fact_checks
            from agents.agent_critic import synthesize_grounded_review
            from agents.agent_extractor import process_paper_extractions
            from agents.agent_synthesizer import synthesize_review
            from utilities.output_generator import save_review_markdown, save_consolidated_csv
            from utilities.helpers import load_yaml_config
            from core.data_models import GroundedReview, LiteratureContext
            import time as time_mod

            literature_config = load_yaml_config("config/literature_sources.yaml")
        else:
            from agents.agent_extractor import process_paper_extractions
            from agents.agent_synthesizer import synthesize_review
            from utilities.output_generator import save_review_markdown, save_consolidated_csv

        final_reviews = []
        total_cost = 0.0

        for i, paper in enumerate(papers, 1):
            if _cancel_flags.get(run_id, False):
                emitter.emit(Event(event_type="error", stage_name="cancel", message="Run cancelled by user", recoverable=True))
                break

            paper_start = time.time()

            if mode == "literature":
                # Stage 1: Librarian
                emitter.emit(StageStarted(stage_name="Librarian", paper_filename=paper.filename))
                t0 = time.time()
                baseline = None
                try:
                    baseline = create_baseline_reference(paper, config)
                except Exception as e:
                    emitter.emit(Error(stage_name="Librarian", message=str(e)))
                emitter.emit(StageCompleted(stage_name="Librarian", duration_s=time.time()-t0,
                                           result_summary=f"{len(baseline.baseline_papers) if baseline else 0} papers"))

                # Stage 2: Reader/Extractor
                emitter.emit(StageStarted(stage_name="Extraction", paper_filename=paper.filename))
                t0 = time.time()
                if baseline:
                    extractions = process_lit(paper, config, baseline=baseline)
                else:
                    extractions = process_paper_extractions(paper, config)
                emitter.emit(StageCompleted(stage_name="Extraction", duration_s=time.time()-t0,
                                           result_summary=f"{len(extractions)} criteria"))

                if not extractions:
                    continue

                # Stage 3: Fact-Checker
                fact_checks = []
                if baseline:
                    emitter.emit(StageStarted(stage_name="Fact-Check", paper_filename=paper.filename))
                    t0 = time.time()
                    try:
                        fact_checks = run_fact_checks(extractions, config.get_criteria(), config, literature_config) or []
                    except Exception:
                        pass
                    emitter.emit(StageCompleted(stage_name="Fact-Check", duration_s=time.time()-t0,
                                               result_summary=f"{len(fact_checks)} checks"))

                # Stage 4: Critic
                emitter.emit(StageStarted(stage_name="Synthesis", paper_filename=paper.filename))
                t0 = time.time()
                review = synthesize_grounded_review(paper, extractions, config, baseline=baseline, fact_checks=fact_checks)
                emitter.emit(StageCompleted(stage_name="Synthesis", duration_s=time.time()-t0,
                                           result_summary=f"Score: {review.overall_score:.1f}" if review else "Failed"))
            else:
                # Standard pipeline
                emitter.emit(StageStarted(stage_name="Extraction", paper_filename=paper.filename))
                t0 = time.time()
                extractions = process_paper_extractions(paper, config)
                emitter.emit(StageCompleted(stage_name="Extraction", duration_s=time.time()-t0,
                                           result_summary=f"{len(extractions)} criteria"))

                if not extractions:
                    continue

                emitter.emit(StageStarted(stage_name="Synthesis", paper_filename=paper.filename))
                t0 = time.time()
                review = synthesize_review(paper, extractions, config)
                emitter.emit(StageCompleted(stage_name="Synthesis", duration_s=time.time()-t0,
                                           result_summary=f"Score: {review.overall_score:.1f}" if review else "Failed"))

            if not review:
                continue

            # Save review
            paper_base = os.path.splitext(paper.filename)[0]
            ext_model = review.extractor_model_used.replace("/", "_")
            syn_model = review.synthesizer_model_used.replace("/", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            review_fn = f"{paper_base}_{ext_model}_{syn_model}_{ts}.md"
            output_path = os.path.join(dirs["reviews_dir"], review_fn)
            save_review_markdown(review=review, output_path=output_path, paper=paper, config=config)

            paper_duration = time.time() - paper_start
            total_cost += review.total_cost

            emitter.emit(PaperCompleted(
                paper_filename=paper.filename,
                score=review.overall_score,
                recommendation=review.recommendation,
                cost=review.total_cost,
                duration_s=paper_duration,
            ))

            elapsed = time.time() - start_time
            avg_per_paper = elapsed / i
            remaining = avg_per_paper * (len(papers) - i)
            emitter.emit(RunProgress(
                papers_done=i, papers_total=len(papers),
                elapsed_s=elapsed, estimated_remaining_s=remaining,
            ))
            emitter.emit(CostUpdate(paper_cost=review.total_cost, total_cost=total_cost))

            final_reviews.append(review)

        # Save consolidated CSV
        if final_reviews:
            save_consolidated_csv(final_reviews, dirs["reports_dir"])

        total_time = time.time() - start_time
        emitter.emit(RunCompleted(
            total_papers=len(final_reviews),
            total_cost=total_cost,
            total_time_s=total_time,
            output_dir=dirs["outputs_dir"],
        ))

    except Exception as e:
        import traceback
        emitter.emit(Error(stage_name="pipeline", message=f"{e}", recoverable=False))
        traceback.print_exc()
    finally:
        _active_runs.pop(run_id, None)
        _cancel_flags.pop(run_id, None)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return f.read()


@app.get("/api/runs")
async def list_runs():
    return {"runs": _find_run_dirs()}


@app.post("/api/start")
async def start_run(request: Request):
    body = await request.json()
    run_dir = body.get("run_dir")
    mode = body.get("mode", "standard")
    config_overrides = body.get("config_overrides", {})

    if not run_dir:
        raise HTTPException(status_code=400, detail="run_dir is required")

    # Resolve to absolute path
    if not os.path.isabs(run_dir):
        root = os.path.dirname(os.path.abspath(__file__))
        run_dir = os.path.join(root, run_dir)

    if not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail=f"Directory not found: {run_dir}")

    run_id = f"{os.path.basename(run_dir)}_{int(time.time())}"

    # Ensure SSE backend is active
    emitter = get_emitter()
    sse = get_sse_backend()
    if not sse:
        sse = SSEBackend()
        emitter.add_backend(sse)

    _active_runs[run_id] = {
        "run_dir": run_dir,
        "mode": mode,
        "started_at": time.time(),
        "status": "running",
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(run_dir, run_id, mode, config_overrides),
        daemon=True,
    )
    thread.start()

    return {"run_id": run_id, "status": "started"}


@app.get("/api/events/{run_id}")
async def event_stream(run_id: str):
    """SSE endpoint for real-time progress."""
    async def event_generator():
        import asyncio as aio
        queue = asyncio.Queue()
        sse = get_sse_backend()
        if not sse:
            sse = SSEBackend()
            get_emitter().add_backend(sse)

        sse.add_listener(queue)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"event": "progress", "data": data}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            sse.remove_listener(queue)

    return EventSourceResponse(event_generator())


@app.get("/api/status/{run_id}")
async def get_status(run_id: str):
    if run_id in _active_runs:
        return {"status": "running", **_active_runs[run_id]}
    return {"status": "completed"}


@app.post("/api/stop/{run_id}")
async def stop_run(run_id: str):
    if run_id not in _active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    _cancel_flags[run_id] = True
    return {"status": "cancelling"}


@app.get("/api/config/{run_dir:path}")
async def get_config(run_dir: str):
    root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(root, run_dir) if not os.path.isabs(run_dir) else run_dir
    return {"config": _load_config_safe(full_path)}


@app.get("/api/results/{run_dir:path}")
async def get_results(run_dir: str):
    root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(root, run_dir) if not os.path.isabs(run_dir) else run_dir
    return {"results": _parse_csv_results(full_path)}


@app.get("/api/review/{run_dir:path}/{filename:path}")
async def get_review(run_dir: str, filename: str):
    root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(root, run_dir) if not os.path.isabs(run_dir) else run_dir
    review_path = os.path.join(full_path, "outputs", "reviews", filename)
    if not os.path.exists(review_path):
        raise HTTPException(status_code=404, detail="Review not found")
    return FileResponse(review_path, media_type="text/markdown")


@app.get("/api/criteria/{run_dir:path}")
async def get_criteria(run_dir: str):
    root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(root, run_dir) if not os.path.isabs(run_dir) else run_dir
    return {"criteria": _load_criteria(full_path)}


@app.get("/api/reviews/{run_dir:path}")
async def list_reviews(run_dir: str):
    root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(root, run_dir) if not os.path.isabs(run_dir) else run_dir
    return {"reviews": _get_reviews(full_path)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Academic Review System - Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind")
    args = parser.parse_args()

    import uvicorn
    print(f"Starting Academic Review System dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
