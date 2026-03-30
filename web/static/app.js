/**
 * Academic Review System - Dashboard Frontend
 */

// State
let currentRunDir = "";
let currentRunId = null;
let eventSource = null;
let runStartTime = null;
let elapsedInterval = null;
let sortColumn = null;
let sortAsc = true;

// DOM refs
const runSelector = document.getElementById("run-selector");
const modeSelector = document.getElementById("mode-selector");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const progressEta = document.getElementById("progress-eta");
const costDisplay = document.getElementById("cost-display");
const elapsedDisplay = document.getElementById("elapsed-display");
const currentPaperDisplay = document.getElementById("current-paper-display");
const resultsTbody = document.getElementById("results-tbody");
const noResults = document.getElementById("no-results");
const logContent = document.getElementById("log-content");
const configContent = document.getElementById("config-content");
const criteriaContent = document.getElementById("criteria-content");
const reviewModal = document.getElementById("review-modal");
const reviewModalTitle = document.getElementById("review-modal-title");
const reviewModalBody = document.getElementById("review-modal-body");
const reviewModalClose = document.getElementById("review-modal-close");

// ---- Initialization ----

async function init() {
    await loadRuns();
    runSelector.addEventListener("change", onRunSelected);
    startBtn.addEventListener("click", startRun);
    stopBtn.addEventListener("click", stopRun);
    reviewModalClose.addEventListener("click", () => reviewModal.style.display = "none");

    // Sort headers
    document.querySelectorAll("#results-table th[data-sort]").forEach(th => {
        th.addEventListener("click", () => {
            const col = th.dataset.sort;
            if (sortColumn === col) sortAsc = !sortAsc;
            else { sortColumn = col; sortAsc = true; }
            sortAndRenderResults();
        });
    });
}

// ---- API helpers ----

async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
    return res.json();
}

// ---- Run Directory Selection ----

async function loadRuns() {
    const data = await api("/api/runs");
    runSelector.innerHTML = '<option value="">Select run directory...</option>';
    for (const run of data.runs) {
        const opt = document.createElement("option");
        opt.value = run.name;
        opt.textContent = `${run.name} (${run.paper_count} papers)`;
        runSelector.appendChild(opt);
    }
}

async function onRunSelected() {
    currentRunDir = runSelector.value;
    startBtn.disabled = !currentRunDir;

    if (!currentRunDir) {
        configContent.innerHTML = '<p class="placeholder-text">Select a run directory to view config</p>';
        criteriaContent.innerHTML = '<p class="placeholder-text">Select a run directory to view criteria</p>';
        return;
    }

    // Load config
    try {
        const data = await api(`/api/config/${currentRunDir}`);
        let html = "";
        for (const [k, v] of Object.entries(data.config)) {
            html += `<div class="config-item"><span class="config-key">${k}</span><span class="config-value">${v}</span></div>`;
        }
        configContent.innerHTML = html || '<p class="placeholder-text">No config found</p>';
    } catch {
        configContent.innerHTML = '<p class="placeholder-text">Could not load config</p>';
    }

    // Load criteria
    try {
        const data = await api(`/api/criteria/${currentRunDir}`);
        const criteria = data.criteria?.criteria || [];
        let html = "";
        for (const c of criteria) {
            html += `<div class="config-item"><span class="config-key">${c.id}</span><span class="config-value">${c.weight}%</span></div>`;
        }
        criteriaContent.innerHTML = html || '<p class="placeholder-text">No criteria found</p>';
    } catch {
        criteriaContent.innerHTML = '<p class="placeholder-text">Could not load criteria</p>';
    }

    // Load existing results
    await loadResults();
    await loadReviews();
}

// ---- Start / Stop ----

async function startRun() {
    if (!currentRunDir) return;

    startBtn.disabled = true;
    stopBtn.style.display = "inline-block";
    runStartTime = Date.now();
    logContent.innerHTML = "";
    clearStages();

    try {
        const data = await api("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                run_dir: currentRunDir,
                mode: modeSelector.value,
                config_overrides: {},
            }),
        });
        currentRunId = data.run_id;
        connectSSE();
        startElapsedTimer();
    } catch (e) {
        addLog("Failed to start run: " + e.message, "error");
        startBtn.disabled = false;
        stopBtn.style.display = "none";
    }
}

async function stopRun() {
    if (!currentRunId) return;
    try {
        await api(`/api/stop/${currentRunId}`, { method: "POST" });
        addLog("Cancellation requested...", "warning");
    } catch (e) {
        addLog("Stop failed: " + e.message, "error");
    }
}

// ---- SSE ----

function connectSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`/api/events/${currentRunId}`);

    eventSource.addEventListener("progress", (e) => {
        try {
            const evt = JSON.parse(e.data);
            handleEvent(evt);
        } catch {}
    });

    eventSource.addEventListener("ping", () => {});

    eventSource.onerror = () => {
        // Reconnect will happen automatically
    };
}

function handleEvent(evt) {
    switch (evt.event_type) {
        case "run_started":
            addLog(`Run started: ${evt.mode} mode, ${evt.paper_count} papers`, "info");
            break;

        case "stage_started":
            activateStage(evt.stage_name);
            addLog(`[${evt.stage_name}] Starting for ${evt.paper_filename}`, "info");
            currentPaperDisplay.textContent = evt.paper_filename;
            break;

        case "stage_progress":
            addLog(`[${evt.stage_name}] ${evt.current}/${evt.total} ${evt.detail}`, "info");
            break;

        case "stage_completed":
            completeStage(evt.stage_name);
            addLog(`[${evt.stage_name}] Done in ${evt.duration_s?.toFixed(1)}s - ${evt.result_summary}`, "success");
            break;

        case "paper_completed":
            addLog(`Paper done: ${evt.paper_filename} | Score: ${evt.score?.toFixed(1)} | ${evt.recommendation} | $${evt.cost?.toFixed(4)}`, "success");
            break;

        case "run_progress": {
            const pct = evt.papers_total > 0 ? (100 * evt.papers_done / evt.papers_total) : 0;
            progressBar.style.width = pct + "%";
            progressText.textContent = `${evt.papers_done}/${evt.papers_total} papers (${pct.toFixed(0)}%)`;
            if (evt.estimated_remaining_s > 0) {
                progressEta.textContent = `ETA: ${formatDuration(evt.estimated_remaining_s)}`;
            }
            break;
        }

        case "cost_update":
            costDisplay.textContent = `$${evt.total_cost?.toFixed(4)}`;
            break;

        case "error":
            addLog(`[${evt.stage_name}] ${evt.message}`, evt.recoverable ? "warning" : "error");
            break;

        case "run_completed":
            addLog(`Run completed: ${evt.total_papers} papers, $${evt.total_cost?.toFixed(4)}, ${formatDuration(evt.total_time_s)}`, "success");
            progressBar.style.width = "100%";
            progressText.textContent = "Complete";
            progressEta.textContent = "";
            finishRun();
            break;
    }
}

function finishRun() {
    startBtn.disabled = false;
    stopBtn.style.display = "none";
    if (elapsedInterval) clearInterval(elapsedInterval);
    if (eventSource) { eventSource.close(); eventSource = null; }
    loadResults();
}

// ---- Results Table ----

let resultsData = [];

async function loadResults() {
    if (!currentRunDir) return;
    try {
        const data = await api(`/api/results/${currentRunDir}`);
        resultsData = data.results || [];
        sortAndRenderResults();
    } catch {
        resultsData = [];
        renderResults([]);
    }
}

async function loadReviews() {
    // Reviews are loaded on demand when clicking "View"
}

function sortAndRenderResults() {
    let sorted = [...resultsData];
    if (sortColumn) {
        sorted.sort((a, b) => {
            let va = a[sortColumn], vb = b[sortColumn];
            if (typeof va === "string") va = va.toLowerCase();
            if (typeof vb === "string") vb = vb.toLowerCase();
            if (va < vb) return sortAsc ? -1 : 1;
            if (va > vb) return sortAsc ? 1 : -1;
            return 0;
        });
    }
    renderResults(sorted);
}

function renderResults(results) {
    resultsTbody.innerHTML = "";
    noResults.style.display = results.length ? "none" : "block";

    for (const r of results) {
        const tr = document.createElement("tr");

        const scoreClass = r.overall_score >= 70 ? "score-high" : r.overall_score >= 50 ? "score-mid" : "score-low";
        const recClass = getRecClass(r.recommendation);

        tr.innerHTML = `
            <td>${escapeHtml(r.paper_filename || r.title || "")}</td>
            <td><span class="score-badge ${scoreClass}">${r.overall_score?.toFixed(1) || "-"}</span></td>
            <td><span class="rec-badge ${recClass}">${escapeHtml(r.recommendation || "-")}</span></td>
            <td>$${r.total_cost_usd?.toFixed(4) || "-"}</td>
            <td>${r.confidence != null ? (r.confidence * 100).toFixed(0) + "%" : "-"}</td>
            <td><button class="btn-view" data-paper="${escapeHtml(r.paper_filename || "")}">View</button></td>
        `;
        resultsTbody.appendChild(tr);
    }

    // Attach view handlers
    resultsTbody.querySelectorAll(".btn-view").forEach(btn => {
        btn.addEventListener("click", () => showReview(btn.dataset.paper));
    });
}

async function showReview(paperFilename) {
    if (!currentRunDir) return;
    reviewModalTitle.textContent = `Review: ${paperFilename}`;
    reviewModalBody.innerHTML = "Loading...";

    try {
        // Find the review file for this paper
        const data = await api(`/api/reviews/${currentRunDir}`);
        const reviews = data.reviews || [];
        const match = reviews.find(r => r.filename.toLowerCase().includes(paperFilename.toLowerCase()));
        if (!match) {
            reviewModalBody.innerHTML = "Review file not found";
            return;
        }

        const res = await fetch(`/api/review/${currentRunDir}/${match.filename}`);
        const text = await res.text();
        reviewModalBody.innerHTML = renderMarkdown(text);
        reviewModal.style.display = "flex";
    } catch (e) {
        reviewModalBody.innerHTML = "Error loading review: " + e.message;
    }
}

function getRecClass(rec) {
    if (!rec) return "";
    const l = rec.toLowerCase();
    if (l.includes("accept") && !l.includes("revision")) return "rec-accept";
    if (l.includes("accept with revision") || l.includes("minor")) return "rec-revision";
    if (l.includes("revise") || l.includes("resubmit") || l.includes("major")) return "rec-resubmit";
    if (l.includes("reject")) return "rec-reject";
    return "";
}

// ---- Stage Indicators ----

const stageMap = {
    "Ingestion": "stage-ingest",
    "Librarian": "stage-ingest",
    "Extraction": "stage-extraction",
    "Fact-Check": "stage-extraction",
    "Synthesis": "stage-synthesis",
    "Output": "stage-output",
};

function activateStage(name) {
    const id = stageMap[name];
    if (id) {
        const el = document.getElementById(id);
        el?.classList.add("active");
        el?.classList.remove("completed");
    }
}

function completeStage(name) {
    const id = stageMap[name];
    if (id) {
        const el = document.getElementById(id);
        el?.classList.remove("active");
        el?.classList.add("completed");
    }
}

function clearStages() {
    document.querySelectorAll(".stage-item").forEach(el => {
        el.classList.remove("active", "completed");
    });
}

// ---- Log ----

function addLog(message, level = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry log-${level}`;
    const now = new Date();
    const time = now.toLocaleTimeString("en-US", { hour12: false });
    entry.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${escapeHtml(message)}</span>`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

// ---- Helpers ----

function startElapsedTimer() {
    if (elapsedInterval) clearInterval(elapsedInterval);
    elapsedInterval = setInterval(() => {
        if (runStartTime) {
            const s = (Date.now() - runStartTime) / 1000;
            elapsedDisplay.textContent = formatDuration(s);
        }
    }, 1000);
}

function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
    // Simple markdown-to-HTML for display
    let html = escapeHtml(text);
    // Headers
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // Lists
    html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
    html = html.replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>");
    // Paragraphs (double newline)
    html = html.replace(/\n\n/g, "</p><p>");
    html = "<p>" + html + "</p>";
    // Code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Horizontal rule
    html = html.replace(/^---$/gm, "<hr>");
    return html;
}

// Init
document.addEventListener("DOMContentLoaded", init);
