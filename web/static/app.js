/**
 * Academic Review System - Dashboard Frontend
 */

// State
let currentRunDir = "";
let currentRunId = null;
let eventSource = null;
let runStartTime = null;
let elapsedInterval = null;
let sortColumn = "display_date";
let sortAsc = false;  // default: newest first
let promptFiles = [];        // [{filename, content}, ...]
let currentPromptFilename = "";
let configDirty = {};        // {key: value} pending saves
let originalConfig = {};     // snapshot before edits

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
const reviewModal = document.getElementById("review-modal");
const reviewModalTitle = document.getElementById("review-modal-title");
const reviewModalBody = document.getElementById("review-modal-body");
const reviewModalClose = document.getElementById("review-modal-close");
const promptSelector = document.getElementById("prompt-selector");
const promptsEditor = document.getElementById("prompts-editor");
const promptVarsHint = document.getElementById("prompt-vars-hint");
const promptSaveBtn = document.getElementById("prompt-save-btn");
const promptReloadBtn = document.getElementById("prompt-reload-btn");
const promptValidation = document.getElementById("prompt-validation");
const criteriaEditorWrap = document.getElementById("criteria-editor-wrap");
const criteriaSaveBtn = document.getElementById("criteria-save-btn");
const criteriaReloadBtn = document.getElementById("criteria-reload-btn");
const criteriaValidation = document.getElementById("criteria-validation");
const sourcesEditor = document.getElementById("sources-editor");
const sourcesSaveBtn = document.getElementById("sources-save-btn");
const sourcesReloadBtn = document.getElementById("sources-reload-btn");
const sourcesValidation = document.getElementById("sources-validation");
const configSaveBtn = document.getElementById("config-save-btn");
const judgeBtn = document.getElementById("judge-btn");
const viewVerdictsBtn = document.getElementById("view-verdicts-btn");
const judgeModal = document.getElementById("judge-modal");
const judgeModalBody = document.getElementById("judge-modal-body");
const judgeModalClose = document.getElementById("judge-modal-close");

// ---- Initialization ----

async function init() {
    await loadRuns();
    runSelector.addEventListener("change", onRunSelected);
    startBtn.addEventListener("click", startRun);
    stopBtn.addEventListener("click", stopRun);
    reviewModalClose.addEventListener("click", () => reviewModal.style.display = "none");

    // Sidebar tabs
    document.querySelectorAll(".sidebar-tab").forEach(tab => {
        tab.addEventListener("click", () => switchSidebarTab(tab.dataset.tab));
    });

    // Prompt editor
    promptSelector.addEventListener("change", onPromptSelected);
    promptSaveBtn.addEventListener("click", savePrompt);
    promptReloadBtn.addEventListener("click", reloadPrompt);

    // Criteria editor
    criteriaSaveBtn.addEventListener("click", saveCriteria);
    criteriaReloadBtn.addEventListener("click", reloadCriteria);

    // Sources editor
    sourcesSaveBtn.addEventListener("click", saveSources);
    sourcesReloadBtn.addEventListener("click", reloadSources);

    // Config save
    configSaveBtn.addEventListener("click", saveConfig);

    // Judge
    judgeBtn.addEventListener("click", startJudge);
    judgeModalClose.addEventListener("click", () => judgeModal.style.display = "none");
    if (viewVerdictsBtn) viewVerdictsBtn.addEventListener("click", showJudgeVerdicts);

    // Load global resources
    await loadPrompts();
    await loadSources();

    // Sort headers
    document.querySelectorAll("#results-table th[data-sort]").forEach(th => {
        th.addEventListener("click", () => {
            const col = th.dataset.sort;
            if (sortColumn === col) sortAsc = !sortAsc;
            else { sortColumn = col; sortAsc = true; }
            sortAndRenderResults();
        });
    });

    // Refresh reports button
    document.getElementById("refresh-reports-btn").addEventListener("click", loadResults);
}

// ---- API helpers ----

async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    if (!res.ok) {
        let detail = `${res.status}: ${res.statusText}`;
        try {
            const body = await res.json();
            if (body.detail) detail = body.detail;
        } catch {}
        throw new Error(detail);
    }
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
        criteriaEditorWrap.innerHTML = '<p class="placeholder-text">Select a run directory to edit criteria</p>';
        return;
    }

    // Load editable config
    configDirty = {};
    await loadConfig();

    // Load criteria as raw YAML
    await loadCriteria();

    // Load existing results
    await loadResults();
    await loadReviews();
    await checkJudgeVerdicts();
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
    judgeBtn.disabled = false;
    if (elapsedInterval) clearInterval(elapsedInterval);
    if (eventSource) { eventSource.close(); eventSource = null; }
    loadResults();
    checkJudgeVerdicts();
}

// ---- Reports Table ----

let reportsData = [];

async function loadResults() {
    if (!currentRunDir) return;
    try {
        const data = await api(`/api/all-reviews/${currentRunDir}`);
        reportsData = data.reviews || [];
        sortAndRenderResults();
    } catch {
        reportsData = [];
        renderResults([]);
    }
}

async function loadReviews() {
    // Reviews are loaded on demand when clicking "View"
}

function sortAndRenderResults() {
    let sorted = [...reportsData];
    if (sortColumn) {
        sorted.sort((a, b) => {
            let va = a[sortColumn], vb = b[sortColumn];
            if (va == null) va = "";
            if (vb == null) vb = "";
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
    judgeBtn.disabled = results.length === 0;

    for (const r of results) {
        const tr = document.createElement("tr");

        const scoreVal = r.overall_score;
        const scoreClass = scoreVal != null && scoreVal !== "" ? (scoreVal >= 70 ? "score-high" : scoreVal >= 50 ? "score-mid" : "score-low") : "";
        const recClass = getRecClass(r.recommendation);

        tr.innerHTML = `
            <td>${escapeHtml(r.paper || "")}</td>
            <td class="td-date">${escapeHtml(r.display_date || "")}</td>
            <td class="td-model">${escapeHtml(r.extractor_model || "")}</td>
            <td class="td-model">${escapeHtml(r.synthesizer_model || "")}</td>
            <td><span class="score-badge ${scoreClass}">${scoreVal != null && scoreVal !== "" ? Number(scoreVal).toFixed(1) : "-"}</span></td>
            <td><span class="rec-badge ${recClass}">${escapeHtml(r.recommendation || "-")}</span></td>
            <td>${r.total_cost != null && r.total_cost !== "" ? "$" + Number(r.total_cost).toFixed(4) : "-"}</td>
            <td>${r.confidence != null && r.confidence !== "" ? (Number(r.confidence) * 100).toFixed(0) + "%" : "-"}</td>
            <td><button class="btn-view" data-filename="${escapeHtml(r.filename)}">View</button></td>
        `;
        resultsTbody.appendChild(tr);
    }

    // Attach view handlers — direct filename match, no guessing
    resultsTbody.querySelectorAll(".btn-view").forEach(btn => {
        btn.addEventListener("click", () => showReviewByFilename(btn.dataset.filename));
    });
}

async function showReviewByFilename(filename) {
    if (!currentRunDir) return;
    const paperName = filename.split("_20")[0] || filename;
    reviewModalTitle.textContent = `Review: ${paperName}`;
    reviewModalBody.innerHTML = "Loading...";

    try {
        const res = await fetch(`/api/review/${currentRunDir}/${filename}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
    "Judge-Compare": "stage-extraction",
    "Judge-Adjudicate": "stage-synthesis",
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

// ---- Sidebar Tab Switching ----

function switchSidebarTab(tabName) {
    document.querySelectorAll(".sidebar-tab").forEach(t => t.classList.remove("active"));
    document.querySelector(`.sidebar-tab[data-tab="${tabName}"]`).classList.add("active");
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    document.getElementById(`tab-${tabName}`).classList.add("active");
}

// ---- Config Editor ----

const PROVIDER_OPTIONS = ["openai", "anthropic", "deepseek", "google", "gemini", "mistral", "ollama"];
const PROVIDER_KEYS = ["PROVIDER_EXTRACTION", "PROVIDER_SYNTHESIS", "JUDGE_PROVIDER"];
const NUMBER_KEYS = ["TEMPERATURE", "TEMPERATURE_EXTRACTION", "TEMPERATURE_SYNTHESIS", "JUDGE_TEMPERATURE", "MAX_TOKENS_EXTRACTION", "MAX_TOKENS_SYNTHESIS", "EXTRACTION_BATCH_SIZE", "MAX_PARALLEL_EXTRACTIONS", "MAX_RETRIES", "CONCURRENCY"];

async function loadConfig() {
    try {
        const data = await api(`/api/config/${currentRunDir}`);
        originalConfig = data.config || {};
        renderConfigEditor(originalConfig);
    } catch {
        configContent.innerHTML = '<p class="placeholder-text">Could not load config</p>';
    }
}

function renderConfigEditor(config) {
    let html = "";
    for (const [k, v] of Object.entries(config)) {
        const isMasked = v === "***masked***";
        if (isMasked) {
            html += `<div class="config-item config-locked">
                <span class="config-key">${escapeHtml(k)}</span>
                <span class="config-value locked">***<span class="lock-hint">edit .env directly</span></span>
            </div>`;
            continue;
        }
        const isProvider = PROVIDER_KEYS.includes(k);
        const isNumber = NUMBER_KEYS.includes(k);
        let inputHtml;
        if (isProvider) {
            const opts = PROVIDER_OPTIONS.map(o =>
                `<option value="${o}" ${o === v ? "selected" : ""}>${o}</option>`
            ).join("");
            inputHtml = `<select class="config-input config-select" data-key="${escapeHtml(k)}">${opts}</select>`;
        } else if (isNumber) {
            const step = k.includes("TEMPERATURE") ? 'step="0.1"' : "";
            inputHtml = `<input type="number" class="config-input config-number" data-key="${escapeHtml(k)}" value="${escapeHtml(v)}" ${step}>`;
        } else {
            inputHtml = `<input type="text" class="config-input config-text" data-key="${escapeHtml(k)}" value="${escapeHtml(v)}">`;
        }
        html += `<div class="config-item editable-config">
            <span class="config-key">${escapeHtml(k)}</span>
            ${inputHtml}
        </div>`;
    }
    configContent.innerHTML = html || '<p class="placeholder-text">No config found</p>';

    // Track changes
    configContent.querySelectorAll(".config-input").forEach(el => {
        el.addEventListener("change", () => {
            // Always store as string to avoid NaN in JSON
            configDirty[el.dataset.key] = String(el.value);
            configSaveBtn.style.display = Object.keys(configDirty).length ? "inline-block" : "none";
        });
    });
}

async function saveConfig() {
    if (!currentRunDir || !Object.keys(configDirty).length) return;
    try {
        // Send each changed key one by one (API handles one key at a time)
        for (const [key, value] of Object.entries(configDirty)) {
            const data = await api(`/api/config/${currentRunDir}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key, value }),
            });
            originalConfig = data.config || originalConfig;
        }
        configDirty = {};
        renderConfigEditor(originalConfig);
        configSaveBtn.style.display = "none";
        addLog("Config saved", "success");
    } catch (e) {
        addLog("Failed to save config: " + e.message, "error");
    }
}

// ---- Criteria Editor ----

let criteriaTextarea = null;

async function loadCriteria() {
    if (!currentRunDir) return;
    try {
        const data = await api(`/api/criteria-raw/${currentRunDir}`);
        criteriaEditorWrap.innerHTML = "";
        criteriaTextarea = document.createElement("textarea");
        criteriaTextarea.className = "code-editor";
        criteriaTextarea.value = data.content || "";
        criteriaEditorWrap.appendChild(criteriaTextarea);
        criteriaSaveBtn.style.display = "inline-block";
        criteriaReloadBtn.style.display = "inline-block";
        criteriaValidation.textContent = "";
    } catch (e) {
        criteriaEditorWrap.innerHTML = `<p class="placeholder-text">Could not load criteria: ${escapeHtml(e.message)}</p>`;
        criteriaSaveBtn.style.display = "none";
        criteriaReloadBtn.style.display = "none";
    }
}

async function saveCriteria() {
    if (!currentRunDir || !criteriaTextarea) return;
    criteriaValidation.textContent = "";
    try {
        await api(`/api/criteria/${currentRunDir}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: criteriaTextarea.value }),
        });
        criteriaValidation.textContent = "Saved!";
        criteriaValidation.className = "validation-msg success";
        addLog("Criteria saved", "success");
    } catch (e) {
        criteriaValidation.textContent = e.message;
        criteriaValidation.className = "validation-msg error";
    }
}

async function reloadCriteria() {
    await loadCriteria();
}

// ---- Prompts Editor ----

const PROMPT_TEMPLATE_VARS = {
    "extractor_system.txt": "No template variables (system instructions)",
    "extractor_user.txt": "{domain}, {paper_markdown}, {criterion_name}, {criterion_description}, {scale_definition}",
    "synthesizer_system.txt": "No template variables (system instructions)",
    "synthesizer_user.txt": "{paper_title}, {paper_abstract}, {json_dump_of_extractions}, {weights_table}, {calculated_score}, {calculated_recommendation}",
};

async function loadPrompts() {
    try {
        const data = await api("/api/prompts");
        promptFiles = data.prompts || [];
        promptSelector.innerHTML = "";
        for (const p of promptFiles) {
            const opt = document.createElement("option");
            opt.value = p.filename;
            opt.textContent = p.filename.replace(/\.txt$/, "");
            promptSelector.appendChild(opt);
        }
        if (promptFiles.length > 0) {
            promptSelector.value = promptFiles[0].filename;
            onPromptSelected();
        }
    } catch {
        promptSelector.innerHTML = '<option value="">Error loading prompts</option>';
    }
}

function onPromptSelected() {
    const fn = promptSelector.value;
    if (!fn) return;
    currentPromptFilename = fn;
    const file = promptFiles.find(p => p.filename === fn);
    promptsEditor.value = file ? file.content : "";
    promptVarsHint.textContent = PROMPT_TEMPLATE_VARS[fn] || "";
    promptSaveBtn.disabled = false;
    promptValidation.textContent = "";
}

async function savePrompt() {
    if (!currentPromptFilename) return;
    promptValidation.textContent = "";
    try {
        await api(`/api/prompts/${currentPromptFilename}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: promptsEditor.value }),
        });
        // Update local cache
        const idx = promptFiles.findIndex(p => p.filename === currentPromptFilename);
        if (idx >= 0) promptFiles[idx].content = promptsEditor.value;
        promptValidation.textContent = "Saved!";
        promptValidation.className = "validation-msg success";
        addLog(`Prompt '${currentPromptFilename}' saved`, "success");
    } catch (e) {
        promptValidation.textContent = e.message;
        promptValidation.className = "validation-msg error";
    }
}

async function reloadPrompt() {
    try {
        const data = await api("/api/prompts");
        promptFiles = data.prompts || [];
        onPromptSelected();
        promptValidation.textContent = "Reloaded from disk";
        promptValidation.className = "validation-msg success";
    } catch (e) {
        promptValidation.textContent = e.message;
        promptValidation.className = "validation-msg error";
    }
}

// ---- Literature Sources Editor ----

async function loadSources() {
    try {
        const data = await api("/api/literature-sources");
        sourcesEditor.value = data.content || "";
        sourcesValidation.textContent = "";
    } catch {
        sourcesEditor.value = "";
        sourcesEditor.placeholder = "Could not load literature sources";
    }
}

async function saveSources() {
    sourcesValidation.textContent = "";
    try {
        await api("/api/literature-sources", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: sourcesEditor.value }),
        });
        sourcesValidation.textContent = "Saved!";
        sourcesValidation.className = "validation-msg success";
        addLog("Literature sources saved", "success");
    } catch (e) {
        sourcesValidation.textContent = e.message;
        sourcesValidation.className = "validation-msg error";
    }
}

async function reloadSources() {
    await loadSources();
}

// ---- Judge ----

async function checkJudgeVerdicts() {
    if (!currentRunDir) return;
    try {
        const data = await api(`/api/judge/results/${currentRunDir}`);
        const verdicts = data.verdicts || [];
        if (viewVerdictsBtn) viewVerdictsBtn.style.display = verdicts.length > 0 ? "inline-block" : "none";
    } catch {
        if (viewVerdictsBtn) viewVerdictsBtn.style.display = "none";
    }
}

async function startJudge() {
    if (!currentRunDir) return;
    judgeBtn.disabled = true;
    logContent.innerHTML = "";
    clearStages();

    try {
        const data = await api(`/api/judge/${currentRunDir}`, { method: "POST" });
        currentRunId = data.run_id;
        runStartTime = Date.now();
        connectSSE();
        startElapsedTimer();
        addLog("Judge pipeline started...", "info");
    } catch (e) {
        const msg = e.message || "Unknown error";
        if (msg.includes("2 consolidated") || msg.includes("400")) {
            addLog("Cannot judge: run reviews with at least 2 different models first, then compare.", "warning");
        } else {
            addLog("Judge failed: " + msg, "error");
        }
        judgeBtn.disabled = false;
    }
}

async function showJudgeVerdicts() {
    if (!currentRunDir) return;
    judgeModalBody.innerHTML = "Loading verdicts...";

    try {
        const data = await api(`/api/judge/results/${currentRunDir}`);
        const verdicts = data.verdicts || [];
        if (!verdicts.length) {
            judgeModalBody.innerHTML = '<p class="placeholder-text">No judge verdicts yet. Run "Compare & Judge" first.</p>';
            judgeModal.style.display = "flex";
            return;
        }

        let html = '<table class="results-table judge-table"><thead><tr>';
        html += "<th>Paper</th><th>Judge Decision</th><th>Winner</th><th>Rationale</th><th>Cost</th>";
        html += "</tr></thead><tbody>";

        for (const v of verdicts) {
            const winClass = v.winning_review === "A" ? "rec-accept" :
                             v.winning_review === "B" ? "rec-reject" : "rec-revision";
            html += "<tr>";
            html += `<td>${escapeHtml(v.paper_filename || "")}</td>`;
            html += `<td><span class="rec-badge rec-revision">${escapeHtml(v.judge_recommendation || "")}</span></td>`;
            html += `<td><span class="rec-badge ${winClass}">${escapeHtml(v.winning_review || "")}</span></td>`;
            html += `<td>${escapeHtml(v.judge_rationale || "")}</td>`;
            html += `<td>$${parseFloat(v.judge_cost || 0).toFixed(4)}</td>`;
            html += "</tr>";
        }

        html += "</tbody></table>";
        judgeModalBody.innerHTML = html;
        judgeModal.style.display = "flex";
    } catch (e) {
        judgeModalBody.innerHTML = "Error: " + e.message;
    }
}

// Init
document.addEventListener("DOMContentLoaded", init);
