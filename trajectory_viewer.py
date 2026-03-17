"""Trajectory viewer — a local web UI for browsing agent trajectory files.

Usage:
    uv run python trajectory_viewer.py [--port 8501] [--results-dir results/]

Opens a browser with a UI showing:
- Run selector (each subdirectory in results/)
- Overall config (config.yaml) at the top
- Session list with metadata
- Expandable conversation view per session
"""

import argparse
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def find_sessions(run_dir: Path) -> list[dict]:
    """Find all trajectory JSON files under a run directory."""
    sessions = []
    for traj_file in sorted(run_dir.rglob("*_trajectory.json")):
        try:
            with open(traj_file) as f:
                data = json.load(f)
            # Truncate long tool results to keep HTML size reasonable
            for step in data.get("trajectory", []):
                if step.get("tool_result") and len(step["tool_result"]) > 2000:
                    step["tool_result"] = (
                        step["tool_result"][:2000] + "\n\n... [truncated] ..."
                    )
            sessions.append(data)
        except (json.JSONDecodeError, ValueError):
            continue  # skip empty or malformed files
    return sessions


def load_config(run_dir: Path) -> dict | None:
    """Load config.yaml from a run directory if it exists."""
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


def load_summary(run_dir: Path) -> dict | None:
    """Parse outputs_judged.summary.txt into a dict of stats."""
    summary_path = run_dir / "outputs_judged.summary.txt"
    if not summary_path.exists():
        return None
    import re

    text = summary_path.read_text()
    stats: dict = {}
    # Total items
    m = re.search(r"Total items:\s+(\d+)", text)
    if m:
        stats["total"] = int(m.group(1))
    # Correct
    m = re.search(r"Correct:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", text)
    if m:
        stats["correct"] = int(m.group(1))
        stats["correct_pct"] = float(m.group(3))
    # Incorrect
    m = re.search(r"Incorrect:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", text)
    if m:
        stats["incorrect"] = int(m.group(1))
        stats["incorrect_pct"] = float(m.group(3))
    # No answer
    m = re.search(r"No answer:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", text)
    if m:
        stats["no_answer"] = int(m.group(1))
        stats["no_answer_pct"] = float(m.group(3))
    # Avg steps
    m = re.search(r"Avg steps:\s+([\d.]+)", text)
    if m:
        stats["avg_steps"] = float(m.group(1))
    # Avg tool calls
    m = re.search(r"Avg tool calls:\s+([\d.]+)", text)
    if m:
        stats["avg_tool_calls"] = float(m.group(1))
    # Judge model
    m = re.search(r"Judge model:\s+(\S+)", text)
    if m:
        stats["judge_model"] = m.group(1)
    return stats if stats else None


def load_run_data(run_dir: Path) -> dict:
    """Load config and sessions for a single run directory."""
    config = load_config(run_dir)
    sessions = find_sessions(run_dir)
    summary = load_summary(run_dir)

    # Merge judge verdicts from outputs_judged.jsonl into session metadata
    judged_path = run_dir / "outputs_judged.jsonl"
    if judged_path.exists():
        verdicts: dict[str, str] = {}
        with open(judged_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    fid = row.get("financebench_id", "")
                    verdict = row.get("judge_verdict", "")
                    if fid and verdict:
                        verdicts[fid] = verdict
                except (json.JSONDecodeError, ValueError):
                    continue
        for session in sessions:
            task_id = (session.get("metadata") or {}).get("task_id", "")
            if task_id and task_id in verdicts:
                session["metadata"]["judge_verdict"] = verdicts[task_id]

    return {"config": config, "sessions": sessions, "summary": summary}


def build_html(results_dir: Path) -> str:
    """Build the full single-page HTML app.

    Only embeds run names; data is loaded lazily via fetch() or inline for
    static-file mode.
    """
    # Discover runs
    runs = sorted(
        [d.name for d in results_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    runs_json = json.dumps(runs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trajectory Viewer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #eef1f5; color: #333; }}

/* Header */
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #fff; padding: 18px 28px; display: flex; align-items: center; gap: 20px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }}
.header h1 {{ font-size: 20px; font-weight: 700; white-space: nowrap; letter-spacing: -0.3px; }}
.header h1::before {{ content: '\1f9e0'; margin-right: 8px; }}
.header select {{ padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.1); color: #eee; font-size: 14px; max-width: 700px; flex: 1; backdrop-filter: blur(4px); cursor: pointer; transition: background 0.2s; }}
.header select:hover {{ background: rgba(255,255,255,0.18); }}
.header select:focus {{ outline: none; border-color: rgba(255,255,255,0.35); background: rgba(255,255,255,0.15); }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}

/* Config panel */
.config-panel {{ background: #fff; border: 1px solid #dde1e6; border-radius: 10px; padding: 0; margin-bottom: 20px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.config-panel h2 {{ font-size: 13px; font-weight: 700; padding: 12px 20px; color: #fff; background: linear-gradient(135deg, #1a1a2e, #16213e); text-transform: uppercase; letter-spacing: 0.8px; }}
.config-groups {{ display: flex; flex-direction: column; }}
.config-group {{ border-bottom: 1px solid #eef0f3; }}
.config-group:last-child {{ border-bottom: none; }}
.config-group-header {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; background: #f1f5f9; padding: 8px 20px; border-bottom: 1px solid #e2e8f0; cursor: pointer; display: flex; align-items: center; gap: 8px; user-select: none; transition: background 0.15s; }}
.config-group-header:hover {{ background: #e2e8f0; }}
.config-group-header .config-chevron {{ font-size: 10px; color: #94a3b8; transition: transform 0.2s; }}
.config-group-header.open .config-chevron {{ transform: rotate(90deg); }}
.config-grid {{ display: none; width: 100%; font-size: 13px; border-collapse: collapse; }}
.config-grid.open {{ display: table; }}
.config-item {{ display: table-row; transition: background 0.15s; }}
.config-item:nth-child(odd) {{ background: #f7f8fa; }}
.config-item:nth-child(even) {{ background: #fff; }}
.config-item:hover {{ background: #edf0f5; }}
.config-key {{ display: table-cell; font-weight: 600; color: #64748b; white-space: nowrap; padding: 7px 16px 7px 36px; vertical-align: top; border-bottom: 1px solid #f0f0f0; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }}
.config-val {{ display: table-cell; color: #1e293b; padding: 7px 20px 7px 0; word-break: break-word; border-bottom: 1px solid #f0f0f0; }}

/* Summary bar */
.summary-bar {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.summary-stat {{ background: #fff; border: 1px solid #dde1e6; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); text-align: center; transition: transform 0.15s, box-shadow 0.15s; }}
.summary-stat:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
.summary-stat .label {{ display: block; font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; margin-bottom: 4px; }}
.summary-stat .value {{ display: block; font-size: 22px; font-weight: 700; color: #1e293b; }}
.summary-stat.stat-success .value {{ color: #16a34a; }}
.summary-stat.stat-warn .value {{ color: #d97706; }}
.summary-stat.stat-error .value {{ color: #dc2626; }}
.summary-stat.stat-info .value {{ color: #2563eb; }}
.summary-stat .value-pct {{ display: block; font-size: 12px; color: #94a3b8; font-weight: 500; }}

/* Session list */
.session-card {{ background: #fff; border: 1px solid #dde1e6; border-radius: 10px; margin-bottom: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.03); transition: box-shadow 0.2s; }}
.session-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
.session-header {{ padding: 14px 20px; cursor: pointer; display: flex; align-items: center; gap: 14px; transition: background 0.15s; }}
.session-header:hover {{ background: #f8f9fb; }}
.session-header .query {{ flex: 1; font-size: 14px; font-weight: 500; color: #1e293b; line-height: 1.4; }}
.session-header .badges {{ display: flex; gap: 6px; flex-shrink: 0; }}
.badge {{ padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.2px; }}
.badge-success {{ background: #dcfce7; color: #15803d; }}
.badge-error {{ background: #fee2e2; color: #b91c1c; }}
.badge-warn {{ background: #fef3c7; color: #b45309; }}
.badge-steps {{ background: #e0e7ff; color: #4338ca; }}
.chevron {{ font-size: 12px; color: #94a3b8; transition: transform 0.2s; flex-shrink: 0; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; border-radius: 4px; }}
.chevron.open {{ transform: rotate(90deg); color: #475569; }}

/* Trajectory view */
.trajectory {{ display: none; padding: 4px 20px 20px; border-top: 1px solid #f0f0f0; background: #f8f9fb; }}
.trajectory.open {{ display: block; }}
.step {{ margin-top: 10px; padding: 14px 18px; border-radius: 10px; font-size: 13px; line-height: 1.7; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
.step-user {{ background: #eff6ff; border-left: 4px solid #3b82f6; }}
.step-assistant {{ background: #faf5ff; border-left: 4px solid #8b5cf6; }}
.step-tool {{ background: #fffbeb; border-left: 4px solid #f59e0b; }}
.step-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: inline-block; padding: 2px 8px; border-radius: 4px; }}
.step-user .step-label {{ color: #1d4ed8; background: rgba(59,130,246,0.1); }}
.step-assistant .step-label {{ color: #7c3aed; background: rgba(139,92,246,0.1); }}
.step-tool .step-label {{ color: #d97706; background: rgba(245,158,11,0.1); }}
.step-content {{ white-space: pre-wrap; word-break: break-word; color: #334155; }}
.tool-call-box {{ background: rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.06); border-radius: 8px; padding: 12px 14px; margin-top: 8px; font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 12px; line-height: 1.5; }}
.tool-name {{ font-weight: 700; color: #c2410c; }}
.tool-args {{ color: #475569; }}
.tool-result-content {{ max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; font-size: 12px; color: #475569; }}
.tool-result-toggle {{ cursor: pointer; color: #2563eb; font-size: 12px; margin-top: 6px; display: inline-block; font-weight: 500; }}
.tool-result-toggle:hover {{ text-decoration: underline; }}
.session-id {{ font-size: 11px; color: #94a3b8; font-family: 'SF Mono', 'Fira Code', monospace; padding: 8px 0 4px; }}

/* Search & filter */
.filter-bar {{ display: flex; gap: 12px; margin-bottom: 18px; align-items: center; }}
.filter-bar input {{ flex: 1; padding: 10px 16px; border: 1px solid #dde1e6; border-radius: 8px; font-size: 14px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: border-color 0.2s, box-shadow 0.2s; }}
.filter-bar input:focus {{ outline: none; border-color: #93c5fd; box-shadow: 0 0 0 3px rgba(59,130,246,0.12); }}
.filter-bar select {{ padding: 10px 14px; border: 1px solid #dde1e6; border-radius: 8px; font-size: 13px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.04); cursor: pointer; }}
.filter-bar select:focus {{ outline: none; border-color: #93c5fd; }}

/* Loading state */
.loading {{ text-align: center; color: #94a3b8; padding: 60px 20px; font-size: 15px; }}
</style>
</head>
<body>

<div class="header">
    <h1>Trajectory Viewer</h1>
    <select id="runSelect" onchange="selectRun(this.value)">
        <option value="">-- Select a run --</option>
    </select>
</div>

<div class="container">
    <div id="configPanel" class="config-panel" style="display:none;"></div>
    <div id="summaryBar" class="summary-bar" style="display:none;"></div>
    <div class="filter-bar">
        <input type="text" id="searchInput" placeholder="Search queries..." oninput="filterSessions()">
        <select id="statusFilter" onchange="filterSessions()">
            <option value="all">All statuses</option>
            <option value="correct">Correct</option>
            <option value="incorrect">Incorrect</option>
            <option value="no_answer">No answer</option>
        </select>
    </div>
    <div id="sessionList"></div>
</div>

<script>
const RUNS = {runs_json};
const DATA_CACHE = {{}};

// Populate run selector
const runSelect = document.getElementById('runSelect');
RUNS.forEach(name => {{
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    runSelect.appendChild(opt);
}});

// Auto-select first run
if (RUNS.length > 0) {{
    runSelect.value = RUNS[0];
    selectRun(RUNS[0]);
}}

let currentSessions = [];

async function selectRun(runName) {{
    if (!runName) return;

    // Fetch run data lazily
    if (!DATA_CACHE[runName]) {{
        document.getElementById('sessionList').innerHTML = '<div class="loading">Loading run data...</div>';
        try {{
            const resp = await fetch('/api/run/' + encodeURIComponent(runName));
            DATA_CACHE[runName] = await resp.json();
        }} catch(e) {{
            document.getElementById('sessionList').innerHTML = '<p style="color:red;padding:20px;">Failed to load run data.</p>';
            return;
        }}
    }}

    const run = DATA_CACHE[runName];
    currentSessions = run.sessions;

    // Render config
    const cp = document.getElementById('configPanel');
    if (run.config) {{
        cp.style.display = 'block';
        cp.innerHTML = renderConfig(run.config);
    }} else {{
        // Infer config from first session metadata
        if (run.sessions.length > 0) {{
            cp.style.display = 'block';
            cp.innerHTML = renderConfig(run.sessions[0].metadata, true);
        }} else {{
            cp.style.display = 'none';
        }}
    }}

    // Summary
    renderSummary(run.summary, run.sessions);

    // Sessions
    renderSessions(run.sessions);
}}

function renderConfig(config, fromMetadata) {{
    let title = fromMetadata ? 'Run Config (inferred from session metadata)' : 'Run Config';

    // Flatten into key-value pairs
    const entries = [];
    function flatten(obj, prefix) {{
        for (const [k, v] of Object.entries(obj)) {{
            const key = prefix ? prefix + '.' + k : k;
            if (v && typeof v === 'object' && !Array.isArray(v)) {{
                flatten(v, key);
            }} else {{
                const val = Array.isArray(v) ? v.join(', ') : String(v);
                entries.push({{ key, val }});
            }}
        }}
    }}
    flatten(config, '');

    // Group by first segment (before the dot), or "general" for top-level keys
    const groups = {{}};
    entries.forEach(e => {{
        const dotIdx = e.key.indexOf('.');
        const group = dotIdx > 0 ? e.key.substring(0, dotIdx) : 'general';
        const shortKey = dotIdx > 0 ? e.key.substring(dotIdx + 1) : e.key;
        if (!groups[group]) groups[group] = [];
        groups[group].push({{ key: shortKey, val: e.val }});
    }});

    let html = `<h2>${{title}}</h2><div class="config-groups">`;
    let gIdx = 0;
    for (const [group, items] of Object.entries(groups)) {{
        const isGeneral = group === 'general';
        const openCls = isGeneral ? ' open' : '';
        html += `<div class="config-group">`;
        html += `<div class="config-group-header${{openCls}}" onclick="toggleConfigGroup(this)"><span class="config-chevron">&#9654;</span>${{esc(group)}}</div>`;
        html += `<div class="config-grid${{openCls}}">`;
        items.forEach(item => {{
            html += `<div class="config-item"><span class="config-key">${{esc(item.key)}}</span><span class="config-val">${{esc(item.val)}}</span></div>`;
        }});
        html += `</div></div>`;
        gIdx++;
    }}
    html += `</div>`;
    return html;
}}

function renderSummary(summary, sessions) {{
    const bar = document.getElementById('summaryBar');
    if (!summary && sessions.length === 0) {{ bar.style.display = 'none'; return; }}
    bar.style.display = 'grid';

    if (summary) {{
        const total = summary.total || sessions.length;
        const correct = summary.correct ?? '—';
        const correctPct = summary.correct_pct != null ? ` (${{summary.correct_pct}}%)` : '';
        const incorrect = summary.incorrect ?? '—';
        const incorrectPct = summary.incorrect_pct != null ? ` (${{summary.incorrect_pct}}%)` : '';
        const noAnswer = summary.no_answer ?? '—';
        const noAnswerPct = summary.no_answer_pct != null ? ` (${{summary.no_answer_pct}}%)` : '';
        const avgSteps = summary.avg_steps != null ? summary.avg_steps : '—';
        const avgToolCalls = summary.avg_tool_calls != null ? summary.avg_tool_calls : '—';
        bar.innerHTML = `
            <div class="summary-stat"><span class="label">Total</span><span class="value">${{total}}</span></div>
            <div class="summary-stat stat-success"><span class="label">Correct</span><span class="value">${{correct}}</span><span class="value-pct">${{correctPct}}</span></div>
            <div class="summary-stat stat-error"><span class="label">Incorrect</span><span class="value">${{incorrect}}</span><span class="value-pct">${{incorrectPct}}</span></div>
            <div class="summary-stat stat-warn"><span class="label">No Answer</span><span class="value">${{noAnswer}}</span><span class="value-pct">${{noAnswerPct}}</span></div>
            <div class="summary-stat stat-info"><span class="label">Avg Steps</span><span class="value">${{avgSteps}}</span></div>
            <div class="summary-stat stat-info"><span class="label">Avg Tool Calls</span><span class="value">${{avgToolCalls}}</span></div>
        `;
    }} else {{
        // Fallback: compute from session metadata
        const total = sessions.length;
        const success = sessions.filter(s => s.metadata?.status === 'success').length;
        const maxExceeded = sessions.filter(s => s.metadata?.status === 'max_steps_exceeded').length;
        const errors = total - success - maxExceeded;
        const avgSteps = (sessions.reduce((a, s) => a + (s.metadata?.steps || 0), 0) / total).toFixed(1);
        bar.innerHTML = `
            <div class="summary-stat"><span class="label">Sessions</span><span class="value">${{total}}</span></div>
            <div class="summary-stat stat-success"><span class="label">Success</span><span class="value">${{success}}</span></div>
            <div class="summary-stat stat-warn"><span class="label">Max Steps</span><span class="value">${{maxExceeded}}</span></div>
            <div class="summary-stat stat-error"><span class="label">Errors</span><span class="value">${{errors}}</span></div>
            <div class="summary-stat stat-info"><span class="label">Avg Steps</span><span class="value">${{avgSteps}}</span></div>
        `;
    }}
}}

function renderSessions(sessions) {{
    const list = document.getElementById('sessionList');
    list.innerHTML = '';
    sessions.forEach((s, i) => {{
        const card = document.createElement('div');
        card.className = 'session-card';
        card.dataset.status = s.metadata?.judge_verdict || s.metadata?.status || '';
        card.dataset.query = (s.metadata?.query || '').toLowerCase();

        const verdict = s.metadata?.judge_verdict;
        const status = verdict || s.metadata?.status || 'unknown';
        const badgeClass = status === 'correct' ? 'badge-success' : status === 'incorrect' ? 'badge-error' : status === 'no_answer' ? 'badge-warn' : status === 'success' ? 'badge-success' : status === 'max_steps_exceeded' ? 'badge-warn' : 'badge-error';
        const steps = s.metadata?.steps || '?';

        card.innerHTML = `
            <div class="session-header" onclick="toggleTrajectory(${{i}})">
                <span class="chevron" id="chev-${{i}}">&#9654;</span>
                <span class="query">${{esc(s.metadata?.query || '(no query)')}}</span>
                <div class="badges">
                    <span class="badge badge-steps">${{steps}} step${{steps !== 1 ? 's' : ''}}</span>
                    <span class="badge ${{badgeClass}}">${{status}}</span>
                </div>
            </div>
            <div class="trajectory" id="traj-${{i}}">
                <div class="session-id">${{s.metadata?.task_id ? esc(s.metadata.task_id) + ' &middot; ' : ''}}${{esc(s.session_id || '')}}</div>
                ${{renderTrajectory(s.trajectory || [])}}
            </div>
        `;
        list.appendChild(card);
    }});
    renderLatex();
}}

function renderTrajectory(trajectory) {{
    return trajectory.map(step => {{
        const role = step.role || 'unknown';
        let cls = 'step-' + role;
        let label = role.toUpperCase();
        let body = '';

        if (role === 'user') {{
            body = `<div class="step-content">${{esc(step.content || '')}}</div>`;
        }} else if (role === 'assistant') {{
            if (step.tool_call) {{
                let args = step.tool_call.arguments || '';
                try {{ args = JSON.stringify(JSON.parse(args), null, 2); }} catch(e) {{}}
                body = `<div class="tool-call-box"><span class="tool-name">${{esc(step.tool_call.name)}}</span>(<span class="tool-args">${{esc(args)}}</span>)</div>`;
            }} else {{
                body = `<div class="step-content">${{esc(step.content || '')}}</div>`;
            }}
        }} else if (role === 'tool') {{
            const result = step.tool_result || '';
            const truncated = result.length > 500;
            const shortResult = truncated ? result.slice(0, 500) + '...' : result;
            const toolName = step.tool_call?.name || 'tool';
            body = `
                <div class="tool-call-box" style="margin-bottom:6px"><span class="tool-name">${{esc(toolName)}}</span> result</div>
                <div class="tool-result-content" id="">${{esc(shortResult)}}</div>
                ${{truncated ? '<span class="tool-result-toggle" onclick="toggleToolResult(this, &quot;' + btoa(unescape(encodeURIComponent(result))) + '&quot;)">Show full result</span>' : ''}}
            `;
        }}

        return `<div class="step ${{cls}}"><div class="step-label">${{label}}</div>${{body}}</div>`;
    }}).join('');
}}

function toggleConfigGroup(header) {{
    header.classList.toggle('open');
    header.nextElementSibling.classList.toggle('open');
}}

function toggleTrajectory(idx) {{
    const traj = document.getElementById('traj-' + idx);
    const chev = document.getElementById('chev-' + idx);
    const isOpen = traj.classList.contains('open');
    traj.classList.toggle('open');
    chev.classList.toggle('open');
    if (!isOpen) renderLatex();
}}

function toggleToolResult(el, b64) {{
    const parent = el.previousElementSibling;
    if (el.textContent === 'Show full result') {{
        parent.textContent = decodeURIComponent(escape(atob(b64)));
        el.textContent = 'Collapse';
    }} else {{
        const full = decodeURIComponent(escape(atob(b64)));
        parent.textContent = full.slice(0, 500) + '...';
        el.textContent = 'Show full result';
    }}
}}

function filterSessions() {{
    const query = document.getElementById('searchInput').value.toLowerCase();
    const status = document.getElementById('statusFilter').value;
    const cards = document.querySelectorAll('.session-card');
    cards.forEach(card => {{
        const matchQuery = !query || card.dataset.query.includes(query);
        const matchStatus = status === 'all' || card.dataset.status === status;
        card.style.display = (matchQuery && matchStatus) ? '' : 'none';
    }});
}}

function esc(s) {{
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}}

function renderLatex() {{
    if (typeof renderMathInElement === 'undefined') {{
        // KaTeX not loaded yet, retry shortly
        setTimeout(renderLatex, 100);
        return;
    }}
    document.querySelectorAll('.step-assistant .step-content').forEach(el => {{
        renderMathInElement(el, {{
            delimiters: [
                {{ left: '$$', right: '$$', display: true }},
                {{ left: '$', right: '$', display: false }},
                {{ left: '\\\\(', right: '\\\\)', display: false }},
                {{ left: '\\\\[', right: '\\\\]', display: true }},
            ],
            throwOnError: false,
        }});
    }});
}}
</script>
</body>
</html>"""


def build_static_html(results_dir: Path) -> str:
    """Build a self-contained HTML file with all run data embedded inline.

    Used for --output mode and server-fallback mode where fetch() won't work.
    """
    runs = sorted(
        [d.name for d in results_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    all_data: dict[str, dict] = {}
    for run_name in runs:
        run_dir = results_dir / run_name
        all_data[run_name] = load_run_data(run_dir)

    # Escape </script> sequences so the HTML parser doesn't break
    data_json = json.dumps(all_data).replace("</", r"<\/")

    html = build_html(results_dir)
    # Inject inline data and override selectRun to use it instead of fetch
    inject = f"""<script>
const INLINE_DATA = {data_json};
// Override selectRun to use inline data instead of fetch
selectRun = async function(runName) {{
    if (!runName) return;
    DATA_CACHE[runName] = INLINE_DATA[runName];
    const run = DATA_CACHE[runName];
    if (!run) return;
    currentSessions = run.sessions;
    const cp = document.getElementById('configPanel');
    if (run.config) {{
        cp.style.display = 'block';
        cp.innerHTML = renderConfig(run.config);
    }} else if (run.sessions.length > 0) {{
        cp.style.display = 'block';
        cp.innerHTML = renderConfig(run.sessions[0].metadata, true);
    }} else {{
        cp.style.display = 'none';
    }}
    renderSummary(run.sessions);
    renderSessions(run.sessions);
}};
// Re-trigger first run selection
if (RUNS.length > 0) selectRun(RUNS[0]);
</script>"""
    # Insert before closing </body>
    return html.replace("</body>", inject + "\n</body>")


class ViewerHandler(SimpleHTTPRequestHandler):
    html_content: str = ""
    results_dir: Path = Path("results")

    def do_GET(self):
        if self.path.startswith("/api/run/"):
            run_name = self.path[len("/api/run/") :]
            # URL-decode the run name
            from urllib.parse import unquote

            run_name = unquote(run_name)
            run_dir = self.results_dir / run_name
            if not run_dir.exists() or not run_dir.is_dir():
                self.send_response(404)
                self.end_headers()
                return
            data = load_run_data(run_dir)
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_content.encode())

    def log_message(self, format, *args):
        pass  # Suppress request logs


def main():
    parser = argparse.ArgumentParser(description="Trajectory Viewer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to the results directory",
    )
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write HTML to file instead of starting a server (e.g. --output viewer.html)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return

    logger.info("Loading trajectories from %s...", results_dir)
    html_content = build_html(results_dir)

    # If --output is given (or server fails), write to file and open directly
    if args.output:
        out_path = Path(args.output).resolve()
        logger.info("Building static HTML with all run data...")
        static_html = build_static_html(results_dir)
        out_path.write_text(static_html, encoding="utf-8")
        logger.info("HTML written to %s", out_path)
        import webbrowser

        webbrowser.open(f"file://{out_path}")
        return

    # Try to start the server; fall back to file if port binding fails
    try:
        ViewerHandler.html_content = html_content
        ViewerHandler.results_dir = results_dir
        server = HTTPServer(("localhost", args.port), ViewerHandler)
        url = f"http://localhost:{args.port}"
        logger.info("Serving trajectory viewer at %s", url)

        import webbrowser

        webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            server.server_close()
    except OSError:
        # Port binding failed (sandbox, port in use, etc.) — fall back to file
        out_path = results_dir / "trajectory_viewer.html"
        logger.info("Building static HTML with all run data...")
        static_html = build_static_html(results_dir)
        out_path.write_text(static_html, encoding="utf-8")
        logger.warning(
            "Could not bind port %d. HTML written to %s", args.port, out_path
        )
        import webbrowser

        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
