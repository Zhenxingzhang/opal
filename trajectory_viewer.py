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
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import yaml


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


def build_html(results_dir: Path) -> str:
    """Build the full single-page HTML app."""
    # Discover runs
    runs = sorted(
        [d.name for d in results_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    # Pre-load all data as JSON for the frontend
    all_data: dict[str, dict] = {}
    for run_name in runs:
        run_dir = results_dir / run_name
        config = load_config(run_dir)
        sessions = find_sessions(run_dir)
        all_data[run_name] = {"config": config, "sessions": sessions}

    data_json = json.dumps(all_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trajectory Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1a1a2e; color: #fff; padding: 16px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }}
.header h1 {{ font-size: 18px; font-weight: 600; white-space: nowrap; }}
.header select {{ padding: 6px 12px; border-radius: 6px; border: 1px solid #444; background: #16213e; color: #eee; font-size: 14px; max-width: 600px; flex: 1; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}

/* Config panel */
.config-panel {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }}
.config-panel h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 10px; color: #1a1a2e; }}
.config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px 24px; }}
.config-item {{ display: flex; gap: 8px; font-size: 13px; }}
.config-key {{ font-weight: 600; color: #555; white-space: nowrap; }}
.config-val {{ color: #111; }}

/* Summary bar */
.summary-bar {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 20px; margin-bottom: 20px; display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; }}
.summary-stat {{ display: flex; gap: 6px; }}
.summary-stat .label {{ color: #888; }}
.summary-stat .value {{ font-weight: 600; }}

/* Session list */
.session-card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
.session-header {{ padding: 12px 20px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 12px; transition: background 0.15s; }}
.session-header:hover {{ background: #f9f9f9; }}
.session-header .query {{ flex: 1; font-size: 14px; font-weight: 500; }}
.session-header .badges {{ display: flex; gap: 8px; flex-shrink: 0; }}
.badge {{ padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
.badge-success {{ background: #e6f4ea; color: #1e7e34; }}
.badge-error {{ background: #fce8e6; color: #c5221f; }}
.badge-warn {{ background: #fef7e0; color: #b45309; }}
.badge-steps {{ background: #e8eaf6; color: #3f51b5; }}
.chevron {{ font-size: 18px; color: #999; transition: transform 0.2s; flex-shrink: 0; }}
.chevron.open {{ transform: rotate(90deg); }}

/* Trajectory view */
.trajectory {{ display: none; padding: 0 20px 16px; }}
.trajectory.open {{ display: block; }}
.step {{ margin-top: 12px; padding: 12px 16px; border-radius: 8px; font-size: 13px; line-height: 1.6; }}
.step-user {{ background: #e3f2fd; border-left: 4px solid #1976d2; }}
.step-assistant {{ background: #f3e5f5; border-left: 4px solid #7b1fa2; }}
.step-tool {{ background: #fff8e1; border-left: 4px solid #f9a825; }}
.step-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.step-user .step-label {{ color: #1976d2; }}
.step-assistant .step-label {{ color: #7b1fa2; }}
.step-tool .step-label {{ color: #f9a825; }}
.step-content {{ white-space: pre-wrap; word-break: break-word; }}
.tool-call-box {{ background: rgba(0,0,0,0.04); border-radius: 6px; padding: 10px 12px; margin-top: 6px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }}
.tool-name {{ font-weight: 700; color: #e65100; }}
.tool-args {{ color: #333; }}
.tool-result-content {{ max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
.tool-result-toggle {{ cursor: pointer; color: #1976d2; font-size: 12px; margin-top: 4px; text-decoration: underline; }}
.session-id {{ font-size: 11px; color: #aaa; font-family: monospace; }}

/* Search & filter */
.filter-bar {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }}
.filter-bar input {{ flex: 1; padding: 8px 14px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
.filter-bar select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; }}
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
            <option value="success">Success</option>
            <option value="max_steps_exceeded">Max steps exceeded</option>
            <option value="error">Error</option>
        </select>
    </div>
    <div id="sessionList"></div>
</div>

<script>
const DATA = {data_json};

// Populate run selector
const runSelect = document.getElementById('runSelect');
Object.keys(DATA).forEach(name => {{
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    runSelect.appendChild(opt);
}});

// Auto-select first run
if (Object.keys(DATA).length > 0) {{
    runSelect.value = Object.keys(DATA)[0];
    selectRun(Object.keys(DATA)[0]);
}}

let currentSessions = [];

function selectRun(runName) {{
    if (!runName || !DATA[runName]) return;
    const run = DATA[runName];
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
    renderSummary(run.sessions);

    // Sessions
    renderSessions(run.sessions);
}}

function renderConfig(config, fromMetadata) {{
    let title = fromMetadata ? 'Run Config (inferred from session metadata)' : 'Run Config';
    let items = '';
    function flatten(obj, prefix) {{
        for (const [k, v] of Object.entries(obj)) {{
            const key = prefix ? prefix + '.' + k : k;
            if (v && typeof v === 'object' && !Array.isArray(v)) {{
                flatten(v, key);
            }} else {{
                const val = Array.isArray(v) ? v.join(', ') : String(v);
                items += `<div class="config-item"><span class="config-key">${{esc(key)}}:</span><span class="config-val">${{esc(val)}}</span></div>`;
            }}
        }}
    }}
    flatten(config, '');
    return `<h2>${{title}}</h2><div class="config-grid">${{items}}</div>`;
}}

function renderSummary(sessions) {{
    const bar = document.getElementById('summaryBar');
    if (sessions.length === 0) {{ bar.style.display = 'none'; return; }}
    bar.style.display = 'flex';
    const total = sessions.length;
    const success = sessions.filter(s => s.metadata?.status === 'success').length;
    const maxExceeded = sessions.filter(s => s.metadata?.status === 'max_steps_exceeded').length;
    const errors = total - success - maxExceeded;
    const avgSteps = (sessions.reduce((a, s) => a + (s.metadata?.steps || 0), 0) / total).toFixed(1);
    bar.innerHTML = `
        <div class="summary-stat"><span class="label">Sessions:</span><span class="value">${{total}}</span></div>
        <div class="summary-stat"><span class="label">Success:</span><span class="value">${{success}}</span></div>
        <div class="summary-stat"><span class="label">Max steps exceeded:</span><span class="value">${{maxExceeded}}</span></div>
        <div class="summary-stat"><span class="label">Errors:</span><span class="value">${{errors}}</span></div>
        <div class="summary-stat"><span class="label">Avg steps:</span><span class="value">${{avgSteps}}</span></div>
    `;
}}

function renderSessions(sessions) {{
    const list = document.getElementById('sessionList');
    list.innerHTML = '';
    sessions.forEach((s, i) => {{
        const card = document.createElement('div');
        card.className = 'session-card';
        card.dataset.status = s.metadata?.status || '';
        card.dataset.query = (s.metadata?.query || '').toLowerCase();

        const status = s.metadata?.status || 'unknown';
        const badgeClass = status === 'success' ? 'badge-success' : status === 'max_steps_exceeded' ? 'badge-warn' : 'badge-error';
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
                <div class="session-id">${{esc(s.session_id || '')}}</div>
                ${{renderTrajectory(s.trajectory || [])}}
            </div>
        `;
        list.appendChild(card);
    }});
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

function toggleTrajectory(idx) {{
    const traj = document.getElementById('traj-' + idx);
    const chev = document.getElementById('chev-' + idx);
    const isOpen = traj.classList.contains('open');
    traj.classList.toggle('open');
    chev.classList.toggle('open');
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
</script>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    html_content: str = ""

    def do_GET(self):
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
        print(f"Error: results directory not found: {results_dir}")
        return

    print(f"Loading trajectories from {results_dir}...")
    html_content = build_html(results_dir)

    # If --output is given (or server fails), write to file and open directly
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.write_text(html_content, encoding="utf-8")
        print(f"HTML written to {out_path}")
        import webbrowser

        webbrowser.open(f"file://{out_path}")
        return

    # Try to start the server; fall back to file if port binding fails
    try:
        ViewerHandler.html_content = html_content
        server = HTTPServer(("localhost", args.port), ViewerHandler)
        url = f"http://localhost:{args.port}"
        print(f"Serving trajectory viewer at {url}")

        import webbrowser

        webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            server.server_close()
    except OSError:
        # Port binding failed (sandbox, port in use, etc.) — fall back to file
        out_path = results_dir / "trajectory_viewer.html"
        out_path.write_text(html_content, encoding="utf-8")
        print(f"Could not bind port {args.port}. HTML written to {out_path}")
        import webbrowser

        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    main()
