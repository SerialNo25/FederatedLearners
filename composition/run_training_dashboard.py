"""Composition root for the browser-based training dashboard."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.parse

import tomli

from domain.dashboard.training_run_monitor import TrainingRunMonitor, TrainingRunMonitorConfig
from stages.training_dashboard.config import TrainingDashboardConfig


def run_training_dashboard(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = TrainingDashboardConfig.from_dict(tomli.loads(path.read_text(encoding="utf-8")))
    monitor = TrainingRunMonitor(
        TrainingRunMonitorConfig(
            experiments_dir=config.experiments_dir,
            active_timeout_seconds=config.active_timeout_seconds,
        )
    )

    server = _DashboardServer((config.host, config.port), monitor, config)
    url = f"http://{config.host}:{config.port}"
    print(f"Training dashboard listening at {url}")
    print("Press Ctrl+C to stop the dashboard.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return config.experiments_dir


class _DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        monitor: TrainingRunMonitor,
        config: TrainingDashboardConfig,
    ) -> None:
        self.monitor = monitor
        self.dashboard_config = config
        super().__init__(server_address, _DashboardRequestHandler)


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    server: _DashboardServer

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_INDEX_HTML)
            return
        if parsed.path == "/api/config":
            self._send_json({"poll_interval_seconds": self.server.dashboard_config.poll_interval_seconds})
            return
        if parsed.path == "/api/runs":
            self._send_json({"runs": self.server.monitor.list_runs()})
            return
        if parsed.path.startswith("/api/runs/"):
            run_name = urllib.parse.unquote(parsed.path.removeprefix("/api/runs/"))
            try:
                self._send_json(self.server.monitor.get_run(run_name))
            except KeyError:
                self._send_json({"error": "run not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --text: #14171a;
      --muted: #586069;
      --line: #d8dee4;
      --panel: #ffffff;
      --teal: #00897b;
      --red: #c62828;
      --green: #2e7d32;
      --amber: #9a6700;
      --ink: #24292f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    header {
      min-height: 64px;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      gap: 14px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    header img { width: 34px; height: 34px; }
    h1 { font-size: 20px; margin: 0; font-weight: 700; }
    .subtle { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      min-height: calc(100vh - 64px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #ffffff;
      overflow: auto;
    }
    .aside-head, .content-head { padding: 18px 20px; border-bottom: 1px solid var(--line); }
    .run-list { display: grid; }
    .run-button {
      appearance: none;
      width: 100%;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      padding: 14px 20px;
      cursor: pointer;
      min-height: 86px;
    }
    .run-button:hover, .run-button.active { background: #eef8f6; }
    .run-title { display: flex; justify-content: space-between; gap: 10px; font-weight: 700; }
    .run-title span:first-child { overflow-wrap: anywhere; }
    .status {
      border-radius: 8px;
      padding: 2px 8px;
      font-size: 12px;
      line-height: 20px;
      color: #ffffff;
      background: var(--muted);
      white-space: nowrap;
    }
    .status.running { background: var(--teal); }
    .status.completed { background: var(--green); }
    .status.stale { background: var(--amber); }
    .run-meta { margin-top: 8px; color: var(--muted); font-size: 13px; display: grid; gap: 3px; }
    section { min-width: 0; overflow: auto; }
    .content-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      background: #ffffff;
    }
    h2 { margin: 0 0 6px 0; font-size: 22px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 1px;
      background: var(--line);
      border-bottom: 1px solid var(--line);
    }
    .metric {
      background: #ffffff;
      padding: 16px 20px;
      min-height: 84px;
    }
    .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .metric-value { font-size: 23px; font-weight: 750; margin-top: 6px; overflow-wrap: anywhere; }
    .chart-wrap { padding: 20px; background: #ffffff; border-bottom: 1px solid var(--line); }
    #chart { width: 100%; height: 360px; display: block; }
    .logs { padding: 20px; }
    pre {
      margin: 0;
      min-height: 220px;
      max-height: 340px;
      overflow: auto;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111111;
      color: #eeeeee;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .empty { padding: 30px 20px; color: var(--muted); }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 44vh; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .content-head { display: block; }
    }
  </style>
</head>
<body>
  <header>
    <img alt="" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='8' fill='%2300897b'/%3E%3Cpath d='M14 42h8V24h-8v18Zm14 0h8V14h-8v28Zm14 0h8V30h-8v12Z' fill='white'/%3E%3C/svg%3E">
    <div>
      <h1>Local Training Runs</h1>
      <div class="subtle">Live metrics from experiment artifacts</div>
    </div>
  </header>
  <main>
    <aside>
      <div class="aside-head">
        <strong>Runs</strong>
        <div id="refresh-note" class="subtle">Loading...</div>
      </div>
      <div id="run-list" class="run-list"></div>
    </aside>
    <section>
      <div class="content-head">
        <div>
          <h2 id="run-name">Select a run</h2>
          <div id="run-path" class="subtle">Waiting for local training metrics.</div>
        </div>
        <span id="run-status" class="status">idle</span>
      </div>
      <div class="metrics">
        <div class="metric"><div class="metric-label">Epoch</div><div id="metric-epoch" class="metric-value">-</div></div>
        <div class="metric"><div class="metric-label">Train Loss</div><div id="metric-train" class="metric-value">-</div></div>
        <div class="metric"><div class="metric-label">Val Loss</div><div id="metric-val" class="metric-value">-</div></div>
        <div class="metric"><div class="metric-label">PR-AUC</div><div id="metric-pr" class="metric-value">-</div></div>
      </div>
      <div class="chart-wrap">
        <svg id="chart" viewBox="0 0 900 360" role="img" aria-label="Training and validation loss"></svg>
      </div>
      <div class="logs">
        <h2>Training Log</h2>
        <pre id="logs">No run selected.</pre>
      </div>
    </section>
  </main>
  <script>
    const state = { selected: null, pollMs: 2000 };
    const fmt = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-";
    const byId = (id) => document.getElementById(id);

    async function loadConfig() {
      const response = await fetch("/api/config");
      const config = await response.json();
      state.pollMs = Math.max(500, Number(config.poll_interval_seconds || 2) * 1000);
    }

    async function loadRuns() {
      const response = await fetch("/api/runs");
      const payload = await response.json();
      renderRuns(payload.runs || []);
      if (!state.selected && payload.runs && payload.runs.length) {
        state.selected = payload.runs[0].name;
      }
      if (state.selected) {
        await loadRun(state.selected);
      }
      byId("refresh-note").textContent = `Updated ${new Date().toLocaleTimeString()}`;
      setTimeout(loadRuns, state.pollMs);
    }

    function renderRuns(runs) {
      const list = byId("run-list");
      if (!runs.length) {
        list.innerHTML = '<div class="empty">No local training runs found.</div>';
        return;
      }
      list.innerHTML = "";
      for (const run of runs) {
        const button = document.createElement("button");
        button.className = `run-button ${run.name === state.selected ? "active" : ""}`;
        button.onclick = () => { state.selected = run.name; loadRun(run.name); renderRuns(runs); };
        const epoch = run.epoch ? `${run.epoch}/${run.total_epochs || "?"}` : "-";
        button.innerHTML = `
          <div class="run-title"><span>${escapeHtml(run.name)}</span><span class="status ${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></div>
          <div class="run-meta">
            <span>${escapeHtml(run.institution_id || "unknown institution")}</span>
            <span>${escapeHtml(run.experiment_name || "experiment")} / ${escapeHtml(run.run_id || "run")}</span>
            <span>Epoch ${escapeHtml(epoch)}</span>
            <span>${escapeHtml(formatTime(run.updated_at))}</span>
          </div>`;
        list.appendChild(button);
      }
    }

    async function loadRun(name) {
      const response = await fetch(`/api/runs/${encodeURIComponent(name)}`);
      if (!response.ok) return;
      const run = await response.json();
      renderRun(run);
    }

    function renderRun(run) {
      const latest = run.latest_metrics || {};
      const nested = latest.metrics || {};
      byId("run-name").textContent = run.name;
      byId("run-path").textContent = `${run.experiment_name || "experiment"} / ${run.run_id || "run"} | ${run.path}`;
      const status = byId("run-status");
      status.textContent = run.status;
      status.className = `status ${run.status}`;
      byId("metric-epoch").textContent = run.epoch ? `${run.epoch}/${run.total_epochs || "?"}` : "-";
      byId("metric-train").textContent = fmt(latest.train_loss);
      byId("metric-val").textContent = fmt(latest.val_loss);
      byId("metric-pr").textContent = fmt(nested.val_pr_auc);
      byId("logs").textContent = (run.log_tail || []).join("\\n") || "No log lines yet.";
      drawChart(run.metrics_history || []);
    }

    function drawChart(metrics) {
      const svg = byId("chart");
      const width = 900, height = 360, left = 58, right = 22, top = 24, bottom = 46;
      svg.innerHTML = "";
      const points = metrics
        .filter(row => row.epoch && Number.isFinite(Number(row.train_loss)) && Number.isFinite(Number(row.val_loss)))
        .map(row => ({ epoch: Number(row.epoch), train: Number(row.train_loss), val: Number(row.val_loss) }));
      if (!points.length) {
        svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#586069">Waiting for epoch metrics</text>`;
        return;
      }
      const epochs = points.map(p => p.epoch);
      const values = points.flatMap(p => [p.train, p.val]);
      let minX = Math.min(...epochs), maxX = Math.max(...epochs);
      let minY = Math.min(...values), maxY = Math.max(...values);
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const pad = (maxY - minY) * 0.12;
      minY -= pad; maxY += pad;
      const x = (v) => left + ((v - minX) / (maxX - minX)) * (width - left - right);
      const y = (v) => top + (1 - ((v - minY) / (maxY - minY))) * (height - top - bottom);
      const line = (key) => points.map(p => `${x(p.epoch).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
      const grid = [];
      for (let i = 0; i <= 5; i++) {
        const gy = top + (i / 5) * (height - top - bottom);
        const label = maxY - (i / 5) * (maxY - minY);
        grid.push(`<line x1="${left}" y1="${gy}" x2="${width - right}" y2="${gy}" stroke="#d8dee4"/>`);
        grid.push(`<text x="${left - 8}" y="${gy + 4}" text-anchor="end" font-size="12" fill="#586069">${fmt(label)}</text>`);
      }
      svg.innerHTML = `
        ${grid.join("")}
        <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#24292f"/>
        <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#24292f"/>
        <polyline points="${line("train")}" fill="none" stroke="#00897b" stroke-width="3"/>
        <polyline points="${line("val")}" fill="none" stroke="#c62828" stroke-width="3"/>
        <text x="${left}" y="18" fill="#00897b" font-size="13">Train loss</text>
        <text x="${left + 92}" y="18" fill="#c62828" font-size="13">Val loss</text>
        <text x="${width / 2}" y="${height - 12}" text-anchor="middle" fill="#586069" font-size="13">Epoch</text>`;
    }

    function formatTime(value) {
      if (!value) return "not started";
      return new Date(value).toLocaleString();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    loadConfig().then(loadRuns).catch(error => {
      byId("refresh-note").textContent = error.message;
    });
  </script>
</body>
</html>
"""
