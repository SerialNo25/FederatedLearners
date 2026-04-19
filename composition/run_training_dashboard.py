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
            stage = self._stage_from_query(parsed.query)
            self._send_json({"runs": self.server.monitor.list_runs(stage=stage)})
            return
        if parsed.path.startswith("/api/runs/"):
            run_name = urllib.parse.unquote(parsed.path.removeprefix("/api/runs/"))
            stage = self._stage_from_query(parsed.query)
            try:
                self._send_json(self.server.monitor.get_run(run_name, stage=stage))
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

    @staticmethod
    def _stage_from_query(query: str) -> str:
        params = urllib.parse.parse_qs(query)
        stage = params.get("stage", ["local_training"])[0]
        if stage in {"local_training", "federated_training"}:
            return stage
        return "local_training"


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
      --teal: #00897b;
      --red: #c62828;
      --green: #2e7d32;
      --amber: #9a6700;
      --blue: #0969da;
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
    h2 { margin: 0 0 6px 0; font-size: 22px; }
    h3 { margin: 0 0 12px 0; font-size: 16px; }
    .subtle { color: var(--muted); font-size: 13px; }
    .menu-button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      width: 38px;
      height: 38px;
      display: grid;
      gap: 4px;
      justify-content: center;
      align-content: center;
      cursor: pointer;
    }
    .menu-button span { display: block; width: 18px; height: 2px; background: var(--ink); }
    .nav {
      display: flex;
      gap: 8px;
      margin-left: auto;
    }
    .nav button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      padding: 8px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    .nav button.active { background: #eef8f6; border-color: var(--teal); color: #00695c; }
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
    .status.pending, .status.unknown, .status.idle { background: var(--muted); }
    .run-meta { margin-top: 8px; color: var(--muted); font-size: 13px; display: grid; gap: 3px; }
    section { min-width: 0; overflow: auto; }
    .content-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      background: #ffffff;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(110px, 1fr));
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
    .metric-value { font-size: 22px; font-weight: 750; margin-top: 6px; overflow-wrap: anywhere; }
    .chart-wrap, .table-wrap { padding: 20px; background: #ffffff; border-bottom: 1px solid var(--line); }
    #chart { width: 100%; height: 360px; display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
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
    @media (max-width: 920px) {
      header { flex-wrap: wrap; }
      .nav { display: none; width: 100%; margin-left: 0; }
      .nav.open { display: flex; }
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 44vh; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .content-head { display: block; }
    }
  </style>
</head>
<body>
  <header>
    <button id="menu-toggle" class="menu-button" aria-label="Open dashboard navigation">
      <span></span><span></span><span></span>
    </button>
    <img alt="" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='8' fill='%2300897b'/%3E%3Cpath d='M14 42h8V24h-8v18Zm14 0h8V14h-8v28Zm14 0h8V30h-8v12Z' fill='white'/%3E%3C/svg%3E">
    <div>
      <h1 id="page-title">Local Training Runs</h1>
      <div id="page-subtitle" class="subtle">Live metrics from experiment artifacts</div>
    </div>
    <nav id="dashboard-nav" class="nav" aria-label="Dashboard navigation">
      <button type="button" data-stage="local_training" class="active">Local</button>
      <button type="button" data-stage="federated_training">Federated</button>
    </nav>
  </header>
  <main>
    <aside>
      <div class="aside-head">
        <strong id="run-list-title">Runs</strong>
        <div id="refresh-note" class="subtle">Loading...</div>
      </div>
      <div id="run-list" class="run-list"></div>
    </aside>
    <section>
      <div class="content-head">
        <div>
          <h2 id="run-name">Select a run</h2>
          <div id="run-path" class="subtle">Waiting for metrics.</div>
        </div>
        <span id="run-status" class="status idle">idle</span>
      </div>
      <div id="metrics" class="metrics"></div>
      <div class="chart-wrap">
        <svg id="chart" viewBox="0 0 900 360" role="img" aria-label="Training metrics"></svg>
      </div>
      <div id="federated-table" class="table-wrap" hidden>
        <h3>Institution Progress</h3>
        <table>
          <thead>
            <tr>
              <th>Institution</th>
              <th>Samples</th>
              <th>Local Loss</th>
              <th>Eval Loss</th>
              <th>PR-AUC</th>
              <th>F1</th>
              <th>Delta L2</th>
            </tr>
          </thead>
          <tbody id="institution-rows"></tbody>
        </table>
      </div>
      <div class="logs">
        <h2>Training Log</h2>
        <pre id="logs">No run selected.</pre>
      </div>
    </section>
  </main>
  <script>
    const stages = {
      local_training: {
        title: "Local Training Runs",
        subtitle: "Live metrics from experiment artifacts",
        empty: "No local training runs found.",
        progressLabel: "Epoch",
        waiting: "Waiting for local training metrics."
      },
      federated_training: {
        title: "Federated Training Runs",
        subtitle: "Global and exclusive run supervision",
        empty: "No federated training runs found.",
        progressLabel: "Round",
        waiting: "Waiting for federated round metrics."
      }
    };
    const state = { stage: "local_training", selectedByStage: {}, pollMs: 2000, timer: null };
    const fmt = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-";
    const byId = (id) => document.getElementById(id);

    async function loadConfig() {
      const response = await fetch("/api/config");
      const config = await response.json();
      state.pollMs = Math.max(500, Number(config.poll_interval_seconds || 2) * 1000);
    }

    async function loadRuns() {
      clearTimeout(state.timer);
      const response = await fetch(`/api/runs?stage=${encodeURIComponent(state.stage)}`);
      const payload = await response.json();
      const runs = payload.runs || [];
      renderRuns(runs);
      if (!state.selectedByStage[state.stage] && runs.length) {
        state.selectedByStage[state.stage] = runs[0].name;
      }
      if (state.selectedByStage[state.stage]) {
        await loadRun(state.selectedByStage[state.stage]);
      } else {
        renderEmptySelection();
      }
      byId("refresh-note").textContent = `Updated ${new Date().toLocaleTimeString()}`;
      state.timer = setTimeout(loadRuns, state.pollMs);
    }

    function switchStage(stage) {
      state.stage = stage;
      document.querySelectorAll("[data-stage]").forEach(button => {
        button.classList.toggle("active", button.dataset.stage === stage);
      });
      const copy = stages[stage];
      byId("page-title").textContent = copy.title;
      byId("page-subtitle").textContent = copy.subtitle;
      byId("run-list-title").textContent = stage === "federated_training" ? "Federated Runs" : "Runs";
      renderEmptySelection();
      loadRuns();
    }

    function renderRuns(runs) {
      const list = byId("run-list");
      if (!runs.length) {
        list.innerHTML = `<div class="empty">${stages[state.stage].empty}</div>`;
        return;
      }
      list.innerHTML = "";
      for (const run of runs) {
        const button = document.createElement("button");
        button.className = `run-button ${run.name === state.selectedByStage[state.stage] ? "active" : ""}`;
        button.onclick = () => {
          state.selectedByStage[state.stage] = run.name;
          loadRun(run.name);
          renderRuns(runs);
        };
        const progress = run.epoch ? `${run.epoch}/${run.total_epochs || "?"}` : "-";
        const population = state.stage === "federated_training"
          ? `${run.run_type || "federated"} | ${(run.institution_ids || []).join(", ") || "unknown institutions"}`
          : run.institution_id || "unknown institution";
        button.innerHTML = `
          <div class="run-title"><span>${escapeHtml(run.name)}</span><span class="status ${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></div>
          <div class="run-meta">
            <span>${escapeHtml(population)}</span>
            <span>${escapeHtml(run.experiment_name || "experiment")} / ${escapeHtml(run.run_id || "run")}</span>
            <span>${stages[state.stage].progressLabel} ${escapeHtml(progress)}</span>
            <span>${escapeHtml(formatTime(run.updated_at))}</span>
          </div>`;
        list.appendChild(button);
      }
    }

    async function loadRun(name) {
      const response = await fetch(`/api/runs/${encodeURIComponent(name)}?stage=${encodeURIComponent(state.stage)}`);
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
      if (state.stage === "federated_training") {
        renderFederatedMetrics(run, latest, nested);
      } else {
        renderLocalMetrics(run, latest, nested);
      }
      byId("logs").textContent = (run.log_tail || []).join("\\n") || "No log lines yet.";
      drawChart(run.metrics_history || []);
    }

    function renderLocalMetrics(run, latest, nested) {
      byId("federated-table").hidden = true;
      renderMetricCards([
        ["Epoch", run.epoch ? `${run.epoch}/${run.total_epochs || "?"}` : "-"],
        ["Train Loss", fmt(latest.train_loss)],
        ["Val Loss", fmt(latest.val_loss)],
        ["PR-AUC", fmt(nested.val_pr_auc)],
        ["Institution", run.institution_id || "-"],
        ["Status", run.status || "-"]
      ]);
    }

    function renderFederatedMetrics(run, latest, nested) {
      renderMetricCards([
        ["Round", run.epoch ? `${run.epoch}/${run.total_epochs || run.num_rounds || "?"}` : "-"],
        ["Run Type", run.run_type || "federated"],
        ["Clients", (run.institution_ids || []).length || "-"],
        ["FedProx Mu", fmt(run.proximal_mu)],
        ["Train Loss", fmt(latest.train_loss)],
        ["PR-AUC", fmt(latest.pr_auc)]
      ]);
      renderInstitutionTable(nested);
    }

    function renderMetricCards(items) {
      byId("metrics").innerHTML = items.map(([label, value]) => `
        <div class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div></div>
      `).join("");
    }

    function renderInstitutionTable(metrics) {
      const table = byId("federated-table");
      const rows = byId("institution-rows");
      const samples = metrics.local_num_samples || {};
      const losses = metrics.local_loss || {};
      const deltas = metrics.local_parameter_delta_l2 || {};
      const evaluations = metrics.institution_evaluation || {};
      const institutions = Array.from(new Set([
        ...Object.keys(samples),
        ...Object.keys(losses),
        ...Object.keys(deltas),
        ...Object.keys(evaluations)
      ])).sort();
      table.hidden = institutions.length === 0;
      rows.innerHTML = institutions.map(id => {
        const evaluation = evaluations[id] || {};
        return `
          <tr>
            <td>${escapeHtml(id)}</td>
            <td>${escapeHtml(samples[id] ?? "-")}</td>
            <td>${escapeHtml(fmt(losses[id]))}</td>
            <td>${escapeHtml(fmt(evaluation.loss))}</td>
            <td>${escapeHtml(fmt(evaluation.pr_auc))}</td>
            <td>${escapeHtml(fmt(evaluation.f1))}</td>
            <td>${escapeHtml(fmt(deltas[id]))}</td>
          </tr>`;
      }).join("");
    }

    function renderEmptySelection() {
      byId("run-name").textContent = "Select a run";
      byId("run-path").textContent = stages[state.stage].waiting;
      byId("run-status").textContent = "idle";
      byId("run-status").className = "status idle";
      byId("logs").textContent = "No run selected.";
      byId("federated-table").hidden = true;
      renderMetricCards([
        [stages[state.stage].progressLabel, "-"],
        ["Train Loss", "-"],
        ["Val Loss", "-"],
        ["PR-AUC", "-"],
        ["Clients", "-"],
        ["Status", "-"]
      ]);
      byId("chart").innerHTML = '<text x="450" y="180" text-anchor="middle" fill="#586069">Waiting for metrics</text>';
    }

    function drawChart(metrics) {
      const svg = byId("chart");
      const width = 900, height = 360, left = 58, right = 22, top = 24, bottom = 46;
      svg.innerHTML = "";
      const points = metrics
        .filter(row => row.epoch && Number.isFinite(Number(row.train_loss)) && Number.isFinite(Number(row.val_loss)))
        .map(row => ({
          epoch: Number(row.epoch),
          train: Number(row.train_loss),
          val: Number(row.val_loss)
        }));
      if (!points.length) {
        svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#586069">Waiting for ${state.stage === "federated_training" ? "round" : "epoch"} metrics</text>`;
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
      const line = (key) => points.filter(p => Number.isFinite(p[key])).map(p => `${x(p.epoch).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
      const grid = [];
      for (let i = 0; i <= 5; i++) {
        const gy = top + (i / 5) * (height - top - bottom);
        const label = maxY - (i / 5) * (maxY - minY);
        grid.push(`<line x1="${left}" y1="${gy}" x2="${width - right}" y2="${gy}" stroke="#d8dee4"/>`);
        grid.push(`<text x="${left - 8}" y="${gy + 4}" text-anchor="end" font-size="12" fill="#586069">${fmt(label)}</text>`);
      }
      const progressLabel = state.stage === "federated_training" ? "Round" : "Epoch";
      svg.innerHTML = `
        ${grid.join("")}
        <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#24292f"/>
        <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#24292f"/>
        <polyline points="${line("train")}" fill="none" stroke="#00897b" stroke-width="3"/>
        <polyline points="${line("val")}" fill="none" stroke="#c62828" stroke-width="3"/>
        <text x="${left}" y="18" fill="#00897b" font-size="13">Train loss</text>
        <text x="${left + 92}" y="18" fill="#c62828" font-size="13">Val loss</text>
        <text x="${width / 2}" y="${height - 12}" text-anchor="middle" fill="#586069" font-size="13">${progressLabel}</text>`;
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

    byId("menu-toggle").addEventListener("click", () => byId("dashboard-nav").classList.toggle("open"));
    document.querySelectorAll("[data-stage]").forEach(button => {
      button.addEventListener("click", () => switchStage(button.dataset.stage));
    });
    renderEmptySelection();
    loadConfig().then(loadRuns).catch(error => {
      byId("refresh-note").textContent = error.message;
    });
  </script>
</body>
</html>
"""
