"""`jevlab site`: a self-contained, interactive HTML leaderboard built from summary.json.

The page has no external dependencies: data is embedded as JSON and the charts are
drawn as inline SVG by a small script, so it can be opened from disk or served by
GitHub Pages as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev Calibration Lab</title>
<meta name="description" content="Independent calibration benchmark for System One decision models on human-labeled data.">
<style>
:root {
  --bg: #fafaf9; --fg: #1c1917; --muted: #78716c; --line: #e7e5e4; --card: #ffffff;
  --accent: #2563eb; --warn-bg: #fff7ed; --warn-fg: #9a3412; --warn-line: #fdba74;
  --c0: #2563eb; --c1: #d97706; --c2: #7c3aed; --c3: #059669; --c4: #dc2626; --c5: #0891b2;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0c0a09; --fg: #f5f5f4; --muted: #a8a29e; --line: #292524; --card: #1c1917;
    --accent: #60a5fa; --warn-bg: #2a1708; --warn-fg: #fdba74; --warn-line: #9a3412;
    --c0: #60a5fa; --c1: #fbbf24; --c2: #a78bfa; --c3: #34d399; --c4: #f87171; --c5: #22d3ee;
  }
}
:root[data-theme="dark"] {
  --bg: #0c0a09; --fg: #f5f5f4; --muted: #a8a29e; --line: #292524; --card: #1c1917;
  --accent: #60a5fa; --warn-bg: #2a1708; --warn-fg: #fdba74; --warn-line: #9a3412;
  --c0: #60a5fa; --c1: #fbbf24; --c2: #a78bfa; --c3: #34d399; --c4: #f87171; --c5: #22d3ee;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 36px 0 10px; }
p.lede { color: var(--muted); margin: 0 0 16px; max-width: 70ch; }
.banner { background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-line);
  border-radius: 8px; padding: 10px 14px; margin: 16px 0; }
.badge { display: inline-block; font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .04em; padding: 1px 6px; border-radius: 4px; margin-left: 6px;
  background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-line); vertical-align: 1px; }
.badge.real { background: transparent; color: var(--c3); border-color: var(--c3); }
.controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center;
  padding: 12px 14px; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
.controls label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
select { font: inherit; padding: 4px 6px; background: var(--card); color: var(--fg);
  border: 1px solid var(--line); border-radius: 6px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--card); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; font-size: 13px; color: var(--muted); cursor: pointer; user-select: none; }
th.sorted::after { content: " \25BE"; }
th.sorted.asc::after { content: " \25B4"; }
tr:last-child td { border-bottom: none; }
td .ci { color: var(--muted); font-size: 12px; margin-left: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
.card h3 { font-size: 14px; margin: 0 0 6px; }
svg { display: block; width: 100%; height: auto; }
svg text { fill: var(--muted); font-size: 10px; }
.axis { stroke: var(--line); }
.diag { stroke: var(--muted); stroke-dasharray: 3 3; opacity: .6; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; margin: 6px 0; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
footer { color: var(--muted); font-size: 13px; margin-top: 40px; }
a { color: var(--accent); }
@media (max-width: 600px) { th, td { padding: 6px 7px; } h1 { font-size: 22px; } }
</style>
</head>
<body>
<main>
<h1>Jev Calibration Lab</h1>
<p class="lede">An independent check of how well System One decision models' probabilities match
human ground truth. Each arm answers typed decisions from public, human-labeled datasets; the page
reports accuracy, calibration, selective prediction, cost and latency.</p>
<div id="banner"></div>

<div class="controls">
  <label>Dataset <select id="dataset"></select></label>
  <span id="armToggles"></span>
  <label><input type="checkbox" id="hideSynthetic"> hide synthetic arms</label>
</div>

<h2>Leaderboard</h2>
<div class="table-wrap"><table id="board"></table></div>

<h2>Reliability diagrams</h2>
<p class="lede">Bars: accuracy per confidence bin. Ticks: mean confidence in the bin. A calibrated model sits on the dashed diagonal.</p>
<div class="grid" id="reliability"></div>

<h2>Selective prediction</h2>
<p class="lede">Error rate among the most confident predictions as coverage grows. Lower curves abstain more usefully.</p>
<div class="card"><div class="legend" id="rcLegend"></div><div id="riskCoverage"></div></div>

<h2>Cascade</h2>
<p class="lede" id="cascadeLede"></p>
<div class="card"><div id="cascade"></div></div>

<h2>Post-hoc recalibration</h2>
<p class="lede">ECE before and after cheap fixes fitted on held-out folds. If these close the gap, native calibration buys less.</p>
<div class="table-wrap"><table id="recal"></table></div>

<footer>Generated __GENERATED__ by <code>jevlab site</code>. Source and method:
<a href="__REPO__">__REPO__</a>.</footer>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const S = JSON.parse(document.getElementById('data').textContent);
const arms = Object.keys(S.arms);
const color = Object.fromEntries(arms.map((a, i) => [a, `var(--c${i % 6})`]));
const isSyn = a => !!S.arms[a].synthetic;
const state = { dataset: 'macro', hidden: new Set(), hideSyn: false, sort: 'ece', asc: true };
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? 'n/a' : Number(x).toFixed(d);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const badge = a => isSyn(a) ? '<span class="badge">synthetic</span>' : '<span class="badge real">real</span>';
const visible = () => arms.filter(a => !state.hidden.has(a) && !(state.hideSyn && isSyn(a)));

const synArms = arms.filter(isSyn);
if (synArms.length) {
  document.getElementById('banner').innerHTML = `<div class="banner"><strong>Synthetic demo output.</strong>
  Arms badged <em>synthetic</em> (${synArms.map(esc).join(', ')}) come from a dry run with invented generators
  that mimic each provider's reply format. They are not measurements of those models. Real numbers appear once the
  benchmark is run with API keys.</div>`;
}

const dsSel = document.getElementById('dataset');
[['macro', 'All datasets (macro average)'], ...Object.entries(S.datasets).map(([k, v]) => [k, v.title])]
  .forEach(([k, t]) => dsSel.insertAdjacentHTML('beforeend', `<option value="${esc(k)}">${esc(t)}</option>`));
dsSel.onchange = () => { state.dataset = dsSel.value; render(); };
document.getElementById('armToggles').innerHTML = arms.map(a =>
  `<label><input type="checkbox" data-arm="${esc(a)}" checked><span class="swatch" style="background:${color[a]}"></span>${esc(a)}</label>`).join(' ');
document.querySelectorAll('[data-arm]').forEach(cb => cb.onchange = () => {
  cb.checked ? state.hidden.delete(cb.dataset.arm) : state.hidden.add(cb.dataset.arm); render(); });
document.getElementById('hideSynthetic').onchange = e => { state.hideSyn = e.target.checked; render(); };

const COLS = [
  ['accuracy', 'Accuracy', 3, false], ['ece', 'ECE', 3, true], ['ece_adaptive', 'Adaptive ECE', 3, true],
  ['brier', 'Brier', 3, true], ['nll', 'NLL', 3, true], ['aurc', 'AURC', 3, true],
  ['p50_ms', 'p50 ms', 0, true], ['p95_ms', 'p95 ms', 0, true], ['cost_per_1k', '$ / 1k', 4, true]];

function renderBoard() {
  const rows = S.metrics.filter(m => m.dataset === state.dataset && visible().includes(m.arm));
  rows.sort((a, b) => (state.asc ? 1 : -1) * ((a[state.sort] ?? 0) - (b[state.sort] ?? 0)));
  const head = '<tr><th>Arm</th>' + COLS.map(([k, t]) =>
    `<th data-k="${k}" class="${state.sort === k ? 'sorted' + (state.asc ? ' asc' : '') : ''}">${t}</th>`).join('') + '</tr>';
  const body = rows.map(m => '<tr><td>' + esc(m.arm) + badge(m.arm) + '</td>' + COLS.map(([k, , d]) => {
    const ci = m.ci && m.ci[k] ? `<span class="ci">[${fmt(m.ci[k][0], d)}, ${fmt(m.ci[k][1], d)}]</span>` : '';
    return `<td>${fmt(m[k], d)}${ci}</td>`; }).join('') + '</tr>').join('');
  const t = document.getElementById('board');
  t.innerHTML = head + body;
  t.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => {
    const k = th.dataset.k; state.asc = state.sort === k ? !state.asc : COLS.find(c => c[0] === k)[3]; state.sort = k; renderBoard(); });
}

function frame(w, h, pad, xl, yl) {
  return `<line class="axis" x1="${pad}" y1="${h - pad}" x2="${w - 8}" y2="${h - pad}"/>
  <line class="axis" x1="${pad}" y1="8" x2="${pad}" y2="${h - pad}"/>
  <text x="${(w + pad) / 2}" y="${h - 6}" text-anchor="middle">${xl}</text>
  <text x="10" y="${(h - pad) / 2}" transform="rotate(-90 10 ${(h - pad) / 2})" text-anchor="middle">${yl}</text>`;
}

function renderReliability() {
  const ds = state.dataset === 'macro' ? Object.keys(S.datasets)[0] : state.dataset;
  const W = 240, H = 220, P = 30;
  const x = v => P + v * (W - P - 8), y = v => (H - P) - v * (H - P - 8);
  document.getElementById('reliability').innerHTML = visible().map(a => {
    const bins = (S.reliability[a] || {})[ds] || [];
    const bars = bins.map(b => `<rect x="${x(b.lo) + 1}" y="${y(b.acc)}" width="${Math.max(1, x(b.hi) - x(b.lo) - 2)}"
      height="${y(0) - y(b.acc)}" fill="${color[a]}" opacity=".75"><title>${fmt(b.lo, 1)}-${fmt(b.hi, 1)}: acc ${fmt(b.acc)}, conf ${fmt(b.conf)}, n=${b.count}</title></rect>
      <line x1="${x(b.lo) + 2}" x2="${x(b.hi) - 2}" y1="${y(b.conf)}" y2="${y(b.conf)}" stroke="currentColor" stroke-width="2"/>`).join('');
    return `<div class="card"><h3>${esc(a)}${badge(a)}</h3><svg viewBox="0 0 ${W} ${H}" style="color:var(--fg)">
      ${frame(W, H, P, 'confidence', 'accuracy')}<line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>${bars}</svg>
      <div style="font-size:12px;color:var(--muted)">${esc(S.datasets[ds].title)}</div></div>`;
  }).join('');
}

function lineChart(el, series, xl, yl, xlog) {
  const W = 760, H = 300, P = 44;
  const pts = series.flatMap(s => s.points);
  if (!pts.length) { el.innerHTML = '<p class="lede">No data for this selection.</p>'; return; }
  const tx = v => xlog ? Math.log10(Math.max(v, 1e-4)) : v;
  const xs = pts.map(p => tx(p[0])), ys = pts.map(p => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs) + 1e-9], [y0, y1] = [Math.min(...ys), Math.max(...ys) + 1e-9];
  const sx = v => P + (tx(v) - x0) / (x1 - x0) * (W - P - 12), sy = v => (H - P) - (v - y0) / (y1 - y0) * (H - P - 12);
  const ticks = [0, .25, .5, .75, 1].map(f => y0 + f * (y1 - y0));
  const paths = series.map(s => `<path d="${s.points.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1)).join('')}"
    fill="none" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>` +
    (s.dots ? s.points.map(p => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="3" fill="${s.color}"><title>${p[2] || ''}</title></circle>`).join('') : '')).join('');
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${frame(W, H, P, xl, yl)}
    ${ticks.map(t => `<text x="${P - 4}" y="${sy(t) + 3}" text-anchor="end">${fmt(t, 2)}</text>`).join('')}${paths}</svg>`;
}

function renderRiskCoverage() {
  const ds = state.dataset === 'macro' ? Object.keys(S.datasets)[0] : state.dataset;
  const series = visible().filter(a => (S.risk_coverage[a] || {})[ds]).map(a =>
    ({ points: S.risk_coverage[a][ds], color: color[a], dash: isSyn(a) }));
  document.getElementById('rcLegend').innerHTML = visible().map(a =>
    `<span><span class="swatch" style="background:${color[a]}"></span>${esc(a)}${isSyn(a) ? ' (synthetic)' : ''}</span>`).join('') +
    `<span style="color:var(--muted)">${esc(S.datasets[ds].title)}</span>`;
  lineChart(document.getElementById('riskCoverage'), series, 'coverage', 'risk (error rate)', false);
}

function renderCascade() {
  const ds = state.dataset === 'macro' ? Object.keys(S.cascade)[0] : state.dataset;
  const c = S.cascade[ds];
  const lede = document.getElementById('cascadeLede');
  if (!c) { lede.textContent = 'No cascade for this dataset.'; document.getElementById('cascade').innerHTML = ''; return; }
  lede.innerHTML = `${esc(c.first)} answers first; items whose top probability is below &tau; go to ${esc(c.second)}.
    Each dot is one &tau; (hover for details), plotted as cost per 1k decisions (log scale) against accuracy on
    ${esc(S.datasets[ds].title)}.${c.synthetic ? ' <span class="badge">synthetic</span>' : ''}`;
  const pts = c.points.map(p => [p.cost_per_1k, p.accuracy,
    `tau ${fmt(p.tau, 2)}${p.always_escalate ? ' (always escalate)' : ''}: escalate ${fmt(100 * p.escalation_rate, 0)}%, acc ${fmt(p.accuracy)}, $${fmt(p.cost_per_1k, 4)}/1k`]);
  lineChart(document.getElementById('cascade'), [{ points: pts, color: 'var(--accent)', dots: true }], '$ per 1k decisions (log)', 'accuracy', true);
}

function renderRecal() {
  const methods = ['temperature', 'platt', 'isotonic'];
  const rows = visible().map(a => {
    const rs = S.recalibration.filter(r => r.arm === a && (state.dataset === 'macro' || r.dataset === state.dataset));
    if (!rs.length) return '';
    const avg = (m, k) => { const v = rs.filter(r => r.method === m).map(r => r[k]); return v.reduce((x, y) => x + y, 0) / v.length; };
    return `<tr><td>${esc(a)}${badge(a)}</td><td>${fmt(avg('temperature', 'ece_before'))}</td>${methods.map(m => `<td>${fmt(avg(m, 'ece_after'))}</td>`).join('')}</tr>`;
  }).join('');
  document.getElementById('recal').innerHTML = '<tr><th>Arm</th><th>Raw ECE</th><th>Temperature</th><th>Platt</th><th>Isotonic</th></tr>' + rows;
}

function render() { renderBoard(); renderReliability(); renderRiskCoverage(); renderCascade(); renderRecal(); }
render();
</script>
</body>
</html>
"""

REPO_URL = "https://github.com/veermshah/jev-calibration-lab"


def render_site(summary: dict[str, Any], repo_url: str = REPO_URL) -> str:
    from .report import json_safe

    data = json.dumps(json_safe(summary), allow_nan=False).replace("</", "<\\/")
    return (
        TEMPLATE.replace("__DATA__", data)
        .replace("__GENERATED__", str(summary.get("generated", "")))
        .replace("__REPO__", repo_url)
    )


def build_site(summary_path: Path, out_dir: Path) -> Path:
    summary = json.loads(summary_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(render_site(summary), encoding="utf-8")
    return path
