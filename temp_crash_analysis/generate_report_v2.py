#!/usr/bin/env python3
"""Generate updated crash report HTML with interactive network map + tabs."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import sys, re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config

OUT_DIR = Path(__file__).resolve().parent
CRASH_LAT, CRASH_LON = 40.524, -111.893

# ── 1. Extract network map data ────────────────────────────────────

def extract_map_data():
    tree = ET.parse(str(config.NETWORK_XML))
    root = tree.getroot()
    nodes = {}
    for n in root.find('nodes'):
        lat, lon = float(n.get('y')), float(n.get('x'))
        if abs(lat - CRASH_LAT) < 0.038 and abs(lon - CRASH_LON) < 0.048:
            nodes[n.get('id')] = (lat, lon)

    links = []
    for l in root.find('links'):
        fn, tn = l.get('from'), l.get('to')
        if fn in nodes and tn in nodes:
            lat1, lon1 = nodes[fn]
            lat2, lon2 = nodes[tn]
            mid_lat, mid_lon = (lat1+lat2)/2, (lon1+lon2)/2
            spd = float(l.get('freespeed'))
            lns = float(l.get('permlanes'))
            lng = float(l.get('length'))
            dist = ((mid_lat-CRASH_LAT)**2 + (mid_lon-CRASH_LON)**2)**0.5
            is_sb = lat2 < lat1
            cat = 0  # other
            if spd >= 25 and dist < 0.06:
                cat = 1 if is_sb else 2  # 1=SB, 2=NB
            elif 11 <= spd < 25 and lns >= 2 and dist < 0.04:
                cat = 3  # surface
            cs = 1 if (cat == 1 and dist < 0.025) else 0
            links.append([
                round(lat1,6), round(lon1,6),
                round(lat2,6), round(lon2,6),
                cat, cs, l.get('id'),
                round(spd,1), int(lns), int(lng)
            ])
    return links

print("Extracting map data...")
map_links = extract_map_data()
cats = {}
for l in map_links:
    c = l[4]
    cats[c] = cats.get(c, 0) + 1
print(f"  {len(map_links)} links: other={cats.get(0,0)}, SB={cats.get(1,0)}, NB={cats.get(2,0)}, surface={cats.get(3,0)}")

# ── 2. Load conditional results ────────────────────────────────────

with open(OUT_DIR / 'conditional_results.json') as f:
    cond_results = json.load(f)

# ── 3. Read existing HTML & extract body content ───────────────────

html_old = (OUT_DIR / 'crash_report.html').read_text()
# Extract sections 1–12 (between first <section and last </section>)
m_start = html_old.find('<section id="event">')
m_end = html_old.rfind('</section>') + len('</section>')
original_sections = html_old[m_start:m_end]

# ── 4. Build map JS data (compact) ────────────────────────────────

map_js = json.dumps(map_links, separators=(',', ':'))

# ── 5. Build conditional experiment HTML ───────────────────────────

def build_cond_html():
    """Build the conditional experiment tab content."""
    # Group results
    window_rows = [r for r in cond_results if r['in_window']]
    outside_rows = [r for r in cond_results if not r['in_window']]

    html = """
<section>
  <h2>Experiment: Conditional Disruption Modeling</h2>
  <div class="card highlight">
    <h3>Research Question</h3>
    <p>Can we improve crash-day traffic prediction by encoding crash conditions into the
    Two-Phase PageRank transition matrix? We test four conditioning levels using the previous
    Thursday (Sept 13) as input and compare against crash-day (Sept 20) truth.</p>
  </div>

  <h3>Methodology</h3>
  <p>We activate speed sensitivity (<span class="inline-code">&alpha;<sub>s</sub>=1.0</span>,
  <span class="inline-code">&alpha;<sub>l</sub>=0.5</span>) so that speed/lane reductions on
  crash links propagate through the transition weights:</p>

  <div class="pipeline">
    <div class="step">Build M<sub>std</sub><br>(&alpha;<sub>s</sub>=1.0)</div>
    <div class="arrow">&rarr;</div>
    <div class="step">Modify crash<br>link attributes</div>
    <div class="arrow">&rarr;</div>
    <div class="step">Build M<sub>crash</sub></div>
    <div class="arrow">&rarr;</div>
    <div class="step">Run 4-level<br>predictions</div>
    <div class="arrow">&rarr;</div>
    <div class="step">Compare to<br>crash-day truth</div>
  </div>

  <h3>Four Conditioning Levels</h3>
  <table>
    <tr><th>Level</th><th>Matrix</th><th>Teleportation</th><th>Description</th></tr>
    <tr><td><strong>L0</strong></td><td>Standard</td><td>Sept 13</td><td>No crash knowledge (baseline)</td></tr>
    <tr><td><strong>L1</strong></td><td>Crash</td><td>Sept 13</td><td>Matrix encodes crash physics (speed/lane reduction)</td></tr>
    <tr><td><strong>L2</strong></td><td>Standard</td><td>Crash-conditioned</td><td>Input data adjusted for diversion</td></tr>
    <tr><td><strong>L3</strong></td><td>Crash</td><td>Crash-conditioned</td><td>Both matrix and input conditioned</td></tr>
  </table>

  <h3>Crash Link Modifications</h3>
  <table>
    <tr><th>Zone</th><th>Links</th><th>Speed Change</th><th>Lane Change</th><th>Physical Model</th></tr>
    <tr><td>Crash site (5 links)</td><td>59081, 59114, 59128, 65491, 69344</td>
        <td>31.3 &rarr; 2.0 m/s</td><td>6&ndash;7 &rarr; 1</td><td>Near-total closure</td></tr>
    <tr><td>Upstream SB (5 links)</td><td>59088, 59100, 59102, 59103, 59104</td>
        <td>&times; 0.30</td><td>&times; 0.50</td><td>Queue spillback</td></tr>
    <tr><td>I-15 NB (28 links)</td><td>Full NB corridor</td>
        <td>&times; 0.70</td><td>Unchanged</td><td>Rubbernecking</td></tr>
  </table>
</section>

<section>
  <h2>Figures</h2>
  <div class="figure">
    <div class="fig-title">Figure 8: Prediction Error by Conditioning Level</div>
    <img src="fig6_conditional_mre.png" alt="Conditional MRE over time">
    <div class="caption">
      Corridor aggregate error (%) at each time slot for four conditioning levels.
      <strong>Red:</strong> unconditioned. <strong>Orange:</strong> matrix-only.
      <strong>Green:</strong> teleportation-only. <strong>Blue:</strong> fully conditioned.
      Shaded region marks the crash window (07:30&ndash;10:00).
    </div>
  </div>

  <div class="figure">
    <div class="fig-title">Figure 9: Corridor Mass at Peak Disruption</div>
    <img src="fig7_mass_redistribution.png" alt="Mass redistribution at peak">
    <div class="caption">
      Grouped bar chart showing PageRank corridor mass at the peak disruption time.
      Truth (blue) vs. unconditioned (red) vs. matrix-conditioned (orange) vs. fully conditioned (blue).
      Annotations show the error percentage for each prediction.
    </div>
  </div>

  <div class="figure">
    <div class="fig-title">Figure 10: Summary &mdash; Average Error &amp; Improvement</div>
    <img src="fig8_improvement_summary.png" alt="Improvement summary">
    <div class="caption">
      <strong>(A)</strong> Average prediction error during the crash window for each conditioning level.
      <strong>(B)</strong> Error reduction (%) relative to the unconditioned baseline.
    </div>
  </div>
</section>

<section>
  <h2>Results</h2>
  <p>Full results for crash-window time slots (07:30&ndash;10:00):</p>
  <table>
    <tr><th>Time</th><th>Corridor</th><th>Truth</th><th>L0 Err</th><th>L1 Err</th><th>L2 Err</th><th>L3 Err</th></tr>
"""
    for r in window_rows:
        tm = f"{r['truth']:.6f}"
        html += f"    <tr><td>{r['time']}</td><td>{r['corridor']}</td><td>{tm}</td>"
        for k in ['e0','e1','e2','e3']:
            v = r[k]
            cls = 'positive' if v < 20 else ('negative' if v > 50 else '')
            html += f'<td class="{cls}">{v:.1f}%</td>'
        html += "</tr>\n"

    html += """  </table>

  <h3>Crash-Window Averages</h3>
  <table>
    <tr><th>Corridor</th><th>L0 (Uncond)</th><th>L1 (Matrix)</th><th>L2 (Tele)</th><th>L3 (Full)</th></tr>
"""
    for cn in ['I-15 SB', 'I-15 NB', 'Surface']:
        cw = [r for r in window_rows if r['corridor'] == cn]
        avgs = {k: sum(r[k] for r in cw)/len(cw) for k in ['e0','e1','e2','e3']}
        html += f"    <tr><td><strong>{cn}</strong></td>"
        for k in ['e0','e1','e2','e3']:
            html += f"<td>{avgs[k]:.1f}%</td>"
        html += "</tr>\n"

    html += """  </table>
</section>

<section>
  <h2>Analysis &amp; Key Findings</h2>

  <div class="card warning">
    <h3>Why Matrix Conditioning Has Limited Effect</h3>
    <p>With the default parameters (&alpha;<sub>s</sub>=0, &alpha;<sub>l</sub>=0, &mu;=20), the transition
    matrix is dominated by <strong>self-loops</strong> (&gt;99% probability) and
    <strong>teleportation</strong> (20% direct injection). Modifying 21 out of 99,716 links changes
    only ~0.02% of the flowing mass. Even with &alpha;<sub>s</sub>=1.0 activated, the self-loop
    dominance limits the propagation of local crash conditions through the matrix.</p>
  </div>

  <div class="card success">
    <h3>Individual Time-Slot Successes</h3>
    <ul style="margin-left:20px;">
      <li><strong>I-15 SB at 09:00</strong> (matrix conditioning): Error improved from 1.8% to
      <strong>0.6%</strong> &mdash; a 67% relative improvement during peak crash.</li>
      <li><strong>I-15 NB at 09:00</strong> (teleportation conditioning): Error dropped from 30.7% to
      <strong>4.6%</strong> &mdash; an 85% improvement, correctly predicting rubbernecking effects.</li>
      <li><strong>Surface at 08:00</strong> (teleportation conditioning): Error improved from 9.8% to
      <strong>8.3%</strong>, correctly predicting increased diversion to arterials.</li>
    </ul>
  </div>

  <div class="card highlight">
    <h3>The Crash Is Temporally Complex</h3>
    <p>The key challenge is that the crash effect is <strong>not monotonic</strong>:</p>
    <ul style="margin-left:20px;">
      <li><strong>07:30 (onset):</strong> SB mass is <em>higher</em> than baseline (vehicles piling up)</li>
      <li><strong>08:00 (active):</strong> SB mass is <em>lower</em> (diversion taking effect)</li>
      <li><strong>09:00 (peak):</strong> SB mass approximately equals baseline (equilibrium)</li>
      <li><strong>10:00 (clearance):</strong> SB mass is 6&times; <em>higher</em> (rebound surge)</li>
    </ul>
    <p>Our uniform &ldquo;reduce SB mass&rdquo; conditioning only matches one of these four phases.
    A time-varying conditioning model matched to the incident timeline would be needed for
    consistent improvement across all phases.</p>
  </div>

  <div class="card">
    <h3>Structural Insight: Detection vs. Prediction</h3>
    <p>This experiment reveals a fundamental property of the Two-Phase PageRank model:</p>
    <ul style="margin-left:20px;">
      <li>The model excels at <strong>anomaly detection</strong> (Tab 1): comparing crash-day vs.
      baseline distributions successfully identifies the disruption signature.</li>
      <li>For <strong>forward prediction</strong> from a different week&rsquo;s data, week-to-week
      variation (30&ndash;95% on freeway corridors) dominates the crash signal.</li>
      <li>The model&rsquo;s <strong>stability</strong> (high &mu;, moderate damping) is both a
      strength for accurate steady-state prediction and a limitation for scenario analysis.</li>
    </ul>
  </div>

  <h3>Runtime</h3>
  <p>With optimized NPZ caching (single load), the full experiment completes in
  <strong>11.4 seconds</strong> (7 time slots &times; up to 5 PageRank runs each, ~60ms per run).</p>
</section>
"""
    return html


cond_html = build_cond_html()

# ── 6. Assemble final HTML ─────────────────────────────────────────

final_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>I-15 Crash Anomaly Analysis &mdash; Two-Phase PageRank</title>
<style>
  :root {{
    --accent: #1565C0;
    --accent-light: #e3f2fd;
    --red: #c62828;
    --green: #2e7d32;
    --bg: #fafafa;
    --card: #ffffff;
    --text: #212121;
    --muted: #757575;
    --border: #e0e0e0;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7; padding: 0;
  }}
  .hero {{
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 60%, #42a5f5 100%);
    color: white; padding: 60px 40px 50px; text-align: center;
  }}
  .hero h1 {{ font-size: 2.4em; margin-bottom: 10px; font-weight: 700; }}
  .hero .subtitle {{ font-size: 1.2em; opacity: 0.9; margin-bottom: 20px; }}
  .hero .meta {{
    display: inline-flex; gap: 30px; flex-wrap: wrap; justify-content: center;
    font-size: 0.95em; opacity: 0.85; margin-top: 10px;
  }}
  .hero .meta span {{ display: flex; align-items: center; gap: 6px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 30px; }}
  section {{ margin-bottom: 50px; }}
  h2 {{
    font-size: 1.6em; color: var(--accent); margin-bottom: 18px;
    padding-bottom: 8px; border-bottom: 3px solid var(--accent);
  }}
  h3 {{ font-size: 1.2em; color: #333; margin: 20px 0 10px; }}
  p {{ margin-bottom: 14px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 24px 28px; margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .card.highlight {{ border-left: 5px solid var(--accent); background: var(--accent-light); }}
  .card.warning {{ border-left: 5px solid var(--red); background: #fef2f2; }}
  .card.success {{ border-left: 5px solid var(--green); background: #f1f8e9; }}
  .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 16px 0; }}
  .info-item {{ display: flex; gap: 8px; }}
  .info-label {{ font-weight: 600; color: var(--muted); min-width: 140px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.95em; }}
  th {{ background: var(--accent); color: white; padding: 10px 14px; text-align: left; font-weight: 600; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .positive {{ color: var(--green); font-weight: 600; }}
  .negative {{ color: var(--red); font-weight: 600; }}
  .figure {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin: 24px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .figure img {{ width: 100%; height: auto; border-radius: 6px; border: 1px solid #eee; }}
  .figure .caption {{
    margin-top: 14px; padding: 12px 16px; background: #f5f5f5;
    border-radius: 6px; font-size: 0.93em; color: #555; line-height: 1.6;
  }}
  .figure .caption strong {{ color: var(--accent); }}
  .figure .fig-title {{ font-weight: 700; font-size: 1.05em; color: var(--accent); margin-bottom: 12px; }}
  pre {{
    background: #263238; color: #eeffff; padding: 18px 22px;
    border-radius: 8px; overflow-x: auto; font-size: 0.88em; line-height: 1.6; margin: 14px 0;
  }}
  code {{ font-family: 'Fira Code', 'Consolas', monospace; }}
  .inline-code {{
    background: #eceff1; padding: 2px 7px; border-radius: 4px;
    font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.9em; color: #d32f2f;
  }}
  .pipeline {{
    display: flex; align-items: center; gap: 0; margin: 20px 0;
    flex-wrap: wrap; justify-content: center;
  }}
  .pipeline .step {{
    background: var(--accent); color: white; padding: 10px 18px;
    border-radius: 8px; font-size: 0.9em; font-weight: 600; text-align: center; min-width: 120px;
  }}
  .pipeline .arrow {{ font-size: 1.5em; color: var(--accent); padding: 0 8px; }}
  .timeline {{ position: relative; padding-left: 30px; margin: 20px 0; }}
  .timeline::before {{
    content: ''; position: absolute; left: 10px; top: 0; bottom: 0;
    width: 3px; background: var(--accent);
  }}
  .timeline .event {{ position: relative; margin-bottom: 20px; padding-left: 20px; }}
  .timeline .event::before {{
    content: ''; position: absolute; left: -24px; top: 6px;
    width: 12px; height: 12px; background: var(--accent);
    border-radius: 50%; border: 2px solid white; box-shadow: 0 0 0 2px var(--accent);
  }}
  .timeline .event.crash::before {{ background: var(--red); box-shadow: 0 0 0 2px var(--red); }}
  .timeline .event .time {{ font-weight: 700; color: var(--accent); }}
  .timeline .event.crash .time {{ color: var(--red); }}
  footer {{
    text-align: center; padding: 30px; color: var(--muted);
    font-size: 0.9em; border-top: 1px solid var(--border); margin-top: 40px;
  }}
  .toc {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 24px 30px; margin-bottom: 40px; }}
  .toc h3 {{ margin-top: 0; color: var(--accent); }}
  .toc ol {{ padding-left: 24px; }}
  .toc li {{ margin: 6px 0; }}
  .toc a {{ color: var(--accent); text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}

  /* ── Tabs ── */
  .tab-bar {{
    display: flex; gap: 0; margin-bottom: 0; border-bottom: 3px solid var(--accent);
  }}
  .tab-btn {{
    padding: 14px 28px; font-size: 1.05em; font-weight: 600; cursor: pointer;
    border: 1px solid var(--border); border-bottom: none;
    border-radius: 10px 10px 0 0; background: #e8e8e8; color: var(--muted);
    transition: all 0.2s;
  }}
  .tab-btn:hover {{ background: #f0f0f0; }}
  .tab-btn.active {{
    background: var(--accent); color: white; border-color: var(--accent);
  }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* ── Network Map ── */
  #map-section {{ margin-bottom: 40px; }}
  #map-wrap {{
    position: relative; background: #1a1a2e; border-radius: 12px;
    overflow: hidden; border: 2px solid var(--border);
  }}
  #map-canvas {{ display: block; width: 100%; cursor: crosshair; }}
  #map-tooltip {{
    display: none; position: absolute; background: rgba(0,0,0,0.88);
    color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 0.85em;
    pointer-events: none; z-index: 100; max-width: 280px; line-height: 1.5;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }}
  .map-legend {{
    display: flex; gap: 20px; flex-wrap: wrap; margin-top: 14px;
    justify-content: center; font-size: 0.9em;
  }}
  .map-legend span {{ display: flex; align-items: center; gap: 6px; }}
  .map-legend .swatch {{
    display: inline-block; width: 28px; height: 4px; border-radius: 2px;
  }}
  .map-legend .swatch.crash {{ height: 6px; }}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <h1>I-15 Multi-Vehicle Crash Anomaly Analysis</h1>
  <div class="subtitle">Validating the Two-Phase PageRank Model Against a Real-World Traffic Incident</div>
  <div class="meta">
    <span>Salt Lake City, Utah</span>
    <span>September 20, 2018</span>
    <span>I-15 Southbound at 14400 South, Draper</span>
  </div>
</div>

<div class="container">

<!-- ============================================================== -->
<!-- INTERACTIVE NETWORK MAP -->
<!-- ============================================================== -->
<section id="map-section">
  <h2>Interactive Network Map &mdash; I-15 Crash Corridor</h2>
  <p>Hover over any link to see its properties. The map shows {len(map_links):,} road links
  within ~4 km of the crash site. Crash-site links pulse in bright red.</p>
  <div id="map-wrap">
    <canvas id="map-canvas" width="1060" height="620"></canvas>
    <div id="map-tooltip"></div>
  </div>
  <div class="map-legend">
    <span><span class="swatch crash" style="background:#ff1744"></span> Crash-site I-15 SB</span>
    <span><span class="swatch" style="background:#ef5350"></span> I-15 Southbound</span>
    <span><span class="swatch" style="background:#ffa726"></span> I-15 Northbound</span>
    <span><span class="swatch" style="background:#66bb6a"></span> Surface Arterials</span>
    <span><span class="swatch" style="background:#455a64"></span> Other Roads</span>
    <span><span class="swatch" style="background:#ffeb3b; height:8px; width:8px; border-radius:50%"></span> Crash Site</span>
  </div>
</section>

<!-- ============================================================== -->
<!-- TABS -->
<!-- ============================================================== -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('detection')">Crash Detection Analysis</button>
  <button class="tab-btn" onclick="showTab('conditional')">Conditional Experiment</button>
</div>

<!-- TAB 1: Original crash detection content -->
<div id="tab-detection" class="tab-content active">
{original_sections}
</div>

<!-- TAB 2: Conditional experiment -->
<div id="tab-conditional" class="tab-content">
{cond_html}
</div>

</div><!-- /container -->

<footer>
  I-15 Crash Anomaly Analysis &mdash; Two-Phase PageRank with Travel-Time Self-Loops
  &mdash; Salt Lake City Road Network (N = 99,716 links) &mdash; Generated April 2026
</footer>

<script>
// ── Map data: [lat1, lon1, lat2, lon2, cat, crashSite, id, speed, lanes, length] ──
const ML = {map_js};
const CRASH = [40.524, -111.893];
const CAT_NAMES = ['Other road', 'I-15 Southbound', 'I-15 Northbound', 'Surface Arterial'];
const CAT_COLORS = ['#546e7a', '#ef5350', '#ffa726', '#66bb6a'];
const CRASH_COLOR = '#ff1744';

// ── Canvas setup ──
const canvas = document.getElementById('map-canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('map-tooltip');
const W = canvas.width, H = canvas.height;
const PAD = 30;

// Compute bounds
let minLat=90, maxLat=-90, minLon=180, maxLon=-180;
ML.forEach(l => {{
  minLat=Math.min(minLat,l[0],l[2]); maxLat=Math.max(maxLat,l[0],l[2]);
  minLon=Math.min(minLon,l[1],l[3]); maxLon=Math.max(maxLon,l[1],l[3]);
}});
const latR = maxLat-minLat, lonR = maxLon-minLon;

function toX(lon) {{ return PAD + (lon-minLon)/lonR*(W-2*PAD); }}
function toY(lat) {{ return PAD + (maxLat-lat)/latR*(H-2*PAD); }}

// ── Sort: draw other first, then surface, NB, SB, crash-site last ──
const order = [0, 3, 2, 1];
const sorted = [];
order.forEach(cat => ML.forEach((l,i) => {{ if(l[4]===cat && !l[5]) sorted.push(i); }}));
ML.forEach((l,i) => {{ if(l[5]) sorted.push(i); }});

// ── Draw ──
let animPhase = 0;
function draw() {{
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 0.5;
  for(let lat=Math.ceil(minLat*100)/100; lat<=maxLat; lat+=0.01) {{
    const y = toY(lat);
    ctx.beginPath(); ctx.moveTo(PAD,y); ctx.lineTo(W-PAD,y); ctx.stroke();
  }}
  for(let lon=Math.ceil(minLon*100)/100; lon<=maxLon; lon+=0.01) {{
    const x = toX(lon);
    ctx.beginPath(); ctx.moveTo(x,PAD); ctx.lineTo(x,H-PAD); ctx.stroke();
  }}

  // Links
  sorted.forEach(i => {{
    const l = ML[i];
    const x1=toX(l[1]), y1=toY(l[0]), x2=toX(l[3]), y2=toY(l[2]);
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    if(l[5]) {{
      ctx.strokeStyle = CRASH_COLOR;
      ctx.lineWidth = 3.5 + 1.5*Math.sin(animPhase);
      ctx.globalAlpha = 0.7 + 0.3*Math.sin(animPhase);
    }} else {{
      ctx.strokeStyle = CAT_COLORS[l[4]];
      ctx.lineWidth = l[4]===0 ? 0.8 : (l[4]===3 ? 1.5 : 2.5);
      ctx.globalAlpha = l[4]===0 ? 0.35 : 0.85;
    }}
    ctx.stroke();
    ctx.globalAlpha = 1;
  }});

  // Crash site marker
  const cx=toX(CRASH[1]), cy=toY(CRASH[0]);
  ctx.beginPath();
  ctx.arc(cx, cy, 8+2*Math.sin(animPhase), 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,235,59,0.3)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI*2);
  ctx.fillStyle = '#ffeb3b';
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Label
  ctx.fillStyle = '#ffeb3b';
  ctx.font = 'bold 11px sans-serif';
  ctx.fillText('CRASH SITE', cx+12, cy+4);

  animPhase += 0.06;
  requestAnimationFrame(draw);
}}
draw();

// ── Hit detection ──
canvas.addEventListener('mousemove', function(e) {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = W / rect.width, scaleY = H / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;
  let best = -1, bestDist = 12;

  for(let i=0; i<ML.length; i++) {{
    const l = ML[i];
    const x1=toX(l[1]), y1=toY(l[0]), x2=toX(l[3]), y2=toY(l[2]);
    const dx=x2-x1, dy=y2-y1;
    const len2 = dx*dx + dy*dy;
    if(len2 < 1) continue;
    let t = ((mx-x1)*dx + (my-y1)*dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const px = x1+t*dx, py = y1+t*dy;
    const d = Math.sqrt((mx-px)*(mx-px) + (my-py)*(my-py));
    if(d < bestDist) {{ bestDist = d; best = i; }}
  }}

  if(best >= 0) {{
    const l = ML[best];
    const catName = l[5] ? '<span style="color:#ff1744;font-weight:bold">CRASH-SITE LINK</span>' :
                   CAT_NAMES[l[4]];
    tooltip.innerHTML =
      '<div style="margin-bottom:4px;font-weight:bold">Link #' + l[6] + '</div>' +
      '<div>' + catName + '</div>' +
      '<div>Speed: ' + l[7] + ' m/s (' + Math.round(l[7]*2.237) + ' mph)</div>' +
      '<div>Lanes: ' + l[8] + '</div>' +
      '<div>Length: ' + l[9] + ' m</div>';
    tooltip.style.display = 'block';
    const tipX = e.clientX - canvas.getBoundingClientRect().left;
    const tipY = e.clientY - canvas.getBoundingClientRect().top;
    tooltip.style.left = (tipX + 15) + 'px';
    tooltip.style.top = (tipY - 10) + 'px';
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
canvas.addEventListener('mouseleave', () => {{ tooltip.style.display='none'; }});

// ── Tab switching ──
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>'''

# ── 7. Write output ────────────────────────────────────────────────

out_path = OUT_DIR / 'crash_report.html'
out_path.write_text(final_html)
print(f"\nWrote {out_path} ({len(final_html):,} bytes)")
