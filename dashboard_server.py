from flask import Flask, Response, jsonify
import json
import os
import logging

# 禁用 Flask 日志刷屏
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>MilCube MVP Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg-primary: #0f1419;
      --bg-secondary: #1a1f2e;
      --bg-card: #232b3e;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --border: #334155;
      --shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.2);
    }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      padding: 20px;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }

    .header h1 {
      font-size: 24px;
      font-weight: 600;
      background: linear-gradient(135deg, var(--accent), #8b5cf6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .header-controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      background: var(--bg-card);
      border: 1px solid var(--border);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    .status-dot.online { background: var(--success); }
    .status-dot.offline { background: var(--danger); animation: none; }
    .status-dot.paused { background: var(--warning); animation: none; }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .btn-primary { background: var(--accent); color: white; }
    .btn-primary:hover { background: var(--accent-hover); transform: translateY(-1px); }
    .btn-secondary { background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border); }
    .btn-secondary:hover { background: var(--bg-secondary); border-color: var(--accent); }

    .grid { display: grid; gap: 16px; }
    .grid-2 { grid-template-columns: repeat(2, 1fr); }
    .grid-3 { grid-template-columns: 2fr 1fr; }

    @media (max-width: 1200px) {
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--bg-card);
      border-radius: 12px;
      border: 1px solid var(--border);
      overflow: hidden;
      box-shadow: var(--shadow);
    }

    .card-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .card-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .card-subtitle { font-size: 12px; color: var(--text-muted); }
    .card-body { padding: 16px 20px; }
    .card-body.no-padding { padding: 0; }

    .video-container {
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
    }

    .video-container img {
      width: 100%;
      height: 360px;
      object-fit: contain;
      display: block;
    }

    .video-overlay {
      position: absolute;
      top: 12px;
      left: 12px;
      display: flex;
      gap: 8px;
    }

    .video-tag {
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      background: rgba(0,0,0,0.6);
      backdrop-filter: blur(4px);
    }

    .video-tag.live { background: rgba(239,68,68,0.8); }

    .kpi-row {
      display: flex;
      gap: 16px;
      padding: 12px 16px;
      background: var(--bg-secondary);
      border-radius: 8px;
      margin-top: 12px;
      font-size: 12px;
      flex-wrap: wrap;
    }

    .kpi-item { display: flex; align-items: center; gap: 6px; }
    .kpi-label { color: var(--text-muted); }
    .kpi-value { color: var(--text-primary); font-weight: 600; font-family: 'Consolas', monospace; }

    .pair-stats {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }

    .stat-box {
      background: var(--bg-secondary);
      padding: 12px 16px;
      border-radius: 8px;
      text-align: center;
    }

    .stat-value { font-size: 24px; font-weight: 700; color: var(--accent); }
    .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }

    .pair-list { margin-top: 12px; }

    .pair-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 12px;
      background: var(--bg-secondary);
      border-radius: 8px;
      margin-bottom: 8px;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s;
      border: 2px solid transparent;
    }

    .pair-item:hover { border-color: var(--accent); background: rgba(59,130,246,0.1); }
    .pair-item.selected { border-color: var(--success); background: rgba(16,185,129,0.15); }

    .pair-link { font-family: 'Consolas', monospace; color: var(--accent); font-weight: 500; }
    .pair-meta { color: var(--text-muted); font-size: 11px; }

    .chart-container { height: 180px; padding: 8px; }

    .heat-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 16px;
    }

    .heat-canvas { border-radius: 8px; background: var(--bg-secondary); }
    .heat-label { margin-top: 8px; font-size: 11px; color: var(--text-muted); }

    .hint {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 12px;
      padding: 8px 12px;
      background: var(--bg-secondary);
      border-radius: 6px;
      border-left: 3px solid var(--accent);
    }

    .offline-msg { color: var(--danger); font-weight: 500; }

    .method-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      margin-left: 8px;
    }
    .method-badge.color { background: var(--success); color: white; }
    .method-badge.basic { background: var(--warning); color: black; }

    /* Focus Panel */
    .focus-panel {
      display: none;
      margin-top: 16px;
    }
    .focus-panel.active { display: block; }

    .focus-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .focus-card {
      background: var(--bg-secondary);
      border-radius: 12px;
      padding: 16px;
      text-align: center;
    }

    .focus-card h4 {
      margin-bottom: 12px;
      color: var(--accent);
    }

    .focus-crop {
      width: 100%;
      height: 200px;
      object-fit: contain;
      border-radius: 8px;
      border: 2px solid var(--accent);
      background: #000;
    }

    .focus-crop-container {
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      min-height: 200px;
    }

    .focus-crop-label {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0,0,0,0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 11px;
      color: #fff;
    }

    .color-bar {
      height: 20px;
      border-radius: 4px;
      margin-top: 8px;
      display: flex;
      overflow: hidden;
    }

    .color-bar-segment {
      flex: 1;
      min-width: 0;
    }

    .similarity-display {
      text-align: center;
      padding: 20px;
      background: var(--bg-secondary);
      border-radius: 12px;
    }

    .similarity-value {
      font-size: 48px;
      font-weight: 700;
      color: var(--success);
    }

    .similarity-label {
      font-size: 14px;
      color: var(--text-muted);
      margin-top: 8px;
    }
  </style>
</head>
<body>

  <div class="header">
    <h1>MilCube MVP Dashboard</h1>
    <div class="header-controls">
      <div class="status-badge">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">Connecting...</span>
      </div>
      <div class="status-badge" id="methodBadgeContainer">
        <span id="methodStatus">Matching: COLOR</span>
      </div>
      <button class="btn btn-secondary" onclick="pauseBoth(true)">Pause</button>
      <button class="btn btn-primary" onclick="pauseBoth(false)">Resume</button>
    </div>
  </div>

  <div class="grid grid-2" style="margin-bottom: 16px;">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Camera 5</span>
        <a href="http://127.0.0.1:5000/video" target="_blank" class="card-subtitle" style="text-decoration: none; color: var(--accent);">Fullscreen</a>
      </div>
      <div class="card-body no-padding">
        <div class="video-container">
          <img id="vid5" src="http://127.0.0.1:5000/video" alt="cam5" />
          <div class="video-overlay">
            <span class="video-tag live">LIVE</span>
            <span class="video-tag">CAM5</span>
          </div>
        </div>
        <div class="kpi-row" id="kpi5"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span class="card-title">Camera 6</span>
        <a href="http://127.0.0.1:5001/video" target="_blank" class="card-subtitle" style="text-decoration: none; color: var(--accent);">Fullscreen</a>
      </div>
      <div class="card-body no-padding">
        <div class="video-container">
          <img id="vid6" src="http://127.0.0.1:5001/video" alt="cam6" />
          <div class="video-overlay">
            <span class="video-tag live">LIVE</span>
            <span class="video-tag">CAM6</span>
          </div>
        </div>
        <div class="kpi-row" id="kpi6"></div>
      </div>
    </div>
  </div>

  <!-- Focus Panel (shows when a pair is clicked) -->
  <div class="focus-panel" id="focusPanel">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Person Focus <span class="method-badge color" id="focusBadge">MATCHED</span></span>
        <button class="btn btn-secondary" onclick="closeFocus()" style="padding: 6px 12px;">Close</button>
      </div>
      <div class="card-body">
        <div class="focus-grid" style="grid-template-columns: 1fr 160px 1fr;">
          <div class="focus-card">
            <h4>CAM5 - Track <span id="focusT5">?</span></h4>
            <div class="focus-crop-container">
              <img id="crop5" class="focus-crop" src="" alt="Person from CAM5" />
              <span class="focus-crop-label">CAM5</span>
            </div>
            <div class="color-bar" id="colorBar5"></div>
          </div>
          <div class="similarity-display">
            <div class="similarity-value" id="simValue">--</div>
            <div class="similarity-label">Color Similarity</div>
            <div style="margin-top: 16px; font-size: 12px; color: var(--text-muted);">
              <div>Position Dist: <span id="posDist">--</span></div>
              <div>Score: <span id="combScore">--</span></div>
            </div>
          </div>
          <div class="focus-card">
            <h4>CAM6 - Track <span id="focusT6">?</span></h4>
            <div class="focus-crop-container">
              <img id="crop6" class="focus-crop" src="" alt="Person from CAM6" />
              <span class="focus-crop-label">CAM6</span>
            </div>
            <div class="color-bar" id="colorBar6"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="grid" style="margin-bottom: 16px;">
    <div class="card">
      <div class="card-header">
        <span class="card-title">
          Cross-Camera Association
          <span class="method-badge color" id="matchMethodBadge">COLOR + POSITION</span>
        </span>
        <span class="card-subtitle">Click a pair to focus</span>
      </div>
      <div class="card-body">
        <div class="pair-stats" id="pairKpi">
          <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">Pairs</div></div>
          <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">Stable</div></div>
          <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">Avg Similarity</div></div>
          <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">Churn</div></div>
        </div>
        <div class="pair-list" id="pairTop"></div>
        <div class="hint" id="pairHint">Matching uses color histogram + position</div>
      </div>
    </div>
  </div>

  <div class="grid" style="grid-template-columns: 2fr 1fr 1fr; gap: 16px;">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Time Series</span>
      </div>
      <div class="card-body no-padding">
        <div class="chart-container"><canvas id="chart"></canvas></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Density Heatmap</span>
      </div>
      <div class="card-body">
        <div class="heat-container">
          <canvas id="heat" class="heat-canvas" width="200" height="120"></canvas>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Eval Stats</span>
        <button class="btn btn-secondary" onclick="exportLog()" style="padding:4px 8px;font-size:11px;">Export</button>
      </div>
      <div class="card-body">
        <div style="font-size:11px;color:var(--text-secondary);">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span>Frames:</span><span id="statFrames" style="color:var(--text-primary)">0</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span>Changes:</span><span id="statChanges" style="color:var(--text-primary)">0</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span>Stability:</span><span id="statStability" style="color:var(--success)">--%</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span>Locked:</span><span id="statLocked" style="color:var(--accent)">0</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- 实时日志 -->
  <div class="card" style="margin-top:16px;">
    <div class="card-header">
      <span class="card-title">Live Matching Log</span>
      <button class="btn btn-secondary" onclick="clearLog()" style="padding:4px 8px;font-size:11px;">Clear</button>
    </div>
    <div class="card-body" style="max-height:150px;overflow-y:auto;font-family:monospace;font-size:11px;" id="liveLog">
    </div>
  </div>

<script>
const CAM5 = "http://127.0.0.1:5000";
const CAM6 = "http://127.0.0.1:5001";

// Global state
let latestM5 = null;
let latestM6 = null;
let selectedPair = null;

async function pauseBoth(v){
  try {
    await fetch(CAM5 + "/control/pause", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({paused:v})});
    await fetch(CAM6 + "/control/pause", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({paused:v})});
  } catch(e) {}
}

// ========== Feature Matching (Re-ID + Histogram) ==========

// Cosine similarity for Re-ID features (L2-normalized, so dot product = cosine)
function cosineSimilarity(f1, f2) {
  if (!f1 || !f2 || f1.length !== f2.length) return 0;
  let dot = 0;
  for (let i = 0; i < f1.length; i++) {
    dot += f1[i] * f2[i];
  }
  // Result is in [-1, 1], but for normalized features it's typically [0, 1]
  return Math.max(0, dot);
}

// Histogram intersection for color histograms
function histogramIntersection(h1, h2) {
  if (!h1 || !h2 || h1.length !== h2.length) return 0;
  let sum = 0;
  for (let i = 0; i < h1.length; i++) {
    sum += Math.min(h1[i], h2[i]);
  }
  return sum; // 0~1, higher = more similar
}

function bhattacharyyaDistance(h1, h2) {
  if (!h1 || !h2 || h1.length !== h2.length) return 1;
  let sum = 0;
  for (let i = 0; i < h1.length; i++) {
    sum += Math.sqrt(h1[i] * h2[i]);
  }
  return 1 - sum; // 0~1, lower = more similar
}

// Unified similarity function that handles both Re-ID and histogram
function computeSimilarity(p5, p6) {
  // Prefer Re-ID features if available
  if (p5.reid_feat && p6.reid_feat) {
    return {
      sim: cosineSimilarity(p5.reid_feat, p6.reid_feat),
      method: "reid"
    };
  }
  // Fallback to histogram
  if (p5.color_hist && p6.color_hist) {
    return {
      sim: histogramIntersection(p5.color_hist, p6.color_hist),
      method: "histogram"
    };
  }
  return { sim: 0, method: "none" };
}

// ========== Improved Matcher with Re-ID / Color ==========
let currentMatchMethod = "detecting...";

function matchTracks(m5, m6) {
  const a = (m5 && m5.tracks_xy) ? m5.tracks_xy : [];
  const b = (m6 && m6.tracks_xy) ? m6.tracks_xy : [];

  // 返回未匹配的 tracks
  const unmatched5 = [];
  const unmatched6 = [];

  if (!a.length && !b.length) return {pairs: [], unmatched5: [], unmatched6: [], method: "none"};
  if (!a.length) return {pairs: [], unmatched5: [], unmatched6: b.map(t => t.tid), method: "none"};
  if (!b.length) return {pairs: [], unmatched5: a.map(t => t.tid), unmatched6: [], method: "none"};

  // Detect feature type from first track
  const featureType = a[0].feature_type || (a[0].reid_feat ? "reid" : "histogram");
  currentMatchMethod = featureType === "reid" ? "Re-ID (OSNet)" : "Color Histogram";

  const w5 = m5.frame_w || 1, h5 = m5.frame_h || 1;
  const w6 = m6.frame_w || 1, h6 = m6.frame_h || 1;

  // 计算所有候选配对的综合分数
  const candidates = [];
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      const p5 = a[i], p6 = b[j];

      // 计算相似度（支持 Re-ID 和 Histogram）
      const simResult = computeSimilarity(p5, p6);
      const similarity = simResult.sim;

      // cost = 1 - similarity
      const cost = 1 - similarity;
      const posDist = 0; // 不使用位置

      candidates.push({
        i, j, cost, posDist,
        colorSim: similarity,  // 统一命名为 colorSim 以兼容现有代码
        matchMethod: simResult.method,
        t5: p5.tid, t6: p6.tid,
        hist5: p5.color_hist || null,
        hist6: p6.color_hist || null,
        reid5: p5.reid_feat || null,
        reid6: p6.reid_feat || null,
        box5: m5.tracks_box ? m5.tracks_box.find(t => t.tid === p5.tid) : null,
        box6: m6.tracks_box ? m6.tracks_box.find(t => t.tid === p6.tid) : null
      });
    }
  }

  // 按综合分数排序
  candidates.sort((p, q) => p.cost - q.cost);

  // 贪婪匹配 - Re-ID 使用更严格的阈值
  const usedA = new Set();
  const usedB = new Set();
  const pairs = [];

  // Re-ID 阈值更严格（余弦相似度通常更高）
  const COST_TH = featureType === "reid" ? 0.25 : 0.30;
  const MIN_SIM = featureType === "reid" ? 0.70 : 0.65;

  // 第一步：处理已锁定的配对（优先级最高）
  for (const c of candidates) {
    if (usedA.has(c.i) || usedB.has(c.j)) continue;
    // 检查是否是锁定配对
    if (lockedPairs.has(c.t5) && lockedPairs.get(c.t5) === c.t6) {
      usedA.add(c.i);
      usedB.add(c.j);
      pairs.push({
        t5: c.t5,
        t6: c.t6,
        cost: Number(c.cost.toFixed(3)),
        colorSim: Number(c.colorSim.toFixed(3)),
        posDist: 0,
        hist5: c.hist5,
        hist6: c.hist6,
        box5: c.box5,
        box6: c.box6,
        locked: true
      });
    }
  }

  // 第二步：处理新配对（需要满足阈值，且不能违反锁定）
  for (const c of candidates) {
    if (c.cost > COST_TH) break;
    if (c.colorSim < MIN_SIM) continue;
    if (usedA.has(c.i) || usedB.has(c.j)) continue;
    // 检查锁定约束
    if (!canPair(c.t5, c.t6)) continue;
    usedA.add(c.i);
    usedB.add(c.j);
    pairs.push({
      t5: c.t5,
      t6: c.t6,
      cost: Number(c.cost.toFixed(3)),
      colorSim: Number(c.colorSim.toFixed(3)),
      posDist: 0,
      hist5: c.hist5,
      hist6: c.hist6,
      reid5: c.reid5,
      reid6: c.reid6,
      box5: c.box5,
      box6: c.box6,
      locked: false,
      matchMethod: c.matchMethod
    });
  }

  // 找出未匹配的 tracks
  for (let i = 0; i < a.length; i++) {
    if (!usedA.has(i)) unmatched5.push(a[i].tid);
  }
  for (let j = 0; j < b.length; j++) {
    if (!usedB.has(j)) unmatched6.push(b[j].tid);
  }

  pairs.sort((x, y) => y.colorSim - x.colorSim);
  return {pairs, unmatched5, unmatched6, method: "color+position"};
}

// Pair stability memory
const pairMem = new Map();
let lastPairsCount = 0;
let churn = 0;

// 配对锁定：稳定配对后锁定，防止跳变
const lockedPairs = new Map();  // t5 -> t6
const lockedPairsRev = new Map();  // t6 -> t5
const LOCK_AFTER_FRAMES = 4;  // 稳定 4 帧后锁定（更快锁定）
const LOCK_TIMEOUT_MS = 2000;  // 2秒无更新则解锁

function updatePairMem(mm){
  const now = Date.now();
  const curPairs = (mm && mm.pairs) ? mm.pairs : [];

  for(const p of curPairs){
    const key = `${p.t5}-${p.t6}`;
    if(!pairMem.has(key)){
      pairMem.set(key, {last:now, frames:1, simAvg: p.colorSim, t5: p.t5, t6: p.t6});
    }else{
      const s = pairMem.get(key);
      s.last = now;
      s.frames += 1;
      s.simAvg = 0.8 * s.simAvg + 0.2 * p.colorSim;
      pairMem.set(key, s);

      // 稳定后锁定
      if(s.frames >= LOCK_AFTER_FRAMES){
        lockedPairs.set(p.t5, p.t6);
        lockedPairsRev.set(p.t6, p.t5);
      }
    }
  }

  // 清理过期的 pairMem
  for(const [k,v] of pairMem.entries()){
    if(now - v.last > 2000) pairMem.delete(k);
  }

  // 清理过期的锁定
  for(const [t5, t6] of lockedPairs.entries()){
    const key = `${t5}-${t6}`;
    const mem = pairMem.get(key);
    if(!mem || now - mem.last > LOCK_TIMEOUT_MS){
      lockedPairs.delete(t5);
      lockedPairsRev.delete(t6);
    }
  }

  const cur = curPairs.length;
  churn = 0.7*churn + 0.3*Math.abs(cur - lastPairsCount);
  lastPairsCount = cur;
}

// 检查是否可以配对（考虑锁定）
function canPair(t5, t6){
  // 如果 t5 已锁定
  if(lockedPairs.has(t5)){
    return lockedPairs.get(t5) === t6;  // 只能和锁定的对象配对
  }
  // 如果 t6 已锁定
  if(lockedPairsRev.has(t6)){
    return lockedPairsRev.get(t6) === t5;  // 只能和锁定的对象配对
  }
  return true;  // 都没锁定，可以配对
}

function summarizePairs(mm){
  const curPairs = (mm && mm.pairs) ? mm.pairs : [];
  const enriched = curPairs.map(p=>{
    const key = `${p.t5}-${p.t6}`;
    const s = pairMem.get(key);
    return {
      ...p,
      frames: s ? s.frames : 0,
      simAvg: s ? s.simAvg : p.colorSim,
    };
  });

  enriched.sort((a,b)=> (b.frames - a.frames) || (b.simAvg - a.simAvg));
  const stable = enriched.filter(p => p.frames >= 6);
  const avgSim = enriched.length ? (enriched.reduce((x,y) => x + y.simAvg, 0) / enriched.length) : 0;

  return {enriched, stable, avgSim};
}

// ========== Focus Panel ==========
let cropRefreshTimer = null;

function showFocus(pair) {
  selectedPair = pair;
  document.getElementById("focusPanel").classList.add("active");
  document.getElementById("focusT5").textContent = pair.t5;
  document.getElementById("focusT6").textContent = pair.t6;
  document.getElementById("simValue").textContent = (pair.colorSim * 100).toFixed(0) + "%";
  document.getElementById("posDist").textContent = pair.posDist.toFixed(3);
  document.getElementById("combScore").textContent = pair.cost.toFixed(3);

  // Update focus badge based on matching method
  const focusBadge = document.getElementById("focusBadge");
  if (pair.matchMethod === "reid" || pair.reid5) {
    focusBadge.textContent = "Re-ID";
    focusBadge.style.background = "#10b981";
  } else {
    focusBadge.textContent = "HISTOGRAM";
    focusBadge.style.background = "#f59e0b";
  }

  // Color bars (only show for histogram mode)
  if (pair.hist5 && pair.hist6) {
    renderColorBar("colorBar5", pair.hist5);
    renderColorBar("colorBar6", pair.hist6);
  } else {
    document.getElementById("colorBar5").innerHTML = '<div style="color:#10b981;font-size:11px;">Re-ID: 512-dim vector</div>';
    document.getElementById("colorBar6").innerHTML = '<div style="color:#10b981;font-size:11px;">Re-ID: 512-dim vector</div>';
  }

  // Update crop images immediately
  updateCropImages(pair.t5, pair.t6);

  // Start periodic refresh of crop images
  if (cropRefreshTimer) clearInterval(cropRefreshTimer);
  cropRefreshTimer = setInterval(() => {
    if (selectedPair) {
      updateCropImages(selectedPair.t5, selectedPair.t6);
    }
  }, 200);  // 5 FPS for crop updates
}

function updateCropImages(t5, t6) {
  const img5 = document.getElementById("crop5");
  const img6 = document.getElementById("crop6");
  const ts = Date.now();  // Cache busting

  img5.src = CAM5 + "/crop/" + t5 + "?t=" + ts;
  img6.src = CAM6 + "/crop/" + t6 + "?t=" + ts;

  // Handle load errors gracefully
  img5.onerror = () => { img5.src = ""; };
  img6.onerror = () => { img6.src = ""; };
}

function closeFocus() {
  selectedPair = null;
  document.getElementById("focusPanel").classList.remove("active");
  document.querySelectorAll(".pair-item").forEach(el => el.classList.remove("selected"));

  // Stop crop refresh
  if (cropRefreshTimer) {
    clearInterval(cropRefreshTimer);
    cropRefreshTimer = null;
  }

  // Clear images
  document.getElementById("crop5").src = "";
  document.getElementById("crop6").src = "";
}

function renderColorBar(id, hist) {
  const el = document.getElementById(id);
  if (!hist) {
    el.innerHTML = '<div style="color:#666;font-size:11px;">No histogram</div>';
    return;
  }

  // HSV hue to RGB
  function hueToRgb(h) {
    const s = 1, v = 1;
    const c = v * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = v - c;
    let r, g, b;
    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    return `rgb(${Math.round((r+m)*255)},${Math.round((g+m)*255)},${Math.round((b+m)*255)})`;
  }

  el.innerHTML = hist.map((v, i) => {
    const hue = (i / 16) * 180;
    const height = Math.max(5, v * 100);
    return `<div class="color-bar-segment" style="background:${hueToRgb(hue * 2)};opacity:${0.3 + v * 0.7}"></div>`;
  }).join('');
}

function renderPairStability(m5, m6, mm){
  const kpi = document.getElementById("pairKpi");
  const top = document.getElementById("pairTop");
  const hint = document.getElementById("pairHint");

  if(!m5 || !m6){
    kpi.innerHTML = `
      <div class="stat-box"><div class="stat-value offline-msg">OFF</div><div class="stat-label">Pairs</div></div>
      <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">CAM5 Only</div></div>
      <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">CAM6 Only</div></div>
      <div class="stat-box"><div class="stat-value">-</div><div class="stat-label">Avg Sim</div></div>
    `;
    top.innerHTML = `<div class="pair-item"><span class="offline-msg">Cameras offline</span></div>`;
    hint.textContent = "Start both camera processes";
    return;
  }

  updatePairMem(mm);
  const sum = summarizePairs(mm);

  const pairsNow = mm.pairs.length;
  const unmatched5 = mm.unmatched5 || [];
  const unmatched6 = mm.unmatched6 || [];

  kpi.innerHTML = `
    <div class="stat-box"><div class="stat-value" style="color:#10b981">${pairsNow}</div><div class="stat-label">Matched</div></div>
    <div class="stat-box"><div class="stat-value" style="color:#3b82f6">${unmatched5.length}</div><div class="stat-label">CAM5 Only</div></div>
    <div class="stat-box"><div class="stat-value" style="color:#8b5cf6">${unmatched6.length}</div><div class="stat-label">CAM6 Only</div></div>
    <div class="stat-box"><div class="stat-value">${(sum.avgSim * 100).toFixed(0)}%</div><div class="stat-label">Avg Sim</div></div>
  `;

  let html = "";

  // 已匹配的 pairs
  const topK = sum.enriched.slice(0, 6);
  if(topK.length > 0){
    html += topK.map((p, idx) => {
      const isSelected = selectedPair && selectedPair.t5 === p.t5 && selectedPair.t6 === p.t6;
      const isLocked = lockedPairs.has(p.t5) && lockedPairs.get(p.t5) === p.t6;
      const lockIcon = isLocked ? '🔒' : '';
      return `<div class="pair-item ${isSelected ? 'selected' : ''}" onclick='selectPair(${JSON.stringify(p)})'>
        <span class="pair-link" style="color:#10b981">${lockIcon} cam5:#${p.t5} ⟷ cam6:#${p.t6}</span>
        <span class="pair-meta">similarity: ${(p.simAvg * 100).toFixed(0)}% ${isLocked ? '| LOCKED' : ''}</span>
      </div>`;
    }).join("");
  }

  // 未匹配的显示
  if(unmatched5.length > 0){
    html += `<div class="pair-item" style="opacity:0.7;cursor:default">
      <span style="color:#3b82f6">CAM5 only: #${unmatched5.join(', #')}</span>
      <span class="pair-meta">no match in CAM6</span>
    </div>`;
  }
  if(unmatched6.length > 0){
    html += `<div class="pair-item" style="opacity:0.7;cursor:default">
      <span style="color:#8b5cf6">CAM6 only: #${unmatched6.join(', #')}</span>
      <span class="pair-meta">no match in CAM5</span>
    </div>`;
  }

  if(html === ""){
    html = `<div class="pair-item"><span class="offline-msg">No people detected</span></div>`;
  }

  top.innerHTML = html;
  hint.textContent = pairsNow > 0 ? "Click a matched pair to see comparison" : "Waiting for matches...";
}

function selectPair(p) {
  showFocus(p);
  document.querySelectorAll(".pair-item").forEach(el => el.classList.remove("selected"));
  event.currentTarget.classList.add("selected");
}

// ========== Chart ==========
const ctx = document.getElementById("chart").getContext("2d");
const chart = new Chart(ctx, {
  type: "line",
  data: { labels: [], datasets: [
    { label: "cam5 count", data: [], borderColor: "#3b82f6", tension: 0.3 },
    { label: "cam6 count", data: [], borderColor: "#8b5cf6", tension: 0.3 },
    { label: "matched pairs", data: [], borderColor: "#10b981", tension: 0.3 },
  ]},
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12, font: { size: 10 } } }
    },
    scales: {
      x: { display: false },
      y: { grid: { color: 'rgba(51,65,85,0.5)' }, ticks: { color: '#64748b' } }
    }
  }
});

function turboColor(v){
  const x = Math.max(0, Math.min(1, v));
  const r = Math.floor(255 * Math.max(0, Math.min(1, (x-0.5)*2)));
  const g = Math.floor(255 * Math.max(0, Math.min(1, 1 - Math.abs(x-0.5)*2)));
  const b = Math.floor(255 * Math.max(0, Math.min(1, (0.5-x)*2)));
  return `rgb(${r},${g},${b})`;
}

function drawHeat(grid){
  const cvs = document.getElementById("heat");
  const g = cvs.getContext("2d");
  g.clearRect(0, 0, cvs.width, cvs.height);
  if(!grid || !grid.length) return;

  const gh = grid.length, gw = grid[0].length;
  const cellW = cvs.width / gw, cellH = cvs.height / gh;

  for(let y = 0; y < gh; y++){
    for(let x = 0; x < gw; x++){
      g.fillStyle = turboColor(grid[y][x] ?? 0);
      g.fillRect(x * cellW, y * cellH, cellW, cellH);
    }
  }
}

function fmtKpi(m){
  if(!m) return `<div class="kpi-item"><span class="kpi-value offline-msg">OFFLINE</span></div>`;
  return `
    <div class="kpi-item"><span class="kpi-label">Count:</span><span class="kpi-value">${m.confirmed_count || 0}</span></div>
    <div class="kpi-item"><span class="kpi-label">Latency:</span><span class="kpi-value">${m.latency_ms || 0}ms</span></div>
    <div class="kpi-item"><span class="kpi-label">Infer:</span><span class="kpi-value">${Math.round(m.infer_ms_wide || 0)}ms</span></div>
  `;
}

function updateStatus(m5, m6) {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");

  const c5 = m5 ? (m5.paused ? "paused" : "online") : "offline";
  const c6 = m6 ? (m6.paused ? "paused" : "online") : "offline";

  if (c5 === "offline" && c6 === "offline") {
    dot.className = "status-dot offline";
    text.textContent = "Both offline";
  } else if (c5 === "paused" || c6 === "paused") {
    dot.className = "status-dot paused";
    text.textContent = `cam5:${c5} | cam6:${c6}`;
  } else {
    dot.className = "status-dot online";
    text.textContent = `cam5:${c5} | cam6:${c6}`;
  }
}

let matchedPairsHistory = [];

// ========== 评估统计 ==========
let evalStats = {
  frames: 0,
  pairChanges: 0,
  lastPairKey: "",
  log: []  // 每帧的匹配记录
};

function updateEvalStats(mm) {
  evalStats.frames++;

  // 计算当前配对的 key
  const pairKey = mm.pairs.map(p => `${p.t5}-${p.t6}`).sort().join("|");

  // 检测配对变化
  const changed = evalStats.lastPairKey !== "" && evalStats.lastPairKey !== pairKey;
  if (changed) {
    evalStats.pairChanges++;
  }
  evalStats.lastPairKey = pairKey;

  // 记录日志
  const logEntry = {
    ts: Date.now(),
    pairs: mm.pairs.map(p => ({t5: p.t5, t6: p.t6, sim: p.colorSim})),
    unmatched5: mm.unmatched5,
    unmatched6: mm.unmatched6
  };
  evalStats.log.push(logEntry);

  // 只保留最近 1000 条
  if (evalStats.log.length > 1000) evalStats.log.shift();

  // 更新 UI
  const stability = evalStats.frames > 10 ?
    ((1 - evalStats.pairChanges / evalStats.frames) * 100).toFixed(1) : "--";

  document.getElementById("statFrames").textContent = evalStats.frames;
  document.getElementById("statChanges").textContent = evalStats.pairChanges;
  document.getElementById("statStability").textContent = stability + "%";
  document.getElementById("statLocked").textContent = lockedPairs.size;

  // 更新实时日志（只显示有变化或每10帧显示一次）
  if (changed || evalStats.frames % 10 === 0) {
    addLogLine(logEntry, changed);
  }
}

function addLogLine(entry, isChange) {
  const logDiv = document.getElementById("liveLog");
  const time = new Date(entry.ts).toLocaleTimeString();
  const pairs = entry.pairs.map(p => `${p.t5}↔${p.t6}(${(p.sim*100).toFixed(0)}%)`).join(", ") || "none";
  const color = isChange ? "color:#ef4444" : "color:#64748b";
  const prefix = isChange ? "⚠ CHANGE" : "  ";

  const line = document.createElement("div");
  line.style.cssText = color;
  line.textContent = `[${time}] ${prefix} Pairs: ${pairs}`;
  logDiv.appendChild(line);

  // 保持最多 50 行
  while (logDiv.children.length > 50) {
    logDiv.removeChild(logDiv.firstChild);
  }

  // 自动滚动到底部
  logDiv.scrollTop = logDiv.scrollHeight;
}

function clearLog() {
  document.getElementById("liveLog").innerHTML = "";
}

function exportLog() {
  const data = {
    exportTime: new Date().toISOString(),
    stats: {
      totalFrames: evalStats.frames,
      pairChanges: evalStats.pairChanges,
      stabilityScore: evalStats.frames > 0 ? (1 - evalStats.pairChanges / evalStats.frames) : 0
    },
    log: evalStats.log.slice(-500)  // 最近 500 条
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `matching_log_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function poll(){
  let m5 = null, m6 = null;
  try { m5 = await (await fetch(CAM5 + "/metrics")).json(); } catch(e){}
  try { m6 = await (await fetch(CAM6 + "/metrics")).json(); } catch(e){}

  latestM5 = m5;
  latestM6 = m6;

  const t = new Date().toLocaleTimeString();
  chart.data.labels.push(t);
  if (chart.data.labels.length > 60) chart.data.labels.shift();

  chart.data.datasets[0].data.push(m5 ? (m5.confirmed_count ?? 0) : null);
  chart.data.datasets[1].data.push(m6 ? (m6.confirmed_count ?? 0) : null);

  const mm = matchTracks(m5, m6);
  chart.data.datasets[2].data.push(mm.pairs.length);

  chart.data.datasets.forEach(ds => { if (ds.data.length > 60) ds.data.shift(); });
  chart.update();

  // Update matching method display
  document.getElementById("methodStatus").textContent = "Matching: " + currentMatchMethod;
  const badge = document.getElementById("matchMethodBadge");
  badge.textContent = currentMatchMethod;
  badge.className = currentMatchMethod.includes("Re-ID") ? "method-badge color" : "method-badge basic";

  document.getElementById("kpi5").innerHTML = fmtKpi(m5);
  document.getElementById("kpi6").innerHTML = fmtKpi(m6);

  renderPairStability(m5, m6, mm);
  drawHeat(m5 ? m5.grid_norm : null);
  updateStatus(m5, m6);
  updateEvalStats(mm);

  // Update focus panel if open
  if (selectedPair && mm.pairs.length > 0) {
    const updated = mm.pairs.find(p => p.t5 === selectedPair.t5 && p.t6 === selectedPair.t6);
    if (updated) {
      document.getElementById("simValue").textContent = (updated.colorSim * 100).toFixed(0) + "%";
      renderColorBar("colorBar5", updated.hist5);
      renderColorBar("colorBar6", updated.hist6);
    }
  }
}

setInterval(poll, 500);
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9000, threaded=True, use_reloader=False)
