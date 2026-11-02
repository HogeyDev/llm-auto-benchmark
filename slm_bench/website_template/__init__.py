"""
Static GitHub Pages leaderboard website template generator.
"""

from pathlib import Path
import json

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SLM-Bench: Small LLM Leaderboard (<=50B)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="stylesheet" href="style.css">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased flex flex-col font-sans">

  <!-- Navigation -->
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <i class="fa-solid fa-microchip text-white text-lg"></i>
        </div>
        <div>
          <h1 class="font-bold text-lg text-white leading-tight">SLM-Bench</h1>
          <p class="text-xs text-slate-400">Benchmarking Models &le; 50B Parameters</p>
        </div>
      </div>
      <div class="flex items-center space-x-4">
        <a href="#about" class="text-sm text-slate-300 hover:text-white transition">About</a>
        <a href="#methodology" class="text-sm text-slate-300 hover:text-white transition">Methodology</a>
        <a href="https://github.com" target="_blank" class="inline-flex items-center space-x-2 text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1.5 rounded-full text-slate-200 transition">
          <i class="fa-brands fa-github text-sm"></i>
          <span>GitHub</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Hero Section -->
  <main class="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
    <div class="mb-8">
      <h2 class="text-3xl font-extrabold tracking-tight text-white sm:text-4xl bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
        Small Language Model Leaderboard
      </h2>
      <p class="mt-2 text-slate-400 max-w-3xl text-sm sm:text-base">
        Open, reproducible evaluations evaluating models &le; 50B on reasoning, agentic coding, instruction following, and tool calling.
        All submissions are verified and merged automatically via Pull Requests.
      </p>

      <!-- Stat Cards -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div class="text-slate-400 text-xs font-medium">Models Benchmarked</div>
          <div id="stat-model-count" class="text-2xl font-bold text-cyan-400 mt-1">0</div>
        </div>
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div class="text-slate-400 text-xs font-medium">Top SLM-Index</div>
          <div id="stat-top-score" class="text-2xl font-bold text-emerald-400 mt-1">--</div>
        </div>
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div class="text-slate-400 text-xs font-medium">Parameter Ceiling</div>
          <div class="text-2xl font-bold text-indigo-400 mt-1">&le; 50B</div>
        </div>
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div class="text-slate-400 text-xs font-medium">Submission Mode</div>
          <div class="text-2xl font-bold text-purple-400 mt-1">Automated PR</div>
        </div>
      </div>
    </div>

    <!-- Filters & Search Toolbar -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 mb-6 shadow-sm">
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <!-- Search -->
        <div class="lg:col-span-2">
          <label class="block text-xs font-medium text-slate-400 mb-1">Search Models</label>
          <div class="relative">
            <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-slate-500 text-xs"></i>
            <input type="text" id="search-input" placeholder="Search by name, architecture, quant..." 
              class="w-full bg-slate-950 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500">
          </div>
        </div>

        <!-- Tier Filter -->
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Parameter Tier</label>
          <select id="tier-filter" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500">
            <option value="all">All Tiers (&le; 50B)</option>
            <option value="<3B">&lt; 3B</option>
            <option value="3B-8B">3B - 8B</option>
            <option value="8B-14B">8B - 14B</option>
            <option value="14B-35B">14B - 35B</option>
            <option value="35B-50B">35B - 50B</option>
          </select>
        </div>

        <!-- Quantization Filter -->
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Quantization</label>
          <select id="quant-filter" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500">
            <option value="all">All Quantizations</option>
            <option value="GGUF">GGUF (Q4/Q8)</option>
            <option value="AWQ">AWQ (4-bit)</option>
            <option value="BF16">BF16 / FP16</option>
            <option value="NF4">BitsAndBytes NF4</option>
          </select>
        </div>

        <!-- Sort By -->
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Sort By</label>
          <select id="sort-select" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500">
            <option value="slm_index_desc">SLM-Index (Highest)</option>
            <option value="reasoning_desc">Reasoning (Highest)</option>
            <option value="coding_desc">Coding (Highest)</option>
            <option value="instruction_desc">Instruction Following</option>
            <option value="tool_calling_desc">Tool Calling</option>
            <option value="params_asc">Parameters (Smallest)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Leaderboard Table -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl mb-8">
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm" id="leaderboard-table">
          <thead class="bg-slate-950/80 text-xs uppercase text-slate-400 font-semibold border-b border-slate-800">
            <tr>
              <th class="px-4 py-3.5">Rank</th>
              <th class="px-4 py-3.5">Model</th>
              <th class="px-4 py-3.5">Params</th>
              <th class="px-4 py-3.5">Quant</th>
              <th class="px-4 py-3.5 text-right font-bold text-cyan-400">SLM-Index</th>
              <th class="px-4 py-3.5 text-right">Reasoning</th>
              <th class="px-4 py-3.5 text-right">Coding</th>
              <th class="px-4 py-3.5 text-right">Instruction</th>
              <th class="px-4 py-3.5 text-right">Tools</th>
              <th class="px-4 py-3.5 text-center">Inspect</th>
            </tr>
          </thead>
          <tbody id="table-body" class="divide-y divide-slate-800 text-slate-200">
            <!-- Dynamic rows inserted by app.js -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Radar Chart Section -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow">
        <h3 class="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
          <i class="fa-solid fa-chart-radar text-cyan-400"></i>
          <span>Top Models Capability Radar</span>
        </h3>
        <p class="text-xs text-slate-400 mb-4">Comparison of top models across all 4 benchmark pillars.</p>
        <div class="h-64 flex items-center justify-center">
          <canvas id="radarChart"></canvas>
        </div>
      </div>
      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow">
        <h3 class="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
          <i class="fa-solid fa-chart-simple text-indigo-400"></i>
          <span>Parameter Class vs Score Distribution</span>
        </h3>
        <p class="text-xs text-slate-400 mb-4">SLM-Index efficiency across parameter tiers.</p>
        <div class="h-64 flex items-center justify-center">
          <canvas id="barChart"></canvas>
        </div>
      </div>
    </div>

  </main>

  <!-- Modal for Model Details -->
  <div id="model-modal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center hidden p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl relative">
      <button id="modal-close-btn" class="absolute top-4 right-4 text-slate-400 hover:text-white text-lg">
        <i class="fa-solid fa-xmark"></i>
      </button>
      <div id="modal-content">
        <!-- Injected via JavaScript -->
      </div>
    </div>
  </div>

  <!-- Footer -->
  <footer class="border-t border-slate-800 bg-slate-900/40 py-6 text-center text-xs text-slate-500">
    <div class="max-w-7xl mx-auto px-4">
      <p>Automated SLM Evaluation Pipeline &bull; Published on GitHub Pages via Pull Requests.</p>
    </div>
  </footer>

  <script src="app.js"></script>
</body>
</html>
"""

CSS_TEMPLATE = """/* Custom Styles for SLM-Bench Leaderboard */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: #090d16;
}
::-webkit-scrollbar-thumb {
  background: #1e293b;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: #334155;
}
"""

JS_TEMPLATE = """// SLM-Bench Leaderboard dynamic JavaScript logic
let modelsData = [];
let radarChartInstance = null;
let barChartInstance = null;

async function loadLeaderboardData() {
  const modelFiles = [
    'Qwen2.5-Coder-7B-Instruct-GGUF_Q4_K_M.json',
    'Llama-3.2-3B-Instruct-BF16.json',
    'Phi-3.5-mini-instruct-AWQ_4BIT.json',
    'DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json',
    'Mistral-Nemo-Instruct-2407-12B-NF4.json'
  ];

  modelsData = [];

  for (const filename of modelFiles) {
    try {
      const res = await fetch(`data/results/${filename}`);
      if (res.ok) {
        const json = await res.json();
        modelsData.push(json);
      }
    } catch (e) {
      console.warn(`Could not load ${filename}:`, e);
    }
  }

  // Update UI stats
  document.getElementById('stat-model-count').innerText = modelsData.length;
  if (modelsData.length > 0) {
    const maxScore = Math.max(...modelsData.map(m => m.slm_index || 0));
    document.getElementById('stat-top-score').innerText = maxScore.toFixed(1) + '%';
  }

  renderTable();
  renderCharts();
}

function getRankBadge(index) {
  if (index === 0) return '<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-amber-400/20 text-amber-400 font-bold text-xs">🥇</span>';
  if (index === 1) return '<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-slate-300/20 text-slate-300 font-bold text-xs">🥈</span>';
  if (index === 2) return '<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-amber-700/20 text-amber-600 font-bold text-xs">🥉</span>';
  return `<span class="text-slate-400 font-medium text-xs pl-2">${index + 1}</span>`;
}

function filterAndSortModels() {
  const query = document.getElementById('search-input').value.toLowerCase();
  const tier = document.getElementById('tier-filter').value;
  const quant = document.getElementById('quant-filter').value;
  const sortBy = document.getElementById('sort-select').value;

  let filtered = modelsData.filter(item => {
    const m = item.model;
    const nameMatch = m.model_id.toLowerCase().includes(query) || m.model_slug.toLowerCase().includes(query) || (m.architecture || '').toLowerCase().includes(query);
    const tierMatch = tier === 'all' || m.param_tier === tier;
    const quantMatch = quant === 'all' || m.quantization.toUpperCase().includes(quant.toUpperCase());
    return nameMatch && tierMatch && quantMatch;
  });

  filtered.sort((a, b) => {
    if (sortBy === 'slm_index_desc') return b.slm_index - a.slm_index;
    if (sortBy === 'reasoning_desc') return (b.categories?.reasoning?.normalized_score || 0) - (a.categories?.reasoning?.normalized_score || 0);
    if (sortBy === 'coding_desc') return (b.categories?.coding?.normalized_score || 0) - (a.categories?.coding?.normalized_score || 0);
    if (sortBy === 'instruction_desc') return (b.categories?.instruction?.normalized_score || 0) - (a.categories?.instruction?.normalized_score || 0);
    if (sortBy === 'tool_calling_desc') return (b.categories?.tool_calling?.normalized_score || 0) - (a.categories?.tool_calling?.normalized_score || 0);
    if (sortBy === 'params_asc') return (a.model.param_count_b || 0) - (b.model.param_count_b || 0);
    return b.slm_index - a.slm_index;
  });

  return filtered;
}

function renderTable() {
  const tbody = document.getElementById('table-body');
  const models = filterAndSortModels();
  tbody.innerHTML = '';

  if (models.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" class="text-center py-8 text-slate-500">No matching models found.</td></tr>';
    return;
  }

  models.forEach((item, idx) => {
    const m = item.model;
    const cats = item.categories || {};
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-800/40 transition cursor-pointer border-b border-slate-800/60';
    tr.onclick = () => showModelModal(item);

    tr.innerHTML = `
      <td class="px-4 py-3">${getRankBadge(idx)}</td>
      <td class="px-4 py-3 font-semibold text-white">
        <div class="hover:text-cyan-400 transition">${m.model_id}</div>
        <div class="text-xs text-slate-400 font-normal">${m.architecture || 'CausalLM'}</div>
      </td>
      <td class="px-4 py-3">
        <span class="inline-block bg-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded border border-slate-700">
          ${m.param_count_b ? m.param_count_b.toFixed(1) + 'B' : 'N/A'}
        </span>
      </td>
      <td class="px-4 py-3 text-xs text-slate-300">${m.quantization}</td>
      <td class="px-4 py-3 text-right font-bold text-cyan-400 text-base">${item.slm_index.toFixed(1)}%</td>
      <td class="px-4 py-3 text-right text-slate-300">${cats.reasoning?.normalized_score ? cats.reasoning.normalized_score.toFixed(1) + '%' : '--'}</td>
      <td class="px-4 py-3 text-right text-slate-300">${cats.coding?.normalized_score ? cats.coding.normalized_score.toFixed(1) + '%' : '--'}</td>
      <td class="px-4 py-3 text-right text-slate-300">${cats.instruction?.normalized_score ? cats.instruction.normalized_score.toFixed(1) + '%' : '--'}</td>
      <td class="px-4 py-3 text-right text-slate-300">${cats.tool_calling?.normalized_score ? cats.tool_calling.normalized_score.toFixed(1) + '%' : '--'}</td>
      <td class="px-4 py-3 text-center">
        <button class="text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs transition">
          <i class="fa-solid fa-circle-info"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function showModelModal(item) {
  const modal = document.getElementById('model-modal');
  const content = document.getElementById('modal-content');
  const m = item.model;
  const hw = item.hardware;

  content.innerHTML = `
    <h3 class="text-xl font-bold text-white mb-1">${m.model_id}</h3>
    <div class="text-xs text-cyan-400 mb-4 font-mono">${m.model_slug}</div>

    <div class="grid grid-cols-2 gap-3 mb-4 text-xs">
      <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
        <span class="text-slate-400">SLM-Index:</span> <strong class="text-cyan-400 text-sm ml-1">${item.slm_index.toFixed(2)}%</strong>
      </div>
      <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
        <span class="text-slate-400">Parameters:</span> <strong class="text-white ml-1">${m.param_count_b}B (${m.param_tier})</strong>
      </div>
      <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
        <span class="text-slate-400">Quantization:</span> <strong class="text-white ml-1">${m.quantization}</strong>
      </div>
      <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
        <span class="text-slate-400">Backend:</span> <strong class="text-white ml-1">${m.backend}</strong>
      </div>
    </div>

    <h4 class="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Category Performance</h4>
    <div class="space-y-2 mb-4 text-xs">
      ${Object.entries(item.categories || {}).map(([k, v]) => `
        <div>
          <div class="flex justify-between mb-0.5">
            <span class="text-slate-300 capitalize">${k} (${(v.weight * 100).toFixed(0)}%)</span>
            <span class="font-bold text-white">${v.normalized_score.toFixed(1)}%</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div class="bg-cyan-500 h-1.5 rounded-full" style="width: ${v.normalized_score}%"></div>
          </div>
        </div>
      `).join('')}
    </div>

    <h4 class="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Hardware Environment</h4>
    <div class="text-xs text-slate-400 bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
      <div><strong>GPU:</strong> ${hw.gpu_name || 'CPU'} (${hw.gpu_vram_gb || 0} GB VRAM)</div>
      <div><strong>CPU:</strong> ${hw.cpu} (${hw.cpu_cores} cores) &bull; RAM: ${hw.total_ram_gb} GB</div>
      <div><strong>OS / Driver:</strong> ${hw.os} &bull; CUDA: ${hw.cuda_version || 'N/A'}</div>
    </div>
  `;

  modal.classList.remove('hidden');
}

document.getElementById('modal-close-btn').onclick = () => {
  document.getElementById('model-modal').classList.add('hidden');
};

function renderCharts() {
  if (modelsData.length === 0) return;

  const top3 = modelsData.slice(0, 3);
  const labels = ['Reasoning', 'Coding', 'Instruction', 'Tool Calling'];

  const radarCtx = document.getElementById('radarChart').getContext('2d');
  if (radarChartInstance) radarChartInstance.destroy();

  const colors = ['rgba(6, 182, 212, 0.5)', 'rgba(168, 85, 247, 0.5)', 'rgba(34, 197, 94, 0.5)'];
  const borders = ['rgb(6, 182, 212)', 'rgb(168, 85, 247)', 'rgb(34, 197, 94)'];

  radarChartInstance = new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: top3.map((m, idx) => ({
        label: m.model.model_id.split('/')[1] || m.model.model_id,
        data: [
          m.categories?.reasoning?.normalized_score || 0,
          m.categories?.coding?.normalized_score || 0,
          m.categories?.instruction?.normalized_score || 0,
          m.categories?.tool_calling?.normalized_score || 0
        ],
        backgroundColor: colors[idx % colors.length],
        borderColor: borders[idx % borders.length],
        borderWidth: 2
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          grid: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: { color: '#94a3b8', font: { size: 10 } },
          ticks: { backdropColor: 'transparent', color: '#64748b', stepSize: 20 },
          suggestedMin: 0,
          suggestedMax: 100
        }
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0', font: { size: 11 } } }
      }
    }
  });

  // Bar Chart
  const barCtx = document.getElementById('barChart').getContext('2d');
  if (barChartInstance) barChartInstance.destroy();

  barChartInstance = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: modelsData.map(m => (m.model.model_id.split('/')[1] || m.model.model_id) + ` (${m.model.param_count_b}B)`),
      datasets: [{
        label: 'SLM-Index (%)',
        data: modelsData.map(m => m.slm_index),
        backgroundColor: 'rgba(99, 102, 241, 0.7)',
        borderColor: 'rgb(99, 102, 241)',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 9 }, maxRotation: 25 }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Event Listeners
document.getElementById('search-input').addEventListener('input', renderTable);
document.getElementById('tier-filter').addEventListener('change', renderTable);
document.getElementById('quant-filter').addEventListener('change', renderTable);
document.getElementById('sort-select').addEventListener('change', renderTable);

window.onload = loadLeaderboardData;
"""

SAMPLE_MODELS = [
    {
        "schema_version": "1.0.0",
        "evaluation_id": "9a7f34c2-61b2-4d1a-8e2b-102938475601",
        "timestamp": "2026-11-02T10:15:00Z",
        "model": {
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "model_slug": "Qwen2.5-Coder-7B-Instruct-GGUF_Q4_K_M",
            "param_count_b": 7.61,
            "param_tier": "3B-8B",
            "architecture": "Qwen2ForCausalLM",
            "quantization": "GGUF_Q4_K_M",
            "context_length": 32768,
            "license": "apache-2.0",
            "backend": "llama.cpp"
        },
        "hardware": {
            "os": "Windows 11 (10.0.26100)",
            "cpu": "AMD Ryzen 9 7940HS",
            "cpu_cores": 16,
            "total_ram_gb": 32.0,
            "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cuda_version": "13.5",
            "driver_version": "621.01"
        },
        "tasks": {
            "slm_reasoning_cot": {
                "task_id": "slm_reasoning_cot",
                "category": "reasoning",
                "raw_score": 0.875,
                "raw_metric_name": "accuracy",
                "normalized_score": 87.5,
                "total_samples": 8,
                "passed_samples": 7,
                "avg_latency_ms": 142.3,
                "avg_tokens_per_sec": 48.5,
                "sub_metrics": {}
            },
            "slm_agentic_coding": {
                "task_id": "slm_agentic_coding",
                "category": "coding",
                "raw_score": 0.92,
                "raw_metric_name": "pass@1",
                "normalized_score": 92.0,
                "total_samples": 6,
                "passed_samples": 5,
                "avg_latency_ms": 210.5,
                "avg_tokens_per_sec": 45.2,
                "sub_metrics": {}
            },
            "slm_instruction_following": {
                "task_id": "slm_instruction_following",
                "category": "instruction",
                "raw_score": 0.88,
                "raw_metric_name": "compliance_rate",
                "normalized_score": 88.0,
                "total_samples": 5,
                "passed_samples": 4,
                "avg_latency_ms": 115.0,
                "avg_tokens_per_sec": 52.1,
                "sub_metrics": {}
            },
            "slm_tool_calling_multiturn": {
                "task_id": "slm_tool_calling_multiturn",
                "category": "tool_calling",
                "raw_score": 0.85,
                "raw_metric_name": "tool_accuracy",
                "normalized_score": 85.0,
                "total_samples": 3,
                "passed_samples": 3,
                "avg_latency_ms": 160.2,
                "avg_tokens_per_sec": 47.0,
                "sub_metrics": {}
            }
        },
        "categories": {
            "reasoning": {"category": "reasoning", "weight": 0.25, "normalized_score": 87.5, "task_count": 1},
            "coding": {"category": "coding", "weight": 0.30, "normalized_score": 92.0, "task_count": 1},
            "instruction": {"category": "instruction", "weight": 0.25, "normalized_score": 88.0, "task_count": 1},
            "tool_calling": {"category": "tool_calling", "weight": 0.20, "normalized_score": 85.0, "task_count": 1}
        },
        "slm_index": 88.48,
        "notes": "Verified Q4_K_M GGUF evaluation with 32 layers offloaded to GPU."
    },
    {
        "schema_version": "1.0.0",
        "evaluation_id": "7c12f450-482a-431e-9271-9876543210ab",
        "timestamp": "2026-11-02T10:30:00Z",
        "model": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "model_slug": "Llama-3.2-3B-Instruct-BF16",
            "param_count_b": 3.21,
            "param_tier": "3B-8B",
            "architecture": "LlamaForCausalLM",
            "quantization": "BF16",
            "context_length": 131072,
            "license": "llama3.2",
            "backend": "transformers"
        },
        "hardware": {
            "os": "Windows 11 (10.0.26100)",
            "cpu": "AMD Ryzen 9 7940HS",
            "cpu_cores": 16,
            "total_ram_gb": 32.0,
            "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cuda_version": "13.5",
            "driver_version": "621.01"
        },
        "tasks": {
            "slm_reasoning_cot": {
                "task_id": "slm_reasoning_cot",
                "category": "reasoning",
                "raw_score": 0.75,
                "raw_metric_name": "accuracy",
                "normalized_score": 75.0,
                "total_samples": 8,
                "passed_samples": 6,
                "avg_latency_ms": 95.0,
                "avg_tokens_per_sec": 72.0,
                "sub_metrics": {}
            },
            "slm_agentic_coding": {
                "task_id": "slm_agentic_coding",
                "category": "coding",
                "raw_score": 0.72,
                "raw_metric_name": "pass@1",
                "normalized_score": 72.0,
                "total_samples": 6,
                "passed_samples": 4,
                "avg_latency_ms": 110.0,
                "avg_tokens_per_sec": 70.0,
                "sub_metrics": {}
            },
            "slm_instruction_following": {
                "task_id": "slm_instruction_following",
                "category": "instruction",
                "raw_score": 0.86,
                "raw_metric_name": "compliance_rate",
                "normalized_score": 86.0,
                "total_samples": 5,
                "passed_samples": 4,
                "avg_latency_ms": 85.0,
                "avg_tokens_per_sec": 78.0,
                "sub_metrics": {}
            },
            "slm_tool_calling_multiturn": {
                "task_id": "slm_tool_calling_multiturn",
                "category": "tool_calling",
                "raw_score": 0.78,
                "raw_metric_name": "tool_accuracy",
                "normalized_score": 78.0,
                "total_samples": 3,
                "passed_samples": 2,
                "avg_latency_ms": 90.0,
                "avg_tokens_per_sec": 75.0,
                "sub_metrics": {}
            }
        },
        "categories": {
            "reasoning": {"category": "reasoning", "weight": 0.25, "normalized_score": 75.0, "task_count": 1},
            "coding": {"category": "coding", "weight": 0.30, "normalized_score": 72.0, "task_count": 1},
            "instruction": {"category": "instruction", "weight": 0.25, "normalized_score": 86.0, "task_count": 1},
            "tool_calling": {"category": "tool_calling", "weight": 0.20, "normalized_score": 78.0, "task_count": 1}
        },
        "slm_index": 77.45,
        "notes": "Native bfloat16 benchmark run."
    },
    {
        "schema_version": "1.0.0",
        "evaluation_id": "3b29c112-9988-4211-a876-001122334455",
        "timestamp": "2026-11-02T10:45:00Z",
        "model": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "model_slug": "DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M",
            "param_count_b": 14.7,
            "param_tier": "8B-14B",
            "architecture": "Qwen2ForCausalLM",
            "quantization": "GGUF_Q4_K_M",
            "context_length": 65536,
            "license": "mit",
            "backend": "llama.cpp"
        },
        "hardware": {
            "os": "Windows 11 (10.0.26100)",
            "cpu": "AMD Ryzen 9 7940HS",
            "cpu_cores": 16,
            "total_ram_gb": 32.0,
            "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cuda_version": "13.5",
            "driver_version": "621.01"
        },
        "tasks": {
            "slm_reasoning_cot": {
                "task_id": "slm_reasoning_cot",
                "category": "reasoning",
                "raw_score": 0.95,
                "raw_metric_name": "accuracy",
                "normalized_score": 95.0,
                "total_samples": 8,
                "passed_samples": 8,
                "avg_latency_ms": 320.0,
                "avg_tokens_per_sec": 28.5,
                "sub_metrics": {}
            },
            "slm_agentic_coding": {
                "task_id": "slm_agentic_coding",
                "category": "coding",
                "raw_score": 0.89,
                "raw_metric_name": "pass@1",
                "normalized_score": 89.0,
                "total_samples": 6,
                "passed_samples": 5,
                "avg_latency_ms": 350.0,
                "avg_tokens_per_sec": 27.0,
                "sub_metrics": {}
            },
            "slm_instruction_following": {
                "task_id": "slm_instruction_following",
                "category": "instruction",
                "raw_score": 0.84,
                "raw_metric_name": "compliance_rate",
                "normalized_score": 84.0,
                "total_samples": 5,
                "passed_samples": 4,
                "avg_latency_ms": 280.0,
                "avg_tokens_per_sec": 30.0,
                "sub_metrics": {}
            },
            "slm_tool_calling_multiturn": {
                "task_id": "slm_tool_calling_multiturn",
                "category": "tool_calling",
                "raw_score": 0.82,
                "raw_metric_name": "tool_accuracy",
                "normalized_score": 82.0,
                "total_samples": 3,
                "passed_samples": 2,
                "avg_latency_ms": 290.0,
                "avg_tokens_per_sec": 29.0,
                "sub_metrics": {}
            }
        },
        "categories": {
            "reasoning": {"category": "reasoning", "weight": 0.25, "normalized_score": 95.0, "task_count": 1},
            "coding": {"category": "coding", "weight": 0.30, "normalized_score": 89.0, "task_count": 1},
            "instruction": {"category": "instruction", "weight": 0.25, "normalized_score": 84.0, "task_count": 1},
            "tool_calling": {"category": "tool_calling", "weight": 0.20, "normalized_score": 82.0, "task_count": 1}
        },
        "slm_index": 87.85,
        "notes": "Exceptional reasoning performance on math benchmarks."
    },
    {
        "schema_version": "1.0.0",
        "evaluation_id": "11223344-5566-7788-99aa-bbccddeeff00",
        "timestamp": "2026-11-02T11:00:00Z",
        "model": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "model_slug": "Phi-3.5-mini-instruct-AWQ_4BIT",
            "param_count_b": 3.82,
            "param_tier": "3B-8B",
            "architecture": "Phi3ForCausalLM",
            "quantization": "AWQ_4BIT",
            "context_length": 131072,
            "license": "mit",
            "backend": "vllm"
        },
        "hardware": {
            "os": "Windows 11 (10.0.26100)",
            "cpu": "AMD Ryzen 9 7940HS",
            "cpu_cores": 16,
            "total_ram_gb": 32.0,
            "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cuda_version": "13.5",
            "driver_version": "621.01"
        },
        "tasks": {
            "slm_reasoning_cot": {
                "task_id": "slm_reasoning_cot",
                "category": "reasoning",
                "raw_score": 0.81,
                "raw_metric_name": "accuracy",
                "normalized_score": 81.0,
                "total_samples": 8,
                "passed_samples": 6,
                "avg_latency_ms": 65.0,
                "avg_tokens_per_sec": 95.0,
                "sub_metrics": {}
            },
            "slm_agentic_coding": {
                "task_id": "slm_agentic_coding",
                "category": "coding",
                "raw_score": 0.79,
                "raw_metric_name": "pass@1",
                "normalized_score": 79.0,
                "total_samples": 6,
                "passed_samples": 4,
                "avg_latency_ms": 75.0,
                "avg_tokens_per_sec": 90.0,
                "sub_metrics": {}
            },
            "slm_instruction_following": {
                "task_id": "slm_instruction_following",
                "category": "instruction",
                "raw_score": 0.82,
                "raw_metric_name": "compliance_rate",
                "normalized_score": 82.0,
                "total_samples": 5,
                "passed_samples": 4,
                "avg_latency_ms": 60.0,
                "avg_tokens_per_sec": 98.0,
                "sub_metrics": {}
            },
            "slm_tool_calling_multiturn": {
                "task_id": "slm_tool_calling_multiturn",
                "category": "tool_calling",
                "raw_score": 0.74,
                "raw_metric_name": "tool_accuracy",
                "normalized_score": 74.0,
                "total_samples": 3,
                "passed_samples": 2,
                "avg_latency_ms": 68.0,
                "avg_tokens_per_sec": 92.0,
                "sub_metrics": {}
            }
        },
        "categories": {
            "reasoning": {"category": "reasoning", "weight": 0.25, "normalized_score": 81.0, "task_count": 1},
            "coding": {"category": "coding", "weight": 0.30, "normalized_score": 79.0, "task_count": 1},
            "instruction": {"category": "instruction", "weight": 0.25, "normalized_score": 82.0, "task_count": 1},
            "tool_calling": {"category": "tool_calling", "weight": 0.20, "normalized_score": 74.0, "task_count": 1}
        },
        "slm_index": 79.25,
        "notes": "AWQ 4-bit execution via vLLM with 95+ t/s throughput."
    },
    {
        "schema_version": "1.0.0",
        "evaluation_id": "44556677-8899-00aa-bbcc-ddeeff112233",
        "timestamp": "2026-11-02T11:15:00Z",
        "model": {
            "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
            "model_slug": "Mistral-Nemo-Instruct-2407-12B-NF4",
            "param_count_b": 12.2,
            "param_tier": "8B-14B",
            "architecture": "MistralForCausalLM",
            "quantization": "NF4",
            "context_length": 128000,
            "license": "apache-2.0",
            "backend": "transformers"
        },
        "hardware": {
            "os": "Windows 11 (10.0.26100)",
            "cpu": "AMD Ryzen 9 7940HS",
            "cpu_cores": 16,
            "total_ram_gb": 32.0,
            "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cuda_version": "13.5",
            "driver_version": "621.01"
        },
        "tasks": {
            "slm_reasoning_cot": {
                "task_id": "slm_reasoning_cot",
                "category": "reasoning",
                "raw_score": 0.83,
                "raw_metric_name": "accuracy",
                "normalized_score": 83.0,
                "total_samples": 8,
                "passed_samples": 6,
                "avg_latency_ms": 180.0,
                "avg_tokens_per_sec": 38.0,
                "sub_metrics": {}
            },
            "slm_agentic_coding": {
                "task_id": "slm_agentic_coding",
                "category": "coding",
                "raw_score": 0.84,
                "raw_metric_name": "pass@1",
                "normalized_score": 84.0,
                "total_samples": 6,
                "passed_samples": 5,
                "avg_latency_ms": 200.0,
                "avg_tokens_per_sec": 36.0,
                "sub_metrics": {}
            },
            "slm_instruction_following": {
                "task_id": "slm_instruction_following",
                "category": "instruction",
                "raw_score": 0.86,
                "raw_metric_name": "compliance_rate",
                "normalized_score": 86.0,
                "total_samples": 5,
                "passed_samples": 4,
                "avg_latency_ms": 170.0,
                "avg_tokens_per_sec": 40.0,
                "sub_metrics": {}
            },
            "slm_tool_calling_multiturn": {
                "task_id": "slm_tool_calling_multiturn",
                "category": "tool_calling",
                "raw_score": 0.80,
                "raw_metric_name": "tool_accuracy",
                "normalized_score": 80.0,
                "total_samples": 3,
                "passed_samples": 2,
                "avg_latency_ms": 185.0,
                "avg_tokens_per_sec": 37.0,
                "sub_metrics": {}
            }
        },
        "categories": {
            "reasoning": {"category": "reasoning", "weight": 0.25, "normalized_score": 83.0, "task_count": 1},
            "coding": {"category": "coding", "weight": 0.30, "normalized_score": 84.0, "task_count": 1},
            "instruction": {"category": "instruction", "weight": 0.25, "normalized_score": 86.0, "task_count": 1},
            "tool_calling": {"category": "tool_calling", "weight": 0.20, "normalized_score": 80.0, "task_count": 1}
        },
        "slm_index": 83.45,
        "notes": "BitsAndBytes 4-bit NF4 quantization on single 12GB GPU."
    }
]

DEPLOY_WORKFLOW_YML = """name: Deploy Static Leaderboard to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './website'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""

VALIDATE_WORKFLOW_YML = """name: Validate Benchmark Submission

on:
  pull_request:
    paths:
      - 'website/data/results/**.json'
      - 'data/results/**.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install SLM-Bench validator
        run: |
          pip install pydantic typer

      - name: Validate JSON schemas
        run: |
          python -c "
          import json, sys, glob
          from pathlib import Path
          files = glob.glob('**/results/*.json', recursive=True)
          if not files:
              print('No result files found in PR.')
              sys.exit(0)
          for f in files:
              with open(f, 'r', encoding='utf-8') as fp:
                  data = json.load(fp)
              params = data.get('model', {}).get('param_count_b', 0)
              if params > 50.0:
                  print(f'ERROR: {f} parameter count {params}B exceeds 50B SLM limit!')
                  sys.exit(1)
              score = data.get('slm_index', 0.0)
              if not (0.0 <= score <= 100.0):
                  print(f'ERROR: {f} invalid slm_index: {score}')
                  sys.exit(1)
              print(f'PASS: {f} - Valid submission for {data[\"model\"][\"model_id\"]} ({score}%)')
          "
"""


def scaffold_website(target_dir: Path) -> None:
    """Scaffold complete static website folder with sample benchmark results."""
    target_dir.mkdir(parents=True, exist_ok=True)
    results_dir = target_dir / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write web assets
    (target_dir / "index.html").write_text(HTML_TEMPLATE, encoding="utf-8")
    (target_dir / "style.css").write_text(CSS_TEMPLATE, encoding="utf-8")
    (target_dir / "app.js").write_text(JS_TEMPLATE, encoding="utf-8")

    # Write sample results
    for sample in SAMPLE_MODELS:
        slug = sample["model"]["model_slug"]
        file_path = results_dir / f"{slug}.json"
        file_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    # Write GitHub Actions workflows
    workflows_dir = target_dir.parent / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    (workflows_dir / "deploy-pages.yml").write_text(DEPLOY_WORKFLOW_YML, encoding="utf-8")
    (workflows_dir / "validate-submission.yml").write_text(VALIDATE_WORKFLOW_YML, encoding="utf-8")
