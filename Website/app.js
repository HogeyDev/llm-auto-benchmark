/**
 * AI Model Benchmark Leaderboard - Static Client Application
 * Pure client-side JavaScript for GitHub Pages
 */

// Application State
let rawLeaderboardData = null;
let allModels = [];
let filteredModels = [];
let selectedModelIds = new Set();
let currentSort = { column: 'overall', direction: 'desc' };

let radarChartInstance = null;
let barChartInstance = null;

// Embedded Fallback Data (empty until local evaluations are recorded)
const EMBEDDED_FALLBACK_DATA = {
  metadata: {
    total_models: 0,
    last_updated: "",
    highest_overall: 0.0,
    best_by_benchmark: {}
  },
  models: []
};

// Markdown PR template content
const PR_TEMPLATE_TEXT = `### What this adds
- **Model:** org/model
- **Precision:** GGUF Q4_K_M
- **Hardware:** NVIDIA RTX 3060 12GB / Apple M2, 16GB
- **Result file:** \`Website/data/results/<model-slug>.json\`

### Checklist
- [ ] Open weights, and the license permits local use.
- [ ] Ran on consumer hardware: a consumer GPU, Apple Silicon, or system RAM.
- [ ] GPU, VRAM, CPU, RAM, and OS are recorded in the JSON.
- [ ] Sample counts are non-zero and match the run.
- [ ] Only the result file changed in this pull request.

### Hardware
| Resource | Value |
|:--|:--|
| GPU | NVIDIA RTX 3060 12GB |
| CPU | AMD Ryzen 5 5600X, 6 cores |
| System RAM | 32 GB |
| OS / driver | Ubuntu 24.04 / CUDA 12.4 / 550.54 |
| Backend | llama.cpp, 33 layers offloaded |

### Log tail
\`\`\`text
[2026-11-02 10:00:15] Model loaded
[2026-11-02 10:11:02] Benchmark suite finished. SLM-Index: 82.10
\`\`\`
`;

// Initialize Application
async function initApp() {
  setupEventListeners();
  await loadLeaderboardData();
}

// Data Fetching: tries static leaderboard.json, then falls back to results/index.json, then embedded
async function loadLeaderboardData() {
  try {
    const res = await fetch('data/leaderboard.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    rawLeaderboardData = data;
    allModels = data.models || [];
    console.log('[Leaderboard] Successfully fetched data/leaderboard.json');
  } catch (err) {
    console.warn('[Leaderboard] Could not load data/leaderboard.json via HTTP (likely local file:// origin or static build):', err.message);
    try {
      // Secondary attempt: load manifest
      const manifestRes = await fetch('data/results/index.json', { cache: 'no-cache' });
      if (manifestRes.ok) {
        const manifest = await manifestRes.json();
        const loadedModels = [];
        for (const filename of manifest.files) {
          try {
            const fRes = await fetch(`data/results/${filename}`);
            if (fRes.ok) {
              const fData = await fRes.json();
              loadedModels.push(normalizeEntry(fData, filename));
            }
          } catch (e) {}
        }
        if (loadedModels.length > 0) {
          loadedModels.sort((a, b) => b.scores.overall - a.scores.overall);
          loadedModels.forEach((m, i) => m.rank = i + 1);
          allModels = loadedModels;
          rawLeaderboardData = { metadata: { total_models: loadedModels.length }, models: loadedModels };
        }
      }
    } catch (e2) {}

    // Fallback to embedded data if still empty
    if (!allModels || allModels.length === 0) {
      console.log('[Leaderboard] Using embedded benchmark dataset.');
      rawLeaderboardData = EMBEDDED_FALLBACK_DATA;
      allModels = EMBEDDED_FALLBACK_DATA.models;
    }
  }

  // Populate dynamic UI elements
  populateLicenseFilter();
  updateTopStats();
  applyFiltersAndSort();
  renderCharts();
  loadPRTemplate();
}

function normalizeEntry(data, filename) {
  const model = data.model || {};
  const summary = data.summary_scores || {};
  const tasks = data.tasks || {};
  const hw = data.hardware || {};

  const gsm8k = summary.gsm8k || tasks.gsm8k?.score || 0;
  const humaneval = summary.humaneval || tasks.humaneval?.score || 0;
  const arc = summary.arc || tasks.arc?.score || 0;
  const ifeval = summary.ifeval || tasks.ifeval?.score || 0;
  const tool_calling = summary.tool_calling || tasks.tool_calling?.score || 0;
  const overall = summary.overall_score || data.slm_index || ((gsm8k * 0.25) + (humaneval * 0.25) + (arc * 0.2) + (ifeval * 0.15) + (tool_calling * 0.15));

  return {
    filename: filename,
    file_path: `data/results/${filename}`,
    evaluation_id: data.evaluation_id || "eval-" + Math.random().toString(36).substring(7),
    timestamp: data.timestamp || new Date().toISOString(),
    submitter: data.submitter || { name: "Community Contributor", github_handle: "open-source" },
    model: {
      model_id: model.model_id || "Unknown",
      model_slug: model.model_slug || (model.model_id || "").replace('/', '-'),
      hf_url: model.hf_url || `https://huggingface.co/${model.model_id}`,
      param_count_b: parseFloat(model.param_count_b || 0),
      param_tier: model.param_tier || "7B–14B",
      architecture: model.architecture || "CausalLM",
      quantization: model.quantization || "BF16",
      context_length: model.context_length || 32768,
      license: model.license || "Open",
      backend: model.backend || "transformers"
    },
    scores: {
      overall: parseFloat(overall.toFixed(2)),
      gsm8k: parseFloat(gsm8k.toFixed(2)),
      humaneval: parseFloat(humaneval.toFixed(2)),
      arc: parseFloat(arc.toFixed(2)),
      ifeval: parseFloat(ifeval.toFixed(2)),
      tool_calling: parseFloat(tool_calling.toFixed(2))
    },
    hardware: hw,
    tasks: tasks,
    raw_logs_preview: data.raw_logs_preview || data.notes || "Evaluation completed."
  };
}

// Populate Licenses Dropdown dynamically
function populateLicenseFilter() {
  const licenseSelect = document.getElementById('license-filter');
  const licenses = new Set();
  allModels.forEach(m => {
    if (m.model?.license) licenses.add(m.model.license);
  });

  licenseSelect.innerHTML = '<option value="all">All Licenses</option>';
  Array.from(licenses).sort().forEach(lic => {
    const opt = document.createElement('option');
    opt.value = lic;
    opt.textContent = lic;
    licenseSelect.appendChild(opt);
  });
}

// Update Top Metric Stat Cards
function updateTopStats() {
  const modelCountEl = document.getElementById('stat-model-count');
  const topScoreEl = document.getElementById('stat-top-score');
  const topModelEl = document.getElementById('stat-top-model');
  const topCodingEl = document.getElementById('stat-top-coding');
  const topCodingModelEl = document.getElementById('stat-top-coding-model');
  const topMathEl = document.getElementById('stat-top-math');
  const topMathModelEl = document.getElementById('stat-top-math-model');
  const updatedDateEl = document.getElementById('last-updated-text');

  if (!allModels || allModels.length === 0) {
    if (modelCountEl) modelCountEl.textContent = '0';
    if (topScoreEl) topScoreEl.textContent = '—';
    if (topModelEl) topModelEl.textContent = '—';
    if (topCodingEl) topCodingEl.textContent = '—';
    if (topCodingModelEl) topCodingModelEl.textContent = '—';
    if (topMathEl) topMathEl.textContent = '—';
    if (topMathModelEl) topMathModelEl.textContent = '—';
    if (updatedDateEl) updatedDateEl.textContent = '—';
    return;
  }

  modelCountEl.textContent = allModels.length;

  const topOverall = allModels.reduce((prev, cur) => (cur.scores.overall > prev.scores.overall) ? cur : prev, allModels[0]);
  topScoreEl.textContent = topOverall.scores.overall.toFixed(1) + '%';
  topModelEl.textContent = topOverall.model.model_id;

  const topCoding = allModels.reduce((prev, cur) => (cur.scores.humaneval > prev.scores.humaneval) ? cur : prev, allModels[0]);
  topCodingEl.textContent = topCoding.scores.humaneval.toFixed(1) + '%';
  topCodingModelEl.textContent = topCoding.model.model_id;

  const topMath = allModels.reduce((prev, cur) => (cur.scores.gsm8k > prev.scores.gsm8k) ? cur : prev, allModels[0]);
  topMathEl.textContent = topMath.scores.gsm8k.toFixed(1) + '%';
  topMathModelEl.textContent = topMath.model.model_id;

  const updatedDate = rawLeaderboardData?.metadata?.last_updated ? new Date(rawLeaderboardData.metadata.last_updated).toLocaleDateString() : 'Recently';
  if (updatedDateEl) updatedDateEl.textContent = updatedDate;
}

// Filter and Sort Engine
function applyFiltersAndSort() {
  const searchTerm = (document.getElementById('search-input').value || '').trim().toLowerCase();
  const tierFilter = document.getElementById('tier-filter').value;
  const licenseFilter = document.getElementById('license-filter').value;
  const quantFilter = document.getElementById('quant-filter').value;

  filteredModels = allModels.filter(item => {
    const m = item.model || {};
    const sub = item.submitter || {};

    // Search filter
    if (searchTerm) {
      const match = (
        (m.model_id || '').toLowerCase().includes(searchTerm) ||
        (m.model_slug || '').toLowerCase().includes(searchTerm) ||
        (m.architecture || '').toLowerCase().includes(searchTerm) ||
        (m.license || '').toLowerCase().includes(searchTerm) ||
        (m.backend || '').toLowerCase().includes(searchTerm) ||
        (m.quantization || '').toLowerCase().includes(searchTerm) ||
        (sub.name || '').toLowerCase().includes(searchTerm) ||
        (sub.github_handle || '').toLowerCase().includes(searchTerm)
      );
      if (!match) return false;
    }

    // Parameter Tier filter
    if (tierFilter !== 'all') {
      const p = m.param_count_b || 0;
      if (tierFilter === '< 3B' && p >= 3.0) return false;
      if (tierFilter === '3B–7B' && (p < 3.0 || p > 7.0)) return false;
      if (tierFilter === '7B–14B' && (p <= 7.0 || p > 14.0)) return false;
      if (tierFilter === '14B–32B' && (p <= 14.0 || p > 32.0)) return false;
      if ((tierFilter === '32B+' || tierFilter === '32B–50B' || tierFilter === '> 32B') && p <= 32.0) return false;
    }

    // License filter
    if (licenseFilter !== 'all') {
      if (m.license !== licenseFilter) return false;
    }

    // Quantization filter
    if (quantFilter !== 'all') {
      const q = (m.quantization || '').toUpperCase();
      if (quantFilter === 'BF16' && !q.includes('BF16') && !q.includes('FP16')) return false;
      if (quantFilter === 'GGUF' && !q.includes('GGUF')) return false;
      if (quantFilter === 'AWQ' && !q.includes('AWQ')) return false;
      if (quantFilter === 'NF4' && !q.includes('NF4')) return false;
    }

    return true;
  });

  // Sort
  sortModels(currentSort.column, currentSort.direction);
  renderTable();
  updateCompareDock();
}

function sortModels(column, direction) {
  filteredModels.sort((a, b) => {
    let valA, valB;

    switch (column) {
      case 'rank':
        valA = a.rank || 0;
        valB = b.rank || 0;
        break;
      case 'name':
        valA = a.model.model_id.toLowerCase();
        valB = b.model.model_id.toLowerCase();
        break;
      case 'params':
        valA = a.model.param_count_b || 0;
        valB = b.model.param_count_b || 0;
        break;
      case 'overall':
        valA = a.scores.overall;
        valB = b.scores.overall;
        break;
      case 'gsm8k':
        valA = a.scores.gsm8k;
        valB = b.scores.gsm8k;
        break;
      case 'humaneval':
        valA = a.scores.humaneval;
        valB = b.scores.humaneval;
        break;
      case 'arc':
        valA = a.scores.arc;
        valB = b.scores.arc;
        break;
      case 'ifeval':
        valA = a.scores.ifeval;
        valB = b.scores.ifeval;
        break;
      case 'tool_calling':
        valA = a.scores.tool_calling;
        valB = b.scores.tool_calling;
        break;
      default:
        valA = a.scores.overall;
        valB = b.scores.overall;
    }

    if (valA < valB) return direction === 'asc' ? -1 : 1;
    if (valA > valB) return direction === 'asc' ? 1 : -1;
    return 0;
  });
}

// Render Leaderboard Table
function renderTable() {
  const tbody = document.getElementById('table-body');
  const countText = document.getElementById('results-count-text');
  tbody.innerHTML = '';

  if (allModels.length === 0) {
    countText.textContent = `0 models`;
    tbody.innerHTML = `
      <tr>
        <td colspan="11">
          <div class="empty-state">
            <strong>No benchmarks recorded yet</strong>
            <span>Run <code>slm-bench evaluate</code> to generate benchmark results.</span>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  countText.textContent = `Showing ${filteredModels.length} of ${allModels.length} models`;

  if (filteredModels.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="11">
          <div class="empty-state">
            <strong>No matching models</strong>
            <span>Try adjusting your search query or filter selections.</span>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  filteredModels.forEach((item, index) => {
    const m = item.model;
    const s = item.scores;
    const isSelected = selectedModelIds.has(m.model_slug || m.model_id);

    const tr = document.createElement('tr');
    tr.className = `model-row${isSelected ? ' is-selected' : ''}`;
    
    // Clicking anywhere on row (except checkbox & link) opens modal
    tr.onclick = (e) => {
      if (e.target.closest('.no-modal-click')) return;
      showModelModal(item);
    };

    // Rank marker
    let rankBadge = `<span class="rank">${item.rank}</span>`;
    if (item.rank === 1 || item.rank === 2 || item.rank === 3) {
      rankBadge = `<span class="rank rank--${item.rank}">${item.rank}</span>`;
    }

    // Score colour ramp
    const getScoreBadge = (score) => {
      let colorClass = 'score-val--low';
      if (score >= 80) colorClass = 'score-val--high';
      else if (score >= 70) colorClass = 'score-val--mid';
      return `<span class="score-val ${colorClass}">${score.toFixed(1)}%</span>`;
    };

    tr.innerHTML = `
      <td class="col-check no-modal-click" onclick="event.stopPropagation()">
        <input type="checkbox" class="checkbox model-select-cb"
          data-id="${m.model_slug || m.model_id}" ${isSelected ? 'checked' : ''}
          aria-label="Select ${m.model_id} for comparison">
      </td>
      <td class="col-rank cell-rank">${rankBadge}</td>
      <td class="col-model">
        <a href="${m.hf_url}" target="_blank" rel="noopener noreferrer"
          class="model-link no-modal-click" title="Open the model card on Hugging Face"
          onclick="event.stopPropagation()">
          <span class="truncate">${m.model_id}</span>
          <span class="model-arrow">&#8599;</span>
        </a>
        <div class="model-meta">
          <span class="tag">${m.architecture}</span>
          <span class="tag tag--quant">${m.quantization}</span>
          <span class="tag">${m.license}</span>
          ${item.notes ? `<span class="tag tag--notes">notes</span>` : ''}
        </div>
      </td>
      <td class="col-params">
        <span class="param">${m.param_count_b ? m.param_count_b.toFixed(1) + 'B' : 'N/A'}</span>
        ${m.model_ram_gb ? `<span class="param-ram">${m.model_ram_gb.toFixed(1)} GB</span>` : ''}
      </td>
      <td class="col-overall">
        <span class="score-overall">${s.overall.toFixed(1)}%</span>
      </td>
      <td class="col-score">${getScoreBadge(s.gsm8k)}</td>
      <td class="col-score">${getScoreBadge(s.humaneval)}</td>
      <td class="col-score">${getScoreBadge(s.arc)}</td>
      <td class="col-score">${getScoreBadge(s.ifeval)}</td>
      <td class="col-score">${getScoreBadge(s.tool_calling)}</td>
      <td class="col-action no-modal-click" onclick="event.stopPropagation()">
        <button type="button" class="icon-btn inspect-btn" title="Open the full run record" aria-label="Inspect ${m.model_id}">&rarr;</button>
      </td>
    `;

    // Connect row inspect button
    tr.querySelector('.inspect-btn').onclick = (e) => {
      e.stopPropagation();
      showModelModal(item);
    };

    // Connect checkbox
    const cb = tr.querySelector('.model-select-cb');
    cb.onchange = (e) => {
      const id = e.target.getAttribute('data-id');
      if (e.target.checked) {
        if (selectedModelIds.size >= 3) {
          alert("You can compare up to 3 models at a time.");
          e.target.checked = false;
          return;
        }
        selectedModelIds.add(id);
      } else {
        selectedModelIds.delete(id);
      }
      updateCompareDock();
      renderCharts();
    };

    tbody.appendChild(tr);
  });
}

// Update Compare Floating Dock & Nav Indicator
function updateCompareDock() {
  const dock = document.getElementById('compare-dock');
  const countSpan = document.getElementById('dock-count');
  const navCount = document.getElementById('compare-count');
  const navBtn = document.getElementById('nav-compare-btn');

  const count = selectedModelIds.size;
  countSpan.textContent = count;
  navCount.textContent = count;

  if (count >= 1) {
    navBtn.classList.remove('hidden');
  } else {
    navBtn.classList.add('hidden');
  }

  if (count >= 2) {
    dock.classList.remove('hidden');
  } else {
    dock.classList.add('hidden');
  }
}

// Show Model Inspector Modal
function showModelModal(item) {
  const modal = document.getElementById('model-modal');
  const m = item.model;
  const s = item.scores;
  const hw = item.hardware || {};
  const tasks = item.tasks || {};

  document.getElementById('modal-rank-badge').textContent = `Rank #${item.rank || '--'}`;
  document.getElementById('modal-tier-badge').textContent = m.param_tier || `${m.param_count_b}B`;
  document.getElementById('modal-quant-badge').textContent = m.quantization || 'BF16';
  document.getElementById('modal-model-title').textContent = m.model_id;
  
  const hfLink = document.getElementById('modal-hf-link');
  hfLink.href = m.hf_url;
  
  document.getElementById('modal-eval-id').textContent = `ID: ${item.evaluation_id}`;
  document.getElementById('modal-score-overall').textContent = `${s.overall.toFixed(2)}%`;
  document.getElementById('modal-param-count').textContent = `${m.param_count_b}B`;

  const modelRam = m.model_ram_gb || hw.model_ram_gb;
  document.getElementById('modal-model-ram').textContent = modelRam ? `${modelRam.toFixed(1)} GB` : 'N/A';
  document.getElementById('modal-hw-model-ram').textContent = modelRam ? `${modelRam.toFixed(2)} GB RAM/VRAM` : 'N/A';

  document.getElementById('modal-architecture').textContent = m.architecture || 'CausalLM';
  document.getElementById('modal-license').textContent = m.license || 'Open';

  // Notes section
  const notesContainer = document.getElementById('modal-notes-section');
  const notesText = document.getElementById('modal-notes-text');
  if (item.notes && item.notes.trim()) {
    notesText.textContent = item.notes;
    notesContainer.classList.remove('hidden');
  } else {
    notesContainer.classList.add('hidden');
  }

  // Submitter details
  document.getElementById('modal-sub-name').textContent = item.submitter?.name || 'Open Contributor';
  document.getElementById('modal-sub-github').textContent = item.submitter?.github_handle ? `@${item.submitter.github_handle}` : '@community';
  document.getElementById('modal-sub-timestamp').textContent = new Date(item.timestamp).toLocaleString();

  // Hardware Details
  document.getElementById('modal-hw-gpu').textContent = `${hw.gpu_name || 'N/A'} (x${hw.gpu_count || 1})`;
  document.getElementById('modal-hw-vram').textContent = `${hw.gpu_vram_gb || 0} GB VRAM`;
  document.getElementById('modal-hw-cuda').textContent = `CUDA ${hw.cuda_version || 'N/A'} / Driver ${hw.driver_version || 'N/A'}`;
  document.getElementById('modal-hw-backend').textContent = m.backend || 'transformers';
  document.getElementById('modal-hw-cpu').textContent = hw.cpu || 'N/A';
  document.getElementById('modal-hw-cores').textContent = `${hw.cpu_cores || 0} cores`;
  document.getElementById('modal-hw-ram').textContent = `${hw.total_ram_gb || 0} GB Host RAM`;
  document.getElementById('modal-hw-os').textContent = hw.os || 'Linux';

  // Tasks Progress Bars
  const breakdownDiv = document.getElementById('modal-tasks-breakdown');
  const taskDefinitions = [
    { key: 'gsm8k', label: 'GSM8K (Math Reasoning)', score: s.gsm8k, desc: 'Grade School Math 8K strict answer accuracy' },
    { key: 'humaneval', label: 'HumanEval (Python Coding)', score: s.humaneval, desc: 'Code generation pass@1 execution accuracy' },
    { key: 'arc', label: 'ARC Challenge (Reasoning)', score: s.arc, desc: 'Grade school science multiple-choice reasoning' },
    { key: 'ifeval', label: 'IFEval (Instruction Following)', score: s.ifeval, desc: 'Verifiable constraint satisfaction & format compliance' },
    { key: 'tool_calling', label: 'Tool Calling & Multiturn', score: s.tool_calling, desc: 'Function calling JSON schema adherence and execution' }
  ];

  breakdownDiv.innerHTML = taskDefinitions.map(t => {
    const taskData = tasks[t.key] || {};
    const passed = taskData.passed_samples ? `passed ${taskData.passed_samples}/${taskData.total_samples || '?'}` : '';
    const speed = taskData.avg_tokens_per_sec ? `${taskData.avg_tokens_per_sec} tok/s` : '';

    return `
      <div class="task">
        <div class="task-head">
          <div>
            <span class="task-name">${t.label}</span>
            <span class="task-desc">${t.desc}</span>
          </div>
          <span class="task-score">${t.score.toFixed(1)}%</span>
        </div>
        <div class="task-bar"><div class="task-fill" style="width: ${t.score}%"></div></div>
        <div class="task-meta">
          <span>${taskData.metric || 'accuracy'}${passed ? ' &middot; ' + passed : ''}</span>
          <span>${speed}</span>
        </div>
      </div>
    `;
  }).join('');

  // Raw Logs
  document.getElementById('modal-raw-logs').textContent = item.raw_logs_preview || 'No logs attached.';
  
  // Raw JSON
  document.getElementById('modal-json-filename').textContent = item.filename || 'benchmark.json';
  const jsonString = JSON.stringify(item, null, 2);
  document.getElementById('modal-raw-json').textContent = jsonString;

  // Download JSON button
  const blob = new Blob([jsonString], { type: 'application/json' });
  const downloadBtn = document.getElementById('download-json-btn');
  downloadBtn.href = URL.createObjectURL(blob);
  downloadBtn.download = item.filename || 'result.json';

  // Copy buttons
  document.getElementById('copy-logs-btn').onclick = () => {
    navigator.clipboard.writeText(item.raw_logs_preview || '');
    alert('Logs copied to clipboard!');
  };
  document.getElementById('copy-json-btn').onclick = () => {
    navigator.clipboard.writeText(jsonString);
    alert('JSON copied to clipboard!');
  };

  // Reset tab to overview
  switchModalTab('tab-overview');
  modal.classList.remove('hidden');
}

function switchModalTab(tabId) {
  document.querySelectorAll('.modal-tab-btn').forEach(btn => {
    btn.classList.toggle('is-active', btn.getAttribute('data-tab') === tabId);
  });

  document.querySelectorAll('.modal-tab-content').forEach(content => {
    if (content.id === tabId) {
      content.classList.remove('hidden');
    } else {
      content.classList.add('hidden');
    }
  });
}

// Compare Modal (Side-by-Side Head to Head)
function showCompareModal() {
  const modal = document.getElementById('compare-modal');
  const container = document.getElementById('compare-content');

  const selectedModels = allModels.filter(m => selectedModelIds.has(m.model.model_slug || m.model.model_id));
  if (selectedModels.length === 0) {
    alert("Select two or three models from the leaderboard to compare them.");
    return;
  }

  const scoreRows = [
    ['Math', 'gsm8k'],
    ['Code', 'humaneval'],
    ['ARC', 'arc'],
    ['Format', 'ifeval'],
    ['Tool use', 'tool_calling']
  ];

  container.innerHTML = `
    <table class="compare">
      <thead>
        <tr>
          <th>Metric</th>
          ${selectedModels.map(m => `
            <th>
              <div class="compare-model">${m.model.model_id}</div>
              <div class="compare-rank">Rank #${m.rank} &middot; ${m.model.param_count_b}B</div>
            </th>
          `).join('')}
        </tr>
      </thead>
      <tbody>
        <tr class="is-total">
          <td class="metric-name">Overall</td>
          ${selectedModels.map(m => `<td class="total">${m.scores.overall.toFixed(1)}%</td>`).join('')}
        </tr>
        ${scoreRows.map(([label, key]) => `
          <tr>
            <td class="metric-name">${label}</td>
            ${selectedModels.map(m => `<td class="num">${m.scores[key].toFixed(1)}%</td>`).join('')}
          </tr>
        `).join('')}
        <tr>
          <td class="metric-name">Parameters</td>
          ${selectedModels.map(m => `<td class="num">${m.model.param_count_b}B &middot; ${m.model.param_tier}</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Model footprint</td>
          ${selectedModels.map(m => `<td class="num">${m.model.model_ram_gb ? m.model.model_ram_gb.toFixed(1) + ' GB' : (m.hardware?.model_ram_gb ? m.hardware.model_ram_gb.toFixed(1) + ' GB' : 'N/A')}</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Precision</td>
          ${selectedModels.map(m => `<td class="num">${m.model.quantization}</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Architecture</td>
          ${selectedModels.map(m => `<td class="num">${m.model.architecture}</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Backend</td>
          ${selectedModels.map(m => `<td class="num">${m.model.backend}</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">GPU</td>
          ${selectedModels.map(m => `<td class="num">${m.hardware?.gpu_name || 'N/A'}</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Notes</td>
          ${selectedModels.map(m => `<td class="num">${m.notes || '—'}</td>`).join('')}
        </tr>
      </tbody>
    </table>
  `;

  modal.classList.remove('hidden');
}

// Chart.js Visualizations
function renderCharts() {
  const figuresSection = document.getElementById('figures');
  if (allModels.length === 0) {
    if (figuresSection) figuresSection.classList.add('hidden');
    return;
  }
  if (figuresSection) figuresSection.classList.remove('hidden');
  if (allModels.length === 0) return;

  // Radar Chart: top 3 models or selected models
  let modelsForRadar = allModels.slice(0, 3);
  if (selectedModelIds.size > 0) {
    const selected = allModels.filter(m => selectedModelIds.has(m.model.model_slug || m.model.model_id));
    if (selected.length > 0) modelsForRadar = selected;
  }

  const radarLabels = ['GSM8K (Math)', 'HumanEval (Code)', 'ARC (Reasoning)', 'IFEval (Format)', 'Tool Calling'];
  const colors = [
    { bg: 'rgba(163, 58, 28, 0.16)', border: '#a33a1c' },
    { bg: 'rgba(47, 90, 70, 0.16)', border: '#2f5a46' },
    { bg: 'rgba(138, 103, 20, 0.16)', border: '#8a6714' }
  ];

  const radarCtx = document.getElementById('radarChart').getContext('2d');
  if (radarChartInstance) radarChartInstance.destroy();

  radarChartInstance = new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: radarLabels,
      datasets: modelsForRadar.map((item, idx) => ({
        label: item.model.model_id.split('/')[1] || item.model.model_id,
        data: [
          item.scores.gsm8k,
          item.scores.humaneval,
          item.scores.arc,
          item.scores.ifeval,
          item.scores.tool_calling
        ],
        backgroundColor: colors[idx % colors.length].bg,
        borderColor: colors[idx % colors.length].border,
        borderWidth: 1.5,
        pointBackgroundColor: colors[idx % colors.length].border,
        pointBorderColor: '#f8f5ec',
        pointRadius: 2.5,
        pointHoverRadius: 5
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(28, 26, 20, 0.12)' },
          grid: { color: 'rgba(28, 26, 20, 0.12)' },
          pointLabels: { color: '#4b473c', font: { size: 9, family: 'IBM Plex Mono' } },
          ticks: { backdropColor: 'transparent', color: '#7a7464', stepSize: 20, font: { size: 8, family: 'IBM Plex Mono' } },
          suggestedMin: 50,
          suggestedMax: 100
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#1c1a14', font: { size: 10, family: 'IBM Plex Sans' }, boxWidth: 10, boxHeight: 10 }
        }
      }
    }
  });

  // Scatter: overall score against parameter count
  const barCtx = document.getElementById('barChart').getContext('2d');
  if (barChartInstance) barChartInstance.destroy();

  const sortedBySize = [...allModels].sort((a, b) => (a.model.param_count_b || 0) - (b.model.param_count_b || 0));

  barChartInstance = new Chart(barCtx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Models',
        data: sortedBySize.map(m => ({
          x: m.model.param_count_b || 0,
          y: m.scores.overall,
          model: m.model.model_id
        })),
        backgroundColor: sortedBySize.map(m => {
          if (m.scores.overall >= 85) return 'rgba(163, 58, 28, 0.85)';
          if (m.scores.overall >= 80) return 'rgba(138, 103, 20, 0.72)';
          return 'rgba(122, 116, 100, 0.55)';
        }),
        borderColor: 'rgba(28, 26, 20, 0.35)',
        borderWidth: 1,
        pointRadius: 5,
        pointHoverRadius: 7
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0,
          title: {
            display: true,
            text: 'parameters (billions)',
            color: '#6a6455',
            font: { size: 9, family: 'IBM Plex Mono' }
          },
          grid: { color: 'rgba(28, 26, 20, 0.08)' },
          ticks: { color: '#6a6455', font: { size: 9, family: 'IBM Plex Mono' } }
        },
        y: {
          min: 50,
          max: 100,
          title: {
            display: true,
            text: 'overall score (%)',
            color: '#6a6455',
            font: { size: 9, family: 'IBM Plex Mono' }
          },
          grid: { color: 'rgba(28, 26, 20, 0.08)' },
          ticks: { color: '#6a6455', font: { size: 9, family: 'IBM Plex Mono' } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.model} · ${ctx.raw.y}%`
          }
        }
      }
    }
  });
}

// PR Guide & Local Validator Logic
function loadPRTemplate() {
  document.getElementById('pr-template-preview').textContent = PR_TEMPLATE_TEXT;
}

function setupEventListeners() {
  // Search
  const searchInput = document.getElementById('search-input');
  const clearSearchBtn = document.getElementById('clear-search-btn');

  searchInput.addEventListener('input', () => {
    if (searchInput.value.trim().length > 0) {
      clearSearchBtn.classList.remove('hidden');
    } else {
      clearSearchBtn.classList.add('hidden');
    }
    applyFiltersAndSort();
  });

  clearSearchBtn.addEventListener('click', () => {
    searchInput.value = '';
    clearSearchBtn.classList.add('hidden');
    applyFiltersAndSort();
  });

  // Filters
  document.getElementById('tier-filter').addEventListener('change', applyFiltersAndSort);
  document.getElementById('license-filter').addEventListener('change', applyFiltersAndSort);
  document.getElementById('quant-filter').addEventListener('change', applyFiltersAndSort);
  
  // Sort select dropdown
  document.getElementById('sort-select').addEventListener('change', (e) => {
    const val = e.target.value;
    const parts = val.split('_');
    const dir = parts.pop();
    const col = parts.join('_');
    currentSort = { column: col, direction: dir };
    updateHeaderSortIcons();
    applyFiltersAndSort();
  });

  // Header Table Column Click to Sort
  document.querySelectorAll('#leaderboard-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-sort');
      if (currentSort.column === col) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
      } else {
        currentSort.column = col;
        currentSort.direction = (col === 'name' || col === 'rank') ? 'asc' : 'desc';
      }
      updateHeaderSortIcons();
      applyFiltersAndSort();
    });
  });

  // Reset Filters
  document.getElementById('reset-filters-btn').addEventListener('click', () => {
    searchInput.value = '';
    clearSearchBtn.classList.add('hidden');
    document.getElementById('tier-filter').value = 'all';
    document.getElementById('license-filter').value = 'all';
    document.getElementById('quant-filter').value = 'all';
    document.getElementById('sort-select').value = 'overall_desc';
    currentSort = { column: 'overall', direction: 'desc' };
    updateHeaderSortIcons();
    applyFiltersAndSort();
  });

  // Export buttons
  document.getElementById('export-csv-btn').addEventListener('click', exportToCSV);
  document.getElementById('export-json-btn').addEventListener('click', exportToJSON);

  // Compare Navigation and Dock buttons
  document.getElementById('nav-compare-btn').addEventListener('click', showCompareModal);
  document.getElementById('dock-compare-btn').addEventListener('click', showCompareModal);
  document.getElementById('dock-clear-btn').addEventListener('click', () => {
    selectedModelIds.clear();
    applyFiltersAndSort();
    renderCharts();
  });
  document.getElementById('compare-modal-close').addEventListener('click', () => {
    document.getElementById('compare-modal').classList.add('hidden');
  });

  // Inspector Modal Tab Switchers
  document.querySelectorAll('.modal-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchModalTab(btn.getAttribute('data-tab'));
    });
  });

  // Inspector Modal Close
  document.getElementById('modal-close-btn').addEventListener('click', () => {
    document.getElementById('model-modal').classList.add('hidden');
  });
  document.getElementById('modal-footer-close').addEventListener('click', () => {
    document.getElementById('model-modal').classList.add('hidden');
  });

  // PR Guide Modal triggers
  const prModal = document.getElementById('pr-guide-modal');
  const openPrBtns = [
    document.getElementById('open-pr-guide-btn'),
    document.getElementById('cta-pr-guide-btn'),
    document.getElementById('cta-pr-guide-btn-secondary')
  ];
  openPrBtns.forEach(btn => {
    if (btn) btn.addEventListener('click', () => {
      switchPRTab('pr-tab-steps');
      prModal.classList.remove('hidden');
    });
  });

  document.getElementById('cta-test-validator-btn').addEventListener('click', () => {
    switchPRTab('pr-tab-validator');
    prModal.classList.remove('hidden');
  });

  document.getElementById('pr-guide-close-btn').addEventListener('click', () => {
    prModal.classList.add('hidden');
  });
  document.getElementById('pr-guide-close-bottom').addEventListener('click', () => {
    prModal.classList.add('hidden');
  });

  // PR Tabs
  document.querySelectorAll('.pr-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchPRTab(btn.getAttribute('data-prtab'));
    });
  });

  // PR Template copy button
  document.getElementById('copy-pr-template-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(PR_TEMPLATE_TEXT);
    alert('PR Template copied to clipboard!');
  });

  // Validator Buttons
  document.getElementById('validate-btn').addEventListener('click', runLocalJsonValidation);
  document.getElementById('load-sample-btn').addEventListener('click', () => {
    const sample = {
      schema_version: "1.1.0",
      evaluation_id: "eval-submission-sample",
      timestamp: new Date().toISOString(),
      submitter: { name: "Your Name", github_handle: "your-gh-handle" },
      model: {
        model_id: "org/sample-model-7b",
        param_count_b: 7.2,
        param_tier: "7B–14B",
        quantization: "GGUF Q4_K_M",
        license: "Apache-2.0",
        backend: "llama.cpp"
      },
      hardware: {
        gpu_name: "NVIDIA RTX 4090",
        gpu_vram_gb: 24.0,
        cpu: "AMD Ryzen 9",
        total_ram_gb: 64.0,
        os: "Ubuntu 24.04",
        cuda_version: "12.4"
      },
      summary_scores: {
        overall_score: 82.5,
        gsm8k: 84.0,
        humaneval: 80.5,
        arc: 85.0,
        ifeval: 81.0,
        tool_calling: 78.0
      }
    };
    document.getElementById('validator-input').value = JSON.stringify(sample, null, 2);
  });
}

function switchPRTab(tabId) {
  document.querySelectorAll('.pr-tab-btn').forEach(btn => {
    btn.classList.toggle('is-active', btn.getAttribute('data-prtab') === tabId);
  });

  document.querySelectorAll('.pr-tab-content').forEach(c => {
    if (c.id === tabId) {
      c.classList.remove('hidden');
    } else {
      c.classList.add('hidden');
    }
  });
}

// In-Browser Submission JSON Validator
function runLocalJsonValidation() {
  const input = document.getElementById('validator-input').value.trim();
  const feedback = document.getElementById('validator-feedback');

  if (!input) {
    feedback.className = 'feedback feedback--error';
    feedback.innerHTML = '<strong>No JSON provided</strong>Paste result JSON to validate.';
    feedback.classList.remove('hidden');
    return;
  }

  try {
    const data = JSON.parse(input);
    const errors = [];

    if (!data.model) errors.push("Missing 'model' object.");
    else {
      if (!data.model.model_id) errors.push("Missing 'model.model_id'.");
      if (!data.model.param_count_b) errors.push("Missing 'model.param_count_b'.");
      if (!data.model.quantization) errors.push("Missing 'model.quantization'.");
      if (!data.model.license) errors.push("Missing 'model.license'.");
    }

    if (!data.hardware) errors.push("Missing 'hardware' (GPU, CPU, RAM).");
    if (!data.summary_scores && !data.tasks) errors.push("Missing scores ('summary_scores' or 'tasks').");

    if (errors.length > 0) {
      feedback.className = 'feedback feedback--error';
      feedback.innerHTML = `<strong>${errors.length} field${errors.length === 1 ? '' : 's'} missing</strong><ul>${errors.map(e => `<li>${e}</li>`).join('')}</ul>`;
      feedback.classList.remove('hidden');
    } else {
      feedback.className = 'feedback feedback--ok';
      feedback.innerHTML = `<strong>Valid</strong>All required fields are present.`;
      feedback.classList.remove('hidden');
    }
  } catch (err) {
    feedback.className = 'feedback feedback--error';
    feedback.innerHTML = `<strong>Invalid JSON</strong>${err.message}`;
    feedback.classList.remove('hidden');
  }
}

// Update table header sort markers
function updateHeaderSortIcons() {
  const cols = ['rank', 'name', 'params', 'overall', 'gsm8k', 'humaneval', 'arc', 'ifeval', 'tool_calling'];
  cols.forEach(col => {
    const icon = document.getElementById(`sort-icon-${col}`);
    if (!icon) return;

    if (currentSort.column === col) {
      icon.textContent = currentSort.direction === 'asc' ? '\u2191' : '\u2193';
      icon.className = 'sort-mark is-active';
    } else {
      icon.textContent = '\u00b7';
      icon.className = 'sort-mark';
    }
  });
}

// Export CSV of current filtered view
function exportToCSV() {
  if (filteredModels.length === 0) {
    alert("No models to export.");
    return;
  }

  const headers = ['Rank', 'Model ID', 'Parameters (B)', 'Tier', 'Overall Score', 'GSM8K', 'HumanEval', 'ARC', 'IFEval', 'Tool Calling', 'Quantization', 'Architecture', 'License', 'Backend', 'GPU'];
  const rows = filteredModels.map(item => [
    item.rank,
    `"${item.model.model_id}"`,
    item.model.param_count_b,
    `"${item.model.param_tier}"`,
    item.scores.overall,
    item.scores.gsm8k,
    item.scores.humaneval,
    item.scores.arc,
    item.scores.ifeval,
    item.scores.tool_calling,
    `"${item.model.quantization}"`,
    `"${item.model.architecture}"`,
    `"${item.model.license}"`,
    `"${item.model.backend}"`,
    `"${item.hardware?.gpu_name || 'N/A'}"`
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `ai-model-leaderboard-${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Export JSON of current filtered view
function exportToJSON() {
  if (filteredModels.length === 0) {
    alert("No models to export.");
    return;
  }
  const jsonContent = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(filteredModels, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", jsonContent);
  link.setAttribute("download", `ai-model-leaderboard-${new Date().toISOString().slice(0, 10)}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Start application when DOM is ready
window.addEventListener('DOMContentLoaded', initApp);
