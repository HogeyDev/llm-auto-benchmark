/**
 * SLM-Bench Leaderboard Application (Version 2.0).
 * Connects to data/leaderboard.json, provides interactive search, filtering by
 * model size, status (Official FULL vs Partial), precision, and license.
 * Visualizes official benchmarks: MMLU-Pro, GPQA Diamond, BBH, GSM8K, MATH-500, IFEval, EvalPlus, BFCL, XSTest.
 */

let allModels = [];
let filteredModels = [];
let selectedModelIds = new Set();
let currentSort = { column: 'rank', direction: 'asc' };
let rawLeaderboardData = null;

let radarChartInstance = null;
let barChartInstance = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  setupEventListeners();
  await loadLeaderboardData();
}

function setupEventListeners() {
  // Search bar
  const searchInput = document.getElementById('search-input');
  const clearSearchBtn = document.getElementById('clear-search-btn');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      if (clearSearchBtn) {
        if (e.target.value.trim().length > 0) {
          clearSearchBtn.classList.remove('hidden');
        } else {
          clearSearchBtn.classList.add('hidden');
        }
      }
      applyFiltersAndSort();
    });
  }

  if (clearSearchBtn) {
    clearSearchBtn.addEventListener('click', () => {
      if (searchInput) {
        searchInput.value = '';
        clearSearchBtn.classList.add('hidden');
        applyFiltersAndSort();
      }
    });
  }

  // Filters
  const tierFilter = document.getElementById('tier-filter');
  const statusFilter = document.getElementById('status-filter');
  const profileFilter = document.getElementById('profile-filter');
  const quantFilter = document.getElementById('quant-filter');
  const licenseFilter = document.getElementById('license-filter');
  const sortSelect = document.getElementById('sort-select');

  [tierFilter, statusFilter, profileFilter, quantFilter, licenseFilter].forEach(el => {
    if (el) el.addEventListener('change', () => applyFiltersAndSort());
  });

  if (sortSelect) {
    sortSelect.addEventListener('change', (e) => {
      const [col, dir] = e.target.value.split('_');
      currentSort = { column: col, direction: dir || 'desc' };
      applyFiltersAndSort();
    });
  }

  // Reset Filters button
  const resetBtn = document.getElementById('reset-filters-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (searchInput) searchInput.value = '';
      if (clearSearchBtn) clearSearchBtn.classList.add('hidden');
      if (tierFilter) tierFilter.value = 'all';
      if (statusFilter) statusFilter.value = 'official_only';
      if (profileFilter) profileFilter.value = 'all';
      if (quantFilter) quantFilter.value = 'all';
      if (licenseFilter) licenseFilter.value = 'all';
      if (sortSelect) sortSelect.value = 'overall_desc';
      currentSort = { column: 'rank', direction: 'asc' };
      applyFiltersAndSort();
    });
  }

  // Table header click-to-sort
  const headers = document.querySelectorAll('#leaderboard-table th[data-sort]');
  headers.forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-sort');
      if (currentSort.column === col) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
      } else {
        currentSort.column = col;
        currentSort.direction = (col === 'rank' || col === 'name' || col === 'params') ? 'asc' : 'desc';
      }
      applyFiltersAndSort();
    });
  });

  // Modal close buttons
  setupModalDismiss();

  // Export buttons
  const csvBtn = document.getElementById('export-csv-btn');
  const jsonBtn = document.getElementById('export-json-btn');
  if (csvBtn) csvBtn.addEventListener('click', exportCSV);
  if (jsonBtn) jsonBtn.addEventListener('click', exportJSON);
}

// Data Fetching
async function loadLeaderboardData() {
  const countText = document.getElementById('results-count-text');
  if (countText) countText.textContent = "Loading benchmarks...";

  try {
    const res = await fetch('data/leaderboard.json');
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    rawLeaderboardData = await res.json();
    
    // Process models
    if (rawLeaderboardData && Array.isArray(rawLeaderboardData.models)) {
      allModels = rawLeaderboardData.models;
    } else {
      allModels = [];
    }

    populateLicenseFilter();
    updateStatCards();
    applyFiltersAndSort();
    renderCharts();
  } catch (err) {
    console.warn("Could not load data/leaderboard.json directly, searching results...", err);
    await loadDirectResults();
  }
}

async function loadDirectResults() {
  const resultFiles = [
    'spark-x2.6-4b-int4.json',
    'llama-3.2-3b-bf16.json',
    'qwen2.5-3b-instruct-bf16.json',
    'phi-3.5-mini-bf16.json'
  ];

  const loaded = [];
  for (const f of resultFiles) {
    try {
      const res = await fetch(`data/results/${f}`);
      if (res.ok) {
        const data = await res.json();
        loaded.push(normalizeEntry(data, f));
      }
    } catch (e) {
      // ignore
    }
  }

  allModels = loaded;
  populateLicenseFilter();
  updateStatCards();
  applyFiltersAndSort();
  renderCharts();
}

function normalizeEntry(data, filename) {
  const model = data.model || {};
  const tasks = data.tasks || {};
  const categories = data.categories || {};
  const hw = data.hardware || {};
  const telem = data.telemetry || hw.telemetry || {};

  const isLegacy = data.is_legacy || false;
  const status = isLegacy ? "INVALID" : (data.status || "FULL");
  const eligible = isLegacy ? false : (data.eligible_for_leaderboard !== undefined ? data.eligible_for_leaderboard : (status === "FULL"));

  // Canonical Official Benchmarks
  const mmlu_pro = tasks.mmlu_pro?.normalized_score || null;
  const gpqa = tasks.gpqa_diamond?.normalized_score || tasks.gpqa?.normalized_score || null;
  const bbh = tasks.bbh?.normalized_score || null;
  const gsm8k = tasks.gsm8k?.normalized_score || null;
  const math500 = tasks.math500?.normalized_score || null;
  const ifeval = tasks.ifeval?.normalized_score || null;
  const evalplus = tasks.evalplus?.normalized_score || tasks.evalplus_humanevalplus?.normalized_score || null;
  const bfcl = tasks.bfcl?.normalized_score || null;
  const xstest = tasks.xstest?.normalized_score || tasks.safety?.normalized_score || null;

  // Category Pillars
  const reasoning = categories.reasoning?.normalized_score || (mmlu_pro || gpqa || 0);
  const math = categories.mathematics?.normalized_score || (gsm8k || math500 || 0);
  const instruction = categories.instruction?.normalized_score || (ifeval || 0);
  const coding = categories.coding?.normalized_score || (evalplus || null);
  const tool_calling = categories.tool_calling?.normalized_score || (bfcl || null);

  const overall = data.slm_index || 0.0;
  const modelRam = hw.model_ram_gb || model.model_ram_gb || data.model_ram_gb || (model.param_count_b ? (model.param_count_b * 0.65) : 8.0);
  const score_per_gb = modelRam ? parseFloat((overall / Math.max(1.0, modelRam)).toFixed(2)) : 0.0;
  const tps = telem.decode_tok_per_sec || data.avg_tokens_per_sec || 0.0;

  return {
    filename: filename,
    file_path: `data/results/${filename}`,
    evaluation_id: data.evaluation_id || "eval-" + Math.random().toString(36).substring(7),
    timestamp: data.timestamp || new Date().toISOString(),
    profile: data.profile || "standard",
    status: status,
    eligible_for_leaderboard: eligible,
    is_legacy: isLegacy,
    submitter: data.submitter || { name: "Open Contributor", github_handle: "community" },
    model: {
      model_id: model.model_id || "Unknown",
      model_slug: model.model_slug || (model.model_id || "").replace('/', '-'),
      hf_url: model.hf_url || `https://huggingface.co/${model.model_id}`,
      param_count_b: parseFloat(model.param_count_b || 0),
      param_tier: model.param_tier || "3B–8B",
      architecture: model.architecture || "Transformer",
      quantization: model.quantization || "BF16",
      model_ram_gb: modelRam,
      context_length: model.context_length || 32768,
      license: model.license || "Open",
      backend: model.backend || "transformers",
      reasoning_effort: model.reasoning_effort,
    },
    scores: {
      overall: parseFloat(overall.toFixed(2)),
      ci_lower: data.ci_lower || null,
      ci_upper: data.ci_upper || null,
      coding_agentic: data.coding_agentic_index || null,
      score_per_gb: score_per_gb,
      speed: parseFloat(tps.toFixed(1)),
      reasoning: reasoning ? parseFloat(reasoning.toFixed(2)) : null,
      mathematics: math ? parseFloat(math.toFixed(2)) : null,
      instruction: instruction ? parseFloat(instruction.toFixed(2)) : null,
      coding: coding !== null ? parseFloat(coding.toFixed(2)) : null,
      tool_calling: tool_calling !== null ? parseFloat(tool_calling.toFixed(2)) : null,
      mmlu_pro: mmlu_pro,
      gpqa_diamond: gpqa,
      bbh: bbh,
      gsm8k: gsm8k,
      math500: math500,
      ifeval: ifeval,
      evalplus: evalplus,
      bfcl: bfcl,
      xstest: xstest,
    },
    hardware: hw,
    telemetry: {
      ttft_ms: telem.ttft_ms || 0,
      decode_tok_per_sec: tps,
      tpot_ms: telem.tpot_ms || 0,
      peak_ram_mb: telem.peak_ram_mb || 0,
      peak_vram_mb: telem.peak_vram_mb || 0,
    },
    categories: categories,
    tasks: tasks,
    notes: data.notes || "",
  };
}

function populateLicenseFilter() {
  const licenseSelect = document.getElementById('license-filter');
  if (!licenseSelect) return;
  const licenses = new Set();
  allModels.forEach(m => {
    if (m.model?.license) licenses.add(m.model.license);
  });

  licenseSelect.innerHTML = '<option value="all">All licenses</option>';
  Array.from(licenses).sort().forEach(lic => {
    const opt = document.createElement('option');
    opt.value = lic;
    opt.textContent = lic;
    licenseSelect.appendChild(opt);
  });
}

function updateStatCards() {
  const modelCountEl = document.getElementById('stat-model-count');
  const topModelEl = document.getElementById('stat-top-model');
  const topScoreEl = document.getElementById('stat-top-score');
  const topSpeedEl = document.getElementById('stat-top-speed');
  const topSpeedModelEl = document.getElementById('stat-top-speed-model');

  const eligibleModels = allModels.filter(m => m.eligible_for_leaderboard);
  const displayCount = eligibleModels.length > 0 ? eligibleModels.length : allModels.length;

  if (modelCountEl) modelCountEl.textContent = displayCount;

  if (eligibleModels.length > 0) {
    const sorted = [...eligibleModels].sort((a, b) => b.scores.overall - a.scores.overall);
    const top = sorted[0];
    if (topModelEl) topModelEl.textContent = top.model.model_id;
    if (topScoreEl) topScoreEl.textContent = `${top.scores.overall.toFixed(1)}%`;
  }

  if (allModels.length > 0) {
    const topSpeed = allModels.reduce((prev, cur) => {
      const spdCur = cur.telemetry?.decode_tok_per_sec || cur.scores?.speed || 0;
      const spdPrev = prev.telemetry?.decode_tok_per_sec || prev.scores?.speed || 0;
      return spdCur > spdPrev ? cur : prev;
    }, allModels[0]);
    if (topSpeedEl) {
      const spdVal = topSpeed.telemetry?.decode_tok_per_sec || topSpeed.scores?.speed || 0;
      topSpeedEl.textContent = spdVal.toFixed(1) + ' tok/s';
      if (topSpeedModelEl) topSpeedModelEl.textContent = topSpeed.model.model_id;
    }
  }
}

// Filter and Sort Engine
function applyFiltersAndSort() {
  const searchInput = document.getElementById('search-input');
  const searchTerm = (searchInput?.value || '').trim().toLowerCase();
  const tierFilter = document.getElementById('tier-filter')?.value || 'all';
  const statusFilter = document.getElementById('status-filter')?.value || 'official_only';
  const profileFilter = document.getElementById('profile-filter')?.value || 'all';
  const licenseFilter = document.getElementById('license-filter')?.value || 'all';
  const quantFilter = document.getElementById('quant-filter')?.value || 'all';

  filteredModels = allModels.filter(item => {
    const m = item.model || {};
    const prof = (item.profile || 'standard').toLowerCase();

    // Status filter: only official FULL runs vs all runs
    if (statusFilter === 'official_only') {
      if (!item.eligible_for_leaderboard) return false;
    }

    // Search filter
    if (searchTerm) {
      const match = (
        (m.model_id || '').toLowerCase().includes(searchTerm) ||
        (m.model_slug || '').toLowerCase().includes(searchTerm) ||
        (m.architecture || '').toLowerCase().includes(searchTerm) ||
        (m.backend || '').toLowerCase().includes(searchTerm) ||
        (m.quantization || '').toLowerCase().includes(searchTerm)
      );
      if (!match) return false;
    }

    // Parameter Tier filter
    if (tierFilter !== 'all') {
      const p = m.param_count_b || 0;
      if (tierFilter === '< 3B' && p >= 3.0) return false;
      if (tierFilter === '3B–8B' && (p < 3.0 || p > 8.0)) return false;
      if (tierFilter === '8B–14B' && (p <= 8.0 || p > 14.0)) return false;
      if (tierFilter === '14B–35B' && (p <= 14.0 || p > 35.0)) return false;
      if (tierFilter === '35B+' && p <= 35.0) return false;
    }

    // Profile filter
    if (profileFilter !== 'all') {
      if (prof !== profileFilter.toLowerCase()) return false;
    }

    // License filter
    if (licenseFilter !== 'all') {
      if (m.license?.toLowerCase() !== licenseFilter.toLowerCase()) return false;
    }

    // Quantization filter
    if (quantFilter !== 'all') {
      const q = (m.quantization || '').toUpperCase();
      if (quantFilter === 'BF16' && !q.includes('BF16') && !q.includes('FP16')) return false;
      if (quantFilter === 'GGUF' && !q.includes('GGUF')) return false;
      if (quantFilter === 'AWQ' && !q.includes('AWQ')) return false;
      if (quantFilter === 'NF4' && !q.includes('NF4')) return false;
      if (quantFilter === 'INT4' && !q.includes('INT4') && !q.includes('4BIT') && !q.includes('Q4')) return false;
    }

    return true;
  });

  sortModels(currentSort.column, currentSort.direction);
  renderTable();
  updateCompareDock();
}

function sortModels(column, direction) {
  filteredModels.sort((a, b) => {
    let valA, valB;

    switch (column) {
      case 'rank':
        valA = a.rank && a.rank !== '-' ? a.rank : 9999;
        valB = b.rank && b.rank !== '-' ? b.rank : 9999;
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
        valA = a.scores.overall || 0;
        valB = b.scores.overall || 0;
        break;
      case 'reasoning':
        valA = a.scores.reasoning || a.scores.mmlu_pro || 0;
        valB = b.scores.reasoning || b.scores.mmlu_pro || 0;
        break;
      case 'math':
        valA = a.scores.mathematics || a.scores.gsm8k || 0;
        valB = b.scores.mathematics || b.scores.gsm8k || 0;
        break;
      case 'instruction':
        valA = a.scores.instruction || a.scores.ifeval || 0;
        valB = b.scores.instruction || b.scores.ifeval || 0;
        break;
      case 'coding':
        valA = a.scores.coding || a.scores.evalplus || 0;
        valB = b.scores.coding || b.scores.evalplus || 0;
        break;
      case 'tool_calling':
        valA = a.scores.tool_calling || a.scores.bfcl || 0;
        valB = b.scores.tool_calling || b.scores.bfcl || 0;
        break;
      case 'speed':
        valA = a.telemetry?.decode_tok_per_sec || a.scores?.speed || 0;
        valB = b.telemetry?.decode_tok_per_sec || b.scores?.speed || 0;
        break;
      case 'efficiency':
        valA = a.scores.score_per_gb || 0;
        valB = b.scores.score_per_gb || 0;
        break;
      default:
        valA = a.scores.overall || 0;
        valB = b.scores.overall || 0;
    }

    if (valA < valB) return direction === 'asc' ? -1 : 1;
    if (valA > valB) return direction === 'asc' ? 1 : -1;
    return 0;
  });
}

function renderTable() {
  const tbody = document.getElementById('table-body');
  const countText = document.getElementById('results-count-text');
  tbody.innerHTML = '';

  if (allModels.length === 0) {
    if (countText) countText.textContent = `0 models`;
    tbody.innerHTML = `
      <tr>
        <td colspan="13">
          <div class="empty-state">
            <strong>No benchmarks recorded yet</strong>
            <span>Run <code>slm-bench evaluate</code> to generate benchmark results.</span>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  if (countText) countText.textContent = `Showing ${filteredModels.length} of ${allModels.length} models`;

  if (filteredModels.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="13">
          <div class="empty-state">
            <strong>No matching models</strong>
            <span>Try adjusting your filter options (e.g. switch status from "Official FULL Only" to "All Runs").</span>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  filteredModels.forEach((item, index) => {
    const m = item.model;
    const s = item.scores;
    const telem = item.telemetry || {};
    const isSelected = selectedModelIds.has(m.model_slug || m.model_id);
    const prof = item.profile || 'standard';
    const status = item.status || 'FULL';

    const tr = document.createElement('tr');
    tr.className = `model-row${isSelected ? ' is-selected' : ''}`;
    
    tr.onclick = (e) => {
      if (e.target.closest('.no-modal-click')) return;
      showModelModal(item);
    };

    let rankBadge = `<span class="rank">${item.rank || (index + 1)}</span>`;
    if (item.rank === 1 || item.rank === 2 || item.rank === 3) {
      rankBadge = `<span class="rank rank--${item.rank}">${item.rank}</span>`;
    }

    const getScoreBadge = (score) => {
      if (score === null || score === undefined) return '<span style="color:#64748b;">—</span>';
      let colorClass = 'score-val--low';
      if (score >= 75) colorClass = 'score-val--high';
      else if (score >= 60) colorClass = 'score-val--mid';
      return `<span class="score-val ${colorClass}">${score.toFixed(1)}%</span>`;
    };

    let statusTag = '';
    if (item.is_legacy) {
      statusTag = '<span class="tag" style="background:#374151;color:#9ca3af;" title="Deprecated custom benchmark record">DEPRECATED</span>';
    } else if (status === 'FULL') {
      statusTag = '<span class="tag" style="background:#065f46;color:#a7f3d0;" title="Complete official evaluation split">FULL</span>';
    } else {
      statusTag = '<span class="tag" style="background:#854d0e;color:#fef08a;" title="Sampled evaluation — ineligible for full leaderboard">PARTIAL</span>';
    }

    const speedVal = telem.decode_tok_per_sec || s.speed || 0;
    const effVal = s.score_per_gb || (m.model_ram_gb ? (s.overall / m.model_ram_gb).toFixed(1) : 0);

    tr.innerHTML = `
      <td class="col-check no-modal-click" onclick="event.stopPropagation()">
        <input type="checkbox" class="checkbox model-select-cb"
          data-id="${m.model_slug || m.model_id}" ${isSelected ? 'checked' : ''}
          aria-label="Select ${m.model_id} for comparison">
      </td>
      <td class="col-rank cell-rank">${rankBadge}</td>
      <td class="col-model">
        <a href="${m.hf_url}" target="_blank" rel="noopener noreferrer"
          class="model-link no-modal-click" title="Open model on Hugging Face"
          onclick="event.stopPropagation()">
          <span class="truncate">${m.model_id}</span>
          <span class="model-arrow">&#8599;</span>
        </a>
        <div class="model-meta">
          ${statusTag}
          <span class="tag tag--quant">${m.quantization}</span>
          <span class="tag">${m.backend}</span>
          <span class="tag">${m.architecture}</span>
        </div>
      </td>
      <td class="col-params">
        <span class="param">${m.param_count_b ? m.param_count_b.toFixed(1) + 'B' : 'N/A'}</span>
        ${m.model_ram_gb ? `<span class="param-ram">${m.model_ram_gb.toFixed(1)} GB</span>` : ''}
      </td>
      <td class="col-overall">
        <span class="score-overall">${s.overall.toFixed(1)}%</span>
        ${s.ci_lower ? `<span class="ci-range">[${s.ci_lower.toFixed(1)}%, ${s.ci_upper.toFixed(1)}%]</span>` : ''}
      </td>
      <td class="col-score">${getScoreBadge(s.reasoning)}</td>
      <td class="col-score">${getScoreBadge(s.mathematics)}</td>
      <td class="col-score">${getScoreBadge(s.instruction)}</td>
      <td class="col-score">${getScoreBadge(s.coding)}</td>
      <td class="col-score">${getScoreBadge(s.tool_calling)}</td>
      <td class="col-score">
        <span class="score-val ${speedVal >= 60 ? 'score-val--high' : 'score-val--mid'}">${speedVal ? speedVal.toFixed(1) : '—'}</span>
      </td>
      <td class="col-score track-col-efficiency">
        <span class="score-val ${effVal >= 10 ? 'score-val--high' : 'score-val--mid'}">${typeof effVal === 'number' ? effVal.toFixed(1) : effVal}</span>
      </td>
      <td class="col-action no-modal-click" onclick="event.stopPropagation()">
        <button type="button" class="icon-btn inspect-btn" title="Open the full run record" aria-label="Inspect ${m.model_id}">&rarr;</button>
      </td>
    `;

    tr.querySelector('.inspect-btn').onclick = (e) => {
      e.stopPropagation();
      showModelModal(item);
    };

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

function updateCompareDock() {
  const dock = document.getElementById('compare-dock');
  const countSpan = document.getElementById('dock-count');
  const navCount = document.getElementById('compare-count');
  const navBtn = document.getElementById('nav-compare-btn');

  const count = selectedModelIds.size;
  if (countSpan) countSpan.textContent = count;
  if (navCount) navCount.textContent = count;

  if (count >= 1) {
    if (navBtn) navBtn.classList.remove('hidden');
  } else {
    if (navBtn) navBtn.classList.add('hidden');
  }

  if (count >= 2) {
    if (dock) dock.classList.remove('hidden');
  } else {
    if (dock) dock.classList.add('hidden');
  }
}

function showModelModal(item) {
  const modal = document.getElementById('model-modal');
  if (!modal) return;
  const m = item.model;
  const s = item.scores;
  const hw = item.hardware || {};
  const tasks = item.tasks || {};
  const telem = item.telemetry || {};

  document.getElementById('modal-rank-badge').textContent = `Rank #${item.rank || '--'}`;
  document.getElementById('modal-tier-badge').textContent = m.param_tier || `${m.param_count_b}B`;
  document.getElementById('modal-quant-badge').textContent = m.quantization || 'BF16';
  document.getElementById('modal-model-title').textContent = m.model_id;

  const profileBadge = document.getElementById('modal-profile-badge');
  if (profileBadge) {
    profileBadge.textContent = (item.status || "FULL").toUpperCase();
    profileBadge.className = `chip ${item.status === 'FULL' ? 'tag--profile-standard' : 'tag--profile-quick'}`;
  }

  const hfLink = document.getElementById('modal-hf-link');
  if (hfLink) hfLink.href = m.hf_url;

  document.getElementById('modal-eval-id').textContent = `ID: ${item.evaluation_id}`;
  document.getElementById('modal-score-overall').textContent = `${s.overall.toFixed(2)}%`;

  const ciRangeEl = document.getElementById('modal-ci-range');
  if (ciRangeEl) {
    if (s.ci_lower !== null && s.ci_upper !== null) {
      ciRangeEl.textContent = `95% Wilson CI: [${s.ci_lower.toFixed(1)}%, ${s.ci_upper.toFixed(1)}%]`;
    } else {
      ciRangeEl.textContent = `95% CI: N/A`;
    }
  }

  const throughputEl = document.getElementById('modal-throughput');
  if (throughputEl) {
    const spd = telem.decode_tok_per_sec || s.speed || 0;
    throughputEl.textContent = spd ? `${spd.toFixed(1)} tok/s` : 'N/A';
  }

  document.getElementById('modal-param-count').textContent = `${m.param_count_b || 'N/A'}B`;
  const modelRam = m.model_ram_gb || hw.model_ram_gb;
  document.getElementById('modal-model-ram').textContent = modelRam ? `${modelRam.toFixed(1)} GB` : 'N/A';

  // Hardware Details
  document.getElementById('modal-hw-gpu').textContent = `${hw.gpu_name || 'CPU'} (x${hw.gpu_count || 1})`;
  document.getElementById('modal-hw-vram').textContent = `${hw.gpu_vram_gb || 0} GB VRAM`;
  document.getElementById('modal-hw-backend').textContent = m.backend || 'transformers';
  document.getElementById('modal-hw-cpu').textContent = hw.cpu || 'N/A';
  document.getElementById('modal-hw-ram').textContent = `${hw.total_ram_gb || 0} GB Host RAM`;
  document.getElementById('modal-hw-os').textContent = hw.os || 'Linux';

  // Dynamic Tasks Breakdown
  const breakdownDiv = document.getElementById('modal-tasks-breakdown');
  const taskKeys = Object.keys(tasks);

  if (taskKeys.length > 0) {
    breakdownDiv.innerHTML = taskKeys.map(k => {
      const taskData = tasks[k];
      const score = taskData.normalized_score !== undefined ? taskData.normalized_score : (taskData.raw_score * 100 || 0);
      const passed = taskData.passed_samples !== undefined ? `${taskData.passed_samples}/${taskData.total_samples || '?'}` : '';
      const ci = (taskData.ci_lower !== undefined && taskData.ci_upper !== undefined) ? `[${taskData.ci_lower}%, ${taskData.ci_upper}%]` : '';
      const title = k.replace(/_/g, ' ').toUpperCase();

      return `
        <div class="task">
          <div class="task-head">
            <div>
              <span class="task-name">${title}</span>
              ${ci ? `<span class="task-ci">95% CI: ${ci}</span>` : ''}
            </div>
            <span class="task-score">${score.toFixed(1)}%</span>
          </div>
          <div class="task-bar"><div class="task-fill" style="width: ${Math.min(100, score)}%"></div></div>
          <div class="task-meta">
            <span>${taskData.raw_metric_name || 'accuracy'} &middot; passed ${passed}</span>
            <span>Status: ${taskData.status || 'FULL'}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  // Raw JSON
  const jsonString = JSON.stringify(item, null, 2);
  const rawJsonEl = document.getElementById('modal-raw-json');
  if (rawJsonEl) rawJsonEl.textContent = jsonString;

  modal.classList.remove('hidden');
}

function setupModalDismiss() {
  const modal = document.getElementById('model-modal');
  const closeBtns = document.querySelectorAll('.modal-dismiss');
  closeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (modal) modal.classList.add('hidden');
    });
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
      modal.classList.add('hidden');
    }
  });
}

function renderCharts() {
  const radarCanvas = document.getElementById('radarChart');
  const barCanvas = document.getElementById('barChart');

  if (!radarCanvas || !barCanvas) return;

  const modelsToPlot = selectedModelIds.size > 0
    ? allModels.filter(m => selectedModelIds.has(m.model.model_slug || m.model.model_id))
    : allModels.filter(m => m.eligible_for_leaderboard).slice(0, 3);

  if (modelsToPlot.length === 0) return;

  // Radar Chart: Official Benchmark Pillars
  const labels = ['Reasoning', 'Mathematics', 'Instruction', 'Coding', 'Tool Calling'];
  const datasets = modelsToPlot.map((m, idx) => {
    const colors = ['rgba(6, 182, 212, ', 'rgba(99, 102, 241, ', 'rgba(236, 72, 153, '];
    const c = colors[idx % colors.length];
    return {
      label: m.model.model_id,
      data: [
        m.scores.reasoning || 0,
        m.scores.mathematics || 0,
        m.scores.instruction || 0,
        m.scores.coding || 0,
        m.scores.tool_calling || 0
      ],
      backgroundColor: c + '0.2)',
      borderColor: c + '1.0)',
      borderWidth: 2,
      pointBackgroundColor: c + '1.0)',
    };
  });

  if (radarChartInstance) radarChartInstance.destroy();
  radarChartInstance = new Chart(radarCanvas, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { color: '#94a3b8', backdropColor: 'transparent' },
          grid: { color: '#334155' },
          pointLabels: { color: '#e2e8f0', font: { size: 12 } }
        }
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0' } }
      }
    }
  });

  // Scatter Chart: Score vs Memory Footprint
  const scatterData = allModels.map(m => ({
    x: m.model.model_ram_gb || 8.0,
    y: m.scores.overall,
    model: m.model.model_id
  }));

  if (barChartInstance) barChartInstance.destroy();
  barChartInstance = new Chart(barCanvas, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Models (Score vs RAM GB)',
        data: scatterData,
        backgroundColor: '#06b6d4',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'RAM / VRAM Footprint (GB)', color: '#94a3b8' },
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          min: 0,
          max: 100,
          title: { display: true, text: 'Composite SLM-Index (%)', color: '#94a3b8' },
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0' } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.model}: ${ctx.raw.y.toFixed(1)}% (${ctx.raw.x.toFixed(1)} GB)`
          }
        }
      }
    }
  });
}

function exportCSV() {
  if (!filteredModels.length) return;
  const headers = ['Model ID', 'Parameters (B)', 'Quantization', 'Status', 'SLM-Index', 'Reasoning', 'Math', 'Instruction', 'Coding', 'Tools', 'RAM GB', 'Speed tok/s'];
  const rows = filteredModels.map(m => [
    `"${m.model.model_id}"`,
    m.model.param_count_b || '',
    `"${m.model.quantization}"`,
    m.status,
    m.scores.overall,
    m.scores.reasoning || '',
    m.scores.mathematics || '',
    m.scores.instruction || '',
    m.scores.coding || '',
    m.scores.tool_calling || '',
    m.model.model_ram_gb || '',
    m.telemetry?.decode_tok_per_sec || ''
  ]);
  const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'slm_bench_leaderboard.csv';
  a.click();
}

function exportJSON() {
  if (!filteredModels.length) return;
  const jsonContent = JSON.stringify(filteredModels, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'slm_bench_leaderboard.json';
  a.click();
}
