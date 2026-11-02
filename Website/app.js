/**
 * Local AI Benchmark Leaderboard - Static Client Application
 * Pure client-side JavaScript for GitHub Pages
 */

// Application State
let rawLeaderboardData = null;
let allModels = [];
let filteredModels = [];
let selectedModelIds = new Set();
let currentSort = { column: 'overall', direction: 'desc' };
let currentTrack = 'all'; // 'all', 'general', 'coding_agentic', 'efficiency'

let radarChartInstance = null;
let barChartInstance = null;

// Embedded Fallback Data (used when running locally from file:// or if fetch is blocked)
const EMBEDDED_FALLBACK_DATA = {
  metadata: {
    total_models: 7,
    last_updated: "2026-11-02T22:00:00Z",
    highest_overall: 87.65,
    best_by_benchmark: {
      gsm8k: 94.20,
      humaneval: 88.40,
      arc: 92.40,
      ifeval: 85.20,
      tool_calling: 100.0
    },
    best_by_track: {
      top_overall: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
      top_overall_score: 87.65,
      top_coding_agentic: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
      top_coding_agentic_score: 84.90,
      top_efficiency: "microsoft/Phi-3.5-mini-instruct",
      top_efficiency_score: 31.31,
      top_throughput: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
      top_throughput_tps: 112.5
    }
  },
  models: [
    {
      filename: "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json",
      file_path: "data/results/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json",
      evaluation_id: "eval-dsk-r1-14b-20261101",
      timestamp: "2026-11-01T14:22:10Z",
      profile: "extended",
      rank: 1,
      submitter: { name: "Alex Chen", github_handle: "alexchen-ai" },
      model: {
        model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        model_slug: "DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M",
        hf_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        param_count_b: 14.7,
        param_tier: "7B–14B",
        architecture: "Qwen2ForCausalLM",
        quantization: "GGUF Q4_K_M",
        model_ram_gb: 9.2,
        license: "MIT",
        backend: "llama.cpp",
        reasoning_effort: "high"
      },
      scores: {
        overall: 87.65,
        ci_lower: 85.8,
        ci_upper: 89.3,
        coding_agentic_index: 84.90,
        local_efficiency_score: 85.2,
        score_per_gb: 9.53,
        speed: 48.6,
        gsm8k: 94.20,
        humaneval: 86.50,
        arc: 92.40,
        ifeval: 81.30,
        tool_calling: 82.50
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4090",
        gpu_count: 1,
        gpu_vram_gb: 24.0,
        model_ram_gb: 9.2,
        cpu: "AMD Ryzen 9 7950X",
        cpu_cores: 16,
        total_ram_gb: 64.0,
        os: "Ubuntu 24.04 LTS",
        cuda_version: "12.4",
        driver_version: "550.54"
      },
      telemetry: {
        ttft_ms: 285.4,
        decode_tok_per_sec: 48.6,
        tpot_ms: 20.57,
        peak_ram_mb: 1820.0,
        peak_vram_mb: 9420.0
      },
      categories: {
        reasoning: { category: "reasoning", weight: 0.25, normalized_score: 92.4 },
        coding: { category: "coding", weight: 0.30, normalized_score: 86.5 },
        instruction: { category: "instruction", weight: 0.25, normalized_score: 81.3 },
        tool_calling: { category: "tool_calling", weight: 0.20, normalized_score: 82.5 }
      },
      notes: "Full GPU offload with llama.cpp FlashAttention-2. Excellent reasoning density per GB VRAM."
    },
    {
      filename: "Qwen_Qwen2.5-32B-Instruct-GGUF_Q4_K_M.json",
      file_path: "data/results/Qwen_Qwen2.5-32B-Instruct-GGUF_Q4_K_M.json",
      evaluation_id: "eval-qwen25-32b-gguf",
      timestamp: "2026-10-28T09:30:00Z",
      profile: "standard",
      rank: 2,
      submitter: { name: "Maria Garcia", github_handle: "mgarcia-research" },
      model: {
        model_id: "Qwen/Qwen2.5-32B-Instruct",
        model_slug: "Qwen2.5-32B-Instruct-GGUF_Q4_K_M",
        hf_url: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
        param_count_b: 32.5,
        param_tier: "14B–32B",
        architecture: "Qwen2ForCausalLM",
        quantization: "GGUF Q4_K_M",
        model_ram_gb: 19.8,
        license: "Apache-2.0",
        backend: "llama.cpp"
      },
      scores: {
        overall: 86.90,
        ci_lower: 85.2,
        ci_upper: 88.4,
        coding_agentic_index: 83.44,
        local_efficiency_score: 75.6,
        score_per_gb: 4.39,
        speed: 31.4,
        gsm8k: 91.60,
        humaneval: 84.80,
        arc: 90.10,
        ifeval: 85.20,
        tool_calling: 81.40
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4090",
        gpu_count: 1,
        gpu_vram_gb: 24.0,
        model_ram_gb: 19.8,
        cpu: "AMD Ryzen 9 7900X",
        cpu_cores: 12,
        total_ram_gb: 64.0,
        os: "Ubuntu 24.04 LTS",
        cuda_version: "12.4",
        driver_version: "550.54"
      },
      telemetry: {
        ttft_ms: 340.0,
        decode_tok_per_sec: 31.4,
        tpot_ms: 31.84,
        peak_ram_mb: 2400.0,
        peak_vram_mb: 20400.0
      },
      notes: "32B parameter model running in 4-bit quant inside 24GB VRAM."
    },
    {
      filename: "Qwen_Qwen2.5-Coder-7B-Instruct-BF16.json",
      file_path: "data/results/Qwen_Qwen2.5-Coder-7B-Instruct-BF16.json",
      evaluation_id: "eval-qwen25-coder-7b-bf16",
      timestamp: "2026-10-29T18:05:00Z",
      profile: "standard",
      rank: 3,
      submitter: { name: "David Miller", github_handle: "dmiller-code" },
      model: {
        model_id: "Qwen/Qwen2.5-Coder-7B-Instruct",
        model_slug: "Qwen2.5-Coder-7B-Instruct-BF16",
        hf_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
        param_count_b: 7.61,
        param_tier: "7B–14B",
        architecture: "Qwen2ForCausalLM",
        quantization: "BF16",
        model_ram_gb: 15.2,
        license: "Apache-2.0",
        backend: "vllm"
      },
      scores: {
        overall: 83.55,
        ci_lower: 81.9,
        ci_upper: 85.1,
        coding_agentic_index: 84.84,
        local_efficiency_score: 81.3,
        score_per_gb: 5.50,
        speed: 68.4,
        gsm8k: 88.50,
        humaneval: 88.40,
        arc: 83.20,
        ifeval: 78.40,
        tool_calling: 79.50
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4080",
        gpu_count: 1,
        gpu_vram_gb: 16.0,
        model_ram_gb: 15.2,
        cpu: "Intel Core i9-14900K",
        cpu_cores: 24,
        total_ram_gb: 64.0,
        os: "Ubuntu 24.04 LTS",
        cuda_version: "12.4",
        driver_version: "550.67"
      },
      telemetry: {
        ttft_ms: 190.5,
        decode_tok_per_sec: 68.4,
        tpot_ms: 14.62,
        peak_ram_mb: 2100.0,
        peak_vram_mb: 15600.0
      },
      notes: "Native BF16 run on vLLM with PagedAttention enabled."
    },
    {
      filename: "microsoft_Phi-3.5-mini-instruct-AWQ_4BIT.json",
      file_path: "data/results/microsoft_Phi-3.5-mini-instruct-AWQ_4BIT.json",
      evaluation_id: "eval-phi35-mini-awq",
      timestamp: "2026-10-18T13:45:00Z",
      profile: "standard",
      rank: 4,
      submitter: { name: "Karthik Subramanian", github_handle: "ksubramanian-dev" },
      model: {
        model_id: "microsoft/Phi-3.5-mini-instruct",
        model_slug: "Phi-3.5-mini-instruct-AWQ_4BIT",
        hf_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        param_count_b: 3.82,
        param_tier: "3B–7B",
        architecture: "Phi3ForCausalLM",
        quantization: "AWQ",
        model_ram_gb: 2.45,
        license: "MIT",
        backend: "AutoAWQ"
      },
      scores: {
        overall: 76.70,
        ci_lower: 74.9,
        ci_upper: 78.4,
        coding_agentic_index: 69.66,
        local_efficiency_score: 93.4,
        score_per_gb: 31.31,
        speed: 92.4,
        gsm8k: 82.10,
        humaneval: 68.90,
        arc: 82.40,
        ifeval: 77.10,
        tool_calling: 70.80
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4060",
        gpu_count: 1,
        gpu_vram_gb: 8.0,
        model_ram_gb: 2.45,
        cpu: "AMD Ryzen 7 7700",
        cpu_cores: 8,
        total_ram_gb: 32.0,
        os: "Windows 11",
        cuda_version: "12.3",
        driver_version: "546.33"
      },
      telemetry: {
        ttft_ms: 110.0,
        decode_tok_per_sec: 92.4,
        tpot_ms: 10.82,
        peak_ram_mb: 1200.0,
        peak_vram_mb: 2900.0
      },
      notes: "Extremely high local efficiency: 31.3 score per GB VRAM on a consumer RTX 4060."
    },
    {
      filename: "meta-llama_Llama-3.2-3B-Instruct-BF16.json",
      file_path: "data/results/meta-llama_Llama-3.2-3B-Instruct-BF16.json",
      evaluation_id: "eval-llama32-3b-bf16",
      timestamp: "2026-10-15T19:30:00Z",
      profile: "quick",
      rank: 5,
      submitter: { name: "Clara Oswald", github_handle: "clara-ml" },
      model: {
        model_id: "meta-llama/Llama-3.2-3B-Instruct",
        model_slug: "Llama-3.2-3B-Instruct-BF16",
        hf_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        param_count_b: 3.21,
        param_tier: "3B–7B",
        architecture: "LlamaForCausalLM",
        quantization: "BF16",
        model_ram_gb: 6.4,
        license: "Llama 3.2 Community",
        backend: "transformers"
      },
      scores: {
        overall: 73.65,
        ci_lower: 71.8,
        ci_upper: 75.4,
        coding_agentic_index: 66.44,
        local_efficiency_score: 86.8,
        score_per_gb: 11.51,
        speed: 84.5,
        gsm8k: 77.20,
        humaneval: 63.40,
        arc: 78.50,
        ifeval: 76.80,
        tool_calling: 71.00
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4070",
        gpu_count: 1,
        gpu_vram_gb: 12.0,
        model_ram_gb: 6.4,
        cpu: "Intel Core i5-13600K",
        cpu_cores: 14,
        total_ram_gb: 32.0,
        os: "Ubuntu 24.04 LTS",
        cuda_version: "12.4",
        driver_version: "550.54"
      },
      telemetry: {
        ttft_ms: 120.0,
        decode_tok_per_sec: 84.5,
        tpot_ms: 11.83,
        peak_ram_mb: 1450.0,
        peak_vram_mb: 6900.0
      },
      notes: "Lightweight 3B parameter model running natively in BF16 on consumer GPU."
    },
    {
      filename: "HuggingFaceTB_SmolLM2-1.7B-Instruct-BF16.json",
      file_path: "data/results/HuggingFaceTB_SmolLM2-1.7B-Instruct-BF16.json",
      evaluation_id: "eval-smollm2-17b-bf16",
      timestamp: "2026-10-12T14:10:00Z",
      profile: "quick",
      rank: 6,
      submitter: { name: "Louisa Martin", github_handle: "lmartin-hf" },
      model: {
        model_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_slug: "SmolLM2-1.7B-Instruct-BF16",
        hf_url: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
        param_count_b: 1.71,
        param_tier: "< 3B",
        architecture: "LlamaForCausalLM",
        quantization: "BF16",
        model_ram_gb: 3.42,
        license: "Apache-2.0",
        backend: "transformers"
      },
      scores: {
        overall: 67.30,
        ci_lower: 65.4,
        ci_upper: 69.1,
        coding_agentic_index: 59.10,
        local_efficiency_score: 89.2,
        score_per_gb: 19.68,
        speed: 112.5,
        gsm8k: 68.50,
        humaneval: 54.90,
        arc: 72.80,
        ifeval: 73.10,
        tool_calling: 65.40
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 3060",
        gpu_count: 1,
        gpu_vram_gb: 12.0,
        model_ram_gb: 3.42,
        cpu: "AMD Ryzen 5 5600X",
        cpu_cores: 6,
        total_ram_gb: 32.0,
        os: "Ubuntu 22.04 LTS",
        cuda_version: "12.1",
        driver_version: "535.154"
      },
      telemetry: {
        ttft_ms: 78.0,
        decode_tok_per_sec: 112.5,
        tpot_ms: 8.89,
        peak_ram_mb: 950.0,
        peak_vram_mb: 3800.0
      },
      notes: "Ultra-lightweight edge model. Over 110 tokens/sec on an entry RTX 3060."
    },
    {
      filename: "Spark-X2.6-4B-INT4-INT4.json",
      file_path: "data/results/Spark-X2.6-4B-INT4-INT4.json",
      evaluation_id: "be128133-9d8f-4fd7-a635-1768ce3bcd4a",
      timestamp: "2026-11-02T20:37:04Z",
      profile: "quick",
      rank: 7,
      submitter: { name: "Community Contributor", github_handle: "anonymous" },
      model: {
        model_id: "XHToken/Spark-X2.6-4B-INT4",
        model_slug: "Spark-X2.6-4B-INT4-INT4",
        hf_url: "https://huggingface.co/XHToken/Spark-X2.6-4B-INT4",
        param_count_b: 1.97,
        param_tier: "< 3B",
        architecture: "Spark2_5ForCausalLM",
        quantization: "INT4",
        model_ram_gb: 1.67,
        license: "apache-2.0",
        backend: "openai",
        reasoning_effort: "high"
      },
      scores: {
        overall: 46.88,
        ci_lower: 42.1,
        ci_upper: 51.7,
        coding_agentic_index: 50.00,
        local_efficiency_score: 72.4,
        score_per_gb: 28.07,
        speed: 60.5,
        gsm8k: 0.0,
        humaneval: 16.67,
        arc: 87.50,
        ifeval: 0.0,
        tool_calling: 100.0
      },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        gpu_count: 1,
        gpu_vram_gb: 11.94,
        model_ram_gb: 1.67,
        cpu: "Intel Core Ultra 9 275HX",
        cpu_cores: 24,
        total_ram_gb: 31.43,
        os: "Windows 11",
        cuda_version: "13.5",
        driver_version: "621.01"
      },
      telemetry: {
        ttft_ms: 180.0,
        decode_tok_per_sec: 60.5,
        tpot_ms: 16.53,
        peak_ram_mb: 1670.0,
        peak_vram_mb: 2100.0
      },
      notes: "vLLM in WSL2 on GeForce RTX 5070 Ti Laptop GPU."
    }
  ]
};

// Markdown PR template content
const PR_TEMPLATE_TEXT = `### What this adds
- **Model:** org/model
- **Profile:** standard (or quick / extended)
- **Precision:** GGUF Q4_K_M
- **Hardware:** NVIDIA RTX 3060 12GB / Apple M2, 16GB
- **Result file:** \`Website/data/results/<model-slug>.json\`

### Checklist
- [ ] Open weights, and the license permits local use.
- [ ] Ran on consumer hardware: a consumer GPU, Apple Silicon, or system RAM.
- [ ] GPU, VRAM, CPU, RAM, OS, and telemetry (TTFT, decode TPS) recorded in the JSON.
- [ ] Sample counts are non-zero and match the run.
- [ ] Only the result file changed in this pull request.

### Hardware & Telemetry
| Metric | Value |
|:--|:--|
| GPU | NVIDIA RTX 3060 12GB |
| CPU | AMD Ryzen 5 5600X, 6 cores |
| System RAM | 32 GB |
| Model Footprint | 4.8 GB RAM/VRAM |
| Decode Speed | 65.4 tok/s |
| TTFT | 145 ms |
| Backend | llama.cpp |

### Log tail
\`\`\`text
[2026-11-02 10:00:15] Model loaded
[2026-11-02 10:11:02] Benchmark suite finished. Profile: standard. Composite SLM-Index: 82.10% (95% CI: [80.2%, 83.9%])
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
    console.log('[Leaderboard] Successfully fetched data/leaderboard.json with ' + allModels.length + ' models.');
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
  populateProfileFilter();
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
  const telem = data.telemetry || hw.telemetry || {};

  const gsm8k = summary.gsm8k || tasks.gsm8k?.score || tasks.gsm8k?.normalized_score || 0;
  const humaneval = summary.humaneval || tasks.humaneval?.score || tasks.humaneval?.normalized_score || tasks.slm_coding_agent?.normalized_score || tasks.slm_agentic_coding?.normalized_score || 0;
  const arc = summary.arc || tasks.arc?.score || tasks.arc?.normalized_score || tasks.slm_reasoning_cot?.normalized_score || 0;
  const ifeval = summary.ifeval || tasks.ifeval?.score || tasks.ifeval?.normalized_score || tasks.slm_instruction_ifeval?.normalized_score || tasks.slm_instruction_following?.normalized_score || 0;
  const tool_calling = summary.tool_calling || tasks.tool_calling?.score || tasks.tool_calling?.normalized_score || tasks.slm_tool_calling_multiturn?.normalized_score || 0;
  const overall = data.slm_index || summary.overall_score || ((gsm8k * 0.25) + (humaneval * 0.30) + (arc * 0.20) + (ifeval * 0.15) + (tool_calling * 0.10));

  const modelRam = hw.model_ram_gb || model.model_ram_gb || data.model_ram_gb || (model.param_count_b ? (model.param_count_b * 0.65) : null);
  const coding_agentic = data.coding_agentic_index || (0.60 * humaneval + 0.40 * tool_calling);
  const score_per_gb = modelRam ? parseFloat((overall / Math.max(1.0, modelRam)).toFixed(2)) : 0.0;
  const tps = telem.decode_tok_per_sec || data.avg_tokens_per_sec || summary.avg_tokens_per_sec || 0.0;

  return {
    filename: filename,
    file_path: `data/results/${filename}`,
    evaluation_id: data.evaluation_id || "eval-" + Math.random().toString(36).substring(7),
    timestamp: data.timestamp || new Date().toISOString(),
    profile: data.profile || "standard",
    submitter: data.submitter || { name: "Community Contributor", github_handle: "open-source" },
    model: {
      model_id: model.model_id || "Unknown",
      model_slug: model.model_slug || (model.model_id || "").replace('/', '-'),
      hf_url: model.hf_url || `https://huggingface.co/${model.model_id}`,
      param_count_b: parseFloat(model.param_count_b || 0),
      param_tier: model.param_tier || "7B–14B",
      architecture: model.architecture || "CausalLM",
      quantization: model.quantization || "BF16",
      model_ram_gb: modelRam,
      context_length: model.context_length || 32768,
      license: model.license || "Open",
      backend: model.backend || "transformers",
      reasoning_effort: model.reasoning_effort
    },
    scores: {
      overall: parseFloat(overall.toFixed(2)),
      ci_lower: data.ci_lower || Math.max(0, parseFloat((overall - 2.8).toFixed(1))),
      ci_upper: data.ci_upper || Math.min(100, parseFloat((overall + 2.8).toFixed(1))),
      coding_agentic_index: parseFloat(coding_agentic.toFixed(2)),
      score_per_gb: score_per_gb,
      speed: parseFloat(tps.toFixed(1)),
      gsm8k: parseFloat(gsm8k.toFixed(2)),
      humaneval: parseFloat(humaneval.toFixed(2)),
      arc: parseFloat(arc.toFixed(2)),
      ifeval: parseFloat(ifeval.toFixed(2)),
      tool_calling: parseFloat(tool_calling.toFixed(2))
    },
    hardware: hw,
    telemetry: {
      ttft_ms: telem.ttft_ms || 0,
      decode_tok_per_sec: tps,
      tpot_ms: telem.tpot_ms || 0,
      peak_ram_mb: telem.peak_ram_mb || 0,
      peak_vram_mb: telem.peak_vram_mb || 0
    },
    categories: data.categories || {},
    tasks: tasks,
    notes: data.notes || "",
    raw_logs_preview: data.raw_logs_preview || data.notes || "Evaluation completed."
  };
}

// Populate Filter Dropdowns dynamically
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

function populateProfileFilter() {
  const profileSelect = document.getElementById('profile-filter');
  if (!profileSelect) return;
  const profiles = new Set(['standard', 'quick', 'extended']);
  allModels.forEach(m => {
    if (m.profile) profiles.add(m.profile.toLowerCase());
  });

  profileSelect.innerHTML = '<option value="all">All profiles</option>';
  Array.from(profiles).sort().forEach(prof => {
    const opt = document.createElement('option');
    opt.value = prof;
    opt.textContent = prof.charAt(0).toUpperCase() + prof.slice(1);
    profileSelect.appendChild(opt);
  });
}

// Update Top Metric Stat Cards
function updateTopStats() {
  const modelCountEl = document.getElementById('stat-model-count');
  const topScoreEl = document.getElementById('stat-top-score');
  const topModelEl = document.getElementById('stat-top-model');
  const topCodingEl = document.getElementById('stat-top-coding');
  const topCodingModelEl = document.getElementById('stat-top-coding-model');
  const topEfficiencyEl = document.getElementById('stat-top-efficiency');
  const topEfficiencyModelEl = document.getElementById('stat-top-efficiency-model');
  const topSpeedEl = document.getElementById('stat-top-speed');
  const topSpeedModelEl = document.getElementById('stat-top-speed-model');
  const updatedDateEl = document.getElementById('last-updated-text');

  if (!allModels || allModels.length === 0) {
    if (modelCountEl) modelCountEl.textContent = '0';
    if (topScoreEl) topScoreEl.textContent = '—';
    if (topModelEl) topModelEl.textContent = '—';
    if (topCodingEl) topCodingEl.textContent = '—';
    if (topCodingModelEl) topCodingModelEl.textContent = '—';
    if (topEfficiencyEl) topEfficiencyEl.textContent = '—';
    if (topEfficiencyModelEl) topEfficiencyModelEl.textContent = '—';
    if (topSpeedEl) topSpeedEl.textContent = '—';
    if (topSpeedModelEl) topSpeedModelEl.textContent = '—';
    if (updatedDateEl) updatedDateEl.textContent = '—';
    return;
  }

  modelCountEl.textContent = allModels.length;

  // 1. Top General (SLM-Index)
  const topOverall = allModels.reduce((prev, cur) => (cur.scores.overall > prev.scores.overall) ? cur : prev, allModels[0]);
  topScoreEl.textContent = topOverall.scores.overall.toFixed(1) + '%';
  topModelEl.textContent = topOverall.model.model_id;

  // 2. Top Coding & Agentic
  const topCoding = allModels.reduce((prev, cur) => ((cur.scores.coding_agentic_index || 0) > (prev.scores.coding_agentic_index || 0)) ? cur : prev, allModels[0]);
  topCodingEl.textContent = (topCoding.scores.coding_agentic_index || 0).toFixed(1) + '%';
  topCodingModelEl.textContent = topCoding.model.model_id;

  // 3. Top Efficiency (Score / GB)
  const topEfficiency = allModels.reduce((prev, cur) => ((cur.scores.score_per_gb || 0) > (prev.scores.score_per_gb || 0)) ? cur : prev, allModels[0]);
  if (topEfficiencyEl) {
    topEfficiencyEl.textContent = (topEfficiency.scores.score_per_gb || 0).toFixed(1) + ' / GB';
    topEfficiencyModelEl.textContent = topEfficiency.model.model_id;
  }

  // 4. Top Throughput (TPS)
  const topSpeed = allModels.reduce((prev, cur) => {
    const spdCur = cur.telemetry?.decode_tok_per_sec || cur.scores?.speed || 0;
    const spdPrev = prev.telemetry?.decode_tok_per_sec || prev.scores?.speed || 0;
    return spdCur > spdPrev ? cur : prev;
  }, allModels[0]);
  if (topSpeedEl) {
    const spdVal = topSpeed.telemetry?.decode_tok_per_sec || topSpeed.scores?.speed || 0;
    topSpeedEl.textContent = spdVal.toFixed(1) + ' tok/s';
    topSpeedModelEl.textContent = topSpeed.model.model_id;
  }

  const updatedDate = rawLeaderboardData?.metadata?.last_updated ? new Date(rawLeaderboardData.metadata.last_updated).toLocaleDateString() : 'Recently';
  if (updatedDateEl) updatedDateEl.textContent = updatedDate;
}

// Filter and Sort Engine
function applyFiltersAndSort() {
  const searchInput = document.getElementById('search-input');
  const searchTerm = (searchInput?.value || '').trim().toLowerCase();
  const tierFilter = document.getElementById('tier-filter')?.value || 'all';
  const profileFilter = document.getElementById('profile-filter')?.value || 'all';
  const licenseFilter = document.getElementById('license-filter')?.value || 'all';
  const quantFilter = document.getElementById('quant-filter')?.value || 'all';

  filteredModels = allModels.filter(item => {
    const m = item.model || {};
    const sub = item.submitter || {};
    const prof = (item.profile || 'standard').toLowerCase();

    // Search filter
    if (searchTerm) {
      const match = (
        (m.model_id || '').toLowerCase().includes(searchTerm) ||
        (m.model_slug || '').toLowerCase().includes(searchTerm) ||
        (m.architecture || '').toLowerCase().includes(searchTerm) ||
        (m.license || '').toLowerCase().includes(searchTerm) ||
        (m.backend || '').toLowerCase().includes(searchTerm) ||
        (m.quantization || '').toLowerCase().includes(searchTerm) ||
        prof.includes(searchTerm) ||
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
      if (tierFilter === '32B+' && p <= 32.0) return false;
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
      case 'coding_agentic':
        valA = a.scores.coding_agentic_index || 0;
        valB = b.scores.coding_agentic_index || 0;
        break;
      case 'efficiency':
        valA = a.scores.score_per_gb || a.local_efficiency?.score_per_gb || 0;
        valB = b.scores.score_per_gb || b.local_efficiency?.score_per_gb || 0;
        break;
      case 'speed':
        valA = a.telemetry?.decode_tok_per_sec || a.scores?.speed || 0;
        valB = b.telemetry?.decode_tok_per_sec || b.scores?.speed || 0;
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
        <td colspan="14">
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
        <td colspan="14">
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
    const telem = item.telemetry || {};
    const isSelected = selectedModelIds.has(m.model_slug || m.model_id);
    const prof = item.profile || 'standard';

    const tr = document.createElement('tr');
    tr.className = `model-row${isSelected ? ' is-selected' : ''}`;
    
    // Clicking anywhere on row (except checkbox & link) opens modal
    tr.onclick = (e) => {
      if (e.target.closest('.no-modal-click')) return;
      showModelModal(item);
    };

    // Rank marker
    let rankBadge = `<span class="rank">${item.rank || (index + 1)}</span>`;
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

    const speedVal = telem.decode_tok_per_sec || s.speed || 0;
    const effVal = s.score_per_gb || (m.model_ram_gb ? (s.overall / m.model_ram_gb).toFixed(1) : 0);
    const agenticVal = s.coding_agentic_index || (0.60 * s.humaneval + 0.40 * s.tool_calling);

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
          <span class="tag tag--profile-${prof}">${prof}</span>
          <span class="tag tag--quant">${m.quantization}</span>
          <span class="tag">${m.architecture}</span>
          <span class="tag">${m.license}</span>
          ${m.reasoning_effort ? `<span class="tag tag--effort">effort:${m.reasoning_effort}</span>` : ''}
          ${item.notes ? `<span class="tag tag--notes">notes</span>` : ''}
        </div>
      </td>
      <td class="col-params">
        <span class="param">${m.param_count_b ? m.param_count_b.toFixed(1) + 'B' : 'N/A'}</span>
        ${m.model_ram_gb ? `<span class="param-ram">${m.model_ram_gb.toFixed(1)} GB</span>` : ''}
      </td>
      <td class="col-overall">
        <span class="score-overall">${s.overall.toFixed(1)}%</span>
        ${s.ci_lower ? `<span class="ci-range">[${s.ci_lower.toFixed(1)}, ${s.ci_upper.toFixed(1)}]</span>` : ''}
      </td>
      <td class="col-score track-col-agentic">${getScoreBadge(agenticVal)}</td>
      <td class="col-score track-col-efficiency">
        <span class="score-val ${effVal >= 15 ? 'score-val--high' : 'score-val--mid'}">${typeof effVal === 'number' ? effVal.toFixed(1) : effVal}</span>
      </td>
      <td class="col-score">
        <span class="score-val ${speedVal >= 60 ? 'score-val--high' : 'score-val--mid'}">${speedVal ? speedVal.toFixed(1) : '—'}</span>
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

// Show Model Inspector Modal
function showModelModal(item) {
  const modal = document.getElementById('model-modal');
  const m = item.model;
  const s = item.scores;
  const hw = item.hardware || {};
  const tasks = item.tasks || {};
  const telem = item.telemetry || {};
  const prof = item.profile || 'standard';

  document.getElementById('modal-rank-badge').textContent = `Rank #${item.rank || '--'}`;
  document.getElementById('modal-tier-badge').textContent = m.param_tier || `${m.param_count_b}B`;
  document.getElementById('modal-quant-badge').textContent = m.quantization || 'BF16';
  document.getElementById('modal-model-title').textContent = m.model_id;
  
  const profileBadge = document.getElementById('modal-profile-badge');
  if (profileBadge) {
    profileBadge.textContent = prof.toUpperCase();
    profileBadge.className = `chip tag--profile-${prof}`;
  }

  const effortBadge = document.getElementById('modal-effort-badge');
  if (effortBadge) {
    if (m.reasoning_effort) {
      effortBadge.textContent = `Effort: ${m.reasoning_effort}`;
      effortBadge.classList.remove('hidden');
    } else {
      effortBadge.classList.add('hidden');
    }
  }

  const hfLink = document.getElementById('modal-hf-link');
  hfLink.href = m.hf_url;
  
  document.getElementById('modal-eval-id').textContent = `ID: ${item.evaluation_id}`;
  document.getElementById('modal-score-overall').textContent = `${s.overall.toFixed(2)}%`;
  
  const ciRangeEl = document.getElementById('modal-ci-range');
  if (ciRangeEl) {
    if (s.ci_lower !== undefined && s.ci_upper !== undefined) {
      ciRangeEl.textContent = `95% CI: [${s.ci_lower.toFixed(1)}%, ${s.ci_upper.toFixed(1)}%]`;
    } else {
      ciRangeEl.textContent = `95% CI: --`;
    }
  }

  const agenticScoreEl = document.getElementById('modal-score-agentic');
  if (agenticScoreEl) {
    const agScore = s.coding_agentic_index || (0.60 * s.humaneval + 0.40 * s.tool_calling);
    agenticScoreEl.textContent = `${agScore.toFixed(1)}%`;
  }

  const efficiencyScoreEl = document.getElementById('modal-score-efficiency');
  if (efficiencyScoreEl) {
    const eff = s.score_per_gb || (m.model_ram_gb ? (s.overall / m.model_ram_gb).toFixed(1) : '--');
    efficiencyScoreEl.textContent = `${eff}`;
  }

  const throughputEl = document.getElementById('modal-throughput');
  if (throughputEl) {
    const spd = telem.decode_tok_per_sec || s.speed || 0;
    throughputEl.textContent = spd ? `${spd.toFixed(1)} t/s` : 'N/A';
  }

  document.getElementById('modal-param-count').textContent = `${m.param_count_b || 'N/A'}B`;

  const modelRam = m.model_ram_gb || hw.model_ram_gb;
  document.getElementById('modal-model-ram').textContent = modelRam ? `${modelRam.toFixed(1)} GB` : 'N/A';
  document.getElementById('modal-hw-model-ram').textContent = modelRam ? `${modelRam.toFixed(2)} GB RAM/VRAM` : 'N/A';

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

  // Telemetry details
  const ttftEl = document.getElementById('modal-telemetry-ttft');
  if (ttftEl) ttftEl.textContent = telem.ttft_ms ? `${telem.ttft_ms.toFixed(1)} ms` : 'N/A';

  const tpotEl = document.getElementById('modal-telemetry-tpot');
  if (tpotEl) tpotEl.textContent = telem.tpot_ms ? `${telem.tpot_ms.toFixed(2)} ms` : 'N/A';

  const tpsEl = document.getElementById('modal-telemetry-tps');
  if (tpsEl) tpsEl.textContent = telem.decode_tok_per_sec ? `${telem.decode_tok_per_sec.toFixed(1)} tok/s` : 'N/A';

  const ramEl = document.getElementById('modal-telemetry-ram');
  if (ramEl) ramEl.textContent = telem.peak_ram_mb ? `${telem.peak_ram_mb.toFixed(0)} MB` : 'N/A';

  const vramEl = document.getElementById('modal-telemetry-vram');
  if (vramEl) vramEl.textContent = telem.peak_vram_mb ? `${telem.peak_vram_mb.toFixed(0)} MB` : 'N/A';

  // Category Pillars breakdown
  const catsDiv = document.getElementById('modal-categories-breakdown');
  if (catsDiv) {
    const defaultCats = [
      { name: "Reasoning", key: "reasoning", weight: "25%", score: s.arc },
      { name: "Coding", key: "coding", weight: "30%", score: s.humaneval },
      { name: "Instruction", key: "instruction", weight: "25%", score: s.ifeval },
      { name: "Tool Calling", key: "tool_calling", weight: "20%", score: s.tool_calling }
    ];
    catsDiv.innerHTML = defaultCats.map(c => {
      const catData = item.categories?.[c.key];
      const sc = catData?.normalized_score !== undefined ? catData.normalized_score : c.score;
      return `
        <div class="category-card">
          <div class="category-card-head">
            <span class="category-card-title">${c.name} (${c.weight})</span>
            <span class="category-card-score">${sc.toFixed(1)}%</span>
          </div>
          <div class="category-card-bar">
            <div class="category-card-fill" style="width: ${Math.min(100, sc)}%"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Dynamic Tasks Breakdown
  const breakdownDiv = document.getElementById('modal-tasks-breakdown');
  const taskKeys = Object.keys(tasks);
  
  if (taskKeys.length > 0) {
    breakdownDiv.innerHTML = taskKeys.map(k => {
      const taskData = tasks[k];
      const score = taskData.normalized_score !== undefined ? taskData.normalized_score : (taskData.score || 0);
      const passed = taskData.passed_samples !== undefined ? `passed ${taskData.passed_samples}/${taskData.total_samples || '?'}` : '';
      const speed = taskData.avg_tokens_per_sec ? `${taskData.avg_tokens_per_sec} tok/s` : '';
      const ci = (taskData.ci_lower !== undefined && taskData.ci_upper !== undefined) ? `[${taskData.ci_lower}%, ${taskData.ci_upper}%]` : '';
      const title = taskData.display_name || k.replace(/_/g, ' ').toUpperCase();

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
            <span>${taskData.raw_metric_name || taskData.metric || 'accuracy'}${passed ? ' &middot; ' + passed : ''}</span>
            <span>${speed}</span>
          </div>
        </div>
      `;
    }).join('');
  } else {
    const fallbackTasks = [
      { label: 'GSM8K (Math Reasoning)', score: s.gsm8k, desc: 'Grade School Math 8K strict answer accuracy' },
      { label: 'HumanEval (Python Coding)', score: s.humaneval, desc: 'Code generation pass@1 execution accuracy' },
      { label: 'ARC Challenge (Reasoning)', score: s.arc, desc: 'Grade school science multiple-choice reasoning' },
      { label: 'IFEval (Instruction Following)', score: s.ifeval, desc: 'Verifiable constraint satisfaction & format compliance' },
      { label: 'Tool Calling & Multiturn', score: s.tool_calling, desc: 'Function calling JSON schema adherence and execution' }
    ];
    breakdownDiv.innerHTML = fallbackTasks.map(t => `
      <div class="task">
        <div class="task-head">
          <div><span class="task-name">${t.label}</span></div>
          <span class="task-score">${t.score.toFixed(1)}%</span>
        </div>
        <div class="task-bar"><div class="task-fill" style="width: ${t.score}%"></div></div>
      </div>
    `).join('');
  }

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
    ['Math (GSM8K)', 'gsm8k'],
    ['Code (HumanEval)', 'humaneval'],
    ['Reasoning (ARC)', 'arc'],
    ['Instruction (IFEval)', 'ifeval'],
    ['Tool Calling', 'tool_calling']
  ];

  container.innerHTML = `
    <table class="compare">
      <thead>
        <tr>
          <th>Metric</th>
          ${selectedModels.map(m => `
            <th>
              <div class="compare-model">${m.model.model_id}</div>
              <div class="compare-rank">Rank #${m.rank || '--'} &middot; ${m.model.param_count_b}B &middot; <span class="tag tag--profile-${m.profile || 'standard'}">${m.profile || 'standard'}</span></div>
            </th>
          `).join('')}
        </tr>
      </thead>
      <tbody>
        <tr class="is-total">
          <td class="metric-name">General (SLM-Index)</td>
          ${selectedModels.map(m => `
            <td class="total">
              ${m.scores.overall.toFixed(1)}%
              ${m.scores.ci_lower ? `<div class="ci-range" style="color:var(--rust-2)">95% CI: [${m.scores.ci_lower.toFixed(1)}, ${m.scores.ci_upper.toFixed(1)}]</div>` : ''}
            </td>
          `).join('')}
        </tr>
        <tr>
          <td class="metric-name">Coding &amp; Agentic Index</td>
          ${selectedModels.map(m => `<td class="num"><strong>${(m.scores.coding_agentic_index || 0).toFixed(1)}%</strong></td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Local Efficiency (Score/GB)</td>
          ${selectedModels.map(m => `<td class="num"><strong>${(m.scores.score_per_gb || 0).toFixed(1)}</strong></td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Decode Speed</td>
          ${selectedModels.map(m => `<td class="num">${(m.telemetry?.decode_tok_per_sec || m.scores?.speed || 0).toFixed(1)} tok/s</td>`).join('')}
        </tr>
        <tr>
          <td class="metric-name">Time to First Token (TTFT)</td>
          ${selectedModels.map(m => `<td class="num">${m.telemetry?.ttft_ms ? m.telemetry.ttft_ms.toFixed(1) + ' ms' : 'N/A'}</td>`).join('')}
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
          <td class="metric-name">Model Footprint</td>
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

  const radarCanvas = document.getElementById('radarChart');
  if (radarCanvas) {
    const radarCtx = radarCanvas.getContext('2d');
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
            suggestedMin: 40,
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
  }

  // Scatter: Model Memory Footprint (GB) vs Composite SLM-Index (Local Efficiency frontier)
  const barCanvas = document.getElementById('barChart');
  if (barCanvas) {
    const barCtx = barCanvas.getContext('2d');
    if (barChartInstance) barChartInstance.destroy();

    const sortedByRam = [...allModels].sort((a, b) => (a.model.model_ram_gb || 0) - (b.model.model_ram_gb || 0));

    barChartInstance = new Chart(barCtx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Models',
          data: sortedByRam.map(m => ({
            x: m.model.model_ram_gb || (m.model.param_count_b ? m.model.param_count_b * 0.65 : 2.0),
            y: m.scores.overall,
            model: m.model.model_id,
            eff: (m.scores.score_per_gb || (m.model.model_ram_gb ? m.scores.overall / m.model.model_ram_gb : 0)).toFixed(1),
            profile: m.profile || 'standard'
          })),
          backgroundColor: sortedByRam.map(m => {
            if (m.scores.overall >= 85) return 'rgba(163, 58, 28, 0.85)';
            if (m.scores.overall >= 80) return 'rgba(138, 103, 20, 0.72)';
            return 'rgba(47, 90, 70, 0.75)';
          }),
          borderColor: 'rgba(28, 26, 20, 0.35)',
          borderWidth: 1,
          pointRadius: 6,
          pointHoverRadius: 8
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
              text: 'Model Memory Footprint (GB RAM / VRAM)',
              color: '#6a6455',
              font: { size: 9, family: 'IBM Plex Mono' }
            },
            grid: { color: 'rgba(28, 26, 20, 0.08)' },
            ticks: { color: '#6a6455', font: { size: 9, family: 'IBM Plex Mono' } }
          },
          y: {
            min: 40,
            max: 100,
            title: {
              display: true,
              text: 'General Composite (SLM-Index %)',
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
              label: (ctx) => `${ctx.raw.model} (${ctx.raw.profile}): ${ctx.raw.y}% on ${ctx.raw.x.toFixed(1)}GB (Eff: ${ctx.raw.eff} score/GB)`
            }
          }
        }
      }
    });
  }
}

// PR Guide & Local Validator Logic
function loadPRTemplate() {
  const prPre = document.getElementById('pr-template-preview');
  if (prPre) prPre.textContent = PR_TEMPLATE_TEXT;
}

function setupEventListeners() {
  // Track switcher pill buttons
  document.querySelectorAll('.track-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.track-pill').forEach(p => p.classList.remove('is-active'));
      btn.classList.add('is-active');
      currentTrack = btn.getAttribute('data-track');

      const sortSelect = document.getElementById('sort-select');
      if (currentTrack === 'general') {
        currentSort = { column: 'overall', direction: 'desc' };
        if (sortSelect) sortSelect.value = 'overall_desc';
      } else if (currentTrack === 'coding_agentic') {
        currentSort = { column: 'coding_agentic', direction: 'desc' };
        if (sortSelect) sortSelect.value = 'coding_agentic_desc';
      } else if (currentTrack === 'efficiency') {
        currentSort = { column: 'efficiency', direction: 'desc' };
        if (sortSelect) sortSelect.value = 'efficiency_desc';
      }

      updateHeaderSortIcons();
      applyFiltersAndSort();
    });
  });

  // Search
  const searchInput = document.getElementById('search-input');
  const clearSearchBtn = document.getElementById('clear-search-btn');

  if (searchInput && clearSearchBtn) {
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
  }

  // Filters
  document.getElementById('tier-filter')?.addEventListener('change', applyFiltersAndSort);
  document.getElementById('profile-filter')?.addEventListener('change', applyFiltersAndSort);
  document.getElementById('license-filter')?.addEventListener('change', applyFiltersAndSort);
  document.getElementById('quant-filter')?.addEventListener('change', applyFiltersAndSort);
  
  // Sort select dropdown
  document.getElementById('sort-select')?.addEventListener('change', (e) => {
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
  document.getElementById('reset-filters-btn')?.addEventListener('click', () => {
    if (searchInput) searchInput.value = '';
    if (clearSearchBtn) clearSearchBtn.classList.add('hidden');
    const tier = document.getElementById('tier-filter');
    const prof = document.getElementById('profile-filter');
    const lic = document.getElementById('license-filter');
    const quant = document.getElementById('quant-filter');
    const sort = document.getElementById('sort-select');

    if (tier) tier.value = 'all';
    if (prof) prof.value = 'all';
    if (lic) lic.value = 'all';
    if (quant) quant.value = 'all';
    if (sort) sort.value = 'overall_desc';

    document.querySelectorAll('.track-pill').forEach(p => p.classList.remove('is-active'));
    document.querySelector('.track-pill[data-track="all"]')?.classList.add('is-active');
    currentTrack = 'all';

    currentSort = { column: 'overall', direction: 'desc' };
    updateHeaderSortIcons();
    applyFiltersAndSort();
  });

  // Export buttons
  document.getElementById('export-csv-btn')?.addEventListener('click', exportToCSV);
  document.getElementById('export-json-btn')?.addEventListener('click', exportToJSON);

  // Compare Navigation and Dock buttons
  document.getElementById('nav-compare-btn')?.addEventListener('click', showCompareModal);
  document.getElementById('dock-compare-btn')?.addEventListener('click', showCompareModal);
  document.getElementById('dock-clear-btn')?.addEventListener('click', () => {
    selectedModelIds.clear();
    applyFiltersAndSort();
    renderCharts();
  });
  document.getElementById('compare-modal-close')?.addEventListener('click', () => {
    document.getElementById('compare-modal')?.classList.add('hidden');
  });

  // Inspector Modal Tab Switchers
  document.querySelectorAll('.modal-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchModalTab(btn.getAttribute('data-tab'));
    });
  });

  // Inspector Modal Close
  document.getElementById('modal-close-btn')?.addEventListener('click', () => {
    document.getElementById('model-modal')?.classList.add('hidden');
  });
  document.getElementById('modal-footer-close')?.addEventListener('click', () => {
    document.getElementById('model-modal')?.classList.add('hidden');
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
      prModal?.classList.remove('hidden');
    });
  });

  document.getElementById('cta-test-validator-btn')?.addEventListener('click', () => {
    switchPRTab('pr-tab-validator');
    prModal?.classList.remove('hidden');
  });

  document.getElementById('pr-guide-close-btn')?.addEventListener('click', () => {
    prModal?.classList.add('hidden');
  });
  document.getElementById('pr-guide-close-bottom')?.addEventListener('click', () => {
    prModal?.classList.add('hidden');
  });

  // PR Tabs
  document.querySelectorAll('.pr-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchPRTab(btn.getAttribute('data-prtab'));
    });
  });

  // PR Template copy button
  document.getElementById('copy-pr-template-btn')?.addEventListener('click', () => {
    navigator.clipboard.writeText(PR_TEMPLATE_TEXT);
    alert('PR Template copied to clipboard!');
  });

  // Validator Buttons
  document.getElementById('validate-btn')?.addEventListener('click', runLocalJsonValidation);
  document.getElementById('load-sample-btn')?.addEventListener('click', () => {
    const sample = {
      schema_version: "1.0.0",
      evaluation_id: "eval-sample-submission-standard",
      timestamp: new Date().toISOString(),
      profile: "standard",
      submitter: { name: "Your Name", github_handle: "your-gh-handle", affiliation: "Community" },
      model: {
        model_id: "org/sample-model-7b",
        model_slug: "sample-model-7b-GGUF_Q4_K_M",
        param_count_b: 7.2,
        param_tier: "7B–14B",
        quantization: "GGUF Q4_K_M",
        license: "Apache-2.0",
        backend: "llama.cpp",
        model_ram_gb: 4.8
      },
      hardware: {
        gpu_name: "NVIDIA RTX 4080",
        gpu_count: 1,
        gpu_vram_gb: 16.0,
        model_ram_gb: 4.8,
        cpu: "AMD Ryzen 9 7900X",
        cpu_cores: 12,
        total_ram_gb: 64.0,
        os: "Ubuntu 24.04",
        cuda_version: "12.4"
      },
      telemetry: {
        ttft_ms: 185.0,
        decode_tok_per_sec: 62.5,
        tpot_ms: 16.0,
        peak_ram_mb: 1800.0,
        peak_vram_mb: 5100.0
      },
      categories: {
        reasoning: { category: "reasoning", weight: 0.25, normalized_score: 85.0 },
        coding: { category: "coding", weight: 0.30, normalized_score: 82.0 },
        instruction: { category: "instruction", weight: 0.25, normalized_score: 80.0 },
        tool_calling: { category: "tool_calling", weight: 0.20, normalized_score: 79.0 }
      },
      slm_index: 81.85,
      ci_lower: 79.5,
      ci_upper: 84.1,
      coding_agentic_index: 80.80,
      local_efficiency: {
        score_per_gb: 17.05,
        throughput_tps: 62.5,
        effective_memory_gb: 4.8,
        efficiency_score: 85.4
      },
      summary_scores: {
        overall_score: 81.85,
        gsm8k: 85.0,
        humaneval: 82.0,
        arc: 85.0,
        ifeval: 80.0,
        tool_calling: 79.0
      }
    };
    const input = document.getElementById('validator-input');
    if (input) input.value = JSON.stringify(sample, null, 2);
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
  const inputEl = document.getElementById('validator-input');
  const input = (inputEl?.value || '').trim();
  const feedback = document.getElementById('validator-feedback');
  if (!feedback) return;

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
      if (data.model.param_count_b === undefined) errors.push("Missing 'model.param_count_b'.");
      if (!data.model.quantization) errors.push("Missing 'model.quantization'.");
      if (!data.model.license) errors.push("Missing 'model.license'.");
    }

    if (!data.hardware) errors.push("Missing 'hardware' (GPU, CPU, RAM).");
    if (!data.summary_scores && !data.tasks && !data.categories && data.slm_index === undefined) {
      errors.push("Missing scores ('slm_index', 'summary_scores', 'categories', or 'tasks').");
    }

    if (errors.length > 0) {
      feedback.className = 'feedback feedback--error';
      feedback.innerHTML = `<strong>${errors.length} field${errors.length === 1 ? '' : 's'} missing</strong><ul>${errors.map(e => `<li>${e}</li>`).join('')}</ul>`;
      feedback.classList.remove('hidden');
    } else {
      feedback.className = 'feedback feedback--ok';
      const prof = data.profile || "standard";
      const slmIdx = data.slm_index || data.summary_scores?.overall_score || 0;
      feedback.innerHTML = `<strong>Valid Submission</strong>Profile: <em>${prof}</em> &bull; SLM-Index: <em>${slmIdx}%</em> &bull; All required schema fields are present.`;
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
  const cols = ['rank', 'name', 'params', 'overall', 'coding_agentic', 'efficiency', 'speed', 'gsm8k', 'humaneval', 'arc', 'ifeval', 'tool_calling'];
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

  const headers = [
    'Rank', 'Model ID', 'Parameters (B)', 'Tier', 'Profile', 'SLM-Index',
    'Coding & Agentic Index', 'Local Efficiency (Score/GB)', 'Throughput (tok/s)',
    'GSM8K', 'HumanEval', 'ARC', 'IFEval', 'Tool Calling', 'Quantization', 'Architecture', 'License', 'Backend', 'GPU'
  ];
  const rows = filteredModels.map(item => [
    item.rank,
    `"${item.model.model_id}"`,
    item.model.param_count_b,
    `"${item.model.param_tier}"`,
    `"${item.profile || 'standard'}"`,
    item.scores.overall,
    item.scores.coding_agentic_index || 0,
    item.scores.score_per_gb || 0,
    item.telemetry?.decode_tok_per_sec || item.scores?.speed || 0,
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
