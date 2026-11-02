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

// Embedded Fallback Data (ensures 100% offline & file:// protocol functionality)
const EMBEDDED_FALLBACK_DATA = {
  metadata: {
    total_models: 10,
    last_updated: "2026-11-02T16:52:55Z",
    highest_overall: 87.65,
    best_by_benchmark: { gsm8k: 94.2, humaneval: 88.4, arc: 92.4, ifeval: 85.2, tool_calling: 81.4 }
  },
  models: [
    {
      filename: "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json",
      file_path: "data/results/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json",
      evaluation_id: "eval-dsk-r1-14b-20261101",
      timestamp: "2026-11-01T14:22:10Z",
      submitter: { name: "Alex Chen", github_handle: "alexchen-ai", affiliation: "Independent ML Researcher" },
      model: {
        model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        model_slug: "DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M",
        hf_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        param_count_b: 14.7,
        param_tier: "7B–14B",
        architecture: "Qwen2ForCausalLM",
        quantization: "GGUF Q4_K_M",
        context_length: 65536,
        license: "MIT",
        backend: "llama.cpp"
      },
      scores: { overall: 87.65, gsm8k: 94.20, humaneval: 86.50, arc: 92.40, ifeval: 81.30, tool_calling: 79.80 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4090", gpu_count: 1, gpu_vram_gb: 24.0,
        cpu: "AMD Ryzen 9 7950X 16-Core Processor", cpu_cores: 16, total_ram_gb: 64.0,
        os: "Ubuntu 24.04 LTS (Linux 6.8.0-generic)", cuda_version: "12.4", driver_version: "550.54.14"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 94.2, total_samples: 1319, passed_samples: 1242, metric: "strict_acc", avg_tokens_per_sec: 38.6 },
        humaneval: { display_name: "HumanEval (Coding)", score: 86.5, total_samples: 164, passed_samples: 142, metric: "pass@1", avg_tokens_per_sec: 36.8 },
        arc: { display_name: "ARC Challenge", score: 92.4, total_samples: 1172, passed_samples: 1083, metric: "acc_norm", avg_tokens_per_sec: 41.2 },
        ifeval: { display_name: "IFEval (Instruction)", score: 81.3, total_samples: 541, passed_samples: 440, metric: "strict_acc", avg_tokens_per_sec: 39.5 },
        tool_calling: { display_name: "Tool Calling", score: 79.8, total_samples: 300, passed_samples: 239, metric: "schema_acc", avg_tokens_per_sec: 35.1 }
      },
      raw_logs_preview: "[2026-11-01 14:05:00] Initialized llama.cpp backend with -ngl 99\n[2026-11-01 14:05:02] Model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B loaded (VRAM allocated: 9.8 GB)\n[2026-11-01 14:08:15] Task gsm8k completed: 1242/1319 (94.20%)\n[2026-11-01 14:12:40] Task humaneval completed: 142/164 (86.50%)\n[2026-11-01 14:16:11] Task arc completed: 1083/1172 (92.40%)\n[2026-11-01 14:19:05] Task ifeval completed: 440/541 (81.30%)\n[2026-11-01 14:22:08] Task tool_calling completed: 239/300 (79.80%)\n[2026-11-01 14:22:10] Benchmark suite finished. Overall Score: 87.65%",
      rank: 1
    },
    {
      filename: "Qwen_Qwen2.5-32B-Instruct-GGUF_Q4_K_M.json",
      file_path: "data/results/Qwen_Qwen2.5-32B-Instruct-GGUF_Q4_K_M.json",
      evaluation_id: "eval-qwen25-32b-20261028",
      timestamp: "2026-10-28T09:15:33Z",
      submitter: { name: "Sarah Lin", github_handle: "slin-openllm", affiliation: "Hugging Face Community" },
      model: {
        model_id: "Qwen/Qwen2.5-32B-Instruct",
        model_slug: "Qwen2.5-32B-Instruct-GGUF_Q4_K_M",
        hf_url: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
        param_count_b: 32.5,
        param_tier: "14B–32B",
        architecture: "Qwen2ForCausalLM",
        quantization: "GGUF Q4_K_M",
        context_length: 131072,
        license: "Apache-2.0",
        backend: "vLLM"
      },
      scores: { overall: 86.90, gsm8k: 91.60, humaneval: 84.80, arc: 90.10, ifeval: 85.20, tool_calling: 81.40 },
      hardware: {
        gpu_name: "NVIDIA RTX 6000 Ada Generation", gpu_count: 1, gpu_vram_gb: 48.0,
        cpu: "Intel Xeon w9-3495X 56-Cores", cpu_cores: 56, total_ram_gb: 128.0,
        os: "Red Hat Enterprise Linux 9.4", cuda_version: "12.3", driver_version: "545.23.08"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 91.6, total_samples: 1319, passed_samples: 1208 },
        humaneval: { display_name: "HumanEval (Coding)", score: 84.8, total_samples: 164, passed_samples: 139 },
        arc: { display_name: "ARC Challenge", score: 90.1, total_samples: 1172, passed_samples: 1056 },
        ifeval: { display_name: "IFEval (Instruction)", score: 85.2, total_samples: 541, passed_samples: 461 },
        tool_calling: { display_name: "Tool Calling", score: 81.4, total_samples: 300, passed_samples: 244 }
      },
      raw_logs_preview: "[2026-10-28 09:00:00] Initialized vLLM engine with Qwen2.5-32B-Instruct\n[2026-10-28 09:15:30] Full evaluation completed on 5 benchmarks. Average accuracy 86.90%.",
      rank: 2
    },
    {
      filename: "Qwen_Qwen2.5-Coder-7B-Instruct-BF16.json",
      file_path: "data/results/Qwen_Qwen2.5-Coder-7B-Instruct-BF16.json",
      evaluation_id: "eval-qwen25-coder-7b-bf16",
      timestamp: "2026-10-29T18:05:00Z",
      submitter: { name: "David Miller", github_handle: "dmiller-code", affiliation: "CodeAI Labs" },
      model: {
        model_id: "Qwen/Qwen2.5-Coder-7B-Instruct",
        model_slug: "Qwen2.5-Coder-7B-Instruct-BF16",
        hf_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
        param_count_b: 7.61,
        param_tier: "7B–14B",
        architecture: "Qwen2ForCausalLM",
        quantization: "BF16",
        context_length: 131072,
        license: "Apache-2.0",
        backend: "transformers"
      },
      scores: { overall: 83.55, gsm8k: 88.50, humaneval: 88.40, arc: 83.20, ifeval: 78.40, tool_calling: 76.50 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4080", gpu_count: 1, gpu_vram_gb: 16.0,
        cpu: "Intel Core i9-14900K", cpu_cores: 24, total_ram_gb: 64.0,
        os: "Ubuntu 24.04 LTS", cuda_version: "12.4", driver_version: "550.67"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 88.5, total_samples: 1319, passed_samples: 1167 },
        humaneval: { display_name: "HumanEval (Coding)", score: 88.4, total_samples: 164, passed_samples: 145 },
        arc: { display_name: "ARC Challenge", score: 83.2, total_samples: 1172, passed_samples: 975 },
        ifeval: { display_name: "IFEval (Instruction)", score: 78.4, total_samples: 541, passed_samples: 424 },
        tool_calling: { display_name: "Tool Calling", score: 76.5, total_samples: 300, passed_samples: 230 }
      },
      raw_logs_preview: "[2026-10-29 18:00:00] Evaluating Qwen2.5-Coder-7B in BF16 unquantized mode.\n[2026-10-29 18:04:50] HumanEval pass@1 reached 88.40% (state-of-the-art for 7B).",
      rank: 3
    },
    {
      filename: "Qwen_Qwen2.5-7B-Instruct-GGUF_Q4_K_M.json",
      file_path: "data/results/Qwen_Qwen2.5-7B-Instruct-GGUF_Q4_K_M.json",
      evaluation_id: "eval-qwen25-7b-gguf-q4",
      timestamp: "2026-11-02T10:11:00Z",
      submitter: { name: "Marcus Vance", github_handle: "marcus-vance", affiliation: "Open Weights Alliance" },
      model: {
        model_id: "Qwen/Qwen2.5-7B-Instruct",
        model_slug: "Qwen2.5-7B-Instruct-GGUF_Q4_K_M",
        hf_url: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        param_count_b: 7.61,
        param_tier: "7B–14B",
        architecture: "Qwen2ForCausalLM",
        quantization: "GGUF Q4_K_M",
        context_length: 32768,
        license: "Apache-2.0",
        backend: "llama.cpp"
      },
      scores: { overall: 82.10, gsm8k: 86.80, humaneval: 82.30, arc: 85.10, ifeval: 81.00, tool_calling: 72.80 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 3090", gpu_count: 1, gpu_vram_gb: 24.0,
        cpu: "AMD Ryzen 9 5900X", cpu_cores: 12, total_ram_gb: 32.0,
        os: "Windows 11 (10.0.26100)", cuda_version: "12.2", driver_version: "551.86"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 86.8, total_samples: 1319, passed_samples: 1145 },
        humaneval: { display_name: "HumanEval (Coding)", score: 82.3, total_samples: 164, passed_samples: 135 },
        arc: { display_name: "ARC Challenge", score: 85.1, total_samples: 1172, passed_samples: 997 },
        ifeval: { display_name: "IFEval (Instruction)", score: 81.0, total_samples: 541, passed_samples: 438 },
        tool_calling: { display_name: "Tool Calling", score: 72.8, total_samples: 300, passed_samples: 218 }
      },
      raw_logs_preview: "[2026-11-02 10:00:15] llama.cpp evaluation finished for Qwen2.5-7B GGUF Q4_K_M.",
      rank: 4
    },
    {
      filename: "google_gemma-2-9b-it-BF16.json",
      file_path: "data/results/google_gemma-2-9b-it-BF16.json",
      evaluation_id: "eval-gemma2-9b-it-bf16",
      timestamp: "2026-10-25T11:40:00Z",
      submitter: { name: "Elena Rostova", github_handle: "elena-ml", affiliation: "EuroLLM Working Group" },
      model: {
        model_id: "google/gemma-2-9b-it",
        model_slug: "gemma-2-9b-it-BF16",
        hf_url: "https://huggingface.co/google/gemma-2-9b-it",
        param_count_b: 9.24,
        param_tier: "7B–14B",
        architecture: "Gemma2ForCausalLM",
        quantization: "BF16",
        context_length: 8192,
        license: "Gemma Terms of Use",
        backend: "vLLM"
      },
      scores: { overall: 81.85, gsm8k: 84.20, humaneval: 74.40, arc: 88.60, ifeval: 84.10, tool_calling: 76.50 },
      hardware: {
        gpu_name: "NVIDIA A100-SXM4-80GB", gpu_count: 1, gpu_vram_gb: 80.0,
        cpu: "AMD EPYC 7742", cpu_cores: 32, total_ram_gb: 128.0,
        os: "Ubuntu 22.04 LTS", cuda_version: "12.4", driver_version: "550.54"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 84.2, total_samples: 1319, passed_samples: 1111 },
        humaneval: { display_name: "HumanEval (Coding)", score: 74.4, total_samples: 164, passed_samples: 122 },
        arc: { display_name: "ARC Challenge", score: 88.6, total_samples: 1172, passed_samples: 1038 },
        ifeval: { display_name: "IFEval (Instruction)", score: 84.1, total_samples: 541, passed_samples: 455 },
        tool_calling: { display_name: "Tool Calling", score: 76.5, total_samples: 300, passed_samples: 230 }
      },
      raw_logs_preview: "[2026-10-25 11:20:00] Executed vLLM benchmark on Google Gemma 2 9B IT in BF16 precision.",
      rank: 5
    },
    {
      filename: "meta-llama_Llama-3.1-8B-Instruct-GGUF_Q8_0.json",
      file_path: "data/results/meta-llama_Llama-3.1-8B-Instruct-GGUF_Q8_0.json",
      evaluation_id: "eval-llama31-8b-gguf-q8",
      timestamp: "2026-10-22T16:50:00Z",
      submitter: { name: "Tobias Huber", github_handle: "tobi-huber", affiliation: "LocalAI Initiative" },
      model: {
        model_id: "meta-llama/Llama-3.1-8B-Instruct",
        model_slug: "Llama-3.1-8B-Instruct-GGUF_Q8_0",
        hf_url: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        param_count_b: 8.03,
        param_tier: "7B–14B",
        architecture: "LlamaForCausalLM",
        quantization: "GGUF Q8_0",
        context_length: 131072,
        license: "Llama 3.1 Community",
        backend: "llama.cpp"
      },
      scores: { overall: 80.95, gsm8k: 84.50, humaneval: 72.60, arc: 85.40, ifeval: 83.20, tool_calling: 77.30 },
      hardware: {
        gpu_name: "Apple M3 Max", gpu_count: 1, gpu_vram_gb: 48.0,
        cpu: "Apple M3 Max (16-core)", cpu_cores: 16, total_ram_gb: 48.0,
        os: "macOS 15.1 Sequoia", cuda_version: "Metal 3", driver_version: "N/A"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 84.5, total_samples: 1319, passed_samples: 1115 },
        humaneval: { display_name: "HumanEval (Coding)", score: 72.6, total_samples: 164, passed_samples: 119 },
        arc: { display_name: "ARC Challenge", score: 85.4, total_samples: 1172, passed_samples: 1001 },
        ifeval: { display_name: "IFEval (Instruction)", score: 83.2, total_samples: 541, passed_samples: 450 },
        tool_calling: { display_name: "Tool Calling", score: 77.3, total_samples: 300, passed_samples: 232 }
      },
      raw_logs_preview: "[2026-10-22 16:30:10] Metal unified memory acceleration enabled. Run completed.",
      rank: 6
    },
    {
      filename: "mistralai_Mistral-Nemo-Instruct-2407-12B-NF4.json",
      file_path: "data/results/mistralai_Mistral-Nemo-Instruct-2407-12B-NF4.json",
      evaluation_id: "eval-mistral-nemo-12b-nf4",
      timestamp: "2026-10-20T08:12:00Z",
      submitter: { name: "Jean-Luc Picard", github_handle: "jlpicard-ai", affiliation: "French Open Source AI" },
      model: {
        model_id: "mistralai/Mistral-Nemo-Instruct-2407",
        model_slug: "Mistral-Nemo-Instruct-2407-12B-NF4",
        hf_url: "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
        param_count_b: 12.2,
        param_tier: "7B–14B",
        architecture: "MistralForCausalLM",
        quantization: "NF4",
        context_length: 128000,
        license: "Apache-2.0",
        backend: "bitsandbytes"
      },
      scores: { overall: 78.80, gsm8k: 78.40, humaneval: 71.30, arc: 83.90, ifeval: 82.50, tool_calling: 77.20 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4070 Ti SUPER", gpu_count: 1, gpu_vram_gb: 16.0,
        cpu: "Intel Core i7-13700K", cpu_cores: 16, total_ram_gb: 32.0,
        os: "Ubuntu 24.04 LTS", cuda_version: "12.4", driver_version: "550.54"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 78.4, total_samples: 1319, passed_samples: 1034 },
        humaneval: { display_name: "HumanEval (Coding)", score: 71.3, total_samples: 164, passed_samples: 117 },
        arc: { display_name: "ARC Challenge", score: 83.9, total_samples: 1172, passed_samples: 983 },
        ifeval: { display_name: "IFEval (Instruction)", score: 82.5, total_samples: 541, passed_samples: 446 },
        tool_calling: { display_name: "Tool Calling", score: 77.2, total_samples: 300, passed_samples: 232 }
      },
      raw_logs_preview: "[2026-10-20 08:00:00] bitsandbytes 4-bit NF4 quantization applied with bfloat16 compute.",
      rank: 7
    },
    {
      filename: "microsoft_Phi-3.5-mini-instruct-AWQ_4BIT.json",
      file_path: "data/results/microsoft_Phi-3.5-mini-instruct-AWQ_4BIT.json",
      evaluation_id: "eval-phi35-mini-awq",
      timestamp: "2026-10-18T13:45:00Z",
      submitter: { name: "Karthik Subramanian", github_handle: "ksubramanian-dev", affiliation: "Edge LLM Benchmarks" },
      model: {
        model_id: "microsoft/Phi-3.5-mini-instruct",
        model_slug: "Phi-3.5-mini-instruct-AWQ_4BIT",
        hf_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        param_count_b: 3.82,
        param_tier: "3B–7B",
        architecture: "Phi3ForCausalLM",
        quantization: "AWQ",
        context_length: 131072,
        license: "MIT",
        backend: "AutoAWQ"
      },
      scores: { overall: 76.70, gsm8k: 82.10, humaneval: 68.90, arc: 82.40, ifeval: 77.10, tool_calling: 70.80 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4060", gpu_count: 1, gpu_vram_gb: 8.0,
        cpu: "AMD Ryzen 7 7700", cpu_cores: 8, total_ram_gb: 32.0,
        os: "Windows 11 (10.0.22631)", cuda_version: "12.3", driver_version: "546.33"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 82.1, total_samples: 1319, passed_samples: 1083 },
        humaneval: { display_name: "HumanEval (Coding)", score: 68.9, total_samples: 164, passed_samples: 113 },
        arc: { display_name: "ARC Challenge", score: 82.4, total_samples: 1172, passed_samples: 966 },
        ifeval: { display_name: "IFEval (Instruction)", score: 77.1, total_samples: 541, passed_samples: 417 },
        tool_calling: { display_name: "Tool Calling", score: 70.8, total_samples: 300, passed_samples: 212 }
      },
      raw_logs_preview: "[2026-10-18 13:30:00] AutoAWQ quantization verified. High efficiency in memory constrained setups.",
      rank: 8
    },
    {
      filename: "meta-llama_Llama-3.2-3B-Instruct-BF16.json",
      file_path: "data/results/meta-llama_Llama-3.2-3B-Instruct-BF16.json",
      evaluation_id: "eval-llama32-3b-bf16",
      timestamp: "2026-10-15T19:30:00Z",
      submitter: { name: "Clara Oswald", github_handle: "clara-ml", affiliation: "TinyModels Org" },
      model: {
        model_id: "meta-llama/Llama-3.2-3B-Instruct",
        model_slug: "Llama-3.2-3B-Instruct-BF16",
        hf_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        param_count_b: 3.21,
        param_tier: "3B–7B",
        architecture: "LlamaForCausalLM",
        quantization: "BF16",
        context_length: 131072,
        license: "Llama 3.2 Community",
        backend: "transformers"
      },
      scores: { overall: 73.65, gsm8k: 77.20, humaneval: 63.40, arc: 78.50, ifeval: 76.80, tool_calling: 71.00 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 4070", gpu_count: 1, gpu_vram_gb: 12.0,
        cpu: "Intel Core i5-13600K", cpu_cores: 14, total_ram_gb: 32.0,
        os: "Ubuntu 24.04 LTS", cuda_version: "12.4", driver_version: "550.54"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 77.2, total_samples: 1319, passed_samples: 1018 },
        humaneval: { display_name: "HumanEval (Coding)", score: 63.4, total_samples: 164, passed_samples: 104 },
        arc: { display_name: "ARC Challenge", score: 78.5, total_samples: 1172, passed_samples: 920 },
        ifeval: { display_name: "IFEval (Instruction)", score: 76.8, total_samples: 541, passed_samples: 415 },
        tool_calling: { display_name: "Tool Calling", score: 71.0, total_samples: 300, passed_samples: 213 }
      },
      raw_logs_preview: "[2026-10-15 19:15:00] Evaluated Llama-3.2-3B-Instruct in native BF16.",
      rank: 9
    },
    {
      filename: "HuggingFaceTB_SmolLM2-1.7B-Instruct-BF16.json",
      file_path: "data/results/HuggingFaceTB_SmolLM2-1.7B-Instruct-BF16.json",
      evaluation_id: "eval-smollm2-17b-bf16",
      timestamp: "2026-10-12T14:10:00Z",
      submitter: { name: "Louisa Martin", github_handle: "lmartin-hf", affiliation: "Hugging Face" },
      model: {
        model_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_slug: "SmolLM2-1.7B-Instruct-BF16",
        hf_url: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
        param_count_b: 1.71,
        param_tier: "< 3B",
        architecture: "LlamaForCausalLM",
        quantization: "BF16",
        context_length: 8192,
        license: "Apache-2.0",
        backend: "transformers"
      },
      scores: { overall: 67.30, gsm8k: 68.50, humaneval: 54.90, arc: 72.80, ifeval: 73.10, tool_calling: 65.40 },
      hardware: {
        gpu_name: "NVIDIA GeForce RTX 3060", gpu_count: 1, gpu_vram_gb: 12.0,
        cpu: "AMD Ryzen 5 5600X", cpu_cores: 6, total_ram_gb: 32.0,
        os: "Ubuntu 22.04 LTS", cuda_version: "12.1", driver_version: "535.154"
      },
      tasks: {
        gsm8k: { display_name: "GSM8K (Math)", score: 68.5, total_samples: 1319, passed_samples: 903 },
        humaneval: { display_name: "HumanEval (Coding)", score: 54.9, total_samples: 164, passed_samples: 90 },
        arc: { display_name: "ARC Challenge", score: 72.8, total_samples: 1172, passed_samples: 853 },
        ifeval: { display_name: "IFEval (Instruction)", score: 73.1, total_samples: 541, passed_samples: 395 },
        tool_calling: { display_name: "Tool Calling", score: 65.4, total_samples: 300, passed_samples: 196 }
      },
      raw_logs_preview: "[2026-10-12 14:00:00] Evaluating SmolLM2-1.7B lightweight sub-3B model.",
      rank: 10
    }
  ]
};

// Markdown PR template content
const PR_TEMPLATE_TEXT = `### Model Submission Checklist
- [x] Model parameter size is \\le 50B parameters.
- [x] Benchmark evaluations were run with reproducible harness settings.
- [x] Result JSON file follows the standardized schema.
- [x] Result file placed in \`/data/results/<organization>_<model_slug>.json\`.
- [x] No proprietary or restricted access models (must have open weights).

### Evaluation Summary
- **Model ID:** [e.g., Qwen/Qwen2.5-7B-Instruct]
- **Hugging Face Link:** [e.g., https://huggingface.co/Qwen/Qwen2.5-7B-Instruct]
- **Parameter Count:** [e.g., 7.61B]
- **Quantization:** [e.g., GGUF Q4_K_M]
- **License:** [e.g., Apache-2.0]

### Hardware Environment
| Resource | Specification |
|:---|:---|
| **GPU Model** | NVIDIA RTX 4090 (24GB) |
| **CPU** | AMD Ryzen 9 7950X |
| **RAM** | 64 GB DDR5 |
| **OS / CUDA** | Ubuntu 24.04 LTS / CUDA 12.4 |
| **Inference Backend** | llama.cpp |

### Proof of Run
Attached evaluation log snippet and hash confirmation:
\`\`\`text
[2026-11-01 14:05:00] Initialized llama.cpp backend with -ngl 99
[2026-11-01 14:22:10] Benchmark suite finished. Overall Score: 87.65%
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
  if (allModels.length === 0) return;

  document.getElementById('stat-model-count').textContent = allModels.length;

  const topOverall = allModels.reduce((prev, cur) => (cur.scores.overall > prev.scores.overall) ? cur : prev, allModels[0]);
  document.getElementById('stat-top-score').textContent = topOverall.scores.overall.toFixed(1) + '%';
  document.getElementById('stat-top-model').textContent = topOverall.model.model_id;

  const topCoding = allModels.reduce((prev, cur) => (cur.scores.humaneval > prev.scores.humaneval) ? cur : prev, allModels[0]);
  document.getElementById('stat-top-coding').textContent = topCoding.scores.humaneval.toFixed(1) + '%';
  document.getElementById('stat-top-coding-model').textContent = topCoding.model.model_id;

  const topMath = allModels.reduce((prev, cur) => (cur.scores.gsm8k > prev.scores.gsm8k) ? cur : prev, allModels[0]);
  document.getElementById('stat-top-math').textContent = topMath.scores.gsm8k.toFixed(1) + '%';
  document.getElementById('stat-top-math-model').textContent = topMath.model.model_id;

  const updatedDate = rawLeaderboardData?.metadata?.last_updated ? new Date(rawLeaderboardData.metadata.last_updated).toLocaleDateString() : 'Today';
  document.getElementById('last-updated-text').textContent = updatedDate;
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
      if (tierFilter === '32B–50B' && (p <= 32.0 || p > 50.0)) return false;
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

  countText.textContent = `Showing ${filteredModels.length} of ${allModels.length} evaluated models`;

  if (filteredModels.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="11" class="text-center py-16 text-slate-400">
          <i class="fa-solid fa-filter-circle-xmark text-3xl text-slate-600 mb-2"></i>
          <div class="text-sm font-semibold text-slate-300">No models match the current filters</div>
          <div class="text-xs text-slate-500 mt-1">Try clearing your search query or broadening parameter tiers.</div>
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
    tr.className = `leaderboard-row cursor-pointer transition ${isSelected ? 'bg-indigo-950/30' : ''}`;
    
    // Clicking anywhere on row (except checkbox & link) opens modal
    tr.onclick = (e) => {
      if (e.target.closest('.no-modal-click')) return;
      showModelModal(item);
    };

    // Rank Badge logic
    let rankBadge = `<span class="font-mono text-slate-400 text-xs font-semibold">#${item.rank}</span>`;
    if (item.rank === 1) {
      rankBadge = `<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/50 text-xs font-extrabold" title="Rank #1"><i class="fa-solid fa-crown text-[10px]"></i></span>`;
    } else if (item.rank === 2) {
      rankBadge = `<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-slate-400/20 text-slate-200 border border-slate-400/50 text-xs font-bold" title="Rank #2">2</span>`;
    } else if (item.rank === 3) {
      rankBadge = `<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-amber-700/20 text-amber-400 border border-amber-700/50 text-xs font-bold" title="Rank #3">3</span>`;
    }

    // Score coloring helper
    const getScoreBadge = (score) => {
      let colorClass = 'text-slate-300';
      if (score >= 90) colorClass = 'text-emerald-400 font-bold';
      else if (score >= 80) colorClass = 'text-cyan-300 font-semibold';
      else if (score >= 70) colorClass = 'text-indigo-300';
      return `<span class="${colorClass}">${score.toFixed(1)}%</span>`;
    };

    tr.innerHTML = `
      <td class="px-3 py-3 text-center no-modal-click" onclick="event.stopPropagation()">
        <input type="checkbox" class="model-select-cb rounded bg-slate-900 border-slate-700 text-indigo-500 focus:ring-0 focus:ring-offset-0 cursor-pointer" 
          data-id="${m.model_slug || m.model_id}" ${isSelected ? 'checked' : ''}>
      </td>
      <td class="px-3 py-3 text-center">${rankBadge}</td>
      <td class="px-4 py-3">
        <div class="flex items-center space-x-2">
          <a href="${m.hf_url}" target="_blank" rel="noopener noreferrer" 
            class="no-modal-click text-slate-100 hover:text-cyan-400 font-bold text-sm tracking-tight transition flex items-center gap-1.5"
            title="Open repository on Hugging Face" onclick="event.stopPropagation()">
            <span class="truncate max-w-[210px] sm:max-w-[280px]">${m.model_id}</span>
            <i class="fa-solid fa-arrow-up-right-from-square text-[10px] text-slate-500 hover:text-cyan-300"></i>
          </a>
        </div>
        <div class="flex items-center flex-wrap gap-1.5 mt-1 text-[11px] text-slate-400">
          <span class="inline-block px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono text-[10px]">${m.architecture}</span>
          <span class="inline-block px-1.5 py-0.2 rounded bg-slate-800/80 text-cyan-300 border border-cyan-900/60 font-mono text-[10px]">${m.quantization}</span>
          <span class="text-slate-500 text-[10px] truncate max-w-[120px]">${m.license}</span>
        </div>
      </td>
      <td class="px-3 py-3 font-mono text-xs">
        <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700">
          ${m.param_count_b ? m.param_count_b.toFixed(1) + 'B' : 'N/A'}
        </span>
      </td>
      <td class="px-4 py-3 text-right bg-cyan-950/20 border-x border-slate-800/50">
        <span class="inline-block px-2.5 py-1 rounded-lg text-sm font-extrabold text-cyan-300 bg-cyan-950/80 border border-cyan-700/60 shadow-sm">
          ${s.overall.toFixed(1)}%
        </span>
      </td>
      <td class="px-3 py-3 text-right font-mono text-xs">${getScoreBadge(s.gsm8k)}</td>
      <td class="px-3 py-3 text-right font-mono text-xs">${getScoreBadge(s.humaneval)}</td>
      <td class="px-3 py-3 text-right font-mono text-xs">${getScoreBadge(s.arc)}</td>
      <td class="px-3 py-3 text-right font-mono text-xs">${getScoreBadge(s.ifeval)}</td>
      <td class="px-3 py-3 text-right font-mono text-xs">${getScoreBadge(s.tool_calling)}</td>
      <td class="px-3 py-3 text-center no-modal-click" onclick="event.stopPropagation()">
        <button class="inspect-btn p-1.5 rounded-lg text-slate-400 hover:text-white bg-slate-800/70 hover:bg-slate-700 transition text-xs" title="Inspect model telemetry">
          <i class="fa-solid fa-chart-simple"></i>
        </button>
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
  document.getElementById('modal-architecture').textContent = m.architecture || 'CausalLM';
  document.getElementById('modal-license').textContent = m.license || 'Open';

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
    const passed = taskData.passed_samples ? ` &bull; Passed: ${taskData.passed_samples}/${taskData.total_samples || '?'}` : '';
    const speed = taskData.avg_tokens_per_sec ? ` &bull; ${taskData.avg_tokens_per_sec} tok/s` : '';

    return `
      <div class="bg-[#050811] p-3.5 rounded-xl border border-slate-800">
        <div class="flex items-center justify-between mb-1 text-xs">
          <div>
            <span class="font-bold text-slate-200">${t.label}</span>
            <span class="text-[11px] text-slate-500 ml-1">${t.desc}</span>
          </div>
          <span class="font-extrabold text-cyan-400 font-mono text-sm">${t.score.toFixed(1)}%</span>
        </div>
        <div class="w-full bg-slate-800/80 h-2 rounded-full overflow-hidden my-1.5">
          <div class="bg-gradient-to-r from-cyan-500 to-emerald-400 h-2 rounded-full transition-all duration-500" style="width: ${t.score}%"></div>
        </div>
        <div class="text-[10px] text-slate-400 flex items-center justify-between">
          <span>Metric: ${taskData.metric || 'accuracy'}${passed}</span>
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
    if (btn.getAttribute('data-tab') === tabId) {
      btn.className = 'modal-tab-btn py-3 px-3 border-b-2 border-cyan-400 text-cyan-300 font-semibold';
    } else {
      btn.className = 'modal-tab-btn py-3 px-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200';
    }
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
    alert("Please select 2 or 3 models from the leaderboard to compare.");
    return;
  }

  const columns = ['Model', 'Parameters', 'Overall Score', 'GSM8K (Math)', 'HumanEval (Coding)', 'ARC (Reasoning)', 'IFEval', 'Tool Calling', 'Architecture', 'Quantization', 'License', 'GPU Hardware'];

  container.innerHTML = `
    <table class="w-full text-xs text-left border-collapse">
      <thead>
        <tr class="border-b border-slate-800 bg-[#050811]">
          <th class="p-3 text-slate-400 uppercase font-semibold">Metric / Feature</th>
          ${selectedModels.map(m => `
            <th class="p-3 text-cyan-300 font-bold text-sm">
              <div>${m.model.model_id}</div>
              <div class="text-[10px] text-slate-400 font-normal">Rank #${m.rank}</div>
            </th>
          `).join('')}
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-800 text-slate-200">
        <tr class="bg-slate-900/40">
          <td class="p-3 font-semibold text-slate-300">Overall Score</td>
          ${selectedModels.map(m => `<td class="p-3 font-extrabold text-cyan-400 text-base font-mono">${m.scores.overall.toFixed(1)}%</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">GSM8K (Math)</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono font-semibold">${m.scores.gsm8k.toFixed(1)}%</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">HumanEval (Coding)</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono font-semibold">${m.scores.humaneval.toFixed(1)}%</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">ARC (Reasoning)</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono font-semibold">${m.scores.arc.toFixed(1)}%</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">IFEval</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono font-semibold">${m.scores.ifeval.toFixed(1)}%</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">Tool Calling</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono font-semibold">${m.scores.tool_calling.toFixed(1)}%</td>`).join('')}
        </tr>
        <tr class="bg-slate-900/40">
          <td class="p-3 font-semibold text-slate-300">Parameter Count</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono">${m.model.param_count_b}B (${m.model.param_tier})</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">Quantization</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono text-cyan-300">${m.model.quantization}</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">Architecture</td>
          ${selectedModels.map(m => `<td class="p-3 text-slate-300">${m.model.architecture}</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">License</td>
          ${selectedModels.map(m => `<td class="p-3 text-slate-300">${m.model.license}</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">Inference Backend</td>
          ${selectedModels.map(m => `<td class="p-3 font-mono text-slate-300">${m.model.backend}</td>`).join('')}
        </tr>
        <tr>
          <td class="p-3 font-semibold text-slate-300">GPU Hardware</td>
          ${selectedModels.map(m => `<td class="p-3 text-slate-300">${m.hardware?.gpu_name || 'N/A'}</td>`).join('')}
        </tr>
      </tbody>
    </table>
  `;

  modal.classList.remove('hidden');
}

// Chart.js Visualizations
function renderCharts() {
  if (allModels.length === 0) return;

  // Radar Chart: top 3 models or selected models
  let modelsForRadar = allModels.slice(0, 3);
  if (selectedModelIds.size > 0) {
    const selected = allModels.filter(m => selectedModelIds.has(m.model.model_slug || m.model.model_id));
    if (selected.length > 0) modelsForRadar = selected;
  }

  const radarLabels = ['GSM8K (Math)', 'HumanEval (Code)', 'ARC (Reasoning)', 'IFEval (Format)', 'Tool Calling'];
  const colors = [
    { bg: 'rgba(6, 182, 212, 0.25)', border: '#06b6d4' },
    { bg: 'rgba(168, 85, 247, 0.25)', border: '#a855f7' },
    { bg: 'rgba(16, 185, 129, 0.25)', border: '#10b981' }
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
        borderWidth: 2,
        pointBackgroundColor: colors[idx % colors.length].border,
        pointBorderColor: '#fff',
        pointHoverRadius: 5
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: { color: '#94a3b8', font: { size: 10, family: 'Plus Jakarta Sans' } },
          ticks: { backdropColor: 'transparent', color: '#64748b', stepSize: 20, font: { size: 9 } },
          suggestedMin: 50,
          suggestedMax: 100
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#f1f5f9', font: { size: 11, family: 'Plus Jakarta Sans' }, boxWidth: 12 }
        }
      }
    }
  });

  // Bar Chart: Parameter Efficiency & Overall Score Distribution
  const barCtx = document.getElementById('barChart').getContext('2d');
  if (barChartInstance) barChartInstance.destroy();

  const sortedBySize = [...allModels].sort((a, b) => (a.model.param_count_b || 0) - (b.model.param_count_b || 0));

  barChartInstance = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: sortedBySize.map(m => `${(m.model.model_id.split('/')[1] || m.model.model_id).substring(0, 14)} (${m.model.param_count_b}B)`),
      datasets: [{
        label: 'Overall Benchmark Score (%)',
        data: sortedBySize.map(m => m.scores.overall),
        backgroundColor: sortedBySize.map(m => {
          if (m.scores.overall >= 85) return 'rgba(16, 185, 129, 0.7)';
          if (m.scores.overall >= 80) return 'rgba(6, 182, 212, 0.7)';
          if (m.scores.overall >= 75) return 'rgba(99, 102, 241, 0.7)';
          return 'rgba(148, 163, 184, 0.5)';
        }),
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          min: 50,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 9 }, maxRotation: 35, minRotation: 20 }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => sortedBySize[items[0].dataIndex].model.model_id,
            label: (item) => `Score: ${item.raw}% | Size: ${sortedBySize[item.dataIndex].model.param_count_b}B`
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
    document.getElementById('cta-pr-guide-btn')
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
    if (btn.getAttribute('data-prtab') === tabId) {
      btn.className = 'pr-tab-btn py-3 px-3 border-b-2 border-cyan-400 text-cyan-300 font-semibold';
    } else {
      btn.className = 'pr-tab-btn py-3 px-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200';
    }
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
    feedback.className = 'p-3 rounded-lg text-xs bg-red-950/70 border border-red-800 text-red-300';
    feedback.innerHTML = '<i class="fa-solid fa-triangle-exclamation mr-1.5"></i> Please paste JSON text to validate.';
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
      else if (data.model.param_count_b > 50.0) errors.push(`Parameter count (${data.model.param_count_b}B) exceeds the 50B SLM limit.`);
      if (!data.model.quantization) errors.push("Missing 'model.quantization'.");
      if (!data.model.license) errors.push("Missing 'model.license'.");
    }

    if (!data.hardware) errors.push("Missing 'hardware' specification object (GPU, CPU, RAM).");
    if (!data.summary_scores && !data.tasks) errors.push("Missing benchmark evaluation scores ('summary_scores' or 'tasks').");

    if (errors.length > 0) {
      feedback.className = 'p-3 rounded-lg text-xs bg-red-950/70 border border-red-800 text-red-300';
      feedback.innerHTML = `<div class="font-bold mb-1"><i class="fa-solid fa-circle-xmark mr-1.5"></i> Validation Failed (${errors.length} issues):</div><ul class="list-disc pl-5 space-y-0.5">${errors.map(e => `<li>${e}</li>`).join('')}</ul>`;
      feedback.classList.remove('hidden');
    } else {
      feedback.className = 'p-3 rounded-lg text-xs bg-emerald-950/70 border border-emerald-800 text-emerald-300';
      feedback.innerHTML = `<div class="font-bold"><i class="fa-solid fa-circle-check mr-1.5"></i> Validation Passed!</div><p class="mt-1">This result file conforms to schema guidelines and is ready to be submitted in a Pull Request.</p>`;
      feedback.classList.remove('hidden');
    }
  } catch (err) {
    feedback.className = 'p-3 rounded-lg text-xs bg-red-950/70 border border-red-800 text-red-300';
    feedback.innerHTML = `<i class="fa-solid fa-bug mr-1.5"></i> Invalid JSON Syntax: ${err.message}`;
    feedback.classList.remove('hidden');
  }
}

// Update Table Header Sort Icons
function updateHeaderSortIcons() {
  const cols = ['rank', 'name', 'params', 'overall', 'gsm8k', 'humaneval', 'arc', 'ifeval', 'tool_calling'];
  cols.forEach(col => {
    const icon = document.getElementById(`sort-icon-${col}`);
    if (!icon) return;

    if (currentSort.column === col) {
      icon.className = currentSort.direction === 'asc' ? 'fa-solid fa-sort-up text-cyan-400 text-xs' : 'fa-solid fa-sort-down text-cyan-400 text-xs';
    } else {
      icon.className = 'fa-solid fa-sort text-slate-600 text-[10px]';
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
