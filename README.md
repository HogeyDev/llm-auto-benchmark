# Local AI Benchmark: Evaluating AI for Everyday Hardware 💻🏆

[![GitHub Pages](https://img.shields.io/badge/Leaderboard-GitHub%20Pages-brightgreen.svg)](https://seandolan0.github.io/llm-auto-benchmark/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Category](https://img.shields.io/badge/Hardware-Everyday%20Local%20AI-purple.svg)](#mission--hardware-philosophy)
[![CI Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-emerald.svg)](tests/)

An automated evaluation harness, benchmark runner, and community submission pipeline designed specifically for benchmarking **new AI models that normal people can reasonably run on hardware they already have** and publishing validated results to a zero-server **GitHub Pages** leaderboard via automated Pull Requests.

🌐 **Live Leaderboard:** [https://seandolan0.github.io/llm-auto-benchmark/](https://seandolan0.github.io/llm-auto-benchmark/)

---

## 💡 Mission & Hardware Philosophy

Most AI benchmarks evaluate massive frontier models running on expensive enterprise cloud clusters. **Our goal is completely different:**

> **Benchmark new AI models that everyday people can realistically run on the hardware they already own.**

Whether you are a developer, hobbyist, or student using:
- 🎮 **A standard gaming PC or laptop** with an NVIDIA RTX 3060, 4060, 4070, 5070, or AMD Radeon GPU (6GB–16GB VRAM)
- 🍏 **An Apple Silicon Mac** (M1/M2/M3/M4 MacBook Air/Pro, Mac Mini, Studio) with unified memory
- 💻 **An everyday desktop or laptop** using CPU inference and standard system RAM (16GB–32GB) with fast quantized formats (GGUF, AWQ)

We focus on what actually runs well locally without needing cloud API subscriptions or enterprise server gear. Model inclusion is curated by the project maintainer to ensure every model on the leaderboard is practical for local use.

---

## 🌟 Key Architecture & Highlights

1. **Focused on Consumer & Everyday Hardware**:
   - Evaluates models across realistic parameter tiers tailored to consumer machines:
     - **`< 3B` — Lightweight:** Runs comfortably on almost any modern laptop, PC, or mobile device.
     - **`3B–7B` — Everyday Sweet Spot:** Ideal for 8GB VRAM GPUs (RTX 3060/4060) and 16GB RAM Macs.
     - **`7B–14B` — Mid-Range Powerhouse:** Optimal on 12GB–16GB VRAM GPUs (RTX 4070) or 24GB Macs.
     - **`14B–32B` — Enthusiast Local:** Runs via 4-bit/8-bit quantization on 16GB+ VRAM or unified memory.
     - **`32B+` — Heavy Local:** Evaluated using GGUF layer-by-layer offloading to test the boundaries of consumer rigs.
2. **Multi-Backend Inference with Low-VRAM Quantization**:
   - **`llama.cpp` / GGUF**: Granular layer-by-layer GPU offloading (`n_gpu_layers`) allowing models to run on standard 8GB–16GB consumer GPUs with system RAM fallback.
   - **`vLLM`**: High-throughput batched inference with PagedAttention and native AWQ / GPTQ / FP8 support.
   - **`Transformers` + `bitsandbytes`**: Direct evaluation of Hugging Face weights in 4-bit (NF4) or 8-bit precision with automatic device mapping.
   - **`MockBackend`**: Deterministic simulation for lightning-fast testing and CI/CD validation without requiring GPU hardware.
3. **Pillar Benchmark Taxonomy & Normalization**:
   - 📐 **GSM8K (Math & Reasoning)** (25% weight): Multi-step Chain-of-Thought with algebraic extraction and strict verification.
   - 💻 **HumanEval (Agentic Coding)** (30% weight): Python function synthesis, docstring contract fulfillment, and test assertion execution in an isolated sandbox.
   - 🧠 **ARC Challenge (Scientific Reasoning)** (20% weight): Advanced scientific and logical question-answering with accuracy normalization.
   - 📋 **IFEval (Instruction Following)** (15% weight): Strict constraint obedience (JSON schemas, negative constraints, bullet counts).
   - 🔧 **Multi-Turn Tool Calling** (10% weight): Function schema catalog selection, structured parameter extraction, and execution.
   - **Composite SLM-Index**: Overall weighted composite score normalized to $0.0 - 100.0\%$.
4. **Automated GitHub Pages PR Workflow**:
   - The CLI (`--submit`) clones the leaderboard repository, creates branch `eval/<model-slug>`, commits result JSON to `website/data/results/<model-slug>.json`, pushes, and opens an automated Pull Request with formatted hardware specs and task metrics.
   - GitHub Actions validates schema and real-world hardware telemetry on every PR.
   - Once merged into `main`, GitHub Pages automatically updates and redeploys the leaderboard!
5. **Static Client-Side Leaderboard Application**:
   - 100% static web application with glassmorphism dark mode, real-time search, multi-column sorting, hardware tier filtering, head-to-head model comparison drawer, Chart.js radar capability diagrams, and deep telemetry inspector modals.

---

## 📂 Repository Structure

```text
llm-auto-benchmark/
├── .github/
│   ├── workflows/
│   │   ├── deploy-pages.yml         # Aggregates data & deploys website/ to GitHub Pages
│   │   ├── validate-submission.yml  # Validates PR schema & local hardware telemetry
│   │   └── ci.yml                   # Automated Pytest suite across Python versions & OS
│   └── pull_request_template.md     # Auto-populated PR description template
├── slm_bench/                       # Python evaluation harness & CLI package
│   ├── backends/                    # Inference engines: llama.cpp, vLLM, HF, Mock
│   ├── benchmarks/                  # Evaluators: GSM8K, Coding, IFEval, Tool Calling
│   ├── models/                      # Hardware telemetry & Pydantic result schemas
│   ├── submission/                  # Git branching & GitHub PR automation
│   ├── utils/                       # Rich console logging & formatting
│   └── cli.py                       # Typer CLI entrypoint (`slm-bench`)
├── tests/                           # Pytest unit tests for evaluation harness
├── website/                         # Static Leaderboard Application (GitHub Pages root)
│   ├── data/
│   │   ├── results/                 # Individual model benchmark results (*.json)
│   │   │   └── index.json           # Manifest listing all evaluation files
│   │   └── leaderboard.json         # Master pre-aggregated leaderboard dataset
│   ├── scripts/
│   │   ├── aggregate.py             # Compiles data/results/*.json into leaderboard.json
│   │   └── generate_sample_data.py  # Populates sample evaluation results
│   ├── app.js                       # Client-side UI logic, charts, filtering & sorting
│   ├── index.html                   # Single-page application entrypoint
│   ├── PR_TEMPLATE.md               # Standalone contributor guide
│   └── style.css                    # Glassmorphism dark-theme styling
├── pyproject.toml                   # Python build metadata & dependencies
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/SeanDolan0/llm-auto-benchmark.git
cd llm-auto-benchmark

# Install in editable mode
pip install -e .

# Or install with optional backend dependencies for local execution
pip install -e ".[transformers,llamacpp]"

# Or install all extras including testing tools
pip install -e ".[all]"
```

---

## 🚀 CLI Quick Start

### 1. Inspect Your Host Hardware
Inspect what hardware your local machine has before running benchmarks:
```bash
slm-bench info --model-id "Qwen/Qwen2.5-7B-Instruct"
```

### 2. Run Benchmark Evaluation Locally
```bash
# Fast mock evaluation (for testing the pipeline)
slm-bench evaluate \
  --model-id "meta-llama/Llama-3.2-3B-Instruct" \
  --backend "mock" \
  --sample-limit 2

# Real evaluation using llama.cpp on consumer GPU or CPU
slm-bench evaluate \
  --model-id "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --quantization "GGUF_Q4_K_M" \
  --backend "llamacpp"
```

### 3. Evaluate and Submit via Pull Request
```bash
slm-bench evaluate \
  --model-id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --quantization "GGUF_Q4_K_M" \
  --backend "llamacpp" \
  --submit \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

### 4. Submit an Existing Result File
```bash
slm-bench submit \
  --result-file "website/data/results/Qwen_Qwen2.5-7B-Instruct-GGUF_Q4_K_M.json" \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

---

## 🌐 Local Website Development & Preview

Because the leaderboard website is completely static, no build step or Node.js environment is required:

```bash
# Aggregate latest benchmark results
python website/scripts/aggregate.py

# Launch local preview web server
python -m http.server -d website 8000
```

Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 🤝 Contributing Benchmark Results

We welcome community benchmark submissions for open-weights models tested on **common local hardware**!

1. Fork and clone the repository.
2. Run your benchmark locally using `slm-bench evaluate`. The tool automatically records your GPU, VRAM, CPU, and RAM telemetry.
3. Save or copy your verified result JSON into `website/data/results/<model_slug>.json`.
4. Run `python website/scripts/aggregate.py` to verify compilation.
5. Open a Pull Request. GitHub Actions will automatically validate your submission schema. The maintainer will review submissions to determine if the model is suitable for the leaderboard, and GitHub Pages will deploy the updated leaderboard upon merge.

---

## 🧪 Running Tests

```bash
pytest -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Leaderboard benchmark data contributed by the community is shared under open and permissive data licenses.
