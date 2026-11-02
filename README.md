# LLM Auto Benchmark: SLM-Bench & Open Leaderboard 🏆

[![GitHub Pages](https://img.shields.io/badge/Leaderboard-GitHub%20Pages-brightgreen.svg)](https://seandolan0.github.io/llm-auto-benchmark/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Category](https://img.shields.io/badge/Category-Small%20Local%20AI-purple.svg)](#small-local-ai)
[![CI Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-emerald.svg)](tests/)

An automated evaluation harness, benchmark runner, and community submission pipeline designed specifically for benchmarking **Small Local AI models** and publishing validated results to a zero-server **GitHub Pages** leaderboard via automated Pull Requests. Model inclusion is curated by the maintainer to ensure models qualify as small local AI.

🌐 **Live Leaderboard:** [https://seandolan0.github.io/llm-auto-benchmark/](https://seandolan0.github.io/llm-auto-benchmark/)

---

## 🌟 Key Architecture & Highlights

1. **Focused on Small Local AI**:
   - Designed for models suited for local execution on consumer and workstation hardware.
   - Model inclusion is curated by the project maintainer to determine if models are small enough for the leaderboard.
   - Categorizes models into practical parameter tiers: `< 3B`, `3B–7B`, `7B–14B`, `14B–32B`, `32B+`.
2. **Multi-Backend Inference with Low-VRAM Quantization**:
   - **`llama.cpp` / GGUF**: Granular layer-by-layer GPU offloading (`n_gpu_layers`) allowing 14B–32B+ models to run comfortably on standard 12GB–16GB consumer GPUs (e.g. RTX 4070/5070) with RAM fallback.
   - **`vLLM`**: High-concurrency batched inference with PagedAttention and native AWQ / GPTQ / FP8 support.
   - **`Transformers` + `bitsandbytes`**: Direct evaluation of arbitrary Hugging Face repo IDs in 4-bit (NF4) or 8-bit precision with `device_map="auto"`.
   - **`MockBackend`**: Deterministic simulation for lightning-fast testing and CI/CD validation.
3. **Pillar Benchmark Taxonomy & Normalization**:
   - 📐 **GSM8K (Math & Reasoning)** (25% weight): Multi-step Chain-of-Thought with algebraic extraction and strict verification.
   - 💻 **HumanEval (Agentic Coding)** (30% weight): Python function synthesis, docstring contract fulfillment, and test assertion execution in an isolated sandbox.
   - 🧠 **ARC Challenge (Scientific Reasoning)** (20% weight): Advanced scientific and logical question-answering with accuracy normalization.
   - 📋 **IFEval (Instruction Following)** (15% weight): Strict constraint obedience (JSON schemas, negative constraints, bullet counts).
   - 🔧 **Multi-Turn Tool Calling** (10% weight): Function schema catalog selection, structured parameter extraction, and execution.
   - **Composite SLM-Index**: Overall weighted composite score normalized to $0.0 - 100.0\%$.
4. **Automated GitHub Pages PR Workflow**:
   - The CLI (`--submit`) clones the leaderboard repository, creates branch `eval/<model-slug>`, commits result JSON to `website/data/results/<model-slug>.json`, pushes, and opens an automated Pull Request with formatted hardware specs and task metrics.
   - GitHub Actions validates schema and hardware specs on every PR.
   - Once merged into `main`, GitHub Pages automatically updates and redeploys the leaderboard!
5. **Static Client-Side Leaderboard Application**:
   - 100% static web application with glassmorphism dark mode, real-time search, multi-column sorting, tier/quantization/license filtering, head-to-head model comparison drawer, Chart.js radar capability diagrams, and deep telemetry inspector modals.

---

## 📂 Repository Structure

```text
llm-auto-benchmark/
├── .github/
│   ├── workflows/
│   │   ├── deploy-pages.yml         # Aggregates data & deploys website to GitHub Pages
│   │   ├── validate-submission.yml  # Validates PR schema & small local AI submissions
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

# Or install with optional backend dependencies
pip install -e ".[transformers,llamacpp]"

# Or install all extras including testing tools
pip install -e ".[all]"
```

---

## 🚀 CLI Quick Start

### 1. Inspect Local Hardware & Hugging Face Model
```bash
slm-bench info --model-id "Qwen/Qwen2.5-7B-Instruct"
```

### 2. Run Benchmark Evaluation
```bash
# Fast mock evaluation (for testing)
slm-bench evaluate \
  --model-id "meta-llama/Llama-3.2-3B-Instruct" \
  --backend "mock" \
  --sample-limit 2

# Real evaluation using llama.cpp (GGUF)
slm-bench evaluate \
  --model-id "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --quantization "GGUF_Q4_K_M" \
  --backend "llamacpp"
```

### 3. Evaluate and Automatically Submit via Pull Request
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
  --result-file "website/data/results/Qwen2.5-7B-Instruct-GGUF_Q4_K_M.json" \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

### 5. Batch Evaluation
```bash
slm-bench batch \
  --batch-file "models.txt" \
  --backend "mock"
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

We welcome community benchmark submissions for open-weights models that qualify as **Small Local AI**!

1. Fork and clone the repository.
2. Run your benchmark using `slm-bench evaluate`.
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
