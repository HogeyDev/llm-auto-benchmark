# AI Model Benchmark Leaderboard 🏆

A modern, responsive, 100% client-side web application designed to host an **Open AI Model Benchmark Leaderboard** directly on **GitHub Pages**. All benchmark data is stored statically as JSON files in the repository, enabling completely transparent, reproducible, and community-driven evaluations with zero server-side databases.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![GitHub Pages](https://img.shields.io/badge/hosting-GitHub%20Pages-brightgreen.svg)
![Evaluations](https://img.shields.io/badge/models-Small%20Local%20AI-cyan.svg)

---

## 🌟 Key Features

- **Pure Static Architecture:** Runs seamlessly on GitHub Pages or any static CDN without any backend servers, Node runtime, or databases.
- **Five Standardized Benchmark Pillars:**
  - 📐 **GSM8K:** Grade School Math 8K (strict numerical answer match).
  - 💻 **HumanEval:** Python Code Generation (pass@1 execution tests).
  - 🧠 **ARC Challenge:** Scientific and logical reasoning multi-choice evaluation.
  - 📋 **IFEval:** Verifiable instruction adherence and formatting constraints.
  - 🔧 **Tool Calling:** Multi-turn function execution and JSON schema adherence.
- **Interactive UI & Leaderboard Table:**
  - Dynamic full-text search across model name, architecture, license, and hardware backend.
  - Multi-column sorting (Rank, Model Name, Parameter Count, Overall Score, GSM8K, HumanEval, ARC, IFEval, Tool Calling).
  - Multi-dimensional filters for **Parameter Tiers** (`< 3B`, `3B–7B`, `7B–14B`, `14B–32B`, `32B+`), **Model Licenses** (Apache-2.0, MIT, Llama Community, etc.), and **Quantization Precision** (BF16, AWQ, GGUF Q4/Q8, NF4).
  - Direct links to model cards on the **Hugging Face Hub**.
- **Deep Model Telemetry & Inspector Modal:**
  - Click any model row to inspect hardware specs (GPU, VRAM, CPU cores, system RAM, OS, CUDA/driver versions).
  - Sub-task pass rates, throughput metrics (tokens/sec), and evaluation logs viewer with one-click copy.
  - Download or copy raw submission JSON.
- **Side-by-Side Head-to-Head Compare:** Select up to 3 models to compare metrics and hardware side-by-side.
- **Interactive Visualizations:**
  - 5-Pillar Capability Radar Chart powered by Chart.js.
  - Parameter Size vs Overall Score efficiency distribution chart.
- **Export Data:** Download current filtered views as CSV or JSON.
- **Automated Pull Request Workflow:** Complete CI validation and submission templates for community evaluations.

---

## 📂 Repository Structure

```text
├── .github/
│   ├── pull_request_template.md     # Auto-populated PR description template
│   └── workflows/
│       ├── deploy-pages.yml         # GitHub Actions deployment to Pages
│       └── validate-submission.yml  # Automated schema & bounds validation on PRs
├── data/
│   ├── leaderboard.json             # Pre-aggregated master leaderboard index
│   └── results/                     # Individual raw evaluation result files
│       ├── index.json               # Manifest listing all evaluation result files
│       ├── deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json
│       ├── Qwen_Qwen2.5-32B-Instruct-GGUF_Q4_K_M.json
│       ├── Qwen_Qwen2.5-Coder-7B-Instruct-BF16.json
│       └── ...
├── scripts/
│   ├── aggregate.py                 # Scans data/results/*.json and builds leaderboard.json
│   └── generate_sample_data.py      # Generates sample benchmark entries
├── app.js                           # Core client-side filtering, sorting, and UI logic
├── index.html                       # Single-page leaderboard application
├── PR_TEMPLATE.md                   # Standalone contributor PR guide
├── style.css                        # Glassmorphism dark-theme styling
└── README.md                        # Documentation
```

---

## 🚀 How Benchmark Data is Loaded

The application supports a dual-loading strategy:
1. **Aggregated Master File (`/data/leaderboard.json`):** On initial load, the client makes a single HTTP GET request to `/data/leaderboard.json`. This provides instant, zero-latency loading of all ranked models and stats.
2. **Individual Files Fallback (`/data/results/*.json`):** If the aggregated file is unavailable, the application fetches the `/data/results/index.json` manifest and requests each individual result JSON in parallel.
3. **Embedded Fallback Dataset:** When opened directly from a local filesystem via `file://` where browser CORS prevents local file `fetch()`, a pre-bundled fallback dataset ensures all interactive features, tables, charts, and modals continue to function.

To update or re-aggregate after adding new result files:
```bash
python scripts/aggregate.py
```

---

## 🤝 Community Pull Request Submission Workflow

We welcome benchmark submissions for any open-weights model with **&le; 50 Billion parameters**!

### Step 1: Fork and Clone
```bash
git clone https://github.com/<your-username>/llm-auto-benchmark-leaderboard.git
cd llm-auto-benchmark-leaderboard
```

### Step 2: Run the Benchmark Suite Locally
Run the evaluation harness on your local machine or compute cluster:
```bash
# Example command using llama.cpp backend:
python -m slm_bench.evaluate \
  --model-id "Qwen/Qwen2.5-7B-Instruct" \
  --quantization "GGUF_Q4_K_M" \
  --backend llama.cpp \
  --benchmarks gsm8k,humaneval,arc,ifeval,tool_calling \
  --output-dir ./data/results/
```

### Step 3: Verify the Generated JSON
Ensure your generated JSON file:
- Is located at `data/results/<organization>_<model_slug>.json`.
- Contains full hardware telemetry (GPU, VRAM, CPU, RAM, OS, CUDA).
- Includes verifiable execution logs.
- Can be validated in-browser using our **PR Guide > Local JSON Validator** tab or via CLI:
  ```bash
  python scripts/aggregate.py
  ```

### Step 4: Open a Pull Request
1. Commit your changes:
   ```bash
   git checkout -b submit/qwen-2.5-7b
   git add data/results/Qwen_Qwen2.5-7B-Instruct-GGUF_Q4_K_M.json
   git commit -m "feat(benchmark): add Qwen2.5-7B-Instruct evaluation"
   git push origin submit/qwen-2.5-7b
   ```
2. Open a Pull Request on GitHub. GitHub will automatically populate our [PR Template](PR_TEMPLATE.md).
3. The `.github/workflows/validate-submission.yml` action will automatically verify your file schema, check small local AI criteria (reviewed by maintainer), and validate scores.
4. Once reviewed and merged, GitHub Pages will automatically re-deploy the updated leaderboard!

---

## 💻 Local Development & Preview

Because this is a pure static web app, no build step or node_modules are required!

You can preview the site with any static web server:

```bash
# Python 3
python -m http.server 8000

# Node.js npx
npx serve .
```

Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Benchmark data contributed by the community remains open under permissive open data licenses.
