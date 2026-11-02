# Local AI Benchmark

A test harness and leaderboard for open-weights models that fit on hardware people actually own.

[Live leaderboard](https://seandolan0.github.io/llm-auto-benchmark/) &middot; [How scoring works](#how-the-score-works) &middot; [Run it](#run-it) &middot; [Submit a result](#submit-a-result)

Most leaderboards measure models on machines nobody has. This one measures them on RTX 3060s,
MacBooks, and laptops with 16&nbsp;GB of RAM — and publishes the machine next to the score, because
a score without hardware context is not reproducible.

---

## What is in this repository

Two things that meet at a JSON file:

1. **`slm_bench/`** — an extensible evaluation harness, interactive Terminal User Interface (TUI),
   and CLI for running standardized benchmark profiles, capturing real hardware telemetry, and generating reproducible results.
2. **`Website/`** — a static, paper-and-ink leaderboard that reads those results, ranks models across
   three distinct capability tracks, and lets you inspect every run. No server, no database, no build step.

A result is not accepted by uploading it to a service. It is a pull request against
`Website/data/results/`. CI checks the schema; a merge republishes the board.

## The tests & evaluation profiles

The test harness provides both rapid smoke testing and full academic/specialist benchmark suites organized by evaluation profile:

* **`quick`**: 4 fast core tests with minimal sample limit for smoke testing and local validation.
* **`standard`** (default): core general intelligence suite combining fast tests with MMLU-Pro, GSM8K, EvalPlus, and BFCL.
* **`extended`**: comprehensive evaluation across all 15 benchmark suites including GPQA, BBH, MATH-500, LiveCodeBench, IFBench, RULER, and Safety.

To list all available benchmarks and profiles:
```bash
slm-bench list-benchmarks
```

### Benchmark Tasks Catalog

| Category | Benchmark Tasks | Metric & Capability |
|:--|:--|:--|
| **Reasoning & Math** | **GSM8K**, **MATH-500**, **MMLU-Pro**, **GPQA Diamond**, **BBH**, `slm_reasoning_cot` | Multi-step word math, competition algebra/number theory, 10-choice professional reasoning, PhD science deduction, and algorithmic logic. |
| **Coding** | **EvalPlus** (HumanEval+ / MBPP+), **LiveCodeBench**, `slm_coding_agent` | Extended 80x edge-case Python function synthesis, post-cutoff competitive programming, and self-repair. |
| **Instruction Following** | **IFEval**, **IFBench**, `slm_instruction_ifeval` | Verifiable countable constraints, JSON structure adherence, and 2-turn dynamic constraint updates. |
| **Tool Calling & Agents** | **BFCL V4**, `slm_tool_calling_multiturn` | Catalog selection, strict parameter schema typing, multi-turn tool output synthesis, and negative refusal detection. |
| **Long Context** | **RULER** (`ruler_context`) | Effective context length testing (needle-in-a-haystack, multi-hop retrieval, variable tracking). |
| **Safety & Reliability** | **HarmBench** & **XSTest** (`safety_reliability`) | Adversarial jailbreak refusal accuracy vs. false refusal avoidance. |

## How the score works

Results are scored and grouped across three distinct leaderboard tracks:

1. **General Intelligence (`slm_index`)**:
   Weighted blend of core capability pillars:
   ```text
   reasoning 25%   coding 30%   instruction 25%   tool_calling 20%
   ```
2. **Coding & Agentic Index (`coding_agentic_index`)**:
   Specialist score combining code generation, unit-test fulfillment, function calling, and multi-turn tool synthesis.
3. **Local Hardware Efficiency**:
   Inference decode tokens/sec, memory footprint (VRAM/RAM), and normalized **Score-per-GB** efficiency.

Every metric computes **Wilson 95% confidence intervals** to distinguish genuine model differences from sample variance.

### Quantization Tradeoff Analysis

Compare capability retention, degradation percentage, and VRAM/RAM savings between model quantizations (e.g. BF16 vs Q4):

```bash
slm-bench compare-quantization --baseline results/Model-BF16.json --quantized results/Model-Q4.json
```

## Interactive Terminal Studio (TUI)

You can launch a full interactive Textual terminal studio to configure models, test endpoint connectivity, stream LLM reasoning and responses live, monitor evaluation progress in real time, and submit verified results:

```bash
slm-bench tui
```

Optional defaults can be provided on launch:
```bash
slm-bench tui --model-id "Qwen/Qwen2.5-7B-Instruct" --backend openai --endpoint-url "http://localhost:11434/v1"
```

## What "runs locally" means here

| Size | What it takes, roughly |
|:--|:--|
| **&lt; 3B** | Any recent laptop. CPU inference is fine. |
| **3B–7B** | 8&nbsp;GB VRAM, or 16&nbsp;GB of system RAM. The comfortable default. |
| **7B–14B** | 12–16&nbsp;GB VRAM, or an Apple Silicon machine with 24&nbsp;GB unified memory. |
| **14B–32B** | A 4-bit quant on 16–24&nbsp;GB VRAM, or layer offloading into system RAM. |
| **32B+** | Possible on consumer gear with offloading. Expect it to be slow. |

Model inclusion is curated: open weights, a license that permits local use, and a real path to
running on one of those machines. A model that technically loads after swapping to disk does not
belong on the board.

## Install

```bash
git clone https://github.com/SeanDolan0/llm-auto-benchmark.git
cd llm-auto-benchmark

pip install -e .                 # CLI and TUI studio
pip install -e ".[llamacpp]"     # llama.cpp / GGUF inference
pip install -e ".[transformers]" # HF weights, bitsandbytes 4-bit / 8-bit
pip install -e ".[vllm]"         # vLLM with AWQ / GPTQ
pip install -e ".[test]"         # pytest and test suite dependencies
pip install -e ".[all]"          # everything except vLLM, plus test dependencies
```

Python 3.10 or newer.

## Run it

Check what the host actually has before trusting a number:

```bash
slm-bench info --model-id "Qwen/Qwen2.5-7B-Instruct"
```

Do a two-sample run first. The mock backend exercises the whole pipeline without touching a GPU:

```bash
slm-bench evaluate \
  --model-id "meta-llama/Llama-3.2-3B-Instruct" \
  --backend mock \
  --sample-limit 2
```

A real GGUF run with layer offloading using the `standard` profile:

```bash
slm-bench evaluate \
  --model-id "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --quantization "GGUF_Q4_K_M" \
  --backend llamacpp \
  --profile standard \
  --model-ram-gb 4.8 \
  --notes "RTX 3060 12GB, 33 layers on GPU"
```

Anything that speaks the OpenAI chat API works: Ollama, LM Studio, vLLM, LocalAI, or your own
runner. Pass the footprint explicitly so the result is comparable:

```bash
slm-bench evaluate \
  --model-id "qwen2.5-coder:7b" \
  --backend openai \
  --endpoint-url "http://localhost:11434/v1" \
  --profile standard \
  --model-ram-gb 4.8 \
  --notes "Ollama on an M2 MacBook Air, 16GB"
```

For models supporting explicit reasoning effort (e.g. reasoning models with thinking modes), pass `--reasoning-effort`:

```bash
slm-bench evaluate \
  --model-id "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --backend openai \
  --endpoint-url "http://localhost:11434/v1" \
  --profile standard \
  --reasoning-effort medium
```

Batch evaluate multiple models listed in a text or JSON file:

```bash
slm-bench batch --batch-file models.txt --backend openai --profile quick
```

Results are written to `Website/data/results/<model-slug>.json` when that directory exists,
otherwise to `data/results/`. Every file contains the scores, per-test sample counts, tokens per
second where the backend reports it, and the host telemetry.

## Submit a result

Automated:

```bash
slm-bench evaluate \
  --model-id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --quantization "GGUF_Q4_K_M" \
  --backend llamacpp \
  --profile standard \
  --model-ram-gb 9.5 \
  --notes "Apple M3 Max, Metal, 96GB unified memory" \
  --submit \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

Or submit an already completed evaluation JSON:

```bash
slm-bench submit \
  --file "Website/data/results/DeepSeek_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json" \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

That clones the repository, creates an `eval/<model-slug>` branch, commits the JSON, pushes, and
opens a pull request with the hardware summary filled in.

Manual, if you would rather control the diff:

1. Run the evaluation.
2. Check the JSON: the GPU, VRAM, CPU, RAM, OS, and (where available) driver are present, and the
   sample counts are not zero.
3. Run `python Website/scripts/aggregate.py` to confirm it compiles.
4. Branch, commit the result file, push, and open a pull request. The template in
   `Website/PR_TEMPLATE.md` is preloaded by GitHub.

The browser can check the required fields too: open **Submit a result → Check JSON** on the site.

## The site, locally

No Node, no bundler:

```bash
python Website/scripts/aggregate.py   # rebuilds Website/data/leaderboard.json
python -m http.server -d Website 8000
```

Then open <http://localhost:8000>. If the JSON cannot be fetched — for example when opening
`index.html` directly from disk — the page falls back to an embedded dataset so the table, charts,
and inspector still work.

## Repository layout

```text
.
├── .github/workflows/
│   ├── ci.yml                    # pytest matrix on every push and pull request
│   ├── deploy-pages.yml          # aggregates results, deploys Website/ to Pages
│   └── validate-submission.yml   # validates result schema and telemetry on PRs
├── slm_bench/
│   ├── backends/                 # llama.cpp, vLLM, transformers, OpenAI-compatible, mock
│   ├── benchmarks/               # benchmark suites (reasoning, coding, instruction, tools, context, safety)
│   ├── execution/                # inference gateway and streaming runtime
│   ├── metrics/                  # telemetry monitoring & Wilson confidence intervals
│   ├── models/                   # hardware detection, RAM estimator, HF metadata, Pydantic schema
│   ├── scoring/                  # composite scoring engine & quantization analysis
│   ├── submission/               # branch + pull request automation
│   ├── utils/                    # logging & helper utilities
│   ├── website_template/         # optional scaffold for a new leaderboard
│   ├── tui.py                    # interactive Textual studio application
│   └── cli.py                    # `slm-bench`
├── tests/                        # unit tests for the harness, TUI, and submission flow
├── Website/
│   ├── data/results/             # one JSON file per evaluation
│   ├── data/leaderboard.json     # generated aggregate
│   ├── scripts/aggregate.py      # results -> leaderboard
│   ├── scripts/generate_sample_data.py
│   ├── index.html                # the ledger
│   ├── style.css                 # paper-and-ink stylesheet
│   └── app.js                    # filtering, sorting, charts, inspector
├── pyproject.toml
└── LICENSE
```

## Tests

```bash
pytest -v
```

## A note on the numbers

This is a small, opinionated benchmark, not a peer-reviewed evaluation. A two-point gap between two
models is usually noise. Read the per-test scores, the sample counts, and the machine before drawing
a conclusion. Results without telemetry or sample counts are demos, not measurements.

If you find a result that looks wrong, open an issue with the evaluation ID. Corrections are more
useful than silent re-runs.

## License

Harness code: [MIT](LICENSE). Benchmark data contributed by the community stays under the license
of the model it describes and is published for comparison, not as a warranty of fitness.
