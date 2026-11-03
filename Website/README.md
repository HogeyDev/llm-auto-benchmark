# Website Leaderboard

The static leaderboard in `Website/` is the public dashboard for the benchmark. It is built with vanilla HTML, CSS, and JavaScript — no framework, no build tool, no server — deployed directly to GitHub Pages.

- **Live URL**: <https://seandolan0.github.io/llm-auto-benchmark/>
- **Test Harness Documentation**: See the [Root README](../README.md)

---

## Benchmark Integrity on the Leaderboard

The leaderboard strictly enforces benchmark integrity:
1. **Official Benchmarks Only**: Ranks models evaluated on authentic upstream benchmarks:
   - **Reasoning**: MMLU-Pro (12,032 test items), GPQA Diamond (198 questions), BBH (23 tasks)
   - **Mathematics**: GSM8K (1,319 problems), MATH-500 (500 competition problems)
   - **Instruction Following**: IFEval (541 verifiable constraint prompts)
   - **Coding**: EvalPlus (HumanEval+ / MBPP+)
   - **Tool Calling**: Berkeley Function Calling Leaderboard (BFCL v4)
2. **Official Splitting & Status Verification**:
   - **`FULL`**: Run evaluated on complete official test splits. Only `FULL` evaluations are assigned an official rank and displayed by default.
   - **`PARTIAL`**: Evaluations run on subsets or sample limits. Transparently flagged and separated via the **Eligibility Filter**.
   - **`INVALID` / `DEPRECATED`**: Any legacy custom benchmark runs are marked deprecated and excluded from the active leaderboard.
3. **Hardware Transparency**:
   - Every entry displays the physical hardware: GPU name, VRAM, system RAM, CPU, OS, quantization format, and actual memory footprint.
   - Telemetry tracks decode tokens per second (TPS) and Time-to-First-Token (TTFT).
4. **Statistical Confidence**:
   - Overall scores and individual benchmark accuracies display **95% Wilson confidence intervals**.

---

## Local Development & Compilation

To recompile the aggregate leaderboard dataset from individual result files:

```bash
# Aggregates Website/data/results/*.json into Website/data/leaderboard.json and index.json
python Website/scripts/aggregate.py

# Launch local preview server
python -m http.server -d Website 8000
```

Open <http://localhost:8000> in your browser.

---

## Directory Structure

```text
Website/
├── index.html                 # Main leaderboard table, filter controls, figures, inspection modal
├── style.css                  # Paper-and-ink typography and layout
├── app.js                     # Filtering, sorting, Chart.js graphs, run inspector, JSON validator
├── data/
│   ├── leaderboard.json       # Aggregated leaderboard database
│   └── results/               # Individual model evaluation JSON results (Schema 2.0.0)
│       ├── index.json
│       ├── Qwen_Qwen2.5-7B-Instruct-BF16.json
│       ├── deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json
│       └── meta-llama_Llama-3.2-3B-Instruct-BF16.json
├── scripts/
│   ├── aggregate.py           # Result aggregator and Wilson CI calculator
│   └── generate_sample_data.py # Sample data generator for Schema 2.0.0
└── README.md
```

---

## Submitting New Evaluation Results

1. Run an evaluation with `slm-bench evaluate` using the `standard` or `extended` profile.
2. Confirm the resulting JSON file has `status: "FULL"` and `eligible_for_leaderboard: true`.
3. Submit via automated PR:
   ```bash
   slm-bench submit --result-file Website/data/results/<model-slug>.json --github-repo SeanDolan0/llm-auto-benchmark
   ```
4. GitHub Actions CI automatically verifies schema conformance and telemetry integrity before merging.
