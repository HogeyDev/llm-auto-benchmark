# Model Submission Pull Request

Thank you for contributing benchmark evaluations to the AI Model Benchmark Leaderboard! Please fill out the sections below to ensure your submission is verified and merged promptly.

---

### 1. Submission Pre-Flight Checklist
Please verify each of the following before requesting review:
- [ ] **Everyday Hardware Scope:** The model can be reasonably run on common local hardware (consumer GPUs 6GB–16GB, Apple Silicon, or system RAM).
- [ ] **Hardware Specs Included:** The evaluation JSON records real host hardware telemetry (GPU, VRAM, CPU, RAM).
- [ ] **Standardized Schema:** The benchmark result JSON file conforms to the schema in `data/results/*.json`.
- [ ] **File Location:** The JSON result file has been placed in `data/results/<organization>_<model_slug>.json` (or `website/data/results/`).
- [ ] **Reproducible Settings:** The evaluation was executed using the benchmark harness without prompt gaming or modified ground truth.
- [ ] **Open Weights:** The model weights are publicly accessible on the Hugging Face Hub under an open or permissive community license.
- [ ] **Clean Git History:** Only the new/updated JSON file in `data/results/` is included in this PR (no unintended scratch files).

---

### 2. Model Metadata
| Field | Value |
|:---|:---|
| **Model ID** | `e.g. Qwen/Qwen2.5-7B-Instruct` |
| **Hugging Face Hub URL** | https://huggingface.co/`org/model` |
| **Exact Parameter Count** | `e.g. 7.61B` |
| **Parameter Tier** | `< 3B` / `3B–7B` / `7B–14B` / `14B–32B` / `32B+` |
| **Architecture** | `e.g. Qwen2ForCausalLM / LlamaForCausalLM` |
| **Quantization / Precision**| `BF16 / AWQ / GGUF Q4_K_M / GGUF Q8_0 / NF4` |
| **Model License** | `e.g. Apache-2.0 / MIT / Llama-3.1 Community` |
| **Context Length Tested** | `e.g. 32,768 tokens` |

---

### 3. Hardware & Runtime Telemetry
To ensure transparency and reproducibility, please provide details on the hardware used during this run:

| Resource | Value |
|:---|:---|
| **GPU / Accelerator** | `e.g. NVIDIA RTX 4090 (24GB) or Apple M3 Max` |
| **GPU Count & VRAM** | `e.g. 1 GPU, 24.0 GB VRAM` |
| **CPU Model & Cores** | `e.g. AMD Ryzen 9 7950X (16 cores, 32 threads)` |
| **System RAM** | `e.g. 64 GB DDR5` |
| **Operating System** | `e.g. Ubuntu 24.04 LTS / Windows 11` |
| **CUDA & Driver Version** | `e.g. CUDA 12.4, Driver 550.54.14` |
| **Inference Backend** | `vLLM / llama.cpp / transformers / AutoAWQ / Ollama` |

---

### 4. Benchmark Score Summary
Fill in the scores reported in your generated JSON file:

| Benchmark | Metric | Score (%) | Samples Passed / Total |
|:---|:---|:---:|:---:|
| **Overall Composite** | Weighted Average | **0.0%** | -- |
| **GSM8K** | Math Reasoning (strict-acc) | 0.0% | / 1319 |
| **HumanEval** | Python Coding (pass@1) | 0.0% | / 164 |
| **ARC Challenge** | Scientific Reasoning (acc_norm) | 0.0% | / 1172 |
| **IFEval** | Instruction Following (strict) | 0.0% | / 541 |
| **Tool Calling** | Multiturn & Schema Accuracy | 0.0% | / 300 |

---

### 5. Proof of Run & Evaluation Logs
Paste a snippet of your terminal evaluation log below (including timestamps, device initialization, and final summary table):

```text
[2026-11-02 12:00:00] Initialized inference engine with model: <model_id>
[2026-11-02 12:05:10] Task gsm8k completed: ...
[2026-11-02 12:12:30] Task humaneval completed: ...
[2026-11-02 12:18:45] Task arc completed: ...
[2026-11-02 12:22:10] Task ifeval completed: ...
[2026-11-02 12:26:00] Task tool_calling completed: ...
[2026-11-02 12:26:02] Benchmark suite finished. Overall Score: XX.X%
```

---

### 6. Submitter Attribution
- **Submitter Name:** (e.g. Jane Doe)
- **GitHub Handle:** @username
- **Affiliation / Organization:** (Independent / University / Lab)
- **Declaration:** I confirm that these evaluations were generated truthfully from the specified model weights.
