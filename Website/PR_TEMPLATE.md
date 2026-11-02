# Model Submission Pull Request Guide

Thank you for contributing benchmark evaluations to the AI Model Benchmark Leaderboard! Please fill out the sections below to ensure your submission is verified and merged promptly.

---

### 1. Submission Pre-Flight Checklist
Please verify each of the following before requesting review:
- [ ] **Small Local AI Scope:** The model qualifies as a small local AI (suitability reviewed and determined by maintainer).
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

### 3. Evaluation Summary Scores
| Benchmark Pillar | Metric | Score / Pass Rate |
|:---|:---|:---|
| **Overall Composite Score** | `SLM-Index` | `e.g. 82.10%` |
| **GSM8K (Math)** | Strict Accuracy | `e.g. 86.80%` |
| **HumanEval (Coding)** | Pass@1 | `e.g. 82.30%` |
| **ARC Challenge (Reasoning)** | Normalized Acc | `e.g. 85.10%` |
| **IFEval (Instruction)** | Strict Accuracy | `e.g. 81.00%` |
| **Tool Calling** | Schema Execution Acc| `e.g. 72.80%` |

---

### 4. Benchmark Hardware Environment
| Resource | Specification |
|:---|:---|
| **GPU Model & Count** | `e.g. 1x NVIDIA RTX 4090 (24GB VRAM)` |
| **CPU Model & Cores** | `e.g. AMD Ryzen 9 7950X (16 cores)` |
| **System RAM** | `e.g. 64 GB DDR5` |
| **Operating System** | `e.g. Ubuntu 24.04 LTS / Windows 11` |
| **CUDA & Driver Version** | `e.g. CUDA 12.4 / Driver 550.54.14` |
| **Inference Backend** | `llama.cpp / vLLM / Transformers / bitsandbytes` |

---

### 5. Execution Proof / Log Snippet
Please paste the final summary lines or console output from your evaluation harness run:
```text
[2026-11-01 14:05:00] Initialized inference engine
[2026-11-01 14:22:10] Benchmark suite finished. Overall Score: 87.65%
```
