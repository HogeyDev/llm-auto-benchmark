#!/usr/bin/env python3
"""
Populates data/results/ with high-quality, realistic benchmark results for top open weights models.
Covers GSM8K, HumanEval, ARC, IFEval, and Tool Calling across multiple tiers, quantizations, and licenses.
"""

import json
from pathlib import Path

SAMPLE_MODELS = [
    {
        "filename": "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M.json",
        "evaluation_id": "eval-dsk-r1-14b-20261101",
        "timestamp": "2026-11-01T14:22:10Z",
        "submitter": {
            "name": "Alex Chen",
            "github_handle": "alexchen-ai",
            "affiliation": "Independent ML Researcher"
        },
        "model": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "model_slug": "DeepSeek-R1-Distill-Qwen-14B-GGUF_Q4_K_M",
            "hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "param_count_b": 14.7,
            "param_tier": "7B–14B",
            "architecture": "Qwen2ForCausalLM",
            "quantization": "GGUF Q4_K_M",
            "context_length": 65536,
            "license": "MIT",
            "backend": "llama.cpp"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "gpu_vram_gb": 24.0,
            "cpu": "AMD Ryzen 9 7950X 16-Core Processor",
            "cpu_cores": 16,
            "total_ram_gb": 64.0,
            "os": "Ubuntu 24.04 LTS (Linux 6.8.0-generic)",
            "cuda_version": "12.4",
            "driver_version": "550.54.14"
        },
        "summary_scores": {
            "overall_score": 87.65,
            "gsm8k": 94.20,
            "humaneval": 86.50,
            "arc": 92.40,
            "ifeval": 81.30,
            "tool_calling": 79.80
        },
        "tasks": {
            "gsm8k": {
                "display_name": "GSM8K (Grade School Math)",
                "category": "Math Reasoning",
                "score": 94.20,
                "metric": "strict_acc",
                "total_samples": 1319,
                "passed_samples": 1242,
                "avg_tokens_per_sec": 38.6,
                "avg_latency_ms": 285.4,
                "sample_preview": "Q: Natalia sold clips to 48 of her friends in April... A: The answer is 72. [CORRECT]"
            },
            "humaneval": {
                "display_name": "HumanEval (Python Coding)",
                "category": "Code Generation",
                "score": 86.50,
                "metric": "pass@1",
                "total_samples": 164,
                "passed_samples": 142,
                "avg_tokens_per_sec": 36.8,
                "avg_latency_ms": 312.0,
                "sample_preview": "Task 045: triangle_area(a, h) -> passed all unit test assertions."
            },
            "arc": {
                "display_name": "ARC Challenge (Reasoning)",
                "category": "Scientific Reasoning",
                "score": 92.40,
                "metric": "acc_norm",
                "total_samples": 1172,
                "passed_samples": 1083,
                "avg_tokens_per_sec": 41.2,
                "avg_latency_ms": 195.0,
                "sample_preview": "ARC-Challenge #812: Multi-step thermodynamic hypothesis validation [PASSED]"
            },
            "ifeval": {
                "display_name": "IFEval (Instruction Following)",
                "category": "Instruction Adherence",
                "score": 81.30,
                "metric": "strict_prompt_acc",
                "total_samples": 541,
                "passed_samples": 440,
                "avg_tokens_per_sec": 39.5,
                "avg_latency_ms": 240.0,
                "sample_preview": "Prompt: Write 4 paragraphs with exactly 3 bullet points in markdown [PASSED]"
            },
            "tool_calling": {
                "display_name": "Tool Calling & Function Execution",
                "category": "Agentic Tool Use",
                "score": 79.80,
                "metric": "schema_and_execution_acc",
                "total_samples": 300,
                "passed_samples": 239,
                "avg_tokens_per_sec": 35.1,
                "avg_latency_ms": 330.0,
                "sample_preview": "Call: get_current_weather(location='Tokyo', unit='celsius') -> Valid JSON schema invocation [PASSED]"
            }
        },
        "raw_logs_preview": "[2026-11-01 14:05:00] Initialized llama.cpp backend with -ngl 99\n[2026-11-01 14:05:02] Model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B loaded (VRAM allocated: 9.8 GB)\n[2026-11-01 14:08:15] Task gsm8k completed: 1242/1319 (94.20%)\n[2026-11-01 14:12:40] Task humaneval completed: 142/164 (86.50%)\n[2026-11-01 14:16:11] Task arc completed: 1083/1172 (92.40%)\n[2026-11-01 14:19:05] Task ifeval completed: 440/541 (81.30%)\n[2026-11-01 14:22:08] Task tool_calling completed: 239/300 (79.80%)\n[2026-11-01 14:22:10] Benchmark suite finished. Overall Score: 87.65%"
    },
    {
        "filename": "Qwen_Qwen2.5-32B-Instruct-GGUF_Q4_K_M.json",
        "evaluation_id": "eval-qwen25-32b-20261028",
        "timestamp": "2026-10-28T09:15:33Z",
        "submitter": {
            "name": "Sarah Lin",
            "github_handle": "slin-openllm",
            "affiliation": "Hugging Face Community"
        },
        "model": {
            "model_id": "Qwen/Qwen2.5-32B-Instruct",
            "model_slug": "Qwen2.5-32B-Instruct-GGUF_Q4_K_M",
            "hf_url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
            "param_count_b": 32.5,
            "param_tier": "14B–32B",
            "architecture": "Qwen2ForCausalLM",
            "quantization": "GGUF Q4_K_M",
            "context_length": 131072,
            "license": "Apache-2.0",
            "backend": "vLLM"
        },
        "hardware": {
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "gpu_count": 1,
            "gpu_vram_gb": 48.0,
            "cpu": "Intel Xeon w9-3495X 56-Cores",
            "cpu_cores": 56,
            "total_ram_gb": 128.0,
            "os": "Red Hat Enterprise Linux 9.4",
            "cuda_version": "12.3",
            "driver_version": "545.23.08"
        },
        "summary_scores": {
            "overall_score": 86.90,
            "gsm8k": 91.60,
            "humaneval": 84.80,
            "arc": 90.10,
            "ifeval": 85.20,
            "tool_calling": 81.40
        },
        "tasks": {
            "gsm8k": {
                "display_name": "GSM8K (Grade School Math)",
                "category": "Math Reasoning",
                "score": 91.60,
                "metric": "strict_acc",
                "total_samples": 1319,
                "passed_samples": 1208,
                "avg_tokens_per_sec": 31.4,
                "avg_latency_ms": 340.0
            },
            "humaneval": {
                "display_name": "HumanEval (Python Coding)",
                "category": "Code Generation",
                "score": 84.80,
                "metric": "pass@1",
                "total_samples": 164,
                "passed_samples": 139,
                "avg_tokens_per_sec": 29.8,
                "avg_latency_ms": 380.0
            },
            "arc": {
                "display_name": "ARC Challenge (Reasoning)",
                "category": "Scientific Reasoning",
                "score": 90.10,
                "metric": "acc_norm",
                "total_samples": 1172,
                "passed_samples": 1056,
                "avg_tokens_per_sec": 34.2,
                "avg_latency_ms": 260.0
            },
            "ifeval": {
                "display_name": "IFEval (Instruction Following)",
                "category": "Instruction Adherence",
                "score": 85.20,
                "metric": "strict_prompt_acc",
                "total_samples": 541,
                "passed_samples": 461,
                "avg_tokens_per_sec": 32.0,
                "avg_latency_ms": 290.0
            },
            "tool_calling": {
                "display_name": "Tool Calling & Function Execution",
                "category": "Agentic Tool Use",
                "score": 81.40,
                "metric": "schema_and_execution_acc",
                "total_samples": 300,
                "passed_samples": 244,
                "avg_tokens_per_sec": 30.5,
                "avg_latency_ms": 350.0
            }
        },
        "raw_logs_preview": "[2026-10-28 09:00:00] Initialized vLLM engine with Qwen2.5-32B-Instruct\n[2026-10-28 09:15:30] Full evaluation completed on 5 benchmarks. Average accuracy 86.90%."
    },
    {
        "filename": "Qwen_Qwen2.5-Coder-7B-Instruct-BF16.json",
        "evaluation_id": "eval-qwen25-coder-7b-bf16",
        "timestamp": "2026-10-29T18:05:00Z",
        "submitter": {
            "name": "David Miller",
            "github_handle": "dmiller-code",
            "affiliation": "CodeAI Labs"
        },
        "model": {
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "model_slug": "Qwen2.5-Coder-7B-Instruct-BF16",
            "hf_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
            "param_count_b": 7.61,
            "param_tier": "7B–14B",
            "architecture": "Qwen2ForCausalLM",
            "quantization": "BF16",
            "context_length": 131072,
            "license": "Apache-2.0",
            "backend": "transformers"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4080",
            "gpu_count": 1,
            "gpu_vram_gb": 16.0,
            "cpu": "Intel Core i9-14900K",
            "cpu_cores": 24,
            "total_ram_gb": 64.0,
            "os": "Ubuntu 24.04 LTS",
            "cuda_version": "12.4",
            "driver_version": "550.67"
        },
        "summary_scores": {
            "overall_score": 83.55,
            "gsm8k": 88.50,
            "humaneval": 88.40,
            "arc": 83.20,
            "ifeval": 78.40,
            "tool_calling": 76.50
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 88.50, "passed_samples": 1167, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 88.40, "passed_samples": 145, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 83.20, "passed_samples": 975, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 78.40, "passed_samples": 424, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 76.50, "passed_samples": 230, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-29 18:00:00] Evaluating Qwen2.5-Coder-7B in BF16 unquantized mode.\n[2026-10-29 18:04:50] HumanEval pass@1 reached 88.40% (state-of-the-art for 7B)."
    },
    {
        "filename": "Qwen_Qwen2.5-7B-Instruct-GGUF_Q4_K_M.json",
        "evaluation_id": "eval-qwen25-7b-gguf-q4",
        "timestamp": "2026-11-02T10:11:00Z",
        "submitter": {
            "name": "Marcus Vance",
            "github_handle": "marcus-vance",
            "affiliation": "Open Weights Alliance"
        },
        "model": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_slug": "Qwen2.5-7B-Instruct-GGUF_Q4_K_M",
            "hf_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "param_count_b": 7.61,
            "param_tier": "7B–14B",
            "architecture": "Qwen2ForCausalLM",
            "quantization": "GGUF Q4_K_M",
            "context_length": 32768,
            "license": "Apache-2.0",
            "backend": "llama.cpp"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_count": 1,
            "gpu_vram_gb": 24.0,
            "cpu": "AMD Ryzen 9 5900X",
            "cpu_cores": 12,
            "total_ram_gb": 32.0,
            "os": "Windows 11 (10.0.26100)",
            "cuda_version": "12.2",
            "driver_version": "551.86"
        },
        "summary_scores": {
            "overall_score": 82.10,
            "gsm8k": 86.80,
            "humaneval": 82.30,
            "arc": 85.10,
            "ifeval": 81.00,
            "tool_calling": 72.80
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 86.80, "passed_samples": 1145, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 82.30, "passed_samples": 135, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 85.10, "passed_samples": 997, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 81.00, "passed_samples": 438, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 72.80, "passed_samples": 218, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-11-02 10:00:15] llama.cpp evaluation finished for Qwen2.5-7B GGUF Q4_K_M."
    },
    {
        "filename": "google_gemma-2-9b-it-BF16.json",
        "evaluation_id": "eval-gemma2-9b-it-bf16",
        "timestamp": "2026-10-25T11:40:00Z",
        "submitter": {
            "name": "Elena Rostova",
            "github_handle": "elena-ml",
            "affiliation": "EuroLLM Working Group"
        },
        "model": {
            "model_id": "google/gemma-2-9b-it",
            "model_slug": "gemma-2-9b-it-BF16",
            "hf_url": "https://huggingface.co/google/gemma-2-9b-it",
            "param_count_b": 9.24,
            "param_tier": "7B–14B",
            "architecture": "Gemma2ForCausalLM",
            "quantization": "BF16",
            "context_length": 8192,
            "license": "Gemma Terms of Use",
            "backend": "vLLM"
        },
        "hardware": {
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_count": 1,
            "gpu_vram_gb": 80.0,
            "cpu": "AMD EPYC 7742",
            "cpu_cores": 32,
            "total_ram_gb": 128.0,
            "os": "Ubuntu 22.04 LTS",
            "cuda_version": "12.4",
            "driver_version": "550.54"
        },
        "summary_scores": {
            "overall_score": 81.85,
            "gsm8k": 84.20,
            "humaneval": 74.40,
            "arc": 88.60,
            "ifeval": 84.10,
            "tool_calling": 76.50
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 84.20, "passed_samples": 1111, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 74.40, "passed_samples": 122, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 88.60, "passed_samples": 1038, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 84.10, "passed_samples": 455, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 76.50, "passed_samples": 230, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-25 11:20:00] Executed vLLM benchmark on Google Gemma 2 9B IT in BF16 precision."
    },
    {
        "filename": "meta-llama_Llama-3.1-8B-Instruct-GGUF_Q8_0.json",
        "evaluation_id": "eval-llama31-8b-gguf-q8",
        "timestamp": "2026-10-22T16:50:00Z",
        "submitter": {
            "name": "Tobias Huber",
            "github_handle": "tobi-huber",
            "affiliation": "LocalAI Initiative"
        },
        "model": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "model_slug": "Llama-3.1-8B-Instruct-GGUF_Q8_0",
            "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
            "param_count_b": 8.03,
            "param_tier": "7B–14B",
            "architecture": "LlamaForCausalLM",
            "quantization": "GGUF Q8_0",
            "context_length": 131072,
            "license": "Llama 3.1 Community",
            "backend": "llama.cpp"
        },
        "hardware": {
            "gpu_name": "Apple M3 Max",
            "gpu_count": 1,
            "gpu_vram_gb": 48.0,
            "cpu": "Apple M3 Max (16-core)",
            "cpu_cores": 16,
            "total_ram_gb": 48.0,
            "os": "macOS 15.1 Sequoia",
            "cuda_version": "Metal 3",
            "driver_version": "N/A"
        },
        "summary_scores": {
            "overall_score": 80.95,
            "gsm8k": 84.50,
            "humaneval": 72.60,
            "arc": 85.40,
            "ifeval": 83.20,
            "tool_calling": 77.30
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 84.50, "passed_samples": 1115, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 72.60, "passed_samples": 119, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 85.40, "passed_samples": 1001, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 83.20, "passed_samples": 450, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 77.30, "passed_samples": 232, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-22 16:30:10] Metal unified memory acceleration enabled. Run completed."
    },
    {
        "filename": "mistralai_Mistral-Nemo-Instruct-2407-12B-NF4.json",
        "evaluation_id": "eval-mistral-nemo-12b-nf4",
        "timestamp": "2026-10-20T08:12:00Z",
        "submitter": {
            "name": "Jean-Luc Picard",
            "github_handle": "jlpicard-ai",
            "affiliation": "French Open Source AI"
        },
        "model": {
            "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
            "model_slug": "Mistral-Nemo-Instruct-2407-12B-NF4",
            "hf_url": "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
            "param_count_b": 12.2,
            "param_tier": "7B–14B",
            "architecture": "MistralForCausalLM",
            "quantization": "NF4",
            "context_length": 128000,
            "license": "Apache-2.0",
            "backend": "bitsandbytes"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "gpu_count": 1,
            "gpu_vram_gb": 16.0,
            "cpu": "Intel Core i7-13700K",
            "cpu_cores": 16,
            "total_ram_gb": 32.0,
            "os": "Ubuntu 24.04 LTS",
            "cuda_version": "12.4",
            "driver_version": "550.54"
        },
        "summary_scores": {
            "overall_score": 78.80,
            "gsm8k": 78.40,
            "humaneval": 71.30,
            "arc": 83.90,
            "ifeval": 82.50,
            "tool_calling": 77.20
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 78.40, "passed_samples": 1034, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 71.30, "passed_samples": 117, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 83.90, "passed_samples": 983, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 82.50, "passed_samples": 446, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 77.20, "passed_samples": 232, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-20 08:00:00] bitsandbytes 4-bit NF4 quantization applied with bfloat16 compute."
    },
    {
        "filename": "microsoft_Phi-3.5-mini-instruct-AWQ_4BIT.json",
        "evaluation_id": "eval-phi35-mini-awq",
        "timestamp": "2026-10-18T13:45:00Z",
        "submitter": {
            "name": "Karthik Subramanian",
            "github_handle": "ksubramanian-dev",
            "affiliation": "Edge LLM Benchmarks"
        },
        "model": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "model_slug": "Phi-3.5-mini-instruct-AWQ_4BIT",
            "hf_url": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
            "param_count_b": 3.82,
            "param_tier": "3B–7B",
            "architecture": "Phi3ForCausalLM",
            "quantization": "AWQ",
            "context_length": 131072,
            "license": "MIT",
            "backend": "AutoAWQ"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4060",
            "gpu_count": 1,
            "gpu_vram_gb": 8.0,
            "cpu": "AMD Ryzen 7 7700",
            "cpu_cores": 8,
            "total_ram_gb": 32.0,
            "os": "Windows 11 (10.0.22631)",
            "cuda_version": "12.3",
            "driver_version": "546.33"
        },
        "summary_scores": {
            "overall_score": 76.70,
            "gsm8k": 82.10,
            "humaneval": 68.90,
            "arc": 82.40,
            "ifeval": 77.10,
            "tool_calling": 70.80
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 82.10, "passed_samples": 1083, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 68.90, "passed_samples": 113, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 82.40, "passed_samples": 966, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 77.10, "passed_samples": 417, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 70.80, "passed_samples": 212, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-18 13:30:00] AutoAWQ quantization verified. High efficiency in memory constrained setups."
    },
    {
        "filename": "meta-llama_Llama-3.2-3B-Instruct-BF16.json",
        "evaluation_id": "eval-llama32-3b-bf16",
        "timestamp": "2026-10-15T19:30:00Z",
        "submitter": {
            "name": "Clara Oswald",
            "github_handle": "clara-ml",
            "affiliation": "TinyModels Org"
        },
        "model": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "model_slug": "Llama-3.2-3B-Instruct-BF16",
            "hf_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
            "param_count_b": 3.21,
            "param_tier": "3B–7B",
            "architecture": "LlamaForCausalLM",
            "quantization": "BF16",
            "context_length": 131072,
            "license": "Llama 3.2 Community",
            "backend": "transformers"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4070",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cpu": "Intel Core i5-13600K",
            "cpu_cores": 14,
            "total_ram_gb": 32.0,
            "os": "Ubuntu 24.04 LTS",
            "cuda_version": "12.4",
            "driver_version": "550.54"
        },
        "summary_scores": {
            "overall_score": 73.65,
            "gsm8k": 77.20,
            "humaneval": 63.40,
            "arc": 78.50,
            "ifeval": 76.80,
            "tool_calling": 71.00
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 77.20, "passed_samples": 1018, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 63.40, "passed_samples": 104, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 78.50, "passed_samples": 920, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 76.80, "passed_samples": 415, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 71.00, "passed_samples": 213, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-15 19:15:00] Evaluated Llama-3.2-3B-Instruct in native BF16."
    },
    {
        "filename": "HuggingFaceTB_SmolLM2-1.7B-Instruct-BF16.json",
        "evaluation_id": "eval-smollm2-17b-bf16",
        "timestamp": "2026-10-12T14:10:00Z",
        "submitter": {
            "name": "Louisa Martin",
            "github_handle": "lmartin-hf",
            "affiliation": "Hugging Face"
        },
        "model": {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "model_slug": "SmolLM2-1.7B-Instruct-BF16",
            "hf_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "param_count_b": 1.71,
            "param_tier": "< 3B",
            "architecture": "LlamaForCausalLM",
            "quantization": "BF16",
            "context_length": 8192,
            "license": "Apache-2.0",
            "backend": "transformers"
        },
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 3060",
            "gpu_count": 1,
            "gpu_vram_gb": 12.0,
            "cpu": "AMD Ryzen 5 5600X",
            "cpu_cores": 6,
            "total_ram_gb": 32.0,
            "os": "Ubuntu 22.04 LTS",
            "cuda_version": "12.1",
            "driver_version": "535.154"
        },
        "summary_scores": {
            "overall_score": 67.30,
            "gsm8k": 68.50,
            "humaneval": 54.90,
            "arc": 72.80,
            "ifeval": 73.10,
            "tool_calling": 65.40
        },
        "tasks": {
            "gsm8k": {"display_name": "GSM8K", "score": 68.50, "passed_samples": 903, "total_samples": 1319},
            "humaneval": {"display_name": "HumanEval", "score": 54.90, "passed_samples": 90, "total_samples": 164},
            "arc": {"display_name": "ARC Challenge", "score": 72.80, "passed_samples": 853, "total_samples": 1172},
            "ifeval": {"display_name": "IFEval", "score": 73.10, "passed_samples": 395, "total_samples": 541},
            "tool_calling": {"display_name": "Tool Calling", "score": 65.40, "passed_samples": 196, "total_samples": 300}
        },
        "raw_logs_preview": "[2026-10-12 14:00:00] Evaluating SmolLM2-1.7B lightweight sub-3B model."
    }
]

def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for item in SAMPLE_MODELS:
        file_path = results_dir / item["filename"]
        # Remove internal filename property before saving
        data = {k: v for k, v in item.items() if k != "filename"}
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        print(f"Generated sample result: {file_path.name}")

if __name__ == "__main__":
    main()
