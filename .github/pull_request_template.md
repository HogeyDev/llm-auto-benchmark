### What this adds

- **Model:** `org/model`
- **Evaluation Profile:** `standard` (or `quick` / `extended`)
- **Precision:** `GGUF Q4_K_M`
- **Hardware:** `NVIDIA RTX 3060 12GB` / `Apple M2, 16GB`
- **Result file:** `Website/data/results/<model-slug>.json`

### Checklist

- [ ] Open weights, and the license permits local use.
- [ ] Ran on consumer hardware: a consumer GPU, Apple Silicon, or system RAM.
- [ ] GPU, VRAM, CPU, RAM, OS, and telemetry (TTFT, decode TPS) are recorded in the JSON.
- [ ] Sample counts are non-zero and match the run.
- [ ] Only the result file changed in this pull request.

### Hardware & Telemetry

| Resource / Telemetry | Value |
|:--|:--|
| GPU | NVIDIA RTX 3060 12GB |
| Model Footprint | 4.8 GB RAM/VRAM |
| CPU | AMD Ryzen 5 5600X, 6 cores |
| System RAM | 32 GB |
| OS / driver | Ubuntu 24.04 / CUDA 12.4 / 550.54 |
| Backend | llama.cpp, 33 layers offloaded |
| Time to First Token (TTFT) | 145 ms |
| Generation Speed | 65.4 tok/s |

### Log tail

```text
[2026-11-02 10:00:15] Model loaded
[2026-11-02 10:11:02] Benchmark suite finished. Profile: standard. Composite SLM-Index: 82.10% (95% CI: [80.2%, 83.9%])
```

---

By opening this pull request I confirm the result was produced by running the model listed above
on the hardware described, without editing scores or sample counts.
