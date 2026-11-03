"""
Static GitHub Pages leaderboard website template generator (Version 2.0).
Scaffolds website assets and official benchmark results.
"""

from pathlib import Path
import json
import shutil

DEPLOY_WORKFLOW_YML = """name: Deploy Static Leaderboard to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './Website'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""

VALIDATE_WORKFLOW_YML = """name: Validate Benchmark Submission

on:
  pull_request:
    paths:
      - 'Website/data/results/**.json'
      - 'data/results/**.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install SLM-Bench validator
        run: |
          pip install pydantic typer

      - name: Validate JSON schemas
        run: |
          python -c "
          import json, sys, glob
          files = glob.glob('**/results/*.json', recursive=True)
          if not files:
              sys.exit(0)
          for f in files:
              with open(f, 'r', encoding='utf-8') as fp:
                  data = json.load(fp)
              score = data.get('slm_index', 0.0)
              if not (0.0 <= score <= 100.0):
                  print(f'ERROR: {f} invalid slm_index: {score}')
                  sys.exit(1)
              print(f'PASS: {f} - Valid official submission')
          "
"""


def scaffold_website(target_dir: Path) -> None:
    """Scaffold complete static website folder with canonical assets and sample results."""
    target_dir.mkdir(parents=True, exist_ok=True)
    results_dir = target_dir / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # If Website/ directory exists in project root, copy the latest production assets
    source_site = Path("Website")
    if source_site.exists():
        for item in ["index.html", "style.css", "app.js", "favicon.svg", "README.md", "PR_TEMPLATE.md"]:
            src = source_site / item
            if src.exists():
                shutil.copy2(src, target_dir / item)

        scripts_dir = target_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        if (source_site / "scripts").exists():
            for sf in (source_site / "scripts").glob("*.py"):
                shutil.copy2(sf, scripts_dir / sf.name)

    # Populate official Schema 2.0.0 sample models
    from ..models.schema import BenchmarkResult, ModelMetadata, HardwareInfo, TaskMetric, CategoryScore
    sample_res = BenchmarkResult(
        schema_version="2.0.0",
        evaluation_id="eval-official-sample-01",
        status="FULL",
        eligible_for_leaderboard=True,
        is_legacy=False,
        profile="standard",
        model=ModelMetadata(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            model_slug="Qwen-Qwen2.5-7B-Instruct-BF16",
            param_count_b=7.6,
            param_tier="3B–8B",
            architecture="Qwen2ForCausalLM",
            quantization="BF16",
            model_ram_gb=15.2,
            context_length=32768,
            license="Apache-2.0",
            backend="vLLM",
        ),
        hardware=HardwareInfo(
            os="Ubuntu 22.04 LTS",
            cpu="Intel Core i9-14900K",
            cpu_cores=24,
            total_ram_gb=64.0,
            gpu_name="NVIDIA RTX 4080",
            gpu_count=1,
            gpu_vram_gb=16.0,
            cuda_version="12.3",
        ),
        tasks={
            "mmlu_pro": TaskMetric(task_id="mmlu_pro", category="reasoning", status="FULL", raw_score=0.554, raw_metric_name="accuracy", normalized_score=55.4, total_samples=12032, passed_samples=6665, ci_lower=54.5, ci_upper=56.3),
            "gpqa_diamond": TaskMetric(task_id="gpqa_diamond", category="reasoning", status="FULL", raw_score=0.414, raw_metric_name="accuracy", normalized_score=41.4, total_samples=198, passed_samples=82, ci_lower=34.8, ci_upper=48.4),
            "gsm8k": TaskMetric(task_id="gsm8k", category="mathematics", status="FULL", raw_score=0.892, raw_metric_name="accuracy", normalized_score=89.2, total_samples=1319, passed_samples=1176, ci_lower=87.4, ci_upper=90.8),
            "math500": TaskMetric(task_id="math500", category="mathematics", status="FULL", raw_score=0.772, raw_metric_name="accuracy", normalized_score=77.2, total_samples=500, passed_samples=386, ci_lower=73.3, ci_upper=80.7),
            "ifeval": TaskMetric(task_id="ifeval", category="instruction", status="FULL", raw_score=0.785, raw_metric_name="prompt_level_strict_accuracy", normalized_score=78.5, total_samples=541, passed_samples=425, ci_lower=74.8, ci_upper=81.8),
        },
        categories={
            "reasoning": CategoryScore(category="reasoning", weight=0.35, normalized_score=48.4, task_count=2, tasks=["mmlu_pro", "gpqa_diamond"]),
            "mathematics": CategoryScore(category="mathematics", weight=0.35, normalized_score=83.2, task_count=2, tasks=["gsm8k", "math500"]),
            "instruction": CategoryScore(category="instruction", weight=0.30, normalized_score=78.5, task_count=1, tasks=["ifeval"]),
        },
        slm_index=69.6,
        notes="Official standard profile sample run.",
    )
    (results_dir / "Qwen-Qwen2.5-7B-Instruct-BF16.json").write_text(sample_res.model_dump_json(indent=2), encoding="utf-8")

    # Write GitHub Actions workflows
    workflows_dir = target_dir.parent / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    (workflows_dir / "deploy-pages.yml").write_text(DEPLOY_WORKFLOW_YML, encoding="utf-8")
    (workflows_dir / "validate-submission.yml").write_text(VALIDATE_WORKFLOW_YML, encoding="utf-8")
