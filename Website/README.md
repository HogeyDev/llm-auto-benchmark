# Website

The static leaderboard in `Website/` is the public face of the benchmark. It is plain HTML, CSS, and
JavaScript — no framework, no bundler, no server — deployed to GitHub Pages from `main`.

- Live: <https://seandolan0.github.io/llm-auto-benchmark/>
- Harness and method: see the [root README](../README.md)

## How it loads data

1. It requests `data/leaderboard.json`, the aggregate built by `scripts/aggregate.py`.
2. If that is unavailable, it reads `data/results/index.json` and fetches each result file.
3. If both fail — for example when `index.html` is opened from disk and `fetch()` is blocked — it
   falls back to a dataset embedded in `app.js` so the table, charts, and inspector still work.

## Add a result

The harness writes one JSON file per evaluation to `data/results/`. To add one by hand:

```bash
python scripts/aggregate.py     # rebuild data/leaderboard.json
python -m http.server 8000      # preview at http://localhost:8000
```

The aggregator ranks by `scores.overall`, recomputes tier assignments, and writes both
`data/leaderboard.json` and `data/results/index.json`.

## Files

```text
Website/
├── index.html                 # the ledger, figures, contribution guide, modals
├── style.css                  # paper-and-ink design system
├── app.js                     # filtering, sorting, charts, inspector, validator
├── data/
│   ├── leaderboard.json       # generated aggregate
│   └── results/*.json         # one evaluation per file
├── scripts/
│   ├── aggregate.py           # results -> leaderboard.json
│   └── generate_sample_data.py
├── PR_TEMPLATE.md             # contributor template
└── README.md
```

## Design notes

The page is built as a printed report rather than a dashboard: warm paper, ink rules, one rust
accent, a serif display face, and monospace for anything numeric. That is a deliberate constraint,
not a theme. There are no gradients, blur panels, or icon cards, and no CDN CSS framework.

Fonts and Chart.js are loaded from public CDNs. Everything else is local. If you need a fully
offline build, vendor those three files and update the references in `index.html`.

## Editing without breaking the page

`app.js` renders the table rows, task bars, comparison table, and validator feedback as HTML
strings. The static page and the generated markup share class names from `style.css`. If you change
one, change the other. Every element ID used by `app.js` exists in `index.html`; keep it that way.

The in-browser checker only verifies required fields. Real schema validation happens in
`.github/workflows/validate-submission.yml` when a pull request is opened.
