# Attribution Analysis Visualizer

React/Vite frontend for inspecting token-level saliency and training correlation reports.

## Run Locally

```bash
cd tools/correlation-report
pnpm install
pnpm dev
```

Open `http://localhost:5173`.

For static deployment:

```bash
cd tools/correlation-report
pnpm run build
cd dist
python3 -m http.server 5173 --bind 0.0.0.0
```

The built app reads bundled experiment files from `dist/data/` and also supports browser-side JSON import.

## Importing External Saliency Files

The New View has an import panel at the top:

- Drag a JSON file into the panel, or choose a local JSON file.
- Paste a JSON URL and click `Load URL`.
- Open a report URL directly with `?reportUrl=...`, for example:

```text
https://your-site.example/visualizer/?reportUrl=https%3A%2F%2Fexample.com%2Fmy_saliency.json
```

Remote `reportUrl` imports require the JSON host to allow browser CORS requests.

This is a static frontend import path. The file is parsed in the browser and is not saved to the server. A persistent `POST /api/upload-report` endpoint would require a separate backend and is not included here.

## Supported Formats

### 1. Native all-token correlation report

The native format is the output used by this repository, usually named:

```text
correlation_matching_results_test{N}_all_tokens.json
```

Minimal shape:

```json
{
  "experiment_meta": {
    "test_sample_index": 58,
    "mode": "all_tokens",
    "tokens_analyzed": 2
  },
  "test_sample_baseline": {
    "full_tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn", "Ġbar"],
    "correct_full_tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn", "Ġbaz"],
    "prompt_len": 6
  },
  "per_token_results": [
    {
      "target_token_index": 7,
      "target_token": "Ġreturn",
      "top_correlations": [
        {
          "source_token": "def",
          "source_token_index": 0,
          "target_token": "Ġreturn",
          "target_token_index": 7,
          "saliency_score": 0.42
        }
      ],
      "correlation_pairs": []
    }
  ],
  "train_sample_details": {}
}
```

If `correlation_pairs` and `train_sample_details` are present, the right-side training correlation panel is enabled. If they are empty, the viewer still shows token saliency and top source tokens.

### 2. Generic saliency-only format

Use this when an external method only has one saliency vector per target token.

```json
{
  "sample_id": "case-001",
  "tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn", "Ġbar"],
  "prompt_len": 6,
  "correct_tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn", "Ġbaz"],
  "saliency_list": [
    {
      "target_token_index": 7,
      "scores": [0.12, 0.04, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0]
    },
    {
      "target_token_index": 8,
      "scores": [0.03, 0.22, 0.0, 0.0, 0.0, 0.0, 0.08, 0.11, 0.0]
    }
  ]
}
```

Field rules:

- `tokens`: tokenizer tokens aligned with every saliency vector. GPT-style whitespace markers such as `Ġ` and newline markers such as `Ċ` are decoded by the viewer.
- `prompt_len`: number of prompt tokens. Target tokens are normally at indices `>= prompt_len`.
- `correct_tokens`: optional reference tokens. If omitted, the viewer uses `tokens` as the reference.
- `saliency_list[].target_token_index`: index of the target token being explained.
- `saliency_list[].scores`: saliency scores aligned to `tokens`; `scores[i]` is the source-token score for token `tokens[i]`.

Accepted aliases:

- `full_tokens` or `token_list` instead of `tokens`.
- `start_index` or `answer_start_index` instead of `prompt_len`.
- `saliency`, `saliencies`, or `targets` instead of `saliency_list`.
- `index` or `target_index` instead of `target_token_index`.
- `saliency`, `saliency_scores`, or `source_scores` instead of `scores`.

Object-map form is also accepted:

```json
{
  "tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn"],
  "prompt_len": 6,
  "saliency_by_target": {
    "7": [0.12, 0.04, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0]
  }
}
```

### 3. Legacy `latest_saliency.json`

The older compare-view saliency file is accepted for import:

```json
{
  "target_test_sample": {
    "before": {
      "full_tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn"],
      "start_index": 6,
      "saliency_list": [
        {
          "index": 7,
          "saliency": [0.12, 0.04, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0]
        }
      ]
    }
  }
}
```

The importer converts this to the generic saliency-only view.
