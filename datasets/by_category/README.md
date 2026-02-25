# Stable Head Mining Dataset Layout

Directory structure:

- `by_category/<category_name>/*.jsonl|*.json|*.txt`

Each JSONL line can be either:

- `{"prompt": "..."}` (recommended)
- `"..."` (plain string JSON)

Examples:

- `capitals/easy.jsonl`
- `antonyms/common.jsonl`
- `arithmetic/basic.jsonl`

CLI usage:

```bash
python3 scripts/stable_head_mining_cli.py \
  --dataset-root /home/head-bang-bang_cli/datasets/by_category \
  --categories capitals,antonyms \
  --intervention-mode resampling \
  --analysis-scope individual
```

Use all categories:

```bash
python3 scripts/stable_head_mining_cli.py \
  --dataset-root /home/head-bang-bang_cli/datasets/by_category \
  --categories all
```
