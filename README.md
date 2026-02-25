# hbb_cli

`hbb_cli`는 프롬프트 데이터셋(`.jsonl`)을 기반으로 attention head resampling 실험을 실행하고, head별/프롬프트별 지표를 저장하는 CLI 프로젝트입니다.

## 요구 사항

- Python 3.10+
- CUDA 사용 시 NVIDIA GPU + CUDA 환경

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## 실행

### 1) 모든 head 스캔

```bash
python3 scripts/head_mining.py \
  --scan-all-heads \
  --dataset-root datasets \
  --output-dir outputs
```

### 2) 특정 head 세트 평가

```bash
python3 scripts/head_mining.py \
  --multi-heads L1.H2,L3.H5 \
  --dataset-root datasets \
  --output-dir outputs
```

## 데이터 형식

- 입력: `datasets/**.jsonl`
- 각 줄은 아래 중 하나여야 합니다.
  - `{"prompt": "..."}`
  - `"..."` (문자열 단독 라인)

## 결과물

- 기본 출력 경로: `outputs/`
- 주요 산출물:
  - `summary_by_head.csv`, `summary_by_head.jsonl`
  - `prompt_by_head.csv`, `prompt_by_head.jsonl`
  - (특정 head 세트 실행 시) `prompt_metrics_*.csv/json`, `summary_*.csv/json`

## GitHub 업로드 예시

```bash
git add .
git commit -m "Initialize hbb_cli for GitHub"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
