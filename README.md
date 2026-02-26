# hbb_cli

`hbb_cli`는 attention head resampling 실험을 실행하고, 프롬프트별/head별 지표를 저장하는 CLI 프로젝트입니다.

## 지표
### 프롬프트 별
* delta : 변화량
* delta_direction : 변화하는 방향 (logit을 올리는지 내리는지)
* top1_changed : 실제 해당 head의 resampling을 통해 출력 값이 변하였는지
* dropped_from_top5 : 원래 출력값의 로짓이 top_5에서 밀려났는지(영향력이 큰가)
* baseline_top1_token(prob) : 원래 출력값과 그 확률
* resampled_top1_token(prob) : 바뀐 출력값과 그 확률 

### head 별
* prompt_count : 실험에 사용한 프롬프트 개수
* delta_mean : 변화량 평균
* delta_variance : 변화량 분산
* decrease_ratio : 전체 프롬프트 중 확률이 떨어진 것의 비율(1.0에 가까울수록 영향력이 큼)
* top1_changed_ratio : 전체 프롬프트 중 top1이 바뀐 것의 비율
* dropped_from_top5_ratio : 전체 프롬프트 중 top_5에서 밀려난 것의 비율
* dropped_from_top20_ratio : 전체 프롬프트 중 top_20에서 밀려난 것의 비율

## 요구 사항

- Python 3.10+
- CUDA 사용 시 NVIDIA GPU + CUDA 환경
- requirement.txt 참고

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

### 1-1) 특정 프롬프트 파일만 사용 (`--prompts-file`)

- `--prompts-file`를 주면 `--dataset-root`는 무시됩니다.
- `.jsonl` 파일 경로를 쉼표(`,`)로 연결해서 여러 파일을 동시에 넣을 수 있습니다.

```bash
python3 scripts/head_mining.py \
  --scan-all-heads \
  --prompts-file "datasets/by_category/country/general.jsonl,datasets/by_category/chemical/chemical_number.jsonl" \
  --output-dir outputs
```

### 2) 특정 head 세트 평가

```bash
python3 scripts/head_mining.py \
  --multi-heads L1.H2,L3.H5 \
  --dataset-root datasets \
  --output-dir outputs
```

## 결과물

- 기본 출력 경로: `outputs/`
- 주요 산출물:
  - `summary_by_head.csv`, `summary_by_head.jsonl`
  - `prompt_by_head.csv`, `prompt_by_head.jsonl`
  - (특정 head 세트 실행 시) `prompt_metrics_*.csv/json`, `summary_*.csv/json`

### `summary_by_head`가 안 생길 수 있는 경우

- 버킷(카테고리/소스 파일) 안 프롬프트가 2개 미만이면 해당 버킷은 스킵됩니다.
- `scan-all-heads`에서 summary 필터를 통과한 head가 하나도 없으면 새 `summary_by_head`가 생성되지 않을 수 있습니다.
  - 필터: `decrease_ratio >= threshold(0.8 -> 0.1 fallback)` 그리고 `delta_mean < -0.01`
- 재실행 시 이미 같은 `(head_label, prompt_count)` 키가 있으면 중복 추가하지 않습니다.

# 결과
### 아래링크 참고
https://headbb.vercel.app/

# 실험결과

## 1. capitals

### africa

- 프롬프트 예시: `What is the capital of Egypt? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|16|-0.2265|0.0116|1|0.3125|
|L17.H6|16|-0.0489|0.002|1|0.125|
|L18.H9|16|-0.0387|0.0032|0.8125|0.0625|
|L15.H3|16|-0.0334|0.0012|0.9375|0.125|
|L20.H11|16|-0.0262|0.0021|0.8125|0.0625|
|L22.H2|16|-0.0186|0.0012|0.8125|0|
|L13.H1|16|-0.0183|0.0005|0.8125|0.0625|
|L23.H9|16|-0.0125|0.0002|0.8125|0|
|L19.H7|16|-0.0106|0.0008|0.8125|0.0625|

### asia

- 프롬프트 예시: `What is the capital of Japan? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|25|-0.1683|0.0166|0.96|0.24|
|L15.H3|25|-0.0372|0.003|0.8|0.04|
|L22.H2|25|-0.0178|0.0006|0.88|0|
|L23.H13|25|-0.0103|0.0001|0.84|0|

### easy

- 프롬프트 예시: `What is the capital of Italy? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|28|-0.1994|0.0161|0.9643|0.3929|
|L17.H6|28|-0.067|0.002|0.9643|0.1071|
|L22.H2|28|-0.0174|0.0003|0.8929|0|
|L23.H9|28|-0.0137|0.0001|0.9286|0|
|L23.H13|28|-0.0118|0.0002|0.8214|0|

### europe

- 프롬프트 예시: `What is the capital of France? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|20|-0.2471|0.0153|1|0.25|
|L20.H11|20|-0.0373|0.002|0.85|0|
|L17.H6|20|-0.0334|0.0008|0.95|0.05|
|L13.H1|20|-0.0169|0.0005|0.8|0|
|L22.H2|20|-0.0153|0.0002|0.9|0|
|L23.H13|20|-0.0141|0.0002|0.8|0|

### north_america

- 프롬프트 예시: `What is the capital of the United States? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|16|-0.1287|0.0091|0.9375|0.375|
|L17.H6|16|-0.0432|0.0016|0.875|0.0625|
|L15.H3|16|-0.0391|0.0018|0.875|0|
|L13.H1|16|-0.022|0.001|0.8125|0.0625|
|L22.H2|16|-0.016|0.0002|1|0|
|L15.H11|16|-0.0135|0.001|0.8125|0|
|L23.H13|16|-0.0125|0.0003|0.8125|0|
|L13.H13|16|-0.0111|0.0005|0.8125|0|

### oceania

- 프롬프트 예시: `What is the capital of Solomon Islands? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|7|-0.0846|0.003|0.8571|0.4286|
|L20.H1|7|-0.0326|0.0016|1|0.5714|
|L17.H6|7|-0.0261|0.0009|0.8571|0|
|L16.H2|7|-0.0221|0.0004|1|0.1429|
|L13.H1|7|-0.016|0.0003|0.8571|0.1429|
|L16.H15|7|-0.0143|0|1|0.1429|
|L15.H3|7|-0.0122|0.0005|0.8571|0.1429|

### south_america

- 프롬프트 예시: `What is the capital of Argentina? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|11|-0.1854|0.0139|1|0.2727|
|L16.H2|11|-0.0655|0.0023|0.9091|0|
|L18.H9|11|-0.0398|0.0015|0.8182|0|
|L22.H2|11|-0.0342|0.0003|1|0.0909|
|L15.H9|11|-0.0272|0.0006|0.8182|0|
|L17.H6|11|-0.0265|0.0002|1|0|
|L19.H7|11|-0.0262|0.0012|0.8182|0|
|L15.H11|11|-0.0198|0.0009|0.9091|0.0909|
|L14.H7|11|-0.0185|0.0004|0.9091|0.0909|
|L23.H13|11|-0.016|0.0002|0.9091|0.0909|
|L16.H6|11|-0.0144|0.0005|0.8182|0.0909|
|L16.H1|11|-0.0102|0.0003|0.8182|0|

## 2. chemical

### chemical_symbols

- 프롬프트 예시: `The chemical symbol for Hydrogen is`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L13.H6|100|-0.1236|0.0207|0.84|0.19|
|L22.H2|100|-0.0556|0.0039|0.82|0.01|


## 3. logical

### order2

- 프롬프트 예시: `1, 2,`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L12.H0|38|-0.2335|0.0615|0.8421|0.6053|

### order3

- 프롬프트 예시: `1, 2, 3,`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L12.H0|32|-0.5218|0.0304|1|0.8438|
|L10.H7|32|-0.0874|0.0062|0.8438|0|
|L12.H7|32|-0.0656|0.0069|0.875|0|
|L22.H2|32|-0.0281|0.0008|0.9062|0|

## 4. mathematics

### add

- 프롬프트 예시: `Cal : 12+35=`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L12.H0|100|-0.2586|0.0304|0.91|0.78|
|L13.H1|100|-0.2113|0.0312|0.93|0.57|
|L13.H6|100|-0.1394|0.0382|0.8|0.41|
|L11.H10|100|-0.0957|0.0115|0.81|0.31|

### arithmetic_geometric_progression

- 프롬프트 예시: `Find the pattern: 3, 7, 15, 31, 63, 127,`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L13.H6|30|-0.0339|0.003|0.8|0.1333|

### arithmetic_progression

- 프롬프트 예시: `Find the pattern: 2, 5, 8, 11,`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L13.H6|30|-0.1044|0.0067|1|0.4333|
|L10.H7|30|-0.0271|0.001|0.8333|0.1333|
|L16.H14|30|-0.0157|0.0007|0.8|0.1333|

### constant

- 프롬프트 예시: `Output only the number. pi =`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L11.H11|50|-0.0115|0.0018|0.58|0.12|

### geometric_progression

- 프롬프트 예시: `Find the pattern: 2, 4, 8, 16,`

유의미한 헤드를 찾지 못함

### mul

- 프롬프트 예시: `Cal : 12*24=`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L13.H6|100|-0.0632|0.0126|0.8|0.41|

### sub

- 프롬프트 예시: `Cal : 39 minus 38=`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L13.H1|100|-0.1541|0.0158|0.96|0.67|
|L11.H10|100|-0.1099|0.0178|0.88|0.53|
|L12.H6|100|-0.0808|0.0129|0.8|0.37|
|L16.H14|100|-0.0774|0.014|0.8|0.49|

## 5. opposite

### opposite

- 프롬프트 예시: `The opposite of 'hot' is '`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L13.H2|100|-0.2209|0.0293|0.97|0.18|
|L14.H12|100|-0.0573|0.0057|0.8|0.07|
|L22.H0|100|-0.0259|0.0013|0.8|0.04|
|L23.H2|100|-0.018|0.0009|0.81|0.01|
|L23.H9|100|-0.0163|0.0004|0.8|0.02|

## 6. place

### architecture

- 프롬프트 예시: `The Eiffel Tower is in`

유의미한 헤드를 찾지 못함

### birthplace

- 프롬프트 예시: `Albert Einstein was born in`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L17.H0|50|-0.0213|0.001|0.8|0.26|

## 7. time

### historical_year

- 프롬프트 예시: `Answer using only a 4-digit number. World War I began in`

유의미한 헤드를 찾지 못함

## 8. country

### general

- 프롬프트 예시: `What country is Paris in? Answer:`

|head|prompt_count|delta_mean|delta_variance|decrease_ratio|top1_changed_ratio|
|---|---|---|---|---|---|
|L15.H7|28|-0.1592|0.0065|1.0|0.0714|
|L17.H6|28|-0.0575|0.0012|0.9643|0.0714|
|L19.H5|28|-0.0436|0.0009|1.0|0.0|
|L22.H2|28|-0.0352|0.0003|1.0|0.0|
|L17.H0|28|-0.0305|0.0005|0.9286|0.0|
|L14.H7|28|-0.0236|0.0008|0.8214|0.0|
|L21.H10|28|-0.0197|0.0002|0.8929|0.0|
|L19.H2|28|-0.0125|0.0002|0.8214|0.0|

## 9. profession

### celebrity

- 프롬프트 예시: `Taylor Swift's profession is`

유의미한 헤드를 찾지 못함
