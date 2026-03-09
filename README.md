# hbb_cli

`hbb_cli`는 attention head replace 실험을 실행하고, 프롬프트별/head별 지표를 저장하는 CLI 프로젝트입니다.

## 지표
### 프롬프트 별
* base_token_prob_delta : base token 확률 변화량
* base_token_prob_direction : base token 확률 변화 방향 (감소/증가)
* base_token_changed : replace 후 base token이 예측 토큰에서 바뀌었는지
* base_token_rank_pre_replace/post_replace/change : base token이 몇 등에서 몇 등으로 바뀌었는지
* donor_token : donor 프롬프트의 원래 토큰
* donor_token_prob_pre_replace/post_replace/delta : donor 토큰의 확률이 얼마나 증가/감소했는지
* donor_token_rank_pre_replace/post_replace/change : donor 토큰이 base 분포에서 몇 등에서 몇 등으로 이동했는지
* baseline_base_token(prob) : 원래 출력값과 그 확률
* replaced_base_token(prob) : 바뀐 출력값과 그 확률 

### head 별
* prompt_count : 실험에 사용한 프롬프트 개수
* base_token_prob_delta_mean : base token 확률 변화량 평균
* base_token_prob_delta_variance : base token 확률 변화량 분산
* base_token_prob_decrease_ratio : 전체 프롬프트 중 base token 확률이 떨어진 비율
* base_token_changed_ratio : 전체 프롬프트 중 base token이 바뀐 비율
* base_token_rank_post_replace_mean : base token의 replace 후 평균 등수
* donor_token_prob_delta_mean : donor 토큰 확률 변화량 평균
* donor_token_prob_delta_variance : donor 토큰 확률 변화량 분산
* donor_token_prob_increase_ratio : donor 토큰 확률이 증가한 비율
* donor_token_rank_up_ratio : donor 토큰 rank가 실제로 상승한 비율
* donor_token_rank_pre_replace_mean : donor 토큰의 replace 전 평균 rank
* donor_token_rank_post_replace_mean : donor 토큰의 replace 후 평균 rank

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
  - `prompt_output_map.csv`, `prompt_output_map.jsonl` (프롬프트와 baseline 출력 토큰 매핑)
  - `summary_by_head.csv`, `summary_by_head.jsonl`
  - `prompt_by_head.csv`, `prompt_by_head.jsonl`
  - (특정 head 세트 실행 시) `prompt_metrics_*.csv/json`, `summary_*.csv/json`

### `summary_by_head`가 안 생길 수 있는 경우

- 버킷(카테고리/소스 파일) 안 프롬프트가 2개 미만이면 해당 버킷은 스킵됩니다.
- `scan-all-heads`에서 summary 필터를 통과한 head가 하나도 없으면 새 `summary_by_head`가 생성되지 않을 수 있습니다.
  - 필터: `base_token_prob_decrease_ratio >= threshold(0.8 -> 0.1 fallback)` 그리고 `base_token_prob_delta_mean < -0.01`
- 재실행 시 이미 같은 `(head_label, prompt_count)` 키가 있으면 중복 추가하지 않습니다.

# 결과
### 아래링크 참고
https://headbb.vercel.app/

# 실험결과

실험 결과 표는 아래 5개 핵심 지표 중심으로 정리했습니다.

- `base_token_prob_delta_mean`
- `base_token_prob_decrease_ratio`
- `donor_token_rank_up_ratio`
- `donor_token_rank_pre_replace_mean`
- `donor_token_rank_post_replace_mean`

## 1. capitals

### africa

- 프롬프트 예시: `What is the capital of Egypt? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|16|-0.2303|0.9375|1|392.875|14.75|
  |L17.H6|16|-0.0484|1|1|392.875|104.625|
  |L15.H3|16|-0.0355|0.875|0.9375|392.875|290.125|
  |L22.H2|16|-0.0204|0.8125|0.875|392.875|332.75|
  |L17.H0|16|-0.0131|0.8125|0.875|392.875|306.4375|
  |L23.H9|16|-0.013|0.875|0.875|392.875|366.75|
  |L19.H7|16|-0.0104|0.8125|0.8125|392.875|375.25|

### asia

- 프롬프트 예시: `What is the capital of Japan? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|25|-0.1726|0.96|0.96|549.92|18.84|
  |L15.H3|25|-0.038|0.8|0.96|549.92|378.6|
  |L17.H6|25|-0.0334|0.8|0.92|549.92|145|
  |L18.H9|25|-0.0287|0.8|0.88|549.92|321.32|
  |L22.H2|25|-0.0183|0.88|0.8|549.92|511.92|
  |L23.H13|25|-0.0101|0.84|0.8|549.92|523|

### easy

- 프롬프트 예시: `What is the capital of Italy? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|28|-0.201|0.9643|1|725.3214|7.3929|
  |L17.H6|28|-0.0687|0.9643|1|725.3214|75.8571|
  |L17.H0|28|-0.0186|0.8571|1|725.3214|376.2857|
  |L23.H9|28|-0.0137|0.9286|0.8571|725.3214|670.8571|

### europe

- 프롬프트 예시: `What is the capital of France? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|20|-0.2513|1|0.95|226.65|3|
  |L17.H6|20|-0.0387|0.95|0.95|226.65|52.35|
  |L22.H2|20|-0.0173|0.8|0.95|226.65|202.95|

### north_america

- 프롬프트 예시: `What is the capital of the United States? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|16|-0.1305|0.9375|1|207.1875|18.125|
  |L17.H6|16|-0.0419|0.9375|0.9375|207.1875|79.125|
  |L15.H3|16|-0.0366|0.875|0.8125|207.1875|131.1875|
  |L22.H2|16|-0.016|0.875|0.8125|207.1875|192.4375|
  |L23.H13|16|-0.0132|0.875|0.875|207.1875|193.4375|

### oceania

- 프롬프트 예시: `What is the capital of Solomon Islands? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|7|-0.102|1|0.8571|44.2857|9.4286|
  |L17.H6|7|-0.0441|1|0.8571|44.2857|21.1429|
  |L13.H1|7|-0.017|0.8571|0.8571|44.2857|30.8571|

### south_america

- 프롬프트 예시: `What is the capital of Argentina? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|11|-0.1811|1|1|93.7273|13.1818|
  |L16.H2|11|-0.0612|0.9091|1|93.7273|67.3636|
  |L22.H2|11|-0.0334|1|1|93.7273|77.3636|
  |L13.H1|11|-0.022|0.8182|1|93.7273|54|
  |L17.H6|11|-0.0204|0.9091|1|93.7273|58.4545|
  |L23.H13|11|-0.0146|0.9091|0.9091|93.7273|87.0909|

## 2. chemical

### chemical_symbols

- 프롬프트 예시: `The chemical symbol for Hydrogen is`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L13.H6|100|-0.1218|0.81|1|136.09|9.35|
  |L22.H2|100|-0.0557|0.82|0.98|136.09|62.48|

## 3. logical

### order2

- 프롬프트 예시: `1, 2,`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L12.H0|38|-0.2342|0.8421|0.7368|408.2632|120.7632|

### order3

- 프롬프트 예시: `1, 2, 3,`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L12.H0|32|-0.527|1|0.9688|1110.75|228.6562|

## 4. mathematics

### add

- 프롬프트 예시: `Cal : 12+35=`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L12.H0|159|-0.2842|0.93|0.84|1893.02|465.83|
  |L22.H2|159|-0.0502|0.86|0.97|1893.02|763.89|

### arithmetic_geometric_progression

- 프롬프트 예시: `Find the pattern: 3, 7, 15, 31, 63, 127,`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L7.H7|30|-0.1946|0.7667|0.7333|2177.4667|168.3333|

### arithmetic_progression

- 프롬프트 예시: `Find the pattern: 2, 5, 8, 11,`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L12.H0|30|-0.1685|0.8|0.8667|32.4|13.9|
  |L13.H6|30|-0.1058|1|0.9333|32.4|20.4667|

### constant

- 프롬프트 예시: `Output only the number. pi =`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L11.H11|50|-0.0119|0.62|0.38|140.22|105.82|
  |L14.H5|50|-0.0107|0.64|0.38|140.22|83.16|

### geometric_progression

- 프롬프트 예시: `Find the pattern: 2, 4, 8, 16,`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L13.H6|30|-0.0679|0.7333|0.7333|110.9333|70.8333|
  |L19.H5|30|-0.0112|0.7|0.7333|110.9333|94.9|

### mul

- 프롬프트 예시: `Cal : 12*24=`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L12.H15|200|-0.1181|0.89|0.81|2857.665|2339.25|
  |L22.H2|200|-0.0472|0.87|0.85|2857.665|1969.34|

### sub

- 프롬프트 예시: `Cal : 39 minus 38=`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L22.H2|100|-0.0156|0.75|0.9|376.62|216.88|

## 5. opposite

### opposite

- 프롬프트 예시: `The opposite of 'hot' is '`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L13.H2|100|-0.2199|0.97|0.97|327.18|80.28|
  |L14.H12|100|-0.0567|0.83|0.95|327.18|196.5|
  |L22.H0|100|-0.0253|0.8|0.93|327.18|235.86|
  |L23.H2|100|-0.0179|0.8|0.86|327.18|283.2|
  |L23.H9|100|-0.0162|0.82|0.92|327.18|281.5|

## 6. place

### architecture

- 프롬프트 예시: `The Eiffel Tower is in`

유의미한 헤드를 찾지 못함

### birthplace

- 프롬프트 예시: `Albert Einstein was born in`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|50|-0.0372|0.76|0.62|275.68|142.02|
  |L17.H0|50|-0.0206|0.74|0.66|275.68|138.4|

## 7. time

### historical_year

- 프롬프트 예시: `Answer using only a 4-digit number. World War I began in`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L14.H2|79|-0.1016|0.7722|0.8734|596.9873|90.51|

## 8. country

### general

- 프롬프트 예시: `What country is Paris in? Answer:`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L15.H7|28|-0.1611|1|1|46.75|8|
  |L17.H6|28|-0.0587|1|0.9643|46.75|23.0357|
  |L19.H5|28|-0.0452|1|0.9286|46.75|30.5357|
  |L22.H2|28|-0.0344|1|0.9286|46.75|38|
  |L17.H0|28|-0.0319|0.9643|0.9286|46.75|31.5714|
  |L21.H10|28|-0.0212|0.8929|0.8571|46.75|37.8214|
  |L19.H2|28|-0.012|0.8214|0.8214|46.75|39.5|

### ISO code

- 프롬프트 예시: `The ISO country code for United States is`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L22.H2|165|-0.043|0.818|0.993|325.424|197.27|
  
## 9. profession

### celebrity

- 프롬프트 예시: `Taylor Swift's profession is`

유의미한 헤드를 찾지 못함

## 10. languages

### blah_corr

- 프롬프트 예시: `The official language of South Korea is`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L17.H6|105|-0.0752|0.8095|0.9810|619.6667|57.6857|


## 11. feelings

### positive

- 프롬프트 예시: `I finally achieved a goal I worked toward for years. The word for this feeling is '`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L16.H4|50|-0.0141|0.76|0.84|191.6|122.34|
  |L18.H2|50|-0.0110|0.78|0.86|191.6|120.72|
  |L13.H13|50|-0.0103|0.78|0.88|191.6|124.62|

### negative
- 프롬프트 예시: `I made a serious mistake that hurt someone. The word for this feeling is '`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L13.H13|50|-0.0225|0.86|0.82|72.8|31.28|

## 12. relations

### object-function relations
- 프롬프트 예시: `A knife is used to '`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L14.H5|180|-0.0223|0.87|0.96|354.45|203.73|
  |L12.H8|180|-0.0170|0.85|0.91|354.45|290.68|
  |L11.H1|180|-0.0104|0.82|0.84|354.45|308.57|

### IATA_airport
- 프롬프트 예시: `The IATA code for Los Angeles International Airport is`

  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L18.H0|110|-0.0645|0.85|0.85|49.34|31.95|
  |L13.H6|110|-0.0652|0.88|0.88|49.34|18.15|
  |L22.H2|110|-0.0450|0.9|0.85|49.34|30.0|
  |L11.H6|110|-0.0242|0.89|0.85|49.34|35.71|

## 13. computer

### http_error_code
- 프롬프트 예시: `The HTTP status code for 'Not Found' is '`
  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L14.H5|13|-0.079|1.0|0.85|14.62|9.31|

### file_extension
- 프롬프트 예시: `The file extension for Python source files is .`
  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L22.H2|100|-0.016|0.82|0.86|407.5|317.64|

### port_number
- 프롬프트 예시: `The default port for HTTP is`
  |head|prompt_count|base_token_prob_delta_mean|base_token_prob_decrease_ratio|donor_token_rank_up_ratio|donor_token_rank_pre_replace_mean|donor_token_rank_post_replace_mean|
  |---|---|---|---|---|---|---|
  |L14.H5|41|-0.133|0.92|0.90|70.36|29.24|
  |L15.H6|41|-0.106|0.92|0.88|70.36|33.12|
  |L15.H7|41|-0.075|0.87|0.80|70.36|47.12|