# Co-occurrence Bias → Hallucination 진단 실험 (go/no-go)

LVLM(LLaVA-1.5-7B)의 object hallucination이 visual encoder에 인코딩된
object co-occurrence bias에서 비롯되는지를 검증하는 진단 실험이다. 논문
전체의 전제를 확립하는 go/no-go 실험이므로, 목표는 좋은 수치가 아니라
**측정의 정확성과 confound 통제**다. 아래 결과에는 가설을 지지하는 것과
지지하지 않는 것이 모두 포함되어 있으며, 결과를 좋게 보이도록 만드는 방향의
판단은 어디에서도 하지 않았다.

**핵심 가설**: 이미지에 존재하지 않는 객체 A에 대해 "A가 있다"는 응답을
유도하는 targeted adversarial attack의 최소 perturbation budget ε*는, A가
이미지 내 객체들과 co-occurrence가 높을 때 더 작다.

## 환경

- LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`, HuggingFace transformers), greedy decoding
- CLIP ViT-L/14-336 (`openai/clip-vit-large-patch14-336`) — LLaVA-1.5의 vision
  tower와 동일 모델. Stage 2의 CLIP text-image similarity, Stage 5의 frozen
  visual feature 모두 이 모델을 재사용해 가설과 같은 시각-의미 공간에서 측정.
- COCO 2017: train2017 annotation으로 co-occurrence 통계, val2017로 실험
  (`/data3/KJE/code/UQ/data/COCO/`)
- GPU: A100 80GB × 5 (실행 시점 기준 GPU0은 다른 프로세스가 상시 점유,
  나머지 GPU를 필요에 따라 사용)
- 재현성: 모든 seed 고정, config를 stage별 yaml로 분리 (`configs/`)

## 파일 구조

```
configs/                                  stage별 yaml 설정
src/cooc_diagnostic/
  coco_index.py                           COCO 이미지별 present-category 인덱싱
  cooccurrence_stats.py                   PMI/lift/raw conditional 계산
  covariates.py                           카테고리별 평균 object area
  clip_similarity.py                      CLIP image-text 유사도 및 image embedding
  strata_sampling.py                      candidate 생성 + high/low strata 분할
  matching.py                             Coarsened Exact Matching + balance table
  stratified_subsample.py                 CEM cell 비율 유지 stratified subsampling
  llava_runtime.py                        LLaVA 전처리/프롬프트/Yes-No 판정
  pgd_attack.py / random_attack.py        L∞ PGD attack, random-noise control
  epsilon_star.py                         ε* 지수탐색+이분탐색
  sanity_checks.py                        3대 필수 sanity check 집계
  survival_analysis.py                    KM, stratified Cox, Weibull AFT, McNemar, Holm
  linear_probe.py                         Stage5 linear probe (excess AUC)
scripts/run_stage{1..5}_*.py              각 stage 실행 스크립트
tests/                                    GPU 없이 도는 순수 로직 단위 테스트 (53개, 전부 통과)
outputs/CooccurrenceHallucinationDiagnostic/stage{1..5}_*/   결과물 (레포 루트 outputs/)
```

## Stage 1 — Co-occurrence 통계

COCO train2017 (118,287장, 80개 카테고리) 전체에서 카테고리 쌍 co-occurrence를
계산. Raw conditional P(B|A)는 marginal frequency에 오염되므로 PMI/lift를
주 지표로, raw conditional은 sanity check로만 사용. 0-count 쌍은 PMI=-inf로
그대로 남기고(스무딩 없음), top/bottom 순위와 이후 stage의 candidate scoring에는
`min_support_count=10` 미달 쌍만 제외(전체 3,160쌍 중 1,407쌍, 44.5%).

**결과**: 의미상 뚜렷한 클러스터(차량/도로시설, 주방/식사, 전자기기, 스포츠 장비)가
PMI heatmap에서 그대로 드러나 계산 정확성을 확인. `person`의 marginal frequency가
54.2%로 압도적 1위 — Stage 2 confound matching이 필수인 이유를 데이터로 확인.
Top pair: mouse-keyboard (PMI=3.65, lift=38.3×). Bottom pair: car-sink (PMI=-3.79).

출력: `outputs/.../stage1_cooccurrence/` (co-occurrence 행렬 4종, marginal
frequency, top/bottom pairs, heatmap/histogram plot).

## Stage 2 — 샘플링과 confound matching

val2017(5,000장) 각 이미지의 present set Y와, 결측 카테고리 A 사이의
co-occurrence 점수 S(A,Y) = mean{PMI(A,y) : y∈Y, count(A,y)≥10}으로 정의
(사용자 확인: mean 집계, 낮은 support 항은 평균에서 제외). Train2017 기준
marginal frequency 1% 미만인 6개 극희귀 카테고리(hair drier, toaster, parking
meter, scissors, bear, toothbrush)는 target 후보에서 제외(사용자 확인) —
eligible 74/80.

302,670개 candidate 중 S(A,Y) 상/하위 33%를 각각 high-co-occurrence
(treatment, 99,881개)/low-co-occurrence (control, 99,897개)로, 중간 33%는
제거(사용자 확인). Confound matching은 **Coarsened Exact Matching (CEM)**을
선택 — marginal frequency/average area는 카테고리 단위 상수이므로 propensity
모델보다 "어느 cell에 몇 개씩 있는지" 그대로 감사(audit) 가능한 CEM이 이
설계에는 더 투명하다고 판단. Freq/area 4분위, CLIP image-text similarity
5분위로 코스닝 후, 같은 cell 내에서 CLIP 유사도 순위 매칭.

**Matching 전/후 balance (SMD)**:

| covariate | before | after |
|---|---|---|
| marginal_freq | +0.076 | −0.036 |
| avg_area | −0.209 | −0.026 |
| clip_sim | +0.290 | +0.005 |

77,828쌍 매칭 성공 (pool의 78%). 세 covariate 모두 matching 후 \|SMD\|<0.04로
양호한 balance 달성.

출력: `outputs/.../stage2_sampling_matching/` (balance_before/after_matching.csv,
matched_pairs.csv).

## Stage 3 — Targeted attack과 ε* 측정

"Is there a {A} in the image?" 질문에서 "Yes" 토큰의 logit을 최대화하는 L∞
PGD. **perturbation은 CLIP normalization 이전의 [0,1] 픽셀 공간(resize→
center-crop→rescale 이후, normalize 이전)에 매 step [0,1] clamp**하며 적용
(`llava_runtime.py`의 `preprocess_to_unit_range`/`normalize` 분리로 강제).
"Yes" 토큰 위치는 하드코딩하지 않고 실제 생성으로 매번 탐지
(`detect_yes_no_decision_point`) — LLaVA 프롬프트가 `"ASSISTANT:"`로 끝나
공백 토큰이 별도로 붙을 걱정을 했으나, 실측 결과 첫 생성 토큰이 곧바로
Yes(3869)/No(1939)였음을 확인. PGD의 margin(logit(yes)-logit(no))은 최적화
proxy로만 쓰고, **최종 flip 판정은 항상 실제 greedy 생성 텍스트**로 authoritative하게
확인(margin과 실제 답변 텍스트를 분리).

**설계가 실측 데이터로 두 번 바뀐 지점** (모두 사용자 확인 후 반영):
1. 최초 계획한 고정 해상도(1/255, 8bit 이미지 양자화 기준) 선형 이분탐색은
   30-sample 파일럿에서 거의 모든 샘플이 1/255에서 flip — 진짜 임계값이
   1/255보다 훨씬 작아 측정이 saturate됨을 발견. → **지수 탐색(eps0=epsilon_max/8192부터
   배로 증가해 bracket 탐색) 후 상대정밀도(10%) 이분탐색**으로 교체.
2. 실패(non-flip) 호출은 early-stop이 안 되어 PGD 예산을 전부 소진 —
   같은 파일럿에서 40 step×3 restart 기준 샘플당 ~267초 소요, 150쌍 전체
   실행 시 예상보다 훨씬 오래 걸림을 발견. → **20 step×2 restart**로 축소
   (실패 호출 비용 3배 절감, PGD-20은 adversarial robustness 문헌에서 흔히
   쓰는 예산).

150쌍(300 샘플, Stage2 matched pairs를 CEM cell 비율 유지하며 stratified
subsampling)을 GPU 4장(1~4)에 분산 병렬 실행.

**필수 sanity check 3종 (4개 shard 전부 통과)**:
1. present-object baseline @ ε=0: Yes rate 93~97%
2. targeted attack @ 16/255 성공률: 100% (모든 shard)
3. **같은 예산의 random noise flip rate: 7~24%** (targeted attack 대비
   76~87%p 격차) — 측정 대상이 co-occurrence가 아니라 단순 노이즈 민감도가
   아님을 확인하는 가장 중요한 control.

censored 샘플: 0/300 (epsilon_max=32/255 안에서 전부 결론).

출력: `outputs/.../stage3_attack/epsilon_star_results.csv` (300 rows).

## Stage 4 — 통계 분석 (survival analysis)

ε*는 (이번 실행에서는 censored가 없었지만 일반적으로) right-censored이므로
단순 평균/t-test 대신 생존분석 사용. ε=0에서 이미 Yes인 already_yes 케이스는
버리지 않고 duration=0인 실제 event로 포함하면서, 별도 rate로도 집계
(brief 요구사항). **Matched-pair 구조는 pair_id를 strata로 하는 stratified
Cox regression**으로 반영 — Stage 2가 만든 confound-matched 짝을 그대로
partial likelihood에 살림.

| 분석 | 결과 |
|---|---|
| **주 검정: stratified Cox (pair_id strata)** | HR = **1.627**, 95% CI [1.177, 2.250], p = 0.00325 |
| already_yes rate (baseline hallucination) | treatment 21.3% (32/150) vs control 6.0% (9/150) |
| Weibull AFT time ratio | **0.585**, 95% CI [0.438, 0.783], p = 0.00031 (treatment는 필요 ε*가 평균 ~42% 작음) |
| paired bootstrap median 차이 | −0.00038, 95% CI [−0.00064, −0.00012] |
| 비matched sensitivity log-rank | p = 0.00026 (방향/크기 일치) |
| McNemar (already_yes, paired) | treatment-only 29 vs control-only 6, p = 0.000117 |
| **Holm 보정 후** (2개 가설: Cox, McNemar) | 둘 다 α=0.05에서 유의 (0.00325, 0.000234) |

5개의 독립적 분석이 모두 같은 방향·크기로 일치했고 다중비교 보정에도
살아남았다 — **가설을 지지하는 일관된 증거**.

출력: `outputs/.../stage4_survival_analysis/` (stage4_report.json, km_curves.png).

## Stage 5 — Linear probe (독립적 보완 증거)

Frozen CLIP visual feature가 "present 집합 Y를 아는 것"을 넘어서는
co-occurrence 정보를 선형적으로 담고 있는지 검증. Y만으로도 Z(high/low
co-occurrence strata 소속)가 거의 결정되므로, **Y-only baseline 대비 excess
AUC**만이 의미 있는 신호(brief 요구사항). Baseline = one-hot(A)+multi-hot(Y),
full = one-hot(A)+frozen CLIP embedding — 두 모델 모두 "어느 카테고리 A를
묻는지"는 task 명세로 동일하게 받고, Y(symbolic) vs raw pixel의 차이만 비교.
Stage 2의 전산 candidate pool(199,778개, matched 150쌍보다 통계적으로
훨씬 강력, 사용자 확인)을 사용, **이미지 단위** train/test 80/20 분리
(train 159,752 / test 40,026)로 leakage 차단.

| Probe | AUC |
|---|---|
| Baseline (Y-only) | 0.799 |
| Full (CLIP feature) | 0.776 |
| **Excess AUC** | **−0.023**, 95% CI [−0.025, −0.020] |

**가설을 지지하지 않는 null 결과**: excess AUC의 신뢰구간이 0을 포함하지 않고
음수 쪽에 있다 — frozen CLIP visual feature는 present 객체 집합을 아는 것
이상의 co-occurrence 정보를 선형적으로 제공하지 못했다. Baseline AUC(0.799)가
높다는 것 자체는 "Y가 Z를 거의 결정한다"는 brief의 전제를 뒷받침하지만, 그
이상을 raw pixel에서 끌어내지는 못했다.

출력: `outputs/.../stage5_linear_probe/stage5_report.json`.

## 종합 결론

| Stage | 질문 | 결과 |
|---|---|---|
| 3–4 | 전체 LLaVA 파이프라인의 행동(적대적 공격 취약성)이 co-occurrence에 따라 다른가 | **지지** (HR=1.63, p=0.003, Holm 보정 후에도 유의) |
| 5 | 그 정보가 frozen CLIP visual encoder 출력에서 선형적으로 읽히는가 | **지지 안 됨** (excess AUC 95% CI 전부 음수) |

두 결과는 서로 반박하지 않는다 — 다른 질문이다. Stage 3/4는 "모델 전체가
다르게 반응하는가"를, Stage 5는 "그 반응의 원인이 vision encoder의
선형적으로 추출 가능한 표현에 있는가"를 본다. 이 조합은 co-occurrence
bias가 (존재한다면) frozen visual encoder의 표현보다는 language model의
학습된 prior나 vision-language fusion 과정 쪽에 더 강하게 자리잡고 있을
가능성을 시사한다 — 다만 이는 사후 해석이며, Stage 5의 null 결과 자체를
근거로 확정할 수 있는 결론은 아니다.

**Go/no-go 판단**: 핵심 가설(co-occurrence가 targeted attack budget에
영향을 준다)에 대한 행동적 증거(Stage 3/4)는 견고하고 다중 sanity
check·confound matching·다중비교 보정을 통과했다. 다만 그 메커니즘이
vision encoder 자체에 있다는 보완 증거(Stage 5)는 얻지 못했으므로, 후속
연구는 "어디서 이 정보가 인코딩되는가"(language model, cross-attention,
또는 vision encoder의 비선형 표현)를 좁혀가는 방향이 되어야 한다.

## 재현

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
export PYTHONPATH=src

# 단위 테스트 (GPU 불필요, 53개 전부 통과해야 함)
/opt/anaconda3/envs/py3_11/bin/python -m unittest discover -s tests -v

# Stage 1
/opt/anaconda3/envs/py3_11/bin/python scripts/run_stage1_cooccurrence.py --config configs/stage1_cooccurrence.yaml

# Stage 2 (GPU 필요: CLIP 유사도)
/opt/anaconda3/envs/py3_11/bin/python scripts/run_stage2_sampling_matching.py --config configs/stage2_sampling_matching.yaml

# Stage 3 (GPU 필요: LLaVA-1.5-7B). --pilot로 먼저 sanity check + timing 확인 권장
/opt/anaconda3/envs/py3_11/bin/python scripts/run_stage3_attack.py --config configs/stage3_attack.yaml --pilot
/opt/anaconda3/envs/py3_11/bin/python scripts/run_stage3_attack.py --config configs/stage3_attack.yaml

# Stage 4 (GPU 불필요)
/opt/anaconda3/envs/py3_11/bin/python scripts/run_stage4_survival_analysis.py --config configs/stage4_survival_analysis.yaml

# Stage 5 (GPU 필요: CLIP feature 추출)
/opt/anaconda3/envs/py3_11/bin/python scripts/run_stage5_linear_probe.py --config configs/stage5_linear_probe.yaml
```

## 알려진 한계

- Stage 3 전체 실행은 Stage 2의 매칭된 77,828쌍 중 150쌍(stratified
  subsample)만 사용 — 계산 비용 때문(샘플당 실측 ~102초). 전체 모집단에
  대한 결론이 아니라 대표 부분표본에 대한 결론이다.
- PGD 예산(20 step×2 restart)은 원래 계획(40×3)보다 축소됐다 — 실측 근거로
  사용자 확인 후 조정했지만, 극히 미세한 ε* 차이에 대한 검정력은 원래
  계획보다 낮을 수 있다.
- Stage 5의 null 결과는 "선형 probe로 탐지되지 않음"을 보인 것이며, 비선형
  관계(예: MLP probe)나 다른 CLIP layer/pooling 방식에서는 다른 결과가
  나올 수 있다 — 이 실험은 그 가능성을 배제하지 않는다.
- Open-ended caption 생성으로의 전이를 보는 추가 시도(closed yes/no 공격을
  그대로 재사용)를 해봤으나, yes/no 목적함수로 찾은 perturbation을
  open-ended 목적에 그대로 재사용하는 것은 설계상 타당한 검증이 아니라고
  판단해 분석/결론에서 제외했다 (코드: `scripts/run_stage6_open_ended_transfer.py`,
  결과: `outputs/.../stage6_open_ended_transfer/`, 참고용으로만 보존).
