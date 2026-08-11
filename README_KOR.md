# Co-occurrence Bias → Hallucination 진단 실험

## 핵심 질문

**이미지에 없는 객체가 주변 객체들과 자주 같이 등장하는 객체일수록 LVLM이 더 쉽게 hallucination하는가?**

이를 확인하기 위해 이미지에 존재하지 않는 target object (A)에 대해 targeted adversarial attack을 수행하고, 모델의 응답을 `No → Yes`로 바꾸는 데 필요한 최소 perturbation budget (\epsilon^*)을 측정함.

핵심 가설은 다음과 같음.

> **Target object (A)가 이미지 내 객체들과 co-occurrence가 높을수록 더 작은 (\epsilon^*)에서 hallucination이 발생할 것이다.**

전체 실험은 LLaVA-1.5-7B와 COCO 2017을 사용함.

---

## Stage 1. Object co-occurrence 계산

먼저 COCO train2017 전체에서 80개 object category 사이의 co-occurrence를 계산함.

단순한 (P(B|A))는 `person`처럼 원래 자주 등장하는 객체의 빈도에 크게 영향을 받기 때문에, **PMI를 주요 co-occurrence score로 사용함.**

* 데이터: COCO train2017 118,287장
* 주요 지표: PMI
* support count < 10인 pair는 이후 분석에서 제외
* 전체 3,160개 pair 중 1,407개 제외

결과적으로 의미적으로 자연스러운 object pair들이 높은 PMI를 보임.

* `mouse–keyboard`: PMI = 3.65, lift = 38.3×
* `car–sink`: PMI = −3.79
* `person` marginal frequency = 54.2%

즉, COCO에서 실제로 강한 object co-occurrence 구조가 존재함을 확인함.

---

## Stage 2. High/Low co-occurrence 그룹 구성

각 val2017 이미지에 대해 이미지에 **존재하지 않는 객체 (A)**를 target으로 설정하고,

[
S(A,Y)
======

\mathrm{mean}
{
PMI(A,y)
:
y\in Y
}
]

로 target (A)와 이미지 내 실제 객체 집합 (Y) 사이의 co-occurrence score를 계산함.

전체 candidate를 기준으로

* 상위 33% → **High co-occurrence**
* 하위 33% → **Low co-occurrence**
* 중간 33% → 제외

로 구분함.

단순히 high/low를 비교하면 object frequency, 크기, 이미지와 target 간 semantic similarity가 confound가 될 수 있기 때문에 **Coarsened Exact Matching(CEM)**을 수행함.

Matching 변수:

* target marginal frequency
* average object area
* CLIP image–text similarity

Matching 전후 SMD는 다음과 같음.

| Covariate          | Before |  After |
| ------------------ | -----: | -----: |
| marginal frequency | +0.076 | −0.036 |
| avg. area          | −0.209 | −0.026 |
| CLIP similarity    | +0.290 | +0.005 |

총 **77,828개의 high/low pair**가 matching되었으며, 세 covariate 모두 matching 후 (|SMD|<0.04)를 만족함.

즉 이후 비교에서 단순한 object frequency나 visual similarity의 영향을 최대한 통제함.

---

## Stage 3. Targeted attack으로 (\epsilon^*) 측정

각 이미지에 대해

> `Is there a {A} in the image?`

라고 질문하고, 원래 `No`라고 답하는 모델을 `Yes`라고 답하도록 만드는 targeted PGD attack을 수행함.

여기서 관심 있는 값은 **attack 성공 여부 자체가 아니라, 응답을 바꾸는 데 필요한 최소 perturbation (\epsilon^*)**임.

즉,

* (\epsilon^*)가 작다
  → 해당 객체를 hallucination시키기 쉬움
* (\epsilon^*)가 크다
  → hallucination시키기 어려움

으로 해석함.

최종 설정:

* L∞ PGD
* 20 steps × 2 restarts
* perturbation은 normalization 이전 [0,1] pixel space에서 적용
* 최대 (\epsilon = 32/255)
* exponential search + binary search로 (\epsilon^*) 탐색

Stage 2의 matched pair 중 150쌍을 stratified sampling하여 총 300개 sample에 대해 실험함.

### Sanity check

실험 자체가 제대로 동작하는지 확인하기 위해 세 가지 control을 함께 확인함.

1. **실제 존재하는 객체 질문**

   ε=0에서 Yes rate = 93–97%

2. **Targeted attack**

   ε=16/255에서 attack success = 100%

3. **Random noise**

   동일 budget의 random noise flip rate = 7–24%

Targeted attack과 random noise 사이에 **76–87%p 차이**가 나타남.

즉 모델이 단순히 noise에 민감해서 응답이 바뀐 것이 아니라, targeted perturbation에 의해 특정 hallucination이 유도되고 있음을 확인함.

---

## Stage 4. High vs Low co-occurrence 비교

이제 핵심 질문인

> **High co-occurrence 조건에서 실제로 더 작은 perturbation으로 hallucination이 발생하는가?**

를 분석함.

Matched pair 구조를 유지하기 위해 `pair_id`를 strata로 하는 stratified Cox regression을 주 분석으로 사용함.

### 결과

| 분석                     | 결과                                                 |
| ---------------------- | -------------------------------------------------- |
| Stratified Cox         | **HR = 1.627**, 95% CI [1.177, 2.250], p = 0.00325 |
| Baseline hallucination | High 21.3% vs Low 6.0%                             |
| Weibull AFT            | **Time ratio = 0.585**, p = 0.00031                |
| Paired bootstrap       | median Δε* = −0.00038                              |
| Log-rank               | p = 0.00026                                        |
| McNemar                | p = 0.000117                                       |

Holm correction 이후에도 주요 결과가 모두 유의함.

특히 Weibull AFT 기준으로는 **High co-occurrence 조건에서 필요한 (\epsilon^*)가 약 42% 작게 나타남.**

즉,

> **이미지에 없는 target object가 현재 이미지의 객체들과 강하게 co-occur할수록 모델이 해당 객체를 hallucination하기 쉬워짐.**

이라는 행동 수준의 결과를 확인함.

---

## Stage 5. 이 bias가 visual encoder feature에 직접 들어있는가?

Stage 3–4는 **LLaVA 전체 모델의 행동**에서 co-occurrence effect가 존재한다는 것을 보여줌.

하지만 이것만으로는 그 원인이 **visual encoder 자체**라고 말할 수 없음.

그래서 frozen CLIP visual feature만으로 high/low co-occurrence condition을 구분할 수 있는지 linear probe를 수행함.

여기서 중요한 것은 이미지에 실제 존재하는 객체 집합 (Y)만 알아도 high/low condition을 상당 부분 예측할 수 있다는 점임.

따라서 단순한 visual feature AUC가 아니라

> **Y-only baseline보다 visual feature가 추가적인 정보를 제공하는가**

를 확인함.

### 결과

| Probe               |              AUC |
| ------------------- | ---------------: |
| Y-only baseline     |            0.799 |
| CLIP visual feature |            0.776 |
| **Excess AUC**      |       **−0.023** |
| 95% CI              | [−0.025, −0.020] |

즉 **visual feature가 Y-only baseline 이상의 정보를 제공하지 못함.**

따라서 현재 결과만으로는

> “co-occurrence bias가 frozen CLIP visual encoder feature에 직접 인코딩되어 있다”

고 볼 수 없음.

---

## 전체 결과

| 질문                                                | 결과            |
| ------------------------------------------------- | ------------- |
| Co-occurrence가 높으면 hallucination이 더 쉽게 발생하는가?     | **Yes**       |
| 그 차이가 단순 frequency / area / CLIP similarity 때문인가? | Matching으로 통제 |
| 단순 random noise에서도 같은 현상이 나타나는가?                  | **No**        |
| Frozen CLIP feature에서 이 bias를 선형적으로 읽을 수 있는가?     | **No**        |

핵심적으로 Stage 3–4에서는 **co-occurrence에 따라 모델의 hallucination susceptibility가 달라진다는 행동적 증거**를 확인함.

반면 Stage 5에서는 해당 정보가 **frozen CLIP visual feature에서 선형적으로 분리되지는 않음.**

따라서 현재까지 확인할 수 있는 것은

[
\text{High Co-occurrence}
\rightarrow
\text{smaller }\epsilon^*
\rightarrow
\text{easier hallucination}
]

이며,

[
\text{Co-occurrence Bias}
\rightarrow
\text{Visual Encoder Representation}
]

까지는 아직 확인되지 않음.

---

## 현재 결론

**Co-occurrence bias 자체는 LLaVA의 hallucination behavior에 영향을 주는 것으로 보임.**

High co-occurrence condition에서 baseline hallucination도 더 많이 발생했고, hallucination을 유도하는 데 필요한 adversarial perturbation도 더 작았음.

다만 현재 실험만으로는 이 현상의 원인이 **visual encoder인지, vision-language fusion인지, language model prior인지** 구분할 수 없음.

특히 Stage 5 결과에서는 frozen CLIP feature에서 추가적인 co-occurrence signal을 확인하지 못했기 때문에,

> **다음 단계에서는 co-occurrence 정보가 실제로 모델의 어느 부분에서 형성되는지를 localization할 필요가 있음.**

---

## 한계

* Stage 3은 전체 77,828 matched pair 중 150쌍만 사용함.
* PGD는 계산 비용 때문에 20 step × 2 restart로 수행함.
* Stage 5는 linear probe만 확인했기 때문에 비선형적으로 정보가 존재할 가능성까지 배제하지는 못함.
* 현재 attack은 closed-form Yes/No 질문을 기준으로 설계되었기 때문에 open-ended caption hallucination에 동일하게 적용된다고 볼 수 없음.

---
## 파일 구조

```text
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

## 재현

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
export PYTHONPATH=src

# 단위 테스트 (GPU 불필요, 53개 전부 통과해야 함)
python -m unittest discover -s tests -v

# Stage 1
python scripts/run_stage1_cooccurrence.py \
  --config configs/stage1_cooccurrence.yaml

# Stage 2 (GPU 필요: CLIP 유사도)
python scripts/run_stage2_sampling_matching.py \
  --config configs/stage2_sampling_matching.yaml

# Stage 3 (GPU 필요: LLaVA-1.5-7B)
# --pilot으로 먼저 sanity check + timing 확인
python scripts/run_stage3_attack.py \
  --config configs/stage3_attack.yaml \
  --pilot

python scripts/run_stage3_attack.py \
  --config configs/stage3_attack.yaml

# Stage 4 (GPU 불필요)
python scripts/run_stage4_survival_analysis.py \
  --config configs/stage4_survival_analysis.yaml

# Stage 5 (GPU 필요: CLIP feature 추출)
python scripts/run_stage5_linear_probe.py \
  --config configs/stage5_linear_probe.yaml
```
