# Co-occurrence Bias → Hallucination 진단 실험

## 연구 가설

이미지에 없는 target object $A$가 이미지 내에 실제로 존재하는 객체들과 co-occurrence가 높을수록, LVLM이 그 target object를 더 쉽게 hallucination한다는 가설을 검증하는 프로젝트.

> **이미지에 없는 target object $A$가 이미지 내 존재하는 객체들과 co-occurrence가 높을수록 hallucination이 더 쉽게 발생한다.**

전체 실험은 LLaVA-1.5-7B와 COCO 2017(co-occurrence 통계는 train2017, 모델 관련 평가는 전부 val2017)을 사용하며, seed=42로 고정.

프로젝트는 아래 4개 Phase로 구성되고, 각 Phase는 이전 Phase의 결과를 다시 서술하지 않고 그 위에 이어지는 구조로 설계되어 있음. Stage별 상세 수치는 각 stage 자신의 output/README(아래 링크)에 있고, 이 문서는 전체 구조 — 무엇이 존재하고, 핵심 결과가 무엇이며, 다음 단계와 어떻게 연결되는지 — 만 정리함.

```
Phase A  행동적 효과 검증                     Stage 1-8
Phase B  표상적 위치 규명                     Stage 9-11 (+ Stage 5)
Phase C  단일 pair 메커니즘 케이스 스터디       Stage 11 case study + cooccurrence_causal_coupling
Phase D  Debiasing 가능성 검증                adversarial_functional_debiasing_pilot + adversarial_signal_debiasing_pilot
```

---

## Phase A — 행동적 효과 (Stage 1-8)

여러 object-category pair를 대상으로 한 population-level 파이프라인. 이 stage들은 별도 README가 없어 이 문서가 결과의 1차 기록임.

| Stage | 검증 내용 | 방법 | 핵심 결과 |
|---|---|---|---|
| **1** | COCO 내 object co-occurrence 구조 존재 확인 | train2017(118,287장) 전체에서 PMI 계산, support < 10인 pair 제외 | `mouse–keyboard` PMI = 3.65 (lift 38.3×); 구조 확인됨 |
| **2** | Co-occurrence 효과와 confound(marginal frequency / object area / CLIP similarity)의 분리 | High/Low co-occurrence 그룹(이미지 내 객체 대상 평균 PMI 상위/하위 33%) 구성 후 Coarsened Exact Matching 적용 | 77,828개 matched pair, matching 후 모든 covariate \|SMD\| < 0.04 |
| **3** | Hallucination 유도 난이도 측정 | `Is there a {A}?`에 대한 targeted PGD attack, 응답을 뒤집는 최소 budget $\epsilon^*$ ($L_\infty$, 20 step × 2 restart, max $\epsilon=32/255$), 150개 matched pair(300 sample) | 3개 필수 sanity check 통과 (attack 성공률 ≫ random-noise flip率) |
| **4** | High co-occurrence 조건에서 $\epsilon^*$가 더 작은지 검증 | Stage 3의 $\epsilon^*$에 대해 `pair_id`를 stratum으로 한 stratified Cox / Weibull AFT / paired bootstrap / McNemar | Cox HR = 1.627 (p = 0.00325); ε\*가 High에서 약 42% 더 작음; Holm correction 후에도 유의 |
| **5** | Bias의 frozen visual encoder 내 linear decodability 검증 | CLIP visual feature에 대한 linear probe vs. Y-only(존재 객체만 사용) baseline, excess AUC 비교 | excess AUC = −0.023, 95% CI [−0.025, −0.020] — 추가 정보 없음 |
| **6** | Yes/No 최적화 attack의 open-ended captioning 전이 여부 검증 | Stage 3의 adversarial image를 재사용, free-form caption 내 target mention률 측정 | 탐색적 실행만 존재 — 이 프로젝트의 rigor bar를 통과하지 못함; `outputs/.../stage6_open_ended_transfer/`에 raw 결과만 보관, 결과로 기재하지 않음 |
| **7** | Caption을 직접 겨냥한 attack으로 Stage 3 효과 재현 시도 | Stage 3의 구조적 복제, readout을 forced yes/no 대신 short-answer 응답 내 target category mention 여부로 변경 | 파이프라인은 구축되어 end-to-end로 동작하나, margin 설계를 여러 번 pivot했고 small N에서 run-to-run noise가 관찰됨 — 통제된 결과로 보지 않음 |
| **8** | Caption-attack $\epsilon^*$에 대한 High vs. Low 검증 | Stage 7의 $\epsilon^*$에 대해 Stage 4와 동일한 survival analysis | Stage 7과 동일한 이유로 보류 — 실행은 완료되었으나 제대로 통제되지 않은 것으로 판단되어 결과로 보고하지 않음(raw 수치는 `outputs/.../stage8_survival_analysis_caption/` 참고) |

**Phase A 정리:** forced yes/no VQA(Stage 3-4) 기준으로 co-occurrence → hallucination 효과가 실재하며 confound가 잘 통제되어 있고, visual-encoder만으로는 설명되지 않음(Stage 5). Stage 6-8은 동일한 효과가 open-ended/caption 스타일 readout에서도 성립하는지 탐색한 단계로, 실행은 완료되었으나 중간 설계 pivot과 small-N noise가 많아 이 프로젝트의 통제 기준을 충족하지 못해 결과로 기재하지 않음(raw output은 참고용으로 보관, 결론을 내리려면 재실행 필요).

---

## Phase B — 표상적 위치 (Stage 9-11, 일반 population)

Phase A가 *attack 하에서의 행동*을 다뤘다면, Phase B는 *clean image(ε=0)의 내부 evidence*를 대상으로 bias가 실제로 모델 어디에 위치하는지를 규명함.

| Stage | 검증 내용(논문 표기) | 방법 | 핵심 결과 |
|---|---|---|---|
| **9** (Exp1) | Attack 없이도 High co-occurrence가 clean target-positive evidence $s_T$ = logit(Yes) − logit(No)를 높이는지 검증 | Stage 3와 동일한 150개 matched High/Low pair, $\epsilon=0$에서의 $s_T$ | mean diff = +0.729, 95% CI [0.399, 1.061], Cohen's $d_z$ = 0.353, p < 0.001 |
| **10** (Exp2) | 이미지를 고정한 상태에서 co-occurrence 관계 강도와 $s_T$의 관계 검증(within-image specificity, 이미지 간 confound와 구분) | 50개 이미지 × 74개 target category(3,039 pair), 이원 fixed-effects 모델 $s_T \sim \text{coocScore} + \text{imageFE} + \text{targetFE}$ | $\beta = 0.406$, 95% CI [0.290, 0.522], p = 6.6e-12, permutation null ≈ 0 |
| **11** (Exp3) | 효과가 LLM의 어느 layer에서 나타나는지 localization | Stage 10의 3,039 pair에 대해 LLaVA decoder 32개 layer 전체에 logit lens(최종 RMSNorm + lm_head) 적용, layer별로 Stage 10과 동일한 FE 모델 재적합 | Layer 1-6 신호 ≈0, layer 7-12 유의하게 음(negative), layer 13에서 양(positive)으로 전환, layer 16부터 마지막 layer(32, Stage 10의 $\beta$ 재현)까지 partial $r \approx 0.21$–$0.26$로 plateau |

**Phase B 정리:** 이 효과는 (a) attack 없이도 이미 존재하고, (b) 단순 frequency confound가 아니라 image-target 관계에 특이적이며, (c) LLM 자체의 decoder stream에서는 mid layer 이후부터 linear하게 decode됨 — frozen visual encoder에 대한 Stage 5의 negative 결과와 대비됨. 이 결과는 효과를 LLM decoder layer ≥13에 localize하며, Phase C에서 다루는 causal 사용 여부 검증으로 이어짐.

---

## Phase C — 단일 Pair 메커니즘 케이스 스터디

Phase A/B는 여러 object-category pair를 pooling한 결과이며, Phase C는 눈으로 직접 확인 가능한 하나의 구체적인 pair(`baseball bat` context → `sports ball` target)에 대해 동일한 검증 체인을 end-to-end로 반복한 뒤, correlation을 넘어 intervention까지 진행하는 케이스 스터디임.

### Stage 11 case study — `baseball bat → sports ball`

전체 write-up: [`outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md`](../../../outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md) (Exp0-Exp7). 요약:

| Exp | 검증 내용 | 결과 |
|---|---|---|
| 0 | Pair 강도 확인 | PMI = 2.342, lift = 10.40, 5,338개 pair 중 59위 — **GO** |
| 1 | Bat context와 attack 취약성의 관계 검증 | median $\epsilon^*$ 0.0001148(G10, bat+ball−) vs 0.0005203(G00 control), Cox HR = 1.760, p = 0.024 |
| 2 | Bat 존재 시 clean $s_\text{ball}$ 상승 여부 검증 | Cohen's $d$ = 0.745 |
| 3 | Clean evidence와 $\epsilon^*$ 간 consistency 확인 | Consistent(정의상 연결되어 있어 독립적 증거가 아닌 consistency check로만 보고) |
| 4/4B | Bat 영역 counterfactual 제거(sham edit과 비교)에 따른 ball evidence 변화 측정 | Bat 제거가 sham edit보다 unsupported ball evidence를 더 크게 감소시킴; 시각적 audit 수행 |
| 5 | Bat→ball 신호의 layer localization | Logit lens 기준 decoder layer 19에서 peak(Δ_bat_to_ball = 1.493 vs. Δ_sham = −0.030) |
| 6 | Layer 19 편집의 causal 효과 검증 | 효과 존재, 그러나 선택적이지 않음 — genuine ball evidence와 bat recognition도 동등하거나 더 크게 감소 |
| 7 | 선택적/완전 decoupled edit의 feasibility 검증(P1-P4) | 확립되지 않음 — Exp6의 non-selectivity에서 gating됨; P4(association knowledge 보존)는 측정되지 않음 |

**요약:** spurious effect를 가장 깨끗하게 감소시키는 single-direction, single-layer edit은 같은 layer에서 genuine object recognition도 함께 파괴함. Localization(Phase B, 여기서는 Exp5)은 확인되었으나, 이 해상도에서의 선택적 causal fix는 확립되지 않음.

### `src/cooccurrence_causal_coupling/` — Causal test의 population 확장

전체 write-up: [`src/cooccurrence_causal_coupling/README.md`](src/cooccurrence_causal_coupling/README.md). Case study의 causal 검증(위 Exp6)을 Stage 10-11의 전체 50개 이미지 / 3,039 pair population으로 확장한 실험. Naive diff-in-means 대신 fixed-effects-controlled 방향 추정을 사용하고, 4개 layer(3, 13, 16, 24)에 대한 λ-grid를 사전 등록한 뒤 train/val/test 이미지 분리 split으로 검증.

- FE-controlled 방향은 downstream $\beta$에 real하고 dose-dependent한 causal effect를 가짐: layer 24, λ=1.0에서 $\beta$가 99% 감소(0.306 → 0.004), λ에 대해 monotonic하며 random/shuffled-direction control을 명확히 상회.
- 동일 intervention이 같은 layer에서 genuine target-recognition evidence를 더 크게 파괴하고(−3.06 logit), low-co-occurrence absent target은 평평하게 유지되지 않고 오히려 상승 — 여전히 선택적이지 않음.
- **결론:** causal load-bearing은 population 규모에서도 확인되지만, 단일-pair case study의 non-selectivity 결론을 뒤집는 것이 아니라 population 규모로 일반화하는 결과임.

---

## Phase D — Debiasing 가능성

Phase C에서 representation editing이 선택적이지 않음을 확인한 것을 전제로, 동일한 `baseball bat → sports ball` pair에서 runtime edit이 아닌 학습(training)으로 spurious pathway를 줄이면서 genuine recognition을 보존할 수 있는지 검증하는 단계.

### `src/adversarial_functional_debiasing_pilot/`

전체 write-up: [`src/adversarial_functional_debiasing_pilot/README.md`](src/adversarial_functional_debiasing_pilot/README.md). LoRA fine-tuning(`q_proj`/`v_proj`만) 2가지 variant — Clean Debias(clean negative로 학습) vs. Adv Debias(ε=16/255 PGD-attacked negative로 학습) — 를 image-ID 기준으로 분리된 held-out clean 이미지에서 평가.

- **GO**: bat→ball coupling $B$가 1.937(Original) → 0.320(Clean Debias) → −0.182(Adv Debias)로 감소, genuine ball/bat recognition은 사전 등록된 10-pp non-selectivity threshold 내에서 유지됨.
- 단서: Adv-vs-Clean 순서(adversarial exposure를 구체적으로 뒷받침하는 패턴)는 n=20 test 이미지 기준 통계적으로 유의하지 않음(paired 95% CI for Adv−Clean: −0.631 ~ +0.439).

### `src/adversarial_signal_debiasing_pilot/`

전체 write-up: [`src/adversarial_signal_debiasing_pilot/README.md`](src/adversarial_signal_debiasing_pilot/README.md). 후속 실험으로, adversarial하게 유도된 layer-19 representation shift를 분해(PCA / PLS)하여 더 선택적인 spurious component 존재 여부를 검증하고, 해당 component만 억제하는 4번째 variant Adv+Decomp Debias를 학습.

- **NO-GO**: 가장 좋은 분해 candidate(PLS2)는 단순 mean-direction baseline은 상회하지만, 20개 random direction 중 최선의 것을 내부 selectivity test에서 상회하지 못함.
- Downstream 학습 결과로도 뒷받침됨: Adv+Decomp Debias는 coupling 측면에서 plain Adv Debias보다 낫지 않고, genuine Ball+ retention에서는 다른 모든 variant보다 낮음.
- Phase C의 결과를 다른 각도에서 재확인: layer-19 shift의 non-selectivity는 단일 diff-in-means 방향 사용에 따른 artifact가 아니며, PCA(unsupervised)와 PLS(supervised) 분해 모두에서 동일하게 유지됨.

---

## 현재까지의 정리

| 검증 항목 | Phase | 결과 |
|---|---|---|
| Co-occurrence가 높을수록 hallucination 유도가 쉬운지(forced yes/no) | A | 확인됨 |
| 동일 효과가 open-ended captioning에서도 성립하는지 | A | 미확립 — 파이프라인은 존재(Stage 6-8), 통제된 결과는 없음 |
| Bias가 frozen visual encoder에서 linear하게 읽히는지 | B | 아니오 |
| Bias가 LLM decoder stream에서 linear하게 읽히는지 | B | 확인됨, layer ≈13부터 |
| 그 표상이 최종 결정에서 causal하게 사용되는지 | C | 확인됨 (단일 pair 및 population 규모 모두) |
| 현재 선택적(non-destructive)인 causal edit이 가능한지 | C | 아니오 — 지금까지 찾은 최선의 edit도 선택적이지 않음 |
| Edit이 아니라 training으로 효과를 줄이면서 recognition을 보존할 수 있는지 | D | Adversarial LoRA fine-tuning은 가능성 있음(GO); decomposition 기반 selectivity(NO-GO)는 추가 이득 없음 |

모든 결과는 단일 모델(LLaVA-1.5-7B), 단일 데이터셋(COCO 2017) 기준. Stage별 상세 limitation은 각 stage 자신의 README에 기술되어 있음.

---

## File Structure

```text
configs/                                   Stage 1-11(일반) YAML config
src/cooc_diagnostic/                       Stage 1-11(일반) 공용 라이브러리
  coco_index.py                            COCO 이미지 단위 present-category 인덱싱
  cooccurrence_stats.py                    PMI/lift/raw conditional 계산
  covariates.py                            category별 평균 object area
  clip_similarity.py                       CLIP image-text similarity 및 image embedding
  strata_sampling.py                       candidate 생성 + high/low strata 구성
  matching.py                              Coarsened Exact Matching + balance table
  stratified_subsample.py                  CEM cell 비율을 유지하는 stratified subsampling
  llava_runtime.py                         LLaVA preprocessing/prompting/Yes-No 판정
  pgd_attack.py / random_attack.py         L∞ PGD attack 및 random-noise control
  epsilon_star.py                          ε* exponential search + binary search
  sanity_checks.py                         3가지 필수 sanity check 집계
  survival_analysis.py                     KM, stratified Cox, Weibull AFT, McNemar, Holm
  linear_probe.py                          Stage 5 linear probe (excess AUC)
  mention_detection.py / caption_attack.py Stage 6-8 open-ended/caption-mention readout + attack
  masking.py                               counterfactual region 제거 (Stage 11 case study)
scripts/run_stage{1..11}_*.py              stage별 실행 스크립트(Stage 11은 여러 개)
tests/                                      CPU-only unit test (62개, 전부 pass)
outputs/CooccurrenceHallucinationDiagnostic/stage{1..11}_*/   stage별 output (이 디렉토리가 아니라 repo-root의 outputs/)

src/cooccurrence_causal_coupling/          Phase C: population 규모 causal intervention (자체 README)
src/adversarial_functional_debiasing_pilot/  Phase D: LoRA debiasing pilot, Clean vs. Adv (자체 README)
src/adversarial_signal_debiasing_pilot/      Phase D: signal decomposition + 선택적 debiasing pilot (자체 README)
```

## Reproduction

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
export PYTHONPATH=src
PY=/opt/anaconda3/envs/py3_11/bin/python

# Unit tests (GPU 불필요; 62개 전부 pass해야 함)
$PY -m unittest discover -s tests -v

# Phase A / B, Stage 1-11(일반 population) -- 순서대로 실행, 각 stage는 이전 output에 의존
$PY scripts/run_stage1_cooccurrence.py            --config configs/stage1_cooccurrence.yaml
$PY scripts/run_stage2_sampling_matching.py       --config configs/stage2_sampling_matching.yaml        # GPU: CLIP
$PY scripts/run_stage3_attack.py                  --config configs/stage3_attack.yaml --pilot           # GPU: LLaVA; pilot 먼저
$PY scripts/run_stage3_attack.py                  --config configs/stage3_attack.yaml
$PY scripts/run_stage4_survival_analysis.py       --config configs/stage4_survival_analysis.yaml
$PY scripts/run_stage5_linear_probe.py            --config configs/stage5_linear_probe.yaml             # GPU: CLIP feature
$PY scripts/run_stage6_open_ended_transfer.py     --config configs/stage6_open_ended_transfer.yaml       # GPU: LLaVA
$PY scripts/run_stage7_caption_attack.py          --config configs/stage7_caption_attack.yaml            # GPU: LLaVA
$PY scripts/run_stage8_survival_analysis_caption.py --config configs/stage8_survival_analysis_caption.yaml
$PY scripts/run_stage9_clean_target_evidence.py   --config configs/stage9_clean_target_evidence.yaml     # GPU: LLaVA
$PY scripts/run_stage9_analysis.py                --config configs/stage9_analysis.yaml
$PY scripts/run_stage10_within_image_evidence.py  --config configs/stage10_within_image_evidence.yaml    # GPU: LLaVA
$PY scripts/run_stage10_analysis.py               --config configs/stage10_analysis.yaml
$PY scripts/run_stage11_layer_localization.py     --config configs/stage11_layer_localization.yaml       # GPU: LLaVA
$PY scripts/run_stage11_analysis.py               --config configs/stage11_analysis.yaml

# Phase C, Stage 11 case study (baseball bat -> sports ball), Exp0-Exp7
$PY scripts/run_stage11_exp0_pair_statistics.py      --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_build_sample.py         --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_attack.py               --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp2_clean_evidence.py       --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp3_boundary_consistency.py --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp4_counterfactual.py       --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp4b_visual_audit.py        --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp5_localization.py         --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp6_causal_intervention.py  --config configs/stage11_case_bat_ball.yaml --layer 19 --device cuda:0  # GPU
$PY scripts/run_stage11_exp7_decoupling.py           --config configs/stage11_case_bat_ball.yaml
# 전체 narrative: outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md

# Phase C, population 규모 causal coupling -- 자세한 내용은 src/cooccurrence_causal_coupling/README.md 참고
cd src/cooccurrence_causal_coupling/scripts
$PY 01_collect_hidden_states.py   --config ../configs/01_collect_hidden_states.yaml   # GPU
$PY 02_estimate_directions.py     --config ../configs/02_estimate_directions.yaml
$PY 03_screen_layers.py           --config ../configs/03_screen_layers.yaml           # GPU
$PY 04_full_intervention_scan.py  --config ../configs/04_full_intervention_scan.yaml   # GPU
$PY 05_analyze_intervention.py    --config ../configs/05_analyze_intervention.yaml

# Phase D, debiasing pilot -- 전체 command는 각 pilot 자신의 README.md 참고
cd ../../adversarial_functional_debiasing_pilot
# build_split.py -> generate_adversarial_forget_set.py -> train_lora_debias.py (2 variant) -> evaluate_model.py -> analysis/
cd ../adversarial_signal_debiasing_pilot
# prepare_data.py -> extract_layer19_shifts.py -> decompose_signal.py -> evaluate_components.py -> train_adv_decomp_debias.py -> evaluate_all_models.py
```
