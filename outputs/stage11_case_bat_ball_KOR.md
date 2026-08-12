# Stage 11 — 케이스 스터디: "baseball bat" → "sports ball"

## 핵심 질문

Stage 1–5에서는 여러 object pair에 걸친 **집단 수준(population-level)** 행동 효과를 확인함: 이미지 내 존재하는 객체들과 co-occurrence가 높은 객체일수록 hallucination이 쉽게 일어남. Stage 11은 이 질문을 **하나의 구체적이고 시각적으로 확인 가능한 pair**에 대해 처음부터 끝까지(end-to-end) 다시 검증하고, 한 단계 더 나아감 — 행동적 증거 → counterfactual 입력 레벨 검증 → 모델 내부 localization → causal intervention 순으로 진행함.

> `baseball bat` ↔ `sports ball`의 co-occurrence가 bat context → 근거 없는 ball evidence → hallucination 취약성으로 이어지는 spurious functional pathway를 만드는가? 그리고 그 pathway를 모델 내부에서 선택적으로(selectively) 제거하면서도 진짜 ball evidence, bat 인식, 전반적인 출력 품질은 그대로 유지할 수 있는가?

모델: `llava-hf/llava-1.5-7b-hf` (Stage 3/9/10과 동일한 runtime). 데이터: COCO 2017. Seed: 42로 고정. 가능한 모든 부분에서 Stage 1–5의 모듈(PMI, matching, LLaVA runtime, PGD/epsilon*, probe fitting)을 그대로 재사용함 — 재사용 내역 전체는 `outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/repository_audit.md` 참고.

이 문서는 방법론 논문이 아니라 진단적 케이스 스터디임: **모든 단계가 실패할 수 있도록 열어두었고**, 실제로 한 단계(Exp6)는 실패함 — 아래에 나온 그대로 보고함.

---

## Exp0 — 이 pair가 실제로 강한 관계인가?

Train2017 (118,287장): N(bat) = 2506, N(ball) = 4262, N(bat & ball) = 939.

$$\text{PMI(bat, ball)} = 2.342 \qquad \text{lift} = 10.40 \qquad P(\text{ball}\mid\text{bat}) = 0.375 \text{ vs. } P(\text{ball}) = 0.036$$

PMI 기준 COCO category pair 5,338개 중 **59위** — 상위 약 1%에 해당하는 강한 연관성. Val2017 그룹 크기: G00 (bat−,ball−) = 4766, **G10 (bat+,ball−) = 65**, G01 (bat−,ball+) = 137, G11 (bat+,ball+) = 32.

**결론: GO.** (G10 65장 전체를 이후 모든 실험에 그대로 사용함 — Stage 11 어디에서도 treatment arm을 추가로 subsampling하지 않음.)

---

## Exp1 — Phenomenon: bat context가 attack 취약성을 높이는가?

G10 65장 vs. matching된 G00 65장 (Stage 2와 동일한 coarsened exact matching 방식을 재사용, scene complexity와 "sports ball"/"baseball bat"에 대한 CLIP 유사도로 matching). Stage 3와 동일한 PGD/$\epsilon^*$ attack (20 step × 2 restart, $\epsilon_{max}=32/255$, 설정 변경 없음).

| 분석 | 결과 |
|---|---|
| Stratified Cox (주 분석) | **HR = 1.760**, 95% CI [1.077, 2.875], p = 0.024 |
| Weibull AFT time ratio | **0.416**, 95% CI [0.254, 0.683], p = 0.0005 |
| 중앙값 $\epsilon^*$ | G10 = 0.000115 vs. G00 = 0.000520 |
| McNemar (clean Yes rate) | p = 0.69 (**유의하지 않음**) |

Bat context가 있는 경우 matching된 대조군 대비 약 **42%의 perturbation budget**만으로 ball을 hallucination시킴 — 다만 이 효과는 연속적인 evidence/취약성 지표에서 나타나며, 이 표본 크기에서 "$\epsilon=0$에서 이미 hallucination" 여부의 binary rate에서는 유의하지 않음. **결론: GO.**

---

## Exp2 — Attack 이전의 clean functional evidence

*"Is there a sports ball in the image?"*에 대한 $s_{ball} = \text{logit(Yes)} - \text{logit(No)}$ (attack 없음):

| 그룹 | n | 평균 $s_{ball}$ | Yes rate |
|---|---|---|---|
| G00 (bat−,ball−) | 65 | −1.260 | 0.215 |
| **G10 (bat+,ball−)** | 65 | **−0.320** | 0.262 |
| G01 (bat−,ball+) | 65 | 2.580 | 0.785 |
| G11 (bat+,ball+) | 32 | 1.665 | 0.813 |

G10 vs G00: 평균 차이 = **+0.940**, Cohen's d = 0.745, bootstrap 95% CI [0.50, 1.36], Mann-Whitney p = 1.75e-6. **결론: GO.**

---

## Exp3 — Boundary Consistency (새로운 증거가 아니라 일관성 체크)

Exp1과 Exp2를 image_id로 join: $s_{ball}$과 $\epsilon^*$ 사이 Spearman ρ = **−0.895**. $\epsilon^*$가 $s_{ball}$과 동일한 Yes/No margin에서 유도되기 때문에 이 강한 관계는 어느 정도 기계적으로(mechanically) 예상되는 결과이며, 독립적인 causal evidence가 아니라 일관성 체크로만 취급함.

---

## Exp4 — Bat 자체 때문인가, 단순 이미지 스타일 때문인가? (Counterfactual)

G10 65장 전체에 대해: bat 영역을 gray-fill로 제거 (다른 실험에서 이미 검증된 `mask_dog_regions` 기법을 임의 category로 일반화) vs. 동일 면적의 **mirror + translate된 sham** mask.

$$\Delta_{bat \to ball} = s_{ball}(\text{원본}) - s_{ball}(\text{bat 제거})$$

| 항목 | 값 |
|---|---|
| $\Delta_{bat \to ball}$ 평균 | **0.238**, 95% CI [0.114, 0.380], Wilcoxon p = 0.00046 |
| $\Delta_{sham}$ 평균 | −0.015 (null intervention이므로 예상대로 ≈ 0) |
| $\Delta_{bat} - \Delta_{sham}$ | **0.254**, 95% CI [0.108, 0.407], p = 0.00085 |

두 가설 모두 성립: bat을 제거하면 근거 없는 ball evidence가 줄어들고, 동일 면적의 sham 개입보다 **더 크게** 줄어듦. (결과 해석 전 필수인) 시각적 감사에서 65장 중 4장(6%)은 mask가 적용되었음에도 시각적 효과가 거의 없었음 — LLaVA 자체의 resize+crop 전처리 과정에서 작거나 가장자리에 있는 bat 영역이 소실된 것으로 보임 — 그리고 65장 중 7장(11%)의 sham 배치는 다른 객체와의 overlap을 완전히 해소하지 못함. 두 경우 모두 통계에서 제외하지 않고 투명하게 기록함.

### Checkpoint 1: **GO**

네 가지 기준(evidence 상승, bat 특이적 감소, sham 초과, artifact 공개 및 결과 무효화 수준 아님) 모두 충족 → 내부 localization으로 진행.

---

## Exp5 — 모델 내부 어디에서 나타나는가?

Readout: LLM의 33개 readout 지점(embedding + decoder 32개 층) 전체에 대해 **logit lens** (final RMSNorm + `lm_head`를 적용하고 yes/no token으로 제한); vision tower와 projector 단계는 **logistic probe** (Stage 5의 `fit_probe`를 그대로 사용, train2017에서 별도로 추출한 500장 학습셋으로 학습).

Vision tower/projector 단계에서는 신호가 작고(0.32 / 0.25, sham은 0.11 / 0.02), LLM 1–13층까지는 작고 noisy하게 유지되다가, 이후 급격히 상승하여 **32층 중 19번째 층에서 최고점**을 찍음 ($\Delta_{bat}=1.493$ vs. $\Delta_{sham}=-0.030$ — 전체 프로파일에서 가장 뚜렷한 분리). 이후 출력 방향으로 갈수록 다시 감소함. 이는 **candidate localization**일 뿐 causal claim이 아님.

---

## Exp6 — 해당 층을 편집하면 선택적으로 문제가 해결되는가?

Exp5의 최고점에서 직접 동기를 얻은 개입: layer 19의 출력에서, G10 이미지들의 (원본 − bat 제거) hidden-state 평균 차이 방향(direction)을 scoring 시점에 모든 token position에서 빼줌 — 사용 가능한 가장 덜 파괴적인 단일 방향 개입.

| 테스트 | $s$ 감소량 | Yes rate (전 → 후) |
|---|---|---|
| **Main: 근거 없는 ball (G10)** | **0.257** (p=1.7e-12) | 0.262 → 0.154 |
| Control 1: 진짜 ball (G01) | 0.275 (p=2.6e-8) | 0.750 → 0.750 |
| Control 1: 진짜 ball (G11) | 0.256 (p=6.8e-7) | 0.813 → 0.688 |
| Control 2: bat 인식 (G10) | 0.320 (p=1.9e-12) | 0.969 → 0.969 |
| Control 3: 일반 caption | 10개 중 8개 완전 동일, 나머지도 표현만 소폭 변경 | 붕괴 없음 |

**네 가지 감소량이 사실상 전부 비슷한 크기임.** 이는 전형적인 **non-selective**(비선택적) 개입의 signature임 — bat-ball에 특이적인 pathway가 아니라, 해당 층에서 yes 쪽으로 기운 일반적인 evidence를 뭉뚱그려 억제하는 것에 가까움. 전반적인 출력 품질은 손상되지 않았지만(Control 3), 선택성(selectivity)에는 실패함.

### Checkpoint 2: **STOP (non-selective)**

---

## Exp7 — 선택적 decoupling이 가능한가?

| 요구 조건 | 결과 | 충족? |
|---|---|---|
| P1: spurious effect 감소 | 감소량 = 0.257, p=1.7e-12 | **YES** |
| P2: 진짜 ball evidence 유지 | main effect와 거의 같은 크기로 감소 | **NO** |
| P3: bat/context 정보 유지 | main effect보다 *더 크게* 감소 | **NO** |
| P4: association 지식 유지 | 측정하지 않음 — 적절한 지표를 이번 범위 안에서 찾지 못함 | 해당 없음 |

**측정 가능한 3개 요구 조건 중 1개만 충족.** 이 개입, 이 층, 이 pair 조합으로는 **functional decoupling이 성립하지 않음.**

---

## 전체 결과

| 질문 | 답 |
|---|---|
| Train set에서 강한 연관성이 있는가? | **Yes** (PMI=2.34, 순위 59/5338) |
| Clean 상태에서 근거 없는 ball evidence가 더 높은가? | **Yes** (d=0.75, p=1.75e-6) |
| Attack 취약성이 더 높은가? | 연속적 evidence 기준 **Yes** (Cox p=0.024); raw clean-Yes rate 기준 **아님** (p=0.69) |
| Sham 대비 bat 자체의 효과인가? | **Yes** ($\Delta_{bat}>\Delta_{sham}$, p=0.00085) |
| 모델 내부에서 localize 되는가? | LLM 19층에서 **candidate localization** |
| 그 층이 causal하고 selective하게 책임이 있는가? | **아니오** — non-selective (Exp6) |
| 선택적 decoupling이 가능한가? | **아니오**, 측정 가능한 요구 조건 3개 중 1개만 충족 |

$$
\text{Bat}
\;\to\;
\underbrace{\text{근거 없는 Ball evidence}}_{\text{Exp2, Exp4 — SUPPORTED}}
\;\to\;
\underbrace{\text{LLM 19층의 candidate readout}}_{\text{Exp5 — localization만}}
\;\to\;
\underbrace{\text{Hallucination 취약성}}_{\text{Exp1 — SUPPORTED}}
$$

"19층이 causal하고 selective하게 이 효과를 매개한다"는 화살표는 **사실로 그리지 않음** — Exp6에서 직접 검증했고 selectivity 기준을 통과하지 못했음. 이는 미완성 단계가 아니라 그 자체로 유의미한 negative result임.

---

## 주장할 수 없는 것

- 단일 pair, 단일 모델 — 다른 pair나 다른 모델로의 일반화는 전혀 함의하지 않음.
- Exp6의 null result는 *이* difference-of-means 개입을 *이* 층에 적용했을 때의 결과이며, 이 pathway에 대해 어떤 selective intervention도 작동할 수 없다는 증명은 아님.
- Exp4의 bat 제거 mask 중 약 6%는 시각적 효과가 없었음(전처리 스케일 artifact); sham 배치 중 약 11%는 다른 객체와의 overlap을 완전히 해소하지 못함.
- Exp1의 matched sample은 scene complexity 기준으로는 잘 balance되어 있지만, "baseball bat"에 대한 CLIP 유사도는 balance되지 않음(당연한 결과: G10 이미지는 실제로 bat을 포함하므로), "sports ball" 유사도도 완전히 balance되지는 않음.
- Association 지식 보존(P4)은 **측정하지 않음** — `outputs/.../stage11_case_bat_ball/association_measure_definition.md` 참고.
- Exp3의 상관관계는 독립적 증거가 아니라 일관성 체크임.

---

## 파일 구조

```text
configs/stage11_case_bat_ball.yaml                   Exp0-7 전체 공통 설정
scripts/run_stage11_exp0_pair_statistics.py          Exp0
scripts/run_stage11_exp1_build_sample.py             Exp1a: CEM 기반 G10/G00 matched sample 구성
scripts/run_stage11_exp1_attack.py                   Exp1b: epsilon* attack + survival 분석
scripts/run_stage11_exp2_clean_evidence.py           Exp2 + Exp2B ranked gallery
scripts/run_stage11_exp3_boundary_consistency.py     Exp3
scripts/run_stage11_exp4_counterfactual.py           Exp4: bat 제거 vs. sham
scripts/run_stage11_exp4b_visual_audit.py            Exp4B: 필수 시각적 감사 figure
scripts/run_stage11_exp5_localization.py             Exp5: logit-lens + probe 기반 layer별 readout
scripts/make_exp5_figure.py                          Exp5 figure 생성
scripts/run_stage11_exp6_causal_intervention.py      Exp6: 19층 direction ablation + control
scripts/run_stage11_exp7_decoupling.py               Exp7: decoupling feasibility 요약
src/cooc_diagnostic/masking.py                        신규: gray-fill 제거 + mirror/translate sham

outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/
  repository_audit.md                                 Step 0: 재사용 내역 vs. 신규 구현 내역
  exp0_pair_statistics.json, exp0_group_membership.csv
  exp1_sample_selection.csv, exp1_sample_balance.json, exp1_epsilon_star.csv, exp1_statistics.json
  exp2_clean_evidence.csv, exp2_statistics.json, exp2_g10_ranked.csv, exp2_g10_selections.json
  exp3_boundary_consistency.csv, exp3_statistics.json
  exp4_counterfactual.csv, exp4_statistics.json, exp4b_selected_examples.json, counterfactual_images/
  exp5_layerwise_evidence.csv, exp5_statistics.json
  exp6_internal_intervention.csv, exp6_statistics.json
  exp7_decoupling_summary.csv, exp7_statistics.json
  checkpoint1.md, checkpoint2.md, association_measure_definition.md, figures/
  README.md                                            전체 서술형 결과 문서 (이 파일의 원본)

notebook/baseball_bat_sports_ball_case_analysis.ipynb   실행 완료(에러 0건) -- 모든 plot/gallery 포함
```

## 재현

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
PY=/opt/anaconda3/envs/py3_11/bin/python

$PY scripts/run_stage11_exp0_pair_statistics.py --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_build_sample.py    --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_attack.py          --config configs/stage11_case_bat_ball.yaml --device cuda:2
$PY scripts/run_stage11_exp2_clean_evidence.py  --config configs/stage11_case_bat_ball.yaml --device cuda:0
$PY scripts/run_stage11_exp3_boundary_consistency.py --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp4_counterfactual.py  --config configs/stage11_case_bat_ball.yaml --device cuda:3
$PY scripts/run_stage11_exp4b_visual_audit.py   --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp5_localization.py    --config configs/stage11_case_bat_ball.yaml --device cuda:2
$PY scripts/make_exp5_figure.py                 --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp6_causal_intervention.py --config configs/stage11_case_bat_ball.yaml --layer 19 --device cuda:2
$PY scripts/run_stage11_exp7_decoupling.py      --config configs/stage11_case_bat_ball.yaml

# 노트북 (저장된 결과만 로드, 추론 재실행 없음):
cd ../../../notebook
jupyter nbconvert --to notebook --execute --inplace baseball_bat_sports_ball_case_analysis.ipynb \
    --ExecutePreprocessor.kernel_name=py3_11
```

전체 서술, 모든 수치, 모든 한계 사항은
`outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md`에 있음.
