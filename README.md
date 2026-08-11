# Co-occurrence Bias → Hallucination Diagnostic Experiment

## Research Question

**Are LVLMs more likely to hallucinate an absent object when that object frequently co-occurs with the objects present in the image?**

To test this, we perform a targeted adversarial attack on a target object (A) that is not present in the image, and measure the minimum perturbation budget (\epsilon^*) required to flip the model response from `No → Yes`.

The main hypothesis is:

> **If the target object (A) has higher co-occurrence with the objects present in the image, hallucination will occur at a smaller (\epsilon^*).**

All experiments use LLaVA-1.5-7B and COCO 2017.

---

## Stage 1. Object Co-occurrence Estimation

We first compute object co-occurrence statistics across all 80 COCO object categories using COCO train2017.

Since a simple conditional probability (P(B|A)) is strongly affected by the marginal frequency of common categories such as `person`, **PMI is used as the main co-occurrence score.**

* Dataset: COCO train2017, 118,287 images
* Main metric: PMI
* Pairs with support count < 10 are excluded from subsequent analysis
* 1,407 out of 3,160 total pairs are excluded

Semantically meaningful object pairs show high PMI values.

* `mouse–keyboard`: PMI = 3.65, lift = 38.3×
* `car–sink`: PMI = −3.79
* `person` marginal frequency = 54.2%

This confirms that COCO contains a clear object co-occurrence structure.

---

## Stage 2. High/Low Co-occurrence Group Construction

For each val2017 image, we define a target object (A) that is **not present in the image**.

The co-occurrence score between the target (A) and the set of objects present in the image (Y) is defined as:

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

Candidates are divided into:

* Top 33% → **High co-occurrence**
* Bottom 33% → **Low co-occurrence**
* Middle 33% → excluded

A direct comparison between the high and low groups may be confounded by object frequency, object size, or semantic similarity between the image and the target object. Therefore, we apply **Coarsened Exact Matching (CEM)**.

Matching variables:

* target marginal frequency
* average object area
* CLIP image–text similarity

The standardized mean difference (SMD) before and after matching is shown below.

| Covariate          | Before |  After |
| ------------------ | -----: | -----: |
| marginal frequency | +0.076 | −0.036 |
| avg. area          | −0.209 | −0.026 |
| CLIP similarity    | +0.290 | +0.005 |

A total of **77,828 high/low pairs** are successfully matched, and all three covariates satisfy (|SMD| < 0.04) after matching.

This allows the subsequent comparison to control for simple frequency and visual similarity effects.

---

## Stage 3. Measuring (\epsilon^*) with a Targeted Attack

For each image, we ask:

> `Is there a {A} in the image?`

We then perform a targeted PGD attack that attempts to change the model response from `No` to `Yes`.

The quantity of interest is **not attack success itself, but the minimum perturbation (\epsilon^*) required to flip the response.**

Interpretation:

* smaller (\epsilon^*)
  → easier to induce hallucination
* larger (\epsilon^*)
  → harder to induce hallucination

Final attack configuration:

* (L_\infty) PGD
* 20 steps × 2 restarts
* perturbations applied in the [0,1] pixel space before normalization
* maximum (\epsilon = 32/255)
* exponential search + binary search for (\epsilon^*)

From the matched pairs in Stage 2, 150 pairs are selected using stratified sampling, resulting in 300 total samples.

### Sanity Checks

Three controls are used to verify that the experiment behaves as intended.

1. **Questions about objects actually present in the image**

   Yes rate at (\epsilon=0): 93–97%

2. **Targeted attack**

   Attack success at (\epsilon=16/255): 100%

3. **Random noise**

   Flip rate under random noise with the same budget: 7–24%

The targeted attack exceeds random-noise flipping by **76–87 percentage points**.

This indicates that the response changes are caused by targeted perturbations rather than general sensitivity to image noise.

---

## Stage 4. High vs. Low Co-occurrence Comparison

The main question is:

> **Does hallucination occur with a smaller perturbation under the high co-occurrence condition?**

To preserve the matched-pair structure, we use stratified Cox regression with `pair_id` as the stratum.

### Results

| Analysis               | Result                                             |
| ---------------------- | -------------------------------------------------- |
| Stratified Cox         | **HR = 1.627**, 95% CI [1.177, 2.250], p = 0.00325 |
| Baseline hallucination | High 21.3% vs Low 6.0%                             |
| Weibull AFT            | **Time ratio = 0.585**, p = 0.00031                |
| Paired bootstrap       | median Δε* = −0.00038                              |
| Log-rank               | p = 0.00026                                        |
| McNemar                | p = 0.000117                                       |

The main results remain significant after Holm correction.

According to the Weibull AFT analysis, the required (\epsilon^*) is approximately **42% smaller under the high co-occurrence condition.**

This provides behavioral evidence that:

> **An absent target object is easier to hallucinate when it strongly co-occurs with the objects present in the image.**

---

## Stage 5. Is the Bias Directly Encoded in Visual Encoder Features?

Stages 3–4 show that **the behavior of the full LLaVA model changes depending on co-occurrence.**

However, this alone does not show that the effect originates from the **visual encoder itself**.

We therefore test whether high/low co-occurrence conditions can be distinguished using frozen CLIP visual features with a linear probe.

A key issue is that the set of present objects (Y) already provides substantial information about whether a sample belongs to the high or low co-occurrence group.

Therefore, instead of looking only at visual-feature AUC, we test whether visual features provide additional information beyond a **Y-only baseline**.

### Results

| Probe               |              AUC |
| ------------------- | ---------------: |
| Y-only baseline     |            0.799 |
| CLIP visual feature |            0.776 |
| **Excess AUC**      |       **−0.023** |
| 95% CI              | [−0.025, −0.020] |

The visual features do **not** provide additional predictive information beyond the Y-only baseline.

Therefore, the current results do not support the claim that:

> “Co-occurrence bias is directly encoded in the frozen CLIP visual encoder features.”

---

## Overall Results

| Question                                                           | Result                      |
| ------------------------------------------------------------------ | --------------------------- |
| Does higher co-occurrence make hallucination easier to induce?     | **Yes**                     |
| Is the difference explained by frequency / area / CLIP similarity? | Controlled through matching |
| Does the same effect appear under random noise?                    | **No**                      |
| Is the bias linearly readable from frozen CLIP features?           | **No**                      |

Stages 3–4 provide **behavioral evidence that hallucination susceptibility changes with object co-occurrence.**

In contrast, Stage 5 does not show that the same information is linearly separable from frozen CLIP visual features.

So far, the supported relationship is:

[
\text{High Co-occurrence}
\rightarrow
\text{smaller }\epsilon^*
\rightarrow
\text{easier hallucination}
]

while the following has **not** yet been established:

[
\text{Co-occurrence Bias}
\rightarrow
\text{Visual Encoder Representation}
]

---

## Current Conclusion

**Co-occurrence bias appears to affect hallucination behavior in LLaVA.**

The high co-occurrence condition shows both a higher baseline hallucination rate and a smaller adversarial perturbation required to induce hallucination.

However, the current experiments do not identify whether this effect originates from the **visual encoder, vision-language fusion, or language-model prior**.

In particular, Stage 5 does not reveal additional co-occurrence information in frozen CLIP features.

> **The next step is therefore to localize where co-occurrence information is actually represented within the model.**

---

## Limitations

* Stage 3 uses only 150 matched pairs out of the full 77,828 matched pairs.
* PGD is run with 20 steps × 2 restarts due to computational cost.
* Stage 5 only tests a linear probe, so nonlinear representations of co-occurrence information are not ruled out.
* The current attack is designed for closed-form Yes/No questions and therefore does not directly establish transfer to open-ended caption hallucination.

---

## File Structure

```text
configs/                                  stage-specific YAML configurations
src/cooc_diagnostic/
  coco_index.py                           COCO image-level present-category indexing
  cooccurrence_stats.py                   PMI/lift/raw conditional computation
  covariates.py                           category-level average object area
  clip_similarity.py                      CLIP image-text similarity and image embeddings
  strata_sampling.py                      candidate generation + high/low strata construction
  matching.py                             Coarsened Exact Matching + balance table
  stratified_subsample.py                 stratified subsampling preserving CEM cell ratios
  llava_runtime.py                        LLaVA preprocessing/prompting/Yes-No decision
  pgd_attack.py / random_attack.py        L∞ PGD attack and random-noise control
  epsilon_star.py                         ε* exponential search + binary search
  sanity_checks.py                        aggregation of three required sanity checks
  survival_analysis.py                    KM, stratified Cox, Weibull AFT, McNemar, Holm
  linear_probe.py                         Stage 5 linear probe (excess AUC)
scripts/run_stage{1..5}_*.py              execution scripts for each stage
tests/                                    CPU-only unit tests (53 tests, all passing)
outputs/CooccurrenceHallucinationDiagnostic/stage{1..5}_*/   stage-wise outputs
```

## Reproduction

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
export PYTHONPATH=src

# Unit tests (GPU not required; all 53 tests should pass)
python -m unittest discover -s tests -v

# Stage 1
python scripts/run_stage1_cooccurrence.py \
  --config configs/stage1_cooccurrence.yaml

# Stage 2 (GPU required for CLIP similarity)
python scripts/run_stage2_sampling_matching.py \
  --config configs/stage2_sampling_matching.yaml

# Stage 3 (GPU required for LLaVA-1.5-7B)
# Run the pilot first to verify sanity checks and runtime
python scripts/run_stage3_attack.py \
  --config configs/stage3_attack.yaml \
  --pilot

python scripts/run_stage3_attack.py \
  --config configs/stage3_attack.yaml

# Stage 4 (GPU not required)
python scripts/run_stage4_survival_analysis.py \
  --config configs/stage4_survival_analysis.yaml

# Stage 5 (GPU required for CLIP feature extraction)
python scripts/run_stage5_linear_probe.py \
  --config configs/stage5_linear_probe.yaml
```