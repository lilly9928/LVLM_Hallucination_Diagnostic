# 共现幻觉诊断 (Co-occurrence Hallucination Diagnostic)

## 研究假设

本项目检验当一个物体经常与图像中已存在的物体共同出现时，LVLM 是否更容易对该缺失物体产生幻觉。

> **如果目标物体 $A$（图像中不存在）与图像中存在的物体具有更高的共现率，则幻觉更容易发生。**

所有实验均使用 LLaVA-1.5-7B 与 COCO 2017（train2017 用于共现统计，val2017 用于所有面向模型的评估），全程 seed=42。

项目分为四个阶段（Phase），每个阶段建立在前一阶段的结果之上，而非重复叙述。任何阶段的详细数据都保存在该阶段自己的 outputs/README 中（下方有链接）；本文件仅追踪当前存在什么内容、其头条级发现，以及它与下一阶段的关联。

```
Phase A  行为效应（是否发生？）                Stage 1-8
Phase B  表征定位（存在于何处？）              Stage 9-11 (+ Stage 5)
Phase C  单对物体的机制性案例研究              Stage 11 case study + cooccurrence_causal_coupling
Phase D  去偏可行性（能否去除？）              adversarial_functional_debiasing_pilot + adversarial_signal_debiasing_pilot
```

---

## Phase A — 行为效应（Stage 1-8）

跨多个物体类别对的群体级流水线。这些阶段没有单独的 README；本节即为这些结果的主要写法。

| Stage | 测试内容 | 方法 | 头条结果 |
|---|---|---|---|
| **1** | COCO 中是否存在真实的物体共现结构 | 在 train2017（118,287 张图像）上计算 PMI，排除支持度 < 10 的对 | `mouse–keyboard` PMI = 3.65（提升 38.3×）；结构得到确认 |
| **2** | 将共现效应与混杂因素分离 | 高/低共现分组（对已存在物体的平均 PMI 的前/后 33%），基于边际频率/物体面积/CLIP 相似度的粗化精确匹配（CEM） | 77,828 个匹配对，匹配后所有协变量的 \|SMD\| < 0.04 |
| **3** | 通过攻击*诱导*幻觉的难度 | 对 `Is there a {A}?` 的定向 PGD 攻击，最小翻转预算 $\epsilon^*$（$L_\infty$，20 步 × 2 次重启，最大 $\epsilon=32/255$），150 个匹配对（300 个样本） | 通过 3 项必要的合理性检查（攻击翻转率 ≫ 随机噪声翻转率） |
| **4** | 高共现是否比低共现需要更小的 $\epsilon^*$ | 对 Stage 3 的 $\epsilon^*$ 进行分层 Cox / Weibull AFT / 配对 bootstrap / McNemar 检验，以 `pair_id` 为分层变量 | Cox HR = 1.627（p = 0.00325）；高共现下 ε\* 约小 42%；通过 Holm 校正后仍显著 |
| **5** | 该偏差在冻结视觉编码器中的线性可解码性 | 在 CLIP 视觉特征上进行线性探针，对比仅使用 Y（已存在物体）的基线，计算超额 AUC | 超额 AUC = −0.023，95% CI [−0.025, −0.020] — 无额外信息 |
| **6** | 面向 yes/no 优化的攻击迁移到开放式看图说话 | 复用 Stage 3 的对抗图像，检测自由格式描述中目标提及率 | 仅探索性 — 尚未按本项目的严谨标准得到受控结果；见 `outputs/.../stage6_open_ended_transfer/`，本文不作总结 |
| **7** | 用*面向描述*的攻击重现 Stage 3 效应 | 结构上克隆 Stage 3，读出方式改为在简短回答中提及目标类别，而非强制 yes/no | 流水线已构建并可端到端运行；经历了多次边界设计调整，且在小 N 下观察到跑次间噪声 — 尚非受控结果 |
| **8** | 高 vs. 低共现是否在描述攻击的 $\epsilon^*$ 上成立 | 对 Stage 7 的 $\epsilon^*$ 应用与 Stage 4 相同的生存分析 | 与 Stage 7 相同的警示 — 该跑次已完成，但被判定未受到良好控制；本文不作为发现报告（原始数据见 `outputs/.../stage8_survival_analysis_caption/`） |

**Phase A 小结：** 共现→幻觉效应对于强制 yes/no VQA 而言是真实且良好受控的（Stage 3-4），且不能仅归结为视觉编码器本身的信息（Stage 5）。Stage 6-8 探索该效应是否在开放式/描述式读出下同样成立；这些跑次已完成，但由于经历了较多中途设计调整以及小 N 噪声，未达到本项目对受控结果的标准，因此不作为发现报告（原始输出保留供参考，若要得出结论需要重新运行）。

---

## Phase B — 表征定位（Stage 9-11，群体层面）

从"攻击下的行为"转向"干净图像（ε=0）的内部证据"，定位偏差实际存在于何处，而不仅仅是它是否改变了攻击预算。

| Stage | 测试内容（论文标签） | 方法 | 头条结果 |
|---|---|---|---|
| **9**（Exp1） | 高共现是否在任何攻击之前就已经提高了干净目标正向证据 $s_T$ = logit(Yes) − logit(No) | 与 Stage 3 相同的 150 个匹配高/低对，在 $\epsilon=0$ 下的 $s_T$ | 均值差 = +0.729，95% CI [0.399, 1.061]，Cohen's $d_z$ = 0.353，p < 0.001 |
| **10**（Exp2） | 固定图像的情况下，与该具体图像更强的共现关系是否会提高 $s_T$（图像内特异性，而非仅是图像间混杂） | 50 张图像 × 74 个目标类别（3,039 对），双向固定效应模型 $s_T \sim \text{coocScore} + \text{imageFE} + \text{targetFE}$ | $\beta = 0.406$，95% CI [0.290, 0.522]，p = 6.6e-12，置换检验零假设 ≈ 0 |
| **11**（Exp3） | 该效应出现的层级定位 | 对 Stage 10 的 3,039 个对，在 LLaVA 全部 32 个解码器层的每一层应用 logit lens（最终 RMSNorm + lm_head），并逐层重新拟合 Stage 10 的精确固定效应模型 | 第 1-6 层信号 ≈0，第 7-12 层显著为负，第 13 层符号转为正，从第 16 层到最终层（第 32 层，恰好重现 Stage 10 的 $\beta$）稳定在偏相关 $r \approx 0.21$–$0.26$ |

**Phase B 小结：** 该效应（a）在无任何攻击的情况下已经存在，（b）与图像-目标的特定关系相关而非一般性频率混杂，且（c）从中间层起可从 LLM 自身的解码器流中线性解码 —— 这与 Stage 5 对冻结视觉编码器的否定结果形成对比。这将效应定位于第 ≥13 层的 LLM 解码器层，并引向 Phase C 对该表征是否被*因果地使用*的测试。

---

## Phase C — 单对物体的机制性案例研究

以上所有内容都是对多个物体类别对的汇总。Phase C 对**一个具体、可视觉检查的物体对**（`baseball bat` 上下文 → `sports ball` 目标）端到端地运行相同的测试链，使每一步都能被肉眼核实，然后从相关性推进到干预。

### Stage 11 案例研究 — `baseball bat → sports ball`

完整写法见：[`outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md`](../../../outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md)（Exp0-Exp7）。摘要：

| Exp | 测试内容 | 结果 |
|---|---|---|
| 0 | 物体对强度检查 | PMI = 2.342，提升 = 10.40，在 5,338 对中排名第 59 — **通过** |
| 1 | bat 上下文是否增加攻击脆弱性 | 中位数 $\epsilon^*$ 0.0001148（G10，bat+ball−）vs. 0.0005203（G00 对照），Cox HR = 1.760，p = 0.024 |
| 2 | 干净的 $s_\text{ball}$ 是否在 bat 存在时已经更高 | Cohen's $d$ = 0.745 |
| 3 | 干净证据与 $\epsilon^*$ 之间的一致性 | 一致（在定义上相关联，作为一致性检查报告，而非独立证据） |
| 4/4B | 移除 bat 区域（反事实）对 ball 证据的影响，与虚假编辑对比 | 移除 bat 比虚假编辑更多地降低了无支持的 ball 证据；已进行视觉审核 |
| 5 | bat→ball 信号的层级定位 | Logit lens 在解码器第 19 层达到峰值（Δ_bat_to_ball = 1.493 vs. Δ_sham = −0.030） |
| 6 | 编辑第 19 层的因果效应 | 效应存在，但**不具选择性** — 以相同或更大的幅度降低了真实的 ball 证据和 bat 识别能力 |
| 7 | 选择性、完全解耦编辑（P1-P4）的可行性 | **未确立** — 受 Exp6 非选择性结果的限制；P4（关联知识保留）甚至未被测量 |

**小结：** 最能干净地降低虚假效应的单方向、单层编辑，同时也在同一层破坏了真实的物体识别能力。定位（Phase B，此处的 Exp5）得到确认；但在此分辨率下的选择性因果修复未能确立。

### `src/cooccurrence_causal_coupling/` — 将因果检验推广至群体规模

完整写法见：[`src/cooccurrence_causal_coupling/README.md`](src/cooccurrence_causal_coupling/README.md)。在 Stage 10-11 的完整 50 张图像 / 3,039 对群体上重新运行案例研究的因果检验（上文 Exp6），使用固定效应控制的方向估计（而非朴素的均值差），并对 4 个层（3、13、16、24）进行预先注册的 λ 网格搜索，采用图像不重叠的训练/验证/测试划分。

- 固定效应控制的方向对下游 $\beta$ 具有真实的、剂量依赖的因果效应：在第 24 层、λ=1.0 时，$\beta$ 下降 99%（0.306 → 0.004），随 λ 单调变化，且明显优于随机/打乱方向的对照。
- 它**仍不具选择性**：同一干预以更大的幅度（−3.06 logits）破坏了同一层的真实目标识别证据，且低共现的缺失目标反而*上升*而非保持不变。
- **结论：** 因果承载作用在群体规模上得到确认，推广了单对案例研究的负向选择性发现，而非推翻它。

---

## Phase D — 去偏可行性

鉴于表征编辑不具选择性（Phase C），Phase D 探讨**训练**（而非运行时编辑）是否能在同一固定的 `baseball bat → sports ball` 对上降低虚假通路，同时保留真实的识别能力。

### `src/adversarial_functional_debiasing_pilot/`

完整写法见：[`src/adversarial_functional_debiasing_pilot/README.md`](src/adversarial_functional_debiasing_pilot/README.md)。对两种变体进行 LoRA 微调（仅 `q_proj`/`v_proj`）—— Clean Debias（在干净负样本上训练）vs. Adv Debias（在 ε=16/255 的 PGD 攻击负样本上训练）——在留出的干净图像上评估，训练/测试图像 ID 不重叠。

- **通过**：bat→ball 耦合度 $B$ 从 1.937（原始）降至 0.320（Clean Debias），再降至 −0.182（Adv Debias），且真实的 ball/bat 识别能力保持在预先注册的 10 个百分点非选择性阈值以内。
- 警示：Adv 与 Clean 之间的*排序*（能够专门支持对抗性暴露训练的模式）在 n=20 个测试图像下不具统计显著性（Adv−Clean 的配对 95% CI 跨越 −0.631 至 +0.439）。

### `src/adversarial_signal_debiasing_pilot/`

完整写法见：[`src/adversarial_signal_debiasing_pilot/README.md`](src/adversarial_signal_debiasing_pilot/README.md)。后续研究：分解对抗诱导的第 19 层表征偏移（PCA / PLS），检验是否存在更*具选择性*的虚假成分，然后训练第四种变体 Adv+Decomp Debias，仅抑制该成分。

- **未通过**：最佳的分解候选（PLS2）优于朴素的均值方向基线，但在内部选择性检验中并未明显优于 20 个随机方向中的最佳者。
- 下游训练结果印证了这一点：Adv+Decomp Debias 在耦合度上并不优于普通的 Adv Debias，且在真实 Ball+ 保留率上劣于其他所有变体。
- 从另一个角度强化了 Phase C 的发现：第 19 层偏移的非选择性并非单一均值差方向的伪影 —— 它在 PCA（无监督）和 PLS（有监督）分解下均依然存在。

---

## 当前进展

| 测试内容 | 阶段 | 结果 |
|---|---|---|
| 更高的共现率是否使幻觉更容易被诱导（强制 yes/no） | A | 已确认 |
| 开放式看图说话是否同样成立 | A | 尚未确立 — 流水线已存在（Stage 6-8），但没有写出受控结果 |
| 该偏差是否可从冻结视觉编码器中线性读出 | B | 不支持 |
| 该偏差是否可从 LLM 解码器流中线性读出 | B | 已确认，从第 ≈13 层起 |
| 该表征是否被最终决策因果性地使用 | C | 已确认（单对，且推广至完整群体） |
| 目前是否可实现选择性（非破坏性）的因果编辑 | C | 未实现 — 目前找到的最佳编辑都不具选择性 |
| 训练而非编辑是否能在保留识别能力的同时降低效应 | D | 对抗性 LoRA 微调有希望（通过）；分解引导的选择性（未通过）未带来额外提升 |

本文中的一切均为单模型（LLaVA-1.5-7B）、单数据集（COCO 2017）；具体局限性请参见各阶段自己的 README。

---

## 文件结构

```text
configs/                                   Stage 1-11（通用）YAML 配置
src/cooc_diagnostic/                       Stage 1-11（通用）共享库
  coco_index.py                            COCO 图像级已存在类别索引
  cooccurrence_stats.py                    PMI/提升/原始条件概率计算
  covariates.py                            类别级平均物体面积
  clip_similarity.py                       CLIP 图文相似度与图像嵌入
  strata_sampling.py                       候选生成 + 高/低分层构建
  matching.py                              粗化精确匹配（CEM）+ 平衡表
  stratified_subsample.py                  保持 CEM 单元比例的分层子采样
  llava_runtime.py                         LLaVA 预处理/提示/Yes-No 决策
  pgd_attack.py / random_attack.py         L∞ PGD 攻击与随机噪声对照
  epsilon_star.py                          ε* 指数搜索 + 二分搜索
  sanity_checks.py                         三项必要合理性检查的汇总
  survival_analysis.py                     KM、分层 Cox、Weibull AFT、McNemar、Holm
  linear_probe.py                          Stage 5 线性探针（超额 AUC）
  mention_detection.py / caption_attack.py Stage 6-8 开放式/描述提及读出 + 攻击
  masking.py                               反事实区域移除（Stage 11 案例研究）
scripts/run_stage{1..11}_*.py              执行脚本，每个阶段一个（Stage 11 有多个）
tests/                                      仅 CPU 单元测试（62 个测试，全部通过）
outputs/CooccurrenceHallucinationDiagnostic/stage{1..11}_*/   各阶段输出（位于仓库根目录 outputs/，不在本目录下）

src/cooccurrence_causal_coupling/          Phase C：群体规模因果干预（独立 README）
src/adversarial_functional_debiasing_pilot/  Phase D：LoRA 去偏试验，Clean vs. Adv（独立 README）
src/adversarial_signal_debiasing_pilot/      Phase D：信号分解 + 选择性去偏试验（独立 README）
```

## 复现方法

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
export PYTHONPATH=src
PY=/opt/anaconda3/envs/py3_11/bin/python

# 单元测试（无需 GPU；62 个测试应全部通过）
$PY -m unittest discover -s tests -v

# Phase A / B，Stage 1-11（通用群体）-- 按顺序运行，每一步依赖前一阶段的输出
$PY scripts/run_stage1_cooccurrence.py            --config configs/stage1_cooccurrence.yaml
$PY scripts/run_stage2_sampling_matching.py       --config configs/stage2_sampling_matching.yaml        # GPU: CLIP
$PY scripts/run_stage3_attack.py                  --config configs/stage3_attack.yaml --pilot           # GPU: LLaVA；先运行 pilot
$PY scripts/run_stage3_attack.py                  --config configs/stage3_attack.yaml
$PY scripts/run_stage4_survival_analysis.py       --config configs/stage4_survival_analysis.yaml
$PY scripts/run_stage5_linear_probe.py            --config configs/stage5_linear_probe.yaml             # GPU: CLIP 特征
$PY scripts/run_stage6_open_ended_transfer.py     --config configs/stage6_open_ended_transfer.yaml       # GPU: LLaVA
$PY scripts/run_stage7_caption_attack.py          --config configs/stage7_caption_attack.yaml            # GPU: LLaVA
$PY scripts/run_stage8_survival_analysis_caption.py --config configs/stage8_survival_analysis_caption.yaml
$PY scripts/run_stage9_clean_target_evidence.py   --config configs/stage9_clean_target_evidence.yaml     # GPU: LLaVA
$PY scripts/run_stage9_analysis.py                --config configs/stage9_analysis.yaml
$PY scripts/run_stage10_within_image_evidence.py  --config configs/stage10_within_image_evidence.yaml    # GPU: LLaVA
$PY scripts/run_stage10_analysis.py               --config configs/stage10_analysis.yaml
$PY scripts/run_stage11_layer_localization.py     --config configs/stage11_layer_localization.yaml       # GPU: LLaVA
$PY scripts/run_stage11_analysis.py               --config configs/stage11_analysis.yaml

# Phase C，Stage 11 案例研究（baseball bat -> sports ball），Exp0-Exp7
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
# 完整叙述见：outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md

# Phase C，群体规模因果耦合 -- 详见 src/cooccurrence_causal_coupling/README.md
cd src/cooccurrence_causal_coupling/scripts
$PY 01_collect_hidden_states.py   --config ../configs/01_collect_hidden_states.yaml   # GPU
$PY 02_estimate_directions.py     --config ../configs/02_estimate_directions.yaml
$PY 03_screen_layers.py           --config ../configs/03_screen_layers.yaml           # GPU
$PY 04_full_intervention_scan.py  --config ../configs/04_full_intervention_scan.yaml   # GPU
$PY 05_analyze_intervention.py    --config ../configs/05_analyze_intervention.yaml

# Phase D，去偏试验 -- 完整命令见各试验自己的 README.md
cd ../../adversarial_functional_debiasing_pilot
# build_split.py -> generate_adversarial_forget_set.py -> train_lora_debias.py（x2 变体） -> evaluate_model.py -> analysis/
cd ../adversarial_signal_debiasing_pilot
# prepare_data.py -> extract_layer19_shifts.py -> decompose_signal.py -> evaluate_components.py -> train_adv_decomp_debias.py -> evaluate_all_models.py
```
