# Future Work / Ideas

Ideas captured during development. Not in current scope — revisit when the
core pipeline is stable.

## 1. Resize: measure the lossy transform

**Context.** Resizing wafer maps to 64×64 with nearest-neighbor is a *lossy*
transform: a sparse defect die can be dropped (a resized cell lands on a
neighboring good die) or duplicated. Acceptable for shape-level classification,
but the loss is currently unmeasured — we resize and move on.

**The idea — don't just transform, measure the loss:**

1. **Die-preservation metric.** Per wafer, compare defect-die count (value == 2)
   before vs after resize. Report distribution + % of wafers downscaled vs
   upscaled (upscaling is near-lossless; risk concentrates in downscaled wafers).
2. **Quality gates (Pandera / Great Expectations).** Assert invariants after each
   transform step, fail-fast on violation:
    - resized values are a subset of {0, 1, 2}
    - defect-die count does not drop by more than a set threshold
    - no class loses all samples after the train/val/test split
    - row counts reconcile across stages
3. **Config-driven transform strategy.** Make the resize method swappable
   (nearest vs. channel-split max-pooling) and let logged metrics — die
   preservation + downstream accuracy — decide, rather than guessing.
   **Channel-split max-pooling (the "correct" alternative).** Split the map into
   binary channels (die-present, die-defective), resize the defect channel with a
   max rule: if any defect die falls in the source region, the output cell is a
   defect. Never drops a defect die. More complex; only build if nearest-neighbor
   hurts the sparse-defect classes (Scratch, Loc).

**When.** Defer to the `src/` refactor — that's where Pandera schemas belong.

## 2. Add a detection stage before classification

**Context.** We dropped `none` (~85% of all wafers), so the current model trains
only on the 25,519 defect wafers. It answers "given a defective wafer, which of
the 8 types?" — but has no concept of "no defect." Feed it a good wafer and it is
forced to pick a defect class, leaning toward the majority (Edge-Ring, ~38%).
This is a deliberate scope decision: classification, not detection.

**The idea — a two-stage pipeline that mirrors a real fab:**

- Stage 1 — detection: is the wafer defective? (binary: none vs defective)
- Stage 2 — classification: if defective, which type? (the current 8-class model)
  Fabs separate these because a single model fighting 85% `none` imbalance is hard
  to train and hard to evaluate per-stage. Two stages keep each model focused and
  let us measure detection and classification accuracy independently.

**Alternative.** A combined 9-class model with heavy `none` downsampling — simpler
to wire, but reintroduces the imbalance problem we dropped `none` to avoid.

**When.** Defer until the 8-class pipeline is solid. The current README must state
the classification-only scope so the assumption is explicit.

## 3. Inference-time prior correction

**Context.** WeightedRandomSampler balances the classes during training so the
model gets enough signal to learn every defect shape, including the rare ones
(Near-full, 104 train samples). But a balanced training distribution is not the
real fab distribution — Near-full is ~65x rarer than Edge-Ring in production.
Baking the real prior into training (keeping the data imbalanced) would starve
the rare classes of signal and the model never learns to detect them. So we
train balanced on purpose and accept that the model's raw outputs assume a
uniform prior, which does not match deployment.

**The idea — separate training representation from deployment prior:**

- Train balanced (sampler) so the model learns the *shape* of every class.
- Apply the real prior at inference, not at training, via logit adjustment:
  subtract the log of the train-time class frequency, add the log of the true
  deployment frequency. Same model, corrected to whatever prior actually holds.
- Tune per-class thresholds on asymmetric misclassification cost — missing a
  Near-full (a severe defect) costs more than a false alarm, so we optimize
  expected cost, not raw accuracy.
  The point: one trained model can serve shifting priors (fab A vs fab B, a new
  process that spikes Near-full) by changing config instead of retraining.

**Alternative.** Retrain on each deployment's real distribution — accurate to that
prior, but loses rare-class signal and needs a full retrain per prior shift.

**When.** Defer until baseline + MLflow + Grad-CAM (must-haves) are done. Ties
into drift detection: when the true prior drifts from the train prior, that *is*
label drift — detecting it and re-correcting the inference prior closes a clean
MLOps loop. Evaluation stays on a natural-distribution test set throughout.