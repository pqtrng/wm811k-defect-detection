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

## 4. Advanced augmentation (beyond 90° rotations + flips)

**Context.** The active roadmap already covers *basic* geometric augmentation —
90°/180°/270° rotations + flips — as the step right after ResNet-18. Those are
safe for this domain: wafer defects are largely orientation-invariant (a rotated
Scratch is still a Scratch), rotations/flips about the center preserve the
edge-vs-center distinction, and 90° rotations only permute pixels so they keep
the discrete {0, 1, 2} values intact (no interpolation). That basic augmentation
is **not** captured here because it is in-scope, not deferred. This section is for
the *harder* augmentation ideas we deliberately postponed.

**The ideas — richer augmentation, each with a catch to design around:**

1. **Learned augmentation policy (RandAugment / AutoAugment / TrivialAugment).**
   Instead of hand-picking transforms, search for an augmentation policy. Only
   worth it if manual 90°+flip plateaus — otherwise it adds tuning cost for
   marginal gain. Most of its default ops (shear, translate, color) are either
   unsafe (translate breaks edge-vs-center) or meaningless (color on a 1-channel
   discrete map), so it would need a wafer-safe op subset first.
2. **Continuous rotation / elastic deformation.** Free-angle rotation and elastic
   warps create richer variants than 90° steps, but both require interpolation,
   which produces off-grid values outside {0, 1, 2} — the same reason we use
   INTER_NEAREST at resize. To do this correctly, split into binary channels
   (die-present, die-defective), deform each, then re-threshold — mirrors the
   channel-split approach in Idea #1. Only build if 90°+flip proves insufficient
   for the ambiguous classes (Loc, Scratch).
3. **MixUp / CutMix.** Blend two wafers (and their labels) to regularize. Powerful
   on natural images, but questionable here: a linear blend of two wafer maps is
   not a physically valid wafer, and a soft label between, say, Scratch and Donut
   has no fab meaning. Would need validation that blended samples help rather than
   teach artifacts. Lowest-priority of the three.

**Alternative.** Stick with basic 90°+flip permanently. If it closes the
Loc/Scratch gap enough, none of the above is needed — augmentation complexity is
only justified by a measured shortfall, not by default.

**When.** Defer until basic augmentation (90°+flip) has been run and measured on
ResNet-18. Let the MLflow numbers decide: only reach for advanced augmentation if
the basic version leaves a clear, quantified gap on the ambiguous classes.

## 5. Browser-side ONNX demo on the portfolio site

**Context.** The portfolio currently *describes* the classifier; a visitor has to
trust the README numbers. A live demo inverts that: a professor clicks, feeds a
wafer, sees the prediction. The constraint is skills-honesty over time — any demo
linked from application documents must still be alive in November 2026 and beyond
with zero maintenance. A hosted API (Cloud Run) can die from billing, cold starts,
or CORS; a dead demo is worse than no demo.

**The idea — run the final model in the browser, no server:**

- Export the canonical checkpoint (resnet18-aug, test macro-F1 0.920) to ONNX;
  the x/2.0 normalization lives inside forward() so it exports with the graph.
- Quantize to int8 (~11 MB from ~43 MB fp32) and commit the artifact to the
  portfolio repo; load with onnxruntime-web. 64×64 single-channel inference is
  effectively instant client-side.
- Input UI matches the data contract, not "upload an image": preloaded sample
  wafers per class from the test set, plus a 64×64 draw-your-own-defect grid
  emitting {0,1,2} values. Invalid input is rejected by the same validation
  rules as the pipeline — the rejection is part of the demo.
- Acceptance before claiming: ONNX output must match PyTorch output on a fixed
  batch (parity check, same discipline as the notebook→CLI refactor).

**Alternative.** Deploy the T10 FastAPI image to Cloud Run and call it from the
static site — demonstrates the real serving stack (registry, model_version,
server-side validation) but adds CORS, cold starts, and a billing dependency
that can silently kill the link. Keep as an interview-only live demo, not as
the always-on portfolio link.

**When.** Defer until after T10 (serving contract settled) or fold into the
pending portfolio-number sweep, whichever comes first. Not part of the T6–T12
sprint; the September 30 deadline takes precedence.