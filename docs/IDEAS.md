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