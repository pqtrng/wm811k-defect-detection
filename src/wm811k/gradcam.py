"""Grad-CAM visualization for wm811k models.

Answers "where does the network look when it predicts this class" by weighting
the final conv feature maps (layer4, [512, 8, 8]) with the spatially-averaged
gradients of the target logit, then upsampling the ReLU'd sum onto the wafer.

Two outputs:
- gradcam.png: one correctly-classified sample per class (8 panels).
- gradcam_confusions.png: misclassified samples for the Loc<->Edge-Loc and
  Loc<->Scratch pairs, with the CAM computed w.r.t. the PREDICTED class --
  showing what the model looked at when it got it wrong.

Inputs are raw dataset tensors (die values {0,1,2}); the model normalizes
x/2.0 internally in forward() -- do NOT normalize here (double-normalization
was a real bug class in the original notebook).

Usage:
    uv run python -m wm811k.gradcam --checkpoint models/resnet18-aug_best.pt \
        --model resnet18 --split test --out docs/figures/gradcam.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on servers/CI
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wm811k.config import Config, load_config
from wm811k.data import build_loaders
from wm811k.models import build_model
from wm811k.seed import set_seed


class GradCAM:
    """Minimal Grad-CAM: forward hook for activations, backward hook for gradients.

    Call remove() when done (or use as a context manager) so the hooks don't
    leak into later uses of the model.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    @torch.enable_grad()
    def generate(
            self, x: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Compute a CAM for one sample.

        Args:
            x: raw wafer tensor [1, 1, 64, 64], values in {0, 1, 2}.
            class_idx: logit to explain; defaults to the predicted class.

        Returns:
            (cam upsampled to 64x64 and normalized to [0, 1], predicted class).
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        pred = int(logits.argmax(dim=1).item())
        target = pred if class_idx is None else class_idx
        logits[0, target].backward()

        # weights: global-average-pooled gradients, one scalar per channel
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, 8, 8]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred


@torch.no_grad()
def _predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """One inference pass over the loader; returns (preds, labels) as arrays."""
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        preds.append(out.argmax(dim=1).cpu())
        labels.append(yb)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def _plot_grid(panels: list[dict], ncols: int, out_path: Path, suptitle: str) -> None:
    """panels: [{'x': [64,64] array, 'cam': [64,64] array, 'title': str}, ...]"""
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes, panels):
        ax.imshow(panel["x"], cmap="gray_r", vmin=0, vmax=2)
        ax.imshow(panel["cam"], cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        ax.set_title(panel["title"], fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _first_index(mask: np.ndarray) -> int | None:
    """Deterministic sample choice: first index where mask holds (dataset is unshuffled)."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM grids for a trained checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt state_dict")
    parser.add_argument(
        "--model", default="resnet18", choices=["resnet18"],
        help="Grad-CAM hooks layer4; only resnet18 is supported",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--out", default="docs/figures/gradcam.png",
        help="Path for the per-class grid; the confusion grid is written next to it "
             "as gradcam_confusions.png",
    )
    args = parser.parse_args()

    config: Config = load_config(args.config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # augment=False always: explanations must be computed on raw data.
    loaders = dict(zip(["train", "val", "test"], build_loaders(config, augment=False)))
    loader = loaders[args.split]
    dataset = loader.dataset

    model = build_model(args.model, num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    preds, labels = _predict_all(model, loader, device)
    label2idx = config.label2idx

    out_path = Path(args.out)
    confusions_path = out_path.with_name("gradcam_confusions.png")

    with GradCAM(model, target_layer=model.layer4) as cam_engine:

        def make_panel(sample_idx: int, title: str, class_idx: int | None = None) -> dict:
            x, _ = dataset[sample_idx]
            x = x.unsqueeze(0).to(device)  # raw {0,1,2}; model normalizes internally
            cam, _ = cam_engine.generate(x, class_idx=class_idx)
            return {"x": x[0, 0].cpu().numpy(), "cam": cam, "title": title}

        # Grid 1: one correctly-classified sample per class, CAM w.r.t. that class.
        class_panels = []
        for name, ci in label2idx.items():
            idx = _first_index((labels == ci) & (preds == ci))
            if idx is None:
                print(f"WARNING: no correctly-classified {name} sample in {args.split}")
                continue
            class_panels.append(make_panel(idx, name, class_idx=ci))
        _plot_grid(
            class_panels, ncols=4, out_path=out_path,
            suptitle="Grad-CAM — one correct prediction per class",
        )

        # Grid 2: Loc<->Edge-Loc and Loc<->Scratch confusions,
        # CAM w.r.t. the PREDICTED class (what the model saw when it was wrong).
        pairs = [("Loc", "Edge-Loc"), ("Edge-Loc", "Loc"), ("Loc", "Scratch"), ("Scratch", "Loc")]
        confusion_panels = []
        for true_name, pred_name in pairs:
            ti, pi = label2idx[true_name], label2idx[pred_name]
            mask = (labels == ti) & (preds == pi)
            for k, idx in enumerate(np.flatnonzero(mask)[:2]):  # up to 2 per direction
                confusion_panels.append(
                    make_panel(int(idx), f"true {true_name} → pred {pred_name}", class_idx=pi)
                )
        _plot_grid(
            confusion_panels, ncols=4, out_path=confusions_path,
            suptitle="Grad-CAM on confusions (CAM w.r.t. predicted class)",
        )


if __name__ == "__main__":
    main()
