# WM-811K defect classifier -- serving image (T10).
# Single-node FastAPI. CPU-only (no GPU in the container): torch via --extra cpu.
# The canonical model is pulled from a GitHub Release at build time and its
# SHA-256 verified, so the image is self-contained and reproducible on any host
# with network access -- no local checkpoint needed in the build context.

FROM python:3.12-slim

# uv: copy the static binary from the official image (faster than pip-installing).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# --- Dependency layer (cached until pyproject/lock change) -------------------
# Copy ONLY the dependency manifests first so `uv sync` is cached across code
# edits. torch CPU is ~200MB; we don't want to reinstall it on every src change.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra cpu --no-dev --frozen --no-install-project

# --- Model layer (cached until the URL/SHA change) ---------------------------
# Pull the canonical checkpoint from the GitHub Release and verify its digest.
# A mismatch fails the build: we never serve a checkpoint we can't verify.
ARG MODEL_URL=https://github.com/pqtrng/wm811k-defect-detection/releases/download/model-v1/resnet18-aug_best.pt
ARG MODEL_SHA256=f91f18970ac7c38f14f9ae95eb47ecec688c99c850a344dfe52b5da6e5460cef
RUN mkdir -p models \
    && (command -v wget >/dev/null || (apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*)) \
    && wget -q -O models/resnet18-aug_best.pt "${MODEL_URL}" \
    && echo "${MODEL_SHA256}  models/resnet18-aug_best.pt" | sha256sum -c -

# --- Source layer (changes most often -> last) -------------------------------
COPY src ./src
COPY configs ./configs
RUN uv sync --extra cpu --no-dev --frozen

EXPOSE 8000

# Serve. app is importable at module scope (wm811k.serve:app); serve.py's
# default checkpoint path is models/resnet18-aug_best.pt, matching the pull above.
CMD ["uv", "run", "--extra", "cpu", "--no-dev", "uvicorn", "wm811k.serve:app", "--host", "0.0.0.0", "--port", "8000"]
