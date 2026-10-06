#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_full_pipeline.sh <activitynet_root> <run_id> <moco_seed> [gpu_id] [moco_mode]

Arguments:
  activitynet_root  ActivityNet v1.3 dataset root
  run_id            MoCo run ID (example: run1)
  moco_seed         MoCo initialization seed in [0, 2**32)
  gpu_id            CUDA_VISIBLE_DEVICES value for the single GPU (default: 0)
  moco_mode         fresh | resume | skip (default: fresh)

Modes:
  fresh   Start a new full-dataset MoCo run.
  resume  Resume from log/moco/<run_id>/resume/latest.pt.
  skip    Skip MoCo and use an already completed final snapshot.

Environment variables:
  PYTHON            Python executable (default: .venv/bin/python)
  FEATURE_DEVICE    Feature extraction device: cuda | cpu (default: cuda)
  PROBE_DEVICE      Linear Probe device: cpu | cuda (default: cpu)
  DISABLE_COMET     true | false (default: false)

Example:
  bash run_full_pipeline.sh \
    /mnt/NAS-TVS872XT/dataset/ActivityNet \
    run1 \
    7 \
    0 \
    fresh

Resume example:
  bash run_full_pipeline.sh \
    /mnt/NAS-TVS872XT/dataset/ActivityNet \
    run1 \
    7 \
    0 \
    resume

Continue downstream evaluation after MoCo already completed:
  bash run_full_pipeline.sh \
    /mnt/NAS-TVS872XT/dataset/ActivityNet \
    run1 \
    7 \
    0 \
    skip
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ $# -ge 3 && $# -le 5 ]] || {
  usage
  exit 2
}

DATASET_ROOT="$1"
RUN_ID="$2"
MOCO_SEED="$3"
GPU_ID="${4:-0}"
MOCO_MODE="${5:-fresh}"

PYTHON="${PYTHON:-.venv/bin/python}"
FEATURE_DEVICE="${FEATURE_DEVICE:-cuda}"
PROBE_DEVICE="${PROBE_DEVICE:-cpu}"
DISABLE_COMET="${DISABLE_COMET:-false}"

MANIFEST_ID="lp-v1"
EXPECTED_BRANCH="dev"

[[ -d "$DATASET_ROOT" ]] || die "ActivityNet root does not exist: $DATASET_ROOT"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid run_id: $RUN_ID"
[[ "$MOCO_SEED" =~ ^[0-9]+$ ]] || die "moco_seed must be a non-negative integer"
(( MOCO_SEED < 4294967296 )) || die "moco_seed must be smaller than 2**32"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "gpu_id must identify exactly one GPU, e.g. 0"
[[ "$MOCO_MODE" == "fresh" || "$MOCO_MODE" == "resume" || "$MOCO_MODE" == "skip" ]] \
  || die "moco_mode must be fresh, resume, or skip"
[[ "$FEATURE_DEVICE" == "cuda" || "$FEATURE_DEVICE" == "cpu" ]] \
  || die "FEATURE_DEVICE must be cuda or cpu"
[[ "$PROBE_DEVICE" == "cuda" || "$PROBE_DEVICE" == "cpu" ]] \
  || die "PROBE_DEVICE must be cuda or cpu"
[[ "$DISABLE_COMET" == "true" || "$DISABLE_COMET" == "false" ]] \
  || die "DISABLE_COMET must be true or false"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die "Run this script from inside the research implementation repository"
cd "$REPO_ROOT"

[[ -x "$PYTHON" ]] || die "Python executable is not executable: $PYTHON"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$EXPECTED_BRANCH" ]] \
  || die "Full run expects branch '$EXPECTED_BRANCH', current branch is '$CURRENT_BRANCH'"

# Production provenance requires a completely clean checkout, including
# untracked files. Check before creating any local log output.
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  git status --short --untracked-files=all >&2
  die "Canonical production requires a clean checkout. Commit/stash/remove the changes above first."
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

PIPELINE_LOG_DIR="log/full_pipeline/${RUN_ID}"
MOCO_RUN_DIR="log/moco/${RUN_ID}"
MANIFEST_DIR="log/linear_probe/manifest/${MANIFEST_ID}"
mkdir -p "$PIPELINE_LOG_DIR"

echo "============================================================"
echo "Full pipeline"
echo "repository : $REPO_ROOT"
echo "branch     : $CURRENT_BRANCH"
echo "commit     : $(git rev-parse HEAD)"
echo "dataset    : $DATASET_ROOT"
echo "run_id     : $RUN_ID"
echo "moco_seed  : $MOCO_SEED"
echo "gpu_id     : $GPU_ID"
echo "moco_mode  : $MOCO_MODE"
echo "Comet      : $([[ "$DISABLE_COMET" == "false" ]] && echo enabled || echo disabled)"
echo "============================================================"

run_logged() {
  local log_file="$1"
  shift
  "$@" 2>&1 | tee "$log_file"
}

# ---------------------------------------------------------------------------
# 1. Full-dataset Stage 6B-v2 Streaming MoCo
# ---------------------------------------------------------------------------
case "$MOCO_MODE" in
  fresh)
    [[ ! -e "$MOCO_RUN_DIR" ]] \
      || die "MoCo run directory already exists. Use resume/skip or choose a new run_id: $MOCO_RUN_DIR"

    run_logged "$PIPELINE_LOG_DIR/01_moco.log" \
      "$PYTHON" -m scripts.moco.train_full_streaming_moco \
      "runtime.dataset_root=$DATASET_ROOT" \
      "runtime.run_id=$RUN_ID" \
      "runtime.seed=$MOCO_SEED" \
      runtime.device=cuda \
      runtime.resume=false \
      "logging.disable_comet=$DISABLE_COMET"
    ;;

  resume)
    [[ -f "$MOCO_RUN_DIR/resume/latest.pt" ]] \
      || die "Resume checkpoint not found: $MOCO_RUN_DIR/resume/latest.pt"

    run_logged "$PIPELINE_LOG_DIR/01_moco_resume.log" \
      "$PYTHON" -m scripts.moco.train_full_streaming_moco \
      "runtime.dataset_root=$DATASET_ROOT" \
      "runtime.run_id=$RUN_ID" \
      "runtime.seed=$MOCO_SEED" \
      runtime.device=cuda \
      runtime.resume=true \
      "logging.disable_comet=$DISABLE_COMET"
    ;;

  skip)
    echo "[1/5] Skipping MoCo; an existing final snapshot will be validated downstream."
    ;;
esac

FINAL_SNAPSHOT="$(
  "$PYTHON" - "$MOCO_RUN_DIR/evaluation_snapshots" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
matches = sorted(
    path for path in root.glob("videos-*_step-*_final")
    if path.is_dir()
)
if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one final Query LoRA snapshot under {root}, found {len(matches)}: "
        + ", ".join(str(path) for path in matches)
    )
print(matches[0])
PY
)"

echo "[1/5] Final Query LoRA snapshot: $FINAL_SNAPSHOT"

# ---------------------------------------------------------------------------
# 2. ActivityNet segment manifest + reproducibility audit
# ---------------------------------------------------------------------------
if [[ ! -d "$MANIFEST_DIR" ]]; then
  run_logged "$PIPELINE_LOG_DIR/02_manifest_build.log" \
    "$PYTHON" -m scripts.linear_probe.build_manifest \
    runtime.command=build \
    "runtime.dataset_root=$DATASET_ROOT" \
    "runtime.manifest_id=$MANIFEST_ID" \
    "logging.disable_comet=$DISABLE_COMET"
else
  echo "[2/5] Existing manifest found; build is skipped and audit will verify it: $MANIFEST_DIR"
fi

run_logged "$PIPELINE_LOG_DIR/02_manifest_audit.log" \
  "$PYTHON" -m scripts.linear_probe.build_manifest \
  runtime.command=audit \
  "runtime.dataset_root=$DATASET_ROOT" \
  "runtime.manifest_id=$MANIFEST_ID" \
  "logging.disable_comet=$DISABLE_COMET"

"$PYTHON" - "$MANIFEST_DIR/metadata.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
metadata = json.loads(path.read_text())
if metadata.get("production") is not True:
    raise SystemExit(f"Manifest is not production: {path}")
status = (metadata.get("gate") or {}).get("status")
if status != "PASS":
    raise SystemExit(f"Manifest Gate is not PASS ({status}): {path}")
print(f"[2/5] Manifest Gate PASS: {path.parent}")
PY

# ---------------------------------------------------------------------------
# 3. Base ViT / final Query LoRA segment features
# ---------------------------------------------------------------------------
extract_feature_dir() {
  local condition="$1"
  local snapshot="${2:-}"
  local log_file="$PIPELINE_LOG_DIR/03_features_${condition}.log"
  local -a cmd=(
    "$PYTHON" -m scripts.linear_probe.extract_features
    "runtime.dataset_root=$DATASET_ROOT"
    "runtime.manifest_id=$MANIFEST_ID"
    "runtime.condition=$condition"
    "runtime.device=$FEATURE_DEVICE"
    "logging.disable_comet=$DISABLE_COMET"
  )

  if [[ -n "$snapshot" ]]; then
    cmd+=("runtime.snapshot=$snapshot")
  fi

  # Keep the full extractor output visible while reserving stdout below for
  # the single resolved artifact path returned to the caller.
  "${cmd[@]}" 2>&1 | tee "$log_file" >&2

  "$PYTHON" - "$log_file" <<'PY'
import json
from pathlib import Path
import sys

path = None
for raw in Path(sys.argv[1]).read_text().splitlines():
    prefix = "Reusing verified feature artifact: "
    if raw.startswith(prefix):
        path = raw[len(prefix):].strip()
        continue
    try:
        row = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if row.get("event") == "features_written":
        path = row["path"]

if path is None:
    raise SystemExit(f"Could not resolve feature artifact path from {sys.argv[1]}")
print(path)
PY
}

BASE_FEATURE_DIR="$(extract_feature_dir base_vit)"
LORA_FEATURE_DIR="$(extract_feature_dir moco_query_lora_final "$FINAL_SNAPSHOT")"

echo "[3/5] Base ViT features : $BASE_FEATURE_DIR"
echo "[3/5] MoCo LoRA features: $LORA_FEATURE_DIR"

# ---------------------------------------------------------------------------
# 4. Linear Probe: 2 conditions x seeds [0, 1, 2]
# ---------------------------------------------------------------------------
PROBE_LOG="$PIPELINE_LOG_DIR/04_linear_probe.log"

run_logged "$PROBE_LOG" \
  "$PYTHON" -m scripts.linear_probe.run_probe \
  "runtime.manifest_id=$MANIFEST_ID" \
  "runtime.base_features=$BASE_FEATURE_DIR" \
  "runtime.lora_features=$LORA_FEATURE_DIR" \
  "runtime.device=$PROBE_DEVICE" \
  "logging.disable_comet=$DISABLE_COMET"

# ---------------------------------------------------------------------------
# 5. Resolve aggregate result path
# ---------------------------------------------------------------------------
AGGREGATE_DIR="$(
  "$PYTHON" - "$PROBE_LOG" <<'PY'
import json
from pathlib import Path
import sys

path = None
for raw in Path(sys.argv[1]).read_text().splitlines():
    prefix = "Reusing identical aggregate result: "
    if raw.startswith(prefix):
        path = raw[len(prefix):].strip()
        continue
    try:
        row = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if row.get("event") == "aggregate":
        path = row["path"]

if path is None:
    raise SystemExit(f"Could not resolve aggregate result path from {sys.argv[1]}")
print(path)
PY
)"

echo "============================================================"
echo "FULL PIPELINE COMPLETED"
echo "MoCo run       : $MOCO_RUN_DIR"
echo "Final snapshot : $FINAL_SNAPSHOT"
echo "Manifest       : $MANIFEST_DIR"
echo "Base features  : $BASE_FEATURE_DIR"
echo "LoRA features  : $LORA_FEATURE_DIR"
echo "Probe aggregate: $AGGREGATE_DIR"
echo "Logs           : $PIPELINE_LOG_DIR"
echo "============================================================"
