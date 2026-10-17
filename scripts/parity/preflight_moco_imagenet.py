"""Gate A0 preflight for Spec A: reference clone, licence texts, imports, ImageNet root, GPUs/NCCL, environment.

    python -m scripts.parity.preflight_moco_imagenet                  # read-only CPU / I/O checks
    python -m scripts.parity.preflight_moco_imagenet --pytest         # + CIFAR/LoRA/MoCo non-regression pytest (CPU)
    CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.preflight_moco_imagenet --nccl-check   # + 2-GPU NCCL test

Read-only: the dataset root and the clone are only listed, hashed and decoded; nothing is downloaded, moved or
reorganized. Writes preflight.json, dataset_inventory.json and verification_summary.md (plus reference_clone.diff
when the clone differs from the pinned commit) under log/moco_imagenet_parity/preflight/<date>/<time>/.
A0 is PASS only when every required section is PASS; a section that was not run makes A0 BLOCKED.
"""

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS

from scripts.parity.moco_parity_state import (
    REFERENCE, REPOSITORY_ROOT, environment, file_sha256, git_info, load_config, reference_argv,
)

SCHEMA_VERSION = "moco_imagenet_preflight/v1"
CONFIG_PATH = REPOSITORY_ROOT / "conf" / "moco_imagenet_parity.yaml"
PASS, FAIL, BLOCKED, NOT_RUN = "PASS", "FAIL", "BLOCKED", "NOT_RUN"
REFERENCE_FILES = ("moco/builder.py", "moco/loader.py", "main_moco.py", "LICENSE", "README.md")
# Standard ILSVRC2012 (ImageNet-1K) counts, used only to describe the inventory.
EXPECTED = {"classes": 1000, "train": 1_281_167, "val": 50_000}
NON_REGRESSION_TESTS = (
    "test/model/test_moco_standalone.py", "test/model/test_moco_cifar10_reporting.py",
    "test/model/test_lora_linear.py", "test/model/test_vit_lora_checkpoint.py",
    "test/model/test_vit_lora_frame_encoder.py", "test/model/test_vit_lora_reporting.py",
    "test/model/test_moco_ddp.py", "test/model/test_moco_parity_harness.py",
)


def reference_section(cfg, output_dir: Path) -> dict:
    root = Path(str(cfg.reference.root))
    info = git_info(root)
    result = {"root": str(root), "pinned_commit": REFERENCE["commit"], "git": info, "status": FAIL}
    if not info.get("commit"):
        result["reason"] = f"no git clone at {root}"
        return result
    diff = subprocess.run(["git", "-C", str(root), "diff", REFERENCE["commit"]], capture_output=True, text=True).stdout
    if diff:
        (output_dir / "reference_clone.diff").write_text(diff, encoding="utf-8")
    files = {name: file_sha256(root / name) for name in REFERENCE_FILES if (root / name).exists()}
    licence = {
        "LICENSE_first_line": next(line for line in (root / "LICENSE").read_text(encoding="utf-8").splitlines()
                                   if line.strip()),
        "builder_header": [line.strip("# ").strip() for line in (root / "moco/builder.py").read_text(
            encoding="utf-8").splitlines()[:8] if "licen" in line.lower()],
        "README_license_section": [line for line in (root / "README.md").read_text(encoding="utf-8").splitlines()
                                   if "license" in line.lower() and not line.startswith("#")],
    }
    licence["note"] = ("LICENSE/ヘッダとREADMEの表記が一致しない（MIT vs CC-BY-NC 4.0）。公開・コピー方針は原本で判断し、"
                       "公式コードを研究repositoryへ転載しない（spec §2.2）")
    ok = info["commit"] == REFERENCE["commit"] and not info["dirty"] and not diff and len(files) == len(REFERENCE_FILES)
    result.update(status=PASS if ok else FAIL, diff_empty=not diff, files_sha256=files, licence=licence,
                  reason=None if ok else "clone is not the clean pinned commit")
    return result


def import_section(cfg) -> dict:
    """Import through the shim in a fresh interpreter and parse the reference MoCo v2 command line."""
    code = (
        "import json, sys\n"
        "from omegaconf import OmegaConf\n"
        "from scripts.parity.moco_parity_state import import_reference, reference_args\n"
        "cfg = OmegaConf.create(json.loads(sys.argv[1]))\n"
        "ref = import_reference(cfg.reference.root)\n"
        "args = reference_args(ref['main_moco'], cfg)\n"
        "print(json.dumps({'provenance': ref['provenance'], 'args': {k: v for k, v in vars(args).items() "
        "if isinstance(v, (str, int, float, bool, list, type(None)))}}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, json.dumps(OmegaConf.to_container(cfg, resolve=True))],
                          cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    if proc.returncode:
        return {"status": FAIL, "stderr": proc.stderr[-4000:]}
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    args = loaded["args"]
    flags_ok = args["mlp"] and args["aug_plus"] and args["cos"] and args["moco_t"] == float(cfg.moco.temperature)
    return {"status": PASS if flags_ok else FAIL, "argv": reference_argv(cfg), **loaded,
            "note": "main_moco.py imports deeplearning.cross_image_ssl.moco.*; resolved by scripts/parity/"
                    "meta_moco_shim to the clone's unmodified moco/builder.py and moco/loader.py (B1)"}


def _scan_class(path: Path) -> dict:
    try:
        names = [entry.name for entry in os.scandir(path) if entry.is_file()]
    except OSError as error:
        return {"error": repr(error), "images": [], "other": 0}
    images = sorted(n for n in names if n.lower().endswith(IMG_EXTENSIONS))
    return {"images": images, "other": len(names) - len(images)}


def dataset_section(cfg, samples_per_split: dict, workers: int) -> tuple[dict, dict]:
    root = Path(str(cfg.data.root))
    result = {"root": str(root), "status": FAIL, "read_only": True}
    if not root.is_dir():
        result["reason"] = f"{root} is not a readable directory"
        return result, {}
    result["top_level"] = sorted(({"name": entry.name, "type": "dir" if entry.is_dir() else "file",
                                   "bytes": entry.stat().st_size if entry.is_file() else None}
                                  for entry in os.scandir(root)), key=lambda entry: entry["name"])
    inventory, splits = {}, {}
    rng = random.Random(0)
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            splits[split] = {"exists": False}
            continue
        started = time.perf_counter()
        classes = sorted(entry.name for entry in os.scandir(split_root) if entry.is_dir())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scanned = dict(zip(classes, pool.map(_scan_class, [split_root / c for c in classes])))
        listing_seconds = time.perf_counter() - started
        counts = {c: len(s["images"]) for c, s in scanned.items()}
        files = [(c, name) for c, s in scanned.items() for name in s["images"]]
        chosen = rng.sample(files, min(samples_per_split[split], len(files)))
        decode = _decode([split_root / c / name for c, name in chosen], workers)
        loose = [entry.name for entry in os.scandir(split_root) if entry.is_file()]
        splits[split] = {
            "exists": True, "classes": len(classes), "images": len(files), "listing_seconds": listing_seconds,
            "empty_classes": [c for c, n in counts.items() if n == 0],
            "unreadable_classes": {c: s["error"] for c, s in scanned.items() if "error" in s},
            "non_image_files": sum(s["other"] for s in scanned.values()), "loose_files_in_split_root": len(loose),
            "min_per_class": min(counts.values(), default=0), "max_per_class": max(counts.values(), default=0),
            "decode": decode,
        }
        inventory[split] = counts
    result["splits"] = splits
    present = [s for s in ("train", "val") if splits.get(s, {}).get("exists")]
    same_classes = len(present) == 2 and inventory["train"].keys() == inventory["val"].keys()
    result["train_val_same_classes"] = same_classes
    result["matches_standard_ilsvrc2012_counts"] = {
        "classes": all(splits[s]["classes"] == EXPECTED["classes"] for s in present),
        **{s: splits[s]["images"] == EXPECTED[s] for s in present}}
    structural = len(present) == 2 and same_classes and all(
        not splits[s]["empty_classes"] and not splits[s]["unreadable_classes"] and splits[s]["decode"]["failures"] == []
        for s in present)
    result["status"] = PASS if structural else FAIL
    if not structural:
        result["reason"] = "train/ and val/ must both be ImageFolder class directories with readable, decodable images"
    return result, inventory


def _decode_one(path: Path) -> dict:
    started = time.perf_counter()
    try:
        with Image.open(path) as image:
            size = image.convert("RGB").size
        return {"ok": True, "bytes": path.stat().st_size, "seconds": time.perf_counter() - started, "size": size}
    except Exception as error:  # record every decode failure, keep scanning
        return {"ok": False, "path": str(path), "error": repr(error)}


def _decode(paths: list, workers: int) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_decode_one, paths))
    wall = time.perf_counter() - started
    ok = [r for r in results if r["ok"]]
    total = sum(r["bytes"] for r in ok)
    return {"sampled": len(paths), "decoded": len(ok), "failures": [r for r in results if not r["ok"]],
            "wall_seconds": wall, "images_per_sec": len(ok) / wall if wall else None,
            "mib_per_sec": total / 2**20 / wall if wall else None, "threads": workers}


def _nccl_worker(rank: int, port: int, out: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank,
                            timeout=timedelta(minutes=2))
    try:
        x = torch.full((1 << 20,), float(rank + 1), device="cuda")
        started = time.perf_counter()
        dist.all_reduce(x)
        gathered = [torch.empty(4, device="cuda") for _ in range(2)]
        dist.all_gather(gathered, torch.full((4,), float(rank), device="cuda"))
        idx = torch.randperm(8, device="cuda") if rank == 0 else torch.empty(8, dtype=torch.long, device="cuda")
        dist.broadcast(idx, src=0)
        torch.cuda.synchronize()
        result = {"rank": rank, "gpu": torch.cuda.get_device_name(rank), "all_reduce_ok": bool((x == 3).all()),
                  "all_gather_ok": [float(g[0]) for g in gathered] == [0.0, 1.0],
                  "broadcast": idx.tolist(), "seconds": time.perf_counter() - started}
        Path(out, f"nccl_rank{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def gpu_section(run_nccl: bool, output_dir: Path) -> dict:
    result = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "nccl_available": dist.is_nccl_available()}
    result["devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                          "total_memory_mib": torch.cuda.get_device_properties(i).total_memory / 2**20}
                         for i in range(torch.cuda.device_count())]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True)
    result["nvidia_smi"] = smi.stdout.strip().splitlines() if smi.returncode == 0 else smi.stderr
    if not run_nccl:
        result.update(status=NOT_RUN, reason="2-GPU NCCL check needs separate approval (--nccl-check)")
        return result
    if torch.cuda.device_count() < 2:
        result.update(status=FAIL, reason="fewer than 2 visible GPUs")
        return result
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_nccl_worker, args=(port, str(output_dir)), nprocs=2, join=True)
    ranks = [json.loads((output_dir / f"nccl_rank{r}.json").read_text(encoding="utf-8")) for r in range(2)]
    same_broadcast = ranks[0]["broadcast"] == ranks[1]["broadcast"]
    ok = all(r["all_reduce_ok"] and r["all_gather_ok"] for r in ranks) and same_broadcast
    result.update(status=PASS if ok else FAIL, nccl_ranks=ranks)
    return result


def pytest_section(run: bool) -> dict:
    if not run:
        return {"status": NOT_RUN, "reason": "run with --pytest"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *NON_REGRESSION_TESTS],
                          cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-2000:]
    return {"status": PASS if proc.returncode == 0 else FAIL, "files": list(NON_REGRESSION_TESTS), "summary": tail,
            "returncode": proc.returncode}


def render(report: dict) -> str:
    lines = [f"# Spec A Gate A0 preflight ({report['created']})", "", f"- A0 status: **{report['status']}**", ""]
    for name, section in report["sections"].items():
        lines.append(f"- {name}: **{section['status']}** {section.get('reason') or ''}")
    data = report["sections"]["dataset"]
    for split, info in data.get("splits", {}).items():
        if info.get("exists"):
            lines.append(f"  - {split}: {info['classes']} classes, {info['images']} images, decode "
                         f"{info['decode']['decoded']}/{info['decode']['sampled']} "
                         f"({info['decode']['images_per_sec'] or 0:.1f} img/s, "
                         f"{info['decode']['mib_per_sec'] or 0:.1f} MiB/s)")
    licence = report["sections"]["reference_clone"].get("licence")
    if licence:
        lines += ["", f"- licence: LICENSE `{licence['LICENSE_first_line']}`, builder header "
                      f"`{licence['builder_header']}`, README `{licence['README_license_section']}`",
                  f"  - {licence['note']}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--nccl-check", action="store_true", help="2-GPU NCCL collectives (separate approval)")
    parser.add_argument("--pytest", action="store_true", help="run the CPU non-regression tests")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(CONFIG_PATH, args.overrides)
    stamp = f"{datetime.now():%Y-%m-%d/%H-%M-%S-%f}"
    output_dir = args.output_dir or Path("log/moco_imagenet_parity/preflight") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "preflight.json").exists():
        raise FileExistsError(f"{output_dir / 'preflight.json'} exists")
    dataset, inventory = dataset_section(cfg, {"train": args.train_samples, "val": args.val_samples}, args.workers)
    sections = {
        "reference_clone": reference_section(cfg, output_dir), "reference_import": import_section(cfg),
        "self_repository": {"status": PASS, **git_info(REPOSITORY_ROOT)},
        "dataset": dataset, "gpu_nccl": gpu_section(args.nccl_check, output_dir),
        "environment": {"status": PASS, **environment()}, "non_regression_pytest": pytest_section(args.pytest),
    }
    statuses = [s["status"] for s in sections.values()]
    status = FAIL if FAIL in statuses else (BLOCKED if NOT_RUN in statuses else PASS)
    report = {"schema_version": SCHEMA_VERSION, "created": datetime.now().isoformat(timespec="seconds"),
              "status": status, "sections": sections}
    (output_dir / "dataset_inventory.json").write_text(json.dumps(inventory, indent=1), encoding="utf-8")
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    (output_dir / "verification_summary.md").write_text(render(report), encoding="utf-8")
    print(f"A0 {status}; saved {output_dir}")
    return report


if __name__ == "__main__":
    main()
