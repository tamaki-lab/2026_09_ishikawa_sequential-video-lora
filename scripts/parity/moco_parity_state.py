"""Shared pieces of the Spec A parity harness: reference import, state-name mapping, seeds, hashes, comparisons.

Canonical names are those of the reference ``moco.builder.MoCo`` (``encoder_q.*``, ``encoder_k.*``, ``queue``,
``queue_ptr``). The self ``DistributedMoCo`` keeps its own module names; the explicit mapping is
``backbone_{q,k}.net.X <-> encoder_{q,k}.X`` and ``projector_{q,k}.N.* <-> encoder_{q,k}.fc.N.*``.
``real_key_count`` is a self-only diagnostic buffer and never enters a comparison.
"""

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import torch

REFERENCE = {
    "repository": "facebookresearch/moco",
    "commit": "a419ef282fc9885015674e77f58c59447206a27f",
}
SHIM_DIR = Path(__file__).resolve().parent / "meta_moco_shim"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SELF_ONLY_BUFFERS = ("real_key_count",)
SHARED_BUFFERS = ("queue", "queue_ptr")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}{tuple(tensor.shape)}".encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(tensor_sha256(state[name]).encode())
    return digest.hexdigest()


def derive_seed(*parts) -> int:
    """Deterministic 63-bit seed from labelled parts, e.g. ``derive_seed(0, "shuffle", step, rank)``."""
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "big") >> 1


def shuffle_seed(fixture_seed: int, step: int, rank: int) -> int:
    """CPU global-RNG seed set on ``rank`` right before the forward of ``step`` (1-based)."""
    return derive_seed(fixture_seed, "shuffle", step, rank)


def expected_shuffle(fixture_seed: int, step: int, global_batch: int) -> torch.Tensor:
    """The permutation rank 0 draws with ``torch.randperm`` once the CPU global RNG is seeded with rank 0's seed."""
    generator = torch.Generator().manual_seed(shuffle_seed(fixture_seed, step, 0))
    return torch.randperm(global_batch, generator=generator)


def synthetic_views(fixture_seed: int, step: int, rank: int, batch_size: int, image_size: int):
    """Fixed (im_q, im_k) of one rank and step: standard normal float32 [B, 3, S, S] from derived seeds."""
    return tuple(
        torch.randn(batch_size, 3, image_size, image_size,
                    generator=torch.Generator().manual_seed(derive_seed(fixture_seed, "input", step, rank, view)))
        for view in ("q", "k")
    )


def self_to_canonical(name: str) -> str | None:
    """Canonical (reference) name of a self ``DistributedMoCo`` state entry; None for self-only buffers."""
    if name in SELF_ONLY_BUFFERS:
        return None
    if name in SHARED_BUFFERS:
        return name
    for branch in ("q", "k"):
        for prefix, target in ((f"backbone_{branch}.net.", f"encoder_{branch}."),
                               (f"projector_{branch}.", f"encoder_{branch}.fc.")):
            if name.startswith(prefix):
                return target + name[len(prefix):]
    raise KeyError(f"no canonical name for self state entry {name!r}")


def canonical_to_self(name: str) -> str:
    if name in SHARED_BUFFERS:
        return name
    for branch in ("q", "k"):
        head, encoder = f"encoder_{branch}.fc.", f"encoder_{branch}."
        if name.startswith(head):
            return f"projector_{branch}." + name[len(head):]
        if name.startswith(encoder):
            return f"backbone_{branch}.net." + name[len(encoder):]
    raise KeyError(f"no self name for canonical state entry {name!r}")


def mapping_table(canonical_state: dict, self_state: dict) -> list[dict]:
    """Explicit canonical <-> self table; raises unless it is a bijection with equal shapes and dtypes."""
    mapped = {}
    for self_name in self_state:
        canonical = self_to_canonical(self_name)
        if canonical is None:
            continue
        if canonical in mapped or canonical_to_self(canonical) != self_name:
            raise KeyError(f"mapping is not one-to-one at {self_name!r} -> {canonical!r}")
        mapped[canonical] = self_name
    missing, extra = sorted(set(canonical_state) - set(mapped)), sorted(set(mapped) - set(canonical_state))
    if missing or extra:
        raise KeyError(f"state mapping mismatch: reference-only {missing[:5]}, self-only {extra[:5]}")
    rows = []
    for canonical, tensor in canonical_state.items():
        other = self_state[mapped[canonical]]
        if tuple(tensor.shape) != tuple(other.shape) or tensor.dtype != other.dtype:
            raise ValueError(f"{canonical}: reference {tuple(tensor.shape)}/{tensor.dtype} "
                             f"!= self {mapped[canonical]} {tuple(other.shape)}/{other.dtype}")
        rows.append({"canonical": canonical, "self": mapped[canonical], "shape": list(tensor.shape),
                     "dtype": str(tensor.dtype)})
    return rows


def canonical_to_self_state(canonical_state: dict, self_state: dict) -> dict:
    """A self state_dict holding the canonical values (copies); self-only buffers are reset to zero."""
    mapping_table(canonical_state, self_state)
    state = {}
    for name, template in self_state.items():
        canonical = self_to_canonical(name)
        state[name] = torch.zeros_like(template) if canonical is None else canonical_state[canonical].clone()
    return state


def import_reference(root) -> dict:
    """Import the unmodified clone: ``main_moco`` through the shim, ``moco.builder`` and ``moco.loader``.

    Returns the modules and the loaded file paths with their sha256; raises if anything resolves outside the clone.
    Bytecode writing is turned off for the import so that the clone stays byte-identical (no __pycache__).
    """
    root = Path(root).resolve()
    for path in (str(SHIM_DIR), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        main_moco = importlib.import_module("main_moco")
        builder = importlib.import_module("moco.builder")
        loader = importlib.import_module("moco.loader")
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    if (sys.modules["deeplearning.cross_image_ssl.moco.builder"] is not builder
            or sys.modules["deeplearning.cross_image_ssl.moco.loader"] is not loader):
        raise ImportError("main_moco.py does not use the clone's moco.builder / moco.loader")
    loaded = {}
    for module, directory in ((main_moco, root), (builder, root / "moco"), (loader, root / "moco")):
        path = Path(module.__file__).resolve()
        if path.parent != directory:
            raise ImportError(f"{module.__name__} was loaded from {path}, not from the clone {root}")
        loaded[module.__name__] = {"path": str(path), "sha256": file_sha256(path)}
    shim = {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in sorted(SHIM_DIR.rglob("*.py"))}
    return {"main_moco": main_moco, "builder": builder, "loader": loader,
            "provenance": {"root": str(root), "modules": loaded, "shim": shim}}


def reference_argv(cfg) -> list[str]:
    """``main_moco.py`` command line for this config: MoCo v2 flags --mlp --moco-t --aug-plus --cos."""
    world_size, local_batch = int(cfg.distributed.world_size), int(cfg.training.local_batch_size)
    argv = [
        str(cfg.data.root), "--arch", str(cfg.model.arch), "--lr", repr(float(cfg.training.learning_rate)),
        "--batch-size", str(world_size * local_batch), "--momentum", repr(float(cfg.training.momentum)),
        "--weight-decay", repr(float(cfg.training.weight_decay)), "--epochs", str(int(cfg.training.epochs)),
        "--seed", str(int(cfg.training.seed)), "--workers", str(world_size * int(cfg.data.num_workers)),
        "--moco-dim", str(int(cfg.model.projection_dim)), "--moco-k", str(int(cfg.moco.queue_size)),
        "--moco-m", repr(float(cfg.moco.momentum)), "--moco-t", repr(float(cfg.moco.temperature)),
        "--aug-plus", "--world-size", "1", "--rank", "0", "--multiprocessing-distributed",
        "--dist-backend", str(cfg.distributed.backend),
    ]
    if cfg.model.mlp:
        argv.append("--mlp")
    if cfg.training.cos:
        argv.append("--cos")
    return argv


def reference_args(main_moco, cfg):
    """main_moco's own parser on ``reference_argv`` with main_worker's per-GPU batch split applied."""
    args = main_moco.parser.parse_args(reference_argv(cfg))
    world_size = int(cfg.distributed.world_size)
    args.world_size = world_size  # ngpus_per_node * nodes, as main() sets it
    args.batch_size = int(args.batch_size / world_size)  # as main_worker does for one GPU per process
    args.workers = int((args.workers + world_size - 1) / world_size)
    if args.batch_size != int(cfg.training.local_batch_size):
        raise ValueError(f"main_moco per-GPU batch {args.batch_size} != training.local_batch_size")
    return args


def reference_two_crops(loader, image_size: int = 224):
    """main_moco.py's ``--aug-plus`` pipeline built from the clone's ``moco.loader`` classes.

    main_worker builds this list inline, so it is restated here with the reference GaussianBlur/TwoCropsTransform;
    with image_size=224 it is the reference pipeline.
    """
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return loader.TwoCropsTransform(transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))


def load_config(config_path, overrides):
    """The Hydra yaml without its ``hydra`` block, with ``key=value`` overrides (for the non-Hydra entry points)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    cfg.pop("hydra", None)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def environment() -> dict:
    import platform

    import numpy
    import torchvision

    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "numpy": numpy.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
            "platform": platform.platform(), "host": platform.node()}


def git_info(path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout

    try:
        status = git("status", "--porcelain", "--untracked-files=all").splitlines()
        return {"path": str(Path(path).resolve()), "commit": git("rev-parse", "HEAD").strip(),
                "branch": git("branch", "--show-current").strip() or None, "dirty": bool(status),
                "dirty_paths": status[:50]}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"path": str(path), "commit": None, "error": repr(error)}


@torch.no_grad()
def group_stats(tensors) -> dict:
    """float64 sum / abs-sum / square-sum / max-abs / non-finite count over a list of tensors (diagnostics)."""
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return {"numel": 0}
    flat = torch.cat([t.detach().reshape(-1).double() for t in tensors])
    finite = torch.isfinite(flat)
    values = flat[finite]
    return {
        "numel": flat.numel(), "nonfinite": int((~finite).sum()),
        "sum": float(values.sum()), "abs_sum": float(values.abs().sum()), "sq_sum": float((values * values).sum()),
        "max_abs": float(values.abs().max()) if values.numel() else None,
    }


@torch.no_grad()
def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor, rtol: float, atol: float) -> dict:
    """Element-wise comparison with torch.testing.assert_close semantics (|c - r| <= atol + rtol * |r|).

    Non-float tensors must match exactly. NaN never matches. ``max_rel_error`` only uses elements with
    |r| > atol so that tiny denominators do not dominate; ``small_denominator_count`` reports how many were skipped.
    """
    result = {"shape": list(reference.shape), "dtype": str(reference.dtype), "numel": reference.numel()}
    if tuple(reference.shape) != tuple(candidate.shape) or reference.dtype != candidate.dtype:
        return {**result, "kind": "exact", "within_tolerance": False, "bitwise_equal": False,
                "reason": f"shape/dtype {tuple(candidate.shape)}/{candidate.dtype} differs"}
    reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
    bitwise = bool(torch.equal(reference, candidate))
    if not reference.is_floating_point():
        mismatch = int((reference != candidate).sum())
        return {**result, "kind": "exact", "bitwise_equal": bitwise, "mismatch_count": mismatch,
                "within_tolerance": mismatch == 0}
    ref, cand = reference.double(), candidate.double()
    nonfinite_ref, nonfinite_cand = int((~torch.isfinite(ref)).sum()), int((~torch.isfinite(cand)).sum())
    diff = (cand - ref).abs()
    same_inf = torch.isinf(ref) & (ref == cand)
    diff = torch.where(same_inf, torch.zeros_like(diff), diff)
    allowed = atol + rtol * ref.abs()
    mismatch = ~(diff <= allowed)  # NaN differences count as mismatches
    finite_diff = diff[torch.isfinite(diff)]
    large = ref.abs() > atol
    rel = (diff[large] / ref.abs()[large]) if bool(large.any()) else torch.zeros(0, dtype=torch.float64)
    rel = rel[torch.isfinite(rel)]
    return {
        **result, "kind": "close", "bitwise_equal": bitwise,
        "max_abs_error": float(finite_diff.max()) if finite_diff.numel() else None,
        "mean_abs_error": float(finite_diff.mean()) if finite_diff.numel() else None,
        "max_rel_error": float(rel.max()) if rel.numel() else 0.0,
        "small_denominator_count": int((~large).sum()),
        "mismatch_count": int(mismatch.sum()), "nonfinite_reference": nonfinite_ref,
        "nonfinite_candidate": nonfinite_cand, "within_tolerance": not bool(mismatch.any()),
    }
