"""Build the canonical Spec A fixture shared by the reference and self parity runs (CPU only, no GPU).

    python -m scripts.parity.build_moco_parity_fixture fixture.kind=synthetic
    python -m scripts.parity.build_moco_parity_fixture fixture.kind=imagenet fixture.imagenet_steps=1
    # small-K A1 fixture: moco.queue_size=64 training.local_batch_size=4 data.image_size=64

The initial state is the reference ``moco.builder.MoCo`` built from the unmodified pinned clone after
``torch.manual_seed(training.seed)``: query/key ResNet50 + MLP, BN buffers, the random unit queue and ptr 0.
Both implementations load exactly this state; the optimizer starts fresh (no momentum buffers).

Inputs per (step, rank): ``synthetic`` draws standard-normal views from derived seeds at run time;
``imagenet`` stores views made once here from the DistributedSampler order of epoch 0 (shuffle, seed 0, as
main_moco.py) with the reference aug-plus pipeline under per-sample derived seeds. Both runs therefore consume
identical tensors, which the runs confirm by hash. Writes fixture.pt and fixture_manifest.json; never overwrites.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from scripts.parity.moco_parity_state import (
    REFERENCE, REPOSITORY_ROOT, derive_seed, environment, file_sha256, git_info, import_reference, load_config,
    reference_two_crops, state_sha256, tensor_sha256,
)

SCHEMA_VERSION = "moco_parity_fixture/v1"
CONFIG_PATH = REPOSITORY_ROOT / "conf" / "moco_imagenet_parity.yaml"
UNIT_NORM_ATOL = 1e-4


def check_reference_clone(cfg) -> dict:
    info = git_info(cfg.reference.root)
    if info.get("commit") != str(cfg.reference.commit) or cfg.reference.commit != REFERENCE["commit"]:
        raise RuntimeError(f"reference clone is at {info.get('commit')}, expected {REFERENCE['commit']}")
    if info["dirty"]:
        raise RuntimeError(f"reference clone has local changes: {info['dirty_paths']}")
    return info


def reference_initial_state(builder, cfg) -> dict:
    """State dict of a freshly built reference MoCo; checks the reference initialization invariants."""
    torch.manual_seed(int(cfg.training.seed))
    model = builder.MoCo(torchvision.models.__dict__[str(cfg.model.arch)], int(cfg.model.projection_dim),
                         int(cfg.moco.queue_size), float(cfg.moco.momentum), float(cfg.moco.temperature),
                         bool(cfg.model.mlp))
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    query = {name[len("encoder_q."):]: t for name, t in state.items() if name.startswith("encoder_q.")}
    key = {name[len("encoder_k."):]: t for name, t in state.items() if name.startswith("encoder_k.")}
    if query.keys() != key.keys() or not all(torch.equal(query[n], key[n]) for n in query):
        raise RuntimeError("reference query and key encoders do not start identical")
    if any(p.requires_grad for p in model.encoder_k.parameters()):
        raise RuntimeError("reference key encoder parameters require grad")
    shared = {p.untyped_storage().data_ptr() for p in model.encoder_q.parameters()} & \
        {p.untyped_storage().data_ptr() for p in model.encoder_k.parameters()}
    if shared:
        raise RuntimeError("reference query and key parameters share storage")
    if float((state["queue"].norm(dim=0) - 1).abs().max()) > UNIT_NORM_ATOL or int(state["queue_ptr"]) != 0:
        raise RuntimeError("reference queue is not K unit columns with pointer 0")
    return state


def imagenet_views(cfg, loader):
    """Fixed views of epoch 0 for each (step, rank) and the per-sample records."""
    root = Path(str(cfg.data.root))
    two_crops = reference_two_crops(loader, int(cfg.data.image_size))
    dataset = ImageFolder(str(root / str(cfg.data.split)))
    world_size, batch, steps = int(cfg.distributed.world_size), int(cfg.training.local_batch_size), \
        int(cfg.fixture.imagenet_steps)
    seed = int(cfg.training.seed)
    inputs, samples = [[None] * world_size for _ in range(steps)], []
    for rank in range(world_size):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)  # shuffle=True, seed=0
        sampler.set_epoch(0)
        order = list(sampler)[:steps * batch]
        for step in range(1, steps + 1):
            views = []
            for position, index in enumerate(order[(step - 1) * batch:step * batch]):
                path, _ = dataset.samples[index]  # the class label is not used by the SSL loss
                augment_seed = derive_seed(seed, "augment", step, rank, position)
                random.seed(augment_seed)  # reference GaussianBlur draws sigma with Python's random
                torch.manual_seed(augment_seed)  # torchvision random transforms use torch's RNG
                views.append(two_crops(dataset.loader(path)))
                samples.append({"step": step, "rank": rank, "position": position, "index": index,
                                "path": str(Path(path).relative_to(root)), "file_bytes": Path(path).stat().st_size,
                                "file_sha256": file_sha256(path), "augment_seed": augment_seed})
            inputs[step - 1][rank] = tuple(torch.stack([v[i] for v in views]) for i in range(2))
    dataset_info = {"root": str(root), "split": str(cfg.data.split), "num_samples": len(dataset),
                    "num_classes": len(dataset.classes), "sampler": "DistributedSampler(shuffle=True, seed=0), "
                    "set_epoch(0), first local_batch_size * imagenet_steps indices per rank",
                    "augmentation": repr(two_crops.base_transform)}
    return inputs, samples, dataset_info


def build(cfg, output_dir: Path) -> dict:
    world_size, batch = int(cfg.distributed.world_size), int(cfg.training.local_batch_size)
    if int(cfg.moco.queue_size) % (world_size * batch):
        raise ValueError("moco.queue_size must be a multiple of the global batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fixture.pt", "fixture_manifest.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} exists; fixtures are never overwritten")
    clone = check_reference_clone(cfg)
    reference = import_reference(cfg.reference.root)
    state = reference_initial_state(reference["builder"], cfg)

    kind = str(cfg.fixture.kind)
    meta = {
        "schema_version": SCHEMA_VERSION, "kind": kind, "created": datetime.now().isoformat(timespec="seconds"),
        "seed": int(cfg.training.seed), "world_size": world_size, "local_batch_size": batch,
        "global_batch_size": world_size * batch, "image_size": int(cfg.data.image_size),
        "arch": str(cfg.model.arch), "mlp": bool(cfg.model.mlp), "projection_dim": int(cfg.model.projection_dim),
        "queue_size": int(cfg.moco.queue_size), "momentum": float(cfg.moco.momentum),
        "temperature": float(cfg.moco.temperature),
        "reference": {**REFERENCE, "clone": clone, "loaded": reference["provenance"]},
        "builder_call": "torch.manual_seed(seed); moco.builder.MoCo(torchvision.models.<arch>, dim, K, m, T, mlp)",
        "state_sha256": state_sha256(state), "state_tensors": len(state),
        "optimizer": "fresh torch.optim.SGD in both runs (no momentum buffers); LR from the epoch-0 schedule",
        "shuffle": "rank r calls torch.default_generator.manual_seed(shuffle_seed(seed, step, r)) right before the "
                   "forward of step (CPU global RNG only); "
                   "the permutation drawn by rank 0 is the one broadcast",
        "self_repository": git_info(REPOSITORY_ROOT), "environment": environment(),
    }
    inputs = None
    if kind == "synthetic":
        meta["inputs"] = "synthetic_views(seed, step, rank, local_batch_size, image_size): torch.randn float32 " \
                         "from derive_seed(seed, 'input', step, rank, view); any number of steps"
    elif kind == "imagenet":
        inputs, samples, dataset_info = imagenet_views(cfg, reference["loader"])
        meta.update(inputs=f"stored real-image views for steps 1..{len(inputs)}", steps=len(inputs),
                    dataset=dataset_info, samples=samples,
                    input_sha256=[[{"im_q": tensor_sha256(q), "im_k": tensor_sha256(k)} for q, k in step]
                                  for step in inputs])
    else:
        raise ValueError(f"fixture.kind must be synthetic or imagenet, got {kind}")

    path = output_dir / "fixture.pt"
    torch.save({"meta": meta, "state": state, "inputs": inputs}, path)
    manifest = {**meta, "fixture_path": str(path), "fixture_sha256": file_sha256(path),
                "state_tensor_sha256": {name: tensor_sha256(t) for name, t in state.items()}}
    (output_dir / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    print(f"Saved {path} ({manifest['fixture_sha256']})")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. fixture.kind=imagenet")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(CONFIG_PATH, args.overrides)
    stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S-%f")
    build(cfg, args.output_dir or Path("log/moco_imagenet_parity/fixtures") / f"{stamp}_{cfg.fixture.kind}")


if __name__ == "__main__":
    main()
