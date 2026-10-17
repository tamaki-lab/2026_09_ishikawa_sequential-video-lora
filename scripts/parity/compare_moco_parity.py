"""Compare two Spec A parity runs step by step, rank by rank and phase by phase (CPU only).

    python -m scripts.parity.compare_moco_parity --reference <meta_reference run> --candidate <self run>
    python -m scripts.parity.compare_moco_parity --pair-kind rerun --reference <run> --candidate <same-impl rerun>
    python -m scripts.parity.compare_moco_parity --pair-kind resume --reference <uninterrupted> --candidate <resumed>

Pair kinds: ``implementation`` (meta_reference vs self), ``rerun`` (same implementation twice: measures the
reproducibility floor used to fix the threshold), ``resume`` (same implementation, candidate resumed from a
checkpoint; rank>0 BN buffers before the first forward are informational because the native checkpoint is saved by
rank 0 only and DDP rebroadcasts rank 0's buffers at that forward).

Discrete values (input hashes, permutation, targets, queue_ptr) must match exactly. Float tensors use
|c - r| <= atol + rtol * |r|. Full tensors exist on the runs' ``trace.full_steps``; every step also has the small
tensors (z_q, z_k, q, k, enqueued keys, positive logits, loss) and float64 group statistics (diagnostic only).
Status: PASS / FAIL only with ``--threshold-status fixed``; with the provisional threshold a clean comparison is
BLOCKED (diagnostic). Exact mismatches, failed runs or failed within-run checks are FAIL either way.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from scripts.parity.moco_parity_state import compare_tensors

SCHEMA_VERSION = "moco_parity_report/v1"
PASS, FAIL, BLOCKED, NOT_RUN = "PASS", "FAIL", "BLOCKED", "NOT_RUN"
PAIR_KINDS = ("implementation", "rerun", "resume")
PHASE_ORDER = ("inputs", "shuffle", "before_forward", "after_buffer_broadcast", "after_query", "after_key_ema",
               "after_key_forward", "after_logits_before_enqueue", "after_enqueue", "loss", "after_backward",
               "after_sgd", "metrics")
SMALL_PHASE = {
    "idx_shuffle": "shuffle", "idx_this": "shuffle", "z_q": "after_query", "z_k_shuffled": "after_key_forward",
    "q": "after_logits_before_enqueue", "k": "after_logits_before_enqueue",
    "positive_logits": "after_logits_before_enqueue", "targets": "after_logits_before_enqueue",
    "queue_ptr_before": "after_buffer_broadcast", "enqueued_keys": "after_enqueue", "queue_ptr_after": "after_enqueue",
    "loss": "loss",
}
IDENTITY_IGNORED = ("implementation",)


def load_run(run_dir: Path) -> dict:
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
    steps = {}
    for rank_dir in sorted((run_dir / "trace").glob("rank*")):
        rank = int(rank_dir.name[len("rank"):])
        for path in rank_dir.glob("step_*.pt"):
            steps.setdefault(int(path.stem[len("step_"):]), {})[rank] = path
    return {"dir": run_dir, "report": report, "config": config, "provenance": provenance, "manifest": manifest,
            "steps": steps}


def comparable_conditions(config: dict) -> dict:
    """Conditions that must agree between the two runs (identity of a comparison)."""
    return {
        "reference_commit": config["reference"]["commit"], "model": config["model"], "moco": config["moco"],
        "image_size": config["data"]["image_size"], "distributed": {k: config["distributed"][k] for k in
                                                                    ("world_size", "broadcast_buffers")},
        "precision": config["precision"],
        "training": {k: config["training"][k]
                     for k in ("seed", "local_batch_size", "learning_rate", "momentum", "weight_decay", "epochs",
                               "cos")},
    }


def _differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out += _differences(left.get(key), right.get(key), f"{prefix}{key}.")
        return out
    return [] if left == right else [f"{prefix[:-1]}: {left!r} != {right!r}"]


def precondition_issues(reference: dict, candidate: dict, pair_kind: str) -> list[str]:
    issues = []
    ref_impl, cand_impl = reference["report"]["implementation"], candidate["report"]["implementation"]
    if pair_kind == "implementation" and (ref_impl, cand_impl) != ("meta_reference", "self"):
        issues.append(f"implementation pair expects meta_reference vs self, got {ref_impl} vs {cand_impl}")
    if pair_kind in ("rerun", "resume") and ref_impl != cand_impl:
        issues.append(f"{pair_kind} pair expects one implementation, got {ref_impl} vs {cand_impl}")
    for run in (reference, candidate):
        if run["report"]["run_kind"] != "parity":
            issues.append(f"{run['dir']} is not a parity run")
    if pair_kind == "resume" and not any(r and r.get("resume") for r in candidate["manifest"]["per_rank"]):
        issues.append("resume pair: the candidate run did not resume from a checkpoint")
    fixtures = [run["manifest"]["fixture"]["sha256"] if run["manifest"].get("fixture") else None
                for run in (reference, candidate)]
    if fixtures[0] is None or fixtures[0] != fixtures[1]:
        issues.append(f"runs used different fixtures: {fixtures}")
    conditions = [comparable_conditions(run["config"]) for run in (reference, candidate)]
    issues += [f"condition differs: {d}" for d in _differences(*conditions)]
    flags = [run["provenance"]["backend_flags"] for run in (reference, candidate)]
    if flags[0] != flags[1]:
        issues.append(f"backend flags differ: {flags}")
    versions = [{k: run["provenance"]["environment"][k] for k in ("torch", "torchvision", "cuda", "cudnn", "nccl")}
                for run in (reference, candidate)]
    if versions[0] != versions[1]:
        issues.append(f"environment versions differ: {versions}")
    return issues


def _relative_stats_difference(ref: dict, cand: dict) -> float | None:
    worst = None
    for key in ("sum", "abs_sum", "sq_sum", "max_abs"):
        a, b = ref.get(key), cand.get(key)
        if a is None or b is None:
            continue
        rel = abs(b - a) / max(abs(a), 1e-30)
        worst = rel if worst is None else max(worst, rel)
    if ref.get("nonfinite") != cand.get("nonfinite") or ref.get("numel") != cand.get("numel"):
        return float("inf")
    return worst


def compare_step(ref_trace: dict, cand_trace: dict, rtol: float, atol: float, informational) -> tuple[list, list]:
    """Returns (judged comparisons, diagnostic statistics) for one (step, rank)."""
    step, rank = ref_trace["step"], ref_trace["rank"]
    judged, diagnostics = [], []

    def add(phase, group, name, result, judged_entry=True):
        entry = {"step": step, "rank": rank, "phase": phase, "group": group, "tensor": name, **result}
        if judged_entry and not informational(phase, group, rank):
            judged.append(entry)
        else:
            diagnostics.append({**entry, "informational": True})

    for view in ("im_q", "im_k"):
        same = ref_trace["input_sha256"][view] == cand_trace["input_sha256"][view]
        add("inputs", "views", view, {"kind": "exact", "within_tolerance": same, "bitwise_equal": same,
                                      "mismatch_count": 0 if same else 1})
    for name, phase in SMALL_PHASE.items():
        if name in ref_trace["small"] or name in cand_trace["small"]:
            if name not in ref_trace["small"] or name not in cand_trace["small"]:
                add(phase, "small", name, {"kind": "exact", "within_tolerance": False, "reason": "missing"})
                continue
            add(phase, "small", name, compare_tensors(ref_trace["small"][name], cand_trace["small"][name], rtol,
                                                      atol))
    lr = (ref_trace["metrics"]["lr"], cand_trace["metrics"]["lr"])
    add("metrics", "optimizer", "lr", compare_tensors(torch.tensor(lr[0], dtype=torch.float64),
                                                      torch.tensor(lr[1], dtype=torch.float64), rtol, atol))

    for phase in PHASE_ORDER:
        ref_phase, cand_phase = ref_trace["phases"].get(phase), cand_trace["phases"].get(phase)
        if ref_phase is None and cand_phase is None:
            continue
        if ref_phase is None or cand_phase is None:
            add(phase, "phase", "*", {"kind": "exact", "within_tolerance": False, "reason": "phase missing"})
            continue
        for group in sorted(set(ref_phase["stats"]) | set(cand_phase["stats"])):
            rel = _relative_stats_difference(ref_phase["stats"].get(group, {}), cand_phase["stats"].get(group, {}))
            diagnostics.append({"step": step, "rank": rank, "phase": phase, "group": group, "tensor": "*stats*",
                                "max_relative_stats_difference": rel})
        for group in sorted(set(ref_phase["full"]) & set(cand_phase["full"])):
            ref_group, cand_group = ref_phase["full"][group], cand_phase["full"][group]
            for name in sorted(set(ref_group) | set(cand_group)):
                if name not in ref_group or name not in cand_group:
                    add(phase, group, name, {"kind": "exact", "within_tolerance": False, "reason": "missing"})
                    continue
                add(phase, group, name, compare_tensors(ref_group[name], cand_group[name], rtol, atol))
    return judged, diagnostics


def summarize(entries: list) -> dict:
    summary = {}
    for entry in entries:
        key = f"{entry['phase']}/{entry['group']}"
        item = summary.setdefault(key, {"compared": 0, "mismatching": 0, "bitwise_equal": 0, "max_abs_error": 0.0,
                                        "max_rel_error": 0.0, "nonfinite": 0})
        item["compared"] += 1
        item["mismatching"] += not entry.get("within_tolerance", False)
        item["bitwise_equal"] += bool(entry.get("bitwise_equal"))
        for name in ("max_abs_error", "max_rel_error"):
            if entry.get(name) is not None:
                item[name] = max(item[name], entry[name])
        item["nonfinite"] += entry.get("nonfinite_reference", 0) + entry.get("nonfinite_candidate", 0)
    return dict(sorted(summary.items(), key=lambda kv: (PHASE_ORDER.index(kv[0].split("/")[0]), kv[0])))


def _phase_rank(entry) -> tuple:
    return entry["step"], PHASE_ORDER.index(entry["phase"]), entry["rank"], entry["group"], entry["tensor"]


def compare(reference_dir: Path, candidate_dir: Path, pair_kind: str, rtol: float, atol: float,
            threshold_status: str, gate: str | None = None) -> dict:
    if pair_kind not in PAIR_KINDS or threshold_status not in ("provisional", "fixed"):
        raise ValueError(f"pair_kind in {PAIR_KINDS}, threshold_status in (provisional, fixed)")
    reference, candidate = load_run(reference_dir), load_run(candidate_dir)
    issues = precondition_issues(reference, candidate, pair_kind)
    common = sorted(set(reference["steps"]) & set(candidate["steps"]))
    world_size = int(reference["config"]["distributed"]["world_size"])

    def informational(phase, group, rank):
        return pair_kind == "resume" and phase == "before_forward" and group in ("query_bn", "key_bn") and rank > 0

    judged, diagnostics, steps_compared = [], [], []
    if not issues:
        first_resumed = min(candidate["steps"], default=None)
        for step in common:
            if set(reference["steps"][step]) != set(range(world_size)) or \
                    set(candidate["steps"][step]) != set(range(world_size)):
                continue
            for rank in range(world_size):
                ref_trace = torch.load(reference["steps"][step][rank], map_location="cpu", weights_only=True)
                cand_trace = torch.load(candidate["steps"][step][rank], map_location="cpu", weights_only=True)
                rank_judged, rank_diag = compare_step(
                    ref_trace, cand_trace, rtol, atol,
                    lambda phase, group, r: informational(phase, group, r) and step == first_resumed)
                judged += rank_judged
                diagnostics += rank_diag
            steps_compared.append(step)

    within_run = {name: run["report"].get("verification", {}) for name, run in
                  (("reference", reference), ("candidate", candidate))}
    within_failures = [f"{side}: {check}" for side, checks in within_run.items() for check, result in checks.items()
                       if isinstance(result, dict) and result.get("status") in (FAIL, NOT_RUN)]
    run_failures = [f"{side} run status {run['report']['status']}: {run['report'].get('failures')}"
                    for side, run in (("reference", reference), ("candidate", candidate))
                    if run["report"]["status"] != "completed"]
    exact_failures = sorted((e for e in judged if e["kind"] == "exact" and not e["within_tolerance"]), key=_phase_rank)
    close_failures = sorted((e for e in judged if e["kind"] == "close" and not e["within_tolerance"]), key=_phase_rank)
    first = (exact_failures + close_failures)
    first = min(first, key=_phase_rank) if first else None

    if run_failures or within_failures or exact_failures:
        exact = [f"{len(exact_failures)} exact mismatches"] if exact_failures else []
        status, reason = FAIL, "; ".join(run_failures + within_failures[:5] + exact)
    elif issues:
        status, reason = BLOCKED, "; ".join(issues)
    elif not steps_compared:
        status, reason = NOT_RUN, "no common step with traces from every rank"
    elif close_failures and threshold_status == "fixed":
        status, reason = FAIL, f"{len(close_failures)} float tensors outside the fixed tolerance"
    elif threshold_status == "fixed":
        status, reason = PASS, "all compared tensors within the fixed tolerance; within-run checks PASS"
    else:
        status = BLOCKED
        reason = (f"threshold is provisional (spec §5.3): diagnostic only; {len(close_failures)} float tensors "
                  f"outside the provisional tolerance")
    return {
        "schema_version": SCHEMA_VERSION, "created": datetime.now().isoformat(timespec="seconds"),
        "gate": gate or candidate["report"].get("gate"), "pair_kind": pair_kind, "status": status, "reason": reason,
        "tolerance": {"rtol": rtol, "atol": atol, "status": threshold_status,
                      "rule": "|candidate - reference| <= atol + rtol * |reference|; NaN never matches"},
        "reference": _run_summary(reference), "candidate": _run_summary(candidate),
        "preconditions": issues, "steps_compared": steps_compared,
        "full_tensor_steps": sorted({e["step"] for e in judged if e["group"] not in ("small", "views", "optimizer")}),
        "first_failure": first,
        "counts": {"judged": len(judged), "exact_failures": len(exact_failures), "close_failures": len(close_failures),
                   "bitwise_equal": sum(bool(e.get("bitwise_equal")) for e in judged)},
        "summary_by_phase_group": summarize(judged),
        "diagnostic_stats_max_relative_difference": _stats_summary(diagnostics),
        "informational": [e for e in diagnostics if e.get("informational")],
        "within_run_checks": {side: {name: result.get("status") for name, result in checks.items()
                                     if isinstance(result, dict) and "status" in result}
                              for side, checks in within_run.items()},
        "comparisons": judged,
    }


def _run_summary(run: dict) -> dict:
    report = run["report"]
    return {"run_dir": str(run["dir"]), "implementation": report["implementation"], "status": report["status"],
            "gate": report.get("gate"), "completed_steps": report.get("completed_steps"),
            "config_identity_sha256": report.get("config_identity_sha256"),
            "fixture": run["manifest"].get("fixture"),
            "self_commit": run["provenance"]["self_repository"].get("commit"),
            "self_dirty": run["provenance"]["self_repository"].get("dirty"),
            "reference_commit": run["provenance"]["reference_clone"].get("commit"),
            "reference_diff_empty": run["provenance"]["reference_clone"].get("diff_empty")}


def _stats_summary(diagnostics: list) -> dict:
    summary = {}
    for entry in diagnostics:
        if entry.get("tensor") != "*stats*" or entry.get("max_relative_stats_difference") is None:
            continue
        key = f"{entry['phase']}/{entry['group']}"
        summary[key] = max(summary.get(key, 0.0), entry["max_relative_stats_difference"])
    return summary


def render_summary(report: dict) -> str:
    lines = [
        f"# MoCo parity comparison ({report['pair_kind']}, gate {report['gate']})", "",
        f"- status: **{report['status']}** — {report['reason']}",
        f"- reference: `{report['reference']['run_dir']}` ({report['reference']['implementation']}, "
        f"{report['reference']['status']})",
        f"- candidate: `{report['candidate']['run_dir']}` ({report['candidate']['implementation']}, "
        f"{report['candidate']['status']})",
        f"- tolerance: rtol={report['tolerance']['rtol']}, atol={report['tolerance']['atol']} "
        f"({report['tolerance']['status']}); {report['tolerance']['rule']}",
        f"- steps compared: {len(report['steps_compared'])} (full tensors at {report['full_tensor_steps']})",
        f"- judged comparisons: {report['counts']['judged']}, bitwise equal: {report['counts']['bitwise_equal']}, "
        f"exact failures: {report['counts']['exact_failures']}, float failures: {report['counts']['close_failures']}",
    ]
    if report["preconditions"]:
        lines += ["", "## Preconditions not met", ""] + [f"- {issue}" for issue in report["preconditions"]]
    if report["first_failure"]:
        f = report["first_failure"]
        lines += ["", "## First failure", "",
                  f"step {f['step']}, rank {f['rank']}, phase `{f['phase']}`, group `{f['group']}`, tensor "
                  f"`{f['tensor']}`: max_abs={f.get('max_abs_error')}, max_rel={f.get('max_rel_error')}, "
                  f"mismatches={f.get('mismatch_count')}, reason={f.get('reason')}"]
    lines += ["", "## Max error by phase / group", "",
              "| phase/group | compared | mismatching | bitwise equal | max abs | max rel | non-finite |",
              "|---|---|---|---|---|---|---|"]
    for key, item in report["summary_by_phase_group"].items():
        lines.append(f"| {key} | {item['compared']} | {item['mismatching']} | {item['bitwise_equal']} | "
                     f"{item['max_abs_error']:.3g} | {item['max_rel_error']:.3g} | {item['nonfinite']} |")
    lines += ["", "## Within-run checks", ""]
    for side, checks in report["within_run_checks"].items():
        failing = {k: v for k, v in checks.items() if v not in (PASS, "INFO")}
        lines.append(f"- {side}: {sum(v == PASS for v in checks.values())} PASS"
                     + (f", not passing: {failing}" if failing else ""))
    if report["informational"]:
        lines += ["", f"Informational (not judged): {len(report['informational'])} entries, e.g. rank>0 BN buffers "
                      "before the first resumed forward."]
    lines += ["", "## Not established", "",
              "- このreportはSpec AのGate判定の一部。表現品質・10 epoch学習・公開精度の再現は主張しない",
              "- provisional閾値での結果は診断値。正式閾値は判定用runより前にレビュー・確定する（spec §5.3）",
              "", "## Reproduction", "",
              f"`python -m scripts.parity.compare_moco_parity --pair-kind {report['pair_kind']} "
              f"--reference {report['reference']['run_dir']} --candidate {report['candidate']['run_dir']} "
              f"--rtol {report['tolerance']['rtol']} --atol {report['tolerance']['atol']} "
              f"--threshold-status {report['tolerance']['status']}`（各runの再現手順は各runの verification_summary.md）"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pair-kind", choices=PAIR_KINDS, default="implementation")
    parser.add_argument("--rtol", type=float, default=None, help="default: the reference run's verification.rtol")
    parser.add_argument("--atol", type=float, default=None, help="default: the reference run's verification.atol")
    parser.add_argument("--threshold-status", choices=("provisional", "fixed"), default="provisional")
    parser.add_argument("--gate", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = yaml.safe_load((args.reference / "resolved_config.yaml").read_text(encoding="utf-8"))
    rtol = args.rtol if args.rtol is not None else float(config["verification"]["rtol"])
    atol = args.atol if args.atol is not None else float(config["verification"]["atol"])
    output_dir = args.output_dir or Path("log/moco_imagenet_parity/compare") / \
        f"{datetime.now():%Y-%m-%d/%H-%M-%S-%f}_{args.pair_kind}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("parity_report.json", "verification_summary.md"):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} exists; reports are never overwritten")
    report = compare(args.reference, args.candidate, args.pair_kind, rtol, atol, args.threshold_status, args.gate)
    (output_dir / "parity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")
    (output_dir / "verification_summary.md").write_text(render_summary(report), encoding="utf-8")
    print(f"{report['status']}: {report['reason']}\nSaved {output_dir}")
    return report


if __name__ == "__main__":
    main()
