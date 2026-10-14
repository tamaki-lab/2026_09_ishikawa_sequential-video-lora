"""Evidence artifacts for the standalone ViT-LoRA CIFAR-10 smoke run.

Every PASS/FAIL/NOT_RUN status is computed from measured values stored in report.json.
The run raises before writing any of these files when a fail-fast check fails.
"""

import math
from collections.abc import Iterable
from pathlib import Path

import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
SERIES_COLORS = ("#2a78d6", "#eb6834")  # LoRA A, LoRA B
ARTIFACTS = (
    ("adapter_head.pt", "LoRA A/B と分類Headの重み（Base ViTは保存しない）"),
    ("resolved_config.yaml", "Hydraで解決済みの実行設定"),
    ("report.json", "このrunの測定値（本サマリーの元データ）"),
    ("loss.png", "stepごとのmini-batch cross-entropy loss"),
    ("lora_grad_norm.png", "stepごとのLoRA A/B別勾配norm"),
    ("verification_summary.md", "本ファイル"),
)


def grad_norm_sum(parameters: Iterable[torch.Tensor]) -> float | None:
    """Sum of per-tensor gradient norms; None (not 0) when no tensor has a gradient."""
    norms = [float(p.grad.detach().norm().item()) for p in parameters if p.grad is not None]
    return sum(norms) if norms else None


def changed_tensors(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> int:
    """Number of tensors that are not bit-exactly equal."""
    return sum(not torch.equal(after[key].detach().cpu(), before[key].detach().cpu()) for key in after)


def update_norm_sum(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return sum((float((after[key].detach().cpu() - before[key].detach().cpu()).norm().item()) for key in after), 0.0)


def _check(passed: bool) -> dict:
    return {"status": PASS if passed else FAIL, "reason": None}


def _not_run(reason: str) -> dict:
    return {"status": NOT_RUN, "reason": reason}


def evaluate_checks(report: dict) -> dict:
    """Judge each verification item from the measured fields of a report."""
    verification = report["verification"]
    lora_enabled = report["lora"]["enabled"]
    if not lora_enabled:
        lora_a = lora_b = _not_run("LoRA disabled: A/Bは注入されていない")
    else:
        lora_b = _check(verification["lora_b_changed_tensors"] > 0)
        if report["steps"] < 2:
            lora_a = _not_run("1 step: Bがゼロ初期化のためstep 1のA勾配は0。Aの更新は2 step以上で検査する")
        else:
            lora_a = _check(verification["lora_a_changed_tensors"] > 0)
    return {
        "base_frozen_requires_grad": _check(verification["base_frozen_requires_grad"] is True),
        "base_no_gradient": _check(verification["base_gradient_tensors_max"] == 0),
        "base_unchanged": _check(verification["base_changed_tensors"] == 0),
        "optimizer_matches_trainable_parameters": _check(
            verification["optimizer_matches_trainable_parameters"] is True
        ),
        "lora_a_updated": lora_a,
        "lora_b_updated": lora_b,
        "head_updated": _check(verification["head_changed_tensors"] > 0),
        "checkpoint_roundtrip": _check(verification["checkpoint_roundtrip_passed"] is True),
    }


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Measured values per step; a missing value becomes NaN (a gap), never zero."""
    steps = [int(item["step"]) for item in history]
    values = [math.nan if item.get(key) is None else float(item[key]) for item in history]
    return steps, values


def _new_axis(steps: list[int]):
    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.add_subplot()
    axis.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    if len(steps) <= 30:
        axis.set_xticks(steps)
    else:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(color="#d9d9d6", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    return figure, axis


def save_loss_plot(history: list[dict], path: Path) -> Figure:
    steps, losses = _series(history, "loss")
    figure, axis = _new_axis(steps)
    axis.plot(steps, losses, color=SERIES_COLORS[0], linewidth=2, marker="o", markersize=5)
    axis.set_title("Training mini-batch cross-entropy loss")
    axis.set_xlabel("Step (each step uses a different mini-batch)")
    axis.set_ylabel("Loss")
    figure.savefig(path, dpi=150)
    return figure


def save_lora_grad_norm_plot(history: list[dict], path: Path) -> Figure:
    figure, axis = _new_axis([int(item["step"]) for item in history])
    for key, label, color, marker in (
        ("lora_a_grad_norm_sum", "LoRA A", SERIES_COLORS[0], "o"),
        ("lora_b_grad_norm_sum", "LoRA B", SERIES_COLORS[1], "s"),
    ):
        steps, values = _series(history, key)
        if not all(math.isnan(value) for value in values):
            axis.plot(steps, values, color=color, linewidth=2, marker=marker, markersize=5, label=label)
    if axis.lines:
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No LoRA A/B gradient values in report.json",
                  ha="center", va="center", transform=axis.transAxes)
    axis.set_title("LoRA gradient norm per step (sum of per-tensor L2 norms)")
    axis.set_xlabel("Step")
    axis.set_ylabel("Gradient norm")
    figure.savefig(path, dpi=150)
    return figure


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _failed_or_not_run(checks: dict, names: tuple[str, ...]) -> list[str]:
    return [f"{name}={checks[name]['status']}" for name in names if checks[name]["status"] != PASS]


def render_verification_summary(report: dict) -> str:
    """Markdown for the professor-facing evidence, generated only from report values."""
    verification, checks, lora = report["verification"], report["checks"], report["lora"]
    history, git = report["history"], report["git"]
    tolerance = verification["checkpoint_roundtrip_tolerance"]
    targets = ", ".join(report["selected_targets"]) or "なし"
    lora_text = (f"enabled / targets {targets} / rank {lora['rank']} / alpha {_fmt(lora['alpha'])} / "
                 f"dropout {_fmt(lora['dropout'])}" if lora["enabled"] else "disabled（注入なし）")

    rows = [
        ("Base: 全parameterが `requires_grad=False`", "base_frozen_requires_grad",
         f"{_fmt(verification['base_frozen_requires_grad'])}（{_fmt(verification['base_parameter_tensors'])} tensors, "
         f"{_fmt(verification['base_parameters'])} params）"),
        ("Base: 勾配を受け取らない", "base_no_gradient",
         f"全stepでのBase勾配tensor数の最大値 {_fmt(verification['base_gradient_tensors_max'])}"),
        ("Base: 学習前後でbit-exact一致（`torch.equal`）", "base_unchanged",
         f"変更 {_fmt(verification['base_changed_tensors'])} / {_fmt(verification['base_parameter_tensors'])} tensors"),
        ("Optimizer対象 = LoRA A/B + Head（param ID集合）", "optimizer_matches_trainable_parameters",
         f"一致 {_fmt(verification['optimizer_matches_trainable_parameters'])}（trainable "
         f"{_fmt(report['trainable_parameters'])} params）"),
        ("LoRA A: 学習前後で変化", "lora_a_updated",
         f"変更 {_fmt(verification['lora_a_changed_tensors'])} / {_fmt(verification['lora_a_tensors'])} tensors, "
         f"update norm sum {_fmt(verification['lora_a_update_norm_sum'])}"),
        ("LoRA B: 学習前後で変化", "lora_b_updated",
         f"変更 {_fmt(verification['lora_b_changed_tensors'])} / {_fmt(verification['lora_b_tensors'])} tensors, "
         f"update norm sum {_fmt(verification['lora_b_update_norm_sum'])}"),
        ("Head: 学習前後で変化", "head_updated",
         f"変更 {_fmt(verification['head_changed_tensors'])} / {_fmt(verification['head_tensors'])} tensors, "
         f"update norm sum {_fmt(report['head_update_norm_sum'])}"),
        ("Checkpoint保存・再読込後のeval logits一致", "checkpoint_roundtrip",
         f"max abs diff {_fmt(verification['reload_logits_max_abs_diff'])}"
         f"（rtol {_fmt(tolerance['rtol'])}, atol {_fmt(tolerance['atol'])}）"),
    ]
    not_run = [(label, checks[name]["reason"]) for label, name, _ in rows if checks[name]["status"] == NOT_RUN]

    base_names = ("base_frozen_requires_grad", "base_no_gradient", "base_unchanged")
    base_issues = _failed_or_not_run(checks, base_names)
    base_conclusion = (
        "Base凍結に関する3検査はすべてPASS。学習中、事前学習済みViTのparameterは勾配を受け取らず、値も変化していない。"
        if not base_issues else f"Base凍結の検証はPASSしていない: {', '.join(base_issues)}。"
    )
    update_names = ("optimizer_matches_trainable_parameters", "lora_a_updated", "lora_b_updated", "head_updated")
    update_issues = _failed_or_not_run(checks, update_names)
    update_conclusion = (
        "Optimizer対象はLoRA A/BとHeadに一致し、LoRA A・LoRA B・Headはいずれも学習前後で変化した。"
        if not update_issues else f"学習による更新の検証にPASS以外の項目がある: {', '.join(update_issues)}。"
    )

    lines = [
        "# Standalone ViT-LoRA CIFAR-10 検証サマリー",
        "",
        "> このファイルは `scripts/smoke/smoke_vit_lora_cifar10.py` が、同じrunの `report.json` から自動生成した。"
        "表の数値と判定はすべてこのrunで測定した値から計算しており、固定のPASSは含まない。"
        "Base凍結・Base勾配・Base不変・optimizer集合・Head更新・checkpoint再読込などのfail-fast検査で例外が出たrunでは、"
        "このファイルは作成されない。",
        "",
        "## 実行設定",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| checkpoint | `{report['model_checkpoint']}` |",
        f"| LoRA | {lora_text} |",
        f"| seed | {_fmt(report['seed'])} |",
        f"| subset | {report['dataset']} から {_fmt(report['train_subset_size'])} 枚"
        "（indexは `report.json` の `dataset_indices`） |",
        f"| batch size | {_fmt(report['batch_size'])} |",
        f"| device | `{report['device']}` |",
        f"| steps | {_fmt(report['steps'])} |",
        f"| Git | HEAD `{git['commit']}` / dirty: {_fmt(git['dirty']) if git['dirty'] is not None else 'unknown'} |",
        "",
        "## 検証結果（このrunの測定値）",
        "",
        "| 検査 | 判定 | 測定値 |",
        "| --- | --- | --- |",
        *[f"| {label} | {checks[name]['status']} | {value} |" for label, name, value in rows],
        "",
    ]
    if not_run:
        lines += ["NOT_RUN の理由:", "", *[f"- {label}: {reason}" for label, reason in not_run], ""]
    lines += [
        "## stepごとの記録",
        "",
        "各stepは異なるmini-batchで計算している。`n/a` は値が存在しないことを示し、0の代用ではない。",
        "",
        "| step | mini-batch CE loss | LoRA A grad norm | LoRA B grad norm | Head grad norm |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *[f"| {item['step']} | {_fmt(item['loss'])} | {_fmt(item.get('lora_a_grad_norm_sum'))} | "
          f"{_fmt(item.get('lora_b_grad_norm_sum'))} | {_fmt(item['head_grad_norm_sum'])} |" for item in history],
        "",
        "![loss](loss.png)",
        "",
        "![LoRA gradient norm](lora_grad_norm.png)",
        "",
        "## 結論",
        "",
        f"- Base凍結: {base_conclusion}",
        f"- 学習: {update_conclusion}この学習は LoRA + classifier head の学習であり、LoRAのみの学習ではない。",
        "- 限界: mini-batch lossの低下だけでは、LoRAが分類性能に寄与したことは示せない"
        "（Headも同時に学習しており、各stepのmini-batchも異なる）。"
        "固定画像での学習前後評価、LoRA ON/OFF推論比較、head-only baseline、未知画像・複数seedでの評価は行っていない。",
        "",
        "## コード由来の前提（このrunでは再測定していない）",
        "",
        "- `model/vit/lora_linear.py` はLoRA Bをゼロで初期化する。そのためstep 1のLoRA A勾配が0になるのは仕様どおりである。",
        "- 学習対象の期待集合は、ViT内の `.lora_a.` / `.lora_b.` parameterと分類Headのparameterとして定義している。",
        "",
        "## Artifacts",
        "",
        *[f"- `{name}`: {description}" for name, description in ARTIFACTS],
        "",
        "数値の元データは同じディレクトリの `report.json`（`verification`, `checks`, `history`）。",
        "",
    ]
    return "\n".join(lines)


def write_verification_artifacts(output_dir: Path, report: dict) -> None:
    save_loss_plot(report["history"], output_dir / "loss.png")
    save_lora_grad_norm_plot(report["history"], output_dir / "lora_grad_norm.png")
    (output_dir / "verification_summary.md").write_text(render_verification_summary(report), encoding="utf-8")
