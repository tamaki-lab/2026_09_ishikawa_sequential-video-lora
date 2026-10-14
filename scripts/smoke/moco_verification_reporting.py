"""Evidence artifacts for the standalone MoCo v2-style CIFAR-10 run.

Every PASS/FAIL/NOT_RUN status is computed from measured values stored in report.json.
Failed runs write the same artifacts, with the failure and the steps measured before it.
Diagnostic values (gradient spikes, norm shrinkage, variance collapse) are described, not judged.
"""

import math
import statistics
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
REFERENCE_LINE_COLOR = "#52514e"
ARTIFACTS = (
    ("resolved_config.yaml", "Hydraで解決済みの実行設定"),
    ("report.json", "このrunの測定値（本サマリーの元データ）"),
    ("loss.png", "optimizer stepごとのInfoNCE loss"),
    ("similarity.png", "positive / mean negative / hardest negative cosine"),
    ("grad_update.png", "Query Backbone/Projectorの勾配normとSGD後の実update norm"),
    ("queue.png", "Queueのrandom残数・実画像key数・pointer"),
    ("verification_summary.md", "本ファイル"),
    ("checkpoint.pt", "Query/Key・Queue・Optimizer・Scheduler・RNG・sampler位置"),
)
OFFICIAL_DIFFERENCES = (
    "Backbone: 公式ResNet50ではなくResNet18（3x3 stride1 stem、max-poolなし、学習済み重みなし）",
    "Data: ImageNetではなくCIFAR-10 train（32x32）。augmentationは公式aug_plusの32x32適応で、"
    "GaussianBlurはkernel 3・sigma 0.1–2.0（公式は224px向けPIL blur）",
    "Device: single-device。DDP・all_gather・batch shuffleなし（all_gatherは恒等）",
    "BatchNorm: Query/Keyとも`train()`。Key BN running bufferはKey forwardでのみ更新し、"
    "parameter（weight/bias）だけEMA。公式のshuffle-BNは再現しない",
    "Queue K=1024、batch 32、SGD lr 0.00375（公式lr 0.03 / batch 256の比例換算）",
)
HISTORY_TABLE_FULL_LIMIT = 40


def _check(passed: bool) -> dict:
    return {"status": PASS if passed else FAIL, "reason": None}


def _not_run(reason: str) -> dict:
    return {"status": NOT_RUN, "reason": reason}


def _measured(value, judge) -> dict:
    """PASS/FAIL from a measured value; NOT_RUN when the value was never measured."""
    if value is None:
        return _not_run("このrunでは測定値がない（失敗またはその段階に未到達）")
    return _check(bool(judge(value)))


def expected_shapes(batch_size: int, queue_size: int, dim: int) -> dict[str, list[int]]:
    return {
        "im_q": [batch_size, 3, 32, 32], "im_k": [batch_size, 3, 32, 32],
        "q": [batch_size, dim], "k": [batch_size, dim], "logits": [batch_size, 1 + queue_size],
    }


def evaluate_checks(report: dict) -> dict:
    """Judge each verification item from the measured fields of a report."""
    verification = report["verification"]
    queue_expected = verification.get("expected_queue")
    queue_final = verification.get("final_queue")
    shapes = verification.get("observed_shapes")
    expected = verification.get("expected_shapes")
    return {
        "initial_query_key_identical": _measured(
            verification.get("initial_query_key"),
            lambda v: v["state_equal"] and v["shared_storage_tensors"] == 0,
        ),
        "key_requires_grad_false": _measured(verification.get("key_requires_grad_tensors"), lambda v: v == 0),
        "optimizer_targets_query_only": _measured(
            verification.get("optimizer_matches_query_parameters"), lambda v: v is True,
        ),
        "optimizer_steps_completed": _check(
            report["optimizer"]["completed_steps"] == report["optimizer"]["max_steps"]
        ),
        "tensor_shapes": _measured(
            shapes if shapes else None,
            lambda v: all(v.get(name) == [shape] for name, shape in expected.items()),
        ),
        "history_finite": _measured(verification.get("history_finite") if report["history"] else None,
                                    lambda v: v is True),
        "key_no_gradient": _measured(verification.get("key_grad_tensors_max"), lambda v: v == 0),
        "ema_within_tolerance": _measured(verification.get("ema_within_tolerance_all_steps"), lambda v: v is True),
        "query_backbone_updated": _measured(verification.get("query_backbone_changed_tensors"), lambda v: v > 0),
        "query_projector_updated": _measured(verification.get("query_projector_changed_tensors"), lambda v: v > 0),
        "queue_pointer_and_counts": _measured(
            queue_final,
            lambda v: all(v[key] == queue_expected[key]
                          for key in ("queue_ptr", "real_key_count", "random_remaining_count")),
        ),
        "queue_unit_norm_finite": _measured(
            queue_final,
            lambda v: v["finite"] and v["column_norm_max_abs_error"] <= verification["unit_norm_atol"],
        ),
        "data_order_and_zero_targets": _measured(
            verification.get("data_order_matches_all_steps") if report["history"] else None,
            lambda v: v is True and verification["targets_all_zero_all_steps"] is True,
        ),
        "checkpoint_roundtrip": _measured(
            verification.get("checkpoint_roundtrip"), lambda v: v["passed"] is True,
        ),
    }


def gate_statuses(report: dict) -> dict:
    """Gate-level status of this run; Gate A and the Gate D test suite run in pytest, not here."""
    all_pass = report["status"] == "completed" and all(
        check["status"] == PASS for check in report["checks"].values()
    )
    run_kind = report["run_kind"]
    gate_b = (PASS if all_pass else FAIL) if run_kind == "gate_b_smoke20" else NOT_RUN
    gate_c = (PASS if all_pass else FAIL) if run_kind == "gate_c_pilot500" else NOT_RUN
    return {
        "A": {"status": NOT_RUN, "note": "CPU unit testsはpytestで別途実行する（このrunでは実行していない）"},
        "B": {"status": gate_b, "note": "20step smoke（実CIFAR-10 50,000枚・既定設定・fresh run）"
              + ("" if gate_b != NOT_RUN else f"。このrunは `{run_kind}`")},
        "C": {"status": gate_c, "note": "500step pilotの工学的検査のみ。研究的な診断は自動判定しない"
              + ("" if gate_c != NOT_RUN else f"。このrunは `{run_kind}`")},
        "D": {"status": report["checks"]["checkpoint_roundtrip"]["status"],
              "note": "run内はcheckpoint保存→別インスタンス復元のみ。resume・非回帰はpytestで別途実行する"},
    }


def _numeric(history: list[dict], key: str) -> list[float]:
    return [float(item[key]) for item in history if item.get(key) is not None]


def describe_series(history: list[dict], key: str) -> dict | None:
    """Descriptive statistics only; thresholds for 'abnormal' are deliberately not applied."""
    values = _numeric(history, key)
    if not values:
        return None
    steps = [int(item["global_step"]) for item in history if item.get(key) is not None]
    peak = max(range(len(values)), key=values.__getitem__)
    median = statistics.median(values)
    return {
        "first": values[0], "last": values[-1], "min": min(values), "max": max(values),
        "median": median, "max_step": steps[peak],
        "max_over_median": values[peak] / median if median > 0 else None,
    }


DIAGNOSTIC_KEYS = (
    ("loss", "InfoNCE loss"),
    ("positive_cosine_mean", "positive cosine（batch平均）"),
    ("hardest_negative_cosine_mean", "hardest negative cosine（batch平均）"),
    ("query_backbone_grad_norm", "Query Backbone grad norm"),
    ("query_projector_grad_norm", "Query Projector grad norm"),
    ("query_backbone_update_norm", "Query Backbone update norm"),
    ("z_q_norm_min", "正規化前 ||z_q|| の最小値"),
    ("z_k_norm_min", "正規化前 ||z_k|| の最小値"),
    ("q_variance_min", "正規化後qの次元別batch variance最小値"),
    ("k_variance_min", "正規化後kの次元別batch variance最小値"),
)


def build_diagnostics(history: list[dict]) -> dict:
    return {key: describe_series(history, key) for key, _ in DIAGNOSTIC_KEYS}


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Measured values per step; a missing value becomes NaN (a gap), never zero."""
    steps = [int(item["global_step"]) for item in history]
    values = [math.nan if item.get(key) is None else float(item[key]) for item in history]
    return steps, values


def _style_axis(axis, steps: list[int]) -> None:
    if steps:
        axis.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    if len(steps) <= 30:
        axis.set_xticks(steps)
    else:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(color="#d9d9d6", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _plot_lines(axis, history: list[dict], series: tuple[tuple[str, ...], ...]) -> None:
    """Each series is (history key, label) or (history key, label, linestyle)."""
    show_markers = len(history) <= 30
    for item, color, marker in zip(series, SERIES_COLORS, ("o", "s", "^")):
        key, label, linestyle = (*item, "-")[:3]
        steps, values = _series(history, key)
        if values and not all(math.isnan(value) for value in values):
            axis.plot(steps, values, color=color, linewidth=2, linestyle=linestyle, label=label,
                      marker=marker if show_markers else None, markersize=5)


def _mark_queue_replaced(axis, history: list[dict], replaced_step: int) -> str:
    """Draw the step after which no initial random key remains (K / batch size); return the x-axis label."""
    if history and int(history[0]["global_step"]) <= replaced_step <= int(history[-1]["global_step"]):
        axis.axvline(replaced_step, color=REFERENCE_LINE_COLOR, linewidth=1, linestyle=":",
                     label=f"step {replaced_step}: initial random keys fully replaced (K/B)")
    elif history:
        return f"Optimizer step (initial random keys remain until step {replaced_step} = K/B, beyond this run)"
    return "Optimizer step"


def _finish_axis(axis, history: list[dict], legend: bool = True) -> None:
    """Legend below the plot area (outside the data) when the axis has more than one labeled artist."""
    handles, labels = axis.get_legend_handles_labels()
    if legend and len(handles) > 1:
        axis.figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    if not history:
        axis.text(0.5, 0.5, "No measured optimizer step in report.json",
                  ha="center", va="center", transform=axis.transAxes)


def save_loss_plot(history: list[dict], path: Path, replaced_step: int) -> Figure:
    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.add_subplot()
    _style_axis(axis, [int(item["global_step"]) for item in history])
    _plot_lines(axis, history, (("loss", "InfoNCE loss"),))
    axis.set_xlabel(_mark_queue_replaced(axis, history, replaced_step))
    _finish_axis(axis, history)
    axis.set_title("InfoNCE loss per optimizer step (each step uses a different mini-batch)")
    axis.set_ylabel("Loss")
    figure.savefig(path, dpi=150)
    return figure


def save_similarity_plot(history: list[dict], path: Path, replaced_step: int) -> Figure:
    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.add_subplot()
    _style_axis(axis, [int(item["global_step"]) for item in history])
    _plot_lines(axis, history, (
        ("positive_cosine_mean", "positive (q·k)"),
        ("negative_cosine_mean", "mean negative (queue before enqueue)"),
        ("hardest_negative_cosine_mean", "hardest negative (max over queue)"),
    ))
    axis.set_xlabel(_mark_queue_replaced(axis, history, replaced_step))
    _finish_axis(axis, history)
    axis.set_title("Cosine similarity per optimizer step (batch mean)")
    axis.set_ylabel("Cosine similarity")
    figure.savefig(path, dpi=150)
    return figure


def save_grad_update_plot(history: list[dict], path: Path) -> Figure:
    figure = Figure(figsize=(7, 6), layout="constrained")
    grad_axis, update_axis = figure.subplots(2, 1, sharex=True)
    steps = [int(item["global_step"]) for item in history]
    for axis, prefix, title, ylabel in (
        (grad_axis, "grad", "Gradient L2 norm (before the SGD step)", "Gradient norm"),
        (update_axis, "update", "Actual parameter update L2 norm (after the SGD step)",
         "||theta_after - theta_before||"),
    ):
        _style_axis(axis, steps)
        _plot_lines(axis, history, (
            (f"query_backbone_{prefix}_norm", "Query Backbone"),
            (f"query_projector_{prefix}_norm", "Query Projector"),
        ))
        _finish_axis(axis, history, legend=axis is update_axis)  # both panels share the same two series
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    update_axis.set_xlabel("Optimizer step")
    figure.savefig(path, dpi=150)
    return figure


def save_queue_plot(history: list[dict], path: Path, replaced_step: int) -> Figure:
    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.add_subplot()
    _style_axis(axis, [int(item["global_step"]) for item in history])
    _plot_lines(axis, history, (
        ("random_remaining_count", "random keys remaining"),
        ("real_key_count", "real-image keys"),
        ("queue_ptr", "queue pointer (dashed)", "--"),
    ))
    axis.set_xlabel(_mark_queue_replaced(axis, history, replaced_step))
    _finish_axis(axis, history)
    axis.set_title("Queue state after each step's enqueue (pointer = real keys until full)")
    axis.set_ylabel("Number of queue columns")
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


def _history_rows(history: list[dict], replaced_step: int) -> list[dict]:
    if len(history) <= HISTORY_TABLE_FULL_LIMIT:
        return history
    last = int(history[-1]["global_step"])
    keep = {1, 2, 3, replaced_step - 1, replaced_step, replaced_step + 1, last}
    return [item for item in history if int(item["global_step"]) in keep or int(item["global_step"]) % 50 == 0]


def _check_rows(report: dict) -> list[tuple[str, str, str]]:
    verification, optimizer = report["verification"], report["optimizer"]
    initial = verification.get("initial_query_key") or {}
    final, expected = verification.get("final_queue") or {}, verification.get("expected_queue") or {}
    roundtrip = verification.get("checkpoint_roundtrip") or {}
    return [
        ("Query/Key初期state一致・storage非共有", "initial_query_key_identical",
         f"state一致 {_fmt(initial.get('state_equal'))}, "
         f"共有storage {_fmt(initial.get('shared_storage_tensors'))} tensors"),
        ("Key全parameterが `requires_grad=False`", "key_requires_grad_false",
         f"requires_grad=True のKey tensor {_fmt(verification.get('key_requires_grad_tensors'))}"),
        ("Optimizer対象 = Query Backbone + Projector（param ID集合）", "optimizer_targets_query_only",
         f"一致 {_fmt(verification.get('optimizer_matches_query_parameters'))}, "
         f"Query {_fmt(report['model'].get('query_parameters'))} params"),
        ("optimizer update回数", "optimizer_steps_completed",
         f"{_fmt(optimizer['completed_steps'])} / {_fmt(optimizer['max_steps'])}"),
        ("im_q/im_k/q/k/logitsの形状（全step）", "tensor_shapes",
         ", ".join(f"{name} {shapes}" for name, shapes in (verification.get("observed_shapes") or {}).items())
         or "n/a"),
        ("history全値finite", "history_finite", f"{_fmt(len(report['history']))} steps"),
        ("Key勾配なし（全step）", "key_no_gradient",
         f"grad付きKey tensor数の最大 {_fmt(verification.get('key_grad_tensors_max'))}"),
        ("EMA = m·θk + (1−m)·θq（Optimizer更新前のθq）", "ema_within_tolerance",
         f"max abs error {_fmt(verification.get('ema_max_abs_error_max'))}"
         f"（rtol {_fmt(verification['ema_tolerance']['rtol'])}, atol {_fmt(verification['ema_tolerance']['atol'])}）"),
        ("Query Backboneが学習前後で変化", "query_backbone_updated",
         f"変更 {_fmt(verification.get('query_backbone_changed_tensors'))} / "
         f"{_fmt(verification.get('query_backbone_tensors'))} tensors"),
        ("Query Projectorが学習前後で変化", "query_projector_updated",
         f"変更 {_fmt(verification.get('query_projector_changed_tensors'))} / "
         f"{_fmt(verification.get('query_projector_tensors'))} tensors"),
        ("Queue pointer / 実key数 / random残数", "queue_pointer_and_counts",
         f"観測 {_fmt(final.get('queue_ptr'))} / {_fmt(final.get('real_key_count'))} / "
         f"{_fmt(final.get('random_remaining_count'))}、期待 {_fmt(expected.get('queue_ptr'))} / "
         f"{_fmt(expected.get('real_key_count'))} / {_fmt(expected.get('random_remaining_count'))}"),
        ("Queue全列unit norm・finite", "queue_unit_norm_finite",
         f"finite {_fmt(final.get('finite'))}, 列norm max abs error {_fmt(final.get('column_norm_max_abs_error'))}"),
        ("sample indexがsampler順と一致・target全0（ラベル未使用）", "data_order_and_zero_targets",
         f"順序一致 {_fmt(verification.get('data_order_matches_all_steps'))}, "
         f"target全0 {_fmt(verification.get('targets_all_zero_all_steps'))}"),
        ("checkpoint保存→別インスタンス復元", "checkpoint_roundtrip",
         f"state一致 {_fmt(roundtrip.get('state_equal'))}, optimizer一致 {_fmt(roundtrip.get('optimizer_equal'))}, "
         f"eval feature max abs diff {_fmt(roundtrip.get('eval_feature_max_abs_diff'))}"),
    ]


def render_verification_summary(report: dict) -> str:
    """Markdown evidence generated only from report values."""
    checks, history, provenance = report["checks"], report["history"], report["provenance"]
    dataset, moco, optimizer, model = report["dataset"], report["moco"], report["optimizer"], report["model"]
    git, reference, gates = provenance["git"], report["reference"], gate_statuses(report)
    replaced_step = moco["queue_size"] // optimizer["batch_size"]
    rows = _check_rows(report)
    not_run = [(label, checks[name]["reason"]) for label, name, _ in rows if checks[name]["status"] == NOT_RUN]
    failure = report.get("failure")

    lines = [
        "# Standalone MoCo v2型 CIFAR-10 検証サマリー",
        "",
        "> このファイルは `scripts/smoke/smoke_moco_cifar10.py` が、同じrunの `report.json` から自動生成した。"
        "判定はすべてこのrunの測定値から計算しており、固定のPASSは含まない。",
        "",
        f"- run status: **{report['status']}** / run kind: `{report['run_kind']}`",
    ]
    if failure:
        lines += [f"- failure: step {_fmt(failure.get('step'))}（{failure['stage']}）で `{failure['type']}`: "
                  f"{failure['message']}"]
    lines += [
        "",
        "## 実行条件",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| dataset | {dataset['name']} {dataset['split']}（root `{dataset['root']}`、"
        f"{_fmt(dataset.get('sample_count'))} 枚、download {_fmt(dataset['download'])}） |",
        f"| data order | {dataset['order_policy']}（seed {_fmt(dataset['order_seed'])}、"
        f"消費 {_fmt(dataset.get('samples_consumed'))} 枚） |",
        f"| backbone | {model['backbone']} |",
        f"| projector | {model['projector']} |",
        f"| MoCo | K {_fmt(moco['queue_size'])}, m {_fmt(moco['momentum'])}, T {_fmt(moco['temperature'])}, "
        f"{moco['queue_init']}、{moco['negatives']} |",
        f"| optimizer | SGD lr {_fmt(optimizer['learning_rate'])}, momentum {_fmt(optimizer['momentum'])}, "
        f"weight decay {_fmt(optimizer['weight_decay'])}, nesterov {_fmt(optimizer['nesterov'])}, "
        f"scheduler `{optimizer['scheduler']}` |",
        f"| steps / batch | {_fmt(optimizer['completed_steps'])} / {_fmt(optimizer['max_steps'])} steps、"
        f"batch {_fmt(optimizer['batch_size'])} |",
        f"| seed / device | {_fmt(report['seed'])} / `{report['device']}` |",
        f"| resume | {report['resume']['resumed_from'] or 'なし（fresh run）'} |",
        f"| Git | `{git['branch']}` HEAD `{git['commit']}` / dirty: "
        f"{_fmt(git['dirty']) if git['dirty'] is not None else 'unknown'} |",
        f"| environment | Python {provenance['python']}, PyTorch {provenance['torch']}, "
        f"torchvision {provenance['torchvision']}, CUDA {provenance['cuda'] or 'n/a'} |",
        f"| 公式参照 | `{reference['repository']}@{reference['commit']}`（{', '.join(reference['files'])}） |",
        "",
        "## Gate別判定",
        "",
        "| Gate | 判定 | 範囲 |",
        "| --- | --- | --- |",
        *[f"| {gate} | {item['status']} | {item['note']} |" for gate, item in gates.items()],
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

    shown = _history_rows(history, replaced_step)
    lines += [
        "## stepごとの記録",
        "",
        "`n/a` は値が存在しないことを示し、0の代用ではない。"
        + ("" if len(shown) == len(history) else f"表は抜粋（全{len(history)} stepは `report.json` の `history`）。"),
        "",
        "| step | lr | loss | pos cos | mean neg cos | hardest neg cos | backbone grad | projector grad | "
        "backbone update | min ‖z_q‖ | min q var | real / random keys | ptr |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {item['global_step']} | {_fmt(item.get('lr'))} | {_fmt(item.get('loss'))} | "
          f"{_fmt(item.get('positive_cosine_mean'))} | {_fmt(item.get('negative_cosine_mean'))} | "
          f"{_fmt(item.get('hardest_negative_cosine_mean'))} | {_fmt(item.get('query_backbone_grad_norm'))} | "
          f"{_fmt(item.get('query_projector_grad_norm'))} | {_fmt(item.get('query_backbone_update_norm'))} | "
          f"{_fmt(item.get('z_q_norm_min'))} | {_fmt(item.get('q_variance_min'))} | "
          f"{_fmt(item.get('real_key_count'))} / {_fmt(item.get('random_remaining_count'))} | "
          f"{_fmt(item.get('queue_ptr'))} |" for item in shown],
        "",
        "![loss](loss.png)",
        "",
        "![similarity](similarity.png)",
        "",
        "![gradient and update norms](grad_update.png)",
        "",
        "![queue](queue.png)",
        "",
        "## 診断値（記述統計のみ・自動判定なし）",
        "",
        "勾配急増・正規化前norm縮小・variance崩壊の判断に使う値。閾値による自動判定は行わず、"
        "原因が説明できない兆候は研究的なPASSとしない。",
        "",
        "| 系列 | first | last | min | max（step） | median | max / median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in DIAGNOSTIC_KEYS:
        stats = report["diagnostics"].get(key)
        if stats is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {label} | {_fmt(stats['first'])} | {_fmt(stats['last'])} | {_fmt(stats['min'])} | "
                         f"{_fmt(stats['max'])}（{stats['max_step']}） | {_fmt(stats['median'])} | "
                         f"{_fmt(stats['max_over_median'])} |")
    lines += [
        "",
        "## 公式MoCo v2との差分",
        "",
        *[f"- {item}" for item in OFFICIAL_DIFFERENCES],
        "",
        "## 結論（限定的）",
        "",
        "- 本runは「MoCo mechanics / CIFAR-10 short controlled learning」の工学的検証である。"
        "NaN/Infなし・更新経路の成立は、Lossの単調減少、CIFAR-10線形評価精度、表現品質、時間情報の獲得を示さない。",
        "- 20stepと500stepは学習率履歴が異なるため、loss曲線を無条件に数値比較しない。",
        "- Linear Probe・ImageNet精度・DDP公式再現はこのrunの対象外。",
        "",
        "## Artifacts",
        "",
        *[f"- [`{name}`]({name}): {description}"
          + ("" if report["artifacts"].get(name, True) else "（このrunでは作成されていない）")
          for name, description in ARTIFACTS],
        "",
        "数値の元データは同じディレクトリの `report.json`（`verification`, `checks`, `history`, `diagnostics`）。",
        "",
    ]
    return "\n".join(lines)


def write_verification_artifacts(output_dir: Path, report: dict) -> None:
    replaced_step = report["moco"]["queue_size"] // report["optimizer"]["batch_size"]
    save_loss_plot(report["history"], output_dir / "loss.png", replaced_step)
    save_similarity_plot(report["history"], output_dir / "similarity.png", replaced_step)
    save_grad_update_plot(report["history"], output_dir / "grad_update.png")
    save_queue_plot(report["history"], output_dir / "queue.png", replaced_step)
    (output_dir / "verification_summary.md").write_text(render_verification_summary(report), encoding="utf-8")
