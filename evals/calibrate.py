"""Calibration metrics computed directly (no external library).

Inputs are parallel lists: ``probs`` is per-sample distributions over options, ``labels``
is the per-sample correct option.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _max_conf(probs: list[dict[str, float]]) -> list[float]:
    return [max(p.values()) for p in probs]


def _predictions(probs: list[dict[str, float]]) -> list[str]:
    return [max(p, key=lambda k: p[k]) for p in probs]


def ece(probs: list[dict[str, float]], labels: list[str], n_bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins."""
    n = len(labels)
    if n == 0:
        return 0.0
    confs = _max_conf(probs)
    preds = _predictions(probs)
    correct = [1.0 if p == label else 0.0 for p, label in zip(preds, labels, strict=True)]

    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_cnt = [0] * n_bins
    for conf, corr in zip(confs, correct, strict=True):
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_conf[idx] += conf
        bin_acc[idx] += corr
        bin_cnt[idx] += 1

    total = 0.0
    for i in range(n_bins):
        if bin_cnt[i] > 0:
            total += (bin_cnt[i] / n) * abs(bin_conf[i] / bin_cnt[i] - bin_acc[i] / bin_cnt[i])
    return total


def brier(
    probs: list[dict[str, float]], labels: list[str], options: list[str] | None = None
) -> float:
    """Multiclass Brier score (mean over samples of summed squared error)."""
    n = len(labels)
    if n == 0:
        return 0.0
    if options is None:
        options = sorted({opt for p in probs for opt in p} | set(labels))
    total = 0.0
    for probs_, label in zip(probs, labels, strict=True):
        for option in options:
            total += (probs_.get(option, 0.0) - (1.0 if option == label else 0.0)) ** 2
    return total / n


def reliability(
    probs: list[dict[str, float]], labels: list[str], n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """Reliability diagram rows: (mean confidence, accuracy, count) for non-empty bins."""
    confs = _max_conf(probs)
    preds = _predictions(probs)
    correct = [1.0 if p == label else 0.0 for p, label in zip(preds, labels, strict=True)]

    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_cnt = [0] * n_bins
    for conf, corr in zip(confs, correct, strict=True):
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_conf[idx] += conf
        bin_acc[idx] += corr
        bin_cnt[idx] += 1

    rows: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        if bin_cnt[i] > 0:
            rows.append((bin_conf[i] / bin_cnt[i], bin_acc[i] / bin_cnt[i], bin_cnt[i]))
    return rows


def recommend_threshold(
    probs: list[dict[str, float]], labels: list[str], target_failure: float = 0.05
) -> float | None:
    """Lowest confidence threshold whose auto-applied tail has error <= target_failure."""
    confs = _max_conf(probs)
    preds = _predictions(probs)
    correct = [1.0 if p == label else 0.0 for p, label in zip(preds, labels, strict=True)]

    for threshold in sorted(set(confs)):
        selected = [i for i in range(len(confs)) if confs[i] >= threshold]
        if not selected:
            continue
        errors = sum(1 for i in selected if correct[i] == 0.0)
        if errors / len(selected) <= target_failure:
            return threshold
    return None


def render_reliability(rows: list[tuple[float, float, int]]) -> str:
    lines = ["confidence  accuracy  count", "-" * 28]
    for conf, acc, count in rows:
        lines.append(f"{conf:>10.2f}  {acc:>8.2f}  {count:>5d}")
    return "\n".join(lines)


def _probs_from_named_scores(named_scores: dict[str, float]) -> dict[str, float]:
    return {k[2:]: v for k, v in named_scores.items() if k.startswith("p_")}


def load_promptfoo_results(path: str | Path) -> dict[str, dict[str, list[Any]]]:
    """Read a promptfoo ``-o results.json`` into per-provider probs/labels."""
    data = json.loads(Path(path).read_text())
    results = data["results"]["results"]
    by_provider: dict[str, dict[str, list[Any]]] = {}
    for result in results:
        provider = result.get("provider") or {}
        provider_id = provider.get("id", "unknown") if isinstance(provider, dict) else "unknown"
        probs = _probs_from_named_scores(
            (result.get("response") or {}).get("namedScores", {})
        )
        label = (result.get("vars") or {}).get("category")
        if not probs or label is None:
            continue
        entry = by_provider.setdefault(provider_id, {"probs": [], "labels": []})
        entry["probs"].append(probs)
        entry["labels"].append(label)
    return by_provider


def main(results_path: str, out_path: str | None = None, target_failure: float = 0.05) -> str:
    by_provider = load_promptfoo_results(results_path)
    lines: list[str] = []
    for provider, data in sorted(by_provider.items()):
        probs = data["probs"]
        labels = data["labels"]
        options = sorted({opt for p in probs for opt in p} | set(labels))
        threshold = recommend_threshold(probs, labels, target_failure=target_failure)
        threshold_text = "none" if threshold is None else f"{threshold:.2f}"
        lines.append(f"== {provider} (n={len(labels)}) ==")
        lines.append(
            f"ECE={ece(probs, labels):.4f}  Brier={brier(probs, labels, options):.4f}  "
            f"threshold({target_failure:.0%} silent failure)={threshold_text}"
        )
        lines.append(render_reliability(reliability(probs, labels)))
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if out_path:
        Path(out_path).write_text(report)
    return report


if __name__ == "__main__":
    import sys

    results = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    main(results, out)

