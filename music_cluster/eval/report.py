"""Reports: one JSON per run, a Markdown summary, and a diff against a baseline.

The JSON is the record. Its ``headline`` numbers per suite are what a baseline
is compared on; everything else is detail for whoever is digging into why.
"""

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build(
    suites: Dict[str, Dict[str, Any]], args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "meta": {
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "version": __version__,
            "commit": _git_commit(),
            "python": platform.python_version(),
            "args": args or {},
        },
        "suites": suites,
    }


def save(report: Dict[str, Any], path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return target


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def compare(
    current: Dict[str, Any], baseline: Dict[str, Any], tolerance: float = 0.02
) -> List[Dict[str, Any]]:
    """Every baseline headline number, with whether this run regressed it.

    A number regresses when it moves the wrong way by more than ``tolerance``
    (absolute), or when a suite that was run no longer reports it at all.
    Suites list their lower-is-better numbers in ``lower_is_better``. Suites
    that were not run this time are left out.
    """
    rows: List[Dict[str, Any]] = []
    current_suites = current.get("suites", {})
    for suite, before in baseline.get("suites", {}).items():
        if not isinstance(before, dict) or not before.get("headline"):
            continue
        data = current_suites.get(suite)
        if data is None:
            # Not run this time: nothing to compare, nothing regressed.
            continue
        lower = set(data.get("lower_is_better", before.get("lower_is_better", [])))
        headline = data.get("headline") or {}
        for name, old in before["headline"].items():
            if not isinstance(old, (int, float)):
                continue
            value = headline.get(name)
            if not isinstance(value, (int, float)):
                # The baseline had a number and this run has none: the suite
                # was skipped, crashed into a skip, or lost the metric. That
                # must fail the gate, not pass it silently.
                rows.append(
                    {
                        "suite": suite,
                        "metric": name,
                        "baseline": float(old),
                        "current": None,
                        "delta": None,
                        "regressed": True,
                    }
                )
                continue
            delta = float(value) - float(old)
            worse = -delta if name not in lower else delta
            rows.append(
                {
                    "suite": suite,
                    "metric": name,
                    "baseline": float(old),
                    "current": float(value),
                    "delta": delta,
                    "regressed": worse > tolerance,
                }
            )
    return rows


def markdown(report: Dict[str, Any], comparison: Optional[List[Dict[str, Any]]] = None) -> str:
    meta = report.get("meta", {})
    lines = [
        "# music-cluster evaluation",
        "",
        f"{meta.get('created', '')} · version {meta.get('version', '?')} · commit {meta.get('commit') or '?'}",
        "",
    ]
    for suite, data in report.get("suites", {}).items():
        lines += [f"## {suite}", "", "| Metric | Value |", "| --- | --- |"]
        for name, value in data.get("headline", {}).items():
            lines.append(f"| {name} | {_fmt(value)} |")
        lines.append("")
        methods = data.get("methods")
        if isinstance(methods, dict) and methods:
            columns = [
                c
                for c in (
                    "top1",
                    "top3",
                    "macro_f1",
                    "ece",
                    "aurc",
                    "weighted",
                    "correct",
                    "accuracy1",
                    "accuracy2",
                )
                if any(isinstance(m, dict) and c in m for m in methods.values())
            ]
            if columns:
                lines += [
                    "| Method | " + " | ".join(columns) + " |",
                    "| --- | " + " | ".join("---" for _ in columns) + " |",
                ]
                for name, values in methods.items():
                    lines.append(
                        f"| {name} | " + " | ".join(_fmt(values.get(c)) for c in columns) + " |"
                    )
                lines.append("")
        points = (data.get("sorter") or {}).get("operating_points")
        if points:
            lines += ["Auto-file operating points (novel tracks count as errors):", ""]
            current = data["sorter"]["current_thresholds"]
            lines.append(
                f"- current (confidence ≥ {current['auto_accept_confidence']}, margin ≥ "
                f"{current['min_margin']}): precision {_fmt(current['precision'])}, "
                f"coverage {_fmt(current['coverage'])}"
            )
            for name, row in points.items():
                if row:
                    lines.append(
                        f"- {name.replace('_', ' ')}: confidence ≥ {row['confidence']}, margin ≥ "
                        f"{row['margin']} → precision {_fmt(row['precision'])}, coverage {_fmt(row['coverage'])}"
                    )
                else:
                    lines.append(f"- {name.replace('_', ' ')}: not reachable")
            lines.append("")
        misses = data.get("beat_misses")
        if misses:
            # Printed in full so a CI log shows them; the JSON report is not
            # always reachable from where the log is read.
            lines += [
                "Wrong beat types (bars: one digit per sixteenth, 0 silent to 9 loudest):",
                "",
                "| Track | Genre | BPM (annotated) | Beat | Kick | Snare | Top scores |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for miss in misses:
                scores = ", ".join(f"{k} {v:.2f}" for k, v in miss.get("scores", {}).items())
                lines.append(
                    f"| {miss['track']} | {miss['genre']} | {_fmt(miss.get('bpm'))} "
                    f"({_fmt(miss.get('annotated_bpm'))}) | {miss['beat']} | "
                    f"`{miss.get('kick', '-')}` | `{miss.get('snare', '-')}` | {scores} |"
                )
            lines.append("")
    if comparison:
        lines += [
            "## Against baseline",
            "",
            "| Suite | Metric | Baseline | Current | Change |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in comparison:
            flag = " ⚠" if row["regressed"] else ""
            if row["current"] is None:
                lines.append(
                    f"| {row['suite']} | {row['metric']} | {row['baseline']:.3f} | missing | ⚠ |"
                )
                continue
            lines.append(
                f"| {row['suite']} | {row['metric']} | {row['baseline']:.3f} | {row['current']:.3f} "
                f"| {row['delta']:+.3f}{flag} |"
            )
        lines.append("")
    return "\n".join(lines)
