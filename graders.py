"""Explicit per-task graders for GradLab."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)


def _history_matches(history: List[Dict[str, Any]], keywords: List[str]) -> int:
    text = " ".join(
        str(item.get("action", {})).lower()
        for item in history
    )
    return sum(1 for keyword in keywords if keyword.lower() in text)


def grade_overfit_rescue(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state.get("history", [])
    inspected = state.get("progress", {}).get("inspected", [])
    score = 0.0
    score += min(len(inspected), 3) * 0.10
    score += min(_history_matches(history, ["overfit", "generalization gap", "dropout", "weight decay", "augmentation", "validation"]), 4) * 0.10
    return {
        "task_id": "overfit_rescue",
        "grader": "deterministic_rule_grader",
        "score": _bounded(score),
        "reward_range": [0.0, 1.0],
    }


def grade_noisy_label_curation(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state.get("history", [])
    inspected = state.get("progress", {}).get("inspected", [])
    score = 0.0
    score += min(len(inspected), 4) * 0.08
    score += min(_history_matches(history, ["label noise", "annotation", "relabel", "audit", "macro f1", "per class"]), 5) * 0.10
    return {
        "task_id": "noisy_label_curation",
        "grader": "deterministic_rule_grader",
        "score": _bounded(score),
        "reward_range": [0.0, 1.0],
    }


def grade_unstable_robustness(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state.get("history", [])
    inspected = state.get("progress", {}).get("inspected", [])
    score = 0.0
    score += min(len(inspected), 5) * 0.07
    score += min(_history_matches(history, ["normalization", "gradient clipping", "shifted validation", "robustness", "seeds", "pre norm"]), 6) * 0.09
    return {
        "task_id": "unstable_robustness",
        "grader": "deterministic_rule_grader",
        "score": _bounded(score),
        "reward_range": [0.0, 1.0],
    }


GRADER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "overfit_rescue": grade_overfit_rescue,
    "noisy_label_curation": grade_noisy_label_curation,
    "unstable_robustness": grade_unstable_robustness,
}

