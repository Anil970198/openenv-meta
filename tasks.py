"""Explicit task catalog for GradLab."""

from __future__ import annotations

from typing import Any, Dict, List

from gradlab_env import TASKS


def list_tasks() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for task in TASKS.values():
        items.append(
            {
                "id": task.task_id,
                "name": task.name,
                "difficulty": task.difficulty,
                "objective": task.objective,
                "max_steps": task.max_steps,
                "grader": {
                    "name": task.grader_name,
                    "type": "deterministic",
                    "score_range": list(task.score_range),
                    "reward_range": list(task.reward_range),
                },
            }
        )
    return items


def get_task(task_id: str) -> Dict[str, Any]:
    for task in list_tasks():
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)
