"""Small HTTP server for Hugging Face Space and validator pings."""

from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI

from gradlab_env import GradLabAction, GradLabEnv, TASKS, make_env


app = FastAPI(title="GradLab OpenEnv", version="1.0.0")
_sessions: Dict[str, GradLabEnv] = {}


def _dump_model(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _session_id(payload: dict) -> str:
    return str(payload.get("session_id") or "default")


@app.get("/")
def index() -> dict:
    return {
        "name": "GradLab",
        "description": "OpenEnv-style benchmark for diagnosing neural network training failures.",
        "tasks": GradLabEnv.task_catalog(),
        "reset": "/reset",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "gradlab"}


@app.post("/reset")
def reset(payload: dict | None = None) -> dict:
    payload = payload or {}
    task_id = payload.get("task_id") or os.getenv("GRADLAB_TASK", "overfit_rescue")
    env = make_env(task_id)
    _sessions[_session_id(payload)] = env
    result = env.reset()
    return _dump_model(result)


@app.post("/step")
def step(payload: dict) -> dict:
    sid = _session_id(payload)
    env = _sessions.get(sid)
    if env is None:
        env = make_env(payload.get("task_id") or os.getenv("GRADLAB_TASK", "overfit_rescue"))
        _sessions[sid] = env
    action_payload = payload.get("action", payload)
    result = env.step(GradLabAction(**action_payload))
    return _dump_model(result)


@app.get("/state")
def state(session_id: str = "default") -> dict:
    env = _sessions.get(session_id)
    if env is None:
        env = make_env(os.getenv("GRADLAB_TASK", "overfit_rescue"))
        _sessions[session_id] = env
    return env.state()


@app.get("/tasks")
def tasks() -> dict:
    return {
        "tasks": GradLabEnv.task_catalog(),
        "count": len(TASKS),
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "env_name": "gradlab",
        "task_count": len(TASKS),
        "tasks": GradLabEnv.task_catalog(),
        "action_schema": {
            "kind": "inspect|diagnose|repair|evaluate|finish",
            "target": "string",
            "value": "string",
            "rationale": "string",
        },
        "observation_fields": [
            "task_id",
            "task_name",
            "difficulty",
            "objective",
            "step",
            "max_steps",
            "symptoms",
            "visible_evidence",
            "available_actions",
            "progress",
            "last_action_error",
        ],
        "score_range": [0.0, 1.0],
        "reward_range": [-1.0, 1.0],
    }


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
