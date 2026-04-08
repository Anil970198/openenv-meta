"""GradLab: a lightweight OpenEnv-style environment for ML debugging tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field


ActionKind = Literal["inspect", "diagnose", "repair", "evaluate", "finish"]


class GradLabAction(BaseModel):
    """Action submitted by an agent."""

    kind: ActionKind = Field(..., description="One of inspect, diagnose, repair, evaluate, finish.")
    target: str = Field("", description="Diagnostic target, fix target, evaluation target, or final label.")
    value: str = Field("", description="Optional proposed fix, test, or final recommendation.")
    rationale: str = Field("", description="Short reason grounded in the observed evidence.")


class GradLabObservation(BaseModel):
    """Observation returned to the agent."""

    task_id: str
    task_name: str
    difficulty: str
    objective: str
    step: int
    max_steps: int
    symptoms: List[str]
    visible_evidence: Dict[str, Any]
    available_actions: List[str]
    progress: Dict[str, Any]
    last_action_error: Optional[str] = None


class GradLabReward(BaseModel):
    """Typed reward details for graders and documentation."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Step reward clipped to [-1.0, 1.0].")
    score: float = Field(..., ge=0.0, le=1.0, description="Current normalized task score.")
    reason: str = Field("", description="Short deterministic grader reason for the reward.")


class GradLabStepResult(BaseModel):
    """Result object returned by reset and step."""

    observation: GradLabObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    difficulty: str
    objective: str
    symptoms: List[str]
    hidden_evidence: Dict[str, Any]
    inspect_targets: Dict[str, str]
    diagnosis_keywords: Set[str]
    repair_keywords: Set[str]
    evaluation_keywords: Set[str]
    trap_keywords: Set[str]
    max_steps: int = 8


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("_", " ").replace("-", " ").split())


def _contains_any(text: str, keywords: Set[str]) -> bool:
    normalized = _norm(text)
    return any(keyword in normalized for keyword in keywords)


TASKS: Dict[str, TaskSpec] = {
    "overfit_rescue": TaskSpec(
        task_id="overfit_rescue",
        name="Overfit Rescue",
        difficulty="easy",
        objective=(
            "Diagnose why a small CNN has high train accuracy but poor validation accuracy, "
            "then recommend a repair plan and a validation check."
        ),
        symptoms=[
            "train_accuracy climbs from 0.62 to 0.99 by epoch 20",
            "validation_accuracy peaks at 0.74 around epoch 6, then falls to 0.58",
            "train_loss keeps dropping while validation_loss rises after epoch 7",
            "the dataset has only 420 labeled images across four classes",
        ],
        hidden_evidence={
            "curves": "Generalization gap widens from 0.04 at epoch 4 to 0.41 at epoch 20.",
            "config": "The model has no dropout, no weight decay, and only horizontal flips as augmentation.",
            "split": "Train and validation splits are disjoint and class-balanced; no leakage is detected.",
        },
        inspect_targets={
            "curves": "Inspect train/validation curves and quantify the generalization gap.",
            "config": "Inspect regularization, augmentation, optimizer, and scheduler settings.",
            "split": "Check whether validation failure is caused by leakage or an invalid split.",
        },
        diagnosis_keywords={"overfit", "generalization gap", "regularization"},
        repair_keywords={"dropout", "weight decay", "augmentation", "early stopping"},
        evaluation_keywords={"validation", "holdout", "ablation", "learning curve"},
        trap_keywords={"increase capacity", "bigger model", "more epochs", "train longer"},
        max_steps=7,
    ),
    "noisy_label_curation": TaskSpec(
        task_id="noisy_label_curation",
        name="Noisy Label Dataset Curation",
        difficulty="medium",
        objective=(
            "Diagnose a vision classifier whose errors cluster around two classes, then choose "
            "data-centric fixes before changing the architecture."
        ),
        symptoms=[
            "macro_f1 is 0.61 while micro_f1 is 0.83",
            "class 'crack' is often predicted as 'scratch'",
            "70 percent of high-confidence errors come from one annotation batch",
            "loss remains high for a small repeated subset of samples",
        ],
        hidden_evidence={
            "confusion_matrix": "Most false negatives are crack->scratch and dent->background.",
            "hard_examples": "18 of the top 25 loss samples have ambiguous or visibly wrong labels.",
            "metadata": "The suspicious labels come from annotator_batch=B17 and low-light captures.",
            "baseline": "A larger backbone improves micro_f1 by only 0.01 and does not fix macro_f1.",
        },
        inspect_targets={
            "confusion_matrix": "Inspect class-level confusion rather than only aggregate accuracy.",
            "hard_examples": "Inspect persistent high-loss samples for mislabeled or ambiguous examples.",
            "metadata": "Inspect annotator/source metadata for correlated label noise.",
            "baseline": "Compare against an architecture-only baseline.",
        },
        diagnosis_keywords={"label noise", "noisy label", "annotation", "class imbalance", "data quality"},
        repair_keywords={"relabel", "remove", "audit", "targeted augmentation", "rebalance"},
        evaluation_keywords={"macro f1", "held out", "clean validation", "per class"},
        trap_keywords={"bigger backbone", "deeper model", "blind architecture", "more layers"},
        max_steps=8,
    ),
    "unstable_robustness": TaskSpec(
        task_id="unstable_robustness",
        name="Unstable Architecture and Robustness Failure",
        difficulty="hard",
        objective=(
            "Diagnose unstable transformer training with distribution-shift brittleness, then "
            "sequence architecture, optimizer, and robustness checks."
        ),
        symptoms=[
            "validation_loss spikes every time the cosine scheduler restarts",
            "gradient_norm jumps above 240 before loss becomes NaN in 3 of 5 seeds",
            "activation_stats show saturated GELU blocks in layers 7-9",
            "clean validation accuracy is 0.79 but shifted validation accuracy is 0.43",
            "removing layer normalization from residual blocks was the latest architecture change",
        ],
        hidden_evidence={
            "gradients": "Gradient clipping was disabled; spikes start in attention block 8.",
            "activations": "Layers 7-9 have high saturation and poor variance after the residual merge.",
            "architecture": "The candidate architecture removed pre-norm and widened the MLP by 4x.",
            "robustness": "Shifted validation contains blur, contrast drop, and rare-class enrichment.",
            "seeds": "Instability reproduces on 3 of 5 seeds, so this is not a single unlucky run.",
        },
        inspect_targets={
            "gradients": "Inspect gradient norms and the first layer/block where instability appears.",
            "activations": "Inspect activation saturation and variance by block.",
            "architecture": "Inspect recent architecture changes around normalization and residual paths.",
            "robustness": "Inspect shifted validation slices and perturbation-specific failures.",
            "seeds": "Inspect seed variance to distinguish stochastic noise from structural instability.",
        },
        diagnosis_keywords={"unstable", "normalization", "gradient", "distribution shift", "robustness"},
        repair_keywords={"pre norm", "layer norm", "gradient clipping", "lower learning rate", "scheduler", "residual"},
        evaluation_keywords={"shifted validation", "robustness", "seed", "ablation", "slice"},
        trap_keywords={"remove normalization", "raise learning rate", "ignore shift", "single seed"},
        max_steps=8,
    ),
}


class GradLabEnv:
    """Deterministic environment with OpenEnv-compatible reset/step/state methods."""

    def __init__(self, task_id: str = "overfit_rescue"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id={task_id!r}. Available tasks: {', '.join(TASKS)}")
        self.task = TASKS[task_id]
        self._step = 0
        self._done = False
        self._visible_evidence: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._seen_actions: Set[str] = set()
        self._progress: Dict[str, Any] = {}
        self._last_action_error: Optional[str] = None
        self.reset()

    @classmethod
    def task_ids(cls) -> List[str]:
        return list(TASKS.keys())

    def reset(self) -> GradLabStepResult:
        self._step = 0
        self._done = False
        self._visible_evidence = {"initial_symptoms": list(self.task.symptoms)}
        self._history = []
        self._seen_actions = set()
        self._progress = {
            "inspected": [],
            "diagnosis_hit": False,
            "repairs": [],
            "evaluation_hit": False,
            "final_score": 0.0,
            "penalties": 0.0,
        }
        self._last_action_error = None
        return GradLabStepResult(observation=self._observation(), reward=0.0, done=False, info={"task_id": self.task.task_id})

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "step": self._step,
            "done": self._done,
            "visible_evidence": dict(self._visible_evidence),
            "progress": self._copy_progress(),
            "history": list(self._history),
            "last_action_error": self._last_action_error,
        }

    def close(self) -> None:
        self._done = True

    def step(self, action: GradLabAction) -> GradLabStepResult:
        if self._done:
            self._last_action_error = "episode already completed"
            return GradLabStepResult(observation=self._observation(), reward=0.0, done=True, info=self._info())

        self._step += 1
        self._last_action_error = None
        reward = 0.0
        action_key = f"{action.kind}:{_norm(action.target)}:{_norm(action.value)}"
        combined = _norm(f"{action.target} {action.value} {action.rationale}")

        if action_key in self._seen_actions:
            reward -= 0.08
            self._progress["penalties"] += 0.08
            self._last_action_error = "repeated action"
        else:
            self._seen_actions.add(action_key)

        if _contains_any(combined, self.task.trap_keywords):
            reward -= 0.18
            self._progress["penalties"] += 0.18
            self._last_action_error = "unsupported or harmful intervention"

        if action.kind == "inspect":
            reward += self._handle_inspect(action)
        elif action.kind == "diagnose":
            reward += self._handle_diagnose(combined)
        elif action.kind == "repair":
            reward += self._handle_repair(combined)
        elif action.kind == "evaluate":
            reward += self._handle_evaluate(combined)
        elif action.kind == "finish":
            final_score = self.score()
            reward += max(0.0, final_score - sum(item["reward"] for item in self._history if item["reward"] > 0) * 0.10)
            self._progress["final_score"] = final_score
            self._done = True

        if self._step >= self.task.max_steps and not self._done:
            self._done = True
            self._progress["final_score"] = self.score()

        reward = max(-1.0, min(1.0, reward))
        self._history.append(
            {
                "step": self._step,
                "action": action.model_dump() if hasattr(action, "model_dump") else action.dict(),
                "reward": reward,
                "score": self.score(),
                "error": self._last_action_error,
            }
        )
        return GradLabStepResult(observation=self._observation(), reward=reward, done=self._done, info=self._info())

    def score(self) -> float:
        inspected = len(self._progress["inspected"])
        evidence_credit = min(inspected / max(2, len(self.task.inspect_targets)), 1.0) * 0.15
        diagnosis_credit = 0.25 if self._progress["diagnosis_hit"] else 0.0
        repair_credit = min(len(self._progress["repairs"]) / 3.0, 1.0) * 0.35
        evaluation_credit = 0.15 if self._progress["evaluation_hit"] else 0.0
        rationale_credit = 0.10 if self._has_grounded_rationale() else 0.0
        raw = evidence_credit + diagnosis_credit + repair_credit + evaluation_credit + rationale_credit
        score = raw - min(self._progress["penalties"], 0.35)
        return round(max(0.0, min(1.0, score)), 3)

    def _handle_inspect(self, action: GradLabAction) -> float:
        target = _norm(action.target)
        for key, description in self.task.inspect_targets.items():
            if _norm(key) in target or target in _norm(description):
                if key not in self._visible_evidence:
                    self._visible_evidence[key] = self.task.hidden_evidence[key]
                    self._progress["inspected"].append(key)
                    return 0.12
                if self._last_action_error is None:
                    self._last_action_error = "evidence already inspected"
                self._progress["penalties"] += 0.04
                return -0.04
        if self._last_action_error is None:
            self._last_action_error = "unknown evidence target"
        self._progress["penalties"] += 0.06
        return -0.06

    def _handle_diagnose(self, combined: str) -> float:
        if _contains_any(combined, self.task.diagnosis_keywords):
            first_hit = not self._progress["diagnosis_hit"]
            self._progress["diagnosis_hit"] = True
            return 0.22 if first_hit else 0.04
        if self._last_action_error is None:
            self._last_action_error = "diagnosis not supported by task evidence"
        self._progress["penalties"] += 0.10
        return -0.10

    def _handle_repair(self, combined: str) -> float:
        matched = {keyword for keyword in self.task.repair_keywords if keyword in combined}
        new_matches = [keyword for keyword in matched if keyword not in self._progress["repairs"]]
        if new_matches:
            self._progress["repairs"].extend(sorted(new_matches))
            return min(0.10 * len(new_matches), 0.24)
        if self._last_action_error is None:
            self._last_action_error = "repair does not address the known failure mode"
        self._progress["penalties"] += 0.08
        return -0.08

    def _handle_evaluate(self, combined: str) -> float:
        if _contains_any(combined, self.task.evaluation_keywords):
            first_hit = not self._progress["evaluation_hit"]
            self._progress["evaluation_hit"] = True
            return 0.16 if first_hit else 0.03
        if self._last_action_error is None:
            self._last_action_error = "evaluation plan is not targeted to the failure mode"
        self._progress["penalties"] += 0.07
        return -0.07

    def _has_grounded_rationale(self) -> bool:
        for entry in self._history:
            action = entry["action"]
            rationale = _norm(action.get("rationale", ""))
            if len(rationale) >= 24 and any(_norm(key) in rationale for key in self._visible_evidence):
                return True
        return False

    def _copy_progress(self) -> Dict[str, Any]:
        return {
            "inspected": list(self._progress["inspected"]),
            "diagnosis_hit": bool(self._progress["diagnosis_hit"]),
            "repairs": list(self._progress["repairs"]),
            "evaluation_hit": bool(self._progress["evaluation_hit"]),
            "final_score": float(self._progress["final_score"]),
            "penalties": round(float(self._progress["penalties"]), 3),
        }

    def _observation(self) -> GradLabObservation:
        return GradLabObservation(
            task_id=self.task.task_id,
            task_name=self.task.name,
            difficulty=self.task.difficulty,
            objective=self.task.objective,
            step=self._step,
            max_steps=self.task.max_steps,
            symptoms=list(self.task.symptoms),
            visible_evidence=dict(self._visible_evidence),
            available_actions=[
                "inspect(target)",
                "diagnose(target, rationale)",
                "repair(target, value, rationale)",
                "evaluate(target, value, rationale)",
                "finish(value, rationale)",
            ],
            progress=self._copy_progress(),
            last_action_error=self._last_action_error,
        )

    def _info(self) -> Dict[str, Any]:
        return {
            "score": self.score(),
            "task_id": self.task.task_id,
            "last_action_error": self._last_action_error,
            "history_length": len(self._history),
        }


def make_env(task_id: str = "overfit_rescue") -> GradLabEnv:
    return GradLabEnv(task_id=task_id)


def run_scripted_baseline(task_id: str) -> Tuple[float, List[float]]:
    """Deterministic local baseline used by tests and README examples."""

    env = make_env(task_id)
    rewards: List[float] = []
    for target in list(TASKS[task_id].inspect_targets)[:2]:
        result = env.step(GradLabAction(kind="inspect", target=target, rationale=f"Check {target} evidence."))
        rewards.append(result.reward)
    diagnosis = {
        "overfit_rescue": "overfit generalization gap regularization",
        "noisy_label_curation": "label noise annotation data quality",
        "unstable_robustness": "unstable normalization gradient distribution shift",
    }[task_id]
    result = env.step(GradLabAction(kind="diagnose", target="root cause", value=diagnosis))
    rewards.append(result.reward)
    for keyword in list(TASKS[task_id].repair_keywords)[:3]:
        result = env.step(GradLabAction(kind="repair", target=keyword, value=f"apply {keyword}"))
        rewards.append(result.reward)
    result = env.step(GradLabAction(kind="evaluate", target="validation", value="run targeted held out validation ablation"))
    rewards.append(result.reward)
    result = env.step(GradLabAction(kind="finish", value="final recommendation"))
    rewards.append(result.reward)
    return env.score(), rewards
