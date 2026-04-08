# GradLab

GradLab is an OpenEnv-style benchmark for ML and deep-learning research workflows. The agent acts as an ML researcher diagnosing failed neural-network experiments from realistic logs, metrics, dataset signals, architecture notes, gradient statistics, and robustness slices.

The environment is deterministic and lightweight. It does not train models live during evaluation; instead it uses curated experiment fixtures and programmatic graders so that runs are reproducible under the hackathon constraints of 2 vCPU, 8 GB RAM, and less than 20 minutes for inference.

## Motivation

Many ML agents are evaluated on coding or office workflows, but research work often involves diagnosing training dynamics, choosing targeted repairs, and proving that a fix actually addresses the failure mode. GradLab turns those research tasks into a step-by-step environment with partial rewards.

## Tasks

| Task | Difficulty | Goal |
| --- | --- | --- |
| `overfit_rescue` | Easy | Diagnose overfitting from train/validation curves and recommend regularization plus validation checks. |
| `noisy_label_curation` | Medium | Diagnose label noise and class-level data quality issues before attempting architecture changes. |
| `unstable_robustness` | Hard | Diagnose unstable transformer training and distribution-shift brittleness using gradient, activation, architecture, and robustness evidence. |

Each task has a deterministic grader with scores in `[0.0, 1.0]`.

## Action Space

Agents submit a JSON action:

```json
{
  "kind": "inspect|diagnose|repair|evaluate|finish",
  "target": "evidence or intervention target",
  "value": "proposed fix, test, or final recommendation",
  "rationale": "short reason grounded in evidence"
}
```

Good trajectories inspect evidence, diagnose the root cause, apply targeted repairs, run a validation or robustness check, and then finish.

## Observation Space

Each observation includes:

- task id, name, difficulty, and objective
- current step and max steps
- initial symptoms
- currently visible evidence
- available action types
- progress summary
- `last_action_error` when the previous action was repeated, unsupported, or harmful

The code also defines a typed `GradLabReward` Pydantic model for grader details: step reward value, current normalized score, and reward reason.

## Reward and Scoring

Rewards provide partial progress:

- positive reward for inspecting useful evidence
- positive reward for correct diagnosis
- positive reward for targeted repair actions
- positive reward for validation, ablation, seed, per-class, or robustness checks
- penalties for repeated actions, unknown evidence targets, harmful fixes, or unsupported conclusions

Final score is deterministic and normalized to `[0.0, 1.0]`.

## Required Environment Variables

`inference.py` reads:

- `API_BASE_URL`, default `https://router.huggingface.co/v1`
- `MODEL_NAME`, default `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN`, required API key

Optional:

- `GRADLAB_TASK`, default `overfit_rescue`
- `GRADLAB_MAX_STEPS`, default `8`
- `GRADLAB_SUCCESS_THRESHOLD`, default `0.70`

## Run Baseline Inference

```bash
export HF_TOKEN=your_token_here
python inference.py
```

The script emits exactly the required structured logs:

```text
[START] task=overfit_rescue env=gradlab model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.12 done=false error=null
[END] success=true steps=6 score=0.850 rewards=0.12,0.22,0.10,0.10,0.16,0.00
```

## Docker

```bash
docker build -t gradlab-openenv .
docker run --rm -p 7860:7860 gradlab-openenv
```

The server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`

The provided validator pings `POST /reset` and expects HTTP 200.

## Local Tests

```bash
python -m unittest discover -s tests
```

If available, also run:

```bash
openenv validate
```

## Expected Baseline Behavior

The LLM baseline should reach a non-zero score by inspecting evidence and proposing a diagnosis, repair, and evaluation plan. Strong models should exceed `0.70` on the easy task and improve on the medium/hard tasks by grounding their actions in evidence.
