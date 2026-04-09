---
title: GradLab OpenEnv
emoji: "🧪"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# GradLab

GradLab is an OpenEnv environment for a kind of work that shows up all the time in machine learning research but is rarely captured well in agent benchmarks: looking at a failed training run, figuring out what actually went wrong, and choosing the next intervention for the right reason.

Most agent environments either reduce ML work to a static question-answer task or hide the research process behind a final metric. That misses the part that matters. In practice, an ML researcher does not jump straight from "loss looks bad" to "change the learning rate." They inspect curves, compare slices, look for data problems, check whether an issue is architectural or optimization-related, and only then decide what to try next. GradLab turns that process into a stepwise environment with explicit state, partial rewards, deterministic grading, and a clear finish condition.

The environment is intentionally lightweight. It does not train a real model during evaluation. Instead, it uses curated experiment fixtures that look and feel like real debugging cases: train/validation curves, confusion patterns, sample-quality clues, activation and gradient symptoms, and robustness failures. That design keeps the benchmark reproducible and makes it practical under the hackathon constraints.

## Why This Environment

The core idea behind GradLab is that ML research work is not a single decision; it is a sequence of evidence-driven choices.

We wanted the environment to reward the right behavior, not just the right final label. A good agent should:

- inspect relevant evidence before making a claim
- identify the failure mode instead of guessing generic fixes
- choose repairs that actually match the diagnosis
- propose evaluation steps that would confirm or falsify the intervention
- avoid harmful or lazy behavior such as repeating actions, escalating capacity without evidence, or ignoring robustness and data quality signals

That is why the environment is structured around five action families: `inspect`, `diagnose`, `repair`, `evaluate`, and `finish`.

## Tasks

GradLab ships with three tasks that follow an easy → medium → hard progression and cover distinct failure modes.

| Task | Difficulty | What the agent must do |
| --- | --- | --- |
| `overfit_rescue` | Easy | Read train/validation symptoms, diagnose overfitting, propose regularization and evaluation changes. |
| `noisy_label_curation` | Medium | Recognize that the bottleneck is data quality, not architecture size, and choose curation-oriented fixes. |
| `unstable_robustness` | Hard | Diagnose training instability plus shift brittleness using architecture, gradient, activation, and robustness evidence. |

### 1. Overfit Rescue

This task starts with a familiar vision-training pattern: training accuracy climbs, validation performance peaks early, and the generalization gap keeps widening. The agent is expected to inspect the curves and training configuration, identify overfitting, and recommend appropriate interventions such as dropout, weight decay, augmentation, and better validation checks.

This task is deliberately straightforward. It confirms that the agent can read standard optimization symptoms and avoid irrelevant fixes like simply training longer or making the model bigger.

### 2. Noisy Label Dataset Curation

This task is about recognizing when the real bottleneck lives in the dataset rather than the model. The observations point toward concentrated annotation issues, class-specific confusion, and a subset of persistently bad examples. A strong agent should notice that a larger backbone does not meaningfully solve the problem and instead choose actions like auditing labels, relabeling suspect samples, and applying targeted data improvements.

This is a more realistic benchmark for data-centric ML work than a generic "clean this dataset" task because the agent has to justify why data curation matters more than model scaling in this case.

### 3. Unstable Architecture and Robustness Failure

The hard task combines two things that often appear together in frontier training work: optimization instability and poor behavior under distribution shift. The agent sees exploding gradients, saturated activations, risky architecture changes around normalization, and a gap between clean validation and shifted validation. The right solution is not one isolated fix but a coherent plan that addresses normalization, gradient control, and targeted robustness evaluation.

This task is meant to feel closer to actual model-debugging work than a simple classifier diagnosis benchmark.

## Environment Design

### Action Space

Agents act with a typed JSON object:

```json
{
  "kind": "inspect|diagnose|repair|evaluate|finish",
  "target": "evidence target, failure mode, or intervention target",
  "value": "proposed fix, test, or final recommendation",
  "rationale": "short justification grounded in evidence"
}
```

The action space is intentionally narrow. We did not want the benchmark to become a free-form essay grader. The agent has to choose a category of action and then ground it in the current state of the environment.

### Observation Space

Each observation includes:

- task id, name, difficulty, and objective
- current step and step budget
- visible symptoms
- visible evidence collected so far
- available actions
- a compact progress summary
- `last_action_error` for repeated, unsupported, or harmful actions

The environment uses typed Pydantic models for observations, actions, and reward details.

### State Management

GradLab implements the OpenEnv-style workflow through:

- `reset()` to initialize a fresh task instance
- `step(action)` to apply one action and return the next observation, reward, done flag, and info
- `state()` to expose the current environment state

Each reset produces clean state. Evidence is revealed only when the agent explicitly inspects the relevant target. Repeated actions are tracked and penalized.

## Reward Design

The reward function is shaped around research behavior rather than just task completion.

The agent gets credit for:

- revealing useful evidence through inspection
- naming the correct failure mode
- applying repairs that match that diagnosis
- proposing a relevant evaluation or validation check

The agent loses credit for:

- repeating the same action
- selecting evidence targets that do not exist
- proposing harmful or unsupported interventions
- making architecture or optimization changes that contradict the observed evidence

The final score is normalized to `[0.0, 1.0]`. Because the environment keeps partial rewards and penalties, the benchmark distinguishes between an agent that stumbles into a decent answer and one that follows a good diagnostic process.

## How This Submission Matches the Hackathon Requirements

### Real-world task simulation

The environment models real ML research and engineering work: diagnosing failed experiments, distinguishing optimization issues from data problems, reasoning about architecture changes, and planning robustness evaluation. These are common tasks in modern model development and evaluation.

### OpenEnv specification compliance

The submission includes:

- typed Pydantic models for actions, observations, and reward details
- environment logic with `reset()`, `step()`, and `state()`
- `openenv.yaml`
- `pyproject.toml`
- `uv.lock`

The project passes:

```bash
openenv validate
```

### Minimum of three tasks with agent graders

GradLab provides three tasks with clear objectives and a deterministic scoring function. The tasks are ordered by difficulty and reward the right sequence of diagnostic actions rather than a single terminal guess.

### Meaningful reward function

Rewards are not sparse. The agent receives feedback throughout the trajectory for evidence gathering, diagnosis quality, repair selection, and evaluation planning.

### Baseline inference script

The root-level `inference.py` uses the OpenAI client exactly as required and reads:

- `API_BASE_URL` with a default
- `MODEL_NAME` with a default
- `HF_TOKEN` without a default

It emits the required structured logs:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

### Deployment on Hugging Face Spaces

The environment is packaged as a Docker Space and exposed through a FastAPI app. The deployment serves the endpoints expected by the validator, including `POST /reset`.

### Containerized execution

The repository includes a working `Dockerfile`, and the image is designed to stay within CPU-only constraints. The environment itself is lightweight and deterministic, so it does not depend on GPU hardware or long-running training jobs.

### Documentation

This README documents the environment’s motivation, task design, action and observation spaces, reward design, setup instructions, and validation workflow.

## Required Environment Variables

`inference.py` reads:

- `API_BASE_URL`, default `https://router.huggingface.co/v1`
- `MODEL_NAME`, default `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN`, required

Optional:

- `GRADLAB_TASK`, optional task id; leave unset to run all three tasks
- `GRADLAB_MAX_STEPS`, default `8`
- `GRADLAB_SUCCESS_THRESHOLD`, default `0.70`

## Running the Baseline

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Example output:

```text
[START] task=overfit_rescue env=gradlab model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.12 done=false error=null
[STEP] step=2 action={...} reward=0.22 done=false error=null
[END] success=true steps=7 score=0.850 rewards=0.12,0.12,0.12,0.22,0.10,0.10,0.10
```

If `GRADLAB_TASK` is not set, the baseline runs all three tasks in sequence and emits one `[START]`/`[END]` block per task.

## Running the Server

### Local Python

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t gradlab-openenv .
docker run --rm -p 7860:7860 gradlab-openenv
```

Useful endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /health`

## Local Validation

Unit tests:

```bash
python -m unittest discover -s tests
```

OpenEnv validation:

```bash
openenv validate
```

## Final Note

GradLab is intentionally small enough to evaluate reliably, but the structure is meant to be extensible. More tasks can be added around hyperparameter rescue, ablation planning, interpretability-assisted debugging, and representation failure analysis without changing the basic interaction model. That makes it a practical hackathon submission today and a reasonable base for a broader research benchmark later.
