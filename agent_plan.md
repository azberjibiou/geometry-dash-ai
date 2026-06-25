# Geometry Dash RL Practice Agent Plan

## 0. Purpose

This document tracks the agent work separately from the broader Geometry Dash
AI roadmap in `plan.md`.

The main project direction is a **level-specific practicing RL agent** that
plays through a stochastic human mock model. The policy chooses intended
actions or intended press/release events. Those intended outputs are never sent
directly to Geometry Dash; they pass through the human model first.

Main loop:

```text
observation
-> RL policy
-> intended action/event
-> human mock model
-> executed input
-> Geometry Dash
-> trace, death, progress, reward
-> RL update / practice memory
```

Imitation learning is optional support only. It can bootstrap a policy,
diagnose event decoding, or provide a baseline, but it does not define the
agent architecture.

---

## 1. Scope

### In Scope

- Repeated attempts on one small local/offline level.
- RL policy outputs as intended button states or intended press/release events.
- Human model wrapper for perception delay, motor delay, timing variance,
  miss/drop probability, input constraints, and correlated error.
- Intended-event logging and executed-input logging.
- Trace capture, death/progress extraction, reward computation, and practice
  summaries.
- Level-specific practice memory that stores timing windows, section stats, and
  confidence instead of only exact macros.
- Optional imitation bootstrap or diagnostic baselines.

### Out of Scope

- Online levels.
- Leaderboard submission or leaderboard-oriented behavior.
- Zero-shot generalization to unseen levels.
- Perfect bot behavior.
- Treating predicted macros as the final agent.
- Large open-ended RL before the local/offline practice loop is reliable.

---

## 2. Current State

Already implemented in the repo:

```text
Phase 1: Python-only human model
Phase 2: trace and macro format
Phase 3: minimal Geometry Dash bridge
Phase 4: deterministic queued replay
Phase 5: screenshot/frame capture
Phase 6: scripted policy + human wrapper end-to-end
```

Support infrastructure already present:

```text
gd_human_model/
  events.py
  profile.py
  observation_buffer.py
  motor_noise.py
  macro_humanizer.py
  humanized_agent.py

gd_trace/
  trace_schema.py
  macro_schema.py
  replay_check.py
  humanized_run.py
  compare_trace.py

gd_env/
  protocol.py
  client.py
  dummy_env.py

scripts/
  run_geode_replay_check.py
  run_humanized_geode_macro.py
  capture_geode_frames.py
```

Optional imitation support already present:

```text
gd_imitation/dataset.py
gd_imitation/image_dataset.py
gd_imitation/baseline.py
gd_imitation/decoder.py
scripts/prepare_imitation_dataset.py
scripts/train_imitation_baseline.py
scripts/predict_imitation_macro.py
```

Verified historical status:

```text
pytest: 107 passed
PyTorch: installed with CUDA support
GPU: NVIDIA GeForce RTX 2060, 6 GB VRAM
```

Latest Phase A implementation status:

```text
Implemented:
  gd_rl/
    actions.py
      - IntendedAction and conversion to canonical Event objects

    policy.py
      - PracticePolicy base contract
      - PracticeContext
      - NoInputPolicy
      - ScriptedEventPolicy
      - RandomEventPolicy

    rewards.py
      - TraceOutcome
      - RewardConfig
      - deterministic reward computation from trace rows and event counts

    results.py
      - AttemptResult
      - PracticeRunSummary aggregation

    runner.py
      - PracticeRunner
      - PracticeAttemptExecutor protocol
      - per-attempt artifact persistence:
        policy_intended_events.json
        human_executed_events.json
        humanization_details.json
        trace.jsonl
        summary.json

    synthetic.py
      - SyntheticTraceExecutor for non-Geode tests and smoke checks

Verification:
  pytest: 113 passed

Current limitation:
  The Phase A runner uses an executor protocol. Synthetic execution is covered
  by tests; a live Geode executor or script wiring should be added next for
  local/offline level runs.
```

The imitation smoke artifact remains useful as a diagnostic fixture, but the
next main task is not to improve imitation. The next main task is the RL
practice environment and reward loop.

---

## 3. Agent Architecture

Runtime pipeline:

```text
Geode observation/frame stream
    |
    v
Observation buffer
    |
    v
RL policy / practice learner
    |
    | intended action or intended press/release event
    v
Level memory / practice adapter
    |
    | intended event
    v
Human mock model
    |
    | executed input event or dropped event
    v
Geometry Dash bridge
    |
    v
Trace, death, progress, clear status
    |
    v
Reward computer and practice update
```

Attempt-level training pipeline:

```text
reset level
-> run policy through human wrapper until death/clear/timeout
-> save trace
-> save intended events
-> save executed events
-> compute rewards
-> update policy and/or level memory
-> start next attempt
```

The important boundary:

```text
Policy output != game input

Policy output is intent.
Human wrapper output is executed input.
```

---

## 4. Agent Contract

Initial policy contract:

```python
class PracticePolicy:
    def reset(self, level_id: str, attempt_index: int) -> None:
        ...

    def act(self, observation) -> "IntendedAction":
        ...

    def update(self, attempt_result: "AttemptResult") -> None:
        ...
```

Initial intended action options:

```text
no_op
press
release
```

Alternative later action spaces:

```text
intended button state per tick
short event timing offset
section/window adjustment
macro-fragment choice
```

The first version may use compact observations:

```text
progress or x-position
current input_down
mode/gravity if available
recent internal-state rows
optional frame stack
```

Screenshot-only RL is not required for the first practice loop.

---

## 5. Human Model Boundary

The human model should be the mandatory wrapper between policy and game.

It should apply:

- visual/perception delay through delayed observations,
- motor delay before intended events execute,
- press/release timing variance,
- interval-dependent timing variance for dense or long-gap clicks,
- correlated timing error,
- miss/drop probability,
- button-state constraints and impossible-event handling.

Every attempt should log:

```text
policy_intended_events.json
human_executed_events.json
humanization_details.json
trace.jsonl
summary.json
```

This separation lets evaluation answer:

```text
Did the policy intend the right thing?
Did the human wrapper execute it late, early, or not at all?
Did the resulting trace improve reward or progress?
```

---

## 6. Reward and Attempt Results

`AttemptResult` should include:

```text
level_id
attempt_index
human_profile
seed
trace_path
intended_events_path
executed_events_path
final_percent
best_percent
death_tick
death_percent
cleared
total_reward
reward_terms
metadata
```

Initial reward terms:

```text
progress_delta
best_progress_bonus
section_survival_bonus
clear_bonus
death_penalty
illegal_or_excessive_input_penalty
```

Reward computation should be deterministic from the attempt trace and metadata.
Changing the human profile should change performance through execution noise,
not through hidden reward changes.

---

## 7. Development Phases

### Phase A - Practice Env Skeleton

Goal:

Create the repeated-attempt wrapper without training a neural policy yet.

Tasks:

1. Add a small `gd_rl/` or equivalent package.
2. Define `IntendedAction`, `AttemptResult`, and `PracticeRunSummary`.
3. Add reward computation from trace rows and terminal state.
4. Add a runner that can execute a scripted or random policy for many attempts.
5. Persist intended events, executed events, traces, and summaries under
   `artifacts/`.

Success criterion:

```text
A scripted/random policy can make repeated local/offline attempts through the
human wrapper and produce reward/progress summaries.
```

### Phase B - Reward Sanity Baselines

Goal:

Verify that reward and logging behave sensibly before training.

Tasks:

1. Run no-input, single-jump, and simple scripted policies.
2. Compare final percent, deaths, rewards, and executed-input distributions.
3. Sweep at least two HumanProfiles.
4. Confirm larger noise generally hurts timing-sensitive fixtures.

Success criterion:

```text
Reward ranking matches obvious behavior on small local/offline fixtures.
```

### Phase C - Minimal RL Practice Agent

Goal:

Train the first policy that improves on one fixed local/offline level.

Candidate algorithms:

```text
random search over event timings
cross-entropy method over event windows
small policy-gradient spike
section/window bandit
```

Use the simplest algorithm that produces a measurable learning curve. The
first policy does not need to be visually sophisticated.

Success criterion:

```text
Repeated attempts improve average progress, best progress, or reward under the
same HumanProfile compared with the initial policy.
```

### Phase D - Level Practice Memory

Goal:

Make successful sections persist as level-specific memory.

Tasks:

1. Store event windows with confidence.
2. Store death histograms and section stats.
3. Update windows from successful clears and near-miss/death context.
4. Combine policy output with memory at runtime.
5. Evaluate whether memory improves post-practice consistency under human
   noise.

Success criterion:

```text
The agent improves on the same level through practice without reducing to a
single exact replay macro.
```

### Phase E - Optional Imitation Bootstrap

Goal:

Use existing imitation tools only when they help RL practice.

Allowed uses:

- initialize event windows,
- pretrain a policy head,
- compare learned windows against a demonstration,
- test event decoders.

Success criterion:

```text
Imitation improves startup or diagnostics while all final evaluation still
runs through RL practice attempts and the human wrapper.
```

---

## 8. Evaluation Metrics

Attempt metrics:

```text
final_percent
best_percent
death_tick
death_percent
cleared
total_reward
reward_terms
intended_event_count
executed_event_count
dropped_event_count
timing_delta_distribution
```

Practice metrics:

```text
attempts_to_first_clear
playtime_to_first_clear_seconds
best_percent_by_attempt
average_progress_by_window
reward_curve
death_histogram_over_time
section_success_rate_over_time
post_practice_clear_rate
memory_confidence_by_section
```

The key early question:

```text
Does repeated practice on the same local/offline level improve progress or
reward under the same stochastic human profile?
```

---

## 9. Data Rules

All training and evaluation should use:

```text
local/offline levels only
artifacts/ for generated traces, frames, datasets, predictions, and summaries
no leaderboard submission
no generated artifacts in Git
```

Small synthetic fixtures in `tests/` are allowed when they make reward,
logging, or update behavior testable.

---

## 10. Near-Term Task List

Immediate next task:

```text
Implement Phase A: practice environment skeleton and reward computation.
```

Recommended order:

1. Add `gd_rl/` package scaffolding.
2. Define attempt/result dataclasses.
3. Implement reward computation from trace rows.
4. Add a repeated-attempt runner using a scripted or random policy.
5. Route intended events through `gd_human_model` before replay/application.
6. Save intended events, executed events, trace paths, rewards, and summaries.
7. Add focused tests for reward computation and summary aggregation.

---

## 11. Prompt For Next Agent Work

```text
We are in the geometry_dash_ai repo. Continue from agent_plan.md.

Main objective:
Implement the Phase A practice environment skeleton for a level-specific RL
practice agent under a stochastic human mock model.

Current state:
- Phases 1 through 6 are implemented.
- Humanized macro execution, replay checks, trace capture, and optional
  imitation support tools exist.
- pytest previously passed: 107 passed.
- artifacts/ is ignored and must remain local-only.

Task:
Add the first RL-practice infrastructure:
- `gd_rl/` package or equivalent
- intended action/event representation
- attempt result and practice summary structures
- reward computation from trace/death/progress/clear data
- repeated-attempt runner for a scripted or random policy
- logging that separates intended policy outputs from executed humanized inputs

Constraints:
- RL policy output is intent, not direct game input.
- All intended outputs must pass through the human mock model before reaching
  Geometry Dash.
- Imitation components are optional support tools only.
- Use one small local/offline level.
- Do not focus on zero-shot generalization, online levels, leaderboard use, or
  perfect bot behavior.
```
