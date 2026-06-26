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

Latest live Geode wiring status:

```text
Implemented:
  gd_rl/geode_executor.py
    - GeodeExecutorConfig
    - GeodePracticeExecutor implementing the PracticeAttemptExecutor protocol
    - live queued macro execution:
      human_executed_events.json
      -> client.load_macro(...)
      -> reset_attempt(...)
      -> run_loaded_macro(...)
      -> trace rows returned to PracticeRunner
    - start/progress guards for wrong-level or stale-attempt protection
    - geode_diagnostics.json per attempt

  scripts/run_rl_practice_geode.py
    - live Phase A CLI wiring
    - policies:
      no-input
      scripted macro
      random event schedule
    - HumanProfile selection from built-ins or profile JSON
    - reward weight options
    - Geode bridge settings and local/offline guard options

Verification:
  python scripts/run_rl_practice_geode.py --help
  pytest: 121 passed

Usage sketch:
  python scripts/run_rl_practice_geode.py \
    --level-id tiny_local_fixture \
    --policy scripted \
    --macro-json examples/macros/single_jump.json \
    --attempts 5 \
    --profile Advanced \
    --stop-on-success \
    --require-start-percent-max 2 \
    --require-start-x-max 50

Current limitation:
  The wiring is unit-tested with a fake Geode client, but the live script still
  needs a manual local/offline Geometry Dash smoke run with the Geode bridge
  active.
```

Live smoke update:

```text
Manual local/offline Geode bridge run succeeded.

Scripted single-jump smoke:
  command:
    python scripts/run_rl_practice_geode.py
      --level-id manual_live_smoke
      --policy scripted
      --macro-json examples/macros/single_jump.json
      --attempts 3
      --profile Advanced
      --max-observations 400
      --require-start-percent-max 2
      --require-start-x-max 50

  artifact:
    artifacts/rl_practice_geode_20260626_002734

  result:
    attempts: 3
    clears: 0
    deaths: 3
    death_tick: 206 in all attempts
    final/best percent: about 21.616 in all attempts
    intended_event_count: 2
    executed_event_count: 2
    dropped_event_count: 0
    attempt 1 executed macro events: press 20, release 28
    attempt 1 observed input transitions: press 20, release 28

No-input baseline smoke:
  artifact:
    artifacts/rl_practice_geode_20260626_002746

  result:
    attempts: 3
    clears: 0
    deaths: 3
    death_tick: 206 in all attempts
    final/best percent: about 21.616 in all attempts
    intended_event_count: 0
    executed_event_count: 0
    observed input transitions: none

Interpretation:
  Live bridge wiring is working. Policy-intended events, humanized executed
  events, Geode input transitions, traces, rewards, and summaries are all being
  persisted through the Phase A runner.

  The current manual level appears not to benefit from the single_jump fixture,
  or the click is irrelevant/too early for that level. The next validation
  should use a known tiny local fixture where no-input and scripted macros
  produce different progress or clear outcomes.
```

Live reset-after-clear bug note:

```text
Observed issue:
  Restarting immediately after a clear can make the Geometry Dash progress bar
  stop updating correctly.

Mitigation implemented:
  PracticeRunConfig.stop_after_first_clear
  scripts/run_rl_practice_geode.py --stop-after-first-clear

Related hardening:
  GeodeExecutorConfig.start_guard_reset_retries
  GeodeExecutorConfig.start_guard_retry_delay_seconds

Verification:
  pytest: 122 passed

Current recommendation:
  For live local/offline smoke tests that may clear the level, run with:
    --stop-on-success
    --stop-after-first-clear

  Do not run repeated post-clear attempts automatically until the Geode/GD
  reset-after-clear progress bar bug is fixed or fully understood.
```

Successful triple-spike live smoke:

```text
Command:
  python scripts/run_rl_practice_geode.py
    --level-id manual_triple_spike_smoke
    --policy scripted
    --macro-json examples/macros/triple_spike_jump.json
    --attempts 5
    --profile Advanced
    --max-observations 1400
    --stop-on-success
    --stop-after-first-clear
    --post-terminal-delay-seconds 0.5
    --require-start-percent-max 2
    --require-start-x-max 50

Artifact:
  artifacts/rl_practice_geode_20260626_003101

Result:
  attempts configured: 5
  attempts run: 1
  stopped after first clear: yes
  clear: true
  final_percent: 100.0
  death_tick: none
  row_count: 625
  last_tick: 624
  playtime: 2.6 seconds
  total_reward: 252.5
  intended_event_count: 2
  executed_event_count: 2
  dropped_event_count: 0
  executed macro events: press 192, release 210
  observed input transitions: press 192, release 210

Interpretation:
  The live Phase A practice runner can now execute a scripted policy through
  the human model into Geode queued replay, clear the local/offline triple-spike
  fixture, persist trace/reward/artifacts, and avoid the known post-clear
  progress-bar reset bug by stopping after the first clear.
```

Post-clear delayed restart smoke:

```text
Command:
  python scripts/run_rl_practice_geode.py
    --level-id manual_triple_spike_restart_delay_5s
    --policy scripted
    --macro-json examples/macros/triple_spike_jump.json
    --attempts 3
    --profile Advanced
    --max-observations 1400
    --stop-on-success
    --post-terminal-delay-seconds 5
    --start-guard-reset-retries 3
    --start-guard-retry-delay-seconds 1
    --require-start-percent-max 2
    --require-start-x-max 50

Artifact:
  artifacts/rl_practice_geode_20260626_003156

Result:
  attempts run: 3
  clears: 3
  deaths: 0
  clear_rate: 1.0
  final_percent_by_attempt: [100.0, 100.0, 100.0]
  reset_attempts: 1 for every attempt
  start_percent: about 0.157 for every attempt
  last_tick: 634 for every attempt

Observed input transitions:
  attempt 1: press 192, release 210
  attempt 2: press 194, release 210
  attempt 3: press 193, release 212

Interpretation:
  A 5 second post-terminal delay allows the live loop to continue across clears
  without triggering the observed progress-bar/reset issue in this smoke run.
  Keep this delay for repeated live clear loops until the underlying Geode/GD
  reset-after-clear behavior is fixed or better characterized.
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

Documented live smoke procedure:

Prerequisites:

```text
Geometry Dash is open on the local/offline triple-spike fixture.
The Geode bridge is active and listening on 127.0.0.1:29430.
Do not use online levels or submit leaderboard runs.
Keep all generated outputs under artifacts/ and do not commit them.
```

Common live guard flags:

```text
--post-terminal-delay-seconds 5
--require-start-percent-max 2
--require-start-x-max 50
```

Run the four baseline policies:

```text
python scripts/run_rl_practice_geode.py \
  --level-id phase_b_triple_spike_no_input \
  --policy no-input \
  --attempts 3 \
  --profile Advanced \
  --max-observations 1400 \
  --stop-on-success \
  --post-terminal-delay-seconds 5 \
  --require-start-percent-max 2 \
  --require-start-x-max 50

python scripts/run_rl_practice_geode.py \
  --level-id phase_b_triple_spike_early_single_jump \
  --policy scripted \
  --macro-json examples/macros/single_jump.json \
  --attempts 3 \
  --profile Advanced \
  --max-observations 1400 \
  --stop-on-success \
  --post-terminal-delay-seconds 5 \
  --require-start-percent-max 2 \
  --require-start-x-max 50

python scripts/run_rl_practice_geode.py \
  --level-id phase_b_triple_spike_correct_scripted \
  --policy scripted \
  --macro-json examples/macros/triple_spike_jump.json \
  --attempts 3 \
  --profile Advanced \
  --max-observations 1400 \
  --stop-on-success \
  --post-terminal-delay-seconds 5 \
  --require-start-percent-max 2 \
  --require-start-x-max 50

python scripts/run_rl_practice_geode.py \
  --level-id phase_b_triple_spike_random \
  --policy random \
  --attempts 3 \
  --profile Advanced \
  --max-observations 1400 \
  --stop-on-success \
  --random-max-events 4 \
  --random-min-tick 120 \
  --random-max-tick 260 \
  --random-min-spacing 8 \
  --post-terminal-delay-seconds 5 \
  --require-start-percent-max 2 \
  --require-start-x-max 50
```

Compare the generated run summaries:

```text
python scripts/compare_practice_runs.py \
  artifacts/<no-input-run>/summary.json \
  artifacts/<single-jump-run>/summary.json \
  artifacts/<triple-spike-run>/summary.json \
  artifacts/<random-run>/summary.json
```

Optional profile sweep:

```text
for profile in TopPlayer Advanced Intermediate Beginner; do
  python scripts/run_rl_practice_geode.py \
    --level-id phase_b_triple_spike_profile_${profile} \
    --policy scripted \
    --macro-json examples/macros/triple_spike_jump.json \
    --attempts 3 \
    --profile ${profile} \
    --max-observations 1400 \
    --stop-on-success \
    --post-terminal-delay-seconds 5 \
    --require-start-percent-max 2 \
    --require-start-x-max 50
done
```

Expected sanity ordering:

```text
correct triple_spike_jump scripted policy should clear or score best.
no-input and early single_jump should die near the same early obstacle.
random should usually be noisy and worse than the correct scripted policy.
Higher-noise profiles should not outperform lower-noise profiles consistently
on this timing-sensitive fixture.
```

Phase B live baseline sweep update:

```text
Date:
  2026-06-26

Common flags:
  --post-terminal-delay-seconds 5
  --require-start-percent-max 2
  --require-start-x-max 50

Runs:
  no-input:
    artifact: artifacts/phase_b_20260626_003931_no_input
    attempts: 3
    clears: 0
    deaths: 3
    average_final_percent: 32.4409
    best_percent: 32.4409
    death_tick_histogram: {"206": 3}
    total_reward: 95.0531

  early single_jump:
    artifact: artifacts/phase_b_20260626_004001_single_jump
    attempts: 3
    clears: 0
    deaths: 3
    average_final_percent: 32.4409
    best_percent: 32.4409
    death_tick_histogram: {"206": 3}
    total_reward: 95.0531

  correct triple_spike_jump scripted:
    artifact: artifacts/phase_b_20260626_004024_triple_spike
    attempts: 3
    clears: 3
    deaths: 0
    clear_rate: 1.0
    average_final_percent: 100.0
    best_percent: 100.0
    total_reward: 657.0276

  random:
    artifact: artifacts/phase_b_20260626_004129_random_retry
    attempts: 3
    clears: 0
    deaths: 3
    average_final_percent: 35.9580
    best_percent: 36.5354
    death_tick_histogram: {"224": 1, "229": 1, "232": 1}
    total_reward: 108.7067

Interpretation:
  Reward ranking matches obvious behavior on the local/offline triple-spike
  fixture. Correct scripted input clears reliably and scores far above
  no-input, early single_jump, and random. Random sometimes survives slightly
  farther than no-input but does not clear.

Live note:
  The first random run immediately after a repeated-clear scripted run failed
  before summary persistence because Geode returned a trace with duplicate
  ticks. Waiting briefly and rerunning random with:
    --start-guard-reset-retries 3
    --start-guard-retry-delay-seconds 1
  succeeded. Keep the 5 second terminal delay and conservative reset retries
  for live post-clear sweeps.
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

Initial Phase C implementation status:

```text
Implemented:
  gd_rl/timing_search.py
    - TimingEventWindow and TimingSearchConfig
    - CEM-style candidate sampling and elite window updates
    - CandidateEvaluation, GenerationResult, and TimingSearchResult
    - JSON loading for event timing windows
    - candidate-level error capture for flaky live traces

  scripts/run_rl_timing_search_geode.py
    - live Geode timing-window search CLI
    - each sampled candidate is converted to intended Event objects
    - intended events still pass through PracticeRunner and the human model
    - candidate artifacts are written under artifacts/<search>/generation_*/candidate_*/
    - search_summary.json, final_windows.json, and best_intended_macro.json
      are persisted locally

  examples/timing_windows/
    - triple_spike_two_orbs_seed.json
      Broad 6-event search over triple spike plus two orb clicks.
    - triple_spike_two_orbs_from_memory.json
      Keeps the known Phase B triple-spike click fixed and searches the two
      newly added orb clicks.

Verification:
  pytest: 129 passed
```

Phase C live smoke on harder local/offline fixture:

```text
Level change:
  The local triple-spike fixture was extended with two jump orbs after the
  triple spike.

Probe:
  command:
    python scripts/run_rl_practice_geode.py
      --level-id phase_c_probe_triple_plus_orbs
      --policy scripted
      --macro-json examples/macros/triple_spike_jump.json
      --attempts 1
      --profile Advanced
      --max-observations 1800
      --stop-on-success
      --post-terminal-delay-seconds 5
      --start-guard-reset-retries 3
      --start-guard-retry-delay-seconds 1
      --require-start-percent-max 2
      --require-start-x-max 50

  artifact:
    artifacts/phase_c_20260626_005454_probe_triple_only

  result:
    clears: 0
    death_tick: 345
    best_percent: 54.3307
    total_reward: 78.0217

Broad 6-event CEM smoke:
  artifact:
    artifacts/phase_c_20260626_005800_timing_search_smoke

  result:
    generations: 2
    population_size: 4
    best_candidate: g000_c003
    best_percent: 51.515
    best_score: 73.548

  interpretation:
    The search machinery worked, but searching the already-solved triple-spike
    click together with the new orb clicks made the first section unstable.

Memory-seeded CEM smoke:
  command:
    python scripts/run_rl_timing_search_geode.py
      --level-id phase_c_triple_spike_two_orbs_memory_cem_smoke
      --window-json examples/timing_windows/triple_spike_two_orbs_from_memory.json
      --generations 2
      --population-size 6
      --elite-fraction 0.5
      --attempts-per-candidate 1
      --profile Advanced
      --max-observations 1800
      --post-terminal-delay-seconds 5
      --start-guard-reset-retries 3
      --start-guard-retry-delay-seconds 1
      --require-start-percent-max 2
      --require-start-x-max 50

  artifact:
    artifacts/phase_c_20260626_005925_timing_search_memory_smoke

  best candidate:
    candidate_id: g001_c005
    intended events:
      press 192
      release 212
      press 334
      release 354
      press 400
      release 417
    best_score: 89.9596
    best_percent in candidate evaluation: 61.6162

Best candidate recheck:
  artifact:
    artifacts/phase_c_20260626_010113_best_candidate_recheck

  result:
    attempts: 3
    clears: 0
    average_final_percent: 61.8687
    best_percent: 62.1212
    death_tick_histogram: {"488": 1, "490": 1, "492": 1}
    total_reward: 209.3485

Interpretation:
  Phase C produced a small but real learning-loop improvement on the harder
  local/offline fixture. The known triple-spike memory plus CEM over the new
  orb click windows improved from the triple-only probe at 54.3307% to a
  rechecked best of 62.1212%. It did not clear yet.

Phase C closure:
  CEM timing search was useful as a diagnostic only. It confirmed that reward,
  logging, artifacts, and the human-model boundary can support a learning loop,
  but it is not the intended long-term agent architecture. Do not continue
  optimizing this fixture with CEM unless needed for a narrow debugging
  baseline.

Next direction:
  Move from queued whole-macro candidates to a live observation-conditioned RL
  environment. The next agent should make per-step intended decisions from
  observations, pass those intentions through an online human model, apply the
  resulting executed inputs to the live local/offline level, and learn from
  incremental rewards and terminal outcomes.
```

### Phase D - Closed-Loop Practice Learner

Goal:

Move the live learner away from exact timing discovery and toward feedback
control. The policy should learn to correct continuously from delayed compact
observations while its intended actions pass through the stochastic human
model.

Core direction:

```text
This is not a "find the 192 tick jump" problem.

Because every policy output is filtered through visual delay, motor delay,
jitter, drops, and accumulated state error, exact tick targets are not stable
control primitives. This becomes most obvious in ship/flying sections, where
survival depends on continuously correcting y-position, velocity, and current
input state rather than replaying fixed press/release timestamps.
```

The next learner should therefore be a small closed-loop controller:

- policy output remains desired input state: idle/hold,
- desired state transitions are converted to intended press/release edges,
- intended edges always pass through the online human model,
- observations stay compact at first: percent/x/y/velocity/mode/gravity/input,
- recent intended state and/or action history should be available to the
  learner so it can reason about delayed execution,
- death feedback should localize failed state/action regions, not produce exact
  replacement click ticks.

Recommended algorithm direction:

```text
Keep the current REINFORCE path as a smoke scaffold.
Do not optimize it as the serious learner.

Implement the next learner as a minimal actor-critic/A2C-style loop before
considering PPO. The value head should help use death-local feedback and reduce
the very high variance of whole-episode REINFORCE.
```

Smallest meaningful experiment:

```text
Use a tiny local/offline fixture that rewards closed-loop correction.

Best target:
  a short ship corridor or flying-control level where survival requires
  maintaining a y-band under humanized input noise.

Fallback:
  a simple existing local/offline fixture can be used for smoke, but it should
  not be treated as proof that the closed-loop direction works unless feedback
  correction matters.
```

Tasks:

1. Define or select a small local/offline closed-loop control fixture,
   preferably an early ship corridor.
2. Keep no-input, random desired-state, and simple fixed-control policies as
   baselines.
3. Add a minimal actor-critic learner around `LivePracticeEnv`.
4. Keep action space as desired idle/hold and reuse `ButtonStateIntentAdapter`.
5. Add recent intended-state/action-history features, or prepare for a tiny
   recurrent state later.
6. Add death-local credit assignment over the preceding observation/action
   window. This may weight recent advantages or penalties, but it must not
   write exact tick corrections.
7. Add only generic regularization if needed, such as decision stride,
   action-repeat, entropy tuning, or input-rate penalties. Do not add
   level-specific cooldown scripts.
8. Evaluate with changed human seeds after training to check whether the policy
   is correcting feedback error rather than memorizing one noisy execution.

Success criterion:

```text
The agent improves survival, progress, or clear rate on the same local/offline
level under stochastic human execution without reducing to a single exact
replay macro or exact-tick timing table.
```

Failure modes to watch:

```text
The policy learns a fixed timestamp-like pattern that fails when the human seed
changes.

The policy collapses to rapid random toggling or permanently held input.

Memory or handcrafted logic starts overriding the policy with level-specific
press/release commands.

Learning only improves on jump timing fixtures and does not transfer to a
fixture where closed-loop correction is actually required.
```

### Phase E - State-Space Practice Memory

Goal:

Persist level-specific practice knowledge as state-space diagnostics and soft
guidance, not as exact event timings.

The memory should answer questions like:

```text
Where does the agent die in observation/state space?
Which y/y_vel/input-state regions are recoverable?
Which sections are stable under different human seeds?
Does the policy recover after humanized execution drifts from intended input?
```

Tasks:

1. Store death clusters by compact state features, section, mode, and human
   profile/seed.
2. Store survival bands or successful state distributions for closed-loop
   sections.
3. Track intended state, executed input, pending human events, and observed
   input transitions around death windows.
4. Keep memory passive at first: summaries, diagnostics, plots/tables, and
   optional learner features.
5. Only later consider soft policy priors from memory. Do not let memory issue
   direct press/release overrides.

Success criterion:

```text
Practice memory helps explain and eventually guide closed-loop improvement
without becoming a macro, timing-window table, or handcrafted controller.
```

### Phase F - Optional Imitation Bootstrap

Goal:

Use existing imitation tools only when they help RL practice.

Allowed uses:

- initialize observation encoders or policy heads,
- pretrain a policy head,
- compare learned closed-loop behavior against a demonstration,
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
desired_input_state_counts
emitted_intent_counts
observed_input_transition_count
pending_not_dispatched_count
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
death_state_cluster_over_time
survival_state_band_over_time
human_seed_eval_delta
closed_loop_recovery_rate
```

The key early question:

```text
Does repeated practice on the same local/offline level improve progress or
reward under the same stochastic human profile?
```

For Phase D, the sharper question is:

```text
When humanized execution creates timing and state error, does the policy use
new observations to correct its next intended button state, or is it merely
rediscovering a brittle fixed timing pattern?
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
Start Phase D: turn the live step learner into a closed-loop practice
controller rather than a timing-search or macro-replay system.
```

Recommended order:

1. Preserve `LivePracticeEnv` as the policy-intent-to-humanized-input boundary.
2. Keep the desired idle/hold input state action space introduced by
   `ButtonStateIntentAdapter`.
3. Add the smallest actor-critic/A2C-style learner that can use a value head
   and death-local feedback.
4. Include recent intended state/action history so the policy can reason about
   delayed observations and delayed execution.
5. Select or create a tiny local/offline closed-loop fixture, preferably a ship
   corridor, where fixed tick timing is not the right abstraction.
6. Compare against no-input, random desired-state, and simple fixed-control
   baselines.
7. Evaluate with changed human seeds to catch brittle timing memorization.
8. Keep state-space practice memory passive until closed-loop learning is
   demonstrated.

Live step environment implementation update:

```text
Implemented:
  gd_human_model/humanized_agent.py
    - added submit_intended_event_result(...) so online execution can log
      per-intent motor delay, jitter, miss/drop, and actual-event provenance.

  gd_rl/live_env.py
    - LivePracticeEnvConfig
    - LivePracticeObservation
    - LiveStepResult
    - LivePracticeEnv reset/step/save_attempt abstraction
    - per-step policy intent is stamped at the current live tick, passed
      through HumanizedAgent, and only due human-executed events are dispatched
      to the bridge with send_event(...)
    - compact bridge observations are used first: tick, percent, x/y,
      velocities, mode, gravity, input state, dead/completed flags
    - delayed policy observations are exposed through the human observation
      buffer while latest observations remain available for logging/alignment
    - episode artifacts are persisted without requiring a full intended macro
      up front:
        policy_intended_events.json
        human_executed_events.json
        humanization_details.json
        trace.jsonl
        geode_diagnostics.json
        summary.json

  gd_rl/__init__.py
    - exported live env types.

  tests/test_live_practice_env.py
    - fake-client coverage for reset/step, visual delay, motor delay,
      intent-vs-executed-input separation, terminal persistence, and start
      guards. Tests do not require Geometry Dash.

Geode bridge capability finding:
  The existing bridge already accepts live press/release through the protocol
  "action" message and applies those commands via the Geode mod. No protocol
  addition was needed for the first live step environment.

Verification:
  python -m pytest -q
  result: 132 passed

Next recommended implementation step:
  Add the smallest learner/driver around LivePracticeEnv, likely a compact
  observation encoder plus a tiny REINFORCE or A2C loop. Keep the first learner
  local/offline, short-running, and artifact-only; do not start a large training
  run yet.
```

Tiny live neural learner implementation update:

```text
Implemented:
  gd_rl/live_learner.py
    - compact observation encoder for LivePracticeObservation
      using the delayed policy-visible bridge observation
    - TinyLivePolicyNetwork, a small PyTorch MLP with action logits for:
        no_op, press, release
    - NeuralActionDecision for logging policy probabilities and intended
      actions
    - REINFORCE-style one-episode update:
        observation -> neural policy -> intended action
        -> LivePracticeEnv -> human model -> executed input
        -> reward/done -> discounted returns -> policy update
    - short-run training summary persistence via run_reinforce_training(...)

  tests/test_live_learner.py
    - encoder tests that do not require Geometry Dash
    - fake one-step live client where a press clears the synthetic attempt
    - verifies the neural policy update increases press probability after a
      rewarded press action

Notes:
  This is the first neural model in the live RL path. It is intentionally small
  and diagnostic: it proves the live step environment can feed a neural policy
  and receive a policy-gradient update without running a large training job.
  PyTorch remains loaded only when the live learner is constructed.

Next recommended implementation step:
  Add a guarded live CLI/smoke driver that can run a very small number of
  local/offline REINFORCE attempts through Geode, with conservative reset
  guards and no checkpoint/artifact commits.
```

Guarded live neural smoke driver implementation update:

```text
Implemented:
  scripts/run_live_rl_practice_geode.py
    - tiny live Geode REINFORCE smoke CLI
    - wires GeometryDashClient -> LivePracticeEnv -> TinyLivePolicyNetwork
      -> run_reinforce_training(...)
    - uses policy intent only; LivePracticeEnv still routes all press/release
      intent through the online HumanizedAgent before sending bridge actions
    - writes run output under artifacts/live_rl_practice_<timestamp> by default
      with training_summary.json plus per-attempt live env artifacts
    - defaults are intentionally small:
        attempts: 1
        max_steps: 600
        device: cpu
        no checkpoint writing
    - conservative live guard defaults:
        --post-terminal-delay-seconds 5
        --start-guard-reset-retries 3
        --start-guard-retry-delay-seconds 1
        --require-start-percent-max 2
        --require-start-x-max 50

  tests/test_run_live_rl_practice_geode.py
    - verifies smoke defaults and config wiring
    - runs the CLI main path with a fake one-step terminal bridge client, so
      Geometry Dash is not required
    - verifies training_summary.json and per-attempt summary persistence

Verification:
  python scripts/run_live_rl_practice_geode.py --help
  python -m pytest -q tests/test_live_practice_env.py tests/test_live_learner.py tests/test_run_live_rl_practice_geode.py
  python -m pytest -q
  result: 137 passed

Manual live smoke update:
  Date:
    2026-06-26

  Command:
    python scripts/run_live_rl_practice_geode.py
      --level-id manual_live_rl_smoke
      --attempts 1
      --max-steps 600
      --profile Advanced
      --post-terminal-delay-seconds 5
      --start-guard-reset-retries 3
      --start-guard-retry-delay-seconds 1
      --require-start-percent-max 2
      --require-start-x-max 50

  Artifact:
    artifacts/live_rl_practice_20260626_013351

  Result:
    attempts: 1
    step_count: 206
    cleared: false
    death_tick: 206
    final/best percent: 26.0430
    row_count: 207
    total_reward: 32.0423
    reset_attempts: 1
    policy action counts: no_op 73, press 48, release 85
    intended_event_count: 133
    executed_event_count: 70
    dropped_event_count: 61
    pending_not_dispatched_count: 2
    observed input transition count in trace: 34

  Interpretation:
    The first guarded neural live smoke successfully ran the tiny
    observation-conditioned REINFORCE loop through the real local/offline Geode
    bridge. The untrained policy was noisy and died early, which is expected
    for this smoke; the important validation is that neural intent, online
    humanization, executed bridge input, trace capture, reward, and summaries
    all completed end to end.

Next recommended implementation step:
  Add a small action regularization or stateful button-head constraint for the
  live neural policy before running multi-attempt training, because the
  untrained no_op/press/release sampler produces dense contradictory intents.
```

Desired input-state adapter implementation update:

```text
Implemented:
  gd_rl/live_learner.py
    - TinyLivePolicyNetwork now outputs desired input states:
        idle, hold
      instead of directly sampling no_op/press/release events.
    - Added ButtonStateIntentAdapter, which tracks intended input state and
      converts desired state transitions into edge intents:
        intended idle -> desired hold: press
        intended hold -> desired hold: no_op
        intended hold -> desired idle: release
        intended idle -> desired idle: no_op
    - The adapter is intentionally based on intended state, not executed bridge
      input state, so human visual/motor delay does not cause repeated press or
      release intents while a humanized event is pending.
    - ReinforceAttemptSummary now reports both:
        action_counts: desired idle/hold counts
        intent_counts: emitted no_op/press/release intent counts
    - Training summary policy metadata now reports desired_input_states and
      intent_action_kinds separately.

  gd_rl/__init__.py
    - exported ButtonStateIntentAdapter and DESIRED_INPUT_STATES.

  tests/test_live_learner.py
    - added fake/unit coverage proving repeated desired hold emits one press
      then no_op, independent of delayed executed input.
    - updated the one-step reward learner test to train toward desired hold
      and verify the emitted intent is press.

Verification:
  python -m pytest -q tests/test_live_learner.py
  python -m pytest -q tests/test_live_practice_env.py tests/test_live_learner.py tests/test_run_live_rl_practice_geode.py
  python scripts/run_live_rl_practice_geode.py --help
  python -m pytest -q
  python -m pytest --collect-only
  result: 138 tests collected; full pytest passed

Manual live retry:
  Attempted a 1-attempt guarded local/offline smoke with the updated adapter:
    artifacts/live_rl_practice_20260626_014424
  The bridge timed out before a summary or diagnostics were persisted:
    error: bridge communication failed: cannot read from timed out object
  This left only an empty attempt_001 directory under artifacts/. Rerun after
  confirming the Geode observation stream is active at the fresh level start.

Successful desired-state live smoke:
  Command:
    python scripts/run_live_rl_practice_geode.py
      --level-id manual_live_rl_desired_state_smoke
      --attempts 1
      --max-steps 600
      --profile Advanced
      --post-terminal-delay-seconds 5
      --start-guard-reset-retries 3
      --start-guard-retry-delay-seconds 1
      --require-start-percent-max 2
      --require-start-x-max 50

  Artifact:
    artifacts/live_rl_practice_20260626_014925

  Result:
    attempts: 1
    step_count: 206
    cleared: false
    death_tick: 206
    final/best percent: 26.0430
    row_count: 207
    total_reward: 32.0423
    reset_attempts: 1
    desired action counts: idle 111, hold 95
    emitted intent counts: no_op 107, press 50, release 49
    intended_event_count: 99
    executed_event_count: 93
    dropped_event_count: 2
    pending_not_dispatched_count: 4
    observed input transition count in trace: 33

  Comparison with previous direct no_op/press/release smoke:
    intended_event_count improved from 133 to 99.
    dropped_event_count improved from 61 to 2.
    observed input transitions stayed similar: 34 -> 33.
    progress/death outcome stayed the same at this untrained one-attempt
    smoke, which is expected; the policy is still random, but the intent stream
    is now much less contradictory.

Next recommended implementation step:
  Add a small input-rate penalty/cooldown or short hold-duration prior before
  running multi-attempt live training, because the desired-state adapter removes
  impossible duplicate edges but still allows rapid random toggling.
```

Minimal actor-critic/A2C implementation update:

```text
Implemented:
  gd_rl/live_learner.py
    - ActorCriticConfig for the first Phase D A2C-style learner.
    - LiveActionHistory and history features for recent desired button state
      plus emitted intent kind, so delayed observation/execution context is
      available to the policy.
    - TinyLiveActorCriticNetwork with a shared compact-observation encoder,
      desired idle/hold actor head, and scalar value head.
    - run_actor_critic_attempt(...) and run_actor_critic_training(...).
    - configurable death-local feedback that weights the recent terminal
      death window in reward/advantage space and persists death-local stats,
      without writing exact replacement click ticks.
    - optional generic input-rate penalty for emitted press/release intents.

  scripts/run_live_rl_practice_geode.py
    - added --algorithm {a2c,reinforce}; default is now a2c.
    - keeps tiny guarded live defaults:
        attempts=1
        max_steps=600
        device=cpu
        no checkpoint writing
    - added A2C flags:
        --value-loss-weight
        --normalize-advantages
        --history-length
        --death-local-window
        --death-local-penalty
        --input-rate-penalty
    - REINFORCE remains available as a smoke fallback with
      --algorithm reinforce.

  gd_rl/__init__.py
    - exported actor-critic config, network, history, encoder, and training
      helpers.

  tests/
    - fake-client coverage for A2C policy update, history feature encoding,
      death-local feedback stats, and CLI A2C wiring.

Verification:
  python -m pytest -q tests/test_live_learner.py
  python -m pytest -q tests/test_run_live_rl_practice_geode.py
  python scripts/run_live_rl_practice_geode.py --help
  python -m pytest -q
  python -m pytest --collect-only
  result: 141 tests collected; full pytest passed

Next recommended implementation step:
  Run a guarded one-attempt local/offline A2C smoke. Prefer a tiny ship or
  flying-control fixture where closed-loop y/y_vel/input correction matters.
  If only the existing spike fixture is available, treat the run as plumbing
  validation only, not proof of closed-loop learning.
```

---

## 11. Prompt For Next Agent Work

```text
We are in the geometry_dash_ai repo. Continue from agent_plan.md.

Main objective:
Validate and iterate on the first Phase D closed-loop actor-critic learner.
The goal is no longer to discover exact press/release ticks. The goal is to
learn an observation-conditioned feedback controller that can correct for
humanized execution error over time.

Current state:
- Phase A RL practice skeleton is implemented and pushed.
- Live Geode queued replay executor is implemented and pushed.
- Phase B reward sanity baselines passed on the local/offline triple-spike
  fixture.
- Phase C CEM timing search is closed as diagnostic only. Do not continue
  optimizing CEM timing windows.
- Live step env exists:
  - `gd_rl/live_env.py`
  - `LivePracticeEnv.reset/step/save_attempt`
  - policy intent -> online HumanizedAgent -> executed input -> Geode bridge
- Tiny live neural learner exists:
  - `gd_rl/live_learner.py`
  - compact observation encoder
  - `TinyLivePolicyNetwork`
  - REINFORCE loop
  - `TinyLiveActorCriticNetwork`
  - LiveActionHistory features
  - A2C-style actor/value update
  - death-local feedback stats
- Guarded live Geode smoke driver exists:
  - `scripts/run_live_rl_practice_geode.py`
  - default --algorithm a2c
  - --algorithm reinforce remains available as a smoke fallback
- Desired button-state adapter is implemented:
  - policy outputs desired idle/hold
  - `ButtonStateIntentAdapter` converts intended state transitions to
    no_op/press/release
- Latest full pytest passed:
  - 141 tests collected
- artifacts/ is ignored and must remain local-only.

Direction:
- The agent is a level-specific practice controller, not a timing-table or
  macro-replay agent.
- Exact ticks such as "press at 192" are not stable primitives because the
  human model introduces visual delay, motor delay, jitter, drops, and
  accumulated state error.
- This matters most in ship/flying sections, where survival depends on
  continuous correction of y-position, velocity, input state, and delayed
  observations.
- Practice memory should start as state-space diagnostics, not as direct
  press/release overrides.

Task:
Run the smallest guarded live A2C validation and prepare the closed-loop
experiment path:
- Keep policy output as desired input state only: idle/hold.
- Keep `ButtonStateIntentAdapter`.
- All emitted press/release intent must still pass through the online
  `HumanizedAgent` inside `LivePracticeEnv`.
- Keep compact bridge observations; do not introduce screenshot RL.
- Use `scripts/run_live_rl_practice_geode.py` with default `--algorithm a2c`
  for one-attempt local/offline smoke validation.
- Prefer a tiny local/offline ship corridor or flying-control fixture where
  y/y_vel/input correction matters.
- If that fixture is not ready, only run a smoke on the existing local/offline
  fixture and do not claim closed-loop learning success from it.
- Add no-input, random desired-state, and simple fixed-control baselines for
  the closed-loop fixture before larger training.
- Evaluate with changed human seeds before claiming learning success.
- Consider only generic regularization if live A2C toggles too quickly:
  entropy tuning, input-rate penalty, decision stride, or action repeat.
  Do not add level-specific cooldown scripts.

Smallest live experiment after tests pass:
- Prefer a tiny local/offline ship corridor or flying-control fixture where
  closed-loop correction matters.
- If that fixture is not ready, only run a smoke on the existing local/offline
  fixture and do not treat it as proof of the closed-loop direction.
- Compare against no-input, random desired-state, and simple fixed-control
  baselines.
- Evaluate with changed human seeds before claiming learning success.

Constraints:
- RL policy output is intent, not direct game input.
- All intended outputs must pass through the human mock model.
- Imitation components are optional support tools only.
- Use local/offline levels only.
- Do not use online levels or leaderboard submission.
- Do not commit generated artifacts, traces, frames, checkpoints, summaries, or
  live run outputs.
- Keep tests runnable without Geometry Dash.
- Do not keep expanding CEM timing search unless a narrow diagnostic requires
  it.
- Do not implement exact-tick practice memory or macro replay as the next
  learning architecture.
```
