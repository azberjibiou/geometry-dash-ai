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
Start the post-CEM RL pivot: build the first live step-based practice
environment for observation-conditioned RL.
```

Recommended order:

1. Inspect the Geode bridge protocol and determine what live input-control
   commands already exist or need to be added.
2. Add a step environment abstraction, such as `LivePracticeEnv`, that exposes
   reset/step semantics for one local/offline level.
3. Keep the policy boundary as intent, not direct game input.
4. Add an online human-action wrapper with delay/noise/drop queues for per-step
   intended actions.
5. Start with compact observations from bridge state rather than screenshots:
   tick, percent, x/y, velocity, input state, dead/clear flags, and recent
   progress.
6. Start with a tiny action space: keep/no-op, intend press, intend release,
   or a short intended tap action.
7. Add fake-client or synthetic tests so the step loop runs without Geometry
   Dash.
8. Only after the step environment is reliable, add a minimal neural learner
   such as REINFORCE/A2C/PPO on the local/offline fixture.

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

---

## 11. Prompt For Next Agent Work

```text
We are in the geometry_dash_ai repo. Continue from agent_plan.md.

Main objective:
Close out Phase C as a diagnostic and start the next phase: build the first
live observation-conditioned RL practice environment. The goal is to move
beyond queued whole-macro candidates toward per-step intended actions that are
filtered through the human mock model before they reach Geometry Dash.

Current state:
- Phase A RL practice skeleton is implemented and pushed.
- Live Geode queued replay executor is implemented and pushed.
- Phase B reward sanity baselines passed on the local/offline triple-spike
  fixture.
- Phase C CEM timing search diagnostic is implemented and tested:
  - `gd_rl/timing_search.py`
  - `scripts/run_rl_timing_search_geode.py`
  - timing window examples under `examples/timing_windows/`
- Phase C live smoke on the harder local/offline fixture
  (triple spike plus two jump orbs) improved progress from:
  - triple-only probe best_percent: 54.3307
  - memory-seeded CEM best recheck: 62.1212
- CEM is not the final direction and should not be further optimized except as
  a debugging baseline.
- pytest previously passed: 129 passed.
- artifacts/ is ignored and must remain local-only.

Task:
Implement the first live step-based practice environment for real RL:
- Inspect existing Geode bridge capabilities and identify whether live
  press/release input commands exist or need to be added.
- Add a `LivePracticeEnv` or equivalent reset/step abstraction.
- Each step should consume an observation, accept a policy intent, pass that
  intent through an online human model, apply only the executed input to the
  live local/offline level, and return next observation, reward, done, and info.
- Persist traces and summaries in the same spirit as the queued practice
  runner, but do not require a full intended macro up front.
- Add tests with fake clients or synthetic step traces so the environment is
  testable without Geometry Dash.
- If bridge support is missing, implement the smallest bridge/protocol addition
  needed for live local input stepping, with fake-client tests first.
- Do not start a large training run yet. Once the step environment is reliable,
  propose the smallest neural RL learner to connect next.

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
```
