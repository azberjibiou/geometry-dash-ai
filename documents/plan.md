# Geometry Dash Human-like RL Practice Agent Plan

## 0. Project Goal

Build a **level-specific practicing RL agent** for Geometry Dash. The agent
should repeatedly attempt one small local/offline level, die, observe progress
and trace information, update its policy or practice memory, and improve under
a stochastic human mock model.

The RL policy should output intended actions or intended press/release events.
Those intended outputs must pass through the human model before reaching
Geometry Dash.

The goal is **not** to build a perfect bot, a leaderboard bot, or a
zero-shot Geometry Dash generalist.

The goal is to model a player who:

- observes the screen with delay,
- makes decisions with delay,
- clicks with motor delay,
- has timing variance,
- becomes less consistent when clicks are too close together,
- may become slightly less precise after long gaps,
- can be configured as Beginner / Intermediate / Advanced / Top Player,
- can run many RL practice attempts to measure learning progress, clear rate, playtime, attempts, and death distribution,
- can remember level-specific timing patterns after practice, like human muscle memory.

Final use case:

```text
Given one local/offline Geometry Dash level, an RL policy, and a human profile,
practice the level over repeated attempts and report:
- attempts to first clear
- playtime to first clear
- progress over attempts
- post-practice clear rate
- average progress
- most common death positions
- hardest sections
- learned timing windows / muscle-memory sections
- sensitivity to reaction delay and timing variance
```

Imitation learning is optional support only. It can be used as a bootstrap,
diagnostic baseline, data tool, or pretraining method, but it does not define
the agent architecture.

---

## 1. Core Principle

The project should separate five layers:

```text
Geometry Dash Environment
        -> Observation / Trace Interface
        -> RL Policy / Practice Learner
        -> Level Memory / Practice Adapter
        -> Human-like Input Wrapper
        -> Reward / Update Loop
```

The RL policy should decide what it *intends* to do from observation and
practice state.

The level memory / practice adapter should use previous attempts on the same level to adjust timing, prefer learned event windows, and build muscle memory.

The human wrapper should decide when that intended input actually reaches the game.

The reward/update loop should turn trace, death, progress, clear status, and
practice memory changes into learning signals for the next attempt.

This separation is important because we want to distinguish:

```text
The RL policy selected an intended action/event
vs.
The level memory has learned this level-specific timing window
vs.
The human-like player executed it imperfectly
vs.
The resulting trace/death/progress changed future practice behavior
```

---

## 2. Target Architecture

```text
+-----------------------------+
| Geometry Dash / Geode Mod   |
|                             |
| - captures observation      |
| - applies input events      |
| - detects death/progress    |
| - sends data to Python      |
+--------------+--------------+
               |
               | observation per tick
               v
+-----------------------------+
| Python Environment Bridge   |
|                             |
| - receives observations     |
| - sends actions             |
| - logs traces               |
| - handles reset/replay      |
+--------------+--------------+
               |
               v
+-----------------------------+
| Observation Buffer          |
|                             |
| - stores recent frames      |
| - returns delayed obs       |
+--------------+--------------+
               |
               v
+-----------------------------+
| RL Policy / Practice        |
| Learner                     |
|                             |
| input: delayed observation  |
| output: intended events     |
|                             |
| events: press / release     |
+--------------+--------------+
               |
               v
+-----------------------------+
| Level Memory / Practice     |
| Adapter                     |
|                             |
| - uses per-level memory     |
| - adjusts event timing      |
| - selects learned windows   |
| - tracks death sections     |
+--------------+--------------+
               |
               v
+-----------------------------+
| Human Input Wrapper         |
|                             |
| - motor delay               |
| - timing jitter             |
| - click interval variance   |
| - miss probability          |
| - correlated errors         |
+--------------+--------------+
               |
               | actual input event
               v
+-----------------------------+
| Geometry Dash               |
+-----------------------------+
               |
               | trace, death, progress, reward
               v
+-----------------------------+
| Practice Update Loop        |
|                             |
| - computes reward           |
| - updates RL policy/memory  |
| - schedules next attempt    |
+-----------------------------+
```

---

## 3. Fixed Experimental Assumptions

For the first version, fix the environment.

```text
FPS: 240
CBF: off
Physics Bypass: off unless explicitly enabled
Online submission / leaderboard usage: prohibited
Test levels: local or offline
```

Every trace and result must store metadata:

```json
{
  "gd_version": "TBD",
  "geode_version": "TBD",
  "fps": 240,
  "cbf": false,
  "physics_bypass": false,
  "input_mode": "vanilla_240",
  "level_name": "TBD",
  "run_id": "TBD"
}
```

---

## 4. Human Model

### 4.1 Delayed Perception

The AI should not receive the current frame immediately.

Instead:

```text
actual game tick = t
AI observes tick = t - visual_delay_frames
```

Example:

```python
visual_delay_frames = 8
```

At 240 FPS:

```text
8 frames ≈ 33.3 ms
```

---

### 4.2 Delayed Motor Execution

After the AI decides to press or release, the input should not be applied immediately.

Instead:

```text
AI decision tick = t
actual input tick = t + motor_delay_frames + jitter
```

Example:

```python
motor_delay_frames = 8
```

At 240 FPS:

```text
8 frames ≈ 33.3 ms
```

Total perception-to-input delay:

```text
visual_delay_frames + motor_delay_frames
```

---

### 4.3 Event-based Input

Do not model input as a noisy button state every frame.

Model input as discrete events:

```text
press
release
no_event
```

Reason:

- Cube needs press timing.
- Ship and wave need hold/release timing.
- Ball, UFO, robot, spider, swing also depend on event timing.
- Human error is more naturally modeled as event timing error.

Canonical event schema:

```python
@dataclass
class Event:
    tick: int
    kind: Literal["press", "release"]
    player: Literal["p1", "p2"] = "p1"
```

---

### 4.4 Timing Variance

Each intended event receives a random timing error.

Basic model:

```text
actual_tick = intended_tick + motor_delay + sampled_error
```

The sampled error should depend on the distance from the previous intended event.

Let:

```text
delta = current_intended_tick - previous_intended_tick
```

Use a U-shaped variance model:

```python
std(delta) =
    base_std
    + close_amp * exp(-delta / close_tau)
    + long_amp * log1p(delta / long_tau)
```

Interpretation:

- Very close clicks are hard because the player must react quickly.
- Medium rhythmic clicks are easier.
- Very long gaps can slightly increase uncertainty.

Example:

```python
base_std = 1.5
close_amp = 3.0
close_tau = 10.0
long_amp = 0.4
long_tau = 120.0
```

At 240 FPS:

```text
10 frames ≈ 41.7 ms
120 frames ≈ 500 ms
```

---

### 4.5 Correlated Timing Error

Human errors should not be fully independent.

If a player is slightly late on one click, they may remain slightly late on the next click.

Use an AR(1)-style error model:

```python
error_i = rho * error_{i-1} + sqrt(1 - rho**2) * std_i * normal()
```

Suggested initial value:

```python
rho = 0.3
```

---

### 4.6 Miss Probability

Some intended events may be dropped.

This models:

- panic,
- finger slip,
- double-click failure,
- extremely close clicks,
- fatigue-like mistakes.

Suggested model:

```python
miss_prob(delta) =
    miss_prob_base
    + miss_prob_close_amp * exp(-delta / miss_prob_close_tau)
```

Example:

```python
miss_prob_base = 0.002
miss_prob_close_amp = 0.03
miss_prob_close_tau = 8.0
```

---

## 5. HumanProfile

Use a configurable profile.

```python
@dataclass
class HumanProfile:
    name: str

    visual_delay_frames: int
    motor_delay_frames: int

    base_press_std_frames: float
    base_release_std_frames: float

    close_amp: float
    close_tau: float
    long_amp: float
    long_tau: float

    error_rho: float

    miss_prob_base: float
    miss_prob_close_amp: float
    miss_prob_close_tau: float

    random_seed: int
```

Initial example profiles:

```python
BEGINNER = HumanProfile(
    name="Beginner",
    visual_delay_frames=18,
    motor_delay_frames=18,
    base_press_std_frames=5.0,
    base_release_std_frames=5.0,
    close_amp=5.0,
    close_tau=12.0,
    long_amp=0.8,
    long_tau=120.0,
    error_rho=0.4,
    miss_prob_base=0.03,
    miss_prob_close_amp=0.07,
    miss_prob_close_tau=10.0,
    random_seed=0,
)

INTERMEDIATE = HumanProfile(
    name="Intermediate",
    visual_delay_frames=12,
    motor_delay_frames=12,
    base_press_std_frames=3.0,
    base_release_std_frames=3.0,
    close_amp=3.0,
    close_tau=10.0,
    long_amp=0.5,
    long_tau=120.0,
    error_rho=0.3,
    miss_prob_base=0.01,
    miss_prob_close_amp=0.03,
    miss_prob_close_tau=8.0,
    random_seed=0,
)

ADVANCED = HumanProfile(
    name="Advanced",
    visual_delay_frames=8,
    motor_delay_frames=8,
    base_press_std_frames=1.5,
    base_release_std_frames=1.5,
    close_amp=1.5,
    close_tau=8.0,
    long_amp=0.3,
    long_tau=120.0,
    error_rho=0.25,
    miss_prob_base=0.003,
    miss_prob_close_amp=0.01,
    miss_prob_close_tau=8.0,
    random_seed=0,
)

TOP_PLAYER = HumanProfile(
    name="TopPlayer",
    visual_delay_frames=5,
    motor_delay_frames=5,
    base_press_std_frames=0.7,
    base_release_std_frames=0.7,
    close_amp=0.8,
    close_tau=6.0,
    long_amp=0.15,
    long_tau=120.0,
    error_rho=0.2,
    miss_prob_base=0.0005,
    miss_prob_close_amp=0.003,
    miss_prob_close_tau=6.0,
    random_seed=0,
)
```

These values are placeholders and must be calibrated later.

---

## 6. Muscle Memory Layer

The agent should be general, but it should be allowed to learn a specific level through repeated practice.

This means the policy should not be only:

```text
observation -> action
```

It should become:

```text
observation
  -> general policy
  -> level memory / practice adapter
  -> intended event
  -> human input wrapper
  -> actual event
```

### 6.1 Purpose

The muscle memory layer stores information learned from previous attempts on the same level.

It should help the agent:

- remember successful click timings,
- remember valid press/release windows,
- identify dangerous sections,
- adjust timings after deaths,
- reuse successful macro fragments,
- improve consistency through practice.

This layer should not replace the general policy.

The general policy should still understand game state, mode, gravity, obstacles, and possible actions.

The level memory should specialize that general behavior for one practiced level.

---

### 6.2 LevelMemory Data

Initial schema:

```python
@dataclass
class LevelMemory:
    level_id: str
    level_name: str
    fps: int
    input_mode: str

    attempts_seen: int
    best_percent: float
    clears: int

    event_windows: list[MemoryEventWindow]
    macro_fragments: list[MacroFragment]
    death_histogram: dict[int, int]
    section_stats: dict[str, SectionStats]

    random_seed: int
```

Event-window schema:

```python
@dataclass
class MemoryEventWindow:
    center_tick: int
    start_tick: int
    end_tick: int
    kind: Literal["press", "release"]
    player: Literal["p1", "p2"] = "p1"
    confidence: float = 0.0
    source: Literal["recording", "practice", "search"] = "practice"
```

Macro-fragment schema:

```python
@dataclass
class MacroFragment:
    start_tick: int
    end_tick: int
    events: list[Event]
    success_count: int
    failure_count: int
    confidence: float
```

---

### 6.3 Practice Update Loop

After each attempt:

```text
1. Save trace and actual input events.
2. Record final percent, death tick, and death position.
3. Identify the last relevant intended events before death.
4. Compare intended events vs actual humanized events.
5. Update death histogram and section stats.
6. If a section was cleared, increase confidence in nearby event windows.
7. If a death likely came from early/late timing, shift or widen the window.
8. If a repeated pattern succeeds, save it as a macro fragment.
```

Important:

```text
The memory should store timing windows and confidence,
not only one exact macro.
```

This keeps the system closer to human muscle memory:

```text
"click around here"
instead of
"click on this exact frame forever"
```

---

### 6.4 Runtime Use

At each tick, the practice adapter should combine:

```text
general policy event probabilities
+ current progress / x-position
+ known event windows
+ macro fragments
+ current button state
= intended press/release event or no_event
```

If memory confidence is high, the adapter may prefer the memorized timing.

If memory confidence is low, the adapter should defer more to the general policy.

Suggested behavior:

```text
early practice:
  mostly general policy

after repeated attempts:
  general policy + remembered event windows

after many successful section clears:
  high-confidence muscle-memory fragments
```

---

### 6.5 Evaluation Metrics

For the current goal, do not require a single difficulty score.

Measure learning directly:

```text
attempts_to_first_clear
playtime_to_first_clear_seconds
best_percent_by_attempt
death_histogram_over_time
section_success_rate_over_time
clear_rate_after_practice
average_attempts_per_clear_after_practice
memory_confidence_by_section
```

Success criterion:

```text
On a fixed level, repeated practice should improve progress,
reduce repeated deaths in learned sections,
and eventually increase clear rate.
```

---

## 7. Development Roadmap

### Phase 1 — Python-only Human Model

Goal:

Build and test the human-like input wrapper without Geometry Dash.

Deliverables:

```text
gd_human_model/
  events.py
  profile.py
  observation_buffer.py
  motor_noise.py
  humanized_agent.py
  tests/
```

Implement:

- `Event`
- `HumanProfile`
- `ObservationBuffer`
- `MotorNoiseModel`
- `HumanizedAgent`

Test cases:

```text
1. Same random seed gives identical actual events.
2. Larger close_amp increases variance for close clicks.
3. Larger visual_delay returns older observations.
4. Larger motor_delay schedules events later.
5. Miss probability can drop events.
6. Press/release order is preserved unless impossible.
7. Event jitter distribution roughly matches configured std.
```

Success criterion:

```text
Given a scripted intended event list,
the wrapper produces deterministic, delayed, noisy actual events.
```

---

### Phase 2 — Trace and Macro Format

Goal:

Create canonical trace and macro formats.

Trace schema:

```text
tick
time_ms
input_down
x
y
x_vel
y_vel
rotation
mode
gravity
percent
dead
death_reason
fps
cbf
physics_bypass
```

Macro schema:

```text
tick
kind: press | release
player: p1 | p2
```

Deliverables:

```text
gd_trace/
  trace_schema.py
  macro_schema.py
  load_trace.py
  save_trace.py
  compare_trace.py
  analyze_click_window.py
  tests/
```

Tools:

```text
gd-trace-validate trace.jsonl
gd-macro-validate macro.json
gd-trace-compare a.jsonl b.jsonl
```

Success criterion:

```text
Synthetic traces and macros can be loaded, validated, compared, and summarized.
```

---

### Phase 3 — Minimal Geometry Dash Bridge

Goal:

Connect Python to Geometry Dash through a Geode mod.

Initial mod responsibilities:

```text
1. Send observation once per physics tick.
2. Receive action event from Python.
3. Apply press/release on the next tick.
4. Log local JSONL trace.
5. Support restart/reset.
```

Initial observation fields:

```text
tick
x
y
y_vel
mode
gravity
percent
dead
input_down
```

Do not require screenshot capture yet.

Python side:

```text
1. Connect to the mod.
2. Send scripted actions.
3. Receive observations.
4. Save traces.
5. Reset after death.
```

Success criterion:

```text
A Python script can make the cube jump in a local level and record the resulting trace.
```

---

### Phase 4 — Deterministic Replay Check

Goal:

Verify that scripted replay is stable.

Run repeated trials with identical input events.

Tests:

```text
1. No input macro.
2. Single jump macro.
3. Ten-click macro.
4. Death macro.
5. Simple clear macro if available.
```

Metrics:

```text
death_tick_std
final_percent_std
x_position_max_diff
y_position_max_diff
success_rate_for_identical_macro
```

Success criterion:

```text
Identical input events produce identical or near-identical outcomes under fixed settings.
```

If this fails, do not proceed to AI training.

---

### Phase 5 — Screenshot Observation

Goal:

Add visual observation for AI.

Observation types:

```text
1. internal state only
2. screenshot only
3. screenshot + progress/x-position
```

Recommended first AI observation:

```text
last 4 grayscale frames
+ progress
+ current button_down state
```

Reason:

- One frame may not reveal velocity.
- Progress helps map memorization.
- Current button state is needed for hold-based modes.

Success criterion:

```text
Python can receive frame observations at a usable rate and store them with aligned input labels.
```

---

### Phase 6 — Scripted Policy + Human Wrapper End-to-End

Goal:

Run the full pipeline without neural networks.

```text
scripted intended macro
→ visual delay
→ motor delay
→ jitter
→ actual macro
→ Geometry Dash
→ trace
→ result summary
```

Run many attempts with the same intended macro and a fixed human profile.

Metrics:

```text
clear_rate
average_progress
death_histogram
actual_event_distribution
```

Success criterion:

```text
Changing HumanProfile parameters changes performance in the expected direction.
```

Examples:

```text
Higher base_std → lower clear rate.
Higher visual_delay → lower clear rate.
Higher close_amp → harder dense-click sections.
Higher miss_prob → more random deaths.
```

---

### Phase 7 - Practice Environment and Reward Loop

Goal:

Build the repeated-attempt RL training loop around one small local/offline
level.

Core loop:

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

Near-term deliverables:

```text
gd_rl/
  env.py
  rewards.py
  attempt_runner.py
  practice_loop.py
  rollout.py
  tests/
```

Implement:

- `PracticeEnv` or equivalent wrapper over the existing bridge/replay tools.
- `RewardComputer` for progress, survival, death, and clear signals.
- `AttemptResult` with trace path, intended events, executed events, final
  percent, death tick/percent, clear status, and reward totals.
- Repeated-attempt runner that resets after death or completion.
- Event/action logging that preserves intended policy output separately from
  humanized executed input.
- Local/offline level guards so training cannot accidentally target online or
  leaderboard use.

Initial reward sketch:

```text
+ progress_delta reward
+ best_progress bonus
+ section survival bonus
+ clear bonus
- death penalty scaled by lost opportunity
- optional excessive/illegal input penalty
```

Success criterion:

```text
A scripted or random policy can run many attempts on the same local/offline
level through the human wrapper, save intended/executed event logs and traces,
compute rewards, and produce a practice summary.
```

---

### Phase 8 - Minimal RL Agent on One Local Level

Goal:

Train the first small RL policy under the stochastic human mock model.

Initial action space options:

```text
Option A: discrete intended button state per tick
  no_op | press | release

Option B: event proposal
  no_event | press_now | release_now

Option C: short timing adjustment around known candidate windows
  shift earlier | keep | shift later | widen/search
```

Start with the simplest action space that can drive the selected local test
level. The policy may use internal state, progress/x-position, current
input_down, and compact visual features. Screenshot-only learning is not
required for the first RL loop.

Training approach:

```text
Use a small, local, repeatable algorithm first:
- random/search baseline for reward sanity
- cross-entropy method or policy-gradient spike
- simple replay/practice memory for successful sections
```

Do not optimize for zero-shot generalization. Do not target online levels.
Do not aim for perfect deterministic bot behavior.

Success criterion:

```text
On a fixed local/offline level, repeated RL practice improves average progress
or best progress under the same HumanProfile compared with the initial policy.
```

---

### Phase 9 - Level Memory / Muscle Memory Adapter

Goal:

Make repeated attempts improve performance on the same level by storing
robust timing windows and section knowledge.

Implement:

```text
LevelMemory
MemoryEventWindow
MacroFragment
SectionStats
PracticeAdapter
PracticeUpdater
```

The adapter should combine:

```text
RL policy output
+ progress / x-position
+ learned event windows
+ learned macro fragments
+ current button state
= intended event
```

The updater should consume:

```text
trace
intended events
actual humanized events
reward terms
death position
clear/failure result
```

and update the level memory after each attempt.

Important:

```text
The memory should store timing windows, confidence, correction history, and
section stats, not only one exact macro.
```

Success criterion:

```text
On a fixed level, the agent improves over repeated attempts without
hard-coding a final exact macro by hand.
```

---

### Phase 10 - Practice Learning Evaluation

Goal:

Measure how well the RL agent learns a specific level through practice.

For each level and human profile:

```text
run repeated practice attempts
store traces, intended events, executed events, rewards, deaths, memory updates,
and clears
```

Report:

```text
attempts_to_first_clear
playtime_to_first_clear_seconds
best_percent_by_attempt
progress_curve
average_progress
median_progress
post_practice_clear_rate
death_histogram
death_histogram_over_time
hardest_sections
section_success_rate_over_time
reward_curve
memory_confidence_by_section
sensitivity_to_delay
sensitivity_to_std
sensitivity_to_close_amp
```

Do not require a single difficulty score for early versions.

Use attempts, playtime, progress, reward, and consistency as the primary
measurements.

Success criterion:

```text
Repeated practice produces a visible learning curve:
best progress increases, repeated deaths move later or disappear, reward
improves, and clear rate improves after memory is learned.
```

---

### Optional Support Track - Imitation and Event Windows

Goal:

Keep imitation learning useful as support infrastructure without making it the
main agent architecture.

Allowed uses:

- Bootstrap an initial policy from a local/offline demonstration.
- Produce diagnostic baselines for event decoding and replay.
- Estimate valid event windows from recorded attempts.
- Pretrain visual features before RL practice.
- Compare RL-learned timing windows against demonstration timing.

Not allowed as the main objective:

- A standalone macro-prediction policy treated as the final agent.
- A replay bot that bypasses the human mock model.
- A zero-shot generalization benchmark.
- Online-level or leaderboard-focused training.

Success criterion:

```text
Imitation tools can help initialize or inspect practice, but the primary
training result is still measured by repeated RL attempts through the human
model on one fixed local/offline level.
```

---

### Phase 11 — Calibration Against Human Data

Goal:

Fit human profiles and practice-learning behavior to real player data.

Collect:

```text
player skill label
attempt count
playtime to first clear
death positions
successful clear macro if available
practice runs
progress by attempt
```

Fit:

```text
visual_delay
motor_delay
base_std
close_amp
miss_prob
memory_confidence_growth
event_window_width
section_learning_rate
```

Target:

```text
simulated death histogram resembles real death histogram
simulated clear rate resembles human clear rate
simulated learning curve resembles human learning curve
```

Success criterion:

```text
HumanProfile and LevelMemory parameters become empirically meaningful instead of arbitrary.
```

---

## 8. Initial Repository Layout

```text
gd-difficulty-ai/
  README.md
  plan.md

  pyproject.toml

  gd_human_model/
    __init__.py
    events.py
    profile.py
    observation_buffer.py
    motor_noise.py
    humanized_agent.py

  gd_trace/
    __init__.py
    trace_schema.py
    macro_schema.py
    load_trace.py
    save_trace.py
    compare_trace.py
    click_window.py

  gd_learning/
    __init__.py
    level_memory.py
    practice_adapter.py
    practice_updater.py
    section_stats.py
    death_analyzer.py

  gd_env/
    __init__.py
    client.py
    protocol.py
    dummy_env.py

  scripts/
    run_humanized_macro.py
    run_practice_session.py
    compare_traces.py
    analyze_attempts.py

  tests/
    test_events.py
    test_observation_buffer.py
    test_motor_noise.py
    test_humanized_agent.py
    test_trace_schema.py
    test_macro_schema.py
    test_level_memory.py
    test_practice_adapter.py

  geode_mod/
    README.md
    # Actual Geode mod implementation later
```

---

## 9. First Codex Task

Ask Codex to implement Phase 1 only.

Prompt:

```text
Implement Phase 1 from plan.md.

Create a Python package `gd_human_model`.

Do not implement Geometry Dash integration.
Do not implement neural networks.

Implement:
- Event
- HumanProfile
- ObservationBuffer
- MotorNoiseModel
- HumanizedAgent

Add pytest tests for:
1. deterministic random seed
2. visual delay
3. motor delay
4. interval-dependent std
5. correlated errors
6. miss probability
7. press/release ordering

Keep the code clean, typed, and documented.
```

---

## 10. Non-goals for Early Versions

Do not do these early:

```text
1. Do not implement full Geometry Dash physics.
2. Do not start large-scale or open-ended RL before the local practice loop works.
3. Do not support all game modes.
4. Do not support online levels or leaderboard use.
5. Do not optimize for speed before correctness.
6. Do not use the bot for online leaderboard submission.
7. Do not require zero-shot generalization to unseen levels.
8. Do not treat imitation learning or macro prediction as the main architecture.
```

---

## 11. Main Risks

### Risk 1 — Environment bridge is unstable

If the same macro does not replay consistently, AI training becomes meaningless.

Mitigation:

```text
Build deterministic replay tests before training.
```

### Risk 2 — Optional supervised labels are ambiguous

There may be multiple valid click frames, but an imitation dataset contains
only one. This matters for bootstrap and diagnostics, but it should not define
the practicing RL objective.

Mitigation:

```text
Move from single-tick labels to event-window labels.
```

### Risk 3 — Human profile parameters are arbitrary

The initial profiles may not match real players.

Mitigation:

```text
Later calibrate profiles against real attempt/death data.
```

### Risk 4 — Screenshot-only observation is too hard

Pure visual input may be inefficient.

Mitigation:

```text
Start with screenshot + progress + input_down.
```

### Risk 5 — Full GD physics is too hard

Direct physics implementation may diverge from the real game.

Mitigation:

```text
Use actual Geometry Dash as the authoritative environment.
Build independent simulator only later if needed.
```

### Risk 6 — Muscle memory becomes a fixed macro

If the level memory stores only exact click frames, the agent may become a replay bot instead of a practicing player.

Mitigation:

```text
Store event windows, confidence, section stats, and correction history.
Use the general policy when memory confidence is low.
Evaluate with human-like jitter so memorized timing must be robust.
```

---

## 12. Current Status

Completed and pushed:

```text
Phase 1:
Python-only Human Model

Phase 2:
Trace and Macro Format

Phase 3:
Minimal Geometry Dash Bridge

Phase 4:
Deterministic Replay Check

Phase 5:
Screenshot Observation

Phase 6:
Scripted Policy + Human Wrapper End-to-End
```

Live smoke-test result:

```text
Commit: e331d43 Implement live Geode bridge
Bridge: Geode mod listening on 127.0.0.1:29430
Level used: Stereo Madness
Python sent: press at tick 20, release at tick 30
Observed: cube jumped in-game
Trace: artifacts/live_geode_smoke_trace.jsonl
Rows: 177
Input down observed: ticks 24..31
Final row: dead=true, death_reason=player_dead
```

The local smoke trace is intentionally ignored by Git through `artifacts/`.

Current local target:

```text
Phase 7:
Practice Environment and Reward Loop
```

Reason:

The non-neural Phase 6 pipeline now works end-to-end with centered timing
noise, queued live Geode replay, profile-sensitive summaries, and optional
imitation support tools. The next milestone is to wrap these pieces in a
repeated-attempt RL practice loop for one small local/offline level.

Phase 4 implementation:

```text
gd_trace/replay_check.py
  reusable metrics for repeated identical replay traces

scripts/run_geode_replay_check.py
  manual live Geode replay runner, queued macro replay by default

gd_env/protocol.py and gd_env/client.py
  load_macro and diagnostic messages for queued replay

geode_mod/src/main.cpp
  mod-side macro storage and attempt-tick playback

examples/macros/no_input.json
examples/macros/single_jump.json
examples/macros/short_repeated_clicks.json
examples/macros/death_macro.json
examples/macros/simple_clear.json
  tiny local/offline replay macros
```

The replay summary reports:

```text
final_percent_std
death_tick_std
x_position_max_diff
y_position_max_diff
success_rate
survival_rate
input_state_mismatch_ticks
observed input latency from macro event ticks to input_down transitions
first movement tick per trial
zero movement step counts
double movement step counts
mod-side macro application tick per intended event
```

Unit tests use synthetic traces only. Live Geometry Dash replay checks remain
manual and write ignored files under `artifacts/`.

Manual live workflow:

```powershell
.\.venv\Scripts\python.exe scripts\run_geode_replay_check.py examples\macros\single_jump.json --trials 5 --max-observations 600
```

The replay check now uses mod-side queued macro replay by default:

```text
Python sends load_macro once
Python sends reset before each trial
Geode applies due macro events by attempt tick inside the update hook
Python collects observations and diagnostic messages
```

To run the older live TCP action-send path for smoke testing:

```powershell
.\.venv\Scripts\python.exe scripts\run_geode_replay_check.py examples\macros\single_jump.json --trials 5 --max-observations 600 --live-send
```

The script resets the open local/offline level before each trial, saves per-trial
JSONL traces under `artifacts/replay_check_<timestamp>/`, and writes
`summary.json` in the same folder.

Live Phase 4 replay check result:

```text
Commit: e2bf09b Implement deterministic replay check

Live check level:
local/offline test level

single_jump.json:
  trials: 5
  deaths: none
  survival_rate: 1.0
  final_percent_std: 0.00399
  x_position_max_diff: 2.60
  y_position_max_diff: 0.0
  input latency: mostly stable, press +1 frame, release +2/+3 frames

no_input.json:
  trials: 5
  deaths: none
  survival_rate: 1.0
  final_percent_std: 0.00798
  x_position_max_diff: 3.89
  y_position_max_diff: 63.85
  input_state_mismatch_ticks: 0

short_repeated_clicks.json:
  trials: 5
  deaths: none
  survival_rate: 1.0
  final_percent_std: 0.0
  x_position_max_diff: 1.30
  y_position_max_diff: 66.36
  input latency: varied around +1 to +4 frames
  input_state_mismatch_ticks: 10
```

Investigation findings:

```text
The bridge is runnable and reset works, but replay is not deterministic enough
yet.

The no-input run diverged even though input_down stayed identical, so the issue
is not only Python macro timing.

First movement ticks for no_input were:
[4, 3, 3, 3, 4]

Adjacent observations sometimes showed zero movement steps and sometimes double
movement steps. This means the current trace tick is a synthetic observation
counter, not an authoritative fixed Geometry Dash physics tick.

Current Geode hook:
  GJBaseGameLayer::update(dt)
  then enqueue observation

Current Python macro behavior:
  Python receives observation tick N
  then sends any events whose macro tick <= N over TCP
  the mod applies received events on a later update

This live TCP loop creates variable input latency, and the update/observation
hook creates variable movement alignment.
```

Decision:

```text
Yes, fix or fully explain observation/physics tick alignment before Phase 5.

Reason:
Queued mod-side macros fixed deterministic input timing, but traces still show
zero/double movement steps and position drift. Screenshot labels, replay
comparisons, imitation-learning labels, and later practice-memory updates will
be hard to trust unless one trace tick corresponds to one real physics step, or
unless the remaining drift is explicitly understood and bounded.
```

Do not move to Phase 5 yet until remaining update/physics tick alignment drift
is understood.

Phase 4 hardening implementation status:

```text
Phase 4 hardening:
Mod-side deterministic macro queue / pre-scheduled replay

Implemented locally:
- protocol message: load_macro
- protocol message: diagnostic
- Python client method: load_macro(...)
- Python client method: run_loaded_macro(...)
- replay script defaults to queued macro replay
- --live-send keeps the older live action-send path
- --live-send clears any stale queued macro before comparison
- Geode mod stores a loaded macro inactive until reset
- Geode mod applies queued events by attempt tick inside the update hook
- summary.json reports movement-step diagnostics and macro application ticks

Verification:
- pytest: 42 passed
- Geode CMake build: succeeded using `C:\Program Files\CMake\bin\cmake.exe`
- Live queued replay validation: completed on local/offline test level
```

Live validation result:

```text
Queued mod-side replay:
  artifact: artifacts/replay_check_20260624_203558
  macro: examples/macros/single_jump.json
  trials: 5
  deaths: none
  survival_rate: 1.0
  input latency: press 0 frames, release 0 frames
  macro application: press tick 20 in all trials, release tick 30 in all trials
  input_state_mismatch_ticks: 0
  first_movement_ticks: [3, 4, 4, 3, 3]
  zero_movement_step_counts: [3, 9, 18, 11, 28]
  double_movement_step_counts: [1, 7, 16, 11, 23]
  x_position_max_diff: 10.386
  y_position_max_diff: 2.467

Live-send comparison:
  artifact: artifacts/replay_check_20260624_203852
  macro: examples/macros/single_jump.json
  trials: 5
  deaths: none
  survival_rate: 1.0
  input latency: press +1 frame in all trials, release +2/+3 frames
  input_latency_mean_frames: 1.6
  input_latency_std_frames: 0.663
  input_state_mismatch_ticks: 1
  first_movement_ticks: [3, 3, 3, 3, 3]
  zero_movement_step_counts: [5, 3, 12, 15, 7]
  double_movement_step_counts: [3, 1, 11, 14, 6]
  x_position_max_diff: 2.597
  y_position_max_diff: 0.0
```

Interpretation:

```text
The queued macro path removes live TCP input latency variance. Intended macro
ticks, mod-side application ticks, and observed input_down transitions now align
exactly for the single-jump macro.

Remaining drift is not from live action delivery. The zero/double movement step
counts and first movement tick variation show the observation/update tick is
still not an authoritative fixed physics tick.
```

Latest Phase 4 validation update:

```text
Bridge alignment:
  observations and queued macro replay now run from GJBaseGameLayer::processCommands
  reset uses a fuller level reset path before resetting bridge attempt state
  Python can stop a live trace on success percent for clear fixtures
  replay checker can fail fast on wrong-level captures with start/progress guards

Death macro:
  macro: examples/macros/death_macro.json
  trials: 5
  death_tick: 1154 in all trials
  x_position_max_diff: 0.0
  y_position_max_diff: 0.0
  input latency: 0 frames

Simple clear macro:
  macro: examples/macros/simple_clear.json
  artifact: artifacts/replay_check_20260625_001824
  level: tiny local/offline clear test level
  command guards:
    --stop-on-success
    --require-start-percent-max 2
    --require-start-x-max 50
    --require-progress-tick 120
    --require-progress-percent-min 10
  trials: 5
  deaths: none
  success_rate: 1.0
  survival_rate: 1.0
  final_percent_std: 0.0
  input_state_mismatch_ticks: 0
  row_counts: [680, 680, 679, 679, 679]
  x_position_max_diff: 4.895
  y_position_max_diff: 2.635

Interpretation:
  Core no-input, jump, repeated-click, and death fixtures now replay with stable
  inputs and deterministic or bounded outcomes under fixed settings.

  The simple clear fixture validates outcome stability rather than exact
  position identity. Small end-of-level position spread remains visible on the
  tiny clear level, but all trials clear with no input mismatches and zero final
  percent variance.
```

Latest Phase 5 validation update:

```text
Screenshot observation capture is implemented and pushed.

Current pushed commit:
  87e5985 Add guarded Geode frame capture controls

Implemented:
  gd_capture/screen_capture.py
    Windows window capture helpers, foreground checks, and window activation

  scripts/capture_geode_frames.py
    manual Geode frame capture runner with:
    - queued macro replay through --macro-json
    - capture stride and start tick controls
    - start/progress guards for wrong-level protection
    - foreground/window guards for visible gameplay capture
    - stop-before-completion and stop-on-success controls
    - manifest JSONL and summary JSON output
    - lightweight image validation

  geode_mod/src/main.cpp
    reset path clears stale completion/dropdown overlays before and after reset
    so frame datasets do not start under the LEVEL COMPLETE dialog

  tests/test_capture_geode_frames.py
    synthetic tests for guard and terminal-capture behavior

Verification:
  pytest: 61 passed
  Geode CMake build: succeeded and installed updated .geode package

Live capture:
  artifact: artifacts/frame_capture_20260625_011230
  macro: examples/macros/simple_clear.json
  local/offline tiny clear level
  frames: 140
  validation_ok: true
  start saved frame: tick 5, about 0.82%
  final saved frame: tick 700, about 96.03% telemetry
  sampled frames: gameplay-only, no LEVEL COMPLETE popup

Interpretation:
  Phase 5 is complete enough to proceed. Python can collect visible Geometry
  Dash gameplay frames aligned with bridge metadata and can reject common bad
  captures such as the wrong level, covered windows, or stale completion UI.
```

Latest Phase 6 validation update:

```text
Scripted policy + human wrapper end-to-end is implemented.

Implemented:
  gd_human_model/motor_noise.py
    HumanizedEventResult exposes per-event provenance:
    - timing std
    - miss probability
    - sampled timing error
    - raw tick
    - actual tick
    - drop reason

  gd_human_model/macro_humanizer.py
    reusable intended-macro to actual-macro adapter
    stores intended events, decision events, actual events, and per-event deltas
    supports deterministic output for a fixed seed

  gd_trace/humanized_run.py
    repeated-attempt summaries:
    - clears and clear_rate
    - average_progress and best_percent
    - final_percent_by_attempt
    - death tick/percent histograms
    - missed event counts
    - actual timing delta distributions

  scripts/run_humanized_geode_macro.py
    manual live Geode runner:
    - loads an intended macro JSON
    - selects a built-in HumanProfile or profile JSON
    - generates one actual macro per attempt
    - uses queued Geode replay
    - saves ignored artifacts under artifacts/
    - writes per-attempt traces, actual macros, humanization JSON, and summary.json
    - supports --post-terminal-delay-seconds to avoid stale completion states

  tests/
    test_macro_humanizer.py
    test_humanized_run.py
    test_run_humanized_geode_macro.py

Important correction:
  Macro humanization now distinguishes timing references:

  target:
    macro ticks are desired click timings.
    actual clicks are centered on those ticks, with jitter/misses around them.
    This is the default for recorded or scripted target macros.

  decision:
    macro ticks are AI/policy decision timings.
    visual and motor delay shift actual clicks later.
    This remains available for future policy-decision simulations.

Reason for correction:
  The first Phase 6 adapter treated macro ticks as decision ticks, which biased
  every profile late by visual_delay_frames + motor_delay_frames. That made
  Intermediate and Beginner always click late instead of clicking early/late
  around the intended timing. The default target mode fixes this.

Verification:
  pytest: 76 passed

Live validation:
  level:
    local/offline triple-spike test level

  raw centered macro:
    press 192, release 212
    result: clear

  corrected target-mode profile sweep:
    intended press: 192
    intended release: 212
    attempts per profile: 10

    TopPlayer:
      clears: 10/10
      timing_mean_frames: 0.00
      timing_std_frames: 0.71

    Advanced:
      clears: 10/10
      timing_mean_frames: -0.10
      timing_std_frames: 1.37

    Intermediate:
      clears: 10/10
      timing_mean_frames: 0.05
      timing_std_frames: 2.78

    Beginner:
      clears: 9/10
      timing_mean_frames: -0.10
      timing_std_frames: 4.49
      deaths: tick 206 once

Interpretation:
  Phase 6 is complete enough to proceed. The same intended macro now produces
  centered timing noise for every profile, while larger timing variance lowers
  consistency on the timing-sensitive triple-spike fixture.

  The current Beginner profile represents a practiced player with noisy motor
  execution, not a first-time sight-reading beginner. Later calibration should
  split profiles such as PracticedBeginner and UnpracticedBeginner.
```

Superseded imitation target (support-only historical note):

The following block records the previous imitation-learning direction. It is
kept because the implemented dataset, baseline, and decoder tools may still be
useful for bootstrap and diagnostics. It is no longer the main project target.

```text
Previous direction:
  Phase 7 imitation learning on one local/offline level.

Current interpretation:
  This work is support infrastructure only. The dataset builder, tiny baseline,
  decoder, and predicted-macro export can help bootstrap or diagnose the RL
  practice agent, but they are not the main objective.

Current main target:
  Phase 7 practice environment and reward loop.
```

---

## 13. Optional Imitation Support Log

This section records support infrastructure that already exists. These tools
can bootstrap, inspect, or compare RL practice, but they should not define the
main agent architecture.

Latest implemented Phase 7 work:

```text
Dataset preparation layer is implemented.

Added:
  gd_imitation/dataset.py
    - DatasetConfig
    - ImitationSample
    - build_imitation_samples(...)
    - load_imitation_samples(...)
    - load_samples_jsonl(...)
    - deterministic train/validation splits

  scripts/capture_geode_frames.py
    - now writes same-run trace.jsonl next to manifest.jsonl
    - summary.json includes trace_path and trace_row_count

  scripts/prepare_imitation_dataset.py
    - manifest.jsonl + trace.jsonl + macro.json -> samples.jsonl
    - writes split.json and summary.json

  gd_imitation/image_dataset.py
    - loads samples.jsonl
    - reads referenced BMP frames without third-party image dependencies
    - converts frames to normalized grayscale
    - downsamples directly to fixed size
    - caches frames across stacked samples
    - pads short initial frame stacks when requested
    - exposes scalar features:
        normalized progress
        input_down
    - exposes labels:
        press_event
        release_event
```

Verification:

```text
pytest: 94 passed

Real local/offline smoke artifact:
  capture:
    artifacts/phase7_single_jump_capture

  dataset:
    artifacts/phase7_single_jump_dataset

  macro:
    examples/macros/single_jump.json

  capture result:
    frames: 120
    trace rows: 120
    ticks: 0..119
    dead: false
    final_percent: about 12.9
    validation_ok: true

  dataset result:
    samples: 120
    train: 96
    validation: 24
    press label tick: 20
    release label tick: 30

  image loader smoke:
    samples loaded: 120
    frame stack shape: 4 x 12 x 16
    press labels: 1
    release labels: 1
    positive label ticks: [20, 30]
```

Important caveat:

```text
The current single_jump dataset is good for proving the pipeline, but it is too
small for meaningful model validation.

The default contiguous split places the positive labels in the train set only
because the press/release happen early. For real training, either:
  - capture a longer level with more events,
  - capture multiple attempts,
  - use shuffled splits for small smoke datasets,
  - or build a small balanced/stratified split helper later.
```

Additional implemented Phase 7 work:

```text
First tiny PyTorch imitation baseline is implemented.

Added:
  gd_imitation/baseline.py
    - tiny image-backed MLP baseline
    - train/validation loss and binary event metrics
    - writes metrics.json and predictions.jsonl
    - optional checkpoint saving

  scripts/train_imitation_baseline.py
    - trains the baseline from a prepared dataset directory
    - writes outputs under a caller-provided artifacts/ directory

  gd_imitation/decoder.py
    - decodes per-tick press/release probabilities into Event objects
    - thresholding plus non-max suppression
    - button-state legality filtering
    - minimum event spacing
    - event-level timing metrics

  scripts/predict_imitation_macro.py
    - decodes an existing predictions.jsonl
    - can run a saved tiny-baseline checkpoint over a prepared dataset
    - writes predicted_macro.json, prediction_summary.json, and
      decoded_events.jsonl
    - exports through the canonical Macro schema
```

Verification:

```text
pytest: 107 passed

Smoke prediction artifact:
  input:
    artifacts/imitation_baseline_smoke/predictions.jsonl

  output:
    artifacts/imitation_baseline_smoke/predicted_macro/predicted_macro.json
    artifacts/imitation_baseline_smoke/predicted_macro/prediction_summary.json
    artifacts/imitation_baseline_smoke/predicted_macro/decoded_events.jsonl

  decoded events:
    press tick: 20
    release tick: 30

  macro validation:
    events: 2
    presses: 1
    releases: 1
    first_tick: 20
    last_tick: 30
```

Optional imitation follow-up:

```text
Replay artifacts/imitation_baseline_smoke/predicted_macro/predicted_macro.json
through the queued Geode replay checker.

Record:
  final_percent
  death_tick/death_percent if any
  progress delta versus the source demonstration
  event timing error versus the source macro

Do not treat this as the main project path.
Do not require zero-shot generalization yet.
Do not use online levels, leaderboard submission, or generated artifacts in Git.
```

Historical prompt for optional imitation follow-up:

```text
We are in the geometry_dash_ai repo. Run the optional imitation support
follow-up from plan.md only if it helps bootstrap or diagnose the RL practice
agent.

Current state:
- Phases 1 through 6 are implemented.
- Phase 7 dataset prep is implemented.
- The first tiny PyTorch imitation baseline exists.
- Event decoding and predicted macro export are implemented.
- scripts/predict_imitation_macro.py decodes predictions.jsonl or runs a saved
  checkpoint over a prepared dataset.
- pytest currently passes: 107 passed.
- artifacts/ is ignored and must remain local-only.

Existing real smoke artifact:
- artifacts/phase7_single_jump_capture
- artifacts/phase7_single_jump_dataset
- artifacts/imitation_baseline_smoke
- macro: examples/macros/single_jump.json
- dataset: 120 samples, press label tick 20, release label tick 30
- image loader smoke: 4 x 12 x 16 frame stacks, positive ticks [20, 30]
- decoded predicted macro:
  artifacts/imitation_baseline_smoke/predicted_macro/predicted_macro.json
- decoded events: press tick 20, release tick 30

Optional task:
Replay the predicted macro through the queued Geode replay checker and compare
progress against the source demonstration.

Constraints:
- Do not treat imitation as the main project direction.
- Do not require zero-shot generalization yet.
- Do not use online levels or leaderboard submission.
- Do not commit artifacts, generated frames, traces, checkpoints, or model files.
- Prefer using the existing gd_imitation image dataset loader.
- If adding PyTorch or another ML dependency is needed, first check whether it
  is already installed locally. If it is not installed, either ask before adding
  the dependency or implement a dependency-free smoke baseline first.
```

---

## 14. Current RL Practice Handoff

Current main direction:

```text
Build a level-specific practicing RL agent that always acts through the
stochastic human mock model.
```

Use the existing imitation pieces only as optional support tools. Do not make
macro prediction, zero-shot generalization, online levels, or perfect bot
behavior the main objective.

Immediate next task:

```text
Implement the Phase 7 practice environment and reward loop.
```

Near-term infrastructure:

- Repeated-attempt runner for one small local/offline level.
- Observation and trace capture aligned to each attempt.
- Intended action/event log from the policy.
- Humanized executed input log after perception delay, motor delay, timing
  variance, miss/drop probability, and input constraints.
- Reward computation from progress, death, section survival, and clears.
- Practice summary with best progress, average progress, death histogram,
  reward curve, and per-attempt metadata.

First implementation target:

```text
scripted or random policy
-> intended events
-> human mock model
-> queued Geode replay or live bridge
-> trace/death/progress
-> reward summary
-> repeated attempt loop
```

Success criterion:

```text
The repo can run many local/offline attempts on the same level through the
human wrapper, persist intended vs executed event logs, compute rewards, and
show whether progress improves or degrades across attempts.
```

Prompt for next Codex chat:

```text
We are in the geometry_dash_ai repo. Continue from plan.md and
agent_plan.md.

Main objective:
Build the Phase 7 practice environment and reward loop for a level-specific
RL practice agent under a stochastic human mock model.

Current state:
- Phases 1 through 6 are implemented.
- Replay, trace capture, screenshot capture, humanized macro execution, and
  optional imitation dataset/baseline tools exist.
- pytest previously passed at 107 tests.
- artifacts/ is ignored and must remain local-only.

Task:
Add the near-term RL infrastructure:
- repeated-attempt runner
- intended action/event logging
- executed humanized input logging
- reward computation from trace/death/progress/clear data
- practice summary for one small local/offline level

Constraints:
- The RL policy outputs intended actions/events only.
- All intended outputs must pass through the human mock model before reaching
  Geometry Dash.
- Imitation components are optional bootstrap or diagnostics only.
- Do not focus on zero-shot generalization, online levels, leaderboard use, or
  perfect bot behavior.
- Prioritize a small local/offline level that can be attempted repeatedly.
```
