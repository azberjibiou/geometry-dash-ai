# Geometry Dash Human-like Practice-Learning AI Plan

## 0. Project Goal

Build an AI system that plays Geometry Dash in a **human-like** way, can practice a specific level over many attempts, and can build level-specific muscle memory while keeping a reusable general policy.

The goal is **not** to build a perfect bot.

The goal is to model a player who:

- observes the screen with delay,
- makes decisions with delay,
- clicks with motor delay,
- has timing variance,
- becomes less consistent when clicks are too close together,
- may become slightly less precise after long gaps,
- can be configured as Beginner / Intermediate / Advanced / Top Player,
- can run many practice attempts to measure learning progress, clear rate, playtime, attempts, and death distribution,
- can remember level-specific timing patterns after practice, like human muscle memory.

Final use case:

```text
Given a Geometry Dash level, a general agent, and a human profile,
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

---

## 1. Core Principle

The project should separate four layers:

```text
Geometry Dash Environment
        ↓
General AI Policy
        ↓
Level Memory / Practice Adapter
        ↓
Human-like Input Wrapper
```

The general AI policy should decide what it *currently thinks* should be done from observation.

The level memory / practice adapter should use previous attempts on the same level to adjust timing, prefer learned event windows, and build muscle memory.

The human wrapper should decide when that intended input actually reaches the game.

This separation is important because we want to distinguish:

```text
The general AI understands the situation
vs.
The level memory has learned this exact timing
vs.
The human-like player fails to execute it precisely
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
| General AI Policy           |
|                             |
| input: delayed observation  |
| output: candidate events    |
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

### Phase 7 — Imitation Learning on One Level

Goal:

Train a policy to imitate a recorded playthrough on a single short level.

Initial policy:

```text
input: last 4 grayscale frames + progress
output:
  - press_event probability
  - release_event probability
```

Training data:

```text
recorded frame sequence
recorded input event sequence
```

Important:

Because of visual and motor delay, labels should be shifted.

If total delay is:

```text
D = visual_delay_frames + motor_delay_frames
```

then frame at tick `t` should learn to predict events needed around tick `t + D`.

Success criterion:

```text
The learned policy can reproduce meaningful progress on the same level.
```

Do not require zero-shot generalization yet.

---

### Phase 8 — Event Window Labels

Goal:

Avoid overly brittle single-tick labels.

Instead of only labeling the exact recorded press tick, estimate a valid event window.

Possible labels:

```text
is_press_window
is_release_window
frames_to_next_press
frames_to_next_release
```

Why:

In Geometry Dash, many jumps allow a small range of valid click timings.

Single-tick supervised labels can incorrectly mark other valid ticks as negative.

Success criterion:

```text
Training becomes less brittle and the policy tolerates small timing variations better.
```

---

### Phase 9 — Level Memory / Muscle Memory Adapter

Goal:

Make repeated attempts improve performance on the same level.

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
general policy output
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
death position
clear/failure result
```

and update the level memory after each attempt.

Success criterion:

```text
On a fixed level, the agent improves over repeated attempts
without hard-coding a final exact macro by hand.
```

---

### Phase 10 — Practice Learning Evaluation

Goal:

Measure how well the general agent learns a specific level through practice.

For each level and human profile:

```text
run repeated practice attempts
store traces, deaths, memory updates, and clears
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
memory_confidence_by_section
sensitivity_to_delay
sensitivity_to_std
sensitivity_to_close_amp
```

Do not require a single difficulty score for early versions.

Use attempts, playtime, progress, and consistency as the primary measurements.

Success criterion:

```text
Repeated practice produces a visible learning curve:
best progress increases, repeated deaths move later or disappear,
and clear rate improves after memory is learned.
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
2. Do not train RL immediately.
3. Do not support all game modes.
4. Do not support online levels at scale.
5. Do not optimize for speed before correctness.
6. Do not use the bot for online leaderboard submission.
7. Do not require zero-shot generalization to unseen levels before the practice-learning loop works.
```

---

## 11. Main Risks

### Risk 1 — Environment bridge is unstable

If the same macro does not replay consistently, AI training becomes meaningless.

Mitigation:

```text
Build deterministic replay tests before training.
```

### Risk 2 — Supervised labels are ambiguous

There may be multiple valid click frames, but the dataset contains only one.

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
Phase 4:
Deterministic Replay Check
```

Reason:

Before adding screenshots, policy learning, or practice memory, verify that the
real Geometry Dash bridge gives stable outcomes for repeated identical macros.
If identical input events do not replay consistently under fixed settings, the
learning loop will be hard to trust.

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

Next target:

```text
Phase 4 hardening follow-up:
authoritative attempt tick / fixed physics-step observation alignment

Investigate whether a better Geode hook can emit exactly one observation per
actual physics step and reset the attempt tick at the same point in the real
level reset lifecycle.
```

---

## 13. Prompt for Next Codex Chat

Copy this into the next Codex chat:

```text
We are in the geometry_dash_ai repo. Please continue from plan.md after the
Phase 4 hardening implementation.

Current status:
- Phase 1, Phase 2, Phase 3, initial Phase 4 replay metrics, and Phase 4
  hardening are implemented.
- The Geode mod builds successfully with:
  C:\Program Files\CMake\bin\cmake.exe
- The replay script uses mod-side queued macro replay by default.
- `--live-send` keeps the older Python live action-send path and clears stale
  queued macro state before comparison.
- Live queued replay validation was run on a local/offline level.
- artifacts/ is ignored and should remain local-only.

Live validation:
- Queued artifact: artifacts/replay_check_20260624_203558
- Live-send artifact: artifacts/replay_check_20260624_203852
- Queued replay applied press at tick 20 and release at tick 30 in every trial.
- Queued observed input_down transitions also happened at ticks 20 and 30 in
  every trial.
- Live-send observed press at tick 21 in every trial and release at tick 32 or
  33.
- Queued path removed live TCP input latency variance.
- Remaining drift appears to come from update/physics observation alignment:
  zero/double movement step counts and first movement tick variation still
  appear in traces.
- pytest passed locally: 42 passed.

Task:
Continue Phase 4 hardening follow-up: make observation ticks authoritative or
identify the correct Geode hook for exactly one observation per real physics
step.

Please:
1. Inspect the current Geode hook and replay diagnostics.
2. Research locally in the Geode/GD bindings for a better physics-step hook or
   stable attempt tick source.
3. Implement a focused change that aligns observations to the real physics
   update if possible.
4. Preserve the queued macro path and canonical trace format.
5. Keep live checks manual and local/offline only.
6. Run synthetic tests and, if Geometry Dash is open on a local/offline level,
   rerun queued replay to compare first_movement_ticks, zero_movement_steps,
   double_movement_steps, and position drift.

Constraints:
- Use local/offline levels only.
- Do not use online leaderboard submission.
- Do not commit generated artifacts or live traces.
- Do not move to Phase 5 until update/physics tick alignment is understood.
```
