# Geometry Dash / Geode Bridge Contract

This folder contains the first real Geode bridge implementation.

The mod starts a local-only TCP server on `127.0.0.1:29430`, sends one
observation message from `GJBaseGameLayer::processCommands` while a Python client is connected,
accepts `action`, `load_macro`, and `reset` messages, and can either apply
live-sent jump press/release events on the next game update or replay a loaded
macro inside the mod by attempt tick.

The Python dummy server still exists for protocol tests, but the live workflow
now connects to the Geode mod directly.

## Safety Rules

- Use local or offline test levels only.
- Do not submit bot runs to online leaderboards.
- Keep FPS, CBF, and physics bypass settings in trace metadata.

## Transport

Default TCP endpoint:

```text
host: 127.0.0.1
port: 29430
encoding: UTF-8 JSON lines
protocol_version: 1
```

Every message is one JSON object followed by `\n`.

## Mod Responsibilities

The current Geode mod:

1. Send one `observation` message per `GJBaseGameLayer::processCommands`
   gameplay step.
2. Receive `action` messages from Python.
3. Apply `press` / `release` on the next gameplay update after receipt.
4. Receive `load_macro` messages and store a complete macro for deterministic
   replay.
5. Arm the loaded macro after reset and apply due events from inside the mod by
   attempt tick immediately before that `processCommands` step.
6. Receive `reset` messages and restart the attempt.

Trace saving is currently handled on the Python side.

## Observation Message

```json
{
  "version": 1,
  "type": "observation",
  "observation": {
    "tick": 123,
    "x": 1200.5,
    "y": 240.0,
    "y_vel": -3.5,
    "mode": "cube",
    "gravity": "normal",
    "percent": 12.3,
    "dead": false,
    "input_down": true,
    "x_vel": 8.0,
    "rotation": 90.0,
    "death_reason": null
  }
}
```

Required initial fields:

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

Optional fields currently accepted by Python:

```text
x_vel
rotation
death_reason
```

## Action Message

```json
{
  "version": 1,
  "type": "action",
  "event": {
    "tick": 123,
    "kind": "press",
    "player": "p1"
  }
}
```

`kind` must be:

```text
press
release
```

`player` must be:

```text
p1
p2
```

## Load Macro Message

```json
{
  "version": 1,
  "type": "load_macro",
  "events": [
    {
      "tick": 20,
      "kind": "press",
      "player": "p1"
    },
    {
      "tick": 30,
      "kind": "release",
      "player": "p1"
    }
  ],
  "metadata": {
    "level_name": "local_test"
  }
}
```

The mod stores the macro as inactive when it is loaded. A later `reset` arms the
macro from event zero, so the queued macro does not accidentally affect the
currently running attempt before the trial starts.

## Reset Message

```json
{
  "version": 1,
  "type": "reset",
  "reason": "death"
}
```

## Ack / Error Messages

```json
{
  "version": 1,
  "type": "ack",
  "tick": 123,
  "message": "macro loaded"
}
```

```json
{
  "version": 1,
  "type": "error",
  "message": "invalid action"
}
```

## Diagnostic Messages

Queued replay emits non-trace diagnostics such as:

```json
{
  "version": 1,
  "type": "diagnostic",
  "kind": "macro_event_applied",
  "tick": 20,
  "data": {
    "event_index": 0,
    "intended_tick": 20,
    "applied_tick": 20,
    "kind": "press",
    "player": "p1"
  }
}
```

Python stores these in `summary.json`. They are intentionally separate from the
canonical trace JSONL rows.

## Python Dummy Check

The current dummy bridge can be exercised with:

```powershell
.\.venv\Scripts\python.exe scripts\run_dummy_bridge_macro.py macro.json trace.jsonl
```

The dummy server is not a Geometry Dash simulator. It only verifies the bridge
wire format and Python client behavior.

## Live Geode Check

Build the mod:

```powershell
cmake -S geode_mod -B geode_mod/build
cmake --build geode_mod/build --config RelWithDebInfo
```

If MSVC or Geode codegen fails because the repo path contains non-ASCII
characters, build through an ASCII-only checkout or junction instead:

```powershell
cmake -S C:\path\to\geometry_dash_ai\geode_mod -B C:\path\to\geometry_dash_ai\geode_mod\build
cmake --build C:\path\to\geometry_dash_ai\geode_mod\build --config RelWithDebInfo
```

Install `geode_mod/build/azberjibiou.geometry_dash_ai.geode` into your Geometry
Dash `geode/mods` folder, launch Geometry Dash, and open a local/offline test
level.

Create a tiny macro such as:

```json
{
  "metadata": {
    "level_name": "local_test"
  },
  "events": [
    {
      "tick": 20,
      "kind": "press",
      "player": "p1"
    },
    {
      "tick": 30,
      "kind": "release",
      "player": "p1"
    }
  ]
}
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_geode_bridge_macro.py macro.json trace.jsonl --max-observations 600
```

The script connects to `127.0.0.1:29430`, sends macro events as observations
arrive, and writes the resulting trace as JSONL.

## Live Deterministic Replay Check

After the one-shot bridge check works, run repeated trials with one identical
macro:

```powershell
.\.venv\Scripts\python.exe scripts\run_geode_replay_check.py examples\macros\single_jump.json --trials 5 --max-observations 600
```

By default, this loads the full macro into the mod before the trials and lets
the mod apply events by attempt tick. To run the older live-send path for smoke
testing, add `--live-send`.

Other tiny checked-in macros:

```text
examples/macros/no_input.json
examples/macros/single_jump.json
examples/macros/short_repeated_clicks.json
```

The replay-check script:

1. Connects to the live bridge on `127.0.0.1:29430`.
2. Loads the macro into the mod once, unless `--live-send` is set.
3. Sends a reset and waits for a fresh tick-0 observation before each trial.
4. Collects observations while the mod-side macro queue applies due events.
5. Saves per-trial traces under `artifacts/replay_check_<timestamp>/`.
6. Writes `summary.json` with deterministic replay metrics.

Important summary fields:

```text
final_percent_std
death_tick_std
x_position_max_diff
y_position_max_diff
success_rate
survival_rate
input_state_mismatch_ticks
input_latency_by_event
first_movement_ticks
zero_movement_step_counts
double_movement_step_counts
macro_application_by_event
```

Keep these live checks on local/offline levels only. `artifacts/` is ignored, so
the generated traces and summaries should remain local.
