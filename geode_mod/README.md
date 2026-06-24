# Geometry Dash / Geode Bridge Contract

This folder contains the first real Geode bridge implementation.

The mod starts a local-only TCP server on `127.0.0.1:29430`, sends one
observation message per gameplay update while a Python client is connected,
accepts `action` and `reset` messages, and applies received jump press/release
events on the next game update.

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

1. Send one `observation` message per physics tick.
2. Receive `action` messages from Python.
3. Apply `press` / `release` on the next gameplay update after receipt.
4. Receive `reset` messages and restart the attempt.

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
  "message": "action queued"
}
```

```json
{
  "version": 1,
  "type": "error",
  "message": "invalid action"
}
```

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

Other tiny checked-in macros:

```text
examples/macros/no_input.json
examples/macros/single_jump.json
examples/macros/short_repeated_clicks.json
```

The replay-check script:

1. Connects to the live bridge on `127.0.0.1:29430`.
2. Sends a reset and waits for a fresh tick-0 observation before each trial.
3. Runs the same macro for each trial.
4. Saves per-trial traces under `artifacts/replay_check_<timestamp>/`.
5. Writes `summary.json` with deterministic replay metrics.

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
```

Keep these live checks on local/offline levels only. `artifacts/` is ignored, so
the generated traces and summaries should remain local.
