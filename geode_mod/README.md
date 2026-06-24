# Geometry Dash / Geode Bridge Contract

This folder will hold the real Geode mod later.

For Phase 3, the implemented piece is the Python-side bridge and a dummy TCP
server that speaks the same JSON-line protocol. The dummy server exists so the
Python client, trace saving, reset handling, and action messages can be tested
before any Geometry Dash hooks are written.

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

The future Geode mod should:

1. Send one `observation` message per physics tick.
2. Receive `action` messages from Python.
3. Apply `press` / `release` on the next physics tick after receipt.
4. Receive `reset` messages and restart the attempt.
5. Optionally log the same observations locally as JSONL.

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
