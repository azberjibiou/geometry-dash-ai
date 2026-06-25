"""Live Geode bridge executor for RL-practice attempts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from gd_env import BridgeDiagnostic, BridgeObservation, GeometryDashClient
from gd_trace import Macro, TraceRow


class GeodeClientLike(Protocol):
    """Subset of GeometryDashClient used by the practice executor."""

    def connect(self) -> "GeodeClientLike":
        ...

    def close(self) -> None:
        ...

    def load_macro(
        self,
        events: Sequence[Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        ...

    def run_loaded_macro(
        self,
        *,
        max_observations: int,
        fps: int = 240,
        cbf: bool = False,
        physics_bypass: bool = False,
        trace_path: str | Path | None = None,
        initial_observation: BridgeObservation | None = None,
        diagnostics: list[BridgeDiagnostic] | None = None,
        stop_percent: float | None = None,
    ) -> list[TraceRow]:
        ...


@dataclass(frozen=True, slots=True)
class GeodeExecutorConfig:
    """Settings for live queued-macro execution through the Geode bridge."""

    host: str = "127.0.0.1"
    port: int = 29430
    timeout_seconds: float = 5.0
    max_observations: int = 1200
    reset_wait_observations: int = 600
    fps: int = 240
    cbf: bool = False
    physics_bypass: bool = False
    success_percent: float = 100.0
    stop_on_success: bool = False
    post_terminal_delay_seconds: float = 0.0
    start_guard_reset_retries: int = 0
    start_guard_retry_delay_seconds: float = 0.0
    require_start_percent_max: float | None = None
    require_start_x_max: float | None = None
    require_progress_tick: int | None = None
    require_progress_percent_min: float | None = None

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("port must be positive")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_observations <= 0:
            raise ValueError("max_observations must be positive")
        if self.reset_wait_observations <= 0:
            raise ValueError("reset_wait_observations must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= self.success_percent <= 100.0:
            raise ValueError("success_percent must be between 0 and 100")
        if self.post_terminal_delay_seconds < 0.0:
            raise ValueError("post_terminal_delay_seconds must be non-negative")
        if self.start_guard_reset_retries < 0:
            raise ValueError("start_guard_reset_retries must be non-negative")
        if self.start_guard_retry_delay_seconds < 0.0:
            raise ValueError("start_guard_retry_delay_seconds must be non-negative")
        if (
            self.require_start_percent_max is not None
            and not 0.0 <= self.require_start_percent_max <= 100.0
        ):
            raise ValueError("require_start_percent_max must be between 0 and 100")
        if self.require_start_x_max is not None and self.require_start_x_max < 0.0:
            raise ValueError("require_start_x_max must be non-negative")
        if self.require_progress_tick is not None and self.require_progress_tick < 0:
            raise ValueError("require_progress_tick must be non-negative")
        if (
            self.require_progress_percent_min is not None
            and not 0.0 <= self.require_progress_percent_min <= 100.0
        ):
            raise ValueError("require_progress_percent_min must be between 0 and 100")
        if (self.require_progress_tick is None) != (
            self.require_progress_percent_min is None
        ):
            raise ValueError(
                "require_progress_tick and require_progress_percent_min "
                "must be used together"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeodePracticeExecutor:
    """PracticeAttemptExecutor implementation backed by the live bridge."""

    def __init__(
        self,
        config: GeodeExecutorConfig | None = None,
        *,
        client_factory: Callable[[], GeodeClientLike] | None = None,
    ) -> None:
        self.config = config or GeodeExecutorConfig()
        self._client_factory = client_factory or self._default_client_factory
        self._client: GeodeClientLike | None = None

    def __enter__(self) -> "GeodePracticeExecutor":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    def connect(self) -> "GeodePracticeExecutor":
        if self._client is None:
            self._client = self._client_factory().connect()
        return self

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def run_attempt(
        self,
        *,
        attempt_index: int,
        executed_macro: Macro,
        attempt_dir: Path,
        metadata: dict[str, Any],
    ) -> Sequence[TraceRow]:
        """Load an executed macro, reset, collect trace, and save diagnostics."""

        client = self._ensure_client()
        diagnostics: list[BridgeDiagnostic] = []
        macro_metadata = dict(executed_macro.metadata)
        macro_metadata.update(
            {
                "attempt_index": attempt_index,
                "executor": "geode_queued_macro",
                "practice_metadata": dict(metadata),
            }
        )

        client.load_macro(executed_macro.events, metadata=macro_metadata)
        initial_observation, reset_attempts = self._reset_until_fresh_start(
            client,
            attempt_index=attempt_index,
            diagnostics=diagnostics,
        )

        rows = client.run_loaded_macro(
            max_observations=self.config.max_observations,
            fps=self.config.fps,
            cbf=self.config.cbf,
            physics_bypass=self.config.physics_bypass,
            initial_observation=initial_observation,
            diagnostics=diagnostics,
            stop_percent=(
                self.config.success_percent if self.config.stop_on_success else None
            ),
        )
        if (
            self.config.post_terminal_delay_seconds > 0.0
            and _is_terminal_trace(rows, success_percent=self.config.success_percent)
        ):
            time.sleep(self.config.post_terminal_delay_seconds)

        _validate_trace_progress(
            rows,
            attempt_index=attempt_index,
            require_tick=self.config.require_progress_tick,
            require_percent_min=self.config.require_progress_percent_min,
        )
        _write_json(
            {
                "executor": "geode_queued_macro",
                "config": self.config.to_dict(),
                "metadata": dict(metadata),
                "reset_attempts": reset_attempts,
                "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
            },
            attempt_dir / "geode_diagnostics.json",
        )
        return rows

    def _reset_until_fresh_start(
        self,
        client: GeodeClientLike,
        *,
        attempt_index: int,
        diagnostics: list[BridgeDiagnostic],
    ) -> tuple[BridgeObservation, int]:
        last_error: TimeoutError | ValueError | None = None
        max_resets = self.config.start_guard_reset_retries + 1

        for reset_index in range(max_resets):
            try:
                initial_observation = client.reset_attempt(
                    f"rl_practice_attempt_{attempt_index}",
                    max_observations=self.config.reset_wait_observations,
                    diagnostics=diagnostics,
                )
                _validate_start_observation(
                    initial_observation,
                    attempt_index=attempt_index,
                    require_percent_max=self.config.require_start_percent_max,
                    require_x_max=self.config.require_start_x_max,
                )
                return initial_observation, reset_index + 1
            except (TimeoutError, ValueError) as exc:
                last_error = exc
                if reset_index + 1 >= max_resets:
                    raise
                if self.config.start_guard_retry_delay_seconds > 0.0:
                    time.sleep(self.config.start_guard_retry_delay_seconds)

        raise RuntimeError("unreachable reset retry state") from last_error

    def _ensure_client(self) -> GeodeClientLike:
        if self._client is None:
            self.connect()
        if self._client is None:
            raise RuntimeError("failed to initialize Geode client")
        return self._client

    def _default_client_factory(self) -> GeodeClientLike:
        return GeometryDashClient(
            host=self.config.host,
            port=self.config.port,
            timeout_seconds=self.config.timeout_seconds,
        )


def _validate_start_observation(
    observation: BridgeObservation,
    *,
    attempt_index: int,
    require_percent_max: float | None,
    require_x_max: float | None,
) -> None:
    failures = []
    if (
        require_percent_max is not None
        and observation.percent > require_percent_max
    ):
        failures.append(
            f"percent {observation.percent:.3f} > {require_percent_max:.3f}"
        )
    if require_x_max is not None and observation.x > require_x_max:
        failures.append(f"x {observation.x:.3f} > {require_x_max:.3f}")
    if failures:
        raise ValueError(
            f"attempt {attempt_index} fresh start check failed: "
            + "; ".join(failures)
        )


def _validate_trace_progress(
    rows: Sequence[TraceRow],
    *,
    attempt_index: int,
    require_tick: int | None,
    require_percent_min: float | None,
) -> None:
    if require_tick is None or require_percent_min is None:
        return

    row = next((candidate for candidate in rows if candidate.tick >= require_tick), None)
    if row is None:
        raise ValueError(
            f"attempt {attempt_index} ended before progress guard tick {require_tick}"
        )
    if row.percent < require_percent_min:
        raise ValueError(
            f"attempt {attempt_index} progress check failed at tick {row.tick}: "
            f"percent {row.percent:.3f} < {require_percent_min:.3f}"
        )


def _is_terminal_trace(
    rows: Sequence[TraceRow],
    *,
    success_percent: float,
) -> bool:
    if not rows:
        return False
    last = rows[-1]
    return last.dead or last.percent >= success_percent


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
