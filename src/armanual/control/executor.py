"""Cooperative scheduler that advances several arm skills against one simulator clock.

Each skill is a generator that writes actuator targets and yields once per control tick; the
scheduler steps the physics. Two consequences the bimanual task needs:

* **true simultaneity** — two arms can run different skills in the same tick, so "hold the mug
  while the other arm pours" is one scheduler call, not a scripted interleaving, and
* **interruptibility** — the scheduler can stop a skill mid-motion when re-observation says the
  plan is stale, which is what makes replanning possible at all.

Physics runs at the model timestep (200 Hz); skills act at ``dt`` (default 20 Hz), the rate a
learned policy would also run at.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Skill = Generator[None, None, Any]


class Status(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


class SkillFailure(RuntimeError):
    """Raised inside a skill when it cannot continue (bad IK, lost grasp, ...)."""

    def __init__(self, message: str, *, kind: str = "manipulation"):
        super().__init__(message)
        self.kind = kind


@dataclass
class TickRecord:
    t: float
    active: tuple[str, ...]


@dataclass
class RunResult:
    status: Status
    seconds: float
    error: str | None = None
    failed_arm: str | None = None
    kind: str | None = None
    ticks: list[TickRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is Status.SUCCESS


class Scheduler:
    """Runs skills for one or more arms, stepping the world between ticks."""

    def __init__(self, world, dt: float = 0.05, max_seconds: float = 90.0):
        self.world = world
        self.dt = dt
        self.max_seconds = max_seconds
        self.steps_per_tick = max(1, int(round(dt / world.model.opt.timestep)))
        self.on_tick: list[Callable[[float], None]] = []

    def run(
        self,
        skills: dict[str, Skill],
        *,
        max_seconds: float | None = None,
        watchdog: Callable[[], str | None] | None = None,
    ) -> RunResult:
        """Advance ``skills`` (keyed by arm) until all finish, one fails, or time runs out.

        ``watchdog`` is polled every tick; returning a string aborts the run with that reason —
        this is the hook the closed-loop executor uses to interrupt on a scene change.
        """
        budget = max_seconds if max_seconds is not None else self.max_seconds
        started = self.world.time
        wall = time.perf_counter()
        active = dict(skills)
        ticks: list[TickRecord] = []

        while active:
            if self.world.time - started > budget:
                return RunResult(Status.TIMEOUT, self.world.time - started,
                                 error=f"exceeded {budget:.1f}s of sim time",
                                 kind="timeout", ticks=ticks)
            finished: list[str] = []
            for arm, skill in list(active.items()):
                try:
                    next(skill)
                except StopIteration:
                    finished.append(arm)
                except SkillFailure as exc:
                    return RunResult(Status.FAILURE, self.world.time - started, error=str(exc),
                                     failed_arm=arm, kind=exc.kind, ticks=ticks)
            for arm in finished:
                active.pop(arm, None)

            self.world.step(self.steps_per_tick)
            ticks.append(TickRecord(t=self.world.time, active=tuple(active)))
            for hook in self.on_tick:
                hook(self.world.time)

            if watchdog is not None and (reason := watchdog()):
                return RunResult(Status.ABORTED, self.world.time - started, error=reason,
                                 kind="replan", ticks=ticks)

        _ = wall
        return RunResult(Status.SUCCESS, self.world.time - started, ticks=ticks)

    def run_sequence(self, arm: str, skills: Iterable[Callable[[], Skill]], **kwargs) -> RunResult:
        """Run skills one after another on a single arm, stopping at the first failure."""
        for make in skills:
            result = self.run({arm: make()}, **kwargs)
            if not result.ok:
                return result
        return RunResult(Status.SUCCESS, 0.0)
