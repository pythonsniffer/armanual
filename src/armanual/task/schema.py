"""The single task representation every input modality produces.

Text, speech and pointing are three ways of filling in the *same* structure. Nothing downstream
knows or cares which one was used — that is the whole point of the design, and it is what keeps
speech from turning into a second, parallel robot controller.

The chain is:

``raw input -> TaskRequest (referents unresolved) -> GroundedTask (referents bound to detections)
-> Plan (ordered steps with arms assigned) -> execution``

Each stage is a plain dataclass that can be serialized into an evaluation record, so a failure can
be attributed to the stage that caused it: a misheard word, an unresolved referent, a bad plan, or
a dropped cup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

Modality = Literal["text", "speech", "pointing", "config"]

#: Spatial relations the grounder understands, and how they read in the table frame
#: (+x is to the user's right, +y is away from the user).
RELATIONS = ("left_of", "right_of", "near", "far_from", "in_front_of", "behind", "on")

#: Verbs the planner can execute. Anything else is reported as unsupported rather than guessed at.
VERBS = ("place", "pick", "pour", "open_drawer", "handoff", "set_table", "stack", "clear")


@dataclass
class Referent:
    """An unresolved reference to something on the table, as the user described it."""

    text: str = ""
    category: str | None = None
    color: str | None = None
    size: str | None = None
    relation: str | None = None
    anchor: "Referent | None" = None
    ordinal: str | None = None  # "leftmost", "rightmost", "nearest", "farthest"
    #: Set when the user pointed instead of (or as well as) describing.
    pointing_xy: tuple[float, float] | None = None

    def describe(self) -> str:
        parts = [p for p in (self.size, self.color, self.category) if p]
        phrase = " ".join(parts) if parts else (self.text or "object")
        if self.relation and self.anchor:
            phrase += f" {self.relation.replace('_', ' ')} the {self.anchor.describe()}"
        if self.ordinal:
            phrase = f"{self.ordinal} {phrase}"
        return phrase

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.category, self.color, self.size, self.relation, self.ordinal, self.pointing_xy)
        )


@dataclass
class Destination:
    """Where something should end up: a named place setting slot, or a spatial relation."""

    slot: str | None = None  # e.g. "place_setting", "tray", "left_of_plate"
    relation: str | None = None
    anchor: Referent | None = None
    absolute_xy: tuple[float, float] | None = None

    def describe(self) -> str:
        if self.slot:
            return self.slot.replace("_", " ")
        if self.relation and self.anchor:
            return f"{self.relation.replace('_', ' ')} the {self.anchor.describe()}"
        if self.absolute_xy:
            return f"({self.absolute_xy[0]:.2f}, {self.absolute_xy[1]:.2f})"
        return "the table"


@dataclass
class ActionSpec:
    """One requested action, still in the user's terms."""

    verb: str
    target: Referent | None = None
    destination: Destination | None = None
    arm_hint: str | None = None  # honoured when feasible; the planner may override and say so
    quantity: int = 1

    def describe(self) -> str:
        text = self.verb.replace("_", " ")
        if self.target:
            text += f" the {self.target.describe()}"
        if self.destination:
            text += f" -> {self.destination.describe()}"
        if self.arm_hint:
            text += f" [{self.arm_hint} arm requested]"
        return text


@dataclass
class TaskRequest:
    """A parsed instruction: what the user wants, before looking at the scene."""

    text: str = ""
    modality: Modality = "text"
    actions: list[ActionSpec] = field(default_factory=list)
    style: str | None = None  # table-setting style, when one was named
    confidence: float = 1.0
    #: Populated by the speech path; lets evaluation separate mishearing from misunderstanding.
    transcript_alternatives: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        head = f"[{self.modality}] {self.text!r}"
        body = "; ".join(a.describe() for a in self.actions) or "(no actions parsed)"
        style = f" style={self.style}" if self.style else ""
        return f"{head} -> {body}{style}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundedTarget:
    """A referent bound to something actually observed."""

    referent: Referent
    name: str | None  # scene object name, when association provides one
    position: np.ndarray
    track_id: int | None = None
    category: str = ""
    color: str = ""
    score: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    #: Runner-up interpretations as (description, score, position, category). The planner falls
    #: back to these when the best match turns out to be unreachable — an unreachable winner
    #: usually means the detector produced a phantom, and the second choice is the real object.
    candidates: list[tuple[str, float, tuple[float, float, float], str]] = field(
        default_factory=list
    )

    @property
    def ambiguous(self) -> bool:
        """True when the runner-up is nearly as good a match as the winner."""
        if not self.alternatives:
            return False
        return (self.score - self.alternatives[0][1]) < 0.15


@dataclass
class GroundedAction:
    """An action whose target (and destination anchor) are resolved to observed objects."""

    spec: ActionSpec
    target: GroundedTarget | None = None
    destination_xy: tuple[float, float] | None = None
    destination_note: str = ""

    def describe(self) -> str:
        target = self.target.name or self.target.referent.describe() if self.target else "-"
        where = self.destination_note or (
            f"({self.destination_xy[0]:.2f}, {self.destination_xy[1]:.2f})"
            if self.destination_xy
            else ""
        )
        return f"{self.spec.verb}({target}){' -> ' + where if where else ''}"


@dataclass
class GroundedTask:
    """The full request, grounded in one observation of the scene."""

    request: TaskRequest
    actions: list[GroundedAction] = field(default_factory=list)
    unresolved: list[Referent] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    observation_stamp: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.actions) and not self.unresolved

    def describe(self) -> str:
        lines = [a.describe() for a in self.actions]
        if self.unresolved:
            lines.append(f"unresolved: {[r.describe() for r in self.unresolved]}")
        if self.ambiguities:
            lines.append(f"ambiguous: {self.ambiguities}")
        return " | ".join(lines)
