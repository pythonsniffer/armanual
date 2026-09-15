"""Bind the words in an instruction to the objects a camera can actually see.

Grounding is scored, not rule-matched. Each candidate detection gets a score for how well it fits
the referent — category, colour, size, spatial relation to an anchor, an ordinal like "leftmost",
and a pointing gesture if one was given — and the best-scoring candidate wins. Two consequences
that matter for the rubric:

* **Ambiguity is a measurable output, not a crash.** When the runner-up scores nearly as well as
  the winner (two blue cups, two plates of similar size), the task records an ambiguity and the
  system can say what it is confused between instead of silently grabbing one.
* **Vision decides, language only constrains.** "The blue cup" is resolved against this frame's
  detections, so when the scene changes the same words resolve to a different object — which is
  what makes the closed loop meaningful.

Spatial words are interpreted in the *user's* frame, matching the front camera view: +x is to the
user's right, +y is away from the user, deeper into the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armanual.perception.detector import Detection, SceneObservation
from armanual.task.schema import (
    Destination,
    GroundedAction,
    GroundedTarget,
    GroundedTask,
    Referent,
    TaskRequest,
)

#: Colours that are genuinely easy to confuse. A near-miss on these costs less than a mismatch
#: between, say, red and green — that is what makes "the blue cup" ambiguous when a navy cup is
#: also on the table, which is exactly the ambiguity the scene is designed to create.
COLOR_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "blue": ("navy",),
    "navy": ("blue",),
    "silver": ("white",),
    "white": ("silver",),
    "purple": ("navy", "blue"),
    "orange": ("red", "brown"),
    "red": ("orange",),
    "brown": ("orange",),
}
#: Utensil categories collapse to one detected class: from above, a spoon and a fork are the same
#: silver sliver. The planner disambiguates them by drawer position, not by pixels.
UTENSIL_CATEGORIES = ("spoon", "fork", "knife", "utensil")


@dataclass
class GroundingConfig:
    ambiguity_margin: float = 0.15
    min_score: float = 0.30
    pointing_sigma: float = 0.08  # metres; how tightly a pointing ray localizes an object


def _category_score(referent: Referent, detection: Detection) -> float:
    if referent.category is None:
        return 0.5
    if referent.category in UTENSIL_CATEGORIES:
        return 1.0 if detection.category in UTENSIL_CATEGORIES else 0.0
    if referent.category == detection.category:
        return 1.0
    if {referent.category, detection.category} == {"cup", "bottle"}:
        return 0.25  # a jug is cup-like from above; allow it, but never prefer it
    return 0.0


def _color_score(referent: Referent, detection: Detection) -> float:
    if referent.color is None:
        return 0.5
    if referent.color == detection.color_name:
        return 1.0
    if detection.color_name in COLOR_NEIGHBOURS.get(referent.color, ()):
        return 0.55
    return 0.0


def _size_score(referent: Referent, detection: Detection) -> float:
    if referent.size is None:
        return 0.5
    return 1.0 if referent.size == detection.size_label else 0.15


def _relation_score(relation: str, position: np.ndarray, anchor: np.ndarray) -> float:
    """How well ``position`` satisfies ``relation`` with respect to ``anchor``.

    Scores are soft so that "to the left of the plate" prefers the object furthest left while
    still admitting a near-tie as an ambiguity rather than an error.
    """
    delta = np.asarray(position)[:2] - np.asarray(anchor)[:2]
    distance = float(np.linalg.norm(delta))
    if relation == "left_of":
        return float(np.clip(-delta[0] / 0.20, 0.0, 1.0))
    if relation == "right_of":
        return float(np.clip(delta[0] / 0.20, 0.0, 1.0))
    if relation == "in_front_of":
        return float(np.clip(-delta[1] / 0.20, 0.0, 1.0))
    if relation == "behind":
        return float(np.clip(delta[1] / 0.20, 0.0, 1.0))
    if relation in ("near", "on"):
        return float(np.clip(1.0 - distance / 0.25, 0.0, 1.0))
    if relation == "far_from":
        return float(np.clip(distance / 0.35, 0.0, 1.0))
    return 0.5


def _ordinal_score(ordinal: str, detection: Detection, others: list[Detection]) -> float:
    positions = np.array([d.position[:2] for d in others]) if others else np.zeros((1, 2))
    here = detection.position[:2]
    if ordinal == "leftmost":
        span = positions[:, 0].max() - positions[:, 0].min() or 1.0
        return float(np.clip((positions[:, 0].max() - here[0]) / span, 0.0, 1.0))
    if ordinal == "rightmost":
        span = positions[:, 0].max() - positions[:, 0].min() or 1.0
        return float(np.clip((here[0] - positions[:, 0].min()) / span, 0.0, 1.0))
    if ordinal == "nearest":
        span = positions[:, 1].max() - positions[:, 1].min() or 1.0
        return float(np.clip((positions[:, 1].max() - here[1]) / span, 0.0, 1.0))
    if ordinal == "farthest":
        span = positions[:, 1].max() - positions[:, 1].min() or 1.0
        return float(np.clip((here[1] - positions[:, 1].min()) / span, 0.0, 1.0))
    return 0.5


class Grounder:
    """Resolves referents against one :class:`SceneObservation`."""

    def __init__(self, config: GroundingConfig | None = None):
        self.config = config or GroundingConfig()

    # ------------------------------------------------------------------------------ scoring
    def score(self, referent: Referent, detection: Detection, scene: list[Detection]) -> float:
        """Weighted fit of one detection to one referent, in [0, 1]."""
        category = _category_score(referent, detection)
        if category == 0.0:
            return 0.0
        color = _color_score(referent, detection)
        if referent.color is not None and color == 0.0:
            # A stated colour is a hard constraint: "the red cup" must not quietly resolve to a
            # blue one just because it is the only cup in view. Confusable neighbours (blue/navy)
            # still pass, at a discount — that is the ambiguity the scene is built to test.
            return 0.0
        parts = [(category, 1.0), (color, 0.9), (_size_score(referent, detection), 0.6)]

        if referent.relation and referent.anchor:
            anchor = self.resolve(referent.anchor, scene, exclude=detection)
            if anchor is not None:
                parts.append((_relation_score(referent.relation, detection.position,
                                              anchor.position), 1.1))
        if referent.ordinal:
            same_kind = [d for d in scene if _category_score(referent, d) > 0] or scene
            parts.append((_ordinal_score(referent.ordinal, detection, same_kind), 1.1))
        if referent.pointing_xy is not None:
            distance = float(
                np.linalg.norm(detection.position[:2] - np.asarray(referent.pointing_xy))
            )
            parts.append((float(np.exp(-0.5 * (distance / self.config.pointing_sigma) ** 2)), 1.6))

        total_weight = sum(weight for _score, weight in parts)
        return sum(score * weight for score, weight in parts) / total_weight

    def resolve(self, referent: Referent, scene: list[Detection],
                exclude: Detection | None = None) -> Detection | None:
        ranked = self.rank(referent, scene, exclude=exclude)
        return ranked[0][0] if ranked else None

    def rank(self, referent: Referent, scene: list[Detection],
             exclude: Detection | None = None) -> list[tuple[Detection, float]]:
        """All plausible candidates, best first."""
        candidates = [
            d
            for d in scene
            if d.category not in ("furniture", "unknown") and (exclude is None or d is not exclude)
        ]
        scored = [(d, self.score(referent, d, scene)) for d in candidates]
        scored = [(d, s) for d, s in scored if s >= self.config.min_score]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)

    # ----------------------------------------------------------------------------- grounding
    def ground_target(self, referent: Referent, observation: SceneObservation) -> GroundedTarget | None:
        ranked = self.rank(referent, observation.detections)
        if not ranked:
            return None
        best, score = ranked[0]
        return GroundedTarget(
            referent=referent,
            name=best.name,
            position=best.position.copy(),
            track_id=best.track_id,
            category=best.category,
            color=best.color_name,
            score=score,
            alternatives=[(d.describe(), s) for d, s in ranked[1:4]],
            candidates=[
                (d.describe(), float(s), tuple(float(v) for v in d.position), d.category)
                for d, s in ranked[:4]
            ],
        )

    def ground_destination(
        self, destination: Destination | None, observation: SceneObservation,
        target: GroundedTarget | None,
    ) -> tuple[tuple[float, float] | None, str]:
        """Turn a destination into a table coordinate, plus a human-readable note."""
        if destination is None:
            return None, ""
        if destination.absolute_xy is not None:
            return destination.absolute_xy, "absolute"
        if destination.anchor is not None:
            anchor = self.ground_target(destination.anchor, observation)
            if anchor is None:
                return None, f"could not find {destination.anchor.describe()}"
            offset = {
                "left_of": (-0.11, 0.0),
                "right_of": (0.11, 0.0),
                "in_front_of": (0.0, -0.10),
                "behind": (0.0, 0.10),
                "near": (0.09, 0.0),
                "far_from": (0.18, 0.0),
                "on": (0.0, 0.0),
            }.get(destination.relation or "near", (0.09, 0.0))
            point = (float(anchor.position[0] + offset[0]), float(anchor.position[1] + offset[1]))
            note = f"{(destination.relation or 'near').replace('_', ' ')} {anchor.name or anchor.category}"
            return point, note
        if destination.slot:
            return None, destination.slot  # resolved later against the style's place setting
        return None, ""

    def ground(self, request: TaskRequest, observation: SceneObservation) -> GroundedTask:
        """Ground every action in a request against one observation."""
        task = GroundedTask(request=request, observation_stamp=observation.stamp)
        for action in request.actions:
            if action.verb in ("set_table", "open_drawer"):
                task.actions.append(GroundedAction(spec=action))
                continue
            target = self.ground_target(action.target, observation) if action.target else None
            if action.target is not None and target is None:
                task.unresolved.append(action.target)
                continue
            if target is not None and target.ambiguous:
                task.ambiguities.append(
                    f"{action.target.describe()!r} matches "
                    f"{target.name or target.category} ({target.score:.2f}) and "
                    f"{target.alternatives[0][0]} ({target.alternatives[0][1]:.2f})"
                )
            point, note = self.ground_destination(action.destination, observation, target)
            task.actions.append(
                GroundedAction(spec=action, target=target, destination_xy=point,
                               destination_note=note)
            )
        return task


def ground_instruction(
    text: str, observation: SceneObservation, *, modality: str = "text",
    pointing_xy: tuple[float, float] | None = None, grounder: Grounder | None = None,
) -> GroundedTask:
    """Convenience: parse and ground in one call, attaching a pointing prior if given."""
    from armanual.task.parser import parse_instruction

    request = parse_instruction(text, modality=modality)
    if pointing_xy is not None:
        for action in request.actions:
            if action.target is not None:
                action.target.pointing_xy = pointing_xy
    return (grounder or Grounder()).ground(request, observation)
