"""Score the camera-based detector against simulator ground truth.

This is the one place allowed to read privileged state, and it exists so that claims about
perception are numbers rather than adjectives. It reports, per seed and aggregated:

* **recall** — of the objects that are genuinely visible, how many were detected,
* **precision** — of the detections, how many correspond to a real object,
* **position error** — planar distance between a detection and its matched object,
* **category / colour accuracy** — on matched detections only, so classification error is not
  confounded with detection error.

Objects hidden inside the closed drawer are excluded from recall: not seeing them is correct
behaviour, and counting them as misses would reward a detector that hallucinates through wood.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from armanual.perception.detector import Detection, SceneObservation

MATCH_RADIUS = 0.06


@dataclass
class PerceptionScore:
    seed: int = 0
    n_truth: int = 0
    n_detected: int = 0
    n_matched: int = 0
    position_errors_mm: list[float] = field(default_factory=list)
    category_correct: int = 0
    color_correct: int = 0
    false_positives: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_truth if self.n_truth else 1.0

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_detected if self.n_detected else 1.0

    @property
    def mean_position_error_mm(self) -> float:
        return float(np.mean(self.position_errors_mm)) if self.position_errors_mm else 0.0

    @property
    def category_accuracy(self) -> float:
        return self.category_correct / self.n_matched if self.n_matched else 1.0

    @property
    def color_accuracy(self) -> float:
        return self.color_correct / self.n_matched if self.n_matched else 1.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            recall=round(self.recall, 3),
            precision=round(self.precision, 3),
            mean_position_error_mm=round(self.mean_position_error_mm, 1),
            category_accuracy=round(self.category_accuracy, 3),
            color_accuracy=round(self.color_accuracy, 3),
        )
        data["position_errors_mm"] = [round(e, 1) for e in self.position_errors_mm]
        return data


def visible_objects(world) -> list:
    """Ground-truth objects a camera could reasonably see: on the table, not shut in the drawer."""
    spec = world.scene.drawer
    closed = spec.present and world.drawer_opening() < 0.03
    out = []
    for obj in world.scene.objects:
        position = world.object_pos(obj.name)
        inside_drawer = (
            closed
            and abs(position[0] - spec.pos[0]) < 0.10
            and abs(position[1] - spec.pos[1]) < 0.10
        )
        if not inside_drawer:
            out.append((obj, position))
    return out


def score_observation(world, observation: SceneObservation, seed: int = 0) -> PerceptionScore:
    """Greedy nearest-neighbour matching between detections and ground-truth objects."""
    truth = visible_objects(world)
    detections: list[Detection] = [
        d for d in observation.detections if d.category not in ("furniture", "unknown")
    ]
    score = PerceptionScore(seed=seed, n_truth=len(truth), n_detected=len(detections))

    # Track by index: Detection holds numpy arrays, so identity comparison (``list.remove``)
    # would try to compare arrays element-wise.
    unmatched = set(range(len(detections)))
    for obj, position in truth:
        best, best_distance = None, MATCH_RADIUS
        for index in unmatched:
            distance = float(np.linalg.norm(detections[index].position[:2] - position[:2]))
            if distance < best_distance:
                best, best_distance = index, distance
        if best is None:
            score.missed.append(obj.name)
            continue
        unmatched.discard(best)
        best = detections[best]
        score.n_matched += 1
        score.position_errors_mm.append(best_distance * 1000)
        expected_category = "utensil" if obj.is_utensil else obj.category
        score.category_correct += int(best.category == expected_category)
        score.color_correct += int(best.color_name == obj.color_name)
    score.false_positives = [
        f"{detections[i].describe()}@{np.round(detections[i].position[:2], 2).tolist()}"
        for i in sorted(unmatched)
    ]
    return score


def aggregate(scores: list[PerceptionScore]) -> dict:
    """Summarize per-seed scores into the numbers that go in the README."""
    if not scores:
        return {}
    errors = [e for s in scores for e in s.position_errors_mm]
    return {
        "seeds": len(scores),
        "recall": round(float(np.mean([s.recall for s in scores])), 3),
        "precision": round(float(np.mean([s.precision for s in scores])), 3),
        "category_accuracy": round(float(np.mean([s.category_accuracy for s in scores])), 3),
        "color_accuracy": round(float(np.mean([s.color_accuracy for s in scores])), 3),
        "position_error_mm": {
            "mean": round(float(np.mean(errors)), 1) if errors else 0.0,
            "p95": round(float(np.percentile(errors, 95)), 1) if errors else 0.0,
            "max": round(float(np.max(errors)), 1) if errors else 0.0,
        },
        "total_false_positives": int(sum(len(s.false_positives) for s in scores)),
        "total_missed": int(sum(len(s.missed) for s in scores)),
    }
