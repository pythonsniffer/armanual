"""The subtask vocabulary the VLA is trained on.

A single policy cannot learn "set the dinner table" from a few hundred demonstrations — that is a
minute-long, multi-object, multi-dependency behaviour. What it *can* learn is the handful of
skills the table-setting workflow is made of, each conditioned on its own sentence. The planner
then issues those sentences in order, re-observing between them, so from the user's side one
instruction still sets the whole table.

Each entry pairs a skill with the natural-language forms it is trained on. Multiple phrasings per
skill are deliberate: the policy should key on meaning, not on one memorized string, and the
evaluation holds out phrasings to check that.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillSpec:
    """One learnable skill and how it is described in language."""

    name: str
    templates: tuple[str, ...]
    bimanual: bool = False
    #: Object categories this skill applies to, for demonstration sampling.
    categories: tuple[str, ...] = ()
    notes: str = ""

    def phrase(self, **kwargs) -> str:
        return self.templates[0].format(**kwargs)

    def all_phrases(self, **kwargs) -> list[str]:
        return [template.format(**kwargs) for template in self.templates]


SKILLS: dict[str, SkillSpec] = {
    "open_drawer": SkillSpec(
        name="open_drawer",
        templates=(
            "open the drawer",
            "pull the drawer open",
            "open the cutlery drawer",
        ),
        notes="Single arm; the prerequisite for every utensil placement.",
    ),
    "place_object": SkillSpec(
        name="place_object",
        templates=(
            "put the {description} {where}",
            "place the {description} {where}",
            "move the {description} {where}",
        ),
        categories=("plate", "cup", "utensil"),
        notes="The workhorse skill: pick a described object and place it at a described spot.",
    ),
    "pour": SkillSpec(
        name="pour",
        templates=(
            "pour water into the {description}",
            "fill the {description} with water",
            "pour some water into the {description}",
        ),
        bimanual=True,
        categories=("cup",),
        notes="One arm steadies the cup, the other tips the bottle over it.",
    ),
    "handoff": SkillSpec(
        name="handoff",
        templates=(
            "pass the {description} to the other arm",
            "hand the {description} over to the other arm",
            "give the {description} to the other hand",
        ),
        bimanual=True,
        categories=("cup", "utensil"),
        notes="Required whenever pick and place sit in different arms' workspaces.",
    ),
}

#: Where things go, in words the grounder already understands.
PLACEMENTS: tuple[str, ...] = (
    "in the middle of the table",
    "to the left of the plate",
    "to the right of the plate",
    "next to the plate",
    "on the right side of the table",
)


@dataclass
class SubgoalPlan:
    """The sentence sequence one high-level instruction expands into."""

    instruction: str
    subgoals: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return " -> ".join(self.subgoals)


def decompose(instruction: str, style: str | None = None) -> SubgoalPlan:
    """Expand a table-setting instruction into the skill sentences the policy was trained on.

    This is the bridge between "set the table" and a policy that knows how to place one object.
    It is intentionally explicit and inspectable: the sequence it produces is written into every
    evaluation record, so what the robot was *told* to do at each stage is never in doubt.
    """
    text = instruction.lower()
    plan = SubgoalPlan(instruction=instruction)

    if any(phrase in text for phrase in ("set the table", "set up the table", "lay the table")):
        if style == "japanese":
            # Japanese arrangement: rice vessel to the diner's left, soup/tea to the right.
            plan.subgoals = [
                "open the drawer",
                "put the plate to the left of the middle of the table",
                "put the cup to the right of the plate",
            ]
        else:
            plan.subgoals = [
                "open the drawer",
                "put the plate in the middle of the table",
                "put the fork to the left of the plate",
                "put the cup to the right of the plate",
            ]
    if "pour" in text or "fill" in text:
        target = "blue cup" if "blue" in text else "cup"
        plan.subgoals.append(f"pour water into the {target}")
    if not plan.subgoals:
        plan.subgoals.append(instruction)
    return plan
