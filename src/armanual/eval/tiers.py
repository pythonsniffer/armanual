"""The benchmark suite: a ladder of tasks, each isolating one capability.

The suite is deliberately a ladder rather than one hero scenario. A single end-to-end demo tells
you only that something broke; six tiers tell you *what* broke, because each tier adds exactly one
capability to the one below it:

===== ==================== ==========================================================
Tier  Name                 What it adds
===== ==================== ==========================================================
1     basic placement      one object, unambiguous, no dependencies
2     disambiguation       several similar objects; language must pick the right one
3     dependency           the drawer must be opened before its contents can be used
4     bimanual             a hand-off or a hold-and-pour that one arm cannot do alone
5     perturbation         full randomization: size, colour, mass, friction, lighting
6     hero                 the whole dinner-table workflow in one episode
===== ==================== ==========================================================

Every tier is defined by data — instruction, randomization, success criteria — so adding a task
never means touching the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from armanual.sim.randomize import RandomizationConfig


@dataclass(frozen=True)
class SuccessCriterion:
    """One checkable statement about the final state of the table."""

    kind: str  # "object_at_slot" | "drawer_open" | "liquid_in_cup" | "object_moved"
    role: str = ""  # setting role, for "object_at_slot"
    category: str = ""  # object category the criterion applies to
    tolerance: float = 0.06  # metres
    minimum: float = 0.0  # for drawer_open (metres) / liquid_in_cup (particle count)
    #: For slot criteria: the object must also have *moved* to get there. Without this a scene
    #: that happens to start with a plate near the middle would score a pass for doing nothing.
    require_moved: bool = True
    moved_threshold: float = 0.015
    description: str = ""

    def describe(self) -> str:
        return self.description or f"{self.kind}({self.role or self.category})"


@dataclass(frozen=True)
class TaskDefinition:
    """One benchmark task."""

    tier: int
    name: str
    instruction: str
    randomization: RandomizationConfig
    criteria: tuple[SuccessCriterion, ...]
    style: str | None = None
    modality: str = "text"
    #: Wording variants used to test that the same task survives being phrased differently.
    paraphrases: tuple[str, ...] = ()
    max_steps: int = 12
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        return f"t{self.tier}_{self.name}"


def _placement_only() -> RandomizationConfig:
    return RandomizationConfig.placement_only()


def _moderate() -> RandomizationConfig:
    return RandomizationConfig(sizes=True, colors=False, mass=True, friction=True,
                               lighting=False, background=False, distractors=True,
                               max_distractors=2)


def _full() -> RandomizationConfig:
    return RandomizationConfig()


TASKS: tuple[TaskDefinition, ...] = (
    # ---------------------------------------------------------------- tier 1: basic placement
    TaskDefinition(
        tier=1,
        name="plate_to_setting",
        instruction="put the plate in the middle of the table",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="object_at_slot", role="plate", category="plate",
                             description="a plate ends up on the place setting"),
        ),
        paraphrases=("place the plate on the table", "move the plate to the middle"),
        max_steps=4,
        tags=("placement",),
    ),
    TaskDefinition(
        tier=1,
        name="cup_to_setting",
        instruction="put the cup above and right of the plate",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="object_at_slot", role="cup", category="cup",
                             description="a cup ends up at the cup position"),
        ),
        max_steps=4,
        tags=("placement",),
    ),
    # ------------------------------------------------------------- tier 2: disambiguation
    TaskDefinition(
        tier=2,
        name="blue_cup",
        instruction="put the blue cup on the plate",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="object_moved", category="cup", tolerance=0.04,
                             description="the *blue* cup moved, not the navy one"),
        ),
        paraphrases=("move the blue mug onto the plate", "place the blue cup on the dish"),
        max_steps=5,
        notes="The scene contains a blue and a navy cup: colour alone must decide.",
        tags=("grounding", "ambiguity"),
    ),
    TaskDefinition(
        tier=2,
        name="left_cup",
        instruction="pick up the cup on the left",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="object_moved", category="cup", tolerance=0.04,
                             description="the leftmost cup was the one picked"),
        ),
        max_steps=4,
        tags=("grounding", "spatial"),
    ),
    # ---------------------------------------------------------------- tier 3: dependency
    TaskDefinition(
        tier=3,
        name="drawer_then_utensil",
        instruction="open the drawer and put the fork to the left of the plate",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="drawer_open", minimum=0.07,
                             description="the drawer was opened"),
            SuccessCriterion(kind="object_at_slot", role="fork", category="utensil",
                             tolerance=0.07,
                             description="a utensil reached the fork position"),
        ),
        max_steps=8,
        notes="The utensils are invisible until the drawer opens — the plan must grow.",
        tags=("dependency", "replanning"),
    ),
    # ------------------------------------------------------------------ tier 4: bimanual
    TaskDefinition(
        tier=4,
        name="pour_with_hold",
        instruction="pour water into the blue cup",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="liquid_in_cup", minimum=1,
                             description="at least one water particle ends up in the cup"),
        ),
        max_steps=6,
        notes="One arm steadies the cup while the other pours — a complementary dual-arm action.",
        tags=("bimanual", "pour"),
    ),
    TaskDefinition(
        tier=4,
        name="cross_table_handoff",
        instruction="put the cup on the left onto the right side of the table",
        randomization=_placement_only(),
        criteria=(
            SuccessCriterion(kind="object_moved", category="cup", tolerance=0.10,
                             description="the cup crossed the table"),
        ),
        max_steps=6,
        notes="Pick and place sit in different arms' workspaces, so a hand-off is required.",
        tags=("bimanual", "handoff"),
    ),
    # -------------------------------------------------------------- tier 5: perturbation
    TaskDefinition(
        tier=5,
        name="set_table_randomized",
        instruction="set the table",
        randomization=_full(),
        criteria=(
            SuccessCriterion(kind="object_at_slot", role="plate", category="plate"),
            SuccessCriterion(kind="object_at_slot", role="cup", category="cup"),
        ),
        max_steps=12,
        notes="Everything varies: size, colour, mass, friction, lighting, background, clutter.",
        tags=("robustness",),
    ),
    TaskDefinition(
        tier=5,
        name="set_table_moderate",
        instruction="set the table",
        randomization=_moderate(),
        criteria=(
            SuccessCriterion(kind="object_at_slot", role="plate", category="plate"),
            SuccessCriterion(kind="object_at_slot", role="cup", category="cup"),
        ),
        max_steps=12,
        tags=("robustness",),
    ),
    # -------------------------------------------------------------------- tier 6: hero
    TaskDefinition(
        tier=6,
        name="hero_dinner_table",
        instruction=(
            "open the drawer and set the table, then pour water into the blue cup"
        ),
        randomization=_moderate(),
        criteria=(
            SuccessCriterion(kind="drawer_open", minimum=0.07),
            SuccessCriterion(kind="object_at_slot", role="plate", category="plate"),
            SuccessCriterion(kind="object_at_slot", role="cup", category="cup"),
            SuccessCriterion(kind="object_at_slot", role="fork", category="utensil",
                             tolerance=0.07),
        ),
        max_steps=16,
        notes="The full workflow: dependency, multi-object placement, bimanual pour.",
        tags=("hero", "bimanual", "dependency"),
    ),
    TaskDefinition(
        tier=6,
        name="hero_japanese_style",
        instruction="set the table in the japanese style",
        randomization=_placement_only(),
        style="japanese",
        criteria=(
            SuccessCriterion(kind="object_at_slot", role="plate", category="plate"),
            SuccessCriterion(kind="object_at_slot", role="cup", category="cup"),
        ),
        max_steps=12,
        notes="Same objects, different layout: the style changes where things go.",
        tags=("hero", "style"),
    ),
)


def tasks_for_tier(tier: int) -> tuple[TaskDefinition, ...]:
    return tuple(task for task in TASKS if task.tier == tier)


def task_by_id(task_id: str) -> TaskDefinition:
    for task in TASKS:
        if task.id == task_id or task.name == task_id:
            return task
    raise KeyError(f"unknown task {task_id!r}; known: {[t.id for t in TASKS]}")
