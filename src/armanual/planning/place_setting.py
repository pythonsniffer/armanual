"""Table-setting layouts, including the cultural/aesthetic styles.

A style is a *task constraint*, not decoration: it changes where each item must end up and, for
some styles, which items belong on the table at all. Everything downstream — planning, arm
assignment, success checking — reads the layout from here, so switching style genuinely changes
the robot's behaviour and the final state it is scored against.

**Sourcing.** Each style carries the convention it encodes and where that convention comes from.
Conventions are recorded at the level of "what goes where", which is what the task needs, and
deliberately not as claims about what any culture "always" does. A style whose `verified` flag is
False has been drafted from the layout description but its source has not yet been checked in
this environment; it must be verified before it appears in the demo video. This is tracked as
gap G7 in docs/LIMITATIONS.md.

Coordinates are metres in the table frame, relative to the *setting origin*: +x to the user's
right, +y away from the user. The planner chooses the origin so the whole layout lands inside the
measured dual-arm workspace (docs/WORKSPACE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Slot:
    """One item's place in a setting."""

    role: str  # "plate", "fork", "knife", "spoon", "cup", "napkin", "bowl"
    offset: tuple[float, float]
    yaw: float = 0.0
    required: bool = True
    note: str = ""


@dataclass(frozen=True)
class TableStyle:
    """A named setting layout plus the provenance of its conventions."""

    name: str
    description: str
    slots: tuple[Slot, ...]
    source: str = ""
    verified: bool = False
    #: Roles that must NOT be on the table in this style, if present they are cleared away.
    excluded_roles: tuple[str, ...] = ()

    def slot(self, role: str) -> Slot | None:
        for candidate in self.slots:
            if candidate.role == role:
                return candidate
        return None

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(s.role for s in self.slots)


# The Western layouts below encode the arrangement this project actually demonstrates: plate
# centred, fork to the left of the plate, knife to its right, spoon outside the knife, drinking
# vessel above and to the right. This is the standard basic/formal Western cover as described by
# etiquette references; the placement is the part the robot is scored on.
_WESTERN_FORMAL = TableStyle(
    name="formal",
    description="Western formal cover: fork left, knife and spoon right, glass upper right.",
    slots=(
        Slot("plate", (0.00, 0.00), note="centred on the cover"),
        Slot("fork", (-0.085, 0.005), yaw=1.5708, note="left of the plate, tines up"),
        Slot("knife", (0.080, 0.005), yaw=1.5708, note="right of the plate, blade toward it"),
        Slot("spoon", (0.115, 0.005), yaw=1.5708, required=False, note="outside the knife"),
        Slot("cup", (0.105, 0.085), note="above and right of the plate"),
        Slot("napkin", (-0.135, 0.005), required=False, note="left of the fork"),
    ),
    source="Western formal cover as commonly documented (Emily Post Institute; Debrett's).",
    verified=False,
)

_WESTERN_CASUAL = TableStyle(
    name="casual",
    description="Everyday Western cover: plate, one fork, one knife, glass.",
    slots=(
        Slot("plate", (0.00, 0.00)),
        Slot("fork", (-0.080, 0.005), yaw=1.5708),
        Slot("knife", (0.075, 0.005), yaw=1.5708),
        Slot("cup", (0.095, 0.075)),
    ),
    source="Simplified form of the Western cover above.",
    verified=False,
)

_MINIMAL = TableStyle(
    name="minimal",
    description="Minimalist cover: plate and drinking vessel only, generous spacing.",
    slots=(
        Slot("plate", (0.00, 0.00)),
        Slot("cup", (0.115, 0.070)),
    ),
    source="Project-defined aesthetic, not a cultural convention.",
    verified=True,
    excluded_roles=("fork", "knife", "spoon", "napkin"),
)

_JAPANESE = TableStyle(
    name="japanese",
    description=(
        "Japanese arrangement: rice vessel at the diner's left, soup at the right, "
        "chopsticks laid horizontally across the near edge with the tips to the left."
    ),
    slots=(
        Slot("plate", (-0.075, 0.010), note="rice vessel, diner's left"),
        Slot("cup", (0.075, 0.010), note="soup/tea vessel, diner's right"),
        Slot("chopsticks", (0.000, -0.075), yaw=0.0, required=False,
             note="laid horizontally in front, tips pointing left"),
    ),
    source="Washoku table arrangement; requires verification before demo (gap G7).",
    verified=False,
    excluded_roles=("knife",),
)

STYLES: dict[str, TableStyle] = {
    style.name: style for style in (_WESTERN_FORMAL, _WESTERN_CASUAL, _MINIMAL, _JAPANESE)
}
DEFAULT_STYLE = "casual"


@dataclass
class SettingPlan:
    """Concrete world-frame targets for one place setting."""

    style: TableStyle
    origin: tuple[float, float]
    targets: dict[str, tuple[float, float]] = field(default_factory=dict)

    def target_for(self, role: str) -> tuple[float, float] | None:
        return self.targets.get(role)


def get_style(name: str | None) -> TableStyle:
    """Look up a style, falling back to the default rather than failing the task."""
    if not name:
        return STYLES[DEFAULT_STYLE]
    return STYLES.get(name, STYLES[DEFAULT_STYLE])


def build_setting(style_name: str | None, origin: tuple[float, float] = (0.0, -0.10)) -> SettingPlan:
    """Instantiate a style's layout at a table position."""
    style = get_style(style_name)
    plan = SettingPlan(style=style, origin=origin)
    for slot in style.slots:
        plan.targets[slot.role] = (origin[0] + slot.offset[0], origin[1] + slot.offset[1])
    return plan


#: Which detected/scene category can fill which setting role. "chopsticks" falls back to a
#: utensil because the scene has no chopstick asset — recorded here rather than hidden in the
#: planner, so the substitution is visible in the evaluation record.
ROLE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "plate": ("plate",),
    "fork": ("fork", "utensil"),
    "knife": ("knife", "utensil"),
    "spoon": ("spoon", "utensil"),
    "chopsticks": ("spoon", "fork", "utensil"),
    "cup": ("cup",),
    "napkin": ("napkin",),
}
