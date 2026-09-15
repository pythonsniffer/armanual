"""Parse a natural-language instruction into a :class:`~armanual.task.schema.TaskRequest`.

This is a deterministic semantic parser, not a language model. That is a deliberate choice for the
*instruction* stage, for three reasons: it runs in microseconds inside a control loop, it is
reproducible across seeds and machines (which the evaluation harness depends on), and when it
fails it fails visibly — an unparsed clause is reported, never silently guessed at.

Language understanding that genuinely needs vision — "the blue cup", "the one on the left",
"the bigger plate" — is *not* done here. The parser only records what was asked; binding those
descriptions to real objects happens in :mod:`armanual.task.grounding`, against the current
camera observation. Keeping the two apart is what makes an ambiguity measurable rather than
accidental.

Supported shapes (with synonyms, filler words and either word order):

    "set the table"                      "set the table, formal style"
    "put the blue cup on the plate"      "place the small plate in front of me"
    "pick up the spoon"                  "open the drawer"
    "pour water into the blue mug"       "give the fork to the other arm"
    "move the cup to the left of the plate"
    "use the left arm to pick up the knife"
"""

from __future__ import annotations

import re

from armanual.task.schema import ActionSpec, Destination, Referent, TaskRequest

# --------------------------------------------------------------------------------- vocabulary
CATEGORY_WORDS: dict[str, str] = {
    "plate": "plate", "plates": "plate", "dish": "plate", "dishes": "plate",
    "cup": "cup", "cups": "cup", "mug": "cup", "mugs": "cup", "glass": "cup", "glasses": "cup",
    "spoon": "spoon", "spoons": "spoon",
    "fork": "fork", "forks": "fork",
    "knife": "knife", "knives": "knife",
    "bottle": "bottle", "bottles": "bottle", "jug": "bottle", "carafe": "bottle",
    "tray": "tray", "trays": "tray", "platter": "tray",
    "napkin": "napkin", "napkins": "napkin", "serviette": "napkin",
    "utensil": "utensil", "utensils": "utensil", "cutlery": "utensil", "silverware": "utensil",
}
COLOR_WORDS = ("white", "blue", "navy", "red", "green", "purple", "orange", "silver", "brown",
               "dark blue", "light blue")
SIZE_WORDS: dict[str, str] = {
    "small": "small", "smaller": "small", "smallest": "small", "little": "small", "tiny": "small",
    "big": "large", "bigger": "large", "biggest": "large", "large": "large", "larger": "large",
    "largest": "large", "medium": "medium",
}
ORDINALS: dict[str, str] = {
    "leftmost": "leftmost", "left-most": "leftmost", "rightmost": "rightmost",
    "right-most": "rightmost", "nearest": "nearest", "closest": "nearest",
    "farthest": "farthest", "furthest": "farthest",
}
RELATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bto (?:the )?left of\b", "left_of"),
    (r"\bleft of\b", "left_of"),
    (r"\bon (?:the )?left(?: side)?(?: of)?\b", "left_of"),
    (r"\bto (?:the )?right of\b", "right_of"),
    (r"\bright of\b", "right_of"),
    (r"\bon (?:the )?right(?: side)?(?: of)?\b", "right_of"),
    (r"\bnext to\b", "near"),
    (r"\bbeside\b", "near"),
    (r"\bnear(?:est to)?\b", "near"),
    (r"\bin front of\b", "in_front_of"),
    (r"\bon top of\b", "on"),
    (r"\bbehind\b", "behind"),
    (r"\bfar from\b", "far_from"),
    (r"\bon top of\b", "on"),
    (r"\bonto\b", "on"),
    (r"\bon\b", "on"),
    (r"\binto\b", "on"),
    (r"\bin\b", "on"),
)
VERB_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bset (?:up )?(?:the )?table\b", "set_table"),
    (r"\blay (?:the )?table\b", "set_table"),
    (r"\bpour\b", "pour"),
    (r"\bfill\b", "pour"),
    (r"\bopen\b", "open_drawer"),
    (r"\bhand (?:it |the .*? )?(?:over|to)\b", "handoff"),
    (r"\bpass\b", "handoff"),
    (r"\bgive\b", "handoff"),
    (r"\bput\b", "place"),
    (r"\bplace\b", "place"),
    (r"\bmove\b", "place"),
    (r"\bbring\b", "place"),
    (r"\barrange\b", "place"),
    (r"\bpick (?:up )?\b", "pick"),
    (r"\bgrab\b", "pick"),
    (r"\btake\b", "pick"),
    (r"\bstack\b", "stack"),
    (r"\bclear\b", "clear"),
)
STYLE_WORDS: dict[str, str] = {
    "formal": "formal", "fine dining": "formal", "casual": "casual", "informal": "casual",
    "minimalist": "minimal", "minimal": "minimal", "simple": "minimal",
    "japanese": "japanese", "washoku": "japanese", "indian": "indian", "thali": "indian",
}
ARM_WORDS: dict[str, str] = {"left arm": "left", "right arm": "right",
                             "left hand": "left", "right hand": "right"}
SLOT_WORDS: dict[str, str] = {
    "place setting": "place_setting", "table": "place_setting", "tray": "tray",
    "drawer": "drawer", "side": "side",
}
# "then" and "and then" are *clause separators*, not filler — stripping them would silently
# merge a two-step instruction into one step.
_FILLER = re.compile(
    r"\b(please|could you|can you|would you|kindly|just|for me|okay|ok|robot|hey)\b"
)
_SPLIT = re.compile(r"\s*(?:,\s*)?(?:\band then\b|\bthen\b|\bafter that\b|\band\b|;)\s*")


def normalize(text: str) -> str:
    """Lower-case, strip filler and punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = _FILLER.sub(" ", text)
    text = re.sub(r"[^\w\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_relation(text: str, *, last: bool = False) -> tuple[str | None, int, int]:
    """First (or last) relation phrase in ``text``.

    ``last=True`` is what a destination needs: in "put the cup on the left onto the tray" the
    *final* relation separates object from destination, while the earlier one belongs to the
    object's own description.
    """
    found: list[tuple[str, int, int]] = []
    for pattern, relation in RELATION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            found.append((relation, match.start(), match.end()))
    if not found:
        return None, -1, -1
    chosen = max(found, key=lambda f: f[1]) if last else min(found, key=lambda f: f[1])
    return chosen


def parse_referent(phrase: str) -> Referent:
    """Parse a noun phrase such as 'the bigger blue cup on the left'."""
    phrase = phrase.strip()
    referent = Referent(text=phrase)
    for word in ("the", "a", "an", "that", "this", "some"):
        phrase = re.sub(rf"\b{word}\b", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()

    for ordinal_word, ordinal in ORDINALS.items():
        if ordinal_word in phrase:
            referent.ordinal = ordinal
            phrase = phrase.replace(ordinal_word, " ")
    for color in sorted(COLOR_WORDS, key=len, reverse=True):
        if re.search(rf"\b{color}\b", phrase):
            referent.color = "navy" if color == "dark blue" else (
                "blue" if color == "light blue" else color
            )
            phrase = re.sub(rf"\b{color}\b", " ", phrase)
            break
    for size_word, size in SIZE_WORDS.items():
        if re.search(rf"\b{size_word}\b", phrase):
            referent.size = size
            phrase = re.sub(rf"\b{size_word}\b", " ", phrase)
            break
    for word, category in CATEGORY_WORDS.items():
        if re.search(rf"\b{word}\b", phrase):
            referent.category = category
            break

    # A trailing relation turns the rest of the phrase into an anchor: "cup next to the plate".
    relation, start, end = _find_relation(phrase)
    if relation is not None:
        tail = phrase[end:].strip()
        if tail:
            anchor = parse_referent(tail)
            if not anchor.is_empty:
                referent.relation = relation
                referent.anchor = anchor
        elif relation in ("left_of", "right_of"):
            # "the cup on the left" — a side of the table, not a relation to another object.
            referent.ordinal = "leftmost" if relation == "left_of" else "rightmost"
    return referent


def _parse_destination(phrase: str) -> Destination | None:
    phrase = phrase.strip()
    if not phrase:
        return None
    relation, _start, end = _find_relation(phrase)
    tail = phrase[end:].strip() if relation else phrase
    for words, slot in SLOT_WORDS.items():
        if re.search(rf"\b{words}\b", tail) and not any(
            re.search(rf"\b{w}\b", tail) for w in CATEGORY_WORDS
        ):
            return Destination(slot=slot, relation=relation)
    anchor = parse_referent(tail)
    if anchor.is_empty:
        return Destination(slot="place_setting", relation=relation)
    return Destination(relation=relation or "near", anchor=anchor)


def _split_clause(clause: str) -> tuple[str, str]:
    """Split a clause into (object phrase, destination phrase) at its last relation word."""
    relation, start, _end = _find_relation(clause, last=True)
    if relation is None:
        return clause, ""
    return clause[:start].strip(), clause[start:].strip()


#: Words that refer back to whatever the previous clause acted on.
PRONOUNS = ("it", "them", "that", "this", "the same one")


def parse_clause(clause: str, previous_target: Referent | None = None) -> ActionSpec | None:
    """Parse one imperative clause into an action.

    ``previous_target`` resolves anaphora: "pick up the spoon **and place it** to the right of the
    plate" is one of the most natural ways to phrase a two-step instruction, and dropping the
    second clause because "it" is not a noun would lose half the task.
    """
    arm_hint = None
    for words, arm in ARM_WORDS.items():
        if words in clause:
            arm_hint = arm
            clause = clause.replace(words, " ")

    verb = None
    for pattern, candidate in VERB_PATTERNS:
        match = re.search(pattern, clause)
        if match:
            verb = candidate
            clause = clause[: match.start()] + " " + clause[match.end() :]
            break
    if verb is None:
        return None
    if verb == "set_table":
        return ActionSpec(verb="set_table", arm_hint=arm_hint)
    if verb == "open_drawer":
        return ActionSpec(verb="open_drawer", arm_hint=arm_hint)

    if verb == "pick":
        # "pick up the cup on the left" has no destination, so the whole phrase describes the
        # object. Splitting it at the relation would hand "on the left" to a destination the verb
        # does not have, and the location would simply be lost.
        object_phrase, destination_phrase = clause, ""
    else:
        object_phrase, destination_phrase = _split_clause(clause)
    target = parse_referent(object_phrase)
    is_pronoun = any(re.search(rf"\b{word}\b", object_phrase) for word in PRONOUNS)
    if target.is_empty and previous_target is not None and (is_pronoun or not object_phrase.strip()):
        target = previous_target
    if verb == "pour":
        # "pour water into the blue cup": the cup is the destination, the bottle is implicit.
        destination = _parse_destination(destination_phrase)
        cup = destination.anchor if destination and destination.anchor else parse_referent(
            destination_phrase
        )
        if cup.is_empty:
            cup = Referent(text="cup", category="cup")
        return ActionSpec(verb="pour", target=cup, arm_hint=arm_hint)
    if verb == "handoff":
        return ActionSpec(verb="handoff", target=target, arm_hint=arm_hint)
    if target.is_empty:
        return None
    destination = _parse_destination(destination_phrase) if destination_phrase else None
    if verb == "pick":
        return ActionSpec(verb="pick", target=target, arm_hint=arm_hint)
    return ActionSpec(verb="place", target=target, destination=destination, arm_hint=arm_hint)


def _has_verb(clause: str) -> bool:
    return any(re.search(pattern, clause) for pattern, _verb in VERB_PATTERNS)


def _merge_verbless(clauses: list[str]) -> list[str]:
    """Re-join fragments that 'and' split in the middle of a phrase.

    "put the cup above and right of the plate" is one instruction, not two: the second fragment
    has no verb, so it belongs to the first clause. Splitting it off would drop half the
    destination and leave a spurious parse warning.
    """
    merged: list[str] = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if merged and not _has_verb(clause):
            merged[-1] = f"{merged[-1]} and {clause}"
        else:
            merged.append(clause)
    return merged


def parse_instruction(text: str, modality: str = "text", confidence: float = 1.0) -> TaskRequest:
    """Parse a whole instruction, which may chain several clauses with 'and' or 'then'."""
    request = TaskRequest(text=text, modality=modality, confidence=confidence)
    normalized = normalize(text)
    if not normalized:
        request.warnings.append("empty instruction")
        return request

    for phrase, style in STYLE_WORDS.items():
        if re.search(rf"\b{phrase}\b", normalized):
            request.style = style
            normalized = re.sub(rf"\b{phrase}\b", " ", normalized)
            break

    previous_target: Referent | None = None
    for clause in _merge_verbless(_SPLIT.split(normalized)):
        clause = clause.strip()
        if not clause:
            continue
        action = parse_clause(clause, previous_target)
        if action is None:
            request.warnings.append(f"could not parse clause: {clause!r}")
            continue
        if action.target is not None:
            previous_target = action.target
        request.actions.append(action)

    if not request.actions and request.style:
        # "set the table in the formal style" with the verb elided.
        request.actions.append(ActionSpec(verb="set_table"))
    if not request.actions:
        request.warnings.append("no supported action found")
    return request
