"""Parser and grounding tests. No simulator needed, so these run in milliseconds."""

from __future__ import annotations

import numpy as np
import pytest

from armanual.perception.detector import Detection, SceneObservation
from armanual.task.grounding import Grounder
from armanual.task.parser import parse_instruction


def detection(category, color, x, y, *, radius=0.03, height=0.06, name=None, elongation=1.0):
    return Detection(
        position=np.array([x, y, height]),
        radius=radius,
        height=height,
        color_name=color,
        color_rgb=np.zeros(3),
        category=category,
        pixel_area=400,
        elongation=elongation,
        name=name,
    )


@pytest.fixture
def scene():
    return SceneObservation(
        detections=[
            detection("plate", "white", -0.04, -0.10, radius=0.05, height=0.018, name="plate_main"),
            detection("plate", "white", 0.22, -0.12, radius=0.037, height=0.014, name="plate_side"),
            detection("cup", "blue", 0.11, -0.03, name="cup_target"),
            detection("cup", "navy", -0.12, -0.03, name="cup_distract"),
            detection("bottle", "green", 0.31, -0.14, height=0.12, name="bottle_water"),
        ]
    )


class TestParser:
    def test_simple_placement(self):
        request = parse_instruction("put the blue cup on the plate")
        assert [a.verb for a in request.actions] == ["place"]
        action = request.actions[0]
        assert action.target.color == "blue"
        assert action.target.category == "cup"
        assert action.destination.anchor.category == "plate"

    def test_two_clauses_with_pronoun(self):
        request = parse_instruction("pick up the spoon and place it to the right of the plate")
        assert [a.verb for a in request.actions] == ["pick", "place"]
        # "it" must resolve to the spoon from the previous clause, not be dropped.
        assert request.actions[1].target.category == "spoon"

    def test_and_inside_one_phrase_is_not_a_clause_break(self):
        request = parse_instruction("put the cup above and right of the plate")
        assert len(request.actions) == 1
        assert not request.warnings

    def test_side_of_table_becomes_an_ordinal(self):
        request = parse_instruction("put the cup on the left onto the tray")
        assert request.actions[0].target.ordinal == "leftmost"

    def test_style_is_extracted(self):
        request = parse_instruction("set the table in the japanese style")
        assert request.style == "japanese"
        assert request.actions[0].verb == "set_table"

    def test_arm_request_is_recorded(self):
        request = parse_instruction("use the left arm to pick up the knife")
        assert request.actions[0].arm_hint == "left"

    def test_unparseable_input_is_reported_not_guessed(self):
        request = parse_instruction("blah blah nonsense")
        assert not request.actions
        assert request.warnings


class TestGrounding:
    def test_colour_selects_between_similar_cups(self, scene):
        request = parse_instruction("pick up the blue cup")
        task = Grounder().ground(request, scene)
        assert task.actions[0].target.name == "cup_target"

    def test_absent_colour_is_unresolved_rather_than_substituted(self, scene):
        request = parse_instruction("pick up the red cup")
        task = Grounder().ground(request, scene)
        assert not task.actions
        assert task.unresolved

    def test_size_disambiguates_plates(self, scene):
        request = parse_instruction("pick up the small plate")
        task = Grounder().ground(request, scene)
        assert task.actions[0].target.name == "plate_side"

    def test_ambiguity_is_reported(self, scene):
        request = parse_instruction("pick up the cup")
        task = Grounder().ground(request, scene)
        assert task.ambiguities, "two similar cups should be flagged as ambiguous"

    def test_pointing_resolves_ambiguity(self, scene):
        request = parse_instruction("pick up the cup")
        for action in request.actions:
            action.target.pointing_xy = (-0.12, -0.03)
        task = Grounder().ground(request, scene)
        assert task.actions[0].target.name == "cup_distract"

    def test_spatial_relation_picks_the_left_object(self, scene):
        request = parse_instruction("pick up the cup on the left")
        task = Grounder().ground(request, scene)
        assert task.actions[0].target.name == "cup_distract"

    def test_destination_is_resolved_to_a_point(self, scene):
        request = parse_instruction("put the blue cup to the left of the plate")
        task = Grounder().ground(request, scene)
        point = task.actions[0].destination_xy
        assert point is not None
        assert point[0] < -0.04, "left of the plate must be at smaller x"
