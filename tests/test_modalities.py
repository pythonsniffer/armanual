"""Every input modality must land in the same TaskRequest, offline and without credentials."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from armanual.modality.speech import OfflineTranscriber, SpeechmaticsTranscriber, transcribe_to_request
from armanual.perception.detector import Detection, SceneObservation
from armanual.task.grounding import Grounder, ground_instruction
from armanual.task.parser import parse_instruction

FIXTURES = Path(__file__).parent / "fixtures" / "audio"


def _scene():
    def detection(category, color, x, y, **kwargs):
        return Detection(position=np.array([x, y, kwargs.get("height", 0.06)]),
                         radius=kwargs.get("radius", 0.03), height=kwargs.get("height", 0.06),
                         color_name=color, color_rgb=np.zeros(3), category=category,
                         pixel_area=400, elongation=1.0, name=kwargs.get("name"))

    return SceneObservation(detections=[
        detection("plate", "white", -0.04, -0.10, radius=0.05, height=0.018, name="plate_main"),
        detection("cup", "blue", 0.11, -0.03, name="cup_target"),
        detection("cup", "navy", -0.12, -0.03, name="cup_distract"),
    ])


class TestSpeechPathway:
    def test_offline_transcriber_reads_sidecar(self):
        transcript = OfflineTranscriber().transcribe_file(FIXTURES / "blue_cup.wav")
        assert transcript.usable
        assert "blue cup" in transcript.text

    def test_speech_and_text_produce_the_same_actions(self):
        request, transcript = transcribe_to_request(OfflineTranscriber(), FIXTURES / "blue_cup.wav")
        typed = parse_instruction(transcript.text)
        assert request.modality == "speech"
        assert [a.describe() for a in request.actions] == [a.describe() for a in typed.actions]

    def test_speech_style_reaches_the_request(self):
        request, _ = transcribe_to_request(OfflineTranscriber(), FIXTURES / "japanese.wav")
        assert request.style == "japanese"

    def test_missing_audio_is_reported_not_raised(self):
        request, transcript = transcribe_to_request(OfflineTranscriber(), FIXTURES / "nope.wav")
        assert not transcript.usable
        assert any("speech:" in w for w in request.warnings)

    def test_speechmatics_without_a_key_reports_instead_of_failing(self, monkeypatch):
        monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
        transcript = SpeechmaticsTranscriber().transcribe_file(FIXTURES / "blue_cup.wav")
        assert not transcript.usable
        assert "SPEECHMATICS_API_KEY" in (transcript.error or "")


class TestPointingPathway:
    def test_pointing_disambiguates_where_words_cannot(self):
        scene = _scene()
        without = ground_instruction("pick up the cup", scene)
        assert without.ambiguities

        with_point = ground_instruction("pick up the cup", scene, modality="pointing",
                                        pointing_xy=(0.11, -0.03))
        assert with_point.actions[0].target.name == "cup_target"

    def test_all_modalities_share_one_schema(self):
        scene = _scene()
        grounder = Grounder()
        for modality, pointing in (("text", None), ("speech", None), ("pointing", (0.11, -0.03))):
            request = parse_instruction("pick up the blue cup", modality=modality)
            if pointing:
                for action in request.actions:
                    action.target.pointing_xy = pointing
            task = grounder.ground(request, scene)
            assert task.actions[0].target.name == "cup_target"
            assert task.request.modality == modality
