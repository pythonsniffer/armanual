"""Speech input via Speechmatics, feeding the same task pathway as text.

The important architectural point: this module produces a **string and a confidence**, nothing
more. It hands them to the same parser and grounder that typed text uses, so speech cannot grow
its own robot controller behind the scenes. A misheard word shows up as a grounding failure with
the transcript attached, which is why `TaskRequest` carries `transcript_alternatives`.

API details verified against the Speechmatics documentation on 2026-09-16:

* package: ``speechmatics-rt`` (``pip install speechmatics-rt``)
* client: ``speechmatics.rt.AsyncClient(api_key=...)``
* session: ``await client.start_session(transcription_config=..., audio_format=...)``
* audio: ``AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000, chunk_size=4096)``
* events: ``ServerMessageType.ADD_TRANSCRIPT`` (finals) and ``ADD_PARTIAL_TRANSCRIPT`` (partials,
  which require ``enable_partials=True``)
* endpoint: ``wss://eu.rt.speechmatics.com/v2/`` (regional) or ``wss://global.rt.speechmatics.com/v2/``

Credentials come from the ``SPEECHMATICS_API_KEY`` environment variable and are never written to
a results file, a log line or this repository.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

#: Only finals are acted on. A partial is a guess that the recognizer may retract, and acting on
#: a retracted guess means the robot starts moving toward something the user did not say.
DEFAULT_LANGUAGE = "en"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK = 4096


@dataclass
class Transcript:
    """One transcription result, with the numbers evaluation needs."""

    text: str
    is_final: bool = True
    confidence: float = 1.0
    latency_ms: float = 0.0
    alternatives: list[str] = field(default_factory=list)
    source: str = "offline"
    error: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.text.strip()) and self.error is None


class SpeechTranscriber(Protocol):
    name: str

    def transcribe_file(self, path: Path) -> Transcript:
        ...


class OfflineTranscriber:
    """Deterministic transcriber for tests and for demos without network access.

    Reads a sidecar ``.txt`` next to the audio file, or falls back to a supplied mapping. This
    exists so the *whole* speech pathway — transcript to parser to grounder to plan — can be
    tested in CI without an API key, which is the part that actually contains project logic.
    """

    name = "offline"

    def __init__(self, fixtures: dict[str, str] | None = None, latency_ms: float = 0.0):
        self.fixtures = fixtures or {}
        self.latency_ms = latency_ms

    def transcribe_file(self, path: Path) -> Transcript:
        path = Path(path)
        if path.stem in self.fixtures:
            return Transcript(self.fixtures[path.stem], latency_ms=self.latency_ms, source=self.name)
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            return Transcript(sidecar.read_text().strip(), latency_ms=self.latency_ms,
                              source=self.name)
        return Transcript("", error=f"no fixture or sidecar transcript for {path.name}",
                          source=self.name)


class SpeechmaticsTranscriber:
    """Real-time transcription through Speechmatics.

    Latency is measured from the first audio chunk sent to each final transcript received, which
    is the number a user actually waits through — not the server-side processing time.
    """

    name = "speechmatics"

    def __init__(self, *, api_key: str | None = None, language: str = DEFAULT_LANGUAGE,
                 url: str | None = None, enable_partials: bool = True,
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.api_key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
        self.language = language
        self.url = url
        self.enable_partials = enable_partials
        self.sample_rate = sample_rate
        self.partials: list[str] = []

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def transcribe_file(self, path: Path) -> Transcript:
        """Stream a PCM/WAV file and return the assembled final transcript."""
        if not self.configured:
            return Transcript(
                "", error="SPEECHMATICS_API_KEY is not set", source=self.name
            )
        try:
            import asyncio

            return asyncio.run(self._transcribe_async(Path(path)))
        except ModuleNotFoundError as exc:
            return Transcript("", error=f"speechmatics-rt not installed: {exc}", source=self.name)
        except Exception as exc:  # noqa: BLE001 - network/auth failures are results, not crashes
            return Transcript("", error=f"{type(exc).__name__}: {exc}", source=self.name)

    async def _transcribe_async(self, path: Path) -> Transcript:
        from speechmatics.rt import (
            AsyncClient,
            AudioEncoding,
            AudioFormat,
            ServerMessageType,
            TranscriptionConfig,
            TranscriptResult,
        )

        finals: list[str] = []
        self.partials = []
        started = time.perf_counter()
        first_final_ms = 0.0

        client_kwargs = {"api_key": self.api_key}
        if self.url:
            client_kwargs["url"] = self.url

        async with AsyncClient(**client_kwargs) as client:

            @client.on(ServerMessageType.ADD_TRANSCRIPT)
            def _on_final(message) -> None:  # pragma: no cover - needs the service
                nonlocal first_final_ms
                text = TranscriptResult.from_message(message).metadata.transcript
                if text:
                    if not finals:
                        first_final_ms = (time.perf_counter() - started) * 1000
                    finals.append(text)

            if self.enable_partials:

                @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
                def _on_partial(message) -> None:  # pragma: no cover - needs the service
                    text = TranscriptResult.from_message(message).metadata.transcript
                    if text:
                        self.partials.append(text)

            config = TranscriptionConfig(
                language=self.language, enable_partials=self.enable_partials
            )
            audio_format = AudioFormat(
                encoding=AudioEncoding.PCM_S16LE,
                sample_rate=self.sample_rate,
                chunk_size=DEFAULT_CHUNK,
            )
            await client.start_session(transcription_config=config, audio_format=audio_format)
            with path.open("rb") as handle:
                while chunk := handle.read(DEFAULT_CHUNK):
                    await client.send_audio(chunk)
            await client.close()

        text = " ".join(part.strip() for part in finals).strip()
        return Transcript(
            text=text,
            is_final=True,
            latency_ms=first_final_ms or (time.perf_counter() - started) * 1000,
            alternatives=list(self.partials[-3:]),
            source=self.name,
            error=None if text else "no final transcript received",
        )


def transcribe_to_request(transcriber: SpeechTranscriber, audio: Path):
    """Transcribe, then parse through the *same* pathway typed text uses.

    Returns ``(TaskRequest, Transcript)``. The request carries the transcript's confidence and the
    recognizer's last partials, so an episode that fails can be attributed to mishearing rather
    than to misunderstanding.
    """
    from armanual.task.parser import parse_instruction

    transcript = transcriber.transcribe_file(Path(audio))
    request = parse_instruction(transcript.text, modality="speech",
                                confidence=transcript.confidence)
    request.transcript_alternatives = list(transcript.alternatives)
    if transcript.error:
        request.warnings.append(f"speech: {transcript.error}")
    return request, transcript
