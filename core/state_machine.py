"""Voice pipeline state machine for ZIA."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

logger = logging.getLogger("zia.state")


class PipelineState(str, Enum):
    START = "START"
    INITIALIZE = "INITIALIZE"
    WAKE_WORD_LISTENING = "WAKE_WORD_LISTENING"
    WAKE_WORD_DETECTED = "WAKE_WORD_DETECTED"
    RECORD_USER_SPEECH = "RECORD_USER_SPEECH"
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    MEMORY_RETRIEVAL = "MEMORY_RETRIEVAL"
    LLM_PROCESSING = "LLM_PROCESSING"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"
    SPEAKER_OUTPUT = "SPEAKER_OUTPUT"
    FOLLOW_UP_LISTENING = "FOLLOW_UP_LISTENING"
    STOPPED = "STOPPED"


ALLOWED_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.START: {PipelineState.INITIALIZE, PipelineState.STOPPED},
    PipelineState.INITIALIZE: {PipelineState.WAKE_WORD_LISTENING, PipelineState.STOPPED},
    PipelineState.WAKE_WORD_LISTENING: {
        PipelineState.WAKE_WORD_DETECTED,
        PipelineState.STOPPED,
    },
    PipelineState.WAKE_WORD_DETECTED: {
        PipelineState.RECORD_USER_SPEECH,
        PipelineState.WAKE_WORD_LISTENING,
        PipelineState.STOPPED,
    },
    PipelineState.RECORD_USER_SPEECH: {
        PipelineState.SPEECH_TO_TEXT,
        PipelineState.WAKE_WORD_LISTENING,
        PipelineState.STOPPED,
    },
    PipelineState.SPEECH_TO_TEXT: {
        PipelineState.MEMORY_RETRIEVAL,
        PipelineState.WAKE_WORD_LISTENING,  # Automatic recovery from empty/failed STT
        PipelineState.TEXT_TO_SPEECH,  # Spoken error prompt if needed
        PipelineState.STOPPED,
    },
    PipelineState.MEMORY_RETRIEVAL: {
        PipelineState.LLM_PROCESSING,
        PipelineState.TEXT_TO_SPEECH,
        PipelineState.STOPPED,
    },
    PipelineState.LLM_PROCESSING: {
        PipelineState.TEXT_TO_SPEECH,
        PipelineState.WAKE_WORD_LISTENING,
        PipelineState.STOPPED,
    },
    PipelineState.TEXT_TO_SPEECH: {
        PipelineState.SPEAKER_OUTPUT,
        PipelineState.WAKE_WORD_LISTENING,
        PipelineState.STOPPED,
    },
    PipelineState.SPEAKER_OUTPUT: {
        PipelineState.FOLLOW_UP_LISTENING,
        PipelineState.WAKE_WORD_LISTENING,
        PipelineState.STOPPED,
    },
    PipelineState.FOLLOW_UP_LISTENING: {
        PipelineState.RECORD_USER_SPEECH,
        PipelineState.SPEECH_TO_TEXT,
        PipelineState.WAKE_WORD_LISTENING,
        PipelineState.STOPPED,
    },
    PipelineState.STOPPED: {PipelineState.START, PipelineState.INITIALIZE},
}


class InvalidStateTransition(RuntimeError):
    """Raised when an illegal pipeline state transition is attempted."""


class VoiceStateMachine:
    """Enforces valid state transitions across the voice interaction pipeline."""

    def __init__(
        self,
        on_transition: Callable[[PipelineState, PipelineState], None] | None = None,
    ) -> None:
        self._state = PipelineState.START
        self._on_transition = on_transition
        self.history: list[PipelineState] = [PipelineState.START]

    @property
    def state(self) -> PipelineState:
        return self._state

    def transition(self, target: PipelineState) -> PipelineState:
        allowed = ALLOWED_TRANSITIONS.get(self._state, set())
        if target not in allowed:
            raise InvalidStateTransition(
                f"Illegal transition: {self._state.value} -> {target.value}"
            )
        prev = self._state
        self._state = target
        self.history.append(target)
        logger.debug("State: %s -> %s", prev.value, target.value)
        if self._on_transition is not None:
            self._on_transition(prev, target)
        return target

    def reset(self) -> None:
        self._state = PipelineState.START
        self.history = [PipelineState.START]
