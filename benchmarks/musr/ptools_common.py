"""Shared interfaces for answer extraction across all MUSR splits."""

from secretagent.core import interface


@interface
def raw_answer(narrative: str, question: str, choices: list) -> str:
    """Read the narrative and answer the multiple-choice question."""


@interface
def extract_index(answer_text: str, choices: list) -> int:
    """Given an answer and choices, return the 0-based index of the matching choice."""
