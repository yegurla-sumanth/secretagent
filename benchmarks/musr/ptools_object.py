"""Interfaces for MUSR object placement (theory of mind)."""

from secretagent.core import interface
from ptools_common import raw_answer, extract_index


@interface
def infer_belief(narrative: str, movements: str, question: str, choices: list) -> str:
    """Determine where the target person believes the target object is located.

    You receive:
    1. The FULL original narrative (re-read it for details extraction may have missed)
    2. Extracted object movements with presence/absence info
    3. Incidental discoveries (someone saw/noticed an object without witnessing the move)
    4. The question identifying the target person and object, and answer choices

    Your task — determine the person's BELIEF about the object's location:

    Step 1: Start with the object's initial location (everyone knows this)
    Step 2: For each movement of the target_object (in chronological order):
       - If target_person is in "present" → they know the new location
       - If target_person is in "absent" → they still believe the old location
    Step 3: Check discoveries — if target_person discovered the target_object
       at a location, that UPDATES their belief to that location
    Step 4: Re-read the narrative to check for anything the extraction missed:
       - Did someone TELL the target person where the object is?
       - Did the target person GO TO the object's location and see it there?
       - Did the target person interact with something NEAR the object?
    Step 5: Match the believed location to one of the answer choices

    IMPORTANT: The question asks where the person would LOOK, which means
    where they BELIEVE the object is — not necessarily where it actually is.
    """


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the narrative and answer where someone would look for an object.
    This is a theory-of-mind task: the answer is based on what the person
    believes, not the object's actual location.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)


@interface
def answer_question_workflow(narrative: str, question: str, choices: list) -> int:
    """Infer belief directly from narrative, then match to choices."""
    text = infer_belief(narrative, "", question, choices)
    return extract_index(text, choices)
