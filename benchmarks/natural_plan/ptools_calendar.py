"""Interfaces for NaturalPlan calendar scheduling task."""

from secretagent.core import interface


@interface
def extract_constraints(prompt: str) -> str:
    """Extract all scheduling constraints from this calendar problem.

    Include: participant names, each participant's busy time slots per day,
    required meeting duration, working hours, available days,
    and scheduling preference (earliest or latest).
    """


@interface
def solve_problem(prompt: str, constraints: str) -> str:
    """Given a calendar scheduling prompt and the extracted constraints,
    determine the optimal meeting time that works for all participants.

    Consider free slot intersections across all participants and
    respect the scheduling preference.
    """


@interface
def format_answer(solution: str) -> str:
    """Format a calendar scheduling solution into the standard answer format.

    Output exactly: Here is the proposed time: {Day}, {HH:MM} - {HH:MM}
    """


@interface
def calendar_scheduling(prompt: str) -> str:
    """Solve a calendar scheduling problem.
    Return: 'Here is the proposed time: {Day}, {HH:MM} - {HH:MM}'
    """
    ...


def calendar_workflow(prompt: str) -> str:
    constraints = extract_constraints(prompt)
    solution = solve_problem(prompt, constraints)
    return format_answer(solution)
