"""Interfaces for NaturalPlan meeting planning task."""

from secretagent.core import interface


@interface
def extract_constraints(prompt: str) -> str:
    """Extract all meeting planning constraints from this problem.

    Include: your starting location and time, each friend's name,
    location, availability window (start and end times),
    required meeting duration, and travel times between all locations.
    """


@interface
def solve_problem(prompt: str, constraints: str) -> str:
    """Given a meeting planning prompt and the extracted constraints,
    find the schedule that maximizes the number of valid meetings.

    Consider travel times between locations, availability windows,
    and meeting durations. Try different orderings to find the optimal one.
    """


@interface
def format_answer(solution: str) -> str:
    """Format a meeting planning solution into the standard answer format.

    Output must start with SOLUTION: followed by steps in this exact format:
    You start at {location} at {time}.
    You travel to {location} in {N} minutes and arrive at {time}.
    You wait until {time}.
    You meet {name} for {N} minutes from {start} to {end}.
    """


@interface
def meeting_planning(prompt: str) -> str:
    """Solve a meeting planning problem.
    Maximize the number of friends you can meet.
    Return a step-by-step schedule starting with SOLUTION:
    """
    ...


def meeting_workflow(prompt: str) -> str:
    constraints = extract_constraints(prompt)
    solution = solve_problem(prompt, constraints)
    return format_answer(solution)
