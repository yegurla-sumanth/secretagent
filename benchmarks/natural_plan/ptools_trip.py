"""Interfaces for NaturalPlan trip planning task."""

from secretagent.core import interface


@interface
def extract_constraints(prompt: str) -> str:
    """Extract all trip planning constraints from this problem.

    Include: total number of days, all cities to visit,
    required duration (days) in each city, all direct flight
    connections (noting which are one-way vs bidirectional),
    and any time-window constraints (must be in city X between day A and day B).
    """


@interface
def solve_problem(prompt: str, constraints: str) -> str:
    """Given a trip planning prompt and the extracted constraints,
    find a valid ordering of cities that satisfies all constraints.

    Ensure: direct flights exist between consecutive cities,
    visit durations are respected, time-window constraints are met,
    and total days add up correctly.
    """


@interface
def format_answer(solution: str) -> str:
    """Format a trip planning solution into the standard answer format.

    Output format:
    First line: Here is the trip plan for visiting the {N} European cities for {total_days} days:
    For each city visit: **Day X-Y:** Visit {city} for {N} days.
    Between cities: **Day X:** Fly from {city1} to {city2}.
    """


@interface
def trip_planning(prompt: str) -> str:
    """Solve a trip planning problem.
    Find a valid ordering of cities satisfying all flight and time constraints.
    Return a day-by-day itinerary with visit and flight lines.
    """
    ...


def trip_workflow(prompt: str) -> str:
    constraints = extract_constraints(prompt)
    solution = solve_problem(prompt, constraints)
    return format_answer(solution)
