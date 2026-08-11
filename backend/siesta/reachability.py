"""Which cards can be moved within a few moves, not just right now.

A card counts as movable at depth ``d`` when some sequence of ``d`` legal moves
ends with that card being moved. Depth 1 is therefore the same as "has a legal
destination right now".

This is an *under*-approximation. The breadth-first walk keeps at most
``DEFAULT_STATE_CAP`` states per level, so on very branchy positions some
reachable cards are missed. Erring this way keeps the marking honest: a card
that lights up really can be moved within the depth shown, while an unmarked
card is only "not found within the budget", never "provably stuck". For the
provable direction, see :mod:`siesta.deadlock`.
"""

from __future__ import annotations

from typing import Dict

from .game import GameState, apply_move, legal_moves


DEFAULT_MAX_DEPTH = 4
DEFAULT_STATE_CAP = 400


def cards_movable_within(
    state: GameState,
    max_depth: int = DEFAULT_MAX_DEPTH,
    state_cap: int = DEFAULT_STATE_CAP,
) -> Dict[str, int]:
    """Map each card code to the fewest moves needed before it can move.

    Cards are keyed by code rather than position, since positions shift as the
    sequence is played out.
    """

    found: Dict[str, int] = {}
    frontier: Dict[str, GameState] = {state.state_hash: state}
    seen = set(frontier)

    for depth in range(1, max_depth + 1):
        next_frontier: Dict[str, GameState] = {}
        for current in frontier.values():
            for move in legal_moves(current):
                column = current.columns[move.from_column]
                for card in column[len(column) - move.run_length :]:
                    found.setdefault(card.code, depth)
                if depth < max_depth and len(next_frontier) < state_cap:
                    child = apply_move(current, move)
                    if child.state_hash not in seen:
                        seen.add(child.state_hash)
                        next_frontier[child.state_hash] = child
        frontier = next_frontier
        if not frontier:
            break

    return found
