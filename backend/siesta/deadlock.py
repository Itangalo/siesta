"""Detection of dead cards and provably unwinnable positions.

A *dead card* is a card that can only ever be moved into an empty column,
because no legal destination card can be exposed for it. Kings are dead by
definition: no rank 14 exists. Lower ranks become dead through a deadlock
argument, resolved rank by rank from the top down.

A card of rank ``r`` is dead when

1. all four cards of rank ``r + 1`` are already on the table (none left in the
   stock, so no fresh destination can be dealt), and
2. every rank ``r + 1`` card is *blocked* – it carries, somewhere above it in
   its column, either another rank ``r`` card or an already-known dead card,
   and
3. the card itself is not resting directly on the matching same-suit ``r + 1``
   card, in which case it travels along as part of a colour run and never
   needs a destination of its own.

The analysis is deliberately conservative: a rank ``r + 1`` card buried under
a card we cannot *prove* dead counts as unblocked. That keeps the result free
of false positives, which is what makes it usable as a hard pruning signal.

Deadness is monotone. Dealing only places cards on top of columns, so it can
never unblock a rank ``r + 1`` card, and condition 1 rules out new ones
arriving. Therefore a column holding a dead card cannot be emptied while no
empty column exists – and if *every* column holds a dead card, no empty column
can ever appear again, so the deal is provably lost.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .game import Card, GameState


CardPosition = Tuple[int, int]
"""A card location as ``(column_index, index_within_column)``."""


def _positions_by_rank(columns: Sequence[Sequence[Card]]) -> Dict[int, List[CardPosition]]:
    positions: Dict[int, List[CardPosition]] = {}
    for column_index, column in enumerate(columns):
        for card_index, card in enumerate(column):
            positions.setdefault(card.rank, []).append((column_index, card_index))
    return positions


def _rests_on_matching_suit(columns: Sequence[Sequence[Card]], position: CardPosition) -> bool:
    column_index, card_index = position
    if card_index == 0:
        return False
    column = columns[column_index]
    card = column[card_index]
    below = column[card_index - 1]
    return below.suit == card.suit and below.rank == card.rank + 1


def _is_blocked(
    columns: Sequence[Sequence[Card]],
    position: CardPosition,
    blocking_rank: int,
    dead: Set[CardPosition],
) -> bool:
    column_index, card_index = position
    column = columns[column_index]
    for above_index in range(card_index + 1, len(column)):
        if column[above_index].rank == blocking_rank:
            return True
        if (column_index, above_index) in dead:
            return True
    return False


def dead_card_positions(state: GameState) -> Set[CardPosition]:
    """Return the positions of every card that can only move into a hole."""

    columns = state.columns
    positions = _positions_by_rank(columns)
    stock_rank_counts: Dict[int, int] = {}
    for card in state.stock:
        stock_rank_counts[card.rank] = stock_rank_counts.get(card.rank, 0) + 1

    dead: Set[CardPosition] = set(positions.get(13, []))

    # Strictly top-down: deadness at rank r may only rest on deadness already
    # established at *higher* ranks, which grounds the recursion at the kings.
    # Iterating this to a fixpoint is unsound – it lets a dead rank-r card be
    # used to prove a rank r+1 card blocked, which is what made it dead.
    for rank in range(12, 0, -1):
        if stock_rank_counts.get(rank + 1, 0) > 0:
            continue
        higher_positions = positions.get(rank + 1, [])
        if len(higher_positions) < 4:
            continue
        if not all(_is_blocked(columns, position, rank, dead) for position in higher_positions):
            continue
        for position in positions.get(rank, []):
            if not _rests_on_matching_suit(columns, position):
                dead.add(position)

    return dead


def locked_column_flags(state: GameState) -> List[bool]:
    """Per column: does it hold at least one dead card?

    A locked column cannot be emptied unless an empty column already exists.
    """

    dead = dead_card_positions(state)
    flags = [False] * len(state.columns)
    for column_index, _ in dead:
        flags[column_index] = True
    return flags


def emptiable_column_count(state: GameState) -> int:
    """Columns that are empty already or hold no dead card."""

    flags = locked_column_flags(state)
    return sum(1 for column_index, column in enumerate(state.columns) if not column or not flags[column_index])


def is_provably_lost(state: GameState) -> bool:
    """True when no empty column can ever be created again.

    Requires that no column is currently empty – an existing hole can absorb a
    dead card and unlock the position.
    """

    if any(not column for column in state.columns):
        return False
    return all(locked_column_flags(state))
