"""Fast immutable engine for Siesta, mirroring backend.siesta.game rules exactly.

Cards are encoded as ints 0..51: id = suit_index * 13 + (rank - 1), where
suit_index follows game.SUITS order ["hearts", "diamonds", "clubs", "spades"].

A tableau is a tuple of 7 tuples of card ids. A full deal is
(cols, stock) where stock is a tuple of remaining card ids.
Tableaus are hashable, enabling transposition tables.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

SUITS = ("hearts", "diamonds", "clubs", "spades")

RANK = tuple(i % 13 + 1 for i in range(52))
SUIT = tuple(i // 13 for i in range(52))

Tableau = Tuple[Tuple[int, ...], ...]
MoveT = Tuple[int, int, int]  # (from_column, to_column, run_length)

_RANK_LABELS = {1: "A", 11: "J", 12: "Q", 13: "K"}


def card_code(cid: int) -> str:
    return f"{_RANK_LABELS.get(RANK[cid], str(RANK[cid]))}{'HDCS'[SUIT[cid]]}"


def describe_tableau(cols: Tableau) -> str:
    return "\n".join(" ".join(card_code(c) for c in col) for col in cols)


def fast_deck_shuffle(seed: Optional[int]) -> Tuple[Tableau, Tuple[int, ...]]:
    """Replicates game.create_game(seed) exactly."""
    ids = list(range(52))
    random.Random(seed).shuffle(ids)
    layout_sizes = [7, 6, 5, 4, 3, 2, 1]
    cursor = 0
    columns: List[Tuple[int, ...]] = []
    for size in layout_sizes:
        columns.append(tuple(ids[cursor : cursor + size]))
        cursor += size
    return tuple(columns), tuple(ids[cursor:])


def state_to_fast(columns: Sequence[Sequence[int]], stock: Sequence[int]) -> Tuple[Tableau, Tuple[int, ...]]:
    return tuple(tuple(int(c) for c in col) for col in columns), tuple(int(c) for c in stock)


def game_state_to_fast(columns: Sequence[Sequence[object]], stock: Sequence[object]) -> Tuple[Tableau, Tuple[int, ...]]:
    """Convert game.Card-based columns/stock into the fast id representation."""
    suit_index = {name: i for i, name in enumerate(SUITS)}

    def cid(card) -> int:
        return suit_index[card.suit] * 13 + card.rank - 1

    return (
        tuple(tuple(cid(card) for card in col) for col in columns),
        tuple(cid(card) for card in stock),
    )


def top_run(col: Tuple[int, ...]) -> int:
    if not col:
        return 0
    run = 1
    i = len(col) - 1
    while i > 0:
        upper = col[i]
        lower = col[i - 1]
        if SUIT[upper] == SUIT[lower] and RANK[lower] == RANK[upper] + 1:
            run += 1
            i -= 1
        else:
            break
    return run


def legal_moves(cols: Tableau) -> List[MoveT]:
    moves: List[MoveT] = []
    top_ranks = [RANK[col[-1]] if col else -1 for col in cols]
    for from_index in range(7):
        col = cols[from_index]
        if not col:
            continue
        run_max = top_run(col)
        for run_length in range(1, run_max + 1):
            wanted = RANK[col[-run_length]] + 1
            for to_index in range(7):
                if to_index == from_index:
                    continue
                if cols[to_index]:
                    if top_ranks[to_index] == wanted:
                        moves.append((from_index, to_index, run_length))
                else:
                    moves.append((from_index, to_index, run_length))
    return moves


def apply_move(cols: Tableau, move: MoveT) -> Tableau:
    frm, to, run_length = move
    source = cols[frm]
    moved = source[-run_length:]
    new_list = list(cols)
    new_list[frm] = source[: len(source) - run_length]
    new_list[to] = cols[to] + moved
    return tuple(new_list)


def apply_deal(cols: Tableau, stock: Tuple[int, ...]) -> Tuple[Tableau, Tuple[int, ...]]:
    n = min(7, len(stock))
    new_list = list(cols)
    for i in range(n):
        new_list[i] = new_list[i] + (stock[i],)
    return tuple(new_list), stock[n:]


def _is_complete_chunk(chunk: Tuple[int, ...]) -> bool:
    if len(chunk) != 13:
        return False
    base = chunk[0]
    suit = SUIT[base]
    king_id = suit * 13 + 12
    return chunk[0] == king_id and all(chunk[k] == king_id - k for k in range(13))


def count_completed_sequences(cols: Tableau) -> int:
    completed = 0
    for col in cols:
        i = 0
        n = len(col)
        while i <= n - 13:
            if _is_complete_chunk(col[i : i + 13]):
                completed += 1
                i += 13
            else:
                i += 1
    return completed


def _column_partitions_complete(col: Tuple[int, ...]) -> bool:
    i = 0
    n = len(col)
    while i < n:
        if not _is_complete_chunk(col[i : i + 13]):
            return False
        i += 13
    return True


def is_won(cols: Tableau) -> bool:
    total = sum(len(col) for col in cols)
    if total != 52:
        return False
    completed = count_completed_sequences(cols)
    if completed != 4:
        return False
    return all(_column_partitions_complete(col) for col in cols)


def dead_columns(cols: Tableau, stock_rank_counts: Optional[Dict[int, int]] = None) -> List[bool]:
    """Port of deadlock.locked_column_flags.

    Returns per-column flags: True when the column holds at least one dead
    card. Uses only the *count* of each rank left in the stock, never their
    order, so it stays fair for a player who cannot see future deals.
    """
    counts: Dict[int, int] = dict(stock_rank_counts or {})
    positions_by_rank: Dict[int, List[Tuple[int, int]]] = {}
    for ci, col in enumerate(cols):
        for i, cid in enumerate(col):
            positions_by_rank.setdefault(RANK[cid], []).append((ci, i))

    dead: Set[Tuple[int, int]] = set(positions_by_rank.get(13, []))

    def rests_on_matching(pos: Tuple[int, int]) -> bool:
        ci, i = pos
        if i == 0:
            return False
        col = cols[ci]
        return SUIT[col[i - 1]] == SUIT[col[i]] and RANK[col[i - 1]] == RANK[col[i]] + 1

    def blocked(pos: Tuple[int, int], blocking_rank: int) -> bool:
        ci, i = pos
        col = cols[ci]
        for j in range(i + 1, len(col)):
            if RANK[col[j]] == blocking_rank:
                return True
            if (ci, j) in dead:
                return True
        return False

    for rank in range(12, 0, -1):
        if counts.get(rank + 1, 0) > 0:
            continue
        higher = positions_by_rank.get(rank + 1, [])
        if len(higher) < 4:
            continue
        if not all(blocked(pos, rank) for pos in higher):
            continue
        for pos in positions_by_rank.get(rank, []):
            if not rests_on_matching(pos):
                dead.add(pos)

    flags = [False] * 7
    for ci, _ in dead:
        flags[ci] = True
    return flags


def is_provably_lost(cols: Tableau, stock_rank_counts: Optional[Dict[int, int]] = None) -> bool:
    """True when no column is empty and every column holds a dead card."""
    for col in cols:
        if not col:
            return False
    return all(dead_columns(cols, stock_rank_counts))


def replay_moves(cols: Tableau, moves: Sequence[MoveT]) -> Tableau:
    for move in moves:
        cols = apply_move(cols, move)
    return cols


def path_to(seen: Dict[Tableau, Optional[Tuple[Tableau, MoveT]]], target: Tableau) -> List[MoveT]:
    """Reconstruct move sequence from start state (parent None) to target."""
    moves: List[MoveT] = []
    cur = target
    while True:
        entry = seen.get(cur)
        if entry is None:
            break
        parent, move = entry
        moves.append(move)
        cur = parent
    moves.reverse()
    return moves
