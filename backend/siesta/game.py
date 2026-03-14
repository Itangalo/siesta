from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Optional


SUITS = ["hearts", "diamonds", "clubs", "spades"]
SUIT_SYMBOLS = {
    "hearts": "H",
    "diamonds": "D",
    "clubs": "C",
    "spades": "S",
}
RANK_LABELS = {
    1: "A",
    11: "J",
    12: "Q",
    13: "K",
}
STATUS_IN_PROGRESS = "in_progress"
STATUS_WON = "won"
STATUS_CONCEDED = "conceded"
STATUS_TRAINING_TERMINATED = "training_terminated"


class InvalidMoveError(ValueError):
    """Raised when a requested move or action is not legal."""


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str

    @property
    def color(self) -> str:
        return "red" if self.suit in {"hearts", "diamonds"} else "black"

    @property
    def rank_label(self) -> str:
        return RANK_LABELS.get(self.rank, str(self.rank))

    @property
    def code(self) -> str:
        return f"{self.rank_label}{SUIT_SYMBOLS[self.suit]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "rank_label": self.rank_label,
            "suit": self.suit,
            "suit_symbol": SUIT_SYMBOLS[self.suit],
            "color": self.color,
            "code": self.code,
        }


@dataclass(frozen=True)
class Move:
    from_column: int
    to_column: int
    run_length: int

    def to_dict(self, state: "GameState") -> Dict[str, Any]:
        cards = state.columns[self.from_column][-self.run_length :]
        lead_card = cards[0]
        dest_card = state.columns[self.to_column][-1] if state.columns[self.to_column] else None
        return {
            "type": "move",
            "from_column": self.from_column,
            "to_column": self.to_column,
            "run_length": self.run_length,
            "lead_card": lead_card.to_dict(),
            "destination_card": dest_card.to_dict() if dest_card else None,
            "description": describe_move(cards, self.to_column, dest_card),
        }


@dataclass
class GameState:
    columns: List[List[Card]]
    stock: List[Card]
    status: str = STATUS_IN_PROGRESS
    moves_played: int = 0
    terminal_reason: Optional[str] = None
    state_hash: str = ""
    completed_sequences: int = 0

    def clone(self) -> "GameState":
        return deepcopy(self)


def create_deck() -> List[Card]:
    return [Card(rank=rank, suit=suit) for suit in SUITS for rank in range(1, 14)]


def create_game(seed: Optional[int] = None) -> GameState:
    deck = create_deck()
    random.Random(seed).shuffle(deck)
    layout_sizes = [7, 6, 5, 4, 3, 2, 1]
    cursor = 0
    columns: List[List[Card]] = []
    for size in layout_sizes:
        columns.append(deck[cursor : cursor + size])
        cursor += size
    stock = deck[cursor:]
    state = GameState(columns=columns, stock=stock)
    return refresh_state(state)


def describe_move(cards: List[Card], to_column: int, dest_card: Optional[Card]) -> str:
    prefix = "-".join(card.code for card in cards)
    if dest_card:
        return f"{prefix} -> {dest_card.code} (col {to_column + 1})"
    return f"{prefix} -> empty col {to_column + 1}"


def top_run_length(column: List[Card]) -> int:
    if not column:
        return 0
    run = 1
    for index in range(len(column) - 1, 0, -1):
        upper = column[index]
        lower = column[index - 1]
        if upper.suit == lower.suit and lower.rank == upper.rank + 1:
            run += 1
        else:
            break
    return run


def movable_cards(column: List[Card]) -> List[List[Card]]:
    if not column:
        return []
    run = top_run_length(column)
    return [column[-length:] for length in range(1, run + 1)]


def legal_moves(state: GameState) -> List[Move]:
    if state.status != STATUS_IN_PROGRESS:
        return []

    moves: List[Move] = []
    for from_index, column in enumerate(state.columns):
        if not column:
            continue
        run = top_run_length(column)
        for run_length in range(1, run + 1):
            moving_cards = column[-run_length:]
            lead = moving_cards[0]
            for to_index, destination in enumerate(state.columns):
                if to_index == from_index:
                    continue
                if not destination:
                    moves.append(Move(from_column=from_index, to_column=to_index, run_length=run_length))
                    continue
                if destination[-1].rank == lead.rank + 1:
                    moves.append(Move(from_column=from_index, to_column=to_index, run_length=run_length))
    unique: Dict[tuple, Move] = {}
    for move in moves:
        unique[(move.from_column, move.to_column, move.run_length)] = move
    return list(unique.values())


def can_deal(state: GameState) -> bool:
    return state.status == STATUS_IN_PROGRESS and bool(state.stock)


def apply_move(state: GameState, move: Move) -> GameState:
    if state.status != STATUS_IN_PROGRESS:
        raise InvalidMoveError("The game is not in progress.")
    if move.from_column == move.to_column:
        raise InvalidMoveError("Source and destination columns must differ.")
    if not 0 <= move.from_column < len(state.columns) or not 0 <= move.to_column < len(state.columns):
        raise InvalidMoveError("Column index out of range.")

    column = state.columns[move.from_column]
    if move.run_length < 1 or move.run_length > len(column):
        raise InvalidMoveError("Invalid run length.")
    if move.run_length > top_run_length(column):
        raise InvalidMoveError("Requested run is not a valid movable suffix.")

    moving_cards = column[-move.run_length :]
    destination = state.columns[move.to_column]
    if destination and destination[-1].rank != moving_cards[0].rank + 1:
        raise InvalidMoveError("Destination must be exactly one rank higher.")

    next_state = state.clone()
    moved = next_state.columns[move.from_column][-move.run_length :]
    next_state.columns[move.from_column] = next_state.columns[move.from_column][:-move.run_length]
    next_state.columns[move.to_column].extend(moved)
    next_state.moves_played += 1
    next_state.terminal_reason = None
    return refresh_state(next_state)


def apply_deal(state: GameState) -> GameState:
    if not can_deal(state):
        raise InvalidMoveError("No cards remain in the stock.")

    next_state = state.clone()
    cards_to_deal = min(7, len(next_state.stock))
    for column_index in range(cards_to_deal):
        next_state.columns[column_index].append(next_state.stock.pop(0))
    next_state.moves_played += 1
    next_state.terminal_reason = None
    return refresh_state(next_state)


def apply_concede(state: GameState) -> GameState:
    if state.status != STATUS_IN_PROGRESS:
        return state
    next_state = state.clone()
    next_state.status = STATUS_CONCEDED
    next_state.terminal_reason = "conceded"
    return refresh_state(next_state)


def terminate_for_training(state: GameState, reason: str) -> GameState:
    next_state = state.clone()
    next_state.status = STATUS_TRAINING_TERMINATED
    next_state.terminal_reason = reason
    return refresh_state(next_state)


def _is_complete_sequence(chunk: List[Card]) -> bool:
    if len(chunk) != 13:
        return False
    suit = chunk[0].suit
    return all(card.suit == suit and card.rank == 13 - index for index, card in enumerate(chunk))


def count_completed_sequences(columns: List[List[Card]]) -> int:
    completed = 0
    for column in columns:
        index = 0
        while index <= len(column) - 13:
            chunk = column[index : index + 13]
            if _is_complete_sequence(chunk):
                completed += 1
                index += 13
            else:
                index += 1
    return completed


def _column_partitions_into_complete_sequences(column: List[Card]) -> bool:
    index = 0
    while index < len(column):
        chunk = column[index : index + 13]
        if not _is_complete_sequence(chunk):
            return False
        index += 13
    return True


def is_won(columns: List[List[Card]]) -> bool:
    total_cards = sum(len(column) for column in columns)
    completed = count_completed_sequences(columns)
    return total_cards == 52 and completed == 4 and all(_column_partitions_into_complete_sequences(column) for column in columns)


def refresh_state(state: GameState) -> GameState:
    state.completed_sequences = max(0, count_completed_sequences(state.columns))
    if state.status == STATUS_IN_PROGRESS and is_won(state.columns):
        state.status = STATUS_WON
        state.terminal_reason = "won"
    state.state_hash = compute_state_hash(state)
    return state


def compute_state_hash(state: GameState) -> str:
    payload = "|".join(
        ",".join(card.code for card in column)
        for column in state.columns
    )
    payload = f"{payload}::{'/'.join(card.code for card in state.stock)}::{state.status}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def serialize_state(state: GameState, game_id: Optional[str] = None, history_depth: int = 0) -> Dict[str, Any]:
    refresh_state(state)
    moves = legal_moves(state)
    columns = []
    for index, column in enumerate(state.columns):
        columns.append(
            {
                "index": index,
                "cards": [card.to_dict() for card in column],
                "movable_run_length": top_run_length(column),
                "height": len(column),
            }
        )
    return {
        "game_id": game_id,
        "status": state.status,
        "terminal_reason": state.terminal_reason,
        "moves_played": state.moves_played,
        "stock_count": len(state.stock),
        "completed_sequences": state.completed_sequences,
        "state_hash": state.state_hash,
        "history_depth": history_depth,
        "can_deal": can_deal(state),
        "columns": columns,
        "legal_moves": [move.to_dict(state) for move in moves],
    }
