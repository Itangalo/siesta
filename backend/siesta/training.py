from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .deadlock import is_provably_lost
from .game import (
    Card,
    GameState,
    Move,
    STATUS_CONCEDED,
    STATUS_IN_PROGRESS,
    STATUS_WON,
    apply_deal,
    apply_move,
    can_deal,
    create_deck,
    create_game,
    legal_moves,
    serialize_state,
    terminate_for_training,
    top_run_length,
)


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_SESSIONS_DIR = DATA_DIR / "sessions"
DEFAULT_MODEL_PATH = MODELS_DIR / "policy-latest.npz"
DEFAULT_METADATA_PATH = MODELS_DIR / "policy-latest.json"
DEFAULT_CANDIDATE_MODEL_PATH = MODELS_DIR / "policy-candidate.npz"
DEFAULT_CANDIDATE_METADATA_PATH = MODELS_DIR / "policy-candidate.json"
DEFAULT_BEST_MODEL_PATH = MODELS_DIR / "policy-best.npz"
DEFAULT_BEST_METADATA_PATH = MODELS_DIR / "policy-best.json"
DEFAULT_HOLE_MODEL_PATH = MODELS_DIR / "hole-policy-latest.npz"
DEFAULT_HOLE_METADATA_PATH = MODELS_DIR / "hole-policy-latest.json"
DEFAULT_HOLE_CANDIDATE_MODEL_PATH = MODELS_DIR / "hole-policy-candidate.npz"
DEFAULT_HOLE_CANDIDATE_METADATA_PATH = MODELS_DIR / "hole-policy-candidate.json"
DEFAULT_COMPACT_MODEL_PATH = MODELS_DIR / "compact-policy-latest.npz"
DEFAULT_COMPACT_METADATA_PATH = MODELS_DIR / "compact-policy-latest.json"
DEFAULT_COMPACT_CANDIDATE_MODEL_PATH = MODELS_DIR / "compact-policy-candidate.npz"
DEFAULT_COMPACT_CANDIDATE_METADATA_PATH = MODELS_DIR / "compact-policy-candidate.json"
DEFAULT_FEEDBACK_PATH = DATA_DIR / "human_feedback.jsonl"
DEFAULT_SEARCH_DEPTH = 2
DEFAULT_SEARCH_BEAM_WIDTH = 4
DEFAULT_SEARCH_DISCOUNT = 0.92
DEFAULT_SEARCH_ROLLOUT_STEPS = 10
DEFAULT_UI_SEARCH_CANDIDATE_LIMIT = 12
DEFAULT_UI_SEARCH_DEPTH = 2
DEFAULT_UI_SEARCH_BEAM_WIDTH = 3
DEFAULT_UI_SEARCH_DISCOUNT = 0.92
DEFAULT_UI_SEARCH_ROLLOUT_STEPS = 6
DEFAULT_TEACHER_SEARCH_DEPTH = 3
DEFAULT_TEACHER_SEARCH_BEAM_WIDTH = 5
DEFAULT_TEACHER_SEARCH_DISCOUNT = 0.94
DEFAULT_TEACHER_SEARCH_ROLLOUT_STEPS = 10
DEFAULT_TEACHER_ALTERNATE_ACTIONS = 1
DEFAULT_TEACHER_MAX_STATES_PER_GAME = 3
DEFAULT_TEACHER_BRANCH_MARGIN = 10.0
DEFAULT_GENERATED_TARGET_BOOST = 0.65
DEFAULT_HUMAN_TARGET_BOOST = 0.82
DEFAULT_EXACT_HUMAN_TARGET_BOOST = 0.9
DEFAULT_IMPORTANT_HUMAN_TARGET_BOOST = 0.9
DEFAULT_IMPORTANT_EXACT_HUMAN_TARGET_BOOST = 0.96
DEFAULT_KEY_HUMAN_TARGET_BOOST = 0.96
DEFAULT_KEY_EXACT_HUMAN_TARGET_BOOST = 0.995
DEFAULT_HUMAN_TARGET_SHARE = 0.18
DEFAULT_MAX_HUMAN_REPEAT_FACTOR = 32
DEFAULT_HOLE_GOAL_HORIZON = 5
DEFAULT_HOLE_GOAL_BEAM_WIDTH = 6
DEFAULT_COMPACT_GOAL_HORIZON = 5
DEFAULT_COMPACT_GOAL_BEAM_WIDTH = 6
DEFAULT_COMPACT_MIDGAME_MIN_STEPS = 6
DEFAULT_COMPACT_MIDGAME_MAX_STEPS = 18
DEFAULT_COMPACT_MIDGAME_SHARE = 0.4
DEFAULT_TRAINING_STATE_SOURCE = "won_initial_states"
VISIBLE_CARD_FEATURES_PER_CARD = 9
MAX_CARD_DEPTH = 51.0
ProgressCallback = Optional[Callable[[str], None]]
CANONICAL_CARDS = tuple(create_deck())


@dataclass(frozen=True)
class Action:
    type: str
    move: Optional[Move] = None

    def to_dict(self, state: GameState) -> Dict[str, Any]:
        if self.type == "deal":
            return {
                "type": "deal",
                "description": f"Deal from stock ({len(state.stock)} left)",
            }
        if not self.move:
            raise ValueError("Move action missing move payload.")
        data = self.move.to_dict(state)
        data["type"] = "move"
        return data


@dataclass
class TrainingCase:
    features: np.ndarray
    target_index: int
    weight: float
    chosen_signature: str
    action_signatures: List[str]
    soft_targets: np.ndarray
    target_boost: float = DEFAULT_GENERATED_TARGET_BOOST
    source: str = "generated"


@dataclass
class HumanFeedbackSummary:
    records_seen: int = 0
    unique_entries: int = 0
    cases_loaded: int = 0
    skipped_missing_state: int = 0
    skipped_missing_action: int = 0
    inferred_stock_cases: int = 0
    invalidated_entries: int = 0
    won_cases: int = 0
    conceded_cases: int = 0
    unknown_outcome_cases: int = 0
    skipped_normal_conceded_cases: int = 0


@dataclass(frozen=True)
class TrainingStartState:
    source_id: str
    state: GameState
    source_type: str


@dataclass
class RolloutTraceStep:
    state: GameState
    action: Action


def emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is not None:
        progress(message)


def current_input_dim() -> int:
    state = create_game(seed=0)
    actions = available_actions(state)
    if not actions:
        raise RuntimeError("Could not determine model input dimension.")
    return int(encode_action(state, actions[0]).shape[0])


def available_actions(state: GameState) -> List[Action]:
    actions = [Action(type="move", move=move) for move in legal_moves(state)]
    if can_deal(state):
        actions.append(Action(type="deal"))
    return actions


def apply_action(state: GameState, action: Action) -> GameState:
    if action.type == "deal":
        return apply_deal(state)
    if action.type == "move" and action.move:
        return apply_move(state, action.move)
    raise ValueError(f"Unsupported action type: {action.type}")


def run_depth(state: GameState) -> int:
    return max((top_run_length(column) for column in state.columns), default=0)


def progress_score(state: GameState) -> float:
    empty_columns = empty_column_count(state)
    hole_bonus = 5.0 * empty_columns
    if empty_columns >= 2:
        hole_bonus += 4.0 * (empty_columns - 1)

    return (
        state.completed_sequences * 100.0
        + hole_bonus
        + 0.9 * hole_access_score(state)
        + 0.45 * hole_setup_score(state)
        + 0.35 * run_depth(state)
        + 0.08 * total_run_depth(state)
    )


def total_run_depth(state: GameState) -> int:
    return sum(top_run_length(column) for column in state.columns if column)


def top_stair_length(column: Sequence[Card]) -> int:
    if not column:
        return 0
    run = 1
    for index in range(len(column) - 1, 0, -1):
        upper = column[index]
        lower = column[index - 1]
        if lower.rank == upper.rank + 1:
            run += 1
        else:
            break
    return run


def empty_column_count(state: GameState) -> int:
    return sum(1 for column in state.columns if not column)


def effective_column_units(column: Sequence[Card]) -> int:
    if not column:
        return 0
    units = 1
    for index in range(1, len(column)):
        lower = column[index - 1]
        upper = column[index]
        if not (lower.suit == upper.suit and lower.rank == upper.rank + 1):
            units += 1
    return units


def effective_column_mass(state: GameState) -> int:
    return sum(effective_column_units(column) for column in state.columns)


def mixed_stair_bonus(state: GameState) -> float:
    bonus = 0.0
    for column in state.columns:
        for index in range(1, len(column)):
            lower = column[index - 1]
            upper = column[index]
            if lower.rank == upper.rank + 1 and lower.suit != upper.suit:
                bonus += 0.35
    return bonus


def hole_setup_score(state: GameState) -> float:
    score = 0.0
    for column in state.columns:
        if not column:
            continue
        movable_prefix = top_run_length(column)
        if movable_prefix == len(column):
            if len(column) == 1:
                score += 7.0
            elif len(column) == 2:
                score += 4.0
            elif len(column) == 3:
                score += 1.5
    return score


STOCK_DESTINATION_WEIGHT = 0.7
"""How much an undealt rank+1 counts next to one already exposed on the table."""

HOLE_TENANT_WEIGHT = 0.0
"""Reward for parking a recoverable card in a hole, per unit of destination supply.

Off by default: two A/B runs found no benefit, one of them confounded and the
other unable to discriminate because the self-play policy never wins. The
primitives above are correct and tested, so raise this once there is an
evaluation that can actually measure a change.
"""


def stock_rank_counts(state: GameState) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for card in state.stock:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def exposed_rank_counts(state: GameState) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for column in state.columns:
        if column:
            counts[column[-1].rank] = counts.get(column[-1].rank, 0) + 1
    return counts


def destination_supply(state: GameState, rank: int) -> float:
    """How readily a card of ``rank`` will find a home on a rank+1 card later.

    Used to judge what to park in a hole before dealing: a six is a good
    tenant while sevens are still undealt, because each deal exposes its card
    on top of a column and hands the six a destination, which wins the hole
    back. A king scores zero - it can never leave, so it spends the hole for
    good.
    """

    if rank >= 13:
        return 0.0
    target = rank + 1
    exposed = exposed_rank_counts(state).get(target, 0)
    in_stock = stock_rank_counts(state).get(target, 0)
    return float(exposed) + STOCK_DESTINATION_WEIGHT * in_stock


def mobility_score(state: GameState) -> float:
    actions = legal_moves(state)
    if not actions:
        return 0.0
    run_total = sum(action.run_length for action in actions)
    distinct_targets = len({action.to_column for action in actions})
    return float(run_total + distinct_targets)


def hole_access_score(state: GameState) -> float:
    actions = legal_moves(state)
    if not actions:
        return 0.0
    best_remaining_by_column: Dict[int, int] = {}
    for move in actions:
        remaining = len(state.columns[move.from_column]) - move.run_length
        current = best_remaining_by_column.get(move.from_column)
        if current is None or remaining < current:
            best_remaining_by_column[move.from_column] = remaining

    score = 0.0
    for remaining in best_remaining_by_column.values():
        if remaining == 0:
            score += 6.0
        elif remaining == 1:
            score += 2.5
        elif remaining == 2:
            score += 0.75
    return score


def top_color_mix_penalty(state: GameState) -> float:
    penalty = 0.0
    for column in state.columns:
        stair_length = top_stair_length(column)
        if stair_length < 2:
            continue
        suits = {card.suit for card in column[-stair_length:]}
        penalty += max(0, len(suits) - 1) * max(0, stair_length - 1)
    return penalty


def _rank_block_weight(rank: int) -> float:
    if rank in {1, 13}:
        return 0.0
    if rank in {2, 12}:
        return 0.35
    if rank == 11:
        return 0.8
    return 1.0


def blocked_mid_rank_risk(state: GameState) -> float:
    rank_counts: Dict[int, int] = {}
    risk = 0.0

    for column in state.columns:
        movable_suffix = top_run_length(column)
        buried_cards = column[:-movable_suffix] if movable_suffix else list(column)
        local_counts: Dict[int, int] = {}
        for index, card in enumerate(buried_cards):
            depth_from_top = len(column) - 1 - index
            if depth_from_top < 2:
                continue
            weight = _rank_block_weight(card.rank)
            if weight <= 0.0:
                continue
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
            local_counts[card.rank] = local_counts.get(card.rank, 0) + 1
            risk += weight * min(4.0, depth_from_top - 1) * 0.45

        for rank, count in local_counts.items():
            if count >= 2:
                risk += _rank_block_weight(rank) * (count - 1) * 2.8

    for rank, count in rank_counts.items():
        if count >= 3:
            risk += _rank_block_weight(rank) * (count - 2) * 3.6

    return risk


def move_creates_hole(state: GameState, action: Action) -> bool:
    if action.type != "move" or action.move is None:
        return False
    return len(state.columns[action.move.from_column]) == action.move.run_length


def hole_bonus(empty_columns: int) -> float:
    if empty_columns <= 0:
        return 0.0
    bonus = 12.0 * empty_columns
    if empty_columns >= 2:
        bonus += 16.0
    if empty_columns >= 3:
        bonus += 10.0
    return bonus


DEAD_STATE_PENALTY = 10000.0
"""Applied to states from which no empty column can ever be created again.

The penalty is subtracted rather than replacing the score, so that relative
ordering survives when every available action leads into a locked position.
"""


def evaluate_action(state: GameState, action: Action) -> float:
    next_state = apply_action(state, action)
    score = 0.0
    if is_provably_lost(next_state):
        score -= DEAD_STATE_PENALTY
    before_empty = empty_column_count(state)
    after_empty = empty_column_count(next_state)
    before_hole_setup = hole_setup_score(state)
    after_hole_setup = hole_setup_score(next_state)
    before_hole_access = hole_access_score(state)
    after_hole_access = hole_access_score(next_state)
    before_blocked_risk = blocked_mid_rank_risk(state)
    after_blocked_risk = blocked_mid_rank_risk(next_state)
    before_mix_penalty = top_color_mix_penalty(state)
    after_mix_penalty = top_color_mix_penalty(next_state)
    before_run_depth = run_depth(state)
    after_run_depth = run_depth(next_state)
    before_total_run_depth = total_run_depth(state)
    after_total_run_depth = total_run_depth(next_state)

    score += 250.0 if next_state.status == "won" else 0.0
    score += 80.0 * (next_state.completed_sequences - state.completed_sequences)
    score += 3.0 * (after_run_depth - before_run_depth)
    score += 1.5 * (after_total_run_depth - before_total_run_depth)
    score += hole_bonus(after_empty)
    score -= hole_bonus(before_empty) * 0.6
    score += 6.0 * (after_hole_setup - before_hole_setup)
    score += 7.0 * (after_hole_access - before_hole_access)
    score += 2.2 * (before_blocked_risk - after_blocked_risk)
    score += 0.22 * (before_mix_penalty - after_mix_penalty)
    score -= 0.45 * len(next_state.stock)
    if action.type == "move" and action.move:
        moved = state.columns[action.move.from_column][-action.move.run_length :]
        score += 2.0 * action.move.run_length
        destination = state.columns[action.move.to_column][-1] if state.columns[action.move.to_column] else None
        if destination and destination.suit == moved[0].suit:
            score += 4.0
        if destination is None:
            # Filling a hole: prefer a tenant we have a good chance of moving
            # on later, which wins the hole back.
            score += HOLE_TENANT_WEIGHT * destination_supply(state, moved[0].rank)
        if move_creates_hole(state, action):
            if before_empty == 0:
                score += 46.0
            elif before_empty == 1:
                score += 58.0
            else:
                score += 24.0
            if after_empty >= 2:
                score += 18.0
        remaining_source = len(state.columns[action.move.from_column]) - action.move.run_length
        if remaining_source == 1:
            score += 5.0
        elif remaining_source == 2:
            score += 1.5
    if action.type == "deal":
        score += 4.0 if not legal_moves(state) else -6.0
        if before_empty > 0:
            score -= 4.0
    return score


def evaluate_state_snapshot(state: GameState) -> float:
    score = 0.0
    empty_columns = empty_column_count(state)
    score += 400.0 if state.status == "won" else 0.0
    score += 90.0 * state.completed_sequences
    score += 3.5 * run_depth(state)
    score += 1.25 * total_run_depth(state)
    score += hole_bonus(empty_columns)
    score += 5.0 * hole_setup_score(state)
    score += 5.5 * hole_access_score(state)
    score += 1.2 * mobility_score(state)
    score -= 2.4 * blocked_mid_rank_risk(state)
    score -= 0.3 * top_color_mix_penalty(state)
    score -= 0.35 * len(state.stock)
    if is_provably_lost(state):
        score -= DEAD_STATE_PENALTY
    return score


def hole_goal_state_score(state: GameState) -> float:
    empty_columns = empty_column_count(state)
    score = 0.0
    if empty_columns >= 1:
        score += 90.0
    if empty_columns >= 2:
        score += 70.0 + 25.0 * (empty_columns - 2)
    score += 10.0 * hole_access_score(state)
    score += 4.0 * hole_setup_score(state)
    score += 0.6 * mobility_score(state)
    score -= 1.5 * blocked_mid_rank_risk(state)
    score -= 0.2 * top_color_mix_penalty(state)
    score += 0.6 * run_depth(state)
    score += 0.15 * total_run_depth(state)
    score += 150.0 * state.completed_sequences
    return score


def compact_goal_hole_bonus(empty_columns: int) -> float:
    if empty_columns <= 0:
        return 0.0
    bonus = 130.0
    if empty_columns >= 2:
        bonus += 180.0
    if empty_columns >= 3:
        bonus += 90.0 * (empty_columns - 2)
    return bonus


def compactness_goal_state_score(state: GameState) -> float:
    empty_columns = empty_column_count(state)
    score = 0.0
    if state.status == STATUS_WON:
        score += 600.0
    score += compact_goal_hole_bonus(empty_columns)
    score += 12.0 * (52 - effective_column_mass(state))
    score += 3.0 * hole_access_score(state)
    score += 1.5 * hole_setup_score(state)
    score += 4.0 * mixed_stair_bonus(state)
    return score


def _beam_states_for_compact_goal(states: Sequence[Tuple[GameState, int]], beam_width: int) -> List[Tuple[GameState, int]]:
    ranked = sorted(states, key=lambda item: compactness_goal_state_score(item[0]), reverse=True)
    selected: List[Tuple[GameState, int]] = []
    seen_hashes = set()
    for state, depth in ranked:
        if state.state_hash in seen_hashes:
            continue
        selected.append((state, depth))
        seen_hashes.add(state.state_hash)
        if len(selected) >= beam_width:
            break
    return selected


def best_compact_goal_outcome(
    state: GameState,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
) -> Dict[str, Any]:
    best_score = compactness_goal_state_score(state)
    best_empty = empty_column_count(state)
    min_effective_mass = effective_column_mass(state)
    first_hole_depth = 0 if best_empty > 0 else None
    frontier: List[Tuple[GameState, int]] = [(state, 0)]
    seen_depths = {state.state_hash: 0}

    for _ in range(horizon):
        expanded: List[Tuple[GameState, int]] = []
        for current_state, depth in frontier:
            current_empty = empty_column_count(current_state)
            best_empty = max(best_empty, current_empty)
            min_effective_mass = min(min_effective_mass, effective_column_mass(current_state))
            best_score = max(best_score, compactness_goal_state_score(current_state))
            if current_empty > 0 and first_hole_depth is None:
                first_hole_depth = depth
            if depth >= horizon or current_state.status != STATUS_IN_PROGRESS:
                continue
            for action in available_actions(current_state):
                next_state = apply_action(current_state, action)
                next_depth = depth + 1
                previous_depth = seen_depths.get(next_state.state_hash)
                if previous_depth is not None and previous_depth <= next_depth:
                    continue
                seen_depths[next_state.state_hash] = next_depth
                expanded.append((next_state, next_depth))
        if not expanded:
            break
        frontier = _beam_states_for_compact_goal(expanded, beam_width=beam_width)

    return {
        "reachable": first_hole_depth is not None,
        "first_hole_depth": first_hole_depth,
        "best_empty_columns": best_empty,
        "min_effective_mass": min_effective_mass,
        "best_score": best_score,
    }


def _beam_states_for_hole_goal(states: Sequence[Tuple[GameState, int]], beam_width: int) -> List[Tuple[GameState, int]]:
    ranked = sorted(states, key=lambda item: hole_goal_state_score(item[0]), reverse=True)
    selected: List[Tuple[GameState, int]] = []
    seen_hashes = set()
    for state, depth in ranked:
        if state.state_hash in seen_hashes:
            continue
        selected.append((state, depth))
        seen_hashes.add(state.state_hash)
        if len(selected) >= beam_width:
            break
    return selected


def best_hole_goal_outcome(
    state: GameState,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
) -> Dict[str, Any]:
    best_score = hole_goal_state_score(state)
    best_empty = empty_column_count(state)
    first_hole_depth = 0 if best_empty > 0 else None
    frontier: List[Tuple[GameState, int]] = [(state, 0)]
    seen_depths = {state.state_hash: 0}

    for _ in range(horizon):
        expanded: List[Tuple[GameState, int]] = []
        for current_state, depth in frontier:
            current_empty = empty_column_count(current_state)
            current_score = hole_goal_state_score(current_state)
            if current_score > best_score:
                best_score = current_score
                best_empty = current_empty
            if current_empty > 0 and first_hole_depth is None:
                first_hole_depth = depth
            if depth >= horizon or current_state.status != STATUS_IN_PROGRESS:
                continue
            for action in available_actions(current_state):
                next_state = apply_action(current_state, action)
                next_depth = depth + 1
                next_empty = empty_column_count(next_state)
                next_score = hole_goal_state_score(next_state)
                if next_score > best_score:
                    best_score = next_score
                    best_empty = next_empty
                if next_empty > 0 and first_hole_depth is None:
                    first_hole_depth = next_depth
                previous_depth = seen_depths.get(next_state.state_hash)
                if previous_depth is not None and previous_depth <= next_depth:
                    continue
                seen_depths[next_state.state_hash] = next_depth
                expanded.append((next_state, next_depth))
        if not expanded:
            break
        frontier = _beam_states_for_hole_goal(expanded, beam_width=beam_width)

    return {
        "reachable": first_hole_depth is not None,
        "first_hole_depth": first_hole_depth,
        "best_empty_columns": best_empty,
        "best_score": best_score,
    }


def choose_heuristic_action(state: GameState, rng: Optional[random.Random] = None) -> Optional[Action]:
    actions = available_actions(state)
    if not actions:
        return None
    rng = rng or random.Random()
    scored = [(evaluate_action(state, action), rng.random(), action) for action in actions]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def choose_random_action(state: GameState, rng: Optional[random.Random] = None) -> Optional[Action]:
    actions = available_actions(state)
    if not actions:
        return None
    rng = rng or random.Random()
    return rng.choice(actions)


def rank_hole_search_actions(
    state: GameState,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
) -> List[Tuple[float, Action]]:
    actions = available_actions(state)
    ranked: List[Tuple[float, Action]] = []
    for action in actions:
        next_state = apply_action(state, action)
        outcome = best_hole_goal_outcome(next_state, horizon=max(0, horizon - 1), beam_width=beam_width)
        score = outcome["best_score"]
        if outcome["reachable"]:
            score += 220.0
            depth = outcome["first_hole_depth"] if outcome["first_hole_depth"] is not None else horizon
            score += max(0.0, 25.0 - 4.0 * depth)
        score += 30.0 * min(2, outcome["best_empty_columns"])
        score += 0.4 * evaluate_action(state, action)
        ranked.append((score, action))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def choose_hole_search_action(
    state: GameState,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
) -> Optional[Action]:
    ranked = rank_hole_search_actions(state, horizon=horizon, beam_width=beam_width)
    return ranked[0][1] if ranked else None


def rank_compact_search_actions(
    state: GameState,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
) -> List[Tuple[float, Action]]:
    actions = available_actions(state)
    ranked: List[Tuple[float, Action]] = []
    for action in actions:
        next_state = apply_action(state, action)
        outcome = best_compact_goal_outcome(next_state, horizon=max(0, horizon - 1), beam_width=beam_width)
        score = outcome["best_score"]
        if outcome["reachable"]:
            depth = outcome["first_hole_depth"] if outcome["first_hole_depth"] is not None else horizon
            score += max(0.0, 18.0 - 3.0 * depth)
        score += 0.35 * evaluate_action(state, action)
        ranked.append((score, action))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def choose_compact_search_action(
    state: GameState,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
) -> Optional[Action]:
    ranked = rank_compact_search_actions(state, horizon=horizon, beam_width=beam_width)
    return ranked[0][1] if ranked else None


def action_signature(state: GameState, action: Action) -> str:
    return action.to_dict(state)["description"]


def action_payload_matches(action: Action, payload: Dict[str, Any]) -> bool:
    if action.type != payload.get("type"):
        return False
    if action.type == "deal":
        return True
    if action.type != "move" or action.move is None:
        return False
    return (
        action.move.from_column == payload.get("from_column")
        and action.move.to_column == payload.get("to_column")
        and action.move.run_length == payload.get("run_length")
    )


def _hole_teacher_targets(
    state: GameState,
    actions: Sequence[Action],
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
) -> Tuple[List[Tuple[float, Action]], Dict[str, float]]:
    ranked_actions = rank_hole_search_actions(state, horizon=horizon, beam_width=beam_width)
    score_map = {action_signature(state, action): score for score, action in ranked_actions}
    for action in actions:
        signature = action_signature(state, action)
        if signature not in score_map:
            score_map[signature] = 0.4 * evaluate_action(state, action)
    return ranked_actions, score_map


def _rollout_tail_value(
    state: GameState,
    steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
    stagnation_limit: int = 6,
    repeat_limit: int = 2,
) -> float:
    rollout_state = state
    best_value = evaluate_state_snapshot(rollout_state)
    repeated_states: Dict[str, int] = {}
    best_progress = progress_score(rollout_state)
    moves_since_progress = 0

    for _ in range(steps):
        if rollout_state.status != STATUS_IN_PROGRESS:
            break
        action = choose_heuristic_action(rollout_state, rng=random.Random(rollout_state.moves_played + len(rollout_state.stock)))
        if action is None:
            break
        rollout_state = apply_action(rollout_state, action)
        best_value = max(best_value, evaluate_state_snapshot(rollout_state))

        current_progress = progress_score(rollout_state)
        if current_progress > best_progress:
            best_progress = current_progress
            moves_since_progress = 0
        else:
            moves_since_progress += 1

        repeated_states[rollout_state.state_hash] = repeated_states.get(rollout_state.state_hash, 0) + 1
        if repeated_states[rollout_state.state_hash] >= repeat_limit or moves_since_progress >= stagnation_limit:
            break
    return best_value


def _search_state_value(
    state: GameState,
    depth: int,
    beam_width: int,
    discount: float,
    rollout_steps: int,
    cache: Dict[Tuple[str, int], float],
    ancestry: Sequence[str],
) -> float:
    cache_key = (state.state_hash, depth)
    if cache_key in cache:
        return cache[cache_key]

    base_value = evaluate_state_snapshot(state)
    if state.status != STATUS_IN_PROGRESS or depth <= 0:
        cache[cache_key] = max(base_value, _rollout_tail_value(state, steps=rollout_steps))
        return cache[cache_key]

    repeated_visits = ancestry.count(state.state_hash)
    if repeated_visits >= 2:
        penalized = base_value - 50.0
        cache[cache_key] = penalized
        return penalized

    actions = available_actions(state)
    if not actions:
        cache[cache_key] = base_value - 25.0
        return cache[cache_key]

    candidates = sorted(actions, key=lambda action: evaluate_action(state, action), reverse=True)[:beam_width]
    best_value = float("-inf")
    next_ancestry = list(ancestry) + [state.state_hash]
    for action in candidates:
        next_state = apply_action(state, action)
        future_value = _search_state_value(
            next_state,
            depth - 1,
            beam_width,
            discount,
            rollout_steps,
            cache,
            next_ancestry,
        )
        value = evaluate_action(state, action) * 0.35 + discount * future_value
        if value > best_value:
            best_value = value
    cache[cache_key] = max(base_value, best_value)
    return cache[cache_key]


def rank_search_actions(
    state: GameState,
    depth: int = DEFAULT_SEARCH_DEPTH,
    beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    discount: float = DEFAULT_SEARCH_DISCOUNT,
    rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> List[Tuple[float, Action]]:
    actions = available_actions(state)
    cache: Dict[Tuple[str, int], float] = {}
    ranked: List[Tuple[float, Action]] = []
    for action in actions:
        next_state = apply_action(state, action)
        future_value = _search_state_value(
            next_state,
            max(0, depth - 1),
            beam_width,
            discount,
            rollout_steps,
            cache,
            [state.state_hash],
        )
        score = evaluate_action(state, action) * 0.35 + discount * future_value
        ranked.append((score, action))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def rank_teacher_search_actions(
    state: GameState,
    depth: int = DEFAULT_SEARCH_DEPTH,
    beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    discount: float = DEFAULT_SEARCH_DISCOUNT,
    rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> List[Tuple[float, Action]]:
    if should_use_hole_opening_signal(state):
        return rank_hole_search_actions(
            state,
            horizon=DEFAULT_HOLE_GOAL_HORIZON,
            beam_width=DEFAULT_HOLE_GOAL_BEAM_WIDTH,
        )
    return rank_search_actions(
        state,
        depth=depth,
        beam_width=beam_width,
        discount=discount,
        rollout_steps=rollout_steps,
    )


def rank_search_actions_subset(
    state: GameState,
    candidate_actions: Sequence[Action],
    depth: int = DEFAULT_SEARCH_DEPTH,
    beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    discount: float = DEFAULT_SEARCH_DISCOUNT,
    rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> List[Tuple[float, Action]]:
    cache: Dict[Tuple[str, int], float] = {}
    ranked: List[Tuple[float, Action]] = []
    for action in candidate_actions:
        next_state = apply_action(state, action)
        future_value = _search_state_value(
            next_state,
            max(0, depth - 1),
            beam_width,
            discount,
            rollout_steps,
            cache,
            [state.state_hash],
        )
        score = evaluate_action(state, action) * 0.35 + discount * future_value
        ranked.append((score, action))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def choose_search_action(
    state: GameState,
    depth: int = DEFAULT_SEARCH_DEPTH,
    beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    discount: float = DEFAULT_SEARCH_DISCOUNT,
    rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> Optional[Action]:
    ranked = rank_search_actions(
        state,
        depth=depth,
        beam_width=beam_width,
        discount=discount,
        rollout_steps=rollout_steps,
    )
    return ranked[0][1] if ranked else None


def global_state_features(state: GameState) -> np.ndarray:
    features: List[float] = []
    for column in state.columns:
        height = len(column) / 52.0
        run_len = top_run_length(column) / 13.0 if column else 0.0
        top_rank = (column[-1].rank / 13.0) if column else 0.0
        top_suit = [0.0, 0.0, 0.0, 0.0]
        if column:
            top_suit[["hearts", "diamonds", "clubs", "spades"].index(column[-1].suit)] = 1.0
        features.extend([height, run_len, top_rank, 1.0 if not column else 0.0])
        features.extend(top_suit)
    features.extend(
        [
            len(state.stock) / 24.0,
            state.completed_sequences / 4.0,
            progress_score(state) / 52.0,
            state.moves_played / 200.0,
            hole_access_score(state) / 24.0,
            blocked_mid_rank_risk(state) / 40.0,
            top_color_mix_penalty(state) / 20.0,
        ]
    )
    return np.asarray(features, dtype=np.float32)


def visible_tableau_card_features(state: GameState) -> np.ndarray:
    features = np.zeros(len(CANONICAL_CARDS) * VISIBLE_CARD_FEATURES_PER_CARD, dtype=np.float32)
    visible_positions: Dict[Card, Tuple[int, int]] = {}
    for column_index, column in enumerate(state.columns):
        for depth_from_bottom, card in enumerate(column):
            visible_positions[card] = (column_index, depth_from_bottom)

    for card_index, card in enumerate(CANONICAL_CARDS):
        offset = card_index * VISIBLE_CARD_FEATURES_PER_CARD
        position = visible_positions.get(card)
        if position is None:
            features[offset] = 1.0
            continue
        column_index, depth_from_bottom = position
        features[offset + 1 + column_index] = 1.0
        features[offset + 8] = depth_from_bottom / MAX_CARD_DEPTH
    return features


def action_features(state: GameState, action: Action) -> np.ndarray:
    features: List[float] = []
    type_flags = [1.0 if action.type == "move" else 0.0, 1.0 if action.type == "deal" else 0.0]
    features.extend(type_flags)
    from_one_hot = [0.0] * 7
    to_one_hot = [0.0] * 7
    run_length = 0.0
    lead_rank = 0.0
    dest_rank = 0.0
    same_suit = 0.0
    hole = 0.0
    if action.type == "move" and action.move:
        from_one_hot[action.move.from_column] = 1.0
        to_one_hot[action.move.to_column] = 1.0
        run_length = action.move.run_length / 13.0
        moved = state.columns[action.move.from_column][-action.move.run_length :]
        lead_rank = moved[0].rank / 13.0
        destination = state.columns[action.move.to_column][-1] if state.columns[action.move.to_column] else None
        if destination:
            dest_rank = destination.rank / 13.0
            same_suit = 1.0 if destination.suit == moved[0].suit else 0.0
        else:
            hole = 1.0
    features.extend(from_one_hot)
    features.extend(to_one_hot)
    features.extend([run_length, lead_rank, dest_rank, same_suit, hole])
    return np.asarray(features, dtype=np.float32)


def encode_action(state: GameState, action: Action) -> np.ndarray:
    return np.concatenate(
        [
            visible_tableau_card_features(state),
            global_state_features(state),
            action_features(state, action),
            action_outcome_features(state, action),
        ]
    )


def action_outcome_features(state: GameState, action: Action) -> np.ndarray:
    if action.type == "deal":
        cards_to_deal = min(7, len(state.stock))
        before_empty = empty_column_count(state)
        empties_filled = sum(1 for column_index in range(cards_to_deal) if not state.columns[column_index])
        after_empty = max(0, before_empty - empties_filled)
        features = [
            state.completed_sequences / 4.0,
            run_depth(state) / 13.0,
            total_run_depth(state) / 52.0,
            after_empty / 7.0,
            hole_setup_score(state) / 20.0,
            hole_access_score(state) / 24.0,
            mobility_score(state) / 30.0,
            (after_empty - before_empty) / 3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if after_empty >= 2 else 0.0,
            max(0, len(state.stock) - cards_to_deal) / 24.0,
            progress_score(state) / 52.0,
        ]
        return np.asarray(features, dtype=np.float32)

    next_state = apply_action(state, action)
    before_empty = empty_column_count(state)
    after_empty = empty_column_count(next_state)
    before_mobility = mobility_score(state)
    after_mobility = mobility_score(next_state)
    before_hole_access = hole_access_score(state)
    after_hole_access = hole_access_score(next_state)
    before_blocked_risk = blocked_mid_rank_risk(state)
    after_blocked_risk = blocked_mid_rank_risk(next_state)
    before_mix_penalty = top_color_mix_penalty(state)
    after_mix_penalty = top_color_mix_penalty(next_state)
    features = [
        next_state.completed_sequences / 4.0,
        run_depth(next_state) / 13.0,
        total_run_depth(next_state) / 52.0,
        after_empty / 7.0,
        hole_setup_score(next_state) / 20.0,
        after_hole_access / 24.0,
        after_mobility / 30.0,
        (after_empty - before_empty) / 3.0,
        (after_hole_access - before_hole_access) / 12.0,
        (after_mobility - before_mobility) / 30.0,
        (before_blocked_risk - after_blocked_risk) / 20.0,
        (before_mix_penalty - after_mix_penalty) / 12.0,
        1.0 if move_creates_hole(state, action) else 0.0,
        1.0 if after_empty >= 2 else 0.0,
        len(next_state.stock) / 24.0,
        progress_score(next_state) / 52.0,
    ]
    return np.asarray(features, dtype=np.float32)


class NumpyPolicyNetwork:
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        scale = 0.15
        self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, 1).astype(np.float32) * scale
        self.b2 = np.zeros(1, dtype=np.float32)

    def score(self, batch: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, batch @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return logits.reshape(-1)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        logits = self.score(batch)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def train(self, features: np.ndarray, labels: np.ndarray, epochs: int = 25, learning_rate: float = 0.02) -> Dict[str, float]:
        losses: List[float] = []
        for _ in range(epochs):
            hidden_linear = features @ self.w1 + self.b1
            hidden = np.maximum(0.0, hidden_linear)
            logits = hidden @ self.w2 + self.b2
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            loss = -np.mean(labels.reshape(-1, 1) * np.log(probs + 1e-8) + (1.0 - labels.reshape(-1, 1)) * np.log(1.0 - probs + 1e-8))
            losses.append(float(loss))

            grad_logits = (probs - labels.reshape(-1, 1)) / len(features)
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ self.w2.T
            grad_hidden[hidden_linear <= 0.0] = 0.0
            grad_w1 = features.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            self.w2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2
            self.w1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1
        accuracy = float(np.mean((self.predict(features) >= 0.5) == labels))
        return {"loss": losses[-1] if losses else 0.0, "accuracy": accuracy}

    def train_ranked(
        self,
        cases: Sequence[TrainingCase],
        epochs: int = 25,
        learning_rate: float = 0.02,
        progress: ProgressCallback = None,
    ) -> Dict[str, float]:
        if not cases:
            raise RuntimeError("No training cases available.")

        losses: List[float] = []
        accuracies: List[float] = []
        rng = random.Random(1234)
        total_weight = sum(case.weight for case in cases)

        for epoch in range(epochs):
            order = list(range(len(cases)))
            rng.shuffle(order)
            weighted_loss = 0.0
            weighted_correct = 0.0

            for index in order:
                case = cases[index]
                hidden_linear = case.features @ self.w1 + self.b1
                hidden = np.maximum(0.0, hidden_linear)
                logits = (hidden @ self.w2 + self.b2).reshape(-1)
                shifted = logits - np.max(logits)
                exp_logits = np.exp(np.clip(shifted, -30.0, 30.0))
                probs = exp_logits / np.sum(exp_logits)
                target_distribution = case.soft_targets * (1.0 - case.target_boost)
                target_distribution[case.target_index] += case.target_boost
                loss = -np.sum(target_distribution * np.log(probs + 1e-8))
                weighted_loss += case.weight * float(loss)
                if int(np.argmax(probs)) == case.target_index:
                    weighted_correct += case.weight

                grad_logits = probs - target_distribution
                grad_logits *= case.weight
                grad_logits = grad_logits.reshape(-1, 1)

                grad_w2 = hidden.T @ grad_logits
                grad_b2 = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ self.w2.T
                grad_hidden[hidden_linear <= 0.0] = 0.0
                grad_w1 = case.features.T @ grad_hidden
                grad_b1 = grad_hidden.sum(axis=0)

                scale = max(1.0, case.weight)
                self.w2 -= learning_rate * grad_w2 / scale
                self.b2 -= learning_rate * grad_b2 / scale
                self.w1 -= learning_rate * grad_w1 / scale
                self.b1 -= learning_rate * grad_b1 / scale

            losses.append(weighted_loss / total_weight)
            accuracies.append(weighted_correct / total_weight)
            emit_progress(
                progress,
                f"[train] epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f} top1={accuracies[-1]:.3f}",
            )

        return {"loss": losses[-1], "accuracy": accuracies[-1]}

    def save(self, path: Path) -> None:
        np.savez(
            path,
            input_dim=np.asarray([self.input_dim]),
            hidden_dim=np.asarray([self.hidden_dim]),
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyPolicyNetwork":
        data = np.load(path)
        model = cls(int(data["input_dim"][0]), int(data["hidden_dim"][0]))
        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = data["b2"]
        return model


def build_training_examples(
    num_games: int = 60,
    max_moves: int = 180,
    stagnation_limit: int = 30,
    repeat_limit: int = 3,
    seed_offset: int = 0,
    teacher_policy: str = "search",
    search_depth: int = DEFAULT_TEACHER_SEARCH_DEPTH,
    beam_width: int = DEFAULT_TEACHER_SEARCH_BEAM_WIDTH,
    discount: float = DEFAULT_TEACHER_SEARCH_DISCOUNT,
    rollout_steps: int = DEFAULT_TEACHER_SEARCH_ROLLOUT_STEPS,
    alternate_actions: int = DEFAULT_TEACHER_ALTERNATE_ACTIONS,
    max_states_per_game: int = DEFAULT_TEACHER_MAX_STATES_PER_GAME,
    branch_margin: float = DEFAULT_TEACHER_BRANCH_MARGIN,
    state_source: str = "random_seed",
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cases = build_training_cases(
        num_games=num_games,
        max_moves=max_moves,
        stagnation_limit=stagnation_limit,
        repeat_limit=repeat_limit,
        seed_offset=seed_offset,
        teacher_policy=teacher_policy,
        search_depth=search_depth,
        beam_width=beam_width,
        discount=discount,
        rollout_steps=rollout_steps,
        alternate_actions=alternate_actions,
        max_states_per_game=max_states_per_game,
        branch_margin=branch_margin,
        state_source=state_source,
        sessions_dir=sessions_dir,
        progress=progress,
    )
    examples: List[np.ndarray] = []
    labels: List[float] = []
    for case in cases:
        for action_index, feature_row in enumerate(case.features):
            examples.append(feature_row)
            labels.append(1.0 if action_index == case.target_index else 0.0)
    if not examples:
        raise RuntimeError("No training examples were generated.")
    return np.vstack(examples), np.asarray(labels, dtype=np.float32)


def training_game_seeds(num_games: int, seed_offset: int = 0) -> List[int]:
    rng = random.Random(17 + seed_offset)
    seeds: List[int] = []
    seen = set()
    while len(seeds) < num_games:
        candidate = rng.randrange(1_000_000_000)
        if candidate in seen:
            continue
        seen.add(candidate)
        seeds.append(candidate)
    return seeds


def _deserialize_session_state(payload: Dict[str, Any]) -> GameState:
    return GameState(
        columns=[
            [Card(rank=int(card["rank"]), suit=str(card["suit"])) for card in column]
            for column in payload["columns"]
        ],
        stock=[Card(rank=int(card["rank"]), suit=str(card["suit"])) for card in payload["stock"]],
        status=str(payload["status"]),
        moves_played=int(payload["moves_played"]),
        terminal_reason=payload.get("terminal_reason"),
        state_hash=str(payload.get("state_hash", "")),
        completed_sequences=int(payload.get("completed_sequences", 0)),
    )


def load_winning_initial_states(sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> List[TrainingStartState]:
    states: List[TrainingStartState] = []
    if not sessions_dir.exists():
        return states

    for path in sorted(sessions_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = payload.get("history") or []
        if not history:
            continue
        last_state = history[-1]
        if str(last_state.get("status", STATUS_IN_PROGRESS)) != STATUS_WON:
            continue
        try:
            initial_state = _deserialize_session_state(history[0])
        except (KeyError, TypeError, ValueError):
            continue
        states.append(
            TrainingStartState(
                source_id=str(payload.get("game_id", path.stem)),
                state=initial_state,
                source_type="won_initial_state",
            )
        )
    return states


def training_start_states(
    num_games: int,
    seed_offset: int = 0,
    state_source: str = "random_seed",
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> List[TrainingStartState]:
    if state_source == "random_seed":
        return [
            TrainingStartState(source_id=str(seed), state=create_game(seed=seed), source_type="random_seed")
            for seed in training_game_seeds(num_games=num_games, seed_offset=seed_offset)
        ]
    if state_source != "won_initial_states":
        raise ValueError(f"Unsupported training state source: {state_source}")

    pool = load_winning_initial_states(sessions_dir=sessions_dir)
    if not pool:
        raise RuntimeError(f"No winning initial states found in {sessions_dir}.")

    rng = random.Random(17 + seed_offset)
    ordered_pool = list(pool)
    rng.shuffle(ordered_pool)
    selected: List[TrainingStartState] = []
    while len(selected) < num_games:
        for item in ordered_pool:
            selected.append(
                TrainingStartState(
                    source_id=item.source_id,
                    state=item.state.clone(),
                    source_type=item.source_type,
                )
            )
            if len(selected) >= num_games:
                break
        if len(selected) < num_games:
            rng.shuffle(ordered_pool)
    emit_progress(
        progress,
        f"[data] loaded {len(pool)} winning initial states from {sessions_dir}, sampling {num_games}",
    )
    return selected


def build_training_cases(
    num_games: int = 60,
    max_moves: int = 180,
    stagnation_limit: int = 30,
    repeat_limit: int = 3,
    seed_offset: int = 0,
    teacher_policy: str = "search",
    search_depth: int = DEFAULT_TEACHER_SEARCH_DEPTH,
    beam_width: int = DEFAULT_TEACHER_SEARCH_BEAM_WIDTH,
    discount: float = DEFAULT_TEACHER_SEARCH_DISCOUNT,
    rollout_steps: int = DEFAULT_TEACHER_SEARCH_ROLLOUT_STEPS,
    alternate_actions: int = DEFAULT_TEACHER_ALTERNATE_ACTIONS,
    max_states_per_game: int = DEFAULT_TEACHER_MAX_STATES_PER_GAME,
    branch_margin: float = DEFAULT_TEACHER_BRANCH_MARGIN,
    state_source: str = "random_seed",
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> List[TrainingCase]:
    cases: List[TrainingCase] = []
    rng = random.Random(17 + seed_offset)
    start_states = training_start_states(
        num_games=num_games,
        seed_offset=seed_offset,
        state_source=state_source,
        sessions_dir=sessions_dir,
        progress=progress,
    )
    progress_every = max(1, num_games // 10)
    emit_progress(
        progress,
        f"[data] generating training cases from {num_games} games using {teacher_policy} "
        f"(state_source={state_source})",
    )

    for game_index, start in enumerate(start_states, start=1):
        state_queue: List[GameState] = [start.state.clone()]
        queued_hashes = {state_queue[0].state_hash}
        explored_states = 0

        while state_queue and explored_states < max_states_per_game:
            state = state_queue.pop(0)
            explored_states += 1
            emit_progress(
                progress,
                "[data] "
                f"game={game_index}/{num_games} start={start.source_type}:{start.source_id} "
                f"exploring branch {explored_states}/{max_states_per_game} queue={len(state_queue)}",
            )
            repeated_states: Dict[str, int] = {}
            best_progress = progress_score(state)
            moves_since_progress = 0

            for _ in range(max_moves):
                actions = available_actions(state)
                if not actions:
                    break
                if teacher_policy == "search":
                    ranked_actions = rank_teacher_search_actions(
                        state,
                        depth=search_depth,
                        beam_width=beam_width,
                        discount=discount,
                        rollout_steps=rollout_steps,
                    )
                    chosen = ranked_actions[0][1] if ranked_actions else None
                    teacher_scores = [score for score, _ in ranked_actions]
                    teacher_order = [action_signature(state, action) for _, action in ranked_actions]
                    enqueue_alternate_states(
                        state,
                        ranked_actions,
                        state_queue,
                        queued_hashes,
                        max_states_per_game=max_states_per_game,
                        alternate_actions=alternate_actions,
                        branch_margin=branch_margin,
                    )
                elif teacher_policy == "heuristic":
                    chosen = choose_heuristic_action(state, rng=rng)
                    scored = sorted(
                        ((evaluate_action(state, action), action) for action in actions),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    teacher_scores = [score for score, _ in scored]
                    teacher_order = [action_signature(state, action) for _, action in scored]
                elif teacher_policy == "hole_search":
                    ranked_actions, score_map = _hole_teacher_targets(
                        state,
                        actions,
                        horizon=DEFAULT_HOLE_GOAL_HORIZON,
                        beam_width=DEFAULT_HOLE_GOAL_BEAM_WIDTH,
                    )
                    chosen = ranked_actions[0][1] if ranked_actions else None
                    teacher_order = [action_signature(state, action) for _, action in ranked_actions]
                    teacher_scores = [score_map[signature] for signature in teacher_order]
                else:
                    raise ValueError(f"Unsupported teacher policy: {teacher_policy}")
                if chosen is None:
                    break
                chosen_signature = action_signature(state, chosen)
                action_rows = [encode_action(state, action) for action in actions]
                signatures = [action_signature(state, action) for action in actions]
                target_index = signatures.index(chosen_signature)
                ranked_map = {signature: score for signature, score in zip(teacher_order, teacher_scores)}
                best_score = ranked_map[chosen_signature]
                competing_scores = [score for signature, score in ranked_map.items() if signature != chosen_signature]
                second_best = max(competing_scores) if competing_scores else best_score
                gap = max(0.0, best_score - second_best)
                soft_targets = teacher_soft_targets(signatures, ranked_map)
                weight = 1.0 + min(3.0, gap / 25.0)
                if move_creates_hole(state, chosen):
                    weight += 0.75
                if empty_column_count(apply_action(state, chosen)) >= 2:
                    weight += 0.75
                cases.append(
                    TrainingCase(
                        features=np.vstack(action_rows),
                        target_index=target_index,
                        weight=weight,
                        chosen_signature=chosen_signature,
                        action_signatures=signatures,
                        soft_targets=soft_targets,
                        target_boost=DEFAULT_GENERATED_TARGET_BOOST,
                        source="generated",
                    )
                )
                state = apply_action(state, chosen)
                if state.status != STATUS_IN_PROGRESS:
                    break
                current_progress = progress_score(state)
                if current_progress > best_progress:
                    best_progress = current_progress
                    moves_since_progress = 0
                else:
                    moves_since_progress += 1

                repeated_states[state.state_hash] = repeated_states.get(state.state_hash, 0) + 1
                if repeated_states[state.state_hash] >= repeat_limit or moves_since_progress >= stagnation_limit:
                    state = terminate_for_training(state, "stagnation")
                    break
        completed = game_index
        if completed % progress_every == 0 or completed == num_games:
            emit_progress(
                progress,
                f"[data] processed {completed}/{num_games} games, cases={len(cases)}, explored_states={explored_states}",
            )
    if not cases:
        raise RuntimeError("No training examples were generated.")
    return cases


def build_hole_opening_cases(
    num_games: int = 60,
    seed_offset: int = 0,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> List[TrainingCase]:
    cases: List[TrainingCase] = []
    game_seeds = training_game_seeds(num_games=num_games, seed_offset=seed_offset)
    progress_every = max(1, num_games // 10)
    emit_progress(progress, f"[hole-data] generating opening cases from {num_games} games")

    for game_index, game_seed in enumerate(game_seeds, start=1):
        state = create_game(seed=game_seed)
        actions = available_actions(state)
        if not actions:
            continue
        ranked_actions, score_map = _hole_teacher_targets(state, actions, horizon=horizon, beam_width=beam_width)
        if not ranked_actions:
            continue
        chosen = ranked_actions[0][1]
        signatures = [action_signature(state, action) for action in actions]
        chosen_signature = action_signature(state, chosen)
        target_index = signatures.index(chosen_signature)
        soft_targets = teacher_soft_targets(signatures, score_map)
        best_outcome = best_hole_goal_outcome(apply_action(state, chosen), horizon=max(0, horizon - 1), beam_width=beam_width)
        weight = 1.0
        if best_outcome["reachable"]:
            weight += 1.5
        if best_outcome["best_empty_columns"] >= 2:
            weight += 0.5
        cases.append(
            TrainingCase(
                features=np.vstack([encode_action(state, action) for action in actions]),
                target_index=target_index,
                weight=weight,
                chosen_signature=chosen_signature,
                action_signatures=signatures,
                soft_targets=soft_targets,
                target_boost=0.82,
                source="hole_opening",
            )
        )
        if game_index % progress_every == 0 or game_index == num_games:
            emit_progress(progress, f"[hole-data] processed {game_index}/{num_games} games, cases={len(cases)}")

    if not cases:
        raise RuntimeError("No hole-opening examples were generated.")
    return cases


def build_compact_opening_cases(
    num_games: int = 60,
    seed_offset: int = 0,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> List[TrainingCase]:
    cases: List[TrainingCase] = []
    game_seeds = training_game_seeds(num_games=num_games, seed_offset=seed_offset)
    progress_every = max(1, num_games // 10)
    emit_progress(progress, f"[compact-data] generating opening cases from {num_games} games")

    for game_index, game_seed in enumerate(game_seeds, start=1):
        state = create_game(seed=game_seed)
        actions = available_actions(state)
        if not actions:
            continue
        ranked_actions = rank_compact_search_actions(state, horizon=horizon, beam_width=beam_width)
        if not ranked_actions:
            continue
        chosen = ranked_actions[0][1]
        ranked_map = {action_signature(state, action): score for score, action in ranked_actions}
        signatures = [action_signature(state, action) for action in actions]
        chosen_signature = action_signature(state, chosen)
        target_index = signatures.index(chosen_signature)
        soft_targets = teacher_soft_targets(signatures, ranked_map)
        best_outcome = best_compact_goal_outcome(apply_action(state, chosen), horizon=max(0, horizon - 1), beam_width=beam_width)
        weight = 1.0
        if best_outcome["reachable"]:
            weight += 1.25
        if best_outcome["best_empty_columns"] >= 2:
            weight += 0.75
        if best_outcome["min_effective_mass"] <= effective_column_mass(state) - 2:
            weight += 0.5
        cases.append(
            TrainingCase(
                features=np.vstack([encode_action(state, action) for action in actions]),
                target_index=target_index,
                weight=weight,
                chosen_signature=chosen_signature,
                action_signatures=signatures,
                soft_targets=soft_targets,
                target_boost=0.86,
                source="compact_opening",
            )
        )
        if game_index % progress_every == 0 or game_index == num_games:
            emit_progress(progress, f"[compact-data] processed {game_index}/{num_games} games, cases={len(cases)}")

    if not cases:
        raise RuntimeError("No compact-opening examples were generated.")
    return cases


def _compact_midgame_state(
    seed: int,
    min_steps: int = DEFAULT_COMPACT_MIDGAME_MIN_STEPS,
    max_steps: int = DEFAULT_COMPACT_MIDGAME_MAX_STEPS,
) -> Optional[GameState]:
    state = create_game(seed=seed)
    rng = random.Random(seed + 7000)
    target_steps = rng.randint(min_steps, max_steps)

    for _ in range(target_steps):
        if state.status != STATUS_IN_PROGRESS:
            break
        action = choose_search_action(
            state,
            depth=DEFAULT_UI_SEARCH_DEPTH,
            beam_width=DEFAULT_UI_SEARCH_BEAM_WIDTH,
            discount=DEFAULT_UI_SEARCH_DISCOUNT,
            rollout_steps=DEFAULT_UI_SEARCH_ROLLOUT_STEPS,
        )
        if action is None:
            break
        state = apply_action(state, action)

    if state.status != STATUS_IN_PROGRESS:
        return None
    if state.moves_played < min_steps:
        return None
    if not available_actions(state):
        return None
    return state


def build_compact_midgame_cases(
    num_games: int = 40,
    seed_offset: int = 0,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
    min_steps: int = DEFAULT_COMPACT_MIDGAME_MIN_STEPS,
    max_steps: int = DEFAULT_COMPACT_MIDGAME_MAX_STEPS,
    progress: ProgressCallback = None,
) -> List[TrainingCase]:
    cases: List[TrainingCase] = []
    game_seeds = training_game_seeds(num_games=num_games, seed_offset=seed_offset)
    progress_every = max(1, num_games // 10)
    emit_progress(progress, f"[compact-midgame] generating midgame cases from {num_games} games")

    for game_index, game_seed in enumerate(game_seeds, start=1):
        state = _compact_midgame_state(
            game_seed,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        if state is None:
            continue
        actions = available_actions(state)
        if not actions:
            continue
        ranked_actions = rank_compact_search_actions(state, horizon=horizon, beam_width=beam_width)
        if not ranked_actions:
            continue
        chosen = ranked_actions[0][1]
        ranked_map = {action_signature(state, action): score for score, action in ranked_actions}
        signatures = [action_signature(state, action) for action in actions]
        chosen_signature = action_signature(state, chosen)
        target_index = signatures.index(chosen_signature)
        soft_targets = teacher_soft_targets(signatures, ranked_map)
        best_outcome = best_compact_goal_outcome(apply_action(state, chosen), horizon=max(0, horizon - 1), beam_width=beam_width)
        weight = 1.1
        if best_outcome["reachable"]:
            weight += 1.0
        if best_outcome["best_empty_columns"] >= 2:
            weight += 0.75
        if best_outcome["min_effective_mass"] <= effective_column_mass(state) - 2:
            weight += 0.65
        cases.append(
            TrainingCase(
                features=np.vstack([encode_action(state, action) for action in actions]),
                target_index=target_index,
                weight=weight,
                chosen_signature=chosen_signature,
                action_signatures=signatures,
                soft_targets=soft_targets,
                target_boost=0.9,
                source="compact_midgame",
            )
        )
        if game_index % progress_every == 0 or game_index == num_games:
            emit_progress(progress, f"[compact-midgame] processed {game_index}/{num_games} games, cases={len(cases)}")

    if not cases:
        raise RuntimeError("No compact-midgame examples were generated.")
    return cases


def train_policy_model(
    num_games: int = 60,
    epochs: int = 30,
    learning_rate: float = 0.02,
    hidden_dim: int = 64,
    teacher_policy: str = "search",
    benchmark_games: int = 20,
    training_state_source: str = DEFAULT_TRAINING_STATE_SOURCE,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    generated_cases = build_training_cases(
        num_games=num_games,
        teacher_policy=teacher_policy,
        state_source=training_state_source,
        sessions_dir=sessions_dir,
        progress=progress,
    )
    include_human_feedback = training_state_source != "won_initial_states"
    if include_human_feedback:
        human_cases, human_feedback_summary = load_human_feedback_cases(progress=progress)
    else:
        human_cases = []
        human_feedback_summary = HumanFeedbackSummary()
        emit_progress(progress, "[feedback] disabled for won_initial_states training")
    cases, human_repeat_factor = expand_human_cases_for_training(generated_cases, human_cases, progress=progress)
    benchmark_seeds = list(range(benchmark_games))
    active_model_before = load_latest_model()
    active_metadata_before = load_model_metadata()
    model = NumpyPolicyNetwork(input_dim=cases[0].features.shape[1], hidden_dim=hidden_dim)
    emit_progress(
        progress,
        f"[train] starting ranked training on {len(cases)} decision states "
        f"({len(human_cases)} human raw, {len(generated_cases)} generated, "
        f"{len(human_cases) * human_repeat_factor} human effective)",
    )
    metrics = model.train_ranked(cases, epochs=epochs, learning_rate=learning_rate, progress=progress)
    emit_progress(progress, "[benchmark] evaluating heuristic")
    heuristic_eval = summarize_evaluation(
        evaluate_policy(policy="heuristic", seeds=benchmark_seeds, progress=progress)
    )
    emit_progress(progress, "[benchmark] evaluating search")
    search_eval = summarize_evaluation(
        evaluate_policy(policy="search", seeds=benchmark_seeds, progress=progress)
    )
    emit_progress(progress, "[benchmark] evaluating candidate model")
    candidate_eval = summarize_evaluation(
        evaluate_policy(policy="model", seeds=benchmark_seeds, model=model, progress=progress)
    )
    active_eval_before = None
    if active_model_before is not None:
        emit_progress(progress, "[benchmark] evaluating active model")
        active_eval_before = summarize_evaluation(
            evaluate_policy(policy="model", seeds=benchmark_seeds, model=active_model_before, progress=progress)
        )

    promoted_to_active = model_is_better(candidate_eval, active_eval_before)
    active_eval = candidate_eval if promoted_to_active or active_eval_before is None else active_eval_before
    eligible_for_ui = (
        active_eval["win_rate"] > heuristic_eval["win_rate"]
        or (
            active_eval["win_rate"] == heuristic_eval["win_rate"]
            and active_eval["average_progress"] >= heuristic_eval["average_progress"]
        )
    )
    candidate_metadata = {
        "num_games": num_games,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "teacher_policy": teacher_policy,
        "training_state_source": training_state_source,
        "human_feedback_enabled": include_human_feedback,
        "benchmark_games": benchmark_games,
        "training_cases": len(cases),
        "generated_training_cases": len(generated_cases),
        "human_feedback_cases": len(human_cases),
        "effective_human_training_cases": len(human_cases) * human_repeat_factor,
        "human_repeat_factor": human_repeat_factor,
        "human_feedback_records_seen": human_feedback_summary.records_seen,
        "human_feedback_unique_entries": human_feedback_summary.unique_entries,
        "human_feedback_inferred_stock_cases": human_feedback_summary.inferred_stock_cases,
        "human_feedback_won_cases": human_feedback_summary.won_cases,
        "human_feedback_conceded_cases": human_feedback_summary.conceded_cases,
        "human_feedback_unknown_outcome_cases": human_feedback_summary.unknown_outcome_cases,
        "human_feedback_skipped_normal_conceded_cases": human_feedback_summary.skipped_normal_conceded_cases,
        "teacher_search_depth": DEFAULT_TEACHER_SEARCH_DEPTH,
        "teacher_search_beam_width": DEFAULT_TEACHER_SEARCH_BEAM_WIDTH,
        "teacher_search_rollout_steps": DEFAULT_TEACHER_SEARCH_ROLLOUT_STEPS,
        "teacher_alternate_actions": DEFAULT_TEACHER_ALTERNATE_ACTIONS,
        "teacher_max_states_per_game": DEFAULT_TEACHER_MAX_STATES_PER_GAME,
        "metrics": metrics,
        "heuristic_evaluation": heuristic_eval,
        "search_evaluation": search_eval,
        "candidate_model_evaluation": candidate_eval,
        "active_model_before_evaluation": active_eval_before,
        "active_model_evaluation": active_eval,
        "model_evaluation": active_eval,
        "promoted_to_active": promoted_to_active,
        "active_model_path": str(DEFAULT_MODEL_PATH),
        "eligible_for_ui": eligible_for_ui,
    }
    save_candidate_checkpoint(model, candidate_metadata)
    if promoted_to_active:
        active_metadata = dict(candidate_metadata)
        active_metadata["eligible_for_ui"] = eligible_for_ui
        active_metadata["promoted_to_active"] = True
        active_metadata["previous_active_metadata"] = active_metadata_before if active_metadata_before else None
        promote_candidate_checkpoint(active_metadata)
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "candidate_model_path": str(DEFAULT_CANDIDATE_MODEL_PATH),
        **candidate_metadata,
    }


def improve_policy_from_winning_rollouts(
    num_games: int = 20,
    rollouts_per_game: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.02,
    hidden_dim: int = 256,
    benchmark_games: int = 20,
    temperature: float = 1.0,
    top_k: int = 3,
    epsilon_random: float = 0.08,
    max_moves: int = 180,
    stagnation_limit: int = 30,
    repeat_limit: int = 3,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    start_states = training_start_states(
        num_games=num_games,
        state_source="won_initial_states",
        sessions_dir=sessions_dir,
        progress=progress,
    )
    active_model_before = load_latest_model()
    active_metadata_before = load_model_metadata()
    if active_model_before is not None and active_model_before.hidden_dim == hidden_dim:
        model = clone_policy_model(active_model_before)
        initialization = "active_checkpoint"
    else:
        model = NumpyPolicyNetwork(input_dim=current_input_dim(), hidden_dim=hidden_dim)
        initialization = "fresh"
        if active_model_before is not None and active_model_before.hidden_dim != hidden_dim:
            emit_progress(
                progress,
                f"[self-win] active model hidden_dim={active_model_before.hidden_dim} incompatible with requested hidden_dim={hidden_dim}; starting fresh",
            )

    report_every = max(1, len(start_states) // 5)
    successful_episodes: List[Dict[str, Any]] = []
    rollout_stats: List[Dict[str, Any]] = []
    total_rollouts = 0

    for game_index, start in enumerate(start_states, start=1):
        seed_successes = 0
        seed_best_progress = 0.0
        for attempt in range(rollouts_per_game):
            total_rollouts += 1
            rng = random.Random(1000 * game_index + attempt)
            episode = play_self_win_rollout(
                start.state,
                model=model,
                rng=rng,
                max_moves=max_moves,
                stagnation_limit=stagnation_limit,
                repeat_limit=repeat_limit,
                temperature=temperature,
                top_k=top_k,
                epsilon_random=epsilon_random,
            )
            seed_best_progress = max(seed_best_progress, float(episode["progress_score"]))
            if episode["status"] == STATUS_WON:
                successful_episodes.append(episode)
                seed_successes += 1
        rollout_stats.append(
            {
                "source_id": start.source_id,
                "rollouts": rollouts_per_game,
                "wins": seed_successes,
                "best_progress": seed_best_progress,
            }
        )
        if game_index % report_every == 0 or game_index == len(start_states):
            emit_progress(
                progress,
                f"[self-win] processed {game_index}/{len(start_states)} start states "
                f"rollouts={total_rollouts} wins={len(successful_episodes)}",
            )

    cases = build_cases_from_winning_rollouts(successful_episodes)
    emit_progress(
        progress,
        f"[self-win] collected {len(successful_episodes)} winning rollouts and {len(cases)} unique training cases",
    )

    metrics = None
    if cases:
        emit_progress(progress, f"[self-win] training on {len(cases)} self-win cases")
        metrics = model.train_ranked(cases, epochs=epochs, learning_rate=learning_rate, progress=progress)

    benchmark_seeds = list(range(benchmark_games))
    emit_progress(progress, "[benchmark] evaluating heuristic")
    heuristic_eval = summarize_evaluation(
        evaluate_policy(policy="heuristic", seeds=benchmark_seeds, progress=progress)
    )
    emit_progress(progress, "[benchmark] evaluating search")
    search_eval = summarize_evaluation(
        evaluate_policy(policy="search", seeds=benchmark_seeds, progress=progress)
    )
    emit_progress(progress, "[benchmark] evaluating candidate model")
    candidate_eval = summarize_evaluation(
        evaluate_policy(policy="model", seeds=benchmark_seeds, model=model, progress=progress)
    )
    active_eval_before = None
    if active_model_before is not None:
        emit_progress(progress, "[benchmark] evaluating active model")
        active_eval_before = summarize_evaluation(
            evaluate_policy(policy="model", seeds=benchmark_seeds, model=active_model_before, progress=progress)
        )

    promoted_to_active = bool(cases) and model_is_better(candidate_eval, active_eval_before)
    active_eval = candidate_eval if promoted_to_active or active_eval_before is None else active_eval_before
    eligible_for_ui = (
        active_eval["win_rate"] > heuristic_eval["win_rate"]
        or (
            active_eval["win_rate"] == heuristic_eval["win_rate"]
            and active_eval["average_progress"] >= heuristic_eval["average_progress"]
        )
    )
    candidate_metadata = {
        "training_mode": "self_win_rollouts",
        "num_games": num_games,
        "rollouts_per_game": rollouts_per_game,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "training_state_source": "won_initial_states",
        "human_feedback_enabled": False,
        "benchmark_games": benchmark_games,
        "temperature": temperature,
        "top_k": top_k,
        "epsilon_random": epsilon_random,
        "max_moves": max_moves,
        "stagnation_limit": stagnation_limit,
        "repeat_limit": repeat_limit,
        "model_initialization": initialization,
        "winning_rollouts": len(successful_episodes),
        "rollout_success_rate": (len(successful_episodes) / total_rollouts) if total_rollouts else 0.0,
        "winning_start_states": sum(1 for stat in rollout_stats if stat["wins"] > 0),
        "training_cases": len(cases),
        "generated_training_cases": len(cases),
        "human_feedback_cases": 0,
        "effective_human_training_cases": 0,
        "human_repeat_factor": 1,
        "human_feedback_records_seen": 0,
        "human_feedback_unique_entries": 0,
        "human_feedback_inferred_stock_cases": 0,
        "human_feedback_won_cases": 0,
        "human_feedback_conceded_cases": 0,
        "human_feedback_unknown_outcome_cases": 0,
        "human_feedback_skipped_normal_conceded_cases": 0,
        "metrics": metrics,
        "rollout_stats": rollout_stats,
        "heuristic_evaluation": heuristic_eval,
        "search_evaluation": search_eval,
        "candidate_model_evaluation": candidate_eval,
        "active_model_before_evaluation": active_eval_before,
        "active_model_evaluation": active_eval,
        "model_evaluation": active_eval,
        "promoted_to_active": promoted_to_active,
        "active_model_path": str(DEFAULT_MODEL_PATH),
        "eligible_for_ui": eligible_for_ui,
    }
    save_candidate_checkpoint(model, candidate_metadata)
    if promoted_to_active:
        active_metadata = dict(candidate_metadata)
        active_metadata["eligible_for_ui"] = eligible_for_ui
        active_metadata["promoted_to_active"] = True
        active_metadata["previous_active_metadata"] = active_metadata_before if active_metadata_before else None
        promote_candidate_checkpoint(active_metadata)
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "candidate_model_path": str(DEFAULT_CANDIDATE_MODEL_PATH),
        **candidate_metadata,
    }


def improve_policy_from_beam_search_wins(
    num_games: int = 19,
    epochs: int = 10,
    learning_rate: float = 0.02,
    hidden_dim: int = 256,
    benchmark_games: int = 20,
    beam_width: int = 48,
    branching_factor: int = 6,
    max_nodes_per_game: int = 4000,
    max_moves: int = 180,
    max_wins_per_game: int = 1,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    start_states = training_start_states(
        num_games=num_games,
        state_source="won_initial_states",
        sessions_dir=sessions_dir,
        progress=progress,
    )
    active_model_before = load_latest_model()
    active_metadata_before = load_model_metadata()
    if active_model_before is not None and active_model_before.hidden_dim == hidden_dim:
        model = clone_policy_model(active_model_before)
        initialization = "active_checkpoint"
    else:
        model = NumpyPolicyNetwork(input_dim=current_input_dim(), hidden_dim=hidden_dim)
        initialization = "fresh"
        if active_model_before is not None and active_model_before.hidden_dim != hidden_dim:
            emit_progress(
                progress,
                f"[beam-win] active model hidden_dim={active_model_before.hidden_dim} incompatible with requested hidden_dim={hidden_dim}; starting fresh",
            )

    report_every = max(1, len(start_states) // 5)
    successful_episodes: List[Dict[str, Any]] = []
    search_stats: List[Dict[str, Any]] = []
    total_expanded_nodes = 0

    for game_index, start in enumerate(start_states, start=1):
        result = search_winning_episodes_from_state(
            start.state,
            model=model,
            beam_width=beam_width,
            branching_factor=branching_factor,
            max_nodes=max_nodes_per_game,
            max_moves=max_moves,
            max_wins=max_wins_per_game,
        )
        successful_episodes.extend(result["winning_episodes"])
        total_expanded_nodes += int(result["expanded_nodes"])
        search_stats.append(
            {
                "source_id": start.source_id,
                "wins": len(result["winning_episodes"]),
                "expanded_nodes": result["expanded_nodes"],
                "best_progress": result["best_progress"],
            }
        )
        if game_index % report_every == 0 or game_index == len(start_states):
            emit_progress(
                progress,
                f"[beam-win] processed {game_index}/{len(start_states)} start states "
                f"expanded_nodes={total_expanded_nodes} wins={len(successful_episodes)}",
            )

    cases = build_cases_from_winning_rollouts(successful_episodes)
    emit_progress(
        progress,
        f"[beam-win] collected {len(successful_episodes)} winning episodes and {len(cases)} unique training cases",
    )

    metrics = None
    if cases:
        emit_progress(progress, f"[beam-win] training on {len(cases)} beam-win cases")
        metrics = model.train_ranked(cases, epochs=epochs, learning_rate=learning_rate, progress=progress)

    benchmark_seeds = list(range(benchmark_games))
    emit_progress(progress, "[benchmark] evaluating heuristic")
    heuristic_eval = summarize_evaluation(
        evaluate_policy(policy="heuristic", seeds=benchmark_seeds, progress=progress)
    )
    emit_progress(progress, "[benchmark] evaluating search")
    search_eval = summarize_evaluation(
        evaluate_policy(policy="search", seeds=benchmark_seeds, progress=progress)
    )
    emit_progress(progress, "[benchmark] evaluating candidate model")
    candidate_eval = summarize_evaluation(
        evaluate_policy(policy="model", seeds=benchmark_seeds, model=model, progress=progress)
    )
    active_eval_before = None
    if active_model_before is not None:
        emit_progress(progress, "[benchmark] evaluating active model")
        active_eval_before = summarize_evaluation(
            evaluate_policy(policy="model", seeds=benchmark_seeds, model=active_model_before, progress=progress)
        )

    promoted_to_active = bool(cases) and model_is_better(candidate_eval, active_eval_before)
    active_eval = candidate_eval if promoted_to_active or active_eval_before is None else active_eval_before
    eligible_for_ui = (
        active_eval["win_rate"] > heuristic_eval["win_rate"]
        or (
            active_eval["win_rate"] == heuristic_eval["win_rate"]
            and active_eval["average_progress"] >= heuristic_eval["average_progress"]
        )
    )
    candidate_metadata = {
        "training_mode": "beam_search_wins",
        "num_games": num_games,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "training_state_source": "won_initial_states",
        "human_feedback_enabled": False,
        "benchmark_games": benchmark_games,
        "beam_width": beam_width,
        "branching_factor": branching_factor,
        "max_nodes_per_game": max_nodes_per_game,
        "max_moves": max_moves,
        "max_wins_per_game": max_wins_per_game,
        "model_initialization": initialization,
        "winning_episodes": len(successful_episodes),
        "winning_start_states": sum(1 for stat in search_stats if stat["wins"] > 0),
        "training_cases": len(cases),
        "generated_training_cases": len(cases),
        "human_feedback_cases": 0,
        "effective_human_training_cases": 0,
        "human_repeat_factor": 1,
        "human_feedback_records_seen": 0,
        "human_feedback_unique_entries": 0,
        "human_feedback_inferred_stock_cases": 0,
        "human_feedback_won_cases": 0,
        "human_feedback_conceded_cases": 0,
        "human_feedback_unknown_outcome_cases": 0,
        "human_feedback_skipped_normal_conceded_cases": 0,
        "metrics": metrics,
        "search_stats": search_stats,
        "heuristic_evaluation": heuristic_eval,
        "search_evaluation": search_eval,
        "candidate_model_evaluation": candidate_eval,
        "active_model_before_evaluation": active_eval_before,
        "active_model_evaluation": active_eval,
        "model_evaluation": active_eval,
        "promoted_to_active": promoted_to_active,
        "active_model_path": str(DEFAULT_MODEL_PATH),
        "eligible_for_ui": eligible_for_ui,
    }
    save_candidate_checkpoint(model, candidate_metadata)
    if promoted_to_active:
        active_metadata = dict(candidate_metadata)
        active_metadata["eligible_for_ui"] = eligible_for_ui
        active_metadata["promoted_to_active"] = True
        active_metadata["previous_active_metadata"] = active_metadata_before if active_metadata_before else None
        promote_candidate_checkpoint(active_metadata)
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "candidate_model_path": str(DEFAULT_CANDIDATE_MODEL_PATH),
        **candidate_metadata,
    }


def _hole_model_is_better(candidate_eval: Dict[str, Any], reference_eval: Optional[Dict[str, Any]]) -> bool:
    if not reference_eval:
        return True
    candidate_key = (
        float(candidate_eval.get("success_rate", 0.0)),
        float(candidate_eval.get("average_best_empty_columns", 0.0)),
        -float(candidate_eval.get("average_first_hole_depth", 999.0)),
        float(candidate_eval.get("average_best_score", 0.0)),
    )
    reference_key = (
        float(reference_eval.get("success_rate", 0.0)),
        float(reference_eval.get("average_best_empty_columns", 0.0)),
        -float(reference_eval.get("average_first_hole_depth", 999.0)),
        float(reference_eval.get("average_best_score", 0.0)),
    )
    return candidate_key > reference_key


def save_hole_candidate_checkpoint(model: "NumpyPolicyNetwork", metadata: Dict[str, Any]) -> None:
    model.save(DEFAULT_HOLE_CANDIDATE_MODEL_PATH)
    DEFAULT_HOLE_CANDIDATE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def promote_hole_candidate_checkpoint(candidate_metadata: Dict[str, Any]) -> None:
    shutil.copyfile(DEFAULT_HOLE_CANDIDATE_MODEL_PATH, DEFAULT_HOLE_MODEL_PATH)
    DEFAULT_HOLE_METADATA_PATH.write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")


def save_compact_candidate_checkpoint(model: "NumpyPolicyNetwork", metadata: Dict[str, Any]) -> None:
    model.save(DEFAULT_COMPACT_CANDIDATE_MODEL_PATH)
    DEFAULT_COMPACT_CANDIDATE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def promote_compact_candidate_checkpoint(candidate_metadata: Dict[str, Any]) -> None:
    shutil.copyfile(DEFAULT_COMPACT_CANDIDATE_MODEL_PATH, DEFAULT_COMPACT_MODEL_PATH)
    DEFAULT_COMPACT_METADATA_PATH.write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")


def train_hole_opening_model(
    num_games: int = 120,
    epochs: int = 30,
    learning_rate: float = 0.02,
    hidden_dim: int = 64,
    benchmark_games: int = 50,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    cases = build_hole_opening_cases(
        num_games=num_games,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )
    benchmark_seeds = list(range(benchmark_games))
    active_model_before = load_hole_model()
    model = NumpyPolicyNetwork(input_dim=cases[0].features.shape[1], hidden_dim=hidden_dim)
    emit_progress(progress, f"[hole-train] training on {len(cases)} opening states")
    metrics = model.train_ranked(cases, epochs=epochs, learning_rate=learning_rate, progress=progress)

    heuristic_eval = benchmark_hole_goal(
        policies=("heuristic",),
        games=benchmark_games,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )["heuristic"]
    search_eval = benchmark_hole_goal(
        policies=("search", "hole_search"),
        games=benchmark_games,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )
    candidate_eval = evaluate_hole_goal(
        policy="model",
        seeds=benchmark_seeds,
        model=model,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )
    active_eval_before = None
    if active_model_before is not None:
        active_eval_before = evaluate_hole_goal(
            policy="model",
            seeds=benchmark_seeds,
            model=active_model_before,
            horizon=horizon,
            beam_width=beam_width,
            progress=progress,
        )

    promoted_to_active = _hole_model_is_better(candidate_eval, active_eval_before)
    active_eval = candidate_eval if promoted_to_active or active_eval_before is None else active_eval_before
    metadata = {
        "num_games": num_games,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "benchmark_games": benchmark_games,
        "horizon": horizon,
        "beam_width": beam_width,
        "training_cases": len(cases),
        "metrics": metrics,
        "heuristic_hole_evaluation": heuristic_eval,
        "search_hole_evaluation": search_eval["search"],
        "hole_search_evaluation": search_eval["hole_search"],
        "candidate_model_evaluation": candidate_eval,
        "active_model_before_evaluation": active_eval_before,
        "active_model_evaluation": active_eval,
        "model_evaluation": active_eval,
        "promoted_to_active": promoted_to_active,
    }
    save_hole_candidate_checkpoint(model, metadata)
    if promoted_to_active:
        promote_hole_candidate_checkpoint(metadata)
    return {
        "model_path": str(DEFAULT_HOLE_MODEL_PATH),
        "candidate_model_path": str(DEFAULT_HOLE_CANDIDATE_MODEL_PATH),
        **metadata,
    }


def train_compact_opening_model(
    num_games: int = 120,
    epochs: int = 30,
    learning_rate: float = 0.02,
    hidden_dim: int = 64,
    benchmark_games: int = 50,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
    midgame_share: float = DEFAULT_COMPACT_MIDGAME_SHARE,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    midgame_games = max(0, min(num_games, int(round(num_games * midgame_share))))
    opening_games = max(1, num_games - midgame_games)
    opening_cases = build_compact_opening_cases(
        num_games=opening_games,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )
    midgame_cases: List[TrainingCase] = []
    if midgame_games > 0:
        midgame_cases = build_compact_midgame_cases(
            num_games=midgame_games,
            seed_offset=opening_games,
            horizon=horizon,
            beam_width=beam_width,
            progress=progress,
        )
    cases = opening_cases + midgame_cases
    benchmark_seeds = list(range(benchmark_games))
    active_model_before = load_compact_model()
    model = NumpyPolicyNetwork(input_dim=cases[0].features.shape[1], hidden_dim=hidden_dim)
    emit_progress(
        progress,
        f"[compact-train] training on {len(cases)} states "
        f"({len(opening_cases)} opening, {len(midgame_cases)} midgame)",
    )
    metrics = model.train_ranked(cases, epochs=epochs, learning_rate=learning_rate, progress=progress)

    heuristic_eval = benchmark_compact_goal(
        policies=("heuristic",),
        games=benchmark_games,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )["heuristic"]
    search_eval = benchmark_compact_goal(
        policies=("search", "hole_search", "compact_search"),
        games=benchmark_games,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )
    candidate_eval = evaluate_compact_goal(
        policy="model",
        seeds=benchmark_seeds,
        model=model,
        horizon=horizon,
        beam_width=beam_width,
        progress=progress,
    )
    active_eval_before = None
    if active_model_before is not None:
        active_eval_before = evaluate_compact_goal(
            policy="model",
            seeds=benchmark_seeds,
            model=active_model_before,
            horizon=horizon,
            beam_width=beam_width,
            progress=progress,
        )

    promoted_to_active = _hole_model_is_better(candidate_eval, active_eval_before)
    active_eval = candidate_eval if promoted_to_active or active_eval_before is None else active_eval_before
    metadata = {
        "num_games": num_games,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "benchmark_games": benchmark_games,
        "horizon": horizon,
        "beam_width": beam_width,
        "training_cases": len(cases),
        "opening_training_cases": len(opening_cases),
        "midgame_training_cases": len(midgame_cases),
        "midgame_share": midgame_share,
        "metrics": metrics,
        "heuristic_compact_evaluation": heuristic_eval,
        "search_compact_evaluation": search_eval["search"],
        "hole_search_compact_evaluation": search_eval["hole_search"],
        "compact_search_evaluation": search_eval["compact_search"],
        "candidate_model_evaluation": candidate_eval,
        "active_model_before_evaluation": active_eval_before,
        "active_model_evaluation": active_eval,
        "model_evaluation": active_eval,
        "promoted_to_active": promoted_to_active,
    }
    save_compact_candidate_checkpoint(model, metadata)
    if promoted_to_active:
        promote_compact_candidate_checkpoint(metadata)
    return {
        "model_path": str(DEFAULT_COMPACT_MODEL_PATH),
        "candidate_model_path": str(DEFAULT_COMPACT_CANDIDATE_MODEL_PATH),
        **metadata,
    }


def load_latest_model() -> Optional[NumpyPolicyNetwork]:
    if not DEFAULT_MODEL_PATH.exists():
        return None
    model = NumpyPolicyNetwork.load(DEFAULT_MODEL_PATH)
    if model.input_dim != current_input_dim():
        return None
    return model


def load_model_metadata() -> Dict[str, Any]:
    if not DEFAULT_METADATA_PATH.exists():
        return {}
    return json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))


def load_hole_model() -> Optional[NumpyPolicyNetwork]:
    if not DEFAULT_HOLE_MODEL_PATH.exists():
        return None
    model = NumpyPolicyNetwork.load(DEFAULT_HOLE_MODEL_PATH)
    if model.input_dim != current_input_dim():
        return None
    return model


def load_hole_model_metadata() -> Dict[str, Any]:
    if not DEFAULT_HOLE_METADATA_PATH.exists():
        return {}
    return json.loads(DEFAULT_HOLE_METADATA_PATH.read_text(encoding="utf-8"))


def load_compact_model() -> Optional[NumpyPolicyNetwork]:
    if not DEFAULT_COMPACT_MODEL_PATH.exists():
        return None
    model = NumpyPolicyNetwork.load(DEFAULT_COMPACT_MODEL_PATH)
    if model.input_dim != current_input_dim():
        return None
    return model


def load_compact_model_metadata() -> Dict[str, Any]:
    if not DEFAULT_COMPACT_METADATA_PATH.exists():
        return {}
    return json.loads(DEFAULT_COMPACT_METADATA_PATH.read_text(encoding="utf-8"))


def choose_model_action(state: GameState, model: NumpyPolicyNetwork) -> Optional[Action]:
    actions = available_actions(state)
    if not actions:
        return None
    features = np.vstack([encode_action(state, action) for action in actions])
    scores = model.score(features)
    best_index = int(np.argmax(scores))
    return actions[best_index]


def sample_model_action(
    state: GameState,
    model: NumpyPolicyNetwork,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int = 0,
    epsilon_random: float = 0.0,
) -> Optional[Action]:
    actions = available_actions(state)
    if not actions:
        return None
    if epsilon_random > 0.0 and rng.random() < epsilon_random:
        return rng.choice(actions)

    features = np.vstack([encode_action(state, action) for action in actions])
    scores = model.score(features)
    candidate_indices = np.arange(len(actions))
    if 0 < top_k < len(actions):
        candidate_indices = np.argsort(scores)[-top_k:]
    candidate_scores = scores[candidate_indices]
    probs = _softmax_probabilities(candidate_scores, temperature=temperature)
    sampled_offset = rng.choices(range(len(candidate_indices)), weights=probs.tolist(), k=1)[0]
    return actions[int(candidate_indices[sampled_offset])]


def play_self_win_rollout(
    initial_state: GameState,
    model: NumpyPolicyNetwork,
    rng: random.Random,
    max_moves: int = 180,
    stagnation_limit: int = 30,
    repeat_limit: int = 3,
    temperature: float = 1.0,
    top_k: int = 0,
    epsilon_random: float = 0.0,
) -> Dict[str, Any]:
    state = initial_state.clone()
    repeated_states: Dict[str, int] = {}
    best_progress = progress_score(state)
    moves_since_progress = 0
    trace: List[RolloutTraceStep] = []

    for _ in range(max_moves):
        if state.status != STATUS_IN_PROGRESS:
            break
        action = sample_model_action(
            state,
            model=model,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            epsilon_random=epsilon_random,
        )
        if action is None:
            state = terminate_for_training(state, "no_actions")
            break
        trace.append(RolloutTraceStep(state=state.clone(), action=action))
        state = apply_action(state, action)
        if state.status != STATUS_IN_PROGRESS:
            break

        current_progress = progress_score(state)
        if current_progress > best_progress:
            best_progress = current_progress
            moves_since_progress = 0
        else:
            moves_since_progress += 1

        repeated_states[state.state_hash] = repeated_states.get(state.state_hash, 0) + 1
        if repeated_states[state.state_hash] >= repeat_limit:
            state = terminate_for_training(state, "loop")
            break
        if moves_since_progress >= stagnation_limit:
            state = terminate_for_training(state, "stagnation")
            break

    return {
        "status": state.status,
        "terminal_reason": state.terminal_reason,
        "moves_played": state.moves_played,
        "progress_score": progress_score(state),
        "completed_sequences": state.completed_sequences,
        "trace": trace,
        "final_state": state,
    }


def build_cases_from_winning_rollouts(episodes: Sequence[Dict[str, Any]]) -> List[TrainingCase]:
    deduped: Dict[Tuple[str, str], TrainingCase] = {}
    for episode in episodes:
        for step in episode.get("trace", []):
            actions = available_actions(step.state)
            if not actions:
                continue
            signatures = [action_signature(step.state, action) for action in actions]
            chosen_signature = action_signature(step.state, step.action)
            if chosen_signature not in signatures:
                continue
            target_index = signatures.index(chosen_signature)
            key = (step.state.state_hash, chosen_signature)
            if key in deduped:
                deduped[key].weight += 1.0
                continue
            soft_targets = np.zeros(len(actions), dtype=np.float32)
            soft_targets[target_index] = 1.0
            deduped[key] = TrainingCase(
                features=np.vstack([encode_action(step.state, action) for action in actions]),
                target_index=target_index,
                weight=1.0,
                chosen_signature=chosen_signature,
                action_signatures=signatures,
                soft_targets=soft_targets,
                target_boost=1.0,
                source="self_win",
            )
    return list(deduped.values())


def clone_policy_model(model: NumpyPolicyNetwork) -> NumpyPolicyNetwork:
    cloned = NumpyPolicyNetwork(input_dim=model.input_dim, hidden_dim=model.hidden_dim)
    cloned.w1 = model.w1.copy()
    cloned.b1 = model.b1.copy()
    cloned.w2 = model.w2.copy()
    cloned.b2 = model.b2.copy()
    return cloned


def rank_actions_for_win_search(
    state: GameState,
    model: NumpyPolicyNetwork,
    model_weight: float = 0.4,
) -> List[Tuple[float, Action]]:
    actions = available_actions(state)
    if not actions:
        return []
    features = np.vstack([encode_action(state, action) for action in actions])
    model_scores = model.score(features)
    model_ranked = sorted(
        ((float(score), action) for score, action in zip(model_scores, actions)),
        key=lambda item: item[0],
        reverse=True,
    )
    heuristic_ranked = sorted(
        ((evaluate_action(state, action), action) for action in actions),
        key=lambda item: item[0],
        reverse=True,
    )
    model_rank_weights = _rank_weight_map(model_ranked, state)
    heuristic_rank_weights = _rank_weight_map(heuristic_ranked, state)
    heuristic_weight = max(0.0, 1.0 - model_weight)
    combined: List[Tuple[float, Action]] = []
    for action in actions:
        signature = action_signature(state, action)
        combined_score = (
            model_weight * model_rank_weights.get(signature, 0.0)
            + heuristic_weight * heuristic_rank_weights.get(signature, 0.0)
        )
        combined.append((combined_score, action))
    combined.sort(key=lambda item: item[0], reverse=True)
    return combined


def search_winning_episodes_from_state(
    initial_state: GameState,
    model: NumpyPolicyNetwork,
    beam_width: int = 48,
    branching_factor: int = 6,
    max_nodes: int = 4000,
    max_moves: int = 180,
    max_wins: int = 1,
) -> Dict[str, Any]:
    frontier: List[Dict[str, Any]] = [{"state": initial_state.clone(), "trace": [], "score": 0.0}]
    winning_episodes: List[Dict[str, Any]] = []
    expanded_nodes = 0
    best_progress = progress_score(initial_state)
    best_seen_score = {initial_state.state_hash: 0.0}

    for _ in range(max_moves):
        if not frontier or expanded_nodes >= max_nodes or len(winning_episodes) >= max_wins:
            break
        expanded: List[Dict[str, Any]] = []
        for node in frontier:
            state = node["state"]
            if state.status == STATUS_WON:
                winning_episodes.append(
                    {
                        "status": state.status,
                        "terminal_reason": state.terminal_reason,
                        "moves_played": state.moves_played,
                        "progress_score": progress_score(state),
                        "completed_sequences": state.completed_sequences,
                        "trace": node["trace"],
                        "final_state": state,
                    }
                )
                if len(winning_episodes) >= max_wins:
                    break
                continue
            if state.status != STATUS_IN_PROGRESS:
                continue
            ranked_actions = rank_actions_for_win_search(state, model=model)
            for combined_score, action in ranked_actions[:branching_factor]:
                if expanded_nodes >= max_nodes:
                    break
                next_state = apply_action(state, action)
                trace = list(node["trace"]) + [RolloutTraceStep(state=state.clone(), action=action)]
                best_progress = max(best_progress, progress_score(next_state))
                node_score = node["score"] + combined_score + 0.015 * evaluate_state_snapshot(next_state)
                expanded_nodes += 1
                if next_state.status == STATUS_WON:
                    winning_episodes.append(
                        {
                            "status": next_state.status,
                            "terminal_reason": next_state.terminal_reason,
                            "moves_played": next_state.moves_played,
                            "progress_score": progress_score(next_state),
                            "completed_sequences": next_state.completed_sequences,
                            "trace": trace,
                            "final_state": next_state,
                        }
                    )
                    if len(winning_episodes) >= max_wins:
                        break
                    continue
                previous_best = best_seen_score.get(next_state.state_hash)
                if previous_best is not None and previous_best >= node_score:
                    continue
                best_seen_score[next_state.state_hash] = node_score
                expanded.append({"state": next_state, "trace": trace, "score": node_score})
            if expanded_nodes >= max_nodes or len(winning_episodes) >= max_wins:
                break

        if len(winning_episodes) >= max_wins or expanded_nodes >= max_nodes:
            break
        if not expanded:
            frontier = []
            break

        next_frontier: List[Dict[str, Any]] = []
        seen_hashes = set()
        for node in sorted(expanded, key=lambda item: item["score"], reverse=True):
            state_hash = node["state"].state_hash
            if state_hash in seen_hashes:
                continue
            next_frontier.append(node)
            seen_hashes.add(state_hash)
            if len(next_frontier) >= beam_width:
                break
        frontier = next_frontier

    return {
        "winning_episodes": winning_episodes,
        "expanded_nodes": expanded_nodes,
        "best_progress": best_progress,
    }


def play_hole_goal_episode(
    policy: str,
    seed: int,
    model: Optional[NumpyPolicyNetwork] = None,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    state = create_game(seed=seed)
    best_empty = empty_column_count(state)
    best_score = hole_goal_state_score(state)
    first_hole_depth = 0 if best_empty > 0 else None
    rng = random.Random(seed + 4000)

    for depth in range(horizon):
        if empty_column_count(state) > 0 and first_hole_depth is None:
            first_hole_depth = depth
        if state.status != STATUS_IN_PROGRESS:
            break
        action = choose_policy_action(
            state,
            policy=policy,
            rng=rng,
            model=model,
        )
        if action is None:
            break
        if policy == "hole_search":
            action = choose_hole_search_action(state, horizon=max(1, horizon - depth), beam_width=beam_width)
            if action is None:
                break
        state = apply_action(state, action)
        best_empty = max(best_empty, empty_column_count(state))
        best_score = max(best_score, hole_goal_state_score(state))
        if empty_column_count(state) > 0 and first_hole_depth is None:
            first_hole_depth = depth + 1

    label = f"hole-game seed={seed}"
    emit_progress(
        progress,
        f"[hole-eval:{policy}] {label} success={first_hole_depth is not None} best_empty={best_empty}",
    )
    return {
        "seed": seed,
        "success": first_hole_depth is not None,
        "first_hole_depth": first_hole_depth,
        "best_empty_columns": best_empty,
        "best_score": best_score,
    }


def evaluate_hole_goal(
    policy: str = "heuristic",
    seeds: Optional[Sequence[int]] = None,
    model: Optional[NumpyPolicyNetwork] = None,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    if policy == "model" and model is None:
        model = load_hole_model()
    if policy == "model" and model is None:
        emit_progress(progress, "[hole-eval:model] no compatible model checkpoint found")
        return {
            "policy": policy,
            "games": 0,
            "success_rate": 0.0,
            "average_best_empty_columns": 0.0,
            "average_first_hole_depth": float(horizon + 1),
            "average_best_score": 0.0,
            "episodes": [],
            "available": False,
        }
    seeds = list(seeds or range(20))
    episodes = []
    report_every = max(1, len(seeds) // 5)
    emit_progress(progress, f"[hole-eval:{policy}] running {len(seeds)} games horizon={horizon}")
    for index, seed in enumerate(seeds, start=1):
        episodes.append(
            play_hole_goal_episode(
                policy=policy,
                seed=seed,
                model=model,
                horizon=horizon,
                beam_width=beam_width,
                progress=progress,
            )
        )
        if index % report_every == 0 or index == len(seeds):
            success_rate = sum(1 for episode in episodes if episode["success"]) / len(episodes)
            emit_progress(progress, f"[hole-eval:{policy}] completed {index}/{len(seeds)} games success_rate={success_rate:.2f}")

    successful_depths = [episode["first_hole_depth"] for episode in episodes if episode["first_hole_depth"] is not None]
    average_depth = float(sum(successful_depths) / len(successful_depths)) if successful_depths else float(horizon + 1)
    return {
        "policy": policy,
        "games": len(episodes),
        "success_rate": sum(1 for episode in episodes if episode["success"]) / len(episodes),
        "average_best_empty_columns": float(sum(episode["best_empty_columns"] for episode in episodes) / len(episodes)),
        "average_first_hole_depth": average_depth,
        "average_best_score": float(sum(episode["best_score"] for episode in episodes) / len(episodes)),
        "episodes": episodes,
    }


def benchmark_hole_goal(
    policies: Sequence[str] = ("heuristic", "search", "hole_search", "model"),
    games: int = 20,
    horizon: int = DEFAULT_HOLE_GOAL_HORIZON,
    beam_width: int = DEFAULT_HOLE_GOAL_BEAM_WIDTH,
    model: Optional[NumpyPolicyNetwork] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Dict[str, Any]]:
    benchmark: Dict[str, Dict[str, Any]] = {}
    seeds = list(range(games))
    for policy in policies:
        loaded_model = model
        if policy == "model" and loaded_model is None:
            loaded_model = load_hole_model()
        benchmark[policy] = {
            key: value
            for key, value in evaluate_hole_goal(
                policy=policy,
                seeds=seeds,
                model=loaded_model,
                horizon=horizon,
                beam_width=beam_width,
                progress=progress,
            ).items()
            if key != "episodes"
        }
    return benchmark


def play_compact_goal_episode(
    policy: str,
    seed: int,
    model: Optional[NumpyPolicyNetwork] = None,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    state = create_game(seed=seed)
    best_empty = empty_column_count(state)
    min_mass = effective_column_mass(state)
    best_score = compactness_goal_state_score(state)
    first_hole_depth = 0 if best_empty > 0 else None
    rng = random.Random(seed + 5000)

    for depth in range(horizon):
        if empty_column_count(state) > 0 and first_hole_depth is None:
            first_hole_depth = depth
        if state.status != STATUS_IN_PROGRESS:
            break
        action = choose_policy_action(
            state,
            policy=policy,
            rng=rng,
            model=model,
        )
        if action is None:
            break
        if policy == "compact_search":
            action = choose_compact_search_action(state, horizon=max(1, horizon - depth), beam_width=beam_width)
            if action is None:
                break
        state = apply_action(state, action)
        best_empty = max(best_empty, empty_column_count(state))
        min_mass = min(min_mass, effective_column_mass(state))
        best_score = max(best_score, compactness_goal_state_score(state))
        if empty_column_count(state) > 0 and first_hole_depth is None:
            first_hole_depth = depth + 1

    label = f"compact-game seed={seed}"
    emit_progress(
        progress,
        f"[compact-eval:{policy}] {label} success={first_hole_depth is not None} best_empty={best_empty} min_mass={min_mass}",
    )
    return {
        "seed": seed,
        "success": first_hole_depth is not None,
        "first_hole_depth": first_hole_depth,
        "best_empty_columns": best_empty,
        "min_effective_mass": min_mass,
        "best_score": best_score,
    }


def evaluate_compact_goal(
    policy: str = "heuristic",
    seeds: Optional[Sequence[int]] = None,
    model: Optional[NumpyPolicyNetwork] = None,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    if policy == "model" and model is None:
        model = load_compact_model()
    if policy == "model" and model is None:
        emit_progress(progress, "[compact-eval:model] no compatible model checkpoint found")
        return {
            "policy": policy,
            "games": 0,
            "success_rate": 0.0,
            "average_best_empty_columns": 0.0,
            "average_first_hole_depth": float(horizon + 1),
            "average_min_effective_mass": 52.0,
            "average_best_score": 0.0,
            "episodes": [],
            "available": False,
        }
    seeds = list(seeds or range(20))
    episodes = []
    report_every = max(1, len(seeds) // 5)
    emit_progress(progress, f"[compact-eval:{policy}] running {len(seeds)} games horizon={horizon}")
    for index, seed in enumerate(seeds, start=1):
        episodes.append(
            play_compact_goal_episode(
                policy=policy,
                seed=seed,
                model=model,
                horizon=horizon,
                beam_width=beam_width,
                progress=progress,
            )
        )
        if index % report_every == 0 or index == len(seeds):
            success_rate = sum(1 for episode in episodes if episode["success"]) / len(episodes)
            emit_progress(progress, f"[compact-eval:{policy}] completed {index}/{len(seeds)} games success_rate={success_rate:.2f}")

    successful_depths = [episode["first_hole_depth"] for episode in episodes if episode["first_hole_depth"] is not None]
    average_depth = float(sum(successful_depths) / len(successful_depths)) if successful_depths else float(horizon + 1)
    return {
        "policy": policy,
        "games": len(episodes),
        "success_rate": sum(1 for episode in episodes if episode["success"]) / len(episodes),
        "average_best_empty_columns": float(sum(episode["best_empty_columns"] for episode in episodes) / len(episodes)),
        "average_first_hole_depth": average_depth,
        "average_min_effective_mass": float(sum(episode["min_effective_mass"] for episode in episodes) / len(episodes)),
        "average_best_score": float(sum(episode["best_score"] for episode in episodes) / len(episodes)),
        "episodes": episodes,
    }


def benchmark_compact_goal(
    policies: Sequence[str] = ("heuristic", "search", "hole_search", "compact_search", "model"),
    games: int = 20,
    horizon: int = DEFAULT_COMPACT_GOAL_HORIZON,
    beam_width: int = DEFAULT_COMPACT_GOAL_BEAM_WIDTH,
    model: Optional[NumpyPolicyNetwork] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Dict[str, Any]]:
    benchmark: Dict[str, Dict[str, Any]] = {}
    seeds = list(range(games))
    for policy in policies:
        loaded_model = model
        if policy == "model" and loaded_model is None:
            loaded_model = load_compact_model()
        benchmark[policy] = {
            key: value
            for key, value in evaluate_compact_goal(
                policy=policy,
                seeds=seeds,
                model=loaded_model,
                horizon=horizon,
                beam_width=beam_width,
                progress=progress,
            ).items()
            if key != "episodes"
        }
    return benchmark


def _rank_weight_map(
    ranked_pairs: Sequence[Tuple[float, Action]],
    state: GameState,
) -> Dict[str, float]:
    total = len(ranked_pairs)
    if total <= 1:
        return {action_signature(state, action): 1.0 for _, action in ranked_pairs}
    return {
        action_signature(state, action): (total - index - 1) / (total - 1)
        for index, (_, action) in enumerate(ranked_pairs)
    }


def should_use_hole_opening_signal(state: GameState) -> bool:
    return (
        state.status == STATUS_IN_PROGRESS
        and state.moves_played <= 2
        and state.completed_sequences == 0
        and empty_column_count(state) == 0
        and len(state.stock) >= 17
    )


def should_use_compact_opening_signal(state: GameState) -> bool:
    return (
        state.status == STATUS_IN_PROGRESS
        and state.moves_played <= 4
        and state.completed_sequences == 0
        and empty_column_count(state) <= 1
        and len(state.stock) >= 14
    )


def choose_policy_action(
    state: GameState,
    policy: str,
    rng: Optional[random.Random] = None,
    model: Optional[NumpyPolicyNetwork] = None,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    search_beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    search_discount: float = DEFAULT_SEARCH_DISCOUNT,
    search_rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> Optional[Action]:
    if policy == "heuristic":
        return choose_heuristic_action(state, rng=rng)
    if policy == "random":
        return choose_random_action(state, rng=rng)
    if policy == "search":
        return choose_search_action(
            state,
            depth=search_depth,
            beam_width=search_beam_width,
            discount=search_discount,
            rollout_steps=search_rollout_steps,
        )
    if policy == "hole_search":
        return choose_hole_search_action(state)
    if policy == "compact_search":
        return choose_compact_search_action(state)
    if policy == "model" and model is not None:
        return choose_model_action(state, model)
    return choose_heuristic_action(state, rng=rng)


def play_episode(
    policy: str,
    seed: int,
    model: Optional[NumpyPolicyNetwork] = None,
    max_moves: int = 180,
    stagnation_limit: int = 30,
    repeat_limit: int = 3,
    progress: ProgressCallback = None,
    progress_label: Optional[str] = None,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    search_beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    search_discount: float = DEFAULT_SEARCH_DISCOUNT,
    search_rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> Dict[str, Any]:
    state = create_game(seed=seed)
    repeated_states: Dict[str, int] = {}
    best_progress = progress_score(state)
    moves_since_progress = 0
    rng = random.Random(seed + 1000)

    for _ in range(max_moves):
        if state.status != STATUS_IN_PROGRESS:
            break
        action = choose_policy_action(
            state,
            policy=policy,
            rng=rng,
            model=model,
            search_depth=search_depth,
            search_beam_width=search_beam_width,
            search_discount=search_discount,
            search_rollout_steps=search_rollout_steps,
        )

        if action is None:
            state = terminate_for_training(state, "no_actions")
            break
        state = apply_action(state, action)
        if state.status != STATUS_IN_PROGRESS:
            break

        current_progress = progress_score(state)
        if current_progress > best_progress:
            best_progress = current_progress
            moves_since_progress = 0
        else:
            moves_since_progress += 1

        repeated_states[state.state_hash] = repeated_states.get(state.state_hash, 0) + 1
        if repeated_states[state.state_hash] >= repeat_limit:
            state = terminate_for_training(state, "loop")
            break
        if moves_since_progress >= stagnation_limit:
            state = terminate_for_training(state, "stagnation")
            break

    label = f"{progress_label} " if progress_label else ""
    emit_progress(progress, f"[eval:{policy}] {label}seed={seed} progress={progress_score(state)} status={state.status}")

    return {
        "seed": seed,
        "status": state.status,
        "terminal_reason": state.terminal_reason,
        "completed_sequences": state.completed_sequences,
        "progress_score": progress_score(state),
        "moves_played": state.moves_played,
        "state": serialize_state(state),
    }


def evaluate_policy(
    policy: str = "heuristic",
    seeds: Optional[Sequence[int]] = None,
    model: Optional[NumpyPolicyNetwork] = None,
    progress: ProgressCallback = None,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    search_beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    search_discount: float = DEFAULT_SEARCH_DISCOUNT,
    search_rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> Dict[str, Any]:
    if policy == "model" and model is None:
        model = load_latest_model()
    if policy == "model" and model is None:
        emit_progress(progress, "[eval:model] no compatible model checkpoint found")
        return {
            "policy": policy,
            "games": 0,
            "win_rate": 0.0,
            "average_progress": 0.0,
            "average_completed_sequences": 0.0,
            "episodes": [],
            "available": False,
        }
    seeds = list(seeds or range(20))
    emit_progress(progress, f"[eval:{policy}] running {len(seeds)} games")
    episodes = []
    report_every = max(1, len(seeds) // 5)
    for index, seed in enumerate(seeds, start=1):
        episodes.append(
            play_episode(
                policy=policy,
                seed=seed,
                model=model,
                progress=progress,
                progress_label=f"game={index}/{len(seeds)}",
                search_depth=search_depth,
                search_beam_width=search_beam_width,
                search_discount=search_discount,
                search_rollout_steps=search_rollout_steps,
            )
        )
        if index % report_every == 0 or index == len(seeds):
            avg_progress = sum(episode["progress_score"] for episode in episodes) / len(episodes)
            emit_progress(progress, f"[eval:{policy}] completed {index}/{len(seeds)} games avg_progress={avg_progress:.2f}")
    wins = sum(1 for episode in episodes if episode["status"] == "won")
    return {
        "policy": policy,
        "games": len(episodes),
        "win_rate": wins / len(episodes),
        "average_progress": float(sum(episode["progress_score"] for episode in episodes) / len(episodes)),
        "average_completed_sequences": float(sum(episode["completed_sequences"] for episode in episodes) / len(episodes)),
        "episodes": episodes,
    }


def summarize_evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "policy": result["policy"],
        "games": result["games"],
        "win_rate": result["win_rate"],
        "average_progress": result["average_progress"],
        "average_completed_sequences": result["average_completed_sequences"],
    }
    if "available" in result:
        summary["available"] = result["available"]
    return summary


def benchmark_policies(
    policies: Sequence[str] = ("heuristic", "search", "model"),
    games: int = 20,
    model: Optional[NumpyPolicyNetwork] = None,
    progress: ProgressCallback = None,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    search_beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH,
    search_discount: float = DEFAULT_SEARCH_DISCOUNT,
    search_rollout_steps: int = DEFAULT_SEARCH_ROLLOUT_STEPS,
) -> Dict[str, Dict[str, Any]]:
    benchmark: Dict[str, Dict[str, Any]] = {}
    for policy in policies:
        loaded_model = model
        if policy == "model" and loaded_model is None:
            loaded_model = load_latest_model()
        emit_progress(progress, f"[benchmark] starting {policy} over {games} games")
        benchmark[policy] = summarize_evaluation(
            evaluate_policy(
                policy=policy,
                seeds=range(games),
                model=loaded_model,
                progress=progress,
                search_depth=search_depth,
                search_beam_width=search_beam_width,
                search_discount=search_discount,
                search_rollout_steps=search_rollout_steps,
            )
        )
    return benchmark


def _ui_candidate_actions(
    state: GameState,
    actions: Sequence[Action],
    heuristic_scores: Dict[str, float],
    candidate_limit: int,
) -> List[Action]:
    if len(actions) <= candidate_limit:
        return list(actions)

    ranked_actions = sorted(
        actions,
        key=lambda action: heuristic_scores[action_signature(state, action)],
        reverse=True,
    )
    selected: List[Action] = []
    seen = set()

    def add_action(action: Action) -> None:
        signature = action_signature(state, action)
        if signature in seen:
            return
        selected.append(action)
        seen.add(signature)

    for action in ranked_actions[:candidate_limit]:
        add_action(action)

    if len(selected) < candidate_limit:
        for action in ranked_actions:
            if move_creates_hole(state, action):
                add_action(action)
            if len(selected) >= candidate_limit:
                break

    if can_deal(state):
        for action in ranked_actions:
            if action.type == "deal":
                add_action(action)
                break

    return selected[:candidate_limit]


def rank_actions_for_state(state: GameState, limit: int = 5) -> Dict[str, Any]:
    actions = available_actions(state)
    heuristic_scores = {action_signature(state, action): evaluate_action(state, action) for action in actions}
    ui_candidates = _ui_candidate_actions(
        state,
        actions,
        heuristic_scores,
        candidate_limit=max(limit * 2, DEFAULT_UI_SEARCH_CANDIDATE_LIMIT),
    )
    search_scores = {
        action_signature(state, action): score
        for score, action in rank_search_actions_subset(
            state,
            ui_candidates,
            depth=DEFAULT_UI_SEARCH_DEPTH,
            beam_width=DEFAULT_UI_SEARCH_BEAM_WIDTH,
            discount=DEFAULT_UI_SEARCH_DISCOUNT,
            rollout_steps=DEFAULT_UI_SEARCH_ROLLOUT_STEPS,
        )
    }
    heuristic_ranked = sorted(
        (
            (search_scores.get(action_signature(state, action), heuristic_scores[action_signature(state, action)]), action)
            for action in ui_candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    model = load_latest_model()
    metadata = load_model_metadata()
    model_eligible = bool(model is not None and metadata.get("eligible_for_ui"))
    model_scores: Dict[str, float] = {}
    hole_model = load_hole_model()
    hole_metadata = load_hole_model_metadata()
    hole_model_eligible = bool(hole_model is not None and hole_metadata)
    hole_model_scores: Dict[str, float] = {}
    compact_model = load_compact_model()
    compact_metadata = load_compact_model_metadata()
    compact_model_eligible = bool(compact_model is not None and compact_metadata)
    compact_model_scores: Dict[str, float] = {}
    if model is not None and actions:
        features = np.vstack([encode_action(state, action) for action in actions])
        predictions = model.score(features)
        for action, score in zip(actions, predictions):
            model_scores[action.to_dict(state)["description"]] = float(score)

    opening_hole_signal_active = bool(hole_model_eligible and should_use_hole_opening_signal(state))
    if opening_hole_signal_active and hole_model is not None and actions:
        features = np.vstack([encode_action(state, action) for action in actions])
        predictions = hole_model.score(features)
        for action, score in zip(actions, predictions):
            hole_model_scores[action.to_dict(state)["description"]] = float(score)

    opening_compact_signal_active = bool(compact_model_eligible and should_use_compact_opening_signal(state))
    if opening_compact_signal_active and compact_model is not None and actions:
        features = np.vstack([encode_action(state, action) for action in actions])
        predictions = compact_model.score(features)
        for action, score in zip(actions, predictions):
            compact_model_scores[action.to_dict(state)["description"]] = float(score)

    base_ranked = list(heuristic_ranked)
    if model_eligible:
        base_ranked.sort(
            key=lambda item: model_scores.get(item[1].to_dict(state)["description"], -1.0),
            reverse=True,
        )

    if opening_hole_signal_active and hole_model_scores:
        base_rank_weights = _rank_weight_map(base_ranked, state)
        hole_ranked = sorted(
            ((hole_model_scores.get(action_signature(state, action), -1.0), action) for _, action in base_ranked),
            key=lambda item: item[0],
            reverse=True,
        )
        hole_rank_weights = _rank_weight_map(hole_ranked, state)
        base_ranked.sort(
            key=lambda item: (
                base_rank_weights.get(action_signature(state, item[1]), 0.0)
                + 0.65 * hole_rank_weights.get(action_signature(state, item[1]), 0.0)
            ),
            reverse=True,
        )

    if opening_compact_signal_active and compact_model_scores:
        base_rank_weights = _rank_weight_map(base_ranked, state)
        compact_ranked = sorted(
            ((compact_model_scores.get(action_signature(state, action), -1.0), action) for _, action in base_ranked),
            key=lambda item: item[0],
            reverse=True,
        )
        compact_rank_weights = _rank_weight_map(compact_ranked, state)
        base_ranked.sort(
            key=lambda item: (
                base_rank_weights.get(action_signature(state, item[1]), 0.0)
                + 0.7 * compact_rank_weights.get(action_signature(state, item[1]), 0.0)
            ),
            reverse=True,
        )

    # The model heads score actions without any notion of a locked position, so
    # they can promote a move that seals the deal. Demote those last of all.
    dead_signatures = {
        action_signature(state, action)
        for _, action in base_ranked
        if is_provably_lost(apply_action(state, action))
    }
    if dead_signatures:
        base_ranked.sort(key=lambda item: action_signature(state, item[1]) in dead_signatures)

    ranked: List[Dict[str, Any]] = []
    for heuristic_score, action in base_ranked[:limit]:
        action_dict = action.to_dict(state)
        ranked.append(
            {
                **action_dict,
                "heuristic_score": heuristic_scores[action_dict["description"]],
                "search_score": search_scores.get(action_dict["description"]),
                "model_score": model_scores.get(action_dict["description"]),
                "hole_model_score": hole_model_scores.get(action_dict["description"]),
                "compact_model_score": compact_model_scores.get(action_dict["description"]),
                "locks_position": action_signature(state, action) in dead_signatures,
            }
        )
    ranking_source = "model" if model_eligible else "search"
    if opening_hole_signal_active and hole_model_scores:
        ranking_source = f"{ranking_source}+hole"
    if opening_compact_signal_active and compact_model_scores:
        ranking_source = f"{ranking_source}+compact"
    return {
        "suggestions": ranked,
        "model_loaded": model is not None,
        "model_eligible": model_eligible,
        "hole_model_loaded": hole_model is not None,
        "hole_model_active": opening_hole_signal_active and bool(hole_model_scores),
        "compact_model_loaded": compact_model is not None,
        "compact_model_active": opening_compact_signal_active and bool(compact_model_scores),
        "ranking_source": ranking_source,
    }


def teacher_soft_targets(
    signatures: Sequence[str],
    ranked_map: Dict[str, float],
    temperature: float = 6.0,
) -> np.ndarray:
    raw_scores = np.asarray([ranked_map.get(signature, min(ranked_map.values()) - 10.0) for signature in signatures], dtype=np.float32)
    shifted = (raw_scores - np.max(raw_scores)) / temperature
    exp_scores = np.exp(np.clip(shifted, -30.0, 30.0))
    probs = exp_scores / np.sum(exp_scores)
    return probs.astype(np.float32)


def _softmax_probabilities(raw_scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted_temperature = max(1e-3, temperature)
    shifted = (raw_scores - np.max(raw_scores)) / adjusted_temperature
    exp_scores = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp_scores / np.sum(exp_scores)


def _make_card(card_data: Dict[str, Any]) -> Card:
    return Card(rank=int(card_data["rank"]), suit=str(card_data["suit"]))


def reconstruct_feedback_state(snapshot: Dict[str, Any]) -> Tuple[GameState, bool]:
    columns = [[_make_card(card) for card in column["cards"]] for column in snapshot["columns"]]
    stock_cards = snapshot.get("stock_cards")
    if stock_cards is not None:
        stock = [_make_card(card) for card in stock_cards]
        exact_stock = True
    else:
        dealt_codes = {card.code for column in columns for card in column}
        remaining_cards = [card for card in create_deck() if card.code not in dealt_codes]
        stock_count = int(snapshot.get("stock_count", len(remaining_cards)))
        stock = remaining_cards[:stock_count]
        exact_stock = False
    state = GameState(
        columns=columns,
        stock=stock,
        status=str(snapshot.get("status", STATUS_IN_PROGRESS)),
        moves_played=int(snapshot.get("moves_played", 0)),
        terminal_reason=snapshot.get("terminal_reason"),
    )
    return state, exact_stock


def _feedback_strength_weight(strength: str) -> float:
    weights = {
        "normal": 3.5,
        "important": 18.0,
        "key_move": 36.0,
    }
    return weights.get(strength, 4.0)


def _feedback_target_boost(strength: str, exact_stock: bool) -> float:
    if strength == "key_move":
        return DEFAULT_KEY_EXACT_HUMAN_TARGET_BOOST if exact_stock else DEFAULT_KEY_HUMAN_TARGET_BOOST
    if strength == "important":
        return DEFAULT_IMPORTANT_EXACT_HUMAN_TARGET_BOOST if exact_stock else DEFAULT_IMPORTANT_HUMAN_TARGET_BOOST
    return DEFAULT_EXACT_HUMAN_TARGET_BOOST if exact_stock else DEFAULT_HUMAN_TARGET_BOOST


def _load_game_outcomes(sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> Dict[str, str]:
    outcomes: Dict[str, str] = {}
    if not sessions_dir.exists():
        return outcomes
    for path in sessions_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = payload.get("history") or []
        if not history:
            continue
        last_state = history[-1]
        status = str(last_state.get("status", STATUS_IN_PROGRESS))
        game_id = str(payload.get("game_id", path.stem))
        outcomes[game_id] = status
    return outcomes


def _outcome_weight_multiplier(outcome: str) -> float:
    if outcome == STATUS_WON:
        return 1.6
    if outcome == STATUS_CONCEDED:
        return 0.82
    return 1.0


def human_case_repeat_factor(
    generated_count: int,
    human_count: int,
    target_share: float = DEFAULT_HUMAN_TARGET_SHARE,
    max_repeat_factor: int = DEFAULT_MAX_HUMAN_REPEAT_FACTOR,
) -> int:
    if human_count <= 0:
        return 1
    if generated_count <= 0:
        return 1
    required = (target_share * generated_count) / max(1e-6, (1.0 - target_share) * human_count)
    return max(1, min(max_repeat_factor, int(np.ceil(required))))


def expand_human_cases_for_training(
    generated_cases: Sequence[TrainingCase],
    human_cases: Sequence[TrainingCase],
    progress: ProgressCallback = None,
) -> Tuple[List[TrainingCase], int]:
    repeat_factor = human_case_repeat_factor(len(generated_cases), len(human_cases))
    expanded_cases = list(generated_cases)
    if human_cases:
        expanded_cases.extend(list(human_cases) * repeat_factor)
    emit_progress(
        progress,
        f"[feedback] effective human repeat factor={repeat_factor} "
        f"(generated={len(generated_cases)}, human={len(human_cases)}, total={len(expanded_cases)})",
    )
    return expanded_cases, repeat_factor


def model_is_better(candidate_eval: Dict[str, Any], reference_eval: Optional[Dict[str, Any]]) -> bool:
    if not reference_eval:
        return True
    candidate_key = (
        float(candidate_eval.get("win_rate", 0.0)),
        float(candidate_eval.get("average_progress", 0.0)),
        float(candidate_eval.get("average_completed_sequences", 0.0)),
    )
    reference_key = (
        float(reference_eval.get("win_rate", 0.0)),
        float(reference_eval.get("average_progress", 0.0)),
        float(reference_eval.get("average_completed_sequences", 0.0)),
    )
    return candidate_key > reference_key


def save_candidate_checkpoint(model: "NumpyPolicyNetwork", metadata: Dict[str, Any]) -> None:
    model.save(DEFAULT_CANDIDATE_MODEL_PATH)
    DEFAULT_CANDIDATE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def promote_candidate_checkpoint(candidate_metadata: Dict[str, Any]) -> None:
    shutil.copyfile(DEFAULT_CANDIDATE_MODEL_PATH, DEFAULT_MODEL_PATH)
    DEFAULT_METADATA_PATH.write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")
    shutil.copyfile(DEFAULT_CANDIDATE_MODEL_PATH, DEFAULT_BEST_MODEL_PATH)
    DEFAULT_BEST_METADATA_PATH.write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")


def _feedback_record_key(record: Dict[str, Any]) -> str:
    existing = record.get("feedback_key")
    if existing:
        return str(existing)
    chosen_action = record.get("chosen_action", {})
    return json.dumps(
        {
            "state_hash": str(record.get("state_hash", "")),
            "chosen_action": {
                "type": chosen_action.get("type"),
                "from_column": chosen_action.get("from_column"),
                "to_column": chosen_action.get("to_column"),
                "run_length": chosen_action.get("run_length"),
                "description": chosen_action.get("description"),
            },
        },
        sort_keys=True,
    )


def _filter_invalidated_feedback_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    invalidated_keys = {
        str(record.get("feedback_key", ""))
        for record in records
        if str(record.get("record_type", "feedback")) == "invalidation"
    }
    filtered = []
    invalidated_count = 0
    for record in records:
        if str(record.get("record_type", "feedback")) != "feedback":
            continue
        if _feedback_record_key(record) in invalidated_keys:
            invalidated_count += 1
            continue
        filtered.append(record)
    return filtered, invalidated_count


def _dedupe_feedback_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    strength_rank = {"normal": 0, "important": 1, "key_move": 2}
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        state_hash = str(record.get("state_hash", ""))
        chosen_action = record.get("chosen_action", {})
        dedupe_key = (
            state_hash,
            json.dumps(
                {
                    "type": chosen_action.get("type"),
                    "from_column": chosen_action.get("from_column"),
                    "to_column": chosen_action.get("to_column"),
                    "run_length": chosen_action.get("run_length"),
                    "description": chosen_action.get("description"),
                },
                sort_keys=True,
            ),
        )
        current = deduped.get(dedupe_key)
        if current is None:
            deduped[dedupe_key] = record
            continue
        current_rank = strength_rank.get(str(current.get("feedback_strength", "normal")), 0)
        next_rank = strength_rank.get(str(record.get("feedback_strength", "normal")), 0)
        current_ts = str(current.get("timestamp", ""))
        next_ts = str(record.get("timestamp", ""))
        if next_rank > current_rank or (next_rank == current_rank and next_ts > current_ts):
            deduped[dedupe_key] = record
    return list(deduped.values())


def load_human_feedback_cases(
    path: Path = DEFAULT_FEEDBACK_PATH,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    progress: ProgressCallback = None,
) -> Tuple[List[TrainingCase], HumanFeedbackSummary]:
    summary = HumanFeedbackSummary()
    if not path.exists():
        emit_progress(progress, f"[feedback] no human feedback file at {path}")
        return [], summary

    game_outcomes = _load_game_outcomes(sessions_dir=sessions_dir)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary.records_seen = len(rows)
    filtered_rows, invalidated_count = _filter_invalidated_feedback_records(rows)
    summary.invalidated_entries = invalidated_count
    deduped_rows = _dedupe_feedback_records(filtered_rows)
    summary.unique_entries = len(deduped_rows)

    cases: List[TrainingCase] = []
    for record in deduped_rows:
        snapshot = record.get("state_snapshot")
        chosen_payload = record.get("chosen_action", {})
        if not snapshot:
            summary.skipped_missing_state += 1
            continue

        state, exact_stock = reconstruct_feedback_state(snapshot)
        actions = available_actions(state)
        if not actions:
            summary.skipped_missing_action += 1
            continue

        signatures = [action_signature(state, action) for action in actions]
        matched_index = next((index for index, action in enumerate(actions) if action_payload_matches(action, chosen_payload)), None)
        if matched_index is None:
            summary.skipped_missing_action += 1
            continue

        heuristic_scores = {signature: evaluate_action(state, action) for signature, action in zip(signatures, actions)}
        chosen_signature = signatures[matched_index]
        best_non_chosen = max((score for signature, score in heuristic_scores.items() if signature != chosen_signature), default=0.0)
        ranked_map = dict(heuristic_scores)
        ranked_map[chosen_signature] = max(ranked_map[chosen_signature], best_non_chosen + 30.0)
        soft_targets = teacher_soft_targets(signatures, ranked_map)
        strength = str(record.get("feedback_strength", "normal"))
        game_outcome = game_outcomes.get(str(record.get("game_id", "")), STATUS_IN_PROGRESS)
        if strength == "normal" and game_outcome == STATUS_CONCEDED:
            summary.skipped_normal_conceded_cases += 1
            continue

        weight = _feedback_strength_weight(strength)
        weight *= _outcome_weight_multiplier(game_outcome)
        if game_outcome == STATUS_WON:
            summary.won_cases += 1
        elif game_outcome == STATUS_CONCEDED:
            summary.conceded_cases += 1
        else:
            summary.unknown_outcome_cases += 1
        if move_creates_hole(state, actions[matched_index]):
            weight += 1.0
        if not exact_stock:
            weight *= 0.8
            summary.inferred_stock_cases += 1

        cases.append(
            TrainingCase(
                features=np.vstack([encode_action(state, action) for action in actions]),
                target_index=matched_index,
                weight=weight,
                chosen_signature=chosen_signature,
                action_signatures=signatures,
                soft_targets=soft_targets,
                target_boost=_feedback_target_boost(strength, exact_stock),
                source="human",
            )
        )

    summary.cases_loaded = len(cases)
    emit_progress(
        progress,
        "[feedback] loaded "
        f"{summary.cases_loaded}/{summary.unique_entries} human cases "
        f"(records={summary.records_seen}, invalidated={summary.invalidated_entries}, "
        f"won={summary.won_cases}, conceded={summary.conceded_cases}, unknown={summary.unknown_outcome_cases}, "
        f"skipped_normal_conceded={summary.skipped_normal_conceded_cases}, "
        f"inferred_stock={summary.inferred_stock_cases}, "
        f"skipped_missing_state={summary.skipped_missing_state}, skipped_missing_action={summary.skipped_missing_action})",
    )
    return cases, summary


def enqueue_alternate_states(
    state: GameState,
    ranked_actions: Sequence[Tuple[float, Action]],
    queue: List[GameState],
    queued_hashes: set,
    max_states_per_game: int,
    alternate_actions: int,
    branch_margin: float,
) -> int:
    if not ranked_actions or alternate_actions <= 0:
        return 0
    added = 0
    best_score = ranked_actions[0][0]
    for score, action in ranked_actions[1 : alternate_actions + 1]:
        if score < best_score - branch_margin:
            continue
        next_state = apply_action(state, action)
        if next_state.status != STATUS_IN_PROGRESS:
            continue
        if next_state.state_hash in queued_hashes:
            continue
        if len(queued_hashes) >= max_states_per_game:
            break
        queue.append(next_state)
        queued_hashes.add(next_state.state_hash)
        added += 1
    return added
