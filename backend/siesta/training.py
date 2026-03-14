from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .game import (
    Card,
    GameState,
    Move,
    STATUS_IN_PROGRESS,
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
DEFAULT_MODEL_PATH = MODELS_DIR / "policy-latest.npz"
DEFAULT_METADATA_PATH = MODELS_DIR / "policy-latest.json"
DEFAULT_CANDIDATE_MODEL_PATH = MODELS_DIR / "policy-candidate.npz"
DEFAULT_CANDIDATE_METADATA_PATH = MODELS_DIR / "policy-candidate.json"
DEFAULT_BEST_MODEL_PATH = MODELS_DIR / "policy-best.npz"
DEFAULT_BEST_METADATA_PATH = MODELS_DIR / "policy-best.json"
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
DEFAULT_HUMAN_TARGET_SHARE = 0.18
DEFAULT_MAX_HUMAN_REPEAT_FACTOR = 32
ProgressCallback = Optional[Callable[[str], None]]


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


def progress_score(state: GameState) -> int:
    return state.completed_sequences * 13 + run_depth(state)


def total_run_depth(state: GameState) -> int:
    return sum(top_run_length(column) for column in state.columns if column)


def empty_column_count(state: GameState) -> int:
    return sum(1 for column in state.columns if not column)


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


def mobility_score(state: GameState) -> float:
    actions = legal_moves(state)
    if not actions:
        return 0.0
    run_total = sum(action.run_length for action in actions)
    distinct_targets = len({action.to_column for action in actions})
    return float(run_total + distinct_targets)


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


def evaluate_action(state: GameState, action: Action) -> float:
    next_state = apply_action(state, action)
    score = 0.0
    score += 250.0 if next_state.status == "won" else 0.0
    score += 80.0 * (next_state.completed_sequences - state.completed_sequences)
    score += 7.0 * run_depth(next_state)
    before_empty = empty_column_count(state)
    after_empty = empty_column_count(next_state)
    score += hole_bonus(after_empty)
    score -= hole_bonus(before_empty) * 0.4
    score += 5.0 * hole_setup_score(next_state)
    score -= 2.0 * len(next_state.stock)
    if action.type == "move" and action.move:
        moved = state.columns[action.move.from_column][-action.move.run_length :]
        score += 4.0 * action.move.run_length
        destination = state.columns[action.move.to_column][-1] if state.columns[action.move.to_column] else None
        if destination and destination.suit == moved[0].suit:
            score += 10.0
        if move_creates_hole(state, action):
            score += 25.0
            if after_empty >= 2:
                score += 18.0
    if action.type == "deal":
        score += 6.0 if not legal_moves(state) else -5.0
    return score


def evaluate_state_snapshot(state: GameState) -> float:
    score = 0.0
    score += 400.0 if state.status == "won" else 0.0
    score += 90.0 * state.completed_sequences
    score += 9.0 * run_depth(state)
    score += 3.5 * total_run_depth(state)
    score += hole_bonus(empty_column_count(state))
    score += 4.0 * hole_setup_score(state)
    score += 1.2 * mobility_score(state)
    score -= 1.5 * len(state.stock)
    return score


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
        ]
    )
    return np.asarray(features, dtype=np.float32)


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
    return np.concatenate([global_state_features(state), action_features(state, action), action_outcome_features(state, action)])


def action_outcome_features(state: GameState, action: Action) -> np.ndarray:
    next_state = apply_action(state, action)
    before_empty = empty_column_count(state)
    after_empty = empty_column_count(next_state)
    before_mobility = mobility_score(state)
    after_mobility = mobility_score(next_state)
    features = [
        next_state.completed_sequences / 4.0,
        run_depth(next_state) / 13.0,
        total_run_depth(next_state) / 52.0,
        after_empty / 7.0,
        hole_setup_score(next_state) / 20.0,
        after_mobility / 30.0,
        (after_empty - before_empty) / 3.0,
        (after_mobility - before_mobility) / 30.0,
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
    progress: ProgressCallback = None,
) -> List[TrainingCase]:
    cases: List[TrainingCase] = []
    rng = random.Random(17 + seed_offset)
    progress_every = max(1, num_games // 10)
    emit_progress(progress, f"[data] generating training cases from {num_games} games using {teacher_policy}")

    for game_seed in range(seed_offset, seed_offset + num_games):
        state_queue: List[GameState] = [create_game(seed=game_seed)]
        queued_hashes = {state_queue[0].state_hash}
        explored_states = 0

        while state_queue and explored_states < max_states_per_game:
            state = state_queue.pop(0)
            explored_states += 1
            emit_progress(
                progress,
                f"[data] seed={game_seed} exploring branch {explored_states}/{max_states_per_game} queue={len(state_queue)}",
            )
            repeated_states: Dict[str, int] = {}
            best_progress = progress_score(state)
            moves_since_progress = 0

            for _ in range(max_moves):
                actions = available_actions(state)
                if not actions:
                    break
                if teacher_policy == "search":
                    ranked_actions = rank_search_actions(
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
        completed = game_seed - seed_offset + 1
        if completed % progress_every == 0 or completed == num_games:
            emit_progress(
                progress,
                f"[data] processed {completed}/{num_games} games, cases={len(cases)}, explored_states={explored_states}",
            )
    if not cases:
        raise RuntimeError("No training examples were generated.")
    return cases


def train_policy_model(
    num_games: int = 60,
    epochs: int = 30,
    learning_rate: float = 0.02,
    hidden_dim: int = 64,
    teacher_policy: str = "search",
    benchmark_games: int = 20,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    generated_cases = build_training_cases(num_games=num_games, teacher_policy=teacher_policy, progress=progress)
    human_cases, human_feedback_summary = load_human_feedback_cases(progress=progress)
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
        "benchmark_games": benchmark_games,
        "training_cases": len(cases),
        "generated_training_cases": len(generated_cases),
        "human_feedback_cases": len(human_cases),
        "effective_human_training_cases": len(human_cases) * human_repeat_factor,
        "human_repeat_factor": human_repeat_factor,
        "human_feedback_records_seen": human_feedback_summary.records_seen,
        "human_feedback_unique_entries": human_feedback_summary.unique_entries,
        "human_feedback_inferred_stock_cases": human_feedback_summary.inferred_stock_cases,
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


def choose_model_action(state: GameState, model: NumpyPolicyNetwork) -> Optional[Action]:
    actions = available_actions(state)
    if not actions:
        return None
    features = np.vstack([encode_action(state, action) for action in actions])
    scores = model.score(features)
    best_index = int(np.argmax(scores))
    return actions[best_index]


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

    emit_progress(progress, f"[eval:{policy}] seed={seed} progress={progress_score(state)} status={state.status}")

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
    if model is not None and actions:
        features = np.vstack([encode_action(state, action) for action in actions])
        predictions = model.score(features)
        for action, score in zip(actions, predictions):
            model_scores[action.to_dict(state)["description"]] = float(score)

    if model_eligible:
        heuristic_ranked.sort(
            key=lambda item: model_scores.get(item[1].to_dict(state)["description"], -1.0),
            reverse=True,
        )

    ranked: List[Dict[str, Any]] = []
    for heuristic_score, action in heuristic_ranked[:limit]:
        action_dict = action.to_dict(state)
        ranked.append(
            {
                **action_dict,
                "heuristic_score": heuristic_scores[action_dict["description"]],
                "search_score": search_scores.get(action_dict["description"]),
                "model_score": model_scores.get(action_dict["description"]),
            }
        )
    return {"suggestions": ranked, "model_loaded": model is not None, "model_eligible": model_eligible}


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
        "normal": 8.0,
        "important": 14.0,
        "key_move": 22.0,
    }
    return weights.get(strength, 4.0)


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
    progress: ProgressCallback = None,
) -> Tuple[List[TrainingCase], HumanFeedbackSummary]:
    summary = HumanFeedbackSummary()
    if not path.exists():
        emit_progress(progress, f"[feedback] no human feedback file at {path}")
        return [], summary

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary.records_seen = len(rows)
    deduped_rows = _dedupe_feedback_records(rows)
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
        weight = _feedback_strength_weight(str(record.get("feedback_strength", "normal")))
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
                target_boost=DEFAULT_EXACT_HUMAN_TARGET_BOOST if exact_stock else DEFAULT_HUMAN_TARGET_BOOST,
                source="human",
            )
        )

    summary.cases_loaded = len(cases)
    emit_progress(
        progress,
        "[feedback] loaded "
        f"{summary.cases_loaded}/{summary.unique_entries} human cases "
        f"(records={summary.records_seen}, inferred_stock={summary.inferred_stock_cases}, "
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
