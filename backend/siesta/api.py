from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import random as _random
import threading
import time as _time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .game import (
    Card,
    GameState,
    InvalidMoveError,
    Move,
    apply_concede,
    apply_deal,
    apply_move,
    can_deal,
    create_game,
    legal_moves,
    serialize_state,
)
from .deadlock import (
    dead_card_positions,
    is_provably_lost,
)
from .reachability import DEFAULT_MAX_DEPTH, cards_movable_within
from .training import train_policy_model
from .fastgame import (
    RANK as FAST_RANK,
    apply_move as fast_apply_move,
    card_code,
    dead_columns as fast_dead_columns,
    game_state_to_fast,
    is_won as fast_is_won,
)
from .solver import (
    TableauScorer,
    choose_handoff_target,
    choose_segment_target,
    solve_endgame,
    path_to as fast_path_to,
    sample_stock_orders,
    segment_search as fast_segment_search,
    segment_search_beam,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "backend" / "data"
FEEDBACK_LOG_PATH = DATA_DIR / "human_feedback.jsonl"
SESSIONS_DIR = DATA_DIR / "sessions"


@dataclass
class GameSession:
    game_id: str
    history: List[GameState] = field(default_factory=list)
    actions: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    @property
    def state(self) -> GameState:
        return self.history[-1]


class SessionStore:
    def __init__(self, storage_dir: Path = SESSIONS_DIR) -> None:
        self._games: Dict[str, GameSession] = {}
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, game_id: str) -> Path:
        return self._storage_dir / f"{game_id}.json"

    def _save(self, session: GameSession) -> None:
        payload = {
            "game_id": session.game_id,
            "history": [_serialize_game_state(state) for state in session.history],
            "actions": session.actions,
        }
        self._session_path(session.game_id).write_text(json.dumps(payload), encoding="utf-8")

    def _load(self, game_id: str) -> GameSession:
        path = self._session_path(game_id)
        if not path.exists():
            raise KeyError(game_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_actions = payload.get("actions", [])
        session = GameSession(
            game_id=payload["game_id"],
            history=[_deserialize_game_state(state_data) for state_data in payload["history"]],
            actions=[action if isinstance(action, dict) else None for action in raw_actions],
        )
        expected_actions = max(0, len(session.history) - 1)
        if len(session.actions) < expected_actions:
            session.actions.extend([None] * (expected_actions - len(session.actions)))
        elif len(session.actions) > expected_actions:
            session.actions = session.actions[:expected_actions]
        self._games[game_id] = session
        return session

    def create(self, seed: Optional[int] = None) -> GameSession:
        game_id = uuid.uuid4().hex[:12]
        session = GameSession(game_id=game_id, history=[create_game(seed=seed)], actions=[])
        self._games[game_id] = session
        self._save(session)
        return session

    def get(self, game_id: str) -> GameSession:
        if game_id not in self._games:
            return self._load(game_id)
        return self._games[game_id]

    def push(self, game_id: str, state: GameState, action: Optional[Dict[str, Any]] = None) -> GameSession:
        session = self.get(game_id)
        session.history.append(state)
        session.actions.append(action)
        self._save(session)
        return session

    def undo(self, game_id: str) -> tuple[GameSession, Optional[Dict[str, Any]]]:
        session = self.get(game_id)
        if len(session.history) <= 1:
            raise InvalidMoveError("No earlier state available.")
        session.history.pop()
        undone_action = session.actions.pop() if session.actions else None
        self._save(session)
        return session, undone_action


class TrainingJob(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class NewGameRequest(BaseModel):
    seed: Optional[int] = None


class MoveRequest(BaseModel):
    game_id: str
    from_column: int = Field(ge=0, le=6)
    to_column: int = Field(ge=0, le=6)
    run_length: int = Field(ge=1)


class GameActionRequest(BaseModel):
    game_id: str


class TrainingRunRequest(BaseModel):
    num_games: int = Field(default=60, ge=10, le=500)
    epochs: int = Field(default=30, ge=1, le=200)
    learning_rate: float = Field(default=0.02, gt=0.0, le=1.0)
    hidden_dim: int = Field(default=64, ge=8, le=512)


class FeedbackRequest(BaseModel):
    game_id: str
    state_hash: str
    ai_source: str = Field(default="none")
    model_loaded: bool = False
    model_eligible: bool = False
    feedback_strength: str = Field(default="normal")
    applies_to: str = Field(default="last_move")
    recommended_action: Optional[Dict[str, Any]] = None
    chosen_action: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]] = Field(default_factory=list)
    state_snapshot: Dict[str, Any]
    note: Optional[str] = None


class SolveAdviceRequest(BaseModel):
    game_id: str
    budget_s: float = Field(default=15.0, ge=1.0, le=120.0)
    midgame_nodes: int = Field(default=150000, ge=10000, le=1000000)


store = SessionStore()
training_jobs: Dict[str, TrainingJob] = {}
feedback_lock = threading.Lock()

app = FastAPI(title="Siesta", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _annotate_deadlock(snapshot: Dict[str, Any], state: GameState) -> Dict[str, Any]:
    """Mark cards that can only ever move into a hole, and flag a locked deal.

    Kings are dead by definition and carry no information, so they are reported
    separately from the ranks that were *derived* dead - those are the ones that
    say something about how the position is going.
    """

    dead = dead_card_positions(state)
    per_column: Dict[int, List[int]] = {}
    for column_index, card_index in dead:
        per_column.setdefault(column_index, []).append(card_index)

    for column_index, card_indices in per_column.items():
        for card_index in card_indices:
            is_king = state.columns[column_index][card_index].rank == 13
            card = snapshot["columns"][column_index]["cards"][card_index]
            card["dead"] = True
            # Kings are dead by definition rather than derived, but they are
            # marked too: it makes the marking verifiable by eye.
            card["dead_by_rule"] = not is_king
            card["dead_marked"] = True

    snapshot["provably_lost"] = is_provably_lost(state)
    snapshot["dead_card_count"] = len(dead)
    return snapshot


def _serialize_session(session: GameSession) -> Dict[str, Any]:
    return _annotate_deadlock(
        serialize_state(session.state, game_id=session.game_id, history_depth=len(session.history) - 1),
        session.state,
    )


def _serialize_game_state(state: GameState) -> Dict[str, Any]:
    return {
        "columns": [[card.to_dict() for card in column] for column in state.columns],
        "stock": [card.to_dict() for card in state.stock],
        "status": state.status,
        "moves_played": state.moves_played,
        "terminal_reason": state.terminal_reason,
        "state_hash": state.state_hash,
        "completed_sequences": state.completed_sequences,
    }


def _deserialize_card(payload: Dict[str, Any]) -> Card:
    return Card(rank=int(payload["rank"]), suit=str(payload["suit"]))


def _deserialize_game_state(payload: Dict[str, Any]) -> GameState:
    return GameState(
        columns=[[_deserialize_card(card) for card in column] for column in payload["columns"]],
        stock=[_deserialize_card(card) for card in payload["stock"]],
        status=str(payload["status"]),
        moves_played=int(payload["moves_played"]),
        terminal_reason=payload.get("terminal_reason"),
        state_hash=str(payload.get("state_hash", "")),
        completed_sequences=int(payload.get("completed_sequences", 0)),
    )


def _get_session(game_id: str) -> GameSession:
    try:
        return store.get(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown game_id.") from exc


def _card_dict(card: Card) -> Dict[str, Any]:
    return card.to_dict()


def _snapshot_with_stock(state: GameState, game_id: str, history_depth: int) -> Dict[str, Any]:
    snapshot = _annotate_deadlock(serialize_state(state, game_id=game_id, history_depth=history_depth), state)
    snapshot["stock_cards"] = [_card_dict(card) for card in state.stock]
    return snapshot


def _find_state_by_hash(session: GameSession, state_hash: str) -> Optional[tuple[int, GameState]]:
    for index in range(len(session.history) - 1, -1, -1):
        candidate = session.history[index]
        if candidate.state_hash == state_hash:
            return index, candidate
    return None


def _normalize_action_payload(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not action:
        return {}
    return {
        "type": action.get("type"),
        "from_column": action.get("from_column"),
        "to_column": action.get("to_column"),
        "run_length": action.get("run_length"),
        "description": action.get("description"),
    }


def _feedback_record_key(state_hash: str, chosen_action: Optional[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "state_hash": state_hash,
            "chosen_action": _normalize_action_payload(chosen_action),
        },
        sort_keys=True,
    )


def _append_feedback_log(record: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with feedback_lock:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _invalidate_feedback_for_action(game_id: str, state_hash: str, chosen_action: Optional[Dict[str, Any]]) -> None:
    if not chosen_action:
        return
    record = {
        "record_type": "invalidation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "game_id": game_id,
        "state_hash": state_hash,
        "chosen_action": _normalize_action_payload(chosen_action),
        "feedback_key": _feedback_record_key(state_hash, chosen_action),
    }
    _append_feedback_log(record)


def _apply_state(game_id: str, state: GameState, action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session = store.push(game_id, state, action=action)
    return _serialize_session(session)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/game/new")
def new_game(request: NewGameRequest) -> Dict[str, Any]:
    session = store.create(seed=request.seed)
    return _serialize_session(session)


@app.get("/game/state")
def get_game_state(game_id: str) -> Dict[str, Any]:
    session = _get_session(game_id)
    return _serialize_session(session)


@app.post("/game/move")
def make_move(request: MoveRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    try:
        next_state = apply_move(
            session.state,
            Move(
                from_column=request.from_column,
                to_column=request.to_column,
                run_length=request.run_length,
            ),
        )
    except InvalidMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    action = {
        "type": "move",
        "from_column": request.from_column,
        "to_column": request.to_column,
        "run_length": request.run_length,
        "description": f"kol {request.from_column + 1} -> kol {request.to_column + 1} ({request.run_length})",
    }
    return _apply_state(request.game_id, next_state, action=action)


@app.post("/game/deal")
def deal(request: GameActionRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    try:
        next_state = apply_deal(session.state)
    except InvalidMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    action = {
        "type": "deal",
        "description": f"Deal from stock ({len(session.state.stock)} left)",
    }
    return _apply_state(request.game_id, next_state, action=action)


@app.post("/game/concede")
def concede(request: GameActionRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    next_state = apply_concede(session.state)
    return _apply_state(request.game_id, next_state, action={"type": "concede", "description": "Concede game"})


@app.post("/game/undo")
def undo(request: GameActionRequest) -> Dict[str, Any]:
    try:
        session, undone_action = store.undo(request.game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown game_id.") from exc
    except InvalidMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _invalidate_feedback_for_action(request.game_id, session.state.state_hash, undone_action)
    return _serialize_session(session)


@app.get("/game/legal-moves")
def get_legal_moves(game_id: str) -> Dict[str, Any]:
    session = _get_session(game_id)
    return {
        "game_id": game_id,
        "legal_moves": [move.to_dict(session.state) for move in legal_moves(session.state)],
        "can_deal": can_deal(session.state),
    }


@app.get("/game/reachable-moves")
def get_reachable_moves(game_id: str, depth: int = DEFAULT_MAX_DEPTH) -> Dict[str, Any]:
    """Cards movable within `depth` moves.

    Deliberately its own endpoint rather than part of the snapshot: at depth 4
    this costs up to ~340ms on a branchy position, which would land on every
    move. Here it is only paid when the marking is switched on.
    """

    if depth < 1 or depth > 4:
        raise HTTPException(status_code=400, detail="depth must be between 1 and 4.")
    session = _get_session(game_id)
    return {
        "game_id": game_id,
        "state_hash": session.state.state_hash,
        "depth": depth,
        "movable_within": cards_movable_within(session.state, max_depth=depth),
    }


@app.post("/ai/evaluate-move")
def evaluate_move(request: GameActionRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    return {"game_id": request.game_id, **_solver_ranking(session.state)}


SOLVER_MIDGAME_NODES = 60000
SOLVER_ENDGAME_BUDGET_S = 2.0


def _solver_ranking(state: GameState) -> Dict[str, Any]:
    """Search-engine move suggestions for the UI.

    Endgame: hunt for a proven winning line; if one exists its first moves are
    served as ordered, playable suggestions. Mid-game: steer toward the best
    tableau reachable before the next deal and serve the first plan steps.
    Only visible cards and the multiset of unseen ranks are ever used.
    """
    import time as _time

    started = _time.time()
    scorer = TableauScorer()
    cols, stock = game_state_to_fast(state.columns, state.stock)

    def finish(suggestions: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
        return {
            "suggestions": suggestions,
            "model_loaded": False,
            "model_eligible": False,
            "hole_model_loaded": False,
            "hole_model_active": False,
            "compact_model_loaded": False,
            "compact_model_active": False,
            "ranking_source": source,
        }

    base_score = scorer.score(cols, deal_size=min(7, len(stock)))

    def move_suggestion(cols_before, move, label_prefix: str) -> Dict[str, Any]:
        action = _fast_move_action(cols_before, move)
        child = _apply_fast_move(cols_before, move)
        action["heuristic_score"] = round(
            scorer.score(child, deal_size=0) - scorer.score(cols_before, deal_size=0), 1
        )
        if label_prefix:
            action["description"] = f"{label_prefix} {action['description']}"
        return action

    if not stock:
        remaining = SOLVER_ENDGAME_BUDGET_S - (_time.time() - started)
        path = solve_endgame(cols, time_budget=max(0.5, remaining))
        if path:
            suggestions = []
            cursor = cols
            for index, move in enumerate(path[:6], start=1):
                suggestions.append(
                    move_suggestion(
                        cursor,
                        move,
                        f"Vinstlinje {index}/{len(path)}:",
                    )
                )
                cursor = _apply_fast_move(cursor, move)
            return finish(suggestions, "solver+vinstlinje")

    seen, mobility_map = fast_segment_search(
        cols,
        budget=SOLVER_MIDGAME_NODES if stock else max(20000, SOLVER_MIDGAME_NODES // 3),
        max_depth=18,
    )
    deal_size = min(7, len(stock))
    rng = _random.Random(state.moves_played)
    samples = sample_stock_orders(stock, deal_size, 10, rng) if stock else None
    target, _score = choose_segment_target(
        seen,
        scorer,
        deal_size=deal_size,
        stock_rank_counts=dict(Counter(FAST_RANK[cid] for cid in stock)),
        lock_penalty=40.0,
        stock_sample=samples,
        w_post_deal_mobility=6.0,
    )

    if target is not None and target != cols:
        path = fast_path_to(seen, target)
        suggestions = []
        cursor = cols
        for index, move in enumerate(path[:6], start=1):
            suggestions.append(move_suggestion(cursor, move, f"Plan {index}/{len(path)}:"))
            cursor = _apply_fast_move(cursor, move)
        return finish(suggestions, "solver")

    if can_deal(state):
        return finish(
            [{
                "type": "deal",
                "description": f"Dela kort ({len(state.stock)} kvar)",
                "heuristic_score": 0.0,
            }],
            "solver",
        )
    return finish([], "solver")


def _fast_move_action(cols, move) -> Dict[str, Any]:
    """Describe one fast-engine move in the UI's action format."""
    from_column, to_column, run_length = move
    moving = cols[from_column][len(cols[from_column]) - run_length :]
    codes = "-".join(card_code(card) for card in moving)
    if cols[to_column]:
        description = f"{codes} -> {card_code(cols[to_column][-1])} (kol {to_column + 1})"
    else:
        description = f"{codes} -> tomt kol {to_column + 1}"
    return {
        "type": "move",
        "from_column": from_column,
        "to_column": to_column,
        "run_length": run_length,
        "description": description,
    }


def _apply_fast_move(cols, move):
    new_cols = list(cols)
    source = cols[move[0]]
    moved = source[len(source) - move[2] :]
    new_cols[move[0]] = source[: len(source) - move[2]]
    new_cols[move[1]] = cols[move[1]] + moved
    return tuple(new_cols)


@app.post("/ai/solve-advice")
def solve_advice(request: SolveAdviceRequest) -> Dict[str, Any]:
    """Search-based advice for the current position.

    Endgame (stock empty): runs the bidirectional solver for `budget_s`
    seconds. A found line is a *proven* win; failure to find one is not proof
    of loss unless the deadlock analysis says so.

    Mid-game: steers toward the best tableau reachable before the next deal
    and recommends its first move. Uses only visible cards plus the multiset
    of unseen ranks - never the stock order.
    """
    started = _time.time()
    session = _get_session(request.game_id)
    state = session.state
    base = {
        "game_id": request.game_id,
        "state_hash": state.state_hash,
        "stock_count": len(state.stock),
    }
    if state.status == "won" or fast_is_won(game_state_to_fast(state.columns, [])[0]):
        return {**base, "mode": "won", "can_win": True, "win_line": [],
                "recommended_action": None, "message": "Partiet är redan vunnet."}
    if state.status != "in_progress":
        return {**base, "mode": "over", "can_win": False, "win_line": [],
                "recommended_action": None, "message": "Partiet är avslutat."}

    cols, stock = game_state_to_fast(state.columns, state.stock)

    if not stock:
        empty_rank_counts = {rank: 0 for rank in range(1, 14)}
        if all(fast_dead_columns(cols, empty_rank_counts)):
            return {**base, "mode": "stuck", "can_win": False, "win_line": [],
                    "recommended_action": None,
                    "elapsed_s": round(_time.time() - started, 2),
                    "message": "Bevisligt förlorat: varje kolumn har ett dött kort och inget hål kan skapas."}
        path = solve_endgame(cols, time_budget=request.budget_s)
        elapsed = round(_time.time() - started, 2)
        if path is not None:
            actions = []
            cursor = cols
            for move in path:
                actions.append(_fast_move_action(cursor, move))
                cursor = _apply_fast_move(cursor, move)
            return {**base, "mode": "endgame", "can_win": True, "win_line": actions,
                    "recommended_action": actions[0] if actions else None,
                    "line_length": len(actions),
                    "elapsed_s": elapsed,
                    "message": f"Vinst finns! Bevisad vinstlinje på {len(actions)} drag. Spela exakt dessa drag."}
        return {**base, "mode": "endgame", "can_win": None, "win_line": [],
                "recommended_action": None, "elapsed_s": elapsed,
                "message": f"Ingen vinstlinje hittades inom {request.budget_s:.0f}s – positionen kan vara förlorad, men det är inte bevisat. Försök igen med längre budget vid behov."}

    # Mid-game: best reachable tableau before the next deal, searched with
    # the same beam + handoff machinery as full-game play: beam extends the
    # node budget far deeper than breadth-first search, and the final segment
    # (this deal empties the stock) is ranked by predicted endgame
    # winnability with survival weighting, not raw tableau score.
    scorer = TableauScorer()
    stock_rank_counts = dict(Counter(FAST_RANK[cid] for cid in stock))
    seen, _mobility_map = segment_search_beam(
        cols,
        scorer,
        budget=request.midgame_nodes,
        bfs_depth=6,
        max_depth=40,
        beam_width=2500,
        stock_rank_counts=stock_rank_counts,
        lock_penalty=40.0,
    )
    deal_size = min(7, len(stock))
    target = None
    if len(stock) <= 7:
        target = choose_handoff_target(
            seen,
            scorer,
            stock,
            deal_size=deal_size,
            lock_penalty=40.0,
            samples=14,
            seed_base=state.moves_played * 131 + (24 - len(stock)) // 7,
            survive_weight=3.0,
        )
    if target is None:
        rng = _random.Random(state.moves_played)
        samples = sample_stock_orders(stock, deal_size, 10, rng)
        target, _score = choose_segment_target(
            seen,
            scorer,
            deal_size=deal_size,
            stock_rank_counts=stock_rank_counts,
            lock_penalty=40.0,
            stock_sample=samples,
            w_post_deal_mobility=6.0,
        )
    elapsed = round(_time.time() - started, 2)
    if target is not None and target != cols:
        path = fast_path_to(seen, target)
        first = _fast_move_action(cols, path[0])
        message = (
            f"Rekommenderat drag: {first['description']} "
            f"(första steget mot bästa nåbara tablå, {len(path)} drag framåt)."
        )
        return {**base, "mode": "midgame", "can_win": None, "win_line": [],
                "recommended_action": first, "plan_length": len(path),
                "elapsed_s": elapsed, "message": message}
    if can_deal(state):
        action = {"type": "deal",
                  "description": f"Dela kort ({len(state.stock)} kvar i talongen)"}
        return {**base, "mode": "midgame", "can_win": None, "win_line": [],
                "recommended_action": action, "elapsed_s": elapsed,
                "message": "Inga förbättrande drag hittade – dela nästa kort."}
    return {**base, "mode": "stuck", "can_win": None, "win_line": [],
            "recommended_action": None, "elapsed_s": elapsed,
            "message": "Inga lagliga drag och talongen är tom."}


@app.post("/ai/feedback")
def save_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    matched_state = _find_state_by_hash(session, request.state_hash)
    if matched_state is None:
        authoritative_snapshot = request.state_snapshot
    else:
        history_index, history_state = matched_state
        authoritative_snapshot = _snapshot_with_stock(history_state, request.game_id, history_index)

    record = {
        "record_type": "feedback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "game_id": request.game_id,
        "state_hash": request.state_hash,
        "ai_source": request.ai_source,
        "model_loaded": request.model_loaded,
        "model_eligible": request.model_eligible,
        "feedback_strength": request.feedback_strength,
        "applies_to": request.applies_to,
        "recommended_action": request.recommended_action,
        "chosen_action": request.chosen_action,
        "candidate_actions": request.candidate_actions,
        "state_snapshot": authoritative_snapshot,
        "note": request.note,
        "feedback_key": _feedback_record_key(request.state_hash, request.chosen_action),
    }
    _append_feedback_log(record)
    return {"saved": True, "path": str(FEEDBACK_LOG_PATH)}


def _run_training_job(job_id: str, request: TrainingRunRequest) -> None:
    try:
        result = train_policy_model(
            num_games=request.num_games,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            hidden_dim=request.hidden_dim,
        )
        training_jobs[job_id] = TrainingJob(job_id=job_id, status="completed", result=result)
    except Exception as exc:  # pragma: no cover - surfaced via API
        training_jobs[job_id] = TrainingJob(job_id=job_id, status="failed", error=str(exc))


@app.post("/training/run")
def run_training(request: TrainingRunRequest) -> TrainingJob:
    job_id = uuid.uuid4().hex[:12]
    training_jobs[job_id] = TrainingJob(job_id=job_id, status="running")
    thread = threading.Thread(target=_run_training_job, args=(job_id, request), daemon=True)
    thread.start()
    return training_jobs[job_id]


@app.get("/training/status/{job_id}")
def get_training_status(job_id: str) -> TrainingJob:
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Unknown training job.")
    return training_jobs[job_id]
