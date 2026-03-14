from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
import uuid
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
from .training import rank_actions_for_state, train_policy_model


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "backend" / "data"
FEEDBACK_LOG_PATH = DATA_DIR / "human_feedback.jsonl"
SESSIONS_DIR = DATA_DIR / "sessions"


@dataclass
class GameSession:
    game_id: str
    history: List[GameState] = field(default_factory=list)

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
        }
        self._session_path(session.game_id).write_text(json.dumps(payload), encoding="utf-8")

    def _load(self, game_id: str) -> GameSession:
        path = self._session_path(game_id)
        if not path.exists():
            raise KeyError(game_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        session = GameSession(
            game_id=payload["game_id"],
            history=[_deserialize_game_state(state_data) for state_data in payload["history"]],
        )
        self._games[game_id] = session
        return session

    def create(self, seed: Optional[int] = None) -> GameSession:
        game_id = uuid.uuid4().hex[:12]
        session = GameSession(game_id=game_id, history=[create_game(seed=seed)])
        self._games[game_id] = session
        self._save(session)
        return session

    def get(self, game_id: str) -> GameSession:
        if game_id not in self._games:
            return self._load(game_id)
        return self._games[game_id]

    def push(self, game_id: str, state: GameState) -> GameSession:
        session = self.get(game_id)
        session.history.append(state)
        self._save(session)
        return session

    def undo(self, game_id: str) -> GameSession:
        session = self.get(game_id)
        if len(session.history) <= 1:
            raise InvalidMoveError("No earlier state available.")
        session.history.pop()
        self._save(session)
        return session


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
    ai_source: str
    model_loaded: bool = False
    model_eligible: bool = False
    feedback_strength: str = Field(default="normal")
    applies_to: str = Field(default="last_move")
    recommended_action: Dict[str, Any]
    chosen_action: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]] = Field(default_factory=list)
    state_snapshot: Dict[str, Any]
    note: Optional[str] = None


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


def _serialize_session(session: GameSession) -> Dict[str, Any]:
    return serialize_state(session.state, game_id=session.game_id, history_depth=len(session.history) - 1)


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
    snapshot = serialize_state(state, game_id=game_id, history_depth=history_depth)
    snapshot["stock_cards"] = [_card_dict(card) for card in state.stock]
    return snapshot


def _find_state_by_hash(session: GameSession, state_hash: str) -> Optional[tuple[int, GameState]]:
    for index in range(len(session.history) - 1, -1, -1):
        candidate = session.history[index]
        if candidate.state_hash == state_hash:
            return index, candidate
    return None


def _apply_state(game_id: str, state: GameState) -> Dict[str, Any]:
    session = store.push(game_id, state)
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
    return _apply_state(request.game_id, next_state)


@app.post("/game/deal")
def deal(request: GameActionRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    try:
        next_state = apply_deal(session.state)
    except InvalidMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _apply_state(request.game_id, next_state)


@app.post("/game/concede")
def concede(request: GameActionRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    next_state = apply_concede(session.state)
    return _apply_state(request.game_id, next_state)


@app.post("/game/undo")
def undo(request: GameActionRequest) -> Dict[str, Any]:
    try:
        session = store.undo(request.game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown game_id.") from exc
    except InvalidMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_session(session)


@app.get("/game/legal-moves")
def get_legal_moves(game_id: str) -> Dict[str, Any]:
    session = _get_session(game_id)
    return {
        "game_id": game_id,
        "legal_moves": [move.to_dict(session.state) for move in legal_moves(session.state)],
        "can_deal": can_deal(session.state),
    }


@app.post("/ai/evaluate-move")
def evaluate_move(request: GameActionRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    ranking = rank_actions_for_state(session.state)
    return {"game_id": request.game_id, **ranking}


@app.post("/ai/feedback")
def save_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    session = _get_session(request.game_id)
    matched_state = _find_state_by_hash(session, request.state_hash)
    if matched_state is None:
        authoritative_snapshot = request.state_snapshot
    else:
        history_index, history_state = matched_state
        authoritative_snapshot = _snapshot_with_stock(history_state, request.game_id, history_index)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
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
    }
    with feedback_lock:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
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
