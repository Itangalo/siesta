import json

from fastapi.testclient import TestClient

import backend.siesta.api as api_module


app = api_module.app


client = TestClient(app)


def test_new_game_is_seeded_and_deterministic():
    first = client.post("/game/new", json={"seed": 11})
    second = client.post("/game/new", json={"seed": 11})
    assert first.status_code == 200
    assert second.status_code == 200
    first_codes = [[card["code"] for card in column["cards"]] for column in first.json()["columns"]]
    second_codes = [[card["code"] for card in column["cards"]] for column in second.json()["columns"]]
    assert first_codes == second_codes


def test_invalid_move_returns_bad_request():
    response = client.post("/game/new", json={"seed": 4})
    game_id = response.json()["game_id"]
    invalid = client.post(
        "/game/move",
        json={"game_id": game_id, "from_column": 0, "to_column": 0, "run_length": 1},
    )
    assert invalid.status_code == 400


def test_existing_game_state_can_be_reloaded():
    response = client.post("/game/new", json={"seed": 5})
    game_id = response.json()["game_id"]
    loaded = client.get("/game/state", params={"game_id": game_id})
    assert loaded.status_code == 200
    assert loaded.json()["game_id"] == game_id


def test_session_store_recovers_game_after_memory_is_cleared(tmp_path):
    store = api_module.SessionStore(storage_dir=tmp_path / "sessions")
    created = store.create(seed=6)
    game_id = created.game_id
    created.history.append(created.state.clone())
    store._save(created)

    reloaded_store = api_module.SessionStore(storage_dir=tmp_path / "sessions")
    recovered = reloaded_store.get(game_id)
    assert recovered.game_id == game_id
    assert len(recovered.history) == 2


def test_concede_changes_status():
    response = client.post("/game/new", json={"seed": 8})
    game_id = response.json()["game_id"]
    conceded = client.post("/game/concede", json={"game_id": game_id})
    assert conceded.status_code == 200
    assert conceded.json()["status"] == "conceded"


def test_ai_suggestions_endpoint_returns_ranked_actions():
    response = client.post("/game/new", json={"seed": 9})
    game_id = response.json()["game_id"]
    suggestions = client.post("/ai/evaluate-move", json={"game_id": game_id})
    assert suggestions.status_code == 200
    payload = suggestions.json()
    assert "suggestions" in payload
    assert isinstance(payload["suggestions"], list)


def test_feedback_endpoint_saves_human_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "FEEDBACK_LOG_PATH", tmp_path / "human_feedback.jsonl")
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)

    response = client.post("/game/new", json={"seed": 9})
    snapshot = response.json()
    game_id = snapshot["game_id"]
    suggestions = client.post("/ai/evaluate-move", json={"game_id": game_id}).json()

    feedback = client.post(
        "/ai/feedback",
        json={
            "game_id": game_id,
            "state_hash": snapshot["state_hash"],
            "ai_source": "search",
            "model_loaded": suggestions["model_loaded"],
            "model_eligible": suggestions["model_eligible"],
            "feedback_strength": "key_move",
            "applies_to": "planned_move",
            "recommended_action": suggestions["suggestions"][0],
            "chosen_action": {"type": "deal", "description": "Deal from stock"},
            "candidate_actions": suggestions["suggestions"],
            "state_snapshot": snapshot,
            "note": "human override",
        },
    )
    assert feedback.status_code == 200
    saved = json.loads((tmp_path / "human_feedback.jsonl").read_text(encoding="utf-8").strip())
    assert saved["game_id"] == game_id
    assert saved["chosen_action"]["type"] == "deal"
    assert saved["feedback_strength"] == "key_move"
    assert saved["applies_to"] == "planned_move"
    assert "stock_cards" in saved["state_snapshot"]
    assert len(saved["state_snapshot"]["stock_cards"]) == saved["state_snapshot"]["stock_count"]


def test_feedback_endpoint_allows_move_logging_without_ai_context(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "FEEDBACK_LOG_PATH", tmp_path / "human_feedback.jsonl")
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)

    response = client.post("/game/new", json={"seed": 10})
    snapshot = response.json()
    game_id = snapshot["game_id"]

    feedback = client.post(
        "/ai/feedback",
        json={
            "game_id": game_id,
            "state_hash": snapshot["state_hash"],
            "feedback_strength": "normal",
            "applies_to": "last_move",
            "chosen_action": {"type": "deal", "description": "Deal from stock"},
            "candidate_actions": [],
            "state_snapshot": snapshot,
            "note": "auto-saved human move",
        },
    )
    assert feedback.status_code == 200
    saved = json.loads((tmp_path / "human_feedback.jsonl").read_text(encoding="utf-8").strip())
    assert saved["ai_source"] == "none"
    assert saved["recommended_action"] is None
    assert saved["chosen_action"]["type"] == "deal"


def test_undo_appends_feedback_invalidation_for_last_action(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "FEEDBACK_LOG_PATH", tmp_path / "human_feedback.jsonl")
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_module, "store", api_module.SessionStore(storage_dir=tmp_path / "sessions"))

    response = client.post("/game/new", json={"seed": 12})
    snapshot = response.json()
    game_id = snapshot["game_id"]

    client.post(
        "/ai/feedback",
        json={
            "game_id": game_id,
            "state_hash": snapshot["state_hash"],
            "feedback_strength": "normal",
            "applies_to": "last_move",
            "chosen_action": {"type": "deal", "description": f"Deal from stock ({snapshot['stock_count']} left)"},
            "candidate_actions": [],
            "state_snapshot": snapshot,
            "note": "auto-saved human deal",
        },
    )
    undo_target = client.post("/game/deal", json={"game_id": game_id})
    assert undo_target.status_code == 200

    undone = client.post("/game/undo", json={"game_id": game_id})
    assert undone.status_code == 200

    rows = [
        json.loads(line)
        for line in (tmp_path / "human_feedback.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[-1]["record_type"] == "invalidation"
    assert rows[-1]["state_hash"] == snapshot["state_hash"]
    assert rows[-1]["chosen_action"]["type"] == "deal"
