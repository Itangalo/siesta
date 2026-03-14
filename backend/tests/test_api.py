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
