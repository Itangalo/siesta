import json

import numpy as np

import backend.siesta.training as training_module
from backend.siesta.game import Card, GameState
from backend.siesta.training import (
    HumanFeedbackSummary,
    NumpyPolicyNetwork,
    TrainingCase,
    benchmark_policies,
    build_training_cases,
    build_training_examples,
    choose_search_action,
    expand_human_cases_for_training,
    evaluate_policy,
    human_case_repeat_factor,
    load_human_feedback_cases,
    rank_actions_for_state,
)


def make_state(columns, stock=None):
    return GameState(columns=columns, stock=stock or [])


def test_search_action_returns_legal_action():
    state = make_state(
        [
            [Card(6, "hearts"), Card(5, "hearts"), Card(4, "hearts")],
            [Card(7, "spades")],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    action = choose_search_action(state)
    assert action is not None
    payload = action.to_dict(state)
    assert payload["type"] == "move"
    assert payload["description"]


def test_search_teacher_generates_training_examples():
    features, labels = build_training_examples(
        num_games=2,
        max_moves=10,
        teacher_policy="search",
        search_depth=2,
        beam_width=3,
        rollout_steps=4,
        alternate_actions=0,
        max_states_per_game=1,
    )
    assert len(features) == len(labels)
    assert features.shape[1] > 0
    assert labels.sum() > 0


def test_training_cases_capture_ranked_choice():
    cases = build_training_cases(
        num_games=2,
        max_moves=10,
        teacher_policy="search",
        search_depth=2,
        beam_width=3,
        rollout_steps=4,
        alternate_actions=0,
        max_states_per_game=1,
    )
    assert cases
    assert all(case.features.shape[0] == len(case.action_signatures) for case in cases)
    assert all(0 <= case.target_index < len(case.action_signatures) for case in cases)
    assert all(case.weight >= 1.0 for case in cases)
    assert all(case.soft_targets.shape[0] == len(case.action_signatures) for case in cases)
    assert all(abs(float(case.soft_targets.sum()) - 1.0) < 1e-5 for case in cases)


def test_search_policy_evaluation_and_benchmark_work():
    evaluation = evaluate_policy(
        policy="search",
        seeds=range(2),
        search_depth=1,
        search_beam_width=2,
        search_rollout_steps=2,
    )
    assert evaluation["policy"] == "search"
    assert evaluation["games"] == 2

    benchmark = benchmark_policies(
        policies=("heuristic", "search"),
        games=2,
        search_depth=1,
        search_beam_width=2,
        search_rollout_steps=2,
    )
    assert set(benchmark) == {"heuristic", "search"}
    assert benchmark["search"]["games"] == 2


def test_search_prefers_creating_hole_when_available():
    state = make_state(
        [
            [Card(6, "hearts")],
            [Card(7, "spades")],
            [Card(5, "clubs"), Card(4, "clubs")],
            [Card(5, "diamonds")],
            [],
            [],
            [],
        ]
    )
    action = choose_search_action(state)
    assert action is not None
    assert action.move is not None
    source_height = len(state.columns[action.move.from_column])
    assert action.move.run_length == source_height


def test_ui_ranking_returns_compact_suggestions():
    state = make_state(
        [
            [Card(13, "spades"), Card(12, "spades"), Card(11, "spades")],
            [Card(13, "hearts"), Card(12, "hearts"), Card(11, "hearts")],
            [Card(13, "clubs"), Card(12, "clubs"), Card(11, "clubs")],
            [Card(13, "diamonds"), Card(12, "diamonds"), Card(11, "diamonds")],
            [Card(10, "spades")],
            [Card(10, "hearts")],
            [Card(10, "clubs")],
        ],
        stock=[Card(1, "diamonds")],
    )
    ranking = rank_actions_for_state(state, limit=5)
    assert len(ranking["suggestions"]) <= 5
    assert ranking["suggestions"]
    assert all("description" in suggestion for suggestion in ranking["suggestions"])


def test_load_human_feedback_cases_uses_strongest_duplicate(tmp_path):
    state_snapshot = {
        "status": "in_progress",
        "terminal_reason": None,
        "moves_played": 0,
        "stock_count": 0,
        "completed_sequences": 0,
        "state_hash": "demo",
        "columns": [
            {"index": 0, "cards": [{"rank": 6, "suit": "hearts"}], "movable_run_length": 1, "height": 1},
            {"index": 1, "cards": [{"rank": 7, "suit": "spades"}], "movable_run_length": 1, "height": 1},
            {"index": 2, "cards": [], "movable_run_length": 0, "height": 0},
            {"index": 3, "cards": [], "movable_run_length": 0, "height": 0},
            {"index": 4, "cards": [], "movable_run_length": 0, "height": 0},
            {"index": 5, "cards": [], "movable_run_length": 0, "height": 0},
            {"index": 6, "cards": [], "movable_run_length": 0, "height": 0},
        ],
        "stock_cards": [],
    }
    rows = [
        {
            "timestamp": "2026-03-14T10:00:00+00:00",
            "state_hash": "demo",
            "feedback_strength": "normal",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": state_snapshot,
        },
        {
            "timestamp": "2026-03-14T10:00:01+00:00",
            "state_hash": "demo",
            "feedback_strength": "important",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": state_snapshot,
        },
    ]
    path = tmp_path / "human_feedback.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    cases, summary = load_human_feedback_cases(path=path)
    assert len(cases) == 1
    assert summary.records_seen == 2
    assert summary.unique_entries == 1
    assert cases[0].weight >= 14.0
    assert cases[0].target_boost >= 0.9
    assert cases[0].source == "human"


def test_human_cases_are_repeated_to_gain_training_share():
    generated_cases = build_training_cases(
        num_games=2,
        max_moves=8,
        teacher_policy="search",
        search_depth=1,
        beam_width=2,
        rollout_steps=2,
        alternate_actions=0,
        max_states_per_game=1,
    )[:12]
    human_cases = generated_cases[:2]
    for case in human_cases:
        case.source = "human"
        case.target_boost = 0.9

    repeat_factor = human_case_repeat_factor(len(generated_cases), len(human_cases))
    expanded, actual_repeat_factor = expand_human_cases_for_training(generated_cases, human_cases)
    assert repeat_factor == actual_repeat_factor
    assert actual_repeat_factor > 1
    assert len(expanded) == len(generated_cases) + len(human_cases) * actual_repeat_factor


def test_training_does_not_replace_active_model_with_worse_candidate(tmp_path, monkeypatch):
    active_model_path = tmp_path / "policy-latest.npz"
    active_metadata_path = tmp_path / "policy-latest.json"
    candidate_model_path = tmp_path / "policy-candidate.npz"
    candidate_metadata_path = tmp_path / "policy-candidate.json"
    best_model_path = tmp_path / "policy-best.npz"
    best_metadata_path = tmp_path / "policy-best.json"

    monkeypatch.setattr(training_module, "DEFAULT_MODEL_PATH", active_model_path)
    monkeypatch.setattr(training_module, "DEFAULT_METADATA_PATH", active_metadata_path)
    monkeypatch.setattr(training_module, "DEFAULT_CANDIDATE_MODEL_PATH", candidate_model_path)
    monkeypatch.setattr(training_module, "DEFAULT_CANDIDATE_METADATA_PATH", candidate_metadata_path)
    monkeypatch.setattr(training_module, "DEFAULT_BEST_MODEL_PATH", best_model_path)
    monkeypatch.setattr(training_module, "DEFAULT_BEST_METADATA_PATH", best_metadata_path)

    case = TrainingCase(
        features=np.zeros((2, 4), dtype=np.float32),
        target_index=0,
        weight=1.0,
        chosen_signature="a",
        action_signatures=["a", "b"],
        soft_targets=np.asarray([0.9, 0.1], dtype=np.float32),
    )
    monkeypatch.setattr(training_module, "build_training_cases", lambda **kwargs: [case])
    monkeypatch.setattr(training_module, "load_human_feedback_cases", lambda **kwargs: ([], HumanFeedbackSummary()))

    active_model = NumpyPolicyNetwork(input_dim=4, hidden_dim=8)
    active_model.save(active_model_path)
    previous_metadata = {"model_evaluation": {"win_rate": 0.0, "average_progress": 2.6, "average_completed_sequences": 0.0}}
    active_metadata_path.write_text(json.dumps(previous_metadata), encoding="utf-8")
    monkeypatch.setattr(training_module, "load_latest_model", lambda: active_model)
    monkeypatch.setattr(training_module, "load_model_metadata", lambda: previous_metadata)

    def fake_evaluate_policy(policy, model=None, seeds=None, progress=None, **kwargs):
        if policy == "heuristic":
            return {"policy": "heuristic", "games": len(list(seeds)), "win_rate": 0.0, "average_progress": 2.1, "average_completed_sequences": 0.0}
        if policy == "search":
            return {"policy": "search", "games": len(list(seeds)), "win_rate": 0.0, "average_progress": 2.7, "average_completed_sequences": 0.0}
        if model is active_model:
            return {"policy": "model", "games": len(list(seeds)), "win_rate": 0.0, "average_progress": 2.6, "average_completed_sequences": 0.0}
        return {"policy": "model", "games": len(list(seeds)), "win_rate": 0.0, "average_progress": 2.4, "average_completed_sequences": 0.0}

    monkeypatch.setattr(training_module, "evaluate_policy", fake_evaluate_policy)

    result = training_module.train_policy_model(num_games=1, epochs=1, benchmark_games=4)
    active_metadata = json.loads(active_metadata_path.read_text(encoding="utf-8"))
    candidate_metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))

    assert result["promoted_to_active"] is False
    assert result["candidate_model_evaluation"]["average_progress"] == 2.4
    assert result["active_model_evaluation"]["average_progress"] == 2.6
    assert active_metadata["model_evaluation"]["average_progress"] == 2.6
    assert candidate_metadata["candidate_model_evaluation"]["average_progress"] == 2.4
