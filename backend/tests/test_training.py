import json

import numpy as np

import backend.siesta.training as training_module
from backend.siesta.game import Card, GameState
from backend.siesta.training import (
    best_hole_goal_outcome,
    benchmark_hole_goal,
    HumanFeedbackSummary,
    NumpyPolicyNetwork,
    TrainingCase,
    blocked_mid_rank_risk,
    benchmark_policies,
    build_hole_opening_cases,
    build_training_cases,
    build_training_examples,
    choose_search_action,
    expand_human_cases_for_training,
    evaluate_policy,
    hole_access_score,
    human_case_repeat_factor,
    load_human_feedback_cases,
    progress_score,
    rank_actions_for_state,
    training_game_seeds,
    top_color_mix_penalty,
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


def test_training_game_seeds_are_deterministic_but_not_sequential():
    seeds_a = training_game_seeds(6, seed_offset=0)
    seeds_b = training_game_seeds(6, seed_offset=0)
    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == len(seeds_a)
    assert seeds_a != list(range(6))


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
            [Card(8, "spades")],
            [Card(9, "hearts")],
            [Card(10, "clubs")],
        ]
    )
    action = choose_search_action(state)
    assert action is not None
    assert action.move is not None
    source_height = len(state.columns[action.move.from_column])
    assert action.move.run_length == source_height


def test_best_hole_goal_outcome_finds_hole_within_horizon():
    state = make_state(
        [
            [Card(6, "hearts")],
            [Card(7, "spades")],
            [Card(9, "clubs")],
            [Card(10, "diamonds")],
            [Card(11, "clubs")],
            [Card(12, "hearts")],
            [Card(13, "spades")],
        ]
    )
    outcome = best_hole_goal_outcome(state, horizon=1, beam_width=4)
    assert outcome["reachable"] is True
    assert outcome["first_hole_depth"] == 1
    assert outcome["best_empty_columns"] >= 1


def test_build_hole_opening_cases_generates_ranked_examples():
    cases = build_hole_opening_cases(num_games=3, horizon=3, beam_width=4)
    assert cases
    assert all(case.source == "hole_opening" for case in cases)
    assert all(case.features.shape[0] == len(case.action_signatures) for case in cases)


def test_benchmark_hole_goal_returns_expected_metrics():
    result = benchmark_hole_goal(policies=("heuristic", "hole_search"), games=2, horizon=3)
    assert set(result) == {"heuristic", "hole_search"}
    assert result["heuristic"]["games"] == 2
    assert "success_rate" in result["hole_search"]


def test_blocked_mid_rank_risk_penalizes_buried_middle_duplicates():
    risky = make_state(
        [
            [Card(9, "hearts"), Card(7, "hearts"), Card(7, "spades"), Card(7, "clubs"), Card(6, "clubs")],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    safe = make_state(
        [
            [Card(9, "hearts"), Card(4, "spades"), Card(3, "spades")],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    assert blocked_mid_rank_risk(risky) > blocked_mid_rank_risk(safe)


def test_hole_access_score_rewards_states_near_creating_hole():
    state = make_state(
        [
            [Card(6, "hearts")],
            [Card(7, "spades")],
            [Card(8, "clubs"), Card(7, "clubs")],
            [],
            [],
            [],
            [],
        ]
    )
    assert hole_access_score(state) > 0.0


def test_color_mix_penalty_grows_with_more_mixed_suits():
    two_suit_mix = make_state(
        [
            [Card(8, "spades"), Card(7, "hearts"), Card(6, "spades")],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    four_suit_mix = make_state(
        [
            [Card(8, "spades"), Card(7, "hearts"), Card(6, "clubs"), Card(5, "diamonds")],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    assert top_color_mix_penalty(four_suit_mix) > top_color_mix_penalty(two_suit_mix)


def test_progress_score_prefers_holes_over_longer_top_run():
    hole_state = make_state(
        [
            [],
            [Card(7, "spades")],
            [Card(6, "hearts")],
            [Card(10, "clubs")],
            [Card(9, "diamonds")],
            [Card(8, "clubs")],
            [Card(5, "hearts")],
        ]
    )
    run_state = make_state(
        [
            [Card(8, "spades"), Card(7, "spades"), Card(6, "spades")],
            [Card(10, "hearts")],
            [Card(9, "clubs")],
            [Card(8, "diamonds")],
            [Card(7, "clubs")],
            [Card(6, "hearts")],
            [Card(5, "spades")],
        ]
    )
    assert progress_score(hole_state) > progress_score(run_state)


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


def test_ui_ranking_can_use_hole_model_signal_in_opening(monkeypatch):
    state = make_state(
        [
            [Card(6, "hearts")],
            [Card(7, "spades")],
            [Card(9, "clubs")],
            [Card(10, "diamonds")],
            [Card(11, "clubs")],
            [Card(12, "hearts")],
            [Card(13, "spades")],
        ],
        stock=[Card(1, "clubs")] * 24,
    )

    class StubModel:
        def score(self, batch):
            return np.linspace(0.0, 1.0, len(batch), dtype=np.float32)

    monkeypatch.setattr(training_module, "load_latest_model", lambda: None)
    monkeypatch.setattr(training_module, "load_model_metadata", lambda: {})
    monkeypatch.setattr(training_module, "load_hole_model", lambda: StubModel())
    monkeypatch.setattr(training_module, "load_hole_model_metadata", lambda: {"ready": True})

    ranking = rank_actions_for_state(state, limit=3)
    assert ranking["hole_model_active"] is True
    assert ranking["ranking_source"] == "search+hole"
    assert any(suggestion.get("hole_model_score") is not None for suggestion in ranking["suggestions"])


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


def test_load_human_feedback_cases_skips_invalidated_entries(tmp_path):
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
    chosen_action = {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1}
    feedback_key = json.dumps(
        {
            "state_hash": "demo",
            "chosen_action": {
                "type": "move",
                "from_column": 0,
                "to_column": 1,
                "run_length": 1,
                "description": None,
            },
        },
        sort_keys=True,
    )
    rows = [
        {
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:00+00:00",
            "state_hash": "demo",
            "feedback_strength": "key_move",
            "chosen_action": chosen_action,
            "state_snapshot": state_snapshot,
            "feedback_key": feedback_key,
        },
        {
            "record_type": "invalidation",
            "timestamp": "2026-03-14T10:00:01+00:00",
            "state_hash": "demo",
            "chosen_action": chosen_action,
            "feedback_key": feedback_key,
        },
    ]
    path = tmp_path / "human_feedback.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    cases, summary = load_human_feedback_cases(path=path)
    assert len(cases) == 0
    assert summary.records_seen == 2
    assert summary.invalidated_entries == 1
    assert summary.unique_entries == 0


def test_load_human_feedback_cases_prioritizes_important_and_key_move(tmp_path):
    state_snapshot = {
        "status": "in_progress",
        "terminal_reason": None,
        "moves_played": 0,
        "stock_count": 0,
        "completed_sequences": 0,
        "state_hash": "weight-demo",
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
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:00+00:00",
            "state_hash": "weight-demo-normal",
            "feedback_strength": "normal",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "weight-demo-normal"},
        },
        {
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:01+00:00",
            "state_hash": "weight-demo-important",
            "feedback_strength": "important",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "weight-demo-important"},
        },
        {
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:02+00:00",
            "state_hash": "weight-demo-key",
            "feedback_strength": "key_move",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "weight-demo-key"},
        },
    ]
    path = tmp_path / "human_feedback.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    cases, _ = load_human_feedback_cases(path=path)
    weights = sorted(case.weight for case in cases)
    boosts = sorted(case.target_boost for case in cases)
    assert len(cases) == 3
    assert weights[0] < weights[1] < weights[2]
    assert boosts[0] < boosts[1] < boosts[2]


def test_load_human_feedback_cases_weights_won_games_above_conceded(tmp_path):
    state_snapshot = {
        "status": "in_progress",
        "terminal_reason": None,
        "moves_played": 0,
        "stock_count": 0,
        "completed_sequences": 0,
        "state_hash": "outcome-demo",
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
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:00+00:00",
            "game_id": "won-game",
            "state_hash": "outcome-demo-won",
            "feedback_strength": "normal",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "outcome-demo-won"},
        },
        {
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:01+00:00",
            "game_id": "conceded-game",
            "state_hash": "outcome-demo-conceded",
            "feedback_strength": "important",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "outcome-demo-conceded"},
        },
    ]
    path = tmp_path / "human_feedback.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "won-game.json").write_text(
        json.dumps({"game_id": "won-game", "history": [{"status": "won"}]}),
        encoding="utf-8",
    )
    (sessions_dir / "conceded-game.json").write_text(
        json.dumps({"game_id": "conceded-game", "history": [{"status": "conceded"}]}),
        encoding="utf-8",
    )

    cases, summary = load_human_feedback_cases(path=path, sessions_dir=sessions_dir)
    weights = sorted(case.weight for case in cases)
    assert len(cases) == 2
    assert weights[0] < weights[1]
    assert summary.won_cases == 1
    assert summary.conceded_cases == 1


def test_load_human_feedback_cases_excludes_normal_from_conceded_games(tmp_path):
    state_snapshot = {
        "status": "in_progress",
        "terminal_reason": None,
        "moves_played": 0,
        "stock_count": 0,
        "completed_sequences": 0,
        "state_hash": "filter-demo",
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
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:00+00:00",
            "game_id": "conceded-game-normal",
            "state_hash": "filter-demo-normal",
            "feedback_strength": "normal",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "filter-demo-normal"},
        },
        {
            "record_type": "feedback",
            "timestamp": "2026-03-14T10:00:01+00:00",
            "game_id": "conceded-game-important",
            "state_hash": "filter-demo-important",
            "feedback_strength": "important",
            "chosen_action": {"type": "move", "from_column": 0, "to_column": 1, "run_length": 1},
            "state_snapshot": {**state_snapshot, "state_hash": "filter-demo-important"},
        },
    ]
    path = tmp_path / "human_feedback.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "conceded-game-normal.json").write_text(
        json.dumps({"game_id": "conceded-game-normal", "history": [{"status": "conceded"}]}),
        encoding="utf-8",
    )
    (sessions_dir / "conceded-game-important.json").write_text(
        json.dumps({"game_id": "conceded-game-important", "history": [{"status": "conceded"}]}),
        encoding="utf-8",
    )

    cases, summary = load_human_feedback_cases(path=path, sessions_dir=sessions_dir)
    assert len(cases) == 1
    assert summary.skipped_normal_conceded_cases == 1
    assert summary.conceded_cases == 1
    assert cases[0].weight >= 18.0 * 0.82


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
