from backend.siesta.game import Card, GameState
from backend.siesta.training import (
    benchmark_policies,
    build_training_cases,
    build_training_examples,
    choose_search_action,
    evaluate_policy,
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
