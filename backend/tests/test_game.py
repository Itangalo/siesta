from backend.siesta.game import (
    Card,
    GameState,
    Move,
    STATUS_CONCEDED,
    STATUS_TRAINING_TERMINATED,
    STATUS_WON,
    apply_concede,
    apply_deal,
    apply_move,
    create_game,
    legal_moves,
    serialize_state,
    terminate_for_training,
)


def make_state(columns, stock=None):
    return GameState(columns=columns, stock=stock or [])


def test_new_game_deals_7_to_1_layout():
    state = create_game(seed=7)
    assert [len(column) for column in state.columns] == [7, 6, 5, 4, 3, 2, 1]
    assert len(state.stock) == 24


def test_allows_single_card_move_onto_exactly_one_higher_rank():
    state = make_state(
        [
            [Card(6, "hearts")],
            [Card(7, "spades")],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    next_state = apply_move(state, Move(from_column=0, to_column=1, run_length=1))
    assert [card.code for card in next_state.columns[1]] == ["7S", "6H"]


def test_disallows_broken_sequence_move():
    state = make_state(
        [
            [Card(6, "hearts"), Card(5, "hearts"), Card(3, "hearts")],
            [Card(7, "clubs")],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    moves = legal_moves(state)
    assert (0, 1, 2) not in {(move.from_column, move.to_column, move.run_length) for move in moves}


def test_allows_suffix_of_top_run_to_move():
    state = make_state(
        [
            [Card(6, "hearts"), Card(5, "hearts"), Card(4, "hearts")],
            [Card(6, "spades")],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    next_state = apply_move(state, Move(from_column=0, to_column=1, run_length=2))
    assert [card.code for card in next_state.columns[0]] == ["6H"]
    assert [card.code for card in next_state.columns[1]] == ["6S", "5H", "4H"]


def test_move_to_empty_column_is_allowed():
    state = make_state(
        [
            [Card(10, "clubs")],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
    )
    next_state = apply_move(state, Move(from_column=0, to_column=1, run_length=1))
    assert next_state.columns[0] == []
    assert [card.code for card in next_state.columns[1]] == ["10C"]


def test_deal_adds_cards_left_to_right_and_handles_short_final_deal():
    state = make_state(
        [[], [], [], [], [], [], []],
        stock=[
            Card(13, "hearts"),
            Card(12, "hearts"),
            Card(11, "hearts"),
        ],
    )
    next_state = apply_deal(state)
    assert [len(column) for column in next_state.columns] == [1, 1, 1, 0, 0, 0, 0]
    assert next_state.stock == []


def test_win_detection_accepts_four_complete_suit_sequences():
    hearts = [Card(rank, "hearts") for rank in range(13, 0, -1)]
    diamonds = [Card(rank, "diamonds") for rank in range(13, 0, -1)]
    clubs = [Card(rank, "clubs") for rank in range(13, 0, -1)]
    spades = [Card(rank, "spades") for rank in range(13, 0, -1)]
    state = make_state([hearts, diamonds, clubs, spades, [], [], []])
    serialized = serialize_state(state)
    assert serialized["status"] == STATUS_WON
    assert serialized["completed_sequences"] == 4


def test_concede_and_training_termination_are_distinct_terminal_states():
    state = create_game(seed=1)
    conceded = apply_concede(state)
    terminated = terminate_for_training(state, "loop")
    assert conceded.status == STATUS_CONCEDED
    assert terminated.status == STATUS_TRAINING_TERMINATED
    assert terminated.terminal_reason == "loop"
