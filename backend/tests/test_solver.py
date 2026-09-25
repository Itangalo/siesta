from backend.siesta.fastgame import apply_move, is_won
from backend.siesta.solver import astar_endgame, misplaced_count, solve_endgame


def suit_stack(suit, top=13, bottom=1):
    return tuple(suit * 13 + rank - 1 for rank in range(top, bottom - 1, -1))


def test_misplaced_count_is_zero_only_for_settled_tableau():
    won = (suit_stack(0), suit_stack(1), suit_stack(2), suit_stack(3), (), (), ())
    assert misplaced_count(won) == 0
    split = (suit_stack(0), suit_stack(1), suit_stack(2), suit_stack(3, 13, 7), suit_stack(3, 6, 1), (), ())
    assert misplaced_count(split) == 1  # the 6 of spades rests on nothing


def test_astar_finds_and_verifies_a_winning_line():
    # Spades split in three pieces, the middle one covered by the ace of clubs.
    cols = (
        suit_stack(0),
        suit_stack(1),
        suit_stack(2, 13, 2),
        suit_stack(3, 13, 9),
        suit_stack(3, 8, 4) + (2 * 13 + 0,),
        suit_stack(3, 3, 1),
        (),
    )
    path, proven_lost = astar_endgame(cols, time_budget=10)
    assert not proven_lost
    assert path is not None
    for move in path:
        cols = apply_move(cols, move)
    assert is_won(cols)


def test_astar_proves_loss_when_search_space_is_exhausted():
    # Fewer than 52 cards can never be won, so the search must exhaust.
    cols = (suit_stack(0, 5, 1), (2 * 13 + 6,), (1 * 13 + 3, 3 * 13 + 9), (), (), (), ())
    path, proven_lost = astar_endgame(cols, time_budget=10)
    assert path is None
    assert proven_lost


def test_solve_endgame_returns_line_from_astar():
    cols = (suit_stack(0), suit_stack(1), suit_stack(2), suit_stack(3, 13, 7), suit_stack(3, 6, 1), (), ())
    path = solve_endgame(cols, time_budget=5)
    assert path == [(4, 3, 6)]


def card(rank, suit):
    return suit * 13 + rank - 1


def test_king_ladder_length_counts_mixed_suits_from_column_bottom():
    from backend.siesta.solver import TableauScorer

    scorer = TableauScorer()
    full = tuple(card(rank, rank % 4) for rank in range(13, 0, -1))
    assert scorer.king_ladder_length(full) == 13
    assert scorer.king_ladder_length(full[:4] + (card(5, 0),)) == 4
    assert scorer.king_ladder_length((card(9, 0),) + full) == 0  # no king at the bottom


def test_ladder_safe_burial_does_not_penalise_king_ladders():
    from backend.siesta.solver import TableauScorer

    ladder = tuple(card(rank, rank % 4) for rank in range(13, 0, -1))
    covered = ladder + (card(9, 0), card(4, 2))
    naive = TableauScorer(w_buried_pairs=10)
    aware = TableauScorer(w_buried_pairs=10, ladder_safe_burial=True)
    assert naive.column_shape(covered)[0] > 0
    assert aware.column_shape(covered)[0] == 0


def test_super_moves_are_legal_move_sequences():
    from backend.siesta.fastgame import legal_moves
    from backend.siesta.solver import super_moves

    # A mixed 8-7-6-5 ladder of three suited segments, a 9 to land on, two holes.
    ladder_col = (card(2, 0), card(8, 1), card(7, 1), card(6, 2), card(5, 3))
    cols = (ladder_col, (card(9, 0),), (), (), (card(12, 3),), (card(3, 1),), (card(11, 2),))
    macros = super_moves(cols)
    assert macros
    for moves, result in macros:
        state = cols
        for move in moves:
            assert move in legal_moves(state)
            state = apply_move(state, move)
        assert state == result
    onto_nine = [result for _moves, result in macros if result[1][-4:] == ladder_col[1:]]
    assert onto_nine and onto_nine[0][0] == (card(2, 0),)
