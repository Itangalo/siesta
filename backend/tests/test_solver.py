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
