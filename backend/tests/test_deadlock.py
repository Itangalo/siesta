from backend.siesta.deadlock import (
    dead_card_positions,
    emptiable_column_count,
    is_provably_lost,
    locked_column_flags,
)
from backend.siesta.game import Card, GameState, create_deck


SUITS = ["hearts", "diamonds", "clubs", "spades"]


def make_state(columns, stock=None):
    return GameState(columns=columns, stock=stock or [])


def card(rank, suit):
    return Card(rank=rank, suit=suit)


def test_kings_are_always_dead():
    state = make_state([[card(13, "hearts"), card(5, "spades")], [], [], [], [], [], []])
    assert dead_card_positions(state) == {(0, 0)}


def test_six_is_not_dead_while_a_seven_remains_in_the_stock():
    columns = [[card(7, suit), card(6, suit_above)] for suit, suit_above in zip(SUITS[:3], SUITS[1:])]
    columns += [[] for _ in range(7 - len(columns))]
    state = make_state(columns, stock=[card(7, "spades")])
    dead = dead_card_positions(state)
    assert not any(state.columns[c][i].rank == 6 for c, i in dead)


def test_six_is_dead_when_every_seven_carries_a_six():
    # Each seven is covered by a six of a different suit: no six can ever land
    # on a seven, so none of them can move without a hole.
    columns = [
        [card(7, "hearts"), card(6, "diamonds")],
        [card(7, "diamonds"), card(6, "clubs")],
        [card(7, "clubs"), card(6, "spades")],
        [card(7, "spades"), card(6, "hearts")],
        [],
        [],
        [],
    ]
    state = make_state(columns)
    dead = dead_card_positions(state)
    assert {(0, 1), (1, 1), (2, 1), (3, 1)} <= dead


def test_six_resting_on_its_own_seven_is_not_dead():
    columns = [
        [card(7, "hearts"), card(6, "hearts")],
        [card(7, "diamonds"), card(6, "clubs")],
        [card(7, "clubs"), card(6, "spades")],
        [card(7, "spades"), card(6, "diamonds")],
        [],
        [],
        [],
    ]
    state = make_state(columns)
    assert (0, 1) not in dead_card_positions(state)


def test_seven_under_an_unprovable_card_is_not_blocked():
    # The nine on top of the first seven might still be movable onto a ten, so
    # that seven counts as a live destination and no six is dead.
    columns = [
        [card(7, "hearts"), card(9, "hearts")],
        [card(7, "diamonds"), card(6, "clubs")],
        [card(7, "clubs"), card(6, "spades")],
        [card(7, "spades"), card(6, "hearts")],
        [],
        [],
        [],
    ]
    state = make_state(columns)
    dead = dead_card_positions(state)
    assert not any(state.columns[c][i].rank == 6 for c, i in dead)


def test_dead_cards_cascade_downwards():
    # Every seven is blocked by a king, which is dead by definition. That kills
    # the sixes, which in turn blocks every six as a destination for the fives.
    columns = [
        [card(7, suit), card(6, suit), card(13, suit)] for suit in SUITS
    ] + [[card(5, "hearts")], [], []]
    state = make_state(columns)
    dead = dead_card_positions(state)
    assert (4, 0) in dead, "the loose five should be dead"


def test_deadness_never_argues_in_a_circle():
    # Every eight is covered by a six, and those sixes are genuinely dead
    # because all four sevens sit under dead kings. It is tempting to conclude
    # that the eights are blocked too and kill the sevens - but that reasoning
    # uses the sixes' deadness, which was derived from the sevens being stuck.
    # Deadness must only ever rest on strictly higher ranks.
    columns = [
        [card(8, "hearts"), card(6, "clubs")],
        [card(8, "diamonds"), card(6, "spades")],
        [card(8, "clubs"), card(6, "hearts")],
        [card(8, "spades"), card(6, "diamonds")],
        [card(7, "hearts"), card(13, "hearts"), card(7, "diamonds"), card(13, "diamonds")],
        [card(7, "clubs"), card(13, "clubs")],
        [card(7, "spades"), card(13, "spades")],
    ]
    state = make_state(columns)
    dead = dead_card_positions(state)
    ranks = {state.columns[c][i].rank for c, i in dead}
    assert 6 in ranks, "the sixes are dead: every seven sits under a dead king"
    assert 8 not in ranks, "the eights must not be killed by the sixes they block"
    assert 7 not in ranks, "and so the sevens must not be killed either"


def test_not_lost_while_an_empty_column_exists():
    columns = [
        [card(7, "hearts"), card(6, "diamonds")],
        [card(7, "diamonds"), card(6, "clubs")],
        [card(7, "clubs"), card(6, "spades")],
        [card(7, "spades"), card(6, "hearts")],
        [card(13, "hearts")],
        [card(13, "diamonds")],
        [],
    ]
    state = make_state(columns)
    assert not is_provably_lost(state)


def test_lost_when_every_column_holds_a_dead_card():
    columns = [
        [card(7, "hearts"), card(6, "diamonds")],
        [card(7, "diamonds"), card(6, "clubs")],
        [card(7, "clubs"), card(6, "spades")],
        [card(7, "spades"), card(6, "hearts")],
        [card(13, "hearts")],
        [card(13, "diamonds")],
        [card(13, "clubs")],
    ]
    state = make_state(columns)
    assert locked_column_flags(state) == [True] * 7
    assert emptiable_column_count(state) == 0
    assert is_provably_lost(state)


def test_a_won_layout_is_never_reported_as_lost():
    columns = [
        [card(rank, suit) for rank in range(13, 0, -1)] for suit in SUITS
    ] + [[], [], []]
    state = make_state(columns)
    assert not is_provably_lost(state)


def test_full_deck_on_the_table_is_analysable():
    deck = create_deck()
    columns = [deck[index::7] for index in range(7)]
    state = make_state(columns)
    assert len(dead_card_positions(state)) >= 4  # at minimum, the four kings
