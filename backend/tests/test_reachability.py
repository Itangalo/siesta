from backend.siesta.game import Card, GameState, create_game
from backend.siesta.reachability import cards_movable_within


def make_state(columns, stock=None):
    return GameState(columns=columns, stock=stock or [])


def card(rank, suit):
    return Card(rank=rank, suit=suit)


def test_depth_one_matches_cards_with_a_legal_move():
    columns = [[card(7, "spades")], [card(6, "hearts")], [card(2, "clubs")], [], [], [], []]
    found = cards_movable_within(make_state(columns), max_depth=1)
    # every top card can reach the empty column, so all three move at depth 1
    assert found == {"7S": 1, "6H": 1, "2C": 1}


def test_a_buried_card_needs_the_card_above_it_to_move_first():
    columns = [
        [card(5, "hearts"), card(9, "clubs")],
        [card(10, "spades")],
        [card(6, "diamonds")],
        [card(2, "clubs")],
        [card(4, "spades")],
        [card(8, "hearts")],
        [card(12, "clubs")],
    ]
    state = make_state(columns)
    # No holes, so the five is stuck under the nine. Only once the nine has
    # gone onto the ten does the five reach the six.
    shallow = cards_movable_within(state, max_depth=1)
    assert shallow["9C"] == 1
    assert "5H" not in shallow
    assert cards_movable_within(state, max_depth=2)["5H"] == 2


def test_depth_is_the_shortest_sequence_found():
    state = create_game(seed=5)
    shallow = cards_movable_within(state, max_depth=1)
    deep = cards_movable_within(state, max_depth=3)
    for code, depth in shallow.items():
        assert deep[code] == depth, "a deeper search must not report a longer route"


def test_deeper_search_never_finds_fewer_cards():
    state = create_game(seed=11)
    previous = set()
    for depth in (1, 2, 3):
        found = set(cards_movable_within(state, max_depth=depth))
        assert previous <= found
        previous = found


def test_no_moves_means_nothing_is_reachable():
    # Seven occupied columns, so no holes, and no top card has a rank+1
    # anywhere on top - nothing can move at any depth.
    columns = [
        [card(2, "spades")],
        [card(2, "hearts")],
        [card(5, "spades")],
        [card(5, "hearts")],
        [card(8, "spades")],
        [card(8, "hearts")],
        [card(11, "spades")],
    ]
    assert cards_movable_within(make_state(columns), max_depth=3) == {}
