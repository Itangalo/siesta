import json

from backend.siesta import solvability
from backend.siesta.fastgame import SUITS


def suit_stack(suit, top=13, bottom=1):
    return tuple(suit * 13 + rank - 1 for rank in range(top, bottom - 1, -1))


def card(cid):
    return {"rank": cid % 13 + 1, "suit": SUITS[cid // 13]}


def write_session(sessions_dir, game_id, cols, stock, status, moves_played=5):
    first = {"columns": [[card(c) for c in col] for col in cols], "stock": [card(c) for c in stock],
             "status": "in_progress", "moves_played": 0}
    last = {**first, "status": status, "moves_played": moves_played}
    (sessions_dir / f"{game_id}.json").write_text(json.dumps({"game_id": game_id, "history": [first, last]}))


# One card short of won: the ace of spades is still in the stock.
NEAR_WON = ((suit_stack(0), suit_stack(1), suit_stack(2), suit_stack(3, 13, 2), (), (), ()), (3 * 13,))


def test_finished_games_skips_unplayed_and_unfinished(tmp_path):
    cols, stock = NEAR_WON
    write_session(tmp_path, "aaaaaaaaaaaa", cols, stock, "won")
    write_session(tmp_path, "bbbbbbbbbbbb", cols, stock, "conceded")
    write_session(tmp_path, "cccccccccccc", cols, stock, "in_progress")
    write_session(tmp_path, "dddddddddddd", cols, stock, "conceded", moves_played=0)
    games = solvability.finished_games(tmp_path)
    assert set(games) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert games["aaaaaaaaaaaa"] == (NEAR_WON, True)
    assert games["bbbbbbbbbbbb"][1] is False


def test_run_records_verified_wins_and_skips_known_solutions(tmp_path):
    sessions, runs = tmp_path / "sessions", tmp_path / "runs.jsonl"
    sessions.mkdir()
    cols, stock = NEAR_WON
    write_session(sessions, "aaaaaaaaaaaa", cols, stock, "won")
    write_session(sessions, "bbbbbbbbbbbb", cols, stock, "conceded")

    records = solvability.run(endgame_time=5, workers=1, sessions_dir=sessions, runs_path=runs)
    assert [r["game_id"] for r in records] == ["bbbbbbbbbbbb"]  # the player already solved the other
    assert records[0]["result"] == "won"
    assert solvability.replays_to_win(NEAR_WON, records[0]["actions"])
    assert solvability.load_runs(runs) == records

    assert solvability.run(endgame_time=5, workers=1, sessions_dir=sessions, runs_path=runs) == []
    assert "Known solvable:               2  100.0%" in solvability.report(sessions, runs)
