"""Which of the player's own deals can be solved, by the player or the engine.

Every engine run on a recorded deal is appended to ``engine_runs.jsonl`` with
the engine version (git commit) and config, so a better engine only needs to
run the deals that still lack a known solution, and versions can be compared
on the same deals. Wins carry the full action list, replayed and checked
before it is saved, so each one is a proof that the deal is solvable.

    python3 -m backend.siesta.solvability run                 # deals with no known solution
    python3 -m backend.siesta.solvability run --all           # every finished deal
    python3 -m backend.siesta.solvability report
"""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .fastgame import SUITS, Tableau, apply_deal, apply_move, is_won

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "backend" / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
RUNS_PATH = DATA_DIR / "engine_runs.jsonl"

Deal = Tuple[Tableau, Tuple[int, ...]]


def _card_id(card: Dict[str, object]) -> int:
    return SUITS.index(card["suit"]) * 13 + int(card["rank"]) - 1


def finished_games(sessions_dir: Path = SESSIONS_DIR) -> Dict[str, Tuple[Deal, bool]]:
    """game_id -> (opening deal, player won) for every game the player played
    to a result (won or conceded). Games with no moves are skipped: they were
    never really played."""
    games: Dict[str, Tuple[Deal, bool]] = {}
    for path in sorted(sessions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("history") or []
        if not history:
            continue
        last = history[-1]
        if last.get("moves_played", 0) == 0 or last.get("status") not in ("won", "conceded"):
            continue
        first = history[0]
        cols = tuple(tuple(_card_id(c) for c in col) for col in first["columns"])
        stock = tuple(_card_id(c) for c in first["stock"])
        games[str(payload.get("game_id", path.stem))] = ((cols, stock), last["status"] == "won")
    return games


def load_runs(path: Path = RUNS_PATH) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def engine_version() -> str:
    """Current git commit, suffixed "-dirty" when engine code has local changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain", "--", "backend/siesta") else "")


def replays_to_win(deal: Deal, actions: Sequence[object]) -> bool:
    cols, stock = deal
    for action in actions:
        if action == "deal":
            cols, stock = apply_deal(cols, stock)
        else:
            cols = apply_move(cols, tuple(action))
    return is_won(cols)


def _play(job: Tuple[str, Deal, int, float]) -> Dict[str, object]:
    from .solver import SolverConfig, play_game

    game_id, deal, seed, endgame_time = job
    result = play_game(seed, SolverConfig(endgame_time=endgame_time), deal=deal)
    won = result["status"] == "won"
    record: Dict[str, object] = {"game_id": game_id, "result": "won" if won else "lost",
                                 "seconds": result["seconds"], "seed": seed}
    if won:
        if not replays_to_win(deal, result["actions"]):
            raise RuntimeError(f"{game_id}: engine reported a win its own actions do not replay")
        record["actions"] = result["actions"]
    return record


def run(
    game_ids: Optional[Sequence[str]] = None,
    all_games: bool = False,
    endgame_time: float = 30.0,
    seed_offset: int = 0,
    workers: int = 8,
    sessions_dir: Path = SESSIONS_DIR,
    runs_path: Path = RUNS_PATH,
) -> List[Dict[str, object]]:
    """Run the engine on finished deals and append one record per deal.

    By default only deals without a known solution (neither the player nor any
    earlier engine run won them). The seed per deal is derived from its
    game_id; ``seed_offset`` varies the engine's deal-preview sampling."""
    from concurrent.futures import ProcessPoolExecutor

    games = finished_games(sessions_dir)
    if game_ids is not None:
        targets = [g for g in game_ids if g in games]
    elif all_games:
        targets = list(games)
    else:
        solved = {r["game_id"] for r in load_runs(runs_path) if r["result"] == "won"}
        targets = [g for g, (_deal, player_won) in games.items() if not player_won and g not in solved]

    version = engine_version()
    date = datetime.date.today().isoformat()
    config = {"endgame_time": endgame_time}
    jobs = [(g, games[g][0], (int(g, 16) + seed_offset) % 2**31, endgame_time) for g in targets]
    records: List[Dict[str, object]] = []
    wins = 0
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, runs_path.open("a", encoding="utf-8") as out:
        for n, record in enumerate(pool.map(_play, jobs), 1):
            record = {**record, "date": date, "engine": version, "config": config}
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
            wins += record["result"] == "won"
            print(f"{n}/{len(jobs)} {record['game_id']} {record['result']} {record['seconds']:.0f}s wins={wins}",
                  flush=True)
    return records


def report(sessions_dir: Path = SESSIONS_DIR, runs_path: Path = RUNS_PATH) -> str:
    games = finished_games(sessions_dir)
    runs = [r for r in load_runs(runs_path) if r["game_id"] in games]
    player = {g for g, (_deal, won) in games.items() if won}
    engine = {r["game_id"] for r in runs if r["result"] == "won"}
    total = len(games)

    def pct(n: int) -> str:
        return f"{n:4d}  {n / total:6.1%}" if total else f"{n:4d}"

    lines = [
        f"Finished deals:            {total}",
        f"Player won:                {pct(len(player))}",
        f"Engine won (any run):      {pct(len(engine))}",
        f"  both                     {pct(len(player & engine))}",
        f"  only player              {pct(len(player - engine))}",
        f"  only engine              {pct(len(engine - player))}",
        f"Known solvable:            {pct(len(player | engine))}",
        f"No known solution:         {pct(total - len(player | engine))}",
        "",
        "Engine runs per version and config (deals run, won):",
    ]
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in runs:
        groups.setdefault(f"{r['engine']}  {json.dumps(r['config'], sort_keys=True)}", []).append(r)
    for key, rows in groups.items():
        dates = sorted({r["date"] for r in rows})
        won = sum(r["result"] == "won" for r in rows)
        lines.append(f"  {key}  {dates[0]}..{dates[-1]}  {won}/{len(rows)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="solvability", description="Track which played deals can be solved.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the engine on finished deals")
    p_run.add_argument("--all", action="store_true", help="every finished deal, not only those without a solution")
    p_run.add_argument("--game-ids", nargs="+")
    p_run.add_argument("--endgame-time", type=float, default=30.0)
    p_run.add_argument("--seed-offset", type=int, default=0)
    p_run.add_argument("--workers", type=int, default=8)
    sub.add_parser("report", help="summarise player and engine results")
    args = parser.parse_args()

    if args.command == "run":
        run(args.game_ids, args.all, args.endgame_time, args.seed_offset, args.workers)
    print(report())
