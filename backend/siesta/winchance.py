"""Estimated chance that the search engine wins from a position.

Mid-game the future deals are unknown, so the estimate is a probability: the
engine's plan (best tableau reachable before the next deal) is dealt with
fair previews drawn from the unseen multiset, and a per-stage logistic model
scores each dealt tableau. Once the stock is empty the answer is exact when
the A* endgame solver settles it within budget: proven win or proven loss.

The models are trained on the engine's own games (position -> did the engine
win), so the number means "this engine's chance", not a perfect player's.
Retrain after changing the engine (the features use its scorer weights):
``python3 -m backend.siesta.winchance collect 2000 3000 states.pkl`` then
``python3 -m backend.siesta.winchance train states.pkl``.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .fastgame import RANK, Tableau, apply_deal, dead_columns, is_won, legal_moves
from .solver import (
    SolverConfig,
    astar_endgame,
    choose_handoff_target,
    choose_segment_target,
    sample_stock_orders,
    segment_search_beam,
    _winnability_features,
)

WEIGHTS_PATH = Path(__file__).with_name("winchance_weights.json")
_SCORER = SolverConfig().make_scorer()
_MODELS: Optional[Dict[int, Dict[str, object]]] = None


def stage_of(cols: Tableau) -> int:
    """Deals completed so far, from the number of visible cards (28 + 7k,
    capped at 4 once all 52 are out)."""
    cards = sum(len(col) for col in cols)
    return min(4, (cards - 28 + 6) // 7)


def features(cols: Tableau) -> List[float]:
    visible = {card for col in cols for card in col}
    stock_counts = Counter(RANK[card] for card in range(52) if card not in visible)
    stock_size = 52 - len(visible)
    pairs = 0.0
    one_unit = two_units = 0
    ladder = full = 0
    for col in cols:
        if not col:
            continue
        col_pairs, segments = _SCORER.column_shape(col)
        pairs += col_pairs
        one_unit += segments == 1
        two_units += segments == 2
        length = _SCORER.king_ladder_length(col)
        ladder += length
        full += length == 13
    dead = sum(dead_columns(cols, dict(stock_counts)))
    holes = sum(1 for col in cols if not col)
    deal = min(7, stock_size)
    return list(_winnability_features(cols)) + [
        pairs, float(one_unit), float(two_units), float(ladder), float(full),
        float(dead), float(holes), float(len(legal_moves(cols))),
        _SCORER.score(cols, deal_size=deal) / 100.0,
    ]


def _models() -> Dict[int, Dict[str, object]]:
    global _MODELS
    if _MODELS is None:
        raw = json.loads(WEIGHTS_PATH.read_text())
        _MODELS = {int(stage): model for stage, model in raw["stages"].items()}
    return _MODELS


def predict(cols: Tableau) -> float:
    """Probability that the engine wins from ``cols`` (a post-deal position)."""
    model = _models()[stage_of(cols)]
    z = model["bias"]
    for x, mu, sd, w in zip(features(cols), model["mean"], model["std"], model["weights"]):
        z += w * (x - mu) / sd
    scale, shift = model.get("calibration", (1.0, 0.0))
    z = scale * z + shift
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def estimate(
    cols: Tableau,
    stock: Tuple[int, ...],
    time_budget: float = 1.5,
    segment_budget: int = 30000,
    samples: int = 24,
    seed: int = 0,
) -> Dict[str, object]:
    """Win-chance report for a position.

    ``mode`` is "won", "proven_win", "proven_loss", "estimate" or "unknown"
    (endgame not settled in time; ``percent`` then carries the model's guess).
    """
    if is_won(cols):
        return {"mode": "won", "percent": 100.0}
    if not stock:
        empty = {rank: 0 for rank in range(1, 14)}
        if all(dead_columns(cols, empty)):
            return {"mode": "proven_loss", "percent": 0.0}
        path, proven_lost = astar_endgame(cols, time_budget=time_budget)
        if path is not None:
            return {"mode": "proven_win", "percent": 100.0, "line_length": len(path)}
        if proven_lost:
            return {"mode": "proven_loss", "percent": 0.0}
        return {"mode": "unknown", "percent": round(100.0 * predict(cols), 1)}

    config = SolverConfig()
    scorer = config.make_scorer()
    stock_rank_counts = dict(Counter(RANK[card] for card in stock))
    seen, _mobility = segment_search_beam(
        cols,
        scorer,
        budget=segment_budget,
        bfs_depth=config.segment_beam_bfs_depth,
        max_depth=config.segment_beam_max_depth,
        beam_width=config.segment_beam_width,
        stock_rank_counts=stock_rank_counts,
        lock_penalty=config.lock_penalty,
    )
    deal_size = min(7, len(stock))
    rng = random.Random(seed)
    if len(stock) <= 7 and config.handoff_predict:
        target = choose_handoff_target(
            seen, scorer, stock, deal_size=deal_size, lock_penalty=config.lock_penalty,
            samples=6, candidates=60, seed_base=seed,
            survive_weight=config.handoff_survive_weight,
        )
    else:
        previews = sample_stock_orders(stock, deal_size, config.deal_samples, rng)
        target, _score = choose_segment_target(
            seen, scorer, deal_size=deal_size, stock_rank_counts=stock_rank_counts,
            lock_penalty=config.lock_penalty, stock_sample=previews,
            w_post_deal_mobility=config.w_post_deal_mobility,
        )
    plan = target if target is not None else cols
    total = 0.0
    for order in sample_stock_orders(stock, deal_size, samples, rng):
        dealt, _rest = apply_deal(plan, order)
        total += predict(dealt)
    return {"mode": "estimate", "percent": round(100.0 * total / samples, 1)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_models(records: Sequence[Tuple[int, int, Tableau, bool]], verbose: bool = True) -> Dict[str, object]:
    """Fit one L2 logistic regression per stage from (seed, deals_used, cols,
    won) records. The L2 strength is chosen on held-out games (every 4th
    seed) by log-loss, then the model is refit on all games.

    Raw logistic regression is overconfident at the top here (a predicted
    60% won about 45%), so each stage also gets Platt scaling: a 1-D
    logistic fit of the outcome on out-of-fold logits (4 folds by seed),
    applied on top of the final model's logit."""
    import numpy as np

    def fit(X, y, l2, iters=4000, lr=0.2):
        w = np.zeros(X.shape[1])
        rate = min(max(y.mean(), 1e-3), 1 - 1e-3)
        b = math.log(rate / (1 - rate))
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
            w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y))
            b -= lr * float((p - y).mean())
        return w, b

    def log_loss(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    out: Dict[str, object] = {"stages": {}, "games": len({r[0] for r in records})}
    for stage in range(5):
        rows = [r for r in records if r[1] == stage]
        X = np.array([features(r[2]) for r in rows], dtype=float)
        y = np.array([1.0 if r[3] else 0.0 for r in rows])
        held = np.array([r[0] % 4 == 0 for r in rows])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        best = None
        for l2 in (1.0, 10.0, 100.0, 1000.0):
            w, b = fit(Xs[~held], y[~held], l2)
            loss = log_loss(1.0 / (1.0 + np.exp(-(Xs[held] @ w + b))), y[held])
            if best is None or loss < best[0]:
                best = (loss, l2)
        folds = np.array([r[0] % 4 for r in rows])
        oof = np.zeros(len(y))
        for k in range(4):
            wk, bk = fit(Xs[folds != k], y[folds != k], best[1])
            oof[folds == k] = Xs[folds == k] @ wk + bk
        scale, shift = fit(oof.reshape(-1, 1), y, 1e-6)
        calibration = (float(scale[0]), float(shift))
        calibrated_loss = log_loss(1.0 / (1.0 + np.exp(-(calibration[0] * oof + calibration[1]))), y)
        w, b = fit(Xs, y, best[1])
        if verbose:
            print(f"stage {stage}: n={len(y)} win rate {y.mean():.1%} held-out log-loss {best[0]:.3f} "
                  f"(l2={best[1]:g}), calibrated out-of-fold {calibrated_loss:.3f} "
                  f"(scale {calibration[0]:.2f}, shift {calibration[1]:+.2f})")
        out["stages"][str(stage)] = {
            "mean": mu.tolist(), "std": sd.tolist(), "weights": w.tolist(), "bias": b, "l2": best[1],
            "calibration": list(calibration),
        }
    return out


def _record_game(seed: int) -> List[Tuple[int, int, Tableau, bool]]:
    from .fastgame import fast_deck_shuffle
    from .solver import play_game

    result = play_game(seed, SolverConfig(record_states=True))
    opening, _stock = fast_deck_shuffle(seed)
    states = [(0, opening)] + list(result["states"] or [])
    won = result["status"] == "won"
    return [(seed, deals, cols, won) for deals, cols in states]


def collect(start: int, games: int, workers: int = 8) -> List[Tuple[int, int, Tableau, bool]]:
    """Play ``games`` deals (seeds start..) with the default engine and return
    (seed, deals_used, cols, won) records: the opening and every post-deal
    tableau. The unseen stock follows from the visible cards."""
    from concurrent.futures import ProcessPoolExecutor

    records: List[Tuple[int, int, Tableau, bool]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(_record_game, range(start, start + games)):
            records.extend(rows)
    return records


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(prog="winchance", description="Collect games and train the win-chance model.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_collect = sub.add_parser("collect", help="play games and save (state, outcome) records")
    p_collect.add_argument("start", type=int)
    p_collect.add_argument("games", type=int)
    p_collect.add_argument("out")
    p_collect.add_argument("--workers", type=int, default=8)
    p_train = sub.add_parser("train", help=f"fit models and write {WEIGHTS_PATH.name}")
    p_train.add_argument("records", nargs="+")
    args = parser.parse_args()

    if args.command == "collect":
        rows = collect(args.start, args.games, args.workers)
        pickle.dump(rows, open(args.out, "wb"))
        print(f"saved {len(rows)} states from {args.games} games to {args.out}")
    else:
        records: List[Tuple[int, int, Tableau, bool]] = []
        for path in args.records:
            records.extend(pickle.load(open(path, "rb")))
        models = fit_models(records)
        WEIGHTS_PATH.write_text(json.dumps(models))
        print(f"wrote {WEIGHTS_PATH} from {models['games']} games")
