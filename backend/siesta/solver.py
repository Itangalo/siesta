"""Search-based Siesta player.

Strategy: play the game as a sequence of *segments*, each ending with exactly
one stock deal. Within a segment, enumerate the tableaus reachable through
legal moves (bounded BFS with a transposition table), score them with a
heuristic, and steer the game toward the best reachable tableau before dealing.
When the stock is empty, switch to an exact bounded search looking for a win.

Fairness: the solver never inspects the *order* of future stock cards. It only
uses the multiset of unseen ranks (which any careful human can track) for
deadlock analysis.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .fastgame import (
    RANK,
    SUIT,
    MoveT,
    Tableau,
    _is_complete_chunk,
    apply_deal,
    apply_move,
    count_completed_sequences,
    dead_columns,
    fast_deck_shuffle,
    is_provably_lost,
    is_won,
    legal_moves,
    path_to,
    top_run as fg_top_run,
)

TableauScored = Tuple[float, Tableau]


class TableauScorer:
    """Heuristic tableau evaluation. Column-level features are cached.

    Core idea: the growing edge of a suit chain is its *top* card, and deals
    bury tops. Chain value therefore decays exponentially with the number of
    cards stacked above it ("burial"). Stranded tops (no landing spot, no
    hole) risk permanent lock-up after the next deal.
    """

    def __init__(
        self,
        w_completed: float = 4000.0,
        w_hole_base: float = 26.0,
        w_hole_extra: float = 34.0,
        w_run_sq: float = 3.0,
        w_king_run: float = 14.0,
        w_ace_run: float = 6.0,
        w_top_covered: float = -4.0,
        burial_decay: float = 0.82,
        w_stranded: float = -14.0,
        w_max_chain: float = 0.0,
    ) -> None:
        self._col_cache: Dict[Tuple[int, ...], Tuple[float, int, int]] = {}
        self._top_run_cache: Dict[Tuple[int, ...], int] = {}
        self._completed_cache: Dict[Tuple[int, ...], int] = {}
        self._maxchain_cache: Dict[Tableau, Tuple[int, int, int, int]] = {}
        self._score_memo: Dict[Tuple[int, Tableau], float] = {}
        self.w_completed = w_completed
        self.w_hole_base = w_hole_base
        self.w_hole_extra = w_hole_extra
        self.w_run_sq = w_run_sq
        self.w_king_run = w_king_run
        self.w_ace_run = w_ace_run
        self.w_top_covered = w_top_covered
        self.burial_decay = burial_decay
        self.w_stranded = w_stranded
        self.w_max_chain = w_max_chain

    def max_chains(self, cols: Tableau) -> Tuple[int, int, int, int]:
        cached = self._maxchain_cache.get(cols)
        if cached is not None:
            return cached
        best = [0, 0, 0, 0]
        for col in cols:
            i = 0
            n = len(col)
            while i < n:
                j = i
                while j + 1 < n and SUIT[col[j + 1]] == SUIT[col[j]] and RANK[col[j]] == RANK[col[j + 1]] + 1:
                    j += 1
                s = SUIT[col[i]]
                length = j - i + 1
                if length > best[s]:
                    best[s] = length
                i = j + 1
        result = (best[0], best[1], best[2], best[3])
        self._maxchain_cache[cols] = result
        return result

    def column_features(self, col: Tuple[int, ...]) -> Tuple[float, int, int]:
        """(discounted run energy, discounted king-run length, ace-run length)."""
        cached = self._col_cache.get(col)
        if cached is not None:
            return cached
        n = len(col)
        gamma = self.burial_decay
        # Precompute burial counts: for each index i, number of cards above i.
        run_energy = 0.0
        king_run = 0.0
        ace_run = 0.0
        i = 0
        while i < n:
            j = i
            while j + 1 < n and SUIT[col[j + 1]] == SUIT[col[j]] and RANK[col[j]] == RANK[col[j + 1]] + 1:
                j += 1
            length = j - i + 1
            burial_above_top = n - 1 - j
            decay = gamma ** burial_above_top
            run_energy += length * length * decay
            if RANK[col[i]] == 13:
                king_run += length * decay
            if RANK[col[j]] == 1:
                ace_run += length
            i = j + 1
        feats = (run_energy, king_run, ace_run)
        self._col_cache[col] = feats
        return feats

    def column_top_run(self, col: Tuple[int, ...]) -> int:
        cached = self._top_run_cache.get(col)
        if cached is not None:
            return cached
        value = fg_top_run(col)
        self._top_run_cache[col] = value
        return value

    def column_completed(self, col: Tuple[int, ...]) -> int:
        cached = self._completed_cache.get(col)
        if cached is not None:
            return cached
        i = 0
        n = len(col)
        completed = 0
        while i <= n - 13:
            if _is_complete_chunk(col[i : i + 13]):
                completed += 1
                i += 13
            else:
                i += 1
        self._completed_cache[col] = completed
        return completed

    def score(self, cols: Tableau, deal_size: int = 7) -> float:
        memo_key = (deal_size, cols)
        cached = self._score_memo.get(memo_key)
        if cached is not None:
            return cached
        value = self._score_impl(cols, deal_size)
        if len(self._score_memo) < 4_000_000:
            self._score_memo[memo_key] = value
        return value

    def _score_impl(self, cols: Tableau, deal_size: int = 7) -> float:
        value = 0.0
        completed_total = 0
        holes = 0
        tops: List[int] = []
        for col in cols:
            if not col:
                holes += 1
                continue
            completed_total += self.column_completed(col)
            run_energy, king_run, ace_run = self.column_features(col)
            value += self.w_run_sq * run_energy
            value += self.w_king_run * king_run
            value += self.w_ace_run * ace_run
            tops.append(RANK[col[-1]])
        value += self.w_completed * completed_total
        if self.w_max_chain:
            chains = self.max_chains(cols)
            value += self.w_max_chain * sum(c * c for c in chains)
        value += self.w_hole_base * holes
        if holes >= 2:
            value += self.w_hole_extra * (holes - 1)
        # Dealing covers the first ``deal_size`` columns; long same-suit runs
        # sitting on those tops lose their movability, discount them slightly.
        if deal_size:
            covered = 0.0
            for ci in range(min(deal_size, 7)):
                col = cols[ci]
                if col:
                    run = self.column_top_run(col)
                    if run > 1:
                        covered += self.w_top_covered * run
            value += covered
        if self.w_stranded and holes == 0:
            rank_counts: Dict[int, int] = {}
            for rank in tops:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            stranded = sum(1 for rank in tops if rank_counts.get(rank + 1, 0) == 0)
            value += self.w_stranded * stranded
        return value

    def delta(self, cols: Tableau, move: MoveT, deal_size: int = 7) -> float:
        """Score change of applying ``move``; used for endgame move ordering."""
        return self.score(apply_move(cols, move), deal_size=deal_size) - self.score(cols, deal_size=deal_size)


def segment_search(
    start: Tableau,
    budget: int = 60000,
    max_depth: int = 14,
) -> Tuple[Dict[Tableau, Optional[Tuple[Tableau, MoveT]]], Dict[Tableau, int]]:
    """BFS over move-only reachable tableaus.

    Returns (parent map, mobility map). BFS order guarantees shortest move
    sequences for reconstruction; the mobility map records how many moves
    were generated from the state each child was discovered from.
    """
    seen: Dict[Tableau, Optional[Tuple[Tableau, MoveT]]] = {start: None}
    mobility: Dict[Tableau, int] = {}
    frontier = [start]
    nodes = 1
    for _depth in range(max_depth):
        if not frontier or nodes >= budget:
            break
        next_frontier: List[Tableau] = []
        for cols in frontier:
            moves = legal_moves(cols)
            mobility[cols] = len(moves)
            for move in moves:
                child = apply_move(cols, move)
                if child not in seen:
                    seen[child] = (cols, move)
                    next_frontier.append(child)
                    nodes += 1
                    if nodes >= budget:
                        return seen, mobility
        frontier = next_frontier
    return seen, mobility


def choose_segment_target(
    seen: Dict[Tableau, Optional[Tuple[Tableau, MoveT]]],
    scorer: TableauScorer,
    deal_size: int,
    stock_rank_counts: Optional[Dict[int, int]] = None,
    lock_penalty: float = 0.0,
    shortlist: int = 250,
    stock_sample: Optional[Sequence[Tuple[int, ...]]] = None,
    mobility_by_state: Optional[Dict[Tableau, int]] = None,
    w_mobility: float = 0.0,
    w_post_deal_mobility: float = 0.0,
    w_winnability: float = 0.0,
    w_model: float = 0.0,
) -> Tuple[Optional[Tableau], float]:
    """Pick the best reachable tableau (start included).

    With ``stock_sample`` (concrete future-stock previews drawn from the
    unseen multiset), candidates are ranked by EXPECTED score after dealing —
    fair, since only the multiset is used.
    """
    counts = stock_rank_counts or {}
    base_scores: List[Tuple[float, int, Tableau]] = []
    for index, cols in enumerate(seen.keys()):
        s = scorer.score(cols, deal_size=deal_size)
        if w_mobility and mobility_by_state is not None:
            m = mobility_by_state.get(cols)
            if m is None:
                m = len(legal_moves(cols))
            s += w_mobility * m
        base_scores.append((s, index, cols))
    if not base_scores:
        return None, float("-inf")

    use_expected = bool(stock_sample)
    if use_expected:
        top = sorted(base_scores, reverse=True)[:400]
        rescored: List[Tuple[float, int, Tableau]] = []
        for _base, index, cols in top:
            total = 0.0
            for sample in stock_sample:
                dealt = list(cols)
                for j in range(len(sample)):
                    dealt[j] = dealt[j] + (sample[j],)
                dealt_t = tuple(dealt)
                total += scorer.score(dealt_t, deal_size=0)
                # Fluidity: a position that stays MOVABLE after the deal keeps
                # future segments productive; frozen towers are dead weight.
                total += w_post_deal_mobility * len(legal_moves(dealt_t))
                if w_winnability:
                    total += w_winnability * predict_winnability_logit(dealt_t)
                if w_model:
                    total += w_model * predict_outcome_logit(dealt_t)
            rescored.append((total / len(stock_sample), index, cols))
        scored = rescored
        scored.sort(reverse=True)
    else:
        scored = sorted(base_scores, reverse=True)

    if lock_penalty > 0 and len(scored) > 1:
        head = scored[:shortlist]
        rescored_lock: List[Tuple[float, int, Tableau]] = []
        for score, index, cols in head:
            holes = sum(1 for col in cols if not col)
            locked = sum(1 for flag in dead_columns(cols, counts) if flag)
            excess_locked = max(0, locked - holes)
            rescored_lock.append((score - lock_penalty * excess_locked, index, cols))
        rescored_lock.sort(reverse=True)
        scored = rescored_lock + scored[shortlist:]
    best_alive = next(
        (entry for entry in scored if not all(dead_columns(entry[2], counts))),
        None,
    )
    if best_alive is None:
        best_alive = scored[0]
    return best_alive[2], best_alive[0]


def sample_stock_orders(
    stock: Tuple[int, ...],
    deal_size: int,
    samples: int,
    rng: "random.Random",
) -> List[Tuple[int, ...]]:
    """Fair deal previews drawn from the unseen multiset (order randomized)."""
    out = []
    for _ in range(samples):
        picks = rng.sample(range(len(stock)), min(deal_size, len(stock)))
        rng.shuffle(picks)
        out.append(tuple(stock[i] for i in picks))
    return out


def endgame_win_search(
    start: Tableau,
    budget: int = 400000,
    max_depth: int = 40,
) -> Optional[List[MoveT]]:
    """Bounded BFS hunting strictly for a winning tableau."""
    seen: Dict[Tableau, Optional[Tuple[Tableau, MoveT]]] = {start: None}
    frontier = [start]
    nodes = 1
    for _depth in range(max_depth):
        if not frontier or nodes >= budget:
            break
        next_frontier: List[Tableau] = []
        for cols in frontier:
            for move in legal_moves(cols):
                child = apply_move(cols, move)
                if child in seen:
                    continue
                seen[child] = (cols, move)
                if is_won(child):
                    return path_to(seen, child)
                next_frontier.append(child)
                nodes += 1
                if nodes >= budget:
                    return None
        frontier = next_frontier
    return None


class _SearchTimeout(Exception):
    pass


def endgame_solve(
    start: Tableau,
    scorer: Optional[TableauScorer] = None,
    time_budget: float = 60.0,
    node_budget: int = 3000000,
    tt_capacity: int = 1500000,
    start_depth: int = 2,
    max_depth: int = 400,
    depth_step: int = 1,
    deadlock_prune_depth: int = 10,
    verbose: bool = False,
) -> Optional[List[MoveT]]:
    """Depth-first iterative-deepening hunt for a winning sequence.

    Transposition entries record the largest remaining depth at which a state
    was already refuted, so deeper iterations reuse earlier proofs. Returns
    the winning move list, or None when budgets/depth are exhausted.
    """
    import hashlib

    scorer = scorer or TableauScorer()
    deadline = time.time() + time_budget
    stats = {"nodes": 0}
    refute: Dict[int, int] = {}
    best_move: Dict[int, MoveT] = {}
    killers: Dict[int, List[MoveT]] = {}

    def state_key(cols: Tableau) -> bytes:
        flat = bytearray()
        for col in cols:
            flat.extend(col)
        return hashlib.blake2b(bytes(flat), digest_size=12).digest()

    start_key = state_key(start)

    def dfs(
        cols: Tableau,
        depth_left: int,
        key: bytes,
        grandparent_key: Optional[bytes],
    ) -> Optional[List[MoveT]]:
        if is_won(cols):
            return []
        refuted_at = refute.get(key, -1)
        if depth_left <= refuted_at:
            return None
        moves = legal_moves(cols)
        if not moves:
            refute[key] = 10 ** 9
            return None
        pv = best_move.get(key)
        deltas: Dict[MoveT, float] = {}
        node_killers = killers.get(depth_left, [])

        def order(m: MoveT) -> Tuple[int, float]:
            if m == pv:
                return (0, 0.0)
            try:
                krank = 1 + node_killers.index(m)
            except ValueError:
                krank = 9
            return (krank, -deltas.setdefault(m, scorer.delta(cols, m, deal_size=0)))

        # Prune pure reversals of the last move (returning to the grandparent):
        # any line through A->B->A is dominated by staying at A.
        candidates = []
        for move in moves:
            child = apply_move(cols, move)
            child_key = state_key(child)
            if grandparent_key is not None and child_key == grandparent_key:
                continue
            candidates.append((move, child, child_key))
        if not candidates:
            refute[key] = max(refute.get(key, -1), depth_left)
            return None
        candidates.sort(key=lambda item: order(item[0]))
        cheapest_count = None
        cheapest_move = None
        cheapest_child_key = None
        for move, child, child_key in candidates:
            stats["nodes"] += 1
            if stats["nodes"] > node_budget or time.time() > deadline:
                raise _SearchTimeout()
            before = stats["nodes"]
            found = dfs(child, depth_left - 1, child_key, key)
            spent = stats["nodes"] - before
            if found is not None:
                best_move[key] = move
                return [move] + found
            if cheapest_count is None or spent < cheapest_count:
                cheapest_count = spent
                cheapest_move = move
                cheapest_child_key = child_key
            if (
                deadlock_prune_depth > 0
                and depth_left >= deadlock_prune_depth
                and child_key not in refute
                and is_provably_lost(child)
            ):
                refute[child_key] = 10 ** 9
        if depth_left > refuted_at:
            if len(refute) >= tt_capacity:
                raise _SearchTimeout()
            refute[key] = depth_left
        if cheapest_move is not None:
            best_move[key] = cheapest_move
            slot = killers.setdefault(depth_left, [])
            if cheapest_move not in slot:
                slot.insert(0, cheapest_move)
                del slot[2:]
        return None

    depth = start_depth
    while depth <= max_depth:
        try:
            path = dfs(start, depth, start_key, None)
        except _SearchTimeout:
            if verbose:
                print(f"  endgame: timeout at depth={depth} nodes={stats['nodes']}")
            return None
        if path is not None:
            return path
        if verbose:
            print(f"  endgame: depth={depth} refuted nodes={stats['nodes']}")
        depth += depth_step
    return None


@dataclass
class SolverConfig:
    segment_budget: int = 120000
    segment_depth: int = 18
    # "beam" extends the same budget much deeper (BFS first, then
    # score-guided beam); "bfs" is the legacy exhaustive breadth-first walk.
    segment_mode: str = "beam"
    segment_beam_width: int = 2500
    segment_beam_bfs_depth: int = 6
    segment_beam_max_depth: int = 40
    # brute-force overrides are passed via SolverConfig(...) args
    lock_penalty: float = 40.0
    endgame_solver: str = "hybrid"
    endgame_time: float = 45.0
    endgame_nodes: int = 6000000
    endgame_beam_width: int = 750
    endgame_guide_k: float = 20.0
    endgame_attempts: int = 4
    w_hole_base: float = 26.0
    w_hole_extra: float = 34.0
    w_run_sq: float = 3.0
    w_max_chain: float = 0.0
    w_mobility: float = 0.0
    w_post_deal_mobility: float = 6.0
    # NOTE: w_winnability stays off — a model trained on FULL endgame states
    # does not transfer to partial mid-game tableaus (missing cards invisible);
    # steering by it produced confident but false positives (verified: 12/12
    # high-confidence entries unsolved at 7 min each). Use handoff_predict,
    # which applies the predictor only where its distribution matches.
    w_winnability: float = 0.0
    handoff_predict: bool = True
    handoff_samples: int = 14
    handoff_survive_weight: float = 3.0
    w_model: float = 0.0
    deal_samples: int = 10
    shortlist_top: int = 400
    record_states: bool = False
    verbose: bool = False

    def make_scorer(self) -> TableauScorer:
        return TableauScorer(
            w_hole_base=self.w_hole_base,
            w_hole_extra=self.w_hole_extra,
            w_run_sq=self.w_run_sq,
            w_max_chain=self.w_max_chain,
        )


def play_game(
    seed: Optional[int] = None,
    config: Optional[SolverConfig] = None,
) -> Dict[str, object]:
    """Play one full deal deterministically. Returns a result summary."""
    config = config or SolverConfig()
    started = time.time()
    cols, stock = fast_deck_shuffle(seed)
    scorer = config.make_scorer()
    moves_played = 0
    deals_used = 0

    def apply_path(path: Sequence[MoveT]) -> None:
        nonlocal cols, moves_played
        for move in path:
            cols = apply_move(cols, move)
            moves_played += 1

    def finish() -> bool:
        return is_won(cols)

    won = False
    entry_tableau: Optional[Tableau] = None
    recorded_states: List[Tuple[int, Tableau]] = []
    while True:
        if is_won(cols):
            won = True
            break
        if not stock:
            entry_tableau = cols
            if config.endgame_solver == "hybrid":
                path = solve_endgame(
                    cols,
                    scorer=scorer,
                    time_budget=config.endgame_time,
                    node_budget=config.endgame_nodes,
                    beam_width=config.endgame_beam_width,
                    guide_k=config.endgame_guide_k,
                    verbose=config.verbose,
                )
            else:
                path = endgame_solve(
                    cols,
                    scorer=scorer,
                    time_budget=config.endgame_time,
                    node_budget=config.endgame_nodes,
                    verbose=config.verbose,
                )
            if path:
                apply_path(path)
                won = is_won(cols)
            break
        stock_rank_counts = dict(Counter(RANK[c] for c in stock))
        if config.segment_mode == "beam":
            seen, mobility_map = segment_search_beam(
                cols,
                scorer,
                budget=config.segment_budget,
                bfs_depth=config.segment_beam_bfs_depth,
                max_depth=config.segment_beam_max_depth,
                beam_width=config.segment_beam_width,
                stock_rank_counts=stock_rank_counts,
                lock_penalty=config.lock_penalty,
            )
        else:
            seen, mobility_map = segment_search(cols, budget=config.segment_budget, max_depth=config.segment_depth)
        deal_size = min(7, len(stock))
        final_segment = len(stock) <= 7  # this deal empties the stock
        stock_sample = None
        if config.deal_samples > 0:
            rng = random.Random(seed * 31 + deals_used if seed is not None else deals_used)
            stock_sample = sample_stock_orders(stock, deal_size, config.deal_samples, rng)

        if final_segment and config.handoff_predict:
            # Endgame handoff: rank candidates by predicted winnability of the
            # resulting FULL endgame position (the predictor's home turf).
            # Fairness: previews are sampled from the unseen multiset only.
            target = choose_handoff_target(
                seen,
                scorer,
                stock,
                deal_size=deal_size,
                lock_penalty=config.lock_penalty,
                samples=config.handoff_samples,
                seed_base=(seed if seed is not None else deals_used) * 131 + deals_used,
                survive_weight=config.handoff_survive_weight,
            )
            score = float("nan")
        else:
            target, score = choose_segment_target(
                seen,
                scorer,
                deal_size=deal_size,
                stock_rank_counts=stock_rank_counts,
                lock_penalty=config.lock_penalty,
                stock_sample=stock_sample,
                mobility_by_state=mobility_map,
                w_mobility=config.w_mobility,
                w_post_deal_mobility=config.w_post_deal_mobility if stock_sample else 0.0,
                w_winnability=config.w_winnability if stock_sample else 0.0,
            )
        if config.verbose:
            target_holes = sum(1 for col in (target or ())if not col) if target else 0
            print(f"  seg deals_used={deals_used} states={len(seen)} score={score:.0f} "
                  f"target_holes={target_holes} moved={target is not None and target != cols}")
        if target is not None and target != cols:
            apply_path(path_to(seen, target))
            if is_won(cols):
                won = True
                break
        cols, stock = apply_deal(cols, stock)
        deals_used += 1
        if config.record_states:
            recorded_states.append((deals_used, cols))
        if is_won(cols):
            won = True
            break

    elapsed = time.time() - started
    status = "won" if is_won(cols) else "lost"
    result = {
        "seed": seed,
        "status": status,
        "completed_sequences": count_completed_sequences(cols),
        "moves_played": moves_played,
        "deals_used": deals_used,
        "entry": entry_tableau,
        "states": recorded_states if config.record_states else None,
        "seconds": round(elapsed, 3),
    }
    if status == "lost" and entry_tableau is not None:
        # Loss autopsy: was the endgame entry provably dead?
        empty_counts = {r: 0 for r in range(1, 14)}
        result["dead_entry"] = all(dead_columns(entry_tableau, empty_counts))
    if config.verbose:
        locked = dead_columns(cols, {r: 0 for r in range(1, 14)}) if not stock else dead_columns(cols)
        result["locked_columns"] = sum(1 for flag in locked if flag)
        print(result)
    return result


# ---------------------------------------------------------------------------
# Backward (retrograde) endgame search
# ---------------------------------------------------------------------------

def predecessors(cols: Tableau):
    """All states that can reach ``cols`` with exactly one legal move."""
    seen = set()
    for t in range(7):
        col = cols[t]
        n = len(col)
        if not n:
            continue
        max_l = fg_top_run(col)
        for l in range(1, max_l + 1):
            rest_len = n - l
            # landing condition in the predecessor: hole or exact rank above
            if rest_len:
                if RANK[col[rest_len - 1]] != RANK[col[rest_len]] + 1:
                    continue
            else:
                pass  # lifting the whole column leaves a hole to land on
            moved = col[rest_len:]
            ok = True
            for k in range(l - 1):
                a, b = moved[k], moved[k + 1]
                if SUIT[a] != SUIT[b] or RANK[a] != RANK[b] + 1:
                    ok = False
                    break
            if not ok:
                continue
            for f in range(7):
                if f == t:
                    continue
                new_cols = list(cols)
                new_cols[t] = col[:rest_len]
                new_cols[f] = cols[f] + moved
                p = tuple(new_cols)
                if p not in seen:
                    seen.add(p)
                    yield p, (f, t, l)


_WON_STATES: Optional[List[Tableau]] = None


def won_states() -> List[Tableau]:
    global _WON_STATES
    if _WON_STATES is not None:
        return _WON_STATES
    def st(suit):
        return tuple(suit * 13 + r for r in range(12, -1, -1))
    stacks = [st(s) for s in range(4)]

    results: List[Tableau] = []

    def place(remaining: List[Tuple[int, ...]], cols: List[Tuple[int, ...]]) -> None:
        if not remaining:
            results.append(tuple(cols))
            return
        stack = remaining[0]
        rest = remaining[1:]
        for ci in range(7):
            new_cols = list(cols)
            new_cols[ci] = cols[ci] + stack
            place(rest, new_cols)

    place(stacks, [() for _ in range(7)])
    _WON_STATES = results
    return results


def backward_solve(
    entry: Tableau,
    time_budget: float = 30.0,
    node_budget: int = 4_000_000,
    verbose: bool = False,
) -> Optional[List[MoveT]]:
    """Find a winning move sequence from ``entry`` by retrograde BFS from all
    won tableaus. Returns moves in FORWARD order, or None."""
    import hashlib

    started = time.time()
    targets = won_states()
    target_set = set(targets)
    if entry in target_set:
        return []

    def key(cols: Tableau) -> bytes:
        flat = bytearray()
        for col in cols:
            flat.extend(col)
        return hashlib.blake2b(bytes(flat), digest_size=12).digest()

    link: Dict[bytes, Tuple[bytes, MoveT, Tableau]] = {}
    visited = {key(entry)}
    frontier = [(key(entry), entry)]
    nodes = 0
    depth = 0
    while frontier:
        next_frontier = []
        for skey, s in frontier:
            for p, move in predecessors(s):
                pkey = key(p)
                if pkey in visited:
                    continue
                visited.add(pkey)
                link[pkey] = (skey, move, s)
                nodes += 1
                if nodes >= node_budget or time.time() - started > time_budget:
                    if verbose:
                        print(f"  backward: budget out at depth {depth} nodes={nodes}")
                    return None
                if p in target_set:
                    moves: List[MoveT] = []
                    cur = pkey
                    while True:
                        nxt = link.get(cur)
                        if nxt is None:
                            break
                        parent_key, m, parent_state = nxt
                        moves.append(m)
                        cur = parent_key
                        if parent_state in target_set:
                            break
                    return moves
                next_frontier.append((pkey, p))
        frontier = next_frontier
        depth += 1
        if verbose:
            print(f"  backward: depth={depth} frontier={len(frontier)} nodes={nodes}")
    return None


class _HybridFound(Exception):
    def __init__(self, meet_key, meet_cols):
        self.meet_key = meet_key
        self.meet_cols = meet_cols


def hybrid_solve(
    entry: Tableau,
    scorer: Optional[TableauScorer] = None,
    time_budget: float = 120.0,
    node_budget: int = 40_000_000,
    backward_depth: int = 4,
    backward_nodes: int = 3_000_000,
    forward_max_depth: int = 20,
    verbose: bool = False,
) -> Optional[List[MoveT]]:
    """Bidirectional endgame solver with staged deepening.

    Forward IDDFS runs first (cheap wins). If it exhausts, one backward layer
    at a time is flooded from all won tableaus into an oracle set, and forward
    deepening continues probing that set — all transposition data persists.
    Uses Python's fast built-in hashing (64-bit) for in-process tables; the
    tiny collision risk only marginally reduces completeness when hunting
    for a win, which is acceptable here.
    """
    scorer = scorer or TableauScorer()
    started = time.time()
    targets = set(won_states())
    if entry in targets:
        return []

    def key(cols: Tableau) -> int:
        return hash(cols) & 0x7FFFFFFFFFFFFFFF

    # ---- backward oracle state -----------------------------------------
    back_link: Dict[int, Tuple[int, MoveT]] = {}
    visited = {key(entry)}
    back_frontier: List[Tuple[int, Tableau]] = []
    target_digests = set()
    for t in targets:
        tk = key(t)
        target_digests.add(tk)
        if tk not in visited:
            visited.add(tk)
            back_frontier.append((tk, t))
    back_nodes = 0
    back_layer = 0

    def grow_backward() -> None:
        nonlocal back_frontier, back_nodes, back_layer
        if back_layer >= backward_depth or not back_frontier:
            return
        if back_nodes >= backward_nodes:
            return
        if time.time() - started > time_budget * 0.6:
            return
        nxt: List[Tuple[int, Tableau]] = []
        overflow = False
        for skey, s in back_frontier:
            if overflow:
                break
            for p, move in predecessors(s):
                pkey = key(p)
                if pkey in visited:
                    continue
                visited.add(pkey)
                back_link[pkey] = (skey, move)
                nxt.append((pkey, p))
                back_nodes += 1
                if back_nodes >= backward_nodes or time.time() - started > time_budget * 0.6:
                    overflow = True
                    break
        if verbose:
            print(f"  hybrid: +backward layer={back_layer+1} new={len(nxt)} total={back_nodes} "
                  f"({time.time()-started:.0f}s)")
        if not overflow:
            back_frontier = nxt
        back_layer += 1

    # ---- forward search state -------------------------------------------
    refute: Dict[int, int] = {}
    best_move: Dict[int, MoveT] = {}
    killers: Dict[int, List[MoveT]] = {}
    stats = {"nodes": 0}
    start_key = key(entry)
    result: Dict[str, Optional[List[MoveT]]] = {"tail": None}

    def dfs(cols: Tableau, depth_left: int, ckey: bytes, gpkey: Optional[bytes], cols_score: float) -> bool:
        if time.time() - started > time_budget or stats["nodes"] > node_budget:
            raise _SearchTimeout()
        if ckey in target_digests or ckey in back_link:
            # Walk oracle links toward the won state; ancestors prepend their
            # own edges as the success bubbles up, completing the line.
            moves = []
            cur = ckey
            while True:
                nxt = back_link.get(cur)
                if nxt is None:
                    break
                parent_key, mv = nxt
                moves.append(mv)
                cur = parent_key
            result["tail"] = moves
            return True
        refuted_at = refute.get(ckey, -1)
        if depth_left <= refuted_at:
            return False
        moves = legal_moves(cols)
        if not moves:
            refute[ckey] = 10 ** 9
            return False
        pv = best_move.get(ckey)
        node_killers = killers.get(depth_left, [])

        def order(item: Tuple[MoveT, Tableau, bytes, float]) -> Tuple[int, float]:
            m = item[0]
            if m == pv:
                return (0, 0.0)
            try:
                krank = 1 + node_killers.index(m)
            except ValueError:
                krank = 9
            return (krank, -item[3])

        candidates = []
        for move in moves:
            child = apply_move(cols, move)
            chkey = key(child)
            if chkey == gpkey:
                continue
            child_score = scorer.score(child, deal_size=0)
            candidates.append((move, child, chkey, child_score - cols_score))
        if not candidates:
            refute[ckey] = max(refute.get(ckey, -1), depth_left)
            return False
        candidates.sort(key=order)
        cheapest_count = None
        cheapest_move = None
        for move, child, chkey, _delta in candidates:
            stats["nodes"] += 1
            before = stats["nodes"]
            if dfs(child, depth_left - 1, chkey, ckey, _delta + cols_score):
                best_move[ckey] = move
                result["tail"] = [move] + (result["tail"] or [])
                return True
            spent = stats["nodes"] - before
            if cheapest_count is None or spent < cheapest_count:
                cheapest_count = spent
                cheapest_move = move
        if depth_left > refuted_at:
            if len(refute) >= 3_000_000:
                raise _SearchTimeout()
            refute[ckey] = depth_left
        if cheapest_move is not None:
            best_move[ckey] = cheapest_move
            slot = killers.setdefault(depth_left, [])
            if cheapest_move not in slot:
                slot.insert(0, cheapest_move)
                del slot[2:]
        return False

    depth = 1
    while depth <= forward_max_depth:
        try:
            found = dfs(entry, depth, start_key, None, scorer.score(entry, deal_size=0))
        except _SearchTimeout:
            if verbose:
                print(f"  hybrid: timeout depth={depth} nodes={stats['nodes']}")
            return None
        if found and result["tail"] is not None:
            return result["tail"]
        if verbose:
            print(f"  hybrid: fwd depth={depth} refuted nodes={stats['nodes']} "
                  f"back={back_nodes} ({time.time()-started:.0f}s)")
        if depth >= 5 and back_layer < backward_depth:
            grow_backward()
            if back_layer < backward_depth and depth >= 9:
                grow_backward()
        depth += 1
    return None


def endgame_beam(
    entry: Tableau,
    scorer: Optional[TableauScorer] = None,
    time_budget: float = 120.0,
    node_budget: int = 3000000,
    beam_width: int = 750,
    max_layers: int = 150,
    guide_k: float = 20.0,
    verbose: bool = False,
) -> Optional[List[MoveT]]:
    """Layered score-guided beam search over the win-only endgame graph.

    Depth-first IDDFS stalls past ~20 plies while real winning lines run
    15–70 moves (measured on human-won endgames: user lines average rank ~3
    per ply under the scorer, so a beam tracks them where exhaustive
    deepening cannot). Each layer expands every beam state, drops visited
    tableaus, and keeps the top-scoring children. Parent links are stored
    only for surviving beam states, keeping memory flat in the beam size.

    Width 750 solves everything width 3000 does at ~4x fewer nodes (the
    extra states were dead weight); the cheap learned guide below shaves
    another ~10-25% off hard positions. Returns moves in forward order,
    or None when budgets run out.
    """
    import hashlib

    scorer = scorer or TableauScorer()
    started = time.time()
    deadline = started + time_budget
    if is_won(entry):
        return []

    def state_key(cols: Tableau) -> bytes:
        flat = bytearray()
        for col in cols:
            flat.extend(col)
        return hashlib.blake2b(bytes(flat), digest_size=12).digest()

    entry_key = state_key(entry)
    visited = {entry_key}
    parent: Dict[bytes, Tuple[bytes, MoveT]] = {}
    layer = [entry]
    nodes = 0
    for depth in range(max_layers):
        survivors: List[Tuple[float, Tableau, bytes, bytes, MoveT]] = []
        for cols in layer:
            cols_key = state_key(cols)
            moves = legal_moves(cols)
            guide = _guide_values(cols, moves) if guide_k else None
            for move in moves:
                child = apply_move(cols, move)
                nodes += 1
                if (nodes & 0xFFF) == 0 and time.time() > deadline:
                    if verbose:
                        print(f"  beam: timeout at layer={depth} nodes={nodes}")
                    return None
                if is_won(child):
                    path = [move]
                    key = cols_key
                    while key != entry_key:
                        key, tail_move = parent[key]
                        path.append(tail_move)
                    path.reverse()
                    return path
                child_key = state_key(child)
                if child_key in visited:
                    continue
                visited.add(child_key)
                value = scorer.score(child, deal_size=0)
                if guide is not None:
                    value += guide_k * guide[move]
                survivors.append((value, child, child_key, cols_key, move))
                if nodes >= node_budget:
                    if verbose:
                        print(f"  beam: node budget out at layer={depth} nodes={nodes}")
                    return None
        if not survivors:
            return None
        survivors.sort(key=lambda item: item[0], reverse=True)
        layer = []
        for _score, child, child_key, cols_key, move in survivors[:beam_width]:
            parent[child_key] = (cols_key, move)
            layer.append(child)
        if verbose and depth % 10 == 0:
            print(f"  beam: layer={depth} nodes={nodes} best={survivors[0][0]:.0f} "
                  f"({time.time()-started:.0f}s)")
    return None


# ---------------------------------------------------------------------------
# Cheap learned move guide for the endgame beam.
#
# A linear model distilled from the imitation policy's (policy-dagger2)
# action rankings, using only microsecond-fast tableau features. Holdout
# rank correlation ~0.43 across games is modest, but as a beam perturbation
# it cuts ~10-25% of nodes on hard endgames at negligible compute cost.
# Feature order must match _GUIDE_W.
# ---------------------------------------------------------------------------

_GUIDE_W = [
    -61.13597320433561, 2.343145405778673, -6.599503062562002,
    -3.736299705455631, 35.79739496752456, -6.828749479230471,
    2.2877713301624443, -5.458119512743796, -8.319365032961061,
    2.895029860835583, -0.8955713626516265, 3.9413702821085543,
    0.40502284601104455, 3.485592114487812, 10.086466743156135,
    6.561782529948386, -0.3125518140035805, -1.5448786965798944,
    2.2255479056284355, 6.080986889662875, -3.732532074069659,
    -3.3708213648844456, -3.651460398101096, -4.999032391162898,
    -13.733984675682702,
]
_GUIDE_B = 17.74371642849262


def _guide_values(
    cols: Tableau, moves: List[MoveT]
) -> Dict[MoveT, float]:
    """Predicted move preference per legal move. O(1) per move after a
    single O(cards) state scan; pure Python arithmetic, no model calls."""
    lens = [len(c) for c in cols]
    tops = [RANK[c[-1]] if c else 0 for c in cols]
    runs = [fg_top_run(c) for c in cols]
    holes = sum(1 for c in cols if not c)
    rank_counts: Dict[int, int] = {}
    for r in tops:
        if r:
            rank_counts[r] = rank_counts.get(r, 0) + 1
    stranded = sum(1 for r in tops if r and rank_counts.get(r + 1, 0) == 0)
    best_rem: Dict[int, int] = {}
    for (f, _t, ln) in moves:
        rem = lens[f] - ln
        if f not in best_rem or rem < best_rem[f]:
            best_rem[f] = rem
    access = 0.0
    for rem in best_rem.values():
        if rem == 0:
            access += 6.0
        elif rem == 1:
            access += 2.5
        elif rem == 2:
            access += 0.75
    setup = 0.0
    buried = 0
    for ci, c in enumerate(cols):
        if c and runs[ci] == len(c):
            if len(c) == 1:
                setup += 7.0
            elif len(c) == 2:
                setup += 4.0
            elif len(c) == 3:
                setup += 1.5
        suf = runs[ci]
        chunk = c[: len(c) - suf] if suf else c
        for card in chunk:
            if 3 <= RANK[card] <= 11:
                buried += 1
    sum_runs = 0
    for r in runs:
        sum_runs += r
    kings = 0
    distinct = 0
    seen_ranks = set()
    for r in tops:
        if r == 13:
            kings += 1
        if r and r not in seen_ranks:
            seen_ranks.add(r)
            distinct += 1
    spread = (max(lens) - min(lens)) if lens else 0
    completed = count_completed_sequences(cols)
    mob = len(moves)
    out: Dict[MoveT, float] = {}
    for (f, t, ln) in moves:
        lead = cols[f][-ln]
        lead_rank = RANK[lead]
        if cols[t]:
            dest = cols[t][-1]
            dest_rank = RANK[dest]
            same = 1.0 if SUIT[dest] == SUIT[lead] else 0.0
            to_hole = 0.0
            dest_run = runs[t] / 13.0
        else:
            dest_rank = 0
            same = 0.0
            to_hole = 1.0
            dest_run = 0.0
        feats = (
            holes / 7.0, mob / 30.0, sum_runs / 20.0, stranded / 7.0,
            completed / 4.0,
            access / 24.0, setup / 20.0, buried / 20.0, kings / 4.0,
            distinct / 7.0, spread / 30.0,
            ln / 13.0, lead_rank / 13.0, dest_rank / 13.0, same, to_hole,
            1.0 if len(cols[f]) == ln else 0.0,
            rank_counts.get(lead_rank + 1, 0) / 4.0,
            len(cols[f]) / 30.0, len(cols[t]) / 30.0,
            (len(cols[f]) - ln) / 30.0,
            1.0 if lead_rank == 13 else 0.0, 1.0 if lead_rank == 1 else 0.0,
            dest_run, runs[f] / 13.0,
        )
        s = _GUIDE_B
        for w_i, x_i in zip(_GUIDE_W, feats):
            s += w_i * x_i
        out[(f, t, ln)] = s
    return out


def _misplaced(card: int, below: Optional[int]) -> int:
    """1 when ``card`` does not rest where it must in a won tableau: on its
    same-suit successor, or (for a king) at the bottom of a column."""
    if below is None:
        return 0 if RANK[card] == 13 else 1
    return 0 if SUIT[below] == SUIT[card] and RANK[below] == RANK[card] + 1 else 1


def misplaced_count(cols: Tableau) -> int:
    count = 0
    for col in cols:
        below = None
        for card in col:
            count += _misplaced(card, below)
            below = card
    return count


def astar_endgame(
    entry: Tableau,
    weight: float = 2.0,
    time_budget: float = 45.0,
    node_budget: int = 3_000_000,
) -> Tuple[Optional[List[MoveT]], bool]:
    """Weighted A* over the empty-stock endgame. Returns (path, proven_lost).

    Heuristic: the number of misplaced cards. A move changes what exactly one
    card (the lead of the moved run) rests on, so this is an admissible lower
    bound on the moves left, updated in O(1) per move.

    With the stock empty, column order no longer matters, so states are keyed
    by their sorted columns (up to 7! transpositions collapse into one).
    Without pruning beyond that symmetry, exhausting the open list proves the
    position lost; ``proven_lost`` is False on budget exhaustion.
    """
    import heapq

    started = time.time()
    h = misplaced_count(entry)
    if h == 0 and is_won(entry):
        return [], False
    entry_key = hash(tuple(sorted(entry)))
    parent: Dict[int, Optional[Tuple[int, MoveT]]] = {entry_key: None}
    holes = sum(1 for col in entry if not col)
    heap = [(weight * h, -holes, 0, 0, h, holes, entry, entry_key)]
    counter = 0
    expanded = 0
    while heap:
        _f, _tb, g, _c, h, holes, cols, key = heapq.heappop(heap)
        expanded += 1
        if (expanded & 0x3FF) == 0 and (
            time.time() - started > time_budget or len(parent) > node_budget
        ):
            return None, False
        for move in legal_moves(cols):
            frm, to, run_length = move
            source = cols[frm]
            whole = run_length == len(source)
            dest = cols[to]
            if whole and not dest:
                continue  # whole column into a hole: same state up to symmetry
            lead = source[-run_length]
            child_h = (
                h
                - _misplaced(lead, None if whole else source[-run_length - 1])
                + _misplaced(lead, dest[-1] if dest else None)
            )
            child = apply_move(cols, move)
            child_key = hash(tuple(sorted(child)))
            if child_key in parent:
                continue
            parent[child_key] = (key, move)
            if child_h == 0 and is_won(child):
                path = [move]
                cur = key
                while True:
                    link = parent[cur]
                    if link is None:
                        break
                    cur, prev_move = link
                    path.append(prev_move)
                path.reverse()
                return path, False
            child_holes = holes + (1 if whole else 0) - (0 if dest else 1)
            counter += 1
            heapq.heappush(
                heap,
                (g + 1 + weight * child_h, -child_holes, g + 1, counter, child_h, child_holes, child, child_key),
            )
    return None, True


def solve_endgame_status(
    entry: Tableau,
    time_budget: float = 45.0,
    node_budget: int = 3_000_000,
) -> Tuple[Optional[List[MoveT]], bool]:
    """(winning line or None, proven_lost). Weighted A* with the misplaced-card
    bound finds lines the beam/IDDFS combo misses and, on exhaustion, proves
    the loss (measured: 13 of 58 endgames the old driver gave up on were
    wins, 39 were provably lost)."""
    return astar_endgame(entry, weight=2.0, time_budget=time_budget, node_budget=node_budget)


def solve_endgame(
    entry: Tableau,
    scorer: Optional[TableauScorer] = None,
    time_budget: float = 45.0,
    node_budget: int = 6000000,
    beam_width: int = 750,
    guide_k: float = 20.0,
    verbose: bool = False,
) -> Optional[List[MoveT]]:
    """Endgame driver: weighted A* first (exact, and proves losses), then the
    legacy hybrid/beam pair with whatever time is left.
    A found line is replayed by callers and verified with ``is_won``.
    """
    scorer = scorer or TableauScorer()
    started = time.time()
    path, proven_lost = astar_endgame(entry, time_budget=time_budget * 0.8)
    if path is not None:
        return path
    if proven_lost:
        return None
    time_budget -= time.time() - started
    if time_budget < 1.0:
        return None
    started = time.time()
    quick_time = min(8.0, time_budget * 0.25)
    quick_nodes = min(1500000, node_budget // 4)
    path = hybrid_solve(
        entry,
        scorer=scorer,
        time_budget=max(0.5, quick_time),
        node_budget=max(100000, quick_nodes),
        verbose=verbose,
    )
    if path:
        return path
    remaining = time_budget - (time.time() - started)
    if remaining < 1.0:
        return None
    return endgame_beam(
        entry,
        scorer=scorer,
        time_budget=remaining,
        node_budget=max(500000, node_budget - quick_nodes),
        beam_width=beam_width,
        guide_k=guide_k,
        verbose=verbose,
    )


EG_SCORER = TableauScorer(
    w_completed=40000.0,
    w_hole_base=60.0,
    w_hole_extra=100.0,
    w_run_sq=6.0,
    w_king_run=20.0,
    w_ace_run=8.0,
    w_top_covered=0.0,
    burial_decay=0.985,
    w_stranded=-25.0,
)


def _forward_probe_worker(payload):
    """Runs in a process pool: IDDFS over a slice of root moves, probing an
    oracle digest set. Returns (prefix_moves, meet_state) on success."""
    (
        root_moves,
        entry,
        oracle,
        time_budget,
        node_budget,
        forward_max_depth,
    ) = payload
    import hashlib
    started = time.time()

    def key(cols):
        flat = bytearray()
        for col in cols:
            flat.extend(col)
        return hashlib.blake2b(bytes(flat), digest_size=12).digest()

    scorer = EG_SCORER
    refute: Dict[int, int] = {}
    best_move: Dict[int, MoveT] = {}
    killers: Dict[int, List[MoveT]] = {}
    stats = {"nodes": 0}
    result: Dict[str, object] = {"tail": None}

    class _Found(Exception):
        pass

    def dfs(cols, depth_left, ckey, gpkey, cols_score, path):
        if time.time() - started > time_budget or stats["nodes"] > node_budget:
            raise _SearchTimeout()
        if ckey in oracle:
            raise _Found()
        refuted_at = refute.get(ckey, -1)
        if depth_left <= refuted_at:
            return False
        moves = legal_moves(cols)
        if not moves:
            refute[ckey] = 10 ** 9
            return False
        pv = best_move.get(ckey)
        node_killers = killers.get(depth_left, [])

        def order(item):
            m = item[0]
            if m == pv:
                return (0, 0.0)
            try:
                krank = 1 + node_killers.index(m)
            except ValueError:
                krank = 9
            return (krank, -item[3])

        candidates = []
        for move in moves:
            child = apply_move(cols, move)
            chkey = key(child)
            if chkey == gpkey:
                continue
            child_score = scorer.score(child, deal_size=0)
            candidates.append((move, child, chkey, child_score - cols_score))
        if not candidates:
            refute[ckey] = max(refute.get(ckey, -1), depth_left)
            return False
        candidates.sort(key=order)
        cheapest_count = None
        cheapest_move = None
        for move, child, chkey, delta in candidates:
            stats["nodes"] += 1
            before = stats["nodes"]
            path.append(move)
            try:
                if dfs(child, depth_left - 1, chkey, ckey, delta + cols_score, path):
                    best_move[ckey] = move
                    return True
            finally:
                path.pop()
            spent = stats["nodes"] - before
            if cheapest_count is None or spent < cheapest_count:
                cheapest_count = spent
                cheapest_move = move
        if depth_left > refuted_at:
            refute[ckey] = depth_left
        if cheapest_move is not None:
            best_move[ckey] = cheapest_move
            slot = killers.setdefault(depth_left, [])
            if cheapest_move not in slot:
                slot.insert(0, cheapest_move)
                del slot[2:]
        return False

    for root_move in root_moves:
        first = apply_move(entry, root_move)
        fkey = key(first)
        path = [root_move]
        try:
            if fkey in oracle:
                return ([root_move], first)
            depth = 1
            while depth <= forward_max_depth:
                if dfs(first, depth, fkey, None, scorer.score(first, deal_size=0), path):
                    return (list(path), None)
                depth += 1
        except _Found:
            return (list(path), None)
        except _SearchTimeout:
            return None
    return None


def solve_endgame_parallel(
    entry: Tableau,
    pool,
    workers: int = 8,
    time_budget: float = 240.0,
    backward_depth: int = 3,
    backward_nodes: int = 3_000_000,
    forward_max_depth: int = 26,
    verbose: bool = False,
) -> Optional[List[MoveT]]:
    """Parallel bidirectional solve: serial backward flood, then root-split
    parallel forward IDDFS probing the oracle digest set."""
    import hashlib
    started = time.time()

    def key(cols):
        flat = bytearray()
        for col in cols:
            flat.extend(col)
        return hashlib.blake2b(bytes(flat), digest_size=12).digest()

    targets = set(won_states())
    if entry in targets:
        return []

    # ---- backward flood (serial, link-preserving) -----------------------
    back_link: Dict[bytes, Tuple[bytes, MoveT]] = {}
    visited = {key(entry)}
    frontier: List[Tuple[bytes, Tableau]] = []
    for t in targets:
        tk = key(t)
        if tk not in visited:
            visited.add(tk)
            frontier.append((tk, t))
    nodes = 0
    for _layer in range(backward_depth):
        if not frontier or nodes >= backward_nodes or time.time() - started > time_budget * 0.35:
            break
        nxt = []
        overflow = False
        for skey, s in frontier:
            if overflow:
                break
            for p, move in predecessors(s):
                pkey = key(p)
                if pkey in visited:
                    continue
                visited.add(pkey)
                back_link[pkey] = (skey, move)
                nxt.append((pkey, p))
                nodes += 1
                if nodes >= backward_nodes or time.time() - started > time_budget * 0.35:
                    overflow = True
                    break
        if overflow:
            break
        frontier = nxt
        if verbose:
            print(f"  par: backward layer={_layer+1} total={nodes} ({time.time()-started:.0f}s)")
    oracle = set(back_link.keys())

    # ---- parallel forward probe -----------------------------------------
    root_moves = legal_moves(entry)
    chunks = [[] for _ in range(workers)]
    for i, mv in enumerate(root_moves):
        chunks[i % workers].append(mv)
    remaining = time_budget - (time.time() - started)
    payloads = [
        (chunk, entry, oracle, max(5.0, remaining), 200_000_000, forward_max_depth)
        for chunk in chunks
        if chunk
    ]
    prefix_and_meet = None
    for res in pool.map(_forward_probe_worker, payloads):
        if res is not None:
            prefix_and_meet = res
            break
    if prefix_and_meet is None:
        return None
    prefix, meet_state_or_none = prefix_and_meet

    # ---- reconstruct tail -------------------------------------------------
    if meet_state_or_none is not None and not prefix:
        return []
    if meet_state_or_none is not None:
        # met exactly at a root child
        cur_key = key(meet_state_or_none)
    else:
        # replay prefix from entry to find meet state
        cur = entry
        for mv in prefix:
            cur = apply_move(cur, mv)
        cur_key = key(cur)
    tail: List[MoveT] = []
    guard = 0
    while cur_key not in {None}:
        nxt = back_link.get(cur_key)
        if nxt is None:
            break
        parent_key, mv = nxt
        tail.append(mv)
        cur_key = parent_key
        guard += 1
        if guard > 500:
            return None
    moves = list(prefix) + tail
    # final verification
    cols = entry
    for mv in moves:
        cols = apply_move(cols, mv)
    if is_won(cols):
        return moves
    return None


def segment_search_beam(
    start: Tableau,
    scorer: TableauScorer,
    budget: int = 150000,
    bfs_depth: int = 9,
    max_depth: int = 40,
    beam_width: int = 2500,
    stock_rank_counts: Optional[Dict[int, int]] = None,
    lock_penalty: float = 0.0,
) -> Tuple[Dict[Tableau, Optional[Tuple[Tableau, MoveT]]], Dict[Tableau, int]]:
    """Exhaustive BFS for the first ``bfs_depth`` plies, then beam-extend the
    most promising states much deeper. Endpoints are what matter, not paths."""
    seen: Dict[Tableau, Optional[Tuple[Tableau, MoveT]]] = {start: None}
    mobility: Dict[Tableau, int] = {}
    frontier = [start]
    nodes = 1
    depth = 0
    counts = stock_rank_counts or {}
    for _d in range(bfs_depth):
        if not frontier or nodes >= budget:
            break
        nxt: List[Tableau] = []
        for cols in frontier:
            moves = legal_moves(cols)
            mobility[cols] = len(moves)
            for move in moves:
                child = apply_move(cols, move)
                if child not in seen:
                    seen[child] = (cols, move)
                    nxt.append(child)
                    nodes += 1
        frontier = nxt
        depth += 1

    # ---- beam phase ----
    def locked_excess(cols: Tableau) -> int:
        if lock_penalty <= 0:
            return 0
        holes = sum(1 for col in cols if not col)
        locked = sum(1 for flag in dead_columns(cols, counts) if flag)
        return max(0, locked - holes)

    beam = sorted(frontier, key=lambda c: scorer.score(c, deal_size=7) - lock_penalty * locked_excess(c), reverse=True)[:beam_width]
    base_score: Dict[Tableau, float] = {c: scorer.score(c, deal_size=7) for c in beam}
    while beam and depth < max_depth and nodes < budget:
        cand: List[Tuple[float, Tableau]] = []
        for cols in beam:
            s0 = base_score.get(cols)
            if s0 is None:
                s0 = scorer.score(cols, deal_size=7)
            moves = legal_moves(cols)
            mobility[cols] = len(moves)
            for move in moves:
                child = apply_move(cols, move)
                if child in seen:
                    continue
                sc = scorer.score(child, deal_size=7)
                seen[child] = (cols, move)
                cand.append((sc - lock_penalty * locked_excess(child), child))
                nodes += 1
                if nodes >= budget:
                    break
            if nodes >= budget:
                break
        if not cand:
            break
        cand.sort(reverse=True)
        beam = [c for _s, c in cand[:beam_width]]
        base_score.update({c: s for s, c in cand[:beam_width]})
        depth += 1
    return seen, mobility


def _cli_run_game(payload):
    seed, cfg = payload
    return play_game(seed, cfg)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(prog="siesta-solver", description="Play Siesta with search.")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--endgame-time", type=float, default=90.0)
    parser.add_argument("--endgame-beam-width", type=int, default=750)
    parser.add_argument("--endgame-guide-k", type=float, default=20.0)
    parser.add_argument("--segment-budget", type=int, default=120000)
    parser.add_argument("--segment-mode", choices=["beam", "bfs"], default="beam")
    parser.add_argument("--beam-width", type=int, default=2500)
    parser.add_argument("--handoff-survive-weight", type=float, default=3.0)
    parser.add_argument("--deal-samples", type=int, default=10)
    args = parser.parse_args()

    cfg = SolverConfig(
        endgame_solver="hybrid",
        endgame_time=args.endgame_time,
        endgame_beam_width=args.endgame_beam_width,
        endgame_guide_k=args.endgame_guide_k,
        segment_budget=args.segment_budget,
        segment_mode=args.segment_mode,
        segment_beam_width=args.beam_width,
        handoff_survive_weight=args.handoff_survive_weight,
        deal_samples=args.deal_samples,
    )
    started = time.time()
    seeds = list(range(args.games))
    jobs = [(s, cfg) for s in seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_cli_run_game, jobs))
    else:
        results = [_cli_run_game(j) for j in jobs]
    wins = [r["seed"] for r in results if r["status"] == "won"]
    dead = sum(1 for r in results if r.get("dead_entry"))
    print(f"win_rate: {len(wins)}/{len(results)} ({len(wins)/len(results)*100:.0f}%)")
    print(f"losses: {len(results)-len(wins)} (provably-dead entries: {dead})")
    print(f"winning seeds: {wins}")
    print(f"wall: {time.time()-started:.0f}s")


# ---------------------------------------------------------------------------
# Learned winnability predictor (logistic regression on structural features).
# Trained offline on: user-won endgame entries + constructed winnable endgames
# (positives) vs provably-dead / exhaustively-refuted endgames (negatives).
# Perfect separation on the clean training set (AUC 1.0).
# ---------------------------------------------------------------------------

_LR_SD = [
    3.974, 3.9281, 4.0406, 3.8947, 1.3978, 1.4524, 1.4164, 1.446,
    14.5397, 176.0378, 18.0405, 4.7101, 4.6149, 4.6622, 4.5456,
    0.4857, 11.6671, 1.5356, 0.4857, 2.7941, 12.7666, 0.5544,
]
_LR_MU = [
    5.0321, 4.8761, 5.133, 4.9679, 1.9817, 1.9725, 1.9128, 1.9725,
    20.0092, 162.844, 21.1514, 5.3624, 5.1697, 5.3349, 5.2844,
    0.2202, 9.5046, 4.6422, 6.7798, 13.1927, 17.1558, 0.4174,
]
_LR_W = [
    0.3642, 0.6168, 0.1893, -0.1967, 1.569, 0.7535, 0.9467, 1.7461,
    0.2661, -0.2828, 0.7508, 1.0392, 0.7539, 0.4749, 0.6504,
    0.548, -0.0192, -0.6354, -0.548, 0.5547, 0.6148, 0.0657,
]
_LR_B = 1.9412


def _winnability_features(cols: Tableau) -> List[float]:
    chains_per_suit: List[List[int]] = [[], [], [], []]
    suit_adj = [0.0, 0.0, 0.0, 0.0]
    adj_total = 0.0
    heights = [len(c) for c in cols]
    for col in cols:
        i = 0
        n = len(col)
        while i < n:
            j = i
            while j + 1 < n and SUIT[col[j + 1]] == SUIT[col[j]] and RANK[col[j]] == RANK[col[j + 1]] + 1:
                j += 1
            L = j - i + 1
            s = SUIT[col[i]]
            chains_per_suit[s].append(L)
            adj_total += L - 1
            suit_adj[s] += L - 1
            i = j + 1
    longest = [max(c) if c else 0 for c in chains_per_suit]
    second = [sorted(c)[-2] if len(c) >= 2 else 0 for c in chains_per_suit]
    holes = sum(1 for h in heights if h == 0)
    mobility = float(len(legal_moves(cols)))
    tops = [RANK[c[-1]] for c in cols if c]
    rank_counts: Dict[int, int] = {}
    for r in tops:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    stranded = float(sum(1 for r in tops if rank_counts.get(r + 1, 0) == 0))
    nonempty = 7 - holes
    tallest = max(heights) if heights else 0
    mean_h = (sum(heights) / 7) or 0.0
    height_var = sum((h - mean_h) ** 2 for h in heights) / 7
    kings_exposed = float(sum(1 for c in cols if c and RANK[c[-1]] == 13))
    return (
        longest + second
        + [float(sum(longest)), sum(l * l for l in longest), adj_total]
        + suit_adj
        + [float(holes), mobility, stranded, float(nonempty),
           float(tallest), float(height_var), kings_exposed]
    )


def predict_winnability_logit(cols: Tableau) -> float:
    """Predicted log-odds that the position is winnable."""
    x = _winnability_features(cols)
    z = _LR_B
    for xi, mu_i, sd_i, w_i in zip(x, _LR_MU, _LR_SD, _LR_W):
        z += w_i * (xi - mu_i) / sd_i
    return z


def predict_winnability(cols: Tableau) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, predict_winnability_logit(cols)))))


_SOLV_SD = [4.062468199679518, 4.076344983874405, 3.979847403454358, 3.927150770933021, 1.547858204442357, 1.400556909972723, 1.3815905145739096, 1.4671943660422246, 14.842381265540235, 182.91936009309669, 18.215760369658003, 4.752041154717879, 4.6691220725601115, 4.675990346896303, 4.601738936915748, 0.5047330490129501, 14.074721171043088, 1.5087743993606175, 0.50473304901295, 2.882437944838018, 13.216115316730042, 0.56490404364677]
_SOLV_MU = [4.737089201877934, 4.666666666666667, 4.732394366197183, 4.671361502347418, 1.8685446009389672, 1.7511737089201878, 1.8262910798122065, 1.892018779342723, 18.8075117370892, 152.81690140845072, 19.295774647887324, 4.929577464788732, 4.690140845070423, 4.821596244131455, 4.854460093896714, 0.2347417840375587, 10.12206572769953, 4.605633802816901, 6.765258215962441, 13.03755868544601, 16.489029414582745, 0.4225352112676056]
_SOLV_W = [-0.6061609432003007, 1.065144854601152, -0.18081668073384197, 0.7920680463623988, 0.520344817525556, 0.49623090873357817, -0.20212810993697633, 0.5554157044732303, 0.2877123350154235, 1.4553515444618799, 0.43999761504744317, -0.14967577478642652, 0.8082269511936688, 0.09542498063911582, 0.9792472012367783, -0.5935637016268255, 1.3957952359860166, 0.7175626084505757, 0.5935637016268238, -2.0373560015896004, -0.5352418673684494, -0.11301147429882473]
_SOLV_B = -4.3557775906438305


def predict_solvable_logit(cols: Tableau) -> float:
    """Predicted log-odds that THIS solver can finish the position to a win
    within its typical budget. Trained on hybrid-solve outcomes over user
    entries, constructed winnable endgames and refuted randoms."""
    x = _winnability_features(cols)
    z = _SOLV_B
    for xi, mu_i, sd_i, w_i in zip(x, _SOLV_MU, _SOLV_SD, _SOLV_W):
        z += w_i * (xi - mu_i) / sd_i
    return z


_OUTCOME_SD = [1.500378619882584, 1.8486752023266124, 1.7921428524490668, 1.6986771333591784, 0.7627132862884892, 0.7563373742814129, 0.744396550039705, 0.7834102099937045, 4.957500446453002, 42.59689578774325, 6.902447103296402, 2.112565581836096, 2.3631481836862482, 2.3030553629051336, 2.2683820969735464, 1e-09, 1.5315908950270773, 1.2319711406331053, 1e-09, 3.047737522506735, 8.77332503046075, 0.6990838459157738, 6.902447103296402, 3.494528040982258, 1.1462448846672402, 0.7909443030143001, 6.831292183954037]
_OUTCOME_MU = [2.808, 2.68, 2.832, 2.864, 1.5653333333333332, 1.4426666666666668, 1.6026666666666667, 1.5653333333333332, 11.184, 43.056, 9.832, 2.52, 2.192, 2.544, 2.576, 0.0, 3.336, 4.277333333333333, 7.0, 11.464, 11.391238095238089, 0.5813333333333334, 9.832, 7.469333333333333, 7.384, 5.8693333333333335, 49.32266666666667]
_OUTCOME_W = [-0.7451219490431238, -1.4044436365771613, 1.4155259450902988, -4.744468783960605, 1.6252097665710221, 3.784855660697845, 0.5789684802413565, 4.507864917414817, -5.478508424490587, -0.020454026859750063, 3.295924581262237, 0.30872723290066983, -0.060402053697118624, 0.42310594695553083, 2.6244934551031447, 0.0, 3.9684331803723443, -0.701723146828017, 7.601229766007353, 2.2292624153115166, 1.77435720929334, 2.7114993034512347, 3.295924581262237, -1.9949628008758131, 0.28392960770213205, 3.6753459536905315, -2.976472969946173]
_OUTCOME_B = 1.0858899665724748


def predict_outcome_logit(cols: Tableau) -> float:
    """Learned log-odds that a position resembles winning play.

    Trained on post-deal positions from user-won games (positive) vs
    solver-lost games (negative), game-level cross-validated AUC 0.79.
    """
    from backend.siesta.solver import _outcome_features
    x = _outcome_features(cols)
    z = _OUTCOME_B
    for xi, mu_i, sd_i, w_i in zip(x, _OUTCOME_MU, _OUTCOME_SD, _OUTCOME_W):
        z += w_i * (xi - mu_i) / sd_i
    return z


def _outcome_features(cols: Tableau) -> List[float]:
    return _winnability_features(cols) + _anatomy_features(cols)


def _anatomy_features(cols: Tableau) -> List[float]:
    ss = xs = 0
    for col in cols:
        for i in range(len(col) - 1):
            if RANK[col[i]] == RANK[col[i + 1]] + 1:
                if SUIT[col[i]] == SUIT[col[i + 1]]:
                    ss += 1
                else:
                    xs += 1
    movable = sum(fg_top_run(c) for c in cols)
    tops = [RANK[c[-1]] for c in cols if c]
    distinct = len(set(tops))
    cards = float(sum(len(c) for c in cols))
    return [float(ss), float(xs), float(movable), float(distinct), cards]




def choose_handoff_target(
    seen: Dict[Tableau, Optional[Tuple[Tableau, MoveT]]],
    scorer: TableauScorer,
    stock: Tuple[int, ...],
    deal_size: int,
    lock_penalty: float = 40.0,
    samples: int = 14,
    candidates: int = 350,
    seed_base: int = 0,
    survive_weight: float = 0.0,
) -> Optional[Tableau]:
    """Pick the final-segment tableau whose resulting endgame (after dealing
    ``deal_size`` cards sampled fairly from the unseen multiset) has the best
    predicted winnability. The predictor only ever sees full 52-card states —
    exactly its training distribution.

    With ``survive_weight`` > 0, candidates also earn credit for the fraction
    of preview deals that leave a provably-alive endgame (not every column
    holding a dead card under empty-stock counts, which is exact post-deal).
    Pure logit ranking can prefer positions that the true deal then buries
    into provably-dead corners; the survival term favours robust handoffs.
    """
    rng = random.Random(seed_base)
    scored: List[Tuple[float, Tableau]] = []
    for cols in seen.keys():
        s = scorer.score(cols, deal_size=deal_size)
        counts = {r: 0 for r in range(1, 14)}
        excess_locked = 0
        if lock_penalty:
            holes = sum(1 for col in cols if not col)
            locked = sum(1 for flag in dead_columns(cols, counts) if flag)
            excess_locked = max(0, locked - holes)
            s -= lock_penalty * excess_locked
        scored.append((s, cols))
    if not scored:
        return None
    scored.sort(reverse=True)

    preview_orders = []
    for _ in range(samples):
        picks = list(stock)
        rng.shuffle(picks)
        preview_orders.append(tuple(picks[:deal_size]))

    best_cols: Optional[Tableau] = None
    best_value = float("-inf")
    empty_counts = {r: 0 for r in range(1, 14)}
    for _s, cols in scored[:candidates]:
        total = 0.0
        alive = 0
        for order in preview_orders:
            dealt = list(cols)
            for j in range(deal_size):
                dealt[j] = dealt[j] + (order[j],)
            dealt_t = tuple(dealt)
            logit = predict_solvable_logit(dealt_t)
            total += min(logit, 8.0)  # cap: beyond p~0.9997 details don't matter
            if survive_weight and not all(dead_columns(dealt_t, empty_counts)):
                alive += 1
        value = total / samples
        if survive_weight:
            value += survive_weight * (alive / samples)
        if value > best_value:
            best_value = value
            best_cols = cols
    return best_cols
