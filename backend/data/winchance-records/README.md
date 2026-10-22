# Win-chance training records

Collected `(seed, deals_used, cols, won)` records for retraining the win-chance model. The pickles themselves are gitignored; this note tracks what exists and what is missing.

Why a retrain is pending: the run-borne blocker fix changed `fastgame.dead_columns`, which feeds feature 27 (`dead`, the number of locked columns) in `winchance.features`. The feature moved in 32 % of positions, mean 0.51 fewer locked columns, so the committed weights in `backend/siesta/winchance_weights.json` were fitted against values the engine no longer produces. Measured drift in the displayed percentage: median ~5 pp, p90 19 pp, max 47 pp, worst in late stages. Only the estimated percentage is affected – `proven_win` and `proven_loss` are exact and were never touched.

Collected so far:

- `wc-seeds-0-999.pkl` – seeds 0–999, 5000 records, 1000 games (213 won), 1000 records per stage 0–4. Took 44.5 min on 8 workers.

Missing for a 3000-game sample matching the current weights:

- seeds 1000–1999
- seeds 2000–2999

To resume, from the project root:

```bash
for start in 1000 2000; do
  PYTHONPATH=. python3.9 -m backend.siesta.winchance collect $start 1000 \
    backend/data/winchance-records/wc-seeds-$start-$((start+999)).pkl --workers 8
done
```

Roughly 1.5 h for both chunks; wrap the loop in `caffeinate -i` to keep the machine awake. Then fit and overwrite the weights:

```bash
PYTHONPATH=. python3.9 -m backend.siesta.winchance train backend/data/winchance-records/*.pkl
```

Features are recomputed from the stored tableaus at train time, so these records stay valid across further changes to `dead_columns` – only a change to the record format would force a re-collect.

Seeds are contiguous from 0. Policy training draws its seeds randomly from 0–1e9 (`training.training_game_seeds`), so overlap is negligible.
