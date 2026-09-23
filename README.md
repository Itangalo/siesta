# Siesta

Lokal implementation av patiensen Siesta med:

- Python/FastAPI-backend för spelregler och API
- statisk webbapp för att spela i browsern
- NumPy-baserad träningspipeline för en första neuronnätsagent

## Kör servern

```bash
python3 -m uvicorn backend.siesta.api:app --reload
```

Öppna sedan [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Träna modellen

```bash
python3 -m backend.siesta.cli train-policy --games 80 --epochs 40 --teacher-policy search --benchmark-games 24
```

## Utvärdera policy

```bash
python3 -m backend.siesta.cli evaluate-policy --policy heuristic --games 20
python3 -m backend.siesta.cli evaluate-policy --policy search --games 20
python3 -m backend.siesta.cli evaluate-policy --policy model --games 20
python3 -m backend.siesta.cli benchmark --games 20
```

## API i korthet

- `POST /game/new`
- `POST /game/move`
- `POST /game/deal`
- `POST /game/concede`
- `POST /game/undo`
- `GET /game/legal-moves`
- `POST /ai/evaluate-move` – sökmotorn driver förslagen: plansteg mot bästa nåbara tablå i mellan spelet, bevisad vinstlinje i slutspelen
- `POST /ai/solve-advice` – fördjupad slutspelsanalys på begäran: bevisad vinstlinje, "ej bevisat" eller "bevisligt förlorat"
- `POST /training/run`
- `GET /training/status/{job_id}`
