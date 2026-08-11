from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .training import (
    benchmark_compact_goal,
    benchmark_policies,
    benchmark_hole_goal,
    build_training_examples,
    evaluate_compact_goal,
    evaluate_policy,
    evaluate_hole_goal,
    improve_policy_from_beam_search_wins,
    improve_policy_from_winning_rollouts,
    load_compact_model,
    load_latest_model,
    load_hole_model,
    train_compact_opening_model,
    train_hole_opening_model,
    train_policy_model,
)


def console_progress(message: str) -> None:
    print(message, flush=True)


def strip_episodes(payload):
    if isinstance(payload, dict):
        return {key: strip_episodes(value) for key, value in payload.items() if key != "episodes"}
    if isinstance(payload, list):
        return [strip_episodes(value) for value in payload]
    return payload


def command_generate_data(args: argparse.Namespace) -> None:
    features, labels = build_training_examples(
        num_games=args.games,
        teacher_policy=args.teacher_policy,
        state_source=args.state_source,
        progress=console_progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, features=features, labels=labels)
    print(json.dumps({"output": str(output), "examples": int(len(labels)), "teacher_policy": args.teacher_policy}, indent=2))


def command_train_policy(args: argparse.Namespace) -> None:
    result = train_policy_model(
        num_games=args.games,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        teacher_policy=args.teacher_policy,
        benchmark_games=args.benchmark_games,
        training_state_source=args.state_source,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def command_improve_policy_wins(args: argparse.Namespace) -> None:
    result = improve_policy_from_winning_rollouts(
        num_games=args.games,
        rollouts_per_game=args.rollouts_per_game,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        benchmark_games=args.benchmark_games,
        temperature=args.temperature,
        top_k=args.top_k,
        epsilon_random=args.epsilon_random,
        max_moves=args.max_moves,
        stagnation_limit=args.stagnation_limit,
        repeat_limit=args.repeat_limit,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def command_improve_policy_beam_wins(args: argparse.Namespace) -> None:
    result = improve_policy_from_beam_search_wins(
        num_games=args.games,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        benchmark_games=args.benchmark_games,
        beam_width=args.beam_width,
        branching_factor=args.branching_factor,
        max_nodes_per_game=args.max_nodes_per_game,
        max_moves=args.max_moves,
        max_wins_per_game=args.max_wins_per_game,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def command_evaluate_policy(args: argparse.Namespace) -> None:
    model = load_latest_model() if args.policy == "model" else None
    result = evaluate_policy(policy=args.policy, model=model, seeds=range(args.games), progress=console_progress)
    print(json.dumps(strip_episodes(result), indent=2))


def command_benchmark(args: argparse.Namespace) -> None:
    result = benchmark_policies(games=args.games, progress=console_progress)
    print(json.dumps(strip_episodes(result), indent=2))


def command_evaluate_hole_goal(args: argparse.Namespace) -> None:
    model = load_hole_model() if args.policy == "model" else None
    result = evaluate_hole_goal(
        policy=args.policy,
        model=model,
        seeds=range(args.games),
        horizon=args.horizon,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def command_benchmark_hole_goal(args: argparse.Namespace) -> None:
    result = benchmark_hole_goal(games=args.games, horizon=args.horizon, progress=console_progress)
    print(json.dumps(strip_episodes(result), indent=2))


def command_evaluate_compact_goal(args: argparse.Namespace) -> None:
    model = load_compact_model() if args.policy == "model" else None
    result = evaluate_compact_goal(
        policy=args.policy,
        model=model,
        seeds=range(args.games),
        horizon=args.horizon,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def command_benchmark_compact_goal(args: argparse.Namespace) -> None:
    result = benchmark_compact_goal(games=args.games, horizon=args.horizon, progress=console_progress)
    print(json.dumps(strip_episodes(result), indent=2))


def command_train_compact_policy(args: argparse.Namespace) -> None:
    result = train_compact_opening_model(
        num_games=args.games,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        benchmark_games=args.benchmark_games,
        horizon=args.horizon,
        midgame_share=args.midgame_share,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def command_train_hole_policy(args: argparse.Namespace) -> None:
    result = train_hole_opening_model(
        num_games=args.games,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        benchmark_games=args.benchmark_games,
        horizon=args.horizon,
        progress=console_progress,
    )
    print(json.dumps(strip_episodes(result), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siesta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data")
    generate.add_argument("--games", type=int, default=60)
    generate.add_argument("--output", default="backend/models/training-data.npz")
    generate.add_argument("--teacher-policy", choices=["heuristic", "search"], default="search")
    generate.add_argument("--state-source", choices=["random_seed", "won_initial_states"], default="random_seed")
    generate.set_defaults(func=command_generate_data)

    train = subparsers.add_parser("train-policy")
    train.add_argument("--games", type=int, default=60)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=0.02)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--teacher-policy", choices=["heuristic", "search"], default="search")
    train.add_argument("--benchmark-games", type=int, default=20)
    train.add_argument("--state-source", choices=["random_seed", "won_initial_states"], default="won_initial_states")
    train.set_defaults(func=command_train_policy)

    improve = subparsers.add_parser("improve-policy-wins")
    improve.add_argument("--games", type=int, default=20)
    improve.add_argument("--rollouts-per-game", type=int, default=32)
    improve.add_argument("--epochs", type=int, default=20)
    improve.add_argument("--learning-rate", type=float, default=0.02)
    improve.add_argument("--hidden-dim", type=int, default=256)
    improve.add_argument("--benchmark-games", type=int, default=20)
    improve.add_argument("--temperature", type=float, default=1.0)
    improve.add_argument("--top-k", type=int, default=3)
    improve.add_argument("--epsilon-random", type=float, default=0.08)
    improve.add_argument("--max-moves", type=int, default=180)
    improve.add_argument("--stagnation-limit", type=int, default=30)
    improve.add_argument("--repeat-limit", type=int, default=3)
    improve.set_defaults(func=command_improve_policy_wins)

    beam_improve = subparsers.add_parser("improve-policy-beam-wins")
    beam_improve.add_argument("--games", type=int, default=19)
    beam_improve.add_argument("--epochs", type=int, default=10)
    beam_improve.add_argument("--learning-rate", type=float, default=0.02)
    beam_improve.add_argument("--hidden-dim", type=int, default=256)
    beam_improve.add_argument("--benchmark-games", type=int, default=20)
    beam_improve.add_argument("--beam-width", type=int, default=48)
    beam_improve.add_argument("--branching-factor", type=int, default=6)
    beam_improve.add_argument("--max-nodes-per-game", type=int, default=4000)
    beam_improve.add_argument("--max-moves", type=int, default=180)
    beam_improve.add_argument("--max-wins-per-game", type=int, default=1)
    beam_improve.set_defaults(func=command_improve_policy_beam_wins)

    evaluate = subparsers.add_parser("evaluate-policy")
    evaluate.add_argument("--policy", choices=["heuristic", "random", "search", "model"], default="heuristic")
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.set_defaults(func=command_evaluate_policy)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--games", type=int, default=20)
    benchmark.set_defaults(func=command_benchmark)

    evaluate_hole = subparsers.add_parser("evaluate-hole-goal")
    evaluate_hole.add_argument("--policy", choices=["heuristic", "random", "search", "hole_search", "model"], default="heuristic")
    evaluate_hole.add_argument("--games", type=int, default=20)
    evaluate_hole.add_argument("--horizon", type=int, default=5)
    evaluate_hole.set_defaults(func=command_evaluate_hole_goal)

    benchmark_hole = subparsers.add_parser("benchmark-hole-goal")
    benchmark_hole.add_argument("--games", type=int, default=20)
    benchmark_hole.add_argument("--horizon", type=int, default=5)
    benchmark_hole.set_defaults(func=command_benchmark_hole_goal)

    evaluate_compact = subparsers.add_parser("evaluate-compact-goal")
    evaluate_compact.add_argument("--policy", choices=["heuristic", "random", "search", "hole_search", "compact_search", "model"], default="heuristic")
    evaluate_compact.add_argument("--games", type=int, default=20)
    evaluate_compact.add_argument("--horizon", type=int, default=5)
    evaluate_compact.set_defaults(func=command_evaluate_compact_goal)

    benchmark_compact = subparsers.add_parser("benchmark-compact-goal")
    benchmark_compact.add_argument("--games", type=int, default=20)
    benchmark_compact.add_argument("--horizon", type=int, default=5)
    benchmark_compact.set_defaults(func=command_benchmark_compact_goal)

    train_compact = subparsers.add_parser("train-compact-policy")
    train_compact.add_argument("--games", type=int, default=120)
    train_compact.add_argument("--epochs", type=int, default=30)
    train_compact.add_argument("--learning-rate", type=float, default=0.02)
    train_compact.add_argument("--hidden-dim", type=int, default=64)
    train_compact.add_argument("--benchmark-games", type=int, default=50)
    train_compact.add_argument("--horizon", type=int, default=5)
    train_compact.add_argument("--midgame-share", type=float, default=0.4)
    train_compact.set_defaults(func=command_train_compact_policy)

    train_hole = subparsers.add_parser("train-hole-policy")
    train_hole.add_argument("--games", type=int, default=120)
    train_hole.add_argument("--epochs", type=int, default=30)
    train_hole.add_argument("--learning-rate", type=float, default=0.02)
    train_hole.add_argument("--hidden-dim", type=int, default=64)
    train_hole.add_argument("--benchmark-games", type=int, default=50)
    train_hole.add_argument("--horizon", type=int, default=5)
    train_hole.set_defaults(func=command_train_hole_policy)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
