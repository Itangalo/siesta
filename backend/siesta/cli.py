from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .training import (
    benchmark_policies,
    benchmark_hole_goal,
    build_training_examples,
    evaluate_policy,
    evaluate_hole_goal,
    load_latest_model,
    load_hole_model,
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
    generate.set_defaults(func=command_generate_data)

    train = subparsers.add_parser("train-policy")
    train.add_argument("--games", type=int, default=60)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=0.02)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--teacher-policy", choices=["heuristic", "search"], default="search")
    train.add_argument("--benchmark-games", type=int, default=20)
    train.set_defaults(func=command_train_policy)

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
