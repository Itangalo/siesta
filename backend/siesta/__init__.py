"""Core package for Siesta game logic, API, and training."""

from .game import GameState, Move, create_game, serialize_state

__all__ = ["GameState", "Move", "create_game", "serialize_state"]
