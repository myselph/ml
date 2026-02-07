"""
Serialization utilities for Scout game types.
Converts between Python dataclasses and JSON-compatible dictionaries.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scout'))

from common import Scout, Show, ScoutAndShow, Move, Card
from game_state import GameState


def serialize_card(card: Card) -> list[int]:
    """Convert a card tuple to a list."""
    return [card[0], card[1]]


def serialize_move(move: Move) -> dict:
    """Convert a Move (Scout/Show/ScoutAndShow) to a JSON-compatible dict."""
    if isinstance(move, Scout):
        return {
            "type": "scout",
            "first": move.first,
            "flip": move.flip,
            "insertPos": move.insertPos
        }
    elif isinstance(move, Show):
        return {
            "type": "show",
            "startPos": move.startPos,
            "length": move.length
        }
    elif isinstance(move, ScoutAndShow):
        return {
            "type": "scout_and_show",
            "scout": serialize_move(move.scout),
            "show": serialize_move(move.show)
        }
    else:
        raise ValueError(f"Unknown move type: {type(move)}")


def deserialize_move(data: dict) -> Move:
    """Convert a JSON dict back to a Move object."""
    move_type = data.get("type")
    
    if move_type == "scout":
        return Scout(
            first=data["first"],
            flip=data["flip"],
            insertPos=data["insertPos"]
        )
    elif move_type == "show":
        return Show(
            startPos=data["startPos"],
            length=data["length"]
        )
    elif move_type == "scout_and_show":
        return ScoutAndShow(
            scout=deserialize_move(data["scout"]),
            show=deserialize_move(data["show"])
        )
    else:
        raise ValueError(f"Unknown move type: {move_type}")


def serialize_game_state(gs: GameState) -> dict:
    """
    Serialize the full GameState to a JSON-compatible dict.
    Includes all player hands (frontend can implement debug mode).
    """
    return {
        "num_players": gs.num_players,
        "dealer": gs.dealer,
        "current_player": gs.current_player,
        "scout_benefactor": gs.scout_benefactor,
        "hands": [[serialize_card(card) for card in hand] for hand in gs.hands],
        "table": [serialize_card(card) for card in gs.table],
        "scores": gs.scores,
        "can_scout_and_show": gs.can_scout_and_show,
        "is_finished": gs.is_finished()
    }
