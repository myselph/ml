"""
Flask server for Scout card game API.
Provides HTTP endpoints for game state management and move execution.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import time
from typing import Optional

from game_state import GameState
from players import PlanningPlayer
from serialization import serialize_game_state, serialize_move, deserialize_move

app = Flask(__name__)
CORS(app)

# In-memory session storage
# sessions[session_id] = {
#     "game_state": GameState,
#     "players": [None, PlanningPlayer(), ...],  # Human is player 0
#     "created_at": timestamp,
#     "last_access": timestamp
# }
sessions = {}


def get_session(session_id: str) -> Optional[dict]:
    """Retrieve a session and update last access time."""
    if session_id not in sessions:
        return None
    
    session = sessions[session_id]
    session["last_access"] = time.time()
    return session


@app.route('/new_game', methods=['POST'])
def new_game():
    """
    Create a new game session.
    
    Request body:
    {
        "num_players": int (3-5),
        "dealer": int (0 to num_players-1)
    }
    
    Response:
    {
        "session_id": str
    }
    """
    data = request.json
    num_players = data.get('num_players')
    dealer = data.get('dealer')
    
    # Validate inputs
    if not isinstance(num_players, int) or num_players < 3 or num_players > 5:
        return jsonify({"error": "num_players must be between 3 and 5"}), 400
    
    if not isinstance(dealer, int) or dealer < 0 or dealer >= num_players:
        return jsonify({"error": f"dealer must be between 0 and {num_players - 1}"}), 400
    
    # Create game state
    game_state = GameState(num_players, dealer)
    
    # Create AI players (human is player 0, so None for index 0)
    players = [None] + [PlanningPlayer() for _ in range(num_players - 1)]
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Store session
    sessions[session_id] = {
        "game_state": game_state,
        "players": players,
        "created_at": time.time(),
        "last_access": time.time()
    }
    
    return jsonify({"session_id": session_id})


@app.route('/state', methods=['GET'])
def get_state():
    """
    Get the current game state.
    
    Query params:
        session_id: str
    
    Response:
    {
        "game_state": {
            "current_player": int,
            "dealer": int,
            "hands": [[card, ...], ...],
            "table": [card, ...],
            "scores": [int, ...],
            "can_scout_and_show": [bool, ...],
            "is_finished": bool
        },
        "possible_moves": [move, ...] | null  // Only if current_player == 0
    }
    """
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    
    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Invalid session_id"}), 404
    
    game_state = session["game_state"]
    
    # Serialize game state
    state_data = serialize_game_state(game_state)
    
    # If it's the human player's turn and game is not finished, include possible moves
    possible_moves = None
    if game_state.current_player == 0 and not game_state.is_finished():
        if game_state.initial_flip_executed:
            info_state = game_state.info_state()
            possible_moves = [serialize_move(move) for move in info_state.possible_moves()]
    
    return jsonify({
        "game_state": state_data,
        "possible_moves": possible_moves
    })


@app.route('/flip_hand', methods=['POST'])
def flip_hand():
    """
    Execute the initial hand flip for all players.
    
    Request body:
    {
        "session_id": str,
        "flip": bool  // Human player's flip decision
    }
    
    Response:
    {
        "status": "ok"
    }
    """
    data = request.json
    session_id = data.get('session_id')
    human_flip = data.get('flip')
    
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    
    if not isinstance(human_flip, bool):
        return jsonify({"error": "flip must be a boolean"}), 400
    
    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Invalid session_id"}), 404
    
    game_state = session["game_state"]
    players = session["players"]
    
    if game_state.initial_flip_executed:
        return jsonify({"error": "Hand flip already executed"}), 400
    
    # Build flip functions for all players
    flip_fns = []
    for i in range(game_state.num_players):
        if i == 0:
            # Human player - use provided decision
            flip_fns.append(lambda hand: human_flip)
        else:
            # AI player - use their flip_hand method
            player = players[i] 
            flip_fns.append(player.flip_hand)
    
    # Execute flip
    game_state.maybe_flip_hand(flip_fns)
    
    return jsonify({"status": "ok"})


@app.route('/advance', methods=['POST'])
def advance():
    """
    Advance the game by one move.
    
    If it's the human player's turn (current_player == 0), the move must be provided.
    If it's an AI player's turn, the AI will select and execute the move automatically.
    
    Request body:
    {
        "session_id": str,
        "move": dict | null  // Required only if current_player == 0
    }
    
    Response:
    {
        "status": "ok",
        "current_player": int  // Player index after the move
    }
    """
    data = request.json
    session_id = data.get('session_id')
    move_data = data.get('move')
    
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    
    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Invalid session_id"}), 404
    
    game_state = session["game_state"]
    players = session["players"]
    
    if not game_state.initial_flip_executed:
        return jsonify({"error": "Must call /flip_hand before /advance"}), 400
    
    if game_state.is_finished():
        return jsonify({"error": "Game is already finished"}), 400
    
    current_player = game_state.current_player
    
    if current_player == 0:
        # Human player's turn - move must be provided
        if move_data is None:
            return jsonify({"error": "move is required for human player"}), 400
        
        try:
            move = deserialize_move(move_data)
        except (ValueError, KeyError) as e:
            return jsonify({"error": f"Invalid move format: {str(e)}"}), 400
        
        # Validate move
        info_state = game_state.info_state()
        if move not in info_state.possible_moves():
            return jsonify({"error": "Invalid move"}), 400
        
        # Execute move
        game_state.move(move)
    else:
        # AI player's turn - select and execute move
        player = players[current_player]
        info_state = game_state.info_state()
        move = player.select_move(info_state)
        game_state.move(move)
    
    return jsonify({
        "status": "ok",
        "current_player": game_state.current_player
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
