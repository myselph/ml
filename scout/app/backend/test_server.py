"""
Integration tests for Scout backend API.
Tests game flow from initialization through completion.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scout'))

import pytest
import requests
import time
from typing import Optional

# Base URL for the API
BASE_URL = "http://localhost:5000"


class TestClient:
    """Helper class for making API requests."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session_id: Optional[str] = None
    
    def new_game(self, num_players: int, dealer: int) -> str:
        """Create a new game and store the session ID."""
        response = requests.post(
            f"{self.base_url}/new_game",
            json={"num_players": num_players, "dealer": dealer}
        )
        assert response.status_code == 200, f"Failed to create game: {response.text}"
        data = response.json()
        self.session_id = data["session_id"]
        return self.session_id
    
    def get_state(self):
        """Get the current game state."""
        assert self.session_id, "No active session"
        response = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id}
        )
        assert response.status_code == 200, f"Failed to get state: {response.text}"
        return response.json()
    
    def flip_hand(self, flip: bool):
        """Execute hand flip."""
        assert self.session_id, "No active session"
        response = requests.post(
            f"{self.base_url}/flip_hand",
            json={"session_id": self.session_id, "flip": flip}
        )
        assert response.status_code == 200, f"Failed to flip hand: {response.text}"
        return response.json()
    
    def advance(self, move: Optional[dict] = None):
        """Advance the game by one move."""
        assert self.session_id, "No active session"
        response = requests.post(
            f"{self.base_url}/advance",
            json={"session_id": self.session_id, "move": move}
        )
        assert response.status_code == 200, f"Failed to advance: {response.text}"
        return response.json()


def test_new_game():
    """Test creating a new game session."""
    client = TestClient()
    session_id = client.new_game(num_players=3, dealer=0)
    
    assert session_id is not None
    assert len(session_id) > 0
    
    # Verify we can get the initial state
    state = client.get_state()
    assert state["game_state"]["num_players"] == 3
    assert state["game_state"]["dealer"] == 0
    assert state["game_state"]["current_player"] == 0
    assert state["game_state"]["is_finished"] is False


def test_flip_hand():
    """Test the hand flip functionality."""
    client = TestClient()
    client.new_game(num_players=3, dealer=0)
    
    # Flip hand
    result = client.flip_hand(flip=False)
    assert result["status"] == "ok"
    
    # Verify we can now get possible moves
    state = client.get_state()
    assert state["possible_moves"] is not None
    assert len(state["possible_moves"]) > 0


def test_invalid_move():
    """Test that invalid moves are rejected."""
    client = TestClient()
    client.new_game(num_players=3, dealer=0)
    client.flip_hand(flip=False)
    
    # Try an invalid move
    invalid_move = {
        "type": "show",
        "startPos": 0,
        "length": 100  # Invalid - hand doesn't have 100 cards
    }
    
    response = requests.post(
        f"{BASE_URL}/advance",
        json={"session_id": client.session_id, "move": invalid_move}
    )
    assert response.status_code == 400


def test_full_game_human_dealer():
    """Play a complete game with human as dealer."""
    client = TestClient()
    client.new_game(num_players=3, dealer=0)
    client.flip_hand(flip=False)
    
    move_count = 0
    max_moves = 200  # Safety limit
    
    while move_count < max_moves:
        state = client.get_state()
        
        if state["game_state"]["is_finished"]:
            print(f"Game finished after {move_count} moves")
            print(f"Final scores: {state['game_state']['scores']}")
            break
        
        current_player = state["game_state"]["current_player"]
        
        if current_player == 0:
            # Human player - pick first available move
            possible_moves = state["possible_moves"]
            assert possible_moves is not None and len(possible_moves) > 0
            move = possible_moves[0]
            client.advance(move=move)
        else:
            # AI player - no move needed
            client.advance()
        
        move_count += 1
    
    assert move_count < max_moves, "Game did not finish within move limit"
    
    # Verify final state
    final_state = client.get_state()
    assert final_state["game_state"]["is_finished"] is True


def test_full_game_human_not_dealer():
    """Play a complete game with human not as dealer."""
    client = TestClient()
    client.new_game(num_players=3, dealer=1)
    client.flip_hand(flip=True)
    
    move_count = 0
    max_moves = 200
    
    while move_count < max_moves:
        state = client.get_state()
        
        if state["game_state"]["is_finished"]:
            print(f"Game finished after {move_count} moves")
            print(f"Final scores: {state['game_state']['scores']}")
            break
        
        current_player = state["game_state"]["current_player"]
        
        if current_player == 0:
            # Human player
            possible_moves = state["possible_moves"]
            assert possible_moves is not None and len(possible_moves) > 0
            move = possible_moves[0]
            client.advance(move=move)
        else:
            # AI player
            client.advance()
        
        move_count += 1
    
    assert move_count < max_moves, "Game did not finish within move limit"
    
    final_state = client.get_state()
    assert final_state["game_state"]["is_finished"] is True


def test_session_isolation():
    """Test that multiple sessions don't interfere with each other."""
    client1 = TestClient()
    client2 = TestClient()
    
    # Create two separate games
    session1 = client1.new_game(num_players=3, dealer=0)
    session2 = client2.new_game(num_players=4, dealer=1)
    
    assert session1 != session2
    
    # Verify they have different states
    state1 = client1.get_state()
    state2 = client2.get_state()
    
    assert state1["game_state"]["num_players"] == 3
    assert state2["game_state"]["num_players"] == 4
    assert state1["game_state"]["dealer"] == 0
    assert state2["game_state"]["dealer"] == 1


if __name__ == "__main__":
    print("Running tests...")
    print("Make sure the server is running on http://localhost:5000")
    print()
    pytest.main([__file__, "-v"])
