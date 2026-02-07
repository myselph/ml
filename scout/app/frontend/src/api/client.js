/**
 * API client for Scout card game backend.
 * All functions log requests and responses to console for debugging.
 */

const API_BASE_URL = 'http://localhost:5000';

/**
 * Helper function to log API calls
 */
function logAPI(method, endpoint, params, result) {
    console.log(`[API] ${method} ${endpoint}`, {
        params,
        result
    });
}

/**
 * Create a new game session
 * @param {number} numPlayers - Number of players (3-5)
 * @param {number} dealer - Dealer index (0 to numPlayers-1)
 * @returns {Promise<{session_id: string}>}
 */
export async function newGame(numPlayers, dealer) {
    const params = { num_players: numPlayers, dealer };

    const response = await fetch(`${API_BASE_URL}/new_game`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });

    if (!response.ok) {
        const error = await response.json();
        logAPI('POST', '/new_game', params, { error });
        throw new Error(error.error || 'Failed to create new game');
    }

    const result = await response.json();
    logAPI('POST', '/new_game', params, result);
    return result;
}

/**
 * Get current game state
 * @param {string} sessionId - Session ID
 * @returns {Promise<{game_state: object, possible_moves: array|null}>}
 */
export async function getState(sessionId) {
    const params = { session_id: sessionId };

    const response = await fetch(`${API_BASE_URL}/state?session_id=${sessionId}`, {
        method: 'GET'
    });

    if (!response.ok) {
        const error = await response.json();
        logAPI('GET', '/state', params, { error });
        throw new Error(error.error || 'Failed to get game state');
    }

    const result = await response.json();
    logAPI('GET', '/state', params, result);
    return result;
}

/**
 * Execute initial hand flip for all players
 * @param {string} sessionId - Session ID
 * @param {boolean} flip - Human player's flip decision
 * @returns {Promise<{status: string}>}
 */
export async function flipHand(sessionId, flip) {
    const params = { session_id: sessionId, flip };

    const response = await fetch(`${API_BASE_URL}/flip_hand`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });

    if (!response.ok) {
        const error = await response.json();
        logAPI('POST', '/flip_hand', params, { error });
        throw new Error(error.error || 'Failed to flip hand');
    }

    const result = await response.json();
    logAPI('POST', '/flip_hand', params, result);
    return result;
}

/**
 * Advance game by one move
 * @param {string} sessionId - Session ID
 * @param {object|null} move - Move object (required for human player, null for AI)
 * @returns {Promise<{status: string, current_player: number}>}
 */
export async function advance(sessionId, move = null) {
    const params = { session_id: sessionId, move };

    const response = await fetch(`${API_BASE_URL}/advance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });

    if (!response.ok) {
        const error = await response.json();
        logAPI('POST', '/advance', params, { error });
        throw new Error(error.error || 'Failed to advance game');
    }

    const result = await response.json();
    logAPI('POST', '/advance', params, result);
    return result;
}
