/**
 * Move validation and hand manipulation utilities
 */

/**
 * Check if a Scout move is valid
 * @param {array} possibleMoves - Array of possible moves from backend
 * @param {boolean} first - Whether selecting first (left) card
 * @param {boolean} flip - Whether to flip the card
 * @param {number} insertPos - Position to insert in hand
 * @returns {boolean}
 */
export function isValidScout(possibleMoves, first, flip, insertPos) {
    if (!possibleMoves) return false;

    return possibleMoves.some(move =>
        move.type === 'scout' &&
        move.first === first &&
        move.flip === flip &&
        move.insertPos === insertPos
    );
}

/**
 * Check if a Show move is valid
 * @param {array} possibleMoves - Array of possible moves from backend
 * @param {number} startPos - Starting position in hand
 * @param {number} length - Number of cards to show
 * @returns {boolean}
 */
export function isValidShow(possibleMoves, startPos, length) {
    if (!possibleMoves) return false;

    return possibleMoves.some(move =>
        move.type === 'show' &&
        move.startPos === startPos &&
        move.length === length
    );
}

/**
 * Find a matching ScoutAndShow move
 * @param {array} possibleMoves - Array of possible moves from backend
 * @param {object} scoutMove - Scout move object {first, flip, insertPos}
 * @param {number} showStartPos - Show start position (in temporary hand after scout)
 * @param {number} showLength - Show length
 * @returns {object|null} - The matching scout_and_show move or null
 */
export function findScoutAndShowMove(possibleMoves, scoutMove, showStartPos, showLength) {
    if (!possibleMoves) return null;

    return possibleMoves.find(move =>
        move.type === 'scout_and_show' &&
        move.scout.first === scoutMove.first &&
        move.scout.flip === scoutMove.flip &&
        move.scout.insertPos === scoutMove.insertPos &&
        move.show.startPos === showStartPos &&
        move.show.length === showLength
    ) || null;
}

/**
 * Compute temporary hand after a Scout move
 * @param {array} hand - Current hand (array of [top, bottom] cards)
 * @param {array} table - Current table (array of [top, bottom] cards)
 * @param {boolean} first - Whether scouting first (left) card
 * @param {boolean} flip - Whether to flip the card
 * @param {number} insertPos - Position to insert in hand
 * @returns {array} - New hand after scout
 */
export function computeTempHandAfterScout(hand, table, first, flip, insertPos) {
    // Get the card from table
    const scoutedCard = first ? table[0] : table[table.length - 1];

    // Flip if needed
    const cardToInsert = flip ? [scoutedCard[1], scoutedCard[0]] : scoutedCard;

    // Insert into hand at specified position
    const newHand = [...hand];
    newHand.splice(insertPos, 0, cardToInsert);

    return newHand;
}

/**
 * Check if there are any scout_and_show moves available
 * @param {array} possibleMoves - Array of possible moves from backend
 * @returns {boolean}
 */
export function hasScoutAndShowMoves(possibleMoves) {
    if (!possibleMoves) return false;
    return possibleMoves.some(move => move.type === 'scout_and_show');
}
