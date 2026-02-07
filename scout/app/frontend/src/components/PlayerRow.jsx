import React from 'react';
import Card from './Card';
import './PlayerRow.css';

/**
 * PlayerRow component
 * Displays a player's cards, score, and action buttons
 * 
 * @param {number} playerIndex - Player index
 * @param {array} hand - Player's hand (array of [top, bottom] cards)
 * @param {number} score - Player's score
 * @param {boolean} canScoutAndShow - Whether player can scout and show
 * @param {boolean} isHuman - Whether this is the human player
 * @param {boolean} isCurrentPlayer - Whether it's this player's turn
 * @param {boolean} debugMode - Whether to show AI cards
 * @param {array} selectedIndices - Array of selected card indices
 * @param {function} onCardClick - Click handler for cards
 * @param {array} insertionPoints - Array of insertion point indices for Scout mode
 * @param {function} onInsertionPointClick - Click handler for insertion points
 * @param {boolean} showShowButton - Whether to show the Show button
 * @param {function} onShowClick - Click handler for Show button
 * @param {boolean} showScoutAndShowButton - Whether to show Scout & Show button
 * @param {boolean} scoutAndShowActive - Whether Scout & Show mode is active
 * @param {function} onScoutAndShowClick - Click handler for Scout & Show button
 */
export default function PlayerRow({
    playerIndex,
    hand = [],
    score = 0,
    canScoutAndShow = false,
    isHuman = false,
    isCurrentPlayer = false,
    debugMode = false,
    selectedIndices = [],
    onCardClick = null,
    insertionPoints = [],
    onInsertionPointClick = null,
    showShowButton = false,
    onShowClick = null,
    showScoutAndShowButton = false,
    scoutAndShowActive = false,
    onScoutAndShowClick = null
}) {
    const showCards = isHuman || debugMode;
    const isInteractive = isHuman && isCurrentPlayer;

    return (
        <div
            className={`player-row ${isCurrentPlayer ? 'player-row-active' : ''}`}
            data-testid={`player-row-${playerIndex}`}
        >
            {/* Header row with player info, score, S&S tag, and buttons */}
            <div className="player-header">
                <div className="player-label">
                    {isHuman ? 'You' : `Player ${playerIndex + 1}`}
                    {isCurrentPlayer && <span className="turn-indicator"> ← Turn</span>}
                </div>

                <div className="player-score" data-testid={`player-${playerIndex}-score`}>
                    Score: {score}
                </div>

                <div
                    className={`scout-show-indicator ${canScoutAndShow ? 'active' : 'inactive'}`}
                    data-testid={`player-${playerIndex}-ss-indicator`}
                >
                    S&S
                </div>

                {/* Action buttons for human player */}
                {isHuman && isCurrentPlayer && (
                    <div className="player-actions">
                        {showShowButton && (
                            <button
                                className="action-button show-button"
                                onClick={onShowClick}
                                data-testid="show-button"
                            >
                                Show
                            </button>
                        )}

                        {showScoutAndShowButton && (
                            <button
                                className={`action-button scout-show-button ${scoutAndShowActive ? 'active' : ''}`}
                                onClick={onScoutAndShowClick}
                                data-testid="scout-and-show-button"
                            >
                                {scoutAndShowActive ? 'Cancel S&S' : 'Scout & Show'}
                            </button>
                        )}
                    </div>
                )}
            </div>

            {/* Cards row */}
            <div className="player-hand">
                {hand.map((card, index) => (
                    <React.Fragment key={index}>
                        {/* Insertion point before card */}
                        {insertionPoints.includes(index) && (
                            <div
                                className="insertion-point"
                                onClick={() => onInsertionPointClick && onInsertionPointClick(index)}
                                data-testid={`insertion-point-${index}`}
                            >
                                <div className="insertion-point-marker">+</div>
                            </div>
                        )}

                        <Card
                            top={card[0]}
                            bottom={card[1]}
                            isHidden={!showCards}
                            isSelected={selectedIndices.includes(index)}
                            onClick={isInteractive && onCardClick ? () => onCardClick(index) : null}
                            dataTestId={`player-${playerIndex}-card-${index}`}
                        />
                    </React.Fragment>
                ))}

                {/* Insertion point after last card */}
                {insertionPoints.includes(hand.length) && (
                    <div
                        className="insertion-point"
                        onClick={() => onInsertionPointClick && onInsertionPointClick(hand.length)}
                        data-testid={`insertion-point-${hand.length}`}
                    >
                        <div className="insertion-point-marker">+</div>
                    </div>
                )}
            </div>
        </div>
    );
}
