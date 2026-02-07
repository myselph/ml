import React, { useState } from 'react';
import Card from './Card';
import './FlipHandModal.css';

/**
 * FlipHandModal component
 * Modal for initial hand flip decision with preview
 * 
 * @param {boolean} isOpen - Whether modal is open
 * @param {array} hand - Player's initial hand
 * @param {function} onFlip - Handler for flip decision (flip: boolean) => void
 */
export default function FlipHandModal({ isOpen, hand = [], onFlip }) {
    const [isFlipped, setIsFlipped] = useState(false);

    if (!isOpen) return null;

    // Display hand based on flip state
    const displayHand = isFlipped
        ? hand.map(card => [card[1], card[0]]) // Swap top and bottom
        : hand;

    const handleConfirm = () => {
        onFlip(isFlipped);
        setIsFlipped(false); // Reset for next time
    };

    return (
        <div className="modal-overlay" data-testid="flip-hand-modal">
            <div className="modal-content">
                <h2>Your Initial Hand</h2>
                <p>Preview your hand flipped or not, then confirm to start the game.</p>

                <div className="modal-cards">
                    {displayHand.map((card, index) => (
                        <Card
                            key={index}
                            top={card[0]}
                            bottom={card[1]}
                            dataTestId={`flip-hand-card-${index}`}
                        />
                    ))}
                </div>

                <div className="modal-buttons">
                    <button
                        className={`modal-button ${isFlipped ? 'active' : ''}`}
                        onClick={() => setIsFlipped(!isFlipped)}
                        data-testid="flip-toggle-button"
                    >
                        {isFlipped ? 'Unflip' : 'Flip'} Preview
                    </button>
                    <button
                        className="modal-button primary"
                        onClick={handleConfirm}
                        data-testid="confirm-flip-button"
                    >
                        Confirm {isFlipped ? '(Flipped)' : '(Not Flipped)'}
                    </button>
                </div>
            </div>
        </div>
    );
}
