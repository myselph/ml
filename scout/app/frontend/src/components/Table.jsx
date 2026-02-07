import React from 'react';
import Card from './Card';
import './Table.css';

/**
 * Table component
 * Displays the cards currently on the table
 * 
 * @param {array} cards - Array of [top, bottom] card tuples
 * @param {number|null} selectedIndex - Index of selected card (0 for left, cards.length-1 for right)
 * @param {boolean} selectedCardFlip - Whether selected card should show flip indicator
 * @param {function} onCardClick - Click handler for cards (index) => void
 * @param {function} onBackgroundClick - Click handler for table background (cancel action)
 * @param {boolean} isInteractive - Whether cards are clickable
 */
export default function Table({
    cards = [],
    selectedIndex = null,
    selectedCardFlip = false,
    onCardClick = null,
    onBackgroundClick = null,
    isInteractive = false
}) {
    const handleTableClick = (e) => {
        // If clicking on table background (not a card), trigger background click
        if (e.target.classList.contains('table') ||
            e.target.classList.contains('table-cards') ||
            e.target.classList.contains('table-label')) {
            if (onBackgroundClick) {
                onBackgroundClick();
            }
        }
    };

    return (
        <div className="table" data-testid="table" onClick={handleTableClick}>
            <div className="table-label">Table</div>
            <div className="table-cards">
                {cards.length === 0 ? (
                    <div className="table-empty">No cards on table</div>
                ) : (
                    cards.map((card, index) => (
                        <Card
                            key={index}
                            top={card[0]}
                            bottom={card[1]}
                            isSelected={selectedIndex === index}
                            isFlipped={selectedIndex === index && selectedCardFlip}
                            onClick={isInteractive && onCardClick ? () => onCardClick(index) : null}
                            dataTestId={`table-card-${index}`}
                        />
                    ))
                )}
            </div>
        </div>
    );
}
