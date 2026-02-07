import React from 'react';
import './Card.css';

/**
 * Card component
 * Displays a Scout card with top and bottom numbers
 * 
 * @param {number} top - Top number (1-10)
 * @param {number} bottom - Bottom number (1-10)
 * @param {boolean} isHidden - Whether to show card back instead of face
 * @param {boolean} isSelected - Whether card is currently selected
 * @param {boolean} isFlipped - Whether card shows flip indicator (for table cards)
 * @param {function} onClick - Click handler
 * @param {string} className - Additional CSS classes
 * @param {string} dataTestId - Test ID for UI testing
 */
export default function Card({
    top,
    bottom,
    isHidden = false,
    isSelected = false,
    isFlipped = false,
    onClick = null,
    className = '',
    dataTestId = ''
}) {
    const cardClasses = [
        'card',
        isHidden ? 'card-hidden' : 'card-visible',
        isSelected ? 'card-selected' : '',
        onClick ? 'card-clickable' : '',
        className
    ].filter(Boolean).join(' ');

    // If flipped, swap top and bottom values
    const displayTop = isFlipped ? bottom : top;
    const displayBottom = isFlipped ? top : bottom;

    return (
        <div
            className={cardClasses}
            onClick={onClick}
            data-testid={dataTestId}
        >
            {isHidden ? (
                <div className="card-back">?</div>
            ) : (
                <>
                    <div className="card-top">{displayTop}</div>
                    <div className="card-bottom">{displayBottom}</div>
                </>
            )}
        </div>
    );
}
