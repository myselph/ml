import React from 'react';
import './ControlBar.css';

/**
 * ControlBar component
 * Top control bar with game controls
 * 
 * @param {function} onNewGame - New game button handler
 * @param {function} onStep - Step button handler
 * @param {function} onPause - Pause button handler
 * @param {function} onContinue - Continue button handler
 * @param {function} onDebugToggle - Debug toggle handler
 * @param {function} onSpeedChange - Speed change handler (speed) => void
 * @param {string} playMode - 'auto' or 'manual'
 * @param {boolean} debugMode - Whether debug mode is active
 * @param {number} speed - Current speed (0.5, 1, or 2)
 */
export default function ControlBar({
    onNewGame,
    onStep,
    onPause,
    onContinue,
    onDebugToggle,
    onSpeedChange,
    playMode = 'manual',
    debugMode = false,
    speed = 1
}) {
    return (
        <div className="control-bar" data-testid="control-bar">
            <button
                className="control-button"
                onClick={onNewGame}
                data-testid="new-game-button"
            >
                New Game
            </button>

            <div className="control-group">
                {playMode === 'manual' ? (
                    <>
                        <button
                            className="control-button icon-button"
                            onClick={onStep}
                            data-testid="step-button"
                            title="Step"
                        >
                            ⏭️
                        </button>
                        <button
                            className="control-button icon-button"
                            onClick={onContinue}
                            data-testid="continue-button"
                            title="Continue"
                        >
                            ▶️
                        </button>
                    </>
                ) : (
                    <button
                        className="control-button icon-button"
                        onClick={onPause}
                        data-testid="pause-button"
                        title="Pause"
                    >
                        ⏸️
                    </button>
                )}
            </div>


            <button
                className="control-button speed-toggle"
                onClick={onSpeedChange}
                data-testid="speed-toggle-button"
            >
                {speed}x
            </button>

            <button
                className={`control-button debug-button ${debugMode ? 'active' : ''}`}
                onClick={onDebugToggle}
                data-testid="debug-button"
            >
                🐞
            </button>
        </div>
    );
}
