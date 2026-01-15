# A Scout (card game) simulator.
from dataclasses import dataclass
import time
from abc import abstractmethod
from typing import Callable
from game_state import GameState
from common import Player
from players import PlanningPlayer, GreedyShowPlayerWithFlip, RandomPlayer
from ismcts_player import IsmctsPlayer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_games",
    type=int,
    help="Number of games per setting",
    default=100
)
parser.add_argument(
    '--num_rollouts', 
    type=lambda s: [int(item) for item in s.split(',')],
    help="Comma-separated list of the number of MCTS rollouts to run"
)
args = parser.parse_args()

# Play a single round - that is, a single deck of cards - and return
# the scores.
def play_round(players: list[Player], dealer: int) -> list[int]:
    game_state = GameState(len(players), dealer)
    game_state.maybe_flip_hand([p.flip_hand for p in players])
    current_player = dealer
    while not game_state.is_finished():
        info_state = game_state.info_state()
        move = players[current_player].select_move(info_state)
        game_state.move(move)
        current_player = (current_player + 1) % len(players)
    return game_state.scores


def play_game(players: list[Player]) -> list[int]:
    # Play a game - that is num_players rounds, with each player dealing once.
    # Return the cumulative scores.
    scores = []
    for dealer in range(0, len(players)):
        scores.append(play_round(players, dealer))
    return [sum(y) for y in zip(*scores)]


# Let two Player implementations compete against each other in a tournament.
# Returns ratio of player A wins. The precise definition of that metric is
# implementation specific, see below.
def play_tournament(
    player_a_factory_fn: Callable[[],
                                  Player],
    player_b_factory_fn: Callable[[],
                                  Player]) -> float:
    # For now: let one player A compete against 4 player B's. There's a lot of
    # other types of setups we could use, such as 2A vs 3B, with different
    # player sequences; other numbers of players (3, 4); or average between
    # the 1-vs-4 and 4-vs-1 setting. But until I get a better
    # understanding for how to produce rankings for multi-player games, this
    # will do.
    # The resulting scores get normalized - that is, we multiply the win rate
    # of A by 4 before computing the ratio.
    # At the very least, this should be symmetric, ie playing A against B should
    # give the complement (1-p) of playing B against A.
    players = [player_a_factory_fn()] + [player_b_factory_fn()
                                         for _ in range(4)]
    wins = [0] * len(players)

    start_time = time.time()
    num_games = args.num_games
    for reps in range(0, num_games):
        scores = play_game(players)
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        wins[winner_index] += 1
    end_time = time.time()

    a_win_rate = wins[0] / (wins[0] + sum(wins[1:]) / 4)
    wins = list(map(lambda i: i / sum(wins), wins))
    print(
        f"wins %: {wins}, a_win_rate normalized: {
            a_win_rate:.3f}, dt/game: {
            (
                end_time -
                start_time) /
            num_games}")
    return a_win_rate


def main():
    for i in args.num_rollouts:
        awr = play_tournament(lambda: IsmctsPlayer(5, i),
                        lambda: GreedyShowPlayerWithFlip())
        print(f"{i}: {awr}")


if __name__ == '__main__':
    main()
