# A Scout (card game) simulator.
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod
from typing import Callable
from game_state import GameState
from common import Player, is_group, is_run, is_move_valid, Scout, Show, ScoutAndShow, InformationState
from players import PlanningPlayer, GreedyShowPlayerWithFlip


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

# Play a game - that is num_players rounds, with each player dealing once.
# Return the cumulative scores.


def play_game(players: list[Player]) -> list[int]:
    scores = []
    for dealer in range(0, len(players)):
        scores.append(play_round(players, dealer))
    return [sum(y) for y in zip(*scores)]


# Let two Player implementations compete against each other in a tournament.
# Returns ratio of player A wins. The precise definition of that metric is
# implementation specific, see below.
def play_tournament(player_a_factory_fn: Callable[[], Player], player_b_factory_fn: Callable[[], Player]) -> float:
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
    players = [player_a_factory_fn()] + [player_b_factory_fn()] * 4
    wins = [0]*len(players)

    start_time = time.time()
    num_games = 200
    for reps in range(0, num_games):
        scores = play_game(players)
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        wins[winner_index] += 1
    end_time = time.time()

    a_win_rate = wins[0] / (wins[0] + sum(wins[1:])/4)
    wins = list(map(lambda i: i / sum(wins), wins))
    print(
        f"wins %: {wins}, a_win_rate normalized: {a_win_rate:.3f}, dt/game: {(end_time-start_time)/num_games}")
    return a_win_rate


def main():
    play_tournament(lambda: PlanningPlayer(), lambda: GreedyShowPlayerWithFlip())
            
    
# TODO: Move into separate file.
def tests():
    game_state = GameState(5, 0)
    # Tests for is_group, is_run.
    assert is_group([])
    assert is_group([1])
    assert is_group([1, 1])
    assert not is_group([1, 2])
    assert not is_group([1, 1, 2])
    assert is_run([])
    assert is_run([1])
    assert is_run([1, 2])
    assert is_run([1, 2, 3])
    assert is_run([3, 2, 1])
    assert not is_run([1, 1])
    assert not is_run([1, 3])
    assert not is_run([1, 2, 4])
    assert not is_run([1, 3, 3])

    # Tests for is_move_valid.
    # Scouts
    assert not is_move_valid([(1, 2)], [], False, Scout(False, False, 0))
    assert not is_move_valid([(1, 2)], [], False, Scout(False, False, 1))
    table = [(2, 9), (3, 4), (4, 1)]
    assert is_move_valid([(5, 6)], table, False, Scout(True, True, 0))
    assert is_move_valid([(5, 6)], table, False, Scout(True, True, 1))
    assert not is_move_valid([(5, 6)], table, False, Scout(True, True, 2))
    # Shows - groups vs. run
    assert is_move_valid([(1, 6), (1, 7), (1, 8)], table, False, Show(0, 3))
    assert is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                         table, False, Show(0, 3))
    assert is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                         table, False, Show(0, 4))
    assert is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                         table, False, Show(1, 3))
    assert not is_move_valid(
        [(1, 6), (1, 7), (1, 8), (1, 9)], table, False, Show(1, 2))
    # Shows - runs vs. run
    assert not is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(0, 3))
    assert is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                         (1, 9)], table, False, Show(1, 3))
    assert is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                         (1, 9)], table, False, Show(0, 4))
    assert not is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(1, 4))
    assert is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                         (1, 9)], table, False, Show(0, 4))
    assert not is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(2, 3))

    # Scout and Show. TODO:
    hand = [(3, 1), (5, 8), (6, 9)]
    # 3,4,5,6 wins over 2,3
    assert is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 4)))
    assert not is_move_valid(hand, table, False, ScoutAndShow(
        Scout(False, False, 1), Show(0, 4)))
    # 3,4,5 wins over 2,3
    assert is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 3)))
    # 3,4 wins over 2,3
    assert is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 2)))
    # 1, 3 is not valid
    assert not is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, True, 1), Show(0, 2)))
    # 2,3 loses to 3,4
    assert not is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, False, 0), Show(0, 2)))
    # 5,6 wins over 3,4
    assert is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, False, 0), Show(2, 2)))
    # illegal to play 3,5,6
    assert not is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, True, 0), Show(1, 3)))

    # InformationState tests - specifically, the valid move generator.
    # 2 cards in hand, none on table -> only Shows.
    hand = [(4, 7), (5, 8)]
    info_state = InformationState(5, 0, 0, -1, hand, [], [0]*5, [True]*5, [])
    assert info_state.possible_moves() == [Show(0, 1), Show(0, 2), Show(1, 1)]

    # 2 cards in hand, 1 on table -> 6 Scouts, 3 Shows (4, 5, (4,5)),
    # For Scout & Shows:
    # 6 scout moves; for each, 3 single card shows -> 18; also:
    # 4 (4,5) shows (insert 1 or 3, left or right)
    # 1 (3,4,5) show
    # 1 (3,4) show, one (4,3) show
    # So overall, 25 S&S moves.
    info_state = InformationState(
        5, 0, 0, -1, hand, [(3, 1)], [0]*5, [True]*5, [])
    moves = info_state.possible_moves()
    for m in moves:
        print(m)
    assert 6 == len([m for m in moves if isinstance(m, Scout)])
    assert 3 == len([m for m in moves if isinstance(m, Show)])
    assert 25 == len([m for m in moves if isinstance(m, ScoutAndShow)])

    # Expected: 12 scout moves; one show move (pair); and for S&S:
    # for each of the 12 scout moves, import timetwo single card shows (4&5)
    # for 3 of the scout "3" moves, a single card show (3);
    # for 8 of the scout moves, a double card show (4&5) - 8 scouts exist that do not break up that sequence
    # when inserting the 3 before the 4, two new show moves - "3,4" and "3,4,5"
    # when inserting the 3 after the for, a new double "4, 3".
    # So 38 ScoutAndShow moves.
    info_state = InformationState(
        5, 0, 0, 0, hand, [(2, 1), (3, 1)], [0]*5, [True]*5, [])
    moves = info_state.possible_moves()
    assert 12 == len([m for m in moves if isinstance(m, Scout)])
    assert 1 == len([m for m in moves if isinstance(m, Show)])
    assert 38 == len([m for m in moves if isinstance(m, ScoutAndShow)])
    assert 27 == len([m for m in moves if isinstance(
        m, ScoutAndShow) and m.show.length == 1])
    assert 10 == len([m for m in moves if isinstance(
        m, ScoutAndShow) and m.show.length == 2])
    assert 1 == len([m for m in moves if isinstance(
        m, ScoutAndShow) and m.show.length == 3])

    # GameState tests. TODO: Add test c'tor to inject my own decks;
    # for now, just test scouting and showing and that scoring works.
    game_state = GameState(5, 1)
    assert not game_state.table
    assert game_state.scores[1] == -9
    game_state.move(Show(0, 1))
    assert game_state.scores[1] == -8
    game_state.move(Scout(True, False, 0))
    assert game_state.scores[1] == -7
    assert game_state.scores[2] == -10

    print("All tests passed")


if __name__ == '__main__':
    # tests()
    main()
