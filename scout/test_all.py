# Tests for scout
from dataclasses import dataclass
from abc import abstractmethod
from game_state import GameState
from common import Util, Scout, Show, ScoutAndShow, InformationState


def test_utils():
    game_state = GameState(5, 0)
    # Tests for is_group, is_run.
    assert Util.is_group([])
    assert Util.is_group([1])
    assert Util.is_group([1, 1])
    assert not Util.is_group([1, 2])
    assert not Util.is_group([1, 1, 2])
    assert Util.is_run([])
    assert Util.is_run([1])
    assert Util.is_run([1, 2])
    assert Util.is_run([1, 2, 3])
    assert Util.is_run([3, 2, 1])
    assert not Util.is_run([1, 1])
    assert not Util.is_run([1, 3])
    assert not Util.is_run([1, 2, 4])
    assert not Util.is_run([1, 3, 3])

    # Tests for is_move_valid.
    # Scouts
    assert not Util.is_move_valid([(1, 2)], [], False, Scout(False, False, 0))
    assert not Util.is_move_valid([(1, 2)], [], False, Scout(False, False, 1))
    table = [(2, 9), (3, 4), (4, 1)]
    assert Util.is_move_valid([(5, 6)], table, False, Scout(True, True, 0))
    assert Util.is_move_valid([(5, 6)], table, False, Scout(True, True, 1))
    assert not Util.is_move_valid([(5, 6)], table, False, Scout(True, True, 2))
    # Shows - groups vs. run
    assert Util.is_move_valid(
        [(1, 6), (1, 7), (1, 8)], table, False, Show(0, 3))
    assert Util.is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                              table, False, Show(0, 3))
    assert Util.is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                              table, False, Show(0, 4))
    assert Util.is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                              table, False, Show(1, 3))
    assert not Util.is_move_valid(
        [(1, 6), (1, 7), (1, 8), (1, 9)], table, False, Show(1, 2))
    # Shows - runs vs. run
    assert not Util.is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(0, 3))
    assert Util.is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                               (1, 9)], table, False, Show(1, 3))
    assert Util.is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                               (1, 9)], table, False, Show(0, 4))
    assert not Util.is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(1, 4))
    assert Util.is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                               (1, 9)], table, False, Show(0, 4))
    assert not Util.is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(2, 3))

    # Scout and Show.
    hand = [(3, 1), (5, 8), (6, 9)]
    # 3,4,5,6 wins over 2,3
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 4)))
    assert not Util.is_move_valid(hand, table, False, ScoutAndShow(
        Scout(False, False, 1), Show(0, 4)))
    # 3,4,5 wins over 2,3
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 3)))
    # 3,4 wins over 2,3
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 2)))
    # 1, 3 is not valid
    assert not Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, True, 1), Show(0, 2)))
    # 2,3 loses to 3,4
    assert not Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, False, 0), Show(0, 2)))
    # 5,6 wins over 3,4
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, False, 0), Show(2, 2)))
    # illegal to play 3,5,6
    assert not Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, True, 0), Show(1, 3)))


def test_InformationState():
    # InformationState tests - specifically, the valid move generator.
    # 2 cards in hand, none on table -> only Shows.
    hand = [(4, 7), (5, 8)]
    info_state = InformationState(
        5, 0, 0, -1, hand, [], [2]*5, [0]*5, [True]*5, [])
    assert info_state.possible_moves() == [Show(0, 1), Show(0, 2), Show(1, 1)]

    # 2 cards in hand, 1 on table -> 6 Scouts, 3 Shows (4, 5, (4,5)),
    # For Scout & Shows:
    # 6 scout moves; for each, 3 single card shows -> 18; also:
    # 4 (4,5) shows (insert 1 or 3, left or right)
    # 1 (3,4,5) show
    # 1 (3,4) show, one (4,3) show
    # So overall, 25 S&S moves.
    info_state = InformationState(
        5, 0, 0, -1, hand, [(3, 1)], [2]*5, [0]*5, [True]*5, [])
    moves = info_state.possible_moves()
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
        5, 0, 0, 0, hand, [(2, 1), (3, 1)], [2]*5, [0]*5, [True]*5, [])
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


def test_GameState():
    # GameState tests. TODO: Add test c'tor to inject my own decks;
    # for now, just test scouting and showing and that scoring works.
    game_state = GameState(5, 1)
    assert not game_state.table
    assert not game_state.initial_flip_executed
    game_state.maybe_flip_hand([lambda _: False]*5)
    assert game_state.scores[1] == -9
    game_state.move(Show(0, 1))
    assert game_state.scores[1] == -8
    game_state.move(Scout(True, False, 0))
    assert game_state.scores[1] == -7
    assert game_state.scores[2] == -10

    # Test the GameState generator. Make a couple of moves, create a
    # determinization, and ensure it is a valid representation.
    game_state = GameState(5, 0)
    card1 = game_state.hands[0][0]
    game_state.maybe_flip_hand([lambda _: False]*5)
    game_state.move(Show(0, 1))
    game_state.move(Scout(True, False, 0))
    card2 = game_state.hands[2][0]
    game_state.move(Show(0, 1))
    game_state.move(Scout(True, True, 3))
    card3 = game_state.hands[4][4]
    game_state.move(Show(4, 1))
    game_state.move(ScoutAndShow(Scout(True, True, 6), Show(6, 1)))
    game_state.move(Scout(True, True, 2))
    
    info_state = game_state.info_state()
    determinization = GameState.sample_from_info_state(info_state)
    assert [len(h) for h in determinization.hands] == [len(h)
                                                       for h in game_state.hands]
    assert not determinization.table
    assert card1 == determinization.hands[1][0]
    assert (card2[1], card2[0]) == determinization.hands[3][3]
    assert card3 == determinization.hands[1][2]
    assert info_state.hand == determinization.hands[2]
