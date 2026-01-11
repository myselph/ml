# A Scout (card game) simulator.
import numpy as np
import random
from dataclasses import dataclass
import time


def generate_hands(num_players: int):
    # The whole deck consists of all pairs of 1-10 (sampling w/o replacement), ie 45 cards:
    # [(1,2), (1,3), ..., (2,1), ..., (8,9), (8,10), (9,10)], but which cards are used
    # depends on the number of players. For now I only support 3-5.
    full_deck = [(i, j) for i in range(1, 10) for j in range(i, 11) if i != j]
    if num_players < 3 or num_players > 5:
        raise "Only 3-5 players supported"
    if num_players == 3:
        # 36 cards, skip the 10s -> 12 cards/player
        N = 12
        deck = [r for r in full_deck if r[1] != 10]
    elif num_players == 4:
        # skip (9,10) -> 11 cards / player
        N = 11
        deck = full_deck[:-1]
    else:
        N = 9
        deck = full_deck
    # Shuffle and serve.
    random.shuffle(deck)
    return [deck[i*N:(i+1)*N] for i in range(0, num_players)]


# Classes that represent moves a player can make.
@dataclass
class Scout:
    firstCard: bool
    flipCard: bool
    insertPosition: int


@dataclass
class Show:
    startPos: int
    length: int


@dataclass
class ScoutAndShow:
    scout: Scout
    show: Show


Move = Scout | Show | ScoutAndShow

# Helper functions to determine whether a move is legal.
# The entry point is is_move_valid; best to proceed from there to understand what's going on.


def is_group(cards: list[int]):
    # First check if it is a group (cards with the same )
    is_group = True
    for i in range(1, len(cards)):
        if cards[i] != cards[0]:
            is_group = False
            break
    return is_group


def is_run(cards: list[int]):
    # Check if it is an ascending run.
    is_ascending_run = True
    for i in range(1, len(cards)):
        if cards[i] != cards[i-1] + 1:
            is_ascending_run = False
            break
    if is_ascending_run:
        return True
    # Check if it's a descending run.
    is_descending_run = True
    for i in range(1, len(cards)):
        if cards[i] != cards[i-1] - 1:
            is_descending_run = False
            break
    return is_descending_run


def is_scout_valid(hand_values: list[int], table_values: list[int], scout: Scout):
    return table_values and scout.insertPosition >= 0 and scout.insertPosition <= len(hand_values)


def is_show_valid(hand_values: list[int], table_values: list[int], show: Show):
    # Basic range checks
    if show.startPos < 0 or show.length < 1 or show.startPos+show.length > len(hand_values):
        return False
    if show.length < len(table_values):
        return False

    # The meld (cards being played) must be either a group or a run.
    meld_values = hand_values[show.startPos:show.startPos+show.length]
    meld_is_group = is_group(meld_values)
    if not meld_is_group:
        meld_is_run = is_run(meld_values)
    if not meld_is_group and not meld_is_run:
        return False
    # If the meld is longer than what's on the table, it's a legal move.
    if show.length > len(table_values):
        return True

    # If the number of cards on the table and in the meld are the same, and we need to decide which one wins.
    # Groups win over runs, higher groups win over lower groups, higher runs win over lower runs.
    # NB table is guaranteed to be either a group or a run by induction - groups and runs are the only
    # valid melds to be played, and any subsequent action either replaces them with another group or meld,
    # or scouts cards such that the group / meld property is retained.
    table_is_group = is_group(table_values)
    if meld_is_group:
        return not table_is_group or table_values[0] < meld_values[0]
    else:
        return not table_is_group and max(table_values) < max(meld_values)


def is_move_valid(hand: list[tuple[int, int]], table: list[tuple[int, int]], can_scout_and_show: bool, move: Move):
    hand_values = [h[0] for h in hand]
    table_values = [t[0] for t in table]
    if isinstance(move, Scout):
        return is_scout_valid(hand_values, table_values, move)
    elif isinstance(move, Show):
        return is_show_valid(hand_values, table_values, move)
    else:  # Scout & Show
        if not can_scout_and_show:
            return False
        if not is_scout_valid(hand_values, table_values, move.scout):
            return False
        # Simulate the Scout move
        if move.scout.firstCard:  # pick first card or last?
            table_values = table_values[1:]
            # Flip card or not?
            scouted_value = table[0][1] if move.scout.flipCard else table[0][0]
        else:
            table_values = table_values[:-1]
            scouted_value = table[-1][1] if move.scout.flipCard else table[-1][0]
        hand_values.insert(move.scout.insertPosition, scouted_value)
        # Check the Show move.
        return is_show_valid(hand_values, table_values, move.show)


# InformationState is a class that represents the information available to a single player.
# It is therefore a subset of (and constructed from) the entire game state, notably
# excluding what cards the other players have.
# This is the information that player implementations can use to decide what moves to make.
# The information encoded within is somewhat overcomplete; e.g. can_scout_and_show could be
# derived from the history; but it is more convenient (and easier to construct) that way.
# I don't love how this class has mostly (except hand) the same variables as GameState,
# and shares the underlying memory (ie as GameState changes, so will this class), and may
# rework it in the future; for now, this class must only be used by a player until it executes
# a move.
@dataclass
class InformationState:
    num_players: int
    dealer: int
    current_player: int
    scout_benefactor: int
    hand: list[tuple[int, int]]
    table: list[tuple[int, int]]
    scores: list[int]
    can_scout_and_show: list[bool]
    history: list[Move]

    def sample_game_state(self):
        # Returns a random GameState consistent with the information state.
        # This function can be used to run ISMCTS explorations, i.e. sample many possible
        # game states and run simulations on each to get an aggregate tree.
        # What needs to be sampled to get to a GameState is what cards the other players hold,
        # and that sample needs to be consistent with all the observations taken so far.
        # We narrow down those choices by
        # 1) sampling a deck
        # 2) removing the cards known to be held by the current player or being on the table.
        # 3) going through history to remove cards that have been removed from the game
        #    in Show and Scout & Show moves
        # 4) build a card -> player map from history, tracking cards that have been scouted.
        #    Using a map and going through history allows us to only consider the last
        #    scout for a given card.
        # 5) Go through history to count how many cards each player has.
        # We then end up with a set of N cards of unknown assignment, which we assign to the
        # remaining players such that they have the correct number of cards.
        # I am not sure if this is guaranteed to be a valid aissngmnet but I think so. Might
        # be possible to confirm via simulations.
        raise "Not implemented"

    def possible_moves(self):
        # Return a list of legal moves the player can make.
        # This can be used in a policy to pick a move, e.g. by randomly sampling moves,
        # or ranking them with heuristics or learned functions.
        # First, generate Scout candidates.
        firstCardOptions = [True, False] if len(self.table) >1 else [True]
        scout_candidates = [Scout(first, flip, insertPos) for first in firstCardOptions for flip in [
            False, True] for insertPos in range(0, len(self.hand)+1)]
        # TODO: Skip the below, but add a check that the table isn't empty.
        scouts = [s for s in scout_candidates if is_move_valid(
            self.hand, self.table, self.can_scout_and_show[self.current_player], s)]

        # Show candidates - generate possible ones (at least as many cards as there are on the table),
        # then filter by validity (group or sequence?)
        show_candidates = [Show(start, length) for start in range(
            0, len(self.hand) - (len(self.table) - 1)) for length in range(len(self.table), len(self.hand) + 1 - start)]
        shows = [s for s in show_candidates if is_move_valid(
            self.hand, self.table, self.can_scout_and_show[self.current_player], s)]

        # Scout and Show candidates.
        # This could probably be sped up somehow.
        # TODO: Coalesce combos that lead to the same outcome - specifically, showing the same
        # card that was just scouted should count as a single option, but leads to len(hand)+1
        # separate moves due to the insert positions.
        scout_and_shows = []
        if self.can_scout_and_show[self.current_player]:
            # Generate possible ranges for the Show moves - like above, but assuming the table has
            # one card less and our hand has one card more from the Scout move. The index math below
            # hurts my head. Tests FTW.            
            show_moves = [Show(start, length) for start in range(
                0, len(self.hand) - (len(self.table) - 1 - 1) + 1) for length in range(len(self.table) - 1, len(self.hand) + 1 - start + 1)]
            for scout in scouts:
                for show in show_moves:
                    move = ScoutAndShow(scout, show)
                    if is_move_valid(self.hand, self.table, self.can_scout_and_show[self.current_player], move):
                        scout_and_shows.append(move)

        return scouts + shows + scout_and_shows


class GameState:
    num_players: int
    dealer: int  # index of first player
    current_player: int  # index of next player to call move() and info_state()
    scout_benefactor: int # index of player who did the last Show or ScoutAndShow, for accounting.
    hands: list[list[tuple[int, int]]]
    table: list[tuple[int, int]]
    scores: list[int] # Per-player running scores: scout points + cards collected - cards in hand.
    can_scout_and_show: list[bool] # Whether a player has used their Scout & Show capability yet.
    history: list[Move]  # History of moves.
    finished: bool  # Whether the game is over.

    def __init__(self, num_players: int, dealer: int):
        self.num_players = num_players
        self.hands = generate_hands(num_players)
        self.scores = [-len(h) for h in self.hands]
        self.can_scout_and_show = [True] * num_players
        self.dealer = dealer
        self.current_player = self.dealer
        self.scout_benefactor = -1
        self.table = []
        self.history = []
        self.finished = False

    def move(self, m: Move):
        assert not self.finished
        assert is_move_valid(self.hands[self.current_player], self.table,
                             self.can_scout_and_show[self.current_player], m)
        if isinstance(m, Scout):
            self._scout(m)
            if (self.current_player + 1) % self.num_players == self.scout_benefactor:
                self.finished = True
        elif isinstance(m, Show):
            self._show(m)
        else:
            self._scout(m.scout)
            self._show(m.show)
            self.can_scout_and_show[self.current_player] = False
        if not self.hands[self.current_player]:
            self.finished = True
        self.history.append(m)
        self.current_player = (self.current_player + 1) % self.num_players

    def is_finished(self):
        return self.finished

    def info_state(self):
        # Returns the information state for the current player.
        return InformationState(
            self.num_players, self.dealer, self.current_player, self.scout_benefactor,
            self.hands[self.current_player], self.table, self.scores, self.can_scout_and_show, self.history)

    def _scout(self, m: Scout):
        hand = self.hands[self.current_player]
        if m.firstCard:
            card = self.table[0]
            self.table = self.table[1:]
        else:
            card = self.table[-1]
            self.table = self.table[:-1]
        if m.flipCard:
            card = (card[1], card[0])
        hand.insert(m.insertPosition, card)
        self.scores[self.scout_benefactor] += 1
        self.scores[self.current_player] -= 1

    def _show(self, m: Show):
        hand = self.hands[self.current_player]
        self.scores[self.current_player] += len(self.table) + m.length
        self.table = hand[m.startPos:m.startPos+m.length]
        self.hands[self.current_player] = hand[:m.startPos] + \
            hand[m.startPos+m.length:]
        self.scout_benefactor = self.current_player



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
    info_state = InformationState(5, 0, 0, -1, hand, [(3,1)], [0]*5, [True]*5, [])
    moves = info_state.possible_moves()
    for m in moves:
        print(m)
    assert 6 == len([m for m in moves if isinstance(m, Scout)])
    assert 3 == len([m for m in moves if isinstance(m, Show)])
    assert 25 == len([m for m in moves if isinstance(m, ScoutAndShow)])
    
    # Expected: 12 scout moves; one show move (pair); and for S&S:
    # for each of the 12 scout moves, two single card shows (4&5)
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
    game_state.move(Show(0,1))
    assert game_state.scores[1] == -8
    game_state.move(Scout(True, False, 0))
    assert game_state.scores[1] == -7
    assert game_state.scores[2] == -10


    print("All tests passed")


def main():
    # How long does a game take when random moves are chosen?
    move_numbers = []
    start_time = time.time()
    num_games = 1000
    for reps in range(0, num_games):
        player = 0
        num_players = 5
        game_state = GameState(num_players, player)
        move_index = 0
        while not game_state.is_finished():            
            info_state = game_state.info_state()
            moves = info_state.possible_moves()
            # policy 1: random moves.
            # Interestingly I found that coalescing identical moves (firstCard=True/False
            # when there is only one card on the table) shortens the game; I assume this
            # bug made Scout moves more likely by overrepresenting them, which under random
            # sampling leads to longer games.
            move = random.choice(moves)
            # policy 2: pick the largest Show; and, if there is a ScoutAndShow that's > 2, pick that.
            # This siginficantly shortens games from ~113 moves down to ~34 moves.
            # TODO: Use max() instead of sorting and taking the last element.
            #shows = [m for m in moves if isinstance(m, Show)]
            #scout_and_shows = [m for m in moves if isinstance(m, ScoutAndShow)]
            #if shows:
            #    shows.sort(key=lambda s: s.length)
            #    move = shows[-1]
            #if scout_and_shows:
            #    scout_and_shows.sort(key=lambda s: s.show.length)
            #    if shows and scout_and_shows[-1].show.length > move.length + 2:
            #        move = scout_and_shows[-1]
            game_state.move(move)
            player = (player + 1) % num_players
            move_index += 1
        move_numbers.append(move_index)

    end_time = time.time()
    elapsed_time = end_time - start_time
    mean = np.mean(move_numbers)
    std = np.std(move_numbers)
    print(f"#moves: {mean}+-{std}; time,[ms]/move: {elapsed_time*1000/np.sum(move_numbers)}")


if __name__ == '__main__':
    #tests()
    main()
