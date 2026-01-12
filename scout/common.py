# Self-contained module for shared types and functionality.
from dataclasses import dataclass
from abc import ABC, abstractmethod


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

# Utility functions to operate on a set or sets of cards.
class Util:
    def is_group(cards: list[int]):
        # Check if it is a group (cards with the same )
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
        meld_is_group = Util.is_group(meld_values)
        if not meld_is_group:
            meld_is_run = Util.is_run(meld_values)
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
        table_is_group = Util.is_group(table_values)
        if meld_is_group:
            return not table_is_group or table_values[0] < meld_values[0]
        else:
            return not table_is_group and max(table_values) < max(meld_values)

    def is_move_valid(hand: list[tuple[int, int]], table: list[tuple[int, int]], can_scout_and_show: bool, move: Move):
        hand_values = [h[0] for h in hand]
        table_values = [t[0] for t in table]
        if isinstance(move, Scout):
            return Util.is_scout_valid(hand_values, table_values, move)
        elif isinstance(move, Show):
            return Util.is_show_valid(hand_values, table_values, move)
        else:  # Scout & Show
            if not can_scout_and_show:
                return False
            if not Util.is_scout_valid(hand_values, table_values, move.scout):
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
            return Util.is_show_valid(hand_values, table_values, move.show)


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
@dataclass(frozen=True)
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
        firstCardOptions = [True, False] if len(self.table) > 1 else [True]
        scout_candidates = [Scout(first, flip, insertPos) for first in firstCardOptions for flip in [
            False, True] for insertPos in range(0, len(self.hand)+1)]
        # TODO: Skip the below, but add a check that the table isn't empty.
        scouts = [s for s in scout_candidates if Util.is_move_valid(
            self.hand, self.table, self.can_scout_and_show[self.current_player], s)]

        # Show candidates - generate possible ones (at least as many cards as there are on the table),
        # then filter by validity (group or sequence?)
        show_candidates = [Show(start, length) for start in range(
            0, len(self.hand) - (len(self.table) - 1)) for length in range(len(self.table), len(self.hand) + 1 - start)]
        shows = [s for s in show_candidates if Util.is_move_valid(
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
                    if Util.is_move_valid(self.hand, self.table, self.can_scout_and_show[self.current_player], move):
                        scout_and_shows.append(move)

        return scouts + shows + scout_and_shows


class Player(ABC):
    # Abstract base class. The player interface is very simple - the game engine
    # calls it with the subset of the game state the player could have observed,
    # and the player picks a move in what can be an arbitrarily complex process,
    # including starefulness by eg caching computation results.
    @abstractmethod
    def flip_hand(self, hand: list[tuple[int, int]]) -> bool:
        pass

    @abstractmethod
    def select_move(self, info_state: InformationState) -> Move:
        pass
