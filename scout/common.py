# Self-contained module for shared types and functionality.
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Sequence

type Card = tuple[int, int]

# Classes that represent moves a player can make.


@dataclass(frozen=True)
class Scout:
    first: bool
    flip: bool
    insertPos: int


@dataclass(frozen=True)
class Show:
    startPos: int
    length: int


@dataclass(frozen=True)
class ScoutAndShow:
    scout: Scout
    show: Show


type Move = Scout | Show | ScoutAndShow


class Util:
    # Utility functions to operate on a set or sets of cards.
    @staticmethod
    def is_group(cards: Sequence[int]):
        # Check if it is a group (cards with the same )
        is_group = True
        for i in range(1, len(cards)):
            if cards[i] != cards[0]:
                is_group = False
                break
        return is_group

    @staticmethod
    def is_run(cards: Sequence[int]):
        # Check if it is an ascending run.
        is_ascending_run = True
        for i in range(1, len(cards)):
            if cards[i] != cards[i - 1] + 1:
                is_ascending_run = False
                break
        if is_ascending_run:
            return True
        # Check if it's a descending run.
        is_descending_run = True
        for i in range(1, len(cards)):
            if cards[i] != cards[i - 1] - 1:
                is_descending_run = False
                break
        return is_descending_run

    @staticmethod
    def is_scout_valid(
            hand_values: Sequence[int],
            table_values: Sequence[int],
            scout: Scout):
        return table_values and scout.insertPos >= 0 and scout.insertPos <= len(
            hand_values)

    @staticmethod
    def is_show_valid(
            hand_values: Sequence[int],
            table_values: Sequence[int],
            show: Show):
        # Basic range checks
        if show.startPos < 0 or show.length < 1 or show.startPos + \
                show.length > len(hand_values):
            return False
        if show.length < len(table_values):
            return False

        # The meld (cards being played) must be either a group or a run.
        meld_values = hand_values[show.startPos:show.startPos + show.length]
        meld_is_group = Util.is_group(meld_values)
        if not meld_is_group:
            if not Util.is_run(meld_values):
                return False
        # If the meld is longer than what's on the table, it's a legal move.
        if show.length > len(table_values):
            return True

        # If the number of cards on the table and in the meld are the same, and
        # we need to decide which one wins. Groups win over runs, higher groups
        # win over lower groups, higher runs win over lower runs. NB table is
        # guaranteed to be either a group or a run by induction - groups and
        # runs are the only valid melds to be played, and any subsequent action
        # either replaces them with another group or meld, or scouts cards such
        # that the group / meld property is retained.
        table_is_group = Util.is_group(table_values)
        if meld_is_group:
            return not table_is_group or table_values[0] < meld_values[0]
        else:
            return not table_is_group and max(table_values) < max(meld_values)

    @staticmethod
    def is_move_valid(
            hand: Sequence[Card],
            table: Sequence[Card],
            can_scout_and_show: bool,
            move: Move):
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
            if move.scout.first:  # pick first card or last?
                table_values = table_values[1:]
                # Flip card or not?
                scouted_value = table[0][1] if move.scout.flip else table[0][0]
            else:
                table_values = table_values[:-1]
                scouted_value = table[-1][1] if move.scout.flip else table[-1][0]
            hand_values.insert(move.scout.insertPos, scouted_value)
            # Check the Show move.
            return Util.is_show_valid(hand_values, table_values, move.show)


@dataclass(frozen=True)
class RecordedScout:
    move: Scout
    card: Card


@dataclass(frozen=True)
class RecordedShow:
    move: Show
    shown: tuple[Card, ...]
    removed: tuple[Card, ...]


@dataclass(frozen=True)
class RecordedScoutAndShow:
    scout: RecordedScout
    show: RecordedShow


type RecordedMove = RecordedScout | RecordedShow | RecordedScoutAndShow


@dataclass(frozen=True)
class InformationState:
    # InformationState is a class that represents the information available to a
    # single player. It is therefore a subset of (and constructed from) the entire
    # game state, notably excluding what cards the other players have (though a
    # subset of that can be reconstructed from the moves they made.
    # Notably, the details of every player's moves - position information, card
    # flipping - are public knowledge (ie a player cannot hide where a scouted card
    # is inserted, if it's flipped, etc.).
    # InformationState needs to be hashable so we can use it as key in dictionaries
    # (necessary in ISMCTS). This requirs converting the lists in GameState to
    # tuples and vice versa.
    num_players: int
    dealer: int
    current_player: int
    scout_benefactor: int
    hand: tuple[Card, ...]
    table: tuple[Card, ...]
    num_cards: tuple[int, ...]
    scores: tuple[int, ...]
    can_scout_and_show: tuple[bool, ...]
    history: tuple[RecordedMove, ...]

    def possible_moves(self) -> tuple[Move, ...]:
        # Return a list of legal moves the player can make.
        # First, generate Scout candidates. We do not call is_move_valid to
        # speed things up, instead simply avoiding invalid moves by
        # construction
        if self.table:
            firstCardOptions = [True, False] if len(self.table) > 1 else [True]
            scouts = [
                Scout(
                    first, flip, insertPos)
                for first in firstCardOptions
                for flip in [False, True]
                for insertPos in range(0, len(self.hand) + 1)]
        else:
            scouts = []

        # Show candidates - generate possible ones (at least as many cards as
        # there are on the table), then filter by validity (group or sequence?)
        show_candidates = [Show(start, length)
                           for start in range(0, len(self.hand) - (len(self.table) - 1))
                           for length in range(len(self.table), len(self.hand) + 1 - start)]
        shows = [s for s in show_candidates if Util.is_move_valid(
            self.hand, self.table, self.can_scout_and_show[self.current_player], s)]

        # Scout and Show candidates.
        # This could probably be sped up somehow.
        scout_and_shows = []
        # TODO: More coallescing - when inserting a scouted card right next to
        # a sequence that will be shown, it doesn't matter if we insert left
        # or right.
        if self.can_scout_and_show[self.current_player]:
            # Generate possible ranges for the Show moves - like above, but
            # assuming the table has one card less and our hand has one card
            # more from the Scout move. The index math below hurts my head.
            show_moves = [Show(start, length)
                          for start in range(0, len(self.hand) - (len(self.table) - 1 - 1) + 1)
                          for length in range(len(self.table) - 1, len(self.hand) + 1 - start + 1)]
            coalesced_scouts: list[tuple[bool, bool]] = []
            for scout in scouts:
                for show in show_moves:
                    move = ScoutAndShow(scout, show)
                    if Util.is_move_valid(
                            self.hand, self.table, self.can_scout_and_show[self.current_player], move):
                        # Coalesce Scout & Show moves: If we scout a card and
                        # play it again right away, the insert position does not
                        # matter -> only keep the first one we encounter.
                        if show.startPos == scout.insertPos and show.length == 1:
                            if (scout.first, scout.flip) in coalesced_scouts:
                                continue
                            else:
                                coalesced_scouts.append((scout.first, scout.flip))
                        scout_and_shows.append(move)

        return tuple(scouts + shows + scout_and_shows)


class Player(ABC):
    # Abstract base class. The player interface is very simple - the game engine
    # calls it with the subset of the game state the player could have observed,
    # and the player picks a move in what can be an arbitrarily complex process,
    # including starefulness by eg caching computation results.
    @abstractmethod
    def flip_hand(self, hand: Sequence[Card]) -> bool:
        pass

    @abstractmethod
    def select_move(self, info_state: InformationState) -> Move:
        pass
