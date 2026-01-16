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
    def _find_groups(
            x: list[int], min_length: int = 2) -> set[tuple[int, int]]:
        # Given a sequence of integers, computes all groups (sub sequences of equal
        # numbers) of size min_length or larger.
        # Algorithm: Find largest groups in one pass through sequence; then
        # further add smaller groups by spliting those large ones into smaller sets
        # if possible.
        if len(x) == 0:
            return set()

        groups: set[tuple[int, int]] = set()
        left = 0
        right = 0

        while x[left] == x[right]:
            right = right + 1
            if right == len(x):
                if right - left >= min_length:
                    groups.add((left, right - 1))
                break
            elif x[left] != x[right]:
                # could be single length. Could add length check here.
                # Could also expand here.
                if right - left >= min_length:
                    groups.add((left, right - 1))
                left = right

        # Create subsets from larger sets.
        new_groups: set[tuple[int, int]] = set()
        for pos in groups:
            original_group_len = pos[1] - pos[0] + 1
            for group_len in range(min_length, original_group_len):
                for i in range(pos[0], pos[1] + 2 - group_len):
                    new_groups.add((i, i + group_len - 1))

        return groups | new_groups

    @staticmethod
    def _find_runs(x: list[int], min_length: int = 2) -> set[tuple[int, int]]:
        # Given a sequence of integers, find all runs (ascending or descending)
        # of size min_length or larger.
        if len(x) == 0:
            return set()

        runs: set[tuple[int, int]] = set()
        left = 0
        right = 0

        # Pass 1 - find ascending runs.
        while x[left] == x[right] - (right - left):
            right = right + 1
            if right == len(x):
                if right - left >= min_length:
                    runs.add((left, right - 1))
                break
            elif x[left] != x[right] - (right - left):
                if right - left >= min_length:
                    runs.add((left, right - 1))
                left = right

        # Pass 2 - find descencing runs
        left = 0
        right = 0
        while x[left] == x[right] + (right - left):
            right = right + 1
            if right == len(x):
                if right - left >= min_length:
                    runs.add((left, right - 1))
                break
            elif x[left] != x[right] + (right - left):
                if right - left >= min_length:
                    runs.add((left, right - 1))
                left = right

        # Create subsets from larger sets.
        new_runs: set[tuple[int, int]] = set()
        for pos in runs:
            original_run_len = pos[1] - pos[0] + 1
            for run_len in range(min_length, original_run_len):
                for i in range(pos[0], pos[1] + 2 - run_len):
                    new_runs.add((i, i + run_len - 1))

        return runs | new_runs

    @staticmethod
    def find_shows(
            hand_values: list[int],
            table_values: list[int]) -> set[Show]:
        show_min_group_length = max(1, len(table_values))
        table_is_group = False if (not table_values) \
            or table_values[0] != table_values[-1] else True
        if not table_values or table_is_group:
            # If the table is empty, we look for groups of size 1+ and runs of
            # size 2+ so as to avoid double counting single cards as runs *and*
            # groups.
            show_min_run_length = show_min_group_length + 1
        else:  # This means the table is a run of length >= 2.
            show_min_run_length = show_min_group_length

        groups = Util._find_groups(hand_values, show_min_group_length)
        runs = Util._find_runs(hand_values, show_min_run_length)
        if not table_values:
            return {Show(x[0], x[1] - x[0] + 1) for x in groups | runs}
        shows = []
        for g in groups:
            show_cand = Show(g[0], g[1] - g[0] + 1)
            if show_cand.length > len(table_values) \
               or not table_is_group or hand_values[g[0]] > table_values[0]:
                shows.append(show_cand)

        for r in runs:
            show_cand = Show(r[0], r[1] - r[0] + 1)
            if show_cand.length > len(table_values):
                shows.append(show_cand)
            elif not table_is_group \
                    and max(hand_values[r[0]], hand_values[r[1]]) > max(table_values[0], table_values[-1]):
                shows.append(show_cand)
        return set(shows)

    @staticmethod
    def is_group(cards: Sequence[int]):
        for i in range(1, len(cards)):
            if cards[i] != cards[0]:
                return False
        return True

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
        if isinstance(move, Show):
            return Util.is_show_valid(hand_values, table_values, move)
        elif isinstance(move, ScoutAndShow):  # Scout & Show
            if not can_scout_and_show:
                return False
            # Scout valid?
            if not if not Util.is_scout_valid(hand_values, table_values, move.scout):
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
        else:
            return Util.is_scout_valid(hand_values, table_values, move.scout)


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

    # This function should return a set to reflect that order doesn't matter
    # and that there should be no duplicates. However, the callers typically
    # sample from the returned collection, and that can't be done easily with
    # sets (requires conversion to tuple or list). I'm therefore returning a
    # tuple - that still allows for hashing, but also allows for sampling.
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

        # Show candidates - generate possible ones in an efficient manner, then
        # filter out those (and only those) that have the same length as the
        # table, but won't beat it. It is crucial that this code runs.
        hand_values = [c[0] for c in self.hand]
        table_values = [c[0] for c in self.table]
        shows = list(Util.find_shows(hand_values, table_values))

        # Scout and Show candidates.
        scout_and_shows = []
        if self.can_scout_and_show[self.current_player]:
            for scout in scouts:
                # Post-scout deck
                assert self.table  # Make the linter happy...
                scouted_card = self.table[0] if scout.first else self.table[-1]
                scouted_value = scouted_card[1] if scout.flip else scouted_card[0]
                new_hand_values = hand_values[:scout.insertPos] + \
                    [scouted_value] + hand_values[scout.insertPos:]
                new_table_values = table_values[1:] \
                    if scout.first else table_values[:-1]
                show_moves = Util.find_shows(new_hand_values, new_table_values)
                for show in show_moves:
                    # 1. If we scout a card and play it again right away, the
                    # insert position does not matter -> skip insert positions
                    # other than 0.
                    if show.startPos == scout.insertPos and show.length == 1 \
                            and scout.insertPos != 0:
                        continue
                    # 2. If we scout a card and insert it right next to the
                    # Show sequence, it does not matter if we insert it left
                    # or right of that sequence -> skip the left insert. We know
                    # there is a matching right insert.
                    if scout.insertPos == show.startPos - 1:
                        continue
                    scout_and_shows.append(ScoutAndShow(scout, show))

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
