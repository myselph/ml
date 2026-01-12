# A Scout (card game) simulator.
import random
from dataclasses import dataclass
from common import Player, InformationState, Scout, Show, ScoutAndShow, Move, is_run, is_group

class RandomPlayer(Player):
    # A baseline player that randomly selects from the possible moves.
    def flip_hand(self, hand: list[tuple[int, int]]) -> bool:
        return random.choice([True, False])

    def select_move(self, info_state: InformationState) -> Move:
        return random.choice(info_state.possible_moves())


class GreedyShowPlayer(Player):
    # A player that maximizes for gettng rid of its cards (ie it picks the highest
    # Show and ScoutAndShow moves).
    def flip_hand(self, hand: list[tuple[int, int]]) -> bool:
        return random.choice([True, False])

    def select_move(self, info_state: InformationState) -> Move:
        moves = info_state.possible_moves()
        scouts = [m for m in moves if isinstance(m, Scout)]
        shows = [m for m in moves if isinstance(m, Show)]
        scout_and_shows = [m for m in moves if isinstance(m, ScoutAndShow)]
        if scouts:
            next_move = random.choice(scouts)
        if shows:
            next_move = max(shows, key=lambda m: m.length)
        if scout_and_shows:
            best_scout_and_show = max(
                scout_and_shows, key=lambda m: m.show.length)
            # Pick the Scout&Show over a Scout only if we to dump at least 3
            # cards more - because we a) increase our hand by one b) can S&S
            # only once c) another player scores points.
            if not shows or next_move.length < best_scout_and_show.show.length+2:
                next_move = best_scout_and_show
        return next_move


class GreedyShowPlayerWithFlip(GreedyShowPlayer):
    # Like GreedyShowPlayer, but with non-random flip - improves performance.
    def flip_hand(self, hand: list[tuple[int, int]]) -> bool:
        up_value = self._hand_value([h[0] for h in hand])
        down_value = self._hand_value([h[1] for h in hand])
        return up_value < down_value

    def _count_groups_and_runs(self, values: list[int]):
        group_counts = [0] * len(values)
        run_counts = [0] * len(values)
        for start_pos in range(len(values)):  # 0, 1, N-1
            for meld_size in range(1, len(values) + 1 - start_pos):
                if is_group(values[start_pos:start_pos+meld_size]):
                    group_counts[meld_size-1] += 1
                if is_run(values[start_pos:start_pos+meld_size]):
                    run_counts[meld_size-1] += 1
        return group_counts, run_counts

    def _hand_value(self, values: list[int]):
        # Compute a heuristic value of this hand, the better, the higher.
        # This is *super* heuristic; I don't even count for overlaps (eg a triple counts
        # as both triple and double and single).
        (group_counts, run_counts) = self._count_groups_and_runs(values)
        value = 0
        for i, c in enumerate(run_counts):
            # ignore the singles
            if i == 0 or c == 0:
                continue
            value += i*c
        for i, c in enumerate(group_counts):
            if i == 0 or c == 0:
                continue
            # groups count more than runs of the same size, but not as much as a longer run.
            value += (i+0.5)*c
        return value


class PlanningPlayer(GreedyShowPlayerWithFlip):
    # A player with a heuristic value function that simulates all moves and picks the one
    # with the highest value. Best perorming heurstic player. There are various knobs in
    # the value function one could tune through RL or grid search.
    c: float

    def __init__(self):
        self.c = 0.25 # found via grid search - self-play and against GreedyShowPlayerWithFlip

    def select_move(self, info_state: InformationState) -> Move:
        moves = info_state.possible_moves()
        best_value = None
        best_move = None
        hand_values = [h[0] for h in info_state.hand]
        for move in moves:
            value = self._value(info_state, move)
            if not best_value or value > best_value:
                best_value = value
                best_move = move
        return best_move

    def _value(self, info_state: InformationState, move: Move):
        # Calculates a heuristic value for the state of the game after the given move.
        # This involved simulating every move and calculating the new value.
        hand_values = [h[0] for h in info_state.hand]
        if isinstance(move, Scout):
            hand_values_new = self._simulate_scout(
                hand_values, info_state.table, move)
            return self.c*self._hand_value(hand_values_new) - len(hand_values_new) - 1
        elif isinstance(move, Show):
            hand_values_new = self._simulate_show(hand_values, move)
            return self.c*self._hand_value(hand_values_new) - len(hand_values_new) + len(info_state.table)
        else:
            hand_values_new = self._simulate_scout(
                hand_values, info_state.table, move.scout)
            hand_values_new = self._simulate_show(hand_values_new, move.show)
            return self.c*self._hand_value(hand_values_new) - len(hand_values_new) + len(info_state.table) - 1

    def _simulate_scout(self,  hand_values: list[int], table: list[tuple[int, int]], scout: Scout):
        card = table[0] if scout.firstCard else table[-1]
        card_value = card[1] if scout.flipCard else card[0]
        return hand_values[:scout.insertPosition] + [card_value] + hand_values[scout.insertPosition:]

    def _simulate_show(self,  hand_values: list[int], show: Show):
        return hand_values[:show.startPos] + hand_values[show.startPos+show.length:]

