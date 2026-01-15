from collections.abc import Sequence
import math
import statistics
import time
from common import Move, Card, InformationState, Player, Util
from players import RandomPlayer
from game_state import GameState
from typing import Self, Callable
from math import inf, sqrt, log
from dataclasses import dataclass
import random


@dataclass
class Node:
    possible_moves: tuple[Move, ...]
    untried_moves: list[Move]  # subset of possible_moves
    N: dict[Move, int]  # action -> visit count
    W: dict[Move, int]  # action -> sum of points
    # action -> information_state hash -> Node
    children: dict[Move, dict[int, Self]]


def sample_move_uct(node: Node) -> Move:
    # Only call this once node is fully expanded (all actions have been tried.)
    max_uct_val = -inf
    best_move = None
    num_visits = sum(node.N.values())
    return max(
        node.possible_moves,
        key=lambda m: node.W[m] /
        node.N[m] + sqrt(2 * log(num_visits) / node.N[m]))


def backprop(path: list[tuple[Node, Move]], score: int):
    for (node, move) in path:
        node.N[move] += 1
        node.W[move] += score


def ismcts(node: Node, game_state: GameState, players: list[Player],
           my_index: int):
    # 1. Selection phase: Descend while nodes are expanded, chosing actions
    # via UCT.
    path: list[tuple[Node, Move]] = []
    while not node.untried_moves and not game_state.is_finished():
        move = sample_move_uct(node)
        path.append((node, move))
        game_state.move(move)
        if game_state.is_finished():
            backprop(path, game_state.scores[my_index])
            return
        for p in players[my_index + 1:] + players[:my_index]:
            game_state.move(p.select_move(game_state.info_state()))
            if game_state.is_finished():
                backprop(path, game_state.scores[my_index])
                return
        info_state = game_state.info_state()
        hash = info_state.__hash__()
        # NB a move having been tried does not imply that a child node was
        # created (the game can have ended before we reached a new info_state);
        # therefore, we need to be careful accessing node.children[move.]
        if move in node.children and hash in node.children[move]:
            node = node.children[move][hash]
        else:
            # Construct new node. Not technically "expansion" because we have
            # tried the action already.
            possible_moves = info_state.possible_moves()
            new_node = Node(
                possible_moves, list(possible_moves), {}, {}, {})
            if move not in node.children:
                node.children[move] = {hash: new_node}
            else:
                node.children[move][hash] = new_node
            node = new_node

    # Is this even possible?
    if game_state.is_finished():
        backprop(path, game_state.scores[my_index])
        return
    # 2. Expansion phase: There are untried moves. Pick one, add a node, then
    # continue rolling out.
    move = random.choice(node.untried_moves)
    node.N[move] = 0
    node.W[move] = 0
    path.append((node, move))
    node.untried_moves.remove(move)
    game_state.move(move)
    if game_state.is_finished():
        backprop(path, game_state.scores[my_index])
        return
    for p in players[my_index + 1:] + players[:my_index]:
        game_state.move(p.select_move(game_state.info_state()))
        if game_state.is_finished():
            backprop(path, game_state.scores[my_index])
            return
    info_state = game_state.info_state()
    hash = info_state.__hash__()
    # Construct new node.
    possible_moves = info_state.possible_moves()
    node.children[move] = {}
    node.children[move][hash] = Node(
        possible_moves[:], list(possible_moves), {}, {}, {})
    node = node.children[move][hash]

    # 3. Roll-out phase - finish the game; don't add new nodes, just finish
    # fast, determin the score, and backprop.
    # Execute the first move before calling rollout() so we can add it to path
    # for bookkeeping; then roll away.
    move = players[my_index].select_move(game_state.info_state())
    node.N[move] = 0
    node.W[move] = 0
    path.append((node, move))
    game_state.move(move)
    if game_state.is_finished():
        backprop(path, game_state.scores[my_index])
        return
    # Play the game till the end and return the player's score.
    p = (my_index + 1) % len(players)
    while not game_state.is_finished():
        game_state.move(players[p].select_move(game_state.info_state()))
        p = (p + 1) % len(players)

    # 4. Backprop phase - Increment N and possible W for all nodes visited.
    backprop(path, game_state.scores[my_index])


@dataclass
class IsmctsStats:
    # We use floats to support averages
    cards_left: float
    num_moves: float
    num_children: float
    max_move_visits: float
    max_move_children: float


class IsmctsPlayer(Player):
    # The number of roll-outs we perform.
    # On reasonable values:
    #   I measured the win rate as a function of number of simulations When
    #   playing against GreedyShowPlayerWithFlip. We get ~0 at 40, about 50% at
    #   120 rollouts, and 75% at 250; unsurprisingly it follows a logarithm
    #   curve. Using randomly selected moves,
    # On timing:
    #   10 games, 5 rounds each, 10 roll-outs per select_move() takes 30s, when
    #   playing against 4 GreedyShowPlayerWithFlip.
    #   So a single round with one roll-out takes about 60ms.
    #   100 games with 100 roll-outs thus take ~3000s.
    _num_simulations: int
    # List of simulated players.
    _players: list[Player]
    # Cached trees for the action last taken.
    # We hardly ever hit a cached tree, and when we do, it has a visit count
    # of 1 so we save ourselves a single roll-out which is hardly worth it.
    # But it doesn't hurt performance and might help in future scenarios when
    # the game state cardinality is significantly reduced, so keeping it for
    # now
    _cached_trees: dict[int, Node]
    # can be used to record stats about the search tree.
    _record_stats: bool
    _stats: list[IsmctsStats]
    # If >0, select_move will finish after exceeding this many seconds.
    _move_time_limit_seconds: int

    # TODOs:
    # 1. Do a perf run to eliminate bottlenecks
    # 2. check the code again
    # 3. Play against a better player - PlanningPlayer?

    def __init__(
            self,
            num_players: int,
            num_simulations: int = 2_000,
            generate_player_fn: Callable[[],
                                         Player] = lambda: RandomPlayer(),
            move_time_limit_seconds: int = 0,
            record_stats: bool = False):
        self._num_simulations = num_simulations
        self._players = [generate_player_fn() for _ in range(num_players)]
        self._cached_trees = {}
        self._record_stats = record_stats
        self._stats = []
        self._move_time_limit_seconds = move_time_limit_seconds

    def flip_hand(self, hand: Sequence[Card]) -> bool:
        up_value = self._hand_value([h[0] for h in hand])
        down_value = self._hand_value([h[1] for h in hand])
        return up_value < down_value

    def select_move(self, info_state: InformationState) -> Move:
        start_time = time.time()
        possible_moves = info_state.possible_moves()
        my_index = info_state.current_player
        hash = info_state.__hash__()
        if hash in self._cached_trees:
            root = self._cached_trees[hash]
        else:
            root = Node(possible_moves[:], list(possible_moves), {}, {}, {})

        for _ in range(self._num_simulations):
            game_state = GameState.sample_from_info_state(info_state)
            ismcts(root, game_state, self._players, my_index)
            if self._move_time_limit_seconds > 0 and time.time(
            ) - start_time > self._move_time_limit_seconds:
                break

        # Find the most visited move.
        move = max(root.N.keys(), key=lambda k: root.N[k])
        if move in root.children:
            self._cached_trees = root.children[move]
        else:
            self._cached_trees = {}
        # Record statistics:
        if self._record_stats:
            self._stats.append(IsmctsStats(
                sum(info_state.num_cards), len(root.possible_moves),
                sum([len(root.children[m]) for m in root.children]),
                root.N[move], len(root.children[move]) if move in root.children else 0))
        return move

    def _hand_value(self, values: list[int]):
        # Compute a heuristic value of this hand, the better, the higher.
        # This is pretty heuristic. I don't count for overlaps (eg a triple counts
        # as both triple and double). But I lack a clear intuition for what
        # would be a better metric, and feel that learning a value function
        # is the only way to really improve on the below. E.g. I tried to use
        # pow(i, const)*c and do a grid search over const but that did not
        # seem to make a difference, and feels like manual ML.
        # TODO: Same as in GreedyPlayerWithFlip, deduplicate.
        (group_counts, run_counts) = self._count_groups_and_runs(values)
        value = 0
        for i, c in enumerate(run_counts):
            # ignore the singles
            if i == 0 or c == 0:
                continue
            value += i * c
        for i, c in enumerate(group_counts):
            if i == 0 or c == 0:
                continue
            # groups count more than runs of the same size, but not as much as
            # a longer run.
            value += (i + 0.5) * c
        return value

    def _count_groups_and_runs(self, values: Sequence[int]):
        group_counts = [0] * len(values)
        run_counts = [0] * len(values)
        for start_pos in range(len(values)):  # 0, 1, N-1
            for meld_size in range(1, len(values) + 1 - start_pos):
                if Util.is_group(values[start_pos:start_pos + meld_size]):
                    group_counts[meld_size - 1] += 1
                if Util.is_run(values[start_pos:start_pos + meld_size]):
                    run_counts[meld_size - 1] += 1
        return group_counts, run_counts

    def stats(self) -> IsmctsStats:
        assert self._record_stats
        # Returns averaged stats.
        N = len(self._stats)
        return IsmctsStats(
            sum([s.cards_left for s in self._stats]) / N,
            sum([s.num_moves for s in self._stats]) / N,
            sum([s.num_children for s in self._stats]) / N,
            sum([s.max_move_visits for s in self._stats]) / N,
            sum([s.max_move_children for s in self._stats]) / N
        )
