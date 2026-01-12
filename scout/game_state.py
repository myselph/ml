# A Scout (card game) simulator.
import random
from typing import Callable
from common import Scout, Show, Move, InformationState, Util

# Helper function to deal a whole deck.
def _generate_hands(num_players: int):
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
    # Randomly flip, shuffle and serve.
    flips = random.choices([True,False], k=len(deck))
    deck = [c if not flip else (c[1],c[0]) for (c, flip) in zip(deck, flips)]
    random.shuffle(deck)
    return [deck[i*N:(i+1)*N] for i in range(0, num_players)]

class GameState:
    num_players: int
    dealer: int  # index of first player
    current_player: int  # index of next player to call move() and info_state()
    # index of player who did the last Show or ScoutAndShow, for accounting.
    scout_benefactor: int
    hands: list[list[tuple[int, int]]]
    table: list[tuple[int, int]]
    # Per-player running scores: scout points + cards collected - cards in hand.
    scores: list[int]
    # Whether a player has used their Scout & Show capability yet.
    can_scout_and_show: list[bool]
    history: list[Move]  # History of moves.
    initial_flip_executed: bool  # Whether the initial flip has been executed.
    finished: bool  # Whether the game is over.

    def __init__(self, num_players: int, dealer: int):
        self.num_players = num_players
        self.hands = _generate_hands(num_players)
        self.scores = [-len(h) for h in self.hands]
        self.can_scout_and_show = [True] * num_players
        self.dealer = dealer
        self.current_player = self.dealer
        self.scout_benefactor = -1
        self.table = []
        self.history = []
        self.initial_flip_executed = False
        self.finished = False

    def move(self, m: Move):
        assert self.initial_flip_executed
        assert not self.finished
        assert Util.is_move_valid(self.hands[self.current_player], self.table,
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

    def maybe_flip_hand(self, flip_fns: list[Callable[[list[tuple[int, int]]], bool]]):
        assert not self.initial_flip_executed
        # Give each player the option to flip their hand
        assert not self.history
        assert len(flip_fns) == self.num_players
        for player in range(self.num_players):
            if flip_fns[player](self.hands[player]):
                self.hands[player] = list(
                    map(lambda c: (c[1], c[0]), self.hands[player]))
        self.initial_flip_executed = True

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

