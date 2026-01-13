# A Scout (card game) simulator.
import random
from typing import Callable, Self
from common import Scout, Show, Move, InformationState, Util, RecordedMove, Card
import copy


def _generate_hands(num_players: int) -> list[list[Card]]:
    # Helper function to deal a whole deck.
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
    flips = random.choices([True, False], k=len(deck))
    deck = [c if not flip else (c[1], c[0]) for (c, flip) in zip(deck, flips)]
    random.shuffle(deck)
    return [deck[i*N:(i+1)*N] for i in range(0, num_players)]


def _normalize_card(c: Card) -> Card:
    return c if c[0] < c[1] else (c[1], c[0])


def _find_removed_and_scouted_cards(
    num_players: int, dealer: int, history: list[RecordedMove]) \
        -> tuple[list[Card], dict[Card, int]]:
    removed_cards = []  # cards that were removed from table.
    scouted_cards = {}  # dict from normalized card to player holding it.
    player_index = dealer
    for i, h in enumerate(history):
        # Memorize who scouted a card...
        if h.scouted:
            scouted_cards[_normalize_card(h.scouted)] = player_index
        # ...and forget about it when it got played.
        # NB this covers Scout&Show moves as well.
        if h.shown:
            for c in h.shown:
                nc = _normalize_card(c)
                if nc in scouted_cards:
                    del scouted_cards[nc]
            removed_cards += [_normalize_card(c) for c in h.removed]
        player_index = (player_index+1) % num_players
    return removed_cards, scouted_cards


class GameState:
    num_players: int
    dealer: int  # index of first player
    current_player: int  # index of next player to call move() and info_state()
    # index of player who did the last Show or ScoutAndShow, for accounting.
    scout_benefactor: int
    hands: list[list[Card]]
    table: list[Card]
    # Per-player running scores: scout points + cards collected - cards in hand.
    scores: list[int]
    # Whether a player has used their Scout & Show capability yet.
    can_scout_and_show: list[bool]
    history: list[RecordedMove]
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
        recorded_move = RecordedMove(None, [], [])
        if isinstance(m, Scout):
            recorded_move.scouted = self._scout(m)
            if (self.current_player + 1) % self.num_players == self.scout_benefactor:
                self.finished = True
        elif isinstance(m, Show):
            (s, r) = self._show(m)[:]
            recorded_move.shown = s
            recorded_move.removed = r
        else:
            recorded_move.scouted = self._scout(m.scout)
            (s, r) = self._show(m.show)[:]
            recorded_move.shown = s
            recorded_move.removed = r
            self.can_scout_and_show[self.current_player] = False
        if not self.hands[self.current_player]:
            self.finished = True
        self.history.append(recorded_move)
        self.current_player = (self.current_player + 1) % self.num_players

    def is_finished(self):
        return self.finished

    def maybe_flip_hand(self, flip_fns: list[Callable[[list[Card]], bool]]):
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
            self.hands[self.current_player], self.table,
            [len(self.hands[i]) for i in range(self.num_players)],
            self.scores, self.can_scout_and_show, self.history)

    def sample_from_info_state(info_state: InformationState) -> Self:
        # Static factory method to create a GameState consistent with the
        # provided InformationState.
        # This function can be used to run ISMCTS explorations, i.e. sample many possible
        # game states and run simulations on each to get an aggregate tree.
        game_state = GameState(info_state.num_players, info_state.dealer)
        game_state.current_player = info_state.current_player
        game_state.scout_benefactor = info_state.scout_benefactor
        game_state.table = copy.deepcopy(info_state.table)
        game_state.scores = info_state.scores[:]
        game_state.can_scout_and_show = info_state.can_scout_and_show[:]
        game_state.history = info_state.history[:]
        game_state.initial_flip_executed = True
        game_state.finished = False
        game_state.hands = [[] for _ in range(game_state.num_players)]
        game_state.hands[game_state.current_player] = info_state.hand[:]

        # 1. Generate a hand, and flatten it (get rid of assignments).
        #    NB cards in random_deck are not normalized (ie some may have been flipped).
        random_deck = _generate_hands(
            game_state.num_players)
        random_deck = [card for hand in random_deck for card in hand]
        # 2. Remove player's cards and cards on the table.
        normalized_hand = [_normalize_card(c) for c in info_state.hand]
        normalized_table = [_normalize_card(c) for c in info_state.table]
        random_deck = [
            c for c in random_deck if not _normalize_card(c) in normalized_hand
            and not _normalize_card(c) in normalized_table]
        # 3. Remove cards that are not in the game anymore. NB both return values
        #    use normalized cards.
        (removed_cards, scouted_cards) = _find_removed_and_scouted_cards(
            info_state.num_players, info_state.dealer, game_state.history)
        random_deck = [
            c for c in random_deck if not _normalize_card(c) in removed_cards]
        # 4. Seed the remaining hands with the known scouted cards, and remove those from the random deck.
        for c, p in scouted_cards.items():
            # "unnormalize"
            if not c in random_deck:
                c = (c[1], c[0])
            game_state.hands[p].append(c)
            random_deck.remove(c)

        # Now distribute the remaining cards in random_deck across the players.
        # I believe we don't need to shuffle a second time.
        card_index = 0
        for p in range(info_state.num_players):
            if p == info_state.current_player:
                continue
            num_missing_cards = info_state.num_cards[p] - \
                len(game_state.hands[p])
            game_state.hands[p] += random_deck[card_index:card_index +
                                               num_missing_cards]
            card_index += num_missing_cards

        print(num_missing_cards)
        print(card_index)
        print(len(random_deck))
        assert card_index == len(random_deck)
        return game_state

    def _scout(self, m: Scout) -> Card:
        hand = self.hands[self.current_player]
        if m.firstCard:
            card = self.table[0]
            scouted_card = card
            self.table = self.table[1:]
        else:
            card = self.table[-1]
            scouted_card = card
            self.table = self.table[:-1]
        if m.flipCard:
            card = (card[1], card[0])
        hand.insert(m.insertPosition, card)
        self.scores[self.scout_benefactor] += 1
        self.scores[self.current_player] -= 1
        return scouted_card

    def _show(self, m: Show) -> tuple[list[Card], list[Card]]:
        hand = self.hands[self.current_player]
        self.scores[self.current_player] += len(self.table) + m.length
        shown_cards = hand[m.startPos:m.startPos+m.length]
        removed_cards = self.table[:]
        self.table = shown_cards
        self.hands[self.current_player] = hand[:m.startPos] + \
            hand[m.startPos+m.length:]
        self.scout_benefactor = self.current_player
        return shown_cards, removed_cards
