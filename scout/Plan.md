# Rough plan for implementing an AI that plays Scout.
#
# First, get a simulator (scoring, computing valid moves) up and running with random policies.
# Then try some heuristics, and infra for comparing different policies.
# Then try ISMCTS.
# And then maybe neural policies.
# The below is a bit of a braindump and likely out of date.

# Take care of scoring, storing later. First, get a single game between two agents to work, one of them being ISMCTS.
# That way I think the interfaces become somewhat stable.
# Highest level:
# scores = Scores() # keep track of scores across rounds.
# for round in range(0, num_players):
#   state = State(round) # shuffle + serve cards
#   player_index = round
#   players = [] # initialize Player objects. This is also where they may flip their hand.
#   while not state.is_terminal():
#     info_state = state.get_info_state(player_index)
#     action = players[player_index].play(info_state)
#     state.apply_action(playerIndex, action)
#     player_index = mod(player_index + 1, num_players)
#   scores.update(state)
# 
# About re-initializing players: They don't need to keep state across rounds (scoring is done by game), it doesn't
# help them with decision making either since the best strategy is to maximize the number of points in each round independently.
# It may be useful knowledge to them who served, but that can / should be encoded in the information state.
# 
# state = new State() // full game state - inc. dealing cards, history, scores.
#   - information_state(i) - return the information state - what is visible to player i from the game state.
#   - apply_action(i, a) - player i taking action a (must be valid), progressing game state
# state = new State(information_state) - sample a game state (determinization) coherent with the given information state.
#   This is probably useful only for(IS)MCTS, and used for simulations.
# InformationState:
#   - get_possible_actions() - returns a list of actions the player can take.
#     - Scout - triple of (bool, bool, int) - 2 (L/R) * 2 (Flip/NoFlip) * (N+1) (insert position) options.
#     - Show - tuples of (int, int)
#     - ScoutAndShow - valid combos of the above. Computing this may be expensive when done inefficiently, such
#       as creating the cross-product then filtering out invalid moves (O(N^3)). Like say we have 11 cards. There are
#       48 possible Scout moves, and 132 Show moves before checking for legality, that's 6336 combinations to check.
#       At the least, optimize the Show part with early stopping - if (i, m) is invalid (not a group nor run), no
#       need to check (i, m+1); and if there are M cards on the table, no need to check (i, m-d). Starting to feel
#       this might be a bottleneck because it's called a lot during ISMCTS rollouts.
#   - hash(). An optimization to make node lookups in a tree more efficient (I doubt it matters much, might even hurt)
# Player has two functions only - __init__ (where they may flip their hand), and action(info_state).
#   - the simplest player would just randomly sample from info_state.get_possible_actions()
#   - Slightly improved baselines would use single heuristics to pick the best action. This can get arbitrarily complex - 
#     one might hand-define a value function that takes into account the number of points a move would result in, how strong
#     the hand is after the move, how many cards the other players have left, what's on the table etc.
#     I don't want to spend a ton of time on improving this, but having some basic heuristics to be better than chance is important
#     so we can check whether the whole engine - scoring, player rankings etc. - actually works. Such players may also be useful to
#     initialize policy networks or value functions before improving them with self-play.
# Tree structure: Each node represents an information state for the player building the tree. Child nodes represent the next states
# we have explored after playing our action and after all other players have plaid theirs.
# So while there may be only N possible actions, there will be many more child nodes because (in ISMCTS) the other players' hands
# are unknown, and they might even behave non-deterministically.
#   - untried_actions
#     - initialized with get_possible_actions(); node is fully expanded if len == 0.
#       As long as there are untried actions, we will "expand", ie take one of the unexplored actions.
#       This will create a new node in the tree. Taking previously picked actions may also result in new
#       nodes if the game state has changed or if opponents are non-deterministic.
#     - map<hash,Node> children - Maps from an info_state hash to a Node.
#   - possible_actions - when untried_actions = [], UCT kicks in and uses this array.
#   - N[a] - map from action to visit count (how often we took that action)
#   - W[a] - map from action to sum of points we got for it.
#   - children[a][key] - for each possible action, a map from info_state hash to a child node.


# Plan:
# 1. Implement the most minimal version - a bunch of random players.
# 2. Start working on MCTS.
# 3. Add heuristics.