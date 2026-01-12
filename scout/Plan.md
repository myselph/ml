# Rough plan for implementing an AI that plays Scout.
#
# 1. Done - get a simulator (scoring, computing valid moves) up and running
#    with random policies.
# 2. Try some heuristics, and infra for comparing different policies.    
# 3. Try ISMCTS.
# 4. Neural policies, value functions?
# The below is a bit of a braindump and likely out of date.
#
# I do proceed in num_player rounds where each player gets to deal and play
# the first card once, to level the playing field, and stay consistent with
# the official rules. But I do not keep state inside players across those
# rounds, such as letting a player know how they did in the past rounds -
# maximizing the per-round score should be sufficient.
# 
# ISMCTS:
# Tree structure: Each node represents an information state for the player
# building the tree. Child nodes represent the next states we have explored
# after playing our action and after all other players have plaid theirs.
# So while there may be only N possible actions, there will be many more child
# nodes because (in ISMCTS) the other players' hands are unknown, and they
# might even behave non-deterministically. A node will have:
#   - untried_moves: list[Move]
#     - initialized with InformationState.possible_moves(); node is fully
#       expanded if len == 0. As long as there are untried actions, we will
#       "expand", ie take one of the unexplored actions (+ simulate other
#       players). This will create a new node in the tree.
#       Once the node is fully explored, we will take previously picked actions
#       that may also result in new nodes if the game state has changed or if
#       opponents are non-deterministic.
#   - possible_actions - when untried_actions = [], UCT kicks in and uses this array.
#   - N[a] - map from action to visit count (how often we took that action)
#   - W[a] - map from action to sum of points we got for it.
#   - children[a][key] - for each possible action, a map from info_state hash to a child node.
