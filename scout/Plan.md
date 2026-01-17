Rough plan for implementing an AI that plays Scout.

1. Done - get a simulator (scoring, computing valid moves) up and running
   with random policies.
2. Done - Try some heuristics, and infra for comparing different policies.    
3. WIP: ISMCTS experiments
4. Neural policies value functions?

ISMCTS experiments:
1. Using a single ISMCTS player against 4 GreedyShowPlayerWithFlip, I plotted
different num_rollout values against the win_rate for the ISMCTS player and
found a saturating (not quite logarithmic) relationship, roughly 0% WR at 40, 50% at 125, and 75% at
250. It seems the slope is roughly 44*ln(x)-160. To get a good enough description
of the curve, running win rate experiments at 50, 100, 200, 400 should be enough.
But I went with 40, 70, 100, 130, 160, 200, 250, 300 - 200 games each. Also better
to make fits. I subsequently found that score := diff from opponent avg score is
a much better score function - it roughly gives the same performance at half the
number of roll-outs (see below)
1. ISMCTS roll-outs are still very slow after speeding up things by 3x - getting
   reliable averages of eg win rates takes a lot of time, and to win against
   my best heuristic player (PlanningPlayer), even 20k rollouts aren't enough -
   and that already takes too long. Since I spent a lot of time on getting the
   code to run fast, it's time to change the algorithm:
   1. I am changing the scores I'm backpropagating.
      1. I tried win (1) or lose (0), because that would be easier to train I
         think than predicting raw scores. But I found the win rates tank if I 
         do that, because too much info is lost about how good a branch is.
      1. Justo ensure I can reproduce the above results, I went back to the
         original, raw scores and confirmed I can still generate the same win
         rates. That was important to me because with all the small changes and
         performance optimizations, I wanted to make sure none of them had any
         adverse impact on the win rate. So this CL here (1/16/26, 8pm) is a 
         baseline working ISMCTS version. I am still collecting the final win
         rates (they look similar enough) with more games and will replace the
         ones above with those new numbers.
         They are, for 40, 70, 100, 130, 160, 200, 250, 300, respectively:
         0, 22, 40, 48, 64, 68, 77, 81.
      1. Repeated experiments with score := diff from opponent avg.
         As hoped (but not expected), significant difference.
         For 40, 70, 100, 130, 160, 200, 250, 300, respectively:
              4, 56,  70,  75,  81,  83,   ?,  91
         Adopted this as the new score.
   1. See if Epsilon-Greedy player works - I would expect slightly higher
      performnce for same number of rollouts and a high epsilon (not a small
      one, at least initially, to keep it close to baseline). 
      I am changing the purely random move selection to something very very
      simple but slightly better - epsilon-greedy: pick random action with
      probability epsilon, and pick the action that most improves the score
      with P(1-epsilon). Score is defined as the current score in the game
      (- cards in hand + scores from scouting + scores from card removal), and
      we can quickly estimate which move makes the biggest dent. There are lots
      of improvements possible, but I hope this alone will reduce the tree depth
      and make the traces more relevant than the randomly explored ones.
      The only real way to compare such agents would be to see who's doing better
      in a given time budget.
      Results: Using an EpsilonGreedy(eps=0.8), this gives
      for 40, 70, 100, 130, 160, 200, 250, 300:
           2, 58,  71,  81,  83,  88,  89,  94
      Using EpsilonGreedy(eps=0.5) is a bit better:
      for 40, 70, 100, 130, 160, 200, 250, 300:
           9, 57,  83,  85,  91,  94,  97,  96
      Using EpsilonGreedy(eps=0.2):
      for 40, 70, 100, 130, 160, 200, 250, 300:
           2, 61,  83,  91,  93,  97,   ?,   ?
      Not sure yet what to make of the EpsilonGreedy stuff. It works significantly
      better than just picking random moves (eps=1), but I'm a bit concerned about
      overfitting (does this extend to other players?)/underexploring, and
      having another hyperparameter (epsilon).
   1. Try playing against PlanningPlayer again - maybe the new scoring method
      makes a difference. Woah yes - with EpsilonGreedyPlayer(0.5), I get ~50% at
      1000 rollouts! I had single digit percent rates when trying 20k moves before
      my improvements. Yay!
   1. I will add early stopping - after 20 or 30 moves or so, I will just assume the
      highest scoring player wins. This can then be improved by using more
      advanced value functions, starting with hard-coded ones, that take into
      account how strong the hand (see above).
   1. Finally, I want to record rollout traces and use them to train a neural
      net. Plan:
      1. We record the scores we see when *expanding*, ie trying out a new
         action and thus adding a child node (I think not when adding child
         nodes for already tried actions). Things to capture:
         * current scores
         * number of cards
         * table
         * hand
         Record about 4M of those in a num_rollouts=200 config (what I hope to
         use later on); with 5 rounds/game, ~4 actions per round, and ~200
         expansions per action, we need to play about 1,000 games.
      1. Train an MLP
         Use hand-coded features first, before doing Transformer stuff etc. At
         this point my goal is still to build a fast, competitive ISMCTS agent,
         so we can generate data for neural policy bootstrapping, and have a
         strong baseline. Features:
         * 5 num_cards integers, pad with 0s if fewer than 5 players (but I
           only use five right now - should probably try others as well).
         * 5 scores integers
         * table features:
           * meld_or_group
           * length
           * highest value
         * hand features
           * length
           * num_runs
           * num_groups
           * longest_run
           * longest_group
           * avg_run_len
           * avg_group_len
           * rank histogram (10)
           * run-length histogram (10?)
           * can scout & show
           * group length histogram (9?)
           * advanced (function of table):
             * max playable run length
             * max playable group length
             * num new groups after scout
             * num new runs after scour
        * scale features with constants to roughly [-1, 1] or [0, 1]
        * So 60-90 features; use an MLP of size 128:64:32:1 w/ RELUs, output
          is score diff.





