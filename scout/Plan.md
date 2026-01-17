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
to make fits.
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
         0, 22, 40, 48, 64, 68, ?, ? (WIP).
      1. Based on above, repeat experiments with score := diff from opponent avg.
         This should be as good as or better.
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
   1. I will add early stopping - after 20 or 30 moves or so, I will just assume the
      highest scoring player wins. This can then be improved by using more
      advanced value functions, starting with hard-coded ones, that take into
      account how strong the hand (see above).
   1. Finally, I want to record rollout traces and use them to train a neural
      net. I am not sure yet about what features to pick but I'm excited to try
      that out.



