Rough plan for implementing an AI that plays Scout.

1. Done - get a simulator (scoring, computing valid moves) up and running
   with random policies.
2. Done - Try some heuristics, and infra for comparing different policies.    
3. Done: ISMCTS experiments
4. WIP: ISMCTS + value function
4. Neural policies value functions?

ISMCTS experiments:
1. My original ISMCTS version used absolute scores and was fairly slow.
   It barely won 1% against PlanningPlayer even with 20k roll-outs, so I used
   GreedyShowPlayerWithFlip in the experiments here and below.
   For the original scoring and all others, I found a saturating relationship
   between rollouts and win rate. This one was roughly
   0% at 40, 50% at 125, and 75%
1. Next I sped things up 3x hrough different move generation and confirmed the
   win rate / rollout relationship was the same. So this CL here (1/16/26, 8pm)
   is a baseline working ISMCTS version:
   For 40, 70, 100, 130, 160, 200, 250, 300, respectively:
   0, 22, 40, 48, 64, 68, 77, 81.
1. Changing the scores from absolute to score := diff from opponent avg. made a
   huge difference, and is now the new score definition.
   For 40, 70, 100, 130, 160, 200, 250, 300, respectively:
   4, 56,  70,  75,  81,  83,   ?,  91
1. Using EpsilonGreedyPlayer during ISMCTS rollouts instead of randomly picking
   moves also helped:
      Using an EpsilonGreedy(eps=0.8), this gives
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
1. Experiments against PlanningPlayer: With the new scoring method and
   with EpsilonGreedyPlayer(0.5), I get for 150, 500, 1000, 2000, 3000, 5000 sims:
   2, 16, 43, 54, 62, 60. Great - had single digit percent rates when trying 20k
   moves before my improvements. However, given the duration of these
   simulations (collecting that data took >10h), it also begs the question
   about MCTS efficiency.
1. Neural value function experiments.
  1. I implemented recording traces, following ChatGPT's advice, which I have
     medium high confidence in.
     Algorithm: When expanding (whether it's a new or existing action), record
     the info state and the resulting score; collect 1M data points, train a
     neural net predicting the score. Use that net to essentially replace
     rollouts.
    * The details of the algorithm and implementation are a bit messy, and
       it's possible I got something wrong. The ISMCTS algorithm still proceeds
       through the stages of selection, expansion, rollout and backprop, but
       the roll-out is adjusted to stop as soon as possible, ie the first time
       it is the player's turn again.
    * On the data I collected (200 rollouts, 1M traces), the neural net MSE loss
      is not very confidence inspiring - a randomly initialized net gets ~45, merely
      predicting the mean score gives ~37, and a converged net gives ~29 on the
      validation set. I suspect that the problem of predicting scores is a very
      hard one due to the high variance. The usual "tricks" - different archs,
      nonlinearities, input normalization weight decay etc. all didn't help, and
      the net mostly converges to its lowest loss after a single epoch.
    * The unoptimized version is maybe 30% faster than using full roll-outs.
      It should be possible to improve this by using a significantly smaller
      network and batching NN calls (a bit tricky because ISMCTS and UCT are
      both inherently sequential, and would require rewriting ismcts()). For
      now, it makes more sense to focus on fully measuring win rates and first.
    * I compared ISMCTS with a GreedyEpsilonPlayer(0.5) against a PlanningPlayer
      for different roll-outs numbers in two configurations - without, and with
      the neural value function. The win rates are a bit puzzling.
      * w/o value function: for 40, 70, 100, 150, 200, 300, 400, 500, 1000, 2000,
      3000, 5000, I get 0,0,0,2,?,?,?,16,43,54,62,60.
      * w/ value function: for 40, 70, 100, 150, 200, 300, 400, 500, 1000, 2000,
      3000, 5000, I get 0,6,8,9,23,?,?,53,45,48,57.
      The value function curve looks much less smooth (as for all numbers above,
      I used 200 games, each 5 rounds) and gets worse after 500 rollouts. A bit
      hard to tell noise from trend. It is encouraging to see that for low
      numbers of roll-outs, the net adds value by providing more smoothed
      numbers (eg I get a win rate of 23% for 200 rollouts when using a neural
      value function vs. ? without) - that at least suggests it's learning
      something. But the breakdown shows that quickly, the neural net value
      function stops adding value and even gets worse than epsilon-greedy
      exploration. I can think of a couple of causes:
      * The net was trained on data from 200 rollouts and won't perform well on
        larger rollouts anymore, due to biases in the sampling procedure. I
        don't find this explanation very convincing - I expect increasing the 
        number of rollouts primarily creates more of the same training data, and
        whether I use 200 rollouts in 1000 games or 2000 rollouts in 100 games
        should not matter (ChatGPT strongly disagrees).
      * Roll-outs take advantage of more info than the neural net has; the
        determinization procedure takes into account what cards other players
        are known to have from scouting. It's possible that this gives it some
        edge the more we sample. Put differently, sampling more when using a
        neural value function gives you more of the same distribution; sampling
        more without a neural value function lets you converge towards the
        actual distribution. I can't think of a good way how to let the neural
        net make use of that info.
      * The neural net needs better features or more information. E.g. it does
        not currently consider how good a hand or scores are after taking
        actions (but this goes way into feature engineering). Eg I could add
        some info about the score diffs for the N best actions. But at that
        point, I would have effectively built PlanningPlayer++
      I am rather surprised how hard it seems to be to beat PlanningPlayer,
      because that is essentially a 1-level (40 moves?) search tree with a
      hard-coded value function, without any form of modeling opponent behavior.
      Prompting ChatGPT, it says that's expected, not the norm (but it says that
      about most observations I give it regardless of its prior predictions).
      But the arguments (obvious - how ISMCTS is handicapped, how a strong
      heuristic bypasses learning obvious things and works really well when
      my actions matter and dominate long term combinatorial effect).
      That's quite interesting, and may also explain heuristic approaches were
      (and still are!) so popular in game AIs. They may be really hard to beat
      if designed by somebody with a good understanding of the game; and while
      MCTS will always beat them in the limit of countless simulations, the
      number of sims you need to get to that point may be excessive (or at
      least hardly worth the risk). So it makes sense that combining the best
      of both worlds - strong hand-engineerd value or action-value functions
      plus MCTS - may provide the biggest bang for the buck.
      Of course I wanted a different answer - just throw a ton of data at it
      and get good. I am still pursuing that avenue - I plan to make the
      neural net function more powerful, and I then want to move on to neural
      policy functions.
    * I traced ISMCTS with stronger simulated players - 40% EpsilonGreedy, 60%
      PlanningPlayer - playing against PlanningPlayer's. The neural net trained
      on that exhibits lower loss (MSE=20, down from 29) so things become more
      predictable / lower variance. Results when using that value_fn are mixed/
      high variance - 0% for 200, 53% for 400.
    * An idea for how to speed up ISMCTS w/ neural value fns is to try all
      untried actions on a "virgin" node at once. In classic ISMCTS, we pick
      actions on unexpanded nodes (such as the first one) until there are no
      untried actions left, but each one of those looks the same, and we don't
      use UCT until a node is fully expanded. It will lead to different outcomes
      though - today, we can end up with many partially expanded nodes.
   1. Using stronger opponents during ISMCTS roll-outs:
     1. W/o value_fn:
     1. for value_fn training: MSE loss strongly reduced (now 20), meaning the
        scores are much better predictable.
   1. Next steps:
     1. Use stronger opponents during the ISMCTS roll-outs, but retain some
       randomness, e.g. by extending PlanningPlayer with epsilon-greedy
       selection, and/or combos strong and weak players (both with randomness).
     1. Use Transformer encoders for the table and hand, but only after above,
        because clean labels / non-garbage data needs to come before more
        advanced neural nets.
     1. Move on to neural policies and self-play.
1. Blog post. Contents?
  1. Rough journey
    1. instead of starting right away with neural policies,
    co-designed w/ AI -> suggested reasonable path. heuristics, ISMCTS, w/
    neural value functions, full blown neural policies w/ self play.
    1. Then build basic infra - the interfaces Players need
    to implement, a simple random player, and a simulator. More work than I
    expected - in particular, finding all valid moves in an efficient manner
    wasn't exactly trivial and took a bit of iterating.
    1. Starting to rank players - 3 heuristic players. Very easy to implement,
    but as it turns out, strongest not easy to beat.
    1. ISMCTS. Type safety becoming crucial to get anything done, hashing 
    InformationStates. Sampling possible game states. Performance becoming a
    problem -> hot spot analysis, optimizations. Heavily relying on talking w/
    LLM about edge cases; in hindsight I feel I should have just read the paper
    to get a deeper understanding and feel more confident about my 
    implementation because occasional LLM slips. Hard to tell when to go that
    deep and when to just rely on AI. Slow but better than many players;
    redefining score to be relative was crucial. Hidden state makes this a
    rather different setting - for each node there's pretty much an infinite
    number of child nodes, because a) there are 5 players, each with O(10)
    moves, especially early on when card counts are high - that's 100k child
    nodes, and we have a rollout of 200! b) the variance (difference in scores
    when picking an action and playing out results) is very high - even towards
    the end of the game. So I'm not sure how useful UCT is as a strategy as
    compared to either just random sampling (then playouts with strong policies)
    or a heuristic and randomized policy chosing strong moves.
    1. Use of neural value functions - trace, train, skip rollouts. Worked
    better for low number of roll-outs by reducing variance, and offers 
    speedups by skipping rollouts and allowing for batching, but still hard to
    compete against best heuristic player. At this point, impressed how far the
    classic methods go; it took me minutes to write the heuristic policy, and
    I could easily make it better by incorporating more of my own player
    behaviors. Even just replicating this with tree search, and neural value
    functions, is a significant endeavour in terms of engineering and
    computational requirements. Depending on what the goal is - build the best
    possible player for a competitive environment? build a player good enough
    for entertainment purposes (hobby players, runs on phones) - a heuristic
    player may be far preferable. I was somehow hoping or expecting that thanks
    to self-play and neural net advances, I would very quickly surpass hand-
    written functions or players.

     







