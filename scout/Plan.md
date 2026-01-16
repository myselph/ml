Rough plan for implementing an AI that plays Scout.

1. Done - get a simulator (scoring, computing valid moves) up and running
   with random policies.
2. Done - Try some heuristics, and infra for comparing different policies.    
3. Try ISMCTS.
4. Neural policies value functions?

### Performance profiling
ISMCST runs do take a significant amount of time. I did a profiling run and the
results are interesting. I let one ISMCST player play against 4 PlaningPlayers,
with 10 rollouts (not a lot - I'll need >100 for competitive players) and 1
game.
Of a ~10s execution time, 8.4s are spent in the ISMCST player (not surprising),
almost entirely in the ismcts function (not surprising either).
Around 8.3s of the total runtime is spent in player's possible_moves(), which
is shared between the ISMCTS and the other four players. But since the other
four players together only take (10s-8.4s), the majority of those 8.3s in
possible_moves must come from ISMCTS, which makes sense because it calls that
function a lot during roll-out.
6 of those 8.3s are spent in is_move_valid() and subfunctions; each individual
call seems to be cheap (0.6ms), but the function is called 13,800 times, and
is_move_valid a whopping 1.1M times. I'm not sure there will be much to optimize
there but looking into obvious inefficiencies, caching, and as a last resort a
C++ impl may be in order.
E.g.
1. get rid of the scout is_move_valid checks (though I suspect they are cheap).
1. we keep re-identifying runs and sets when looking for Shows. Maybe there are
   ways to cache them? And skip unnecessary Shows, eg if Show(i,j) is a group
   or set but is not valid (too low), then neither will be Show(i, j-1).
1. Caching:
   1. Util.is_group(table) keeps getting called for the same table. Cache.
   1. The table may change only minimally from one player to the next,
   and it will always represent a run or a set. So maybe storing some data
   alongside the table such as "set of 1s" or "3-run with max 5" could help
   speed up at least the checks.
1. Could get rid of GameState.move() call to Util.is_move_valid. I assume that
   is called ~30 times per game (1), and for every step (100?) in simulated
   games (20). So could be 2000 out of 2M calls; probably not worth it.
   But revisit (or disable via flag.)
1. Move isinstance(move,Scout) in is_move_valid to back - it should be called
   very infrequently.
1. Make range checks in move validity asserts, and optional? They should not
   ever trigger unless there's a bug.
1. Maybe avoid function the huge amount of function calls in Util by operating
   on lists of moves, or inlining.
1. Coalesce scout moves where it doesn't matter if we insert left or right of
   the Show.

How I measure performance:
fix seed so we get the same execution trace.
Use one IsmctsPlayer and 4 PlanningPlayer (wins a lot so shorter games)
python -m cProfile main.py --num_rollouts=50 --num_games=2 --fix_seed=True | head -50
Goal: Get down possible_moves time - it's .36ms per call, and 90.0% of the total time.
Time is pretty stable at 54.1s. Probably varies by a second.

Baseline:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.004    0.000   54.114   54.114 {built-in method builtins.exec}
        1    0.000    0.000   54.114   54.114 main.py:1(<module>)
        1    0.000    0.000   54.090   54.090 main.py:102(main)
        1    0.000    0.000   54.090   54.090 main.py:63(play_tournament)
        2    0.000    0.000   54.090   27.045 main.py:51(play_game)
       10    0.007    0.001   54.090    5.409 main.py:39(play_round)
       59    0.025    0.000   52.214    0.885 ismcts_player.py:191(select_move)
     2950    0.334    0.000   51.196    0.017 ismcts_player.py:41(ismcts)
   134916    8.287    0.000   48.655    0.000 common.py:181(possible_moves)
   131704    0.356    0.000   45.664    0.000 players.py:12(select_move)
 10167120   12.164    0.000   35.340    0.000 common.py:106(is_move_valid)
  8013655    7.960    0.000   18.239    0.000 common.py:71(is_show_valid)
  6151007    4.319    0.000    5.377    0.000 common.py:45(is_run)
46549452/46549344    4.271    0.000    4.271    0.000 {built-in method builtins.len}
  8584627    3.325    0.000    4.069    0.000 common.py:35(is_group)
 18503441    2.195    0.000    2.195    0.000 {built-in method builtins.isinstance}
  6454097    1.575    0.000    2.103    0.000 common.py:63(is_scout_valid)
      245    0.001    0.000    1.853    0.008 players.py:96(select_move)
    28116    0.016    0.000    1.289    0.000 {built-in method builtins.max}
    17258    0.008    0.000    1.219    0.000 players.py:98(<lambda>)
    17258    0.036    0.000    1.212    0.000 players.py:100(_value)
   135527    0.433    0.000    1.187    0.000 game_state.py:120(move)
    17338    0.027    0.000    1.161    0.000 players.py:66(_hand_value)
    17338    0.433    0.000    1.134    0.000 players.py:55(_count_groups_and_runs)
   134917    0.492    0.000    0.870    0.000 game_state.py:160(info_state)
     2950    0.188    0.000    0.800    0.000 game_state.py:171(sample_from_info_state)
  4371417    0.709    0.000    0.709    0.000 {method 'insert' of 'list' objects}


Experiment 1: Turn of Scout coallescing.
per-call time seems to go down ever so slightly (0.35ms), but might be noise,
and regardless we need coallescing; otherwise we oversample ScoutAndShows, and
the search tree becomes even bigger. So ignore that result.

Experiment 2: Skip Scout validity checks.
This simple change reduces the time per-call to 0.33ms (8%), and overall time by
almost 8-9%. I am pretty surprised by that since the is_move_valid check for
Scout is trivial.

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.008    0.000   50.150   50.150 {built-in method builtins.exec}
        1    0.000    0.000   50.150   50.150 main.py:1(<module>)
        1    0.000    0.000   50.102   50.102 main.py:102(main)
        1    0.000    0.000   50.102   50.102 main.py:63(play_tournament)
        2    0.000    0.000   50.102   25.051 main.py:51(play_game)
       10    0.007    0.001   50.102    5.010 main.py:39(play_round)
       59    0.025    0.000   48.260    0.818 ismcts_player.py:191(select_move)
     2950    0.324    0.000   47.251    0.016 ismcts_player.py:41(ismcts)
   134916    7.259    0.000   44.785    0.000 common.py:181(possible_moves)
   131704    0.322    0.000   41.886    0.000 players.py:12(select_move)
  8080470   10.710    0.000   32.940    0.000 common.py:106(is_move_valid)
  8013655    7.895    0.000   18.049    0.000 common.py:71(is_show_valid)
  6151007    4.268    0.000    5.304    0.000 common.py:45(is_run)
45051710/45051602    4.076    0.000    4.076    0.000 {built-in method builtins.len}
  8584627    3.279    0.000    4.017    0.000 common.py:35(is_group)
 16416791    1.997    0.000    1.997    0.000 {built-in method builtins.isinstance}
      245    0.001    0.000    1.819    0.007 players.py:96(select_move)
  4367447    1.151    0.000    1.547    0.000 common.py:63(is_scout_valid)
    28117    0.016    0.000    1.270    0.000 {built-in method builtins.max}
    17258    0.008    0.000    1.203    0.000 players.py:98(<lambda>)
    17258    0.035    0.000    1.195    0.000 players.py:100(_value)
   135527    0.427    0.000    1.175    0.000 game_state.py:120(move)
    17338    0.026    0.000    1.146    0.000 players.py:66(_hand_value)
    17338    0.428    0.000    1.120    0.000 players.py:55(_count_groups_and_runs)
   134917    0.477    0.000    0.851    0.000 game_state.py:160(info_state)
     2950    0.186    0.000    0.794    0.000 game_state.py:171(sample_from_info_state)
  4371417    0.707    0.000    0.707    0.000 {method 'insert' of 'list' objects}

Experiment 3-N: Tons of extra code to cache Util.is_group(table) and make the checks less safe, seems to be not worth it - same or slightly worse performance.
Quite possible some of the changes helped, others were counter-productive, so
I'm exploring making smaller more isolated ones to see if they help.

Experiment 3: Slightly rewrite is_valid_move to move the isinstance(move, Scout)
check to the back (and a small change to is_group) - because we don't call the
function much with Scout moves at all. This lead to a pretty consistent 2%
improvement in possible_moves().
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.003    0.000   49.306   49.306 {built-in method builtins.exec}
        1    0.000    0.000   49.306   49.306 main.py:1(<module>)
        1    0.000    0.000   49.282   49.282 main.py:102(main)
        1    0.000    0.000   49.282   49.282 main.py:63(play_tournament)
        2    0.000    0.000   49.282   24.641 main.py:51(play_game)
       10    0.007    0.001   49.282    4.928 main.py:39(play_round)
       59    0.025    0.000   47.438    0.804 ismcts_player.py:191(select_move)
     2950    0.323    0.000   46.428    0.016 ismcts_player.py:41(ismcts)
   134916    7.309    0.000   43.915    0.000 common.py:178(possible_moves)
   131704    0.322    0.000   41.057    0.000 players.py:12(select_move)
  8080470   10.198    0.000   32.033    0.000 common.py:103(is_move_valid)
  8013655    7.940    0.000   18.127    0.000 common.py:68(is_show_valid)
  6151007    4.317    0.000    5.357    0.000 common.py:42(is_run)
45051710/45051602    4.106    0.000    4.106    0.000 {built-in method builtins.len}
  8584627    3.250    0.000    3.990    0.000 common.py:35(is_group)
      245    0.001    0.000    1.820    0.007 players.py:96(select_move)
  4367447    1.165    0.000    1.570    0.000 common.py:60(is_scout_valid)
 12770583    1.499    0.000    1.499    0.000 {built-in method builtins.isinstance}
    28117    0.016    0.000    1.275    0.000 {built-in method builtins.max}
    17258    0.008    0.000    1.207    0.000 players.py:98(<lambda>)
    17258    0.035    0.000    1.199    0.000 players.py:100(_value)
   135527    0.429    0.000    1.192    0.000 game_state.py:120(move)
    17338    0.027    0.000    1.150    0.000 players.py:66(_hand_value)
    17338    0.429    0.000    1.123    0.000 players.py:55(_count_groups_and_runs)
   134917    0.490    0.000    0.863    0.000 game_state.py:160(info_state)
     2950    0.187    0.000    0.798    0.000 game_state.py:171(sample_from_info_state)
  4371417    0.710    0.000    0.710    0.000 {method 'insert' of 'list' objects}
     5910    0.068    0.000    0.486    0.000 game_state.py:19(_generate_hands)

Experiment 4:
Stop generating empty moves. Saves another 2s?
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.004    0.000   46.645   46.645 {built-in method builtins.exec}
        1    0.000    0.000   46.645   46.645 main.py:1(<module>)
        1    0.000    0.000   46.623   46.623 main.py:102(main)
        1    0.000    0.000   46.623   46.623 main.py:63(play_tournament)
        2    0.000    0.000   46.623   23.311 main.py:51(play_game)
       10    0.007    0.001   46.623    4.662 main.py:39(play_round)
       59    0.025    0.000   44.785    0.759 ismcts_player.py:191(select_move)
     2950    0.327    0.000   43.781    0.015 ismcts_player.py:41(ismcts)
   134916    6.684    0.000   41.289    0.000 common.py:183(possible_moves)
   131704    0.307    0.000   38.597    0.000 players.py:12(select_move)
  7282550    9.290    0.000   30.461    0.000 common.py:105(is_move_valid)
  7215735    7.736    0.000   17.834    0.000 common.py:61(is_show_valid)
  6151007    4.283    0.000    5.322    0.000 common.py:42(is_run)
43880225/43880117    3.961    0.000    3.961    0.000 {built-in method builtins.len}
  8584627    3.212    0.000    3.949    0.000 common.py:35(is_group)
      245    0.001    0.000    1.815    0.007 players.py:96(select_move)
  3942711    1.043    0.000    1.402    0.000 common.py:96(is_scout_valid)
 11547927    1.361    0.000    1.361    0.000 {built-in method builtins.isinstance}
   168451    0.048    0.000    1.302    0.000 {built-in method builtins.max}
    17258    0.008    0.000    1.203    0.000 players.py:98(<lambda>)
    17258    0.035    0.000    1.195    0.000 players.py:100(_value)
   135527    0.430    0.000    1.190    0.000 game_state.py:120(move)
    17338    0.027    0.000    1.146    0.000 players.py:66(_hand_value)
    17338    0.430    0.000    1.119    0.000 players.py:55(_count_groups_and_runs)
   134917    0.482    0.000    0.859    0.000 game_state.py:160(info_state)
     2950    0.188    0.000    0.794    0.000 game_state.py:171(sample_from_info_state)
  3946681    0.643    0.000    0.643    0.000 {method 'insert' of 'list' objects}
     5910    0.067    0.000    0.482    0.000 game_state.py:19(_generate_hands)

Experiment 5: Inline is_scout_valid
Another second.
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.003    0.000   45.694   45.694 {built-in method builtins.exec}
        1    0.000    0.000   45.694   45.694 main.py:1(<module>)
        1    0.000    0.000   45.664   45.664 main.py:102(main)
        1    0.000    0.000   45.664   45.664 main.py:63(play_tournament)
        2    0.000    0.000   45.664   22.832 main.py:51(play_game)
       10    0.007    0.001   45.664    4.566 main.py:39(play_round)
       59    0.025    0.000   43.838    0.743 ismcts_player.py:191(select_move)
     2950    0.328    0.000   42.831    0.015 ismcts_player.py:41(ismcts)
   134916    6.645    0.000   40.319    0.000 common.py:183(possible_moves)
   131704    0.310    0.000   37.754    0.000 players.py:12(select_move)
  7282550    9.095    0.000   29.514    0.000 common.py:105(is_move_valid)
  7215735    7.789    0.000   17.992    0.000 common.py:61(is_show_valid)
  6151007    4.312    0.000    5.363    0.000 common.py:42(is_run)
43880226/43880118    4.095    0.000    4.095    0.000 {built-in method builtins.len}
  8584627    3.258    0.000    4.004    0.000 common.py:35(is_group)
      245    0.001    0.000    1.802    0.007 players.py:96(select_move)
 11547927    1.419    0.000    1.419    0.000 {built-in method builtins.isinstance}
   168452    0.048    0.000    1.323    0.000 {built-in method builtins.max}
    17258    0.008    0.000    1.222    0.000 players.py:98(<lambda>)
    17258    0.036    0.000    1.214    0.000 players.py:100(_value)
   135527    0.434    0.000    1.168    0.000 game_state.py:120(move)
    17338    0.028    0.000    1.164    0.000 players.py:66(_hand_value)
    17338    0.433    0.000    1.137    0.000 players.py:55(_count_groups_and_runs)
   134917    0.482    0.000    0.848    0.000 game_state.py:160(info_state)
     2950    0.187    0.000    0.804    0.000 game_state.py:171(sample_from_info_state)
  3946681    0.632    0.000    0.632    0.000 {method 'insert' of 'list' objects}
     5910    0.068    0.000    0.491    0.000 game_state.py:19(_generate_hands)

Experiment 6: Efficient Show move generation
Instead of generating all possible subsets of a hand (>= len(table)), then
check if they are runs or moves, use an efficient run+move generator. We can
identify all the longest contiguous runs and moves in a hand in linear time
by going through the hand with two pointers; then generate shorter subruns and
subgroups from those, down to a minimum required length (the table length).
This had a stunning impact, reducing the runtime of possible_moves from 0.30ms
to 0.078ms, and the overall runtime from 46s to 15.6s. NB the overall time
reduction isn't that meaningful because the new algorithm changes the order
in which elements are created, which affects which moves are being sampled even
with a fixed seed, thus leading to different trajectories and shorter or longer
times.
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.004    0.000   15.585   15.585 {built-in method builtins.exec}
        1    0.000    0.000   15.585   15.585 main.py:1(<module>)
        1    0.000    0.000   15.563   15.563 main.py:102(main)
        1    0.000    0.000   15.563   15.563 main.py:63(play_tournament)
        2    0.000    0.000   15.563    7.781 main.py:51(play_game)
       10    0.009    0.001   15.562    1.556 main.py:39(play_round)
       55    0.024    0.000   14.017    0.255 ismcts_player.py:191(select_move)
     2750    0.306    0.000   13.215    0.005 ismcts_player.py:41(ismcts)
   130791    1.549    0.000   10.272    0.000 common.py:292(possible_moves)
   127734    0.244    0.000   10.197    0.000 players.py:12(select_move)
   227079    2.020    0.000    7.430    0.000 common.py:115(find_shows)
   227079    0.224    0.000    1.852    0.000 common.py:108(_find_runs)
   227079    1.310    0.000    1.762    0.000 common.py:36(_find_groups)
   454158    1.268    0.000    1.629    0.000 common.py:74(_find_runs_impl)
      218    0.000    0.000    1.521    0.007 players.py:96(select_move)
   251499    0.058    0.000    1.504    0.000 {built-in method builtins.max}
    19403    0.009    0.000    1.394    0.000 players.py:98(<lambda>)
    19403    0.042    0.000    1.385    0.000 players.py:100(_value)
    19483    0.031    0.000    1.326    0.000 players.py:66(_hand_value)
    19483    0.496    0.000    1.294    0.000 players.py:55(_count_groups_and_runs)
   131338    0.418    0.000    1.187    0.000 game_state.py:120(move)
10719664/10719556    1.031    0.000    1.031    0.000 {built-in method builtins.len}
   130792    0.439    0.000    0.788    0.000 game_state.py:160(info_state)
     2750    0.174    0.000    0.745    0.000 game_state.py:171(sample_from_info_state)
   669322    0.404    0.000    0.501    0.000 common.py:156(is_run)
   964106    0.386    0.000    0.473    0.000 common.py:149(is_group)
     5510    0.063    0.000    0.456    0.000 game_state.py:19(_generate_hands)
   131338    0.187    0.000    0.414    0.000 common.py:209(is_move_valid)

Experiment 7: Further Scout&Show coallescing
I already had coallescing of Scout & Show moves where the scouted card is shown
again right away - in that case, the insert position doesn't matter.
I found another bug - if a scouted card is inserted right next to the sequence
that will be shown, it doesn't matter if the insert position was left or right
of that sequence. I don't think this should have made possible_moves() faster,
but apparently it did. Anyway, this one was a necessary fix, we don't want to
oversample Scout&Shows. Also I optimized the coalescing logic which saved a bit
extra.

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.004    0.000   16.145   16.145 {built-in method builtins.exec}
        1    0.000    0.000   16.145   16.145 main.py:1(<module>)
        1    0.000    0.000   16.122   16.122 main.py:102(main)
        1    0.000    0.000   16.122   16.122 main.py:63(play_tournament)
        2    0.000    0.000   16.122    8.061 main.py:51(play_game)
       10    0.007    0.001   16.122    1.612 main.py:39(play_round)
       56    0.026    0.000   14.725    0.263 ismcts_player.py:191(select_move)
     2800    0.343    0.000   13.918    0.005 ismcts_player.py:41(ismcts)
   150903    0.271    0.000   10.565    0.000 players.py:12(select_move)
   154105    1.624    0.000   10.509    0.000 common.py:299(possible_moves)
   248591    2.151    0.000    7.511    0.000 common.py:121(find_shows)
   248591    1.431    0.000    1.919    0.000 common.py:36(_find_groups)
   248591    1.229    0.000    1.631    0.000 common.py:74(_find_runs)
   154512    0.485    0.000    1.387    0.000 game_state.py:120(move)
      228    0.000    0.000    1.375    0.006 players.py:96(select_move)
   278683    0.063    0.000    1.367    0.000 {built-in method builtins.max}
    17204    0.008    0.000    1.259    0.000 players.py:98(<lambda>)
    17204    0.036    0.000    1.251    0.000 players.py:100(_value)
    17284    0.028    0.000    1.200    0.000 players.py:66(_hand_value)
    17284    0.451    0.000    1.172    0.000 players.py:55(_count_groups_and_runs)
10980852/10980744    1.080    0.000    1.080    0.000 {built-in method builtins.len}
   154105    0.512    0.000    0.921    0.000 game_state.py:160(info_state)
     2800    0.173    0.000    0.747    0.000 game_state.py:171(sample_from_info_state)
   154512    0.219    0.000    0.488    0.000 common.py:216(is_move_valid)
     5610    0.064    0.000    0.459    0.000 game_state.py:19(_generate_hands)
   608364    0.370    0.000    0.458    0.000 common.py:163(is_run)
   400487    0.251    0.000    0.384    0.000 random.py:245(_randbelow_with_getrandbits)
   153647    0.125    0.000    0.355    0.000 random.py:345(choice)
  2013690    0.344    0.000    0.344    0.000 {method 'add' of 'set' objects}
1810995/1646232    0.290    0.000    0.334    0.000 {built-in method builtins.hash}
   687603    0.260    0.000    0.318    0.000 common.py:156(is_group)
     5610    0.067    0.000    0.259    0.000 random.py:354(shuffle)
     2810    0.009    0.000    0.248    0.000 game_state.py:107(__init__)
  1380748    0.225    0.000    0.225    0.000 {method 'append' of 'list' objects}
    78453    0.123    0.000    0.219    0.000 common.py:181(is_show_valid)



At this point, I suspect most low hanging fruits have been picked.
- a lot of safety checks I could have gotten rid of in is_valid_move now are
  hardly called anymore.
- possible_moves and its subfunctions are as optimized as I can think of without
  going to C++ or parallelization, and it still takes 2/3 of the overall
  time, so even if we make the rest faster this will be the bottleneck.
- We may be able to cache a player's Shows for the next round, but I feel that
  may not be worth it - scouting will invalidate many of them, and the logic
  necessary for maintaining that cache and checking its contents is probably
  destroying the gains.

I expect further speedups might require not calling possible_moves as much, by
e.g. using smarter rollout-policies to make the search tree shallower, or
value functions to stop exploration, or ...

Well, I sped things up by around a factor of 3x, that's awesome.