from __future__ import annotations
import copy
from math import inf
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Dict, Any
from common import InformationState, Move, Player, StateAndScoreRecord
from game_state import GameState
from ismcts_player import gen_ssr
from main import play_game, play_round, play_tournament
from neural_value_function import featurize
from players import GreedyShowPlayerWithFlip, PlanningPlayer, RandomPlayer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr",
    type=float,
    help="Policy Learning Rate",
    default=3e-3
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Minibatch size for PPO updates",
    default=512
)
parser.add_argument(
    "--iterations",
    type=int,
    help="Number of training iterations",
    default=80
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs per training iteration",
    default=2
)
parser.add_argument(
    "--episodes",
    type=int,
    help="Number of episodes per training iteration",
    default=40
)


def encode_information_state(
    info_state: InformationState
) -> torch.Tensor:
    """
    Return a tensor of shape [obs_dim].
    """
    # For now, just return the current player's score repeated to fill; that
    # way, the network has something to learn on.
    # Next, I hope to move to the full featurization used in neural_value_function.
    # Finally, I hope to use Transformers or RNNs over annotated hand/table
    # sequences, e.g. "4,4,9,2,3,4,7" ->
    # "<set2>,4,<single>,9,<run3>,2,3,4,<single>,7" etc.
    # Unclear whether "3,2,1,2" should be run2,run2 or run3,single.
    ssr = StateAndScoreRecord(
        info_state.current_player,
        tuple(c[0] for c in info_state.hand),
        info_state.table,
        info_state.num_cards,
        info_state.scores,
        info_state.can_scout_and_show, inf
    )

    return featurize([ssr])[0]


def encode_post_move_states(
    post_move_states: tuple[InformationState, ...]
) -> torch.Tensor:
    return torch.stack([encode_information_state(s) for s in post_move_states])


class PolicyNet(nn.Module):
    # ----------------------------------------------------------------------
    # The Policy network is a ranking model - it takes the state visible to
    # the current player, and the state after each possible move, and "ranks"
    # the moves by producing a logit for each post move state.
    # ----------------------------------------------------------------------
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
            self,
            pre_move_state: torch.Tensor,
            post_move_states: torch.Tensor) -> torch.Tensor:
        """
        pre_move_state: [1, D]
        post_move_states: [num_moves, D]
        return logits: [1, num_moves]
        """

        # Expand obs to [num_moves, obs_dim] using broadcasting (no memory
        # copy)
        # Consider removing this - I don't think it adds extra info beyond
        # the post_move_states, but since all the plubming is in place, I'm
        # keeping for now.
        # pre_move_state = pre_move_state.expand(post_move_states.size(0), -1)

        # Concatenate to [num_moves,   obs_dim + move_dim]
        # combined = torch.cat([pre_move_state, post_move_states], dim=-1)

        combined = post_move_states  # [N, D]

        # Output: [N, 1] -> squeeze to [1, N] or [N]
        return self.net(combined).view(1, -1)


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Return the score, for now.
        return self.net(state)

# ----------------------------------------------------------------------
# Data structures for rollout storage
# ----------------------------------------------------------------------


@dataclass
class Transition:
    """
    One step of experience.
    """
    pre_move_state: torch.Tensor
    action_idx: int             # index into the action list
    logprob: float              # scalar
    reward: float               # scalar
    value: float                # scalar
    done: bool                  # episode termination flag
    post_move_states: torch.Tensor


@dataclass
class Trajectory:
    """
    Sequence of Transitions for a single agent over one episode.
    """
    transitions: List[Transition]



class NeuralPlayer(PlanningPlayer):
    # A player wrapping an agent that wraps a neural policy and value network.
    def __init__(self, agent: Agent):
        self.agent = agent

    def select_move(self, info_state: InformationState) -> Move:
        moves, raw_post_move_states = info_state.post_move_states()
        pre_move_state, post_move_states = self.agent.encode_inputs(
            info_state, raw_post_move_states)

        action_idx, _, _ = self.agent.select_action(
            pre_move_state,
            post_move_states
        )
        return moves[action_idx]

    def flip_hand(self, hand: Sequence[tuple[int, int]]) -> bool:
        # TODO: Implement. Would be nice to use value_fn for that, but I'm not
        # sure how to derive an information state.
        # For now, use PlanningPlayer's heuristic method.
        return super().flip_hand(hand)


# ----------------------------------------------------------------------
# PPO Loss
# ----------------------------------------------------------------------

def ppo_loss(
    policy: PolicyNet,
    value_fn: nn.Module,
    pre_move_states: list[torch.Tensor],  # batch list of tensor [D]
    post_move_states_list: list[torch.Tensor],  # batch list of tensor [N_i, D]
    action_idx: torch.Tensor,     # shape: [batch]
    old_logprob: torch.Tensor,    # shape: [batch]
    returns: torch.Tensor,        # shape: [batch]
    advantages: torch.Tensor,     # shape: [batch]
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    minibatch_size: int = 512
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO clipped surrogate objective.
    We assume:
    - policy(obs, move_batch) -> logits for each move in move_batch
    - value_fn(obs) -> scalar value prediction

    NOTE: Because legal action sets differ per item, we must compute
    action logprobs one sample at a time.
    """

    batch_size = len(pre_move_states)
    new_logprobs = []
    entropies = []

    for i in range(batch_size):
        pre_move_state = pre_move_states[i]
        post_move_states = post_move_states_list[i]

        logits = policy(pre_move_state,
                        post_move_states)    # shape: [1, num_actions]
        logprobs = torch.log_softmax(logits, dim=-1)  # [1, num_actions]

        # Select logprob of performed action
        chosen = action_idx[i]
        new_logprobs.append(logprobs[0, chosen])

        # Entropy for this decision
        entropy = -(logprobs * torch.exp(logprobs)).sum()
        entropies.append(entropy)

    new_logprobs = torch.stack(new_logprobs)  # shape: [batch]
    entropies = torch.stack(entropies)        # shape: [batch]

    # Probability ratio
    ratio = torch.exp(new_logprobs - old_logprob)

    # PPO objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.sum(torch.min(unclipped, clipped)) / minibatch_size

    # Value loss
    pre_move_states_enc = torch.stack(pre_move_states)  # shape: [batch, D]
    values = value_fn(pre_move_states_enc).squeeze(-1)        # shape: [batch]
    value_loss = vf_coef * torch.sum((returns - values) ** 2) / minibatch_size

    # Entropy bonus
    entropy_bonus = ent_coef * entropies.sum() / minibatch_size

    total_loss = policy_loss + value_loss - entropy_bonus

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropies.mean().item()
    }

    return total_loss, metrics


# ----------------------------------------------------------------------
# Advantage calculation (GAE)
# ----------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,       # [T]
    values: torch.Tensor,        # [T+1]
    dones: torch.Tensor,         # [T]
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.
    NB the parameters are concatenations of trajectories for multiple episodes,
    and dones indicate episode boundaries.
    """
    T = rewards.size(0)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        # compute TD: difference between what we saw (reward_t + V(s_{t+1})) and
        # what we expected to see (V(s_t))
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return returns, advantages


class Agent:
    # The agent class wraps a policy network and a value function.
    def __init__(self, policy: PolicyNet, optim: torch.optim.Optimizer, value_fn: nn.Module):
        self.policy = policy
        self.optim = optim
        self.value_fn = value_fn

    def encode_inputs(
        self,
        pre_move_state: InformationState,
        post_move_states: tuple[InformationState, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Separate function for encoding inputs so we can use those not just
        # for inference (select_action) but also for training.
        encoded_pre_move_state = encode_information_state(
            pre_move_state).unsqueeze(0)  # [1, D]
        encoded_post_move_states = encode_post_move_states(
            post_move_states)  # [N, D]
        return encoded_pre_move_state, encoded_post_move_states

    def select_action(
        self,
        encoded_pre_move_state: torch.Tensor,         # shape: [obs_dim]
        encoded_post_move_states: torch.Tensor,        # shape: [N, obs_dim]
    ) -> Tuple[int, float, float]:
        """
        Compute distribution over legal moves and sample one.
        Returns:
            action_idx, logprob of sampled move, value_prediction
        """
        logits = self.policy(
            encoded_pre_move_state,
            encoded_post_move_states)       # [1, N]
        logprobs = torch.log_softmax(logits, dim=-1)[0]  # [N]
        probs = torch.exp(logprobs)
        dist = torch.distributions.Categorical(probs)
        a = int(dist.sample().item())

        value = self.value_fn(encoded_pre_move_state).item()

        return a, float(logprobs[a].item()), value


# ----------------------------------------------------------------------
# Rollout generation
# ----------------------------------------------------------------------

def collect_episodes(
    agents: List[Agent],
    env_constructor: Callable[[int], GameState],
    min_episodes: int,
    num_players: int,
    min_examples_per_player: int
) -> Dict[int, List[Trajectory]]:
    """
    All agents play together in a multi-player environment.
    Returns dict: agent_id -> list of Trajectory (one per episode this agent
    played in).
    We play at least num_episodes, but possibly more to ensure we collect at
    least min_examples_per_player for each agent.
    """
    data: Dict[int, List[Trajectory]] = {i: [] for i in range(len(agents))}

    episode = 0
    while episode < min_episodes or any(not traj for traj in data.values()) or \
        any(sum(len(traj.transitions) for traj in data[i]) < min_examples_per_player for i in data):
        episode += 1
        episode_agent_indices = random.sample(range(len(agents)), num_players)
        episode_agents = [agents[i] for i in episode_agent_indices]
        env = env_constructor(0)
        # For now, we just flip like PlanningPlayer would.
        # Eventually, we should learn a policy for that, but it complicates
        # training a bit because the flip decision requires another network.
        # TODO: Can we just use the neural value function?
        pp = PlanningPlayer()
        env.maybe_flip_hand([lambda h: pp.flip_hand(h)
                            for _ in episode_agents])
        traj = {i: [] for i in episode_agent_indices}

        done = False
        while not done:
            player = env.current_player
            agent = episode_agents[player]

            raw_pre_move_state = env.info_state()
            moves, raw_post_move_states = raw_pre_move_state.post_move_states()
            pre_move_state, post_move_states = agent.encode_inputs(
                raw_pre_move_state, raw_post_move_states)
            action_idx, logp, val = agent.select_action(
                pre_move_state, post_move_states)
            env.move(moves[action_idx])
            reward = 0
            done = env.is_finished()

            traj[episode_agent_indices[player]].append(Transition(
                pre_move_state=pre_move_state,
                action_idx=action_idx,
                logprob=logp,
                reward=reward,
                value=val,
                done=done,
                post_move_states=post_move_states,
            ))
        # Game over - assign final rewards: the difference to the average opponent
        # score. This strikes a balance between using raw scores (not indicative
        # if we won or lost) and just win/loss (too sparse).
        sum_scores = sum(env.scores)
        for i, j in enumerate(episode_agent_indices):
            avg_opp_score = (sum_scores - env.scores[i]) / (num_players - 1)
            traj[j][-1].reward = env.scores[i] - avg_opp_score
        for i in traj:
            data[i].append(Trajectory(traj[i]))

    return data


# ----------------------------------------------------------------------
# Prepare minibatches from trajectories
# ----------------------------------------------------------------------

def flatten_trajectories(
    trajectories: Dict[int, List[Trajectory]]
) -> Dict[int, Dict[str, Any]]:
    """
    Flatten per-agent trajectories into tensors suitable for PPO updates.
    """
    out = {}

    for agent_id, traj_list in trajectories.items():
        pre_move_state_list: list[torch.Tensor] = []
        act_list = []
        logp_list = []
        rew_list = []
        val_list = []
        done_list = []
        post_move_states_list: list[torch.Tensor] = []

        for traj in traj_list:
            for tr in traj.transitions:
                pre_move_state_list.append(tr.pre_move_state)
                act_list.append(tr.action_idx)
                logp_list.append(tr.logprob)
                rew_list.append(tr.reward)
                val_list.append(tr.value)
                done_list.append(float(tr.done))
                post_move_states_list.append(tr.post_move_states)

        val_list.append(0.0)  # bootstrap value for final state

        actions = torch.tensor(act_list, dtype=torch.long)   # [T]
        old_logprobs = torch.tensor(logp_list, dtype=torch.float32)  # [T]
        rewards = torch.tensor(rew_list, dtype=torch.float32)  # [T]
        dones = torch.tensor(done_list)                       # [T]
        values = torch.tensor(val_list, dtype=torch.float32)  # [T+1]

        returns, advantages = compute_gae(
            rewards, values, dones,
            gamma=0.99, lam=0.95
        )  # each: [T]

        out[agent_id] = {
            "pre_move_states": pre_move_state_list,
            "actions": actions,
            "post_move_states_list": post_move_states_list,
            "old_logprobs": old_logprobs,
            "returns": returns,
            "advantages": advantages,
        }
        num_steps = len(actions)
        num_steps_per_game = num_steps / len(traj_list)
        print(
            f"Agent {agent_id} - collected {len(out[agent_id]['actions'])} steps - {num_steps_per_game:.1f} steps per game.")

    return out


# ----------------------------------------------------------------------
# PPO Update
# ----------------------------------------------------------------------
def ppo_update(
    agents: List[Agent],
    value_fn: nn.Module,
    value_optimizer: Adam,
    data_by_agent: Dict[int, Dict[str, Any]],
    minibatch_size: int,
    epochs: int
):
    """
    Run PPO updates for each agent separately.
    NOTE: all agents share value_fn, so value_optimizer updates it globally.
    """

    for agent_id, data in data_by_agent.items():
        pre_move_states = data["pre_move_states"]
        actions = data["actions"]
        post_move_states_list = data["post_move_states_list"]
        old_logprobs = data["old_logprobs"]
        returns = data["returns"]
        advantages = data["advantages"]

        N = len(pre_move_states)        
        for _ in range(epochs):
            perm = torch.randperm(N)
            # TODO: We sample episodes until we have at least one minibatch
            # per agent; so for all agents we typically have just a little bit
            # more than one minibatch; and that last partial batch could be ignored.            
            for start in range(0, N, minibatch_size):
                batch = perm[start:start + minibatch_size]
                prms = [pre_move_states[i] for i in batch]
                psms = [post_move_states_list[i] for i in batch]
                loss, metrics = ppo_loss(
                    policy=agents[agent_id].policy,
                    value_fn=value_fn,
                    pre_move_states=prms,
                    post_move_states_list=psms,
                    action_idx=actions[batch],
                    old_logprob=old_logprobs[batch],
                    returns=returns[batch],
                    advantages=advantages[batch],
                    minibatch_size=minibatch_size
                )
                
                agents[agent_id].optim.zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                agents[agent_id].optim.step()
                value_optimizer.step()


# ----------------------------------------------------------------------
# Evaluation and ranking
# ----------------------------------------------------------------------
def rank_players(
        game_results,
        num_players,
        first_player_skill_val=1.0,
        iterations=100,
        lr=0.1):
    # Rank the players using a Plackett-Luce model. The first player is assumed
    # to have a known skill level (first_player_skill_val), while the others are
    # learned, so the first player serves as a reference point.
    skills_improving = torch.zeros(num_players - 1, requires_grad=True)

    # Only pass the improving agents to the optimizer
    optimizer = torch.optim.Adam([skills_improving], lr=lr)

    # Pre-wrap fixed skill as a non-grad tensor
    fixed_skill = torch.log(torch.tensor([float(first_player_skill_val)]))

    for i in range(iterations):
        optimizer.zero_grad()

        # Combine fixed agent (index 0) with improving agents
        # This creates a full vector [fixed, agent1, agent2, ...]
        all_skills = torch.cat([fixed_skill, skills_improving])
        total_nll = torch.tensor(0.0)

        for participants, winner_idx in game_results:
            # We index into the combined vector
            participant_skills = all_skills[list(participants)]
            winner_skill = all_skills[winner_idx]

            # NLL = log(sum(exp(participants))) - winner_skill
            log_prob = winner_skill - \
                torch.logsumexp(participant_skills, dim=0)
            total_nll -= log_prob

        total_nll.backward()
        optimizer.step()

    with torch.no_grad():
        final_skills_log = torch.cat([fixed_skill, skills_improving])
        final_skills_exp = torch.exp(final_skills_log)
        order = torch.argsort(final_skills_exp, descending=True)

    return final_skills_exp.tolist(), order.tolist()


def evaluate(agents: List[Agent], num_players: int) -> Tuple[List[int], List[float]]:
    """
    Evaluate agents by playing them against each other in a series of games
    with randomly picked players including known baseline players, and compute
    their ranks + skills.
    Return: Tuple of
      - Agent indices ordered from best to worst
      - The respective skills (same order)
    """
    # 1. Create players from agents, and mix in PlanningPlayer as a baseline.
    players = [PlanningPlayer()] + [NeuralPlayer(a) for a in agents]
    # 2. Play rounds (not games - due to random selection, they should all get
    # their fair share of being dealer). We aim for ~50 games per player.
    num_games = int(100 * len(players) / num_players)
    game_results = []
    print("Evaluating agents...")
    for _ in range(num_games):
        selected_indices = random.sample(range(len(players)), num_players)
        selected_players = [players[i] for i in selected_indices]
        scores = play_round(selected_players, 0)
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        winning_player_index = selected_indices[winner_index]
        game_results.append((set(selected_indices), winning_player_index))
    # 3. Rank players using Plackett-Luce model
    skills, order = rank_players(game_results, len(players))
    order = [index for index in order if index != 0]
    # 4. Omit PlanningPlayer from returned list, return agents in ranked order.
    skills = [skills[i] for i in order]
    order = [i - 1 for i in order]
    ftd_skill_list = ", ".join([f"{x:.2f}" for x in skills])    
    print(f"Skills + order (1.0 == PlanningPlayer): {ftd_skill_list}, {order}")

    return order, skills


# ----------------------------------------------------------------------
# High-level training loop
# ----------------------------------------------------------------------

def train(
    num_iterations: int,
    episodes_per_iter: int,
    num_players: int,
    minibatch_size: int = 512,
    epochs: int = 2
):
    agents = []
    # Number of agents we train in an iteration.
    num_agents_train = 2 * num_players
    # We keep copies of the best ones.
    num_best_agents = int(0.2 * num_agents_train)
    num_best_agents = 0
    # So overall there are num_agents + num_best_agents agents.
    state_dim = 57
    value_net = ValueNet(state_dim)
    value_optim = Adam(value_net.parameters(), lr=1e-3)

    for _ in range(num_agents_train):
        policy_net = PolicyNet(state_dim)
        policy_optim = Adam(policy_net.parameters(), lr=args.lr)
        agents.append(Agent(policy_net, policy_optim, value_net))
    best_agents: dict[float, Agent] = {}

    def env_constructor(dealer): return GameState(
        num_players=num_players,
        dealer=dealer)

    for iteration in range(num_iterations):
        # 1. Self-play
        trajectories = collect_episodes(
            agents,
            env_constructor,
            episodes_per_iter,
            num_players,
            min_examples_per_player=minibatch_size
        )

        # 2. Flatten storage and compute GAE
        data = flatten_trajectories(trajectories)

        # 3. PPO update
        ppo_update(
            agents,
            value_net,
            value_optim,
            data,
            minibatch_size,
            epochs
        )

        # 4. Evaluation & shuffling.
        if iteration % 5 == 0 and iteration > 0:
            order, skills = evaluate(agents + list(best_agents.values()), num_players)
            agents = [agents[i] for i in order[:num_agents_train]]     
            best_agents = {skills[i]: copy.deepcopy(agents[i]) for i in range(num_best_agents)}
            print(f"Best agents' skills: {list(best_agents.keys())}")

        print(f"Iteration {iteration} completed.")
        
    return agents


def main():
    num_players = 5
    agents = train(
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        num_players=num_players,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Find the best agent.
    print("Finding best agent...")
    players = [lambda: NeuralPlayer(agents[i]) for i in range(num_players)]
    wins = [0] * len(players)
    for reps in range(0, 50):
        scores = play_game([p() for p in players])
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        wins[winner_index] += 1
    print(f"Agent wins: {wins}")
    best_agent_index = max(range(len(wins)), key=lambda i: wins[i])
    print(f"Best agent is Agent {best_agent_index}.")

    # Play a tournament against RandomPlayer. This is a good sanity check that
    # training works - an untrained net shouldn't do any better than random,
    # while a trained net should win even with a very primitive feature set.
    print("Tournament against GreedyShowPlayerWithFlip:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: GreedyShowPlayerWithFlip(),
        num_games=200
    )
    print("Tournament against PlanningPlayer:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: PlanningPlayer(),
        num_games=200
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main()
