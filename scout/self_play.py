from __future__ import annotations
from math import inf
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Dict, Any
from common import InformationState, Move, Player
from game_state import GameState
from main import play_game, play_tournament
from players import GreedyShowPlayerWithFlip, PlanningPlayer, RandomPlayer

# -----------------------------
# TODO:
# P1:
# - Fix encode_information_state to use proper features. Consider at least the
#   ones available to PlanningPlayer (number of sets, runs) - neural net should
#   be able to get about as good as.
# - Find a way to measure progres (other than just playing against PlanningPlayer)
# P2:
# - Check the rewards. Right now we use points, as they are awarded along the
#   way. I wonder if a) this makes training unstable because it may lead to
#   mutual scout + show loops where both players gain points b) lack of
#   normalization - see my ISMCTS experiences, trees with high scores weren't
#   necessarily indicative of winning, it was the difference to other scores.
#   c) lower scores but winning is better than everybody getting high scores.
#   d) what matters is score *difference* at end of game, not absolute score
#      during game. Magntiude of score difference still holds signal, but worried
#      about reward hacking.
#   Maybe use (player_score - mean_opponent_score) / max_possible_score?
#   Preliminary experiemnts - subtracting -0.5 every time step to encourage
#   faster wins - seem to help already: The episodes get shorter, but more
#   importantly the trained agent wins more often against GreedyShowPlayerWithFlip.
# - Fix post_move_states to avoid code duplication, or at least write tests.
# - Tests for compute_gae
# P3:
# - Can we do imitation learning to bootstrap training? Instantiate a
#   PlanningPlayer and see which action it picks, then replace the action
#   index with that, not the one we'd have picked. Feels like the loss is
#   wrong though.
# - Keep older, good agents around, instead of just the same 5 all the time.
#   Diversity!
# -----------------------------


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
    return torch.tensor(
        [info_state.scores[info_state.current_player]] * 57, dtype=torch.float)


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
    pre_move_state: InformationState
    action_idx: int             # index into the action list
    logprob: float              # scalar
    reward: float               # scalar
    value: float                # scalar
    done: bool                  # episode termination flag
    post_move_states: tuple[InformationState, ...]


@dataclass
class Trajectory:
    """
    Sequence of Transitions for a single agent over one episode.
    """
    transitions: List[Transition]


# ----------------------------------------------------------------------
# PPO Loss
# ----------------------------------------------------------------------

def ppo_loss(
    policy: PolicyNet,
    value_fn: nn.Module,
    pre_move_states: list[InformationState],
    post_move_states_list: List[tuple[InformationState, ...]],
    action_idx: torch.Tensor,     # shape: [batch]
    old_logprob: torch.Tensor,    # shape: [batch]
    returns: torch.Tensor,        # shape: [batch]
    advantages: torch.Tensor,     # shape: [batch]
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
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
        post_move_states = post_move_states_list[i]
        encoded_pre_move_state = encode_information_state(
            pre_move_states[i]).unsqueeze(0)  # [1, D]
        encoded_post_move_states = encode_post_move_states(
            post_move_states)  # [N, D]

        logits = policy(encoded_pre_move_state,
                        encoded_post_move_states)    # shape: [1, num_actions]
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
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    # Value loss
    # TODO: Cache this from above, or featurize outside this function.
    pre_move_states_enc = torch.stack(
        [encode_information_state(s) for s in pre_move_states])
    values = value_fn(pre_move_states_enc).squeeze(-1)        # shape: [batch]
    value_loss = vf_coef * torch.mean((returns - values) ** 2)

    # Entropy bonus
    entropy_bonus = ent_coef * entropies.mean()

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


# ----------------------------------------------------------------------
# Agent wrapper (policy only — value is shared globally)
# ----------------------------------------------------------------------

class Agent:
    def __init__(self, policy: PolicyNet, value_fn: nn.Module):
        self.policy = policy
        self.value_fn = value_fn

    def select_action(
        self,
        pre_move_state: InformationState,         # shape: [obs_dim]
        post_move_states: tuple[InformationState, ...],
    ) -> Tuple[int, float, float]:
        """
        Compute distribution over legal moves and sample one.
        Returns:
            action_idx, logprob of sampled move, value_prediction
        """
        encoded_pre_move_state = encode_information_state(
            pre_move_state).unsqueeze(0)  # [1, D]
        encoded_post_move_states = encode_post_move_states(
            post_move_states)  # [N, D]

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
    num_episodes: int,
) -> Dict[int, List[Trajectory]]:
    """
    All agents play together in a multi-player environment.
    Returns dict: agent_id -> list of Trajectory (one per episode)
    """
    data: Dict[int, List[Trajectory]] = {i: [] for i in range(len(agents))}

    # Avoid disadvantaging players by ensuring each deals equal number of times
    assert num_episodes % len(agents) == 0, \
        "Number of episodes must be multiple of number of agents."
   
    for episode in range(num_episodes):
        dealer = episode % len(agents)
        env = env_constructor(dealer)
        # For now, we just flip like GreedyShowPlayer would.
        # Eventually, we should learn a policy for that, but it complicates
        # training a bit because the flip decision requires another network.
        gsp = GreedyShowPlayerWithFlip()
        env.maybe_flip_hand([lambda h: gsp.flip_hand(h) for _ in agents])
        traj = {i: [] for i in range(len(agents))}

        done = False
        while not done:
            player = env.current_player
            agent = agents[player]

            pre_move_state = env.info_state()
            moves, post_move_states = pre_move_state.post_move_states()
            action_idx, logp, val = agent.select_action(
                pre_move_state, post_move_states)
            previous_score = env.scores[player]
            env.move(moves[action_idx])
            reward = env.scores[player] - previous_score
            done = env.is_finished()
            

            traj[player].append(Transition(
                pre_move_state=pre_move_state,
                action_idx=action_idx,
                logprob=logp,
                reward=reward,
                value=val,
                done=done,
                post_move_states=post_move_states,
            ))

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
        pre_move_state_list: List[InformationState] = []
        act_list = []
        logp_list = []
        rew_list = []
        val_list = []
        done_list = []
        post_move_states_list: List[tuple[InformationState, ...]] = []

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

        actions = torch.tensor(act_list, dtype=torch.long)  # [T]
        old_logprobs = torch.tensor(logp_list, dtype=torch.float32)  # [T]
        rewards = torch.tensor(rew_list, dtype=torch.float32)       # [T]
        dones = torch.tensor(done_list)                    # [T]
        values = torch.tensor(val_list, dtype=torch.float32)        # [T+1]

        returns, advantages = compute_gae(
            rewards, values, dones,
            gamma=0.99, lam=0.95
        )                                                 # each: [T]

        # TODO: Consider featurizing in here, not in ppo_update
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
        print(f"Agent {agent_id} - collected {len(out[agent_id]['actions'])} steps - {num_steps_per_game} steps per game.")
    


    return out


# ----------------------------------------------------------------------
# PPO Update
# ----------------------------------------------------------------------

def ppo_update(
    agents: List[Agent],
    value_fn: nn.Module,
    optimizers: List[Adam],
    value_optimizer: Adam,
    data_by_agent: Dict[int, Dict[str, Any]],
    minibatch_size: int,
    epochs: int,
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
        idxs = torch.arange(N)

        for _ in range(epochs):
            perm = torch.randperm(N)
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
                )
                print(f"Agent {agent_id} - Loss: {loss.item():.4f}, "
                      f"Policy Loss: {metrics['policy_loss']:.4f}, "
                      f"Value Loss: {metrics['value_loss']:.4f}, "
                      f"Entropy: {metrics['entropy']:.4f}")

                optimizers[agent_id].zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                optimizers[agent_id].step()
                value_optimizer.step()


# ----------------------------------------------------------------------
# High-level training loop
# ----------------------------------------------------------------------

def train(
    agents: List[Agent],
    value_fn: nn.Module,
    policy_optims: List[Adam],
    value_optim: Adam,
    env_constructor,
    num_iterations: int,
    episodes_per_iter: int,
    minibatch_size: int = 512,
    epochs: int = 4,
):
    for iteration in range(num_iterations):
        # 1. Self-play
        trajectories = collect_episodes(
            agents,
            env_constructor,
            episodes_per_iter,
        )

        # 2. Flatten storage and compute GAE
        data = flatten_trajectories(trajectories)

        # 3. PPO update
        ppo_update(
            agents,
            value_fn,
            policy_optims,
            value_optim,
            data,
            minibatch_size,
            epochs,
        )

        print(f"Iteration {iteration} completed.")


class NeuralPlayer(PlanningPlayer):
    # A player wrapping an agent that wraps a neural policy and value network.    
    def __init__(self, agent: Agent):
        self.agent = agent

    def select_move(self, info_state: InformationState) -> Move:
        moves, post_move_states = info_state.post_move_states()
        action_idx, _, _ = self.agent.select_action(
            info_state,
            post_move_states
        )
        return moves[action_idx]

    def flip_hand(self, hand: Sequence[tuple[int, int]]) -> bool:
        # TODO: Implement. Would be nice to use value_fn for that, but I'm not
        # sure how to derive an information state.
        # For now, use PlanningPlayer's heuristic method.
        return super().flip_hand(hand)
        


def main():
    agents = []
    num_agents = 5
    state_dim = 57
    value_net = ValueNet(state_dim)
    for _ in range(num_agents):
        policy_net = PolicyNet(state_dim)
        agents.append(Agent(policy_net, value_net))
    policy_optimizers = [
        Adam(
            agent.policy.parameters(),
            lr=3e-4) for agent in agents]
    value_optimizer = Adam(value_net.parameters(), lr=1e-3)
    train(
        agents,
        value_net,
        policy_optimizers,
        value_optimizer,
        env_constructor=lambda dealer: GameState(
            num_players=num_agents,
            dealer=dealer),
        num_iterations=10,
        episodes_per_iter=20,
    )

    # Find the best agent.
    print("Finding best agent...")
    players = [lambda : NeuralPlayer(agents[i]) for i in range(num_agents)]
    wins = [0] * len(players)
    for reps in range(0, 100):
        scores = play_game([p() for p in players])
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        wins[winner_index] += 1
    print("Agent wins:", wins)        
    best_agent_index = max(range(len(wins)), key=lambda i: wins[i])
    print(f"Best agent is Agent {best_agent_index}.")

    # Play a tournament against RandomPlayer. This is a good sanity check that
    # training works - an untrained net shouldn't do any better than random,
    # while a trained net should win even with a very primitive feature set.
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: GreedyShowPlayerWithFlip(),
    )


if __name__ == "__main__":
    main()
# - Tests for compute_gae
