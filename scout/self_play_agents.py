# This file contains different agents for use in neural self-play.
# An agent is a wrapper around a neural policy and a (possibly shared) value
# network. We train the networks inside; at inference time, the agent is wrapped
# in a Player and used in tournaments.

from collections.abc import Callable
from math import inf
import torch
from torch import nn

from common import InformationState, StateAndScoreRecord
from neural_value_function import featurize


class Agent:
    # This is a base class for all agents.
    # It wraps a policy network and a value function network, as well as an
    # encoder for these two networks.
    def __init__(
            self,
            policy: nn.Module,
            policy_optim: torch.optim.Optimizer,
            value_fn: nn.Module,
            value_optim: torch.optim.Optimizer,
            info_state_encoder: Callable[[InformationState], torch.Tensor]):
        self.policy = policy
        self.policy_optim = policy_optim
        self.value_fn = value_fn
        self.value_optim = value_optim
        self.info_state_encoder = info_state_encoder

    def _encode_info_state_list(
            self, info_states: tuple[InformationState, ...]) -> torch.Tensor:
        # Encode a list of info states into a tensor of shape [N, obs_dim]
        return torch.stack([self.info_state_encoder(s) for s in info_states])

    def encode_inputs(
        self,
        pre_move_state: InformationState,
        post_move_states: tuple[InformationState, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Separate function for encoding inputs so we can use those not just
        # for inference (select_action) but also for training.
        encoded_pre_move_state = self.info_state_encoder(
            pre_move_state).unsqueeze(0)  # [1, D]
        encoded_post_move_states = self._encode_info_state_list(
            post_move_states)  # [N, D]
        return encoded_pre_move_state, encoded_post_move_states

    def select_action(
        self,
        encoded_pre_move_state: torch.Tensor,         # shape: [obs_dim]
        encoded_post_move_states: torch.Tensor,        # shape: [N, obs_dim]
    ) -> tuple[int, float, float]:
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


class AgentCollection:
    @staticmethod
    def create_agents(num_agents: int) -> list[Agent]:
        raise NotImplementedError

    @staticmethod
    def load_agents(
            policy_paths: list[str],
            value_fn_path: str | None = None) -> list[Agent]:
        raise NotImplementedError

    @staticmethod
    def save_agents(
            agents: list[Agent],
            policy_paths: list[str],
            value_fn_path: str):
        raise NotImplementedError


#############################################################################
# A simple-ish baseline policy and value network - lots of feature engineering,
# simple feedforward MLP.
# Reuses featurization code I've originally added for ISMCTS value functions.
##############################################################################
def simple_encode_information_state(
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


class SimplePolicyNet(nn.Module):
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


class SimpleValueNet(nn.Module):
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


class SimpleAgent(Agent):
    def __init__(
            self,
            state_dim: int,
            policy_lr: float,
            value_fn: nn.Module,
            value_optim: torch.optim.Optimizer):
        policy = SimplePolicyNet(state_dim)
        super().__init__(
            policy=policy,
            policy_optim=torch.optim.Adam(policy.parameters(), lr=policy_lr),
            value_fn=value_fn,
            value_optim=value_optim,
            info_state_encoder=simple_encode_information_state
        )


class SimpleAgentCollection(AgentCollection):
    @staticmethod
    def create_agents(num_agents: int) -> list[Agent]:
        state_dim = 57
        policy_lr = 3e-3
        value_fn = SimpleValueNet(state_dim)
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
        return [SimpleAgent(state_dim, policy_lr, value_fn, value_optim)
                for _ in range(num_agents)]

    @staticmethod
    def load_agents(
            policy_paths: list[str],
            value_fn_path: str |None = "") -> list[Agent]:
        # This function allows for loading agents from disk; this can be useful
        # for checkpointing/resumption, or simply to load a previously trained agent
        # for evaluation.
        # When used for evaluation, the value function is not used, so the
        # value_fn_path can be left empty, and the value function will be randomly
        # initialized.
        agents = SimpleAgentCollection.create_agents(len(policy_paths))
        for i, policy_path in enumerate(policy_paths):
            agents[i].policy.load_state_dict(torch.load(policy_path))
        if value_fn_path:
            agents[0].value_fn.load_state_dict(torch.load(value_fn_path))
        return agents

    @staticmethod
    def save_agents(
            agents: list[Agent],
            policy_paths: list[str],
            value_fn_path: str):
        for i, agent in enumerate(agents):
            torch.save(agent.policy.state_dict(), policy_paths[i])
        torch.save(agents[0].value_fn.state_dict(), value_fn_path)


