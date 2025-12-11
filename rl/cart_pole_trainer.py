import time
import torch
import torch.nn as nn
import numpy as np
import cart_pole_simulator
import matplotlib.pyplot as plt
import datetime
from torch.utils.tensorboard import SummaryWriter

# Uses tensorboard for logging - start in the venv via
# # tensorboard --logdir=runs --reload_interval 5
NUM_EPOCHS = 400
BATCH_SIZE = 512
TIME_HORIZON = 10

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def experiment_name():
    now = datetime.datetime.now()
    return "pole_trainer" + now.strftime('%Y%m%d_%H%M%S')

# Log book: So far, tried vanilla REINFORCE w/o baseline.
# Lots of combinations that work and that don't work.
# One combo that works reproducibly: B=512, network 4->128->32->5: [-10, -5, 0, 5, 10], lr=1e-3.
# Key aspects:
# - Large batch sizes stabilize training - 512 much better (meaning incremental progress) than
#   128 (which is noisy and jumping to a good solution seems to be a matter of luck).
# - Larger networks work better than smaller ones
# - GELU/ReLU works a bit better than Tanh, and Sigmoid hardly works at all
# - More outputs (5) work better than fewer (3), and giving high force options works better too.
# - Weight decay, even small values, doesn't work.
# - The value of the objective function / the expected rewards, as training progresses, looks
#   pretty different from standard supervised learning - there are plateaus, there are areas of
#   rapid improvements, and I feel the absolute value of the objective function is not very
#   informative (eg getting from surviving 1.5s to 2s seems to be much harder than getting from
#   2s to 6s, according to how the objective function / rewards evolve).
#   I start to understand intuitively why baselines matter.


class Policy(nn.Module):
    """
    Four float inputs: cart position, cart velocity, pole angle, pole angular velocity
    Output is log of probabilities for one of five classes. Can be converted to force via
    output_to_force.
    Inputs are assumed to be batched, first dimension being batch.
    """

    def __init__(self):
        super().__init__()
        self.forces_lut = torch.tensor([-10., -5., 0.0, 5., 10.0])
        self.linear_stack = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, len(self.forces_lut)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.linear_stack(x)

    def output_to_force(self, x):
        # TODO: Make the constant a learnable parameter. Not sure how.
        # Or don't use a softmax but output the force directly - but not sure how
        # the math works without discrete actions. We need some form of
        # sampling actions, so if it's not a categorical distribution on top of softmax,
        # we'd need something like a normal distribution around the network output.
        return self.forces_lut[x]


# The behavior of the objective function is *very* counter-intuitive.
# We *maximize* a function that's effectively \sum_b \sum_t log p(a_t|s_t) G_t.
# That's an estimator of the expected return (G_t being the discounted rewards from t
# onwards). We differentiate that and tell an optimizer to *maximize* it. Yet, as
# training proceeds, this value becomes more and more *negative* (e.g -50 -> -200 etc.),
# but training does work (the actually observed rewards / episode lengths get longer).
# Why? Mathematically, it makes sense that the objective function is negative - we multiply
# a negative log-prob with a positive return. But if we maximize this, shouldn't the function
# get closer to 0, not become more negative?
# I *think* the reason lies in the episodes taking longer as we get better. The optimizer
# changes the policy such that the expression above - \sum_b \sum_t log p(a_t|s_t) G_t -
# becomes higher (less negative, closer to 0), by making "good actions" more likely (that's
# the actions that led to higher rewards - here, the ones from trajectories that took longer).
# If we recompute this objective function without resampling, it would indeed be higher.
# However, in the next episode, this new policy and improved policy will now lead to longer
# episodes, thus higher rewards, thus - because we multiply it with negative log-probs - to
# a more negative objective function. This is a subtle but crucial difference between the
# reinforcement and supervised setting: In the supervised setting, we keep computing the
# objective function (loss) over the same (or identically distributed) data, whereas in
# RL (on-policy at least), that data keeps changing, and thus what the value of the objective
# function means keeps changing.
# This also explains another counter-intuitive observation: The typical shape of the objective
# function across epochs is first somewhat flat, then a rapid decay as the episodes get
# longer and we thus see more rewards; then the objective function increases again. Why?
# This increase is the part where things behave as expected: The policy is now so good that
# the duration of the episodes - and hence the rewards - do not change anymore; and thus
# we're now in the regime where the optimizer behaves as expected: it increases the value
# of the objective function by making the log-likelihoods of the best actions more likely
# (while the rewards part of the objective function stays constant). It seems even more
# important than in SL to plot not just this objective function, but other metrics of
# success like observed rewards / time of simulations etc.
#
# A side note, it is very counter-intuitive to me that the value is negative to begin with,
# because if we only give positive rewards, how can the estimator for expected returns
# be negative?
if __name__ == "__main__":
    policy = Policy().to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3, maximize=True)
    writer = SummaryWriter('runs/' + experiment_name(), flush_secs=5)
    for epoch in np.arange(0, NUM_EPOCHS):
        t_epoch_start = time.time()
        expected_rewards = []
        logprobs = []  # T array of B tensors.
        rewards = []  # T array of B numpy arrays
        # Initialize a batch of simulations with mildly random initial points.
        x0s = np.random.uniform(-0.5, 0.5, size=BATCH_SIZE).astype(np.float32)
        v0s = np.random.uniform(-0.05, 0.05, size=BATCH_SIZE).astype(np.float32)
        angle0s = 2*np.random.uniform(-0.5, 0.5, size=BATCH_SIZE).astype(np.float32)*10*np.pi/180
        sim = cart_pole_simulator.SimpleCartPole(
            x=x0s, v=v0s, angle=angle0s, visualize=False)

        # Run the simulation in a batched manner. That is, at every time step, we
        # simulate a whole batch of agents, and collect their actions & rewards.
        for i, t in enumerate(np.arange(0, TIME_HORIZON, sim.dt)):
            inputs = torch.tensor(sim.get_state(), dtype=torch.float32)
            def normalize_inputs(x):
                # x is a B x 4 tensor, the four dims being x, v, angle, rot_v.
                x[0] = x[0] / sim.TRACK_LENGTH/2
                x[1] = x[1] / 10.0  # empirically, velocities are mostly < 10.
                x[2] = x[2] / (np.pi/2)
                return x

            inputs = normalize_inputs(inputs)
            outputs = policy(inputs)  # torch.tensor B x NumActions - log probs.
            action = torch.distributions.Categorical(
                logits=outputs).sample()  # torch.tensor B - sampled action indices.
            sim.apply_and_step(policy.output_to_force(action).detach().numpy())

            # Abort if < 5% of simulations still running. This is mathematically a bit questionable,
            # but it saves a lot of time because of the Powerlav distribution of simulation times
            # (ie a few of them dominate the overall time), and allows us to make fast progress during
            # # the early epochs when most simulations fail quickly.
            sim_state = sim.ok(quiet=True)            
            if sim_state.sum() / sim_state.size < 0.05:
                break

            # Store the rewards and the logprob tensors of the sampled actions; we will use them
            # after the simulation ends to compute our objective function.
            rewards.append(np.where(sim_state, sim.dt, 0.0))
            logprobs.append(torch.gather(
                outputs, dim=1, index=action.unsqueeze(1)).flatten())
        # Calculate returns aka discounted rewards
        returns = []  # T array of B np arrays
        gamma = 0.99
        G_t = np.zeros(BATCH_SIZE, dtype=np.float32)
        for r in reversed(rewards):
            G_t = r + gamma*G_t
            returns.insert(0, G_t)
        
        # The objective function that we'll maximize.
        expected_returns = torch.sum(
            torch.stack(logprobs, dim=1) * torch.tensor(returns).T) / BATCH_SIZE
        epoch_dur = time.time() - t_epoch_start
        print(f"epoch: {epoch}, dur: {epoch_dur:.2f}s, E[R]: {expected_returns:.2f}, "
              f"observed <reward>: {np.sum(rewards)/BATCH_SIZE:.1f}")

        optim.zero_grad()
        expected_returns.backward()
        optim.step()

        # Write Tensorboard stats.
        writer.add_scalar('Train/Expected Returns',
                          expected_returns.item(), epoch)
        writer.add_scalar('Train/Observed <reward>',
                          np.sum(rewards)/BATCH_SIZE, epoch)
        #for name, param in policy.named_parameters():
        #    writer.add_histogram(f'Weights/{name}', param.data, epoch)
        #    # Log the parameter's gradient distribution (if gradients exist)
        #    if param.grad is not None:
        #        writer.add_histogram(
        #            f'Gradients/{name}', param.grad.data, epoch)

    writer.close()

    # Run a simulation with the model.
    sim = cart_pole_simulator.SimpleCartPole(
        x=np.array([0.0]), v=np.array([0.0]), angle=np.array([5*np.pi/180]), visualize=True)
    for t in np.arange(0, 10, sim.dt):
        inputs = torch.tensor(sim.get_state(), dtype=torch.float32)
        outputs = policy(inputs)
        action = torch.distributions.Categorical(logits=outputs).sample()
        sim.apply_and_step(policy.output_to_force(action).detach().numpy())
        sim.maybe_plot()
        if not sim.ok():
            print("Fail!")
            break
