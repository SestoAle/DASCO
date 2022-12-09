import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *
from torch.distributions import Categorical, Beta, Normal
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.pi = (torch.acos(torch.zeros(1)) * 2).to(device)

EPS = 1e-6
LOG_SIG_MAX = 2
LOG_SIG_MIN = -5

class Policy(nn.Module):
    def __init__(self, state_dim, embedding_arch, action_size=4, action_type='discrete', max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(Policy, self).__init__()

        # Policy hyperparameters
        self.action_size = action_size
        self.state_dim = state_dim
        self.max_action_value = max_action_value
        self.min_action_value = min_action_value
        self.action_type = action_type

        # Layers specification
        self.embedding_l = embedding_arch(state_dim)

        self.mean = nn.Linear(self.embedding_l.output_dim, self.action_size)
        self.log_std = nn.Linear(self.embedding_l.output_dim, self.action_size)

    def forward(self, inputs):
        state = torch.reshape(inputs, (-1, self.state_dim)).float()
        x = self.embedding_l(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        x = torch.cat([mean, log_std], dim=1)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, embedding_arch, **kwargs):
        super(Critic, self).__init__()

        # Layers specification
        self.embedding_q1_l = embedding_arch(state_dim + action_dim)
        self.q1_l = nn.Linear(self.embedding_q1_l.output_dim, 1)

        self.embedding_q2_l = embedding_arch(state_dim + action_dim)
        self.q2_l = nn.Linear(self.embedding_q1_l.output_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.embedding_q1_l(x)
        q1 = self.q1_l(q1)

        q2 = self.embedding_q2_l(x)
        q2 = self.q2_l(q2)
        return q1, q2

    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.embedding_q1_l1(x))
        q1 = F.relu(self.embedding_q1_l2(q1))
        q1 = self.q1_l(q1)
        return q1

# Actor-Critic PPO. The Actor is independent by the Critic.
class CQLAgent(nn.Module):
    # SAC agent
    def __init__(self, state_dim, policy_embedding, critic_embedding, discount=0.99, p_lr=3e-4, v_lr=3e-4,
                 frequency_mode='episodes', memory=50, policy_freq=1, alpha=0.2, tau=0.005, batch_size=32,
                 num_itr=20, name='sac', action_size=4, max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(CQLAgent, self).__init__()
        # Model parameters
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.batch_size = batch_size
        self.num_itr = num_itr
        self.name = name
        self.frequency_mode = frequency_mode
        self.state_dim = state_dim
        self.action_type = "continuous"
        # Functions that define input and network specifications
        self.alpha = alpha
        self.policy_freq = policy_freq
        self.tau = tau
        self.model_name = name

        # CLQ param
        self.with_lagrange = False
        self.temperature = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 10.0


        # PPO hyper-parameters
        self.discount = discount
        # Action hyper-parameters
        # Types permitted: 'discrete' or 'continuous'. Default: 'discrete'
        self.action_size = action_size
        # min and max values for continuous actions
        self.action_min_value = min_action_value
        self.action_max_value = max_action_value
        self.alpha_tuning = True
        self.lagrange = False
        # Distribution type for continuous actions

        self.memory = memory

        self.policy = Policy(state_dim, policy_embedding, self.action_size, self.action_type,
                             self.action_max_value, self.action_min_value).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.p_lr, betas=(0.9, 0.999))

        if self.alpha_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((action_size, )).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=v_lr)

        self.critic = Critic(self.state_dim, self.action_size, critic_embedding).to(device)
        self.critic_loss = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.v_lr, betas=(0.9, 0.999))

        # Define the targets and init them
        self.policy_target = Policy(state_dim, policy_embedding, self.action_size, self.action_type,
                             self.action_max_value, self.action_min_value).to(device)
        self.critic_target = Critic(self.state_dim, self.action_size, critic_embedding).to(device)
        self.copy_target(self.policy_target, self.policy, self.tau, True)
        self.copy_target(self.critic_target, self.critic, self.tau, True)

        self.total_itr = 0

    def forward(self, state, distribution="gaussian", deterministic=False):

        action_scale = (self.action_max_value - self.action_min_value) / 2.
        action_bias = (self.action_max_value + self.action_min_value) / 2.
        probs = self.policy(state)

        if distribution == "gaussian":
            mean = probs[:, :self.action_size]
            log_std = probs[:, self.action_size:]
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * action_scale + action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + EPS)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * action_scale + action_bias
        elif distribution == 'beta':
            alpha = probs[:, :self.action_size]
            beta = probs[:, self.action_size:]
            alpha = F.softplus(alpha) + 1
            beta = F.softplus(beta) + 1
            dist = Beta(alpha, beta)
            # Sample an action from beta distribution
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # If there are more than 1 continuous actions, do the mean of log_probs
            if self.action_size > 1:
                log_prob = torch.sum(log_prob, dim=1, keepdim=True)
            # Standardize the action between min value and max value
            action = self.action_min_value + (
                    self.action_max_value - self.action_min_value) * action

        return action, log_prob, probs, None

    def set_dataset(self, dataset):
        self.states = torch.from_numpy(np.asarray(dataset['states'])).to(device)
        self.next_states = torch.from_numpy(np.asarray(dataset['next_states'])).to(device)
        self.dones = torch.from_numpy(np.asarray(dataset['dones'])).to(device)
        self.actions = torch.from_numpy(np.asarray(np.asarray(dataset['actions']))).to(device)
        self.rewards = torch.from_numpy(np.asarray(np.asarray(dataset['rewards']))).to(device).float()

        self.rewards = torch.reshape(self.rewards, (-1, 1))
        self.dones = torch.reshape(self.dones, (-1, 1))

    def diagonal_gaussian_sample(self, mean, log_std):
        eps = torch.randn(mean.shape).to(device)
        std = torch.exp(log_std)
        sample = mean + std * eps
        return sample

    def squash(self, x, input_min, input_scale, output_scale, output_min):
        return (x - input_min) / input_scale * output_scale + output_min

    def clip_but_pass_gradient(self, x, low=-1., high=1.):
        # From Spinning Up implementation
        with torch.no_grad():
            clip_up = (x > high).float()
            clip_low = (x < low).float()

            return x + (high - x) * clip_up + (low - x) * clip_low

    def diagonal_gaussian_logprob(self, gaussian_samples, mean, log_std):
        assert len(gaussian_samples.shape) == 2
        n_dims = gaussian_samples.shape[1]

        std = torch.exp(log_std)
        log_probs_each_dim = -0.5 * torch.log(2 * torch.pi) - log_std - (gaussian_samples - mean) ** 2 / (2 * std ** 2 + EPS)

        # For a diagonal Gaussian, the probability of the random vector is the product of the probabilities
        # of the individual random variables. We're operating in log-space, so we can just sum.
        log_prob = torch.sum(log_probs_each_dim, dim=1, keepdim=True)

        return log_prob

    def tanh_diagonal_gaussian_logprob(self, gaussian_samples, tanh_gaussian_samples, mean, log_std):
        log_prob = self.diagonal_gaussian_logprob(gaussian_samples=gaussian_samples, mean=mean, log_std=log_std)
        # tf.tanh can sometimes be > 1 due to precision errors
        tanh_gaussian_samples = self.clip_but_pass_gradient(tanh_gaussian_samples, low=-1, high=1)

        correction = torch.sum(torch.log(1 - tanh_gaussian_samples ** 2 + EPS), dim=1, keepdim=True)
        log_prob -= correction

        return log_prob

    # Assign to model_a the weights of model_b. Use it for update the target networks weights.
    def copy_target(self, target_model, main_model, tau=1e-2, init=False):
        if init:
            for a, b in zip(target_model.parameters(), main_model.parameters()):
                a.data.copy_(b.data)
        else:
            for a, b in zip(target_model.parameters(), main_model.parameters()):
                a.data.copy_((1 - tau) * a.data + tau * b.data)

    def update(self):
        c_losses = []
        p_losses = []
        for _ in range(self.num_itr):
            self.total_itr += 1
            # Take a mini-batch of batch_size experience
            mini_batch_idxs = np.random.randint(0, len(self.states), size=self.batch_size)

            states_mb = self.states[mini_batch_idxs]
            next_states_mb = self.next_states[mini_batch_idxs]
            actions_mb = self.actions[mini_batch_idxs]
            rewards_mb = self.rewards[mini_batch_idxs]
            rewards_mb = rewards_mb.view(-1, 1)
            dones_mb = self.dones[mini_batch_idxs]
            dones_mb = dones_mb.view(-1, 1)

            actor_loss = None
            action, logprob, probs, dist = self.forward(states_mb)
            current_Q1, current_Q2 = self.critic(states_mb, action)
            q = torch.min(current_Q1, current_Q2)
            p_loss = ((self.alpha * logprob) - q).mean()

            self.policy_optimizer.zero_grad()
            p_loss.backward()
            self.policy_optimizer.step()

            p_losses.append(p_loss.detach().cpu())

            if self.alpha_tuning:
                alpha_loss = -(self.log_alpha * (logprob + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()

            with torch.no_grad():
                target_Q = self.compute_target(next_states_mb, rewards_mb, dones_mb)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states_mb, actions_mb)

            # CQL addon
            num_repeat = 10
            random_actions = torch.FloatTensor(self.batch_size * num_repeat, self.action_size).uniform_(-1, 1).to(device)
            temp_states_mb = states_mb.unsqueeze(1).repeat(1, num_repeat, 1).view(self.batch_size * num_repeat, self.state_dim)
            temp_next_states_mb = next_states_mb.unsqueeze(1).repeat(1, num_repeat, 1).view(self.batch_size * num_repeat, self.state_dim)

            current_pi_values1, current_pi_values2 = self.compute_policy_values(temp_states_mb, temp_states_mb)
            next_pi_values1, next_pi_values2 = self.compute_policy_values(temp_next_states_mb, temp_states_mb)

            random_values1 = self.compute_random_values(temp_states_mb, random_actions, 0).reshape(self.batch_size, num_repeat, 1)
            random_values2 = self.compute_random_values(temp_states_mb, random_actions, 1).reshape(self.batch_size, num_repeat, 1)

            current_pi_values1 = current_pi_values1.reshape(self.batch_size, num_repeat, 1)
            current_pi_values2 = current_pi_values2.reshape(self.batch_size, num_repeat, 1)

            next_pi_values1 = next_pi_values1.reshape(self.batch_size, num_repeat, 1)
            next_pi_values2 = next_pi_values2.reshape(self.batch_size, num_repeat, 1)

            cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], dim=1)
            cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], dim=1)

            cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temperature, dim=1).mean() * self.cql_weight * self.temperature) - current_Q1.mean()) * self.cql_weight
            cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temperature, dim=1).mean() * self.cql_weight * self.temperature) - current_Q2.mean()) * self.cql_weight

            if self.lagrange:
                cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1e6).to(device)
                cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
                cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

                self.cql_alpha_optimizer.zero_grad()
                cql_alpha_loss = (-cql1_scaled_loss - cql2_scaled_loss) * 0.5
                cql_alpha_loss.backward(retain_graph=True)
                self.cql_alpha_optimizer.step()

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + cql1_scaled_loss + F.mse_loss(current_Q2, target_Q) + cql2_scaled_loss

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            c_losses.append(critic_loss.detach().cpu())

            # Update the frozen target models, ONLY THE CRITIC
            self.copy_target(self.critic_target, self.critic, self.tau, False)

        return np.mean(p_losses), np.mean(c_losses), None, None

    def compute_target(self, states_n, rews, dones):

        action, logprob, probs, dist = self.forward(states_n)

        # Compute the target Q value
        current_Q1, current_Q2 = self.critic_target(states_n, action)
        target_Q = torch.min(current_Q1, current_Q2) - self.alpha * logprob
        target_Q = target_Q.view(-1, 1)

        target = rews + (1.0 - dones.long()) * self.discount * target_Q
        target = target.view(-1, 1)

        return target

    def compute_policy_values(self, states_pi, states_q):
        with torch.no_grad():
            action, logprob, probs, dist = self.forward(states_pi)

        q1, q2 = self.critic(states_q, action)
        return q1 - logprob.detach(), q2 - logprob.detach()

    def compute_random_values(self, obs, actions, critic):
        random_values = self.critic(obs, actions)[critic]
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs

    def save_model(self, name, folder='saved', with_barracuda=False):
        torch.save(self.critic.state_dict(), '{}/{}_critic'.format(folder, name))
        torch.save(self.critic_optimizer.state_dict(), '{}/{}_critic_optimizer'.format(folder, name))

        torch.save(self.policy.state_dict(), '{}/{}_policy'.format(folder, name))
        torch.save(self.policy_optimizer.state_dict(), '{}/{}_policy_optimizer'.format(folder, name))
        if with_barracuda:
            # Input to the model
            x = torch.randn(1, self.state_dim).to(device)

            # Export the model
            torch.onnx.export(self.policy,  # model being run
                              x,  # model input (or a tuple for multiple inputs)
                              "{}/{}.onnx".format(folder, self.model_name),
                              # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=9,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['X'],  # the model's input names
                              output_names=['Y']  # the model's output names
                              )

    def load_model(self, name, folder='saved'):
        self.critic.load_state_dict(torch.load('{}/{}_critic'.format(folder, name)))
        self.critic_optimizer.load_state_dict(torch.load('{}/{}_critic_optimizer'.format(folder, name)))

        self.policy.load_state_dict(torch.load('{}/{}_policy'.format(folder, name)))
        self.policy_optimizer.load_state_dict(torch.load('{}/{}_policy_optimizer'.format(folder, name)))

        self.copy_target(self.policy_target, self.policy, self.tau, True)
        self.copy_target(self.critic_target, self.critic, self.tau, True)