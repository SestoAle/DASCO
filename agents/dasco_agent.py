import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import math
from torch.distributions import Categorical, Beta, Normal

EPS = 1e-6
LOG_SIG_MAX = 1.38
LOG_SIG_MIN = -9.21

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_dim, action_size=4, max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(Policy, self).__init__()

        self.action_size = action_size
        self.state_dim = state_dim
        self.max_action_value = max_action_value
        self.min_action_value = min_action_value

        # Layers specification
        self.embedding_l1 = nn.Linear(state_dim, 256)
        self.embedding_l2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, self.action_size)
        self.log_std = nn.Linear(256, self.action_size)

    def forward(self, inputs, deterministic=False):
        state = torch.reshape(inputs, (-1, self.state_dim))
        x = F.relu(self.embedding_l1(state))
        x = F.relu(self.embedding_l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        action_scale = (self.max_action_value - self.min_action_value) / 2.
        action_bias = (self.max_action_value + self.min_action_value) / 2.

        if deterministic:
            return torch.tanh(mean) * action_scale + action_bias
        else:
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
            return action, log_prob, None, None

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(Critic, self).__init__()

        # Layers specification
        self.embedding_q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.embedding_q1_l2 = nn.Linear(256, 256)
        self.q1_l = nn.Linear(256, 1)

        self.embedding_q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.embedding_q2_l2 = nn.Linear(256, 256)
        self.q2_l = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.embedding_q1_l1(x))
        q1 = F.relu(self.embedding_q1_l2(q1))
        q1 = self.q1_l(q1)

        q2 = F.relu(self.embedding_q2_l1(x))
        q2 = F.relu(self.embedding_q2_l2(q2))
        q2 = self.q2_l(q2)
        return q1, q2

    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.embedding_q1_l1(x))
        q1 = F.relu(self.embedding_q1_l2(q1))
        q1 = self.q1_l(q1)
        return q1

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(Discriminator, self).__init__()

        # Layers specification
        self.embedding_l1 = nn.Linear(state_dim + action_dim, 750)
        self.disc_l = nn.Linear(750, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        disc = F.relu(self.embedding_l1(x))
        logit = self.disc_l(disc)
        prob = F.sigmoid(logit)
        logit = logit.view((-1,))
        return prob, logit

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, max_action_value=1, min_action_value=-1, **kwargs):
        super(Generator, self).__init__()
        self.max_action_value = max_action_value
        self.state_dim = state_dim
        self.random_noise = state_dim
        self.latent_space = 750
        self.noise_dim = 3

        # Layers specification
        self.embedding_l1 = nn.Linear(state_dim + self.noise_dim, self.latent_space)
        self.gen_l = nn.Linear(750, action_dim)

    def forward(self, state):
        x = torch.reshape(state, (-1, self.state_dim))
        noise = torch.rand((x.shape[0], self.noise_dim)).to(device)

        x = torch.cat([x, noise], dim=1)
        gen = F.relu(self.embedding_l1(x))
        action = F.tanh(self.gen_l(gen)) * self.max_action_value
        return action


class DASCOAgent(nn.Module):
    def __init__(self, state_dim, discount=0.99, lr=0.001, tau=0.005, w=1., alpha=0.2, policy_freq=2,
                 batch_size=32, num_itr=20, name='dasco', action_size=4, max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(DASCOAgent, self).__init__()
        # Policy information
        self.state_dim = state_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_itr = num_itr
        # The update if the policy is asyncrhonized with respect to critic
        self.total_itr = 0
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.policy_freq = policy_freq
        self.alpha_tuning = True
        # Action specification
        self.action_size = action_size
        self.max_action = max_action_value
        self.min_action_value = min_action_value
        self.model_name = name
        self.w = w

        # Define the entities that we need, policy and actor
        self.policy = Policy(self.state_dim, self.action_size, self.max_action, self.min_action_value).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.action_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.discriminator = Discriminator(self.state_dim, self.action_size).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.generator = Generator(self.state_dim, self.action_size).to(device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)

        if self.alpha_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((action_size, )).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # Define the targets and init them
        self.critic_target = Critic(self.state_dim, self.action_size).to(device)
        self.copy_target(self.critic_target, self.critic, self.tau, True)

        self.total_itr = 0

    def set_dataset(self, dataset):
        self.states = torch.from_numpy(np.asarray(dataset['states'])).to(device)
        self.next_states = torch.from_numpy(np.asarray(dataset['next_states'])).to(device)
        self.dones = torch.from_numpy(np.asarray(dataset['dones'])).to(device)
        self.actions = torch.from_numpy(np.asarray(np.asarray(dataset['actions']))).to(device)
        self.rewards = torch.from_numpy(np.asarray(np.asarray(dataset['rewards']))).to(device).float()

        self.rewards = torch.reshape(self.rewards, (-1, 1))
        self.dones = torch.reshape(self.dones, (-1, 1))

    # Assign to model_a the weights of model_b. Use it for update the target networks weights.
    def copy_target(self, target_model, main_model, tau=1e-2, init=False):
        if init:
            for a, b in zip(target_model.parameters(), main_model.parameters()):
                a.data.copy_(b.data)
        else:
            for a, b in zip(target_model.parameters(), main_model.parameters()):
                a.data.copy_((1 - tau) * a.data + tau * b.data)

    def get_instance_noise(self, actions, std=0.3):
        noise = torch.normal(mean=0, std=torch.ones_like(actions) * std).to(device)
        n = torch.norm(noise, dim=1)
        f = torch.min(n, torch.full(n.shape, 0.3).to(device)) / n
        noise = noise * f.view(-1, 1)

        return noise

    def update(self):
        self.train()
        c_losses = []
        p_losses = []
        d_losses = []
        g_losses = []
        start = time.time()
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

            with torch.no_grad():
                target_Q = self.compute_target(next_states_mb, rewards_mb, dones_mb)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states_mb, actions_mb)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            c_losses.append(critic_loss.detach().cpu())

            actor_loss = None
            action, logprob, probs, dist = self.policy(states_mb)
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

            # Update the frozen target models
            self.copy_target(self.critic_target, self.critic, self.tau, False)

            p_losses.append(p_loss.detach().cpu())

            # Optimize generator
            real_label = torch.full((self.batch_size,), 1).to(device).float()
            action_fake = self.generator(states_mb)

            _, logit = self.discriminator(states_mb, action_fake)
            g_loss = F.binary_cross_entropy_with_logits(logit, real_label)

            self.generator_optimizer.zero_grad()
            g_loss.backward()
            self.generator_optimizer.step()
            g_losses.append(g_loss.detach().cpu())

            # Optimize discriminator
            # We update the discriminator more time than the generator
            for de in range(5):
                # Take a mini-batch of batch_size experience
                mini_batch_idxs = np.random.randint(0, len(self.states), size=self.batch_size)

                states_mb = self.states[mini_batch_idxs]
                actions_mb = self.actions[mini_batch_idxs]

                # Loss on real action
                _, d_real_logit = self.discriminator(states_mb, actions_mb + self.get_instance_noise(actions_mb.detach()))
                # _, d_real_logit = self.discriminator(states_mb, actions_mb)
                real_label = torch.full((self.batch_size,), 1).to(device).float()
                err_d_real = F.mse_loss(F.sigmoid(d_real_logit), real_label) / 2.

                def loss_fake_action(fake_action):
                    fake_label = torch.full((self.batch_size,), 0,).to(device).float()
                    _, d_fake_logit = self.discriminator(states_mb, fake_action.detach() + self.get_instance_noise(fake_action.detach()))
                    # _, d_fake_logit = self.discriminator(states_mb, fake_action.detach())
                    err_d_fake = F.mse_loss(F.sigmoid(d_fake_logit), fake_label) / 2.
                    return err_d_fake

                fake_action_aux = self.generator(states_mb)
                fake_action_pi, _, _, _ = self.policy(states_mb)
                err_d_fake = loss_fake_action(fake_action_aux) + loss_fake_action(fake_action_pi)
                self.discriminator_optimizer.zero_grad()
                (err_d_real + err_d_fake).backward()
                self.discriminator_optimizer.step()
                d_losses.append((err_d_fake + err_d_real).detach().cpu())

        end = time.time()
        print("Time: {}".format(end - start))

        return p_losses, c_losses, d_losses, g_losses

    def compute_target(self, states_n, rews, dones):

        action, logprob, probs, dist = self.policy(states_n)

        # Compute the target Q value
        current_Q1, current_Q2 = self.critic_target(states_n, action)
        target_Q = torch.min(current_Q1, current_Q2) - self.alpha * logprob
        target_Q = target_Q.view(-1, 1)

        target = rews + (1.0 - dones.long()) * self.discount * target_Q
        target = target.view(-1, 1)

        return target

    def save_model(self, folder='saved', with_barracuda=False):
        torch.save(self.critic.state_dict(), '{}/{}_critic'.format(folder, self.model_name))
        torch.save(self.critic_optimizer.state_dict(), '{}/{}_critic_optimizer'.format(folder, self.model_name))

        torch.save(self.policy.state_dict(), '{}/{}_policy'.format(folder, self.model_name))
        torch.save(self.policy_optimizer.state_dict(), '{}/{}_policy_optimizer'.format(folder, self.model_name))

        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator'.format(folder, self.model_name))
        torch.save(self.discriminator_optimizer.state_dict(), '{}/{}_discriminator_optimizer'.format(folder, self.model_name))

        torch.save(self.generator.state_dict(), '{}/{}_generator'.format(folder, self.model_name))
        torch.save(self.generator_optimizer.state_dict(), '{}/{}_generator_optimizer'.format(folder, self.model_name))

    def load_model(self, folder='saved'):
        self.critic.load_state_dict(torch.load('{}/{}_critic'.format(folder, self.model_name)))
        self.critic_optimizer.load_state_dict(torch.load('{}/{}_critic_optimizer'.format(folder, self.model_name)))

        self.policy.load_state_dict(torch.load('{}/{}_policy'.format(folder, self.model_name)))
        self.policy_optimizer.load_state_dict(torch.load('{}/{}_policy_optimizer'.format(folder, self.model_name)))

        self.copy_target(self.policy_target, self.policy, self.tau, True)
        self.copy_target(self.critic_target, self.critic, self.tau, True)
