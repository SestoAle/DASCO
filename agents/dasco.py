import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.action_l = nn.Linear(256, self.action_size)

    def forward(self, inputs):
        state = torch.reshape(inputs, (-1, self.state_dim))
        x = F.relu(self.embedding_l1(state))
        x = F.relu(self.embedding_l2(x))
        x = F.tanh(self.action_l(x)) * self.max_action_value
        return x

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
        self.embedding_l1 = nn.Linear(state_dim + action_dim, 256)
        self.embedding_l2 = nn.Linear(256, 256)
        self.disc_l = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        disc = F.relu(self.embedding_l1(x))
        disc = F.relu(self.embedding_l2(disc))
        logit = self.disc_l(disc)
        prob = F.sigmoid(logit)
        logit = logit.view((-1,))
        return prob, logit

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, max_action_value=1, min_action_value=-1, **kwargs):
        super(Generator, self).__init__()
        self.max_action_value = max_action_value
        self.state_dim = state_dim

        # Layers specification
        self.embedding_l1 = nn.Linear(state_dim, 256)
        self.embedding_l2 = nn.Linear(256, 256)
        self.gen_l = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.reshape(state, (-1, self.state_dim))
        gen = F.relu(self.embedding_l1(x))
        gen = F.relu(self.embedding_l2(gen))
        gen = F.tanh(self.gen_l(gen)) * self.max_action_value
        return gen


class DASCOAgent(nn.Module):
    def __init__(self, state_dim, discount=0.99, lr=0.001, tau=0.005,
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
        # Action specification
        self.action_size = action_size
        self.max_action = max_action_value
        self.min_action_value = min_action_value
        self.model_name = name

        # Define the entities that we need, policy and actor
        self.policy = Policy(self.state_dim, self.action_size, self.max_action, self.min_action_value).to(device)
        self.policy_loss = torch.nn.MSELoss()
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.action_size).to(device)
        self.critic_loss = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.discriminator = Discriminator(self.state_dim, self.action_size).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.generator = Generator(self.state_dim, self.action_size).to(device)
        self.generator_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

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

    def train(self):
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
            dones_mb = self.dones[mini_batch_idxs]

            # Critic update
            with torch.no_grad():
                # TODO: Noise?
                # Select action according to policy and add clipped noise
                next_action = self.policy(next_states_mb)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_states_mb, next_action)

                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards_mb + (1 - dones_mb.long()) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states_mb, actions_mb)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            c_losses.append(critic_loss.detach().cpu())

            # Update critic target weights
            self.copy_target(self.critic_target, self.critic, self.tau, False)

            # Policy update
            pi = self.policy(states_mb)
            q = self.critic.Q1(states_mb, pi)
            # Compute log probability under discriminator
            disc, logit = self.discriminator(states_mb, pi)
            log_d = F.logsigmoid(logit)

            real_action_disc, _ = self.discriminator(states_mb, actions_mb)
            probs = torch.min(disc, real_action_disc)
            probs = probs/real_action_disc

            p_loss = -(probs * q + log_d).mean()

            self.policy_optimizer.zero_grad()
            p_loss.backward()
            self.policy_optimizer.step()
            p_losses.append(p_loss.detach().cpu())

            # Optimize generator
            real_label = torch.full((self.batch_size,), 1).to(device).long()
            action_fake = self.generator(states_mb)

            _, logit = self.discriminator(states_mb, action_fake)
            g_loss = F.binary_cross_entropy_with_logits(logit, real_label)
            self.generator_optimizer.zero_grad()
            g_loss.backward()
            self.generator_optimizer.step()
            g_losses.append(g_loss.detach().cpu())

            # Optimize discriminator
            # Loss on real action
            # TODO: Noise
            _, d_real_logit = self.discriminator(states_mb, actions_mb)
            real_label = torch.full((self.batch_size,), 1).to(device).long()
            err_d_real = F.mse_loss(F.sigmoid(d_real_logit), real_label) / 2.

            # Loss on fake action
            with torch.no_grad:
                fake_action_gen = self.generator(states_mb)
                fake_action_pi = self.policy(states_mb)
            fake_label = torch.full((self.batch_size,), 0).to(device).long()
            # For generator
            # TODO: noise
            _, d_fake_gen_logit = self.discriminator(states_mb, fake_action_gen)
            err_d_fake_gen = F.mse_loss(F.sigmoid(d_fake_gen_logit), fake_label) / 2.
            # For policy
            # TODO: noise
            _, d_fake_pi_logit = self.discriminator(states_mb, fake_action_pi)
            err_d_fake_pi = F.mse_loss(F.sigmoid(d_fake_pi_logit), fake_label) / 2.

            err_d_fake = err_d_fake_pi + err_d_fake_gen
            disc_loss = err_d_fake + err_d_real
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()
            d_losses.append(disc_loss.detach().cpu())

        end = time.time()
        print("Time: {}".format(end - start))

        return p_losses, c_losses, d_losses, g_losses

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
