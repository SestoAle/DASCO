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

class TD3BCAgent(nn.Module):
    def __init__(self, state_dim, discount=0.99, lr=0.001,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5, tau=0.005, batch_size=32,
                 num_itr=20, name='td3bc', action_size=4, max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(TD3BCAgent, self).__init__()
        # Policy information
        self.state_dim = state_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_itr = num_itr
        # The update if the policy is asyncrhonized with respect to critic
        self.total_itr = 0
        self.policy_freq = policy_freq
        # TD3 parameters
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.tau = tau
        self.discount = discount
        # Action specification
        self.action_size = action_size
        self.max_action = max_action_value
        self.min_action_value = min_action_value
        self.model_name = name

        # Define the entities that we need, policy and actor
        self.policy = Policy(self.state_dim, self.action_size, self.max_action, self.min_action_value).to(device)
        self.policy_loss = torch.nn.MSELoss()
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-5)
        self.critic = Critic(self.state_dim, self.action_size).to(device)
        self.critic_loss = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-6)

        # Define the targets and init them
        self.policy_target = Policy(self.state_dim, self.action_size, self.max_action, self.min_action_value).to(device)
        self.critic_target = Critic(self.state_dim, self.action_size).to(device)
        self.copy_target(self.policy_target, self.policy, self.tau, True)
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

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                        torch.randn_like(actions_mb) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                        self.policy_target(next_states_mb) + noise
                ).clamp(-self.max_action, self.max_action)

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

            actor_loss = None
            # Delayed policy updates
            if self.total_itr % self.policy_freq == 0:

                pi = self.policy(states_mb)
                q = self.critic.Q1(states_mb, pi)
                lmbda = self.alpha / q.abs().mean().detach()

                p_loss = -lmbda * q.mean() + self.policy_loss(actions_mb, pi)

                self.policy_optimizer.zero_grad()
                p_loss.backward()
                self.policy_optimizer.step()

                p_losses.append(p_loss.detach().cpu())

                # Update the frozen target models
                self.copy_target(self.critic_target, self.critic, self.tau, False)
                self.copy_target(self.policy_target, self.policy, self.tau, False)
        end = time.time()
        print("Time: {}".format(end - start))

        return p_losses, c_losses, None, None

    def save_model(self, folder='saved', with_barracuda=False):
        torch.save(self.critic.state_dict(), '{}/{}_critic'.format(folder, self.model_name))
        torch.save(self.critic_optimizer.state_dict(), '{}/{}_critic_optimizer'.format(folder, self.model_name))

        torch.save(self.policy.state_dict(), '{}/{}_policy'.format(folder, self.model_name))
        torch.save(self.policy_optimizer.state_dict(), '{}/{}_policy_optimizer'.format(folder, self.model_name))
        if with_barracuda:
            # Input to the model
            x = torch.randn(1, self.state_dim).to(device)

            # Export the model
            torch.onnx.export(self.policy,  # model being run
                              x,  # model input (or a tuple for multiple inputs)
                              "{}/{}.onnx".format(folder, self.model_name),  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=9,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['X'],  # the model's input names
                              output_names=['Y']  # the model's output names
                              )


    def load_model(self, folder='saved'):
        self.critic.load_state_dict(torch.load('{}/{}_critic'.format(folder, self.model_name)))
        self.critic_optimizer.load_state_dict(torch.load('{}/{}_critic_optimizer'.format(folder, self.model_name)))

        self.policy.load_state_dict(torch.load('{}/{}_policy'.format(folder, self.model_name)))
        self.policy_optimizer.load_state_dict(torch.load('{}/{}_policy_optimizer'.format(folder, self.model_name)))

        self.copy_target(self.policy_target, self.policy, self.tau, True)
        self.copy_target(self.critic_target, self.critic, self.tau, True)
