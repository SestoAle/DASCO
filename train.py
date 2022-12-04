from agents.sac_agent import SACAgent
from envs.mujoco_env import MujocoEnvWrapper
from runner.parallel_runner import Runner
from runner.runner import Runner as SRunner
import os
import argparse
import torch
import os
import numpy as np
from architectures.lunar_arch import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the policy", default='sac')
parser.add_argument('-gn', '--game-name', help="The name of the game", default="Hopper")
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=1000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=1)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=1000)
parser.add_argument('-pl', '--parallel', help="How many environments to simulate in parallel. Default is 1", type=int, default=1)

collect_demo = False
args = parser.parse_args()

eps = 1e-12

epis = 0

if not collect_demo:
    def callback(agent, env, runner):
        return
else:
    def callback(agent, env, runner):
        global epis
        epis += 1
        if epis % 12 == 0:
            input('...')
        return

if __name__ == "__main__":

    # RL arguments
    model_name = args.model_name
    game_name = args.game_name
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = None

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'timesteps'
    # Frequency of training (in episode)
    frequency = 1
    # Memory of the agent (in episode)
    memory = 1e6
    # Random initial actions
    random_actions = 25e3
    # Action type of the policy
    action_type = "continuous"

    # Open the environment with all the desired flags
    # If parallel, create more environments
    # Create the environment
    if 'Hopper' in args.game_name:
        action_size = 3
        state_dim = 11
        max_timesteps = 1000
        visualize = False
        env = MujocoEnvWrapper(args.game_name)
    elif 'HalfCheetah' in args.game_name:
        action_size = 6
        state_dim = 16+1
        max_timesteps = 1000
        visualize = False
        env = MujocoEnvWrapper(args.game_name)
    elif 'Walker2D' in args.game_name:
        action_size = 6
        state_dim = 16 + 1
        max_timesteps = 1000
        visualize = False
        env = MujocoEnvWrapper(args.game_name)
    else:
        raise Exception("Environment not available!")

    # Create agent
    # The policy embedding and the critic embedding for the PPO agent are defined in the architecture file
    # You can change those architectures, the PPOAgent will manage the action layers and the value layers
    agent = SACAgent(critic_embedding=CriticEmbedding, policy_embedding=PolicyEmbedding, state_dim=state_dim, lr=3e-5,
                     action_size=action_size, num_itr=1, batch_size=256, frequency_mode=frequency_mode, name=model_name,
                     memory=memory, alpha=0.05)

    # Create runner
    # This class manages the evaluation of the policy and the collection of experience in a parallel setting
    # (not vectorized)
    runner = SRunner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency, random_actions=random_actions,
                    logging=logging, total_episode=total_episode, curriculum=curriculum, demonstrations_name="dems",
                    frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback)

    try:
        runner.run()
    finally:
        env.close()


"%%%%%%%%%%%%%%%%%%%%%%%%%%"
# from utils  import *
# env = envs[0]
# state, done = env.reset(), False
# episode_reward = 0
# episode_timesteps = 0
# random_actions = 0
# episode_num = 0
#
# for t in range(int(1e100)):
#
#     episode_timesteps += 1
#
#     # Select action randomly or according to policy
#     if t < random_actions:
#         action = env.env.action_space.sample()
#         logprobs = 0
#     else:
#         action, logprobs, probs, dist = agent(torch.from_numpy(np.asarray(state)).to(device).float())
#         action = action.detach().cpu().numpy()[0]
#         logprobs = logprobs.detach().cpu().numpy()
#         probs = probs.detach().cpu().numpy()
#
#     # Perform action
#     next_state, reward, done = env.step(action)
#     if episode_timesteps == max_episode_timestep:
#         done = True
#     done_bool = float(done) if episode_timesteps < max_episode_timestep else 0
#
#     # Store data in replay buffer
#     agent.add_to_buffer(state, next_state, action, reward, logprobs, done_bool)
#     # replay_buffer.add(state, action, next_state, reward, done_bool)
#
#     state = next_state
#     episode_reward += reward
#
#     # Train agent after collecting sufficient data
#     if t >= random_actions:
#         agent.update()
#
#     if done:
#         # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
#         print(
#             f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
#         # Reset environment
#         state, done = env.reset(), False
#         episode_reward = 0
#         episode_timesteps = 0
#         episode_num += 1
