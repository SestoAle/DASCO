import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from agents.td3_plus_bc import TD3BCAgent
from agents.dasco_agent import DASCOAgent
from agents.cql_agent import CQLAgent
import d4rl
from architectures.lunar_arch import *
import argparse
from envs.mujoco_env import MujocoEnvWrapper
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_demonstrations_d4rl(env, normalize=False):
    dataset = d4rl.qlearning_dataset(env.env)

    states_mean = None
    states_std = None
    if normalize:
        states_mean = np.mean(dataset['observations'], axis=0)
        states_std = np.std(dataset['observations'], axis=0) + 1e-3

    dems = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])

    for s in dataset['observations']:
        if normalize:
            s = (s - states_mean) / (states_std)

        dems['states'].append(s)
    dems['actions'] = dataset['actions']
    for s in dataset['next_observations']:
        if normalize:
            s = (s - states_mean) / (states_std)

        dems['next_states'].append(s)
    dems['rewards'] = dataset['rewards'].reshape(-1, 1)
    dems['dones'] = dataset['terminals'].reshape(-1, 1)

    print("There are {} transitions in this demonstrations".format(len(dems['states'])))
    # input("Press any key to continue... ")
    return dems, states_mean, states_std

def eval(model, max_test_ep_len, env, state_mean=None, state_std=None):
    print("Testing...")
    global visualize
    global args
    episode_rewards = []
    for episode in range(10):

        # init episode
        running_state = env.reset()['global_in']
        total_reward = 0
        # Run inference on CPU
        for t in range(max_test_ep_len):

            if state_mean is not None:
                running_state = (np.asarray(running_state).reshape(1, -1) - state_mean) / (state_std)

            running_state = torch.from_numpy(running_state).float().to(device)
            if args.algorithm == "cql":
                action, _, _, _ = model(running_state)
            else:
                action, _, _, _ = model.policy(running_state, deterministic=False)
            action = action.detach().cpu().numpy()[0]
            running_state, running_reward, done = env.execute(action, visualize)
            running_state = running_state['global_in']
            total_reward += running_reward

            if done:
                break

        episode_rewards.append(total_reward)

    print("Average reward of last {} episodes: {}".format(10, np.mean(episode_rewards)))
    try:
        d4rl_score = env.env.get_normalized_score(np.mean(episode_rewards)) * 100
        print("Average D$RL reward of last {} episodes: {}".format(10, d4rl_score))

    except Exception as e:
        d4rl_score = None

    return np.mean(episode_rewards), d4rl_score

parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='cql')
parser.add_argument('-gn', '--game-name', help="The name of the game", default="walker2d-medium-expert-v2")
parser.add_argument('-rn', '--run-name', help="The name of the run to save statistics", default="run")
parser.add_argument('-al', '--algorithm', help="The algorithm to use", default="cql")
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=1000, type=int)
parser.add_argument('-e', '--epochs', help="Number of epochs", default=2000, type=int)

args = parser.parse_args()

if __name__ == "__main__":

    # Create the environment
    if 'hopper' in args.game_name:
        action_size = 3
        state_dim = 11
        max_timesteps = 1000
        visualize = False
        env = MujocoEnvWrapper(args.game_name)
        trajectories, state_mean, state_std = load_demonstrations_d4rl(env)
    elif 'halfcheetah' in args.game_name:
        action_size = 6
        state_dim = 16+1
        max_timesteps = 1000
        visualize = False
        env = MujocoEnvWrapper(args.game_name)
        trajectories, state_mean, state_std = load_demonstrations_d4rl(env)
    elif 'walker' in args.game_name:
        action_size = 6
        state_dim = 16 + 1
        max_timesteps = 1000
        visualize = False
        env = MujocoEnvWrapper(args.game_name)
        trajectories, state_mean, state_std = load_demonstrations_d4rl(env)
    else:
        raise Exception("Environment not available!")

    epochs = args.epochs
    if args.algorithm == 'td3':
        model = TD3BCAgent(state_dim=state_dim, lr=3e-4, action_size=action_size, num_itr=5000, batch_size=256,
                       name=args.model_name)
    elif args.algorithm == 'dasco':
        model = DASCOAgent(state_dim=state_dim, lr=3e-4, action_size=action_size, num_itr=5000, batch_size=256,
                           name=args.model_name)
    elif args.algorithm == 'cql':
        model = CQLAgent(critic_embedding=CriticEmbedding, policy_embedding=PolicyEmbedding, state_dim=state_dim, lr=1e-4,
                 action_size=action_size, num_itr=5000, batch_size=256, frequency_mode='timesteps', name=args.model_name,
                 memory=1e6, alpha=0.2)

    # Set trajectories data
    model.set_dataset(trajectories)
    # First evaluation
    eval(model, max_timesteps, env, state_mean, state_std)

    p_losses = []
    c_losses = []
    rewards = []
    d4rl_rewards = []
    print("Start training")
    for e in range(1, epochs+1):
        current_p_losses, current_q_losses, current_d_losses, current_g_losses = model.update()
        print("Policy loss at {} epoch: {}".format(e, np.mean(current_p_losses)))
        print("Critic loss at {} epoch: {}".format(e, np.mean(current_q_losses)))
        if current_d_losses is not None:
            print("Discriminator loss at {} epoch: {}".format(e, np.mean(current_d_losses)))
            print("Generator loss at {} epoch: {}".format(e, np.mean(current_g_losses)))

        reward, d4rl_reward = eval(model, max_timesteps, env, state_mean, state_std)
        p_losses.append(np.mean(current_p_losses))
        c_losses.append(np.mean(current_q_losses))
        rewards.append(reward)
        d4rl_rewards.append(d4rl_reward)
        save_statistics(args.run_name, p_losses, c_losses, rewards, d4rl_rewards, "results")
