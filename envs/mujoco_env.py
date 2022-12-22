import gym

class MujocoEnvWrapper():
    def __init__(self, game_name, _max_episode_timesteps=1000):
        self.env = gym.make(game_name)
        self._max_episode_timesteps = _max_episode_timesteps
        self.ep_reward = 0
        self.step = 0

    def reset(self):
        # Sample engine strength
        state = self.env.reset()
        # state = dict(global_in=state)
        self.ep_reward = 0
        self.step = 0

        return state

    def normalize(self, value, ran):
        return ((value - ran[0]) / (ran[1] - ran[0])) * 2 - 1

    def entropy(self, probs):
        return 0

    def execute(self, actions, visualize=False):
        state, reward, done, _ = self.env.step(actions)
        if visualize:
            self.env.render()
        self.ep_reward += reward
        self.step += 1
        # state = dict(global_in=state)
        return state, reward, done

    def set_config(self, config):
        return
