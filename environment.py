import gym
import numpy as np

class Game:
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        # typecast since problems with multiplication if type(limit) = np.float32
        self.limit = float(self.env.action_space.high[0])

    def reset(self):
        start = self.env.reset()
        return start

    def take_one_step(self, action):
        new_frame, reward, is_done, _ = self.env.step([action])
        return new_frame, reward, is_done

    def sample_action(self):
        return self.env.action_space.sample()
