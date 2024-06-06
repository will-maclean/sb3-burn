import gym
from gym import spaces
import numpy as np

# This class provides a template of how a gym env class

class BaseEnv(gym.Env):
    def __init__(self, obs_space_size, n_actions):
        super(BaseEnv, self).__init__()
        # Example obs and action spaces only
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_space_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)
    
    def step(self, action):
        """
        The step function will take in an action and return the new state, reward, done, and info.
        """
        # Implement logic to apply action and return results
        pass
    
    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        # Implement logic to reset the environment
        pass
    
    def render(self):
        """
        Render the environment to the screen.
        """
        # Implement logic to render the environment
        pass
    
    def close(self):
        """
        Perform any necessary cleanup.
        """
        # Implement logic to clean up the environment
        pass

    def seed(self, seed=None):
        """
        Set the seed for the environment's random number generator(s).
        """
        # Implement logic to seed the environment
        pass
