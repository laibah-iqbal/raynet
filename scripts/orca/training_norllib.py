import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from omnetbind import OmnetGymApi
from gymnasium import spaces
import numpy as np
import os
from collections import deque
import math
import logging

logging.basicConfig(level=logging.DEBUG)

class OmnetGymApiEnv(gym.Env):
    def __init__(self, env_config):
        self.env_config = env_config
        self.stacking = env_config['stacking']
        self.action_space = spaces.Box(low=np.array([-2.0], dtype=np.float32), high=np.array([2.0], dtype=np.float32), dtype=np.float32)
        self.obs_min = np.tile(np.array([-1e9] * 7, dtype=np.float32), self.stacking)
        self.obs_max = np.tile(np.array([1e10] * 7, dtype=np.float32), self.stacking)
        self.currentRecord = None
        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)
        self.runner = OmnetGymApi()
        self.obs = deque(np.zeros(len(self.obs_min)), maxlen=len(self.obs_min))
        self.agentId = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, *, seed=None, options=None):
        logging.debug("Resetting environment")
        self.obs = deque(np.zeros(len(self.obs_min)), maxlen=len(self.obs_min))
        linkrate_range = self.env_config["linkrate_range"]
        rtt_range = self.env_config["rtt_range"]
        buffer_range = self.env_config["buffer_range"]

        linkrate = np.random.uniform(low=linkrate_range[0], high=linkrate_range[1])
        rtt = np.random.uniform(low=rtt_range[0], high=rtt_range[1]) / 2.0
        buffer = np.random.uniform(low=buffer_range[0], high=buffer_range[1])

        original_ini_file = self.env_config["iniPath"]
        worker_ini_file = original_ini_file + f".worker{os.getpid()}_{self.env_config['worker_index']}"

        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()

        ini_string = ini_string.replace("DELAY_PLACEOLDER", f'{round(rtt, 2)}ms')
        ini_string = ini_string.replace("LINKRATE_PLACEHOLDER", f'{round(linkrate)}Mbps')
        ini_string = ini_string.replace("Q_PLACEHOLDER", str(round(buffer)))
        ini_string = ini_string.replace("HOME", os.getenv('HOME'))

        with open(worker_ini_file, 'w') as fout:
            fout.write(ini_string)

        self.runner.initialise(worker_ini_file)
        observations = self.runner.reset()

        if len(observations.keys()) > 1:
            logging.error(f"Expected only 1 flow, but {len(observations.keys())} were found.")

        self.agentId = list(observations.keys())[0]
        observations = observations[self.agentId]
        self.currentRecord = observations
        self.obs.extend(observations)
        observations = np.asarray(list(self.obs), dtype=np.float32)
        self.steps = 0
        logging.debug(f"Environment reset completed: {observations}")
        return observations, {}

    def step(self, action):
        logging.debug(f"Step {self.steps+1}: Action {action}")
        self.steps += 1
        action = 2**action
        actions = {self.agentId: action}

        if math.isnan(action):
            logging.error("Action passed is nan")

        observations, rewards, dones, info_ = self.runner.step(actions)
        logging.debug(f"Observations: {observations}, Rewards: {rewards}, Dones: {dones}")

        if dones[self.agentId]:
            self.runner.shutdown()
            self.runner.cleanup()

        reward = round(rewards[self.agentId], 4)

        if any(np.isnan(np.asarray(observations[self.agentId], dtype=np.float32))):
            logging.error("Observations returned are nan")

        observations = observations[self.agentId]
        self.currentRecord = observations
        self.obs.extend(observations)
        observations = np.asarray(list(self.obs), dtype=np.float32)

        if info_['simDone']:
            dones[self.agentId] = True

        truncated = self.steps >= self.max_steps
        return observations, reward, dones[self.agentId], truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.runner.shutdown()
        self.runner.cleanup()

# Create and register the custom environment
env_config = {
    "iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini",
    "stacking": 10,
    "linkrate_range": [64, 64],
    "rtt_range": [16, 16],
    "buffer_range": [250, 250],
    "worker_index": 0
}

def OmnetGymApienv_creator(env_config):
    return OmnetGymApiEnv(env_config)

# Register the environment with Gym
gym.register(
    id='OmnetppEnv-v0',
    entry_point=OmnetGymApienv_creator,
    kwargs={'env_config': env_config}
)

# Create and check the custom environment
env = gym.make('OmnetppEnv-v0')
check_env(env)

# Create the model
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("sac_custom_env")

# Load the model
model = SAC.load("sac_custom_env")

# Test the model
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()
