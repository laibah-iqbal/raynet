from omnetbind import OmnetGymApi
import gymnasium
from gymnasium import spaces, logger
import numpy as np
import pandas as pd
import random
import ray
import math
from ray import tune
import copy
import nnmodels
import os
from collections import deque
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from nnmodels import KerasBatchNormModel
import gymnasium as gym
from gymnasium import spaces, logger
import numpy as np
import math
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray
import pandas as pd
import os
from collections import deque
import time
from ray.tune.logger import pretty_print
import cProfile

from ray import train, tune, air
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.envs.registration import register
from collections import defaultdict

from ray.tune.registry import register_env

def uniform(low=0, high=1):
    return np.random.uniform(low, high)

def lognuniform(low=0, high=1, size=None, base=np.e):
    return np.power(base, np.random.uniform(low, high, size))

class OmnetGymApiEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self._agent_ids = set(['1', '2', '3', '4'])
        self.env_config = env_config
        self.stacking = env_config['stacking']
        self.action_space = spaces.Box(low=np.array([-2.0], dtype=np.float32), high=np.array([2.0], dtype=np.float32), dtype=np.float32)
        self.obs_min = np.tile(np.array([-10000000000,  
                                 -10000000000,   
                                 -10000000000,   
                                 -10000000000,
                                 -10000000000,
                                 -10000000000,
                                 -10000000000], dtype=np.float32), self.stacking)

        self.obs_max = np.tile(np.array([10000000000, 
                                 10000000000, 
                                 10000000000, 
                                 10000000000,
                                 10000000000,
                                 10000000000,
                                 10000000000],dtype=np.float32), self.stacking)
        self.currentRecord = None
        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)
        self.runner = OmnetGymApi()
        self.obs = deque(np.zeros(len(self.obs_min)),maxlen=len(self.obs_min))

    def reset(self, seed, options):
        self.obs = deque(np.zeros(len(self.obs_min)),maxlen=len(self.obs_min))
        # Draw network parameters from space
        linkrate_range = self.env_config["linkrate_range"]
        rtt_range = self.env_config["rtt_range"]
        buffer_range = self.env_config["buffer_range"]

        linkrate = uniform(low=linkrate_range[0], high=linkrate_range[1])
        rtt = uniform(low=rtt_range[0], high=rtt_range[1])/6.0
        buffer = uniform(low=buffer_range[0], high=buffer_range[1])

        original_ini_file = self.env_config["iniPath"]
        worker_ini_file = original_ini_file + f".worker{self.env_config.worker_index}"

        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        
        ini_string = ini_string.replace("DELAY_PLACEOLDER", f'{round(rtt,2)}ms')
        ini_string = ini_string.replace("LINKRATE_PLACEHOLDER", f'{round(linkrate)}Mbps')
        ini_string = ini_string.replace("Q_PLACEHOLDER", str(round(buffer)))

        with open(worker_ini_file, 'w') as fout:
            fout.write(ini_string)

        self.runner.initialise(worker_ini_file)
        obs = self.runner.reset()
        # obs = np.asarray(obs['5'], dtype=np.float32)
        for key,value in obs.items():
            self.currentRecord = value #[-13:]
            self.obs.extend(value) #[:-13])
            obs = {key: np.asarray(list(self.obs),dtype=np.float32)}
        return obs, {}

    def step(self, actions):
        # self.steps += 1
        
        for agent, action  in actions.items():
            actions[agent] = 2**action
        
        # action = 2**action

        # actions = {self.agentId: action}

        # actions = action

        obs, rewards, dones, info_ = self.runner.step(actions)
        if dones['__all__']:
             self.runner.shutdown()
             self.runner.cleanup()

        if info_['simDone']:
             for agent, done in dones.items():
                 dones[agent] = True

        for key, value in obs.items():
            self.currentRecord = value #[-13:]
            self.obs.extend(value) #[:-13])
            obs = {key: np.asarray(list(self.obs),dtype=np.float32)}
        return  obs, rewards, dones, {}


def OmnetGymApienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetppEnv", OmnetGymApienv_creator)

if __name__ == "__main__":
    config = {
        "env": "OmnetppEnv",
        "env_config": {"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini",
                       "linkrate_range": [96,96],
                       "rtt_range": [40, 40],
                       "buffer_range": [440, 440],
                       "stacking": 5},
        "evaluation_config": {
                                "env_config": {"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini"},
                                "explore": False,
                                "stacking": 5

        },
     "num_workers": 4,
     "horizon": 400,
     "no_done_at_end":True,
     "soft_horizon":False,
     "gamma": 0.5,
     "multiagent": {
         "count_steps_by": "agent_steps"
     }
     }

    ray.init(num_cpus=5, num_gpus=1, object_store_memory=1000000000)
    
    # # Create the Trainer from config.
    # cls = get_trainable_cls("SAC")
    # env = simulationrunnerenv_creator(config['env_config'])
    # agent = cls(env="OmnetppEnv", config=config)

    # checkpoint_path = f"/its/home/lg317/ray_results/explicitstate5/SAC_OmnetppEnv_4700e_00000_0_2022-07-05_16-51-05/checkpoint_009000"
    # checkpoint_file = f"/checkpoint-9000" 
    # agent.restore(checkpoint_path + checkpoint_file)

    tune.run(
        "SAC",
        name="SAC_2",
        stop={"timesteps_total": 500000},
        config=config,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        resume=False,
        # resources_per_trial={"cpu": 5, "gpu": 1},
        max_concurrent_trials=1

    )



