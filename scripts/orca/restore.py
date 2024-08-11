import torch
if torch.cuda.is_available():
    print("CUDA AVAILABLE")
# import sys
# print(sys.path)
from omnetbind import OmnetGymApi
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


# ModelCatalog.register_custom_model("bn_model",KerasBatchNormModel)

def uniform(low=0, high=1):
    return np.random.uniform(low, high)

class OmnetGymApiEnv(gym.Env):
    def __init__(self, env_config):
        self.env_config = env_config
        self.stacking = env_config['stacking']
        self.action_space = spaces.Box(low=np.array([-2.0], dtype=np.float32), high=np.array([2.0], dtype=np.float32), dtype=np.float32)
        self.obs_min = np.tile(np.array([-1000000000,  
                                 -1000000000,   
                                 -1000000000,   
                                 -1000000000,
                                 -1000000000,
                                 -1000000000,
                                 -1000000000], dtype=np.float32), self.stacking)

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
        self.agentId = None
        self.steps = 0
        self.max_episode_steps = 50
        self.currentReward = None
    
    def reset(self, *, seed=None, options=None):
        print("ENVIRONMENT RESET")
        self.obs = deque(np.zeros(len(self.obs_min)),maxlen=len(self.obs_min))
        # Draw network parameters from space
        linkrate_range = self.env_config["linkrate_range"]
        rtt_range = self.env_config["rtt_range"]
        buffer_range = self.env_config["buffer_range"]

        linkrate = uniform(low=linkrate_range[0], high=linkrate_range[1])
        rtt = uniform(low=rtt_range[0], high=rtt_range[1])/2.0
        buffer = uniform(low=buffer_range[0], high=buffer_range[1])

        original_ini_file = self.env_config["iniPath"]
        worker_ini_file = original_ini_file + f".worker{os.getpid()}_{self.env_config.worker_index}"

        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        
        ini_string = ini_string.replace("DELAY_PLACEOLDER", f'{round(rtt,2)}ms')
        ini_string = ini_string.replace("LINKRATE_PLACEHOLDER", f'{round(linkrate)}Mbps')
        ini_string = ini_string.replace("Q_PLACEHOLDER", str(round(buffer)))
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))

        with open(worker_ini_file, 'w') as fout:
            fout.write(ini_string)

        # self.runner = OmnetGymApi() # ADDED FOR TESTING
        self.runner.initialise(worker_ini_file)
        print("before")
        obs = self.runner.reset()

        for key, value in obs.items():
            obs[key] = np.asarray(value, dtype=np.float32)

        print(obs)
        print("after")
        if len(obs.keys()) > 1:
            print(f"************ ERROR: expected only 1 flow, but {len(obs.keys())} were found.") 
        self.agentId = list(obs.keys())[0]
        obs = obs[self.agentId]
        self.currentRecord = obs
        self.obs.extend(obs)
        obs = np.asarray(list(self.obs),dtype=np.float32)
        self.steps = 0
        print("RESET CALLED")
        return obs, {}

    def step(self, action):
        self.steps += 1
        print(self.steps)
        action = 2**action

        actions = {self.agentId: action}

        if math.isnan(action):
            print("====================================== action passed is nan =========================================")
        
        print("STEPS: " + str(self.steps))
        obs, rewards, dones, info_= self.runner.step(actions)

        for key, value in obs.items():
            obs[key] = np.asarray(value, dtype=np.float32)
        
        print("observations: ",obs)
        print("dones:", dones)
        print("info:",info_)
        print("rewards:", rewards)

        if self.agentId in rewards.keys():
            if self.agentId in rewards.keys() and math.isnan(rewards[self.agentId]):
                print("====================================== reward returned is nan =========================================")
            reward = round(rewards[self.agentId],4)
            self.currentReward = reward
            print("REWARD: " + str(reward))
        else:
            reward = self.currentReward
        if self.agentId in obs.keys(): 
            if any(np.isnan(np.asarray(obs[self.agentId], dtype=np.float32))):
                print("====================================== obs returned is nan =========================================")
        
        # completion = defaultdict(int)
            try: 
                obs = obs[self.agentId]
            except:
                print(obs.keys)
            
            self.currentRecord = obs
            self.obs.extend(obs)
            obs = np.asarray(list(self.obs),dtype=np.float32)
        else:
            obs = self.currentRecord
            self.obs.extend(obs)
            obs = np.asarray(list(self.obs),dtype=np.float32)

        terminated = ((self.steps >= self.max_episode_steps) or info_['simDone'])

        if self.steps >= self.max_episode_steps and info_['simDone'] == False:
            truncated = True      
        else:
            truncated = False

        if terminated: #info_['simDone']:
             self.runner.shutdown()
             self.runner.cleanup()
             # self.runner = None

        return  obs, reward, terminated, truncated, {} #reward, dones[self.agentId],truncated, {}


def OmnetGymApienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetppEnv", OmnetGymApienv_creator)

env_config={"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini",
          "stacking": 10,
          "linkrate_range": [6,192],
          "rtt_range": [4, 400],
          "buffer_range": [100, 1000],}

config = (
    SACConfig()
    .env_runners(num_rollout_workers=4, rollout_fragment_length=10, sample_timeout_s=80) #, rollout_fragment_length=100)
    .resources(num_gpus=1)
    .environment("OmnetppEnv", env_config=env_config) #, disable_env_checking=True) # "ns3-v0"
    .framework(
    "torch",
    torch_compile_worker=True,
    torch_compile_worker_dynamo_backend="ipex",
    torch_compile_worker_dynamo_mode="default",)
    .training(
        # n_step=8,
        gamma=0.995,
        lr=0.001,
        train_batch_size=64,
        tau=0.001,
        num_steps_sampled_before_learning_starts = 200,
        replay_buffer_config={
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_batch_size": 10000,
                "replay_sequence_length": 1,
                },
        optimization_config = {"actor_learning_rate": 0.0001, "critic_learning_rate":0.001, "entropy_learning_rate": 3e-4}
    ))

if __name__ == "__main__":

    env_config = {"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini", #ndpconfig_single_flow_train_with_delay.ini",
                       "linkrate_range": [64,128],
                       "rtt_range": [16, 64],
                       "buffer_range": [80, 800],
                       "stacking": 10}

    evaluation_config =  {
                                "env_config": {"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini", 
                                               "linkrate_range": [64,128],
                                               "rtt_range": [4, 10],
                                            "buffer_range": [80, 800],
                                            "stacking": 10}, #/raynet/configs/ndpconfig_single_flow_train_with_delay.ini", "stacking": 10},
                                "explore": False,
                                                            
    }


    ray.init() #local_mode=True)
    
    # # Create the Trainer from config.
    # cls = get_trainable_cls("SAC")
    # env = OmnetGymApienv_creator(config['env_config'])
    # agent = cls(env="OmnetppEnv", config=config)

    # checkpoint_path = f"/its/home/lg317/ray_results/explicitstate5/SAC_OmnetppEnv_4700e_00000_0_2022-07-05_16-51-05/checkpoint_009000"
    # checkpoint_file = f"/checkpoint-9000" 
    # agent.restore(checkpoint_path + checkpoint_file)

    checkpoint_path = f"/home/laibah/ray_results/SAC_1/SAC_OmnetppEnv_62cad_00000_0_2024-08-10_01-20-45/checkpoint_000013/rllib_checkpoint.json"
    trainable_path = f"/home/laibah/ray_results/SAC_1/SAC_OmnetppEnv_62cad_00000_0_2024-08-10_01-20-45/checkpoint_000013/policies/default_policy/rllib_checkpoint.json"
    # tuner = tune.Tuner(
    #     "SAC", 
    #     run_config=air.RunConfig(stop={"timesteps_total": 100000}, 
    #                              name=f"SAC_1",
    #                              checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1,
    #                                                                     checkpoint_at_end=True
    #                                                                     ),
                                
    #                     ),
    #     param_space=config
        
    # )

    tuner = tune.Tuner.run(SACTrainer, restore=checkpoint_path,param_space=config)

    results = tuner.fit()
    print(results)

    ray.shutdown()
    
