import torch
if torch.cuda.is_available():
    print("CUDA AVAILABLE")
import sys
print(sys.path)
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

from ray import train, tune
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.envs.registration import register


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
        self.max_steps = 200
    
    def reset(self, *, seed=None, options=None):
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

        self.runner.initialise(worker_ini_file)
        print("before")
        observations = self.runner.reset()
        print(observations)
        print("after")
        if len(observations.keys()) > 1:
            print(f"************ ERROR: expected only 1 flow, but {len(observations.keys())} were found.") 
        self.agentId = list(observations.keys())[0]
        observations = observations[self.agentId]
        self.currentRecord = observations
        self.obs.extend(observations)
        observations = np.asarray(list(self.obs),dtype=np.float32)
        self.steps = 0
        return observations, {}

    def step(self, action):
        self.steps += 1
        print(self.steps)
        action = 2**action

        actions = {self.agentId: action}

        if math.isnan(action):
            print("====================================== action passed is nan =========================================")
        
        print("STEPS: " + str(self.steps))
        observations, rewards, dones, info_= self.runner.step(actions)
        
        print(rewards)
        
        if dones[self.agentId]:
             self.runner.shutdown()
             self.runner.cleanup()

        if math.isnan(rewards[self.agentId]):
            print("====================================== reward returned is nan =========================================")
        reward = round(rewards[self.agentId],4)
        if any(np.isnan(np.asarray(observations[self.agentId], dtype=np.float32))):
            print("====================================== obs returned is nan =========================================")
        

        observations = observations[self.agentId]
        self.currentRecord = observations
        self.obs.extend(observations)
        observations = np.asarray(list(self.obs),dtype=np.float32)

        if info_['simDone']:
             dones[self.agentId] = True

        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        return  observations, reward, dones[self.agentId],truncated, {}


def OmnetGymApienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetppEnv", OmnetGymApienv_creator)

env_config={"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini",
          "stacking": 10,
          "linkrate_range": [6,192],
          "rtt_range": [4, 400],
          "buffer_range": [100, 1000],}



# context = ray.init()
# print(context.dashboard_url) 

# algo = (
#     PPOConfig()
#     .rollouts(num_rollout_workers=4, sample_timeout_s=600, rollout_fragment_length=100)
#     .resources(num_gpus=0)
#     .environment("OmnetppEnv", env_config=env_config) # "ns3-v0"
#     # .environment("Cartpole-v1")
#     .build())

config = {"env": "OmnetppEnv",
            "env_config": env_config,
            "horizon": 400,
            "framework": "torch",
            "torch_compile_worker": True,
            "torch_compile_worker_dynamo_backend": "ipex",
            "torch_compile_worker_dynamo_mode=":"default",
            "optimization_config":{"actor_learning_rate": 0.0001, "critic_learning_rate":0.001, "entropy_learning_rate": 3e-4},
            "n_step":8,
            "gamma":0.995,
            "lr":0.001,
            "train_batch_size":8000,
            "tau":0.001,
            "num_steps_sampled_before_learning_starts":2,
            "replay_buffer_config":{"_enable_replay_buffer_api": True,"type": "MultiAgentReplayBuffer","capacity": 50000,"replay_batch_size": 32,"replay_sequence_length": 1,},
            "num_rollout_workers": 10,
            "rollout_fragment_length":"auto",
            "num_gpus":1,
            }


# algo = (
#     SACConfig()
#     .env_runners(num_rollout_workers=10, rollout_fragment_length='auto', batch_mode='truncate_episodes') #, rollout_fragment_length=100)
#     .resources(num_gpus=1)
#     .environment("OmnetppEnv", env_config=env_config) #, disable_env_checking=True) # "ns3-v0"
#     .framework(
#     "torch",
#     torch_compile_worker=True,
#     torch_compile_worker_dynamo_backend="ipex",
#     torch_compile_worker_dynamo_mode="default",)
#     .training(
#         n_step=8,
#         gamma=0.995,
#         lr=0.001,
#         train_batch_size=8000,
#         tau=0.001,
#         num_steps_sampled_before_learning_starts = 2,
#         replay_buffer_config={
#                 "_enable_replay_buffer_api": True,
#                 "type": "MultiAgentReplayBuffer",
#                 "capacity": 50000,
#                 "replay_batch_size": 32,
#                 "replay_sequence_length": 1,
#                 },
#         optimization_config = {"actor_learning_rate": 0.0001, "critic_learning_rate":0.001, "entropy_learning_rate": 3e-4}
#     ))
#     # .reporting(min_)
#     # .training(n_step=7, store_buffer_in_checkpoints=True, train_batch_size=100) # , num_sgd_iter=10)
#     # .environment("Cartpole-v1")
#     # .build())

# # algo = (
# #     SACConfig()
# #     .environment(env="OmnetppEnv", env_config=env_config, render_env=False)
# #     .framework('torch')
# #     .training(
# #         gamma=0.99,
# #         lr=0.001,
# #         train_batch_size=256,
# #         target_network_update_freq=500,
# #         tau=0.005
# #     )
# #     # .resources(num_gpus=0, num_cpus=1)
# #     .rollouts(num_rollout_workers=1,sample_timeout_s=30,  # Increase the sample timeout
# #         rollout_fragment_length=50)
# # )

ray.init(ignore_reinit_error=True)

tuner = tune.Tuner(
        "SAC",
        #param_space=algo,
        run_config=train.RunConfig(
            stop={"timesteps_total": 500000},
        ),
        param_space=config,
        # checkpoint_freq=100,
        # checkpoint_at_end=True
    )
start = time.time()
# while True:
#     result = algo.train() 
#     print("Number of steps trained: " + str(result['num_env_steps_trained']))
#     # print("TRAIN ITER: " + str(i)) 

#     if result['num_env_steps_trained'] % 100 == 0:
#         checkpoint = algo.save()
#         print(f"Checkpoint saved at {checkpoint}")

#     if result['num_env_steps_trained'] >= 800000:
#         break
# # while True:
# #     result = algo.train()
# #     print(result['num_env_steps_sampled'])
# #     if result['num_env_steps_sampled'] >= 1000000:
# #             break

results = tuner.fit()
print(time.time() - start)
print(pretty_print(results))  
ray.shutdown()
# # # analysis = ray.tune.run(
# # #     "TD3", name="orca",stop={"training_iteration": 200000}, config=config, checkpoint_freq=50)


# # ray.init()

# # Define SAC configuration
# # config = (
# #     SACConfig()
# #     .environment(env="OmnetppEnv", env_config=env_config, render_env=False)
# #     .framework('torch')
# #     .training(
# #         gamma=0.99,
# #         lr=0.001,
# #         train_batch_size=256,
# #         target_network_update_freq=500,
# #         tau=0.005
# #     )
# #     # .resources(num_gpus=0, num_cpus=1)
# #     .rollouts(num_rollout_workers=1,sample_timeout_s=30,  # Increase the sample timeout
# #         rollout_fragment_length=50)
# # )

# # Run the training process
# # tune.run(
# #     "SAC",
# #     config=config.to_dict(),
# #     stop={"training_iteration": 100},
# #     storage_path="~/ray_results",
# #     checkpoint_at_end=True
# # )
# now = time.time()
# print(start)
# print(now)
