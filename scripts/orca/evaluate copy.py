# from build.omnetbind import SimulationRunner
# import gym
# from gym import spaces, logger
import numpy as np
import pandas as pd
import random
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
import os

from ray.tune.registry import register_env
import argparse
import copy
import math
from collections import deque

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
# from ray.tune.registry import register_env
# from ray.rllib.algorithms.ppo import PPOConfig
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
        self.timeStarted = None
    
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
        worker_ini_file = original_ini_file + f".worker{os.getpid()}" #_{self.env_config.worker_index}"

        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        
        ini_string = ini_string.replace("DELAY_PLACEHOLDER", f'{round(rtt,2)}ms')
        ini_string = ini_string.replace("LINKRATE_PLACEHOLDER", f'{round(linkrate)}Mbps')
        ini_string = ini_string.replace("Q_PLACEHOLDER", str(round(buffer)))
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))

        with open(worker_ini_file, 'w') as fout:
            fout.write(ini_string)

        # self.runner = OmnetGymApi() # ADDED FOR TESTING
        self.runner.initialise(worker_ini_file)
        self.timeStarted = time.time()
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

        terminated = info_['simDone'] # ((self.steps >= self.max_episode_steps)) # or info_['simDone']) # or (time.time() - self.timeStarted >= 80))

        if self.steps >= self.max_episode_steps and info_['simDone'] == False:
            truncated = True      
        else:
            truncated = False

        if terminated: #info_['simDone']:
             self.runner.shutdown()
             self.runner.cleanup()
             # self.runner = None

        return  obs, reward, terminated, truncated, {}


def OmnetGymApienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetppEnv", OmnetGymApienv_creator)

def run_episode(agent, env, explore):
    # rollout = pd.DataFrame(columns=[
    #                                 'bwNorm', 
    #                                 'trimPortion',
    #                                 'rttNorm', 
    #                                 'cwndNorm' , 
    #                                 'time',
    #                                 'cwnd',
    #                                 'paceTime',
    #                                 'bwMean',
    #                                 'bwStd', 
    #                                 'rttMean', 
    #                                 'rttStd', 
    #                                 'bwMax', 
    #                                 'rttminEst',
    #                                 'action',
    #                                 'reward',
    #                                 'lossrate1',
    #                                 'lossrate2'])
    done = False
    obs, info_ = env.reset()

    print(obs)
    # env.agentId = list(obs.keys())[0]
    # obs = obs[env.agentId]
    # env.currentRecord = obs
    # env.obs.extend(obs)
    # obs = np.asarray(list(env.obs),dtype=np.float32)
    # env.steps = 0
    
    # rollout = pd.concat([rollout, pd.DataFrame({'rttNorm': list(obs[-9:])[0],
    #                           'bwNorm': list(obs[-9:])[1], 
    #                           'trimPortion': list(obs[-9:])[2], 
    #                           'cwndNorm': round(list(obs[-9:])[3], 4),
    #                           'badSteps': list(obs[-9:])[4],
    #                           'bdpNorm': list(obs[-9:])[5],
    #                           'alpha':list(obs[-9:])[8],
    #                           'time': env.currentRecord[0],
    #                           'cwnd': env.currentRecord[1],
    #                           'paceTime': env.currentRecord[2],
    #                           'bwMean':  env.currentRecord[3],
    #                           'bwStd':  env.currentRecord[4],
    #                           'rttMean':  env.currentRecord[5],
    #                           'rttStd':  env.currentRecord[6],
    #                           'rttminEst':  env.currentRecord[7],
    #                           'bwMax': env.currentRecord[8]},index=[0])])

    step_counter = 1
    while not done:
        #data_list = [value for key, value in obs.items()]
        # obs = np.array([np.array(a) for a in obs])
        # Convert list of lists to a PyTorch tensor
        # tensor = torch.tensor(obs, dtype=torch.float32)

        # print(tensor)
        action = agent.compute_single_action(obs, explore=False)
        # action = 2**action

        # actions = {env.agentId: action}
        new_obs, reward, terminated, truncated, info_ = env.step(action)
        # rollout = pd.concat([rollout, pd.DataFrame({
        #                       'bwNorm': list(obs[-len(env.features_min):])[0], 
        #                       'trimPortion': list(obs[-len(env.features_min):])[1], 
        #                       'rttNorm': round(list(obs[-len(env.features_min):])[2]),
        #                       'cwndNorm': list(obs[-len(env.features_min):])[3],
        #                       'action': trans,
        #                       'reward': rewards,
        #                       'time': env.currentRecord[0],
        #                       'cwnd': env.currentRecord[1],
        #                       'paceTime': env.currentRecord[2],
        #                       'bwMean':  env.currentRecord[3],
        #                       'bwStd':  env.currentRecord[4],
        #                       'rttMean':  env.currentRecord[5],
        #                       'rttStd':  env.currentRecord[6],
        #                       'rttminEst':  env.currentRecord[7],
        #                       'bwMax': env.currentRecord[8],
        #                       'lossrate1': env.currentRecord[9],
        #                       'lossrate2': env.currentRecord[10]}, index=[0])])

        step_counter += 1
        obs = new_obs
        done = terminated

    # env.runner.shutdown()
    # return rollout

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p','--policy', required=True, type=int, help='policy number to test')
    # parser.add_argument('-r','--rate', required=True, type=int, help='bottleneck linkrate')
    # parser.add_argument('-d','--delay', required=True, type=int, help='rtt')
    # parser.add_argument('-b','--buffer', required=True, type=int, help='buffer size')
    # parser.add_argument('-e','--explore', required=False, action='store_true', help='explore')
    # args = parser.parse_args()

    # HOMEPATH = os.getenv('HOME')

    # os.environ["OMNETPP_NED_PATH"] = f"{HOMEPATH}/RLlibIntegration/model:{HOMEPATH}/inet/src/inet:{HOMEPATH}/inet/examples:{HOMEPATH}/rdp/src/:{HOMEPATH}/rdp/simulations:{HOMEPATH}/RLlibIntegration/rltcp/rltcp/src:{HOMEPATH}/ecmp/src"
    # os.environ["NEDPATH"] = f"{HOMEPATH}/RLlibIntegration/model:{HOMEPATH}/inet/src/inet:{HOMEPATH}/inet/examples:{HOMEPATH}/rdp/src/:{HOMEPATH}/rdp/simulations:{HOMEPATH}/RLlibIntegration/rltcp/rltcp/src:{HOMEPATH}/ecmp/src"


    # bw = args.rate
    # rtt = args.delay
    # buffer = args.buffer
    # explore = args.explore

    ray.init() #address='auto')

    env_config={"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaEval.ini",
          "stacking": 10,
          "linkrate_range": [200,200],
          "rtt_range": [100, 100],
          "buffer_range": [100, 100],}
    
    config = (
    SACConfig()
    .env_runners(num_rollout_workers=2, rollout_fragment_length=10) #, sample_timeout_s=80) #, rollout_fragment_length=100)
    .resources(num_gpus=1)
    .environment("OmnetppEnv", env_config=env_config) #, disable_env_checking=True) # "ns3-v0"
    .framework(
    "torch",
    torch_compile_worker=True,
    torch_compile_worker_dynamo_backend="ipex",
    torch_compile_worker_dynamo_mode="default",)
    .training(
        # n_step=8,
        # sample_async=True,
        twin_q = False,
        target_network_update_freq = 100,
        gamma=0.995,
        lr=0.001,
        train_batch_size=1000,
        tau=0.001,
        num_steps_sampled_before_learning_starts = 1500,
        replay_buffer_config={
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 1000000,
                "replay_batch_size": 64,
                "replay_sequence_length": 1,
                },
       optimization_config = {"actor_learning_rate": 0.0001, "critic_learning_rate":0.001, "entropy_learning_rate": 3e-4})
    )

    # with open(os.getenv('HOME') + "/RLlibIntegration/configs/ndpconfig_single_flow_eval_with_delay_template.ini", 'r') as fin:
    #     config_template = fin.read()

    # ini_file = config_template.replace('DELAY_PLACEHOLDER', f'{round(rtt/6, 2)}ms')
    # ini_file = ini_file.replace('RATE_PLACEHOLDER', f'{bw}Mbps')
    # ini_file = ini_file.replace('BUFFER_PLACEHOLDER', f'{buffer}')

    # with open(os.getenv('HOME') + f"/RLlibIntegration/configs/ndpconfig_single_flow_eval_with_delay_{os.getpid()}.ini", 'w') as fout:
    #     fout.write(ini_file)


    # check_no = args.policy
    # Create the Trainer from config.
    cls = get_trainable_cls("SAC")
    env = OmnetGymApienv_creator(env_config)
    agent = cls(env="OmnetppEnv", config=config)

    checkpoint_path = os.getenv('HOME') + "/ray_results/SAC_1/SAC_OmnetppEnv_d642b_00000_0_2024-08-11_03-24-29/checkpoint_000033"
    #checkpoint_file = "/rllib_checkpoint.json" 
    agent.restore(checkpoint_path) # + checkpoint_file)

    rollout = run_episode(agent, env, False)

    # rollout.to_csv(os.getenv('HOME') +f"/stacking_results/rollout_obsb_stack5_SAC05_{check_no}_15_workers_{buffer}pkts_{bw}mbps_{rtt}ms_{'stoc' if explore else 'det'}.csv")

    # os.remove(os.getenv('HOME') + f"/RLlibIntegration/configs/ndpconfig_single_flow_eval_with_delay_{os.getpid()}.ini")