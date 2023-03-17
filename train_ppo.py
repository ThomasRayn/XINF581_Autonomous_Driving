from time import time
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

# Create log dir
log_dir = "logs/ppo_car_racing/"
os.makedirs(log_dir, exist_ok=True)

####################################################
#                  BE CAREFULL                     #
#      THIS SCRIPT WILL LAUNCH A PPO TRAINING      #
#        THAT WILL TAKE A LONG TIME TO RUN         #
####################################################

model_name = "models/ppo_car_racing_1M"

train = False

# Training of the algorithm
if train:
    
    env = make_vec_env("CarRacing-v0", n_envs=8, monitor_dir=log_dir)
    
    # Credit to the RL-Zoo3 git that gives the hyperparemeters of the algorithm: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    
    model = PPO(
        env=env,
        policy="CnnPolicy",
        batch_size=128,
        n_steps=512,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.0,
        sde_sample_freq=4,
        max_grad_norm=0.5,
        vf_coef=0.5,
        learning_rate=1e-4,
        use_sde=True,
        clip_range=0.2,
        policy_kwargs= {
            "log_std_init":-2,
            "ortho_init":False,
            "activation_fn": nn.GELU,
            "net_arch": {
                "pi": [256],
                "vf": [256]
            }
        }
    )

    t0 = time()
    print("Start learning...")
    model.learn(total_timesteps=1_000_000) ## /!\/!\ CONSIDER total_timesteps=10_000 MAX IF YOU WANT TO TEST /!\/!\ ##
    model.save(model_name)
    t1 = time()
    print("Learning is done!")
    print("Time for learning: ")
    print("-", t1 - t0, "sec")
    print("-", (t1 - t0) / 60, "min")
    print("-", ((t1 - t0) / 60) / 60, "h")
    