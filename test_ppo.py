import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CarRacing_V2 import CarRacing_V2
import torch.nn as nn

####################################################
#                  BE CAREFULL                     #
#      THIS SCRIPT WILL LAUNCH A PPO TRAINING      #
#         THAT WILL TAKE A LONG TIME TO RUN        #
####################################################

env = make_vec_env(CarRacing_V2, n_envs=1)

# Training of the algorithm
model = PPO(
    env=env,
    policy="MlpPolicy",
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
model.learn(total_timesteps=1_000_000) ## /!\/!\ CONSIDER total_timesteps=10_000 MAX IF YOU WANT TO TEST /!\/!\ ##
model.save("ppo_car_racing")
print("Learning is done!")

del model # remove to demonstrate saving and loading


# Launding and Testing the trained model
model = PPO.load("models/ppo_car_racing")

obs = env.reset()
while True:
    try:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()  
    except KeyboardInterrupt:
        break  