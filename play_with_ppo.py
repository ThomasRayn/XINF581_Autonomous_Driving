from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Pre-trained model
model_name = "models/ppo_car_racing_1M"

# Creation of the environment
env = make_vec_env("CarRacing-v0", n_envs=1)

# Creation of the model
model = PPO.load(model_name)

# Launch the game
obs = env.reset()
while True:
    try:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()  
    except KeyboardInterrupt:
        break  