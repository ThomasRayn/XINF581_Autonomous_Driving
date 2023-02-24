# XINF581_Autonomous_Driving

This repository provides the basis of an autonomous driving project solved with reinforcement learning.

Requirements:

- gym package with box2d environment (for the simulation): !pip3 install gym
- StableBaselines 3 packages (for the PPO algorithm): !pip3 install stable-baselines3
- Pillow (for images): !pip3 install Pillow
- torch, torchvision
- numpy
- matplotlib

Description of the files:

- car_racing_humain.py: provides a script to manually run and play the simulation.
- car_racing_autoencoder.ipynb: provides a notebook for the autoencoder **(DO NOT TRAIN, DATA NOT AVAILABLE)**
- CarRacing_V2.py: provides a derivated class of the gym class CarRacing in order to have some controls over the agent
- AutoEncoder.py: provides the basic class for the autoencoder
- test_ppo.py: provides a script to train and run a PPO algorithm **(CAREFULL, VERY COMPUTATION DEMENDING)**
- models : folder that provides the saved models for the encoder and the decoder **(PPO were to heavy to be published)**

For some further documentations:

- Gym CarRacing environment: https://www.gymlibrary.dev/environments/box2d/car_racing/
- StableBaselines3 documentation: https://stable-baselines3.readthedocs.io/en/master/