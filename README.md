# XINF581_Autonomous_Driving

This repository provides the basis of an autonomous driving project solved with reinforcement learning.

## Requirements

All the packages needed to run the files:
- gym package with box2d environment (for the simulation): *!pip3 install gym[box2d]==0.21.0*
- pyglet precise version (for the simulation): *!pip3 install pyglet==1.5.27*
- pygame (for the simulation): *!pip3 install pygame*
- StableBaselines 3 packages (for the PPO algorithm): *!pip3 install stable-baselines3*
- Pillow (for images): *!python3 -m pip install --upgrade Pillow*
- OpenCV (for images): *!pip3 install opencv-python*
- torch, torchvision: *for the installation, refer to https://pytorch.org/*
- numpy
- matplotlib

## Description of the repository

Files:
- car_racing_humain.py: provides a script to manually run and play the simulation.
- AutoEncoder.ipynb: provides a notebook for the autoencoder and the variational autoencoder **(DO NOT TRAIN, DATA NOT AVAILABLE)**
- AutoEncoder.py: provides the basic class for the autoencoder and the variational autoencoder
- DQN.ipynb: provides a notebook for the DQN model **(MODELS NOT PROVIDED)**
- DQN_AutoEncoder.ipynb: provides a notebook for the DQN+VAE model **(MODELS NOT PROVIDED)**
- train_ppo.py: provides a script to train a PPO algorithm **(CAREFULL, VERY COMPUTATION DEMENDING)**
- play_with_ppo.py: provides a script to play with a pre-trained PPO model
- plot_reward_func; provides scripts to plot the performances of the models according to the logs of their training

Folders:
- models : folder that provides the saved models for the Encoder, the Decoder, the VAE and the PPO
- logs: folder that provides log files of the training of the DQN, the DQN+VAE and the PPO models
- results: folder that provides figures showing the performances of the different models

For some further documentations:

- Gym CarRacing environment: https://www.gymlibrary.dev/environments/box2d/car_racing/
- StableBaselines3 documentation: https://stable-baselines3.readthedocs.io/en/master/

## How to use it

To play the game: *python3 car_racing_human.py*

To make PPO Agent play the game: *python3 play_with_ppo.py*