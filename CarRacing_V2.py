from gym.envs.box2d.car_racing import CarRacing, FPS, PLAYFIELD
from gym import spaces
import pygame
from typing import Optional, Union
import numpy as np
import torch
from AutoEncoder import Encoder
from PIL import Image
import matplotlib.pyplot as plt

# The following class heritates for the car agent of gym
# It is used to provide some controls over the agent
# Usefull to display some information or conduct some tests

class CarRacing_V2(CarRacing):
    def __init__(
        self,
        verbose: int = 1,
    ):
        
        super().__init__(verbose)
        self.index = 0
        self.model = Encoder(encoded_space_dim=50)
        self.model.load_state_dict(torch.load("models/encoder"))

        
    def step(self, action: Union[np.ndarray, int]):
        self.index += 1
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self.render("state_pixels")
        rgb_array = self.render("rgb_array")
        
        ####################################################
        #                The next section                  #
        #         is about using the autoencoder           #
        #                for the training                  #
        #      in stead of the full observation space      #
        #             (NOT READY TO USE YET)               #
        ####################################################
        
        # image = self.state.swapaxes(0, -1)
        # image = Image.fromarray(image.astype(np.uint8).transpose(1,2,0))
        # image = self.model.transform_data(image)
        # image = torch.unsqueeze(image, 0)
        # # plt.imshow(image[0], cmap="Greys")
        # # plt.show()
        # self.state = self.model(image)
        # self.state = torch.squeeze(self.state)
        # self.state = self.state.detach().numpy()
        # print(self.state.shape)
        
            
        surface = pygame.surfarray.make_surface(rgb_array)
        
        ####################################################
        #                The next section                  #
        #             is used to collect data              #
        #               from a game session                #
        ####################################################
         
        # if self.index <= 2250 and self.index > 250:
        #     pygame.image.save(surface, 'images/validation/class/validation_car_step_' + str(self.index) + '.png')
        # elif self.index > 2250:
        #     print("2250 images have been saved!")

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}