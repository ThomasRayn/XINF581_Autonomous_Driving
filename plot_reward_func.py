import os
import matplotlib.pyplot as plt
import numpy as np

# Plot the evolution of the reward over time for each simulation of the ppo training
def plot_ppo_reward_simulation(*folders):
    for folder in folders:
        reward_dict = {}

        for file in os.listdir(folder):
            var = file.split(".")[1] + "_" + file.split(".")[0]
            path = folder + file
            
            reward_dict[var] = []
            
            # Read the reward from log file
            with open(path, 'r') as monitor:
                for index, line in enumerate(monitor):
                    if index >= 2:
                        reward = float(line.split(",")[0])
                        reward = np.asarray(reward)
                        reward_dict[var].append(reward)
                
        reward_dict = dict(sorted(reward_dict.items()))

        # Creation of the subplot according to monitors
        fig, axis = plt.subplots(4, 2, figsize=(15, 10))
        axis[0, 0].plot(reward_dict["monitor_0"], label="monitor_0", color='red')
        axis[0, 1].plot(reward_dict["monitor_1"], label="monitor_1", color='green')
        axis[1, 0].plot(reward_dict["monitor_2"], label="monitor_2", color='blue')
        axis[1, 1].plot(reward_dict["monitor_3"], label="monitor_3", color='orange')
        axis[2, 0].plot(reward_dict["monitor_4"], label="monitor_4", color='purple')
        axis[2, 1].plot(reward_dict["monitor_5"], label="monitor_5", color='grey')
        axis[3, 0].plot(reward_dict["monitor_6"], label="monitor_6", color='black')
        axis[3, 1].plot(reward_dict["monitor_7"], label="monitor_7", color='cyan')

        for ax in axis.flat:
            ax.set(xlabel='Episodes', ylabel='Total Reward')
            ax.legend()
            
        fig.tight_layout()
        plt.show()
    
# Plot the mean and the std evolution of the reward over time
def plot_ppo_reward_mean_std(*folders):
    colors = ["blue", "red", "green"]
    plt.figure()
    for index_folder, folder in enumerate(folders):
        reward_dict = {}

        for file in os.listdir(folder):
            var = file.split(".")[1] + "_" + file.split(".")[0]
            path = folder + file
            
            reward_dict[var] = []
            
            # Read the reward from log file
            with open(path, 'r') as monitor:
                for index, line in enumerate(monitor):
                    if index >= 2:
                        reward = float(line.split(",")[0])
                        reward = np.asarray(reward)
                        reward_dict[var].append(reward)
                
        reward_dict = dict(sorted(reward_dict.items()))

        reward = []
        for monitor, rewards in reward_dict.items():
            reward.append(rewards)
            
        reward = np.asarray(reward)

        mean_reward = np.mean(reward, axis=0)
        std_reward = np.std(reward, axis=0)

        x = np.linspace(0, 100, len(mean_reward))

        # Creation of plot for each variation of gamma
        gamma = folder.split("_")[-1].split("/")[0]
        plt.plot(x, mean_reward, color=colors[index_folder], label=f"Gamma: {gamma} -> Mean")
        plt.fill_between(x, mean_reward - std_reward, mean_reward + std_reward, color=colors[index_folder], alpha=0.2, label=f"Gamma: {gamma} -> Mean +- Std")
        plt.legend()
    plt.show()
    
# Plot the evolution of the reward over time for each simulation of the dqn training having gamma a variable
def plot_dqn_reward(*files, autoencoder=False):
    # Plot the evolution of the reward over time if the autoencoder is not added to the architecture
    if not autoencoder:
        colors = ["blue", "red", "green"]
        reward_dict = {}
        
        for file in files:
            reward_dict[file] = []
            
            # Read the reward from log file
            with open(file, "r") as f:
                for line in f:
                    reward = float(line.split(":")[-1][1:])
                    reward_dict[file].append(reward)
                    
        reward_dict = dict(sorted(reward_dict.items()))
        
        # Creation of the subplots according to the value of gamma 
        _, axis = plt.subplots(3, figsize=(15, 10))
        index = 0
        for file, rewards in reward_dict.items():
            axis[index].plot(rewards, color=colors[index], label="Gamma: " + file.split("_")[-2])
            axis[index].axhline(y = 100, color="red", alpha=0.5)
            axis[index].axhline(y = 200, color="green", alpha=0.5)
            axis[index].axhline(y = 300, color="blue", alpha=0.5)
            axis[index].axhline(y = 400, color="orange", alpha=0.5)
            axis[index].axhline(y = 500, color="black", alpha=0.5)
            index += 1
            
        for ax in axis.flat:
            ax.set(xlabel='Episodes', ylabel='Total Reward')
            ax.legend(loc="upper right")
            
        plt.legend()
        plt.show()
        
    # Plot the evolution of the reward over time if the autoencoder is added to the architecture
    else:
        colors = ["blue", "red", "green"]
        reward_dict = {}
        
        for file in files:
            reward_dict[file] = []
            
            # Read the reward from log file
            with open(file, "r") as f:
                for line in f:
                    reward = float(line.split(":")[-1][1:])
                    reward_dict[file].append(reward)
                    
        reward_dict = dict(sorted(reward_dict.items()))
        
        # Creation of the plot
        plt.figure()
        for file, rewards in reward_dict.items():
            plt.plot(rewards, color="red", label="Gamma: 0.99")
            plt.axhline(y = 100, color="red", alpha=0.5)
            plt.axhline(y = 200, color="green", alpha=0.5)
            plt.axhline(y = 300, color="blue", alpha=0.5)
            plt.axhline(y = 400, color="orange", alpha=0.5)
            plt.axhline(y = 500, color="black", alpha=0.5)
            
        plt.legend()
        plt.show()
    
if __name__ == '__main__':

    plot_ppo_reward_simulation("logs/ppo_car_racing/")
    plot_ppo_reward_mean_std("logs/ppo_gamma_0.99/", "logs/ppo_gamma_0.95/", "logs/ppo_gamma_0.90/")
    plot_dqn_reward("logs/dqn/dqn_gamma_0.99_log.txt", "logs/dqn/dqn_gamma_0.90_log.txt", "logs/dqn/dqn_gamma_0.999_log.txt")
    plot_dqn_reward("logs/dqn_ae/dqn_ae_log.txt", autoencoder=True)