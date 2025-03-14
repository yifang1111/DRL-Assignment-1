# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
from train import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent(16, 6)
agent.model.load_state_dict(torch.load("dqn_taxi_model.pth", map_location=torch.device('cpu')))
agent.model.eval()
agent.model.to(device)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = agent.model(obs)
    return torch.argmax(q_values).item()

    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

