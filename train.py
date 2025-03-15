import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from simple_custom_taxi_env import SimpleTaxiEnv


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99 
        self.alpha = 0.0001
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        self.update_target_network()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state, epsilon):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        with torch.no_grad():
            q_values = self.model(state.to(self.device))
        return torch.argmax(q_values).item()
    
    def update(self, state, action, target):
        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)        
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, target.detach())
        loss.backward()
        self.optimizer.step()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = torch.stack(states)  
        actions = torch.stack(actions) 
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        self.update(states, actions, target_q_values)
        


def train_agent(episodes=15000):
    env = SimpleTaxiEnv()
    state_size = 16
    action_size = 6  
    agent = DQNAgent(state_size, action_size)
    epsilon=1.0
    epsilon_end=0.01
    epsilon_decay_rate=0.999 #0.995
    rewards_per_episode = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0 
        
        while not done:
            action = agent.get_action(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        
            agent.replay()

            if step % 100 == 0:
                agent.update_target_network()

            step += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay_rate)
        # print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    
    torch.save(agent.model.state_dict(), "checkpoints/dqn_taxi_model_f50_ep15000_0.999.pth")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_agent()
