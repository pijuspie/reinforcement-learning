import gymnasium as gym 
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

env = gym.make('Acrobot-v1') 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = nn.Sequential(
    nn.Linear(state_dim, 40), nn.ReLU(),
    nn.Linear(40, 40), nn.ReLU(),
    nn.Linear(40, action_dim)
)

target_network = nn.Sequential(
    nn.Linear(state_dim, 40), nn.ReLU(),
    nn.Linear(40, 40), nn.ReLU(),
    nn.Linear(40, action_dim)
)
target_network.load_state_dict(q_network.state_dict())  

optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
memory = deque(maxlen=10000)

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.99
batch_size = 64
target_update_interval = 10

for episode in range(500):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_dim)
        else:
            q_values = q_network(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values).item()
        
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.int64)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(np.array(dones), dtype=torch.bool)
            
            q_values = q_network(states)
            next_q_values = target_network(next_states)
            target = rewards + (gamma * next_q_values.max(dim=1)[0] * ~dones)
            
            q_value = q_values.gather(1, actions.unsqueeze(1))
            loss = nn.MSELoss()(q_value.squeeze(), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if total_reward <= -1000:
            break
    
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    if episode % target_update_interval == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    if (episode + 1) % 1 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
env = gym.make('Acrobot-v1', render_mode='human')

epsilon = 0.0

for episode in range(3):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = torch.argmax(q_network(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0))).item()
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        if total_reward <= -1000:
            break
    
    print(f"Total reward achieved by the agent: {total_reward}")

env.close()
