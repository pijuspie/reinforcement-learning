import torch
import torch.nn as nn
import gym
from utils.utils import train, test

learning_rate = 0.01
num_episodes = 1500
num_tests = 100
log_frequency = 100
epsilon = 1 # randomness
epsilon_decay = 1/num_episodes

env = gym.make('CartPole-v1')

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Linear(4, 24),
    nn.ReLU(),
    nn.Linear(24, 48),
    nn.ReLU(),
    nn.Linear(48, 2)
).to(device)

# Training model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_score = 0
for episode in range(num_episodes):
    epsilon -= epsilon_decay 
    train_score += train(model, optimizer, criterion, env, epsilon)

    if (episode+1) % log_frequency == 0:
        print(f'Episode: [{episode+1}/{num_episodes}], Average Reward: {train_score/log_frequency}')
        train_score = 0

# Testing model
test_score = 0
for _ in range(num_tests):
    test_score += test(model, env)
print(f'Average Testing Reward: {test_score/num_tests}')

env.close()