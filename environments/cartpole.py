import torch
import torch.nn as nn
import gym
import os
from utils.utils import train, test

learning_rate = 0.005
num_episodes = 1000
num_tests = 100
log_frequency = 100
epsilon = 0.1 # randomness
epsilon_decay = 0.0005

env = gym.make('CartPole-v1')

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Linear(4, 24),
    nn.ReLU(),
    nn.Linear(24, 48),
    nn.ReLU(),
    nn.Linear(48, 2)
).to(device)

model_file = './models/cartpole.pth'
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, weights_only=True))
    print(f"Loaded model from {model_file}")

# Training model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_score = 0
for episode in range(num_episodes):
    epsilon -= epsilon_decay
    train_score += train(model, optimizer, criterion, env, epsilon)

    if (episode+1) % log_frequency == 0:
        test_score = 0
        for _ in range(num_tests):
            test_score += test(model, env)

        print(f'Episode: [{episode+1}/{num_episodes}], Average Reward: {train_score/log_frequency}, Average Testing Reward: {test_score/num_tests}')
        train_score = 0

torch.save(model.state_dict(), model_file)
print(f"Saved model to {model_file}")

# Testing model
test_score = 0
for _ in range(num_tests):
    test_score += test(model, env)
print(f'Average Testing Reward: {test_score/num_tests}')

env.close()