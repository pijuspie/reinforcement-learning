import os
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.01
num_episodes = 10000

# Environment
env = gym.make('CartPole-v1')

# Policy Network
model = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.ReLU(),
    nn.Linear(64, env.action_space.n)
).to(device)

# Load the saved model (if it exists)
model_file = './models/cartpole_model.pth'
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, weights_only=True))
    print(f"Loaded model from {model_file}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.tensor(state).float().to(device)
        action_probs = torch.softmax(model(state), dim=0)
        action = torch.multinomial(action_probs, 1).item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        optimizer.zero_grad()
        log_prob = torch.log(action_probs[action])
        loss = -log_prob * reward
        loss.backward()
        optimizer.step()

        state = next_state

    if ((episode+1)%10 == 0):
        print(f"Episode: [{episode+1}/{num_episodes}], Total Reward: {total_reward}")

# Save the trained model
torch.save(model.state_dict(), model_file)
print(f"Saved model to {model_file}")

# Test the model
state = env.reset()
done = False

while not done:
    env.render()
    time.sleep(0.05)
    state = torch.tensor(state).float().to(device)
    action_probs = torch.softmax(model(state), dim=0)
    action = torch.multinomial(action_probs, 1).item()

    next_state, reward, done, _ = env.step(action)
    state = next_state

env.close()