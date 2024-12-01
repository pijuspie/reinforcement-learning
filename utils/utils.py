import torch
import random

def train(model, optimizer, criterion, env, epsilon):
    device = next(model.parameters()).device

    state = torch.tensor(env.reset()).float().to(device)
    done = False
    score = 0

    while not done:
        y = model(state)

        if (random.random() <= epsilon):
            action = env.action_space.sample() 
        else:
            action = y.argmax(dim=-1).item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state).float().to(device)

        y_target = y.clone().detach()
        with torch.no_grad():
            y_target[action] = reward if done else reward + torch.max(model(next_state))

        optimizer.zero_grad()
        loss = criterion(y, y_target)
        loss.backward()
        optimizer.step()   

        state = next_state
        score += 1

    return score

def test(model, env, render=False):
    device = next(model.parameters()).device

    state = env.reset()
    done = False
    score = 0

    while not done:
        if render:
            env.render()
        
        state = torch.tensor(state).float().to(device)
        action = model(state).argmax(dim=-1).item()
        state, _, done, _ = env.step(action)
        score += 1

    return score