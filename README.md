# Reinforcement learning with Python

This project implements reinforcement learning agents using Deep Q-Networks (DQN) to solve two classic control tasks: CartPole-v1 and Acrobot-v1 environments from Gymnasium.

## Usage

Create a new environment:
```
python -m venv rl-env
```

Activate/deactivate the environtment:
```
rl-env\Scripts\activate
deactivate
```

Install dependencies and update requirement list:
```
pip install -r requirements.txt
pip freeze > requirements.txt
```

Run scripts:
```
py cart-pole/main.py
py acrobot/main.py
```