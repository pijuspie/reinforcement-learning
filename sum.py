import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network to sum two numbers using Sequential
model = nn.Sequential(
    nn.Linear(2, 1)
)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate some random input and target data
X = torch.randn(100, 2)  # 100 pairs of random numbers
y = torch.sum(X, dim=1, keepdim=True)  # Sum of each pair

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[3.14, 4.68]])
    predicted_sum = model(test_input)
    print(f"Predicted sum: {predicted_sum.item():.2f}")