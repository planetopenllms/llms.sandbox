import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the network architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 4)  # Input layer to hidden layer
        self.fc2 = nn.Linear(4, 1)  # Hidden layer to output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # He initialization for ReLU
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Prepare data
X = torch.tensor([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=torch.float32)
                
y = torch.tensor([[0],
                  [1],
                  [1],
                  [0]], dtype=torch.float32)

# Initialize model, loss and optimizer
model = NeuralNet()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(60000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10000 == 0:
        print(f'Epoch [{epoch+1}/60000], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predicted = model(X)
    print("\nFinal predictions:")
    print(predicted)


print( "bye" )