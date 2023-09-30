import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load the data from the Excel file
# Replace 'your_file.xlsx' with the actual file path
file_path = 'your_file.xlsx'

# Load the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Step 2: Preprocess the data
# Assume that you have columns 'longitude', 'latitude', and 'date' in your DataFrame
# Convert the date column to a numeric format (e.g., days since a reference date)
df['date'] = pd.to_datetime(df['date'])
reference_date = pd.to_datetime('2000-01-01')  # You can choose a different reference date
df['date'] = (df['date'] - reference_date).dt.days

# Select the input features (longitude, latitude, date)
selected_columns = ['LONGITUDE', 'LATITUDE', 'DATE']
df = df[selected_columns]

# Define a function to calculate fire probability (you can replace this with your own logic)
def calculate_fire_probability(row):
    # Replace this with your logic for calculating fire probability based on longitude, latitude, and date
    # For simplicity, we assume a fixed probability here
    return 0.1

# Apply the calculate_fire_probability function to create the target variable
df['fire_probability'] = df.apply(calculate_fire_probability, axis=1)

# Convert the DataFrame to a numpy array
data_array = df.to_numpy(dtype=np.float32)

# Separate the input features (longitude, latitude, date) and target variable (fire probability)
inputs = data_array[:, :-1]
targets = data_array[:, -1]

# Step 3: Create a PyTorch model
class FireProbabilityModel(nn.Module):
    def __init__(self, input_size):
        super(FireProbabilityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create the model
input_size = inputs.shape[1]
model = FireProbabilityModel(input_size)

# Step 4: Define a loss function and an optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(torch.tensor(inputs))  # Forward pass
    loss = criterion(outputs, torch.tensor(targets).view(-1, 1))  # Calculate the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Use the trained model to make predictions
with torch.no_grad():
    # Replace these values with the input values for prediction (longitude, latitude, date)
    new_input = torch.tensor([input("Enter Longitude"), input("Enter Latitude"), input("Date")], dtype=torch.float32)
    predicted_probability = model(new_input).item()
    print(f'Predicted Fire Probability: {predicted_probability:.4f}')

# Save the model if needed
torch.save(model.state_dict(), 'fire_probability_model.pth')