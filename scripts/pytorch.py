import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data from the CSV file
file_path = 'scripts/data/edmund_data.csv'
df = pd.read_csv(file_path)

# Define inputs based on your selected columns
inputs = df[['LONGITUDE', 'LATITUDE', 'DISCOVERY_DATE']].values

# Create a new column 'FIRE_OCCURRENCE_COLUMN' and set it to 0 for all rows
df['FIRE_OCCURRENCE_COLUMN'] = 0

# Identify unique integer values in DISCOVERY_DATE
unique_dates = df['DISCOVERY_DATE'].unique()

# Generate a range of dates from 2000 to 2010 (integer intervals of 1)
all_dates = np.arange(2000, 2011)

# Identify missing dates and set 'FIRE_OCCURRENCE_COLUMN' to 0 for those dates
missing_dates = np.setdiff1d(all_dates, unique_dates)
df.loc[df['DISCOVERY_DATE'].isin(missing_dates), 'FIRE_OCCURRENCE_COLUMN'] = 0

# Define labels based on 'FIRE_OCCURRENCE_COLUMN'
labels = df['FIRE_OCCURRENCE_COLUMN'].values

# Define a PyTorch model for binary classification
class FireOccurrenceModel(nn.Module):
    def __init__(self, input_size):
        super(FireOccurrenceModel, self).__init__()
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
model = FireOccurrenceModel(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(torch.tensor(inputs, dtype=torch.float32))
    loss = criterion(outputs.view(-1), torch.tensor(labels, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Use the trained model to make predictions
with torch.no_grad():
    new_input = torch.tensor([float(input("Enter Longitude")), float(input("Enter Latitude")), float(input("Date"))], dtype=torch.float32)
    predicted_probability = model(new_input).item()
    print(f'Predicted Fire Probability: {predicted_probability:.4f}')

# Save the model if needed
torch.save(model.state_dict(), 'fire_occurrence_model.pth')
