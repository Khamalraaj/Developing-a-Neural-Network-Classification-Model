# Developing a Neural Network Classification Model

### Name: Khamalraaj S

### Register Number:212224230122

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load and Preprocess the Dataset:

Import the required libraries, load the dataset, handle missing values, encode categorical variables, and normalize the features.

### STEP 2: 

Split the Dataset:

Divide the dataset into training and testing sets.

### STEP 3: 

Build the Neural Network:

Create a Sequential model with an input layer, hidden layers using ReLU activation, and an output layer with Softmax activation for the four customer segments.

### STEP 4: 

Train the Model:

Compile the model using the Adam optimizer and categorical cross-entropy loss function, then train it using the training data.

### STEP 5: 

Evaluate and Predict:

Evaluate the model on the test dataset, measure its accuracy, and use it to predict the customer segment (A, B, C, or D) for new customers.







## PROGRAM & OUTPUT: 
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
```
```
data = pd.read_csv('/content/customers (1).csv')
df
```
```
<img width="1593" height="324" alt="image" src="https://github.com/user-attachments/assets/83591d32-fbcf-405d-b348-cb56222e93b6" />

```
```
df=df.drop(columns=["ID"])
df
```
```
df.columns
```
```
<img width="1092" height="129" alt="image" src="https://github.com/user-attachments/assets/c0b92088-dcfd-415c-8294-1f2ec4e8d8ed" />
```
```
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
```
```
# Encode categorical variables:
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])
```
```
# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3
```
```
# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values
```
```
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)xt=torch.FloatTensor(xt)
```
```
# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```
```
# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
```
```
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4) # Output layer with 4 classes (A, B, C, D)





    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.relu(x)
      x = self.fc3(x)
      return x
```
```
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```
```
# Initialize model
input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
# Train the model
epochs = 50
train_model(model, train_loader, criterion, optimizer, epochs)
```

```
<img width="533" height="157" alt="image" src="https://github.com/user-attachments/assets/48e53f50-655c-4eec-b79f-450eb9478697" />
```

```
## Evaluation:
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

```
```
# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name:Khamalraaj S")
print("Register No: 212224230122")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
```
```
<img width="943" height="547" alt="image" src="https://github.com/user-attachments/assets/4f17343f-e638-4fa9-96b3-833466a99cfd" />

```
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```
```
<img width="1059" height="708" alt="image" src="https://github.com/user-attachments/assets/1554edac-fad2-425f-938f-4fdcf840f6cf" />

```
```
# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name:Khamalraaj S")
print("Register No:212224230122")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
```
```
<img width="519" height="114" alt="image" src="https://github.com/user-attachments/assets/3a6a3cf1-a7ce-47b8-a1e1-08e825522803" />


```

## RESULT:
This program has been executed successfully.
