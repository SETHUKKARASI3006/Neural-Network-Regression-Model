# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This experiment aims to build a neural network for regression tasks using the PyTorch framework. It starts with data preprocessing, where the input features and target values are extracted from a CSV file, split into training and test sets, and normalized using MinMaxScaler. The core model, a multi-layer neural network with three hidden layers, utilizes the ReLU activation function to introduce non-linearity. The model is trained using the Mean Squared Error loss function and optimized using the RMSprop optimizer. During training, the loss is recorded and plotted to visualize performance over epochs. Finally, the model's performance on the test set is evaluated and a single prediction is made to demonstrate its functionality.

## Neural Network Model

![nndiagram](/nn.png)
<br>

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SETHUKKARASI C
### Register Number: 212223230201
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 14)
        self.fc4 = nn.Linear(14, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

![dataset](/dataset.png)
<br>

## OUTPUT

### Training Loss Vs Iteration Plot

![plot](/plot.png)
<br>

### New Sample Data Prediction

![pred](/pred_out.png)
<br>

## RESULT

Thus, a neural network regression model for the given dataset is successfully developed.