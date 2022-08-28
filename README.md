# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![nnet2](https://user-images.githubusercontent.com/75234946/187083510-70c889b3-4a50-4041-8241-7cef275b52ed.png)

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
```python
# Importing Required packages

from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

# Authenticate the Google sheet

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('datasetexp1dl').sheet1

# Construct Data frame using Rows and columns

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df=df.astype({'X':'float'})
df=df.astype({'Y':'float'})
X=df[['X']].values
Y=df[['Y']].values

# Split the testing and training data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_t_scaled = scaler.transform(x_train)
x_t_scaled

# Build the Deep learning Model

ai_brain = Sequential([
    Dense(3,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_t_scaled,y=y_train,epochs=20000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

# Evaluate the Model

scal_x_test=scaler.transform(x_test)
ai_brain.evaluate(scal_x_test,y_test)
input=[[120]]
inp_scale=scaler.transform(input)
inp_scale.shape
ai_brain.predict(inp_scale)
```
## Dataset Information

![image](https://user-images.githubusercontent.com/75234946/186728620-1fa89694-7e59-4bde-a56d-64acd7a65291.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75234946/186728854-2713f92f-3d72-443c-be18-99b81bf11512.png)


### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75234946/186729251-84bcadfc-1785-468d-bf92-7f1a2e2185d9.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75234946/186729400-50ea08f8-31c0-4879-9542-7626737bcc3f.png)


## RESULT
Thus a Neural network for Regression model is Implemented
