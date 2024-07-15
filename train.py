import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

# Load data
data_file = open("./training_data/training_data.csv", "r")
data = []

for line in data_file:
    sub_data = []
    for id, carac in enumerate(line.split(";")):
        if carac == '\n':
            continue
        if id > 63:
            break
        sub_data.append(float(carac))
    data.append(sub_data)

data_file.close()

# Split data into training and testing

#separate the data into training and testing data 

np.random.shuffle(data)

train_data = np.array(data[:int(len(data)*0.8)])
test_data = np.array(data[int(len(data)*0.8):])



if len(train_data) == 0:
    raise ValueError("Training data contains 0 samples. Provide more data.")

# Define callbacks

num_classes = 5
batch_size = 16
epochs = 250

# Prepare training and testing data
x_train = train_data[:, 1:]
y_train = train_data[:, 0]
x_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Reshape data
x_train = x_train.reshape(-1, 63, 1)
x_test = x_test.reshape(-1, 63, 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Define the model
model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(63, 1)),
        keras.layers.Conv1D(32, 3, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(40, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax")
    ]
)

model.summary()

# Compile the model
model.compile(
    loss="categorical_crossentropy", 
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    x_train,
    y_train,    
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)

# Evaluate the model
score = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(score)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save("./saved_models/trained/model.keras")