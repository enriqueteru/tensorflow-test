import tensorflow as tf
import numpy as np

celsius = np.array([0,1,12,45.21,34,99.91], dtype = float)
fahrenheit = np.array([32,33.8,53.6,113.378,93.2,211.838], dtype = float)

layer1 = tf.keras.layers.Dense(units=3, input_shape=[1])
layer2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([layer1, layer2, output])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

print("Start learning...")
historial = model.fit(celsius, fahrenheit, epochs=10000, verbose=False)
print("Finish learning")

import matplotlib.pyplot as plt
plt.xlabel("# Times")
plt.ylabel("lossing magnitude")
plt.plot(historial.history["loss"])

print("Predict the celsius to fahrenheit")
result = model.predict([100])
print(result)

print("inside tensor flow")
print(layer1.get_weights())