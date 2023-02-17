import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,784).astype("float32")/255.0
print(x_train.shape)
x_test = x_test.reshape(-1,784).astype("float32")/255.0


# Sequantial API

model = tf.keras.Sequential([
    layers.InputLayer(input_shape= (28*28)),
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu',name= 'my_layer'),
    layers.Dense(10)
])

model = tf.keras.Model(inputs= model.inputs,
                    outputs = [layer.output for layer in model.layers])

features = model.predict(x_train)
for feature in features:
    print(feature.shape)

# import sys 
# sys.exit()

# # Funcional API
# inputs = tf.keras.Input(shape= (28*28))
# x = layers.Dense(512,activation='relu')(inputs)
# x = layers.Dense(256,activation='relu')(x)
# outputs = layers.Dense(10,activation='softmax')(x)
# model = tf.keras.Model(inputs=inputs, outputs= outputs)


print(model.summary())


model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),# Fucntional API = False,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

