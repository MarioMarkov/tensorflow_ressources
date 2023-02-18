import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print("Train Shape: ", x_train.shape)
# model = tf.keras.Sequential(
#     [
#         tf.keras.Input(shape = (32,32,3)),
#         layers.Conv2D(32, 3, padding='valid',activation='relu'),
#         layers.MaxPool2D(pool_size=(2,2)),
#         layers.Conv2D(64, 3 ,activation='relu'),
#         layers.MaxPool2D(),
#         layers.Conv2D(128,3,activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64,activation = 'relu'),
#         layers.Dense(10)        
#     ]
# )


def my_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(
        x
    )
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        128, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),
    )(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01),)(
        x
    )
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = my_model()
print(model.summary())

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),# Fucntional API = False,
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4),
    metrics = ["accuracy"]
)

model.fit(x_train,y_train, batch_size =64, epochs =50,verbose=2)

model.evaluate(x_test, y_test,batch_size =64, verbose=2)