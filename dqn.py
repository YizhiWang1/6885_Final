import tensorflow as tf
from tensorflow.keras import layers

def create_dqn(state_size, action_size):
    # sequential model
    model = tf.keras.Sequential([
        # input layer
        layers.Input(shape=(state_size,)),
        # hidden layers
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        # output layers
        layers.Dense(action_size, activation='linear')
    ])
    return model
