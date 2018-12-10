import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

L = 1
T = 0.6
n = 10
x = np.linspace(0, L, n)
t = np.linspace(0, T, n)

zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))

x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

def u(x):
    return tf.sin(np.pi*x)

g_analytic = tf.exp(-np.pi**2*t)*tf.sin(np.pi*x)
g_dnn = None


def custom_objective(y_true, y_pred):
    with tf.name_scope('loss'):
        g_trial = (1 - t)*u(x) + x*(1 - x)*t*y_pred
        g_trial_dt = tf.gradients(g_trial, t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)
        loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])
    return loss

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, activation="sigmoid", input_dim = 2))
model.add(tf.keras.layers.Dense(1, activation=None))
model.compile(optimizer = 'sgd', loss = custom_objective)
#epochs=1
#model.fit(x=None, y=g_analytic, batch_size=None, epochs=epochs)

