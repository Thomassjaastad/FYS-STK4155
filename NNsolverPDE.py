import tensorflow as tf
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


Nx = 10
x_np = np.linspace(0, 1, Nx)

Nt = 10
t_np = np.linspace(0, 0.6,Nt)
X,T = np.meshgrid(x_np, t_np)
x = X.ravel()
t = T.ravel()

## The construction phase
zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))

x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

points = tf.concat([x,t],1)

num_iter = 1000000
num_hidden_neurons = [90]

X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)

with tf.variable_scope('dnn'):
    num_hidden_layers = np.size(num_hidden_neurons)

    prev_layer = points
    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(prev_layer, num_hidden_neurons[l], activation=tf.nn.sigmoid)
        prev_layer = current_layer

    dnn_output = tf.layers.dense(prev_layer, 1)

def u(x):
    return tf.sin(np.pi*x)

with tf.name_scope('loss'):
    g_trial = (1 - t)*u(x) + x*(1 - x)*t*dnn_output

    g_trial_dt = tf.gradients(g_trial, t)
    g_trial_d2x = tf.gradients(tf.gradients(g_trial,x), x)

    loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])


learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    traning_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

g_analytic = tf.exp(-np.pi**2*t)*tf.sin(np.pi*x)
g_dnn = None

## The execution phase
with tf.Session() as sess:
    init.run()
    for i in range(num_iter):
        sess.run(traning_op)
        # If one desires to see how the cost function behaves during training
        #if i % 100 == 0:
        #    print(loss.eval())
    g_analytic = g_analytic.eval()
    g_dnn = g_trial.eval()

diff = np.abs(g_analytic - g_dnn)

G_analytic = g_analytic.reshape((Nt, Nx))
G_dnn = g_dnn.reshape((Nt, Nx))

diff = np.abs(G_analytic - G_dnn)

# Plot the results
X,T = np.meshgrid(x_np, t_np)

cmap = cm.viridis
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/%dlayer' % len(num_hidden_neurons))
s = ax.plot_surface(X, T, G_dnn,linewidth = 0, antialiased = False, cmap = cmap)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Time $t$')

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(X,T,G_analytic,linewidth = 0,antialiased=False,cmap = cmap)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Time $t$')

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap = cmap)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Time $t$')
plt.show()