import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D

import gpflow
from gpflow.utilities import print_summary

from matplotlib import cm


# Load data
data_folder = r"..\data\piano_learning_experiment_0"

input_param_dir = os.path.join(data_folder, r"all_input_parameters.json")
output_param_dir = os.path.join(data_folder, r"all_output_parameters.json")
demo_dir = os.path.join(data_folder, r"demo_parameters.json")

# load parameter data
with open(input_param_dir) as input_param_file:
    input_param = pd.DataFrame(json.load(input_param_file))
with open(output_param_dir) as output_param_file:
    output_param = pd.DataFrame(json.load(output_param_file))
with open(demo_dir) as demo_file:
    demo_param = pd.DataFrame(json.load(demo_file))

# preprocess data
input_param = input_param.drop(columns=['wait_time'])
output_param = output_param.drop(columns=["note"])
input_param = input_param.rename(columns={"on_velocity": "Q1",  # downwards velocity
                                          "off_velocity": "Q2",  # upwards velocity
                                          "hold_time": "Q3"})  # wait time between velocities
# output_param = output_param.rename(columns={"on_velocity": "output_on_velocity",
#                                           "off_velocity": "output_off_velocity",
#                                           "hold_time": "output_hold_time"})
all_param = pd.concat([input_param, output_param], axis=1, sort=False)

# remove outliers - might delete
all_param = all_param[all_param['off_velocity'] > 20]
all_param = all_param[all_param['hold_time'] < 5]
all_param = all_param[:3000]

X = np.array(all_param[["on_velocity", "off_velocity", "hold_time"]])
# y = np.array(all_param["Q3"])
y = np.array(all_param[["Q1", "Q2", "Q3"]])

# """Gaussian Process Model"""
#
# # ---------------   fit Q1 hold time ---------------
# plt.figure(0)
# plt.title("fit Q1 to hold time")
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3 / 2) + WhiteKernel(noise_level=1)
# gpr = GaussianProcessRegressor(kernel=kernel)
# gpr.fit(X[:, 0].reshape(-1, 1), y.reshape(-1, 1))
#
# x_plot = np.linspace(0, 3, 100)
# y_mean, y_cov = gpr.predict(x_plot[:, np.newaxis], return_cov=True)
# plt.plot(x_plot, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(x_plot, y_mean.flatten() - np.sqrt(np.diag(y_cov)),
#                  y_mean.flatten() + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')
#
# X_pred = np.array(demo_param[["on_velocity","off_velocity","hold_time"]])[:, 2]
# y_pred, sigma = gpr.predict(X_pred[:, np.newaxis], return_std=True)
# plt.scatter(X_pred, y_pred, c='r')
#
#
# #
# # # ---------------   fit Q2 hold time  ---------------
# plt.figure(1)
# plt.title("fit Q2 to hold time")
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3 / 2) + WhiteKernel(noise_level=1)
# gpr = GaussianProcessRegressor(kernel=kernel)
# gpr.fit(X[:, 1].reshape(-1, 1), y.reshape(-1, 1))
#
# x_plot = np.linspace(0, 120, 100)
# y_mean, y_cov = gpr.predict(x_plot[:, np.newaxis], return_cov=True)
# plt.plot(x_plot, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(x_plot, y_mean.flatten() - np.sqrt(np.diag(y_cov)),
#                  y_mean.flatten() + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')
#
# X_pred = np.array(demo_param[["on_velocity","off_velocity","hold_time"]])[:, 2]
# y_pred, sigma = gpr.predict(X_pred[:, np.newaxis], return_std=True)
# plt.scatter(X_pred, y_pred, c='r')
#

# ---------------   fit Q3 hold time   ---------------
# plt.figure(2)
# plt.title("fit Q3 to hold time")
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3 / 2) + WhiteKernel(noise_level=1.)
# gpr = GaussianProcessRegressor(kernel=kernel)
# gpr.fit(X[:500, 2].reshape(-1, 1), y[:500].reshape(-1, 1))
#
# x_plot = np.linspace(0, 5, 100)
# y_mean, y_cov = gpr.predict(x_plot[:, np.newaxis], return_cov=True)
# plt.plot(x_plot, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(x_plot, y_mean.flatten() - np.sqrt(np.diag(y_cov)),
#                  y_mean.flatten() + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')
#
# X_pred = np.array(demo_param[["on_velocity","off_velocity","hold_time"]])[:, 2]
# y_pred, sigma = gpr.predict(X_pred[:, np.newaxis], return_std=True)
# plt.scatter(X_pred, y_pred, c='r')
#
#
# plt.show()

idxes = np.arange(0, X.shape[0], 200).astype(np.int)
x1_plot = np.linspace(np.min(X[idxes, 0]), np.max(X[idxes, 0]), 20).reshape(-1, 1)
x2_plot = np.linspace(np.min(X[idxes, 1]), np.max(X[idxes, 1]), 20).reshape(-1, 1)
x3_plot = np.linspace(np.min(X[idxes, 2]), np.max(X[idxes, 2]), 20).reshape(-1, 1)
xx, yy, zz = np.meshgrid(x1_plot, x2_plot, x3_plot)

for elem_num in [30, 500, 1000, 3000]:
    k = gpflow.kernels.Matern52()
    m = gpflow.models.GPR(data=(X[:elem_num, :], y[:elem_num, :]), kernel=k, mean_function=None)
    print_summary(m)
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscale.assign(2)
    opt = gpflow.optimizers.Scipy()


    def objective_closure():
        return - m.log_marginal_likelihood()


    opt_logs = opt.minimize(objective_closure,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    print_summary(m)

    # x_plot = np.array([x1_plot, x2_plot, x3_plot]).T
    mean, var = m.predict_f(np.array([X[idxes, 0], X[idxes, 1], X[idxes, 2]]).T)
    y_actual = y[idxes, :]
    mse = np.sum(np.sqrt((mean - y_actual)**2))
    unc = np.sum(var)
    print(" mse: {}\n uncertainty: {}\n".format(mse, unc))




# ## generate 10 samples from posterior
# # samples = m.predict_f_samples(x_plot.T, 10)  # shape (10, 100, 1)
#
# ## plot
# plt.figure(figsize=(12, 6))
# # plt.plot(X, y, 'kx', mew=2)
# plt.plot(x_plot, mean, 'C0', lw=2)
# plt.fill_between(x_plot[:, 0],
#                  mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
#                  mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
#                  color='C0', alpha=0.2)
# plt.scatter(X[:500, 2], y[:500, 2], c='r')
#
# # plt.plot(x_plot, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
# plt.show()

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')
# ax.scatter(mean[:, 0], mean[:, 1], mean[:, 2], c='b')
plt.scatter(y[0:500, 0], y[0:500, 1], y[0:500, 2], c='r')
plt.show()