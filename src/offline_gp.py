import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

PIANO_OUTPUT_PARAMETERS = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
ROBOT_CONTROL_PARAMETERS = ["rx", "f1", "f2", "t1", "t2"]

gp_models = dict()
gp_inverse_models = dict()

control_lengthscale = np.array([10., 10., 4., .5, .5])
# output_lengthscale = np.array([10., 10., 1., 1.])
output_lengthscale = np.array([10., 10., 10., 10.])


def fit_gp_models(X, Y, params, inverse=False):
    for i in range(Y.shape[1]):
        # x_var = params['corr'][ROBOT_CONTROL_PARAMETERS[i]]
        # x_idx = [i for i in range(len(PIANO_OUTPUT_PARAMETERS)) if PIANO_OUTPUT_PARAMETERS[i] == x_var][0]
        # kernel_forward = RBF(
        #     length_scale=100.0,
        #     length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        #     noise_level=1,
        #     noise_level_bounds=(1e-10, 1e+1)
        # )
        # # Instantiate a Gaussian Process model
        # gp_models[i] = GaussianProcessRegressor(
        #     kernel=kernel_forward,
        #     n_restarts_optimizer=10).fit(
        #         Y[:, i].reshape(-1, 1),
        #         X[:, x_idx].reshape(-1, 1)
        # )
        if inverse:
            kernel_inverse = RBF(
                length_scale=output_lengthscale,
                length_scale_bounds=[(1e-2, 1e3)]*len(output_lengthscale)) + WhiteKernel(
                noise_level=.1,
                noise_level_bounds=(1e-10, 1e+1)
            )
            # Instantiate a Gaussian Process model
            gp_inverse_models[i] = GaussianProcessRegressor(
                kernel=kernel_inverse,
                n_restarts_optimizer=10).fit(
                    X,
                    Y[:, i].reshape(-1, 1)
            )
    return gp_models, gp_inverse_models


def predict_gp_models(models, X, inverse=False):
    res_mean = []
    res_var = []
    for i in range(len(models)):
        if not inverse:
            mean, var = models[i].predict(X[:, i].reshape(-1, 1), return_cov=True)
        else:
            mean, var = models[i].predict(X, return_cov=True)

        res_mean += [mean.flatten()]
        res_var += [np.diag(var)]
    mean = np.array(res_mean)[:, :].T
    var = np.array(res_var)[:, :].T
    return mean, var


def highest_uncertainty_idx(gp_models, inputs, all_control_params):
    mesh_mean, mesh_var = predict_gp_models(gp_models, X=inputs)
    control_uncertainty = np.zeros(all_control_params.shape)
    for j in range(inputs.shape[1]):
        for i in range(inputs.shape[0]):
            control_uncertainty[all_control_params[:, j] == inputs[i, j], j] = mesh_mean[i, j]

    control_idx = np.argmax(np.sum(control_uncertainty, axis=1))
    control = all_control_params[control_idx, :]
    all_control_params = np.delete(all_control_params, control_idx, 0)
    return control, all_control_params