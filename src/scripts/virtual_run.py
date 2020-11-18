""" This script performs a virtual run of the experiments with some previously logged data. The "linear fit" performs
        linear regression for the robot control prediction, the "GP fit" performs GP regression for the control
        prediction, however, the GP also re-preforms the run any number of times necessary to perform a search over
        hyperparameters. """
import pandas as pd
import json
import gpflow
import copy
from sklearn.linear_model import LinearRegression
from offline_gp import *
from simple_io import *
from pianolearning import GPModel, GPActionGenerator
np.random.seed(123)

plt.ion()


# ---- CHOSE PARAMETERS FOR VIRTUAL RUN -----
FINESS = 10                 # perform a fit every FINESS new key-presses
RUN_LINEAR = False           # runs virtual linear fit
RUN_GP = True               # runs virtual Gaussian Process fit
RUN_GP_SEARCH = False       # currently unsupported
RUN_LIMIT = 10000           # set this to the number of steps you want to stop the run at
ONLINE_PLOT = True

REFERENCE = "MIDI"          # reference run optimization with respect to MuseScore generated playing styles, alternatively
                            # set this to "human"
# ------------------------------------------

# variables to save files
data_file_name = 'run_data'
data = dict()

# plots for interactive plotting of results
plot_dict = {
    0: {
        'x_label': 'rx',
        'y_label': 'on_velocity',
        'y_idx': 0,
        'ax': None,
        'xs': np.zeros(39),
        'ys': np.zeros(39),
        'min': np.zeros(39),
        'max': np.zeros(39),
    },
    1: {
        'x_label': 'rx',
        'y_label': 'on_velocity',
        'y_idx': 0,
        'ax': None,
        'xs': np.zeros(39),
        'ys': np.zeros(39),
        'min': np.zeros(39),
        'max': np.zeros(39),
    },
    2: {
        'x_label': 'rx',
        'y_label': 'off_velocity',
        'y_idx': 1,
        'ax': None,
        'xs': np.zeros(39),
        'ys': np.zeros(39),
        'min': np.zeros(39),
        'max': np.zeros(39),
    },
    3: {
        'x_label': 'rx',
        'y_label': 'wait_time',
        'y_idx': 2,
        'ax': None,
    },
    4: {
        'x_label': 'rx',
        'y_label': 'hold_time',
        'y_idx': 3,
        'ax': None,
        'xs': np.zeros(39),
        'ys': np.zeros(39),
        'min': np.zeros(39),
        'max': np.zeros(39),
    },
    5: {
        'x_label': 'learning iteration',
        'y_label': 'loss',
        'y_idx': None,
        'ax': None,
        'xs': np.array(range(10000))*FINESS,
        'ys': np.zeros(10000),
    },
}

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.8)

fig.suptitle('Live Run - plots', fontsize=16)
ax[0, 0].set_ylabel('rx')
ax[0, 0].set_xlabel('on_velocity')
plot_dict[0]['ax'] = ax[0, 0]

ax[0, 1].set_ylabel('f1')
ax[0, 1].set_xlabel('on_velocity')
plot_dict[1]['ax'] = ax[0, 1]

ax[0, 2].set_ylabel('f2')
ax[0, 2].set_xlabel('off_velocity')
plot_dict[2]['ax'] = ax[0, 2]

ax[1, 0].set_ylabel('t1')
ax[1, 0].set_xlabel('wait_time')
plot_dict[3]['ax'] = ax[1, 0]

ax[1, 1].set_ylabel('t2')
ax[1, 1].set_xlabel('hold_time')
plot_dict[4]['ax'] = ax[1, 1]

ax[1, 2].set_ylabel('loss')
ax[1, 2].set_xlabel('learning iteration')
plot_dict[5]['ax'] = ax[1, 2]

# parameters necessary for the run
data_keys = ['running_linear_mean_error', 'running_linear_err_error', 'running_linear_error',
             'running_gp_mean_error', 'running_gp_err_error', 'running_gp_error']
y_params = ["rx", "f1", "f2", "t1", "t2"]
x_params = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
play_style = ["normal", "tenuto", "staccatissimo", "staccato", "ff", "f", "mp", "p", "pp", "pppp"]

# data folders
training_data_folder = "./../../data/training_data/"
testing_data_folder = "./../../data/testing_data/"
reference_data_folder = "./../../data/reference_data/"
results_data_folder = "./../../data/results_data/"

if REFERENCE == "MIDI":
    # ----- LOAD MIDI PLAYING STYLES ----------
    filename = "digital_parameters.json"
    with open("{}{}".format(reference_data_folder, filename)) as dig_file:
        dig_audio = pd.DataFrame(json.load(dig_file))
    dig_audio = dig_audio.reset_index(drop=True)
    dig_audio = dig_audio[x_params]
    for idx in range(len(dig_audio)):
        dig_audio.at[idx, "play_style"] = play_style[idx]
    playing_styles_audio = np.array(dig_audio)[:, :-1].astype(np.float64)
else:
    # -----  LOAD HUMAN PLAYING STYLES -----
    filename = "human_parameters.json"
    with open("{}{}".format(reference_data_folder, filename)) as human_file:
        human_audio = pd.DataFrame(json.load(human_file))
    human_audio = human_audio.reset_index(drop=True)
    human_audio = human_audio[x_params]
    human_audio_array = np.array(human_audio)
    playing_styles_audio = np.zeros((int(human_audio_array.shape[0]/4), human_audio_array.shape[1]))
    for i, j in enumerate(range(0, human_audio_array.shape[0], 4)):
        playing_styles_audio[i, :] = np.average(human_audio_array[j:j+4, :], axis=0)


# define function to load data
def load_data(parameters, type='input', folder='', format='.json'):
    file_names = get_filenames(folder, file_ending=format, contains=type)
    loaded_data = None
    for file_name in file_names:
        with open("{}{}".format(folder, file_name)) as param_file:
            pd_object = pd.DataFrame(json.load(param_file))
            if loaded_data is None:
                loaded_data = np.array(pd_object[parameters])
            else:
                loaded_data = np.append(
                    arr=loaded_data,
                    values=np.array(pd_object[parameters]),
                    axis=0
                )
    return loaded_data


# ---------- LOAD TRAINING DATA ----------
training_action = load_data(y_params, folder=training_data_folder, type='input', format='.json')
training_audio = load_data(x_params, folder=training_data_folder, type='output', format='.json')

# ---------- LOAD TEST DATA ----------
testing_action = load_data(y_params, folder=testing_data_folder, type='input', format='.json')
testing_audio = load_data(x_params, folder=testing_data_folder, type='output', format='.json')

# initialize plot parameters
for j in range(training_action.shape[1]):
    length = np.unique(testing_audio[::3, plot_dict[j]['y_idx']]).shape[0]
    plot_dict[j]['xs'] = np.zeros(length)
    plot_dict[j]['ys'] = np.zeros(length)
    plot_dict[j]['min'] = np.zeros(length)
    plot_dict[j]['max'] = np.zeros(length)


# --------------------------------------------------------------------
def find_closest(test_audio_data, test_control_data, predicted_control):
    distances = np.zeros(test_control_data.shape[0])
    for i in range(distances.shape[0]):
        distances[i] = np.sum(np.sqrt((test_control_data[i, :] - predicted_control) ** 2))
    return test_audio_data[np.argmin(distances), :]


def fit_linear(x, y):
    regressors = [LinearRegression() for j in range(y.shape[1])]
    for i in range(len(regressors)):
        regressors[i].fit(x, y[:, i])
    return regressors


def predict_linear(regressors, x_test):
    prediction = np.zeros((x_test.shape[0], len(regressors)))
    for i in range(len(regressors)):
        prediction[:, i] = regressors[i].predict(x_test)
    return prediction


def plot_fit(predictions, x, y, test_x, test_y, plot_dict, loss=None):
    for j in range(training_action.shape[1]):

        xs = np.unique(x[::3, plot_dict[j]['y_idx']])
        ys = []
        ys_avg = []
        ys_std = []
        for var in xs:
            ys += [predictions[
                       np.where((x[::3, plot_dict[j]['y_idx']] == var).astype(np.int32)!=0),
                       j]]
            ys_avg += [np.average(ys[-1])]
            ys_std += [np.std(ys[-1])]
        avg = np.array(ys_avg)
        std = np.array(ys_std)
        plot_dict[j]['xs'][:xs.shape[0]] = xs
        plot_dict[j]['ys'][:xs.shape[0]] = avg
        plot_dict[j]['min'][:xs.shape[0]] = avg - std
        plot_dict[j]['max'][:xs.shape[0]] = avg + std

        for artist in plot_dict[j]['ax'].lines + plot_dict[j]['ax'].collections:
            artist.remove()
        plot_dict[j]['ax'].scatter(x[:, plot_dict[j]['y_idx']], y[:, j], color='r')  # todo
        plot_dict[j]['ax'].plot(plot_dict[j]['xs'][:xs.shape[0]], plot_dict[j]['ys'][:xs.shape[0]], 'k-',
                                zorder=9,
                                linewidth=3,
                                label='mean'
                                )
        plot_dict[j]['ax'].fill_between(plot_dict[j]['xs'][:xs.shape[0]], plot_dict[j]['min'][:xs.shape[0]], plot_dict[j]['max'][:xs.shape[0]],
                                        color='r',
                                        alpha=0.1,
                                        label='$std$'
                                        )
    if loss is not None:
        for artist in plot_dict[5]['ax'].lines + plot_dict[5]['ax'].collections:
            artist.remove()

        plot_dict[5]['ys'][len(loss)-1:len(loss)] = loss[-1]
        plot_dict[5]['ax'].plot(plot_dict[5]['xs'][:len(loss)], plot_dict[5]['ys'][:len(loss)])

    plt.show()
    plt.pause(0.0001)


def pick_gp_action(actions, gp_model):
    _, var = gp_model.predict(X=actions)
    return np.argmin(np.sum(var, axis=1))

# -----------------LINEAR RUN ------------------------------------
if RUN_LINEAR:
    control_predictions = np.zeros((playing_styles_audio.shape[0], training_action.shape[1]))
    running_linear_mean_error = []
    running_linear_err_error = []
    running_linear_error = []
    running_linear_param_style_errors = []
    running_linear_param_outputs = []
    running_linear_param_controls = []
    running_actual_linear_param_controls = []
    style_error = np.zeros(playing_styles_audio.shape[0])
    # linear run
    for i in range(1, min(training_action.shape[0] + 1, RUN_LIMIT)):
        if i % FINESS == 0:
            x_audio_outputs = training_audio[:i, :]
            y_control_actions = training_action[:i, :]

            regressors = fit_linear(x_audio_outputs, y_control_actions)
            preds = predict_linear(regressors, x_audio_outputs[::3, :])

            control_style_error = []
            audio_outputs = []
            for j in range(playing_styles_audio.shape[0]):
                control_predictions[j, :] = predict_linear(regressors, playing_styles_audio[j, :].reshape(1, -1))
                audio_test = find_closest(testing_audio, testing_action, control_predictions[j, :])
                audio_outputs += [list(audio_test)]
                control_style_error += [list(np.sqrt((audio_test - playing_styles_audio[j, :]) ** 2))]
                style_error[j] = np.sqrt(np.sum((audio_test-playing_styles_audio[j, :])**2))

            running_linear_param_style_errors += [copy.deepcopy(control_style_error)]
            running_linear_param_outputs += [copy.deepcopy(audio_outputs)]
            running_linear_param_controls += [copy.deepcopy(control_predictions.tolist())]
            running_linear_mean_error += [np.average(style_error)]
            running_linear_err_error += [np.std(style_error)]
            running_linear_error += [list(style_error.copy())]

            plot_fit(predictions=preds,
                     x=x_audio_outputs,
                     y=y_control_actions,
                     test_x=testing_audio,
                     test_y=testing_action,
                     plot_dict=plot_dict,
                     loss=running_linear_mean_error)

            print("{0} out of {1}. {2:.2f}% --- Error: {3}".format(
                i,
                training_action.shape[0] + 1,
                i/(training_action.shape[0] + 1)*100,
                running_linear_mean_error[-1])
            )

    data['running_linear_mean_error'] = list(running_linear_mean_error)
    data['running_linear_err_error'] = list(running_linear_err_error)
    data['running_linear_error'] = list(running_linear_error)
    data['control_linear_style_error'] = running_linear_param_style_errors
    data['running_linear_param_outputs'] = running_linear_param_outputs
    data['running_linear_param_controls'] = running_linear_param_controls

    with open('{}{}_{}.json'.format(results_data_folder, data_file_name, "linear"), 'w') as outfile:
        json.dump(data, outfile)

params = {
    'action_params_min': np.array([0, 2.083, 2.083, 0.92, 0.92]),   # staccato hard-corded params
    'action_params_max': np.array([90, 8.03, 8.03, 2.58, 2.58]),  # + hold_time exploration
    'grid_action_number': 5,
    'GP_grid_action_number': 5,
    'params_to_optimize': ['on_velocity', "hold_time", "wait_time", "off_velocity"],
    'corr': {
        'f1': 'on_velocity',
        'f2': 'off_velocity',
        't1': 'wait_time',
        't2': 'hold_time',
        'rx': 'on_velocity'
    }
 }

variance_backwards = [
    [0.10, 10.00],
    [10.00, 0.10],
    [10.00, 0.10],
    [0.05, 0.10],
    [10.00, 0.05],
    [0.05, 1.00],
    [1.00, 0.10],
    [10.00, 1.00],
    [0.10, 1.00],
    [1.00, 10.00],
    [0.10, 10.00],
    [0.05, 1.00],
    [0.10, 1.00],
    [1.00, 10.00],
    [0.10, 1.00],
    [1.00, 10.00]
]
lengthscale_backwards = [
    [10.00, 10.00, 10.00, 10.00],
    [0.05, 1.00, 10.00, 10.00],
    [1.00, 1.00, 1.00, 10.00],
    [1.00, 10.00, 10.00, 0.10],
    [1.00, 1.00, 1.00, 10.00],
    [10.00, 1.00, 10.00, 1.00],
    [0.10, 10.00, 10.00, 0.10],
    [0.10, 10.00, 10.00, 0.10],
    [10.00, 1.00, 10.00, 1.00],
    [10.00, 1.00, 10.00, 1.00],
    [10.00, 10.00, 10.00, 1.00],
    [10.00, 1.00, 10.00, 10.00],
    [10.00, 10.00, 10.00, 1.00],
    [10.00, 10.00, 10.00, 1.00],
    [10.00, 0.05, 10.00, 10.00],
    [10.00, 0.05, 10.00, 10.00]
]

if RUN_GP:
    for lengthscale_backward, variance_backward in zip(lengthscale_backwards, variance_backwards):
        control_predictions = np.zeros((playing_styles_audio.shape[0], training_action.shape[1]))
        control_confidence = np.zeros((playing_styles_audio.shape[0], training_action.shape[1]))
        running_gp_mean_error = []
        running_gp_err_error = []
        running_gp_error = []
        running_param_style_errors = []
        running_gp_param_outputs = []
        running_gp_param_controls = []
        running_gp_param_controls_confidence = []
        running_actual_gp_param_controls = []
        style_error = np.zeros(playing_styles_audio.shape[0])

        # gp run 0.01 0.01 5.00 7.50
        kernel_inverse = [None] * training_action.shape[1]
        for i_param in range(training_action.shape[1]):
            kernel_inverse[i_param] = gpflow.kernels.SquaredExponential(
                lengthscales=np.array(lengthscale_backward),
                variance=variance_backward[0]
            ) + gpflow.kernels.Linear(
                variance=variance_backward[0],
            )
        for i in range(1, min(training_action.shape[0] + 1, RUN_LIMIT)):
            if i % FINESS == 0:
                # x_audio_outputs = training_audio[:i, :]
                # y_control_actions = training_action[:i, :]

                gp_model = GPModel(
                    params=params,
                    verbose=False)

                gp_model.update_fit_data(
                    X=np.array(training_audio[:i, :]),
                    Y=np.array(training_action[:i, :]),
                    inverse=True,
                    kernel_inverse=kernel_inverse,
                    likelihood=variance_backward[1]
                )

                preds, _ = gp_model.predict(
                    X=training_audio[:i:3, :],
                    inverse=True)

                # _, gp_inverse_models = fit_gp_models(x_audio_outputs, y_control_actions, params=params, inverse=True)
                control_style_error = []
                audio_outputs = []
                for j in range(playing_styles_audio.shape[0]):
                    control_predictions[j, :], control_confidence[j, :] = gp_model.predict(
                        X=playing_styles_audio[j, :].reshape(1, -1),
                        inverse=True)
                    audio_test = find_closest(testing_audio, testing_action, control_predictions[j, :])
                    audio_outputs += [list(audio_test)]
                    control_style_error += [list(np.sqrt((audio_test - playing_styles_audio[j, :]) ** 2))]
                    style_error[j] = np.sqrt(np.sum((audio_test - playing_styles_audio[j, :])**2))
                running_param_style_errors += [copy.deepcopy(control_style_error)]
                running_gp_param_outputs += [copy.deepcopy(audio_outputs)]
                running_gp_param_controls += [copy.deepcopy(list(control_predictions.tolist()))]
                running_gp_param_controls_confidence += [copy.deepcopy(list(control_confidence.tolist()))]
                running_gp_mean_error += [np.average(style_error)]
                running_gp_err_error += [np.std(style_error)]
                running_gp_error += [list(copy.deepcopy(style_error))]

                if ONLINE_PLOT:
                    plot_fit(predictions=preds,
                             x=training_audio[:i, :],
                             y=training_action[:i, :],
                             test_x=testing_audio,
                             test_y=testing_action,
                             plot_dict=plot_dict,
                             loss=running_gp_mean_error)

                print("{0} out of {1}. {2:.2f}% --- Error: {3}".format(
                    i,
                    training_action.shape[0] + 1,
                    i / (training_action.shape[0] + 1) * 100,
                    running_gp_mean_error[-1])
                )

        data['running_gp_mean_error'] = list(running_gp_mean_error)
        data['running_gp_err_error'] = list(running_gp_err_error)
        data['running_gp_error'] = list(running_gp_error)
        data['control_gp_style_error'] = running_param_style_errors
        data['running_gp_param_outputs'] = running_gp_param_outputs
        data['running_gp_param_controls'] = running_gp_param_controls
        data['running_gp_param_controls_confidence'] = running_gp_param_controls_confidence

        with open('{}{}{}{}.json'.format(results_data_folder, data_file_name, lengthscale_backward, variance_backward), 'w') as outfile:
            json.dump(data, outfile)

# plt.ioff()
# learing_figure = plt.figure(0, figsize=(8, 8))
# ax = learing_figure.add_subplot(111)
# ax.title.set_text('Z Profile')
#
# ax.plot(list(range(len(data['running_linear_mean_error']))), data['running_linear_mean_error'], 'b-',
#         linewidth=2,
#         label='mean linear prediction'
#         )
# ax.fill_between(list(range(len(data['running_linear_mean_error']))),
#                 np.array(data['running_linear_mean_error']) - np.array(data['running_linear_err_error']),
#                 np.array(data['running_linear_mean_error']) + np.array(data['running_linear_err_error']),
#                 color='C0',
#                 alpha=0.1,
#                 label='$linear\ err$'
#                 )
#
# ax.plot(list(range(len(data['running_gp_mean_error']))), data['running_gp_mean_error'], 'k-',
#         # zorder=9,
#         linewidth=3,
#         label='mean gp prediction'
#         )
# ax.fill_between(list(range(len(data['running_gp_mean_error']))),
#                 np.array(data['running_gp_mean_error']) - np.array(data['running_gp_err_error']),
#                 np.array(data['running_gp_mean_error']) + np.array(data['running_gp_err_error']),
#                 color='r',
#                 alpha=0.1,
#                 label='$gp\ error$'
#                 )
#
# ax.set_xlabel('learning itearations')
# ax.set_ylabel('playing error')
# ax.set_autoscale_on(True)
# ax.legend()
# plt.show()
# time.sleep(20)
