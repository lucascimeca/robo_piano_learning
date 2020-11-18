import pandas as pd
import json
import gpflow
from sklearn.linear_model import LinearRegression
from offline_gp import *
from simple_io import *
from pianolearning import GPModel
np.random.seed(123)

plt.ion() ## Note this correction

REFERENCE = "MIDI"
FINESS = 10
RUN_LINEAR = True
RUN_GP = True
RUN_GP_SEARCH = False
RUN_LIMIT = 10000

data_file_name = 'run_data'

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


def plot_fit(predictions, x, y, test_x, test_y, plot_dict):
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
        plot_dict[j]['ax'].scatter(x[:, plot_dict[j]['y_idx']], y[:, j])  # todo
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
    plt.show()
    plt.pause(0.0001)


def pick_gp_action(actions, gp_model):
    _, var = gp_model.predict(X=actions)
    return np.argmin(np.sum(var, axis=1))


if file_exists('{}.json'.format(data_file_name)):
    with open('{}.json'.format(data_file_name)) as json_file:
        data = json.load(json_file)

params = {
             # GP SEARCH
             'action_params_min': np.array([0, 2.083, 0, 0, 0]),  # staccato hard-corded params
             'action_params_max': np.array([0, 8.03, 0, 0, 0]),  # + hold_time exploration

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
variance_backward = [0.10, 10.00]
lengthscale_backward = [10.00, 10.00, 10.00, 10.00]

data_audio = np.concatenate((training_audio, testing_audio), axis=0)
data_action = np.concatenate((training_action, testing_action), axis=0)

varying_f1 = dict()
for i in range(data_action.shape[0]):
    key = (data_action[i, 0],) + tuple(data_action[i, 2:])
    if key not in varying_f1.keys():
        varying_f1[key] = 1
    else:
        varying_f1[key] += 1

search_data = []
for key in varying_f1.keys():
    search_data += [(key, varying_f1[key])]

search_data = sorted(search_data, key=lambda x: x[1], reverse=True)

idx_data_search = 9

data_audio_f1 = np.zeros((search_data[idx_data_search][1], data_audio.shape[1]))
data_action_f1 = np.zeros((search_data[idx_data_search][1], data_action.shape[1]))
j = 0

for i in range(data_action.shape[0]):
    if data_action[i, 0] == search_data[idx_data_search][0][0] \
            and data_action[i, 2] == search_data[idx_data_search][0][1] \
            and data_action[i, 3] == search_data[idx_data_search][0][2] \
            and data_action[i, 4] == search_data[idx_data_search][0][3]:
        data_audio_f1[j, :] = data_audio[i, :]
        data_action_f1[j, :] = data_action[i, :]
        j += 1


print("Maximum number of elements found/chosen: {}\nparameters for maximum: {}".format(
    search_data[idx_data_search][1],
    search_data[idx_data_search][0])
)


# plot with various axes scales
learing_figure = plt.figure(0, figsize=(7, 6))
ax = learing_figure.add_subplot(111)
axes = plt.gca()

# linear
ax.set_ylabel('$f_1(deg)$', fontsize=28)
ax.set_xlabel('normalized on_velocity', fontsize=28)

kernel_inverse = gpflow.kernels.SquaredExponential(
    lengthscales=np.array(lengthscale_backward),
    variance=variance_backward[0]
) + gpflow.kernels.Linear(
    variance=variance_backward[0],
)


# extrapolate
range_ratio = .1
x_range = max(data_audio[:, 0]) - min(data_audio[:, 0])
y_range = max(data_action[:, 1]) - min(data_action[:, 1])
min_on_vel = min(data_audio_f1[:, 0]) - range_ratio*x_range
max_on_vel = max(data_audio_f1[:, 0]) + range_ratio*x_range
min_f1 = min(data_action_f1[:, 1]) - range_ratio*y_range
max_f1 = max(data_action_f1[:, 1]) + range_ratio*y_range

xs = np.linspace(min_on_vel, max_on_vel, 1000)

for i in range(1, data_audio_f1.shape[0]+1):

    gp_model = GPModel(
        params=params,
        output_indexes_to_model=[0],
        search_indexes_to_optimize=[1],
        verbose=False)

    gp_model.update_fit_data(
        X=data_audio_f1[:i, 0].reshape(-1, 1),
        Y=data_action_f1[:i, 1].reshape(-1, 1),
        inverse=True,
        kernel_inverse=kernel_inverse,
        likelihood=variance_backward[1],
        dim=1,
        data_override=True
    )

    preds, var = gp_model.predict(
        X=xs.reshape(-1, 1),
        inverse=True,
        dim=1)

    preds = np.array(preds).flatten()
    var = np.array(var).flatten()

    # clean plot
    for artist in ax.lines + ax.collections:
        artist.remove()


    normalized_xs = (xs-min(data_audio_f1[:, 0]))/(max(data_audio_f1[:, 0])-min(data_audio_f1[:, 0]))
    normalized_audio = (data_audio_f1[:i, 0]-min(data_audio_f1[:, 0]))/(max(data_audio_f1[:, 0])-min(data_audio_f1[:, 0]))
    # (re-)plot
    ax.scatter(normalized_audio, data_action_f1[:i, 1])
    ax.plot(normalized_xs, preds, 'k-',
            zorder=9,
            linewidth=2,
            label='mean'
            )
    ax.fill_between(normalized_xs,
                    preds - var,
                    preds + var,
                    color='C0',
                    alpha=0.1,
                    label='$std$'
                    )

    axes.set_xlim([-.15, 1.2])
    axes.set_ylim([min_f1, max_f1])
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.savefig('{}{}gp1dfit.png'.format("C:/Users/Luca/Downloads/", i), bbox_inches='tight')
    plt.show()
    plt.pause(0.0001)
    print(i)
