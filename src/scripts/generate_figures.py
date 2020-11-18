import os
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
from simple_io import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
np.random.seed(123)

data_file_name = 'run_data'

SHOW = True
REFERENCE = "human"    # select either "MIDI" or "human"
OPTIMIZATION = "CONF"  # optimizes hyperparameters based on confidence if "CONF" is selected
                        # alternatively select "VALID", and the hyperparameters with the lowest validation
                        # error will be picked

# data folders
training_data_folder = "./../../data/training_data/"
testing_data_folder = "./../../data/testing_data/"
reference_data_folder = "./../../data/reference_data/"
if REFERENCE == "MIDI":
    experiment_folder = "./../../data/results_data/MIDI_experiments/"
else:
    experiment_folder = "./../../data/results_data/human_experiments/"


x_params = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
play_styles = ["normal", "tenuto", "staccatissimo", "staccato", "ff", "f", "mp", "p", "pp", "pppp"]
x_labels = ["$on\_velocity$", "$off\_velocity$", "$wait\_time$", "$hold\_time$"]
param_markers = [11, 10, '|', '_']

if REFERENCE == "MIDI":
    # ----- LOAD MIDI PLAYING STYLES ----------
    filename = "digital_parameters.json"
    with open("{}{}".format(reference_data_folder, filename)) as dig_file:
        dig_audio = pd.DataFrame(json.load(dig_file))
    dig_audio = dig_audio.reset_index(drop=True)
    dig_audio = dig_audio[x_params]
    for idx in range(len(dig_audio)):
        dig_audio.at[idx, "play_style"] = play_styles[idx]
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

running_linear_mean_error = None
running_linear_err_error = None
running_linear_error = None
control_linear_style_error = None

running_gp_mean_error = None
running_gp_err_error = None
running_gp_error = None
control_gp_style_error = None

run_files = get_filenames(folder=experiment_folder, file_ending='.json', file_beginning='run_data')
for file in run_files:
    with open(experiment_folder+file) as json_file:
        data = json.load(json_file)

        if running_linear_mean_error is None and 'control_linear_style_error' in data.keys():
            running_linear_mean_error = data['running_linear_mean_error']
            running_linear_err_error = data['running_linear_err_error']
            running_linear_error = data['running_linear_error']
            control_linear_style_error = [data['control_linear_style_error']]
            running_linear_param_outputs = data['running_gp_param_outputs']
            running_linear_param_controls = data['running_gp_param_controls']
        if running_gp_mean_error is None and 'running_gp_mean_error' in data.keys():
            running_gp_mean_error = [data['running_gp_mean_error']]
            running_gp_err_error = [data['running_gp_err_error']]
            running_gp_error = [data['running_gp_error']]
            control_gp_style_error = [data['control_gp_style_error']]
            running_gp_param_outputs = [data['running_gp_param_outputs']]
            running_gp_param_controls = [data['running_gp_param_controls']]
            running_gp_param_controls_confidence = [data['running_gp_param_controls_confidence']]
        elif 'running_gp_mean_error' in data.keys():
            running_gp_mean_error += [data['running_gp_mean_error']]
            running_gp_err_error += [data['running_gp_err_error']]
            running_gp_error += [data['running_gp_error']]
            control_gp_style_error += [data['control_gp_style_error']]
            running_gp_param_outputs += [data['running_gp_param_outputs']]
            running_gp_param_controls += [data['running_gp_param_controls']]
            running_gp_param_controls_confidence += [data['running_gp_param_controls_confidence']]

no_of_files = len(running_gp_mean_error)

running_linear_mean_error = np.array(running_linear_mean_error)
running_linear_err_error = np.array(running_linear_err_error)
running_linear_error = np.array(running_linear_error)
control_linear_style_error = np.array(control_linear_style_error).squeeze()
running_linear_param_outputs = np.array(running_linear_param_outputs).squeeze()
running_linear_param_controls = np.array(running_linear_param_controls).squeeze()

running_gp_mean_error = np.array(running_gp_mean_error)  # running_gp_mean_error.min(axis=0)
running_gp_err_error = np.array(running_gp_err_error)  # running_gp_err_error.min(axis=0)
running_gp_error = np.array(running_gp_error)
control_gp_style_error = np.array(control_gp_style_error)  # control_gp_style_error.min(axis=0)
running_gp_param_outputs = np.array(running_gp_param_outputs).squeeze()
running_gp_param_controls = np.array(running_gp_param_controls).squeeze()
running_gp_param_controls_confidence = np.array(running_gp_param_controls_confidence).squeeze()

mins = running_gp_error.min(axis=1).min(axis=0)
for i in range(running_gp_error.shape[2]):
    res = np.where(running_gp_error[:, :, i] == mins[i])
    found_file = False
    for idx, _ in list(zip(res[0], res[1])):
        if '][' in run_files[idx]:
            print("learning style: {}, hyperparameters: {}".format(
                play_styles[i], run_files[idx].split("data")[1].split('.json')[0]))
            found_file = True
            break
    if found_file == False:
        print("No file found for style: {}".format(play_styles[i]))


running_gp_mean_error = np.array(running_gp_mean_error).min(axis=0)
running_gp_err_error = np.array(running_gp_err_error).min(axis=0)
running_gp_error = np.array(running_gp_error).min(axis=0)
control_gp_style_error = np.array(control_gp_style_error).min(axis=0)
minimum = control_linear_style_error.min(axis=0)
maximum = control_linear_style_error.max(axis=0)
norm_linear_mean_error = (control_linear_style_error - minimum)/(maximum - minimum)
avg_norm_linear_mean_error = np.average(norm_linear_mean_error, axis=2)
norm_linear_error_error = np.std(avg_norm_linear_mean_error, axis=1)
norm_linear_mean_error_plot = np.average(avg_norm_linear_mean_error, axis=1)

minimum = control_gp_style_error.min(axis=0)
maximum = control_gp_style_error.max(axis=0)
norm_gp_mean_error = (control_gp_style_error - minimum)/(maximum - minimum)
avg_norm_gp_mean_error = np.average(norm_gp_mean_error, axis=2)
norm_gp_error_error = np.std(avg_norm_gp_mean_error, axis=1)
norm_gp_mean_error_plot = np.average(avg_norm_gp_mean_error, axis=1)

if SHOW:
    plt.close('all')

    learing_figure = plt.figure(0, figsize=(11, 7))
    ax = learing_figure.add_subplot(111)
    ax.title.set_text('Learning Error')


    ax.plot(np.array(range(len(norm_linear_mean_error_plot)))*10, norm_linear_mean_error_plot, color="k",
            # zorder=9,
            linewidth=2,
            label='Mean Linear Prediction'
            )
    ax.fill_between(np.array(range(len(norm_linear_mean_error_plot)))*10,
                    np.array(norm_linear_mean_error_plot) - np.array(norm_linear_error_error),
                    np.array(norm_linear_mean_error_plot) + np.array(norm_linear_error_error),
                    color='#a8a5a5',
                    alpha=0.1,
                    label='$Linear\ Error$'
                    )

    ax.plot(np.array(range(len(norm_gp_mean_error_plot)))*10, norm_gp_mean_error_plot, color='#8f0404',
            # zorder=9,
            linewidth=2,
            label='Mean GP Prediction'
            )
    ax.fill_between(np.array(range(len(norm_gp_mean_error_plot)))*10,
                    np.array(norm_gp_mean_error_plot) - np.array(norm_gp_error_error),
                    np.array(norm_gp_mean_error_plot) + np.array(norm_gp_error_error),
                    color='#ff6969',
                    alpha=0.1,
                    label='$GP\ Error$'
                    )

    ax.set_xlabel('learning iterations', fontsize=16)
    ax.set_ylabel('normalized cumulative playing error', fontsize=16)
    ax.set_autoscale_on(True)
    plt.ylim(ymin=0)
    ax.legend()
    plt.show()
print("Minimum error for Linear fit: {}".format(min(norm_linear_mean_error_plot)))
print("Minimum error for GP fit: {}".format(min(norm_gp_mean_error_plot)))

if SHOW:
    learing_figure = plt.figure(1, figsize=(11, 7))
    ax = learing_figure.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.title.set_text('Minimum Error')
    xs = list(range(1, running_linear_error.shape[1] + 1))
    plt.scatter(xs, np.average(norm_linear_mean_error[np.argmin(norm_linear_mean_error_plot), :, :], axis=1), marker="+", s=250, linewidth=4, label="Min. linear Error", c='k')
    plt.scatter(xs, np.average(norm_gp_mean_error[np.argmin(norm_gp_mean_error_plot), :, :], axis=1), marker='x', s=250, linewidth=4, label="Min. GP Error", c='r')
    plt.xticks(xs, ["${}$".format(label) for label in play_styles], rotation=20)  # Set text labels.
    plt.legend(fontsize=14)
    plt.show()


if SHOW:
    y_linear = np.average(np.array(control_linear_style_error), axis=0).T
    y_gp = np.average(np.array(control_gp_style_error), axis=0).T
    learing_figure = plt.figure(2, figsize=(11, 7))
    ax = learing_figure.add_subplot(111)
    ax.set_xlabel('play_style', fontsize=16)
    ax.set_ylabel('normalized playing error', fontsize=16)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.title.set_text('Style Comparison')
    xs = list(range(1, y_gp.shape[1] + 1))
    for i in range(y_gp.shape[0]):
        # maximum = max(np.max(control_gp_style_error[:, :, i]), np.max(control_linear_style_error[:, :, i]))
        # minimum = min(np.min(control_gp_style_error[:, :, i]), np.max(control_linear_style_error[:, :, i]))
        for j in range(y_linear.shape[1]):
            # y_lin_norm = (y_linear[i, j]-minimum)/(maximum-minimum)
            # y_gp_norm = (y_gp[i, j]-minimum)/(maximum-minimum)
            ax.scatter(
                xs[j],
                norm_linear_mean_error[np.argmin(norm_linear_mean_error_plot), j, i],
                marker=param_markers[i],
                s=170,
                linewidth=4,
                c='k'
            )
            ax.scatter(
                xs[j],
                norm_gp_mean_error[np.argmin(norm_gp_mean_error_plot), j, i],
                marker=param_markers[i],
                s=170,
                linewidth=4,
                c='r'
            )

    plt.xticks(xs, ["${}$".format(label) for label in play_styles], rotation=20)  # Set text labels.
    legend_elements = [
        Patch(facecolor='k', label='Linear Error'),
        Patch(facecolor='r', label='GP Error'),
        Line2D([0], [0], marker=11, label='$on\_velocity$', color='w', markerfacecolor='white', markeredgecolor='k', markersize=15),
        Line2D([0], [0], marker='_', label='$hold\_time$', color='w', markerfacecolor='white', markeredgecolor='k', markersize=15),
        Line2D([0], [0], marker=10, label='"$off\_velocity$"', color='w', markerfacecolor='white', markeredgecolor='k', markersize=15),
        Line2D([0], [0], marker='|', label="$wait\_time$", color='w', markerfacecolor='white', markeredgecolor='k', markersize=15),
    ]
    ax.legend(handles=legend_elements, loc='best')
    # plt.legend(fontsize=12)
    plt.show()




all_outputs_min = np.append(running_gp_param_outputs.min(axis=0), playing_styles_audio.reshape((1,)+playing_styles_audio.shape), axis=0)
all_outputs_max = np.append(running_gp_param_outputs.max(axis=0), playing_styles_audio.reshape((1,)+playing_styles_audio.shape), axis=0)
minimum = all_outputs_min.min(axis=0).min(axis=0)
maximum = all_outputs_max.max(axis=0).max(axis=0)

running_gp_param_outputs_normalized = (running_gp_param_outputs - minimum)/([x if x != 0 else 1 for x in maximum - minimum])
playing_styles_audio_normalized = (playing_styles_audio - minimum)/([x if x != 0 else 1 for x in maximum - minimum])

print("\n\nGP RESULTS - optimized per style\n")
for ps_idx, play_style in enumerate(play_styles):   # iterate through all styles
    style_idxs = None
    min_error = None
    for i in range(running_gp_param_outputs.shape[0]):  # iterate over all files
        for j in range(running_gp_param_outputs.shape[1]):  # iterate over all learning steps
            if OPTIMIZATION == "CONF":
                error = np.average(running_gp_param_controls_confidence[i, j, ps_idx, :])
            else:
                error = np.sqrt(np.sum((running_gp_param_outputs_normalized[i, j, ps_idx, :] - playing_styles_audio_normalized[ps_idx, :])**2))
            if min_error is None or error < min_error:
                min_error = error
                style_idxs = i, j
    print("\"{}\": [{}, {}, {}, {}, {}],".format(play_style, *list(running_gp_param_controls[style_idxs[0], style_idxs[1], ps_idx, :])))

print("\n\nGP RESULTS - optimized across styles\n")
idxs = None
min_error = None
for i in range(running_gp_param_outputs.shape[0]):
    for j in range(running_gp_param_outputs.shape[1]):  # iterate over all learning steps
        if OPTIMIZATION == "CONF":
            error = np.average(running_gp_param_controls_confidence[i, j, :, :])
        else:
            error = np.average(np.sqrt(np.sum((running_gp_param_outputs_normalized[i, j, :, :] - playing_styles_audio_normalized)**2, axis=1)))
        if min_error is None or error < min_error:
            min_error = error
            idxs = i, j
for j, play_style in enumerate(play_styles):
    print("\"{}\": [{}, {}, {}, {}, {}],".format(play_style, *list(running_gp_param_controls[idxs[0], idxs[1], j, :])))


# FIGURE PLOTTING ALL results from all files
if SHOW:
    for i in range(running_gp_param_outputs_normalized.shape[0]):
        error = []
        for j in range(running_gp_param_outputs_normalized.shape[1]):
            error += [np.average(np.sqrt(np.sum((running_gp_param_outputs_normalized[i, j, :, :] - playing_styles_audio_normalized)**2, axis=1)))]
        plt.figure()
        plt.plot(error)
        plt.show()

# FIGURE PLOTTING results from grid-search from all files
if SHOW:
    errors_by_output = []
    errors_avg = []
    for i in range(running_gp_param_outputs_normalized.shape[1]):
        error_by_hyperparameter = np.sqrt(np.sum((running_gp_param_outputs_normalized[:, i, :, :] - playing_styles_audio_normalized)**2, axis=1))
        errors_by_output += [list(error_by_hyperparameter.min(axis=0))]  # pick best performing hyperparam
        errors_avg += [np.average(errors_by_output[-1])]

    plt.figure()
    plt.plot(errors_by_output)
    plt.show()

    plt.figure()
    plt.plot(errors_avg)
    plt.show()

# plt.close('all')
if SHOW:
    fig_ylabels = ["$Rx(deg)$", "$f_1(Hz)$", "$f_2(Hz)$", "$t_1(s)$", "$t_2(s)$"]
    for i in range(running_gp_param_controls.shape[-1]):
        learing_figure = plt.figure(figsize=(7, 6))
        ax = learing_figure.add_subplot(111)
        xs = list(range(1, running_linear_error.shape[1] + 1))
        plt.errorbar(xs, running_gp_param_controls[idxs[0], idxs[1], :, i], yerr=running_gp_param_controls_confidence[idxs[0], idxs[1], :, i], ms=20, mew=5, fmt='x', linewidth=3, c='b', solid_capstyle='projecting', capsize=8)
        ax.set_ylabel(fig_ylabels[i], fontsize=28)
        ax.set_xlabel("play_style", fontsize=28)
        plt.xticks(xs, ["${}$".format(label) for label in play_styles], rotation=90)  # Set text labels.
        ax.yaxis.set_tick_params(labelsize=24)
        ax.xaxis.set_tick_params(labelsize=24)
        plt.tight_layout()
        if i == 1:
            plt.gcf().subplots_adjust(left=0.20)
        plt.show()
