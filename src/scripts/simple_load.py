"""  simple function to show how to load piano and robot control data"". """

import pandas as pd
import json
from simple_io import *
np.random.seed(123)

REFERENCE = "Human"   # MIDI/Human

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

print("\nThese are tje first three robot control inputs during grid-search\n")
for i in range(3):
    print("Robot key-press itearation number {} - Control: Rx={}, f1={}, f2={}, t1={}, t2={}".format(
        i,
        training_action[i, 0],
        training_action[i, 1],
        training_action[i, 2],
        training_action[i, 3],
        training_action[i, 4]
    ))

print("\nThese are the first three piano corresponding MIDI outputs during grid_search\n")
for i in range(3):
    print("Robot key-press itearation number {} - Control: on_velocity={}, off_velocity={}, wait_time={}".format(
        i, training_audio[i, 0], training_audio[i, 1], training_audio[i, 2], training_audio[i, 3]
    ))

print("\nThese are the first three robot control in the data during testing\n")
for i in range(3):
    print("Robot key-press itearation number {} - Control: Rx={}, f1={}, f2={}, t1={}, t2={}".format(
        i,
        testing_action[i, 0],
        testing_action[i, 1],
        testing_action[i, 2],
        testing_action[i, 3],
        testing_action[i, 4]
    ))

print("\nThese are the first three piano corresponding MIDI outputs during testing\n")
for i in range(3):
    print("Robot key-press itearation number {} - Control: on_velocity={}, off_velocity={}, wait_time={}".format(
        i, testing_audio[i, 0], testing_audio[i, 1], testing_audio[i, 2], testing_audio[i, 3]
    ))