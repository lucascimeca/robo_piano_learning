import threading
import json
import sys
import os
import mido
import datetime
import time
import pandas as pd
import msvcrt
import gpflow
import tensorflow as tf
import copy
import math

from data_operators import *
from rtde.UR5_protocol import UR5
from visualization import *
from data_operators import MIDI_writer
from mido import MidiFile

from gpflow.utilities import print_summary
from gpflow.optimizers import NaturalGradient
from gpflow.optimizers.natgrad import XiSqrtMeanVar
from sklearn.linear_model import LinearRegression

print("Random number with seed 30")
np.random.seed(123)

FRAME_RATE = 0
TIME_LEFT = 10000000
PRESS_DEPTH = 0.032 #0.063  #  0.02
KEY_WIDTH = 0.0236  # 0.0236
TICKS_PER_BEAT = 480
TEMPO = 500000

PIANO_OUTPUT_PARAMETERS = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
ROBOT_CONTROL_PARAMETERS = ["rx", "f1", "f2", "t1", "t2"]


class PianoLearning(object):
    """PianoLearning, when it runs it synchronises events for all subscribed observes to take action,
       variant of MVC pattern

    Attributes:
        interval (double): forces the frame rate of the experiment to an arbitrary frame rate.
        verbose: set to True if you want to the Piano learning to give you frame-rate updates
    """

    def __init__(self, interval, verbose=False):
        print("Initializing Piano Experiment...")
        self.interval = interval
        self._observers = set()
        self.notified = False
        self.t0 = time.clock()
        self.current_time = time.clock()
        self.verbose = verbose
        print("Piano Experiment initialized.")

    def subscribe(self, observer):
        self._observers.add(observer)
        print("Subscribed {} to Piano Experiment.".format(observer.name))

    def unsubscribe(self, observer):
        self._observers.discard(observer)
        print("Unsubscribed {} to Piano Experiment.".format(observer.name))

    def run(self, steps):
        print("Running Piano Experiment updates!")
        self.t0 = time.clock()
        self.current_time = time.clock()

        # notify observers of how many steps to expect
        [observer.init_data(steps) for observer in self._observers]

        notifications = [False, False]
        current_time = time.clock() - self.t0

        idx = 0
        if self.interval > 0:
            idx = np.floor(current_time / self.interval).astype(np.int32)  # get interval index from time
        else:
            idx += 1
        while True:
            prev_time = current_time
            current_time = time.clock() - self.t0

            prev_idx = idx
            if self.interval > 0:
                idx = np.floor(current_time / self.interval).astype(np.int32)  # get interval index from time
            else:
                idx += 1
            if prev_idx != idx:
                notifications[np.mod(prev_idx, 2)] = False

            if not notifications[np.mod(idx, 2)]:
                # notify observers of step
                nxt_state = np.array([observer.update(current_time) for observer in self._observers])
                if any(nxt_state == False):
                    print("At least some observers do not have any more data!")
                    break
                notifications[np.mod(idx, 2)] = True
                global FRAME_RATE
                FRAME_RATE = (FRAME_RATE + 1 / (current_time - prev_time)) / 2
                # if self.verbose:
                    # print("Frame rate: {}Hz".format(1/(current_time-prev_time)))
                    # sys.stdout.write("\rFrame rate: {}Hz".format(1/(current_time-prev_time)))
                    # sys.stdout.flush()
            # print("Run complete!".format(steps))
        self.end_run()

    def end_run(self):
        observers = self._observers.copy()
        print("Un-subscribing observers")
        [self.unsubscribe(observer) for observer in observers]
        print("Releasing...")
        [observer.release() for observer in observers]
        print("End")


class ExperimentObserver(threading.Thread):
    """Observer simulating a general observer for an experiment

    Attributes:
        name (str): name of the object
        data_folder(str): folder where to dump the experiment data in the future.
        pos_arg (str): positional argument for visualization, to be used later
        from_saved (bool): whether the observer should be running experiments from saved data, to be implemented later
        save (bool): whether the observed should save the data it receives
    """

    data = None

    def __init__(self, name, data_folder='./../data/', pos_arg='left', from_saved=False, save=True):
        super().__init__()
        self.name = name
        self.data_folder = data_folder
        self.from_saved = from_saved
        self.save = save
        self.pos_arg = pos_arg  # for viewer
        self.data_buffer = []
        self.nxt_buffer = []
        # connect a viewer to this camera

    # function called at start()
    def run(self):
        pass

    def init_data(self, steps):
        raise NotImplementedError

    # function that should be called at the end
    def release(self):
        self.join()

    def get_snapshot(self):
        raise NotImplementedError

    def update(self, time):
        raise NotImplementedError


class RobotPlayer(ExperimentObserver):
    brain = None
    ears = None

    current_midi_messages = []
    action_msg_buffer = [(0, 0), (0, 0)]

    def __init__(self, name="Robo-player", robot_ip="169.254.71.113", robot_parameters=None, learning=True,
                 verbose=False, resume_previous_experiment=True, melody_replay=False, save=True,
                 test_frequency=30, sort_actions=False, action_repeat_number=10, testing_params=None,
                 gp_search=False, gp_plots=False, *args, **kwargs):
        print("Initializing {} Thread...".format(name.upper()))
        super().__init__(name=name, *args, **kwargs)

        self.verbose = verbose
        self.melody_replay = melody_replay
        self.resume_previous_experiment = resume_previous_experiment

        # ---------------------- DATA ------------------------

        # the code below
        base_name = "piano_learning_experiment"

        # the code below creates an experiment folder with increasing number identifiers
        num = 0
        exp_folder = "{}{}_{}\\".format(self.data_folder, base_name, num)
        prev_folder = exp_folder
        while folder_exists(exp_folder):
            prev_folder = exp_folder
            exp_folder = "{}{}_{}\\".format(self.data_folder, base_name, num)
            num += 1
        if self.resume_previous_experiment:
            if folder_exists(prev_folder):
                self.data_folder = prev_folder
            else:
                self.resume_previous_experiment = False
                folder_create(exp_folder)
        else:
            self.data_folder = exp_folder
            folder_create(exp_folder)

        self.results_folder = self.data_folder + "results/"
        self.robot_parameters = robot_parameters

        # ---------------------- ROBOT BODY ----------------------
        # Initialize robot and go to initial palpation position
        self.touch_joint_start = np.deg2rad([137.20, -47.63, 78.12, -119.34, -91.54, -130.76])
        # self.safe_joint_start = np.deg2rad([94.32, -94.45, 154.66, -253.94, -92.53, -176.99])
        self.safe_joint_start = np.deg2rad([65.83, -92.92, 127.54, -126.56, -91.30, -202.30])
        self.robot = UR5(robot_ip=robot_ip)
        _, self.state = self.get_snapshot()
        print('Waiting for robot to reach start position...')
        self.robot_homing()
        self.initial_cart = copy.deepcopy(self.robot.get_state(("actual_TCP_pose")))
        self.current_cart = self.robot.get_state("actual_TCP_pose")
        print("Start position reached!")
        self.current_time = time.clock()
        self.robot_stop = False
        self.force_limit_procedure = False
        self.force_reset_initiated = False
        self.robot_connection_lost = False
        self.experiment_ready = False
        self.calibrated = False
        self.resume_action = False
        # ---------------------- BRAIN ------------------------
        self.brain = Brain(
            body=self,
            learning=learning,
            testing_params=None, # optimized test parameters

            test_frequency=test_frequency, # testing during training
            save=save,
            verbose=self.verbose,
            sort_actions=sort_actions,
            action_repeat_number=action_repeat_number,
            gp_search=gp_search,
            gp_plots=gp_plots
        )

        # ---------------------- EARS --------------------------
        self.ears = RobotEars(body=self)

        # -------------------- FINALIZE INITIALIZATION -----------
        self._update_robot_state()
        self.brain.initialize_learning()  # initalize melody for robot learning
        if self.resume_previous_experiment:
            self.calibrated = self.brain._load_calibration_details()
            self.resume_action = self.brain._load_experiment_data()
        else:
            self.calibrated = False
            self.resume_action = False
        self._reset_learning()

    # function called at start()
    def init_data(self, steps):
        # connect a viewer to this robot player
        self.data_viewer = PianoPlayerViewer(window_name=self.name, data_folder=self.data_folder, observer=self)

    def _reset_learning(self):
        self.velocities = None
        self.writer = None
        self.force_limit_procedure = False
        self.force_reset_initiated = False
        self.robot_connection_lost = False

    def get_snapshot(self):
        state = self.robot.get_state()
        return True, state

    def stop_robot(self, acc=5.):
        # stop robot
        self.robot.joint_stop(acc=acc)
        self.robot_stop = True

    def robot_homing(self):
        if not self._experiment_ready():
            # move robot up before returning to start position
            self.cart_end = self.state["actual_TCP_pose"]
            self.robot.joint_go_to(self.safe_joint_start, acc=.5, vel=1)
            while not self.robot.reached_point(point=self.safe_joint_start, mode='joint'):
                pass

            self.robot.joint_go_to(self.touch_joint_start, acc=.5, vel=1)
            while not self._experiment_ready():
                pass
        self.experiment_ready = True
        return True

    def _experiment_ready(self):
        return self.robot.reached_point(point=self.touch_joint_start, mode='joint')

    def _update_robot_state(self, time=0):
        # update internal state of the robot
        robot_state, self.state = self.get_snapshot()
        self.current_time = time
        self.is_moving = self.robot.is_moving(state=self.state)
        if self.state is None or self.is_moving is None:
            self.robot_connection_lost = True
        elif np.any(np.array(self.state['actual_TCP_force']) > 130):
            self.force_limit_procedure = True
        # if self.brain.save=True, then the ears will also save the message to a file every time you get it
        self.current_midi_messages = self.ears.get_msg()

        if len(self.current_midi_messages) > 0:
            for msg_tuple in self.current_midi_messages:
                msg = msg_tuple[0]
                if msg.type == "note_on":
                    self.action_msg_buffer[0] = msg_tuple
                elif msg.type == "note_off":
                    self.action_msg_buffer[1] = msg_tuple
        self.robot.kick_watchdog()
        return robot_state is not None

    def update(self, experiment_time):
        # update sensor and robot state
        nxt_state = self._update_robot_state(experiment_time)
        if self.verbose:
            self.data_viewer.show_frame(experiment_time, frame_rate=FRAME_RATE)

        # At the highest priority level the robot should stop if the stop variable be set to true. Very important.
        if not self.robot_stop:
            if not self.force_limit_procedure or self.robot_connection_lost:
                # EXECUTE CALIBRATION
                self.calibrated = self.brain.calibrate()

                # start experiment once calibration is over
                if self.calibrated:
                    if not self.experiment_ready:
                        print("Pausing experiments before learning... Robot going to start position")
                        self.robot_homing()
                        self.brain._reset_key_press()
                    else:
                        learning, score = self.brain.execute_action()

                        # robot stops
                        if not learning:
                            print("stopped learning, stopping robot")
                            self.brain._dump_experiment_data()
                            return False
            else:
                # ERROR STATE: try to reconnect if the robot is in an error state
                # self.robot._enstablish_connection()
                self.brain.discard()
                self._reset_learning()

        else:
            # STOP ROBOT
            if self.verbose:
                print("Stopping the robot!")
            self.stop_robot()
            if not self.is_moving:
                self.robot_stop = False

        # Advice the experiment object to end the experiment after 20 sec
        #        if self.current_time > 60:
        #            return False

        return nxt_state

    def release(self):
        # ----- return to initial position and end experiment -----
        self.brain.discard()
        if self.verbose:
            print("{} releasing files and objects...".format(self.name.upper()))
            print("Stopping robot...")
        self.robot.joint_stop(acc=5.)
        while self.robot.is_moving():
            pass
        if self.verbose:
            print("Going home!")

        self.robot_homing()
        self.ears.release()

        print("All done.")
        # ----- disconnect -----
        self.robot.disconnect()
        pass


class Brain(object):
    piano_keys = {}  # variable holding the location of the piano keys
    melody = None

    reached_new_key_position = False
    key_press_started = False
    key_press_start_time = None
    key_press_duration = None
    target_pose = None
    pose_diff = None
    key_press_start_pose = [0, 0, 0, 0, 0, 0]
    calibration_key_number = -1
    note_optimised = True
    action_executing = False
    pose_ls = []

    note_iter = -1

    input_parameters = pd.Series({
        "id": np.nan,
        "note": np.nan,
        "rx": np.nan,
        "f1": np.nan,
        "f2": np.nan,
        "t1": np.nan,
        "t2": np.nan,
    })
    last_input_parameters = copy.deepcopy(input_parameters)
    calibrate_parameters = pd.Series({
        "rx": 0,  # x rotation
        "f1": 3.0,  # downward frequency (1/s)
        "f2": 3.0,  # upward frequency (1/s)
        "t1": 0.2,  # wait time
        "t2": 0.1,  # hold time
        "num_of_keys": 52
    })
    try_melody_parameters = pd.Series({
        "rx": 0,  # x rotation
        # "ry": 0,  # y rotation
        "f1": 8.0,  # downward frequency (1/s)
        "f2": 5.,  # upward frequency (1/s)
        "t1": 2.5,  # wait time
        "t2": 1.5  # hold time
    })
    output_parameters = None
    all_input_parameters = pd.DataFrame()
    all_output_parameters = pd.DataFrame()
    all_input_parameters_predicted = pd.DataFrame()
    all_output_parameters_predicted = pd.DataFrame()
    predicted_parameters = pd.DataFrame()
    on_note = None
    off_note = None
    note_start_time = None
    note_end_time = None
    previous_off_time = 0
    learning_iteration = None
    perform_predictions = False
    play_style_list = ["normal","tenuto","staccatissimo","staccato","ff","f","mp","p","pp",
                       "pppp"]
    play_style = None

    def __init__(self, body=None, learning=False, save=False, verbose=False, test_frequency=30,
                 sort_actions=False, action_repeat_number=10, testing_params=None,
                 gp_search=False, gp_plots=False, *args, **kwargs):
        self.learning = learning
        self.test_frequency = test_frequency
        self.save = save
        self.verbose = verbose
        if body is None:
            raise ValueError("The brain need a body to be able to get tactile feedback!!")
        self.body = body
        self.GP_search = gp_search
        self.gp_plots = gp_plots

        # create gp model to model music to control (inverse) and control to music (forward)
        if self.learning:
            if gp_search:
                self.gp_model = GPModel(
                    params=self.body.robot_parameters,
                    verbose=True)

            if self.body.robot_parameters is not None:
                if not gp_search:
                    self.action_generator = ActionGenerator(
                        params=self.body.robot_parameters,
                        grid_action_number=self.body.robot_parameters['grid_action_number'],
                        sort_actions=sort_actions, action_repeat_number=action_repeat_number,
                        body=self.body
                    )
                else:
                    self.action_generator = GPActionGenerator(
                        self.gp_model,
                        params=self.body.robot_parameters,
                        precision=self.body.robot_parameters['GP_grid_action_number'],
                        melody_replay=self.body.melody_replay,
                        body=self.body,
                    )
            else:
                raise AttributeError("ERROR: Please provide exploratory robot parameters, or the robot-player "
                                     "will not know what to do.")
        else:
            self.action_generator = TestingActionGenerator(
                params=self.body.robot_parameters,
                testing_params=testing_params,
                melody_replay=self.body.melody_replay,
                body=self.body,
            )

    def initialize_learning(self):
        self.melody = self.get_melody()
        self.current_pose = self.body.current_cart
        self.note_start_time = time.clock()
        self.learning_iteration = 0
        self._reset_key_press()

    def _reset_key_press(self, save_override=False):
        self.body.stop_robot(acc=.8)
        # stopping robot (blocking--bad)
        # while self.body.robot.is_moving():
        #     self.body.stop_robot(acc=.1)

        # iterate key number for calibration
        self.calibration_key_number += 1
        if not self.action_generator.try_melody and not self.perform_predictions:
            self.learning_iteration += 1

        if self.body.calibrated:
            # -- update gaussian process model
            if self.GP_search and (self.output_parameters is not None and self.input_parameters is not None) and \
                    not self.perform_predictions:
                    data_fit = self.gp_model.update_fit_data(
                        X=np.array(self.all_output_parameters[PIANO_OUTPUT_PARAMETERS]),
                        Y=np.array(self.all_input_parameters[ROBOT_CONTROL_PARAMETERS]),
                        inverse=False
                    )

            if self.test_frequency != 0:
                if self.learning_iteration % self.test_frequency == 0 and self.learning_iteration != 0:
                    self.perform_predictions = True
                    if self.verbose:
                        print("ITERATION {}: (RE)TESTING MELODY...".format(self.learning_iteration))
                    action = self.action_generator.get_inference_action(X=self.all_output_parameters,
                                                                        Y=self.all_input_parameters,
                                                                        note_to_try=self.digital_parameters)
                    self.play_style = self.play_style_list[action[0] % 10]

            if not self.perform_predictions:
                # -- send updated process to action generator, to generate new action
                action = self.action_generator.get_action()
                if self.GP_search and self.gp_plots:
                    self.action_generator.view_action_space()
                if not self.learning:
                    print("STYLE PERFORMED: {}".format(self.action_generator.action_idx_to_label(action[0])))

            # DEBUG
            # print('predicting? ', self.perform_predictions, ', action: ', action)

            if action is not None:
                self.last_input_parameters = copy.deepcopy(self.input_parameters)
                if self.action_generator.note_idx < len(self.melody):
                    self.input_parameters["note"] = self.melody.loc[self.action_generator.note_idx, "note"]
                else:
                    self.input_parameters["note"] = None
                self.input_parameters["id"] = self.learning_iteration
                self.input_parameters['rx'], self.input_parameters['f1'], \
                self.input_parameters['f2'], self.input_parameters['t1'], self.input_parameters['t2'] = action[1]
            elif action is None and self.action_generator.try_melody:
                self.action_generator.try_melody = False
                return True
            elif action is None and not self.action_generator.try_melody:
                return False

            # if save is true then dump current messages onto a file
            if self.body.ears is not None:
                if not save_override and self.save and len(self.body.current_midi_messages) > 0:
                    self.body.ears.save()
                self.body.ears.reset_note()

        self.key_press_start_pose = copy.deepcopy(self.body.state["actual_TCP_pose"])
        self.key_press_start_pose[2] = self.body.initial_cart[2]
        if self.key_press_start_pose[3] < 0:
            self.key_press_start_pose[3] += 2*np.pi
        self.target_pose = None
        self.current_midi_messages = []
        self.reached_new_key_position = False
        self.key_press_started = False
        self.key_press_start_time = None
        self.key_press_duration = None

        return True

    # function to dump the details of the experiments into a file, to be possibly loaded later for future use
    def _dump_calibration_details(self):
        # todo: extend this function to dump more experiment details
        calibration_info = {
            "piano_keys": self.piano_keys
        }
        save_dict_to_file(calibration_info,
                          path=self.body.data_folder,
                          filename="calibration_info")

    def _dump_experiment_data(self):
        self.melody.to_json(os.path.join(self.body.data_folder, "demo_parameters.json"))
        self.all_input_parameters.to_json(os.path.join(self.body.data_folder, "all_input_parameters.json"))
        self.all_output_parameters.to_json(os.path.join(self.body.data_folder, "all_output_parameters.json"))
        self.all_input_parameters_predicted.to_json(os.path.join(self.body.data_folder, "all_input_parameters_predicted.json"))
        self.all_output_parameters_predicted.to_json(os.path.join(self.body.data_folder, "all_output_parameters_predicted.json"))

    def _dump_action_data(self):
        df_actions = pd.DataFrame(self.action_generator.actions)
        df_actions.to_json(os.path.join(self.body.data_folder, "remaining_actions.json"))

    # save current stacked motion, and clear
    def save(self):
        if self.learning and self.body.writer is not None:
            # todo: perhaps record midi file content and save to file? -- to do later
            pass

    def discard(self, keep_motion=True):
        if self.learning and self.body.writer is not None:
            # todo: discard object to be saved if something happened
            pass

    def _load_calibration_details(self):
        calibration_filename = None
        try:
            json_files = list(np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith(".json")]))
            if len(json_files) > 0:
                for json_file in json_files:
                    if "experiment_info" in json_file:
                        calibration_filename = os.path.join(self.body.data_folder, json_file)
                    elif "calibration_info" in json_file:
                        calibration_filename = os.path.join(self.body.data_folder, json_file)

                if calibration_filename is None:
                    print("no calibration file found")
                    return False

                else:
                    with open(calibration_filename, 'r') as f:
                        calibration_info = json.load(f)
                    self.piano_keys = calibration_info["piano_keys"]
                    return True
        except Exception as e:
            print(e)
        return False

    def _load_experiment_data(self):
        try:
            json_files = list(np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith(".json")]))
            if len(json_files) > 0:
                for json_file in json_files:
                    if "all_input_parameters_predicted" in json_file:
                        json_filename = os.path.join(self.body.data_folder, json_file)
                        with open(json_filename) as inputp_param_file:
                            self.all_input_parameters_predicted = pd.DataFrame(json.load(inputp_param_file))
                    elif "all_output_parameters_predicted" in json_file:
                        json_filename = os.path.join(self.body.data_folder, json_file)
                        with open(json_filename) as outputp_param_file:
                            self.all_output_parameters_predicted = pd.DataFrame(json.load(outputp_param_file))
                    elif "all_input_parameters" in json_file:
                        json_filename = os.path.join(self.body.data_folder, json_file)
                        with open(json_filename) as input_param_file:
                            self.all_input_parameters = pd.DataFrame(json.load(input_param_file))
                    elif "all_output_parameters" in json_file:
                        json_filename = os.path.join(self.body.data_folder, json_file)
                        with open(json_filename) as output_param_file:
                            self.all_output_parameters = pd.DataFrame(json.load(output_param_file))
                    elif "digital_parameters" in json_file:
                        json_filename = os.path.join(self.body.data_folder, json_file)
                        with open(json_filename) as digital_param_file:
                            self.digital_parameters = pd.DataFrame(json.load(digital_param_file))
                        self.digital_parameters = pd.concat([self.digital_parameters]*3, ignore_index=True)
                    # elif "predicted_parameters" in json_file:
                    #     json_filename = os.path.join(self.body.data_folder, json_file)
                    #     with open(json_filename) as predict_param_file:
                    #         self.predicted_parameters = pd.DataFrame(json.load(predict_param_file))
                if self.all_input_parameters.empty or self.all_output_parameters.empty:
                    print("Input parameters data is empty: ", self.all_input_parameters.empty)
                    print("Output parameters data is empty: ", self.all_output_parameters.empty)
                    return False
                else:
                    return True
        except Exception as e:
            print(e)
        return False

    def _load_action_data(self):
        action_filename = None
        json_files = list(np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith(".json")]))
        if len(json_files) > 0:
            for json_file in json_files:
                if "remaining_actions" in json_file:
                    action_filename = os.path.join(self.body.data_folder, json_file)

        if action_filename is None:
            print("No action file saved")
            return False
        else:
            with open(action_filename) as action_file:
                remaining_actions = pd.DataFrame(json.load(action_file))
                self.action_generator.actions = remaining_actions.values.tolist()
            return True

    def press_key(self, parameters, target_pose):
        key_press_current_time = self.body.current_time - self.key_press_start_time
        if key_press_current_time < self.key_press_duration:
            if key_press_current_time < parameters["t1"]:
                # change key
                pose_to_reach = copy.deepcopy(self.key_press_start_pose)
                if target_pose is None:
                    # calibration
                    x = 0.5 * KEY_WIDTH * (np.cos(np.pi / parameters["t1"] * key_press_current_time) - 1)
                    pose_to_reach[0] += x
                else:
                    change = - 0.5 * self.pose_diff * (np.cos(np.pi / parameters["t1"] * key_press_current_time) - 1)
                    # change = self.pose_diff * key_press_current_time / parameters["t1"]
                    pose_to_reach[0] += change[0]  # x
                    pose_to_reach[3] += change[3]  # rx
                    # pose_to_reach[4] += change[4]  # ry
                    # print("change: ", pose_to_reach)

            elif key_press_current_time < parameters["t1"] + 1 / parameters["f1"]:
                # press key
                if target_pose is None:
                    # calibration
                    pose_to_reach = copy.deepcopy(self.body.initial_cart)
                    pose_to_reach[0] = copy.deepcopy(self.key_press_start_pose[0]) - KEY_WIDTH
                else:
                    pose_to_reach = copy.deepcopy(target_pose)

                z = 0.5 * PRESS_DEPTH * (
                            np.cos(np.pi * parameters["f1"] * (key_press_current_time - parameters["t1"])) - 1)
                pose_to_reach[2] += z
                # print("press: ", pose_to_reach)

            elif key_press_current_time < parameters["t1"] + 1 / parameters["f1"] + parameters["t2"]:
                # hold key
                if target_pose is None:
                    # calibration
                    pose_to_reach = copy.deepcopy(self.body.initial_cart)
                    pose_to_reach[0] = copy.deepcopy(self.key_press_start_pose[0]) - KEY_WIDTH
                else:
                    pose_to_reach = copy.deepcopy(target_pose)

                pose_to_reach[2] -= PRESS_DEPTH
                # print("hold: ", pose_to_reach)
            else:
                # release key
                if target_pose is None:
                    # calibration
                    pose_to_reach = copy.deepcopy(self.body.initial_cart)
                    pose_to_reach[0] = copy.deepcopy(self.key_press_start_pose[0]) - KEY_WIDTH
                else:
                    pose_to_reach = copy.deepcopy(target_pose)

                z = -0.5 * PRESS_DEPTH * (np.cos(np.pi * parameters["f2"] * (key_press_current_time - parameters["t1"] -
                                                                             1 / parameters["f1"] - parameters[
                                                                                 "t2"])) + 1)
                pose_to_reach[2] += z
                # print("release: ", pose_to_reach)
            # print(FRAME_RATE)
            framerate = np.min([FRAME_RATE, 125])
            self.body.robot.servoj_cart(pose_to_reach, t=(1 / framerate), lookhead_time=.15, gain=100)
            return True
        return False

    def calibrate(self):
        if not self.body.calibrated:
            if self.calibration_key_number < self.calibrate_parameters["num_of_keys"]:
                if not self.key_press_started:
                    #  -- robot at new position, establish parameters for action --
                    self.key_press_duration = self.calibrate_parameters["t1"] + 1 / self.calibrate_parameters["f1"] + \
                                              self.calibrate_parameters["t2"] + 1 / self.calibrate_parameters["f2"]
                    self.key_press_start_time = self.body.current_time
                    self.key_press_started = True
                else:
                    #  -- robot is pressing the key --
                    key_pressing = self.press_key(self.calibrate_parameters, None)

                    # -- if the key pressing is over change to new key --
                    if not key_pressing:
                        self._reset_key_press(save_override=True)

                    if len(self.body.current_midi_messages) > 0:
                        for msg_tuple in self.body.current_midi_messages:
                            msg = msg_tuple[0]
                            if msg.type == "note_on":
                                if str(msg.note) not in self.piano_keys:
                                    self.piano_keys[str(msg.note)] = self.body.initial_cart[0] - KEY_WIDTH * (
                                            self.calibration_key_number + 1)
                return False
            else:
                self._dump_calibration_details()
        return True

    def get_melody(self):
        melody = pd.DataFrame(
            columns=["note", "on_velocity", "hold_time", "off_velocity", "wait_time"]
        )
        midi_files = list(np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith("melody.mid")]))

        while len(midi_files) == 0:
            key = ''
            while key not in ['y', 'Y', 'n', 'N']:
                key = input("No melody was found in the data folder. Would you like to generate a new melody? [y,n]\n")
                if key not in ['y', 'Y', 'n', 'N']:
                    print("Invalid input, please enter yes (y) or no (n).")

            if key in ['y', 'Y']:
                self.body.ears.generate_melody()
            else:
                input("Please drop a 'melody.mid' file within the current experiment folder ({}), "
                      "then press any key to continue...".format(self.body.data_folder))
            midi_files = list(
                np.sort([file for file in os.listdir(self.body.data_folder) if file.endswith("melody.mid")]))

        print("Found melody file, loading melody data for the robot player to perform experiments!")

        midi_filename = self.body.data_folder + midi_files.pop()
        demo = MidiFile(midi_filename)
        for i, track in enumerate(demo.tracks):
            for msg in track:
                if msg.type == "note_on":
                    on_note = msg.note
                    on_velocity = msg.velocity
                    wait_time = mido.tick2second(msg.time, TICKS_PER_BEAT, TEMPO)
                elif msg.type == "note_off":
                    off_note = msg.note
                    off_velocity = msg.velocity
                    hold_time = mido.tick2second(msg.time, TICKS_PER_BEAT, TEMPO)
                    if off_note == on_note:
                        note_params = {"note": off_note,
                                       "on_velocity": on_velocity,
                                       "hold_time": hold_time,
                                       "off_velocity": off_velocity}
                        melody = melody.append(note_params, ignore_index=True)
                    else:
                        print("Error: overlapping notes in demo")
                    if len(melody) > 1:
                        melody.at[len(melody) - 2, "wait_time"] = wait_time
        melody.at[len(melody) - 1, "wait_time"] = 0
        print("-----------Demo Melody-----------")
        print(melody)
        return melody

    def execute_action(self):
        continue_experiments = True
        score = None
        if self.body.resume_action:
            print("execute action from checkpoint")
            self._reset_key_press()
            self.action_generator.try_melody = True
        if not self.key_press_started:
            if not math.isnan(self.input_parameters["note"]):
                self.target_pose = copy.deepcopy(self.body.initial_cart)
                self.target_pose[0] = self.piano_keys[str(int(self.input_parameters["note"]))]

                self.target_pose[3] += np.deg2rad(self.input_parameters["rx"])
                # self.target_pose[4] += np.deg2rad(self.input_parameters["ry"])

                self.pose_diff = np.array(self.target_pose) - np.array(self.key_press_start_pose)

                self.key_press_duration = self.input_parameters["t1"] + 1 / self.input_parameters["f1"] + \
                                          self.input_parameters["t2"] + 1 / self.input_parameters["f2"]
                self.key_press_start_time = self.body.current_time
                self.key_press_started = True
            else:
                self._reset_key_press()
        else:

            # -- robot is pressing key --
            key_pressing = self.press_key(self.input_parameters, self.target_pose)

            if not key_pressing:
                # reset variables and get new action
                continue_experiments = self._reset_key_press(save_override=False)

            # -- record midi messages --
            if not self.action_generator.try_melody and \
                    not np.isclose(self.previous_off_time, self.body.action_msg_buffer[1][1], rtol=1e-15) and \
                    self.body.action_msg_buffer[1][1] - self.body.action_msg_buffer[0][1] > 0:

                self.previous_off_time = self.body.action_msg_buffer[1][1]

                # Get message from robot action
                for msg_tuple in self.body.action_msg_buffer:
                    msg = msg_tuple[0]
                    if msg.type == "note_on":
                        self.on_note = msg.note
                        on_velocity = msg.velocity
                        self.note_start_time = msg_tuple[1]
                        if self.note_end_time is not None:
                            wait_time = self.note_start_time - self.note_end_time
                        else:
                            wait_time = 0

                    elif msg.type == "note_off":
                        self.off_note = msg.note
                        off_velocity = msg.velocity
                        self.note_end_time = msg_tuple[1]
                        if self.note_start_time is not None:
                            hold_time = self.note_end_time - self.note_start_time
                        else:
                            hold_time = 0

                if self.off_note is None:
                    print("no off note")
                elif self.off_note != self.on_note:
                    print("error in midi, check that only one key was pressed")
                else:
                    self.note_iter += 1
                    # Evaluate robot action
                    self.output_parameters = pd.Series({"id": self.note_iter,
                                                        "note": self.on_note,
                                                        "on_velocity": on_velocity,
                                                        "hold_time": hold_time,
                                                        "off_velocity": off_velocity,
                                                        "wait_time": wait_time})


                    # # Evaluation
                    print("LEARNING ITEARATION {}".format(self.learning_iteration))
                    print("--------------------------------")
                    print(self.input_parameters)
                    print(self.output_parameters)
                    print("--------------------------------")


                    # Save experiment data
                    if self.perform_predictions:
                        self.input_parameters["learning_set"] = self.learning_iteration // self.test_frequency
                        self.input_parameters["play_style"] = self.play_style
                        self.output_parameters["learning_set"] = self.learning_iteration // self.test_frequency
                        self.output_parameters["play_style"] = self.play_style
                        self.all_input_parameters_predicted = self.all_input_parameters_predicted.append(
                            self.input_parameters, ignore_index=True)
                        self.all_output_parameters_predicted = self.all_output_parameters_predicted.append(
                            self.output_parameters, ignore_index=True)

                    else:
                        self.all_input_parameters = self.all_input_parameters.append(self.last_input_parameters,
                                                                                     ignore_index=True)
                        self.all_output_parameters = self.all_output_parameters.append(self.output_parameters,
                                                                                       ignore_index=True)

                    self._dump_experiment_data()

                # reset perform_predictions only after saving to file
                if len(self.action_generator.predicted_actions) == 0:
                    self.perform_predictions = False
                    self.action_generator.linear_fit_generated = False

                # remove saved try_melody data
                # if not self.all_output_parameters.empty and \
                #         self.all_output_parameters.loc[0, "note"] != self.all_input_parameters.loc[0, "note"]:
                #     print("removing")
                #     self.all_output_parameters = self.all_output_parameters.drop(0)
                #     self.all_input_parameters = self.all_input_parameters.drop(0)
                #     self.all_output_parameters.reset_index(drop=True, inplace=True)
                #     self.all_input_parameters.reset_index(drop=True, inplace=True)


        return continue_experiments, score


class RobotEars(object):
    midi_writer = None
    start_time = None
    msgs = []

    def __init__(self, midi_input=None, ticks_per_beat=480, tempo=500000, body=None):
        # todo: wrong code if multiple pianos connected
        self.body = body
        if midi_input is None:
            inputs = mido.get_input_names()
            midi_input = [in_name for in_name in inputs if 'MIDI' in in_name][0]

        self.in_port = mido.open_input(
            midi_input,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo
        )
        self.reset_note()

    def reset_note(self):
        # (re)create midi file to dump data in
        if self.body.brain.save:
            self.msgs = []

    def get_msg(self):
        msgs = []
        for msg in self.in_port.iter_pending():
            msg_time = time.time()
            msgs += [(msg, msg_time)]
        self.msgs += msgs
        return msgs

    def save(self):
        if self.body is not None:
            if len(self.msgs) > 0:
                self.midi_writer = MIDI_writer(
                    ticks_per_beat=480,
                    tempo=500000,
                    start_time=time.time(),
                    folder=self.body.data_folder,
                    filename="{}-{:.3f}s.{}".format(self.body.brain.action_generator.note_idx, self.body.current_time,
                                                    "mid")
                )
                for msg, msg_time in self.msgs:
                    self.midi_writer.add_to_midi([(msg, msg_time)])
                self.midi_writer.write()
            return True
        else:
            raise AttributeError("ERROR: to be able to save midi data the Ears need a body!!")
        return False

    def generate_melody(self):
        """ Demonstration Save File script """
        print("Play Demo")

        melody_file = MIDI_writer(
            ticks_per_beat=480,
            tempo=500000,
            start_time=time.time(),
            folder=self.body.data_folder,
            filename="melody.mid"
        )

        print("Please play a melody. Press 'q' to stop and save the melody.")
        done = False
        self.msgs = []
        while not done:
            for msg in self.in_port.iter_pending():
                msg_time = time.time()
                self.msgs += [(msg, msg_time)]

            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key.decode() == 'q':
                    print("Saving the melody and moving on...".format(msvcrt.getch()))
                    done = True
        melody_file.add_to_midi(self.msgs)
        melody_file.write()

    def release(self):
        self.save()
        self.in_port.close()


# Generate action profile, by using recursive function - the parameters MUST be in the form:
# [on_velocity, off_velocity, hold_time]
# grid_action_number specifies the number of parameters to search for each variable (determines spacing)
class ActionGenerator(object):
    # todo: refactor try melody
    try_melody = True

    def __init__(self, params, grid_action_number=3,
                 sort_actions=False, action_repeat_number=10,
                 body=None):

        self.params = params
        self.params_min = params['action_params_min']
        self.params_max = params['action_params_max']
        self.all_actions = self.__get_action_profile(self.params_min, self.params_max,
                                                     grid_action_number=grid_action_number,
                                                     sort_actions=sort_actions,
                                                     action_repeat_number=action_repeat_number)
        self.last_action = None
        self.actions = copy.deepcopy(self.all_actions)
        self.body = body
        self.note_idx = -1
        self.audio_param_names = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
        self.control_param_names = ["rx", "f1", "f2", "t1", "t2"]
        self.linear_fit_generated = False
        self.predicted_actions = []

    def __generate_action(self, current_state, idx, min_vec, max_vec, grid_action_number):
        if min_vec.shape[0] <= idx:
            return current_state
        else:
            if min_vec[idx] != 0 or max_vec[idx] != 0:
                if (max_vec[idx] - min_vec[idx] != 0):
                    actions = np.arange(min_vec[idx], max_vec[idx],
                                        (max_vec[idx] - min_vec[idx]) / (grid_action_number + 1))[1:]
                else:
                    actions = np.array([min_vec[idx]])
                if current_state is None:
                    current_state = np.array([[0.0] * min_vec.shape[0]])

                state = current_state.copy()
                for i, act in enumerate(actions):
                    if i == 0:
                        current_state[:, idx] = act
                    else:
                        state_copy = state.copy()
                        state_copy[:, idx] = act
                        current_state = np.concatenate((current_state, state_copy), axis=0)
            return self.__generate_action(current_state, idx + 1, min_vec, max_vec, grid_action_number)

    def __get_action_profile(self, params_min, params_max, grid_action_number=3,
                             sort_actions=False, action_repeat_number=10):

        actions = self.__generate_action(None, 0, params_min, params_max, grid_action_number)
        action_profile = []
        for i in range(actions.shape[0]):
            action_profile += [(i, actions[i, :])]

        # to generate specific plots
        if sort_actions:
            param_ls = []
            for param_idx in range(len(action_profile[0][1])):
                param_ls.append(np.unique([action_profile[i][1][param_idx] for i in range(len(action_profile))]))

            sorted_actions = []
            # action_profile[action][1][param], param_ls[param][unique num]
            for i in range(len(action_profile)):
                # rx plot
                if action_profile[i][1][1] == param_ls[1][5] and action_profile[i][1][2] == param_ls[2][5] \
                        and action_profile[i][1][3] == param_ls[3][7] and action_profile[i][1][4] == param_ls[4][7]:
                    sorted_actions.append(action_profile[i])
                # f1 plot
                elif (action_profile[i][1][0] == param_ls[0][2] or action_profile[i][1][0] == param_ls[0][3])\
                        and action_profile[i][1][2] == param_ls[2][5] \
                        and action_profile[i][1][3] == param_ls[3][7] and action_profile[i][1][4] == param_ls[4][7]:
                    sorted_actions.append(action_profile[i])
                # f2 plot
                elif (action_profile[i][1][0] == param_ls[0][6] or action_profile[i][1][0] == param_ls[0][7])\
                        and action_profile[i][1][1] == param_ls[1][5] \
                        and action_profile[i][1][3] == param_ls[3][7] and action_profile[i][1][4] == param_ls[4][7]:
                    sorted_actions.append(action_profile[i])
            print(len(sorted_actions))
            sorted_actions *= action_repeat_number
            return sorted_actions
        else:
            return action_profile

    def get_action(self):
        # if resume exp, check through input_params json file and pop until no repeat
        if self.body.resume_action:
            self.actions = copy.deepcopy(self.all_actions)
            action_data_saved = self.body.brain._load_action_data()
            print("current action length: ", len(self.actions))

            # identify the last note recorded
            self.note_idx = self.body.brain.all_input_parameters.note.nunique()

            if not action_data_saved:
                last_parameters = self.body.brain.all_input_parameters[self.body.brain.all_input_parameters.note.isin(
                    [self.body.brain.all_input_parameters.iloc[-1]["note"]])]

                for id_parameter in last_parameters.index:
                    parameter = np.array(
                        last_parameters.loc[str(id_parameter), ["rx", "f1", "f2", "t1", "t2"]]).astype(float)
                    # if int(id_parameter) % 100 == 0 or int(id_parameter) > 2800:
                    #     print(id_parameter, len(last_parameters))
                    for id_action in range(len(self.actions)):
                        if np.isclose(parameter, self.actions[-id_action - 1][1], equal_nan=True).all():
                            self.actions.pop(-id_action - 1)
                            self.body.brain.learning_iteration += 1
                            break
                print("finished popping actions from checkpoint")
                # self.body.brain._dump_action_data()
            else:
                self.body.brain.learning_iteration = len(self.body.brain.all_input_parameters)

            self.try_melody = True
            self.note_idx = 0
            self.body.resume_action = False

        if self.try_melody:
            if self.note_idx < len(self.body.brain.melody):
                # if playing melody before training
                self.last_action = (0, np.array(self.body.brain.try_melody_parameters[["rx", "f1", "f2", "t1", "t2"]]))
                self.note_idx += 1

                if self.note_idx < len(self.body.brain.melody):
                    print("Try melody, note: ", self.note_idx)
                return self.last_action
            else:
                self.note_idx = 0
                self.try_melody = False
                print("Melody ended, start learning")

        if len(self.actions) <= 0:
            self.actions = copy.deepcopy(self.all_actions)
            self.note_idx += 1
            if self.note_idx >= len(self.body.brain.melody):
                return None

        self.last_action = self.actions.pop()

        return self.last_action

    # the methods peforms linear inference on the current data, and returns an action based on the linear fit
    def get_inference_action(self, X, Y, note_to_try):
        if not self.linear_fit_generated:
            regressors = [LinearRegression() for i in range(len(self.actions[0][1]))]
            predictions = []
            self.predicted_actions = []
            for i in range(len(regressors)):
                regressors[i].fit(X[self.audio_param_names],
                                  Y[self.control_param_names[i]])
                predictions.append(regressors[i].predict(note_to_try[self.audio_param_names]))
            predictions = pd.DataFrame(predictions).T
            predictions.columns = self.control_param_names

            # todo: neater way to limit the params? It is different from params_min and params_max
            # restrict prediction range
            predictions.loc[(predictions.rx < 0), 'rx'] = 0
            predictions.loc[(predictions.rx > 90), 'rx'] = 90
            predictions.loc[(predictions.f1 < 0), 'f1'] = 0.1
            predictions.loc[(predictions.f2 < 0), 'f1'] = 0.1
            predictions.loc[(predictions.f2 > 10), 'f2'] = 10.0
            predictions.loc[(predictions.f1 > 10), 'f1'] = 10.0
            predictions.loc[(predictions.t1 < 0.1), 't1'] = 0.1
            predictions.loc[(predictions.t2 < 0.1), 't2'] = 0.1

            for idx in range(len(predictions)):
                self.predicted_actions.append((idx, np.array(predictions.loc[idx, self.control_param_names])))

            self.linear_fit_generated = True
        return self.predicted_actions.pop(0)

    def last_action(self):
        return self.last_action


# Generate action profile, by using GP - the parameters MUST be in the form:
# [rx, f1, f2, t1, t2]
# grid_action_number specifies the number of parameters to search for each variable (determines spacing)
class GPActionGenerator(ActionGenerator):
    pos_arg = "right"

    try_melody = True
    precision = None
    mesh_mean = None
    mesh_var = None
    idx = None  # index of parameter to optimize, either given or found

    no_issued_actions = 0
    inputs = None

    def __init__(self, gp_model, params, body=None, precision=10, melody_replay=False):
        super().__init__(params, grid_action_number=1, body=body)
        self.name = "GPActionGenerator"
        self.precision = precision
        self.gp_model = gp_model
        self.note_idx = -1
        self.gp_action_viewer = GPActionViewer(window_name=self.name,
                                               observer=self)

    # idx is the index of the parameter to show the gaussian for
    def _highest_variance_points(self, robot_parameters=None, idx=None):
        res_params = (self.params_max + self.params_min)/2
        if self.gp_model.forward_models is not None:
            xs = []
            if robot_parameters is not None:
                for idx in self.gp_model.search_indexes_to_optimize:
                    xs += [np.linspace(self.params_min[idx],
                                       self.params_max[idx],
                                       self.precision).reshape(-1, 1)]
            # XS = np.squeeze(np.array(xs))
            # for i in range(len(XS)):
            #     XS[i] = [XS[i].flatten()]
            # self.inputs = np.array([*XS]).T
            # if len(self.inputs.shape) >= 3 and np.any(np.array(self.inputs.shape[1:-1]) == 1):
            self.inputs = np.squeeze(np.array(xs)).T

            if self.gp_model.number_of_y_params == 1:
                self.inputs = self.inputs.reshape(-1, 1)

            self.mesh_mean, self.mesh_var = self.gp_model.predict(X=self.inputs)
            max_idx = np.argmax(self.mesh_var, axis=0)
            for i in range(len(xs)):
                res_params[self.gp_model.search_indexes_to_optimize[i]] = self.inputs[max_idx[i], i]
        return res_params

    def view_action_space(self):
        self.gp_action_viewer.show_frame(0,
                                         gp_model=self.gp_model,
                                         inputs=self.inputs,
                                         precision=self.precision,
                                         params=self.params)
        # else:
        #     print("cumulative variance: {}".format(np.sum(self.mesh_var)))

    def get_action(self):
        # if resume exp, check through input_params json file and pop until no repeat
        if self.body.resume_action:
            self.actions = copy.deepcopy(self.all_actions)
            action_data_saved = self.body.brain._load_action_data()
            print("current action length: ", len(self.actions))

            # identify the last note recorded
            self.note_idx = self.body.brain.all_input_parameters.note.nunique()

            if not action_data_saved:
                last_parameters = self.body.brain.all_input_parameters[self.body.brain.all_input_parameters.note.isin(
                    [self.body.brain.all_input_parameters.iloc[-1]["note"]])]

                for id_parameter in last_parameters.index:
                    parameter = np.array(
                        last_parameters.loc[str(id_parameter), ["rx", "f1", "f2", "t1", "t2"]]).astype(float)
                    # if int(id_parameter) % 100 == 0 or int(id_parameter) > 2800:
                    #     print(id_parameter, len(last_parameters))
                    for id_action in range(len(self.actions)):
                        if np.isclose(parameter, self.actions[-id_action - 1][1], equal_nan=True).all():
                            self.actions.pop(-id_action - 1)
                            self.body.brain.learning_iteration += 1
                            break
                print("finished popping actions from checkpoint")
                self.body.brain.learning_iteration = self.body.brain.all_input_parameters.shape[0]
                # self.body.brain._dump_action_data()
            else:
                self.body.brain.learning_iteration = len(self.body.brain.all_input_parameters)

            self.try_melody = True
            self.note_idx = 0
            self.body.resume_action = False

        if self.try_melody:
            if self.note_idx < len(self.body.brain.melody):
                # if playing melody before training
                self.last_action = (0, np.array(self.body.brain.try_melody_parameters[["rx", "f1", "f2", "t1", "t2"]]))
                self.note_idx += 1

                if self.note_idx < len(self.body.brain.melody):
                    print("Try melody, note: ", self.note_idx)
                return self.last_action
            else:
                self.note_idx = 0
                self.try_melody = False
                print("Melody ended, start learning")

        if not self.try_melody and len(self.actions) <= 0:
            # self.note_idx += 1
            params = self._highest_variance_points(self.body.robot_parameters)
            self.last_action = 0, params
            print("-----============== getting action from gp model -----------------")
            self.no_issued_actions += 1
            return self.last_action
        else:
            self.last_action = self.actions.pop()

        return self.last_action

    def last_action(self):
        return self.last_action

    # cheryn todo: linear fit on X & Y, and predict based on note_to_try
    # the methods peforms GP fit on the current data, and returns an action based on the fit
    def get_inference_action(self, X, Y, note_to_try):
        if not self.linear_fit_generated:
            self.gp_model.update_fit_data(
                X=np.array(self.body.brain.all_output_parameters[PIANO_OUTPUT_PARAMETERS]),
                Y=np.array(self.body.brain.all_input_parameters[ROBOT_CONTROL_PARAMETERS]),
                inverse=True
            )
            mean, _ = self.gp_model.predict(
                X=np.array(note_to_try[np.array(PIANO_OUTPUT_PARAMETERS)[self.gp_model.output_indexes_to_model]]),
                inverse=True)
            predictions = pd.DataFrame(np.round(mean,2))
            predictions.columns = self.control_param_names

            # todo: neater way to limit the params? It is different from params_min and params_max
            # restrict prediction range
            predictions.loc[(predictions.rx < 0), 'rx'] = 0
            predictions.loc[(predictions.rx > 90), 'rx'] = 90
            predictions.loc[(predictions.f1 < 0.4), 'f1'] = 0.4
            predictions.loc[(predictions.f2 < 0.4), 'f2'] = 0.4
            predictions.loc[(predictions.f1 > 8.03), 'f1'] = 8.03
            predictions.loc[(predictions.f2 > 8.03), 'f2'] = 8.03
            predictions.loc[(predictions.t1 < 0.2), 't1'] = 0.2
            predictions.loc[(predictions.t2 < 0.2), 't2'] = 0.2
            predictions.loc[(predictions.t1 > 3), 't1'] = 3
            predictions.loc[(predictions.t2 > 3), 't2'] = 3

            for idx in range(len(predictions)):
                self.predicted_actions.append((idx, np.array(predictions.loc[idx, self.control_param_names])))

            self.linear_fit_generated = True
        return self.predicted_actions.pop(0)


class TestingActionGenerator(object):
    try_melody = True

    def __init__(self, params, testing_params, body=None):

        self.params = params
        self.params_min = params['action_params_min']
        self.params_max = params['action_params_max']
        self.all_actions, self.all_labels = self.__generate_action(testing_params)
        self.last_action = None
        self.actions = copy.deepcopy(self.all_actions)
        self.body = body
        self.note_idx = -1
        self.audio_param_names = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
        self.control_param_names = ["rx", "f1", "f2", "t1", "t2"]
        self.linear_fit_generated = False
        self.predicted_actions = []

    def __generate_action(self, testing_params):
        actions = []
        labels = {}
        for i, param in enumerate(testing_params.keys()):
            actions += [(i, np.array(testing_params[param]))]
            labels[i] = param
        return actions, labels

    def get_action(self):
        if self.try_melody:
            if self.note_idx < len(self.body.brain.melody):
                # if playing melody before training
                self.last_action = (0, np.array(self.body.brain.try_melody_parameters[["rx", "f1", "f2", "t1", "t2"]]))
                self.note_idx += 1

                if self.note_idx < len(self.body.brain.melody):
                    print("Try melody, note: ", self.note_idx)
                return self.last_action
            else:
                self.note_idx = 0
                self.try_melody = False
                print("Melody ended, start learning")

        if len(self.actions) <= 0:
            self.actions = copy.deepcopy(self.all_actions)
            self.note_idx += 1
            if self.note_idx >= len(self.body.brain.melody):
                return None

        self.last_action = self.actions.pop()

        return self.last_action

    def last_action(self):
        return self.last_action

    def action_idx_to_label(self, idx):
        return self.all_labels[idx]




class SGPModel(object):
    kernel = None
    mean_function = None
    gp_process_forward = None
    gp_process_inverse = None
    optimizer = None
    opt_logs = None
    likelihood = .1
    lengthscale = 1.
    kernel_variance = 1.0

    X = None
    Y = None

    def __init__(self, X=None, Y=None, kernel=None, mean_function=None, optimizer=gpflow.optimizers.Scipy(),
                 params=None, verbose=False):
        self.verbose = verbose
        self.params = params
        non_zero_params = [x for x in params['action_params_max']-params['action_params_min']]
        self.number_of_y_params = len([x for x in non_zero_params if x!=0])
        self.number_of_x_params = len(self.params['params_to_optimize'])
        self.search_indexes_to_optimize = [i for i in range(len(non_zero_params)) if non_zero_params[i] != 0] # robot control
        self.output_indexes_to_model = [i for i in range(len(PIANO_OUTPUT_PARAMETERS)) if PIANO_OUTPUT_PARAMETERS[i] in params['params_to_optimize']]
        self.optimizer = optimizer
        self.mean_function = mean_function
        self.update_fit_data(X, Y)

    def _inverse_objective_closure(self):
        return - self.gp_process_inverse.log_marginal_likelihood()

    def _forward_objective_closure(self):
        return - self.gp_process_forward.log_marginal_likelihood()

    def discard_data(self):
        self.X = None
        self.Y = None
        self.kernel = None
        self.mean_function = None
        self.gp_process_inverse = None
        self.gp_process_forward = None
        self.optimizer = None
        self.opt_logs = None

    def update_fit_data(self, X, Y):
        if X is not None:
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                Y = Y.reshape(1, -1)

            if self.lengthscale is None:
                self.lengthscale = 2 * np.ones(X.shape[1]).reshape(1, -1)

            # -- update dataset
            if self.X is None:
                self.X = np.squeeze(np.array([X[:, self.output_indexes_to_model]])).astype(np.float64)  # X
                if len(self.X.shape) == 1:
                    self.X = self.X.reshape(1, -1)
                self.Y = np.squeeze(np.array([Y[:, self.search_indexes_to_optimize]])).astype(np.float64)  # Y
                if len(self.Y.shape) == 1:
                    self.Y = self.Y.reshape(1, -1)
            else:
                self.X = np.append(arr=self.X, values=[X[-1, self.output_indexes_to_model].astype(np.float64)], axis=0)
                self.Y = np.append(arr=self.Y, values=[Y[-1, self.search_indexes_to_optimize].astype(np.float64)], axis=0)

            if len(self.X.shape) == 1 or self.X.shape == ():
                self.X = self.X.reshape(-1, 1)
            if len(self.Y.shape) == 1 or self.Y.shape == ():
                self.Y = self.Y.reshape(-1, 1)

            # -- (re)fit inverse model to data ------------------------------

            self.kernel_inverse = gpflow.kernels.RBF(
                self.number_of_x_params,
                lengthscale=np.array([self.lengthscale]*self.number_of_x_params),
            )
            self.gp_process_inverse = gpflow.models.SGPR(
                data=(self.X, self.Y),
                kernel=self.kernel_inverse,
                inducing_variable=self.X[::2],
                num_latent=self.number_of_x_params
            )
            if self.verbose:
                print_summary(self.gp_process_inverse)
            self.gp_process_inverse.likelihood.variance.assign(self.likelihood)

            # optimize hyperparameters of model by log marginal likelyhood
            self.opt_logs = self.optimizer.minimize(self._inverse_objective_closure,
                                                    self.gp_process_inverse.trainable_variables,
                                                    options=dict(maxiter=100))
            if self.verbose:
                print_summary(self.gp_process_inverse)

            # -- (re)fit forward model to data ------------------------------
            self.kernel_forward = gpflow.kernels.RBF(
                self.number_of_y_params,
                lengthscale=np.array([self.lengthscale]*self.number_of_y_params),
            )
            self.gp_process_forward = gpflow.models.SGPR(
                data=(self.Y, self.X),
                kernel=self.kernel_forward,
                inducing_variable=self.Y[::2],
                num_latent=self.number_of_y_params
            )
            if self.verbose:
                print_summary(self.gp_process_forward)
            self.gp_process_forward.likelihood.variance.assign(self.likelihood)

            # optimize hyperparameters of model by log marginal likelyhood
            self.opt_logs = self.optimizer.minimize(self._forward_objective_closure,
                                                    self.gp_process_forward.trainable_variables,
                                                    options=dict(maxiter=100))
            if self.verbose:
                print_summary(self.gp_process_forward)
            return True
        return False

    # returns (mean, var) tuple
    def predict(self, X, inverse=False):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if inverse:
            return self.gp_process_inverse.predict_f(X)
        return self.gp_process_forward.predict_f(X)

    def predict_compute_mse(self, X_test, y_test, inverse=False):
        if inverse:
            mean, var = self.gp_process_inverse.predict_f(X_test)
        else:
            mean, var = self.gp_process_forward.predict_f(X_test)
        mse = np.sum(np.sqrt((mean - y_test) ** 2))
        unc = np.sum(var)
        return mean, var, mse, unc


class GPModel(object):
    kernel = None
    mean_function = None
    gp_process_forward = None
    gp_process_inverse = None
    optimizer = None
    opt_logs = None
    likelihood = .001
    control_lengthscale = np.array([10., 10., 4., .5, .5])
    # output_lengthscale = np.array([5., 1., .1, .05])
    output_lengthscale = np.array([0.01, 0.01, 10., 5.])
    kernel_variance = 1.

    X = None
    Y = None

    forward_models = None
    inverse_models = None

    def __init__(self, X=None, Y=None, mean_function=None, output_indexes_to_model = None,
                 search_indexes_to_optimize=None, optimizer=gpflow.optimizers.Scipy(), params=None,
                 verbose=False):
        self.verbose = verbose
        self.params = params
        non_zero_params = [x for x in params['action_params_max']-params['action_params_min']]
        self.number_of_y_params = len([x for x in non_zero_params if x!=0])
        self.number_of_x_params = len(self.params['params_to_optimize'])
        if search_indexes_to_optimize is None:
            self.search_indexes_to_optimize = [i for i in range(len(non_zero_params)) if non_zero_params[i] != 0] # robot control
        else:
            self.search_indexes_to_optimize = search_indexes_to_optimize
        if output_indexes_to_model is None:
            self.output_indexes_to_model = [i for i in range(len(PIANO_OUTPUT_PARAMETERS)) if PIANO_OUTPUT_PARAMETERS[i] in params['params_to_optimize']]
        else:
            self.output_indexes_to_model = output_indexes_to_model
        self.optimizer = optimizer
        self.mean_function = mean_function
        self.update_fit_data(X, Y)

    def _inverse_objective_closure(self):
        return - self.gp_process_inverse.log_marginal_likelihood()

    def _forward_objective_closure(self):
        return - self.gp_process_forward.log_marginal_likelihood()

    def optimize_model_with_scipy(self, model):
        @tf.function(autograph=False)
        def obj():
            return -model.log_marginal_likelihood()

        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(obj,
                           variables=model.trainable_variables,
                           options={"disp": False, "maxiter": 10}
                           )

    def discard_data(self):
        self.X = None
        self.Y = None
        self.kernel = None
        self.mean_function = None
        self.gp_process_inverse = None
        self.gp_process_forward = None
        self.optimizer = None
        self.opt_logs = None

    def update_fit_data(self, X, Y, dim=None, data_override=False, kernel_forward=None, kernel_inverse=None, likelihood=None, inverse=False):
        if X is not None:
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                Y = Y.reshape(1, -1)

            if not data_override:
                # -- update dataset
                if self.X is None:
                    self.X = np.squeeze(np.array([X[:, self.output_indexes_to_model]])).astype(np.float64)  # X
                    self.Y = np.squeeze(np.array([Y[:, self.search_indexes_to_optimize]])).astype(np.float64)  # Y
                    self.X = self.X.reshape(-1, self.number_of_x_params)
                    self.Y = self.Y.reshape(-1, self.number_of_y_params)
                else:
                    self.X = np.append(arr=self.X, values=[X[-1, self.output_indexes_to_model].astype(np.float64)], axis=0)
                    self.Y = np.append(arr=self.Y, values=[Y[-1, self.search_indexes_to_optimize].astype(np.float64)], axis=0)
            else:
                self.X = X
                self.Y = Y

            if inverse:
                # these models capture the relationship between sound to countrol.
                # So we build 5 multivariate models, each for each control parameter
                if self.inverse_models is None:
                    self.inverse_models = dict()
                    self.kernel_inverse = dict()

                for i_param in range(self.number_of_y_params):

                    # -- (re)fit inverse model to data ------------------------------
                    if kernel_inverse==None:
                        self.kernel_inverse[i_param] = gpflow.kernels.Linear(
                            self.number_of_x_params,
                        ) + gpflow.kernels.RationalQuadratic(
                            self.number_of_x_params,
                            lengthscales=np.array(self.output_lengthscale[self.output_indexes_to_model]),
                        )
                    else:
                        self.kernel_inverse = kernel_inverse
                    if dim is None:
                        self.inverse_models[i_param] = gpflow.models.GPR(
                            data=(self.X, self.Y[:, i_param].reshape(-1, 1)),
                            kernel=self.kernel_inverse[i_param],
                        )
                    else:
                        self.inverse_models[i_param] = gpflow.models.GPR(
                            data=(self.X, self.Y),
                            kernel=self.kernel_inverse,
                        )

                    if likelihood is None:
                        self.inverse_models[i_param].likelihood.variance.assign(self.likelihood)
                    else:
                        self.inverse_models[i_param].likelihood.variance.assign(likelihood)

                    # optimize hyperparameters of model by log marginal likelyhood
                    # self.optimize_model_with_scipy(self.inverse_models[i_param])
                    # if self.verbose:
                    #     print_summary(self.inverse_models[i_param])
            else:
                # these models capture the relationship between control to sound. So we build
                # 5 models each one dimensional using the correlations specified in the main
                if self.forward_models is None:
                    self.forward_models = dict()

                for i_param in range(self.number_of_y_params):
                    # -- (re)fit forward model to data ------------------------------

                    if self.forward_models is None:
                        self.forward_models = dict()
                        self.kernel_forward = dict()

                    if kernel_forward is None:
                        self.kernel_forward[i_param] = gpflow.kernels.Linear() + \
                                                       gpflow.kernels.RationalQuadratic(
                            lengthscales=np.array(self.output_lengthscale[self.output_indexes_to_model]),
                        )
                    else:
                        self.forward_models = kernel_forward

                    x_var = self.params['corr'][ROBOT_CONTROL_PARAMETERS[self.search_indexes_to_optimize[i_param]]]
                    x_idx = [i for i in range(len(PIANO_OUTPUT_PARAMETERS)) if PIANO_OUTPUT_PARAMETERS[i] == x_var][0]
                    self.forward_models[i_param] = gpflow.models.GPR(
                        data=(self.Y[:, i_param].reshape(-1, 1), self.X[:, x_idx].reshape(-1, 1)),
                        kernel=self.kernel_forward[i_param],
                    )
                    self.forward_models[i_param].likelihood.variance.assign(self.likelihood)
                    # optimize hyperparameters of model by log marginal likelyhood
                    # self.optimize_model_with_scipy(self.forward_models[i_param])
                    # if self.verbose:
                    #     print("optim for: {}-{}".format(x_var, ROBOT_CONTROL_PARAMETERS[i_param]))
                    #     print_summary(self.forward_models[i_param])
            return True
        return False

    # returns (mean, var) tuple
    def predict(self, X, dim=None, inverse=False):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        res_mean = []
        res_var = []
        if not inverse:
            for i in range(self.number_of_y_params):
                mean, var = self.forward_models[i].predict_f(X[:, i].reshape(-1, 1))
                res_mean += [mean]
                res_var += [var]
            mean = np.array(res_mean)[:, :, 0].T
            var = np.array(res_var)[:, :, 0].T
        else:
            if dim is None:
                res_mean = np.repeat(
                    ((self.params['action_params_max'] + self.params['action_params_min']) / 2).reshape(1, -1),
                    X.shape[0],
                    axis=0
                )
                res_var = np.zeros(res_mean.shape)
                for i in range(self.number_of_y_params):
                    mean, var = self.inverse_models[i].predict_f(X)
                    res_mean[:, self.search_indexes_to_optimize[i]] = np.array(mean).flatten()
                    res_var[:, self.search_indexes_to_optimize[i]] = np.array(var).flatten()
                mean = res_mean
                var = res_var
            else:
                mean, var = self.inverse_models[0].predict_f(X)
        return mean, var

    def predict_compute_mse(self, X_test, y_test, inverse=False):
        if inverse:
            mean, var = self.gp_process_inverse.predict_f(X_test)
        else:
            mean, var = self.gp_process_forward.predict_f(X_test)
        mse = np.sum(np.sqrt((mean - y_test) ** 2))
        unc = np.sum(var)
        return mean, var, mse, unc


def save_dict_to_file(data=None, path=None, filename=None, format='json'):
    # if don't want to overwrite, check next available name
    file = "{}{}.{}".format(path, filename, format)
    num = 0
    while file_exists(file):
        file = "{}{}-({}).{}".format(path, filename, num, format)
        num += 1
    with open(file, 'w') as exp_file:
        json.dump(data, exp_file)


def get_time(sec):
    verbose = ["sec(s)", "min(s)", "hour(s)", "day(s)"]
    time = []
    time += [sec // (24 * 3600)]  # days
    sec %= (24 * 3600)
    time += [sec // 3600]  # hours
    sec %= 3600
    time += [sec // 60]  # minutes
    sec %= 60
    time += [sec]  # seconds
    time = time[::-1]
    time_output = ""
    for i in range(len(time)):
        val = time.pop()
        tag = verbose.pop()
        if val != 0:
            time_output += "{}{} :".format(int(val), tag)
    return time_output[:-2]