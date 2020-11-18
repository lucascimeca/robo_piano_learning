import numpy as np
import sys
import signal
from pianolearning import (
    PianoLearning,
    RobotPlayer,
)

# optimized with single values
OPTIMIZED_TESTING_ACTIONS_WRT_MIDI = {
    "normal": [54.3128886162477, 7.556126148475099, 4.32609377677284, 0.3564508857015497, 0.8966013571668155],
    "tenuto": [54.94790485422284, 7.249875349543384, 6.413013935026143, 0.7490459138750744, 1.6333549943993466],
    "staccatissimo": [59.70559664083777, 6.646252180041001, 4.353415106110695, 0.839683941606082, 0.4231272084924607],
    "staccato": [55.754185752878286, 9.28818030846551, 4.9393718694376645, 1.0746600776605124, 0.7399746446145209],
    "ff": [54.24719208843263, 8.471212132644547, 4.872625024924235, 0.6954223007596405, 1.163323298960148],
    "f": [55.70383282946662, 7.319653040548909, 4.337056197857326, 0.5868293030733518, 0.9703450329237452],
    "mp": [56.009939865049105, 7.705769279351372, 4.34562526713487, 0.46086564042984113, 0.9326416547854908],
    "p": [44.16163461841167, 6.439862562491358, 6.186711467753606, 0.6739529005654775, 1.4322080278053864],
    "pp": [28.891630319797965, 5.428432561269891, 5.643884268052922, 0.49113119587035664, 0.891897358270851],
    "pppp": [30.404245187613824, 1.8974385647917158, 6.561714936412166, 0.7745048018760567, 0.9201893246245201],
}

OPTIMIZED_TESTING_ACTIONS_WRT_HUMAN = {
    "normal": [32.30819463324322, 15.360479373606296, 12.344512197812996, 0.8413435821985149, 1.278470688949546],
    "tenuto": [35.01467128287456, 7.091906626270991, 7.082581159244362, 0.6893909101380749, 1.331871737270036],
    "staccatissimo": [28.77858609642408, 4.187268139933119, 7.008837617263351, 0.5291160017144829, 0.5291160017144829],
    "staccato": [69.04953337932119, 6.996204268697155, 9.885141770108675, 0.27151283225315404, 0.28522532490293084],
    "ff": [31.48673056222439, 4.611999166287468, 4.548922877592037, 0.6337788733185329, 0.21886522569756933],
    "f": [67.4039851202335, 6.3159121434335415, 6.939793120397385, 0.47553606173380814, 1.181067699370624],
    "mp": [28.782689730592427, 4.032775009769811, 5.214869930460684, 0.5547419381060292, 1.3409395471668895],
    "p": [28.862517314712726, 5.675671294059304, 4.10825802280357, 0.2895266977879686, 0.9634386575403193],
    "pp": [28.846898446683497, 5.395912923973587, 6.1652938094040515, 0.5426974986512274, 0.9362428908092765],
    "pppp": [28.64246401633468, 3.8003634253886167, 6.5336811812528275, 0.6316467596498168, 0.9673463258308954],
}


"""function running the learning live
Attributes:
     time (double): time-step (sec) for image sampling and data collection"""
def run_learning(time_interval=1.,
                 testing_params=None,
                 steps=0,
                 robot_parameters=None,
                 melody_replay=False,
                 test_frequency=30,
                 learning=True,
                 save=False,
                 resume_previous_experiment=True,
                 verbose=False,
                 sort_actions=False,
                 action_repeat_number=10,
                 GP_search=False,
                 gp_plots=False
                 ):

    # ----create instances and threads----
    piano_learning = PianoLearning(interval=time_interval, verbose=True)  # start palpation on main thread
    robot_player = RobotPlayer(
        robot_ip="169.254.94.83",
        robot_parameters=robot_parameters,
        learning=learning,
        testing_params=testing_params,
        melody_replay=melody_replay,
        test_frequency=test_frequency,
        save=save,
        resume_previous_experiment=resume_previous_experiment,
        data_folder='./../data/',
        verbose=verbose,
        sort_actions=sort_actions,
        action_repeat_number=action_repeat_number,
        gp_search=GP_search,
        gp_plots=gp_plots
    )

    # ---- subscribe the robot to the piano playing experiment----
    piano_learning.subscribe(robot_player)

    # start threads
    robot_player.start()

    # ---- Handle for graceful shutdowns -----
    def signal_handler(sig, frame):
        piano_learning.end_run()
        sys.exit()
    signal.signal(signal.SIGINT, signal_handler)  # to handle gracefull shutdowns

    # ----run experiment on main thread----
    try:
        piano_learning.run(steps)

    except Exception as e:
        raise Exception(e)

    finally:
        piano_learning.end_run()


if __name__ == "__main__":

    # these parameters define the robot exploration space

    corr_parameters = {
        'f1': 'on_velocity',
        'f2': 'off_velocity',
        't1': 'wait_time',
        't2': 'hold_time',
        'rx': 'on_velocity'
    }

    robot_params = {
        # # GRID SEARCH 5
        # 'action_params_min': np.array([-18, 0.1, 0.1, 0.5, 0.5]),   # staccato hard-corded params
        # 'action_params_max': np.array([108, 12., 12., 3., 3.]),    # + hold_time exploration

        # # GRID SEARCH 10
        # 'action_params_min': np.array([-10, 0.1, 0.1, 0.5, 0.5]),   # staccato hard-corded params
        # 'action_params_max': np.array([100, 12., 12., 3., 3.]),    # + hold_time exploration

        # GP SEARCH
        'action_params_min': np.array([0, 2.083, 2.083, 0.92, 0.92]),   # staccato hard-corded params
        'action_params_max': np.array([90, 8.03, 8.03, 2.58, 2.58]),    # + hold_time exploration

        # TOREMOVE
        # 'action_params_min': np.array([90, 2.083, 5., 1.5, 1.5]),   # staccato hard-corded params
        # 'action_params_max': np.array([90, 8.03, 5., 1.5, 1.5]),    # + hold_time exploration
        'grid_action_number': 5,
        'GP_grid_action_number': 1000,

        'params_to_optimize': ['on_velocity', "hold_time", "wait_time", "off_velocity"],

        'corr': corr_parameters
    }



    # run the experiment
    run_learning(
        learning=False,
        testing_params=OPTIMIZED_TESTING_ACTIONS_WRT_MIDI,

        time_interval=0,                    # how much does a step last (sec) -- 0 means as little as possible
        steps=0,                            # number of time steps -- 0 means infinite
        test_frequency=30,                  # number of learning iterations after which to re-try the melody
        robot_parameters=robot_params,
        save=True,
        melody_replay=False,
        resume_previous_experiment=True,   # if True the previous experiment if continued from where it was left off
        verbose=False,
        sort_actions=False,
        action_repeat_number=10,
        GP_search=False,
        gp_plots=False,
    )
