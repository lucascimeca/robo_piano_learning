import cv2
import matplotlib.pyplot as plt
from data_operators import SkinWriter
import ctypes
from simple_io import *

PIANO_OUTPUT_PARAMETERS = ["on_velocity", "off_velocity", "wait_time", "hold_time"]
ROBOT_CONTROL_PARAMETERS = ["rx", "f1", "f2", "t1", "t2"]

labeltag_dict = {
    "rrh": "Round Rough Hard",
    "srh": "Square Rough Hard",
    "rsh": "Round Smooth Hard",
    "ssh": "Square Smooth Hard",
    "rrs": "Round Rough Soft",
    "srs": "Square Rough Soft",
    "rss": "Round Smooth Soft",
    "sss": "Square Smooth Soft"
}

label_tag_task_dict = {
    0: {
        'r': 'Round',
        's': 'Edged'
    },
    1: {
        'r': 'Rough',
        's': 'Smooth'
    },
    2: {
        'h': 'Hard',
        's': 'Soft',
    }
}
# r-round, s-square
# s-smooth, r-rough
# s-soft, h-hard
task_letters = {
    1: ['r', 's'],
    2: ['r', 's'],
    3: ['h', 's']
}

task_labels = {
    1: {
        'r': 'round',
        's': 'edged'
    },
    2: {
        'r': 'rough',
        's': 'smooth'
    },
    3: {
        'h': 'hard',
        's': 'soft',
    }
}


class SincDataViewer(object):
    out = None  # variable holding writer object for saving view

    def __init__(self, window_name='cam_0', data_folder='./../data/', observer=None):
        self.window_name = window_name
        self.observer = observer
        self.data_folder = data_folder

        # screen size
        self.screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        # compute where in screen (different depends which Viewer)
        if self.observer.pos_arg == 'left':
            self.x_coord = 0
        elif self.observer.pos_arg == 'center':
            self.x_coord = np.int(self.screen_size[0]/3)
        elif self.observer.pos_arg == 'right':
            self.x_coord = np.int(self.screen_size[0]/3)*2
        else:
            raise ValueError("Please enter the correct screen position value (i.e. one of: 'left', 'center' or 'right')")

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        if self.out:
            self.out.vacuum()
        cv2.destroyAllWindows()

    def show_frame(self, step):
        raise NotImplementedError


class CameraViewer(SincDataViewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frame_width = self.detector.camera.get(3)
        self.frame_height = self.detector.camera.get(4)

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        if self.detector.save and not self.detector.from_saved:
            self.out = cv2.VideoWriter(
                self.data_folder+'out_cam_{}.avi'.format(self.detector.camera_num),
                cv2.VideoWriter_fourcc(*'XVID'),
                25,  # todo: parameter must be based on time interval
                (np.int32(self.frame_width), np.int32(self.frame_height)),
                True
            )

    def show_frame(self, step, roi=None):
        # save only if connected to live cameras
        if self.detector.save and not self.detector.from_saved:
            self.out.write(self.detector.data[step - self.detector.step_delay, :, :, :])

        # print the image in the data at step, unless an image was passed to this function
        if roi is None:
            roi = self.detector.data[step - self.detector.step_delay, :, :, :]
        bw_img = self.detector.bw_data[step - self.detector.step_delay, :, :]

        # show data frames
        cv2.imshow(self.window_name, roi)
        cv2.moveWindow(self.window_name, self.x_coord, 0)

        # show detection
        cv2.imshow(self.window_name + 'detection', bw_img)
        cv2.moveWindow(self.window_name + 'detection', self.x_coord, np.int(self.frame_height + 40))

        cv2.waitKey(1)
        return True

class SkinViewer(SincDataViewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_size = tuple([scr/2 for scr in self.screen_size])

        #compute dimension of windows to fit skin readings
        self.ihb_num = self.detector.skin.shape[0]
        self.module_num = self.detector.skin.shape[1]
        division_to_screen = np.ceil(np.sqrt(self.ihb_num * self.module_num))
        self.frame_height = self.screen_size[1]/division_to_screen
        self.frame_width = self.screen_size[0]/division_to_screen

        self.x_coord = 0
        self.y_coord = 0

        # define writer object to save skin data
        if self.detector.save and not self.detector.from_saved:
            self.out = SkinWriter(
                shape=(0,) + self.detector.data.shape[1:],
                name="skin_out",
                format="h5",
                folder="./../data/"
            )

    def show_frame(self, step):
        if self.detector.save and not self.detector.from_saved:
            self.out.write(self.detector.data[step - self.detector.step_delay-1, :])

        horizontal_count = 0
        for ihb in range(self.ihb_num):
            for module in range(self.module_num):
                heatmap, _ = skin_heatmap(self.detector.data[step - self.detector.step_delay-1, ihb, module, :], max=16000)

                width_to_display = int(np.floor((self.frame_height / heatmap.shape[1]) * heatmap.shape[0]))
                height_to_display = int(self.frame_height)
                reshaped_heatmap = cv2.resize(heatmap, (width_to_display, height_to_display), interpolation=cv2.INTER_CUBIC)

                # show data frames
                cv2.imshow("{}-{}-{}".format(self.window_name, ihb, module), reshaped_heatmap)
                cv2.moveWindow("{}-{}-{}".format(self.window_name, ihb, module), int(self.x_coord), int(self.y_coord))

                if not self.x_coord + self.frame_width > self.screen_size[1]:
                    self.x_coord += self.frame_width
                else:
                    self.x_coord = 0
                    self.y_coord += self.frame_height


        cv2.waitKey(1)

        self.x_coord = 0
        self.y_coord = 0

        return True

    def save_frame(self, step):
        if self.detector.save and not self.detector.from_saved:
            self.out.write(self.detector.data[step - self.detector.step_delay-1, :])

# Viewer for a piano player object
class PianoPlayerViewer(SincDataViewer):
    out = None  # variable holding writer object for saving view

    def __init__(self, memory_frames=500, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ROBOT PARAMS
        self.robot_frame_width = int(self.screen_size[0] / 3)
        self.robot_frame_height = int(self.screen_size[1] / 3)

        self.buffer_length = memory_frames
        self.action_profile_buffer = np.zeros((self.buffer_length, 3)) # holds the x-y-z location of the robot for the last 500 frames

        self.robot_y_coord = 0
        self.robot_x_coord = np.int((self.screen_size[0] / 4))

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        cv2.destroyAllWindows()

    def show_frame(self, time, frame_rate=1):

        tmp_buffer = np.zeros((self.buffer_length, 3))
        tmp_buffer[0:self.buffer_length-1] = self.action_profile_buffer[1:self.buffer_length]
        tmp_buffer[-1, :] = self.observer.state['actual_TCP_pose'][:3]
        self.action_profile_buffer = tmp_buffer

        # -------------------------------
        # ---- SHOW ACTION PROFILE -------
        learing_figure = plt.figure(0, figsize=(8, 6))

        axes_z = learing_figure.add_subplot(311)
        axes_z.title.set_text('X profile')
        axes_z.plot(self.action_profile_buffer[:, 0], 'r--', label="Robot X Positions")
        # axes_z.plot(np.arange(40)*(1/frame_rate), self.state_profile_buffer[:40, 2], 'k-', label="robot position")
        axes_z.set_ylabel('displacement\n(mm)')
        axes_z.set_autoscale_on(True)
        axes_z.legend()

        axes_rx = learing_figure.add_subplot(312)
        axes_rx.title.set_text('Y Profile')
        axes_rx.plot(self.action_profile_buffer[:, 1], 'r--', label="Robot Y Positions")
        # axes_rx.plot(np.arange(self.buffer_length)*(1/frame_rate), self.state_profile_buffer[:, 3], 'k-', label="robot position")
        axes_rx.set_ylabel('displacement\n(mm)')
        axes_rx.set_autoscale_on(True)
        axes_rx.legend()

        axes_ry = learing_figure.add_subplot(313)
        axes_ry.title.set_text('Z Profile')
        axes_ry.plot(self.action_profile_buffer[:, 2], 'r--', label="Robot Z Positions")
        # axes_ry.plot(np.arange(self.buffer_length)*(1/frame_rate), self.state_profile_buffer[:, 4], 'k-', label="robot position")
        axes_ry.set_xlabel('playing time (s)')
        axes_ry.set_ylabel('displacement\n(mm)')
        axes_ry.set_autoscale_on(True)
        axes_ry.legend()
        learing_figure.tight_layout()

        learing_figure.canvas.draw()

        learning_curve_img = np.fromstring(learing_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        learning_curve_img = learning_curve_img.reshape(learing_figure.canvas.get_width_height()[::-1] + (3,))

        # print the image in the data at step, unless an image was passed to this function
        height_to_display = int(np.floor((learning_curve_img.shape[0] * self.robot_frame_height) / self.robot_frame_width))
        reshaped_learningroi = cv2.resize(learning_curve_img, (self.robot_frame_width, height_to_display))
        plt.close('all')

        cv2.imshow(self.window_name + 'action_profile', reshaped_learningroi)
        cv2.moveWindow(self.window_name + 'action_profile', self.robot_x_coord, self.robot_y_coord)

        cv2.waitKey(1)
        return True

# Viewer for a piano player object
class GPActionViewer(SincDataViewer):
    def __init__(self, memory_frames=500, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ROBOT PARAMS
        self.robot_frame_width = int(self.screen_size[0] / 3)
        self.robot_frame_height = int(self.screen_size[1] / 3)

        self.buffer_length = memory_frames
        self.action_profile_buffer = np.zeros((self.buffer_length, 3)) # holds the x-y-z location of the robot for the last 500 frames

        self.robot_y_coord = 0
        self.robot_x_coord = np.int((self.screen_size[0] / 4))

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        cv2.destroyAllWindows()

    def show_frame(self, step=0, roi=None, time=0, gp_model=None, inputs=None, params=None, precision=100):

        X = gp_model.Y
        Y = gp_model.X

        if self.observer.mesh_mean is not None:
            shift = 0
            for i_param in range(gp_model.number_of_y_params):
                # -------------------------------
                # ---- SHOW ACTION PROFILE -------
                learing_figure = plt.figure(0, figsize=(8, 6))
                ax = learing_figure.add_subplot(111)
                xs = inputs[:, i_param]
                ys = self.observer.mesh_mean[:, i_param]
                err = 1.96 * np.sqrt(self.observer.mesh_var[:, i_param])

                ax.plot(xs, ys, 'k', zorder=9, linewidth=3, label='mean prediction')
                ax.fill_between(xs, ys - err, ys + err, color='C0', alpha=0.2, label='1.96$\\times \sigma$')

                if X is not None and Y is not None:
                    y_var = gp_model.params['corr'][ROBOT_CONTROL_PARAMETERS[gp_model.search_indexes_to_optimize[i_param]]]
                    y_idx = [i for i in range(len(PIANO_OUTPUT_PARAMETERS)) if PIANO_OUTPUT_PARAMETERS[i] == y_var][0]
                    plt.scatter(X[:, i_param], Y[:, y_idx], c='r', label='data points')

                plt.title("Iteration: {}  ".format(self.observer.no_issued_actions) +
                          "Cumulative Variance: {0:.2f}".format(np.sum(self.observer.mesh_var[:, i_param])),
                          fontsize=18,
                )
                ax.set_xlabel('${}$'.format(
                    ROBOT_CONTROL_PARAMETERS[gp_model.search_indexes_to_optimize[i_param]]),
                    fontsize=18
                )
                ax.set_ylabel("${}$".format(y_var), fontsize=18)
                ax.yaxis.set_tick_params(labelsize=18)
                ax.xaxis.set_tick_params(labelsize=18)
                ax.set_autoscale_on(True)
                ax.legend(fontsize=14)
                # plt.show()
                learing_figure.tight_layout()

                learing_figure.canvas.draw()

                learning_curve_img = np.fromstring(learing_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                learning_curve_img = learning_curve_img.reshape(learing_figure.canvas.get_width_height()[::-1] + (3,))

                reshaped_learningroi = cv2.resize(learning_curve_img, (300, 180))
                reshaped_learningroi = cv2.cvtColor(reshaped_learningroi, cv2.COLOR_RGB2BGR)
                plt.close('all')

                # print("\nCUMULATIVE MESH VARIANCE: {}\n".format(np.sum(self.observer.mesh_var)))
                cv2.imshow(self.window_name + 'action_profile_' + str(i_param), reshaped_learningroi)
                cv2.moveWindow(self.window_name + 'action_profile_' + str(i_param), shift, 20)
                shift += 350

            cv2.waitKey(1)
        return True
