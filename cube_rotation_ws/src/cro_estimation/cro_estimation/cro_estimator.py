import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from mujoco import mju_mulQuat, mju_negQuat, mju_normalize4
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.estimation import ObeliskEstimator
from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_sensor_msgs.msg import ObkJointEncoders
from pyzed import sl
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from ruamel.yaml import YAML
from rviz_2d_overlay_msgs.msg import OverlayText
from sensor_msgs.msg import Image

from cro.conversion import numpy_to_image
from cro.corrector import WrenchCorrector
from cro.perseus import PerseusWrapper

torch.set_float32_matmul_precision("high")


class ZEDCamera:
    """A class for handling Zed camera I/O."""

    def __init__(
        self,
        serial_number: int,
        depth: bool = True,
        side: str = "left",
        resolution: str = "VGA",
        fps: int = 100,
    ) -> None:
        """Initialize the ZED camera."""
        self.depth = depth
        if side == "left":
            self.rgb_view = sl.VIEW.LEFT
            if self.depth:
                self.depth_measure = sl.MEASURE.DEPTH
        else:
            self.rgb_view = sl.VIEW.RIGHT
            if self.depth:
                self.depth_measure = sl.MEASURE.DEPTH_RIGHT

        if resolution.lower() == "vga":
            sl_resolution = sl.RESOLUTION.VGA
            if fps not in [15, 30, 60, 100]:
                raise ValueError(f"Invalid fps for VGA resolution: {fps}")
        elif resolution.lower() == "720":
            sl_resolution = sl.RESOLUTION.HD720
            if fps not in [15, 30, 60]:
                raise ValueError(f"Invalid fps for HD720 resolution: {fps}")
        elif resolution.lower() == "1080":
            sl_resolution = sl.RESOLUTION.HD1080
            if fps not in [15, 30]:
                raise ValueError(f"Invalid fps for HD1080 resolution: {fps}")
        elif resolution.lower() == "2k":
            sl_resolution = sl.RESOLUTION.HD2K
            if fps != 15:  # noqa: PLR2004
                raise ValueError(f"Invalid fps for HD2K resolution: {fps}")
        else:
            raise ValueError(f"Invalid resolution: {resolution}")

        self.camera = sl.Camera()
        self.rgb_buffer = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        # Set initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_image_flip = sl.FLIP_MODE.OFF
        init_params.camera_resolution = sl_resolution
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL if depth else sl.DEPTH_MODE.NONE
        init_params.depth_stabilization = 100
        init_params.coordinate_units = sl.UNIT.METER

        if depth:
            init_params.depth_minimum_distance = 0.1
            init_params.depth_maximum_distance = 0.5
            self.depth_buffer = sl.Mat()
        else:
            init_params.depth_minimum_distance = 0.3
            init_params.depth_maximum_distance = 1.0
        init_params.set_from_serial_number(serial_number)

        self.runtime_parameters.enable_depth = depth
        if depth:
            self.runtime_parameters.enable_fill_mode = True  # use fill mode to smooth the image out

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Didn't open!")
            sys.exit(1)

        self.frame = None  # np version of the frame

    def get_frame(self) -> None:
        """Get a frame from the camera.

        Returns:
            np.ndarray: The frame as a numpy array. Shape: (256, 256, 3) or (256, 256, 4) if depth is enabled.
                None if the frame retrieval fails.
        """
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.rgb_buffer, self.rgb_view)
            frame = self.rgb_buffer.get_data()[..., :3]  # (H, W, C), bgr, uint8
            frame = frame[..., ::-1] / 255.0  # Convert to RGB, divide by 255 to normalize
            assert frame.shape[2] == 3, "Frame should have 3 channels."  # noqa: PLR2004
            if self.depth:
                self.camera.retrieve_measure(self.depth_buffer, self.depth_measure)
                depth = self.depth_buffer.get_data()
                depth[np.isnan(depth)] = 0
                depth[np.isinf(depth)] = 0
                depth /= 0.035  # unscale to the scale of the training data. 0.035 is the half side length of the cube.
                frame = np.concatenate([frame, depth[..., None]], axis=-1)

            # center crop to 256x256
            H, W = frame.shape[:2]  # noqa: N806
            frame = frame[H // 2 - 128 : H // 2 + 128, W // 2 - 128 : W // 2 + 128, ...]
            self.frame = frame

    def close(self) -> None:
        """Close the camera."""
        self.camera.close()


class ObeliskCubeRotationEstimator(ObeliskEstimator):
    """Node that performs state estimation for the entire system."""

    def __init__(self, node_name: str = "cube_rotation_estimator") -> None:
        """Initializes the cube estimator node."""
        super().__init__(node_name)

        # ############## #
        # ROS PARAMETERS #
        # ############## #

        # general ros parameters
        self.declare_parameter("estimator_type", "perseus")  # estimation method
        self.declare_parameter("corrector_type", "none")  # correction method
        self.declare_parameter("alpha_cube", 0.1)  # exponential smoothing factors for velocity estimates
        self.declare_parameter("alpha_leap", 0.1)

        # perseus-specific ros parameters
        self.declare_parameter("perseus_depth", True)  # whether to use depth images
        self.declare_parameter("perseus_smoother_freq", 100.0)  # frequency in Hz to run the smoother at
        self.declare_parameter("perseus_lookback", 3)

        # corrector-specific ros parameters
        self.declare_parameter("corrector_timestep", 0.002)  # timestep for the internal corrector mj model
        self.declare_parameter("corrector_q_leap", False)  # whether to use q_leap from the corrector
        self.declare_parameter("corrector_v_leap", False)  # whether to use v_leap from the corrector

        # wrench-specific ros parameters
        self.declare_parameter("wrench_kp_pos", 1000.0)  # proportional gain for position error
        self.declare_parameter("wrench_ki_pos", 1.0)  # integral gain for position error
        self.declare_parameter("wrench_kd_pos", 1.0)  # derivative gain for position error
        self.declare_parameter("wrench_kp_rot", 1.0)  # proportional gain for rotation error
        self.declare_parameter("wrench_ki_rot", 1.0)  # integral gain for rotation error
        self.declare_parameter("wrench_kd_rot", 1.0)  # derivative gain for rotation error
        self.declare_parameter("wrench_pass_v_cube", False)  # whether to pass the cube velocity through

        # viz-specific parameters
        self.declare_parameter("viz_images", False)  # whether to visualize the images by publishing as Image type

        # [HACK] manual adjustments
        self.declare_parameter("cam_A_trans_adjust", [0.0, 0.0, 0.0])  # translation adjustment for the A camera
        self.declare_parameter("cam_B_trans_adjust", [0.0, 0.0, 0.0])  # translation adjustment for the B camera
        self.declare_parameter("cam_C_trans_adjust", [0.0, -0.005, 0.01])  # translation adjustment for the C camera
        self.declare_parameter("q_cube_trans_adjust", [0.0, 0.01, 0.0])  # translation adjustment for the cube

        # ###################### #
        # COMPONENT REGISTRATION #
        # ###################### #

        # sensor data subscriptions
        self.register_obk_subscription(
            "sub_q_leap_setting",
            self.q_leap_callback,
            key="sub_q_leap",
            msg_type=ObkJointEncoders,
        )

        # timer for running the smoother
        self.register_obk_timer(
            "timer_smoother_setting",
            self.smoother_callback,
            key="timer_smoother",
        )

        # extra publisher for the cube estimate
        # we publish the leap hand estimated state with the default base publisher separately from the cube because
        # it makes setting up the viz of the cube and hand easier
        self.register_obk_publisher(
            "pub_q_cube_setting",
            key="pub_q_cube",
            msg_type=EstimatedState,
        )

        # corrector-specific components
        corrector_type = self.get_parameter("corrector_type").get_parameter_value().string_value
        if corrector_type in ["wrench"]:
            # publisher for the ghost cube before corrections
            self.register_obk_publisher(
                "pub_q_cube_ghost_setting",
                key="pub_q_cube_ghost",
                msg_type=EstimatedState,
            )

            # viz publisher for the corrector internal leap hand state
            self.register_obk_publisher(
                "pub_q_leap_ghost_setting",
                key="pub_q_leap_ghost",
                msg_type=EstimatedState,
            )

        # viz publisher for the leap hand encoder readings (the leap hand state after correction is published elsewhere)
        self.register_obk_publisher(
            "pub_q_leap_meas_setting",
            key="pub_q_leap_meas",
            msg_type=EstimatedState,
        )

        self.viz_images = self.get_parameter("viz_images").get_parameter_value().bool_value

        # ################### #
        # ROTATION STATISTICS #
        # ################### #

        self.q_cube_goal = None
        self.last_rotation_time = None
        self.rotation_times = []
        savepath = os.getenv("CUBE_ROTATION_OBELISK_ROOT") + "/experiments"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.rotation_times_file = open(
            f"{savepath}/times_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "a", newline=""
        )
        self.create_subscription(
            EstimatedState,
            "/cro/q_cube_goal",
            self._q_cube_goal_callback,
            10,
        )

        # ##### #
        # OTHER #
        # ##### #

        self.leap_joint_names = [
            "if_mcp", "if_rot", "if_pip", "if_dip",  # index finger
            "mf_mcp", "mf_rot", "mf_pip", "mf_dip",  # middle finger
            "rf_mcp", "rf_rot", "rf_pip", "rf_dip",  # ring finger
            "th_cmc", "th_axl", "th_mcp", "th_ipl",  # thumb
        ]  # fmt: skip
        self.is_configured = False

    # ############# #
    # CONFIGURATION #
    # ############# #

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the cube rotation estimator."""
        super().on_configure(state)
        cro_ros_dir = get_package_share_directory("cro_ros")  # CRO directory

        # set up cameras
        # assume that the cameras are ordered A, B, C, as labeled in the physical setup
        # A is on the side (no thumb), B is on the side (close to thumb), C is head on facing the hand
        zed_params_path = Path(cro_ros_dir) / "config/zed_mini_params.yaml"
        cam_params = YAML().load(zed_params_path)
        self.cameras = [None] * len(cam_params)
        self.camera_cbgs = [None] * len(cam_params)
        self.camera_timers = [None] * len(cam_params)

        self.image_publishers = {}  # optional: for visualizing the images
        image_publisher_cbg = ReentrantCallbackGroup()
        self.image_viz_timers = [None] * len(cam_params)

        for i, cam_param in enumerate(cam_params):
            serial_number = cam_param["serial_number"]
            resolution = cam_param["resolution"]
            side = cam_param["side"]
            fps = cam_param["fps"]
            self.cameras[i] = ZEDCamera(serial_number, depth=True, side=side, resolution=resolution, fps=fps)
            self.camera_cbgs[i] = MutuallyExclusiveCallbackGroup()
            self.camera_timers[i] = self.create_timer(
                1.0 / fps, callback=self.cameras[i].get_frame, callback_group=self.camera_cbgs[i]
            )  # the cameras are constantly updating their own buffers at fps Hz

            # optional: for visualizing the images
            # [NOTE] this really slows down the stack! only do this if debugging the camera/cube poses
            if self.viz_images:

                def _viz_callback(i: int) -> None:
                    if self.cameras[i] is not None:
                        frame = self.cameras[i].frame
                        if frame is not None:
                            rgb_frame = (frame.copy()[..., :3] * 255).astype(np.uint8)  # RGB channels only

                            # add keypoints to frame if using perseus
                            if hasattr(self, "perseus_wrapper"):
                                projected_keypoint_pixels = self.perseus_wrapper._projected_keypoint_pixels
                                keypoints = self.perseus_wrapper._keypoints

                                if projected_keypoint_pixels is not None:
                                    for keypoint in projected_keypoint_pixels[i, ...]:
                                        x, y = keypoint
                                        x, y = int(x), int(y)
                                        rgb_frame = cv2.circle(rgb_frame, (x, y), 3, (255, 0, 0), -1)

                                    edge_indices = [
                                        (0, 1),
                                        (0, 2),
                                        (2, 3),
                                        (1, 3),
                                        (0, 4),
                                        (1, 5),
                                        (2, 6),
                                        (3, 7),
                                        (4, 5),
                                        (4, 6),
                                        (5, 7),
                                        (6, 7),
                                    ]
                                    for edge in edge_indices:
                                        start = projected_keypoint_pixels[i, edge[0]]
                                        end = projected_keypoint_pixels[i, edge[1]]
                                        start = int(start[0]), int(start[1])
                                        end = int(end[0]), int(end[1])
                                        rgb_frame = cv2.line(rgb_frame, start, end, (255, 0, 0), 1)

                                if keypoints is not None:
                                    for keypoint in keypoints[i, ...]:
                                        x, y = keypoint
                                        x, y = int(x), int(y)
                                        rgb_frame = cv2.circle(rgb_frame, (x, y), 3, (0, 255, 0), -1)

                            self.image_publishers[f"pub_image_{i}"].publish(numpy_to_image(rgb_frame, "rgb8"))

                self.image_publishers[f"pub_image_{i}"] = self.create_publisher(
                    Image,
                    f"/cro/viz/cam{i}",
                    10,
                    callback_group=image_publisher_cbg,
                    non_obelisk=True,
                )
                self.image_viz_timers[i] = self.create_timer(
                    1.0 / 10.0,  # hardcoded to 10Hz for now
                    callback=lambda i=i: _viz_callback(i),
                    callback_group=image_publisher_cbg,
                )

        # device
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Running on device: {device_str}")
        self.device = torch.device(device_str)

        # image parameters
        self._H = 376
        self._W = 672
        self._H_crop = 256
        self._W_crop = 256
        self._num_imgs = 3  # 3 cameras

        # exponential moving average filter parameters
        self.alpha_cube = self.get_parameter("alpha_cube").get_parameter_value().double_value
        self.alpha_leap = self.get_parameter("alpha_leap").get_parameter_value().double_value
        self.get_logger().info(f"Alpha Cube: {self.alpha_cube}")
        self.get_logger().info(f"Alpha Leap: {self.alpha_leap}")

        # set properties based on estimator type
        self.estimator_type = self.get_parameter("estimator_type").get_parameter_value().string_value.lower()
        if self.estimator_type not in ["perseus"]:
            self.get_logger().error(f"Invalid estimator type: {self.estimator_type}")
            return TransitionCallbackReturn.ERROR

        if "perseus" in self.estimator_type:
            self.get_logger().info("Estimator Type: perseus")
            self.perseus_depth = self.get_parameter("perseus_depth").get_parameter_value().bool_value
            self._C = 3 if not self.get_parameter("perseus_depth").get_parameter_value().bool_value else 4
            if self.perseus_depth:
                weights_path = Path(cro_ros_dir) / "ckpts/perseus_depth.pth"
            else:
                weights_path = Path(cro_ros_dir) / "ckpts/perseus.pth"
            self.perseus_wrapper = PerseusWrapper(
                weights_path,
                zed_params_path,
                self._num_imgs,
                self._C,
                self._H_crop,
                self._W_crop,
                self.device,
                self.get_parameter("perseus_smoother_freq").get_parameter_value().double_value,
                self.get_parameter("perseus_lookback").get_parameter_value().integer_value,
                depth=self.perseus_depth,
                compile=False,  # TODO(ahl): for now, can't compile b/c of torch multithreading bug
                cam_z_adjust=-0.025,  # adjust the position of the cam origin along the viewing axis
                k_huber=1.345,  # Huber loss parameter for robust noise model
                # [HACK] manual adjustments
                cam_A_trans_adjust=self.get_parameter("cam_A_trans_adjust").get_parameter_value().double_array_value,
                cam_B_trans_adjust=self.get_parameter("cam_B_trans_adjust").get_parameter_value().double_array_value,
                cam_C_trans_adjust=self.get_parameter("cam_C_trans_adjust").get_parameter_value().double_array_value,
                q_cube_trans_adjust=self.get_parameter("q_cube_trans_adjust").get_parameter_value().double_array_value,
            )
        else:
            self.get_logger().error(f"Invalid estimator type: {self.estimator_type}")
            return TransitionCallbackReturn.ERROR

        # set properties based on corrector type
        self.corrector = None
        self.corrector_type = self.get_parameter("corrector_type").get_parameter_value().string_value.lower()

        if self.corrector_type == "wrench":
            self.get_logger().info("Corrector Type: wrench")
            self.corrector = WrenchCorrector(
                xml_path=Path(cro_ros_dir) / "mujoco/leap_rh_cube.xml",
                kp_pos=self.get_parameter("wrench_kp_pos").get_parameter_value().double_value,
                ki_pos=self.get_parameter("wrench_ki_pos").get_parameter_value().double_value,
                kd_pos=self.get_parameter("wrench_kd_pos").get_parameter_value().double_value,
                kp_rot=self.get_parameter("wrench_kp_rot").get_parameter_value().double_value,
                ki_rot=self.get_parameter("wrench_ki_rot").get_parameter_value().double_value,
                kd_rot=self.get_parameter("wrench_kd_rot").get_parameter_value().double_value,
                pass_v=self.get_parameter("wrench_pass_v_cube").get_parameter_value().bool_value,
                timestep=self.get_parameter("corrector_timestep").get_parameter_value().double_value,
            )

            # timer for stepping the wrench model forward
            self.wrench_cbg = MutuallyExclusiveCallbackGroup()
            self.timer_wrench_step = self.create_timer(
                self.get_parameter("corrector_timestep").get_parameter_value().double_value,
                callback=self.corrector.timer_callback,
                callback_group=self.wrench_cbg,
            )
            self.timer_wrench_step.cancel()
            self.obk_timers["timer_wrench"] = self.timer_wrench_step

        else:
            self.get_logger().warn(f"Invalid corrector type: {self.corrector_type}. Assuming no corrector!")

        # generic corrector parameters
        if self.corrector is not None:
            # whether to pass the corrector's q_leap and v_leap through to mjpc
            self.corrector_q_leap = self.get_parameter("corrector_q_leap").get_parameter_value().bool_value
            self.corrector_v_leap = self.get_parameter("corrector_v_leap").get_parameter_value().bool_value

        # timer for publishing the number of rotations as an rviz text overlay
        self.num_rotations_pub = self.create_publisher(OverlayText, "/cro/viz/num_rotations", 10, non_obelisk=True)
        self.num_rotations_timer = self.create_timer(0.1, self._publish_num_rotations)

        # stateful quantities
        self.q_leap: Optional[np.ndarray] = None
        self.v_leap: Optional[np.ndarray] = None
        self.t_leap_last: Optional[float] = None
        self.q_cube: Optional[np.ndarray] = None
        self.v_cube: Optional[np.ndarray] = None
        self.q_cube_last: Optional[np.ndarray] = None
        self.t_cube_last: Optional[float] = None

        self.q_leap_meas: Optional[np.ndarray] = None  # for visualization only

        self.is_configured = True
        self.get_logger().info("***********************************")
        self.get_logger().info("Cube Rotation Estimator Configured!")
        self.get_logger().info("***********************************")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the cube rotation estimator."""
        super().on_activate(state)

        # start the corrector timer
        if self.corrector is not None:
            self.corrector.reset_cube()
            if self.corrector_type == "wrench":
                self.timer_wrench_step.reset()

        return TransitionCallbackReturn.SUCCESS

    # ################ #
    # SENSOR CALLBACKS #
    # ################ #

    def smoother_callback(self) -> None:
        """Processes ZED images.

        See: https://github.com/Caltech-AMBER/obelisk/blob/main/obelisk_ws/src/obelisk_msgs/obelisk_sensor_msgs/msg/ObkImage.msg
        """
        if not self.is_configured:
            return

        if not all([cam.frame is not None for cam in self.cameras]):
            return

        images = np.stack([cam.frame for cam in self.cameras], axis=0)  # (num_cams, H_crop, W_crop, C)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)  # (num_cams, C, H_crop, W_crop)

        # estimator pass
        if "perseus" in self.estimator_type:
            q_cube_new, v_cube_new, estimator_reset = self.perseus_wrapper.compute_cube_estimates(images)

        else:
            err_msg = f"Invalid estimator type: {self.estimator_type}"
            self.get_logger().error(err_msg)
            raise ValueError(err_msg)

        # corrector pass
        if self.corrector is not None:
            # publish the ghost cube before corrections
            msg = EstimatedState()
            msg.base_link_name = "cube_ghost"
            msg.q_base = q_cube_new.tolist() if q_cube_new is not None else [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            self.obk_publishers["pub_q_cube_ghost"].publish(msg)

            # check for cube resets
            z_height_threshold = -0.1
            out_of_hand = self.corrector.q_cube[2] < z_height_threshold

            if out_of_hand or estimator_reset:
                self.corrector.reset_cube()
            q_cube_new, v_cube_new = self.corrector.correct(q_cube_new, v_cube_new)

        # update stateful quantities
        if self.v_cube is None:
            self.v_cube = v_cube_new
        else:
            self.v_cube = self.alpha_cube * v_cube_new + (1 - self.alpha_cube) * self.v_cube
        self.q_cube_last = self.q_cube
        self.q_cube = q_cube_new

    def q_leap_callback(self, msg: ObkJointEncoders) -> None:
        """Processes LEAP hand joint encoder readings.

        See: https://github.com/Caltech-AMBER/obelisk/blob/main/obelisk_ws/src/obelisk_msgs/obelisk_sensor_msgs/msg/ObkJointEncoders.msg
        """
        if not self.is_configured:
            return

        # retrieve reading from encoders
        q_leap_new = np.array(msg.joint_pos)
        v_leap_new = None
        self.q_leap_meas = q_leap_new

        # corrector pass
        if self.corrector is not None:
            # update the corrector's estimate of the leap hand
            self.corrector.set_q_leap_est(q_leap_new)

            # if desired, pass the corrector's q_leap and v_leap through to mjpc instead
            q_leap_new = self.corrector.q_leap if self.corrector_q_leap else q_leap_new
            v_leap_new = self.corrector.v_leap if self.corrector_v_leap else v_leap_new

        # if v_leap is not set from the corrector, compute a noisy estimate with finite differencing
        t = self.t
        if v_leap_new is None:
            if self.q_leap is not None and self.t_leap_last is not None:
                dt = t - self.t_leap_last
                v_leap_new = (q_leap_new - self.q_leap) / dt
            else:
                v_leap_new = np.zeros(16)

        # apply EMA filter to v_leap
        if self.v_leap is not None:
            v_leap_new = self.alpha_leap * v_leap_new + (1 - self.alpha_leap) * self.v_leap

        # update stateful quantities
        self.t_leap_last = t
        self.q_leap = np.copy(q_leap_new)
        self.v_leap = np.copy(v_leap_new)

    # ################### #
    # MAIN TIMER CALLBACK #
    # ################### #

    def compute_state_estimate(self) -> None:
        """Computes the state estimate.

        See: https://github.com/Caltech-AMBER/obelisk/blob/main/obelisk_ws/src/obelisk_msgs/obelisk_estimator_msgs/msg/EstimatedState.msg
        """
        # ########## #
        # FUNCTIONAL #
        # ########## #

        # publish estimated state of the leap hand using the default publisher (key "pub_est")
        if self.q_leap is not None and self.v_leap is not None:
            msg = EstimatedState()
            msg.base_link_name = "leap_mount"
            msg.q_joints = self.q_leap.tolist()
            msg.joint_names = self.leap_joint_names
            msg.v_joints = self.v_leap.tolist()
            self.obk_publishers["pub_est"].publish(msg)

        # publish estimated state of the cube using the cube-specific publisher (key "pub_q_cube")
        if self.q_cube is not None and self.v_cube is not None:
            msg = EstimatedState()
            msg.q_base = self.q_cube.tolist()
            msg.base_link_name = "cube"
            msg.v_base = self.v_cube.tolist()
            self.obk_publishers["pub_q_cube"].publish(msg)

        # ############# #
        # VISUALIZATION #
        # ############# #

        # visualize the leap hand encoder readings for both the measured and corrected
        if self.q_leap_meas is not None and self.corrector is not None:
            # ghost is the encoder readings
            msg = EstimatedState()
            msg.base_link_name = "leap_mount_ghost"
            msg.q_joints = self.q_leap_meas.tolist()
            msg.joint_names = self.leap_joint_names
            self.obk_publishers["pub_q_leap_ghost"].publish(msg)

            # solid is the corrected readings (misleading name of the pub key)
            msg = EstimatedState()
            msg.base_link_name = "leap_mount_meas"
            msg.q_joints = self.corrector.q_leap.tolist()
            msg.joint_names = self.leap_joint_names
            self.obk_publishers["pub_q_leap_meas"].publish(msg)

        # if there's no corrector, then the encoder readings are solid
        elif self.q_leap_meas is not None:
            msg = EstimatedState()
            msg.base_link_name = "leap_mount_meas"
            msg.q_joints = self.q_leap.tolist()
            msg.joint_names = self.leap_joint_names
            self.obk_publishers["pub_q_leap_meas"].publish(msg)

    # ##### #
    # UTILS #
    # ##### #

    def _compute_v_cube(self, q_cube: Optional[np.ndarray], q_cube_prev: Optional[np.ndarray]) -> np.ndarray:
        """Computes the cube velocity.

        Args:
            q_cube: The current cube pose in the form (x, y, z, quat_w, quat_x, quat_y, quat
            q_cube_prev: The last cube pose in the form (x, y, z, quat_w, quat_x, quat_y, quat_z).

        Returns:
            v_cube: The cube velocity in the form (vx, vy, vz, omega_x, omega_y, omega_z).
        """
        if q_cube is None:
            err_msg = "q_cube is None! You should never call this function with q_cube=None."
            self.get_logger().error(err_msg)
            raise ValueError(err_msg)

        t = self.t
        if self.t_cube_last is not None and q_cube_prev is not None:
            dt = t - self.t_cube_last
            self.t_cube_last = t

            # angular velocity
            qw, qx, qy, qz = q_cube[3:]
            omega = (2.0 / dt) * (
                np.array(
                    [
                        [qx, -qw, -qz, qy],
                        [qy, qz, -qw, -qx],
                        [qz, -qy, qx, -qw],
                    ]
                )
                @ q_cube_prev[3:]
            )

            # linear velocity
            v = (q_cube[:3] - q_cube_prev[:3]) / dt

            # concatenate
            v_cube = np.concatenate([v, omega])
        else:
            self.t_cube_last = t
            v_cube = np.zeros(6)

        return v_cube

    def _q_cube_goal_callback(self, msg: EstimatedState) -> None:
        """Callback for the cube goal pose."""
        new_q_cube_goal = np.array(msg.q_base)  # (7,) position and wxyz quaternion
        if self.q_cube_goal is None:
            self.q_cube_goal = new_q_cube_goal
            self.last_rotation_time = self.t

        if (new_q_cube_goal != self.q_cube_goal).any():
            t_curr = self.t
            rotation_time = t_curr - self.last_rotation_time

            self.rotation_times_file.write(f"{rotation_time} | {self.q_cube_goal[3:]}\n")
            self.rotation_times_file.flush()

            self.rotation_times.append(rotation_time)
            self.q_cube_goal = new_q_cube_goal
            self.last_rotation_time = t_curr

    def _compute_ang_error(self) -> str:
        """Computes the angular error between the cube and the goal pose."""
        if self.q_cube is None or self.q_cube_goal is None:
            e_cube_rad = "N/A"
        else:
            q_gco_conj = np.zeros(4)  # gco stands for goal cube orientation
            q_diff = np.zeros(4)
            mju_negQuat(q_gco_conj, self.q_cube_goal[3:])
            mju_mulQuat(q_diff, self.q_cube[3:], q_gco_conj)
            mju_normalize4(q_diff)
            if q_diff[0] < 0.0:
                q_diff *= -1.0
            e_cube_rad = f"{2.0 * np.arccos(q_diff[0]):.3f}"
        return e_cube_rad

    def _publish_num_rotations(self) -> None:
        """Publishes the number of rotations as an rviz text overlay."""
        msg = OverlayText()

        if len(self.rotation_times) > 0:
            num_rots = f"{len(self.rotation_times)}"
            curr_time = f"{self.t - self.last_rotation_time:.2f}"
            avg_time = f"{np.mean(self.rotation_times):.2f}"
            std_dev_time = f"{np.std(self.rotation_times):.2f}"
            median_time = f"{np.median(self.rotation_times):.2f}"
            iqr_25_time = f"{np.percentile(self.rotation_times, 25):.2f}"
            iqr_75_time = f"{np.percentile(self.rotation_times, 75):.2f}"
            min_time = f"{np.min(self.rotation_times):.2f}"
            max_time = f"{np.max(self.rotation_times):.2f}"
            num_timeouts = f"{len([t for t in self.rotation_times if t > 80.0])}"  # noqa: PLR2004
            e_cube_rad = self._compute_ang_error()
        else:
            num_rots = 0
            curr_time = f"{self.t - self.last_rotation_time:.2f}" if self.last_rotation_time is not None else "N/A"
            avg_time = "N/A"
            std_dev_time = "N/A"
            median_time = "N/A"
            iqr_25_time = "N/A"
            iqr_75_time = "N/A"
            min_time = "N/A"
            max_time = "N/A"
            num_timeouts = "N/A"
            e_cube_rad = self._compute_ang_error() if self.last_rotation_time is not None else "N/A"

        msg.text = (
            f"Num Rots: {num_rots} | Num Timeouts: {num_timeouts} | Curr Time: {curr_time} | "
            f"Angle Error: {e_cube_rad} | Avg Time: {avg_time}+/-{std_dev_time} | "
            f"Median Time: {median_time} ({iqr_25_time}, {iqr_75_time}) | "
            f"Min/Max Time: {min_time}/{max_time}"
        )
        msg.text_size = 20.0
        msg.fg_color.r = 1.0  # white
        msg.fg_color.g = 1.0
        msg.fg_color.b = 1.0
        msg.fg_color.a = 1.0
        msg.width = 2048
        msg.height = 128
        msg.horizontal_alignment = OverlayText.LEFT
        msg.vertical_alignment = OverlayText.TOP
        self.num_rotations_pub.publish(msg)

    def __del__(self) -> None:
        """Destructor."""
        if hasattr(self, "rotation_times_file"):
            self.rotation_times_file.close()


def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    # TODO(ahl): if we want to compile the model, we have to use a single-threaded executor!
    # spin_obelisk(args, ObeliskCubeRotationEstimator, SingleThreadedExecutor)

    # [NOTE] MultiThreadedExecutor does not work with torch.compiled models! For now, don't compile.
    # see: https://github.com/pytorch/pytorch/issues/123177
    spin_obelisk(args, ObeliskCubeRotationEstimator, MultiThreadedExecutor)


if __name__ == "__main__":
    main()
