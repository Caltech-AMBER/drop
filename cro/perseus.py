import re
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import gtsam
import gtsam_unstable
import kornia
import numpy as np
import torch
from gtsam.symbol_shorthand import V, W, X
from perseus.detector.models import KeypointCNN
from perseus.smoother.factors import ConstantVelocityFactor, KeypointProjectionFactor, PoseDynamicsFactor
from rclpy.impl import rcutils_logger
from ruamel.yaml import YAML

from cro.utils import _torch_compile_warmup

logger = rcutils_logger.RcutilsLogger(name="debug")

UNIT_CUBE_KEYPOINTS = [
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
]


class PerseusWrapper:
    """A wrapper for the perseus smoother."""

    def __init__(
        self,
        weights_path: Path,
        zed_params_path: Path,
        num_imgs: int,
        C: int,
        H: int,
        W: int,
        device: Union[str, torch.device],
        smoother_freq: float,
        lookback: int,
        depth: bool = True,
        compile: bool = False,
        cam_z_adjust: float = 0.0,
        k_huber: float = 1.345,
        # parameters that are manually adjusted
        cam_A_trans_adjust: Optional[np.ndarray] = None,
        cam_B_trans_adjust: Optional[np.ndarray] = None,
        cam_C_trans_adjust: Optional[np.ndarray] = None,
        q_cube_trans_adjust: Optional[np.ndarray] = None,
    ) -> None:
        """Initializes the Perseus smoother."""
        self.weights_path = weights_path
        self.zed_params_path = zed_params_path
        self.num_imgs = num_imgs
        self.C = C
        self.H = H
        self.W = W
        self.device = device
        self.smoother_freq = smoother_freq
        self.lookback = lookback
        self.depth = depth
        self.compile = compile
        self.cam_z_adjust = cam_z_adjust
        self.k_huber = k_huber

        # manual adjustments
        self.cam_A_trans_adjust = cam_A_trans_adjust if cam_A_trans_adjust is not None else np.zeros(3)
        self.cam_B_trans_adjust = cam_B_trans_adjust if cam_B_trans_adjust is not None else np.zeros(3)
        self.cam_C_trans_adjust = cam_C_trans_adjust if cam_C_trans_adjust is not None else np.zeros(3)
        self.q_cube_trans_adjust = q_cube_trans_adjust if q_cube_trans_adjust is not None else np.zeros(3)

        # computation cache
        self._keypoint_factors = None
        self._projected_keypoint_pixels = None
        self._keypoints = None

        self._setup()

    def _setup(self) -> None:
        """Sets up the Perseus smoother."""
        # the camera poses for corners A and B on the physical setup
        self.cam_pose_A = np.array([0.1205, 0.1472, 0.2126, 0.25881905, 0.96592583, 0.0, 0.0])
        self.cam_pose_B = np.array([0.1205, -0.1472, 0.2126, -0.25881905, 0.96592583, 0.0, 0.0])
        self.cam_pose_C = np.array([0.40313696, 0.0, 0.19798922, -0.35355339, 0.61237244, 0.61237244, -0.35355339])

        # manual adjustments
        self.cam_pose_A[:3] += self.cam_A_trans_adjust
        self.cam_pose_B[:3] += self.cam_B_trans_adjust
        self.cam_pose_C[:3] += self.cam_C_trans_adjust

        if self.cam_z_adjust != 0.0:
            z_axis_A = np.array([0.0, -0.5, -np.sqrt(3) / 2.0])  # noqa: N806
            z_axis_B = np.array([0.0, 0.5, -np.sqrt(3) / 2.0])  # noqa: N806
            z_axis_C = np.array([-np.sqrt(3) / 2, 0.0, -0.5])  # noqa: N806
            self.cam_pose_A[:3] += self.cam_z_adjust * z_axis_A
            self.cam_pose_B[:3] += self.cam_z_adjust * z_axis_B
            self.cam_pose_C[:3] += self.cam_z_adjust * z_axis_C

        # smoother priors
        prior_pose_mean_pos = np.array([0.1, 0.0, 0.0])
        prior_pose_mean_rot = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion, wxyz
        self.prior_pose_mean = gtsam.Pose3(gtsam.Rot3(*prior_pose_mean_rot), prior_pose_mean_pos)

        prior_pose_std_diag = np.array(3 * [0.5] + 3 * [1.0])  # stdev of prior pose noise in meters and radians
        self.prior_pose_std = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber), gtsam.noiseModel.Diagonal.Sigmas(prior_pose_std_diag)
        )

        self.prior_tvel_mean = np.zeros(3)  # mean translational and angular velocities
        self.prior_avel_mean = np.zeros(3)
        self.prior_tvel_std = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber),
            gtsam.noiseModel.Diagonal.Sigmas(np.array(3 * [0.01])),  # stdev of prior velocity noise
        )
        self.prior_avel_std = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber),
            gtsam.noiseModel.Diagonal.Sigmas(np.array(3 * [0.2])),
        )

        # smoother noise models
        self.keypoint_px_stdevs = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([3.0, 3.0])),  # stdev in pixels
        )

        cov_pose = np.array(3 * [0.01] + 3 * [0.2])
        self.Q_pose = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber),
            gtsam.noiseModel.Diagonal.Sigmas(cov_pose),  # stdev of pose noise, m & rad
        )

        cov_vels = np.array(3 * [0.01] + 3 * [0.2])  # stdev of velocity noise in m/s and rad/s
        self.Q_tvel = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber),
            gtsam.noiseModel.Diagonal.Sigmas(cov_vels[:3]),
        )
        self.Q_avel = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(self.k_huber),
            gtsam.noiseModel.Diagonal.Sigmas(cov_vels[3:]),
        )

        # integrating camera calibration info with smoother
        self._calibrate_gtsam_cameras(self.zed_params_path)

        # initializing the smoother
        self._init_smoother()

        # loading the model
        if self.depth:
            model = KeypointCNN(num_channels=4)
        else:
            model = KeypointCNN()
        state_dict = torch.load(self.weights_path, weights_only=True)
        for key in list(state_dict.keys()):
            if "module." in key:
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # JIT compiling the model
        if self.compile:
            model = torch.compile(model, mode="reduce-overhead")
            warmup_tensor = torch.rand((self.num_imgs, self.C, self.H, self.W), device=self.device)
            _torch_compile_warmup(model, warmup_tensor)
        self.model = model

    def _calibrate_gtsam_cameras(self, zed_params_path: Path) -> None:
        """Calibrates the gtsam camera objects for the Perseus estimator."""
        cam_params = YAML().load(zed_params_path)

        self.gtsam_cams = []
        self.gtsam_cam_poses = []
        self.calibrations = []
        for i in range(self.num_imgs):
            # parsing camera parameters
            serial_number = cam_params[i]["serial_number"]
            resolution = cam_params[i]["resolution"]
            side = cam_params[i]["side"]

            if resolution.lower() == "vga":
                suffix = "VGA"
            elif resolution.lower() == "720":
                suffix = "HD"
            elif resolution.lower() == "1080":
                suffix = "FHD"
            elif resolution.lower() == "2k":
                suffix = "2K"
            else:
                raise ValueError(f"Invalid resolution: {resolution}! Must be one of [VGA, 720, 1080, 2K]")

            if side.lower() == "left":
                prefix = "LEFT"
            elif side.lower() == "right":
                prefix = "RIGHT"
            else:
                raise ValueError(f"Invalid side: {side}! Must be one of [left, right]")

            # retrieving factory camera calibration info
            settings_file = f"/usr/local/zed/settings/SN{serial_number}.conf"
            with open(settings_file, "r") as f:
                settings_text = f.read()
                section_heading = f"{prefix}_CAM_{suffix}"
                section_pattern = rf"\[{re.escape(section_heading)}\](.*?)\n\s*\["
                match = re.search(section_pattern, settings_text, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    calibration_params = {}
                    for line in section_content.splitlines():
                        key, value = line.split("=")
                        calibration_params[key.strip()] = float(value.strip())
                else:
                    raise ValueError(f"Could not find section [{section_heading}] in {settings_file}!")

            fx, fy = calibration_params["fx"], calibration_params["fy"]
            cx, cy = self.W / 2, self.H / 2
            s = 0.0
            calibration = gtsam.Cal3_S2(fx, fy, s, cx, cy)
            self.calibrations.append(calibration)
            if i == 0:
                cam_pose = gtsam.Pose3(gtsam.Rot3(*self.cam_pose_A[3:]), self.cam_pose_A[:3])
            elif i == 1:
                cam_pose = gtsam.Pose3(gtsam.Rot3(*self.cam_pose_B[3:]), self.cam_pose_B[:3])
            elif i == 2:  # noqa: PLR2004
                cam_pose = gtsam.Pose3(gtsam.Rot3(*self.cam_pose_C[3:]), self.cam_pose_C[:3])
            else:
                raise ValueError(f"Invalid camera index: {i}!")
            self.gtsam_cam_poses.append(cam_pose)
            self.gtsam_cams.append(gtsam.PinholeCameraCal3_S2(cam_pose, calibration))

    def _init_smoother(self) -> None:
        """Initializes the perseus smoother."""
        # scaling keypoints
        cube_size = 0.07  # side length of the cube in meters
        self.object_frame_keypoints = np.array(UNIT_CUBE_KEYPOINTS) * (cube_size / 2.0)

        # smoother setup
        lag = self.lookback / self.smoother_freq  # lag of the smoother in seconds
        lm_params = gtsam.LevenbergMarquardtParams()
        lm_params.setRelativeErrorTol(1e-3)  # default: 1e-5
        lm_params.setAbsoluteErrorTol(1e-3)  # default: 1e-5
        self.smoother = gtsam_unstable.BatchFixedLagSmoother(lag, lm_params)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.new_timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        # add the priors to the graph
        self.new_factors.push_back(gtsam.PriorFactorPose3(X(0), self.prior_pose_mean, self.prior_pose_std))
        self.new_factors.push_back(gtsam.PriorFactorVector(V(0), self.prior_tvel_mean, self.prior_tvel_std))
        self.new_factors.push_back(gtsam.PriorFactorVector(W(0), self.prior_avel_mean, self.prior_avel_std))

        # add initial values to the graph
        self.new_values.insert(X(0), self.prior_pose_mean)
        self.new_values.insert(V(0), self.prior_tvel_mean)
        self.new_values.insert(W(0), self.prior_avel_mean)

        # counters used for the dt/iteration
        self.last_smoother_time = time.time()
        self.smoother_iter = 0

        # add the initial timestamp
        self.new_timestamps.insert((X(0), self.last_smoother_time))
        self.new_timestamps.insert((V(0), self.last_smoother_time))
        self.new_timestamps.insert((W(0), self.last_smoother_time))

        # update the graph and store initial results
        self.smoother.update(self.new_factors, self.new_values, self.new_timestamps)
        self.result = self.smoother.calculateEstimate()
        self.new_timestamps.clear()
        self.new_factors.resize(0)
        self.new_values.clear()

    def compute_cube_estimates(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Processes ZED2 images using the Perseus estimator into cube poses.

        Args:
            images: The images in the form (num_imgs, C, H, W).

        Returns:
            q_cube: The cube pose in the form (x, y, z, quat_w, quat_x, quat_y, quat_z).
            v_cube: The cube velocity in the form (tvel_x, tvel_y, tvel_z, avel_x, avel_y, avel_z).
            reset: Whether the smoother was reset.
        """
        images = images[:, : self.C, :, :]  # the image stack may include additional channels, so we index them out

        # query CNN for the predicted keypoints in pixel space
        with torch.no_grad():
            raw_pixel_coordinates = self.model(images).reshape(-1, self.model.n_keypoints, 2).detach()
            keypoints = kornia.geometry.denormalize_pixel_coordinates(raw_pixel_coordinates, self.H, self.W).cpu()

        # preparing to add relevant factors
        smoother_time = time.time()
        dt = smoother_time - self.last_smoother_time
        self.smoother_iter += 1
        self.new_timestamps.insert((X(self.smoother_iter), smoother_time))
        self.new_timestamps.insert((V(self.smoother_iter), smoother_time))
        self.new_timestamps.insert((W(self.smoother_iter), smoother_time))
        self.last_smoother_time = smoother_time

        # add keypoint factors associated with each camera measurement
        self._keypoint_factors = [[] for _ in range(self.num_imgs)]
        for i in range(self.num_imgs):
            # compute factors for each camera
            cam_pose = self.gtsam_cam_poses[i]
            for j, keypoint in enumerate(keypoints[i]):
                keypoint_measurement = keypoint.numpy()
                point_body_frame = self.object_frame_keypoints[j]
                keypoint_factor = KeypointProjectionFactor(
                    X(self.smoother_iter),
                    self.keypoint_px_stdevs,
                    self.calibrations[i],
                    keypoint_measurement,
                    point_body_frame,
                    camera_pose=cam_pose,
                )
                self._keypoint_factors[i].append(keypoint_factor)
                self.new_factors.push_back(keypoint_factor)

        # add dynamics factors
        self.new_factors.push_back(
            PoseDynamicsFactor(
                X(self.smoother_iter - 1),
                W(self.smoother_iter - 1),
                V(self.smoother_iter - 1),
                X(self.smoother_iter),
                self.Q_pose,
                dt,
            )
        )  # pose dynamics
        self.new_factors.push_back(
            ConstantVelocityFactor(V(self.smoother_iter - 1), V(self.smoother_iter), self.Q_tvel)
        )  # constant velocity factor
        self.new_factors.push_back(
            ConstantVelocityFactor(
                W(self.smoother_iter - 1),
                W(self.smoother_iter),
                self.Q_avel,
            )
        )  # constant angular velocity factor

        # update initial values
        pred_vel = np.concatenate(
            [
                self.result.atVector(V(self.smoother_iter - 1)),
                self.result.atVector(W(self.smoother_iter - 1)),
            ]
        )
        self.new_values.insert(
            X(self.smoother_iter),
            self.result.atPose3(X(self.smoother_iter - 1)).expmap((1 / self.smoother_freq) * pred_vel),
        )
        self.new_values.insert(
            V(self.smoother_iter),
            self.result.atVector(V(self.smoother_iter - 1)),
        )
        self.new_values.insert(
            W(self.smoother_iter),
            self.result.atVector(W(self.smoother_iter - 1)),
        )

        # update the smoother
        try:
            self.smoother.update(self.new_factors, self.new_values, self.new_timestamps)

            # caching intermediates
            self._projected_keypoint_pixels = np.array(
                [[factor.pixel for factor in keypoint_factor] for keypoint_factor in self._keypoint_factors]
            )  # (num_imgs, n_keypoints, 2)
            self._keypoints = keypoints.numpy()  # (num_imgs, n_keypoints, 2)

            reset = False
        except RuntimeError as e:
            # this is thrown when the smoother diverges and thinks the cube is behind a camera
            if e.args[0] == "CheiralityException":
                self._init_smoother()  # reinitialize the smoother to its priors
                reset = True
            else:
                raise e

        self.result = self.smoother.calculateEstimate()
        self.new_timestamps.clear()
        self.new_factors.resize(0)
        self.new_values.clear()

        # get the cube pose
        _cube_pose = self.result.atPose3(X(self.smoother_iter))
        _rot = _cube_pose.rotation().matrix()
        _rot_kubric_mjpc = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        )  # kubric accidentally generated all data y up, this converts it to z up
        rot = gtsam.Rot3(_rot @ _rot_kubric_mjpc)  # correcting for y up convention (kubric) into z up convention (mjpc)
        quat = rot.toQuaternion()
        q_cube = np.array(_cube_pose.translation().tolist() + [quat.w(), quat.x(), quat.y(), quat.z()])

        # manual adjustments
        q_cube[:3] += self.q_cube_trans_adjust

        # get the cube velocity
        cube_tvel = self.result.atVector(V(self.smoother_iter))
        cube_avel = self.result.atVector(W(self.smoother_iter))
        v_cube = np.concatenate([cube_tvel, cube_avel])
        return q_cube, v_cube, reset
