from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from mujoco import MjData, MjModel, mj_forward, mj_step, mju_subQuat


class Corrector(ABC):
    """Base class for cube state correctors."""

    def __init__(self, xml_path: Union[str, Path], timestep: float = 0.002) -> None:
        """Initializes the corrector.

        Assumes that the first 7 states correspond to the cube pose in the order [x, y, z, qw, qx, qy, qz].

        Args:
            xml_path: The path to the MuJoCo XML file.
            timestep: The timestep of the simulation.
        """
        self.mj_model = MjModel.from_xml_path(str(xml_path))
        self.qpos_home = np.array(
            [
                0.1, 0.0, 0.035, 1.0, 0.0, 0.0, 0.0,  # cube
                0.5, -0.75, 0.75, 0.25, 0.5, 0.0, 0.75, 0.25, 0.5, 0.75, 0.75, 0.25, 0.65, 0.9, 0.75, 0.6,  # leap
            ]
        )  # fmt: skip
        self.mj_model.opt.timestep = timestep

        self.mj_data = MjData(self.mj_model)
        self.mj_data.qpos[:] = self.qpos_home
        mj_forward(self.mj_model, self.mj_data)

        self.q_leap_est = None

    @property
    def q_cube(self) -> np.ndarray:
        """Returns the cube pose in the order [x, y, z, qw, qx, qy, qz]."""
        return self.mj_data.qpos[:7]

    @property
    def v_cube(self) -> np.ndarray:
        """Returns the cube velocity in the order [vx, vy, vz, wx, wy, wz]."""
        return self.mj_data.qvel[:6]

    @property
    def q_leap(self) -> np.ndarray:
        """Returns the leap configuration."""
        return self.mj_data.qpos[7:]

    @property
    def v_leap(self) -> np.ndarray:
        """Returns the leap velocity."""
        return self.mj_data.qvel[6:]

    def set_q_leap_est(self, q_leap_est: np.ndarray) -> None:
        """Sets the estimate of the leap pose used as setpoint for the corrector."""
        self.q_leap_est = q_leap_est

    def reset_cube(self) -> None:
        """Resets the cube to its home pose."""
        self.mj_data.qpos[:7] = self.qpos_home[:7]
        self.mj_data.qvel[:6] = 0.0
        mj_forward(self.mj_model, self.mj_data)

    @abstractmethod
    def correct(self, q_cube: np.ndarray, v_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Corrects the cube pose.

        Args:
            q_cube: The uncorrected cube pose in the order [x, y, z, qw, qx, qy, qz].
            v_cube: The uncorrected cube velocity in the order [vx, vy, vz, wx, wy, wz].

        Returns:
            q_cube_corr: The corrected cube pose in the order [x, y, z, qw, qx, qy, qz].
            v_cube_corr: The corrected cube velocity in the order [vx, vy, vz, wx, wy, wz].
        """


class WrenchCorrector(Corrector):
    """Corrector that applies a wrench to the cube to correct its pose.

    This class implements the method `timer_callback`, which is meant to be called by a ROS2 timer to update the
    internal state of the corrector at a fixed rate in a simulation parallel to the actual physics of the system.
    """

    def __init__(
        self,
        xml_path: Path,
        kp_pos: float,
        ki_pos: float,
        kd_pos: float,
        kp_rot: float,
        ki_rot: float,
        kd_rot: float,
        pass_v: bool,
        timestep: float = 0.002,
    ) -> None:
        """Initializes the corrector.

        Args:
            xml_path: The path to the MuJoCo XML file.
            kp_pos: The proportional gain for the position error.
            ki_pos: The integral gain for the position error.
            kd_pos: The derivative gain for the position error.
            kp_rot: The proportional gain for the rotation error.
            ki_rot: The integral gain for the rotation error.
            kd_rot: The derivative gain for the rotation error.
            pass_v: Whether to pass the velocity through or not.
            timestep: The timestep of the simulation.
        """
        super().__init__(xml_path, timestep=timestep)

        # corrector gains
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_rot = kp_rot
        self.ki_rot = ki_rot
        self.kd_rot = kd_rot

        # integrated errors
        self.ei_pos = np.zeros(3)
        self.ei_rot = np.zeros(3)

        # other
        self.pass_v = pass_v

        # estimates of the cube pose and velocity used as setpoints for the corrector
        self.q_cube_est = None
        self.v_cube_est = None
        self.q_leap_est = None

    def correct(self, q_cube: np.ndarray, v_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Corrects the cube pose by applying a wrench to the cube to correct its pose."""
        self.q_cube_est = q_cube
        self.v_cube_est = v_cube
        if self.pass_v:
            return self.q_cube, v_cube  # passthrough v_cube
        else:
            return self.q_cube, self.v_cube  # return internal v_cube

    def timer_callback(self) -> None:
        """Applies a wrench to the cube to correct its pose."""
        if self.q_cube_est is not None and self.v_cube_est is not None:
            # apply a wrench to the cube to correct its pose
            q_cube = self.q_cube
            v_cube = self.v_cube

            # compute errors
            e_pos = q_cube[:3] - self.q_cube_est[:3]
            self.ei_pos += e_pos
            e_v = v_cube[:3] - self.v_cube_est[:3]

            e_rot = np.zeros(3)
            mju_subQuat(e_rot, q_cube[3:], self.q_cube_est[3:])
            self.ei_rot += e_rot
            e_w = v_cube[3:] - self.v_cube_est[3:]

            # compute artificial applied forces
            posfrc = -self.kp_pos * e_pos - self.ki_pos * self.ei_pos - self.kd_pos * e_v
            posfrc -= np.array([0.0, 0.0, 10.0])  # [HACK] heuristic gravity compensation
            rotfrc = -self.kp_rot * e_rot - self.ki_rot * self.ei_rot - self.kd_rot * e_w
            qfrc_cube = np.concatenate((posfrc, rotfrc))

            # step the model forward, return the resulting cube pose
            self.mj_data.ctrl[:] = self.q_leap_est  # set q_leap
            self.mj_data.qfrc_applied[:6] = qfrc_cube
            mj_step(self.mj_model, self.mj_data)
