import os
from typing import Dict, List, Tuple

import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import EmitEvent, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.substitutions import FindPackageShare
from rclpy.impl import rcutils_logger

logger = rcutils_logger.RcutilsLogger(name="cro_bringup")


def launch_args_setup(context: launch.LaunchContext, *args: List, **kwargs: Dict) -> Tuple[List, Dict]:
    """Sets up all launch arguments at once."""
    launch_actions = []
    launch_args = {}
    return launch_actions, launch_args


def obk_setup(context: launch.LaunchContext, launch_args: Dict) -> List:
    """Returns the launch actions associated with the Obelisk stack."""
    launch_actions = []

    # calling the obelisk bringup launch file
    cro_root = os.environ.get("CUBE_ROTATION_OBELISK_ROOT")
    obelisk_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("obelisk_ros"), "launch", "obelisk_bringup.launch.py"])]
        ),
        launch_arguments={
            "config_file_path": f"{cro_root}/cube_rotation_ws/src/cro_ros/config/hardware.yaml",
            "device_name": "onboard",
            "auto_start": "configure",
            "bag": "False",
        }.items(),
    )
    launch_actions += [obelisk_bringup_launch]

    return launch_actions


def cro_setup(context: launch.LaunchContext, launch_args: Dict) -> List:
    """Returns the launch actions associated with the CRO stack."""
    launch_actions = []

    # global state node in obelisk
    global_state_node = LifecycleNode(
        namespace="", package="obelisk_ros", executable="global_state", name="cro", output="both"
    )

    # if the global state node shuts down, so does this launch file
    shutdown_event = EmitEvent(event=launch.events.Shutdown())
    shutdown_event_handler = RegisterEventHandler(
        launch_ros.event_handlers.on_state_transition.OnStateTransition(
            target_lifecycle_node=global_state_node,
            goal_state="finalized",
            entities=[shutdown_event],
        )
    )  # when the global state node enters its shutdown state, the launch file also shuts down
    launch_actions += [shutdown_event_handler]

    return launch_actions


def launch_setup(context: launch.LaunchContext, *args: List, **kwargs: Dict) -> List:
    """Collates all the launch actions for each part of the pipeline."""
    launch_actions, launch_args = launch_args_setup(context, *args, **kwargs)
    launch_actions += obk_setup(context, launch_args)
    launch_actions += cro_setup(context, launch_args)
    return launch_actions


def generate_launch_description() -> LaunchDescription:
    """Generates the launch description.

    Uses the OpaqueFunction to allow for analyzing and manipulating the
    launch arguments that get passed in before running the relevant actions.
    """
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
