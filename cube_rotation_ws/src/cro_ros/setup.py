import os
from glob import glob
from warnings import simplefilter

from setuptools import SetuptoolsDeprecationWarning, setup

simplefilter("ignore", category=SetuptoolsDeprecationWarning)

package_name = "cro_ros"


def generate_data_files(share_path: str, dir: str) -> list:
    """Generate a list of data files to be included in the package.

    Args:
        share_path (str): The path to the share directory.
        dir (str): The path to the directory containing the data files.

    Returns:
        list: A list of data files to be included in the package
    """
    data_files = []

    for path, _, files in os.walk(dir):
        list_entry = (share_path + path, [os.path.join(path, f) for f in files if not f.startswith(".")])
        data_files.append(list_entry)

    return data_files


setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "rviz"), glob(os.path.join("rviz", "*.rviz"))),
        (os.path.join("share", package_name, "ckpts"), glob(os.path.join("ckpts", "*.pth"))),
        (os.path.join("share", package_name, "meshes"), glob(os.path.join("meshes", "*.obj"))),
        (os.path.join("share", package_name, "mujoco"), glob(os.path.join("mujoco", "*.xml"))),
        (os.path.join("share", package_name, "mujoco", "assets"), glob(os.path.join("mujoco/assets", "*.png"))),
    ]
    + generate_data_files("share/" + package_name + "/", "urdf"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="alberthli",
    maintainer_email="alberthli@caltech.edu",
    description="Cube Rotation Obelisk state estimation package.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
