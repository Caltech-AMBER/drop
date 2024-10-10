import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817


def transform_matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """Transforms a 4x4 homogeneous transformation matrix to a 7D pose vector.

    Args:
        T (np.ndarray): 4x4 homogeneous transformation matrix.

    Returns:
        np.ndarray: 7D pose vector (x, y, z, qw, qx, qy, qz).
    """
    pos = T[:3, 3]
    rot = T[:3, :3]

    rot_scipy = R.from_matrix(rot)
    quat_xyzw = rot_scipy.as_quat()  # xyzw
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    pose = np.hstack((pos, quat_wxyz))
    return pose


if __name__ == "__main__":
    """Description of this script.

    This script converts the transformation matrix of the cameras to a 7D pose vector, which is fed into the
    cube rotation estimator. The transformations are computed from the solidworks assembly.

    The ZED2i optical frames corresponding to the origin of the pinhole camera model are placed concentric with the
    camera optical axis on the physical zed camera and on the front plane, flush with the front face of the camera.
    This location was chosen by empirically observing the cube estimates (other possible choices could have placed the
    camera anywhere along the same axis, not necessarily at the front face of the camera).

    For example, this forum post puts the optical frame about 20mm behind the front panel of the left lens:
    https://community.stereolabs.com/t/exact-location-of-the-origin-in-the-back-of-the-left-eye/2932/2

    Similarly, we also place the optical frame origin of the ZED mini cameras at the front face of the camera, in
    particular at the front of the cylinder containing the left lens. This is in opposition to a forum post from 2022
    explaining that the ZED mini has its origin about 3mm away from the back plane:
    https://community.stereolabs.com/t/location-of-the-origin-of-the-reference-frame-of-the-left-camera-sensor/1702

    To come to this conclusion, I checked the reported depth measurements of the camera against certain pixels in the
    scene where I could either measure in the CAD or do some rough measurements using a measuring tape in the real
    world. In particular, I checked the distance of the target points with respect to the plane containing the front
    cylinder faces of the ZED minis, and determined that it seemed most accurately located at the front. I also
    repeated this process for two different ZED minis mounted in the same location.
    """

    # ###### #
    # ZED 2i #
    # ###### #

    # zed2i camera 1
    # [0.00858148, 0.14786571, 0.12599401, 0.42470821, 0.82047324, 0.33985114, 0.1759199]
    T1 = np.array(
        [
            [np.sqrt(2.0) / 2.0, 0.4082483, 0.5773503, 0.008581479],
            [np.sqrt(2.0) / 2.0, -0.4082483, -0.5773503, 0.14786571],
            [0.0, 0.8164966, -0.5773503, 0.12599401],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose1 = transform_matrix_to_pose(T1)
    print(f"zed2i_1 pose: {pose1}")

    # zed2i camera 2
    # [0.00858148, -0.14786571, 0.12599401, 0.11957315, 0.93810438, -0.1494292, 0.28867515]
    T2 = np.array(
        [
            [-np.sqrt(2.0) / 2.0, 0.4082483, 0.5773503, 0.008581479],
            [np.sqrt(2.0) / 2.0, 0.4082483, 0.5773503, -0.14786571],
            [0.0, 0.8164966, -0.5773503, 0.12599401],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose2 = transform_matrix_to_pose(T2)
    print(f"zed2i_2 pose: {pose2}")

    # zed2i camera 3
    # [0.4319421, 0.0, 0.2146199, 0.35355339, 0.61237244, -0.61237244, -0.35355339]
    T3 = np.array(
        [
            [0.0, -0.5, -np.sqrt(3) / 2, 0.4319421],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, np.sqrt(3) / 2, -0.5, 0.2146199],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose3 = transform_matrix_to_pose(T3)
    print(f"zed2i_3 pose: {pose3}")

    # ######## #
    # ZED MINI #
    # ######## #

    # zed mini camera 1
    # [0.1205, 0.1472, 0.2126, 0.25881905, 0.96592583, 0.0, 0.0]
    TA = np.array(
        [
            [1.0, 0.0, 0.0, 0.1205],
            [0.0, -np.sqrt(3) / 2.0, -0.5, 0.1472],
            [0.0, 0.5, -np.sqrt(3) / 2.0, 0.2126],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    poseA = transform_matrix_to_pose(TA)  # noqa: N816
    print(f"zed_mini_1 pose: {poseA}")

    # zed mini camera 2
    # [0.1205, -0.1472, 0.2126, -0.25881905, 0.96592583, 0.0, 0.0]
    TB = np.array(
        [
            [1.0, 0.0, 0.0, 0.1205],
            [0.0, -np.sqrt(3) / 2.0, 0.5, -0.1472],
            [0.0, -0.5, -np.sqrt(3) / 2.0, 0.02126],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    poseB = transform_matrix_to_pose(TB)  # noqa: N816
    print(f"zed_mini_2 pose: {poseB}")

    # zed mini camera 3
    # [0.40313696, 0.0, 0.19798922, -0.35355339, 0.61237244, 0.61237244, -0.35355339]
    TC = np.array(
        [
            [0.0, 0.5, -np.sqrt(3) / 2, 0.40313696],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -np.sqrt(3) / 2, -0.5, 0.19798922],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    poseC = transform_matrix_to_pose(TC)  # noqa: N816
    print(f"zed_mini_3 pose: {poseC}")
