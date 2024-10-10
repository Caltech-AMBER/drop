import sys

import numpy as np
from sensor_msgs.msg import Image

"""This code taken from https://github.com/Box-Robotics/ros2_numpy.

Relevant parts copied out because installing the package would introduce some unwanted overhead, since it must be
installed with colcon.
"""

_to_numpy = {}
_from_numpy = {}


def converts_to_numpy(msgtype: type, plural: bool = False) -> callable:
    """Returns a decorator that registers a function to convert a ROS message to a numpy array."""

    def decorator(f: callable) -> callable:
        _to_numpy[msgtype, plural] = f
        return f

    return decorator


def converts_from_numpy(msgtype: type, plural: bool = False) -> callable:
    """Returns a decorator that registers a function to convert a numpy array to a ROS message."""

    def decorator(f: callable) -> callable:
        _from_numpy[msgtype, plural] = f
        return f

    return decorator


name_to_dtypes = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),
    # OpenCV CvMat types
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4),
}


@converts_to_numpy(Image)
def image_to_numpy(msg: Image) -> np.ndarray:
    """Converts a sensor_msgs/Image message to a numpy array.

    Args:
        msg: The Image message to convert.

    Returns:
        The Image message as a numpy array, shape=(H, W, C).
    """
    if msg.encoding not in name_to_dtypes:
        raise TypeError("Unrecognized encoding {}".format(msg.encoding))

    dtype_class, channels = name_to_dtypes[msg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    shape = (msg.height, msg.width, channels)

    data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
    data.strides = (msg.step, dtype.itemsize * channels, dtype.itemsize)

    if channels == 1:
        data = data[..., 0]
    return data


@converts_from_numpy(Image)
def numpy_to_image(arr: np.ndarray, encoding: str) -> Image:
    """Converts a numpy array to a sensor_msgs/Image message.

    Args:
        arr: The numpy array to convert.
        encoding: The encoding of the image.

    Returns:
        The numpy array as a Image message.
    """
    if encoding not in name_to_dtypes:
        raise TypeError("Unrecognized encoding {}".format(encoding))

    im = Image(encoding=encoding)

    # extract width, height, and channels
    dtype_class, exp_channels = name_to_dtypes[encoding]
    if len(arr.shape) == 2:  # noqa: PLR2004
        im.height, im.width, channels = arr.shape + (1,)
    elif len(arr.shape) == 3:  # noqa: PLR2004
        im.height, im.width, channels = arr.shape
    else:
        raise TypeError("Array must be two or three dimensional")

    # check type and channels
    if exp_channels != channels:
        raise TypeError("Array has {} channels, {} requires {}".format(channels, encoding, exp_channels))
    if dtype_class != arr.dtype.type:
        raise TypeError("Array is {}, {} requires {}".format(arr.dtype.type, encoding, dtype_class))

    # make the array contiguous in memory, as mostly required by the format
    # Only do this if the array isn't already contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    im.data = arr.tobytes()  # Use tobytes() instead of tostring()
    im.step = arr.strides[0]
    im.is_bigendian = arr.dtype.byteorder == ">" or arr.dtype.byteorder == "=" and sys.byteorder == "big"
    return im
