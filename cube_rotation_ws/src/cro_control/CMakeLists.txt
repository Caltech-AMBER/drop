cmake_minimum_required(VERSION 3.16)
project(cro_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall
    -Wextra
    # suppress warnings from building mujoco via mjpc
    -Wno-unused-parameter
    -Wno-type-limits
    -Wno-missing-field-initializers
    -Wno-error=return-type
    # suppress mjpc warnings
    -Wno-sign-compare
    # suppress warning about end of non-void function (for a publisher)
    -Wno-return-type
  )
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ccd REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Threads REQUIRED)

# ABSL warning
set(ABSL_PROPAGATE_CXX_STD ON)

# fetching obelisk (note: OBELISK_ROOT must be set!)
include(FetchContent)
FetchContent_Declare(
  Obelisk
  SOURCE_DIR $ENV{OBELISK_ROOT}/obelisk/cpp
)
FetchContent_MakeAvailable(Obelisk)

# finding/fetching absl, mjpc
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(FindOrFetch)
FindOrFetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  absl
  LIBRARY_NAME
  abseil-cpp
  GIT_REPO
  https://github.com/abseil/abseil-cpp.git
  GIT_TAG
  fb3621f4f897824c0dbe0615fa94543df6192f30
  TARGETS
  absl::core_headers
  EXCLUDE_FROM_ALL
)
FindOrFetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mjpc
  LIBRARY_NAME
  mjpc
  GIT_REPO
  https://github.com/alberthli/mujoco_mpc.git
  GIT_TAG
  85f89c1920fb35e16b204cca56e435f4bb80e57f
  TARGETS
  mjpc::mjpc
  mjpc::threadpool
  EXCLUDE_FROM_ALL
)  # fork of mjpc with LEAP task

set(THIS_PACKAGE_INCLUDE_DEPENDS
  ament_cmake
  rclcpp
  rclcpp_lifecycle
  std_msgs
  sensor_msgs
  mjpc
  Threads
)
include_directories(${THIS_PACKAGE_INCLUDE_DEPENDS})

# cro_controller executable
add_executable(cro_controller src/cro_controller.cpp)
target_include_directories(cro_controller
  PUBLIC
  ${THIS_PACKAGE_INCLUDE_DEPENDS})
target_link_libraries(
  cro_controller
  PUBLIC
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::strings
  libmjpc
  glfw
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
  Obelisk::Core
  Obelisk::Zoo
)
ament_target_dependencies(cro_controller
  PUBLIC
  ament_cmake
  rclcpp
  rclcpp_lifecycle
  std_msgs
  sensor_msgs)

install(TARGETS cro_controller
  DESTINATION lib/${PROJECT_NAME})

ament_package()
