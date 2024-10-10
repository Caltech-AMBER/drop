if(COMMAND FindOrFetch)
  return()
endif()

macro(FindOrFetch)
  if(NOT FetchContent)
    include(FetchContent)
  endif()

  # Parse arguments.
  set(options EXCLUDE_FROM_ALL)
  set(one_value_args
      USE_SYSTEM_PACKAGE
      PACKAGE_NAME
      LIBRARY_NAME
      GIT_REPO
      GIT_TAG
  )
  set(multi_value_args PATCH_COMMAND TARGETS)
  cmake_parse_arguments(
    _ARGS
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  # Check if all targets are found.
  if(NOT _ARGS_TARGETS)
    message(FATAL_ERROR "mujoco::FindOrFetch: TARGETS must be specified.")
  endif()

  set(targets_found TRUE)
  message(CHECK_START
          "mujoco::FindOrFetch: checking for targets in package `${_ARGS_PACKAGE_NAME}`"
  )
  foreach(target ${_ARGS_TARGETS})
    if(NOT TARGET ${target})
      message(CHECK_FAIL "target `${target}` not defined.")
      set(targets_found FALSE)
      break()
    endif()
  endforeach()

  # If targets are not found, use `find_package` or `FetchContent...` to get it.
  if(NOT targets_found)
    if(${_ARGS_USE_SYSTEM_PACKAGE})
      message(CHECK_START
              "mujoco::FindOrFetch: finding `${_ARGS_PACKAGE_NAME}` in system packages..."
      )
      find_package(${_ARGS_PACKAGE_NAME} REQUIRED)
      message(CHECK_PASS "found")
    else()
      message(CHECK_START
              "mujoco::FindOrFetch: Using FetchContent to retrieve `${_ARGS_LIBRARY_NAME}`"
      )
      FetchContent_Declare(
        ${_ARGS_LIBRARY_NAME}
        GIT_REPOSITORY ${_ARGS_GIT_REPO}
        GIT_TAG ${_ARGS_GIT_TAG}
        GIT_SHALLOW FALSE
        PATCH_COMMAND ${_ARGS_PATCH_COMMAND}
        UPDATE_DISCONNECTED TRUE
      )
      if(${_ARGS_EXCLUDE_FROM_ALL})
        FetchContent_GetProperties(${_ARGS_LIBRARY_NAME})
        if(NOT ${${_ARGS_LIBRARY_NAME}_POPULATED})
          FetchContent_Populate(${_ARGS_LIBRARY_NAME})
          add_subdirectory(
            ${${_ARGS_LIBRARY_NAME}_SOURCE_DIR} ${${_ARGS_LIBRARY_NAME}_BINARY_DIR}
            EXCLUDE_FROM_ALL
          )
        endif()
      else()
        FetchContent_MakeAvailable(${_ARGS_LIBRARY_NAME})
      endif()
      message(CHECK_PASS "Done")
    endif()
  else()
    message(CHECK_PASS "found")
  endif()
endmacro()
