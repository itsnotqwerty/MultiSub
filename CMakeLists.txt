cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime backend for model inference" ON)
option(MULTISUB_BUILD_TESTS "Build MultiSub unit and integration tests" ON)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_20)

target_include_directories(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_ONNX_RUNTIME)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime installation")
  mark_as_advanced(ONNXRUNTIME_ROOT)

  if(ONNXRUNTIME_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_ROOT}")
  endif()

  find_package(onnxruntime CONFIG QUIET)

  if(NOT TARGET onnxruntime::onnxruntime AND NOT TARGET onnxruntime)
    find_path(
      MULTISUB_ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime_cxx_api.h
      HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
      PATH_SUFFIXES include include/onnxruntime/core/session onnxruntime/core/session
    )

    find_library(
      MULTISUB_ONNXRUNTIME_LIBRARY
      NAMES onnxruntime libonnxruntime
      HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
      PATH_SUFFIXES lib lib64
    )

    if(MULTISUB_ONNXRUNTIME_INCLUDE_DIR AND MULTISUB_ONNXRUNTIME_LIBRARY)
      add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
      set_target_properties(
        onnxruntime::onnxruntime
        PROPERTIES
          IMPORTED_LOCATION "${MULTISUB_ONNXRUNTIME_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${MULTISUB_ONNXRUNTIME_INCLUDE_DIR}"
      )
    endif()
  endif()

  if(TARGET onnxruntime::onnxruntime)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE MULTISUB_ENABLE_ONNX_RUNTIME=1)
  elseif(TARGET onnxruntime)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE MULTISUB_ENABLE_ONNX_RUNTIME=1)
  else()
    message(
      WARNING
        "ONNX Runtime not found. Building without model inference backend. Set onnxruntime_DIR or ONNXRUNTIME_ROOT to your ONNX Runtime installation."
    )
  endif()
endif()

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    src/plugin-main.cpp
    src/multisub-filter.cpp
    src/audio/audio-capture.cpp
    src/audio/audio-engine.cpp
    src/audio/asr-engine.cpp
    src/audio/noise-classifier.cpp
    src/audio/audio-utils.cpp
    src/fusion/subtitle-fusion.cpp
    src/fusion/multi-channel-renderer.cpp
    src/fusion/subtitle-buffer.cpp
    src/utils/config.cpp
    src/utils/logger.cpp
    src/utils/model-loader.cpp
)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})

if(MULTISUB_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
