cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME GeometryDashAI)

project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES CXX)

if(NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "GEODE_SDK is not set. Install the Geode SDK or set GEODE_SDK to its path.")
endif()

add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode-sdk)

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

setup_geode_mod(${PROJECT_NAME})
