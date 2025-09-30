cmake_minimum_required(VERSION 3.12)

# Project + versions
#-----------------------------------------------------------------------------------------------------------------------

project(ViZDoom VERSION 1.3.0)
set(ViZDoom_VERSION_SUFFIX ".dev3")
set(ViZDoom_VERSION_STR ${ViZDoom_VERSION_MAJOR}.${ViZDoom_VERSION_MINOR}.${ViZDoom_VERSION_PATCH}${ViZDoom_VERSION_SUFFIX})
set(ViZDoom_VERSION_ID ${ViZDoom_VERSION_MAJOR}${ViZDoom_VERSION_MINOR}${ViZDoom_VERSION_PATCH})


# Building options
#-----------------------------------------------------------------------------------------------------------------------

option(BUILD_PYTHON "Build ViZDoom Python (3) binding/module" ON)
option(BUILD_ENGINE "Build ViZDoom Engine (required to build Python package)" ON)
option(CREATE_PYTHON_STUBS "Build ViZDoom Python (3) .pyi type stubs" OFF)


# CMake options
#-----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LINK)

if(POLICY CMP0003)
    cmake_policy(SET CMP0003 NEW)
endif()

if(POLICY CMP0167)
  find_package(Boost 1.70 QUIET CONFIG)
  if(Boost_FOUND)
    cmake_policy(SET CMP0167 NEW)
  else()
    cmake_policy(SET CMP0167 OLD)
  endif()
  unset(Boost_FOUND CACHE)
  unset(Boost_DIR CACHE)
endif()

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

# Set the release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# System dependent settings
#-----------------------------------------------------------------------------------------------------------------------

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

if(UNIX)
    message(STATUS "Unix-like system detected, setting up Unix specific options")
    set(Boost_USE_STATIC_LIBS OFF)
endif()

if(WIN32)
    message(STATUS "Windows detected, setting up Windows specific options")
    set(Boost_USE_STATIC_LIBS ON)

    macro(get_WIN32_WINNT version)
        if(WIN32 AND CMAKE_SYSTEM_VERSION)
            set(ver ${CMAKE_SYSTEM_VERSION})
            string(REPLACE "10" "A" ver ${ver})
            string(REPLACE "." "" ver ${ver})
            string(REGEX REPLACE "([A-F0-9])" "0\\1" ver ${ver})
            string(SUBSTRING ${ver} 0 4 ver)

            set(${version} "0x${ver}")
        endif()
    endmacro()

    get_WIN32_WINNT(ver)
    add_definitions(-D_WIN32_WINNT=${ver})
endif()

if(APPLE)
    message(STATUS "macOS detected, setting up macOS specific options")
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

    if(CMAKE_APPLE_SILICON_PROCESSOR MATCHES "")
        message(STATUS "CMAKE_APPLE_SILICON_PROCESSOR not set")
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
            set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE INTERNAL "" FORCE)
            message(STATUS "Apple Silicon detected, building for arm64")
        else()
            set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL "" FORCE)
            message(STATUS "Intel CPU detected, building for x86_64")
        endif()
    endif()
endif(APPLE)


# Compiler dependent settings
#-----------------------------------------------------------------------------------------------------------------------

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions("-fPIC")
    # Explicitly silence some warnings
    add_definitions("-Wno-implicit-fallthrough -Wno-shift-negative-value -Wno-stringop-truncation")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess -Wno-deprecated-copy") # C++ only
endif()

# Main library
#-----------------------------------------------------------------------------------------------------------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(VIZDOOM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VIZDOOM_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(VIZDOOM_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
set(VIZDOOM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VIZDOOM_LIB_SRC_DIR ${VIZDOOM_SRC_DIR}/lib)
set(VIZDOOM_LIB_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_LIB_SRC_DIR})
set(VIZDOOM_OUTPUT_NAME vizdoom)

if(Boost_VERSION VERSION_GREATER_EQUAL 1.89)
    find_package(Boost 1.89 COMPONENTS filesystem thread date_time chrono regex iostreams REQUIRED)
else()
    find_package(Boost 1.56 COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
endif()
find_package(Threads REQUIRED)

include_directories(${VIZDOOM_LIB_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

set(VIZDOOM_LIBS
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})

if(UNIX AND NOT APPLE)
    set(VIZDOOM_LIBS ${VIZDOOM_LIBS} rt)
endif()

configure_file(
    "${VIZDOOM_LIB_SRC_DIR}/ViZDoomVersion.h.in"
    "${VIZDOOM_LIB_SRC_DIR}/ViZDoomVersion.h"
)

configure_file(
    "${VIZDOOM_SRC_DIR}/vizdoom/src/viz_version.h.in"
    "${VIZDOOM_SRC_DIR}/vizdoom/src/viz_version.h"
)

file(GLOB VIZDOOM_LIB_SOURCES
    ${VIZDOOM_INCLUDE_DIR}/*.h
    ${VIZDOOM_LIB_SRC_DIR}/*.h
    ${VIZDOOM_LIB_SRC_DIR}/*.cpp)

add_library(libvizdoom_static STATIC ${VIZDOOM_LIB_SOURCES})
target_link_libraries(libvizdoom_static PRIVATE ${VIZDOOM_LIBS})

add_library(libvizdoom_shared SHARED ${VIZDOOM_LIB_SOURCES})
target_link_libraries(libvizdoom_shared PRIVATE ${VIZDOOM_LIBS})

set_target_properties(libvizdoom_static libvizdoom_shared
    PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}
    OUTPUT_NAME ${VIZDOOM_OUTPUT_NAME})

set_target_properties(libvizdoom_static
    PROPERTIES
    PROJECT_LABEL "Core static library")

set_target_properties(libvizdoom_shared
    PROPERTIES
    PROJECT_LABEL "Core shared library")


# ViZDoom Engine
#-----------------------------------------------------------------------------------------------------------------------

if(BUILD_ENGINE)
    add_subdirectory(${VIZDOOM_SRC_DIR}/vizdoom)
endif()


# Python bindings
#-----------------------------------------------------------------------------------------------------------------------

if(BUILD_PYTHON)
    add_subdirectory(${VIZDOOM_SRC_DIR}/lib_python)
endif()
