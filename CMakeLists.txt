cmake_minimum_required(VERSION 2.8)
project(ViZDoom)


# Project version
#-----------------------------------------------------------------------------------------------------------------------

set(ViZDoom_MAJOR_VERSION 1)
set(ViZDoom_MINOR_VERSION 1)
set(ViZDoom_PATCH_VERSION 8)
set(ViZDoom_VERSION ${ViZDoom_MAJOR_VERSION}.${ViZDoom_MINOR_VERSION}.${ViZDoom_PATCH_VERSION})
set(ViZDoom_VERSION_STR ${ViZDoom_VERSION})
set(ViZDoom_VERSION_ID ${ViZDoom_MAJOR_VERSION}${ViZDoom_MINOR_VERSION}${ViZDoom_PATCH_VERSION})


# Building options
#-----------------------------------------------------------------------------------------------------------------------

option(DOWNLOAD_FREEDOOM "Download freedoom2.wad." ON)
option(BUILD_PYTHON3 "Build ViZDoom Python 3 binding." OFF)
option(BUILD_PYTHON "Build ViZDoom Python (3 by default) binding." OFF)
option(BUILD_PYTHON2 "Build ViZDoom Python2 binding instead of default Python3." OFF)
option(BUILD_JULIA "Build ViZDoom Julia binding." OFF)


# CMake options
#-----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_LINK)
if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif (COMMAND cmake_policy)

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

# Set the release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


# System dependent settings
#-----------------------------------------------------------------------------------------------------------------------

if (UNIX)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)
endif ()

if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME ON)

    macro(get_WIN32_WINNT version)
        if (WIN32 AND CMAKE_SYSTEM_VERSION)
            set(ver ${CMAKE_SYSTEM_VERSION})
            string(REPLACE "10" "A" ver ${ver})
            string(REPLACE "." "" ver ${ver})
            string(REGEX REPLACE "([A-F0-9])" "0\\1" ver ${ver})
            string(SUBSTRING ${ver} 0 4 ver)

            set(${version} "0x${ver}")
        endif ()
    endmacro()

    get_WIN32_WINNT(ver)
    add_definitions(-D_WIN32_WINNT=${ver})
endif ()

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
endif (APPLE)


# Compiler dependent settings
#-----------------------------------------------------------------------------------------------------------------------

if (CMAKE_COMPILER_IS_GNUCXX)
    add_definitions("-fPIC")
endif ()

if (MSVC)
    string(REPLACE "/MD " "/MT " CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/MD " "/MT " CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_MINSIZEREL})
    string(REPLACE "/MD " "/MT " CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/MDd " "/MTd " CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif ()


# Main library and engine
#-----------------------------------------------------------------------------------------------------------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(VIZDOOM_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(VIZDOOM_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
set(VIZDOOM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VIZDOOM_LIB_SRC_DIR ${VIZDOOM_SRC_DIR}/lib)
set(VIZDOOM_LIB_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_LIB_SRC_DIR})
set(VIZDOOM_OUTPUT_NAME vizdoom)

find_package(Boost COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
find_package(Threads REQUIRED)

include_directories(${VIZDOOM_LIB_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

set(VIZDOOM_LIBS
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})

if (UNIX AND NOT APPLE)
    set(VIZDOOM_LIBS ${VIZDOOM_LIBS} rt)
endif ()

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
    PROJECT_LABEL "static library")

set_target_properties(libvizdoom_shared
    PROPERTIES
    PROJECT_LABEL "shared library")

add_subdirectory(${VIZDOOM_SRC_DIR}/vizdoom)


# Python and Julia
#-----------------------------------------------------------------------------------------------------------------------

if (BUILD_PYTHON OR BUILD_PYTHON3 OR BUILD_PYTHON2)
    add_subdirectory(${VIZDOOM_SRC_DIR}/lib_python)
endif ()

if (BUILD_JULIA)
    add_subdirectory(${VIZDOOM_SRC_DIR}/lib_julia)
endif ()


# Freedoom 2 download
#-----------------------------------------------------------------------------------------------------------------------

if (UNIX AND DOWNLOAD_FREEDOOM)
    add_custom_target(freedoom2 ALL
        COMMAND ${VIZDOOM_SCRIPTS_DIR}/download_freedoom.sh ${VIZDOOM_OUTPUT_DIR}
        COMMENT "Downloading freedoom2.wad")
endif ()

if (WIN32 AND DOWNLOAD_FREEDOOM)
    add_custom_target(freedoom2 ALL
        COMMAND ${VIZDOOM_SCRIPTS_DIR}/download_freedoom.bat ${VIZDOOM_OUTPUT_DIR}
        COMMENT "Downloading freedoom2.wad")
endif ()
