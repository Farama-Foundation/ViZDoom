cmake_minimum_required(VERSION 2.8)
project(ViZDoom)

# Project version
#-----------------------------------------------------------------------------------------------------------------------
set(ViZDoom_MAJOR_VERSION 1)
set(ViZDoom_MINOR_VERSION 1)
set(ViZDoom_PATCH_VERSION 0)
set(ViZDoom_VERSION ${ViZDoom_MAJOR_VERSION}.${ViZDoom_MINOR_VERSION}.${ViZDoom_PATCH_VERSION})
set(ViZDoom_VERSION_STR ${ViZDoom_VERSION})
set(ViZDoom_VERSION_ID ${ViZDoom_MAJOR_VERSION}${ViZDoom_MINOR_VERSION}${ViZDoom_PATCH_VERSION})


# Building options
#-----------------------------------------------------------------------------------------------------------------------
option(BUILD_PYTHON "Build ViZDoom Python binding." OFF)
option(BUILD_PYTHON3 "Build ViZDoom Python3 binding instead of default Python2." OFF)
option(BUILD_LUA "Build ViZDoom Lua binding." OFF) # only for LuaJIT
option(BUILD_JAVA "Build ViZDoom Java binding." OFF)


# CMake options
#-----------------------------------------------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
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
            string(REPLACE "." "" ver ${ver})
            string(REGEX REPLACE "([0-9])" "0\\1" ver ${ver})

            set(${version} "0x${ver}")
        endif ()
    endmacro()

    get_WIN32_WINNT(ver)
    add_definitions(-D_WIN32_WINNT=${ver})
endif ()

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
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


# Targets
#-----------------------------------------------------------------------------------------------------------------------
find_package(Boost COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
find_package(Threads REQUIRED)

set(VIZDOOM_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(VIZDOOM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VIZDOOM_LIB_SRC_DIR ${VIZDOOM_SRC_DIR}/lib)
set(VIZDOOM_LIB_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_LIB_SRC_DIR})
set(VIZDOOM_OUTPUT_NAME vizdoom)

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
target_link_libraries(libvizdoom_static ${VIZDOOM_LIBS})

add_library(libvizdoom_shared SHARED ${VIZDOOM_LIB_SOURCES})
target_link_libraries(libvizdoom_shared ${VIZDOOM_LIBS})

set_target_properties(libvizdoom_static libvizdoom_shared
        PROPERTIES
        LINKER_LANGUAGE CXX
        LIBRARY_OUTPUT_NAME ${VIZDOOM_OUTPUT_NAME}
        LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}
        LIBRARY_OUTPUT_NAME_RELEASE ${VIZDOOM_OUTPUT_NAME}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}
        LIBRARY_OUTPUT_NAME_DEBUG ${VIZDOOM_OUTPUT_NAME}-d
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}
        LIBRARY_OUTPUT_NAME_MINSIZEREL ${VIZDOOM_OUTPUT_NAME}-msr
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}
        LIBRARY_OUTPUT_NAME_RELWITHDEBINFO ${VIZDOOM_OUTPUT_NAME}-rd
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR})

set_target_properties(libvizdoom_static
        PROPERTIES
        PROJECT_LABEL "Static library")

set_target_properties(libvizdoom_shared
        PROPERTIES
        PROJECT_LABEL "Shared library")


add_subdirectory(${VIZDOOM_SRC_DIR}/vizdoom)


# Python binding
#-----------------------------------------------------------------------------------------------------------------------
if (BUILD_PYTHON AND (NOT BUILD_PYTHON3))

    find_package(PythonInterp 2 REQUIRED)

    STRING(REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" python_version ${PYTHON_VERSION_STRING})
    find_package(PythonLibs ${python_version} REQUIRED)

    find_package(Boost COMPONENTS python REQUIRED)

    find_package(NumPy REQUIRED)

    set(VIZDOOM_PYTHON2_OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/python2)
    set(VIZDOOM_PYTHON_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_python)
    set(VIZDOOM_PYTHON_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_PYTHON_SRC_DIR})

    set(VIZDOOM_PYTHON_LIBS
            ${VIZDOOM_LIBS}
            ${Boost_LIBRARIES}
            ${PYTHON_LIBRARIES}
            ${NUMPY_LIBRARIES})

    include_directories(${VIZDOOM_PYTHON_INCLUDE_DIR}
            ${Boost_INCLUDE_DIR}
            ${PYTHON_INCLUDE_DIRS}
            ${NUMPY_INCLUDES})

    set(VIZDOOM_PYTHON_SOURCES
            ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.h
            ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.cpp
            ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomPythonModule.cpp)

    if (WIN32)
        add_definitions(-DBOOST_PYTHON_STATIC_LIB)
    endif ()

    python_add_module(libvizdoom_python ${VIZDOOM_PYTHON_SOURCES})
    target_link_libraries(libvizdoom_python ${VIZDOOM_PYTHON_LIBS} libvizdoom_static)

    set_target_properties(libvizdoom_python
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_PYTHON2_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_PYTHON2_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_PYTHON2_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_PYTHON2_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_PYTHON2_OUTPUT_DIR}
            OUTPUT_NAME vizdoom
            PROJECT_LABEL "Python 2 binding")

    # vizdoom, vizdoom.pk3 and freedoom2.wad are needed to assemble the package
    add_dependencies(libvizdoom_python vizdoom pk3 freedoom2)

    if (UNIX)
        add_custom_command(TARGET libvizdoom_python
                COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/assemble_pip_package.sh 2
                COMMENT "Assembling pip package in ${VIZDOOM_PYTHON2_OUTPUT_DIR}/pip_package")
    endif ()

endif ()


if (BUILD_PYTHON3)

    find_package(PythonInterp 3 REQUIRED)

    # 3.5.2 => 3.5
    STRING(REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" python3_version ${PYTHON_VERSION_STRING})
    find_package(PythonLibs ${python3_version} REQUIRED)

    # 3.5 => 35 (at least on Ubuntu 16.04 the library is called libboost_python-py35.so)
    STRING(REGEX REPLACE "\\." "" python3_version ${python3_version})
    find_package(Boost QUIET COMPONENTS python-py${python3_version})
    if (${Boost_PYTHON-PY${python3_version}_FOUND})
        MESSAGE(STATUS "Found Boost python3:")
        MESSAGE(STATUS "   ${Boost_PYTHON-PY${python3_version}_LIBRARY}")
    else ()
        find_package(Boost QUIET COMPONENTS python3)  # The name under OSX (macports)
        if (${Boost_PYTHON3_FOUND})
            MESSAGE(STATUS "Found Boost python3:")
            Message(STATUS "  ${Boost_PYTHON3_LIBRARY}")
        else ()
            MESSAGE(FATAL_ERROR "Could not find boost python3")
        endif ()
    endif ()

    find_package(NumPy REQUIRED)

    set(VIZDOOM_PYTHON3_OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/python3)
    set(VIZDOOM_PYTHON_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_python)
    set(VIZDOOM_PYTHON_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_PYTHON_SRC_DIR})

    set(VIZDOOM_PYTHON3_LIBS
            ${VIZDOOM_LIBS}
            ${Boost_LIBRARIES}
            ${PYTHON_LIBRARIES}
            ${NUMPY_LIBRARIES})

    include_directories(${VIZDOOM_PYTHON_INCLUDE_DIR}
            ${Boost_INCLUDE_DIR}
            ${PYTHON_INCLUDE_DIRS}
            ${NUMPY_INCLUDES})

    set(VIZDOOM_PYTHON_SOURCES
            ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.h
            ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.cpp
            ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomPythonModule.cpp)

    if (WIN32)
        add_definitions(-DBOOST_PYTHON_STATIC_LIB)
    endif ()

    python_add_module(libvizdoom_python3 ${VIZDOOM_PYTHON_SOURCES})
    target_link_libraries(libvizdoom_python3 ${VIZDOOM_PYTHON3_LIBS} libvizdoom_static)

    set_target_properties(libvizdoom_python3
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_PYTHON3_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_PYTHON3_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_PYTHON3_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_PYTHON3_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_PYTHON3_OUTPUT_DIR}
            OUTPUT_NAME vizdoom
            PROJECT_LABEL "Python 3 binding")

    # vizdoom, vizdoom.pk3 and freedoom2.wad are needed to assemble the package
    add_dependencies(libvizdoom_python3 vizdoom pk3 freedoom2)

    if (UNIX)
        add_custom_command(TARGET libvizdoom_python3
                COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/assemble_pip_package.sh 3
                COMMENT "Assembling pip package in ${VIZDOOM_PYTHON3_OUTPUT_DIR}/pip_package")
    endif ()


endif ()


# Lua binding
#-----------------------------------------------------------------------------------------------------------------------
if (BUILD_LUA AND (NOT BUILD_LUA52))

    # Use ViZDoom's custom version of luabind
    #find_package(Luabind REQUIRED)
    add_subdirectory(${VIZDOOM_SRC_DIR}/lib_lua/luabind)

    find_package(Lua51 REQUIRED)
    #find_package(Torch REQUIRED)

    set(VIZDOOM_LUA_OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/lua)
    set(VIZDOOM_LUA_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_lua)
    set(VIZDOOM_LUA_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_LUA_SRC_DIR})

    set(VIZDOOM_LUA_LIBS
            ${VIZDOOM_LIBS}
            ${Boost_LIBRARIES}
            ${LUA_LIBRARIES}
            #${LUABIND_LIBRARIES}
            #${VIZDOOM_SRC_DIR}/lib_lua/luabind/src/libluabind.a
            luabind)

    include_directories(${VIZDOOM_LUA_INCLUDE_DIR}
            ${Boost_INCLUDE_DIR}
            ${PYTHON_INCLUDE_DIRS}
            ${LUA_INCLUDE_DIR}
            #${LUABIND_INCLUDE_DIRS}
            ${VIZDOOM_SRC_DIR}/lib_lua/luabind)

    set(VIZDOOM_LUA_SOURCES
            ${VIZDOOM_LUA_SRC_DIR}/ViZDoomGameLua.h
            ${VIZDOOM_LUA_SRC_DIR}/ViZDoomGameLua.cpp
            ${VIZDOOM_LUA_SRC_DIR}/ViZDoomLuaModule.cpp)

    add_library(libvizdoom_lua SHARED ${VIZDOOM_LUA_SOURCES})
    target_link_libraries(libvizdoom_lua ${VIZDOOM_LUA_LIBS} libvizdoom_static)

    set_target_properties(libvizdoom_lua
            PROPERTIES
            PREFIX ""
            LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_LUA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_LUA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_LUA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_LUA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_LUA_OUTPUT_DIR}
            OUTPUT_NAME vizdoom
            PROJECT_LABEL "Lua binding")

    # vizdoom, vizdoom.pk3 and freedoom2.wad are needed to assemble the package
    add_dependencies(libvizdoom_lua vizdoom pk3 freedoom2)

    if (UNIX)
        add_custom_command(TARGET libvizdoom_lua
                COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/assemble_luarocks_package.sh
                COMMENT "Assembling luarocks package in bin/lua")
    endif ()

endif ()


# Java binding
#-----------------------------------------------------------------------------------------------------------------------
if (BUILD_JAVA)

    if (NOT DEFINED ENV{JAVA_HOME})
        message(FATAL_ERROR "JAVA_HOME environment variable is not set.")
    endif ()

    find_package(Java REQUIRED)
    find_package(JNI REQUIRED)

    include(UseJava)

    set(VIZDOOM_JAVA_OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/java)
    set(VIZDOOM_JAVA_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_java)
    set(VIZDOOM_JAVA_CLASSES_DIR ${VIZDOOM_JAVA_SRC_DIR}/java_classes)
    set(VIZDOOM_JAVA_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_JAVA_SRC_DIR})

    set(VIZDOOM_JAVA_LIBS
            ${VIZDOOM_LIBS}
            ${Java_LIBRARIES}
            ${JNI_LIBRARIES})

    if (UNIX)
        set(JNI_INCLUDE_DIR ${JNI_INCLUDE_DIR}
                ${_JAVA_HOME}/include
                ${_JAVA_HOME}/include/linux)
    endif ()

    if (WIN32)
        set(JNI_INCLUDE_DIR ${JNI_INCLUDE_DIR}
                ${_JAVA_HOME}/include
                ${_JAVA_HOME}/include/win32)
    endif ()

    include_directories(${VIZDOOM_JAVA_INCLUDE_DIR}
            ${Boost_INCLUDE_DIR}
            ${Java_INCLUDE_DIRS}
            ${JNI_INCLUDE_DIR})

    set(VIZDOOM_JAVA_SOURCES
            ${VIZDOOM_JAVA_SRC_DIR}/ViZDoomJava.h
            ${VIZDOOM_JAVA_SRC_DIR}/ViZDoomGameJava.h
            ${VIZDOOM_JAVA_SRC_DIR}/ViZDoomGameJava.cpp)

    add_library(libvizdoom_java SHARED ${VIZDOOM_JAVA_SOURCES})
    target_link_libraries(libvizdoom_java ${VIZDOOM_JAVA_LIBS} libvizdoom_static)

    set_target_properties(libvizdoom_java
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_JAVA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_JAVA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_JAVA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_JAVA_OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_JAVA_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_JAVA_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_JAVA_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_JAVA_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_JAVA_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_JAVA_OUTPUT_DIR}
            OUTPUT_NAME vizdoom
            PROJECT_LABEL "Java binding")

    add_jar(libvizdoom_java_classes
            ${VIZDOOM_JAVA_CLASSES_DIR}/enums/AutomapMode.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/enums/Button.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/enums/GameVariable.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/enums/Mode.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/enums/ScreenFormat.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/enums/ScreenResolution.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/FileDoesNotExistException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/MessageQueueException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/SharedMemoryException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/SignalException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomErrorException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomIsNotRunningException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomUnexpectedExitException.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/GameState.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/Label.java
            ${VIZDOOM_JAVA_CLASSES_DIR}/DoomGame.java
            OUTPUT_DIR ${VIZDOOM_JAVA_OUTPUT_DIR}
            OUTPUT_NAME vizdoom
            PROJECT_LABEL "ViZDoom Java classes")

endif ()


# Freedoom 2 download
#-----------------------------------------------------------------------------------------------------------------------

if (UNIX)
    add_custom_target(freedoom2 ALL
            COMMAND ./scripts/download_freedoom.sh)
endif ()
