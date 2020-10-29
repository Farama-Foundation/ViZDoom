if (BUILD_PYTHON2)
    find_package(PythonInterp 2 REQUIRED)
else ()
    find_package(PythonInterp 3 REQUIRED)
endif ()

STRING(REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" _PYTHON_VERSION ${PYTHON_VERSION_STRING})
find_package(PythonLibs ${_PYTHON_VERSION} REQUIRED)
find_package(NumPy REQUIRED)

set(VIZDOOM_PYTHON_OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/python${_PYTHON_VERSION})
set(VIZDOOM_PYTHON_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_python)
set(VIZDOOM_PYTHON_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_PYTHON_SRC_DIR})

# Get pybind11
if(NOT EXISTS ${VIZDOOM_PYTHON_SRC_DIR}/pybind11)
    include(GitUtils)
    git_clone(PROJECT_NAME pybind11
            GIT_URL https://github.com/pybind/pybind11.git
            #GIT_BRANCH master # To use master version
            GIT_TAG ${PYBIND11_VERSION}
            DIRECTORY ${VIZDOOM_PYTHON_SRC_DIR})
endif()
add_subdirectory(${VIZDOOM_PYTHON_SRC_DIR}/pybind11)

set(VIZDOOM_PYTHON_LIBS
    ${VIZDOOM_LIBS}
    ${Boost_LIBRARIES}
    ${NUMPY_LIBRARIES})

include_directories(${VIZDOOM_PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${NUMPY_INCLUDES})

set(ViZDoom_PYTHON_INLCUDE_DIR ${VIZDOOM_PYTHON_INCLUDE_DIR})
set(ViZDoom_PYTHON_VERSION_STR ${PYTHON_VERSION_STRING})

configure_file(
    "${VIZDOOM_PYTHON_SRC_DIR}/__init__.py.in"
    "${VIZDOOM_PYTHON_SRC_DIR}/__init__.py"
)

set(VIZDOOM_PYTHON_SOURCES
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.h
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.cpp
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomPythonModule.cpp)

pybind11_add_module(libvizdoom_python ${VIZDOOM_PYTHON_SOURCES})
target_link_libraries(libvizdoom_python PRIVATE ${VIZDOOM_PYTHON_LIBS} libvizdoom_static)
set_target_properties(libvizdoom_python
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_PYTHON_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_PYTHON_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_PYTHON_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_PYTHON_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_PYTHON_OUTPUT_DIR}
    OUTPUT_NAME vizdoom
    PROJECT_LABEL "python${_PYTHON_VERSION} binding")

if (UNIX)
    add_custom_target(python_pip_package ALL
        COMMAND ${VIZDOOM_SCRIPTS_DIR}/assemble_pip_package.sh ${_PYTHON_VERSION} ${VIZDOOM_OUTPUT_DIR} ${CMAKE_SOURCE_DIR}
        COMMENT "Assembling pip package in ${VIZDOOM_PYTHON_OUTPUT_DIR}/pip_package")
elseif (WIN32)
    add_custom_target(python_pip_package ALL
        COMMAND ${VIZDOOM_SCRIPTS_DIR}/assemble_pip_package.bat ${_PYTHON_VERSION} ${VIZDOOM_OUTPUT_DIR} ${CMAKE_SOURCE_DIR}
        COMMENT "Assembling pip package in ${VIZDOOM_PYTHON_OUTPUT_DIR}/pip_package")
endif ()

set_target_properties(python_pip_package
    PROPERTIES
    PROJECT_LABEL "pip package")

# vizdoom, vizdoom.pk3 are needed to assemble the package
add_dependencies(python_pip_package libvizdoom_python vizdoom pk3)