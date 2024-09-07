set(BUILD_PYTHON_VERSION "" CACHE STRING "Version of Python to build bindings for")

if("${BUILD_PYTHON_VERSION}" STREQUAL "")
    # Try to find Python 3 if BUILD_PYTHON_VERSION is not set
    message(STATUS "BUILD_PYTHON_VERSION is not set, trying to find Python 3, Python_ROOT_DIR is ${Python_ROOT_DIR}")
    find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
    set(BUILD_PYTHON_VERSION ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
endif()

set(PYBIND11_PYTHON_VERSION ${BUILD_PYTHON_VERSION})
set(PYBIND11_FINDPYTHON True)

set(VIZDOOM_PYTHON_OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/python${BUILD_PYTHON_VERSION})
set(VIZDOOM_PYTHON_PACKAGE_DIR ${VIZDOOM_PYTHON_OUTPUT_DIR}/vizdoom)
set(VIZDOOM_PYTHON_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_python)
set(VIZDOOM_PYTHON_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_PYTHON_SRC_DIR})

message(STATUS "Specified Python version to use: ${BUILD_PYTHON_VERSION}")

if(NOT EXISTS ${VIZDOOM_PYTHON_SRC_DIR}/pybind11/CMakeLists.txt)
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        message(STATUS "pybind11 submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()

    if(NOT EXISTS ${VIZDOOM_PYTHON_SRC_DIR}/pybind11/CMakeLists.txt)
        message(FATAL_ERROR "The pybind11 submodule was not downloaded!")
    endif()
endif()
add_subdirectory(${VIZDOOM_PYTHON_SRC_DIR}/pybind11)

set(VIZDOOM_PYTHON_LIBS
    ${VIZDOOM_LIBS}
    ${Boost_LIBRARIES})

include_directories(${VIZDOOM_PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR})

set(ViZDoom_PYTHON_INLCUDE_DIR ${VIZDOOM_PYTHON_INCLUDE_DIR})
set(ViZDoom_PYTHON_VERSION_STR ${BUILD_PYTHON_VERSION})

configure_file(
    "${VIZDOOM_PYTHON_SRC_DIR}/__init__.py.in"
    "${VIZDOOM_PYTHON_SRC_DIR}/__init__.py"
)

set(VIZDOOM_PYTHON_SOURCES
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.h
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomGamePython.cpp
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomMethodsDocstrings.h
    ${VIZDOOM_PYTHON_SRC_DIR}/ViZDoomObjectsDocstrings.h
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
    PROJECT_LABEL "python${BUILD_PYTHON_VERSION} binding")

add_custom_target(assemble_package ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:libvizdoom_python> ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${VIZDOOM_PYTHON_SRC_DIR}/__init__.py ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vizdoom> ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${VIZDOOM_OUTPUT_DIR}/vizdoom.pk3 ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${VIZDOOM_SRC_DIR}/bots.cfg ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${VIZDOOM_SRC_DIR}/freedoom2.wad ${VIZDOOM_PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${VIZDOOM_ROOT_DIR}/scenarios ${VIZDOOM_PYTHON_PACKAGE_DIR}/scenarios
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${VIZDOOM_ROOT_DIR}/gymnasium_wrapper ${VIZDOOM_PYTHON_PACKAGE_DIR}/gymnasium_wrapper
    COMMENT "Assembling Python package in ${VIZDOOM_PYTHON_OUTPUT_DIR}/vizdoom"
)

if(WIN32) # Copy DLLs for Windows
    file(GLOB TXT_FILES "${VIZDOOM_OUTPUT_DIR}/*.dll")

    foreach(file ${TXT_FILES})
        add_custom_command(
            TARGET assemble_package
            COMMAND ${CMAKE_COMMAND} -E copy ${file} ${VIZDOOM_PYTHON_PACKAGE_DIR}
        )
    endforeach()
endif()

set_target_properties(assemble_package
    PROPERTIES
    PROJECT_LABEL "Python package")

# vizdoom, vizdoom.pk3 are needed to assemble the package
add_dependencies(assemble_package libvizdoom_python vizdoom pk3)
