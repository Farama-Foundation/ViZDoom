cmake_minimum_required( VERSION 2.8 )
project(ViZDoom)

option(BUILD_PYTHON "Build ViZDoom Python binding." OFF)
option(BUILD_JAVA "Build ViZDoom Java binding." OFF)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

# System dependent settings
if(UNIX)
	set(Boost_USE_STATIC_LIBS OFF)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_RUNTIME OFF)
endif()

if(WIN32)
	set(Boost_USE_STATIC_LIBS ON)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_RUNTIME ON)

	macro(get_WIN32_WINNT version)
		if (WIN32 AND CMAKE_SYSTEM_VERSION)
			set(ver ${CMAKE_SYSTEM_VERSION})
			string(REPLACE "." "" ver ${ver})
			string(REGEX REPLACE "([0-9])" "0\\1" ver ${ver})

			set(${version} "0x${ver}")
		endif()
	endmacro()

	get_WIN32_WINNT(ver)
	add_definitions(-D_WIN32_WINNT=${ver})
endif()

# Compiler dependent settings
if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions("-fPIC")
endif()

if(MSVC)
	string(REPLACE "/MD " "/MT " CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
	string(REPLACE "/MD " "/MT " CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_MINSIZEREL})
	string(REPLACE "/MD " "/MT " CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
	string(REPLACE "/MDd " "/MTd " CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()


# Targets

find_package(Boost COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
find_package(Threads REQUIRED)

set(VIZDOOM_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(VIZDOOM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(VIZDOOM_LIB_SRC_DIR ${VIZDOOM_SRC_DIR}/lib)
set(VIZDOOM_LIB_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_LIB_SRC_DIR})

include_directories(${VIZDOOM_LIB_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

set(VIZDOOM_LIBS
	${Boost_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT} )

if(UNIX AND NOT APPLE)
	set(VIZDOOM_LIBS ${VIZDOOM_LIBS} rt)
endif()

file(GLOB VIZDOOM_LIB_SOURCES
	${VIZDOOM_INCLUDE_DIR}/*.h
	${VIZDOOM_LIB_SRC_DIR}/*.h
	${VIZDOOM_LIB_SRC_DIR}/*.cpp)

add_library(libvizdoom_static STATIC ${VIZDOOM_LIB_SOURCES})
target_link_libraries (libvizdoom_static ${VIZDOOM_LIBS})

add_library(libvizdoom_shared SHARED ${VIZDOOM_LIB_SOURCES})
target_link_libraries (libvizdoom_shared ${VIZDOOM_LIBS})

set_target_properties(libvizdoom_static libvizdoom_shared
	PROPERTIES
	LINKER_LANGUAGE CXX
	ARCHIVE_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}
	OUTPUT_NAME vizdoom)

	add_subdirectory(${VIZDOOM_SRC_DIR}/vizdoom)

if(BUILD_PYTHON)

	#set(Boost_USE_DEBUG_PYTHON OFF)

        find_package(PythonInterp 2.7 REQUIRED)
	# Find the matching Python libs by starting with PYTHON_VERSION_STRING
	# and stripping off the least significant digit (e.g. 3.4.3 -> 3.4)
	STRING( REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" python_version ${PYTHON_VERSION_STRING} )
        find_package(PythonLibs ${python_version} REQUIRED)

	# Find the matching boost python implementation by starting with
	# PYTHONLIBS_VERSION_STRING (e.g. 3.4.3), stripping off the last part of the
	# version, and checking for a boost of that version
	set(version ${PYTHONLIBS_VERSION_STRING})
	STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
	find_package(Boost COMPONENTS "python-py${boost_py_version}")
	set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

	while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
	  STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

	  STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
	  find_package(Boost COMPONENTS "python-py${boost_py_version}")
	  set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

	  STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
	  if("${has_more_version}" STREQUAL "")
	    break()
	  endif()
	endwhile()
	if(NOT Boost_PYTHON_FOUND)
	  find_package(Boost COMPONENTS python REQUIRED)
	endif()
	set(found_boost_python_library ${Boost_LIBRARIES})

        find_package(NumPy REQUIRED)

	set(VIZDOOM_PYTHON_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_python)
	set(VIZDOOM_PYTHON_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_PYTHON_SRC_DIR})

	set(VIZDOOM_PYTHON_LIBS
		${VIZDOOM_LIBS}
		${found_boost_python_library}
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

	if(WIN32)
		add_definitions(-DBOOST_PYTHON_STATIC_LIB)
	endif()

	python_add_module(libvizdoom_python ${VIZDOOM_PYTHON_SOURCES})
	target_link_libraries (libvizdoom_python ${VIZDOOM_PYTHON_LIBS} libvizdoom_static)

	set_target_properties(libvizdoom_python
		PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}/python
		LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}/python
		LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}/python
		LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}/python
		LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}/python
		RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}/python
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}/python
		RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}/python
		RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}/python
		RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}/python
		OUTPUT_NAME vizdoom)

	if(UNIX)
		ADD_CUSTOM_COMMAND(
			TARGET libvizdoom_python
			COMMAND ln -sf ${VIZDOOM_OUTPUT_DIR}/python/vizdoom.so ${CMAKE_CURRENT_SOURCE_DIR}/examples/python/vizdoom.so
			COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/examples/config/bots.cfg ${CMAKE_CURRENT_SOURCE_DIR}/examples/python/bots.cfg
			COMMENT "Adding symlinks to vizdoom.so and bots.cfg in examples/python.")
	endif()

endif()

#Java binding
if(BUILD_JAVA)

	if (NOT DEFINED ENV{JAVA_HOME})
		message(FATAL_ERROR "JAVA_HOME environment variable is not set.")
	endif()

	find_package(Java REQUIRED)
	find_package(JNI REQUIRED)

	include(UseJava)
	set(VIZDOOM_JAVA_SRC_DIR ${VIZDOOM_SRC_DIR}/lib_java)
	set(VIZDOOM_JAVA_CLASSES_DIR ${VIZDOOM_JAVA_SRC_DIR}/java_classes)
	set(VIZDOOM_JAVA_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} ${VIZDOOM_JAVA_SRC_DIR})

	set(VIZDOOM_JAVA_LIBS
		${VIZDOOM_LIBS}
		${Java_LIBRARIES}
		${JNI_LIBRARIES})

	if( UNIX )
		set(JNI_INCLUDE_DIR ${JNI_INCLUDE_DIR}
			${_JAVA_HOME}/include
			${_JAVA_HOME}/include/linux)
	endif()

	if( WIN32 )
		set( JNI_INCLUDE_DIR ${JNI_INCLUDE_DIR}
			${_JAVA_HOME}/include
			${_JAVA_HOME}/include/win32 )
	endif()

	include_directories(${VIZDOOM_JAVA_INCLUDE_DIR}
						${Boost_INCLUDE_DIR}
						${Java_INCLUDE_DIRS}
						${JNI_INCLUDE_DIR})

	set(VIZDOOM_JAVA_SOURCES
		${VIZDOOM_JAVA_SRC_DIR}/ViZDoomGameJava.h
		${VIZDOOM_JAVA_SRC_DIR}/ViZDoomGameJava.cpp)

	add_library(libvizdoom_java SHARED ${VIZDOOM_JAVA_SOURCES})
	target_link_libraries (libvizdoom_java ${VIZDOOM_JAVA_LIBS} libvizdoom_static)

	set_target_properties(libvizdoom_java
		PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}/java
		LIBRARY_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}/java
		LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}/java
		LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}/java
		LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}/java
		RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_OUTPUT_DIR}/java
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_OUTPUT_DIR}/java
		RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_OUTPUT_DIR}/java
		RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_OUTPUT_DIR}/java
		RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_OUTPUT_DIR}/java
		OUTPUT_NAME vizdoom)

	add_jar(libvizdoom_java_classes
		${VIZDOOM_JAVA_CLASSES_DIR}/enums/Button.java
		${VIZDOOM_JAVA_CLASSES_DIR}/enums/GameVariable.java
		${VIZDOOM_JAVA_CLASSES_DIR}/enums/Mode.java
		${VIZDOOM_JAVA_CLASSES_DIR}/enums/ScreenFormat.java
		${VIZDOOM_JAVA_CLASSES_DIR}/enums/ScreenResolution.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomMismatchedVersionException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomErrorException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomIsNotRunningException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/ViZDoomUnexpectedExitException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/MessageQueueException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/FileDoesNotExistException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/exceptions/SharedMemoryException.java
		${VIZDOOM_JAVA_CLASSES_DIR}/GameState.java
		${VIZDOOM_JAVA_CLASSES_DIR}/DoomGame.java
		OUTPUT_DIR ${VIZDOOM_OUTPUT_DIR}/java
		OUTPUT_NAME vizdoom)

endif()

#if( WIN32 )
#	if(NOT EXISTS "${PROJECT_SOURCE_DIR}/scenarios/freedoom.zip")
#		message("-- Downloading Freedoom WADs")
#		message("-- Original Doom WADs can not be provided due to licences")
#
#		file( DOWNLOAD https://github.com/freedoom/freedoom/releases/download/v0.10.1/freedoom-0.10.1.zip ${PROJECT_SOURCE_DIR}/scenarios/freedoom.zip SHOW_PROGRESS )
#	endif()
#endif()

if(UNIX)
	add_custom_target( freedoom2 ALL
		COMMAND ./download_freedoom.sh
	)
endif()
