cmake_minimum_required(VERSION 3.12)
project(ViZDoomC++Examples)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_CXX_STANDARD 11)

# Set the release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Select if we want to link against static or shared libvizdoom
set( LIBVIZDOOM_TYPE STATIC CACHE STRING "Type of libvizdoom library to link against STATIC or SHARED" )

if( LIBVIZDOOM_TYPE STREQUAL "SHARED" )
	add_library( libvizdoom SHARED IMPORTED )
	if (WIN32)
		set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/vizdoom.dll)
	elseif (UNIX AND NOT APPLE)
		set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/libvizdoom.so)
	elseif (APPLE)
		set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/libvizdoom.dylib)
	endif()
else()
	add_library( libvizdoom STATIC IMPORTED )
	if (WIN32)
		set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/vizdoom.dll)
	elseif (UNIX)
		set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/libvizdoom.a)
	endif()
endif()

find_package(Boost COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
find_package(Threads REQUIRED)

set( VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include )
include_directories( ${VIZDOOM_INCLUDE_DIR} )

set( VIZDOOM_EXAMPLES_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/examples )

set( VIZDOOM_LIBS libvizdoom
	${Boost_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT} )

if( UNIX AND NOT APPLE)
	set( VIZDOOM_LIBS ${VIZDOOM_LIBS} rt)
endif()

add_executable( basic Basic.cpp )
target_link_libraries (basic ${VIZDOOM_LIBS})

add_executable( cig CIG.cpp )
target_link_libraries (cig ${VIZDOOM_LIBS})

add_executable( cigbots CIGBots.cpp )
target_link_libraries (cigbots ${VIZDOOM_LIBS})

add_executable( cighost CIGHost.cpp )
target_link_libraries (cighost ${VIZDOOM_LIBS})

add_executable( deltabuttons DeltaButtons.cpp )
target_link_libraries (deltabuttons ${VIZDOOM_LIBS})

add_executable( fpstest FPSTest.cpp )
target_link_libraries (fpstest ${VIZDOOM_LIBS})

add_executable( seed Seed.cpp )
target_link_libraries (seed ${VIZDOOM_LIBS})

add_executable( shaping Shaping.cpp )
target_link_libraries (shaping ${VIZDOOM_LIBS})

add_executable( spectator Spectator.cpp )
target_link_libraries (spectator ${VIZDOOM_LIBS})

set_target_properties( basic cig cigbots cighost deltabuttons fpstest seed shaping spectator
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_EXAMPLES_OUTPUT_DIR} )
