cmake_minimum_required( VERSION 2.4 )
project(ViZDoomC++Examples)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_library(libvizdoom SHARED IMPORTED GLOBAL)
if (WIN32)
	set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/vizdoom.lib)
endif()
if (UNIX AND NOT APPLE)
	set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/libvizdoom.a)
endif()
if (APPLE)
	set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/libvizdoom.lib)
endif()

find_package(Boost COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
find_package(Threads REQUIRED)

set( VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include )
include_directories( ${VIZDOOM_INCLUDE_DIR} ${Boost_INCLUDE_DIR} )

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

add_executable( multiplayer Multiplayer.cpp )
target_link_libraries (multiplayer ${VIZDOOM_LIBS})

add_executable( multiplayerhost MultiplayerHost.cpp )
target_link_libraries (multiplayerhost ${VIZDOOM_LIBS})

add_executable( seed Seed.cpp )
target_link_libraries (seed ${VIZDOOM_LIBS})

add_executable( shaping Shaping.cpp )
target_link_libraries (shaping ${VIZDOOM_LIBS})

add_executable( spectator Spectator.cpp )
target_link_libraries (spectator ${VIZDOOM_LIBS})

set_target_properties( basic cig cigbots cighost deltabuttons multiplayer multiplayerhost seed shaping spectator
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_EXAMPLES_OUTPUT_DIR} )
