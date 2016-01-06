cmake_minimum_required( VERSION 2.4 )
project(ViziaAPI)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory( vizia_doom_src )

find_package(PythonInterp)
find_package(PythonLibs)
#find_package(Threads REQUIRED)  
find_package(SDL2 REQUIRED)
find_package(Boost COMPONENTS system thread chrono python REQUIRED)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
#add_definitions("-Wall")
add_definitions("-fPIC")

set( VIZIA_INCLUDE_DIR vizia_api_src vizia_api_bindings/python )

set( VIZIA_LIBS ${VIZIA_LIBS} 
    ${Boost_LIBRARIES} 
    ${PYTHON_LIBRARIES} 
    ${SDL2_LIBRARY} 
    #${CMAKE_THREAD_LIBS_INIT} 
    rt pthread)
    
include_directories( ${VIZIA_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${SDL2_INCLUDE_DIR} )    

set( VIZIA_API_SOURCES
	vizia_api_src/ViziaDoomController.cpp
	vizia_api_src/ViziaDoomGame.cpp )

add_library(vizia ${VIZIA_API_SOURCES})
target_link_libraries (vizia ${VIZIA_LIBS}) 

set( VIZIA_PYTHON_SOURCES
    vizia_api_bindings/python/ViziaPythonWrapper.cpp
    vizia_api_bindings/python/ViziaDoomGamePython.cpp  )

python_add_module(python_vizia ${VIZIA_PYTHON_SOURCES})
target_link_libraries (python_vizia ${VIZIA_LIBS} vizia) 

add_executable( doomControllerExample 
    examples/c++/ViziaDoomControllerExample.cpp )
target_link_libraries( doomControllerExample ${VIZIA_LIBS} vizia )

add_executable( doomGameExample 
    examples/c++/ViziaDoomGameExample.cpp )
target_link_libraries( doomGameExample ${VIZIA_LIBS} vizia )

add_executable( doomSpectatorExample
        examples/c++/ViziaDoomSpectatorExample.cpp)
target_link_libraries( doomSpectatorExample ${VIZIA_LIBS} vizia )

add_executable( doomAxisExample 
    examples/c++/ViziaDoomAxisExample.cpp )
target_link_libraries( doomAxisExample ${VIZIA_LIBS} vizia )

set_target_properties( vizia doomGameExample doomControllerExample doomSpectatorExample doomAxisExample
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties( python_vizia 
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
    OUTPUT_NAME vizia
)	
