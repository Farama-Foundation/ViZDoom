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
find_package(Boost COMPONENTS system thread python REQUIRED)

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
	 
add_executable( doomControllerExample 
    ${VIZIA_API_SOURCES}
    examples/c++/ViziaDoomControllerExample.cpp )
target_link_libraries( doomControllerExample ${VIZIA_LIBS} vizia )

add_executable( doomGameExample 
    ${VIZIA_API_SOURCES}
    examples/c++/ViziaDoomGameExample.cpp )
target_link_libraries( doomGameExample ${VIZIA_LIBS} vizia )

set_target_properties( vizia doomGameExample doomControllerExample 
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)	
