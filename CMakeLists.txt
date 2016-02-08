cmake_minimum_required( VERSION 2.4 )
project(ViziaAPI)

option(BUILD_VIZIA_PYTHON "Build Vizia Python binding." OFF)
option(BUILD_VIZIA_JAVA "Build Vizia Python binding." OFF)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

add_subdirectory( vizia_doom_src )

find_package(Boost COMPONENTS system filesystem thread REQUIRED)
find_package(Threads REQUIRED)  
add_definitions("-fPIC")

set( VIZIA_INCLUDE_DIR vizia_api_src )

include_directories( ${VIZIA_INCLUDE_DIR} ${Boost_INCLUDE_DIR} )

set( VIZIA_LIBS
    ${Boost_LIBRARIES}  
    ${CMAKE_THREAD_LIBS_INIT} 
    rt )
    
set( VIZIA_API_SOURCES
    vizia_api_src/ViziaDoomUtilities.cpp
	vizia_api_src/ViziaDoomController.cpp
	vizia_api_src/ViziaDoomGame.cpp )

add_library(viziastatic STATIC ${VIZIA_API_SOURCES})
target_link_libraries (viziastatic ${VIZIA_LIBS}) 

add_library(viziashared SHARED ${VIZIA_API_SOURCES})
target_link_libraries (viziashared ${VIZIA_LIBS}) 

set_target_properties( viziastatic viziashared 
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    OUTPUT_NAME vizia
)

if(BUILD_VIZIA_PYTHON)
    find_package(PythonInterp)
    find_package(PythonLibs)
    find_package(Boost COMPONENTS python REQUIRED)

    set( VIZIA_PYTHON_INCLUDE_DIR ${VIZIA_INCLUDE_DIR} vizia_api_bindings/python )

    set( VIZIA_PYTHON_LIBS 
        ${VIZIA_LIBS}
        ${Boost_LIBRARIES}  
        ${PYTHON_LIBRARIES} )
        
    include_directories( ${VIZIA_INCLUDE_DIR} 
                        ${Boost_INCLUDE_DIR} 
                        ${PYTHON_INCLUDE_DIR} )  

    set( VIZIA_PYTHON_SOURCES
	    vizia_api_bindings/python/ViziaPythonBinding.cpp
        vizia_api_bindings/python/ViziaDoomGamePython.cpp )

    python_add_module(viziapython ${VIZIA_PYTHON_SOURCES})
    target_link_libraries (viziapython ${VIZIA_PYTHON_LIBS} viziastatic) 

    set_target_properties( viziapython
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
        OUTPUT_NAME vizia
    )	
endif()

if(BUILD_VIZIA_JAVA)

    find_package(Java REQUIRED)
    find_package(JNI REQUIRED)
    set( VIZIA_JAVA_INCLUDE_DIR ${VIZIA_INCLUDE_DIR} vizia_api_bindings/java )
    
    set( VIZIA_PYTHON_LIBS 
        ${VIZIA_LIBS}
        ${Boost_LIBRARIES}  
        ${Java_LIBRARIES}
        ${JNI_LIBRARIES} )
        
    include_directories( ${VIZIA_INCLUDE_DIR} 
                        ${Boost_INCLUDE_DIR}
                        ${Java_INCLUDE_DIRS} 
                        ${_JAVA_HOME}/include
                        ${JNI_INCLUDE_DIR} )  

    set( VIZIA_JAVA_SOURCES
	    vizia_api_bindings/java/ViziaDoomGameJava.cpp )

    add_library(viziajava SHARED ${VIZIA_JAVA_SOURCES})
    target_link_libraries (viziajava ${VIZIA_PYTHON_LIBS} viziastatic) 

    set_target_properties( viziajava 
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/java
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/java
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/java
        OUTPUT_NAME vizia
    )	
endif()
