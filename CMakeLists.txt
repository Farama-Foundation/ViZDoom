cmake_minimum_required( VERSION 2.4 )
project(ViZDoomEnvironment)

option(BUILD_PYTHON "Build ViZDoom API Python binding." OFF)
option(BUILD_JAVA "Build ViZDoom API Java binding." OFF)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_USE_DEBUG_RUNTIME OFF)

add_subdirectory( vizdoom_src )

find_package(Boost COMPONENTS system filesystem thread REQUIRED)
find_package(Threads REQUIRED)

set( VIZDOOM_INCLUDE_DIR vizdoom_api_src )

include_directories( ${VIZDOOM_INCLUDE_DIR} ${Boost_INCLUDE_DIR} )

set( VIZDOOM_LIBS
    ${Boost_LIBRARIES}  
    ${CMAKE_THREAD_LIBS_INIT} )
    
if(UNIX)

    set( VIZDOOM_LIBS ${VIZDOOM_LIBS} 
    pthread
    rt )

endif()
  
if(CMAKE_COMPILER_IS_GNUCXX)

    add_definitions("-fPIC")

endif()  
    
set( VIZDOOM_API_SOURCES
    vizdoom_api_src/ViZDoomUtilities.cpp
	vizdoom_api_src/ViZDoomController.cpp
	vizdoom_api_src/ViZDoomGame.cpp )

add_library(vizdoomstatic STATIC ${VIZDOOM_API_SOURCES})
target_link_libraries (vizdoomstatic ${VIZDOOM_LIBS})

add_library(vizdoomshared SHARED ${VIZDOOM_API_SOURCES})
target_link_libraries (vizdoomshared ${VIZDOOM_LIBS})

set_target_properties( vizdoomstatic vizdoomshared
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    OUTPUT_NAME vizdoom )

if(BUILD_PYTHON)

    #set(Boost_USE_DEBUG_PYTHON OFF)
    
    find_package(PythonInterp 2.7 REQUIRED)
    find_package(PythonLibs 2.7 REQUIRED)
    find_package(Boost COMPONENTS python REQUIRED)
    
    message( ${PYTHON_INCLUDE_DIR})

    set( VIZDOOM_PYTHON_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} vizdoom_api_bindings/python )

    set( VIZDOOM_PYTHON_LIBS 
        ${VIZDOOM_LIBS}
        ${Boost_LIBRARIES}  
        ${PYTHON_LIBRARIES} )
        
    include_directories( ${VIZDOOM_INCLUDE_DIR} 
                        ${Boost_INCLUDE_DIR} 
                        ${PYTHON_INCLUDE_DIR} )  

    set( VIZDOOM_PYTHON_SOURCES
	    vizdoom_api_bindings/python/ViZDoomPythonBinding.cpp
        vizdoom_api_bindings/python/ViZDoomGamePython.cpp )

    python_add_module(vizdoompython ${VIZDOOM_PYTHON_SOURCES})
    target_link_libraries (vizdoompython ${VIZDOOM_PYTHON_LIBS} vizdoomstatic)

    set_target_properties( vizdoompython
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/python
        OUTPUT_NAME vizdoom )
        
endif()

if(BUILD_JAVA)

    if (NOT DEFINED ENV{JAVA_HOME})
        message( FATAL_ERROR "JAVA_HOME environment variable is not set." )
    endif()
        
    find_package(Java REQUIRED)
    find_package(JNI REQUIRED)
    include(UseJava)
    set( VIZDOOM_JAVA_INCLUDE_DIR ${VIZDOOM_INCLUDE_DIR} vizdoom_api_bindings/java )
    
    set( VIZDOOM_JAVA_LIBS 
        ${VIZDOOM_LIBS}
        ${Boost_LIBRARIES}  
        ${Java_LIBRARIES}
        ${JNI_LIBRARIES} )
        
    include_directories( ${VIZDOOM_INCLUDE_DIR} 
                        ${Boost_INCLUDE_DIR}
                        ${Java_INCLUDE_DIRS} 
                        ${_JAVA_HOME}/include
			            ${_JAVA_HOME}/include/linux
                        ${JNI_INCLUDE_DIR} )  

    set( VIZDOOM_JAVA_SOURCES
	    vizdoom_api_bindings/java/ViZDoomGameJava.cpp )


    add_library(vizdoomjava SHARED ${VIZDOOM_JAVA_SOURCES})
    target_link_libraries (vizdoomjava ${VIZDOOM_JAVA_LIBS} vizdoomstatic)

	exec_program(javac .
             ARGS ./vizdoom_api_bindings/java/State.java)
	exec_program(javac .
             ARGS ./vizdoom_api_bindings/java/errors/*.java)
	exec_program(javac .
		     ARGS ./vizdoom_api_bindings/java/enums/*.java)

    set_target_properties( vizdoomjava
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/java
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/java
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/java
        OUTPUT_NAME vizdoom )
        	
endif()
