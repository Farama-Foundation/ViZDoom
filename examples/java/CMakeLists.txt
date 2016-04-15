cmake_minimum_required( VERSION 3.0 )
project(ViZDoomJavaExamples)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(Java REQUIRED)
include(UseJava)

set(VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/examples )

set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/java/vizdoom.jar)
#set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../vizdoom_api_bindings/java)
	
add_jar(basic Basic.java
        ENTRY_POINT Basic
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(cig CIG.java
        ENTRY_POINT CIG
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(cigbots CIGBots.java
        ENTRY_POINT CIGBots 
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(cighost CIGHost.java
        ENTRY_POINT CIGHost
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(deltabuttons DeltaButtons.java
        ENTRY_POINT DeltaButtons 
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(multiplayer Multiplayer.java 
        ENTRY_POINT Multiplayer
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(multiplayerhost MultiplayerHost.java 
        ENTRY_POINT MultiplayerHost
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(seed Seed.java
        ENTRY_POINT Seed
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(shaping Shaping.java
        ENTRY_POINT Shaping
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})

add_jar(spectator Spectator.java
        ENTRY_POINT Spectator
        OUTPUT_DIR ${VIZDOOM_JAVA_EXAMPLES_OUTPUT_DIR})