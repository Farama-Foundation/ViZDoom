cmake_minimum_required( VERSION 3.0 )
project(ViZDoomJavaExamples)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(Java REQUIRED)
include(UseJava)

set(CMAKE_JAVA_INCLUDE_PATH ../../bin/java/vizdoom.jar)
	
add_jar( basic Basic.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( cig CIG.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( cigbots CIGBots.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( cighost CIGHost.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( deltabuttons DeltaButtons.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( multiplayer Multiplayer.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( multiplayerhost MultiplayerHost.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( seed Seed.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( shaping Shaping.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

add_jar( spectator Spectator.java OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../../bin/java_examples)

