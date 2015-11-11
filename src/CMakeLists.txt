cmake_minimum_required( VERSION 2.4 )
project(ViziaDoomControllerExample)

set(CMAKE_CXX_FLAGS "-g -Wall")

file( GLOB_RECURSE HEADER_FILES
    process/*.hpp
    process/process/*.hpp
    process/process/posix/*.hpp
    process/process/windows/*.hpp
    interprocess/*.hpp
    interprocess/ipc/*.hpp
)

include_directories(
    .
)

add_executable(example
    ${HEADER_FILES}
    ViziaDoomController.cpp
    ViziaDoomControllerExample.cpp
)

target_link_libraries( zdoom ${ZDOOM_LIBS} gdtoa dumb lzma )