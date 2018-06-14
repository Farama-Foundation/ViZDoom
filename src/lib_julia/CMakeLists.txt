find_package(JlCxx REQUIRED)

add_library(vizdoomjl SHARED ViZDoomJuliaModule.cpp)
target_link_libraries(vizdoomjl JlCxx::cxxwrap_julia ${VIZDOOM_LIBS} libvizdoom_static)