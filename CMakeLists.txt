cmake_minimum_required(VERSION 3.1)

project(explode)

list(APPEND LIBS OpenCL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

list(APPEND SRC_FILES
    main.cpp
    util.cpp
    scene.cpp
    simulation.cpp
)

list(APPEND CL_FILES
    simulate.cl
)

add_executable(explode ${SRC_FILES} ${CL_FILES})
target_compile_features(explode PRIVATE cxx_auto_type)
target_link_libraries(explode ${LIBS})
