cmake_minimum_required(VERSION 3.1)

project(explode)

list(APPEND LIBS asprintf)

include_directories(/home/jfuchs/Desktop/boost-compute/include)
list(APPEND LIBS OpenCL)

include_directories(vecmath/include)
add_subdirectory(vecmath)
list(APPEND LIBS vecmath)

list(APPEND SRC_FILES
    main.cpp
    scene.cpp
    util.cpp
    simulation.cpp
)

list(APPEND CL_FILES
    render.cl
    simulate.cl
)

add_executable(explode ${SRC_FILES} ${CL_FILES})
target_compile_features(explode PRIVATE cxx_auto_type)
target_link_libraries(explode ${LIBS})
