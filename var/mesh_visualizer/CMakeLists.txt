cmake_minimum_required(VERSION 2.8.3)
project(mesh_visualizer)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
set(CMAKE_BUILD_TYPE Release)

set(HEADER_FILES
  include/mesh_visualizer/mesh_visualizer.h
)
set(SRC_FILES
  src/mesh_visualizer.cpp
)


cs_add_library(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})
cs_add_executable(${PROJECT_NAME}_node src/mesh_visualizer_talker.cpp)

# target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES} ${YamlCpp_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

# catkin_python_setup()
cs_install()
cs_export()
