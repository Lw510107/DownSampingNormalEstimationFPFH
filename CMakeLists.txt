cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME SamplingNormalEstimationFPFH)
project(${PROJECT_NAME})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} SamplingNormalEstimationFPFH.cpp)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${BOOST_LIBRARIES})
