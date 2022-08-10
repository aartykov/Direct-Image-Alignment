cmake_minimum_required(VERSION 3.0.0)
project(optical_flow VERSION 0.1.0)

include(CTest)
enable_testing()


# Eigen
include_directories( "/usr/include/eigen3" )

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})


add_executable(optical_flow main.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(OF_GN OpticalFlow_GN.cpp)
target_link_libraries(OF_GN ${OpenCV_LIBS})

add_executable(directMethod direct_method.cpp)
target_link_libraries(directMethod ${OpenCV_LIBS} Sophus::Sophus)

add_executable(directMethodG2O direct_method_g2o.cpp)
target_link_libraries(directMethodG2O ${OpenCV_LIBS} Sophus::Sophus ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ) #  ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY}

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
