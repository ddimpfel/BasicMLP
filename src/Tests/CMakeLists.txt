# Separate project to build for testing only
cmake_minimum_required(VERSION 3.31.2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(tests
    LANGUAGES CXX
    VERSION 0.5
) 

add_subdirectory(googletest)
# For Windows: Prevent overriding the parent 
# project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

enable_testing()

set(CPP_FILES  
    ../Network.cpp
    ../Layer.cpp 
    ../Neuron.cpp
)
set(HPP_FILES 
    ../Network.hpp 
    ../nnMath.hpp 
    ../Layer.hpp 
    ../Neuron.hpp
)

add_executable(
    ${PROJECT_NAME}
    ${CPP_FILES}
    ${HPP_FILES}
    NetworkTests.cpp
)
target_link_libraries(
    ${PROJECT_NAME} GTest::gtest_main
)

# Optional grouping in VS Solution
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Tests")

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
