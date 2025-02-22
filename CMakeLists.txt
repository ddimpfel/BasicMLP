cmake_minimum_required(VERSION 3.31.2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(basic_mlp
    LANGUAGES CXX
    VERSION 0.5
) 

add_subdirectory(src)
add_subdirectory(vendor)