cmake_minimum_required(VERSION 3.31.2)

set(PROJECT_NAME cmakeapp)

project(PROJECT_NAME # CHANGE NAME TO PROJECT
    LANGUAGES CXX
    VERSION 0.1
) 

add_subdirectory(src)
add_subdirectory(vendor)