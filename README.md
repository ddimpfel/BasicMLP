# Building the project

## From this directory

Make the app

```sh
# -DBUILD_SHARED_LIBS=OFF sets cmake to use STATIC libraries
cmake -B ./build -DBUILD_SHARED_LIBS=OFF -S ./app
```

Then build the app (for MVSC [--config Debug or Release])

```sh
cmake --build ./build --config Debug
# New build/src directory that the executable is stored in:
cd ./build/src/Debug
basic_mlp
```

## Defining CMakeLists.txt files

### root

Set language requirements and the project name and version

```cmake
cmake_minimum_required(VERSION 3.31.2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(basic_mlp
    LANGUAGES CXX
    VERSION 0.8
) 

# Enable incremental linking
set(CMAKE_INCREMENTAL_LINK ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /INCREMENTAL")

add_subdirectory(src)
add_subdirectory(vendor)
```

### src

This is where your executable and code will be derived from. 
This directory should include header and source files together but only links with sources.
**Incremental linking** used to reduce compile times

```cmake
# Track files in main directory
set(MAIN_SOURCES
    Main.cpp
    etc...
)

# Track subdirectories
add_subdirectory(subdir)
etc...

# Build main executable
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})

# Link against
target_link_libraries(${PROJECT_NAME} PRIVATE
    # Local libaries
    subdir_libA
    etc...

    # Vendor libraries
    ImGui-SFML::ImGui-SFML
    FastNoiseLite
)

# Include directories for main executable
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sets the target project for Visual Studio
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application")
```

### subdir

The PUBLIC and PRIVATE visibility specifiers in target_include_directories and target_link_libraries are crucial:
•	PUBLIC: Dependency is needed in public headers
•	PRIVATE: Dependency is only needed internally

```cmake
set(SUBDIR_SOURCES
    Source.cpp
    etc...
)

add_library(subdir_lib ${NETWORK_SOURCES})

# Include directories so headers can be found
target_include_directories(subdir_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# optional: Link any libraries needed
target_link_libraries(noise_lib
    PRIVATE
    VendorLibrary
)

target_compile_features(network_lib PRIVATE cxx_std_20)
```

### vendor

```cmake
set(IMGUI_DIR "./imgui")
set(IMGUI_SFML_FIND_SFML OFF)
set(IMGUI_SFML_IMGUI_DEMO ON)

add_subdirectory(sfml)
add_subdirectory(imgui-sfml)
add_subdirectory(FastNoiseLite)
```