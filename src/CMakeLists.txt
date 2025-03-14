# Track files in variable
set(MAIN_SOURCES  
    Main.cpp
    SimpleWindow.cpp
    DrawNetwork.cpp
    Network.cpp
    Layer.cpp 
    Neuron.cpp
    Noise.cpp
)

# Build main executable linked to Network.cpp
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})

# Link against FastNoiseLite and ImGui-SFML
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ImGui-SFML::ImGui-SFML
    FastNoiseLite
    box2d::box2d
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sets the target project for Visual Studio
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application")