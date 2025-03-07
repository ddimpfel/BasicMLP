# Track files in variable
set(CPP_FILES  
    Main.cpp
    SimpleWindow.cpp
    DrawNetwork.cpp
    Network.cpp
    Layer.cpp 
    Neuron.cpp
    Noise.cpp
)

# Build main executable linked to Network.cpp
add_executable(
    ${PROJECT_NAME}
    ${CPP_FILES}
)

# Link against FastNoiseLite and ImGui-SFML
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ImGui-SFML::ImGui-SFML
    FastNoiseLite
)


# Sets the target project for Visual Studio
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application")