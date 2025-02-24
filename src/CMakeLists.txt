# Track files in variable
set(CPP_FILES  
    Main.cpp
    SimpleWindow.cpp
    DrawNetwork.cpp
    Network.cpp
    Layer.cpp 
    Neuron.cpp
)
set(HPP_FILES 
    SimpleWindow.hpp
    Network.hpp 
    Neuron.hpp
    Layer.hpp 
    nnMath.hpp 
)

# Build main executable linked to Network.cpp
add_executable(
    ${PROJECT_NAME}
    ${CPP_FILES}
    ${HPP_FILES}
)

# Sets the target project for Visual Studio
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application")