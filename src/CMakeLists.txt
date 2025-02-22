# Track files in variable
set(CPP_FILES  
    Main.cpp
    SimpleWindow.cpp
    Network.cpp
)
set(HPP_FILES 
    SimpleWindow.hpp
    Network.hpp 
    nnMath.hpp 
    Layer.hpp 
    Neuron.hpp
)

# Create object files to link to in Tests/other projects
# add_library(
#     ${PROJECT_NAME} OBJECT
#     ${CPP_FILES}
#     ${HPP_FILES}
# )

# Add folders to group in solution
# source_group("Source Files" FILES ${CPP_FILES})
# source_group("Header Files" FILES ${HPP_FILES})

# Build main executable linked to Network.cpp
add_executable(
    ${PROJECT_NAME}
    ${CPP_FILES}
    ${HPP_FILES}
    # $<TARGET_OBJECTS:${PROJECT_NAME}>
)

# Optional group into VS Solution
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application")