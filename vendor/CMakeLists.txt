# This should only point to vendor code,
# linking is handled in src cmakelists
set(IMGUI_DIR "./imgui")
set(IMGUI_SFML_FIND_SFML OFF)
set(IMGUI_SFML_IMGUI_DEMO ON)

add_subdirectory(box2d)
add_subdirectory(FastNoiseLite)
add_subdirectory(imgui-sfml)
add_subdirectory(sfml)
