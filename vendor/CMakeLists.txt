set(IMGUI_DIR "../imgui")
set(IMGUI_SFML_FIND_SFML OFF)
set(IMGUI_SFML_IMGUI_DEMO ON)

add_subdirectory(sfml)
add_subdirectory(imgui-sfml)
target_link_libraries(
    ${PROJECT_NAME}  
    PRIVATE ImGui-SFML::ImGui-SFML
)
