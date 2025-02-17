#include <SFML/Graphics.hpp>
#include <imgui-SFML.h>
#include <imgui.h>

int main()
{
    sf::RenderWindow window(sf::VideoMode({1280, 720}), "SFML works!");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    sf::Clock deltaClock;
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        while (const std::optional event = window.pollEvent())
        {
            if (event.has_value())                  ImGui::SFML::ProcessEvent(window, event.value());
            if (event->is<sf::Event::Closed>())     window.close();
        }
        
        ImGui::SFML::Update(window, deltaClock.restart());
        ImGui::ShowDemoWindow();

        window.clear();
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
        
    return 0;
}