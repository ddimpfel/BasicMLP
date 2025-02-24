#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <memory>
#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/System/Clock.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/System/Time.hpp>
#include <SFML/Graphics.hpp>

#include "SimpleWindow.hpp"
#include "Network.hpp"
#include "DrawNetwork.cpp"
#include <SFML/Graphics/View.hpp>

static float ActivationSigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

static float DerivativeActivationSigmoid(float output)
{
    return output * (1.f - output);
}

static float LossMSE(int n, float pred, float expected)
{
    return (2.f / static_cast<float>(n)) * (pred - expected);
}

void BuildNetwork(Network& nn, std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{

    std::vector<std::vector<std::vector<float>>> weights{
        {{0.8f, 0.3f}, {0.4f, 0.2f}, {0.9f, 0.1f}},
        {{0.6f, 0.8f, 0.1f}, {0.9f, 0.8f, 0.1f}}
    };

    nn = Network({ 4, 8, 8, 8, 20 }, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE, // weights);
        std::make_unique<std::uniform_real_distribution<float>>(dist),
        std::make_unique<std::mt19937>(gen)
    );
}


int main() 
{
    SimpleWindow m_window{ "Neural Network Builder", {1280, 720} };
    if (!ImGui::SFML::Init(m_window.get()))
        return -1;
    sf::View view{};
    view.setSize(m_window.getWindowSizeF());
    view.setCenter({ 0.f, 0.f });
    m_window.get().setView(view);

    sf::Clock m_clock;
    sf::Time m_deltaTime;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist{ 0.f, 1.f };

    Network nn;
    BuildNetwork(nn, dist, gen);

    std::vector<float> inputs = { dist(gen), dist(gen), dist(gen), dist(gen) };//, dist(gen), dist(gen), dist(gen) };
    std::vector<float> expected = { 0.f, 0.95f, 1.f, 0.0f, 0.f, 0.f, 0.95f, 1.f, 0.0f, 0.f, 0.f, 0.95f, 1.f, 0.0f, 0.f, 0.f, 0.95f, 1.f, 0.0f, 0.f, };

    int framecounter = 0;
    while (m_window.isOpen())
    {
        // Train the network
        if (framecounter % 30 == 0)
            nn.Fit(inputs, expected);

        m_deltaTime = m_clock.restart();
        ImGui::SFML::Update(m_window.get(), m_deltaTime);

        ImGui::ShowMetricsWindow();
        ImGui::Begin("Variable Editor");
        ImGui::Text("Network Expected Outputs");
        for (size_t i = 0; i < expected.size(); i++)
        {
            ImGui::PushID(i);
            ImGui::InputFloat("", &expected[i], 0.05, 0.2);
            ImGui::PopID();
        }
        if (ImGui::Button("Randomize Expected Outputs")) 
        {
            for (size_t i = 0; i < expected.size(); i++)
            {
                expected[i] = dist(gen);
            }
        }
        if (ImGui::Button("Randomize Inputs"))
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                inputs[i] = dist(gen);
            }
        }
        if (ImGui::Button("Randomize Inputs and Outputs"))
        {
            for (size_t i = 0; i < expected.size(); i++)
            {
                expected[i] = dist(gen);
            }
            for (size_t i = 0; i < inputs.size(); i++)
            {
                inputs[i] = dist(gen);
            }
        }
        ImGui::End();

        m_window.processEvents(view); // Also processes ImGui events

        m_window.beginDraw();
        DrawNetwork(nn, m_window, view.getCenter() / 2.f, 20.f);
        ImGui::SFML::Render(m_window.get());
        m_window.endDraw();

        framecounter++;
    }

    ImGui::SFML::Shutdown();

    return 0;
}