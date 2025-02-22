#include <vector>
#include <cmath>
#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/System/Clock.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics.hpp>

#include "SimpleWindow.hpp"
#include "Network.hpp"
#include "Layer.hpp"

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

void BuildNetwork(Network& nn)
{
    std::vector<std::vector<std::vector<float>>> weights{
        {{0.8f, 0.3f}, {0.4f, 0.2f}},
        {{0.6f, 0.8f}, {0.9f, 0.1f}}
    };

    nn = Network(2, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE, weights);

    std::vector<float> inputs = { 0.1f, 0.5f };
    std::vector<float> expected = { 0.05f, 0.95f };
    nn.Fit(inputs, expected);

    auto preds = nn.Predict(inputs);
    for (size_t i = 0; i < preds.size(); i++) {
        std::printf("Prediction output %d before fitting %f\n", (int)i, preds[i]);
    }

    for (size_t i = 0; i < 1000; i++) {
        nn.Fit(inputs, expected);
    }

    preds = nn.Predict(inputs);
    for (size_t i = 0; i < preds.size(); i++) {
        std::printf("Prediction output %d after fitting %f\n", (int)i, preds[i]);
    }
}


int main() {
    SimpleWindow m_window{ "Neural Network Builder", {1280, 720} };
    sf::Clock m_clock;
    float m_deltaTime;
    Network nn;
    BuildNetwork(nn);

    const std::vector<Layer>& layers = nn.GetLayers();

    sf::CircleShape circle(20.f);
    circle.setOutlineThickness(2.f);
    circle.setOutlineColor(sf::Color::White);
    circle.setFillColor(sf::Color::Transparent);
    float xOff = 100.f;
    float yOff = 60.f;
    auto pos = static_cast<sf::Vector2f>(m_window.getWindowSize()) / 2.f - sf::Vector2f{xOff, yOff};
    auto nextPos = pos;

    while (m_window.isOpen()) {
        m_window.processEvents();
        m_deltaTime = m_clock.restart().asSeconds();
        m_window.beginDraw();
        
        for (auto& l : layers) {
            for (auto& n : l.getNeurons()) {
                circle.setPosition(nextPos);
                m_window.draw(circle);
                nextPos += {0.f, yOff};
            }
            nextPos = {nextPos.x + xOff, pos.y};
        }
        nextPos = pos;

        m_window.endDraw();
    }
    return 0;
}