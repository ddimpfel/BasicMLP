#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <cstdint>
#include <vector>
#include <utility>

#include "Network.hpp"
#include "Layer.hpp"
#include "SimpleWindow.hpp"

constexpr float minThickness = 1.0f;
constexpr float thicknessCoeff = 1.f;

// This adds more of a parallelogram than straight line, in that the ends
// are vertcal and not perpendicular to the line direction.
[[maybe_unused]] static size_t AddLine(sf::VertexArray& vertices, size_t idx,
    float xStart, float yStart, float xEnd, float yEnd, float thickness)
{
    vertices[idx++].position = { xStart, yStart - thickness / 2 };
    vertices[idx++].position = { xEnd,   yEnd - thickness / 2 };
    vertices[idx++].position = { xEnd,   yEnd + thickness / 2 };

    vertices[idx++].position = { xEnd,   yEnd + thickness / 2 };
    vertices[idx++].position = { xStart, yStart + thickness / 2 };
    vertices[idx++].position = { xStart, yStart - thickness / 2 };

    return idx;
}

namespace sf
{
namespace nn
{
static void Draw(const Network& nn, SimpleWindow& window,
    const sf::Vector2f& center, float nodeRadius, float horizontalSpacingScalar, float verticalSpacingScalar)
{
    const std::vector<Layer>& layers = nn.getLayers();
    const std::vector<std::vector<float>>& layerOutputs = nn.getLayerOutputs();
    size_t layerCount = layerOutputs.size();

    // Set up circle to represent each neuron
    // TODO change to using vertex array?
    sf::CircleShape circle(nodeRadius);
    circle.setOrigin({ nodeRadius, nodeRadius });
    circle.setOutlineThickness(2.f);
    circle.setOutlineColor(sf::Color::White);

    // Interconnections of two layers = layer_1 neurons * layer_2 neurons
    // Total vertices inb/w layers = interconnects * 6 -> to allow for 
    // lines made of triangles for thickness
    auto& networkArchitecture = nn.getArchitecture();
    size_t totalVertices = 0;
    for (size_t l{ 0 }; l < networkArchitecture.size() - 1; ++l)
    {
        size_t interconnections = networkArchitecture[l] * networkArchitecture[l + 1];
        totalVertices += interconnections * 6;
    }

    // Vertex array for drawing of interconnects between layers
    sf::VertexArray vertices(sf::PrimitiveType::Triangles, totalVertices);
    size_t vertexCounter = 0;

    // Determine the total width for the entire network
    float horizontalGap = nodeRadius * horizontalSpacingScalar;
    float totalWidth = horizontalGap * (layerCount - 1);
    float x = center.x - totalWidth / 2.f;

    // Iterate through all the layers, getting their neurons and outputs
    for (uint64_t l{ 0 }; l < layerCount; ++l)
    {
        auto& layer = layers[l];
        size_t currNeuronsCount = layerOutputs[l].size();
        auto& outputs = layerOutputs[l];

        // Determine the total height for this layer
        float verticalGap = nodeRadius * verticalSpacingScalar;
        float totalHeight = verticalGap * (currNeuronsCount - 1);
        float y = center.y - totalHeight / 2.f;

        for (uint64_t n{ 0 }; n < currNeuronsCount; ++n)
        {
            circle.setPosition({ x, y });
            circle.setFillColor(sf::Color(255, 255, 255,
                static_cast<uint8_t>(255 * outputs[n])));
            window.Draw(circle);

            // Connect this neuron to the next layer's neurons
            if (l < layerCount - 1)
            {
                // Vertices come out of right side of neurons
                float xVert = x + nodeRadius;

                auto& nextLayer = layers[l + 1];
                const auto& nextNeuronsCount = nextLayer.getNeurons().size();
                float xNextVert = x + horizontalGap - nodeRadius;

                // Determine the next layers vertical spacing
                float totalNextHeight = verticalGap * (nextNeuronsCount - 1);
                float yNextVert = center.y - totalNextHeight / 2.f;

                // Add a line connecting to all of the next neurons with thickness proportional to the weights
                for (uint64_t nextN{ 0 }; nextN < nextNeuronsCount; ++nextN)
                {
                    float w = nextLayer.getNeurons()[nextN].getWeights()[n];
                    vertexCounter = AddLine(
                        vertices, vertexCounter,
                        xVert, y, xNextVert, yNextVert,
                        std::max(minThickness, thicknessCoeff * w)
                    );
                    yNextVert += verticalGap;
                }
            }
            y += verticalGap;
        }
        x += horizontalGap;
    }

    window.Draw(vertices);
}
}   // nn
}   // sf