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
#include <FastNoiseLite.h>
#include "Parameters.hpp"
#include <cstdint>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <box2d/types.h>
#include <box2d/box2d.h>


#define TWO_PI 2*3.14


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Neural Network functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    //std::vector<std::vector<std::vector<float>>> weights{
    //    {{0.8f, 0.3f}, {0.4f, 0.2f}, {0.9f, 0.1f}},
    //    {{0.6f, 0.8f, 0.1f}, {0.9f, 0.8f, 0.1f}}
    //};

    nn = Network({ 4, 2, 3, 2, 2, 3, 2 }, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE, // weights);
        std::make_unique<std::uniform_real_distribution<float>>(dist),
        std::make_unique<std::mt19937>(gen)
    );
}

void ShowNetworkVariablesWindow(std::uniform_real_distribution<float>& dist, std::mt19937& gen, 
    std::vector<float>& expected, std::vector<float>& inputs)
{
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
            expected[i] = dist(gen) * 5.f;
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
            expected[i] = dist(gen) * 5.f;
        }
        for (size_t i = 0; i < inputs.size(); i++)
        {
            inputs[i] = dist(gen);
        }
    }
    ImGui::End();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interesting noise generations
/////////////////////////////////////////////////////////////////////////////////////////////////////////


sf::Sprite GenerateHDPerlinNoiseMapEdgy(sf::View& view, sf::Texture& pixelBuffer, std::vector<uint8_t>& pixels)
{
    // Generate Perlin noise map
    FastNoiseLite perlin;
    perlin.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

    std::vector<float> noise(param::WIDTH * param::HEIGHT);

    view.setCenter({ param::WIDTH / 2, param::HEIGHT / 2 });
    int idx = 0;
    for (size_t y = 0; y < param::HEIGHT; y++)
    {
        for (size_t x = 0; x < param::WIDTH; x++)
        {
            // [-1, 1] gives sharp edges with faded centers
            noise[idx] = perlin.GetNoise((float)x, (float)y);
            int pIdx = idx * 4;
            pixels[pIdx] = noise[idx] * 255;
            pixels[pIdx + 1] = noise[idx] * 255;
            pixels[pIdx + 2] = noise[idx] * 255;
            pixels[pIdx + 3] = 255;
            idx++;
        }
    }

    pixelBuffer.update(&pixels[0]);
    sf::Sprite pixelSprite(pixelBuffer);
    return pixelSprite;
}

sf::Sprite GenerateHDPerlinNoiseMapWhack(sf::View& view, sf::Texture& pixelBuffer, std::vector<uint8_t>& pixels)
{
    // Generate Perlin noise map
    FastNoiseLite perlin;
    perlin.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

    std::vector<float> noise(param::WIDTH * param::HEIGHT);

    view.setCenter({ param::WIDTH / 2, param::HEIGHT / 2 });
    int idx = 0;
    for (size_t y = 0; y < param::HEIGHT; y++)
    {
        for (size_t x = 0; x < param::WIDTH; x++)
        {
            // [-0.5, 1.5] gives perlin noise with sharp islands of white or black
            noise[idx] = perlin.GetNoise((float)x, (float)y) + 1.f / 2.f;
            int pIdx = idx * 4;
            pixels[pIdx] = noise[idx] * 255;
            pixels[pIdx + 1] = noise[idx] * 255;
            pixels[pIdx + 2] = noise[idx] * 255;
            pixels[pIdx + 3] = 255;
            idx++;
        }
    }

    pixelBuffer.update(&pixels[0]);
    sf::Sprite pixelSprite(pixelBuffer);
    return pixelSprite;
}

sf::Sprite GenerateHDPerlinNoiseMap(sf::View& view, sf::Texture& pixelBuffer, std::vector<uint8_t>& pixels)
{
    // Generate Perlin noise map
    FastNoiseLite perlin;
    perlin.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    perlin.SetFractalType(FastNoiseLite::FractalType_FBm);

    perlin.SetFrequency(param::fNoiseFreq);      
    perlin.SetFractalOctaves(param::NoiseOctaves);
    perlin.SetFractalLacunarity(param::fNoiseLacunarity);
    perlin.SetFractalGain(param::fNoiseGain);

    std::vector<float> noise(param::WIDTH * param::HEIGHT);

    view.setCenter({ param::WIDTH / 2, param::HEIGHT / 2 });
    int idx = 0;

    for (size_t y = 0; y < param::NoiseSize; y++)
    {
        for (size_t x = 0; x < param::NoiseSize; x++)
        {
            noise[idx] = (perlin.GetNoise((float)x, (float)y) + 1.f) / 2.f;
            int pIdx = idx * 4;
            pixels[pIdx] = noise[idx] * 255;
            pixels[pIdx + 1] = noise[idx] * 255;
            pixels[pIdx + 2] = noise[idx] * 255;
            pixels[pIdx + 3] = 255;
            idx++;
        }
    }

    pixelBuffer.update(&pixels[0]);
    sf::Sprite pixelSprite(pixelBuffer);
    return pixelSprite;
}

sf::Sprite GeneratePerlinNoiseMap(sf::View& view, sf::Texture& noiseBuffer, std::vector<uint8_t>& pixels)
{
    FastNoiseLite perlin;
    perlin.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    perlin.SetFractalType(FastNoiseLite::FractalType_FBm);

    // Modify Noise parameters                           
    perlin.SetFrequency(param::fNoiseFreq);
    perlin.SetFractalOctaves(param::NoiseOctaves);
    perlin.SetFractalLacunarity(param::fNoiseLacunarity);
    perlin.SetFractalGain(param::fNoiseGain);

    // Allocate arrays for noise map and texture buffer
    std::vector<float> noise(param::NoiseSize * param::NoiseSize);
    sf::Vector2u size{ (unsigned int)param::NoiseSize, (unsigned int)param::NoiseSize };
    noiseBuffer.resize(size);
    pixels.resize(param::NoiseSize * param::NoiseSize * 4);

    // Get noise value for each noise tile and add them to texture array
    int idx = 0;
    for (size_t y = 0; y < param::NoiseSize; y++)
    {
        for (size_t x = 0; x < param::NoiseSize; x++)
        {
            noise[idx] = (perlin.GetNoise((float)x, (float)y) + 1.f) / 2.f;
            int pIdx = idx * 4;
            pixels[pIdx] = noise[idx] * 255;
            pixels[pIdx + 1] = noise[idx] * 255;
            pixels[pIdx + 2] = noise[idx] * 255;
            pixels[pIdx + 3] = 255;
            idx++;
        }
    }

    // Update render objects
    noiseBuffer.update(&pixels[0]);
    sf::Sprite noiseSprite(noiseBuffer);

    // Update texture to match view size and position
    const sf::Vector2f& viewSize = view.getSize();
    sf::Vector2f scale(viewSize.x / param::NoiseSize, viewSize.y / param::NoiseSize);
    noiseSprite.scale(scale);
    noiseSprite.setPosition(view.getCenter() - viewSize / 2.f);

    return noiseSprite;
}
void ShowPerlinNoiseWindow(sf::Sprite& pixelSprite, sf::View& view, sf::Texture& pixelBuffer, std::vector<uint8_t>& pixels)
{
    ImGui::Begin("Noise Values");
    bool valuesChanged = false;

    valuesChanged |= ImGui::InputInt("Octaves", &param::NoiseOctaves);
    valuesChanged |= ImGui::InputFloat("Frequency", &param::fNoiseFreq, 0.01, 0.05, "%.3f");
    valuesChanged |= ImGui::InputFloat("Lacunarity", &param::fNoiseLacunarity, 0.05, 0.3, "%.2f");
    valuesChanged |= ImGui::InputFloat("Gain", &param::fNoiseGain, 0.05, 0.3, "%.2f");

    ImGui::End();

    if (valuesChanged)
        pixelSprite = GeneratePerlinNoiseMap(view, pixelBuffer, pixels);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Physics helpers
/////////////////////////////////////////////////////////////////////////////////////////////////////////

enum CollisionCategories
{
    C_WALL = 0b0001,
    C_CIRCLE = 0b0010,
};

b2ChainId CreateChainBody(b2BodyId body, int32_t pointCount, std::vector<b2Vec2>& points)
{

    b2ChainDef chain = b2DefaultChainDef();
    chain.count = pointCount;
    chain.points = points.data();
    chain.filter.categoryBits = C_WALL;
    chain.filter.maskBits = C_CIRCLE;
    chain.isLoop = true;

    return b2CreateChain(body, &chain);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Draw a circle on a 2D Perlin noise map to create a wavy circle for display
/////////////////////////////////////////////////////////////////////////////////////////////////////////


void GeneratePerlinNoiseLoop(sf::VertexArray& vertices, size_t vertexCount)
{
    FastNoiseLite perlin;
    perlin.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    perlin.SetFractalType(FastNoiseLite::FractalType_FBm);

    // Modify Noise parameters                            
    perlin.SetFrequency(param::fNoiseFreq);     
    perlin.SetFractalOctaves(param::NoiseOctaves); 
    perlin.SetFractalLacunarity(param::fNoiseLacunarity); 
    perlin.SetFractalGain(param::fNoiseGain);

    // Polar coordinates loop to create circle from noise
    size_t idx = 0;
    for (size_t a = 0; a < vertexCount; a++)
    {
        // Angle determined here to avoid float issues when looping with a float
        float angle = (static_cast<float>(a) / param::VertexMultiplier);

        // x and y of circle for noise values
        float xOff = (cos(angle) + 1) * param::fNoiseRadialMultiplier;
        float yOff = (sin(angle) + 1) * param::fNoiseRadialMultiplier;

        // determine radius by perlin noise (x, y) value
        float r = (perlin.GetNoise(xOff, yOff) + 1) * 50 + 100; // radius mapped to 100-200
        
        // x, y in world coordinates
        float x = r * cos(angle);
        float y = r * sin(angle);
        vertices[idx++] = sf::Vertex{ {x, y} };
    }
    // Connect last point to beginning
    vertices[idx] = vertices[0];
}

void ShowPerlinNoiseLoopWindow(sf::VertexArray& vertices, size_t vertexCount)
{
    ImGui::Begin("Loop Values");
    bool valuesChanged = false;

    valuesChanged |= ImGui::InputInt("Octaves", &param::NoiseOctaves, 1);
    valuesChanged |= ImGui::DragFloat("Frequency", &param::fNoiseFreq, 0.005f);
    valuesChanged |= ImGui::DragFloat("Lacunarity", &param::fNoiseLacunarity, 0.05f);
    valuesChanged |= ImGui::DragFloat("Gain", &param::fNoiseGain, 0.05f);

    ImGui::TextUnformatted("Path Multipliers");

    valuesChanged |= ImGui::DragInt("Vertex", &param::VertexMultiplier, 1);
    valuesChanged |= ImGui::DragFloat("Offset", &param::fNoiseRadialMultiplier, 1);

    ImGui::End();

    if (valuesChanged)
    {
        vertexCount = TWO_PI * param::VertexMultiplier;
        vertices.resize(vertexCount + 1);
        GeneratePerlinNoiseLoop(vertices, vertexCount);
    }
}

void GenerateSimplexNoisePath(int seed, size_t vertexCount, sf::VertexArray& displayVerticesInner, sf::VertexArray& displayVerticesOuter,
    std::vector<b2Vec2>& chainVerticesInner, std::vector<b2Vec2>& chainVerticesOuter)
{
    FastNoiseLite perlin;
    perlin.SetSeed(seed);
    perlin.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    perlin.SetFractalType(FastNoiseLite::FractalType_FBm);

    // Modify Noise parameters                              // Ideal values for simple loop
    perlin.SetFrequency(param::fNoiseFreq);                 // 0.01
    perlin.SetFractalOctaves(param::NoiseOctaves);          // 1
    perlin.SetFractalLacunarity(param::fNoiseLacunarity);   // 2
    perlin.SetFractalGain(param::fNoiseGain);               // 0.5

    // Polar coordinates loop to create circle from noise
    size_t idx = 0;
    for (size_t a = 0; a < vertexCount; a++)
    {
        // Angle determined here to avoid float issues when looping with a float
        float angle = (static_cast<float>(a) / param::VertexMultiplier);    // 20

        // x and y of circle for noise values
        float xOff = (cos(angle) + 1) * param::fNoiseRadialMultiplier;      // 65
        float yOff = (sin(angle) + 1) * param::fNoiseRadialMultiplier;      // 65

        // Determine inner radius for x and y in world coordinates
        float rInner = (perlin.GetNoise(xOff, yOff) + 1) * param::fInnerRadiusScalar + param::fInnerRadiusMin; // radius mapped from min to (min + 2*scalar)
        float xInner = rInner * cos(angle);
        float yInner = rInner * sin(angle);
        displayVerticesInner[idx] = sf::Vertex{ {xInner, yInner} };
        chainVerticesInner[idx] = b2Vec2{ xInner / param::fBOX2D_SCALE, yInner / param::fBOX2D_SCALE };

        // Determine outer radius based on inner, ensuring always further away
        float rOuter = rInner + param::fPathWidth;
        float xOuter = rOuter * cos(angle);
        float yOuter = rOuter * sin(angle);
        displayVerticesOuter[idx] = sf::Vertex{ {xOuter, yOuter} };
        chainVerticesOuter[idx++] = b2Vec2{ xOuter / param::fBOX2D_SCALE, yOuter / param::fBOX2D_SCALE };
    }
    // Connect last point to beginning
    displayVerticesInner[idx] = displayVerticesInner[0];
    displayVerticesOuter[idx] = displayVerticesOuter[0];

    // Reverse outer chain winding for collision
    std::reverse(chainVerticesOuter.begin(), chainVerticesOuter.end());
}

void ShowSimplexNoisePathWindow(int seed, size_t vertexCount, sf::VertexArray& displayVerticesInner, sf::VertexArray& displayVerticesOuter, 
    std::vector<b2Vec2>& chainVerticesInner, std::vector<b2Vec2>& chainVerticesOuter, b2ChainId* chainOuter, b2BodyId bodyOuter)
{
    ImGui::Begin("Loop Values");
    bool valuesChanged = false;

    valuesChanged |= ImGui::InputInt("Octaves", &param::NoiseOctaves, 1);
    valuesChanged |= ImGui::DragFloat("Frequency", &param::fNoiseFreq, 0.005f);
    valuesChanged |= ImGui::DragFloat("Lacunarity", &param::fNoiseLacunarity, 0.05f);
    valuesChanged |= ImGui::DragFloat("Gain", &param::fNoiseGain, 0.05f);
    valuesChanged |= ImGui::DragFloat("Offset", &param::fNoiseRadialMultiplier, 1.f);

    ImGui::NewLine();
    ImGui::TextUnformatted("Path");

    // Clamp vertices to be non negative, avoid vector underflow
    int clampVertexMultiplier = param::VertexMultiplier;
    valuesChanged |= ImGui::DragInt("Vertices", &clampVertexMultiplier, 1);
    if (valuesChanged && clampVertexMultiplier > 0)
        param::VertexMultiplier = clampVertexMultiplier;

    valuesChanged |= ImGui::DragFloat("Radius Minimum", &param::fInnerRadiusMin, 1.f);
    valuesChanged |= ImGui::DragFloat("Radius Scalar", &param::fInnerRadiusScalar, 1.f);
    valuesChanged |= ImGui::DragFloat("Width", &param::fPathWidth, 1.f);

    ImGui::End();

    if (valuesChanged)
    {
        vertexCount = TWO_PI * param::VertexMultiplier;
        displayVerticesInner.resize(vertexCount + 1);
        displayVerticesOuter.resize(vertexCount + 1);
        chainVerticesInner.resize(vertexCount);
        chainVerticesOuter.resize(vertexCount);
        GenerateSimplexNoisePath(seed, vertexCount, displayVerticesInner, displayVerticesOuter, chainVerticesInner, chainVerticesOuter);
        
        b2DestroyChain(*chainOuter);
        *chainOuter = CreateChainBody(bodyOuter, vertexCount, chainVerticesOuter);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() 
{
    SimpleWindow m_window{ "Neural Network Builder", {param::WIDTH, param::HEIGHT} };
    if (!ImGui::SFML::Init(m_window.get()))
        return -1;

    sf::View view{};
    view.setSize(m_window.getWindowSizeF());
    view.setCenter({ 0.f, 0.f });
    m_window.setView(view);

    sf::Clock m_clock;
    sf::Time m_deltaTime;

    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_real_distribution<float> dist{ 0.f, 1.f };

    //std::vector<uint8_t> pixels(param::WIDTH * param::HEIGHT * 4);
    //sf::Texture pixelBuffer{ sf::Vector2u(param::WIDTH, param::HEIGHT) };
    //sf::Sprite pixelSprite = GeneratePerlinNoiseMap(view, pixelBuffer, pixels);

    //Network nn;
    //BuildNetwork(nn, dist, gen);
    //std::vector<float> inputs = { dist(gen), dist(gen), dist(gen), dist(gen) };//, dist(gen), dist(gen), dist(gen) };
    //std::vector<float> expected = { 0.f, 0.95f };//, 1.f, 0.0f, 0.f, 0.f, 0.95f, 1.f, 0.0f, 0.f, 0.f, 0.95f, 1.f, 0.0f, 0.f, 0.f, 0.95f, 1.f, 0.0f, 0.f, };


    /*!
     *  Code for race track arena and collisions
     */
    size_t vertexCount = TWO_PI * param::VertexMultiplier;

    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = { 0.f, 0.f };
    b2WorldId world = b2CreateWorld(&worldDef);
  
    std::vector<b2Vec2> chainVerticesInner(vertexCount);
    std::vector<b2Vec2> chainVerticesOuter(vertexCount);

    //sf::VertexArray vertices{ sf::PrimitiveType::LineStrip, vertexCount + 1 };
    sf::VertexArray displayVerticesInner{ sf::PrimitiveType::LineStrip, vertexCount + 1 };
    sf::VertexArray displayVerticesOuter{ sf::PrimitiveType::LineStrip, vertexCount + 1 };
    GenerateSimplexNoisePath(param::SEED, vertexCount, displayVerticesInner, displayVerticesOuter, chainVerticesInner, chainVerticesOuter);

    // Create physics bodies
    // Inner body
    b2BodyDef bodyDef1 = b2DefaultBodyDef();
    bodyDef1.type = b2_staticBody;
    b2BodyId bodyInner = b2CreateBody(world, &bodyDef1);
    b2ChainId chainInner = CreateChainBody(bodyInner, vertexCount, chainVerticesInner);

    // Outer body
    b2BodyDef bodyDef2 = b2DefaultBodyDef();
    bodyDef2.type = b2_staticBody;
    b2BodyId bodyOuter = b2CreateBody(world, &bodyDef2);
    b2ChainId chainOuter = CreateChainBody(bodyOuter, vertexCount, chainVerticesOuter);

    // Test collider
    b2BodyDef dynamicBody = b2DefaultBodyDef();
    dynamicBody.type = b2_dynamicBody;
    dynamicBody.position = { -param::WIDTH / 2 / param::fBOX2D_SCALE, 0.f};
    dynamicBody.linearVelocity = { 3.f, 0.f };
    b2BodyId circleBody = b2CreateBody(world, &dynamicBody);
    
    b2Circle circle{};
    circle.radius = 20.f / param::fBOX2D_SCALE;
    circle.center = b2Vec2(0.f, 0.f);

    b2ShapeDef circleDef = b2DefaultShapeDef();
    circleDef.filter.categoryBits = C_CIRCLE;
    circleDef.filter.maskBits = C_WALL;
    circleDef.enableContactEvents = true;
    b2CreateCircleShape(circleBody, &circleDef, &circle);

    sf::CircleShape displayCircle{ circle.radius * param::fBOX2D_SCALE };
    displayCircle.setOrigin({ circle.radius * param::fBOX2D_SCALE, circle.radius * param::fBOX2D_SCALE });

    int framecounter = 0;
    while (m_window.isOpen())
    {
        // Train the network
        //if (framecounter % 10 == 0)
        //  nn.Fit(inputs, expected);

        // Update
        m_deltaTime = m_clock.restart();
        ImGui::SFML::Update(m_window.get(), m_deltaTime);
        b2World_Step(world, m_deltaTime.asSeconds(), 4);

        // Debug windows
        ImGui::ShowMetricsWindow();
        //ShowNetworkVariablesWindow();
        //ShowPerlinNoiseWindow(pixelSprite, view, pixelBuffer, pixels);
        //ShowPerlinNoiseLoopWindow(vertices, vertexCount);
        ShowSimplexNoisePathWindow(param::SEED, vertexCount, displayVerticesInner, displayVerticesOuter,
            chainVerticesInner, chainVerticesOuter, &chainOuter, bodyOuter);

        const b2Vec2& circlePos = b2Body_GetPosition(circleBody);
        displayCircle.setPosition({ circlePos.x * param::fBOX2D_SCALE, circlePos.y * param::fBOX2D_SCALE });

        ImGui::Begin("Physics Debug");
        ImGui::Text("Circle Position: (%.2f, %.2f)", circlePos.x * param::fBOX2D_SCALE, circlePos.y * param::fBOX2D_SCALE);
        ImGui::Text("Circle Velocity: (%.2f, %.2f)",
            b2Body_GetLinearVelocity(circleBody).x,
            b2Body_GetLinearVelocity(circleBody).y);
        ImGui::End();
        
        m_window.ProcessEvents(view); // Also processes ImGui events

        // Display
        m_window.BeginDraw();
        m_window.Draw(displayVerticesInner);
        m_window.Draw(displayVerticesOuter);
        m_window.Draw(displayCircle);

        //m_window.Draw(pixelSprite);
        //DrawNetwork(nn, m_window, view.getCenter() / 2.f, 20.f);

        ImGui::SFML::Render(m_window.get());
        m_window.EndDraw();

        framecounter++;
    }

    ImGui::SFML::Shutdown();
}