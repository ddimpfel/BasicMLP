#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <memory>
#include <cstdint>

#include <imgui.h>
#include <imgui-SFML.h>

#include <SFML/System/Clock.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/System/Time.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/View.hpp>

#include <FastNoiseLite.h>

#include <box2d/types.h>
#include <box2d/box2d.h>
#include <box2d/collision.h>
#include <box2d/id.h>
#include <box2d/math_functions.h>

#include "SimpleWindow.hpp"
#include "Network.hpp"
#include "DrawNetwork.cpp"
#include "Parameters.hpp"
#include "Vehicle.hpp"


#define TWO_PI 2*3.14159265359f
#define PI 3.14159265359f


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

void BuildNetwork(Network& nn, std::vector<int>& architecture, std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{

    //std::vector<std::vector<std::vector<float>>> weights{
    //    {{0.8f, 0.3f}, {0.4f, 0.2f}, {0.9f, 0.1f}},
    //    {{0.6f, 0.8f, 0.1f}, {0.9f, 0.8f, 0.1f}}
    //};

    nn = Network(architecture, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE, // weights);
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

    std::vector<float> noise(param::iWIDTH * param::iHEIGHT);

    view.setCenter({ param::iWIDTH / 2, param::iHEIGHT / 2 });
    int idx = 0;
    for (size_t y = 0; y < param::iHEIGHT; y++)
    {
        for (size_t x = 0; x < param::iWIDTH; x++)
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

    std::vector<float> noise(param::iWIDTH * param::iHEIGHT);

    view.setCenter({ param::iWIDTH / 2, param::iHEIGHT / 2 });
    int idx = 0;
    for (size_t y = 0; y < param::iHEIGHT; y++)
    {
        for (size_t x = 0; x < param::iWIDTH; x++)
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
    perlin.SetFractalOctaves(param::iNoiseOctaves);
    perlin.SetFractalLacunarity(param::fNoiseLacunarity);
    perlin.SetFractalGain(param::fNoiseGain);

    std::vector<float> noise(param::iWIDTH * param::iHEIGHT);

    view.setCenter({ param::iWIDTH / 2, param::iHEIGHT / 2 });
    int idx = 0;

    for (size_t y = 0; y < param::iNoiseSize; y++)
    {
        for (size_t x = 0; x < param::iNoiseSize; x++)
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
    perlin.SetFractalOctaves(param::iNoiseOctaves);
    perlin.SetFractalLacunarity(param::fNoiseLacunarity);
    perlin.SetFractalGain(param::fNoiseGain);

    // Allocate arrays for noise map and texture buffer
    std::vector<float> noise(param::iNoiseSize * param::iNoiseSize);
    sf::Vector2u size{ (unsigned int)param::iNoiseSize, (unsigned int)param::iNoiseSize };
    noiseBuffer.resize(size);
    pixels.resize(param::iNoiseSize * param::iNoiseSize * 4);

    // Get noise value for each noise tile and add them to texture array
    int idx = 0;
    for (size_t y = 0; y < param::iNoiseSize; y++)
    {
        for (size_t x = 0; x < param::iNoiseSize; x++)
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
    sf::Vector2f scale(viewSize.x / param::iNoiseSize, viewSize.y / param::iNoiseSize);
    noiseSprite.scale(scale);
    noiseSprite.setPosition(view.getCenter() - viewSize / 2.f);

    return noiseSprite;
}
void ShowPerlinNoiseWindow(sf::Sprite& pixelSprite, sf::View& view, sf::Texture& pixelBuffer, std::vector<uint8_t>& pixels)
{
    ImGui::Begin("Noise Values");
    bool valuesChanged = false;

    valuesChanged |= ImGui::InputInt("Octaves", &param::iNoiseOctaves);
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
    C_VEHICLE = 0b0010,
};

b2ChainId UpdateChainBody(b2BodyId body, int32_t pointCount, std::vector<b2Vec2>& points)
{

    b2ChainDef chain = b2DefaultChainDef();
    chain.count = pointCount;
    chain.points = points.data();
    chain.filter.categoryBits = C_WALL;
    chain.filter.maskBits = C_VEHICLE;
    chain.isLoop = true;

    return b2CreateChain(body, &chain);
}

struct Wall
{
    std::vector<b2Vec2> physicsVertices{ 0 };
    sf::VertexArray displayVertices = sf::VertexArray{ sf::PrimitiveType::LineStrip, 0 };

    b2BodyId bodyId;
    b2ChainId chainId;

    Wall(size_t vertexCount)
    {
        physicsVertices.resize(vertexCount);
        displayVertices.resize(vertexCount + 1);
    }

    void CreateBody(b2WorldId world)
    {
        b2BodyDef bodyDef1 = b2DefaultBodyDef();
        bodyDef1.type = b2_staticBody;
        bodyId = b2CreateBody(world, &bodyDef1);
        chainId = UpdateChainBody(bodyId, physicsVertices.size(), physicsVertices);
    }
};


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
    perlin.SetFractalOctaves(param::iNoiseOctaves); 
    perlin.SetFractalLacunarity(param::fNoiseLacunarity); 
    perlin.SetFractalGain(param::fNoiseGain);

    // Polar coordinates loop to create circle from noise
    size_t idx = 0;
    for (size_t a = 0; a < vertexCount; a++)
    {
        // Angle determined here to avoid float issues when looping with a float
        float angle = (static_cast<float>(a) / param::iVertexMultiplier);

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

    valuesChanged |= ImGui::InputInt("Octaves", &param::iNoiseOctaves, 1);
    valuesChanged |= ImGui::DragFloat("Frequency", &param::fNoiseFreq, 0.005f);
    valuesChanged |= ImGui::DragFloat("Lacunarity", &param::fNoiseLacunarity, 0.05f);
    valuesChanged |= ImGui::DragFloat("Gain", &param::fNoiseGain, 0.05f);

    ImGui::TextUnformatted("Path Multipliers");

    valuesChanged |= ImGui::DragInt("Vertex", &param::iVertexMultiplier, 1);
    valuesChanged |= ImGui::DragFloat("Offset", &param::fNoiseRadialMultiplier, 1);

    ImGui::End();

    if (valuesChanged)
    {
        vertexCount = TWO_PI * param::iVertexMultiplier;
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
    perlin.SetFractalOctaves(param::iNoiseOctaves);          // 1
    perlin.SetFractalLacunarity(param::fNoiseLacunarity);   // 2
    perlin.SetFractalGain(param::fNoiseGain);               // 0.5

    // Polar coordinates loop to create circle from noise
    size_t idx = 0;
    for (size_t a = 0; a < vertexCount; a++)
    {
        // Angle determined here to avoid float issues when looping with a float
        float angle = (static_cast<float>(a) / param::iVertexMultiplier);    // 20

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

void ShowSimplexNoisePathWindow(int seed, size_t vertexCount, Wall& wallInner, Wall& wallOuter)
{
    ImGui::Begin("Noise Path Values");
    bool valuesChanged = false;

    valuesChanged |= ImGui::InputInt("Octaves", &param::iNoiseOctaves, 1);
    valuesChanged |= ImGui::DragFloat("Frequency", &param::fNoiseFreq, 0.005f);
    valuesChanged |= ImGui::DragFloat("Lacunarity", &param::fNoiseLacunarity, 0.05f);
    valuesChanged |= ImGui::DragFloat("Gain", &param::fNoiseGain, 0.05f);
    valuesChanged |= ImGui::DragFloat("Offset", &param::fNoiseRadialMultiplier, 1.f);

    ImGui::NewLine();
    ImGui::TextUnformatted("Path");

    // Clamp vertices to be non negative, avoid vector underflow
    int clampVertexMultiplier = param::iVertexMultiplier;
    valuesChanged |= ImGui::DragInt("Vertices", &clampVertexMultiplier, 1);
    if (valuesChanged && clampVertexMultiplier > 0)
        param::iVertexMultiplier = clampVertexMultiplier;

    valuesChanged |= ImGui::DragFloat("Radius Minimum", &param::fInnerRadiusMin, 1.f);
    valuesChanged |= ImGui::DragFloat("Radius Scalar", &param::fInnerRadiusScalar, 1.f);
    valuesChanged |= ImGui::DragFloat("Width", &param::fPathWidth, 1.f);

    ImGui::End();

    if (valuesChanged)
    {
        vertexCount = TWO_PI * param::iVertexMultiplier;
        wallInner.displayVertices.resize(vertexCount + 1);
        wallOuter.displayVertices.resize(vertexCount + 1);
        wallInner.physicsVertices.resize(vertexCount);
        wallOuter.physicsVertices.resize(vertexCount);
        GenerateSimplexNoisePath(seed, vertexCount, wallInner.displayVertices, wallOuter.displayVertices, wallInner.physicsVertices, wallOuter.physicsVertices);
        
        b2DestroyChain(wallOuter.chainId);
        wallOuter.chainId = UpdateChainBody(wallOuter.bodyId, vertexCount, wallOuter.physicsVertices);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Vehicle
/////////////////////////////////////////////////////////////////////////////////////////////////////////

b2BodyId CreateVehicle(b2WorldId world, float halfWidth, float halfHeight)
{
    // Define physics parameters
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = { -271.f / param::fBOX2D_SCALE, 0.f };
    bodyDef.linearVelocity = { 0.f, 0.f };
    bodyDef.linearDamping = 0.9f;
    bodyDef.angularDamping = 0.9f;
    bodyDef.rotation = b2MakeRot(-3.1415f / 2.f);
    b2BodyId vehicleBody = b2CreateBody(world, &bodyDef);

    // Make body shape
    b2Polygon box = b2MakeBox(halfWidth, halfHeight);

    // Define collider
    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.filter.categoryBits = C_VEHICLE;
    shapeDef.filter.maskBits = C_WALL;
    shapeDef.enableContactEvents = true;

    b2CreatePolygonShape(vehicleBody, &shapeDef, &box);

    return vehicleBody;
}

void VehicleSense(b2WorldId world, b2BodyId vehicleBody, sf::VertexArray& outDrawableRays, std::vector<b2RayResult>& outResults)
{
    // Body parameters
    const b2Vec2& origin = b2Body_GetPosition(vehicleBody);
    const b2Rot& rotation = b2Body_GetRotation(vehicleBody);
    float rotationAsRadians = b2Rot_GetAngle(rotation);

    // Ray parameters
    float rayLengthMeters = 3.f;
    float angleStep = param::fFieldOfView / (outResults.size() - 1);
    // beginning of field of view is half FOV to left of rotation
    float angleStart = rotationAsRadians - param::fFieldOfView / 2.f; 
    b2QueryFilter rayFilter = { C_VEHICLE, C_WALL };
    b2Vec2 translation = { 0.f, 0.f }; // Initialize translation

    // Shoot rays from vehicle
    for (size_t i = 0; i < outResults.size(); i++)
    {
        // Box2d rot for cos and sin parts
        b2Rot rayRot = b2MakeRot(angleStart + angleStep * i);

        // Translation vector of ray
        translation = { rayRot.c * rayLengthMeters, rayRot.s * rayLengthMeters };

        // Cast and draw
        outResults[i] = b2World_CastRayClosest(world, origin, translation, rayFilter);
        outDrawableRays[i*2] = sf::Vertex{ { origin.x * param::fBOX2D_SCALE, origin.y * param::fBOX2D_SCALE }, sf::Color::Yellow };
        outDrawableRays[i*2+1] = sf::Vertex{ { (origin.x + translation.x) * param::fBOX2D_SCALE, (origin.y + translation.y) * param::fBOX2D_SCALE }, sf::Color::Yellow };
    }
}

void VehicleMove(b2BodyId vehicleBody, b2Vec2& acceleration)
{
    b2Vec2 newVelocity = b2Body_GetLinearVelocity(vehicleBody) + acceleration;
    b2Body_SetLinearVelocity(vehicleBody, newVelocity);
    // Box2d handles updating the position in it's world step
    // new_position = (previous_position + new_velocity) * linear_damping
}

std::vector<float>& VehicleThink(Network& vehicleBrain, const std::vector<float>& inputs)
{
    std::vector<float> outputs = vehicleBrain.Predict(inputs);
    return outputs;
}

void VehicleStep(b2BodyId vehicleBody, Network& vehicleBrain, const std::vector<float>& inputs)
{
    std::vector<float>& outputs = VehicleThink(vehicleBrain, inputs);
    b2Vec2 acceleration = { outputs[0], outputs[1] };
    VehicleMove(vehicleBody, acceleration);
}

void DrawVehicle(sf::RenderWindow& window, b2BodyId vehicleBody, float halfWidth, float halfHeight)
{
    const b2Vec2& position = b2Body_GetPosition(vehicleBody);
    const b2Rot& rotation = b2Body_GetRotation(vehicleBody);

    // Display vehicle
    sf::RectangleShape vehicle({ halfWidth * 2.f * param::fBOX2D_SCALE, halfHeight * 2.f * param::fBOX2D_SCALE });
    vehicle.setOrigin({ halfWidth * param::fBOX2D_SCALE, halfHeight * param::fBOX2D_SCALE });
    vehicle.setFillColor(sf::Color::Transparent);
    vehicle.setOutlineColor(sf::Color::White);
    vehicle.setOutlineThickness(1.f);
    vehicle.setPosition({ position.x * param::fBOX2D_SCALE, position.y * param::fBOX2D_SCALE });
    vehicle.setRotation(sf::radians(b2Rot_GetAngle(rotation)));
    window.draw(vehicle);

    // display rotation
    float radius = 25.f;
    sf::VertexArray rotationLine(sf::PrimitiveType::Lines, 2);
    b2Vec2 pos = b2Body_GetPosition(vehicleBody);
    float screenX = pos.x * param::fBOX2D_SCALE;
    float screenY = pos.y * param::fBOX2D_SCALE;
    rotationLine[0] = sf::Vertex{ { screenX, screenY } };
    rotationLine[1] = sf::Vertex{ { rotation.c * radius + screenX, rotation.s * radius + screenY } };
    window.draw(rotationLine);
}

void ShowBestVehicleStatsWindow(Vehicle vehicle)
{
    const b2Vec2& bodyPos = b2Body_GetPosition(vehicle.m_body);
    ImGui::Begin("Last Gen Best Vehicle Stats");
    ImGui::Text("Score: %.2f", vehicle.GetScore());
    ImGui::Text("Vehicle Position: (%.2f, %.2f)", bodyPos.x * param::fBOX2D_SCALE, bodyPos.y * param::fBOX2D_SCALE);
    ImGui::Text("Vehicle Velocity: (%.2f, %.2f)",
        b2Body_GetLinearVelocity(vehicle.m_body).x,
        b2Body_GetLinearVelocity(vehicle.m_body).y);
    ImGui::End();
}

void ShowVehicleScores(int generation, float bestScore, float lastGenBestScore, const std::vector<Vehicle>& vehicles)
{
    ImGui::Begin("Scores");
    ImGui::Text("Best Score: %.2f", bestScore);
    ImGui::NewLine();
    ImGui::Text("Generation %d", generation+1);
    ImGui::Text("Last Gen Best Score: %.2f", lastGenBestScore);
    ImGui::NewLine();
    for (size_t i = 0; i < vehicles.size(); i++)
    {
        ImGui::Text("Vehicle %d: %.2f", i+1, vehicles[i].GetScore());
    }
    ImGui::End();
}

void CreateVehicles(
    std::vector<int>& architecture, 
    std::vector<Vehicle>& vehicles, 
    b2WorldId world, 
    float halfWidth, 
    float halfHeight,
    std::uniform_real_distribution<float>& dist, 
    std::mt19937& gen
)
{
    float randomSeed = dist(gen) * 100000;
    for (size_t i = 0; i < vehicles.size(); i++)
    {
        // Randomize seed for each vehicle
        gen.seed(randomSeed + i);

        vehicles[i] = Vehicle();
        vehicles[i].InitBody(world, halfWidth, halfWidth, -271.f / 50.f, 0.f, -3.1415f / 2.f);

        // inputs are
        // {
        //  pos.x, 
        //  pos.y,
        //  ray_left fraction,
        //  ray_frwd fraction,
        //  ray_rght fraction,
        // }
        // 
        // ouputs are 
        // {
        //  vel.x, 
        //  vel.y, 
        //  throttle, 
        //  rotation
        // }
        vehicles[i].InitBrain(architecture, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE, dist, gen);

        vehicles[i].InitializeScoring();
    }
}

bool SortByScore(const Vehicle& a, const Vehicle& b)
{
    return a.GetScore() > b.GetScore();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
    SimpleWindow m_window{ "Neural Network Builder", {param::iWIDTH, param::iHEIGHT} };
    if (!ImGui::SFML::Init(m_window.get()))
        return -1;

    sf::View view{};
    view.setSize(m_window.getWindowSizeF());
    view.setCenter({ 0.f, 0.f });
    m_window.setView(view);

    sf::Clock m_clock;
    sf::Time m_deltaTime;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist{ -1.f, 1.f };

    //std::vector<uint8_t> pixels(param::iWIDTH * param::iHEIGHT * 4);
    //sf::Texture pixelBuffer{ sf::Vector2u(param::iWIDTH, param::iHEIGHT) };
    //sf::Sprite pixelSprite = GeneratePerlinNoiseMap(view, pixelBuffer, pixels);

    /*!
     *  Soft Actor-Critic RL??
     *  Should outputs be some velocity unit vector U(x, y) and some acceleration pressure?
     *      Acceleration pressure could be [0,1] * maxAccel, allowing fine tuned control of velocity
     *      Will this have isses generalizing to different surface conditions?
     */


    /*!
     *  Code for race track arena and collisions, currently implementing neuro-evolution
     */
    size_t vertexCount = TWO_PI * param::iVertexMultiplier;

    // Define physics world (box2d)
    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = { 0.f, 0.f };
    b2WorldId world = b2CreateWorld(&worldDef);

    // Create physics walls from Perlin noise
    Wall wallInner(vertexCount);
    Wall wallOuter(vertexCount);
    GenerateSimplexNoisePath(param::iSEED, vertexCount, wallInner.displayVertices, wallOuter.displayVertices, wallInner.physicsVertices, wallOuter.physicsVertices);
    wallInner.CreateBody(world);
    wallOuter.CreateBody(world);

    // Init ray sensing and display
    sf::VertexArray rays = sf::VertexArray{ sf::PrimitiveType::Lines, param::iRayCount * 2 };
    std::vector<b2RayResult> rayResults(param::iRayCount);

    /*!
     *   Create many randomized agents to interact with the environment
     */
    int agentCount = 100;
    float halfWidth = 0.3f; // meters
    float halfHeight = 0.15f; // meters
    std::vector<Vehicle> vehicles(agentCount);
    std::vector<int> architecture = { 7, 5, 4 };
    CreateVehicles(architecture, vehicles, world, halfWidth, halfHeight, dist, gen);

    // Get arena bounds
    float xMinArena = INFINITY, xMaxArena = -INFINITY;
    float yMinArena = INFINITY, yMaxArena = -INFINITY;
    for (auto& vertex : wallOuter.physicsVertices)
    {
        if (xMinArena > vertex.x)
            xMinArena = vertex.x;
        if (xMaxArena < vertex.x)
            xMaxArena = vertex.x;
        if (yMinArena > vertex.y)
            yMinArena = vertex.y;
        if (yMaxArena < vertex.y)
            yMaxArena = vertex.y;
    }

    int frameCounter = 0;
    int generationFrameLimit = 10000;
    float x = -265.f / 50.f;
    float y = 0.f;
    float rotation = -3.1415f / 2.f;
    std::vector<float> normalizedScores(agentCount);
    float bestScore = 0.f;
    float lastGenBestScore = 0.f;
    while (m_window.isOpen())
    {

        // Mutate vehicles and restart simulation
        if (frameCounter % generationFrameLimit == 0)
        {
            // Sort vehicles based on score
            std::sort(vehicles.begin(), vehicles.end(), SortByScore);

            // Normalize scores to use as random selection threshold
            // FIXME This causes them to gravitate to local minimums
            float minScore = vehicles[agentCount-1].GetScore();
            float maxScore = vehicles[0].GetScore();
            lastGenBestScore = maxScore;
            bestScore = maxScore > bestScore ? maxScore : bestScore;
            for (size_t i = 0; i < normalizedScores.size(); i++)
            {
                normalizedScores[i] = (vehicles[i].GetScore() - minScore) / (maxScore - minScore);
            }

            // Natural selection
            size_t vehiclesHalfSize = vehicles.size() / 2;
            for (int i = 0; i < vehicles.size(); i++)
            {
                // Rebirth vehicle network for lower half of vehicles
                bool shouldEvolve = dist(gen) + 1.f / 2.f > 1.f - normalizedScores[i];
                if (shouldEvolve) {
                    // Randomize seed for network brain
                    float newSeed = dist(gen) + i;
                    gen.seed(newSeed);

                    // Evolving from top 25% vehicle
                    int betterVehIdx = vehicles.size() * 0.25f * ((dist(gen) + 1.f) / 2.f);
                    vehicles[i].Evolve(vehicles[betterVehIdx].m_brain, 0.5f, dist, gen);
                    vehicles[i].MutateBrain(0.5f, dist, gen);
                }

                // Reset vehicles bodies
                vehicles[i].ResetBody(x, y, rotation);

                // Random mutation to prevent local minimum
                if ((dist(gen) + 1.f) / 2.f < 0.5f)
                {
                    // Mutate two times randomly (FIXME work around for bad mutation function) 
                    vehicles[i].MutateBrain(0.5f, dist, gen);
                    //vehicles[i].MutateBrain(0.5f, dist, gen);
                }
            }
        }

        // Update
        m_deltaTime = m_clock.restart();
        ImGui::SFML::Update(m_window.get(), m_deltaTime);
        b2World_Step(world, m_deltaTime.asSeconds(), 4);

        // Handle vehicle-wall collisions (box2d)
        b2ContactEvents collisions = b2World_GetContactEvents(world);
        for (size_t i = 0; i < collisions.beginCount; i++)
        {
            const b2ContactBeginTouchEvent* beginEvent = collisions.beginEvents + i;
            
            // Get colliding bodies, one is the vehicle, one is the wall
            // Shape A is the vehicle
            if (b2Shape_GetFilter(beginEvent->shapeIdA).categoryBits == C_VEHICLE)
            {
                // Get vehicle
                auto body = b2Shape_GetBody(beginEvent->shapeIdA);
                Vehicle* vehicle = static_cast<Vehicle*>(b2Body_GetUserData(body));
                vehicle->IncrementWallCollisions();
            }
            else // Shape B
            {
                // Get vehicle
                auto body = b2Shape_GetBody(beginEvent->shapeIdB);
                Vehicle* vehicle = static_cast<Vehicle*>(b2Body_GetUserData(body));
                vehicle->IncrementWallCollisions();
            }
        }

        // Agent-environment interaction
        for (size_t i = 0; i < vehicles.size(); i++)
        {
            // Agent senses it's environment
            std::vector<float>& inputs = vehicles[i].Sense(world, param::fFieldOfView, param::iRayCount, xMinArena, xMaxArena, yMinArena, yMaxArena);

            // Agent acts on environment
            vehicles[i].Act(inputs);

            // Vehicle is scored based on interactions with environment
            vehicles[i].UpdateScore(param::collisionPenalizer);
        }

        // Debug windows
        ImGui::ShowMetricsWindow();
        ShowSimplexNoisePathWindow(param::iSEED, vertexCount, wallInner, wallOuter);
        ShowBestVehicleStatsWindow(vehicles[0]);
        ShowVehicleScores(frameCounter / generationFrameLimit, bestScore, lastGenBestScore, vehicles);
        
        m_window.ProcessEvents(view); // Also processes ImGui events

        // Display
        m_window.BeginDraw();
        m_window.Draw(wallInner.displayVertices);
        m_window.Draw(wallOuter.displayVertices);
        //m_window.Draw(rays);
        for (size_t i = 0; i < vehicles.size(); i++)
        {
            DrawVehicle(m_window.get(), vehicles[i].m_body, halfWidth, halfHeight);
        }

        //m_window.Draw(pixelSprite);
        DrawNetwork(vehicles[0].m_brain, m_window, { view.getCenter().x - param::iWIDTH / 2 + 60.f, view.getCenter().y + param::iHEIGHT / 2 - 60.f }, 7.f);

        ImGui::SFML::Render(m_window.get());
        m_window.EndDraw();

        frameCounter++;
    }

    ImGui::SFML::Shutdown();
}