#include "Vehicle.hpp"

#include "Network.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>
#include <iostream>

#include <box2d/box2d.h>
#include <box2d/types.h>
#include <box2d/collision.h>
#include <box2d/id.h>
#include <box2d/math_functions.h>

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>

Vehicle::Vehicle() :
    m_score(0.f), m_wallCollisions(0) {}

Vehicle::~Vehicle() = default;

void Vehicle::InitBody(b2WorldId world, float halfWidth, float halfHeight, float x, float y, float rotation)
{
    // Define physics parameters
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    // FIXME 50.f = param::fBOX2D_SCALE but parameters.hpp linker error
    bodyDef.position = { x, y };
    bodyDef.linearVelocity = { 0.f, 0.f };
    bodyDef.linearDamping = 0.9f;
    bodyDef.angularDamping = 0.9f;
    bodyDef.rotation = b2MakeRot(rotation);
    bodyDef.enableSleep = false;
    bodyDef.userData = this;

    m_body = b2CreateBody(world, &bodyDef);

    // Make body shape
    b2Polygon box = b2MakeBox(halfWidth, halfHeight);

    // Define collider
    b2ShapeDef shapeDef = b2DefaultShapeDef();;
    // FIXME 2 = param::Vehicle but parameters.hpp linker error
    shapeDef.filter.categoryBits = 2;;
    // FIXME 1 = param::Wall but parameters.hpp linker error
    shapeDef.filter.maskBits = 1;
    shapeDef.enableContactEvents = true;

    b2CreatePolygonShape(m_body, &shapeDef, &box);
}

void Vehicle::InitBrain(
    std::vector<int>& architecture,
    const std::function<float(float)>& ActivationSigmoid,
    const std::function<float(float)>& DerivativeActivationSigmoid,
    const std::function<float(size_t, float, float)>& LossMSE,
    std::uniform_real_distribution<float>& dist, std::mt19937& gen
)
{
    m_brain = Network(architecture, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE,
        std::make_unique<std::uniform_real_distribution<float>>(dist),
        std::make_unique<std::mt19937>(gen)
    );

    m_inputs.resize(architecture.front());
    m_outputs.resize(architecture.back());
}

std::vector<float>& Vehicle::Sense(b2WorldId world, float fov, size_t rayCount, float xMin, float xMax, float yMin, float yMax)
{ 
    // Body parameters
    const b2Vec2& position = b2Body_GetPosition(m_body);
    const b2Rot& rotation = b2Body_GetRotation(m_body);
    float rotationAsRadians = b2Rot_GetAngle(rotation);

    // Ray parameters
    float rayLengthMeters = 3.f;
    float angleStep = fov / (rayCount - 1);
    // beginning of field of view is half FOV to left of rotation
    float angleStart = rotationAsRadians - fov / 2.f; // FIXME param::fFieldOfView / 2.f
    b2QueryFilter rayFilter = { 2, 1 }; // FIXME {param::Vehicle, param::Wall} in main
    b2Vec2 translation = { 0.f, 0.f }; // Initialize translation

    // Shoot rays from vehicle
    std::vector<b2RayResult> rayResults(rayCount);
    for (size_t i = 0; i < rayCount; i++)
    {
        // Box2d rot for cos and sin parts
        b2Rot rayRot = b2MakeRot(angleStart + angleStep * i);

        // Translation vector of ray
        translation = { rayRot.c * rayLengthMeters, rayRot.s * rayLengthMeters };

        // Cast and draw
        rayResults[i] = b2World_CastRayClosest(world, position, translation, rayFilter);
        //m_rays[i * 2] = sf::Vertex{ { position.x * 50.f, position.y * 50.f }, sf::Color::Yellow };
        //m_rays[i * 2 + 1] = sf::Vertex{ { (position.x + translation.x) * 50.f, (position.y + translation.y) * 50.f }, sf::Color::Yellow };
    }

    // Inputs
    // Normalize position inputs
    m_inputs[0] = (position.x - xMin) / (xMax - xMin);
    m_inputs[1] = (position.y - yMin) / (yMax - yMin);

    // Ray inputs
    m_inputs[2] = rayResults[0].fraction;
    m_inputs[3] = rayResults[1].fraction;
    m_inputs[4] = rayResults[2].fraction;

    // Change in position input
    b2Vec2 currentPosition = b2Body_GetPosition(m_body);
    float dx = currentPosition.x - m_previousPosition.x;
    float dy = currentPosition.y - m_previousPosition.y;
    m_inputs[5] = dx;
    m_inputs[6] = dy;

    return m_inputs;
}

void Vehicle::Act(std::vector<float>& inputs, float halfWidth, float halfHeight)
{
    // TODO set m_outputs to named variables
    // Agent moves the vehicle
    // Network outputs are normalized from [0, 1]
    m_outputs = m_brain.Predict(inputs);

    // Transform outputs to represent all possible force vectors
    //  ouputs in range from [-0.5, 0.5]
    m_outputs[0] -= 0.5f;
    m_outputs[1] -= 0.5f;
    b2Vec2 force = b2Normalize({ m_outputs[0], m_outputs[1] });

    // Move agent with force applied to position on body
    // Transform outputs to represent all possible acceleration vectors
    //  ouputs in range from [-1, 1]
    m_outputs[2] *= 2.f - 1.f;
    m_outputs[2] *= halfWidth;
    m_outputs[3] *= 2.f - 1.f;
    m_outputs[3] *= halfHeight;
    b2Vec2 forcePosition = b2Body_GetPosition(m_body);
    forcePosition.x += m_outputs[2];
    forcePosition.y += m_outputs[3];

    b2Body_ApplyForce(m_body, force, forcePosition, true);
}

void Vehicle::Evolve(Network betterBrain, float mutationFactor, std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{
    // Copy better brain from more succesful vehicle
    m_brain = betterBrain;
    m_inputs.resize(m_brain.getArchitecture().front());
    m_outputs.resize(m_brain.getArchitecture().back());

    // Evolve offspring
    MutateBrain(mutationFactor, dist, gen);
}

void Vehicle::Crossover(Vehicle& parent1, Vehicle& parent2, std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{
    std::vector<std::vector<std::vector<float>>> newWeights;
    std::vector<std::vector<float>> newBiases;

    // Combine two parent networks to create a child network. This is only implemented for equal size networks.
    Network& p1Brain = parent1.m_brain;
    Network& p2Brain = parent2.m_brain;

    const std::vector<Layer>& p1Layers = p1Brain.getLayers();
    const std::vector<Layer>& p2Layers = p2Brain.getLayers();

    newWeights.resize(p1Layers.size());
    newBiases.resize(p1Layers.size());

    for (size_t layer = 0; layer < p1Layers.size(); layer++)
    {
        const std::vector<Neuron>& p1LayerNeurons = p1Layers[layer].getNeurons();
        const std::vector<Neuron>& p2LayerNeurons = p2Layers[layer].getNeurons();

        newWeights[layer].resize(p1LayerNeurons.size());
        newBiases[layer].resize(p1LayerNeurons.size());

        // Randomly select neurons for child
        for (size_t neuron = 0; neuron < p1LayerNeurons.size(); neuron++)
        {
            newWeights[layer][neuron].resize(p1LayerNeurons[neuron].getWeights().size());

            if ((dist(gen) + 1.f) / 2.f < 0.5f)
            {
                // Parent 1 neuron selected
                // new bias
                newBiases[layer][neuron] = p1LayerNeurons[neuron].getBias();
                
                // new weights
                const std::vector<float>& p1NeuronWeights = p1LayerNeurons[neuron].getWeights();
                for (size_t w = 0; w < p1LayerNeurons[neuron].getWeights().size(); w++)
                {
                    newWeights[layer][neuron][w] = p1NeuronWeights[w];
                }
            }
            else
            {
                // Parent 2 neuron selected
                // new bias
                newBiases[layer][neuron] = p2LayerNeurons[neuron].getBias();

                // new weights
                const std::vector<float>& p2NeuronWeights = p2LayerNeurons[neuron].getWeights();
                for (size_t w = 0; w < p2LayerNeurons[neuron].getWeights().size(); w++)
                {
                    newWeights[layer][neuron][w] = p2NeuronWeights[w];
                }
            }
        }
    }

    m_brain.setWeightsAndBiases(newWeights, newBiases);
}

void Vehicle::ResetBody(float x, float y, float rotation)
{
    b2Body_SetTransform(m_body, { x, y }, b2MakeRot(rotation));
    b2Body_SetLinearVelocity(m_body, { 0.f, 0.f });
    b2Body_SetAngularVelocity(m_body, 0.f);
    InitializeScoring();
}

void Vehicle::MutateBrain(float mutationFactor, std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{
    std::vector<std::vector<std::vector<float>>> weights = m_brain.copyWeights();
    std::vector<std::vector<float>> biases = m_brain.copyBiases();

    float mutationChance = 0.3f; // Probability to mutate random weight or bias
    float largeChangeChance = 0.05f; // Probability to make a large mutation to escape local minima

    for (size_t layer = 0; layer < weights.size(); layer++)
    {
        // Chance to mutate any weights
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++)
        {
            for (size_t w = 0; w < weights[layer][neuron].size(); w++)
            {
                if (dist(gen) < mutationChance * 2.f - 1.f) // dist is [-1, 1]
                {
                    // Possibility for large change to avoid local minima
                    float changeSize = (dist(gen) < largeChangeChance * 2.f - 1.f) ? 
                        5.0f * mutationFactor : 
                        mutationFactor;

                    // Mutation * [-1, 1]
                    float mutation = dist(gen) * changeSize;
                    weights[layer][neuron][w] += mutation;
                }
            }
        }

        // Chance to mutate any biases
        for (size_t b = 0; b < biases[layer].size(); b++)
        {
            if (dist(gen) < mutationChance * 2.f - 1.f)
            {
                // Possibility for large change to avoid local minima
                float changeSize = (dist(gen) < largeChangeChance * 2.f - 1.f) ?
                    5.0f * mutationFactor :
                    mutationFactor;

                // Mutation * [-1, 1]
                float mutation = dist(gen) * changeSize;
                biases[layer][b] += mutation;
            }
        }
    }

    // Update brain
    m_brain.setWeightsAndBiases(weights, biases);
}

void Vehicle::ScrambleBrain(std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{
    std::vector<std::vector<std::vector<float>>> newWeights;
    std::vector<std::vector<float>> newBiases;

    const std::vector<Layer>& layers = m_brain.getLayers();

    newWeights.resize(layers.size());
    newBiases.resize(layers.size());

    for (size_t layer = 0; layer < layers.size(); layer++)
    {
        const std::vector<Neuron>& neurons = layers[layer].getNeurons();

        newWeights[layer].resize(neurons.size());
        newBiases[layer].resize(neurons.size());

        for (size_t neuron = 0; neuron < neurons.size(); neuron++)
        {
            std::vector<float> weights = neurons[neuron].getWeights();

            newWeights[layer][neuron].resize(weights.size());
            for (size_t weight = 0; weight < weights.size(); weight++)
            {
                weights[weight] = dist(gen);
            }

            newWeights[layer][neuron] = weights;
            newBiases[layer][neuron] = 0.f;
        }
    }
}

void Vehicle::InitializeScoring()
{
    m_wallCollisions = 0;
    m_totalAngleTraversed = 0.f;
    m_score = 0.f;

    // Calculate starting angle
    // Since track is centered around (0, 0), position is the angle offset for the circle
    //  otherwise this would need to be translated to circle's center
    b2Vec2 position = b2Body_GetPosition(m_body);
    m_previousPositionalAngle = atan2f(position.x, position.y);
}

void Vehicle::IncrementWallCollisions() { m_wallCollisions++; }

void Vehicle::UpdateScore(float collisionPenalizer, float distanceMultiplier)
{
    // Since track is centered around (0, 0), position is the angle offset for the circle
    //  otherwise this would need to be translated to circle's center
    b2Vec2 position = b2Body_GetPosition(m_body);
    b2Vec2 velocity = b2Body_GetLinearVelocity(m_body);
    float speed = b2Length(velocity);
    
    // Check if at a standstill and penalize
    float movementSinceLastUpdate = b2Length(position - m_previousPosition);
    float speedScore = 0.f;
    if (movementSinceLastUpdate < 0.05f)
    {
        m_standstillCount++;
        speedScore = -0.1f * m_standstillCount;
    }
    else
    {
        m_standstillCount = 0.f;
        speedScore = speed * 0.2f;
    }

    float currentPositionalAngle = atan2f(position.x, position.y);

    // Calculate angle difference, map to [-pi, pi]
    float angleDiff = currentPositionalAngle - m_previousPositionalAngle;
    if (angleDiff > B2_PI) angleDiff -= 2 * B2_PI;
    if (angleDiff < -B2_PI) angleDiff += 2 * B2_PI;

    // Negate angle diff for clockwise track (angle is decreaseing)
    m_totalAngleTraversed -= angleDiff;

    // Less good than using a center line
    // Calculate alignment with track
    //  position is track radial vector with track origin at (0, 0)
    b2Vec2 tangentialNormal = b2Normalize(b2RightPerp(position));
    b2Rot rotation = b2Body_GetRotation(m_body);
    b2Vec2 rotationVector = { rotation.c, rotation.s };
    float tangentialOffset = b2Dot(tangentialNormal, rotationVector);

    // Reward vehicles moving in the direction they are facing
    b2Vec2 normalizedVelocity = velocity;
    if (b2Length(velocity) > 0.1f)
    {
        normalizedVelocity = b2Normalize(velocity);
    }
    float movementOffset = b2Dot(normalizedVelocity, rotationVector);

    // Penalize wall collisions
    float collisionPenalties = m_wallCollisions * collisionPenalizer;

    // Update score
    m_score = (m_totalAngleTraversed * distanceMultiplier) - collisionPenalties + 
        speedScore + (tangentialOffset * 1.f) + (movementOffset * 1.f);

    // Additional progressive reward for half-laps
    if (m_totalAngleTraversed > B2_PI)
    {
        m_lapsCompleted++;
        m_score += m_lapsCompleted * 50.f;
    }

    // Update previous vars
    m_previousPosition = position;
    m_previousPositionalAngle = currentPositionalAngle;
}

float Vehicle::GetScore() const { return m_score; }

void Vehicle::Draw(sf::RenderWindow& window, sf::Color outlineColor, b2BodyId vehicleBody, float halfWidth, float halfHeight, float box2dScale)
{
    const b2Vec2& position = b2Body_GetPosition(vehicleBody);
    const b2Rot& rotation = b2Body_GetRotation(vehicleBody);

    // Display vehicle
    sf::RectangleShape vehicle({ halfWidth * 2.f * box2dScale, halfHeight * 2.f * box2dScale });
    vehicle.setOrigin({ halfWidth * box2dScale, halfHeight * box2dScale });
    vehicle.setFillColor(sf::Color::Transparent);
    vehicle.setOutlineColor(outlineColor);
    vehicle.setOutlineThickness(1.f);
    vehicle.setPosition({ position.x * box2dScale, position.y * box2dScale });
    vehicle.setRotation(sf::radians(b2Rot_GetAngle(rotation)));
    window.draw(vehicle);

    // display rotation
    float radius = 25.f;
    sf::VertexArray rotationLine(sf::PrimitiveType::Lines, 2);
    b2Vec2 pos = b2Body_GetPosition(vehicleBody);
    float screenX = pos.x * box2dScale;
    float screenY = pos.y * box2dScale;
    rotationLine[0] = sf::Vertex{ { screenX, screenY } };
    rotationLine[1] = sf::Vertex{ { rotation.c * radius + screenX, rotation.s * radius + screenY } };
    window.draw(rotationLine);
}
