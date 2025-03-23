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
    m_score(0.f), m_wallCollisions(0), m_lapsCompleted(0)
{
    m_rays.setPrimitiveType(sf::PrimitiveType::LineStrip);
    m_rays.resize(10);
}

Vehicle::Vehicle(size_t rayCount) :
    m_score(0.f), m_wallCollisions(0), m_lapsCompleted(0) 
{
    m_rays.setPrimitiveType(sf::PrimitiveType::LineStrip);
    m_rays.resize(rayCount*2);
}

Vehicle::~Vehicle() = default;

void Vehicle::InitBody(b2WorldId world, float halfWidth, float halfHeight, float x, float y, float rotation)
{
    // Define physics parameters
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    // FIXME 50.f = param::fBOX2D_SCALE but parameters.hpp linker error
    bodyDef.position = { x, y };
    bodyDef.linearVelocity = { 0.f, 0.f };
    bodyDef.linearDamping = 0.0f; // Handled by simulation
    bodyDef.angularDamping = 0.8f;
    bodyDef.rotation = b2MakeRot(rotation);
    bodyDef.enableSleep = false;  
    bodyDef.fixedRotation = false;
    bodyDef.userData = this;

    m_body = b2CreateBody(world, &bodyDef);

    // Make body shape
    b2Polygon box = b2MakeBox(halfWidth, halfHeight);

    // Define collider
    b2ShapeDef shapeDef = b2DefaultShapeDef();;
    shapeDef.filter.categoryBits = 2;
    shapeDef.filter.maskBits = 1;
    shapeDef.enableContactEvents = true;
    shapeDef.density = 30.f;
    shapeDef.friction = 0.2f;
    shapeDef.restitution = 0.1f;

    b2CreatePolygonShape(m_body, &shapeDef, &box);

    m_previousPosition = { x, y };
    m_previousPositionalAngle = atan2f(x, y);
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

std::vector<float>& Vehicle::Sense(b2WorldId world, float fov, size_t rayCount, float xMin, float xMax, float yMin, float yMax, float b2Scale)
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
        m_rays[i * 2] = sf::Vertex{ { position.x * b2Scale, position.y * b2Scale }, sf::Color::Yellow };
        m_rays[i * 2 + 1] = sf::Vertex{ { (position.x + translation.x) * b2Scale, (position.y + translation.y) * b2Scale }, sf::Color::Yellow };
    }

    // Inputs
    // Normalize position inputs
    m_inputs[0] = (position.x - xMin) / (xMax - xMin);
    m_inputs[1] = (position.y - yMin) / (yMax - yMin);

    // Change in position input
    b2Vec2 currentPosition = b2Body_GetPosition(m_body);
    float dx = currentPosition.x - m_previousPosition.x;
    float dy = currentPosition.y - m_previousPosition.y;
    m_inputs[2] = dx;
    m_inputs[3] = dy;

    // Ray inputs
    for (size_t i = 0; i < rayCount; i++)
    {
        m_inputs[i + 4] = rayResults[i].fraction;
    }

    return m_inputs;
}

void Vehicle::Act(std::vector<float>& inputs, float halfWidth, float halfHeight)
{
    // TODO set m_outputs to named variables
    // Agent moves the vehicle
    // Network outputs are normalized from [0, 1]
    m_outputs = m_brain.Predict(inputs);

    // Current vehicle stats
    b2Vec2 position = b2Body_GetPosition(m_body);
    b2Rot rotation = b2Body_GetRotation(m_body);
    b2Vec2 currentVelocity = b2Body_GetLinearVelocity(m_body);
    float currentAngularVelocity = b2Body_GetAngularVelocity(m_body);

    // ====== CAR CONTROLS ======
    float throttle = m_outputs[0] * 2.f - 1.f;
    float steering = m_outputs[1] * 2.f - 1.f;

    // ====== PHYSICS PARAMETERS ======
    // Adjusted for smaller vehicle size
    float maxEngineForce = 10.f;
    float maxBrakingForce = 15.f;
    float maxSteeringTorque = 0.3f;
    float steeringDampening = 0.9f;

    float lateralFrictionCoeff = 3.f;
    float rollingResistance = 0.2f;

    b2Vec2 forwardDirection = { rotation.c, rotation.s };

    // ======= APPLY ENGINE/BRAKING FORCE =======
    float engineForce = 0.f;
    if (throttle > 0.f) // Accelerate
        engineForce = throttle * maxEngineForce;
    else                // Brake
        engineForce = throttle * maxBrakingForce;

    // Apply force along forward vector
    b2Vec2 propulsion = {
        forwardDirection.x * engineForce,
        forwardDirection.y * engineForce
    };
    b2Body_ApplyForceToCenter(m_body, propulsion, true);

    // ======= STEERING MECHANICS =======
    float steeringTorque = steering * maxSteeringTorque;

    // Steering effectiveness decreases at higher speed
    float currentSpeed = b2Dot(currentVelocity, forwardDirection);
    float steeringFactor = 1.f / (1.f + 0.1f * fabs(currentSpeed));
    steeringTorque *= steeringFactor;

    // Apply angular impulse for turning (dampening prevents oscillation)
    float angularImpulse = steeringTorque - (currentAngularVelocity * steeringDampening);
    b2Body_ApplyAngularImpulse(m_body, angularImpulse, true);

    // ====== LATERAL FRICTION ====== (sideways velocity component)
    b2Vec2 rightDirection = { -forwardDirection.y, forwardDirection.x };
    float lateralVelocity = b2Dot(currentVelocity, rightDirection);

    // Apply lateral friction to prevent easy sliding
    b2Vec2 lateralFrictionForce = {
        -rightDirection.x * lateralVelocity * lateralFrictionCoeff,
        -rightDirection.y * lateralVelocity * lateralFrictionCoeff
    };
    b2Body_ApplyForceToCenter(m_body, lateralFrictionForce, true);

    // ====== ROLLING DISTANCE ======
    b2Vec2 rollingResistanceForce = {
        -currentVelocity.x * rollingResistance,
        -currentVelocity.y * rollingResistance
    };
    b2Body_ApplyForceToCenter(m_body, rollingResistanceForce, true);

    // Very efficient but poor simulation results
    //// Transform outputs to represent all possible force vectors
    ////  ouputs in range from [-0.5, 0.5]
    //m_outputs[0] -= 0.5f;
    //m_outputs[1] -= 0.5f;
    //b2Vec2 force = b2Normalize({ m_outputs[0], m_outputs[1] });
    //
    //// Move agent with force applied to position on body
    //// Transform outputs to represent all possible acceleration vectors
    ////  ouputs in range from [-1, 1]
    //m_outputs[2] *= 2.f - 1.f;
    //m_outputs[2] *= halfWidth;
    //m_outputs[3] *= 2.f - 1.f;
    //m_outputs[3] *= halfHeight;
    //b2Vec2 forcePosition = b2Body_GetPosition(m_body);
    //forcePosition.x += m_outputs[2];
    //forcePosition.y += m_outputs[3];
    //
    //b2Body_ApplyForce(m_body, force, forcePosition, true);
}

void Vehicle::Crossover(Vehicle& parent1, Vehicle& parent2, float parent1Score, float parent2Score,
    std::uniform_real_distribution<float>& dist, std::mt19937& gen)
{
    // Crossover type randomizer
    float r = (dist(gen) + 1.f) / 2.f;

    // Combine two parent networks to create a child network. This is only implemented for equal size networks.
    const Network& p1Brain = parent1.m_brain;
    const Network& p2Brain = parent2.m_brain;

    // Calulate parent bias based on fitness scores
    float totalScore = parent1Score + parent2Score;
    float p1Bias = (totalScore > 0) ? (parent1Score / totalScore) : 0.5f;

    // Child starts as base copy of parent 1
    std::vector<std::vector<std::vector<float>>> newWeights = p1Brain.copyWeights();
    std::vector<std::vector<float>> newBiases = p1Brain.copyBiases();

    std::vector<std::vector<std::vector<float>>> p2Weights = p2Brain.copyWeights();
    std::vector<std::vector<float>> p2Biases = p2Brain.copyBiases();

    // Use differing crossover strategies
    if (r < 0.4f)
    {
        // Strategy 1: fitness weighted neuron mixing
        for (size_t layer = 0; layer < newWeights.size(); layer++)
        {
            for (size_t neuron = 0; neuron < newWeights[layer].size(); neuron++)
            {
                // Decide if we take neuron from parent 1 or 2
                if ((dist(gen) + 1.f) / 2.f > p1Bias)
                {
                    // Take parent 2 neuron
                    newBiases[layer][neuron] = p2Biases[layer][neuron];
                    for (size_t w = 0; w < newWeights[layer][neuron].size(); w++)
                    {
                        newWeights[layer][neuron][w] = p2Weights[layer][neuron][w];
                    }
                }
            }
        }
    }
    else if (r < 0.7f)
    {
        // Strategy 2: layer crossover
        for (size_t layer = 0; layer < newWeights.size(); layer++)
        {
            // Decide if we take layer from parent 1 or 2
            if ((dist(gen) + 1.f) / 2.f > p1Bias)
            {
                for (size_t neuron = 0; neuron < newWeights[layer].size(); neuron++)
                {
                    // Take parent 1 neuron
                    newBiases[layer][neuron] = p2Biases[layer][neuron];
                    for (size_t w = 0; w < newWeights[layer][neuron].size(); w++)
                    {
                        newWeights[layer][neuron][w] = p2Weights[layer][neuron][w];
                    }
                }
            }
        }
    }
    else if (r < 0.9f)
    {
        // Strategy 3: weight level blending
        // Random parent blend
        float blendRatio = ((dist(gen)) + 1.f) / 2.f;

        for (size_t layer = 0; layer < newWeights.size(); layer++)
        {
            for (size_t neuron = 0; neuron < newWeights[layer].size(); neuron++)
            {
                // Bias blend
                newBiases[layer][neuron] = newBiases[layer][neuron] * blendRatio + 
                    p2Biases[layer][neuron] * (1.f - blendRatio);

                for (size_t w = 0; w < newWeights[layer][neuron].size(); w++)
                {
                    // Weight blend
                    newWeights[layer][neuron][w] = newWeights[layer][neuron][w] * blendRatio +
                        p2Weights[layer][neuron][w] * (1.f - blendRatio);
                }
            }
        }
    }
    else if(parent1Score < parent2Score) 
    {
        // Strategy 4: otherwise keep better parent's network
        newWeights = p2Weights;
        newBiases = p2Biases;
    }

    m_brain.setWeightsAndBiases(newWeights, newBiases);
}

void Vehicle::ResetBody(float x, float y, float rotation)
{
    b2Body_SetTransform(m_body, { x, y }, b2MakeRot(rotation));
    b2Body_SetLinearVelocity(m_body, { 0.f, 0.f });
    b2Body_SetAngularVelocity(m_body, 0.f);
    m_previousPosition = { x, y };
    m_previousPositionalAngle = atan2f(x, y);
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

void Vehicle::ZeroScoring()
{
    m_wallCollisions = 0;
    m_totalAngleTraversed = 0.f;
    m_lapsCompleted = 0.f;
    m_standstillCount = 0;
    m_score = 0.f;
}

void Vehicle::IncrementWallCollisions() { m_wallCollisions++; }

void Vehicle::UpdateScore(float collisionPenalizer, float distanceMultiplier, float generationTimer)
{
    // Since track is centered around (0, 0), position is the angle offset for the circle
    //  otherwise this would need to be translated to circle's center
    b2Vec2 position = b2Body_GetPosition(m_body);
    b2Vec2 velocity = b2Body_GetLinearVelocity(m_body);
    float speed = b2Length(velocity);

    // ======= STANDSTILL PENALTY =======
    // Check if at a standstill and penalize
    float movementSinceLastUpdate = b2Length(position - m_previousPosition);
    float speedScore = 0.f;
    if (movementSinceLastUpdate < 0.05f)
    {
        m_standstillCount++;
        speedScore = -0.5f * m_standstillCount / generationTimer;
    }
    else
    {
        m_standstillCount = 0.f;
        speedScore = speed * 0.3f;
    }


    // ======= ANGLE TRAVERSAL (circular track progression) =======
    float currentPositionalAngle = atan2f(position.x, position.y);
    // Calculate angle difference, map to [-pi, pi]
    float angleDiff = currentPositionalAngle - m_previousPositionalAngle;
    if (angleDiff > B2_PI) angleDiff -= 2 * B2_PI;
    if (angleDiff < -B2_PI) angleDiff += 2 * B2_PI;

    // Negate angle diff for clockwise track (angle is decreaseing)
    m_totalAngleTraversed -= angleDiff;

    // ======= TRACK ALIGNMENT REWARD =======
    // Less good than using a center line
    // Calculate alignment with track
    //  position is track radial vector with track origin at (0, 0)
    b2Vec2 tangentialNormal = b2Normalize(b2RightPerp(position));
    b2Rot rotation = b2Body_GetRotation(m_body);
    b2Vec2 rotationVector = { rotation.c, rotation.s };
    float tangentialOffset = b2Dot(tangentialNormal, rotationVector);

    // Use sigmoid-like function to reward alignment without discontinuities
    float tangentialScore = 2.0f / (1.0f + exp(-3.0f * tangentialOffset)) - 1.0f;

    // ======= MOVEMENT DIRECTION ALIGNMENT =======
    // Reward vehicles moving in the direction they are facing
    b2Vec2 normalizedVelocity = velocity;
    if (speed > 0.1f)
    {
        normalizedVelocity = b2Normalize(velocity);
    }
    float movementOffset = b2Dot(normalizedVelocity, rotationVector);
    // Penalize directions not heavilty aligned with facing direction
    if (movementOffset < 0.7f) 
    {
        // Normalize values below 0.5f to [0, -1.5]
        movementOffset = movementOffset - 0.7f;
    }

    // ======= WALL COLLISION PENALTY =======
    float collisionPenalties = m_wallCollisions * collisionPenalizer;

    // Update score
    m_score = 
        m_totalAngleTraversed * distanceMultiplier - 
        collisionPenalties + 
        speedScore + 
        tangentialScore * 6.f +
        movementOffset * 6.f;

    // ======= LAP COUNTING =======
    // Additional progressive reward for half-laps
    if (m_totalAngleTraversed >= (m_lapsCompleted + 0.5f) * 2.0f * B2_PI)
    {
        m_lapsCompleted += 0.5f;
        m_score += 50.f;
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
