#include "Vehicle.hpp"

#include "Network.hpp"
#include <box2d/box2d.h>
#include <box2d/types.h>
#include <box2d/collision.h>
#include <box2d/id.h>
#include <box2d/math_functions.h>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

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
    m_previousPosition = currentPosition;

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

    //float max = dist.max();
    //float min = dist.min();
    //
    //// random layer's biases altered
    //int randomLayer = (dist(gen) - min) / (max - min) * m_brain.getLayers().size() - 1;
    //std::vector<std::vector<float>> biases = m_brain.copyBiases();
    //for (auto& bias : biases[randomLayer])
    //{
    //    // Normalize distribution generator to [-1, 1]
    //    float randomBiasChange = dist(gen);
    //    bias += randomBiasChange * mutationFactor;
    //}

    //// random neurons weights altered
    //randomLayer = (dist(gen) - min) / (max - min) * m_brain.getLayers().size() - 1;
    //std::vector<std::vector<std::vector<float>>> weights = m_brain.copyWeights();
    //std::vector<std::vector<float>>& layer = weights[randomLayer];
    //float randomNeuron = (dist(gen) - min) / (max - min) * m_brain.getLayers().size() - 1;
    //for (auto& weight : layer[randomNeuron])
    //{
    //    // Normalize distribution generator [-1, 1]
    //    float randomWeightChange = dist(gen);
    //    weight += dist(gen) * mutationFactor;
    //}

    //// Update brain
    //m_brain.setWeightsAndBiases(weights, biases);
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
    m_previousAngle = atan2f(position.x, position.y);
}

void Vehicle::IncrementWallCollisions() { m_wallCollisions++; }

void Vehicle::UpdateScore(float collisionPenalizer, float distanceMultiplier)
{
    // Since track is centered around (0, 0), position is the angle offset for the circle
    //  otherwise this would need to be translated to circle's center
    b2Vec2 position = b2Body_GetPosition(m_body);
    float speed = b2LengthSquared(b2Body_GetLinearVelocity(m_body));
    float currentAngle = atan2f(position.x, position.y);

    // Calculate angle difference, map to [-pi, pi]
    float angleDiff = currentAngle - m_previousAngle;
    if (angleDiff > B2_PI) angleDiff -= 2 * B2_PI;
    if (angleDiff < -B2_PI) angleDiff += 2 * B2_PI;

    // Negate angle diff for clockwise track (angle is decreaseing)
    m_totalAngleTraversed -= angleDiff;
    m_previousAngle = currentAngle;

    float collisionPenalties = m_wallCollisions * collisionPenalizer;
    m_score = m_totalAngleTraversed * distanceMultiplier - collisionPenalties + speed * 0.1f; // FIXME 3.f angle traversed mutliplier
}

float Vehicle::GetScore() const { return m_score; }
