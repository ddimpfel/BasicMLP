#include "Vehicle.hpp"
#include <box2d/box2d.h>
#include <box2d/types.h>
#include <cmath>

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
        //outDrawableRays[i * 2] = sf::Vertex{ { origin.x * param::fBOX2D_SCALE, origin.y * param::fBOX2D_SCALE }, sf::Color::Yellow };
        //outDrawableRays[i * 2 + 1] = sf::Vertex{ { (origin.x + translation.x) * param::fBOX2D_SCALE, (origin.y + translation.y) * param::fBOX2D_SCALE }, sf::Color::Yellow };
    }

    // Inputs
    // Normalize position inputs
    m_inputs[0] = (position.x - xMin) / (xMax - xMin);
    m_inputs[1] = (position.y - yMin) / (yMax - yMin);
    // Ray inputs
    m_inputs[2] = rayResults[0].fraction;
    m_inputs[3] = rayResults[1].fraction;
    m_inputs[4] = rayResults[2].fraction;
    // Self velocity input
    //const b2Vec2& velocity = b2Body_GetLinearVelocity(m_body);
    //m_inputs[5] = velocity.x;
    //m_inputs[6] = velocity.x;
    // Change in position input
    b2Vec2 currentPosition = b2Body_GetPosition(m_body);
    float dx = currentPosition.x - m_previousPosition.x;
    float dy = currentPosition.y - m_previousPosition.y;
    m_previousPosition = currentPosition;

    return m_inputs;
}

void Vehicle::Act(std::vector<float>& inputs)
{
    // TODO set m_outputs to named variables
    // Agent moves the vehicle
    // Network outputs are normalized from [0, 1]
    m_outputs = m_brain.Predict(inputs);

    // Transform outputs to represent all possible acceleration vectors
    //  ouputs in range from [-0.5 to 0.5)
    if (m_outputs[0] < 0.5f)
        m_outputs[0] *= -1.f;
    else // m_outputs[0] >= 0.5f
        m_outputs[0] -= 0.5f;

    if (m_outputs[1] < 0.5f)
        m_outputs[1] *= -1.f;
    else // m_outputs[1] >= 0.5f
        m_outputs[1] -= 0.5f;

    // Throttle pressure
    float maxAccel = 1.f;
    //m_outputs[2] *= maxAccel; // mapped to [0, maxAccel]

    // Move agent
    //b2Vec2 acceleration = { m_outputs[0] * m_outputs[2], m_outputs[1] * m_outputs[2] };
    //b2Vec2 velocity = b2Body_GetLinearVelocity(m_body) + acceleration;
    float acceleration = 3.f * m_outputs[2];
    b2Body_SetLinearVelocity(m_body, { m_outputs[0] * acceleration, m_outputs[1] * acceleration });
    //b2Body_ApplyForceToCenter(m_body, { m_outputs[0], m_outputs[1] }, true);

    // Rotate agent
    float maxRotationSpeed = 3.f;
    float desiredRotation = (m_outputs[3] * 2.f - 1.f) * maxRotationSpeed; // mapped to [-max, max] rotation speed
    b2Body_SetAngularVelocity(m_body, desiredRotation);
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
    float max = dist.max();
    float min = dist.min();
    
    // random layer's biases altered
    int randomLayer = (dist(gen) - min) / (max - min) * m_brain.getLayers().size() - 1;
    std::vector<std::vector<float>> biases = m_brain.copyBiases();
    for (auto& bias : biases[randomLayer])
    {
        // Normalize distribution generator to [-1, 1]
        float randomBiasChange = ((dist(gen) - min) / (max - min) * 2.f) - 1.f;
        bias += randomBiasChange * mutationFactor;
    }

    // random layer's weights altered
    randomLayer = (dist(gen) - min) / (max - min) * m_brain.getLayers().size() - 1;
    std::vector<std::vector<std::vector<float>>> weights = m_brain.copyWeights();
    for (auto& neuron : weights[randomLayer])
    {
        for (auto& weight : neuron)
        {
            // Normalize distribution generator to [0, 1]
            float randomWeightChange = ((dist(gen) - min) / (max - min));
            weight += dist(gen) * mutationFactor;
        }
    }

    // Update brain
    m_brain.setWeightsAndBiases(weights, biases);
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

void Vehicle::UpdateScore(float collisionPenalizer)
{
    // Since track is centered around (0, 0), position is the angle offset for the circle
    //  otherwise this would need to be translated to circle's center
    b2Vec2 position = b2Body_GetPosition(m_body);
    float currentAngle = atan2f(position.x, position.y);

    // Calculate angle difference, map to [-pi, pi]
    float angleDiff = currentAngle - m_previousAngle;
    if (angleDiff > B2_PI) angleDiff -= 2 * B2_PI;
    if (angleDiff < -B2_PI) angleDiff += 2 * B2_PI;

    // Negate angle diff for clockwise track (angle is decreaseing)
    m_totalAngleTraversed -= angleDiff;
    m_previousAngle = currentAngle;

    float collisionPenalties = m_wallCollisions * collisionPenalizer;
    m_score = m_totalAngleTraversed * 3.f - collisionPenalties; // FIXME 3.f angle traversed mutliplier
}

float Vehicle::GetScore() const { return m_score; }
