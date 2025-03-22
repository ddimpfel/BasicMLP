#pragma once
#include "Network.hpp"
#include <box2d/id.h>
#include <box2d/math_functions.h>
#include <functional>
#include <random>
#include <vector>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RenderWindow.hpp>

class Vehicle
{
public:
	Vehicle();
	~Vehicle();

    void InitBody(
        b2WorldId world, 
        float halfWidth, 
        float halfHeight, 
        float x, float y, 
        float rotation
    );

    void InitBrain(
        std::vector<int>& architecture,
        const std::function<float(float)>& ActivationSigmoid,
        const std::function<float(float)>& DerivativeActivationSigmoid,
        const std::function<float(size_t, float, float)>& LossMSE,
        std::uniform_real_distribution<float>& dist,
        std::mt19937& gen
    );

    std::vector<float>& Sense(b2WorldId world, float fov, size_t rayCount, float xMin, float xMax, float yMin, float yMax);

    void Act(std::vector<float>& inputs, float halfWidth, float halfHeight);

    void Evolve(Network betterBrain, float mutationFactor, std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void Crossover(Vehicle& parent1, Vehicle& parent2, std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void ResetBody(float x, float y, float rotation);

    void MutateBrain(float mutationFactor, std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void ScrambleBrain(std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void InitializeScoring();

    void IncrementWallCollisions();

    void UpdateScore(float collisionPenalizer, float distanceMultiplier);

    float GetScore() const;

    void Draw(sf::RenderWindow& window, sf::Color outlineColor, b2BodyId vehicleBody, float halfWidth, float halfHeight, float box2dScale);


	b2BodyId m_body;
	Network m_brain;

private:
    // Brain
    std::vector<float> m_inputs;
    std::vector<float> m_outputs;

    // Score
    float m_score;
    int m_wallCollisions;
    b2Vec2 m_previousPosition;
    float m_previousPositionalAngle;
    float m_totalAngleTraversed;
    int m_lapsCompleted;
    int m_standstillCount;
};