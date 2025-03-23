#pragma once
#include "Network.hpp"
#include <box2d/id.h>
#include <box2d/math_functions.h>
#include <functional>
#include <random>
#include <vector>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/VertexArray.hpp>

class Vehicle
{
public:
    Vehicle();
    Vehicle(size_t rayCount);
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

    std::vector<float>& Sense(b2WorldId world, float fov, size_t rayCount, float xMin, float xMax, float yMin, float yMax, float b2Scale);

    void Act(std::vector<float>& inputs, float halfWidth, float halfHeight);

    void Crossover(Vehicle& parent1, Vehicle& parent2, float parent1Score, float parent2Score,
        std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void ResetBody(float x, float y, float rotation);

    void MutateBrain(float mutationFactor, std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void ScrambleBrain(std::uniform_real_distribution<float>& dist, std::mt19937& gen);

    void ZeroScoring();

    void IncrementWallCollisions();

    void UpdateScore(float collisionPenalizer, float distanceMultiplier, float generationTimer);

    float GetScore() const;

    void Draw(sf::RenderWindow& window, sf::Color outlineColor, b2BodyId vehicleBody, float halfWidth, float halfHeight, float box2dScale);


	b2BodyId m_body;
	Network m_brain;
    sf::VertexArray m_rays;

private:
    // Brain
    std::vector<float> m_inputs;
    std::vector<float> m_outputs;

    // Score
    b2Vec2 m_previousPosition;
    float m_previousPositionalAngle;
    float m_score;
    int m_wallCollisions;
    float m_totalAngleTraversed;
    float m_lapsCompleted;
    int m_standstillCount;
};