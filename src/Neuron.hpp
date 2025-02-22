#pragma once
#include <vector>
#include <random>

/// @brief Feed forward neuron, it has a unique weight associated with each input.
/// @note This means there are Layers * Input Dimensions weights
class Neuron 
{
public:
    void InitRandomWeightsAndBias(std::uniform_real_distribution<> &dist, std::mt19937 &gen) 
    {
        for (size_t i = 0; i < m_weights.size(); i++)
        {
            m_weights[i] = dist(gen);
        }
        this->m_bias = dist(gen);
    }

    std::vector<float>& getWeights() { return m_weights; }
    float&   getBias()    { return m_bias; }

    void setWeights(const std::vector<float>&weights) { m_weights = weights; }

private:
    std::vector<float> m_weights;
    float m_bias;
};