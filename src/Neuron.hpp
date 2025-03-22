#pragma once
#include <vector>
#include <random>

/// @brief Feed forward neuron, it has a unique weight associated with each input.
/// @note This means there are Layers * Input Dimensions weights
class Neuron 
{
public:
    void InitRandomWeightsAndBias(int neuronCount, 
        std::uniform_real_distribution<float>& dist,
        std::mt19937& gen);

    const std::vector<float>& getWeights() const;
    float getBias() const;
    std::vector<float>& weights();
    float&              bias();

    void setBias(float bias);
    void setWeights(const std::vector<float>&weights);

private:
    std::vector<float> m_weights;
    float m_bias;
};