#include "Neuron.hpp"

void Neuron::InitRandomWeightsAndBias(int neuronCount, 
    std::uniform_real_distribution<float>& dist, 
    std::mt19937& gen)
{
    m_weights.reserve(neuronCount);
    for (size_t i = 0; i < neuronCount; i++)
    {
        m_weights.emplace_back(dist(gen));
    }
    // Biases should be initialized to 0 to avoid forcing the model to
    // overcome initial bias offsets.
    //m_bias = dist(gen);
    m_bias = 0.f;
}

const std::vector<float>& Neuron::getWeights() const { return m_weights; }
float Neuron::getBias() const { return m_bias; }
std::vector<float>& Neuron::weights() { return m_weights; }
float& Neuron::bias() { return m_bias; }
void Neuron::setBias(float bias) { m_bias = bias; }
void Neuron::setWeights(const std::vector<float>& weights) { m_weights = weights; }
