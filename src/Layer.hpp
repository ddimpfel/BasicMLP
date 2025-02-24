#pragma once
#include <vector>
#include <functional>
#include "Neuron.hpp"

/// @brief Small vector wrapper to make reading the network easier.
class Layer 
{
public:
    Layer(int neuronCount,
        const std::function<float(float)>& ActivationFunction
    );

    std::vector<float>& Forward(const std::vector<float> &inputs);

    const std::vector<float>& getOutputs() const;
    const std::vector<Neuron>& getNeurons() const;
    std::vector<Neuron>& neurons();
    Neuron& neuron(size_t n);

private:
    std::vector<Neuron> m_neurons;
    std::vector<float > m_outputs;  // member variable to avoid reinstantiating each pass
    std::function<float(float)> ActivationFunction;
};