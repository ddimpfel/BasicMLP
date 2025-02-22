#pragma once
#include <vector>
#include <functional>
#include "Neuron.hpp"
#include "NNMath.hpp"

/// @brief Small vector wrapper to make reading the network easier.
class Layer 
{
public:
    Layer(int neuronCount,
        const std::function<float(float)> &ActivationFunction
    ) : pActivationFunction(ActivationFunction)
    {
        m_neurons.resize(neuronCount);
        m_outputs.resize(neuronCount);
    }

    std::vector<float>& Forward(const std::vector<float> &inputs) 
    {
        // Each neuron must loop over each input and then add bias
        // before being passed to the activation funciton. The final
        // activation value is assigned to this neurons output.
        for (size_t node = 0; node < m_neurons.size(); node++) {
            float fNeuronSum = nnMath::dot(inputs, m_neurons[node].getWeights());
            fNeuronSum += m_neurons[node].getBias();

            float fActivationValue = pActivationFunction(fNeuronSum);
            m_outputs[node] = fActivationValue;
        }

        return m_outputs;
    }

    Neuron& neuron(size_t n)            { return m_neurons[n]; }
    const std::vector<Neuron>& getNeurons() const { return m_neurons; }
    std::vector<Neuron>& getNeurons()   { return m_neurons; }
    std::vector<float>& getOutputs()    { return m_outputs; }

private:
    std::vector<Neuron> m_neurons; // TODO using std::vector<neuron> as layer ?? maybe not
    std::vector<float > m_outputs;  // member variable to avoid reinstantiating each pass
    std::function<float(float)> pActivationFunction;
};