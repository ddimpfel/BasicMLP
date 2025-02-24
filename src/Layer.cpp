#include "Layer.hpp"
#include <vector>
#include <functional>
#include "Neuron.hpp"
#include "NNMath.hpp"

Layer::Layer(int neuronCount,
    const std::function<float(float)>& ActivationFunction
) : ActivationFunction(ActivationFunction)
{
    m_neurons.resize(neuronCount);
    m_outputs.resize(neuronCount);
}

std::vector<float>& Layer::Forward(const std::vector<float>& inputs)
{
    // Each neuron must loop over each input and then add bias
    // before being passed to the activation funciton. The final
    // activation value is assigned to this neurons output.
    for (size_t node = 0; node < m_neurons.size(); node++)
    {
        float fNeuronSum = nnMath::dot(inputs, m_neurons[node].weights());
        fNeuronSum += m_neurons[node].bias();

        float fActivationValue = ActivationFunction(fNeuronSum);
        m_outputs[node] = fActivationValue;
    }

    return m_outputs;
}

const std::vector<float>& Layer::getOutputs() const { return m_outputs; }
const std::vector<Neuron>& Layer::getNeurons() const { return m_neurons; }
std::vector<Neuron>& Layer::neurons() { return m_neurons; }
Neuron& Layer::neuron(size_t n) { return m_neurons[n]; }

