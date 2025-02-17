#include "Network.hpp"

#include <functional>
#include <type_traits>
#include <optional>
#include <random>
#include <vector>

#include "Layer.hpp"
#include "nnMath.hpp"

Network::Network() = default;

Network::Network(int layerCount, 
    const std::function<float(float)>& ActivationFunction, 
    const std::function<float(float)>& DerivativeActivationFunction, 
    const std::function<float(size_t, float, float)>& LossFunction, 
    const std::optional<std::vector<std::vector<std::vector<float>>>>& optWeights, 
    const std::optional<std::vector<std::vector<float>>>& optBiases) 
    : 
    pDerivativeActivationFunction(DerivativeActivationFunction),
    pLossFunction(LossFunction) {
    m_layers.reserve(layerCount);
    m_layerOutputs.resize(layerCount + 1);
    InitLayers(layerCount, ActivationFunction, optWeights, optBiases);
}

void Network::InitRandomizer(int lower_bound, int upper_bound, std::optional<int> opt_seed)
{
    m_dist = std::uniform_real_distribution<>(lower_bound, upper_bound);
    std::random_device rd;
    m_gen.seed(opt_seed.value_or(rd()));
}

void Network::InitLayers(int layerCount, const std::function<float(float)>& ActivationFunction, const std::optional<std::vector<std::vector<std::vector<float>>>>& optWeights, const std::optional<std::vector<std::vector<float>>>& optBiases)
{
    if (optWeights.has_value()) {
        if (optBiases.has_value()) {
            for (size_t l = 0; l < layerCount; l++) {
                int n_count = 2; // TODO init to argument
                m_layers.emplace_back(n_count, ActivationFunction);
                auto& neurons = m_layers[l].getNeurons();
                for (size_t n = 0; n < neurons.size(); n++) {
                    neurons[n].setWeights(optWeights.value()[l][n]);
                    neurons[n].getBias() = optBiases.value()[l][n];
                }
            }
        }
        else {
            for (size_t l = 0; l < layerCount; l++) {
                int n_count = 2; // TODO init to argument
                m_layers.emplace_back(n_count, ActivationFunction);
                auto& neurons = m_layers[l].getNeurons();
                for (size_t n = 0; n < neurons.size(); n++) {
                    neurons[n].setWeights(optWeights.value()[l][n]);
                    neurons[n].getBias() = 0.f;
                }
            }
        }
    }
    else { // FIXME
        InitRandomizer(0, 1, 42); // TODO init to arguments
        for (size_t i = 0; i < layerCount; i++) {
            int neurons = 2; // TODO init to argument
            m_layers[i] = Layer(neurons, ActivationFunction);
            m_layerOutputs.resize(neurons);
            for (auto& neuron : m_layers[i].getNeurons()) {
                neuron.InitRandomWeightsAndBias(m_dist, m_gen);
            }
        }
    }
}

std::vector<float> Network::Predict(const std::vector<float>& inputs)
{
    return ForwardPass(inputs);
}

void Network::Fit(const std::vector<float>& inputs, const std::vector<float>& expectedOutputs)
{
    ForwardPass(inputs);
    BackwardPropagation(expectedOutputs);
}

/*!
 *   @brief Pass inputs forward through each layer
 *      
 *      @param [in] inputs: std::vector<float>
 *
 *      @return m_layerOutputs.back() is prediction
 *      @note This function is O(layers * neurons * inputs)
 */
std::vector<float>& Network::ForwardPass(const std::vector<float>& inputs)
{
    // The next layers outputs feeds the current layers inputs
    int i = 0;
    m_layerOutputs[i] = inputs;
    for (auto& layer : m_layers) {
        i++;
        // Feed inputs forward into each layer
        m_layerOutputs[i] = layer.Forward(m_layerOutputs[i - 1]);
    }

    return m_layerOutputs.back();
}

/*!
 *  @brief Propagate the loss from output to input layer.
 *
 *      This is how the network learns. Loss is calculated at the outputs and utilized
 *      to update the weights. The partial derivative of the error with respect to the
 *      outputs gives the error a layer contributes to.  This updates the layers nodes
 *      and is also used to find the previous layers loss.
 *
 *      @param [in] expectedOutputs: std::vector<float>
 *      @param [in] learningRate: float
 */
void Network::BackwardPropagation(const std::vector<float>& expectedOutputs, float learningRate)
{
    std::vector<float>& predicted = m_layerOutputs.back();

    // Get the loss from each output node
    std::vector<float> loss(expectedOutputs.size());
    for (size_t i = 0; i < expectedOutputs.size(); i++) {
        float lossGradient = pLossFunction(expectedOutputs.size(), predicted[i], expectedOutputs[i]);
        float activationGradient = pDerivativeActivationFunction(predicted[i]);

        loss[i] = lossGradient * activationGradient;
    }

    std::vector<std::vector<std::vector<float>>> weightGradients(m_layers.size());
    std::vector<std::vector<float>>              biasGradients(m_layers.size());

    // Find the gradients by iterating backward through the network
    for (int l = m_layers.size() - 1; l >= 0; l--) {
        auto& currentLayer = m_layers[l];
        const auto& layerInputs = m_layerOutputs[l];

        size_t numNeurons = currentLayer.getNeurons().size();
        weightGradients[l].resize(numNeurons);
        biasGradients[l].resize(numNeurons);

        for (size_t n = 0; n < numNeurons; n++) {
            // wGradient = error term * input that went through this weight
            weightGradients[l][n] = nnMath::mult(loss, layerInputs);

            // bGradient is the error term
            biasGradients[l][n] = loss[n];
        }

        // If not at first layer, calculate error terms for previous layer
        if (l > 0) {
            loss = CalculateLossPreviousLayer(loss, l, numNeurons);
        }
    }

    ApplyGradients(weightGradients, biasGradients, learningRate);
}

std::vector<float> Network::CalculateLossPreviousLayer(std::vector<float> currentLoss, size_t currentLayer, size_t numNeurons)
{
    // Loss from previous outputs
    std::vector<float> prevLoss(m_layerOutputs[currentLayer - 1].size(), 0.f);

    // Each neuron in previous layer is affected by all neurons in current layer
    for (size_t i = 0; i < prevLoss.size(); i++) {
        float lossSum = 0.f;

        // Sum up all contributions to this neuron's error
        for (size_t n = 0; n < numNeurons; n++) {
            float weight = m_layers[currentLayer].neuron(n).getWeights()[i];
            lossSum += currentLoss[n] * weight;
        }

        // Multiply by derivative of activation function to get activation effect on loss
        prevLoss[i] = lossSum * pDerivativeActivationFunction(m_layerOutputs[currentLayer - 1][i]);
    }

    return prevLoss;
}

void Network::ApplyGradients(std::vector<std::vector<std::vector<float>>>& weightGradients, std::vector<std::vector<float>>& biasGradients, float learningRate)
{
    for (size_t l = 0; l < m_layers.size(); l++) {
        auto& neurons = m_layers[l].getNeurons();
        for (size_t n = 0; n < neurons.size(); n++) {
            // Update each weight using its specific gradient
            auto& weights = neurons[n].getWeights();
            for (size_t w = 0; w < weights.size(); w++) {
                weights[w] -= learningRate * weightGradients[l][n][w];
            }
            neurons[n].getBias() -= learningRate * biasGradients[l][n];
        }
    }
}

const std::vector<Layer>& Network::GetLayers() const { return m_layers; }

std::vector<std::vector<std::vector<float>>> Network::CopyWeights()
{
    std::vector<std::vector<std::vector<float>>> weights;
    weights.reserve(m_layers.size());
    for (auto& layer : m_layers) {
        std::vector<std::vector<float>> layerWeights;
        layerWeights.reserve(layer.getNeurons().size());
        for (auto& neuron : layer.getNeurons()) {
            layerWeights.push_back(neuron.getWeights());
        }
        weights.push_back(std::move(layerWeights));
    }
    return weights;
}