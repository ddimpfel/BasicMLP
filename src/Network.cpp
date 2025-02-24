#include "Network.hpp"

#include <type_traits>
#include <functional>
#include <stdexcept>
#include <optional>
#include <memory>
#include <random>
#include <vector>

#include "Layer.hpp"
#include "Neuron.hpp"
#include "nnMath.hpp"

Network::Network() = default;

/*!
 *  @brief Create a FFNN using an initial network architecture.
 *
 *      @param [in] initialArchitecture - Represents the input, each layer,
 *      and the neurons in their layers.
 *
 *      **Ex: {{2, 2, 2}}** - network with 2 inputs, 1 hidden
 *      layer with 2 nodes, and 2 output nodes
 *
 *      @param [in] ActivationFunction - Funciton passed to each layer
 *      for neuron output activation.
 *      
 *      @param [in] LossFunction - Funciton used in back propagation to
 *      determine loss from forward pass.
 *
 *      @note Either weights and biases **or** dist and gen are sent
 *
 *      @param [in] dist - uniform distribution with starting range
 *      @param [in] gen - mt19937 generator
 */
Network::Network(
    const std::vector<int>& initialArchitecture,
    const std::function<float(float)>& ActivationFunction, 
    const std::function<float(float)>& DerivativeActivationFunction, 
    const std::function<float(size_t, float, float)>& LossFunction,
    std::unique_ptr<std::uniform_real_distribution<float>> dist,
    std::unique_ptr<std::mt19937> gen
)   : 
    m_architecture(initialArchitecture),
    DerivativeActivationFunction(DerivativeActivationFunction),
    LossFunction(LossFunction) 
{
    if (initialArchitecture.size() < 3)
    {
        std::cerr << __FILE__ << ":" << __LINE__ << 
            " Initial Architecture should be the full network architecture";
        throw std::runtime_error("Initial Architecture should be the full network architecture");
    }
    if (!dist || !gen)
    {
        std::cerr << __FILE__ << ":" << __LINE__ <<
            " Network cannot be initialized without RNG";
        throw std::runtime_error("Network cannot be initialized without RNG");
    }

    m_layers.reserve(initialArchitecture.size());
    m_layerOutputs.resize(initialArchitecture.size());
    InitLayersRandom(ActivationFunction, *dist, *gen);
}

/*!
 *  @brief Create a FFNN using an initial network architecture.
 *
 *      @param [in] initialArchitecture - Represents the input, each layer,
 *      and the neurons in their layers.
 *
 *      **Ex: {{2, 2, 2}}** - network with 2 inputs, 1 hidden
 *      layer with 2 nodes, and 2 output nodes
 *
 *      @param [in] ActivationFunction - Funciton passed to each layer
 *      for neuron output activation.
 *
 *      @param [in] LossFunction - Funciton used in back propagation to
 *      determine loss from forward pass.
 *
 *      @note Either weights and biases **or** dist and gen are sent
 *
 *      @param [in] weights - Preset weights, ensure they are the same
 *      dimensions as initialLayers.
 *      @param [in] biases - Preset biases, ensure they are the same
 *      dimensions as num neurons for each layer.
 */
Network::Network(
    const std::vector<int>& initialArchitecture,
    const std::function<float(float)>& ActivationFunction,
    const std::function<float(float)>& DerivativeActivationFunction,
    const std::function<float(size_t, float, float)>& LossFunction,
    std::vector<std::vector<std::vector<float>>>& weights,
    std::vector<std::vector<float>>& biases
) :
    m_architecture(initialArchitecture),
    DerivativeActivationFunction(DerivativeActivationFunction),
    LossFunction(LossFunction)
{
    if (initialArchitecture.size() < 3)
    {
        std::cerr << __FILE__ << ":" << __LINE__ <<
            " Initial Architecture should be the full network architecture";
        throw std::runtime_error("Initial Architecture should be the full network architecture");
    }

    // TODO check the weights and biases have equivalent values to the architecture passed in
    m_layers.reserve(initialArchitecture.size());
    m_layerOutputs.resize(initialArchitecture.size());
    InitLayersPreset(ActivationFunction, weights, biases);
}

void Network::InitLayersPreset(
    const std::function<float(float)>& ActivationFunction, 
    const std::optional<std::vector<std::vector<std::vector<float>>>>& optWeights, 
    const std::optional<std::vector<std::vector<float>>>& optBiases)
{
    if (optBiases.has_value()) 
    {
        for (size_t l = 0; l < m_architecture.size(); l++) 
        {
            int neuronCount = m_architecture[l];
            m_layers.emplace_back(neuronCount, ActivationFunction);

            // Skip input layer as it has no incoming weights or biases
            if (l == 0) continue;

            auto& neurons = m_layers[l].neurons();
            for (size_t n = 0; n < neurons.size(); n++) 
            {
                neurons[n].setWeights(optWeights.value()[l - 1][n]);
                neurons[n].setBias(optBiases.value()[l - 1][n]);
            }
        }
    }
    else {
        for (size_t l = 0; l < m_architecture.size(); l++)
        {
            int neuronCount = m_architecture[l];
            m_layers.emplace_back(neuronCount, ActivationFunction);

            // Skip input layer as it has no incoming weights or biases
            if (l == 0) continue;

            auto& neurons = m_layers[l].neurons();
            for (size_t n = 0; n < neurons.size(); n++) 
            {
                neurons[n].setWeights(optWeights.value()[l - 1][n]);
                neurons[n].setBias(0.f);
            }
        }
    }
}

void Network::InitLayersRandom(
    const std::function<float(float)>& ActivationFunction,
    std::uniform_real_distribution<float> dist,
    std::mt19937 gen)
{
    for (size_t l = 0; l < m_architecture.size(); l++)
    {
        int neuronCount = m_architecture[l]; // TODO init to argument
        m_layers.emplace_back(neuronCount, ActivationFunction);

        // Skip input layer as it has no incoming weights or biases
        if (l == 0) continue;

        for (auto& neuron : m_layers[l].neurons())
        {
            // First hidden layer has weights equal to the number of inputs
            size_t wCount;
            if (l == 1) wCount = m_architecture[0];
            else        wCount = m_layers[l - 1].getNeurons().size();
            neuron.InitRandomWeightsAndBias(wCount, dist, gen);
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
    // First layer is the input layer
    m_layerOutputs[0] = inputs;
    for (size_t l{1}; l < m_architecture.size(); l++)
    {
        auto& layer = m_layers[l];
        // Feed inputs forward into each layer
        m_layerOutputs[l] = layer.Forward(m_layerOutputs[l - 1]);
    }

    return m_layerOutputs.back();
}

/*!
 *  @brief Propagate the loss from output to input layer.
 *
 *      This is how the network learns. Loss is calculated at the outputs and utilized
 *      to update the weights. The partial derivative of the error with respect to the
 *      outputs of a layer gives the  error a layer  contributes to.  This updates the
 *      layers nodes and is also used to find the previous layers loss.
 *
 *      @param [in] expectedOutputs: correct outputs of network
 *      @param [in] learningRate: scalar effecting gradient descent velocity
 */
void Network::BackwardPropagation(const std::vector<float>& expectedOutputs, float learningRate)
{
    std::vector<float>& predicted = m_layerOutputs.back();

    // Get the loss from each output node
    std::vector<float> loss(expectedOutputs.size());
    for (size_t i = 0; i < expectedOutputs.size(); i++) 
    {
        float lossGradient = LossFunction(expectedOutputs.size(), predicted[i], expectedOutputs[i]);
        float activationGradient = DerivativeActivationFunction(predicted[i]);
        loss[i] = lossGradient * activationGradient;
    }

    // Get gradients only for non input layers
    std::vector<std::vector<std::vector<float>>> weightGradients(m_layers.size());
    std::vector<std::vector<float>>              biasGradients(m_layers.size());

    // Find the gradients by iterating backward through the network
    for (int l = m_layers.size() - 1; l > 0; l--) 
    {
        size_t numNeurons = m_layers[l].getNeurons().size();

        // This layers inputs is the previous layers outputs
        const auto& layerInputs = m_layerOutputs[l - 1];

        // Update this layers gradients
        weightGradients[l] = nnMath::outer(loss, layerInputs);
        biasGradients[l] = loss;

        loss = CalculateLossPreviousLayer(loss, l, numNeurons);
    }

    ApplyGradients(weightGradients, biasGradients, learningRate);
}

std::vector<float> Network::CalculateLossPreviousLayer(std::vector<float> currentLoss, size_t currentLayer, size_t numNeurons)
{
    // Loss from previous outputs
    std::vector<float> prevLoss(m_layerOutputs[currentLayer - 1].size(), 0.f);

    // Each neuron in previous layer is affected by all neurons in current layer
    for (size_t i = 0; i < prevLoss.size(); i++) 
    {
        float lossSum = 0.f;

        // Sum up all contributions to this neuron's error
        for (size_t n = 0; n < numNeurons; n++) 
        {
            float weight = m_layers[currentLayer].neuron(n).weights()[i];
            lossSum += currentLoss[n] * weight;
        }

        // Multiply by derivative of activation function to get activation effect on loss
        prevLoss[i] = lossSum * DerivativeActivationFunction(m_layerOutputs[currentLayer - 1][i]);
    }

    return prevLoss;
}

void Network::ApplyGradients(
    std::vector<std::vector<std::vector<float>>>& weightGradients, 
    std::vector<std::vector<float>>& biasGradients, 
    float learningRate)
{
    // Start from first hidden layer (first with incoming weights/biases)
    for (size_t l = 1; l < m_layers.size(); l++) 
    {
        auto& neurons = m_layers[l].neurons();

        for (size_t n = 0; n < neurons.size(); n++) 
        {
            // Update each weight using its specific gradient
            auto& weights = neurons[n].weights();
            for (size_t w = 0; w < weights.size(); w++) 
            {
                weights[w] -= learningRate * weightGradients[l][n][w];
            }
            neurons[n].bias() -= learningRate * biasGradients[l][n];
        }
    }
}

const std::vector<int>& Network::getArchitecture() const { return m_architecture; }
const std::vector<Layer>& Network::getLayers() const { return m_layers; }
const std::vector<std::vector<float>>& Network::getLayerOutputs() const { return m_layerOutputs; }

std::vector<std::vector<std::vector<float>>> Network::copyWeights()
{
    std::vector<std::vector<std::vector<float>>> weights;
    weights.reserve(m_layers.size());
    for (auto& layer : m_layers) 
    {
        std::vector<std::vector<float>> layerWeights;
        layerWeights.reserve(layer.getNeurons().size());
        for (auto& neuron : layer.neurons()) 
        {
            layerWeights.push_back(neuron.weights());
        }
        weights.push_back(std::move(layerWeights));
    }
    return weights;
}
