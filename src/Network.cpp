/*!
 *  @file Network.cpp
 *  @author Dominick Dimpfel
 *  @date 2025-03-20
 *  @project Basic NN
 *
 *  Implements a feed forward neural network. Networks can be initialized randomly or by passing in layers,
 *      weights, biases, and the functions used with the network.
 *
 *      Still to do:
 *          Implementing back propagation to allow efficient handling of derivative activation functions rather
 *          than user implementation.
 *
 *          Copying of the network to a json for file saving.
 */
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
 *  @brief Create a FFNN using an initial network architecture. Learning rate defaults to 0.1f,
 *      use setLearningRate(float) after construction. Weights and biases are randomly generated.
 *
 *      **Ex initialArchitecture: {{2, 2, 2}}** - network with 2 inputs, 1 hidden
 *      layer with 2 nodes, and 2 output nodes
 *
 *      @param[in] initialArchitecture - Represents the input, each layer,
 *      and the neurons in their layers.
 *
 *      @param[in] ActivationFunction - Funciton passed to each layer
 *      for neuron output activation.
 *      
 *      @param[in] LossFunction - Funciton used in back propagation to
 *      determine loss from forward pass.
 *
 *      @note Either weights and biases **or** dist and gen are sent
 *
 *      @param[in] dist - uniform distribution with starting range
 *      @param[in] gen - mt19937 generator
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
    m_learningRate(0.1f),
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
 *  @brief Create a FFNN using an initial network architecture. Learning rate defaults to 0.1f,
 *      use setLearningRate(float) after construction. Weights and biases are preset and loaded in.
 *
 *      @param[in] initialArchitecture - Represents the input, each layer,
 *      and the neurons in their layers.
 *
 *      **Ex: {{2, 2, 2}}** - network with 2 inputs, 1 hidden
 *      layer with 2 nodes, and 2 output nodes
 *
 *      @param[in] ActivationFunction - Funciton passed to each layer
 *      for neuron output activation.
 *
 *      @param[in] LossFunction - Funciton used in back propagation to
 *      determine loss from forward pass.
 *
 *      @note Either weights and biases **or** dist and gen are sent
 *
 *      @param[in] weights - Preset weights, ensure they are the same
 *      dimensions as initialLayers.
 *      @param[in] biases - Preset biases, ensure they are the same
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
    m_learningRate(0.1f),
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

/*!
 *  @brief Private function to set network layers when passing in weights and biases.
 *
 *      @param[in] ActivationFunction - activation function handed to layers for internal use
 *
 *      @param[in] optWeights - optional weights to initialize the model to
 *      
 *      @param[in] optBiases - optional biases to initialize the model to
 */
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

/*!
 *  @brief Private function to set network layers when passing in weights and biases.
 *
 *      @param[in] ActivationFunction - activation function handed to layers for internal use
 *
 *      @param[in] dist - uniform random real distribution to initialize weights with
 *
 *      @param[in] gen - mt19937 random number generator to create uniform random numbers
 */
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

/*!
 *  @brief Predict outputs by passing inputs forward through the network.
 *
 *  @param[in] inputs - std::vector<float> inputs to predict on
 *
 *  @return std::vector<float> predicted outputs
 */
std::vector<float> Network::Predict(const std::vector<float>& inputs)
{
    return ForwardPass(inputs);
}

/*!
 *  @brief Fit (train) the model based on the expected outputs with the inputs predicted on.
 *
 *  @param[in] inputs - std::vector<float> inputs to predict on
 *
 *  @param[in] expectedOutputs - std::vector<float> expected outputs the model will adjust to match
 *
 */
void Network::Fit(const std::vector<float>& inputs, const std::vector<float>& expectedOutputs)
{
    ForwardPass(inputs);
    BackwardPropagation(expectedOutputs);
}

/*!
 *   @brief Pass inputs forward through each layer
 *
 *      @param[in] inputs - std::vector<float> inputs to predict on
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
 *      @param[in] expectedOutputs: std::vector<float>
 *      @param[in] learningRate: float
 *
 *      @note
 *      This is how the network learns. Loss is calculated at the outputs and utilized
 *      to update the weights. The partial derivative of the error with respect to the
 *      outputs gives the error a layer contributes to.  This updates the layers nodes
 *      and is also used to find the previous layers loss.
 */
void Network::BackwardPropagation(const std::vector<float>& expectedOutputs)
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

    ApplyGradients(weightGradients, biasGradients);
}

/*!
 *  @brief Internal function to calculate each layer's loss in back propagation step.
 *
 *  @param[in] currentLoss - The loss back propagated to the current layer
 *
 *  @param[in] currentLayer - The current layer
 *
 *  @param[in] numNeurons - The number of neurons in the current layer
 *
 *  @return loss left over for the next layer's back propagation
 */
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

/*!
 *  @brief Internally apply the weights and biases gradients to each layer in the network in back propagation.
 *
 *  @param[in] weightGradients - each layer's weight gradients
 *
 *  @param[in] biasGradients - each layer's bias gradients
 */
void Network::ApplyGradients(
    std::vector<std::vector<std::vector<float>>>& weightGradients, 
    std::vector<std::vector<float>>& biasGradients)
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
                weights[w] -= m_learningRate * weightGradients[l][n][w];
            }
            neurons[n].bias() -= m_learningRate * biasGradients[l][n];
        }
    }
}

const std::vector<int>& Network::getArchitecture() const { return m_architecture; }
const std::vector<Layer>& Network::getLayers() const { return m_layers; }
const std::vector<std::vector<float>>& Network::getLayerOutputs() const { return m_layerOutputs; }

void Network::setLearningRate(float lr) { m_learningRate = lr; }

/*!
 *  @brief Set network weights
 */
void Network::setWeights(const std::vector<std::vector<std::vector<float>>>& weights)
{
    for (size_t l = 0; l < m_architecture.size(); l++)
    {
        for (size_t n = 0; n < m_architecture[l]; n++)
        {
            m_layers[l].neuron(n).setWeights(weights[l][n]);
        }
    }
}

/*!
 *  @brief Set network biases
 */
void Network::setBiases(const std::vector<std::vector<float>>& biases)
{
    for (size_t l = 0; l < m_architecture.size(); l++)
    {
        for (size_t n = 0; n < m_architecture[l]; n++)
        {
            m_layers[l].neuron(n).setBias(biases[l][n]);
        }
    }
}

/*!
 *  @brief Set network weights and biases
 */
void Network::setWeightsAndBiases(const std::vector<std::vector<std::vector<float>>>& weights, const std::vector<std::vector<float>>& biases)
{
    for (size_t l = 0; l < m_architecture.size(); l++)
    {
        for (size_t n = 0; n < m_architecture[l]; n++)
        {
            m_layers[l].neuron(n).setWeights(weights[l][n]);
            m_layers[l].neuron(n).setBias(biases[l][n]);
        }
    }
}

/*!
 *  @return hard copy of weights 3d vector
 */
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

/*!
 *  @return hard copy of biases 2d vector
 */
std::vector<std::vector<float>> Network::copyBiases()
{
    std::vector<std::vector<float>> biases;
    biases.reserve(m_layers.size());
    for (auto& layer : m_layers)
    {
        std::vector<float> layerBiases;
        layerBiases.reserve(layer.getNeurons().size());
        for (auto& neuron : layer.neurons())
        {
            layerBiases.push_back(neuron.bias());
        }
        biases.push_back(std::move(layerBiases));
    }
    return biases;
}
