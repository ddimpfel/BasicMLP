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
#pragma once
#include <vector>
#include <random>
#include <memory>
#include <optional>
#include <functional>

#include "Layer.hpp"

class Network 
{
public:
    /*!
     *  @brief Default initializer. This should only be used to instantiate a network. No data is
     *      present in the network (no weights, biases, or functions are set).
     */
    Network();

    /*!
     *  @brief Create a FFNN using an initial network architecture. Learning rate defaults to 0.1f,
     *      use setLearningRate(float) after construction. Weights and biases are randomly generated.
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
    Network(
        const std::vector<int>& initialLayers,
        const std::function<float(float)>& ActivationFunction,
        const std::function<float(float)>& DerivativeActivationFunction,
        const std::function<float(size_t, float, float)>& LossFunction,
        std::unique_ptr<std::uniform_real_distribution<float>> dist = nullptr,
        std::unique_ptr<std::mt19937> gen = nullptr
    );

    /*!
     *  @brief Create a FFNN using an initial network architecture. Learning rate defaults to 0.1f,
     *      use setLearningRate(float) after construction. Weights and biases are preset and loaded in.
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
    Network(
        const std::vector<int>& initialLayers,
        const std::function<float(float)>& ActivationFunction,
        const std::function<float(float)>& DerivativeActivationFunction,
        const std::function<float(size_t, float, float)>& LossFunction,
        std::vector<std::vector<std::vector<float>>>& optWeights,
        std::vector<std::vector<float>>& optBiases
    );

    /*!
     *  @brief Predict outputs by passing inputs forward through the network.
     *
     *  @param[in] inputs - std::vector<float> inputs to predict on
     *
     *  @return std::vector<float> predicted outputs
     */
    std::vector<float> Predict(const std::vector<float>& inputs);

    /*!
     *  @brief Fit (train) the model based on the expected outputs with the inputs predicted on.
     *
     *  @param[in] inputs - std::vector<float> inputs to predict on
     *
     *  @param[in] expectedOutputs - std::vector<float> expected outputs the model will adjust to match
     *
     */
    void Fit(const std::vector<float> &inputs, const std::vector<float> &expectedOutputs);
    
    /*!
     *  @brief Get the model architecture. This represents each layer with an integer for each neuron per layer.
     *
     *      **Immutable**
     *      
     *  @return std::vector<int> architecture
     */
    const std::vector<int>& getArchitecture() const;

    /*!
     *  @brief Get all of the layers used in the model. These hold the actual data defining the models layers.
     *
     *      **Immutable**
     *
     *  @return const std::vector<Layer>& layers
     */
    const std::vector<Layer>& getLayers() const;

    /*!
     *  @brief Get each layer's ouputs. Mostly used internally for back propagation.
     *
     *      **Immutable**
     *
     *  @return const std::vector<std::vector<float>>& layer outputs
     */
    const std::vector<std::vector<float>>& getLayerOutputs() const;

    /*!
     *  @brief Set the learning rate the network uses to update itself
     */
    void setLearningRate(float lr);

    /*!
     *  @brief Set network weights
     */
    void setWeights(const std::vector<std::vector<std::vector<float>>>& weights);

    /*!
     *  @brief Set network biases
     */
    void setBiases(const std::vector<std::vector<float>>& biases);

    /*!
     *  @brief Set network weights and biases
     */
    void setWeightsAndBiases(const std::vector<std::vector<std::vector<float>>>& weights, const std::vector<std::vector<float>>& biases);

    /*!
     *  @return hard copy of weights 3d vector
     */
    std::vector<std::vector<std::vector<float>>> copyWeights();

    /*!
     *  @return hard copy of biases 2d vector
     */
    std::vector<std::vector<float>> copyBiases();

private:
    /*!
     *  @brief Private function to set network layers when passing in weights and biases.
     *
     *      @param[in] ActivationFunction - activation function handed to layers for internal use
     *
     *      @param[in] optWeights - optional weights to initialize the model to
     *
     *      @param[in] optBiases - optional biases to initialize the model to
     */
    void InitLayersPreset(
        const std::function<float(float)> &ActivationFunction,
        const std::optional<std::vector<std::vector<std::vector<float>>>> &optWeights = std::nullopt,
        const std::optional<std::vector<std::vector<float>>> &optBiases = std::nullopt
    );

    /*!
     *  @brief Private function to set network layers when passing in weights and biases.
     *
     *      @param[in] ActivationFunction - activation function handed to layers for internal use
     *
     *      @param[in] dist - uniform random real distribution to initialize weights with
     *
     *      @param[in] gen - mt19937 random number generator to create uniform random numbers
     */
    void InitLayersRandom(
        const std::function<float(float)>& ActivationFunction,
        std::uniform_real_distribution<float> dist,
        std::mt19937 gen
    );

    /*!
     *   @brief Pass inputs forward through each layer
     *
     *      @param[in] inputs - std::vector<float> inputs to predict on
     *
     *      @return m_layerOutputs.back() is prediction
     *      @note This function is O(layers * neurons * inputs)
     */
    std::vector<float>& ForwardPass(const std::vector<float>& inputs);

    /*!
     *  @brief Propagate the loss from output to input layer.
     *
     *      @param[in] expectedOutputs - std::vector<float>
     *      @param[in] learningRate - float
     *
     *      @note 
     *      This is how the network learns. Loss is calculated at the outputs and utilized
     *      to update the weights. The partial derivative of the error with respect to the
     *      outputs gives the error a layer contributes to.  This updates the layers nodes
     *      and is also used to find the previous layers loss.
     */
    void BackwardPropagation(const std::vector<float>& expectedOutputs);

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
    std::vector<float> CalculateLossPreviousLayer(std::vector<float> currentLoss, size_t currentLayer, size_t numNeurons);

    /*!
     *  @brief Internally apply the weights and biases gradients to each layer in the network in back propagation.
     *
     *  @param[in] weightGradients - each layer's weight gradients
     *
     *  @param[in] biasGradients - each layer's bias gradients
     */
    void ApplyGradients(
        std::vector<std::vector<std::vector<float>>>& weightGradients, 
        std::vector<std::vector<float>>& biasGradients
    );

    std::vector<int> m_architecture;
    std::vector<Layer> m_layers;
    std::vector<std::vector<float>> m_layerOutputs;

    float m_learningRate;
    std::function<float(float)> DerivativeActivationFunction; // TODO change to use 'reverse mode automatic differentiation'
    std::function<float(size_t, float, float)> LossFunction;
};